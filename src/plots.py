# Plotting and visualization module

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    figure_dpi: int = 140
    save_dpi: int = 180
    figsize_wide: tuple[float, float] = (11.0, 6.0)
    figsize_square: tuple[float, float] = (8.0, 8.0)
    figsize_heatmap: tuple[float, float] = (9.0, 7.0)
    heatmap_cmap: str = "Blues"
    annotate_heatmap: bool = True
    rotation_x: int = 30
    alpha: float = 0.9
    close_after_save: bool = True


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: str | Path, config: Optional[PlotConfig] = None) -> Path:
    config = config or PlotConfig()
    path = ensure_parent_dir(path)
    fig.savefig(path, dpi=config.save_dpi, bbox_inches="tight")
    if config.close_after_save:
        plt.close(fig)
    return path


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_presence_table(
    normalized_df: pd.DataFrame,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
) -> pd.DataFrame:
    required = {"address", "chain", "total_tx_count", "recent_30d_tx_count"}
    missing = required - set(normalized_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for presence table: {sorted(missing)}")

    df = normalized_df.copy()
    df = _coerce_numeric(df, ["total_tx_count", "recent_30d_tx_count"])

    df["is_present"] = (
        (df["total_tx_count"].fillna(0) >= min_total_tx)
        & (df["recent_30d_tx_count"].fillna(0) >= min_recent_30d_tx)
    )

    presence = (
        df.pivot_table(
            index="address",
            columns="chain",
            values="is_present",
            aggfunc="max",
            fill_value=False,
        )
        .astype(bool)
    )
    return presence


def build_overlap_matrix(
    normalized_df: pd.DataFrame,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
) -> pd.DataFrame:
    """
    overlap[source, target] =
        among addresses active on source,
        what share is also active on target
    """
    presence = _build_presence_table(
        normalized_df,
        min_total_tx=min_total_tx,
        min_recent_30d_tx=min_recent_30d_tx,
    )
    chains = list(presence.columns)
    matrix = pd.DataFrame(index=chains, columns=chains, dtype=float)

    for source in chains:
        source_mask = presence[source]
        denom = int(source_mask.sum())

        for target in chains:
            if denom == 0:
                matrix.loc[source, target] = 0.0
            else:
                both = int((presence[source] & presence[target]).sum())
                matrix.loc[source, target] = both / denom

    return matrix


def build_jaccard_matrix(
    normalized_df: pd.DataFrame,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
) -> pd.DataFrame:
    """
    Jaccard(source, target) = |A ∩ B| / |A ∪ B|
    """
    presence = _build_presence_table(
        normalized_df,
        min_total_tx=min_total_tx,
        min_recent_30d_tx=min_recent_30d_tx,
    )
    chains = list(presence.columns)
    matrix = pd.DataFrame(index=chains, columns=chains, dtype=float)

    for a in chains:
        for b in chains:
            inter = int((presence[a] & presence[b]).sum())
            union = int((presence[a] | presence[b]).sum())
            matrix.loc[a, b] = inter / union if union > 0 else 0.0

    return matrix


def build_transition_matrix(
    classified_df: pd.DataFrame,
    *,
    only_migrators: bool = False,
) -> pd.DataFrame:
    required = {"historical_dominant_chain", "recent_dominant_chain"}
    missing = required - set(classified_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for transition matrix: {sorted(missing)}")

    df = classified_df.copy()

    if only_migrators:
        if "behavioral_group" not in df.columns:
            raise ValueError("Column 'behavioral_group' required when only_migrators=True")
        df = df[df["behavioral_group"] == "migrator-like wallet"].copy()

    df = df.dropna(subset=["historical_dominant_chain", "recent_dominant_chain"])
    if df.empty:
        return pd.DataFrame()

    matrix = pd.crosstab(
        df["historical_dominant_chain"],
        df["recent_dominant_chain"],
        dropna=False,
    )
    return matrix


def _plot_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    config: Optional[PlotConfig] = None,
    fmt: str = ".2f",
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    if matrix.empty:
        raise ValueError("Cannot plot an empty matrix")

    fig, ax = plt.subplots(figsize=config.figsize_heatmap, dpi=config.figure_dpi)

    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=config.heatmap_cmap)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=config.rotation_x, ha="right")
    ax.set_yticklabels(matrix.index)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("value", rotation=90)

    if config.annotate_heatmap:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    format(values[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.tight_layout()
    return fig, ax


def plot_overlap_heatmap(
    normalized_df: pd.DataFrame,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
    title: str = "Cross-chain active address overlap",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix = build_overlap_matrix(
        normalized_df,
        min_total_tx=min_total_tx,
        min_recent_30d_tx=min_recent_30d_tx,
    )
    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Target chain",
        ylabel="Source chain",
        config=config,
        fmt=".2f",
    )


def plot_jaccard_heatmap(
    normalized_df: pd.DataFrame,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
    title: str = "Jaccard similarity of active address sets",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix = build_jaccard_matrix(
        normalized_df,
        min_total_tx=min_total_tx,
        min_recent_30d_tx=min_recent_30d_tx,
    )
    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        config=config,
        fmt=".2f",
    )


def plot_behavioral_group_distribution(
    classified_df: pd.DataFrame,
    *,
    title: str = "Behavioral group distribution",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    if "behavioral_group" not in classified_df.columns:
        raise ValueError("Column 'behavioral_group' not found")

    counts = (
        classified_df["behavioral_group"]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    bars = ax.bar(counts.index, counts.values, alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel("Behavioral group")
    ax.set_ylabel("Wallet count")
    ax.tick_params(axis="x", rotation=config.rotation_x)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return fig, ax


def plot_dominant_chain_distribution(
    classified_df: pd.DataFrame,
    *,
    title: str = "Dominant chain distribution",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    if "dominant_chain" not in classified_df.columns:
        raise ValueError("Column 'dominant_chain' not found")

    counts = (
        classified_df["dominant_chain"]
        .fillna("unknown")
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    bars = ax.bar(counts.index, counts.values, alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel("Dominant chain")
    ax.set_ylabel("Wallet count")
    ax.tick_params(axis="x", rotation=config.rotation_x)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return fig, ax


def plot_transition_heatmap(
    classified_df: pd.DataFrame,
    *,
    only_migrators: bool = False,
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix = build_transition_matrix(
        classified_df,
        only_migrators=only_migrators,
    )

    if matrix.empty:
        raise ValueError("Transition matrix is empty")

    if title is None:
        title = (
            "Historical → recent dominant chain transitions (migrators only)"
            if only_migrators
            else "Historical → recent dominant chain transitions"
        )

    return _plot_heatmap(
        matrix.astype(float),
        title=title,
        xlabel="Recent dominant chain",
        ylabel="Historical dominant chain",
        config=config,
        fmt=".0f",
    )


def plot_activity_score_boxplot(
    normalized_df: pd.DataFrame,
    *,
    title: str = "Activity score by chain",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    required = {"chain", "activity_score"}
    missing = required - set(normalized_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = normalized_df.copy()
    df = _coerce_numeric(df, ["activity_score"])
    chains = sorted(df["chain"].dropna().astype(str).unique().tolist())
    data = [
        df.loc[df["chain"] == chain, "activity_score"].dropna().to_numpy()
        for chain in chains
    ]

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    ax.boxplot(data, tick_labels=chains, vert=True)
    ax.set_title(title)
    ax.set_xlabel("Chain")
    ax.set_ylabel("Activity score")
    ax.tick_params(axis="x", rotation=config.rotation_x)

    fig.tight_layout()
    return fig, ax


def plot_recency_histogram(
    normalized_df: pd.DataFrame,
    *,
    chains: Optional[Sequence[str]] = None,
    bins: int = 30,
    title: str = "Days since last active",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    required = {"chain", "days_since_last_active"}
    missing = required - set(normalized_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = normalized_df.copy()
    df = _coerce_numeric(df, ["days_since_last_active"])

    if chains is not None:
        df = df[df["chain"].isin(chains)].copy()

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)

    for chain, group in df.groupby("chain"):
        series = group["days_since_last_active"].dropna()
        if series.empty:
            continue
        ax.hist(
            series,
            bins=bins,
            alpha=0.45,
            label=str(chain),
        )

    ax.set_title(title)
    ax.set_xlabel("Days since last active")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    return fig, ax


def export_overlap_matrix(
    normalized_df: pd.DataFrame,
    path: str | Path,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
) -> Path:
    path = ensure_parent_dir(path)
    matrix = build_overlap_matrix(
        normalized_df,
        min_total_tx=min_total_tx,
        min_recent_30d_tx=min_recent_30d_tx,
    )
    matrix.to_csv(path)
    return path


def export_jaccard_matrix(
    normalized_df: pd.DataFrame,
    path: str | Path,
    *,
    min_total_tx: int = 3,
    min_recent_30d_tx: int = 0,
) -> Path:
    path = ensure_parent_dir(path)
    matrix = build_jaccard_matrix(
        normalized_df,
        min_total_tx=min_total_tx,
        min_recent_30d_tx=min_recent_30d_tx,
    )
    matrix.to_csv(path)
    return path


def export_transition_matrix(
    classified_df: pd.DataFrame,
    path: str | Path,
    *,
    only_migrators: bool = False,
) -> Path:
    path = ensure_parent_dir(path)
    matrix = build_transition_matrix(classified_df, only_migrators=only_migrators)
    matrix.to_csv(path)
    return path