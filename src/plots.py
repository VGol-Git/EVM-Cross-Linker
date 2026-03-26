# plots.py
# Visualization layer for the block-window EVM cross-chain correlation project.
#
# Focus:
# - active / passive / present overlap heatmaps
# - bar charts for wallet counts
# - histogram of first-activity deltas
# - scatter plots for cross-chain sent value
# - correlation heatmaps
# - window-comparison plots
#
# This module is intentionally matplotlib-only.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================


@dataclass(frozen=True)
class PlotConfig:
    figure_dpi: int = 140
    save_dpi: int = 180

    figsize_wide: tuple[float, float] = (11.0, 6.0)
    figsize_square: tuple[float, float] = (8.0, 8.0)
    figsize_heatmap: tuple[float, float] = (9.0, 7.0)
    figsize_scatter: tuple[float, float] = (8.5, 6.5)

    heatmap_cmap: str = "Blues"
    annotate_heatmap: bool = True
    rotation_x: int = 30
    alpha: float = 0.85
    close_after_save: bool = True
    histogram_bins: int = 30

    scatter_use_log1p: bool = True


# ============================================================
# Basic helpers
# ============================================================


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    config: Optional[PlotConfig] = None,
) -> Path:
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


def _validate_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _plot_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    config: Optional[PlotConfig] = None,
    fmt: str = ".2f",
    colorbar_label: str = "value",
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
    cbar.ax.set_ylabel(colorbar_label, rotation=90)

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


def _pivot_pairwise_metric(
    pairwise_df: pd.DataFrame,
    *,
    metric_col: str,
    row_col: str = "source_chain",
    col_col: str = "target_chain",
) -> pd.DataFrame:
    required = {row_col, col_col, metric_col}
    _validate_columns(pairwise_df, required, "pairwise_df")

    df = pairwise_df.copy()
    df = _coerce_numeric(df, [metric_col])

    matrix = df.pivot_table(
        index=row_col,
        columns=col_col,
        values=metric_col,
        aggfunc="first",
    )

    if matrix.empty:
        return matrix

    row_order = sorted(matrix.index.astype(str).tolist())
    col_order = sorted(matrix.columns.astype(str).tolist())
    matrix = matrix.reindex(index=row_order, columns=col_order)
    return matrix


# ============================================================
# Chain count plots
# ============================================================


def build_chain_status_count_table(
    status_tables_by_chain: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Builds a compact table:
    chain, present_count, active_count, passive_count
    """
    rows = []
    for chain_name, df in status_tables_by_chain.items():
        if df.empty:
            rows.append(
                {
                    "chain": chain_name,
                    "present_count": 0,
                    "active_count": 0,
                    "passive_count": 0,
                }
            )
            continue

        _validate_columns(
            df,
            ["is_present", "is_active", "is_passive"],
            f"status table for chain={chain_name}",
        )

        rows.append(
            {
                "chain": chain_name,
                "present_count": int(df["is_present"].fillna(False).sum()),
                "active_count": int(df["is_active"].fillna(False).sum()),
                "passive_count": int(df["is_passive"].fillna(False).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("chain").reset_index(drop=True)


def plot_chain_status_counts(
    chain_counts_df: pd.DataFrame,
    *,
    title: str = "Wallet counts by chain",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()
    _validate_columns(
        chain_counts_df,
        ["chain", "present_count", "active_count", "passive_count"],
        "chain_counts_df",
    )

    df = chain_counts_df.copy()
    df = _coerce_numeric(df, ["present_count", "active_count", "passive_count"])
    df = df.sort_values("chain").reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)

    bars_present = ax.bar(x - width, df["present_count"], width=width, alpha=config.alpha, label="present")
    bars_active = ax.bar(x, df["active_count"], width=width, alpha=config.alpha, label="active")
    bars_passive = ax.bar(x + width, df["passive_count"], width=width, alpha=config.alpha, label="passive")

    ax.set_title(title)
    ax.set_xlabel("Chain")
    ax.set_ylabel("Wallet count")
    ax.set_xticks(x)
    ax.set_xticklabels(df["chain"], rotation=config.rotation_x, ha="right")
    ax.legend()

    for bars in (bars_present, bars_active, bars_passive):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    return fig, ax


def plot_overlap_count_bars(
    pairwise_df: pd.DataFrame,
    *,
    metric_col: str = "intersection_size",
    title: str = "Pairwise overlap counts",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()
    _validate_columns(
        pairwise_df,
        ["source_chain", "target_chain", metric_col],
        "pairwise_df",
    )

    df = pairwise_df.copy()
    df = _coerce_numeric(df, [metric_col])

    df["pair"] = df["source_chain"].astype(str) + " → " + df["target_chain"].astype(str)
    df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    bars = ax.bar(df["pair"], df[metric_col], alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel("Pair")
    ax.set_ylabel(metric_col)
    ax.tick_params(axis="x", rotation=config.rotation_x)

    for bar, value in zip(bars, df[metric_col].fillna(0).tolist()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig, ax


# ============================================================
# Overlap / Jaccard heatmaps
# ============================================================


def plot_pairwise_overlap_heatmap(
    pairwise_df: pd.DataFrame,
    *,
    metric_col: str = "overlap_of_source",
    title: str = "Pairwise overlap heatmap",
    xlabel: str = "Target chain",
    ylabel: str = "Source chain",
    colorbar_label: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix = _pivot_pairwise_metric(pairwise_df, metric_col=metric_col)

    return _plot_heatmap(
        matrix,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        config=config,
        fmt=".2f",
        colorbar_label=colorbar_label or metric_col,
    )


def plot_pairwise_jaccard_heatmap(
    pairwise_df: pd.DataFrame,
    *,
    title: str = "Pairwise Jaccard similarity",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    return plot_pairwise_overlap_heatmap(
        pairwise_df,
        metric_col="jaccard",
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        colorbar_label="Jaccard",
        config=config,
    )


def plot_pairwise_intersection_heatmap(
    pairwise_df: pd.DataFrame,
    *,
    title: str = "Pairwise overlap count",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix = _pivot_pairwise_metric(pairwise_df, metric_col="intersection_size")
    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Target chain",
        ylabel="Source chain",
        config=config,
        fmt=".0f",
        colorbar_label="count",
    )


# ============================================================
# Feature-alignment plots
# ============================================================


def plot_first_activity_delta_histogram(
    alignment_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    bins: Optional[int] = None,
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    col = "first_activity_delta_seconds"
    _validate_columns(alignment_df, [col], "alignment_df")

    df = _coerce_numeric(alignment_df, [col])
    series = df[col].dropna()

    if series.empty:
        raise ValueError("No non-null first_activity_delta_seconds values to plot")

    if bins is None:
        bins = config.histogram_bins
    if title is None:
        title = f"Δ first activity: {chain_a} vs {chain_b}"

    hours = series / 3600.0

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    ax.hist(hours, bins=bins, alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel("First activity delta (hours)")
    ax.set_ylabel("Frequency")

    median_hours = float(hours.median())
    ax.axvline(median_hours, linestyle="--", linewidth=1.3)
    ax.text(
        median_hours,
        ax.get_ylim()[1] * 0.9,
        f"median={median_hours:.2f}h",
        rotation=90,
        va="top",
        ha="right",
    )

    fig.tight_layout()
    return fig, ax


def plot_value_sent_scatter(
    alignment_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    use_log1p: Optional[bool] = None,
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    x_col = f"{chain_a}_value_sent_wei"
    y_col = f"{chain_b}_value_sent_wei"
    _validate_columns(alignment_df, [x_col, y_col], "alignment_df")

    df = _coerce_numeric(alignment_df, [x_col, y_col]).dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        raise ValueError("No aligned rows with non-null sent value columns")

    if use_log1p is None:
        use_log1p = config.scatter_use_log1p

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if use_log1p:
        x = np.log1p(x)
        y = np.log1p(y)

    if title is None:
        title = f"Sent value comparison: {chain_a} vs {chain_b}"

    fig, ax = plt.subplots(figsize=config.figsize_scatter, dpi=config.figure_dpi)
    ax.scatter(x, y, alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel(f"log1p({x_col})" if use_log1p else x_col)
    ax.set_ylabel(f"log1p({y_col})" if use_log1p else y_col)

    if len(x) >= 2 and len(y) >= 2:
        coeff = np.polyfit(x, y, deg=1)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        line_y = coeff[0] * line_x + coeff[1]
        ax.plot(line_x, line_y, linewidth=1.5)

    fig.tight_layout()
    return fig, ax


def plot_frequency_scatter(
    alignment_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    x_col = f"{chain_a}_tx_frequency_per_day"
    y_col = f"{chain_b}_tx_frequency_per_day"
    _validate_columns(alignment_df, [x_col, y_col], "alignment_df")

    df = _coerce_numeric(alignment_df, [x_col, y_col]).dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        raise ValueError("No aligned rows with non-null daily frequency columns")

    if title is None:
        title = f"Transaction frequency comparison: {chain_a} vs {chain_b}"

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=config.figsize_scatter, dpi=config.figure_dpi)
    ax.scatter(x, y, alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if len(x) >= 2 and len(y) >= 2:
        coeff = np.polyfit(x, y, deg=1)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        line_y = coeff[0] * line_x + coeff[1]
        ax.plot(line_x, line_y, linewidth=1.5)

    fig.tight_layout()
    return fig, ax


# ============================================================
# Correlation heatmaps
# ============================================================


def plot_pairwise_correlation_heatmap(
    correlation_df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    row_col: str = "chain_a",
    col_col: str = "chain_b",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Expected input shape, for example:
    chain_a | chain_b | pearson_r_value_sent | spearman_r_value_sent | ...
    """
    matrix = _pivot_pairwise_metric(
        correlation_df,
        metric_col=metric_col,
        row_col=row_col,
        col_col=col_col,
    )

    # make it symmetric if only upper-triangular pairs were provided
    if not matrix.empty and set(matrix.index) == set(matrix.columns):
        symmetric = matrix.copy()
        for i in symmetric.index:
            for j in symmetric.columns:
                if pd.isna(symmetric.loc[i, j]) and j in symmetric.index and i in symmetric.columns:
                    symmetric.loc[i, j] = symmetric.loc[j, i]
        for chain in symmetric.index:
            if chain in symmetric.columns and pd.isna(symmetric.loc[chain, chain]):
                symmetric.loc[chain, chain] = 1.0
        matrix = symmetric

    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        config=config,
        fmt=".2f",
        colorbar_label=metric_col,
    )


# ============================================================
# Window-comparison plots
# ============================================================


def plot_window_comparison_bars(
    summary_df: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    title: str,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Example inputs:
    window_blocks | chain | active_addresses
    """
    config = config or PlotConfig()
    _validate_columns(summary_df, ["window_blocks", category_col, value_col], "summary_df")

    df = summary_df.copy()
    df = _coerce_numeric(df, ["window_blocks", value_col])
    df[category_col] = df[category_col].astype(str)

    windows = sorted(df["window_blocks"].dropna().unique().tolist())
    categories = sorted(df[category_col].dropna().unique().tolist())

    x = np.arange(len(windows))
    width = 0.8 / max(len(categories), 1)

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)

    for idx, category in enumerate(categories):
        subset = (
            df[df[category_col] == category]
            .set_index("window_blocks")
            .reindex(windows)
        )
        y = subset[value_col].fillna(0).to_numpy(dtype=float)

        shift = (idx - (len(categories) - 1) / 2.0) * width
        bars = ax.bar(x + shift, y, width=width, alpha=config.alpha, label=category)

        for bar, value in zip(bars, y):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("Window size (blocks)")
    ax.set_ylabel(value_col)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(w)) for w in windows])
    ax.legend()

    fig.tight_layout()
    return fig, ax


# ============================================================
# Optional table rendering plot
# ============================================================


def plot_summary_table(
    summary_df: pd.DataFrame,
    *,
    title: str = "Summary table",
    max_rows: int = 20,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()

    if summary_df.empty:
        raise ValueError("Cannot render an empty summary table")

    df = summary_df.head(max_rows).copy()

    fig_height = max(2.5, 0.35 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=config.figure_dpi)
    ax.axis("off")
    ax.set_title(title, pad=12)

    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns.astype(str).tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    fig.tight_layout()
    return fig, ax