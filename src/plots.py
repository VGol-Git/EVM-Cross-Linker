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
# - disk-backed plot input loading
#
# This module is intentionally matplotlib-only.

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classify import (
    address_status_table_path_base,
    pairwise_overlap_table_path_base,
)
from .config import AppConfig, ChainConfig
from .features import (
    address_feature_table_path_base,
    overlapping_feature_table_path_base,
    pairwise_feature_alignment_path_base,
    window_feature_summary_path_base,
)
from .sampling import read_table
from .stats import (
    daily_series_stats_table_path_base,
    feature_stats_table_path_base,
    presence_stats_table_path_base,
    window_stats_summary_path_base,
)

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
    pvalue_floor: float = 1e-12

    # --------------------------------------------------------
    # Global font controls
    # --------------------------------------------------------
    font_family: Optional[str] = None
    base_font_size: float = 14.0

    title_font_scale: float = 1.18
    axis_label_font_scale: float = 1.00
    tick_font_scale: float = 0.95
    legend_font_scale: float = 0.95
    annotation_font_scale: float = 0.82
    table_font_scale: float = 0.82
    colorbar_font_scale: float = 0.95


# ============================================================
# Basic helpers
# ============================================================


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _font_size(base: float, scale: float) -> float:
    return max(1.0, float(base) * float(scale))


def _apply_global_font_config(config: PlotConfig) -> None:
    """
    Apply global matplotlib text style for the current plotting session.
    The main knob is `base_font_size`.
    """
    plt.rcParams["font.size"] = config.base_font_size
    plt.rcParams["axes.titlesize"] = _font_size(config.base_font_size, config.title_font_scale)
    plt.rcParams["axes.labelsize"] = _font_size(config.base_font_size, config.axis_label_font_scale)
    plt.rcParams["xtick.labelsize"] = _font_size(config.base_font_size, config.tick_font_scale)
    plt.rcParams["ytick.labelsize"] = _font_size(config.base_font_size, config.tick_font_scale)
    plt.rcParams["legend.fontsize"] = _font_size(config.base_font_size, config.legend_font_scale)

    if config.font_family:
        plt.rcParams["font.family"] = config.font_family


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


def _load_table_as_dataframe(
    path_without_suffix: Path,
    table_format: str,
) -> pd.DataFrame:
    rows = read_table(path_without_suffix=path_without_suffix, table_format=table_format)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "nan"}:
            return None
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


def _to_bool(value: Any) -> bool:
    parsed = _to_optional_bool(value)
    return bool(parsed) if parsed is not None else False


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_to_bool).fillna(False).astype(bool)


def _coerce_status_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["is_present", "is_active", "is_passive", "seen_as_from", "seen_as_to"]:
        if col in out.columns:
            out[col] = _coerce_bool_series(out[col])
    return out


def _coerce_feature_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["is_present", "is_active", "is_passive"]:
        if col in out.columns:
            out[col] = _coerce_bool_series(out[col])
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
    _apply_global_font_config(config)

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
    cbar.ax.set_ylabel(
        colorbar_label,
        rotation=90,
        fontsize=_font_size(config.base_font_size, config.colorbar_font_scale),
    )
    cbar.ax.tick_params(labelsize=_font_size(config.base_font_size, config.tick_font_scale))

    if config.annotate_heatmap:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    format(values[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=_font_size(config.base_font_size, config.annotation_font_scale),
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


def _symmetrize_if_square(matrix: pd.DataFrame, diagonal_value: Optional[float] = None) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    if set(matrix.index) != set(matrix.columns):
        return matrix

    symmetric = matrix.copy()
    for i in symmetric.index:
        for j in symmetric.columns:
            if pd.isna(symmetric.loc[i, j]) and j in symmetric.index and i in symmetric.columns:
                symmetric.loc[i, j] = symmetric.loc[j, i]
    if diagonal_value is not None:
        for i in symmetric.index:
            if i in symmetric.columns:
                symmetric.loc[i, i] = diagonal_value
    return symmetric


def _filter_feature_stats_for_metric(
    feature_stats_df: pd.DataFrame,
    *,
    feature_name: str,
    metric_col: str,
) -> pd.DataFrame:
    if feature_stats_df.empty:
        return pd.DataFrame()

    required = {"chain_a", "chain_b", "feature_a", "feature_b", metric_col}
    _validate_columns(feature_stats_df, required, "feature_stats_df")

    df = feature_stats_df.copy()
    df = df[
        (df["feature_a"].astype(str) == feature_name)
        & (df["feature_b"].astype(str) == feature_name)
    ].copy()

    if df.empty:
        return df

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    return df


def _default_plot_output_dir(
    app_config: AppConfig,
    window_blocks: int,
) -> Path:
    out = (
        app_config.paths.outputs_dir
        / "pipeline"
        / "plots"
        / f"window_{window_blocks}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# Disk loading helpers
# ============================================================


def load_status_table_for_chain_window(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> pd.DataFrame:
    df = _load_table_as_dataframe(
        path_without_suffix=address_status_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )
    return _coerce_status_boolean_columns(df)


def load_feature_table_for_chain_window(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> pd.DataFrame:
    df = _load_table_as_dataframe(
        path_without_suffix=address_feature_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )
    return _coerce_feature_boolean_columns(df)


def load_pairwise_overlap_table_for_window(
    app_config: AppConfig,
    window_blocks: int,
    overlap_name: str,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=pairwise_overlap_table_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
            overlap_name=overlap_name,
        ),
        table_format=app_config.storage.table_format,
    )


def load_overlapping_feature_table_for_window(
    app_config: AppConfig,
    window_blocks: int,
) -> pd.DataFrame:
    df = _load_table_as_dataframe(
        path_without_suffix=overlapping_feature_table_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )
    return _coerce_feature_boolean_columns(df)


def load_pairwise_feature_alignment_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    chain_a: str,
    chain_b: str,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=pairwise_feature_alignment_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
            chain_a=chain_a,
            chain_b=chain_b,
        ),
        table_format=app_config.storage.table_format,
    )


def load_window_feature_summary(
    app_config: AppConfig,
    window_blocks: int,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=window_feature_summary_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


def load_presence_stats_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    status_name: str,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=presence_stats_table_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
            status_name=status_name,
        ),
        table_format=app_config.storage.table_format,
    )


def load_feature_stats_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
) -> pd.DataFrame:
    frames = []
    chain_names = [chain.name for chain in app_config.enabled_chains]
    for chain_a, chain_b in combinations(chain_names, 2):
        df = _load_table_as_dataframe(
            path_without_suffix=feature_stats_table_path_base(
                app_config=app_config,
                window_blocks=window_blocks,
                chain_a=chain_a,
                chain_b=chain_b,
            ),
            table_format=app_config.storage.table_format,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=0, ignore_index=True)


def load_daily_series_stats_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
) -> pd.DataFrame:
    frames = []
    chain_names = [chain.name for chain in app_config.enabled_chains]
    for chain_a, chain_b in combinations(chain_names, 2):
        df = _load_table_as_dataframe(
            path_without_suffix=daily_series_stats_table_path_base(
                app_config=app_config,
                window_blocks=window_blocks,
                chain_a=chain_a,
                chain_b=chain_b,
            ),
            table_format=app_config.storage.table_format,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=0, ignore_index=True)


def load_window_stats_summary(
    app_config: AppConfig,
    window_blocks: int,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=window_stats_summary_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


def load_plot_inputs_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
) -> Dict[str, Any]:
    status_tables_by_chain: Dict[str, pd.DataFrame] = {}
    feature_tables_by_chain: Dict[str, pd.DataFrame] = {}
    pairwise_alignments: Dict[Tuple[str, str], pd.DataFrame] = {}

    for chain in app_config.enabled_chains:
        status_tables_by_chain[chain.name] = load_status_table_for_chain_window(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        )
        feature_tables_by_chain[chain.name] = load_feature_table_for_chain_window(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        )

    chain_names = [chain.name for chain in app_config.enabled_chains]
    for chain_a, chain_b in combinations(chain_names, 2):
        pairwise_alignments[(chain_a, chain_b)] = load_pairwise_feature_alignment_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            chain_a=chain_a,
            chain_b=chain_b,
        )

    matching = {
        "present_pairwise": load_pairwise_overlap_table_for_window(
            app_config, window_blocks, "present"
        ),
        "active_pairwise": load_pairwise_overlap_table_for_window(
            app_config, window_blocks, "active"
        ),
        "passive_pairwise": load_pairwise_overlap_table_for_window(
            app_config, window_blocks, "passive"
        ),
        "mixed_active_passive": load_pairwise_overlap_table_for_window(
            app_config, window_blocks, "mixed_active_passive"
        ),
    }

    presence_stats = {
        "present": load_presence_stats_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            status_name="present",
        ),
        "active": load_presence_stats_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            status_name="active",
        ),
        "passive": load_presence_stats_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            status_name="passive",
        ),
    }

    stats_outputs = {
        "presence": presence_stats,
        "features": {
            "feature_stats": load_feature_stats_for_window(
                app_config=app_config,
                window_blocks=window_blocks,
            ),
            "daily_series_stats": load_daily_series_stats_for_window(
                app_config=app_config,
                window_blocks=window_blocks,
            ),
        },
        "summary": load_window_stats_summary(
            app_config=app_config,
            window_blocks=window_blocks,
        ),
    }

    feature_outputs = {
        "feature_tables_by_chain": feature_tables_by_chain,
        "overlapping_feature_table": load_overlapping_feature_table_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
        ),
        "pairwise_alignments": pairwise_alignments,
        "summary": load_window_feature_summary(
            app_config=app_config,
            window_blocks=window_blocks,
        ),
    }

    classification_outputs = {
        "status_tables": status_tables_by_chain,
        "matching": matching,
    }

    return {
        "classification": classification_outputs,
        "features": feature_outputs,
        "stats": stats_outputs,
    }


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
                "present_count": int(_coerce_bool_series(df["is_present"]).sum()),
                "active_count": int(_coerce_bool_series(df["is_active"]).sum()),
                "passive_count": int(_coerce_bool_series(df["is_passive"]).sum()),
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
    _apply_global_font_config(config)

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
    ax.legend(fontsize=_font_size(config.base_font_size, config.legend_font_scale))

    for bars in (bars_present, bars_active, bars_passive):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=_font_size(config.base_font_size, config.annotation_font_scale),
            )

    fig.tight_layout()
    return fig, ax


def plot_chain_status_shares(
    chain_counts_df: pd.DataFrame,
    *,
    title: str = "Wallet shares by chain",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()
    _apply_global_font_config(config)

    _validate_columns(
        chain_counts_df,
        ["chain", "present_count", "active_count", "passive_count"],
        "chain_counts_df",
    )

    df = chain_counts_df.copy()
    df = _coerce_numeric(df, ["present_count", "active_count", "passive_count"])
    df = df.sort_values("chain").reset_index(drop=True)

    present = df["present_count"].replace(0, np.nan)
    active_share = (df["active_count"] / present).fillna(0.0)
    passive_share = (df["passive_count"] / present).fillna(0.0)
    other_share = np.maximum(0.0, 1.0 - active_share - passive_share)

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    ax.bar(df["chain"], active_share, alpha=config.alpha, label="active share")
    ax.bar(df["chain"], passive_share, bottom=active_share, alpha=config.alpha, label="passive share")
    ax.bar(df["chain"], other_share, bottom=active_share + passive_share, alpha=config.alpha, label="other present")

    ax.set_title(title)
    ax.set_xlabel("Chain")
    ax.set_ylabel("Share of present wallets")
    ax.tick_params(axis="x", rotation=config.rotation_x)
    ax.legend(fontsize=_font_size(config.base_font_size, config.legend_font_scale))

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
    _apply_global_font_config(config)

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
            fontsize=_font_size(config.base_font_size, config.annotation_font_scale),
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
    matrix = _pivot_pairwise_metric(pairwise_df, metric_col="jaccard")
    matrix = _symmetrize_if_square(matrix, diagonal_value=1.0)

    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        config=config,
        fmt=".2f",
        colorbar_label="Jaccard",
    )


def plot_pairwise_intersection_heatmap(
    pairwise_df: pd.DataFrame,
    *,
    title: str = "Pairwise overlap count",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix = _pivot_pairwise_metric(pairwise_df, metric_col="intersection_size")
    matrix = _symmetrize_if_square(matrix, diagonal_value=np.nan)
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
    _apply_global_font_config(config)

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
        fontsize=_font_size(config.base_font_size, config.annotation_font_scale),
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
    _apply_global_font_config(config)

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
    _apply_global_font_config(config)

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


def plot_overlap_network_count_histogram(
    overlapping_df: pd.DataFrame,
    *,
    title: str = "Distribution of present_network_count",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()
    _apply_global_font_config(config)

    _validate_columns(overlapping_df, ["address", "present_network_count"], "overlapping_df")

    df = overlapping_df[["address", "present_network_count"]].drop_duplicates().copy()
    df = _coerce_numeric(df, ["present_network_count"]).dropna(subset=["present_network_count"])

    if df.empty:
        raise ValueError("No overlapping addresses with present_network_count to plot")

    fig, ax = plt.subplots(figsize=config.figsize_wide, dpi=config.figure_dpi)
    bins = sorted(df["present_network_count"].astype(int).unique().tolist())
    ax.hist(df["present_network_count"], bins=np.arange(min(bins), max(bins) + 2) - 0.5, alpha=config.alpha)

    ax.set_title(title)
    ax.set_xlabel("Number of networks where address is present")
    ax.set_ylabel("Address count")
    fig.tight_layout()
    return fig, ax


# ============================================================
# Correlation / significance heatmaps
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
    matrix = _pivot_pairwise_metric(
        correlation_df,
        metric_col=metric_col,
        row_col=row_col,
        col_col=col_col,
    )
    matrix = _symmetrize_if_square(matrix, diagonal_value=1.0)

    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        config=config,
        fmt=".2f",
        colorbar_label=metric_col,
    )


def plot_pairwise_pvalue_heatmap(
    stats_df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    row_col: str = "chain_a",
    col_col: str = "chain_b",
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()
    _validate_columns(stats_df, [row_col, col_col, metric_col], "stats_df")

    df = stats_df.copy()
    df = _coerce_numeric(df, [metric_col])
    df = df.dropna(subset=[metric_col]).copy()

    if df.empty:
        raise ValueError(f"No non-null values for {metric_col}")

    df[metric_col] = df[metric_col].clip(lower=config.pvalue_floor)
    df["neg_log10_p"] = -np.log10(df[metric_col])

    matrix = _pivot_pairwise_metric(
        df,
        metric_col="neg_log10_p",
        row_col=row_col,
        col_col=col_col,
    )
    matrix = _symmetrize_if_square(matrix, diagonal_value=np.nan)

    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        config=config,
        fmt=".2f",
        colorbar_label="-log10(p)",
    )


def plot_pairwise_feature_sample_size_heatmap(
    feature_stats_df: pd.DataFrame,
    *,
    feature_name: str,
    title: str,
    config: Optional[PlotConfig] = None,
) -> tuple[plt.Figure, plt.Axes]:
    df = _filter_feature_stats_for_metric(
        feature_stats_df,
        feature_name=feature_name,
        metric_col="sample_size",
    )
    if df.empty:
        raise ValueError(f"No feature stats for feature_name={feature_name!r}")

    matrix = _pivot_pairwise_metric(
        df,
        metric_col="sample_size",
        row_col="chain_a",
        col_col="chain_b",
    )
    matrix = _symmetrize_if_square(matrix, diagonal_value=np.nan)

    return _plot_heatmap(
        matrix,
        title=title,
        xlabel="Chain",
        ylabel="Chain",
        config=config,
        fmt=".0f",
        colorbar_label="sample size",
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
    config = config or PlotConfig()
    _apply_global_font_config(config)

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
                fontsize=_font_size(config.base_font_size, config.annotation_font_scale),
            )

    ax.set_title(title)
    ax.set_xlabel("Window size (blocks)")
    ax.set_ylabel(value_col)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(w)) for w in windows])
    ax.legend(fontsize=_font_size(config.base_font_size, config.legend_font_scale))

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
    _apply_global_font_config(config)

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
    table.set_fontsize(_font_size(config.base_font_size, config.table_font_scale))
    table.scale(1.0, 1.2)

    fig.tight_layout()
    return fig, ax


# ============================================================
# Stage-level rendering from disk
# ============================================================


def render_plots_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    classification_outputs_for_window: Dict[str, Any],
    feature_outputs_for_window: Dict[str, Any],
    stats_outputs_for_window: Dict[str, Any],
    plot_config: Optional[PlotConfig] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    plot_config = plot_config or PlotConfig()
    output_dir = output_dir or _default_plot_output_dir(app_config, window_blocks)
    saved_paths: Dict[str, str] = {}

    status_tables_by_chain = classification_outputs_for_window.get("status_tables", {})
    matching = classification_outputs_for_window.get("matching", {})
    pairwise_alignments = feature_outputs_for_window.get("pairwise_alignments", {})
    overlapping_df = feature_outputs_for_window.get("overlapping_feature_table", pd.DataFrame())
    feature_stats_df = stats_outputs_for_window.get("features", {}).get("feature_stats", pd.DataFrame())
    presence_stats = stats_outputs_for_window.get("presence", {})
    summary_df = stats_outputs_for_window.get("summary", pd.DataFrame())

    # 1. Chain counts
    chain_counts_df = build_chain_status_count_table(status_tables_by_chain)

    fig, _ = plot_chain_status_counts(
        chain_counts_df,
        title=f"Wallet counts by chain (window={window_blocks} blocks)",
        config=plot_config,
    )
    path = save_figure(fig, output_dir / "chain_status_counts.png", plot_config)
    saved_paths["chain_status_counts"] = str(path)

    fig, _ = plot_chain_status_shares(
        chain_counts_df,
        title=f"Wallet shares by chain (window={window_blocks} blocks)",
        config=plot_config,
    )
    path = save_figure(fig, output_dir / "chain_status_shares.png", plot_config)
    saved_paths["chain_status_shares"] = str(path)

    # 2. Overlap count bars and heatmaps
    pairwise_plot_specs = [
        ("present_pairwise", "present_intersection_heatmap", "Present overlap count"),
        ("active_pairwise", "active_intersection_heatmap", "Active overlap count"),
        ("passive_pairwise", "passive_intersection_heatmap", "Passive overlap count"),
    ]

    for key, out_name, title in pairwise_plot_specs:
        df = matching.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue

        try:
            fig, _ = plot_pairwise_intersection_heatmap(
                df,
                title=f"{title} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}.png", plot_config)
            saved_paths[out_name] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip %s: %s",
                window_blocks,
                out_name,
                exc,
            )

        try:
            fig, _ = plot_overlap_count_bars(
                df,
                title=f"{title} bars (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}_bars.png", plot_config)
            saved_paths[f"{out_name}_bars"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip %s bars: %s",
                window_blocks,
                out_name,
                exc,
            )

    jaccard_plot_specs = [
        ("present_pairwise", "present_jaccard_heatmap", "Present Jaccard similarity"),
        ("active_pairwise", "active_jaccard_heatmap", "Active Jaccard similarity"),
        ("passive_pairwise", "passive_jaccard_heatmap", "Passive Jaccard similarity"),
        ("mixed_active_passive", "mixed_active_passive_jaccard_heatmap", "Active vs Passive Jaccard similarity"),
    ]

    for key, out_name, title in jaccard_plot_specs:
        df = matching.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        try:
            fig, _ = plot_pairwise_jaccard_heatmap(
                df,
                title=f"{title} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}.png", plot_config)
            saved_paths[out_name] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip %s: %s",
                window_blocks,
                out_name,
                exc,
            )

    # 3. Overlap distribution across networks
    if overlapping_df is not None and not overlapping_df.empty:
        try:
            fig, _ = plot_overlap_network_count_histogram(
                overlapping_df,
                title=f"present_network_count distribution (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / "overlap_network_count_histogram.png", plot_config)
            saved_paths["overlap_network_count_histogram"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip overlap network histogram: %s",
                window_blocks,
                exc,
            )

    # 4. Feature-alignment plots per chain pair
    for (chain_a, chain_b), alignment_df in pairwise_alignments.items():
        if alignment_df is None or alignment_df.empty:
            continue

        pair_prefix = f"{chain_a}_vs_{chain_b}"

        try:
            fig, _ = plot_value_sent_scatter(
                alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                title=f"Sent value: {chain_a} vs {chain_b} ({window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{pair_prefix}_value_sent_scatter.png", plot_config)
            saved_paths[f"{pair_prefix}_value_sent_scatter"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip value scatter for %s vs %s: %s",
                window_blocks,
                chain_a,
                chain_b,
                exc,
            )

        try:
            fig, _ = plot_frequency_scatter(
                alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                title=f"Tx/day: {chain_a} vs {chain_b} ({window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{pair_prefix}_tx_frequency_scatter.png", plot_config)
            saved_paths[f"{pair_prefix}_tx_frequency_scatter"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip frequency scatter for %s vs %s: %s",
                window_blocks,
                chain_a,
                chain_b,
                exc,
            )

        try:
            fig, _ = plot_first_activity_delta_histogram(
                alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                title=f"Δ first activity: {chain_a} vs {chain_b} ({window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{pair_prefix}_first_activity_delta_hist.png", plot_config)
            saved_paths[f"{pair_prefix}_first_activity_delta_hist"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip first-activity histogram for %s vs %s: %s",
                window_blocks,
                chain_a,
                chain_b,
                exc,
            )

    # 5. Correlation heatmaps
    correlation_specs = [
        ("value_sent_wei", "pearson_r", "value_sent_pearson_heatmap", "Pearson r: sent value"),
        ("value_sent_wei", "spearman_r", "value_sent_spearman_heatmap", "Spearman r: sent value"),
        ("tx_frequency_per_day", "pearson_r", "tx_frequency_pearson_heatmap", "Pearson r: tx/day"),
        ("tx_frequency_per_day", "spearman_r", "tx_frequency_spearman_heatmap", "Spearman r: tx/day"),
        ("unique_counterparties", "pearson_r", "counterparties_pearson_heatmap", "Pearson r: counterparties"),
        ("unique_counterparties", "spearman_r", "counterparties_spearman_heatmap", "Spearman r: counterparties"),
    ]

    for feature_name, metric_col, out_name, title in correlation_specs:
        df = _filter_feature_stats_for_metric(
            feature_stats_df,
            feature_name=feature_name,
            metric_col=metric_col,
        )
        if df.empty:
            continue

        try:
            fig, _ = plot_pairwise_correlation_heatmap(
                df,
                metric_col=metric_col,
                title=f"{title} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}.png", plot_config)
            saved_paths[out_name] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip correlation heatmap %s: %s",
                window_blocks,
                out_name,
                exc,
            )

    # 6. New: p-value heatmaps from presence stats
    for status_name, df in presence_stats.items():
        if df is None or df.empty:
            continue

        if "fisher_p_value" in df.columns:
            try:
                fig, _ = plot_pairwise_pvalue_heatmap(
                    df,
                    metric_col="fisher_p_value",
                    title=f"-log10 Fisher p-value ({status_name}, window={window_blocks} blocks)",
                    config=plot_config,
                )
                path = save_figure(fig, output_dir / f"{status_name}_fisher_pvalue_heatmap.png", plot_config)
                saved_paths[f"{status_name}_fisher_pvalue_heatmap"] = str(path)
            except ValueError as exc:
                logger.warning(
                    "[plots][window=%s] Skip %s fisher p-value heatmap: %s",
                    window_blocks,
                    status_name,
                    exc,
                )

        if "empirical_p_value" in df.columns:
            try:
                fig, _ = plot_pairwise_pvalue_heatmap(
                    df,
                    metric_col="empirical_p_value",
                    title=f"-log10 empirical permutation p-value ({status_name}, window={window_blocks} blocks)",
                    config=plot_config,
                )
                path = save_figure(fig, output_dir / f"{status_name}_empirical_pvalue_heatmap.png", plot_config)
                saved_paths[f"{status_name}_empirical_pvalue_heatmap"] = str(path)
            except ValueError as exc:
                logger.warning(
                    "[plots][window=%s] Skip %s empirical p-value heatmap: %s",
                    window_blocks,
                    status_name,
                    exc,
                )

    # 7. New: sample size heatmaps for feature correlations
    for feature_name in ("value_sent_wei", "tx_frequency_per_day"):
        try:
            fig, _ = plot_pairwise_feature_sample_size_heatmap(
                feature_stats_df,
                feature_name=feature_name,
                title=f"Feature correlation sample size: {feature_name} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{feature_name}_sample_size_heatmap.png", plot_config)
            saved_paths[f"{feature_name}_sample_size_heatmap"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[plots][window=%s] Skip sample size heatmap for %s: %s",
                window_blocks,
                feature_name,
                exc,
            )

    # 8. Summary table plot
    if summary_df is not None and not summary_df.empty:
        fig, _ = plot_summary_table(
            summary_df,
            title=f"Statistical summary (window={window_blocks} blocks)",
            max_rows=20,
            config=plot_config,
        )
        path = save_figure(fig, output_dir / "stats_summary_table.png", plot_config)
        saved_paths["stats_summary_table"] = str(path)

    return saved_paths


def run_plots_for_window_from_disk(
    app_config: AppConfig,
    *,
    window_blocks: int,
    plot_config: Optional[PlotConfig] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    disk_inputs = load_plot_inputs_for_window(
        app_config=app_config,
        window_blocks=window_blocks,
    )

    logger.info(
        "[plots][window=%s] Loaded plot artifacts from disk: status_tables=%s pairwise_alignments=%s",
        window_blocks,
        len(disk_inputs["classification"].get("status_tables", {})),
        len(disk_inputs["features"].get("pairwise_alignments", {})),
    )

    return render_plots_for_window(
        app_config=app_config,
        window_blocks=window_blocks,
        classification_outputs_for_window=disk_inputs["classification"],
        feature_outputs_for_window=disk_inputs["features"],
        stats_outputs_for_window=disk_inputs["stats"],
        plot_config=plot_config,
        output_dir=output_dir,
    )


def run_plot_stage_from_disk(
    app_config: AppConfig,
    *,
    plot_config: Optional[PlotConfig] = None,
) -> Dict[int, Dict[str, str]]:
    plot_config = plot_config or PlotConfig()

    outputs: Dict[int, Dict[str, str]] = {}
    for window_blocks in app_config.sampling.windows.block_counts:
        outputs[window_blocks] = run_plots_for_window_from_disk(
            app_config=app_config,
            window_blocks=window_blocks,
            plot_config=plot_config,
        )

    return outputs