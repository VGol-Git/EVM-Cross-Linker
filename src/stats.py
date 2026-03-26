# stats.py
# Statistical testing layer for the block-window EVM cross-chain correlation project.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, pearsonr, spearmanr

from .classify import presence_matrix_path_base
from .config import AppConfig
from .features import pairwise_feature_alignment_path_base
from .sampling import read_table, write_table

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================


@dataclass(frozen=True)
class StatsConfig:
    n_permutations: int = 1000
    random_state: int = 42
    alpha: float = 0.05

    # Whether to drop rows with missing values for pairwise feature correlations.
    dropna_for_correlations: bool = True

    # Minimum number of observations to compute a correlation.
    min_samples_for_correlation: int = 3


# ============================================================
# IO helpers
# ============================================================


def _load_table_as_dataframe(
    path_without_suffix: Path,
    table_format: str,
) -> pd.DataFrame:
    rows = read_table(path_without_suffix=path_without_suffix, table_format=table_format)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _save_dataframe(
    df: pd.DataFrame,
    path_without_suffix: Path,
    table_format: str,
) -> Path:
    rows = df.to_dict(orient="records")
    return write_table(
        rows=rows,
        path_without_suffix=path_without_suffix,
        table_format=table_format,
    )


# ============================================================
# Path helpers
# ============================================================


def presence_stats_table_path_base(
    app_config: AppConfig,
    window_blocks: int,
    status_name: str,
) -> Path:
    return (
        app_config.paths.processed_statistics_dir
        / f"window_{window_blocks}_{status_name}_presence_stats"
    )


def feature_stats_table_path_base(
    app_config: AppConfig,
    window_blocks: int,
    chain_a: str,
    chain_b: str,
) -> Path:
    return (
        app_config.paths.processed_statistics_dir
        / f"window_{window_blocks}_{chain_a}_vs_{chain_b}_feature_stats"
    )


def daily_series_stats_table_path_base(
    app_config: AppConfig,
    window_blocks: int,
    chain_a: str,
    chain_b: str,
) -> Path:
    return (
        app_config.paths.processed_statistics_dir
        / f"window_{window_blocks}_{chain_a}_vs_{chain_b}_daily_series_stats"
    )


def window_stats_summary_path_base(
    app_config: AppConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.processed_statistics_dir
        / f"window_{window_blocks}_stats_summary"
    )


# ============================================================
# Basic helpers
# ============================================================


def _validate_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


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


def _json_series_to_dict(payload: Any) -> Dict[str, float]:
    if payload is None:
        return {}

    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            parsed = _safe_float(value)
            if parsed is not None:
                out[str(key)] = parsed
        return out

    if not isinstance(payload, str):
        return {}

    payload = payload.strip()
    if not payload:
        return {}

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    if not isinstance(decoded, dict):
        return {}

    out: Dict[str, float] = {}
    for key, value in decoded.items():
        parsed = _safe_float(value)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def _compute_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b)
    union_size = int(union.sum())
    if union_size == 0:
        return 0.0
    intersection_size = int(np.logical_and(mask_a, mask_b).sum())
    return intersection_size / union_size


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(x) < 2 or len(y) < 2:
        return None, None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None, None
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(x) < 2 or len(y) < 2:
        return None, None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None, None
    r, p = spearmanr(x, y)
    if np.isnan(r) or np.isnan(p):
        return None, None
    return float(r), float(p)


# ============================================================
# Presence-matrix loading
# ============================================================


def load_presence_matrix(
    app_config: AppConfig,
    *,
    window_blocks: int,
    status_name: str,
) -> pd.DataFrame:
    """
    status_name expected:
    - present
    - active
    - passive
    """
    df = _load_table_as_dataframe(
        path_without_suffix=presence_matrix_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
            status_name=status_name,
        ),
        table_format=app_config.storage.table_format,
    )

    if df.empty:
        return df

    _validate_columns(df, ["address"], f"{status_name} presence matrix")

    chain_cols = [col for col in df.columns if col != "address"]

    for col in chain_cols:
        if not pd.api.types.is_bool_dtype(df[col]):
            logger.info(
                "[stats][window=%s][%s] Coercing presence column '%s' from dtype=%s to bool",
                window_blocks,
                status_name,
                col,
                df[col].dtype,
            )
        df[col] = _coerce_bool_series(df[col])

    df["address"] = df["address"].astype(str).str.lower()
    return df


# ============================================================
# Chi-squared + permutation on presence matrices
# ============================================================


def build_contingency_table(
    presence_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
) -> np.ndarray:
    _validate_columns(presence_df, ["address", chain_a, chain_b], "presence_df")

    mask_a = _coerce_bool_series(presence_df[chain_a]).to_numpy()
    mask_b = _coerce_bool_series(presence_df[chain_b]).to_numpy()

    present_present = int(np.logical_and(mask_a, mask_b).sum())
    present_absent = int(np.logical_and(mask_a, np.logical_not(mask_b)).sum())
    absent_present = int(np.logical_and(np.logical_not(mask_a), mask_b).sum())
    absent_absent = int(np.logical_and(np.logical_not(mask_a), np.logical_not(mask_b)).sum())

    return np.array(
        [
            [present_present, present_absent],
            [absent_present, absent_absent],
        ],
        dtype=int,
    )


def chi_square_test_for_pair(
    presence_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    status_name: str,
    window_blocks: int,
) -> Dict[str, Any]:
    contingency = build_contingency_table(
        presence_df=presence_df,
        chain_a=chain_a,
        chain_b=chain_b,
    )

    mask_a = _coerce_bool_series(presence_df[chain_a]).to_numpy()
    mask_b = _coerce_bool_series(presence_df[chain_b]).to_numpy()

    observed_jaccard = _compute_jaccard(mask_a, mask_b)
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    row: Dict[str, Any] = {
        "window_blocks": window_blocks,
        "status_name": status_name,
        "chain_a": chain_a,
        "chain_b": chain_b,
        "population_size": int(len(presence_df)),
        "count_a": int(mask_a.sum()),
        "count_b": int(mask_b.sum()),
        "intersection_size": int(np.logical_and(mask_a, mask_b).sum()),
        "union_size": int(np.logical_or(mask_a, mask_b).sum()),
        "observed_jaccard": observed_jaccard,
        "chi2_valid": True,
        "chi2_error_reason": None,
        "chi2_stat": None,
        "chi2_p_value": None,
        "chi2_dof": None,
        "fisher_valid": False,
        "fisher_error_reason": None,
        "fisher_odds_ratio": None,
        "fisher_p_value": None,
        "contingency_present_present": int(contingency[0, 0]),
        "contingency_present_absent": int(contingency[0, 1]),
        "contingency_absent_present": int(contingency[1, 0]),
        "contingency_absent_absent": int(contingency[1, 1]),
        "expected_present_present": None,
        "expected_present_absent": None,
        "expected_absent_present": None,
        "expected_absent_absent": None,
    }

    # Fisher exact is robust for sparse 2x2 tables, so we try it regardless.
    try:
        fisher_odds_ratio, fisher_p_value = fisher_exact(contingency)
        row["fisher_valid"] = True
        row["fisher_odds_ratio"] = float(fisher_odds_ratio)
        row["fisher_p_value"] = float(fisher_p_value)
    except Exception as exc:
        row["fisher_valid"] = False
        row["fisher_error_reason"] = str(exc)
        logger.warning(
            "[stats][window=%s][%s][%s vs %s] Fisher exact failed: %s; contingency=%s",
            window_blocks,
            status_name,
            chain_a,
            chain_b,
            exc,
            contingency.tolist(),
        )

    # Chi-square is not valid for some degenerate sparse cases.
    if np.any(row_sums == 0) or np.any(col_sums == 0):
        row["chi2_valid"] = False
        row["chi2_error_reason"] = "degenerate_zero_row_or_column"
        logger.warning(
            "[stats][window=%s][%s][%s vs %s] Skipping chi-square due to degenerate contingency table=%s",
            window_blocks,
            status_name,
            chain_a,
            chain_b,
            contingency.tolist(),
        )
        return row

    try:
        chi2_stat, p_value, dof, expected = chi2_contingency(
            contingency,
            correction=False,
        )
        row["chi2_valid"] = True
        row["chi2_stat"] = float(chi2_stat)
        row["chi2_p_value"] = float(p_value)
        row["chi2_dof"] = int(dof)
        row["expected_present_present"] = float(expected[0, 0])
        row["expected_present_absent"] = float(expected[0, 1])
        row["expected_absent_present"] = float(expected[1, 0])
        row["expected_absent_absent"] = float(expected[1, 1])
    except ValueError as exc:
        row["chi2_valid"] = False
        row["chi2_error_reason"] = str(exc)
        logger.warning(
            "[stats][window=%s][%s][%s vs %s] Chi-square fallback triggered: %s; contingency=%s",
            window_blocks,
            status_name,
            chain_a,
            chain_b,
            exc,
            contingency.tolist(),
        )

    return row


def permutation_test_for_pair(
    presence_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    status_name: str,
    window_blocks: int,
    config: Optional[StatsConfig] = None,
) -> Dict[str, Any]:
    config = config or StatsConfig()

    _validate_columns(presence_df, ["address", chain_a, chain_b], "presence_df")

    mask_a = _coerce_bool_series(presence_df[chain_a]).to_numpy()
    mask_b = _coerce_bool_series(presence_df[chain_b]).to_numpy()

    observed = _compute_jaccard(mask_a, mask_b)

    rng = np.random.default_rng(config.random_state)
    null_scores = np.empty(config.n_permutations, dtype=float)

    for i in range(config.n_permutations):
        shuffled_b = rng.permutation(mask_b)
        null_scores[i] = _compute_jaccard(mask_a, shuffled_b)

    empirical_p = float((1 + np.sum(null_scores >= observed)) / (1 + config.n_permutations))
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))
    null_q95 = float(np.quantile(null_scores, 0.95))
    z_score = float((observed - null_mean) / null_std) if null_std > 0 else None

    return {
        "window_blocks": window_blocks,
        "status_name": status_name,
        "chain_a": chain_a,
        "chain_b": chain_b,
        "observed_jaccard": float(observed),
        "permutation_count": int(config.n_permutations),
        "null_mean_jaccard": null_mean,
        "null_std_jaccard": null_std,
        "null_q95_jaccard": null_q95,
        "empirical_p_value": empirical_p,
        "null_z_score": z_score,
        "observed_gt_null_q95": bool(observed > null_q95),
    }


def run_presence_statistics_for_status(
    presence_df: pd.DataFrame,
    *,
    status_name: str,
    window_blocks: int,
    config: Optional[StatsConfig] = None,
) -> pd.DataFrame:
    config = config or StatsConfig()

    if presence_df.empty:
        logger.info(
            "[stats][window=%s][%s] Presence matrix is empty; skipping pairwise presence statistics",
            window_blocks,
            status_name,
        )
        return pd.DataFrame()

    chain_cols = [col for col in presence_df.columns if col != "address"]
    if len(chain_cols) < 2:
        logger.info(
            "[stats][window=%s][%s] Fewer than 2 chains present; skipping pairwise presence statistics",
            window_blocks,
            status_name,
        )
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for i, chain_a in enumerate(chain_cols):
        for chain_b in chain_cols[i + 1 :]:
            chi_row = chi_square_test_for_pair(
                presence_df=presence_df,
                chain_a=chain_a,
                chain_b=chain_b,
                status_name=status_name,
                window_blocks=window_blocks,
            )
            perm_row = permutation_test_for_pair(
                presence_df=presence_df,
                chain_a=chain_a,
                chain_b=chain_b,
                status_name=status_name,
                window_blocks=window_blocks,
                config=config,
            )
            rows.append({**chi_row, **perm_row})

    return pd.DataFrame(rows)


# ============================================================
# Feature alignment loading
# ============================================================


def load_pairwise_feature_alignment(
    app_config: AppConfig,
    *,
    window_blocks: int,
    chain_a: str,
    chain_b: str,
) -> pd.DataFrame:
    df = _load_table_as_dataframe(
        path_without_suffix=pairwise_feature_alignment_path_base(
            app_config=app_config,
            window_blocks=window_blocks,
            chain_a=chain_a,
            chain_b=chain_b,
        ),
        table_format=app_config.storage.table_format,
    )
    if df.empty:
        return df
    if "address" in df.columns:
        df["address"] = df["address"].astype(str).str.lower()
    return df


# ============================================================
# Pairwise feature correlations
# ============================================================


DEFAULT_FEATURE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("value_sent_wei", "value_sent_wei"),
    ("value_received_wei", "value_received_wei"),
    ("total_value_wei", "total_value_wei"),
    ("sent_tx_count", "sent_tx_count"),
    ("received_tx_count", "received_tx_count"),
    ("total_tx_count", "total_tx_count"),
    ("avg_sent_value_wei", "avg_sent_value_wei"),
    ("avg_received_value_wei", "avg_received_value_wei"),
    ("unique_counterparties", "unique_counterparties"),
    ("unique_outgoing_counterparties", "unique_outgoing_counterparties"),
    ("unique_incoming_counterparties", "unique_incoming_counterparties"),
    ("tx_frequency_per_day", "tx_frequency_per_day"),
    ("sent_tx_frequency_per_day", "sent_tx_frequency_per_day"),
    ("received_tx_frequency_per_day", "received_tx_frequency_per_day"),
    ("tx_frequency_per_block", "tx_frequency_per_block"),
    ("first_activity_timestamp", "first_activity_timestamp"),
    ("last_activity_timestamp", "last_activity_timestamp"),
)


def compute_feature_correlations_for_alignment(
    alignment_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    window_blocks: int,
    feature_pairs: Sequence[Tuple[str, str]] = DEFAULT_FEATURE_PAIRS,
    config: Optional[StatsConfig] = None,
) -> pd.DataFrame:
    config = config or StatsConfig()

    if alignment_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for feature_a, feature_b in feature_pairs:
        col_a = f"{chain_a}_{feature_a}"
        col_b = f"{chain_b}_{feature_b}"

        if col_a not in alignment_df.columns or col_b not in alignment_df.columns:
            continue

        subset = alignment_df[[col_a, col_b]].copy()
        subset = _coerce_numeric(subset, [col_a, col_b])

        if config.dropna_for_correlations:
            subset = subset.dropna(subset=[col_a, col_b])

        if len(subset) < config.min_samples_for_correlation:
            rows.append(
                {
                    "window_blocks": window_blocks,
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "sample_size": int(len(subset)),
                    "pearson_r": None,
                    "pearson_p_value": None,
                    "spearman_r": None,
                    "spearman_p_value": None,
                }
            )
            continue

        x = subset[col_a].to_numpy(dtype=float)
        y = subset[col_b].to_numpy(dtype=float)

        pearson_r_value, pearson_p_value = _safe_pearson(x, y)
        spearman_r_value, spearman_p_value = _safe_spearman(x, y)

        rows.append(
            {
                "window_blocks": window_blocks,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "feature_a": feature_a,
                "feature_b": feature_b,
                "sample_size": int(len(subset)),
                "pearson_r": pearson_r_value,
                "pearson_p_value": pearson_p_value,
                "spearman_r": spearman_r_value,
                "spearman_p_value": spearman_p_value,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Daily-series correlation
# ============================================================


def _aligned_daily_arrays(
    left_series_json: Any,
    right_series_json: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    left = _json_series_to_dict(left_series_json)
    right = _json_series_to_dict(right_series_json)

    keys = sorted(set(left.keys()) | set(right.keys()))
    if not keys:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.array([left.get(key, 0.0) for key in keys], dtype=float)
    y = np.array([right.get(key, 0.0) for key in keys], dtype=float)
    return x, y


def compute_daily_series_correlations(
    alignment_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    window_blocks: int,
    config: Optional[StatsConfig] = None,
) -> pd.DataFrame:
    """
    Per-address daily series correlations, then a compact row table.

    Uses:
    - {chain}_daily_tx_count_series_json
    - {chain}_daily_value_sent_wei_series_json
    - {chain}_daily_value_received_wei_series_json
    """
    config = config or StatsConfig()

    if alignment_df.empty:
        return pd.DataFrame()

    series_specs = [
        ("daily_tx_count_series_json", "daily_tx_count"),
        ("daily_value_sent_wei_series_json", "daily_value_sent_wei"),
        ("daily_value_received_wei_series_json", "daily_value_received_wei"),
    ]

    rows: List[Dict[str, Any]] = []

    for _, row in alignment_df.iterrows():
        address = str(row.get("address") or "").strip().lower()
        if not address:
            continue

        for suffix, series_name in series_specs:
            col_a = f"{chain_a}_{suffix}"
            col_b = f"{chain_b}_{suffix}"
            if col_a not in alignment_df.columns or col_b not in alignment_df.columns:
                continue

            x, y = _aligned_daily_arrays(row.get(col_a), row.get(col_b))
            if len(x) < config.min_samples_for_correlation:
                rows.append(
                    {
                        "window_blocks": window_blocks,
                        "chain_a": chain_a,
                        "chain_b": chain_b,
                        "address": address,
                        "series_name": series_name,
                        "aligned_days": int(len(x)),
                        "pearson_r": None,
                        "pearson_p_value": None,
                        "spearman_r": None,
                        "spearman_p_value": None,
                    }
                )
                continue

            pearson_r_value, pearson_p_value = _safe_pearson(x, y)
            spearman_r_value, spearman_p_value = _safe_spearman(x, y)

            rows.append(
                {
                    "window_blocks": window_blocks,
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "address": address,
                    "series_name": series_name,
                    "aligned_days": int(len(x)),
                    "pearson_r": pearson_r_value,
                    "pearson_p_value": pearson_p_value,
                    "spearman_r": spearman_r_value,
                    "spearman_p_value": spearman_p_value,
                }
            )

    return pd.DataFrame(rows)


def summarize_daily_series_correlations(
    per_address_df: pd.DataFrame,
) -> pd.DataFrame:
    if per_address_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "aligned_days",
        "pearson_r",
        "pearson_p_value",
        "spearman_r",
        "spearman_p_value",
    ]
    df = _coerce_numeric(per_address_df, numeric_cols)

    rows = []
    grouping_cols = ["window_blocks", "chain_a", "chain_b", "series_name"]
    for keys, group in df.groupby(grouping_cols, dropna=False):
        window_blocks, chain_a, chain_b, series_name = keys

        pearson_values = group["pearson_r"].dropna()
        spearman_values = group["spearman_r"].dropna()

        rows.append(
            {
                "window_blocks": int(window_blocks),
                "chain_a": str(chain_a),
                "chain_b": str(chain_b),
                "series_name": str(series_name),
                "address_count": int(group["address"].nunique()),
                "valid_pearson_count": int(pearson_values.shape[0]),
                "valid_spearman_count": int(spearman_values.shape[0]),
                "mean_aligned_days": float(group["aligned_days"].dropna().mean())
                if group["aligned_days"].dropna().shape[0] > 0
                else None,
                "mean_pearson_r": float(pearson_values.mean()) if pearson_values.shape[0] > 0 else None,
                "median_pearson_r": float(pearson_values.median()) if pearson_values.shape[0] > 0 else None,
                "mean_spearman_r": float(spearman_values.mean()) if spearman_values.shape[0] > 0 else None,
                "median_spearman_r": float(spearman_values.median()) if spearman_values.shape[0] > 0 else None,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Window-level orchestration
# ============================================================


def run_presence_statistics_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    config: Optional[StatsConfig] = None,
    save_output: bool = True,
) -> Dict[str, pd.DataFrame]:
    config = config or StatsConfig()

    outputs: Dict[str, pd.DataFrame] = {}

    for status_name in ("present", "active", "passive"):
        presence_df = load_presence_matrix(
            app_config=app_config,
            window_blocks=window_blocks,
            status_name=status_name,
        )
        stats_df = run_presence_statistics_for_status(
            presence_df=presence_df,
            status_name=status_name,
            window_blocks=window_blocks,
            config=config,
        )
        outputs[status_name] = stats_df

        if save_output:
            _save_dataframe(
                stats_df,
                presence_stats_table_path_base(app_config, window_blocks, status_name),
                app_config.storage.table_format,
            )

    return outputs


def run_feature_statistics_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    config: Optional[StatsConfig] = None,
    save_output: bool = True,
) -> Dict[str, pd.DataFrame]:
    config = config or StatsConfig()

    chain_names = [chain.name for chain in app_config.enabled_chains]

    feature_stats_frames: List[pd.DataFrame] = []
    daily_series_summary_frames: List[pd.DataFrame] = []

    for i, chain_a in enumerate(chain_names):
        for chain_b in chain_names[i + 1 :]:
            alignment_df = load_pairwise_feature_alignment(
                app_config=app_config,
                window_blocks=window_blocks,
                chain_a=chain_a,
                chain_b=chain_b,
            )

            feature_stats_df = compute_feature_correlations_for_alignment(
                alignment_df=alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                window_blocks=window_blocks,
                config=config,
            )
            feature_stats_frames.append(feature_stats_df)

            daily_per_address_df = compute_daily_series_correlations(
                alignment_df=alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                window_blocks=window_blocks,
                config=config,
            )
            daily_summary_df = summarize_daily_series_correlations(daily_per_address_df)
            daily_series_summary_frames.append(daily_summary_df)

            if save_output:
                _save_dataframe(
                    feature_stats_df,
                    feature_stats_table_path_base(
                        app_config,
                        window_blocks,
                        chain_a,
                        chain_b,
                    ),
                    app_config.storage.table_format,
                )
                _save_dataframe(
                    daily_summary_df,
                    daily_series_stats_table_path_base(
                        app_config,
                        window_blocks,
                        chain_a,
                        chain_b,
                    ),
                    app_config.storage.table_format,
                )

    feature_stats_out = (
        pd.concat(feature_stats_frames, axis=0, ignore_index=True)
        if feature_stats_frames
        else pd.DataFrame()
    )
    daily_series_out = (
        pd.concat(daily_series_summary_frames, axis=0, ignore_index=True)
        if daily_series_summary_frames
        else pd.DataFrame()
    )

    return {
        "feature_stats": feature_stats_out,
        "daily_series_stats": daily_series_out,
    }


def build_window_stats_summary(
    *,
    window_blocks: int,
    presence_outputs: Dict[str, pd.DataFrame],
    feature_outputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for status_name, df in presence_outputs.items():
        if df.empty:
            rows.append(
                {
                    "window_blocks": window_blocks,
                    "section": "presence",
                    "name": status_name,
                    "row_count": 0,
                    "valid_chi2_pairs": 0,
                    "significant_chi2_pairs": 0,
                    "significant_fisher_pairs": 0,
                    "significant_empirical_pairs": 0,
                }
            )
            continue

        chi_valid = int(_coerce_bool_series(df["chi2_valid"]).sum()) if "chi2_valid" in df.columns else 0
        chi_sig = int(
            (
                pd.to_numeric(df.get("chi2_p_value"), errors="coerce") < 0.05
            ).fillna(False).sum()
        ) if "chi2_p_value" in df.columns else 0
        fisher_sig = int(
            (
                pd.to_numeric(df.get("fisher_p_value"), errors="coerce") < 0.05
            ).fillna(False).sum()
        ) if "fisher_p_value" in df.columns else 0
        emp_sig = int(
            (
                pd.to_numeric(df.get("empirical_p_value"), errors="coerce") < 0.05
            ).fillna(False).sum()
        ) if "empirical_p_value" in df.columns else 0

        rows.append(
            {
                "window_blocks": window_blocks,
                "section": "presence",
                "name": status_name,
                "row_count": int(len(df)),
                "valid_chi2_pairs": chi_valid,
                "significant_chi2_pairs": chi_sig,
                "significant_fisher_pairs": fisher_sig,
                "significant_empirical_pairs": emp_sig,
            }
        )

    feature_stats_df = feature_outputs.get("feature_stats", pd.DataFrame())
    if feature_stats_df.empty:
        rows.append(
            {
                "window_blocks": window_blocks,
                "section": "feature",
                "name": "pairwise_feature_correlations",
                "row_count": 0,
                "significant_pearson_pairs": 0,
                "significant_spearman_pairs": 0,
            }
        )
    else:
        pearson_sig = int(
            (
                pd.to_numeric(feature_stats_df["pearson_p_value"], errors="coerce") < 0.05
            ).fillna(False).sum()
        )
        spearman_sig = int(
            (
                pd.to_numeric(feature_stats_df["spearman_p_value"], errors="coerce") < 0.05
            ).fillna(False).sum()
        )
        rows.append(
            {
                "window_blocks": window_blocks,
                "section": "feature",
                "name": "pairwise_feature_correlations",
                "row_count": int(len(feature_stats_df)),
                "significant_pearson_pairs": pearson_sig,
                "significant_spearman_pairs": spearman_sig,
            }
        )

    daily_stats_df = feature_outputs.get("daily_series_stats", pd.DataFrame())
    if daily_stats_df.empty:
        rows.append(
            {
                "window_blocks": window_blocks,
                "section": "daily_series",
                "name": "daily_series_correlations",
                "row_count": 0,
                "mean_pearson_r": None,
                "mean_spearman_r": None,
            }
        )
    else:
        rows.append(
            {
                "window_blocks": window_blocks,
                "section": "daily_series",
                "name": "daily_series_correlations",
                "row_count": int(len(daily_stats_df)),
                "mean_pearson_r": pd.to_numeric(daily_stats_df["mean_pearson_r"], errors="coerce").mean(),
                "mean_spearman_r": pd.to_numeric(daily_stats_df["mean_spearman_r"], errors="coerce").mean(),
            }
        )

    return pd.DataFrame(rows)


def run_statistics_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    config: Optional[StatsConfig] = None,
    save_output: bool = True,
) -> Dict[str, Any]:
    config = config or StatsConfig()

    presence_outputs = run_presence_statistics_for_window(
        app_config=app_config,
        window_blocks=window_blocks,
        config=config,
        save_output=save_output,
    )

    feature_outputs = run_feature_statistics_for_window(
        app_config=app_config,
        window_blocks=window_blocks,
        config=config,
        save_output=save_output,
    )

    summary_df = build_window_stats_summary(
        window_blocks=window_blocks,
        presence_outputs=presence_outputs,
        feature_outputs=feature_outputs,
    )

    if save_output:
        _save_dataframe(
            summary_df,
            window_stats_summary_path_base(app_config, window_blocks),
            app_config.storage.table_format,
        )

    return {
        "presence": presence_outputs,
        "features": feature_outputs,
        "summary": summary_df,
    }


def run_statistics_for_all_windows(
    app_config: AppConfig,
    *,
    config: Optional[StatsConfig] = None,
    save_output: bool = True,
) -> Dict[int, Dict[str, Any]]:
    config = config or StatsConfig()

    outputs: Dict[int, Dict[str, Any]] = {}
    for window_blocks in app_config.sampling.windows.block_counts:
        outputs[window_blocks] = run_statistics_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            config=config,
            save_output=save_output,
        )
    return outputs