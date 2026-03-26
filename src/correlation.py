"""
correlation.py — Cross-chain correlation analyses (Phase 6).

Implements four correlation types between overlapping addresses across chains:

    6a  Temporal   — Δ between first activity on Chain A vs Chain B (days)
    6b  Volume     — Pearson & Spearman on total activity (tx count / value)
    6c  Frequency  — Per-address Pearson r on time-windowed tx counts → average
    6d  Ratio      — Share of active wallets on Chain A also on Chain B vs null model

All functions accept:
    • feature_df   : the per-address-per-chain feature table (long format)
                     produced by features.py and cached as
                     data/interim/feature_rows/address_chain_features_{window}.csv
    • active_sets  : optional dict chain → frozenset[str]  (from set_ops.load_address_set)
    • passive_sets : optional dict chain → frozenset[str]

Typical usage
-------------
    import pandas as pd
    from correlation import run_all_correlations

    df = pd.read_csv("data/interim/feature_rows/address_chain_features_1blk.csv")
    results = run_all_correlations(df, chains=["ethereum", "polygon", "bnb"])
    results["temporal"].to_csv("outputs/tables/corr_temporal_1blk.csv", index=False)
    results["volume"].to_csv("outputs/tables/corr_volume_1blk.csv", index=False)
    results["frequency"].to_csv("outputs/tables/corr_frequency_1blk.csv", index=False)
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# EVM address space size (for null-model calculations)
_ADDRESS_SPACE: int = 2**160


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load a feature table from CSV or Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_correlation_results(
    results: Dict[str, Optional[pd.DataFrame]],
    output_dir: str | Path,
    window_label: str,
) -> None:
    """
    Save all correlation result DataFrames to CSV files.

    Files written
    -------------
    - corr_temporal_{window}.csv
    - corr_volume_{window}.csv
    - corr_frequency_{window}.csv
    - corr_ratio_{window}.csv   (only when ratio results exist)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = {
        "temporal": f"corr_temporal_{window_label}.csv",
        "volume": f"corr_volume_{window_label}.csv",
        "frequency": f"corr_frequency_{window_label}.csv",
        "ratio": f"corr_ratio_{window_label}.csv",
    }
    for key, filename in mapping.items():
        df = results.get(key)
        if df is not None and not df.empty:
            df.to_csv(out / filename, index=False)
            logger.info("Saved %s → %s", key, out / filename)


# ─────────────────────────────────────────────────────────────────────────────
# Internal pivot helper
# ─────────────────────────────────────────────────────────────────────────────


def _pivot_to_chains(
    df: pd.DataFrame,
    value_col: str,
    chain_a: str,
    chain_b: str,
    index_col: str = "address",
) -> pd.DataFrame:
    """
    Pivot a long feature table to wide format for two chains, keeping only
    addresses that appear on **both** chains (inner join).

    Returns a DataFrame with columns: [index_col, chain_a, chain_b].
    """
    sub = (
        df[df["chain"].isin([chain_a, chain_b])][[index_col, "chain", value_col]]
        .copy()
    )
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    wide = sub.pivot_table(
        index=index_col,
        columns="chain",
        values=value_col,
        aggfunc="first",
    )
    # Keep only rows present on both chains
    if chain_a not in wide.columns or chain_b not in wide.columns:
        return pd.DataFrame(columns=[index_col, chain_a, chain_b])
    wide = wide.dropna(subset=[chain_a, chain_b]).reset_index()
    return wide


# ─────────────────────────────────────────────────────────────────────────────
# 6a — Temporal correlation
# ─────────────────────────────────────────────────────────────────────────────


def temporal_delta_analysis(
    feature_df: pd.DataFrame,
    chain_a: str,
    chain_b: str,
    *,
    ts_col: str = "first_seen_ts",
) -> Dict[str, Any]:
    """
    Compute the time delta (days) between first activity on *chain_a* vs *chain_b*
    for every address present on both chains.

    Sign convention: positive delta → chain_a seen first (earlier timestamp).

    Returns
    -------
    dict with:
        n_overlap            – number of overlapping addresses
        deltas_days          – numpy array of signed deltas
        mean_delta_days      – mean signed delta
        median_delta_days    – median signed delta
        std_delta_days       – standard deviation of deltas
        abs_mean_delta_days  – mean |delta|
        abs_median_delta_days– median |delta|
        pct_a_first          – fraction where chain_a was active first
        pct_b_first          – fraction where chain_b was active first
        pct_same_day         – fraction where |delta| < 1 day
        pearson_r / _p       – Pearson correlation of first_seen_ts values
        spearman_r / _p      – Spearman correlation of first_seen_ts values
    """
    wide = _pivot_to_chains(feature_df, ts_col, chain_a, chain_b)
    if wide.empty:
        logger.warning(
            "No overlapping addresses for %s ∩ %s (temporal)", chain_a, chain_b
        )
        return {
            "chain_a": chain_a,
            "chain_b": chain_b,
            "n_overlap": 0,
            "deltas_days": np.array([]),
        }

    ts_a: np.ndarray = wide[chain_a].values.astype(float)
    ts_b: np.ndarray = wide[chain_b].values.astype(float)

    # Positive = chain_a seen first (smaller timestamp means earlier)
    delta_seconds = ts_b - ts_a  # positive → chain_b is later → chain_a was first
    delta_days = delta_seconds / 86_400.0

    n = len(delta_days)
    pct_a_first = float((delta_days > 0).sum() / n) if n else 0.0
    pct_b_first = float((delta_days < 0).sum() / n) if n else 0.0
    pct_same_day = float((np.abs(delta_days) < 1.0).sum() / n) if n else 0.0

    pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    if n >= 3:
        try:
            pearson_r, pearson_p = scipy_stats.pearsonr(ts_a, ts_b)
            spearman_r, spearman_p = scipy_stats.spearmanr(ts_a, ts_b)
        except Exception as exc:
            logger.warning("Temporal correlation stats failed: %s", exc)

    def _maybe_float(v: float) -> Optional[float]:
        return float(v) if not np.isnan(v) else None

    return {
        "chain_a": chain_a,
        "chain_b": chain_b,
        "n_overlap": n,
        "deltas_days": delta_days,
        "mean_delta_days": float(np.nanmean(delta_days)),
        "median_delta_days": float(np.nanmedian(delta_days)),
        "std_delta_days": float(np.nanstd(delta_days)),
        "abs_mean_delta_days": float(np.nanmean(np.abs(delta_days))),
        "abs_median_delta_days": float(np.nanmedian(np.abs(delta_days))),
        "pct_a_first": pct_a_first,
        "pct_b_first": pct_b_first,
        "pct_same_day": pct_same_day,
        "pearson_r": _maybe_float(pearson_r),
        "pearson_p": _maybe_float(pearson_p),
        "spearman_r": _maybe_float(spearman_r),
        "spearman_p": _maybe_float(spearman_p),
    }


def run_all_temporal(
    feature_df: pd.DataFrame,
    chains: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run temporal analysis for every pair of chains.

    Returns a summary DataFrame (one row per pair).
    Individual ``deltas_days`` arrays are NOT included in the DataFrame;
    call ``temporal_delta_analysis`` directly to access them.
    """
    chains = chains or sorted(feature_df["chain"].dropna().unique().tolist())
    rows = []
    for chain_a, chain_b in itertools.combinations(chains, 2):
        r = temporal_delta_analysis(feature_df, chain_a, chain_b)
        rows.append(
            {
                "chain_a": chain_a,
                "chain_b": chain_b,
                "n_overlap": r.get("n_overlap", 0),
                "abs_mean_delta_days": r.get("abs_mean_delta_days"),
                "abs_median_delta_days": r.get("abs_median_delta_days"),
                "std_delta_days": r.get("std_delta_days"),
                "pct_a_first": r.get("pct_a_first"),
                "pct_b_first": r.get("pct_b_first"),
                "pct_same_day": r.get("pct_same_day"),
                "pearson_r": r.get("pearson_r"),
                "pearson_p": r.get("pearson_p"),
                "spearman_r": r.get("spearman_r"),
                "spearman_p": r.get("spearman_p"),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6b — Volume correlation
# ─────────────────────────────────────────────────────────────────────────────


def volume_correlation(
    feature_df: pd.DataFrame,
    chain_a: str,
    chain_b: str,
    *,
    volume_col: str = "total_tx_count",
) -> Dict[str, Any]:
    """
    Pearson & Spearman correlation of *volume_col* on *chain_a* vs *chain_b*
    for addresses present on both chains.

    Analysis is performed on both the raw values and log1p-transformed values
    (EVM transaction counts are heavily right-skewed).

    Parameters
    ----------
    volume_col : feature column to use as the volume proxy.
                 Defaults to ``"total_tx_count"``.  Use ``"value_sent_eth"``
                 if that column is present in the feature table.
    """
    wide = _pivot_to_chains(feature_df, volume_col, chain_a, chain_b)
    n = len(wide)

    if n < 3:
        logger.warning(
            "Too few overlapping addresses for %s ∩ %s (volume, n=%d)",
            chain_a,
            chain_b,
            n,
        )
        return {
            "chain_a": chain_a,
            "chain_b": chain_b,
            "volume_col": volume_col,
            "n_overlap": n,
        }

    v_a: np.ndarray = wide[chain_a].values.astype(float)
    v_b: np.ndarray = wide[chain_b].values.astype(float)

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        try:
            pr, pp = scipy_stats.pearsonr(x, y)
            sr, sp = scipy_stats.spearmanr(x, y)
            return float(pr), float(pp), float(sr), float(sp)
        except Exception as exc:
            logger.warning("Correlation computation failed: %s", exc)
            return None, None, None, None

    pr, pp, sr, sp = _safe_corr(v_a, v_b)
    log_a, log_b = np.log1p(np.clip(v_a, 0, None)), np.log1p(np.clip(v_b, 0, None))
    pr_log, pp_log, sr_log, sp_log = _safe_corr(log_a, log_b)

    return {
        "chain_a": chain_a,
        "chain_b": chain_b,
        "volume_col": volume_col,
        "n_overlap": n,
        "mean_a": float(np.nanmean(v_a)),
        "mean_b": float(np.nanmean(v_b)),
        "median_a": float(np.nanmedian(v_a)),
        "median_b": float(np.nanmedian(v_b)),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
        "pearson_r_log": pr_log,
        "pearson_p_log": pp_log,
        "spearman_r_log": sr_log,
        "spearman_p_log": sp_log,
        # Raw arrays for scatter-plot use
        "v_a": v_a,
        "v_b": v_b,
    }


def run_all_volume(
    feature_df: pd.DataFrame,
    chains: Optional[List[str]] = None,
    volume_col: str = "total_tx_count",
) -> pd.DataFrame:
    """Run volume correlation for all chain pairs; return summary DataFrame."""
    chains = chains or sorted(feature_df["chain"].dropna().unique().tolist())
    rows = []
    for chain_a, chain_b in itertools.combinations(chains, 2):
        r = volume_correlation(feature_df, chain_a, chain_b, volume_col=volume_col)
        rows.append(
            {
                "chain_a": chain_a,
                "chain_b": chain_b,
                "volume_col": volume_col,
                "n_overlap": r.get("n_overlap"),
                "mean_a": r.get("mean_a"),
                "mean_b": r.get("mean_b"),
                "pearson_r": r.get("pearson_r"),
                "pearson_p": r.get("pearson_p"),
                "spearman_r": r.get("spearman_r"),
                "spearman_p": r.get("spearman_p"),
                "pearson_r_log": r.get("pearson_r_log"),
                "pearson_p_log": r.get("pearson_p_log"),
                "spearman_r_log": r.get("spearman_r_log"),
                "spearman_p_log": r.get("spearman_p_log"),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6c — Activity frequency correlation
# ─────────────────────────────────────────────────────────────────────────────


def frequency_correlation_from_features(
    feature_df: pd.DataFrame,
    chain_a: str,
    chain_b: str,
    *,
    window_buckets: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Frequency correlation using the available time-windowed tx count columns.

    Because per-day time series are not stored in the feature table, we
    proxy "daily activity time series" with the feature columns that capture
    different observation horizons:

        recent_7d_tx_count, recent_30d_tx_count, recent_90d_tx_count, total_tx_count

    For each address present on both chains we build a 4-element vector
    [7d, 30d, 90d, total] and compute the Pearson r between the two chains'
    vectors.  We then average those per-address r values (6c mean).

    Chain-level Pearson/Spearman is also computed for each time bucket.

    Parameters
    ----------
    window_buckets : list of (column_name, label) pairs to use as the
                     frequency proxy; defaults to the four buckets above.

    Returns
    -------
    dict with:
        n_overlap               – overlapping address count
        n_per_address_r_computed– addresses for which r could be computed
        mean_per_address_r      – average per-address Pearson r (6c output)
        per_address_r_distribution – numpy array of per-address r values
        bucket_results          – DataFrame of chain-level stats per bucket
    """
    buckets: List[Tuple[str, str]] = window_buckets or [
        ("recent_7d_tx_count", "7d"),
        ("recent_30d_tx_count", "30d"),
        ("recent_90d_tx_count", "90d"),
        ("total_tx_count", "total"),
    ]
    available_cols = [col for col, _ in buckets if col in feature_df.columns]

    if not available_cols:
        logger.error("No frequency proxy columns found in feature table")
        return {"chain_a": chain_a, "chain_b": chain_b, "n_overlap": 0}

    # Addresses present on both chains
    addr_a = set(
        feature_df.loc[feature_df["chain"] == chain_a, "address"].dropna()
    )
    addr_b = set(
        feature_df.loc[feature_df["chain"] == chain_b, "address"].dropna()
    )
    overlap = addr_a & addr_b

    if len(overlap) < 3:
        logger.warning(
            "Too few overlapping addresses for %s ∩ %s (frequency, n=%d)",
            chain_a,
            chain_b,
            len(overlap),
        )
        return {
            "chain_a": chain_a,
            "chain_b": chain_b,
            "n_overlap": len(overlap),
            "mean_per_address_r": None,
        }

    def _subset(chain: str) -> pd.DataFrame:
        mask = (feature_df["chain"] == chain) & (
            feature_df["address"].isin(overlap)
        )
        return (
            feature_df.loc[mask]
            .set_index("address")[available_cols]
            .apply(pd.to_numeric, errors="coerce")
            .loc[lambda d: ~d.index.duplicated()]
        )

    df_a = _subset(chain_a)
    df_b = _subset(chain_b)
    shared_idx = df_a.index.intersection(df_b.index)
    df_a, df_b = df_a.loc[shared_idx], df_b.loc[shared_idx]

    # ── per-address Pearson r ──────────────────────────────────────────────
    per_address_r: List[float] = []
    for addr in shared_idx:
        vec_a = df_a.loc[addr].values.astype(float)
        vec_b = df_b.loc[addr].values.astype(float)
        # Skip if either vector is constant (r undefined)
        if np.nanstd(vec_a) < 1e-9 or np.nanstd(vec_b) < 1e-9:
            continue
        try:
            r, _ = scipy_stats.pearsonr(vec_a, vec_b)
            if not np.isnan(r):
                per_address_r.append(float(r))
        except Exception:
            pass

    mean_per_address_r = float(np.mean(per_address_r)) if per_address_r else None

    # ── chain-level stats per time bucket ─────────────────────────────────
    bucket_rows: List[Dict] = []
    for col, label in buckets:
        if col not in df_a.columns:
            continue
        v_a = df_a[col].values.astype(float)
        v_b = df_b[col].values.astype(float)
        valid = ~(np.isnan(v_a) | np.isnan(v_b))
        v_a, v_b = v_a[valid], v_b[valid]
        if len(v_a) < 3:
            continue
        try:
            pr, pp = scipy_stats.pearsonr(v_a, v_b)
            sr, sp = scipy_stats.spearmanr(v_a, v_b)
        except Exception:
            pr = pp = sr = sp = float("nan")

        bucket_rows.append(
            {
                "bucket": label,
                "col": col,
                "n": int(len(v_a)),
                "pearson_r": float(pr) if not np.isnan(pr) else None,
                "pearson_p": float(pp) if not np.isnan(pp) else None,
                "spearman_r": float(sr) if not np.isnan(sr) else None,
                "spearman_p": float(sp) if not np.isnan(sp) else None,
            }
        )

    return {
        "chain_a": chain_a,
        "chain_b": chain_b,
        "n_overlap": len(shared_idx),
        "n_per_address_r_computed": len(per_address_r),
        "mean_per_address_r": mean_per_address_r,
        "per_address_r_distribution": np.array(per_address_r),
        "bucket_results": pd.DataFrame(bucket_rows),
    }


def run_all_frequency(
    feature_df: pd.DataFrame,
    chains: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run frequency correlation for all chain pairs; return summary DataFrame."""
    chains = chains or sorted(feature_df["chain"].dropna().unique().tolist())
    rows = []
    for chain_a, chain_b in itertools.combinations(chains, 2):
        r = frequency_correlation_from_features(feature_df, chain_a, chain_b)
        rows.append(
            {
                "chain_a": chain_a,
                "chain_b": chain_b,
                "n_overlap": r.get("n_overlap"),
                "n_r_computed": r.get("n_per_address_r_computed"),
                "mean_per_address_r": r.get("mean_per_address_r"),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6d — Active/Passive ratio analysis
# ─────────────────────────────────────────────────────────────────────────────


def active_passive_ratio_analysis(
    active_sets: Dict[str, FrozenSet[str]],
    passive_sets: Dict[str, FrozenSet[str]],
) -> pd.DataFrame:
    """
    For every directed pair (A, B), compute what fraction of active wallets on
    Chain A also appear on Chain B (in any role).  Compare to the null-model
    expectation (random draw from 2^160 address space → near-zero).

    Columns
    -------
    source_chain                 – Chain A
    target_chain                 – Chain B
    n_active_source              – |active(A)|
    n_all_target                 – |active(B) ∪ passive(B)|
    n_active_on_both             – |active(A) ∩ active(B)|
    n_any_on_target              – |active(A) ∩ (active(B) ∪ passive(B))|
    share_active_also_active     – n_active_on_both / n_active_source
    share_active_any_role        – n_any_on_target / n_active_source
    expected_random_share        – null-model expected share (≈ 0)
    enrichment_factor            – observed / expected (higher = more non-random)
    """
    all_chains = sorted(set(active_sets) | set(passive_sets))
    rows = []

    for chain_a in all_chains:
        active_a = active_sets.get(chain_a, frozenset())
        if not active_a:
            continue

        for chain_b in all_chains:
            if chain_a == chain_b:
                continue

            active_b = active_sets.get(chain_b, frozenset())
            passive_b = passive_sets.get(chain_b, frozenset())
            all_b = active_b | passive_b

            overlap_active_active = active_a & active_b
            overlap_active_any = active_a & all_b

            n = len(active_a)
            share_active = len(overlap_active_active) / n if n else 0.0
            share_any = len(overlap_active_any) / n if n else 0.0

            # Null model: expected overlap fraction = |all_B| / |address_space|
            exp_share = len(all_b) / _ADDRESS_SPACE if _ADDRESS_SPACE > 0 else 0.0
            enrichment = share_any / exp_share if exp_share > 0 else float("inf")

            rows.append(
                {
                    "source_chain": chain_a,
                    "target_chain": chain_b,
                    "n_active_source": n,
                    "n_all_target": len(all_b),
                    "n_active_on_both": len(overlap_active_active),
                    "n_any_on_target": len(overlap_active_any),
                    "share_active_also_active": round(share_active, 8),
                    "share_active_any_role": round(share_any, 8),
                    "expected_random_share": exp_share,
                    "enrichment_factor": enrichment,
                }
            )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner — all four correlations for one window
# ─────────────────────────────────────────────────────────────────────────────


def run_all_correlations(
    feature_df: pd.DataFrame,
    active_sets: Optional[Dict[str, FrozenSet[str]]] = None,
    passive_sets: Optional[Dict[str, FrozenSet[str]]] = None,
    chains: Optional[List[str]] = None,
    volume_col: str = "total_tx_count",
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Run all four correlation analyses (6a–6d) and return the results.

    Parameters
    ----------
    feature_df   : per-address-per-chain feature table (long format)
    active_sets  : dict chain → frozenset[str] (required for 6d; optional otherwise)
    passive_sets : dict chain → frozenset[str] (required for 6d)
    chains       : list of chains to analyse; auto-detected from feature_df if None
    volume_col   : feature column to use for volume correlation (6b)

    Returns
    -------
    dict with keys ``"temporal"``, ``"volume"``, ``"frequency"``, ``"ratio"``
    (``"ratio"`` is ``None`` when active/passive sets are not provided).
    """
    chains = chains or sorted(feature_df["chain"].dropna().unique().tolist())
    logger.info("Running correlation analyses on chains: %s", chains)

    logger.info("6a — Temporal correlation")
    temporal = run_all_temporal(feature_df, chains=chains)

    logger.info("6b — Volume correlation (col=%s)", volume_col)
    volume = run_all_volume(feature_df, chains=chains, volume_col=volume_col)

    logger.info("6c — Frequency correlation")
    frequency = run_all_frequency(feature_df, chains=chains)

    ratio: Optional[pd.DataFrame] = None
    if active_sets is not None and passive_sets is not None:
        logger.info("6d — Active/Passive ratio analysis")
        ratio = active_passive_ratio_analysis(active_sets, passive_sets)
    else:
        logger.warning(
            "Skipping 6d: active_sets and passive_sets must both be provided"
        )

    return {
        "temporal": temporal,
        "volume": volume,
        "frequency": frequency,
        "ratio": ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-window summary helper
# ─────────────────────────────────────────────────────────────────────────────


def build_cross_window_correlation_summary(
    all_results: Dict[str, Dict[str, Optional[pd.DataFrame]]],
) -> Dict[str, pd.DataFrame]:
    """
    Stack correlation result DataFrames from multiple windows into combined tables.

    Parameters
    ----------
    all_results : dict  window_label → results_dict
                  (each results_dict is the return value of run_all_correlations)

    Returns
    -------
    dict with keys ``"temporal"``, ``"volume"``, ``"frequency"``, ``"ratio"``
    each containing a concatenated DataFrame with an extra ``"window"`` column.
    """
    combined: Dict[str, List[pd.DataFrame]] = {
        "temporal": [],
        "volume": [],
        "frequency": [],
        "ratio": [],
    }

    for window_label, results in all_results.items():
        for key in combined:
            df = results.get(key)
            if df is not None and not df.empty:
                df = df.copy()
                df.insert(0, "window", window_label)
                combined[key].append(df)

    return {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in combined.items()
    }
