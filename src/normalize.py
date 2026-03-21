# Data normalization module

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    log_transform_columns: List[str] = field(
        default_factory=lambda: [
            "native_tx_count",
            "erc20_transfer_count",
            "internal_tx_count",
            "total_tx_count",
            "active_days",
            "active_weeks",
            "distinct_counterparties",
            "distinct_outgoing_counterparties",
            "distinct_incoming_counterparties",
            "distinct_contracts",
            "recent_7d_tx_count",
            "recent_30d_tx_count",
            "recent_90d_tx_count",
        ]
    )
    percentile_columns: List[str] = field(
        default_factory=lambda: [
            "total_tx_count",
            "active_days",
            "active_weeks",
            "distinct_counterparties",
            "distinct_contracts",
            "recent_30d_tx_count",
            "recent_90d_tx_count",
            "activity_density",
            "contract_interaction_ratio",
            "outgoing_ratio",
        ]
    )
    zscore_columns: List[str] = field(
        default_factory=lambda: [
            "total_tx_count",
            "active_days",
            "distinct_counterparties",
            "distinct_contracts",
            "recent_30d_tx_count",
            "recent_90d_tx_count",
            "activity_density",
            "days_since_last_active",
            "median_gap_seconds",
        ]
    )
    reverse_percentile_columns: List[str] = field(
        default_factory=lambda: [
            "days_since_last_active",
            "median_gap_seconds",
            "mean_gap_seconds",
        ]
    )
    reverse_zscore_columns: List[str] = field(
        default_factory=lambda: [
            "days_since_last_active",
            "median_gap_seconds",
        ]
    )
    activity_score_components: Dict[str, float] = field(
        default_factory=lambda: {
            "pct_total_tx_count": 0.25,
            "pct_recent_30d_tx_count": 0.25,
            "pct_active_days": 0.20,
            "pct_distinct_counterparties": 0.15,
            "pct_distinct_contracts": 0.10,
            "pct_activity_density": 0.05,
        }
    )
    active_chain_min_total_tx: int = 3
    active_chain_recent_tx: int = 1
    dormant_threshold_days: int = 90
    robust_zscore_eps: float = 1e-9


def load_feature_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Supported formats: .csv, .parquet")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Supported output formats: .csv, .parquet")


def _ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _robust_zscore(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.median(skipna=True)
    mad = (s - median).abs().median(skipna=True)

    if pd.isna(median) or pd.isna(mad) or mad < eps:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    return 0.67448975 * (s - median) / (mad + eps)


def _standard_zscore(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)

    if pd.isna(mean) or pd.isna(std) or std < eps:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    return (s - mean) / (std + eps)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_candidates = [
        "native_tx_count",
        "erc20_transfer_count",
        "internal_tx_count",
        "total_tx_count",
        "active_days",
        "active_weeks",
        "activity_density",
        "distinct_counterparties",
        "distinct_outgoing_counterparties",
        "distinct_incoming_counterparties",
        "distinct_contracts",
        "outgoing_ratio",
        "contract_interaction_ratio",
        "erc20_ratio",
        "internal_ratio",
        "mean_gap_seconds",
        "median_gap_seconds",
        "mean_gas_used",
        "median_gas_used",
        "mean_gas_price_wei",
        "median_gas_price_wei",
        "recent_7d_tx_count",
        "recent_30d_tx_count",
        "recent_90d_tx_count",
        "days_since_last_active",
        "lifetime_days",
        "observation_window_days",
    ]
    out = _ensure_numeric(out, numeric_candidates)

    out["active_days_share_of_lifetime"] = np.where(
        out["lifetime_days"].fillna(0) > 0,
        out["active_days"] / out["lifetime_days"].clip(lower=1.0),
        0.0,
    )

    out["recent_30d_share"] = np.where(
        out["total_tx_count"].fillna(0) > 0,
        out["recent_30d_tx_count"] / out["total_tx_count"],
        0.0,
    )

    out["recent_90d_share"] = np.where(
        out["total_tx_count"].fillna(0) > 0,
        out["recent_90d_tx_count"] / out["total_tx_count"],
        0.0,
    )

    out["contract_diversity_density"] = np.where(
        out["active_days"].fillna(0) > 0,
        out["distinct_contracts"] / out["active_days"].clip(lower=1.0),
        0.0,
    )

    out["counterparty_diversity_density"] = np.where(
        out["active_days"].fillna(0) > 0,
        out["distinct_counterparties"] / out["active_days"].clip(lower=1.0),
        0.0,
    )

    out["txs_per_active_day"] = np.where(
        out["active_days"].fillna(0) > 0,
        out["total_tx_count"] / out["active_days"].clip(lower=1.0),
        0.0,
    )

    return out


def add_log_transforms(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"log_{col}"] = np.log1p(pd.to_numeric(out[col], errors="coerce").fillna(0))
    return out


def add_within_chain_percentiles(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    reverse_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    reverse_columns = set(reverse_columns or [])

    if "chain" not in out.columns:
        raise ValueError("Input DataFrame must contain a 'chain' column")

    for col in columns:
        if col not in out.columns:
            continue

        s = pd.to_numeric(out[col], errors="coerce")
        ranks = out.assign(_v=s).groupby("chain")["_v"].rank(pct=True, method="average")

        if col in reverse_columns:
            ranks = 1.0 - ranks

        out[f"pct_{col}"] = ranks.fillna(0.0)

    return out


def add_within_chain_zscores(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    reverse_columns: Optional[Sequence[str]] = None,
    robust: bool = True,
    eps: float = 1e-9,
) -> pd.DataFrame:
    out = df.copy()
    reverse_columns = set(reverse_columns or [])

    if "chain" not in out.columns:
        raise ValueError("Input DataFrame must contain a 'chain' column")

    for col in columns:
        if col not in out.columns:
            continue

        z_name = f"z_{col}"

        def _group_transform(s: pd.Series) -> pd.Series:
            base = _robust_zscore(s, eps=eps) if robust else _standard_zscore(s, eps=eps)
            return -base if col in reverse_columns else base

        out[z_name] = out.groupby("chain")[col].transform(_group_transform).fillna(0.0)

    return out


def add_activity_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    out = df.copy()

    missing = [col for col in weights if col not in out.columns]
    if missing:
        raise ValueError(
            f"Cannot compute activity score. Missing normalized columns: {missing}"
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of activity score weights must be > 0")

    score = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    for col, w in weights.items():
        score = score + out[col].fillna(0.0) * w

    out["activity_score"] = score / total_weight
    return out


def normalize_feature_table(
    df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
) -> pd.DataFrame:
    """
    Normalize chain-level features so they become comparable across networks.

    Output columns include:
    - derived ratios
    - log_* columns
    - pct_* within-chain percentiles
    - z_* within-chain z-scores
    - activity_score
    """
    config = config or NormalizationConfig()

    out = df.copy()
    out = add_derived_columns(out)
    out = add_log_transforms(out, config.log_transform_columns)

    percentile_cols = list(dict.fromkeys(
        list(config.percentile_columns) + list(config.reverse_percentile_columns)
    ))
    out = add_within_chain_percentiles(
        out,
        percentile_cols,
        reverse_columns=config.reverse_percentile_columns,
    )

    zscore_cols = list(dict.fromkeys(config.zscore_columns))
    out = add_within_chain_zscores(
        out,
        zscore_cols,
        reverse_columns=config.reverse_zscore_columns,
        robust=True,
        eps=config.robust_zscore_eps,
    )

    out = add_activity_score(out, config.activity_score_components)
    return out


def _top_two_chains_by_metric(group: pd.DataFrame, metric: str) -> tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    if metric not in group.columns:
        return None, None, None, None

    tmp = group[["chain", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0.0)
    tmp = tmp.sort_values(metric, ascending=False).reset_index(drop=True)

    if len(tmp) == 0:
        return None, None, None, None

    first_chain = str(tmp.loc[0, "chain"])
    first_value = float(tmp.loc[0, metric])

    if len(tmp) == 1:
        return first_chain, first_value, None, None

    second_chain = str(tmp.loc[1, "chain"])
    second_value = float(tmp.loc[1, metric])

    return first_chain, first_value, second_chain, second_value


def _json_map_from_group(group: pd.DataFrame, metric: str) -> str:
    if metric not in group.columns:
        return "{}"
    m = {
        str(row["chain"]): float(pd.to_numeric(row[metric], errors="coerce") or 0.0)
        for _, row in group.iterrows()
    }
    return json.dumps(m, ensure_ascii=False, sort_keys=True)


def build_cross_chain_address_summary(
    normalized_df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
) -> pd.DataFrame:
    """
    Collapse chain-level normalized features into one row per address.

    This summary is what classify.py will typically consume.
    """
    config = config or NormalizationConfig()

    required_cols = {"address", "chain", "total_tx_count", "recent_30d_tx_count", "recent_90d_tx_count"}
    missing = required_cols - set(normalized_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for summary: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []

    for address, group in normalized_df.groupby("address", dropna=False):
        g = group.copy()

        g["total_tx_count"] = pd.to_numeric(g["total_tx_count"], errors="coerce").fillna(0.0)
        g["recent_30d_tx_count"] = pd.to_numeric(g["recent_30d_tx_count"], errors="coerce").fillna(0.0)
        g["recent_90d_tx_count"] = pd.to_numeric(g["recent_90d_tx_count"], errors="coerce").fillna(0.0)
        g["days_since_last_active"] = pd.to_numeric(
            g.get("days_since_last_active"), errors="coerce"
        )

        total_tx_across_chains = float(g["total_tx_count"].sum())
        recent_30d_tx_across_chains = float(g["recent_30d_tx_count"].sum())
        recent_90d_tx_across_chains = float(g["recent_90d_tx_count"].sum())

        active_mask = g["total_tx_count"] >= config.active_chain_min_total_tx
        active_recent_mask = g["recent_30d_tx_count"] >= config.active_chain_recent_tx

        active_chain_count = int(active_mask.sum())
        active_recent_chain_count = int(active_recent_mask.sum())

        active_chains = g.loc[active_mask, "chain"].astype(str).tolist()
        recent_active_chains = g.loc[active_recent_mask, "chain"].astype(str).tolist()

        dom_chain, dom_value, second_chain, second_value = _top_two_chains_by_metric(g, "total_tx_count")
        recent_dom_chain, recent_dom_value, _, _ = _top_two_chains_by_metric(g, "recent_30d_tx_count")

        g["historical_tx_ex_recent_90d"] = (g["total_tx_count"] - g["recent_90d_tx_count"]).clip(lower=0.0)
        hist_dom_chain, hist_dom_value, _, _ = _top_two_chains_by_metric(g, "historical_tx_ex_recent_90d")

        if total_tx_across_chains > 0 and dom_value is not None:
            dominant_chain_share = dom_value / total_tx_across_chains
            second_chain_share = (second_value or 0.0) / total_tx_across_chains
        else:
            dominant_chain_share = 0.0
            second_chain_share = 0.0

        dominance_gap = dominant_chain_share - second_chain_share

        min_days_since_last = float(g["days_since_last_active"].min()) if g["days_since_last_active"].notna().any() else np.nan
        max_days_since_last = float(g["days_since_last_active"].max()) if g["days_since_last_active"].notna().any() else np.nan

        dormant_like = bool(
            pd.notna(min_days_since_last)
            and min_days_since_last > config.dormant_threshold_days
            and total_tx_across_chains > 0
        )

        mean_activity_score = float(pd.to_numeric(g.get("activity_score"), errors="coerce").fillna(0.0).mean())

        row = {
            "address": address,
            "chain_rows": int(len(g)),
            "total_tx_across_chains": total_tx_across_chains,
            "recent_30d_tx_across_chains": recent_30d_tx_across_chains,
            "recent_90d_tx_across_chains": recent_90d_tx_across_chains,
            "active_chain_count": active_chain_count,
            "active_recent_chain_count": active_recent_chain_count,
            "active_chains": "|".join(sorted(active_chains)),
            "recent_active_chains": "|".join(sorted(recent_active_chains)),
            "dominant_chain": dom_chain,
            "dominant_chain_share": dominant_chain_share,
            "second_chain": second_chain,
            "second_chain_share": second_chain_share,
            "dominance_gap": dominance_gap,
            "recent_dominant_chain": recent_dom_chain,
            "historical_dominant_chain": hist_dom_chain,
            "min_days_since_last_active": min_days_since_last,
            "max_days_since_last_active": max_days_since_last,
            "dormant_like_flag": dormant_like,
            "mean_activity_score": mean_activity_score,
            "chain_tx_map_json": _json_map_from_group(g, "total_tx_count"),
            "chain_recent_30d_map_json": _json_map_from_group(g, "recent_30d_tx_count"),
            "chain_recent_90d_map_json": _json_map_from_group(g, "recent_90d_tx_count"),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df["is_single_chain"] = summary_df["active_chain_count"] == 1
        summary_df["is_multi_chain"] = summary_df["active_chain_count"] >= 2

    return summary_df