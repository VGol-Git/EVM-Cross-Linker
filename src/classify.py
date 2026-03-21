# Classification module

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from normalize import NormalizationConfig, build_cross_chain_address_summary

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """
    Rule-based classification thresholds for cross-chain wallet behavior.
    """

    min_total_tx_across_chains: int = 3
    min_recent_30d_tx_for_active: int = 1

    dormant_threshold_days: int = 90
    dormant_recent_90d_max_tx: int = 0

    migrator_min_active_chains: int = 2
    migrator_min_total_tx_across_chains: int = 10
    migrator_min_recent_30d_total_tx: int = 3

    migrator_historical_share_min: float = 0.60
    migrator_recent_share_min: float = 0.60
    migrator_old_chain_recent_share_max: float = 0.15
    migrator_new_chain_historical_share_max: float = 0.30
    migrator_dominance_gap_min: float = 0.20

    migrator_min_signal_score: int = 5

    inactive_label: str = "inactive / insufficient activity"
    dormant_label: str = "dormant address"
    single_chain_label: str = "single-chain participant"
    multi_chain_label: str = "multi-chain user"
    migrator_label: str = "migrator-like wallet"


def load_dataframe(path: str | Path) -> pd.DataFrame:
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


def _safe_json_dict(value: Any) -> Dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(k): float(v) if pd.notna(v) else 0.0
            for k, v in value.items()
        }
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {
                    str(k): float(v) if pd.notna(v) else 0.0
                    for k, v in parsed.items()
                }
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_share_map(metric_map: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(v, 0.0) for v in metric_map.values()))
    if total <= 0:
        return {k: 0.0 for k in metric_map}
    return {k: max(v, 0.0) / total for k, v in metric_map.items()}


def _historical_map_from_total_and_recent90(
    total_map: Dict[str, float],
    recent90_map: Dict[str, float],
) -> Dict[str, float]:
    keys = set(total_map) | set(recent90_map)
    return {
        key: max(float(total_map.get(key, 0.0)) - float(recent90_map.get(key, 0.0)), 0.0)
        for key in keys
    }


def _top_chain(metric_map: Dict[str, float]) -> Tuple[Optional[str], float, Optional[str], float]:
    if not metric_map:
        return None, 0.0, None, 0.0

    items = sorted(metric_map.items(), key=lambda kv: kv[1], reverse=True)
    first_chain, first_value = items[0]
    if len(items) == 1:
        return first_chain, float(first_value), None, 0.0

    second_chain, second_value = items[1]
    return first_chain, float(first_value), second_chain, float(second_value)


def _confidence_from_score(score: int, strong_threshold: int = 6, medium_threshold: int = 4) -> str:
    if score >= strong_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_classification_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich address-level summary with derived cross-chain share metrics used by the classifier.
    """
    required_cols = [
        "chain_tx_map_json",
        "chain_recent_30d_map_json",
        "chain_recent_90d_map_json",
    ]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(
            f"Cannot derive classification features. Missing columns: {missing}"
        )

    out = summary_df.copy()

    tx_maps = out["chain_tx_map_json"].map(_safe_json_dict)
    recent30_maps = out["chain_recent_30d_map_json"].map(_safe_json_dict)
    recent90_maps = out["chain_recent_90d_map_json"].map(_safe_json_dict)

    total_share_list = []
    recent30_share_list = []
    recent90_share_list = []
    historical_map_list = []
    historical_share_list = []

    historical_dominant_share = []
    recent_dominant_share = []
    old_chain_recent_share = []
    new_chain_historical_share = []
    historical_second_share = []

    for idx, row in out.iterrows():
        total_map = tx_maps.iloc[idx]
        recent30_map = recent30_maps.iloc[idx]
        recent90_map = recent90_maps.iloc[idx]
        historical_map = _historical_map_from_total_and_recent90(total_map, recent90_map)

        total_shares = _normalize_share_map(total_map)
        recent30_shares = _normalize_share_map(recent30_map)
        recent90_shares = _normalize_share_map(recent90_map)
        historical_shares = _normalize_share_map(historical_map)

        total_share_list.append(json.dumps(total_shares, ensure_ascii=False, sort_keys=True))
        recent30_share_list.append(json.dumps(recent30_shares, ensure_ascii=False, sort_keys=True))
        recent90_share_list.append(json.dumps(recent90_shares, ensure_ascii=False, sort_keys=True))
        historical_map_list.append(json.dumps(historical_map, ensure_ascii=False, sort_keys=True))
        historical_share_list.append(json.dumps(historical_shares, ensure_ascii=False, sort_keys=True))

        hist_dom = row.get("historical_dominant_chain")
        recent_dom = row.get("recent_dominant_chain")

        hist_dom_share = float(historical_shares.get(hist_dom, 0.0)) if hist_dom else 0.0
        recent_dom_share_val = float(recent30_shares.get(recent_dom, 0.0)) if recent_dom else 0.0
        old_recent_share_val = float(recent30_shares.get(hist_dom, 0.0)) if hist_dom else 0.0
        new_hist_share_val = float(historical_shares.get(recent_dom, 0.0)) if recent_dom else 0.0

        _, _, hist_second_chain, hist_second_value = _top_chain(historical_shares)
        hist_second_share_val = float(hist_second_value) if hist_second_chain else 0.0

        historical_dominant_share.append(hist_dom_share)
        recent_dominant_share.append(recent_dom_share_val)
        old_chain_recent_share.append(old_recent_share_val)
        new_chain_historical_share.append(new_hist_share_val)
        historical_second_share.append(hist_second_share_val)

    out["chain_tx_share_map_json"] = total_share_list
    out["chain_recent_30d_share_map_json"] = recent30_share_list
    out["chain_recent_90d_share_map_json"] = recent90_share_list
    out["chain_historical_tx_map_json"] = historical_map_list
    out["chain_historical_share_map_json"] = historical_share_list

    out["historical_dominant_share"] = historical_dominant_share
    out["historical_second_share"] = historical_second_share
    out["recent_dominant_share"] = recent_dominant_share
    out["historical_chain_recent_share"] = old_chain_recent_share
    out["recent_chain_historical_share"] = new_chain_historical_share

    out["historical_dominance_gap"] = (
        out["historical_dominant_share"].fillna(0.0) - out["historical_second_share"].fillna(0.0)
    )

    return out


def classify_address_row(
    row: pd.Series,
    config: ClassificationConfig,
) -> Dict[str, Any]:
    """
    Classify one address summary row into a behavioral group.
    """
    total_tx = float(row.get("total_tx_across_chains", 0.0) or 0.0)
    recent30 = float(row.get("recent_30d_tx_across_chains", 0.0) or 0.0)
    recent90 = float(row.get("recent_90d_tx_across_chains", 0.0) or 0.0)

    active_chain_count = int(row.get("active_chain_count", 0) or 0)
    active_recent_chain_count = int(row.get("active_recent_chain_count", 0) or 0)

    min_days_since_last = row.get("min_days_since_last_active")
    min_days_since_last = float(min_days_since_last) if pd.notna(min_days_since_last) else np.nan

    dominant_chain = row.get("dominant_chain")
    recent_dominant_chain = row.get("recent_dominant_chain")
    historical_dominant_chain = row.get("historical_dominant_chain")

    dominant_chain_share = float(row.get("dominant_chain_share", 0.0) or 0.0)
    dominance_gap = float(row.get("dominance_gap", 0.0) or 0.0)

    historical_dominant_share = float(row.get("historical_dominant_share", 0.0) or 0.0)
    recent_dominant_share = float(row.get("recent_dominant_share", 0.0) or 0.0)
    historical_chain_recent_share = float(row.get("historical_chain_recent_share", 0.0) or 0.0)
    recent_chain_historical_share = float(row.get("recent_chain_historical_share", 0.0) or 0.0)
    historical_dominance_gap = float(row.get("historical_dominance_gap", 0.0) or 0.0)

    reasons: list[str] = []

    if total_tx < config.min_total_tx_across_chains:
        return {
            "behavioral_group": config.inactive_label,
            "classification_confidence": "high",
            "classification_reason": (
                f"Total observed activity across chains is below "
                f"min_total_tx_across_chains={config.min_total_tx_across_chains}."
            ),
            "migrator_signal_score": 0,
            "is_dormant": False,
            "is_single_chain": False,
            "is_multi_chain": False,
            "is_migrator_like": False,
        }

    dormant_like = (
        pd.notna(min_days_since_last)
        and min_days_since_last > config.dormant_threshold_days
        and recent90 <= config.dormant_recent_90d_max_tx
    )
    if dormant_like:
        return {
            "behavioral_group": config.dormant_label,
            "classification_confidence": "high",
            "classification_reason": (
                f"No meaningful recent activity in the last 90 days and "
                f"min_days_since_last_active={min_days_since_last:.1f} > "
                f"dormant_threshold_days={config.dormant_threshold_days}."
            ),
            "migrator_signal_score": 0,
            "is_dormant": True,
            "is_single_chain": False,
            "is_multi_chain": False,
            "is_migrator_like": False,
        }

    if active_chain_count == 1:
        chain_name = row.get("active_chains", "") or dominant_chain or "unknown"
        return {
            "behavioral_group": config.single_chain_label,
            "classification_confidence": "high",
            "classification_reason": (
                f"Meaningful activity detected on exactly one chain: {chain_name}."
            ),
            "migrator_signal_score": 0,
            "is_dormant": False,
            "is_single_chain": True,
            "is_multi_chain": False,
            "is_migrator_like": False,
        }

    # Migrator-like scoring
    signal_score = 0

    if active_chain_count >= config.migrator_min_active_chains:
        signal_score += 1
        reasons.append("active on at least two chains")

    if total_tx >= config.migrator_min_total_tx_across_chains:
        signal_score += 1
        reasons.append("enough total cross-chain activity")

    if recent30 >= config.migrator_min_recent_30d_total_tx:
        signal_score += 1
        reasons.append("enough recent activity")

    if historical_dominant_chain and recent_dominant_chain and historical_dominant_chain != recent_dominant_chain:
        signal_score += 1
        reasons.append("historical dominant chain differs from recent dominant chain")

    if historical_dominant_share >= config.migrator_historical_share_min:
        signal_score += 1
        reasons.append(
            f"historical dominant share={historical_dominant_share:.2f} "
            f">= {config.migrator_historical_share_min:.2f}"
        )

    if recent_dominant_share >= config.migrator_recent_share_min:
        signal_score += 1
        reasons.append(
            f"recent dominant share={recent_dominant_share:.2f} "
            f">= {config.migrator_recent_share_min:.2f}"
        )

    if historical_chain_recent_share <= config.migrator_old_chain_recent_share_max:
        signal_score += 1
        reasons.append(
            f"old dominant chain recent share={historical_chain_recent_share:.2f} "
            f"<= {config.migrator_old_chain_recent_share_max:.2f}"
        )

    if recent_chain_historical_share <= config.migrator_new_chain_historical_share_max:
        signal_score += 1
        reasons.append(
            f"new dominant chain historical share={recent_chain_historical_share:.2f} "
            f"<= {config.migrator_new_chain_historical_share_max:.2f}"
        )

    if dominance_gap >= config.migrator_dominance_gap_min:
        signal_score += 1
        reasons.append(
            f"current dominance gap={dominance_gap:.2f} "
            f">= {config.migrator_dominance_gap_min:.2f}"
        )

    if historical_dominance_gap >= config.migrator_dominance_gap_min:
        signal_score += 1
        reasons.append(
            f"historical dominance gap={historical_dominance_gap:.2f} "
            f">= {config.migrator_dominance_gap_min:.2f}"
        )

    if signal_score >= config.migrator_min_signal_score:
        return {
            "behavioral_group": config.migrator_label,
            "classification_confidence": _confidence_from_score(signal_score),
            "classification_reason": "; ".join(reasons),
            "migrator_signal_score": signal_score,
            "is_dormant": False,
            "is_single_chain": False,
            "is_multi_chain": True,
            "is_migrator_like": True,
        }

    # Default multi-chain bucket
    multi_chain_reason = (
        f"Meaningful activity on {active_chain_count} chains; "
        f"recently active on {active_recent_chain_count} chains; "
        f"migrator signal score={signal_score}."
    )
    return {
        "behavioral_group": config.multi_chain_label,
        "classification_confidence": "high" if active_chain_count >= 3 else "medium",
        "classification_reason": multi_chain_reason,
        "migrator_signal_score": signal_score,
        "is_dormant": False,
        "is_single_chain": False,
        "is_multi_chain": True,
        "is_migrator_like": False,
    }


def classify_address_summary(
    summary_df: pd.DataFrame,
    config: Optional[ClassificationConfig] = None,
) -> pd.DataFrame:
    """
    Classify an address-level summary table into behavioral groups.
    """
    config = config or ClassificationConfig()

    out = summary_df.copy()
    numeric_cols = [
        "total_tx_across_chains",
        "recent_30d_tx_across_chains",
        "recent_90d_tx_across_chains",
        "active_chain_count",
        "active_recent_chain_count",
        "dominant_chain_share",
        "dominance_gap",
        "min_days_since_last_active",
        "max_days_since_last_active",
        "mean_activity_score",
    ]
    out = _coerce_numeric(out, numeric_cols)
    out = add_classification_features(out)

    classified_rows = []
    for _, row in out.iterrows():
        classified_rows.append(classify_address_row(row, config))

    classified_df = pd.concat(
        [out.reset_index(drop=True), pd.DataFrame(classified_rows)],
        axis=1,
    )

    classified_df["behavioral_group"] = classified_df["behavioral_group"].astype(str)
    classified_df["classification_confidence"] = classified_df["classification_confidence"].astype(str)

    return classified_df


def classify_from_normalized_df(
    normalized_df: pd.DataFrame,
    *,
    normalization_config: Optional[NormalizationConfig] = None,
    classification_config: Optional[ClassificationConfig] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper:
    normalized chain-level table -> address-level summary -> classified address table
    """
    summary_df = build_cross_chain_address_summary(
        normalized_df,
        config=normalization_config or NormalizationConfig(),
    )
    return classify_address_summary(
        summary_df,
        config=classification_config or ClassificationConfig(),
    )


def behavioral_group_counts(classified_df: pd.DataFrame) -> pd.DataFrame:
    if "behavioral_group" not in classified_df.columns:
        raise ValueError("Column 'behavioral_group' not found")

    counts = (
        classified_df["behavioral_group"]
        .value_counts(dropna=False)
        .rename_axis("behavioral_group")
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts