# features.py
# Feature extraction for overlapping EOA addresses across EVM chains
# in exact block windows (1 / 10 / 100 blocks).

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .classify import address_status_table_path_base
from .config import AppConfig, ChainConfig
from .sampling import (
    enriched_transaction_table_path_base,
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)


# ============================================================
# Data models
# ============================================================


@dataclass(frozen=True)
class AddressNetworkFeatureRecord:
    chain: str
    chain_id: int
    window_blocks: int
    reference_block_number: int
    address: str

    is_eoa: Optional[bool]
    is_present: bool
    is_active: bool
    is_passive: bool
    participation_label: str

    sent_tx_count: int
    received_tx_count: int
    total_tx_count: int

    value_sent_wei: int
    value_received_wei: int
    total_value_wei: int

    avg_sent_value_wei: float
    avg_received_value_wei: float
    avg_total_tx_value_wei: float

    unique_counterparties: int
    unique_outgoing_counterparties: int
    unique_incoming_counterparties: int

    first_activity_timestamp: Optional[int]
    last_activity_timestamp: Optional[int]
    first_sent_timestamp: Optional[int]
    last_sent_timestamp: Optional[int]
    first_received_timestamp: Optional[int]
    last_received_timestamp: Optional[int]

    first_activity_block: Optional[int]
    last_activity_block: Optional[int]
    first_sent_block: Optional[int]
    last_sent_block: Optional[int]
    first_received_block: Optional[int]
    last_received_block: Optional[int]

    active_days: int
    calendar_day_span: int
    tx_frequency_per_day: float
    sent_tx_frequency_per_day: float
    received_tx_frequency_per_day: float
    tx_frequency_per_block: float

    min_nonce_sent: Optional[int]
    max_nonce_sent: Optional[int]

    total_gas_limit: int
    avg_gas_limit: float
    total_gas_price_wei: int
    avg_gas_price_wei: float

    daily_tx_count_series_json: str
    daily_sent_tx_count_series_json: str
    daily_received_tx_count_series_json: str
    daily_value_sent_wei_series_json: str
    daily_value_received_wei_series_json: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Helpers
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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


def _safe_address(value: Any) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    value = value.strip().lower()
    if not value.startswith("0x") or len(value) != 42:
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


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "None", "null", "nan"}:
            return default
        return int(float(value))
    return int(value)


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "None", "null", "nan"}:
        return None
    return _to_int(value)


def _json_dumps_sorted(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _date_key_from_ts(timestamp: int) -> str:
    return datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")


def _calendar_day_span(first_ts: Optional[int], last_ts: Optional[int]) -> int:
    if first_ts is None or last_ts is None:
        return 0
    first_date = datetime.utcfromtimestamp(first_ts).date()
    last_date = datetime.utcfromtimestamp(last_ts).date()
    return (last_date - first_date).days + 1


def _safe_frequency(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator > 0 else 0.0


# ============================================================
# Path helpers
# ============================================================


def address_feature_table_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.processed_features_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_address_features"
    )


def overlapping_feature_table_path_base(
    app_config: AppConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.processed_features_dir
        / f"window_{window_blocks}_overlapping_address_features"
    )


def pairwise_feature_alignment_path_base(
    app_config: AppConfig,
    window_blocks: int,
    chain_a: str,
    chain_b: str,
) -> Path:
    return (
        app_config.paths.processed_features_dir
        / f"window_{window_blocks}_{chain_a}_vs_{chain_b}_feature_alignment"
    )


def window_feature_summary_path_base(
    app_config: AppConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.processed_features_dir
        / f"window_{window_blocks}_feature_summary"
    )


# ============================================================
# Load chain/window inputs
# ============================================================


def load_chain_window_feature_inputs(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        address_status_df, enriched_transactions_df
    """
    status_df = _load_table_as_dataframe(
        path_without_suffix=address_status_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )

    tx_df = _load_table_as_dataframe(
        path_without_suffix=enriched_transaction_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )

    return status_df, tx_df


# ============================================================
# Core feature extraction
# ============================================================


def _prepare_transactions_for_feature_build(tx_df: pd.DataFrame) -> pd.DataFrame:
    if tx_df.empty:
        return tx_df.copy()

    df = tx_df.copy()

    for col in [
        "chain_id",
        "window_blocks",
        "reference_block_number",
        "block_number",
        "block_timestamp",
        "value_wei",
        "nonce",
        "gas",
        "gas_price_wei",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(_to_optional_int if col in {"block_timestamp", "gas", "gas_price_wei"} else _to_int)

    for col in ["from_address", "to_address"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_address)

    for col in ["from_is_eoa", "to_is_eoa"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_optional_bool)

    return df


def _build_feature_row_for_address(
    address: str,
    status_row: pd.Series,
    tx_df: pd.DataFrame,
) -> Dict[str, Any]:
    address = address.lower()

    sent_df = tx_df[tx_df["from_address"] == address].copy()
    received_df = tx_df[tx_df["to_address"] == address].copy()
    related_df = pd.concat([sent_df, received_df], axis=0, ignore_index=True)

    sent_tx_count = int(len(sent_df))
    received_tx_count = int(len(received_df))
    total_tx_count = sent_tx_count + received_tx_count

    value_sent_wei = int(sent_df["value_wei"].fillna(0).sum()) if not sent_df.empty else 0
    value_received_wei = (
        int(received_df["value_wei"].fillna(0).sum()) if not received_df.empty else 0
    )
    total_value_wei = value_sent_wei + value_received_wei

    avg_sent_value_wei = float(value_sent_wei / sent_tx_count) if sent_tx_count > 0 else 0.0
    avg_received_value_wei = (
        float(value_received_wei / received_tx_count) if received_tx_count > 0 else 0.0
    )
    avg_total_tx_value_wei = (
        float(total_value_wei / total_tx_count) if total_tx_count > 0 else 0.0
    )

    first_activity_timestamp = (
        int(related_df["block_timestamp"].dropna().min())
        if not related_df.empty and related_df["block_timestamp"].notna().any()
        else None
    )
    last_activity_timestamp = (
        int(related_df["block_timestamp"].dropna().max())
        if not related_df.empty and related_df["block_timestamp"].notna().any()
        else None
    )

    first_sent_timestamp = (
        int(sent_df["block_timestamp"].dropna().min())
        if not sent_df.empty and sent_df["block_timestamp"].notna().any()
        else None
    )
    last_sent_timestamp = (
        int(sent_df["block_timestamp"].dropna().max())
        if not sent_df.empty and sent_df["block_timestamp"].notna().any()
        else None
    )

    first_received_timestamp = (
        int(received_df["block_timestamp"].dropna().min())
        if not received_df.empty and received_df["block_timestamp"].notna().any()
        else None
    )
    last_received_timestamp = (
        int(received_df["block_timestamp"].dropna().max())
        if not received_df.empty and received_df["block_timestamp"].notna().any()
        else None
    )

    first_activity_block = (
        int(related_df["block_number"].dropna().min())
        if not related_df.empty and related_df["block_number"].notna().any()
        else None
    )
    last_activity_block = (
        int(related_df["block_number"].dropna().max())
        if not related_df.empty and related_df["block_number"].notna().any()
        else None
    )

    first_sent_block = (
        int(sent_df["block_number"].dropna().min())
        if not sent_df.empty and sent_df["block_number"].notna().any()
        else None
    )
    last_sent_block = (
        int(sent_df["block_number"].dropna().max())
        if not sent_df.empty and sent_df["block_number"].notna().any()
        else None
    )

    first_received_block = (
        int(received_df["block_number"].dropna().min())
        if not received_df.empty and received_df["block_number"].notna().any()
        else None
    )
    last_received_block = (
        int(received_df["block_number"].dropna().max())
        if not received_df.empty and received_df["block_number"].notna().any()
        else None
    )

    outgoing_counterparties = set(
        value for value in sent_df["to_address"].dropna().astype(str).tolist() if value
    )
    incoming_counterparties = set(
        value for value in received_df["from_address"].dropna().astype(str).tolist() if value
    )
    counterparties = outgoing_counterparties | incoming_counterparties

    active_days = 0
    calendar_day_span = _calendar_day_span(first_activity_timestamp, last_activity_timestamp)

    daily_tx_count: Dict[str, int] = {}
    daily_sent_tx_count: Dict[str, int] = {}
    daily_received_tx_count: Dict[str, int] = {}
    daily_value_sent_wei: Dict[str, int] = {}
    daily_value_received_wei: Dict[str, int] = {}

    if not related_df.empty and related_df["block_timestamp"].notna().any():
        all_dates = [
            _date_key_from_ts(int(ts))
            for ts in related_df["block_timestamp"].dropna().tolist()
        ]
        active_days = len(set(all_dates))

    if not sent_df.empty:
        for _, row in sent_df.iterrows():
            ts = _to_optional_int(row.get("block_timestamp"))
            if ts is None:
                continue
            day = _date_key_from_ts(ts)
            daily_sent_tx_count[day] = daily_sent_tx_count.get(day, 0) + 1
            daily_tx_count[day] = daily_tx_count.get(day, 0) + 1
            daily_value_sent_wei[day] = (
                daily_value_sent_wei.get(day, 0) + _to_int(row.get("value_wei"), default=0)
            )

    if not received_df.empty:
        for _, row in received_df.iterrows():
            ts = _to_optional_int(row.get("block_timestamp"))
            if ts is None:
                continue
            day = _date_key_from_ts(ts)
            daily_received_tx_count[day] = daily_received_tx_count.get(day, 0) + 1
            daily_tx_count[day] = daily_tx_count.get(day, 0) + 1
            daily_value_received_wei[day] = (
                daily_value_received_wei.get(day, 0)
                + _to_int(row.get("value_wei"), default=0)
            )

    min_nonce_sent = (
        int(sent_df["nonce"].dropna().min())
        if not sent_df.empty and sent_df["nonce"].notna().any()
        else None
    )
    max_nonce_sent = (
        int(sent_df["nonce"].dropna().max())
        if not sent_df.empty and sent_df["nonce"].notna().any()
        else None
    )

    total_gas_limit = int(sent_df["gas"].fillna(0).sum()) if not sent_df.empty else 0
    avg_gas_limit = float(sent_df["gas"].fillna(0).mean()) if not sent_df.empty else 0.0

    total_gas_price_wei = (
        int(sent_df["gas_price_wei"].fillna(0).sum()) if not sent_df.empty else 0
    )
    avg_gas_price_wei = (
        float(sent_df["gas_price_wei"].fillna(0).mean()) if not sent_df.empty else 0.0
    )

    window_blocks = _to_int(status_row.get("window_blocks"), default=0)

    return AddressNetworkFeatureRecord(
        chain=str(status_row.get("chain")),
        chain_id=_to_int(status_row.get("chain_id"), default=0),
        window_blocks=window_blocks,
        reference_block_number=_to_int(status_row.get("reference_block_number"), default=0),
        address=address,

        is_eoa=_to_optional_bool(status_row.get("is_eoa")),
        is_present=_to_bool(status_row.get("is_present")),
        is_active=_to_bool(status_row.get("is_active")),
        is_passive=_to_bool(status_row.get("is_passive")),
        participation_label=str(status_row.get("participation_label") or ""),

        sent_tx_count=sent_tx_count,
        received_tx_count=received_tx_count,
        total_tx_count=total_tx_count,

        value_sent_wei=value_sent_wei,
        value_received_wei=value_received_wei,
        total_value_wei=total_value_wei,

        avg_sent_value_wei=avg_sent_value_wei,
        avg_received_value_wei=avg_received_value_wei,
        avg_total_tx_value_wei=avg_total_tx_value_wei,

        unique_counterparties=len(counterparties),
        unique_outgoing_counterparties=len(outgoing_counterparties),
        unique_incoming_counterparties=len(incoming_counterparties),

        first_activity_timestamp=first_activity_timestamp,
        last_activity_timestamp=last_activity_timestamp,
        first_sent_timestamp=first_sent_timestamp,
        last_sent_timestamp=last_sent_timestamp,
        first_received_timestamp=first_received_timestamp,
        last_received_timestamp=last_received_timestamp,

        first_activity_block=first_activity_block,
        last_activity_block=last_activity_block,
        first_sent_block=first_sent_block,
        last_sent_block=last_sent_block,
        first_received_block=first_received_block,
        last_received_block=last_received_block,

        active_days=active_days,
        calendar_day_span=calendar_day_span,
        tx_frequency_per_day=_safe_frequency(total_tx_count, calendar_day_span),
        sent_tx_frequency_per_day=_safe_frequency(sent_tx_count, calendar_day_span),
        received_tx_frequency_per_day=_safe_frequency(received_tx_count, calendar_day_span),
        tx_frequency_per_block=(
            float(total_tx_count / window_blocks) if window_blocks > 0 else 0.0
        ),

        min_nonce_sent=min_nonce_sent,
        max_nonce_sent=max_nonce_sent,

        total_gas_limit=total_gas_limit,
        avg_gas_limit=avg_gas_limit,
        total_gas_price_wei=total_gas_price_wei,
        avg_gas_price_wei=avg_gas_price_wei,

        daily_tx_count_series_json=_json_dumps_sorted(daily_tx_count),
        daily_sent_tx_count_series_json=_json_dumps_sorted(daily_sent_tx_count),
        daily_received_tx_count_series_json=_json_dumps_sorted(daily_received_tx_count),
        daily_value_sent_wei_series_json=_json_dumps_sorted(daily_value_sent_wei),
        daily_value_received_wei_series_json=_json_dumps_sorted(daily_value_received_wei),
    ).to_dict()


def build_address_network_feature_table(
    status_df: pd.DataFrame,
    tx_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one feature row per address for one chain/window.

    The input status_df must already be address-level output from classify.py.
    The input tx_df must be enriched raw transactions for the same chain/window.
    """
    if status_df.empty:
        return pd.DataFrame()

    prepared_tx_df = _prepare_transactions_for_feature_build(tx_df)
    rows: List[Dict[str, Any]] = []

    for _, status_row in status_df.iterrows():
        address = _safe_address(status_row.get("address"))
        if address is None:
            continue

        rows.append(
            _build_feature_row_for_address(
                address=address,
                status_row=status_row,
                tx_df=prepared_tx_df,
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# Single-chain runner
# ============================================================


def build_features_for_chain_window(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    *,
    save_output: bool = True,
) -> pd.DataFrame:
    status_df, tx_df = load_chain_window_feature_inputs(
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
    )

    feature_df = build_address_network_feature_table(
        status_df=status_df,
        tx_df=tx_df,
    )

    if save_output:
        _save_dataframe(
            feature_df,
            address_feature_table_path_base(app_config, chain, window_blocks),
            app_config.storage.table_format,
        )

    logger.info(
        "[%s][window=%s] feature rows=%s present=%s active=%s passive=%s",
        chain.name,
        window_blocks,
        len(feature_df),
        int(feature_df["is_present"].sum()) if "is_present" in feature_df.columns else 0,
        int(feature_df["is_active"].sum()) if "is_active" in feature_df.columns else 0,
        int(feature_df["is_passive"].sum()) if "is_passive" in feature_df.columns else 0,
    )

    return feature_df


def load_features_for_chain_window(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=address_feature_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


# ============================================================
# Overlapping-address feature tables
# ============================================================


def build_overlapping_feature_table(
    feature_tables_by_chain: Dict[str, pd.DataFrame],
    *,
    min_networks_present: int = 2,
) -> pd.DataFrame:
    """
    Combine all per-chain feature tables into one long table,
    but keep only addresses present on at least `min_networks_present` chains.
    """
    non_empty = [df.copy() for df in feature_tables_by_chain.values() if not df.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, axis=0, ignore_index=True)

    if "address" not in combined.columns or "is_present" not in combined.columns:
        raise ValueError("Combined feature table must contain 'address' and 'is_present'.")

    combined["address"] = combined["address"].astype(str).str.lower()
    present_df = combined[combined["is_present"].fillna(False)].copy()

    network_presence = (
        present_df.groupby("address")["chain"]
        .nunique()
        .reset_index(name="present_network_count")
    )

    active_presence = (
        present_df[present_df["is_active"].fillna(False)]
        .groupby("address")["chain"]
        .nunique()
        .reset_index(name="active_network_count")
    )

    passive_presence = (
        present_df[present_df["is_passive"].fillna(False)]
        .groupby("address")["chain"]
        .nunique()
        .reset_index(name="passive_network_count")
    )

    combined = combined.merge(network_presence, on="address", how="left")
    combined = combined.merge(active_presence, on="address", how="left")
    combined = combined.merge(passive_presence, on="address", how="left")

    combined["present_network_count"] = combined["present_network_count"].fillna(0).astype(int)
    combined["active_network_count"] = combined["active_network_count"].fillna(0).astype(int)
    combined["passive_network_count"] = combined["passive_network_count"].fillna(0).astype(int)

    overlapping = combined[combined["present_network_count"] >= min_networks_present].copy()
    overlapping = overlapping.sort_values(["address", "chain"]).reset_index(drop=True)
    return overlapping


def save_overlapping_feature_table(
    app_config: AppConfig,
    window_blocks: int,
    overlapping_df: pd.DataFrame,
) -> Path:
    return _save_dataframe(
        overlapping_df,
        overlapping_feature_table_path_base(app_config, window_blocks),
        app_config.storage.table_format,
    )


# ============================================================
# Pairwise chain alignment for correlations
# ============================================================


def build_pairwise_feature_alignment(
    overlapping_df: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    active_only: bool = False,
) -> pd.DataFrame:
    """
    Build one row per overlapping address for a specific chain pair.

    This table is designed for:
    - Pearson/Spearman on value_sent / frequency / counterparty counts
    - Δ first activity timestamp
    - scatter plots
    """
    if overlapping_df.empty:
        return pd.DataFrame()

    df = overlapping_df.copy()
    df["address"] = df["address"].astype(str).str.lower()

    if active_only:
        df = df[df["is_active"].fillna(False)].copy()

    left = df[df["chain"] == chain_a].copy()
    right = df[df["chain"] == chain_b].copy()

    if left.empty or right.empty:
        return pd.DataFrame()

    left_cols = {
        "is_present": f"{chain_a}_is_present",
        "is_active": f"{chain_a}_is_active",
        "is_passive": f"{chain_a}_is_passive",
        "sent_tx_count": f"{chain_a}_sent_tx_count",
        "received_tx_count": f"{chain_a}_received_tx_count",
        "total_tx_count": f"{chain_a}_total_tx_count",
        "value_sent_wei": f"{chain_a}_value_sent_wei",
        "value_received_wei": f"{chain_a}_value_received_wei",
        "total_value_wei": f"{chain_a}_total_value_wei",
        "avg_sent_value_wei": f"{chain_a}_avg_sent_value_wei",
        "avg_received_value_wei": f"{chain_a}_avg_received_value_wei",
        "avg_total_tx_value_wei": f"{chain_a}_avg_total_tx_value_wei",
        "unique_counterparties": f"{chain_a}_unique_counterparties",
        "unique_outgoing_counterparties": f"{chain_a}_unique_outgoing_counterparties",
        "unique_incoming_counterparties": f"{chain_a}_unique_incoming_counterparties",
        "first_activity_timestamp": f"{chain_a}_first_activity_timestamp",
        "last_activity_timestamp": f"{chain_a}_last_activity_timestamp",
        "tx_frequency_per_day": f"{chain_a}_tx_frequency_per_day",
        "sent_tx_frequency_per_day": f"{chain_a}_sent_tx_frequency_per_day",
        "received_tx_frequency_per_day": f"{chain_a}_received_tx_frequency_per_day",
        "tx_frequency_per_block": f"{chain_a}_tx_frequency_per_block",
        "daily_tx_count_series_json": f"{chain_a}_daily_tx_count_series_json",
        "daily_value_sent_wei_series_json": f"{chain_a}_daily_value_sent_wei_series_json",
        "daily_value_received_wei_series_json": f"{chain_a}_daily_value_received_wei_series_json",
    }

    right_cols = {
        "is_present": f"{chain_b}_is_present",
        "is_active": f"{chain_b}_is_active",
        "is_passive": f"{chain_b}_is_passive",
        "sent_tx_count": f"{chain_b}_sent_tx_count",
        "received_tx_count": f"{chain_b}_received_tx_count",
        "total_tx_count": f"{chain_b}_total_tx_count",
        "value_sent_wei": f"{chain_b}_value_sent_wei",
        "value_received_wei": f"{chain_b}_value_received_wei",
        "total_value_wei": f"{chain_b}_total_value_wei",
        "avg_sent_value_wei": f"{chain_b}_avg_sent_value_wei",
        "avg_received_value_wei": f"{chain_b}_avg_received_value_wei",
        "avg_total_tx_value_wei": f"{chain_b}_avg_total_tx_value_wei",
        "unique_counterparties": f"{chain_b}_unique_counterparties",
        "unique_outgoing_counterparties": f"{chain_b}_unique_outgoing_counterparties",
        "unique_incoming_counterparties": f"{chain_b}_unique_incoming_counterparties",
        "first_activity_timestamp": f"{chain_b}_first_activity_timestamp",
        "last_activity_timestamp": f"{chain_b}_last_activity_timestamp",
        "tx_frequency_per_day": f"{chain_b}_tx_frequency_per_day",
        "sent_tx_frequency_per_day": f"{chain_b}_sent_tx_frequency_per_day",
        "received_tx_frequency_per_day": f"{chain_b}_received_tx_frequency_per_day",
        "tx_frequency_per_block": f"{chain_b}_tx_frequency_per_block",
        "daily_tx_count_series_json": f"{chain_b}_daily_tx_count_series_json",
        "daily_value_sent_wei_series_json": f"{chain_b}_daily_value_sent_wei_series_json",
        "daily_value_received_wei_series_json": f"{chain_b}_daily_value_received_wei_series_json",
    }

    left = left[["address", "window_blocks", *left_cols.keys()]].rename(columns=left_cols)
    right = right[["address", "window_blocks", *right_cols.keys()]].rename(columns=right_cols)

    merged = left.merge(
        right,
        on=["address", "window_blocks"],
        how="inner",
    )

    if merged.empty:
        return merged

    merged["chain_a"] = chain_a
    merged["chain_b"] = chain_b
    merged["first_activity_delta_seconds"] = (
        merged[f"{chain_b}_first_activity_timestamp"]
        - merged[f"{chain_a}_first_activity_timestamp"]
    )
    merged["absolute_first_activity_delta_seconds"] = (
        merged["first_activity_delta_seconds"].abs()
    )

    return merged.sort_values("address").reset_index(drop=True)


def save_pairwise_feature_alignment(
    app_config: AppConfig,
    window_blocks: int,
    chain_a: str,
    chain_b: str,
    alignment_df: pd.DataFrame,
) -> Path:
    return _save_dataframe(
        alignment_df,
        pairwise_feature_alignment_path_base(app_config, window_blocks, chain_a, chain_b),
        app_config.storage.table_format,
    )


# ============================================================
# Window-level orchestration
# ============================================================


def build_features_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Full feature stage for one observation window.

    Produces:
    - per-chain feature tables
    - one overlapping-address long table
    - pairwise aligned feature tables for each enabled chain pair
    """
    feature_tables_by_chain: Dict[str, pd.DataFrame] = {}

    for chain in app_config.enabled_chains:
        feature_df = build_features_for_chain_window(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
            save_output=save_output,
        )
        feature_tables_by_chain[chain.name] = feature_df

    overlapping_df = build_overlapping_feature_table(
        feature_tables_by_chain=feature_tables_by_chain,
        min_networks_present=2,
    )

    if save_output:
        save_overlapping_feature_table(
            app_config=app_config,
            window_blocks=window_blocks,
            overlapping_df=overlapping_df,
        )

    pairwise_alignments: Dict[Tuple[str, str], pd.DataFrame] = {}
    chain_names = [chain.name for chain in app_config.enabled_chains]

    for i, chain_a in enumerate(chain_names):
        for chain_b in chain_names[i + 1 :]:
            aligned_df = build_pairwise_feature_alignment(
                overlapping_df=overlapping_df,
                chain_a=chain_a,
                chain_b=chain_b,
                active_only=False,
            )
            pairwise_alignments[(chain_a, chain_b)] = aligned_df

            if save_output:
                save_pairwise_feature_alignment(
                    app_config=app_config,
                    window_blocks=window_blocks,
                    chain_a=chain_a,
                    chain_b=chain_b,
                    alignment_df=aligned_df,
                )

    summary_rows = []
    for chain_name, df in feature_tables_by_chain.items():
        summary_rows.append(
            {
                "window_blocks": window_blocks,
                "chain": chain_name,
                "feature_rows": int(len(df)),
                "present_addresses": int(df["is_present"].sum()) if not df.empty else 0,
                "active_addresses": int(df["is_active"].sum()) if not df.empty else 0,
                "passive_addresses": int(df["is_passive"].sum()) if not df.empty else 0,
            }
        )

    summary_rows.append(
        {
            "window_blocks": window_blocks,
            "chain": "__overlapping__",
            "feature_rows": int(len(overlapping_df)),
            "present_addresses": int(overlapping_df["address"].nunique()) if not overlapping_df.empty else 0,
            "active_addresses": int(
                overlapping_df[overlapping_df["is_active"].fillna(False)]["address"].nunique()
            ) if not overlapping_df.empty else 0,
            "passive_addresses": int(
                overlapping_df[overlapping_df["is_passive"].fillna(False)]["address"].nunique()
            ) if not overlapping_df.empty else 0,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    if save_output:
        _save_dataframe(
            summary_df,
            window_feature_summary_path_base(app_config, window_blocks),
            app_config.storage.table_format,
        )

    logger.info(
        "[window=%s] overlapping unique addresses=%s pairwise_tables=%s",
        window_blocks,
        int(overlapping_df["address"].nunique()) if not overlapping_df.empty else 0,
        len(pairwise_alignments),
    )

    return {
        "feature_tables_by_chain": feature_tables_by_chain,
        "overlapping_feature_table": overlapping_df,
        "pairwise_alignments": pairwise_alignments,
        "summary": summary_df,
    }


def build_features_for_all_windows(
    app_config: AppConfig,
    *,
    save_output: bool = True,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for window_blocks in app_config.sampling.windows.block_counts:
        results[window_blocks] = build_features_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            save_output=save_output,
        )
    return results