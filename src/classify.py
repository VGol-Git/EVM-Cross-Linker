# classify.py
# Active / Passive classification and cross-chain matching layer
# for the block-window version of the project.

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .config import AppConfig, ChainConfig
from .sampling import (
    address_code_snapshot_path_base,
    address_observation_table_path_base,
    enriched_transaction_table_path_base,
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class ClassificationConfig:
    """
    Configuration for address-level classification inside one chain/window.
    """

    min_nonce_for_active: int = 1
    require_eoa: bool = True
    allow_unknown_eoa: bool = False

    present_label: str = "present"
    active_label: str = "active"
    passive_label: str = "passive"
    active_and_receiving_label: str = "active_and_receiving"
    unclassified_present_label: str = "present_unclassified"
    absent_label: str = "absent"


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


def address_status_table_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.interim_status_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_address_status"
    )


def presence_matrix_path_base(
    app_config: AppConfig,
    window_blocks: int,
    status_name: str,
) -> Path:
    return (
        app_config.paths.processed_analysis_dir
        / f"window_{window_blocks}_{status_name}_presence_matrix"
    )


def pairwise_overlap_table_path_base(
    app_config: AppConfig,
    window_blocks: int,
    overlap_name: str,
) -> Path:
    return (
        app_config.paths.processed_analysis_dir
        / f"window_{window_blocks}_{overlap_name}_pairwise_overlap"
    )


def triple_overlap_table_path_base(
    app_config: AppConfig,
    window_blocks: int,
    overlap_name: str,
) -> Path:
    return (
        app_config.paths.processed_analysis_dir
        / f"window_{window_blocks}_{overlap_name}_triple_overlap"
    )


# ============================================================
# Basic coercion helpers
# ============================================================


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


def _safe_address(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if not value.startswith("0x"):
        return None
    if len(value) != 42:
        return None
    return value


def _update_first_last_seen(
    state_row: Dict[str, Any],
    block_number: Optional[int],
    block_timestamp: Optional[int],
) -> None:
    if block_number is None:
        return

    current_first_block = state_row["first_seen_block"]
    current_last_block = state_row["last_seen_block"]

    if current_first_block is None or block_number < current_first_block:
        state_row["first_seen_block"] = block_number
        state_row["first_seen_timestamp"] = block_timestamp

    if current_last_block is None or block_number > current_last_block:
        state_row["last_seen_block"] = block_number
        state_row["last_seen_timestamp"] = block_timestamp


def _is_eoa_allowed(
    is_eoa: Optional[bool],
    *,
    require_eoa: bool,
    allow_unknown_eoa: bool,
) -> bool:
    if not require_eoa:
        return True
    if is_eoa is True:
        return True
    if is_eoa is None and allow_unknown_eoa:
        return True
    return False


# ============================================================
# Load chain/window inputs
# ============================================================


def load_chain_window_inputs(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        transactions_enriched_df, address_observations_df, address_code_snapshot_df
    """
    tx_df = _load_table_as_dataframe(
        path_without_suffix=enriched_transaction_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )

    obs_df = _load_table_as_dataframe(
        path_without_suffix=address_observation_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )

    code_df = _load_table_as_dataframe(
        path_without_suffix=address_code_snapshot_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )

    return tx_df, obs_df, code_df


# ============================================================
# Address status construction
# ============================================================


def build_address_status_table(
    transactions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    address_code_df: pd.DataFrame,
    *,
    config: Optional[ClassificationConfig] = None,
) -> pd.DataFrame:
    """
    Build one row per address for one specific chain and one specific block window.

    Output columns include:
    - is_present
    - is_active
    - is_passive
    - sent/received counts
    - sent/received value
    - nonce summary
    - first/last seen
    - unique_counterparties
    """
    config = config or ClassificationConfig()

    state: Dict[str, Dict[str, Any]] = {}

    def ensure(address: str) -> Dict[str, Any]:
        if address not in state:
            state[address] = {
                "chain": None,
                "chain_id": None,
                "window_blocks": None,
                "reference_block_number": None,
                "address": address,
                "is_eoa": None,
                "seen_as_from": False,
                "seen_as_to": False,
                "sent_tx_count": 0,
                "received_tx_count": 0,
                "value_sent_wei": 0,
                "value_received_wei": 0,
                "sender_nonce_min": None,
                "sender_nonce_max": None,
                "first_seen_block": None,
                "last_seen_block": None,
                "first_seen_timestamp": None,
                "last_seen_timestamp": None,
                "counterparties": set(),
            }
        return state[address]

    # --------------------------------------------------------
    # Step 1. Address code snapshot → authoritative EOA flag
    # --------------------------------------------------------
    if not address_code_df.empty:
        for _, row in address_code_df.iterrows():
            address = _safe_address(row.get("address"))
            if address is None:
                continue

            status = ensure(address)
            status["chain"] = row.get("chain", status["chain"])
            status["chain_id"] = _to_optional_int(row.get("chain_id")) or status["chain_id"]
            status["is_eoa"] = _to_optional_bool(row.get("is_eoa"))

    # --------------------------------------------------------
    # Step 2. Observations → presence / first-last seen fallback
    # --------------------------------------------------------
    if not observations_df.empty:
        for _, row in observations_df.iterrows():
            address = _safe_address(row.get("address"))
            if address is None:
                continue

            status = ensure(address)

            status["chain"] = row.get("chain", status["chain"])
            status["chain_id"] = _to_optional_int(row.get("chain_id")) or status["chain_id"]
            status["window_blocks"] = (
                _to_optional_int(row.get("window_blocks")) or status["window_blocks"]
            )
            status["reference_block_number"] = (
                _to_optional_int(row.get("reference_block_number"))
                or status["reference_block_number"]
            )

            status["seen_as_from"] = status["seen_as_from"] or _to_bool(row.get("seen_as_from"))
            status["seen_as_to"] = status["seen_as_to"] or _to_bool(row.get("seen_as_to"))

            # status["sent_tx_count"] = max(
            #     status["sent_tx_count"],
            #     _to_int(row.get("tx_count_as_from"), default=0),
            # )
            # status["received_tx_count"] = max(
            #     status["received_tx_count"],
            #     _to_int(row.get("tx_count_as_to"), default=0),
            # )

            first_seen_block = _to_optional_int(row.get("first_seen_block"))
            last_seen_block = _to_optional_int(row.get("last_seen_block"))
            first_seen_ts = _to_optional_int(row.get("first_seen_timestamp"))
            last_seen_ts = _to_optional_int(row.get("last_seen_timestamp"))

            if first_seen_block is not None:
                _update_first_last_seen(status, first_seen_block, first_seen_ts)
            if last_seen_block is not None:
                _update_first_last_seen(status, last_seen_block, last_seen_ts)

    # --------------------------------------------------------
    # Step 3. Enriched transactions → sender/receiver metrics
    # --------------------------------------------------------
    if not transactions_df.empty:
        for _, row in transactions_df.iterrows():
            chain = row.get("chain")
            chain_id = _to_optional_int(row.get("chain_id"))
            window_blocks = _to_optional_int(row.get("window_blocks"))
            reference_block_number = _to_optional_int(row.get("reference_block_number"))

            block_number = _to_optional_int(row.get("block_number"))
            block_timestamp = _to_optional_int(row.get("block_timestamp"))
            nonce = _to_int(row.get("nonce"), default=0)
            value_wei = _to_int(row.get("value_wei"), default=0)

            from_address = _safe_address(row.get("from_address"))
            to_address = _safe_address(row.get("to_address"))

            from_is_eoa = _to_optional_bool(row.get("from_is_eoa"))
            to_is_eoa = _to_optional_bool(row.get("to_is_eoa"))

            # Sender side
            if from_address is not None and _is_eoa_allowed(
                from_is_eoa,
                require_eoa=config.require_eoa,
                allow_unknown_eoa=config.allow_unknown_eoa,
            ):
                status = ensure(from_address)
                status["chain"] = chain or status["chain"]
                status["chain_id"] = chain_id or status["chain_id"]
                status["window_blocks"] = window_blocks or status["window_blocks"]
                status["reference_block_number"] = (
                    reference_block_number or status["reference_block_number"]
                )
                if status["is_eoa"] is None:
                    status["is_eoa"] = from_is_eoa

                status["seen_as_from"] = True
                status["sent_tx_count"] += 1
                status["value_sent_wei"] += value_wei

                current_nonce_min = status["sender_nonce_min"]
                current_nonce_max = status["sender_nonce_max"]

                if current_nonce_min is None or nonce < current_nonce_min:
                    status["sender_nonce_min"] = nonce
                if current_nonce_max is None or nonce > current_nonce_max:
                    status["sender_nonce_max"] = nonce

                _update_first_last_seen(status, block_number, block_timestamp)

                if to_address is not None:
                    status["counterparties"].add(to_address)

            # Receiver side
            if to_address is not None and _is_eoa_allowed(
                to_is_eoa,
                require_eoa=config.require_eoa,
                allow_unknown_eoa=config.allow_unknown_eoa,
            ):
                status = ensure(to_address)
                status["chain"] = chain or status["chain"]
                status["chain_id"] = chain_id or status["chain_id"]
                status["window_blocks"] = window_blocks or status["window_blocks"]
                status["reference_block_number"] = (
                    reference_block_number or status["reference_block_number"]
                )
                if status["is_eoa"] is None:
                    status["is_eoa"] = to_is_eoa

                status["seen_as_to"] = True
                status["received_tx_count"] += 1
                status["value_received_wei"] += value_wei

                _update_first_last_seen(status, block_number, block_timestamp)

                if from_address is not None:
                    status["counterparties"].add(from_address)

    # --------------------------------------------------------
    # Step 4. Final row construction + classification flags
    # --------------------------------------------------------
    rows: List[Dict[str, Any]] = []

    for address, payload in sorted(state.items()):
        seen_as_from = bool(payload["seen_as_from"])
        seen_as_to = bool(payload["seen_as_to"])
        is_eoa = payload["is_eoa"]

        is_present = (
            (seen_as_from or seen_as_to)
            and _is_eoa_allowed(
                is_eoa,
                require_eoa=config.require_eoa,
                allow_unknown_eoa=config.allow_unknown_eoa,
            )
        )

        sender_nonce_max = payload["sender_nonce_max"]
        is_active = bool(
            is_present
            and seen_as_from
            and sender_nonce_max is not None
            and sender_nonce_max >= config.min_nonce_for_active
        )
        is_passive = bool(is_present and seen_as_to and not seen_as_from)

        if not is_present:
            participation_label = config.absent_label
        elif is_active and seen_as_to:
            participation_label = config.active_and_receiving_label
        elif is_active:
            participation_label = config.active_label
        elif is_passive:
            participation_label = config.passive_label
        else:
            participation_label = config.unclassified_present_label

        sent_tx_count = int(payload["sent_tx_count"])
        received_tx_count = int(payload["received_tx_count"])
        total_tx_count = sent_tx_count + received_tx_count

        value_sent_wei = int(payload["value_sent_wei"])
        value_received_wei = int(payload["value_received_wei"])

        avg_sent_value_wei = (
            value_sent_wei / sent_tx_count if sent_tx_count > 0 else 0.0
        )
        avg_received_value_wei = (
            value_received_wei / received_tx_count if received_tx_count > 0 else 0.0
        )

        rows.append(
            {
                "chain": payload["chain"],
                "chain_id": payload["chain_id"],
                "window_blocks": payload["window_blocks"],
                "reference_block_number": payload["reference_block_number"],
                "address": address,
                "is_eoa": is_eoa,
                "is_present": is_present,
                "is_active": is_active,
                "is_passive": is_passive,
                "seen_as_from": seen_as_from,
                "seen_as_to": seen_as_to,
                "participation_label": participation_label,
                "sent_tx_count": sent_tx_count,
                "received_tx_count": received_tx_count,
                "total_tx_count": total_tx_count,
                "value_sent_wei": value_sent_wei,
                "value_received_wei": value_received_wei,
                "avg_sent_value_wei": avg_sent_value_wei,
                "avg_received_value_wei": avg_received_value_wei,
                "sender_nonce_min": payload["sender_nonce_min"],
                "sender_nonce_max": payload["sender_nonce_max"],
                "first_seen_block": payload["first_seen_block"],
                "last_seen_block": payload["last_seen_block"],
                "first_seen_timestamp": payload["first_seen_timestamp"],
                "last_seen_timestamp": payload["last_seen_timestamp"],
                "unique_counterparties": len(payload["counterparties"]),
                "tx_frequency_per_block": (
                    total_tx_count / payload["window_blocks"]
                    if payload["window_blocks"]
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Single-chain classification runner
# ============================================================


def classify_chain_window(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    *,
    config: Optional[ClassificationConfig] = None,
    save_output: bool = True,
) -> pd.DataFrame:
    config = config or ClassificationConfig()

    tx_df, obs_df, code_df = load_chain_window_inputs(
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
    )

    status_df = build_address_status_table(
        transactions_df=tx_df,
        observations_df=obs_df,
        address_code_df=code_df,
        config=config,
    )

    if save_output:
        _save_dataframe(
            df=status_df,
            path_without_suffix=address_status_table_path_base(
                app_config=app_config,
                chain=chain,
                window_blocks=window_blocks,
            ),
            table_format=app_config.storage.table_format,
        )

    logger.info(
        "[%s][window=%s] address_status rows=%s active=%s passive=%s present=%s",
        chain.name,
        window_blocks,
        len(status_df),
        int(status_df["is_active"].sum()) if "is_active" in status_df.columns else 0,
        int(status_df["is_passive"].sum()) if "is_passive" in status_df.columns else 0,
        int(status_df["is_present"].sum()) if "is_present" in status_df.columns else 0,
    )

    return status_df


def load_address_status_table(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> pd.DataFrame:
    return _load_table_as_dataframe(
        path_without_suffix=address_status_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


# ============================================================
# Set builders
# ============================================================


def build_status_sets(status_df: pd.DataFrame) -> Dict[str, Set[str]]:
    if status_df.empty:
        return {"present": set(), "active": set(), "passive": set()}

    required = {"address", "is_present", "is_active", "is_passive"}
    missing = required - set(status_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in status_df: {sorted(missing)}")

    address_series = status_df["address"].astype(str).str.lower()

    return {
        "present": set(address_series[status_df["is_present"].fillna(False)]),
        "active": set(address_series[status_df["is_active"].fillna(False)]),
        "passive": set(address_series[status_df["is_passive"].fillna(False)]),
    }


def build_presence_matrix(
    status_tables_by_chain: Dict[str, pd.DataFrame],
    *,
    status_column: str,
) -> pd.DataFrame:
    """
    Builds an address-by-chain boolean matrix for one status type:
    is_present / is_active / is_passive
    """
    frames: List[pd.DataFrame] = []

    for chain_name, df in status_tables_by_chain.items():
        if df.empty:
            continue
        if "address" not in df.columns or status_column not in df.columns:
            raise ValueError(
                f"Chain={chain_name}: missing columns 'address' and/or {status_column!r}"
            )

        tmp = df[["address", status_column]].copy()
        tmp["address"] = tmp["address"].astype(str).str.lower()
        tmp = tmp.rename(columns={status_column: chain_name})
        tmp[chain_name] = tmp[chain_name].fillna(False).astype(bool)
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="address", how="outer")

    chain_columns = [col for col in out.columns if col != "address"]
    out[chain_columns] = out[chain_columns].fillna(False).astype(bool)
    out = out.sort_values("address").reset_index(drop=True)
    return out


def compute_pairwise_overlap_table(
    presence_df: pd.DataFrame,
    *,
    status_name: str,
) -> pd.DataFrame:
    """
    Pairwise set metrics for a boolean address-by-chain matrix.
    """
    if presence_df.empty:
        return pd.DataFrame(
            columns=[
                "status_name",
                "source_chain",
                "target_chain",
                "source_size",
                "target_size",
                "intersection_size",
                "union_size",
                "jaccard",
                "overlap_of_source",
                "overlap_of_target",
            ]
        )

    chains = [col for col in presence_df.columns if col != "address"]
    rows: List[Dict[str, Any]] = []

    for source_chain, target_chain in itertools.product(chains, chains):
        source_mask = presence_df[source_chain].astype(bool)
        target_mask = presence_df[target_chain].astype(bool)

        source_size = int(source_mask.sum())
        target_size = int(target_mask.sum())
        intersection_size = int((source_mask & target_mask).sum())
        union_size = int((source_mask | target_mask).sum())

        rows.append(
            {
                "status_name": status_name,
                "source_chain": source_chain,
                "target_chain": target_chain,
                "source_size": source_size,
                "target_size": target_size,
                "intersection_size": intersection_size,
                "union_size": union_size,
                "jaccard": (intersection_size / union_size) if union_size > 0 else 0.0,
                "overlap_of_source": (
                    intersection_size / source_size if source_size > 0 else 0.0
                ),
                "overlap_of_target": (
                    intersection_size / target_size if target_size > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_triple_overlap_table(
    presence_df: pd.DataFrame,
    *,
    status_name: str,
) -> pd.DataFrame:
    if presence_df.empty:
        return pd.DataFrame(
            columns=[
                "status_name",
                "chain_a",
                "chain_b",
                "chain_c",
                "intersection_size",
            ]
        )

    chains = [col for col in presence_df.columns if col != "address"]
    if len(chains) < 3:
        return pd.DataFrame(
            columns=[
                "status_name",
                "chain_a",
                "chain_b",
                "chain_c",
                "intersection_size",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for chain_a, chain_b, chain_c in itertools.combinations(chains, 3):
        mask = (
            presence_df[chain_a].astype(bool)
            & presence_df[chain_b].astype(bool)
            & presence_df[chain_c].astype(bool)
        )
        rows.append(
            {
                "status_name": status_name,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "chain_c": chain_c,
                "intersection_size": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def compute_mixed_overlap_table(
    status_tables_by_chain: Dict[str, pd.DataFrame],
    *,
    left_status_column: str = "is_active",
    right_status_column: str = "is_passive",
    left_status_name: str = "active",
    right_status_name: str = "passive",
) -> pd.DataFrame:
    """
    Mixed intersections like:
    active on Chain A ∩ passive on Chain B
    """
    sets_by_chain: Dict[str, Tuple[Set[str], Set[str]]] = {}

    for chain_name, df in status_tables_by_chain.items():
        if df.empty:
            sets_by_chain[chain_name] = (set(), set())
            continue

        required = {"address", left_status_column, right_status_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Chain={chain_name}: missing required columns {sorted(missing)}"
            )

        address_series = df["address"].astype(str).str.lower()
        left_set = set(address_series[df[left_status_column].fillna(False)])
        right_set = set(address_series[df[right_status_column].fillna(False)])
        sets_by_chain[chain_name] = (left_set, right_set)

    rows: List[Dict[str, Any]] = []
    for source_chain, target_chain in itertools.product(sets_by_chain.keys(), repeat=2):
        source_set, _ = sets_by_chain[source_chain]
        _, target_set = sets_by_chain[target_chain]

        intersection_size = len(source_set & target_set)
        source_size = len(source_set)
        target_size = len(target_set)
        union_size = len(source_set | target_set)

        rows.append(
            {
                "left_status_name": left_status_name,
                "right_status_name": right_status_name,
                "source_chain": source_chain,
                "target_chain": target_chain,
                "source_size": source_size,
                "target_size": target_size,
                "intersection_size": intersection_size,
                "union_size": union_size,
                "jaccard": intersection_size / union_size if union_size > 0 else 0.0,
                "overlap_of_source": (
                    intersection_size / source_size if source_size > 0 else 0.0
                ),
                "overlap_of_target": (
                    intersection_size / target_size if target_size > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Cross-chain matching orchestration
# ============================================================


def run_cross_chain_matching_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    save_output: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Loads per-chain address status tables for one window and computes:
    - present presence matrix
    - active presence matrix
    - passive presence matrix
    - pairwise overlaps for all three
    - triple overlaps for all three
    - mixed active/passive overlaps
    """
    status_tables_by_chain: Dict[str, pd.DataFrame] = {}
    for chain in app_config.enabled_chains:
        status_tables_by_chain[chain.name] = load_address_status_table(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        )

    present_presence = build_presence_matrix(
        status_tables_by_chain=status_tables_by_chain,
        status_column="is_present",
    )
    active_presence = build_presence_matrix(
        status_tables_by_chain=status_tables_by_chain,
        status_column="is_active",
    )
    passive_presence = build_presence_matrix(
        status_tables_by_chain=status_tables_by_chain,
        status_column="is_passive",
    )

    present_pairwise = compute_pairwise_overlap_table(
        present_presence,
        status_name="present",
    )
    active_pairwise = compute_pairwise_overlap_table(
        active_presence,
        status_name="active",
    )
    passive_pairwise = compute_pairwise_overlap_table(
        passive_presence,
        status_name="passive",
    )

    present_triple = compute_triple_overlap_table(
        present_presence,
        status_name="present",
    )
    active_triple = compute_triple_overlap_table(
        active_presence,
        status_name="active",
    )
    passive_triple = compute_triple_overlap_table(
        passive_presence,
        status_name="passive",
    )

    mixed_active_passive = compute_mixed_overlap_table(
        status_tables_by_chain=status_tables_by_chain,
        left_status_column="is_active",
        right_status_column="is_passive",
        left_status_name="active",
        right_status_name="passive",
    )

    outputs = {
        "present_presence": present_presence,
        "active_presence": active_presence,
        "passive_presence": passive_presence,
        "present_pairwise": present_pairwise,
        "active_pairwise": active_pairwise,
        "passive_pairwise": passive_pairwise,
        "present_triple": present_triple,
        "active_triple": active_triple,
        "passive_triple": passive_triple,
        "mixed_active_passive": mixed_active_passive,
    }

    if save_output:
        _save_dataframe(
            present_presence,
            presence_matrix_path_base(app_config, window_blocks, "present"),
            app_config.storage.table_format,
        )
        _save_dataframe(
            active_presence,
            presence_matrix_path_base(app_config, window_blocks, "active"),
            app_config.storage.table_format,
        )
        _save_dataframe(
            passive_presence,
            presence_matrix_path_base(app_config, window_blocks, "passive"),
            app_config.storage.table_format,
        )

        _save_dataframe(
            present_pairwise,
            pairwise_overlap_table_path_base(app_config, window_blocks, "present"),
            app_config.storage.table_format,
        )
        _save_dataframe(
            active_pairwise,
            pairwise_overlap_table_path_base(app_config, window_blocks, "active"),
            app_config.storage.table_format,
        )
        _save_dataframe(
            passive_pairwise,
            pairwise_overlap_table_path_base(app_config, window_blocks, "passive"),
            app_config.storage.table_format,
        )

        _save_dataframe(
            present_triple,
            triple_overlap_table_path_base(app_config, window_blocks, "present"),
            app_config.storage.table_format,
        )
        _save_dataframe(
            active_triple,
            triple_overlap_table_path_base(app_config, window_blocks, "active"),
            app_config.storage.table_format,
        )
        _save_dataframe(
            passive_triple,
            triple_overlap_table_path_base(app_config, window_blocks, "passive"),
            app_config.storage.table_format,
        )

        _save_dataframe(
            mixed_active_passive,
            pairwise_overlap_table_path_base(
                app_config,
                window_blocks,
                "mixed_active_passive",
            ),
            app_config.storage.table_format,
        )

    return outputs


# ============================================================
# Full-layer orchestration
# ============================================================


def run_classification_for_all_chains_and_windows(
    app_config: AppConfig,
    *,
    config: Optional[ClassificationConfig] = None,
    save_output: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Runs:
    1. per-chain address classification
    2. cross-chain matching
    for all enabled chains and all configured block windows.

    Returns:
    {
        1: {
            "status_tables": {"ethereum": df, ...},
            "matching": {...},
        },
        10: {...},
        100: {...},
    }
    """
    config = config or ClassificationConfig()
    results: Dict[int, Dict[str, Any]] = {}

    for window_blocks in app_config.sampling.windows.block_counts:
        status_tables: Dict[str, pd.DataFrame] = {}

        for chain in app_config.enabled_chains:
            status_df = classify_chain_window(
                app_config=app_config,
                chain=chain,
                window_blocks=window_blocks,
                config=config,
                save_output=save_output,
            )
            status_tables[chain.name] = status_df

        matching = run_cross_chain_matching_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            save_output=save_output,
        )

        results[window_blocks] = {
            "status_tables": status_tables,
            "matching": matching,
        }

    return results