# Feature extraction module

from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from .api_client import EtherscanClient
from .config import AppConfig, ChainConfig

logger = logging.getLogger(__name__)


@dataclass
class AddressChainFeatureRecord:
    address: str
    chain: str
    chain_id: int
    snapshot_ts: int
    observation_window_days: int

    native_tx_count: int
    erc20_transfer_count: int
    internal_tx_count: int
    total_tx_count: int

    first_seen_ts: Optional[int]
    last_seen_ts: Optional[int]
    lifetime_days: Optional[float]
    days_since_last_active: Optional[float]

    active_days: int
    active_weeks: int
    activity_density: float

    distinct_counterparties: int
    distinct_outgoing_counterparties: int
    distinct_incoming_counterparties: int
    distinct_contracts: int

    outgoing_ratio: float
    contract_interaction_ratio: float
    erc20_ratio: float
    internal_ratio: float

    mean_gap_seconds: Optional[float]
    median_gap_seconds: Optional[float]

    mean_gas_used: Optional[float]
    median_gas_used: Optional[float]
    mean_gas_price_wei: Optional[float]
    median_gas_price_wei: Optional[float]

    recent_7d_tx_count: int
    recent_30d_tx_count: int
    recent_90d_tx_count: int

    has_activity: bool
    is_query_truncated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_address(address: str | None) -> Optional[str]:
    if not address:
        return None
    address = address.strip().lower()
    if not address.startswith("0x") or len(address) != 42:
        return None
    return address


def _extract_timestamp(tx: Dict[str, Any]) -> Optional[int]:
    for key in ("timeStamp", "timestamp"):
        if key in tx:
            ts = _safe_int(tx.get(key))
            if ts is not None:
                return ts
    return None


def _extract_counterparty(address: str, tx: Dict[str, Any]) -> Optional[str]:
    tx_from = _sanitize_address(tx.get("from"))
    tx_to = _sanitize_address(tx.get("to"))

    if tx_from == address and tx_to:
        return tx_to
    if tx_to == address and tx_from:
        return tx_from
    return None


def _is_outgoing(address: str, tx: Dict[str, Any]) -> bool:
    tx_from = _sanitize_address(tx.get("from"))
    return tx_from == address


def _is_contract_interaction(tx: Dict[str, Any]) -> bool:
    input_data = tx.get("input")
    if isinstance(input_data, str) and input_data not in {"", "0x", "0x0"}:
        return True
    method_id = tx.get("methodId")
    if isinstance(method_id, str) and method_id not in {"", "0x", "0x0"}:
        return True
    function_name = tx.get("functionName")
    if isinstance(function_name, str) and function_name.strip():
        return True
    return False


def _compute_gap_stats(timestamps: Sequence[int]) -> tuple[Optional[float], Optional[float]]:
    if len(timestamps) < 2:
        return None, None
    ts_sorted = sorted(timestamps)
    gaps = [b - a for a, b in zip(ts_sorted[:-1], ts_sorted[1:]) if b >= a]
    if not gaps:
        return None, None
    return float(sum(gaps) / len(gaps)), float(statistics.median(gaps))


def _count_recent_txs(timestamps: Sequence[int], snapshot_ts: int, days: int) -> int:
    threshold = snapshot_ts - days * 24 * 60 * 60
    return sum(1 for ts in timestamps if ts >= threshold)


def _distinct_calendar_days(timestamps: Sequence[int]) -> int:
    if not timestamps:
        return 0
    return len({datetime.utcfromtimestamp(ts).date() for ts in timestamps})


def _distinct_calendar_weeks(timestamps: Sequence[int]) -> int:
    if not timestamps:
        return 0
    return len(
        {
            datetime.utcfromtimestamp(ts).isocalendar()[:2]
            for ts in timestamps
        }
    )


def load_address_list_from_csv(path: str | Path, address_col: str = "address") -> List[str]:
    df = pd.read_csv(path)
    if address_col not in df.columns:
        raise ValueError(f"Column '{address_col}' not found in {path}")
    addresses = (
        df[address_col]
        .astype(str)
        .map(_sanitize_address)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return addresses


def _address_activity_cache_path(app_config: AppConfig, chain: ChainConfig, address: str) -> Path:
    return app_config.paths.raw_dir / "address_activity" / chain.name / f"{address}.json"


def fetch_address_chain_activity(
    client: EtherscanClient,
    app_config: AppConfig,
    chain: ChainConfig,
    address: str,
    *,
    refresh: bool = False,
    max_pages_per_endpoint: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch raw address activity for one address on one chain and cache it to disk.

    Cached structure:
    {
      "address": "...",
      "chain": "...",
      "chain_id": ...,
      "snapshot_ts": ...,
      "max_pages_per_endpoint": ...,
      "is_query_truncated": bool,
      "endpoints": {
        "normal": {"count": ..., "rows": [...]},
        "erc20": {"count": ..., "rows": [...]},
        "internal": {"count": ..., "rows": [...]}
      }
    }
    """
    address = _sanitize_address(address)
    if address is None:
        raise ValueError("Invalid EVM address")

    cache_path = _address_activity_cache_path(app_config, chain, address)
    if app_config.sampling.resume_from_cache and cache_path.exists() and not refresh:
        logger.info("[%s] Loading cached address activity for %s", chain.name, address)
        return read_json(cache_path)

    logger.info("[%s] Fetching address activity for %s", chain.name, address)

    normal_txs = client.get_all_normal_transactions(
        chain=chain,
        address=address,
        max_pages=max_pages_per_endpoint,
    )
    erc20_txs = client.get_all_erc20_transfers(
        chain=chain,
        address=address,
        max_pages=max_pages_per_endpoint,
    )
    internal_txs = client.get_all_internal_transactions(
        chain=chain,
        address=address,
        max_pages=max_pages_per_endpoint,
    )

    is_query_truncated = max_pages_per_endpoint is not None

    payload = {
        "address": address,
        "chain": chain.name,
        "chain_id": chain.chain_id,
        "fetched_at": utc_now_iso(),
        "snapshot_ts": int(datetime.now(tz=timezone.utc).timestamp()),
        "observation_window_days": app_config.sampling.observation_window_days,
        "max_pages_per_endpoint": max_pages_per_endpoint,
        "is_query_truncated": is_query_truncated,
        "endpoints": {
            "normal": {
                "count": len(normal_txs),
                "rows": normal_txs,
            },
            "erc20": {
                "count": len(erc20_txs),
                "rows": erc20_txs,
            },
            "internal": {
                "count": len(internal_txs),
                "rows": internal_txs,
            },
        },
    }

    write_json(cache_path, payload)
    return payload


def _extract_feature_record_from_payload(payload: Dict[str, Any]) -> AddressChainFeatureRecord:
    address = _sanitize_address(payload["address"])
    if address is None:
        raise ValueError("Payload contains invalid address")

    chain = str(payload["chain"])
    chain_id = int(payload["chain_id"])
    snapshot_ts = int(payload["snapshot_ts"])
    observation_window_days = int(payload.get("observation_window_days", 0))
    is_query_truncated = bool(payload.get("is_query_truncated", False))

    endpoints = payload.get("endpoints", {})
    normal_txs = endpoints.get("normal", {}).get("rows", []) or []
    erc20_txs = endpoints.get("erc20", {}).get("rows", []) or []
    internal_txs = endpoints.get("internal", {}).get("rows", []) or []

    native_tx_count = len(normal_txs)
    erc20_transfer_count = len(erc20_txs)
    internal_tx_count = len(internal_txs)
    total_tx_count = native_tx_count + erc20_transfer_count + internal_tx_count

    all_timestamps: List[int] = []
    normal_timestamps: List[int] = []

    counterparties: Set[str] = set()
    outgoing_counterparties: Set[str] = set()
    incoming_counterparties: Set[str] = set()
    contracts: Set[str] = set()

    outgoing_count = 0
    contract_interaction_count = 0

    gas_used_values: List[int] = []
    gas_price_values: List[int] = []

    for tx in normal_txs:
        ts = _extract_timestamp(tx)
        if ts is not None:
            all_timestamps.append(ts)
            normal_timestamps.append(ts)

        cp = _extract_counterparty(address, tx)
        if cp:
            counterparties.add(cp)
            if _is_outgoing(address, tx):
                outgoing_counterparties.add(cp)
            else:
                incoming_counterparties.add(cp)

        if _is_outgoing(address, tx):
            outgoing_count += 1

        if _is_contract_interaction(tx):
            contract_interaction_count += 1
            tx_to = _sanitize_address(tx.get("to"))
            if tx_to:
                contracts.add(tx_to)

        gas_used = _safe_int(tx.get("gasUsed"))
        if gas_used is not None:
            gas_used_values.append(gas_used)

        gas_price = _safe_int(tx.get("gasPrice"))
        if gas_price is not None:
            gas_price_values.append(gas_price)

    for tx in erc20_txs:
        ts = _extract_timestamp(tx)
        if ts is not None:
            all_timestamps.append(ts)

        cp = _extract_counterparty(address, tx)
        if cp:
            counterparties.add(cp)
            if _is_outgoing(address, tx):
                outgoing_counterparties.add(cp)
            else:
                incoming_counterparties.add(cp)

        if _is_outgoing(address, tx):
            outgoing_count += 1

        tx_to = _sanitize_address(tx.get("contractAddress"))
        if tx_to:
            contracts.add(tx_to)

    for tx in internal_txs:
        ts = _extract_timestamp(tx)
        if ts is not None:
            all_timestamps.append(ts)

        cp = _extract_counterparty(address, tx)
        if cp:
            counterparties.add(cp)
            if _is_outgoing(address, tx):
                outgoing_counterparties.add(cp)
            else:
                incoming_counterparties.add(cp)

        if _is_outgoing(address, tx):
            outgoing_count += 1

    first_seen_ts = min(all_timestamps) if all_timestamps else None
    last_seen_ts = max(all_timestamps) if all_timestamps else None

    lifetime_days: Optional[float]
    days_since_last_active: Optional[float]

    if first_seen_ts is not None and last_seen_ts is not None:
        lifetime_days = max((last_seen_ts - first_seen_ts) / 86400.0, 0.0)
        days_since_last_active = max((snapshot_ts - last_seen_ts) / 86400.0, 0.0)
    else:
        lifetime_days = None
        days_since_last_active = None

    active_days = _distinct_calendar_days(all_timestamps)
    active_weeks = _distinct_calendar_weeks(all_timestamps)
    activity_density = (
        active_days / observation_window_days if observation_window_days > 0 else 0.0
    )

    mean_gap_seconds, median_gap_seconds = _compute_gap_stats(all_timestamps)

    mean_gas_used = (
        float(sum(gas_used_values) / len(gas_used_values)) if gas_used_values else None
    )
    median_gas_used = (
        float(statistics.median(gas_used_values)) if gas_used_values else None
    )
    mean_gas_price_wei = (
        float(sum(gas_price_values) / len(gas_price_values)) if gas_price_values else None
    )
    median_gas_price_wei = (
        float(statistics.median(gas_price_values)) if gas_price_values else None
    )

    outgoing_ratio = outgoing_count / total_tx_count if total_tx_count > 0 else 0.0
    contract_interaction_ratio = (
        contract_interaction_count / native_tx_count if native_tx_count > 0 else 0.0
    )
    erc20_ratio = erc20_transfer_count / total_tx_count if total_tx_count > 0 else 0.0
    internal_ratio = internal_tx_count / total_tx_count if total_tx_count > 0 else 0.0

    record = AddressChainFeatureRecord(
        address=address,
        chain=chain,
        chain_id=chain_id,
        snapshot_ts=snapshot_ts,
        observation_window_days=observation_window_days,
        native_tx_count=native_tx_count,
        erc20_transfer_count=erc20_transfer_count,
        internal_tx_count=internal_tx_count,
        total_tx_count=total_tx_count,
        first_seen_ts=first_seen_ts,
        last_seen_ts=last_seen_ts,
        lifetime_days=lifetime_days,
        days_since_last_active=days_since_last_active,
        active_days=active_days,
        active_weeks=active_weeks,
        activity_density=activity_density,
        distinct_counterparties=len(counterparties),
        distinct_outgoing_counterparties=len(outgoing_counterparties),
        distinct_incoming_counterparties=len(incoming_counterparties),
        distinct_contracts=len(contracts),
        outgoing_ratio=outgoing_ratio,
        contract_interaction_ratio=contract_interaction_ratio,
        erc20_ratio=erc20_ratio,
        internal_ratio=internal_ratio,
        mean_gap_seconds=mean_gap_seconds,
        median_gap_seconds=median_gap_seconds,
        mean_gas_used=mean_gas_used,
        median_gas_used=median_gas_used,
        mean_gas_price_wei=mean_gas_price_wei,
        median_gas_price_wei=median_gas_price_wei,
        recent_7d_tx_count=_count_recent_txs(all_timestamps, snapshot_ts, 7),
        recent_30d_tx_count=_count_recent_txs(all_timestamps, snapshot_ts, 30),
        recent_90d_tx_count=_count_recent_txs(all_timestamps, snapshot_ts, 90),
        has_activity=total_tx_count > 0,
        is_query_truncated=is_query_truncated,
    )
    return record


def compute_address_chain_features_from_cache(
    app_config: AppConfig,
    chain: ChainConfig,
    address: str,
) -> AddressChainFeatureRecord:
    address = _sanitize_address(address)
    if address is None:
        raise ValueError("Invalid address")

    cache_path = _address_activity_cache_path(app_config, chain, address)
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached raw activity found for {chain.name}:{address}")

    payload = read_json(cache_path)
    return _extract_feature_record_from_payload(payload)


def fetch_and_compute_address_chain_features(
    client: EtherscanClient,
    app_config: AppConfig,
    chain: ChainConfig,
    address: str,
    *,
    refresh: bool = False,
    max_pages_per_endpoint: Optional[int] = None,
) -> AddressChainFeatureRecord:
    payload = fetch_address_chain_activity(
        client=client,
        app_config=app_config,
        chain=chain,
        address=address,
        refresh=refresh,
        max_pages_per_endpoint=max_pages_per_endpoint,
    )
    return _extract_feature_record_from_payload(payload)


def feature_records_to_dataframe(records: Sequence[AddressChainFeatureRecord]) -> pd.DataFrame:
    rows = [r.to_dict() for r in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_feature_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Supported output formats: .csv, .parquet")


def build_feature_table_for_addresses(
    client: EtherscanClient,
    app_config: AppConfig,
    addresses: Sequence[str],
    *,
    chains: Optional[Sequence[ChainConfig]] = None,
    refresh: bool = False,
    max_pages_per_endpoint: Optional[int] = None,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """
    Build a chain-level feature table for all (address x chain) pairs.

    This function is intended for notebook step 02 / 03:
    - fetch raw cross-chain activity
    - cache raw payloads
    - compute scalar features
    - checkpoint interim feature tables
    """
    clean_addresses: List[str] = []
    seen: Set[str] = set()

    for address in addresses:
        addr = _sanitize_address(address)
        if addr and addr not in seen:
            clean_addresses.append(addr)
            seen.add(addr)

    if chains is None:
        chains = list(app_config.chains.values())

    records: List[AddressChainFeatureRecord] = []
    total_jobs = len(clean_addresses) * len(chains)
    done = 0

    checkpoint_dir = app_config.paths.interim_dir / "feature_rows"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for address in clean_addresses:
        for chain in chains:
            logger.info(
                "Building features for %s on %s (%s/%s)",
                address,
                chain.name,
                done + 1,
                total_jobs,
            )
            record = fetch_and_compute_address_chain_features(
                client=client,
                app_config=app_config,
                chain=chain,
                address=address,
                refresh=refresh,
                max_pages_per_endpoint=max_pages_per_endpoint,
            )
            records.append(record)
            done += 1

            if checkpoint_every > 0 and done % checkpoint_every == 0:
                checkpoint_df = feature_records_to_dataframe(records)
                checkpoint_path = checkpoint_dir / "address_chain_features_checkpoint.csv"
                checkpoint_df.to_csv(checkpoint_path, index=False)

    df = feature_records_to_dataframe(records)

    final_csv = checkpoint_dir / "address_chain_features.csv"
    df.to_csv(final_csv, index=False)

    return df


def build_feature_table_from_seed_csv(
    client: EtherscanClient,
    app_config: AppConfig,
    seed_csv_path: str | Path,
    *,
    address_col: str = "address",
    chains: Optional[Sequence[ChainConfig]] = None,
    refresh: bool = False,
    max_pages_per_endpoint: Optional[int] = None,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    addresses = load_address_list_from_csv(seed_csv_path, address_col=address_col)
    return build_feature_table_for_addresses(
        client=client,
        app_config=app_config,
        addresses=addresses,
        chains=chains,
        refresh=refresh,
        max_pages_per_endpoint=max_pages_per_endpoint,
        checkpoint_every=checkpoint_every,
    )