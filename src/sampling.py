# sampling.py
# Block-window ingestion module for EVM cross-chain wallet correlation project.
#
# Responsibilities:
# - exact last-N-block window planning
# - raw block fetching with cache
# - raw transaction extraction from full block payloads
# - unique address collection from both `from` and `to`
# - EOA lookup / caching via eth_getCode
# - enrichment of transactions with sender/receiver EOA flags
#
# Important notes:
# - observation windows are block-based, not day-based
# - cached transaction reuse is validated against a manifest for the same exact window
# - eth_getCode is resolved in chunks and persisted incrementally
# - parquet writes are protected against uint256-like integer overflow

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:
    import pandas as pd  # optional, only for parquet support
except ImportError:  # pragma: no cover
    pd = None

from .api_client import AddressCodeResult, BlockchainClient
from .config import AppConfig, ChainConfig

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


# ============================================================
# Data classes
# ============================================================


@dataclass(frozen=True)
class BlockWindowPlan:
    chain: str
    chain_id: int
    reference_block_tag: str
    reference_block_number: int
    window_blocks: int
    start_block: int
    end_block: int
    block_numbers: List[int]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawTransactionRecord:
    chain: str
    chain_id: int
    window_blocks: int
    reference_block_number: int

    block_number: int
    block_timestamp: Optional[int]
    tx_hash: str
    tx_index: Optional[int]

    from_address: str
    to_address: Optional[str]

    value_wei: int
    nonce: int
    gas: Optional[int]
    gas_price_wei: Optional[int]

    is_contract_creation: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AddressObservationRecord:
    chain: str
    chain_id: int
    window_blocks: int
    reference_block_number: int
    address: str
    seen_as_from: bool
    seen_as_to: bool
    tx_count_as_from: int
    tx_count_as_to: int
    first_seen_block: int
    last_seen_block: int
    first_seen_timestamp: Optional[int]
    last_seen_timestamp: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AddressCodeCacheRecord:
    chain: str
    chain_id: int
    address: str
    code: str
    is_eoa: bool
    checked_at: str
    block_tag: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Basic IO helpers
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _is_int64_compatible(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Integral):
        return INT64_MIN <= int(value) <= INT64_MAX
    return True


def _prepare_dataframe_for_parquet(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    PyArrow parquet does not safely support arbitrary-size Python integers.
    EVM numeric fields such as *_wei may exceed int64, so such columns are
    converted to strings before writing.
    """
    out = df.copy()

    for col in out.columns:
        series = out[col]
        has_big_int = False

        for value in series.dropna():
            if not _is_int64_compatible(value):
                has_big_int = True
                break

        if has_big_int:
            out[col] = out[col].map(
                lambda x: str(x) if x is not None and not pd.isna(x) else None
            )

    return out


def write_table(
    rows: Sequence[Dict[str, Any]],
    path_without_suffix: Path,
    table_format: str,
) -> Path:
    """
    Save a row-oriented table as parquet or csv depending on config.
    """
    table_format = table_format.strip().lower()

    if table_format == "parquet":
        if pd is None:
            raise RuntimeError(
                "table_format='parquet' requested but pandas is not installed."
            )
        path = path_without_suffix.with_suffix(".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(list(rows))
        df = _prepare_dataframe_for_parquet(df)
        df.to_parquet(path, index=False)

        return path

    if table_format == "csv":
        path = path_without_suffix.with_suffix(".csv")
        write_csv(path, rows)
        return path

    raise ValueError(
        f"Unsupported table_format={table_format!r}. Expected 'parquet' or 'csv'."
    )


def read_table(path_without_suffix: Path, table_format: str) -> List[Dict[str, Any]]:
    table_format = table_format.strip().lower()

    if table_format == "parquet":
        path = path_without_suffix.with_suffix(".parquet")
        if not path.exists():
            return []
        if pd is None:
            raise RuntimeError(
                "table_format='parquet' requested but pandas is not installed."
            )
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    if table_format == "csv":
        path = path_without_suffix.with_suffix(".csv")
        return read_csv(path)

    raise ValueError(
        f"Unsupported table_format={table_format!r}. Expected 'parquet' or 'csv'."
    )


# ============================================================
# Normalization helpers
# ============================================================


def sanitize_address(address: Optional[str]) -> Optional[str]:
    if not address or not isinstance(address, str):
        return None
    address = address.strip().lower()
    if not address.startswith("0x"):
        return None
    if len(address) != 42:
        return None
    return address


def hex_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.startswith("0x"):
            return int(value, 16)
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Optional[int], default: int = 0) -> int:
    return value if value is not None else default


# ============================================================
# Path helpers
# ============================================================


def block_payload_path(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    block_number: int,
) -> Path:
    return (
        app_config.paths.raw_blocks_dir
        / chain.name
        / f"window_{window_blocks}"
        / f"block_{block_number}.json"
    )


def transaction_table_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.raw_transactions_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_transactions"
    )


def address_observation_table_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.interim_status_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_address_observations"
    )


def address_code_cache_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
) -> Path:
    return (
        app_config.paths.raw_address_code_dir
        / chain.name
        / f"{chain.name}_address_code_cache"
    )


def address_code_snapshot_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.interim_status_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_address_code_snapshot"
    )


def enriched_transaction_table_path_base(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.interim_status_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_transactions_enriched"
    )


def manifest_path(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Path:
    return (
        app_config.paths.interim_status_dir
        / chain.name
        / f"{chain.name}_window_{window_blocks}_manifest.json"
    )


# ============================================================
# Manifest helpers
# ============================================================


def load_window_manifest(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Optional[Dict[str, Any]]:
    path = manifest_path(app_config=app_config, chain=chain, window_blocks=window_blocks)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception as exc:
        logger.warning(
            "[%s][window=%s] Failed to read manifest, ignoring cache: %s",
            chain.name,
            window_blocks,
            exc,
        )
        return None

    return payload if isinstance(payload, dict) else None


def manifest_matches_plan(
    manifest: Optional[Dict[str, Any]],
    plan: BlockWindowPlan,
) -> bool:
    if manifest is None:
        return False

    try:
        manifest_block_numbers = [int(x) for x in manifest.get("block_numbers", [])]
    except Exception:
        return False

    return (
        str(manifest.get("chain")) == plan.chain
        and int(manifest.get("chain_id", -1)) == plan.chain_id
        and str(manifest.get("reference_block_tag")) == plan.reference_block_tag
        and int(manifest.get("reference_block_number", -1)) == plan.reference_block_number
        and int(manifest.get("window_blocks", -1)) == plan.window_blocks
        and int(manifest.get("start_block", -1)) == plan.start_block
        and int(manifest.get("end_block", -1)) == plan.end_block
        and manifest_block_numbers == plan.block_numbers
    )


# ============================================================
# Window planning
# ============================================================


def build_exact_block_window(
    reference_block_number: int,
    window_blocks: int,
) -> List[int]:
    if window_blocks <= 0:
        raise ValueError(f"window_blocks must be > 0, got {window_blocks}")
    if reference_block_number < 0:
        raise ValueError(
            f"reference_block_number must be >= 0, got {reference_block_number}"
        )

    start_block = max(0, reference_block_number - window_blocks + 1)
    return list(range(start_block, reference_block_number + 1))


def build_block_window_plan(
    client: BlockchainClient,
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> BlockWindowPlan:
    reference_block_number = client.get_reference_block_number(
        chain=chain,
        block_tag=app_config.sampling.reference_block_tag,
    )
    block_numbers = build_exact_block_window(
        reference_block_number=reference_block_number,
        window_blocks=window_blocks,
    )

    return BlockWindowPlan(
        chain=chain.name,
        chain_id=chain.chain_id,
        reference_block_tag=app_config.sampling.reference_block_tag,
        reference_block_number=reference_block_number,
        window_blocks=window_blocks,
        start_block=block_numbers[0],
        end_block=block_numbers[-1],
        block_numbers=block_numbers,
        created_at=utc_now_iso(),
    )


# ============================================================
# Block ingestion
# ============================================================


def load_or_fetch_block_payload(
    client: BlockchainClient,
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    block_number: int,
) -> JsonDict:
    cache_path = block_payload_path(
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
        block_number=block_number,
    )

    if app_config.storage.resume_from_cache and cache_path.exists():
        logger.info(
            "[%s][window=%s] Loading cached block %s",
            chain.name,
            window_blocks,
            block_number,
        )
        return read_json(cache_path)

    logger.info(
        "[%s][window=%s] Fetching block %s",
        chain.name,
        window_blocks,
        block_number,
    )
    payload = client.get_block_by_number(
        chain=chain,
        block_number=block_number,
        full_transactions=app_config.sampling.full_transactions,
    )

    if app_config.storage.save_raw_block_payloads:
        write_json(cache_path, payload)

    return payload


def fetch_blocks_for_window(
    client: BlockchainClient,
    app_config: AppConfig,
    chain: ChainConfig,
    plan: BlockWindowPlan,
) -> List[JsonDict]:
    payloads: List[JsonDict] = []
    for block_number in plan.block_numbers:
        payload = load_or_fetch_block_payload(
            client=client,
            app_config=app_config,
            chain=chain,
            window_blocks=plan.window_blocks,
            block_number=block_number,
        )
        payloads.append(payload)
    return payloads


# ============================================================
# Transaction extraction
# ============================================================


def extract_transactions_from_block_payload(
    chain: ChainConfig,
    window_blocks: int,
    reference_block_number: int,
    block_payload: JsonDict,
) -> List[RawTransactionRecord]:
    block_number = _safe_int(hex_to_int(block_payload.get("number")), default=0)
    block_timestamp = hex_to_int(block_payload.get("timestamp"))

    txs = block_payload.get("transactions", []) or []
    rows: List[RawTransactionRecord] = []

    for tx in txs:
        from_address = sanitize_address(tx.get("from"))
        if from_address is None:
            continue

        to_address = sanitize_address(tx.get("to"))
        tx_hash = str(tx.get("hash") or "").strip()
        tx_index = hex_to_int(tx.get("transactionIndex"))
        value_wei = _safe_int(hex_to_int(tx.get("value")), default=0)
        nonce = _safe_int(hex_to_int(tx.get("nonce")), default=0)
        gas = hex_to_int(tx.get("gas"))
        gas_price_wei = hex_to_int(tx.get("gasPrice"))

        rows.append(
            RawTransactionRecord(
                chain=chain.name,
                chain_id=chain.chain_id,
                window_blocks=window_blocks,
                reference_block_number=reference_block_number,
                block_number=block_number,
                block_timestamp=block_timestamp,
                tx_hash=tx_hash,
                tx_index=tx_index,
                from_address=from_address,
                to_address=to_address,
                value_wei=value_wei,
                nonce=nonce,
                gas=gas,
                gas_price_wei=gas_price_wei,
                is_contract_creation=(to_address is None),
            )
        )

    return rows


def extract_transactions_from_blocks(
    chain: ChainConfig,
    window_blocks: int,
    reference_block_number: int,
    block_payloads: Sequence[JsonDict],
) -> List[RawTransactionRecord]:
    rows: List[RawTransactionRecord] = []
    for payload in block_payloads:
        rows.extend(
            extract_transactions_from_block_payload(
                chain=chain,
                window_blocks=window_blocks,
                reference_block_number=reference_block_number,
                block_payload=payload,
            )
        )
    return rows


def save_raw_transactions(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    transactions: Sequence[RawTransactionRecord],
) -> Path:
    rows = [tx.to_dict() for tx in transactions]
    return write_table(
        rows=rows,
        path_without_suffix=transaction_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


def load_raw_transactions(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> List[Dict[str, Any]]:
    return read_table(
        path_without_suffix=transaction_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


# ============================================================
# Address observation / aggregation
# ============================================================


def build_address_observations(
    chain: ChainConfig,
    window_blocks: int,
    reference_block_number: int,
    transactions: Sequence[RawTransactionRecord],
) -> List[AddressObservationRecord]:
    """
    Build one row per address within a chain/window.
    Tracks whether the address appears as sender and/or receiver.
    """
    state: Dict[str, Dict[str, Any]] = {}

    def ensure(address: str) -> Dict[str, Any]:
        if address not in state:
            state[address] = {
                "seen_as_from": False,
                "seen_as_to": False,
                "tx_count_as_from": 0,
                "tx_count_as_to": 0,
                "first_seen_block": None,
                "last_seen_block": None,
                "first_seen_timestamp": None,
                "last_seen_timestamp": None,
            }
        return state[address]

    for tx in transactions:
        sender = tx.from_address
        receiver = tx.to_address

        sender_state = ensure(sender)
        sender_state["seen_as_from"] = True
        sender_state["tx_count_as_from"] += 1
        _update_first_last_seen(
            sender_state,
            tx.block_number,
            tx.block_timestamp,
        )

        if receiver:
            receiver_state = ensure(receiver)
            receiver_state["seen_as_to"] = True
            receiver_state["tx_count_as_to"] += 1
            _update_first_last_seen(
                receiver_state,
                tx.block_number,
                tx.block_timestamp,
            )

    out: List[AddressObservationRecord] = []
    for address, payload in sorted(state.items()):
        out.append(
            AddressObservationRecord(
                chain=chain.name,
                chain_id=chain.chain_id,
                window_blocks=window_blocks,
                reference_block_number=reference_block_number,
                address=address,
                seen_as_from=bool(payload["seen_as_from"]),
                seen_as_to=bool(payload["seen_as_to"]),
                tx_count_as_from=int(payload["tx_count_as_from"]),
                tx_count_as_to=int(payload["tx_count_as_to"]),
                first_seen_block=int(payload["first_seen_block"]),
                last_seen_block=int(payload["last_seen_block"]),
                first_seen_timestamp=payload["first_seen_timestamp"],
                last_seen_timestamp=payload["last_seen_timestamp"],
            )
        )

    return out


def _update_first_last_seen(
    state_row: Dict[str, Any],
    block_number: int,
    block_timestamp: Optional[int],
) -> None:
    current_first_block = state_row["first_seen_block"]
    current_last_block = state_row["last_seen_block"]

    if current_first_block is None or block_number < current_first_block:
        state_row["first_seen_block"] = block_number
        state_row["first_seen_timestamp"] = block_timestamp

    if current_last_block is None or block_number > current_last_block:
        state_row["last_seen_block"] = block_number
        state_row["last_seen_timestamp"] = block_timestamp


def save_address_observations(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    observations: Sequence[AddressObservationRecord],
) -> Path:
    rows = [item.to_dict() for item in observations]
    return write_table(
        rows=rows,
        path_without_suffix=address_observation_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


def collect_unique_addresses_from_transactions(
    transactions: Sequence[RawTransactionRecord],
) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []

    for tx in transactions:
        for address in (tx.from_address, tx.to_address):
            normalized = sanitize_address(address)
            if normalized is None:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)

    return out


# ============================================================
# Address code cache
# ============================================================


def load_address_code_cache(
    app_config: AppConfig,
    chain: ChainConfig,
) -> Dict[str, AddressCodeCacheRecord]:
    rows = read_table(
        path_without_suffix=address_code_cache_path_base(app_config, chain),
        table_format=app_config.storage.table_format,
    )

    cache: Dict[str, AddressCodeCacheRecord] = {}
    for row in rows:
        address = sanitize_address(row.get("address"))
        if address is None:
            continue

        is_eoa_raw = row.get("is_eoa")
        if isinstance(is_eoa_raw, str):
            is_eoa = is_eoa_raw.strip().lower() in {"1", "true", "yes"}
        else:
            is_eoa = bool(is_eoa_raw)

        cache[address] = AddressCodeCacheRecord(
            chain=str(row.get("chain")),
            chain_id=int(row.get("chain_id")),
            address=address,
            code=str(row.get("code") or ""),
            is_eoa=is_eoa,
            checked_at=str(row.get("checked_at") or ""),
            block_tag=str(row.get("block_tag") or "latest"),
        )

    return cache


def save_address_code_cache(
    app_config: AppConfig,
    chain: ChainConfig,
    cache_records: Sequence[AddressCodeCacheRecord],
) -> Path:
    rows = [item.to_dict() for item in cache_records]
    return write_table(
        rows=rows,
        path_without_suffix=address_code_cache_path_base(app_config, chain),
        table_format=app_config.storage.table_format,
    )


def resolve_address_codes_for_window(
    client: BlockchainClient,
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    addresses: Sequence[str],
    block_tag: str = "latest",
) -> Dict[str, AddressCodeCacheRecord]:
    """
    Resolve eth_getCode for all unique addresses in the current chain/window,
    reusing chain-level cache so each address is checked once per chain.

    Improvements:
    - works in chunks
    - persists cache after each chunk
    - uses conservative parallelism in explorer-only mode
    """
    chain_cache = load_address_code_cache(app_config=app_config, chain=chain)

    unresolved: List[str] = []
    for address in addresses:
        normalized = sanitize_address(address)
        if normalized is None:
            continue
        if normalized not in chain_cache:
            unresolved.append(normalized)

    if app_config.sampling.max_addresses_for_code_lookup is not None:
        unresolved = unresolved[: app_config.sampling.max_addresses_for_code_lookup]

    if not unresolved:
        logger.info(
            "[%s][window=%s] All %s addresses already present in code cache",
            chain.name,
            window_blocks,
            len(addresses),
        )
        snapshot_records: List[AddressCodeCacheRecord] = [
            chain_cache[address]
            for address in addresses
            if address in chain_cache
        ]
        write_table(
            rows=[item.to_dict() for item in snapshot_records],
            path_without_suffix=address_code_snapshot_path_base(
                app_config=app_config,
                chain=chain,
                window_blocks=window_blocks,
            ),
            table_format=app_config.storage.table_format,
        )
        return {record.address: record for record in snapshot_records}

    logger.info(
        "[%s][window=%s] Resolving eth_getCode for %s uncached addresses",
        chain.name,
        window_blocks,
        len(unresolved),
    )

    max_workers = None if chain.rpc_url else 1
    chunk_size = max(1, int(getattr(app_config.api, "batch_size", 25) or 25))
    if not chain.rpc_url:
        chunk_size = min(chunk_size, 10)

    now = utc_now_iso()

    for start_idx in range(0, len(unresolved), chunk_size):
        chunk = unresolved[start_idx : start_idx + chunk_size]
        end_idx = start_idx + len(chunk)

        logger.info(
            "[%s][window=%s] eth_getCode chunk %s-%s of %s",
            chain.name,
            window_blocks,
            start_idx + 1,
            end_idx,
            len(unresolved),
        )

        chunk_results: List[AddressCodeResult] = client.get_codes(
            chain=chain,
            addresses=chunk,
            block_tag=block_tag,
            max_workers=max_workers,
        )

        for item in chunk_results:
            chain_cache[item.address] = AddressCodeCacheRecord(
                chain=chain.name,
                chain_id=chain.chain_id,
                address=item.address,
                code=item.code,
                is_eoa=item.is_eoa,
                checked_at=now,
                block_tag=block_tag,
            )

        save_address_code_cache(
            app_config=app_config,
            chain=chain,
            cache_records=list(chain_cache.values()),
        )

    snapshot_records: List[AddressCodeCacheRecord] = [
        chain_cache[address]
        for address in addresses
        if address in chain_cache
    ]

    write_table(
        rows=[item.to_dict() for item in snapshot_records],
        path_without_suffix=address_code_snapshot_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )

    return {record.address: record for record in snapshot_records}


# ============================================================
# Transaction enrichment
# ============================================================


def enrich_transactions_with_eoa_flags(
    transactions: Sequence[RawTransactionRecord],
    address_code_lookup: Dict[str, AddressCodeCacheRecord],
) -> List[Dict[str, Any]]:
    enriched_rows: List[Dict[str, Any]] = []

    for tx in transactions:
        sender_meta = address_code_lookup.get(tx.from_address)
        receiver_meta = address_code_lookup.get(tx.to_address) if tx.to_address else None

        row = tx.to_dict()
        row["from_is_eoa"] = sender_meta.is_eoa if sender_meta is not None else None
        row["to_is_eoa"] = receiver_meta.is_eoa if receiver_meta is not None else None
        row["from_code"] = sender_meta.code if sender_meta is not None else None
        row["to_code"] = receiver_meta.code if receiver_meta is not None else None
        enriched_rows.append(row)

    return enriched_rows


def save_enriched_transactions(
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
    enriched_rows: Sequence[Dict[str, Any]],
) -> Path:
    return write_table(
        rows=list(enriched_rows),
        path_without_suffix=enriched_transaction_table_path_base(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        table_format=app_config.storage.table_format,
    )


# ============================================================
# Manifest / orchestration
# ============================================================


def build_window_manifest(
    plan: BlockWindowPlan,
    tx_count: int,
    unique_address_count: int,
    eoa_count: int,
    contract_count: int,
) -> Dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "chain": plan.chain,
        "chain_id": plan.chain_id,
        "reference_block_tag": plan.reference_block_tag,
        "reference_block_number": plan.reference_block_number,
        "window_blocks": plan.window_blocks,
        "start_block": plan.start_block,
        "end_block": plan.end_block,
        "block_numbers": plan.block_numbers,
        "transaction_count": tx_count,
        "unique_address_count": unique_address_count,
        "eoa_count": eoa_count,
        "contract_count": contract_count,
    }


def run_block_window_ingestion_for_chain(
    client: BlockchainClient,
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Dict[str, Any]:
    """
    Full stage runner for one chain and one exact block window.

    Returns a compact dictionary with the in-memory artifacts needed by next stages.
    """
    # 1) Plan exact block window
    plan = build_block_window_plan(
        client=client,
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
    )

    # 2) Reuse cached transactions only if manifest matches this exact window
    cached_tx_rows: List[Dict[str, Any]] = []
    cached_manifest = None
    cache_is_valid = False

    if app_config.storage.resume_from_cache:
        cached_manifest = load_window_manifest(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        )
        cache_is_valid = manifest_matches_plan(cached_manifest, plan)

        if cache_is_valid:
            cached_tx_rows = load_raw_transactions(
                app_config=app_config,
                chain=chain,
                window_blocks=window_blocks,
            )
        else:
            logger.info(
                "[%s][window=%s] Existing cache does not match current block window; rebuilding",
                chain.name,
                window_blocks,
            )

    if cached_tx_rows and cache_is_valid:
        logger.info(
            "[%s][window=%s] Loaded %s cached transaction rows",
            chain.name,
            window_blocks,
            len(cached_tx_rows),
        )
        transactions = [raw_transaction_record_from_dict(row) for row in cached_tx_rows]
    else:
        # 3) Fetch exact blocks
        block_payloads = fetch_blocks_for_window(
            client=client,
            app_config=app_config,
            chain=chain,
            plan=plan,
        )

        # 4) Extract raw transaction rows
        transactions = extract_transactions_from_blocks(
            chain=chain,
            window_blocks=window_blocks,
            reference_block_number=plan.reference_block_number,
            block_payloads=block_payloads,
        )

        if app_config.storage.save_raw_transaction_rows:
            save_raw_transactions(
                app_config=app_config,
                chain=chain,
                window_blocks=window_blocks,
                transactions=transactions,
            )

    # 5) Address observations
    observations = build_address_observations(
        chain=chain,
        window_blocks=window_blocks,
        reference_block_number=plan.reference_block_number,
        transactions=transactions,
    )
    save_address_observations(
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
        observations=observations,
    )

    # 6) Unique address set from both sender and receiver sides
    unique_addresses = collect_unique_addresses_from_transactions(transactions)

    # 7) Resolve/cached EOA status
    address_code_lookup = resolve_address_codes_for_window(
        client=client,
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
        addresses=unique_addresses,
        block_tag=plan.reference_block_tag,
    )

    # 8) Enrich raw transactions with EOA flags
    enriched_rows = enrich_transactions_with_eoa_flags(
        transactions=transactions,
        address_code_lookup=address_code_lookup,
    )
    save_enriched_transactions(
        app_config=app_config,
        chain=chain,
        window_blocks=window_blocks,
        enriched_rows=enriched_rows,
    )

    # 9) Save manifest
    eoa_count = sum(1 for item in address_code_lookup.values() if item.is_eoa)
    contract_count = sum(1 for item in address_code_lookup.values() if not item.is_eoa)

    manifest = build_window_manifest(
        plan=plan,
        tx_count=len(transactions),
        unique_address_count=len(unique_addresses),
        eoa_count=eoa_count,
        contract_count=contract_count,
    )
    write_json(
        manifest_path(
            app_config=app_config,
            chain=chain,
            window_blocks=window_blocks,
        ),
        manifest,
    )

    logger.info(
        "[%s][window=%s] blocks=%s tx=%s unique_addresses=%s eoa=%s contracts=%s",
        chain.name,
        window_blocks,
        len(plan.block_numbers),
        len(transactions),
        len(unique_addresses),
        eoa_count,
        contract_count,
    )

    return {
        "plan": plan,
        "transactions": transactions,
        "observations": observations,
        "address_code_lookup": address_code_lookup,
        "enriched_transactions": enriched_rows,
        "manifest": manifest,
    }


def run_block_window_ingestion_for_all_chains(
    client: BlockchainClient,
    app_config: AppConfig,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Run the ingestion stage for all enabled chains and all configured block windows.

    Output shape:
    {
        "ethereum": {
            1: {...},
            10: {...},
            100: {...},
        },
        "polygon": {...},
        ...
    }
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for chain in app_config.enabled_chains:
        out[chain.name] = {}
        for window_blocks in app_config.sampling.windows.block_counts:
            out[chain.name][window_blocks] = run_block_window_ingestion_for_chain(
                client=client,
                app_config=app_config,
                chain=chain,
                window_blocks=window_blocks,
            )
    return out


# ============================================================
# Deserialization helpers
# ============================================================


def raw_transaction_record_from_dict(row: Dict[str, Any]) -> RawTransactionRecord:
    return RawTransactionRecord(
        chain=str(row["chain"]),
        chain_id=int(row["chain_id"]),
        window_blocks=int(row["window_blocks"]),
        reference_block_number=int(row["reference_block_number"]),
        block_number=int(row["block_number"]),
        block_timestamp=(
            int(row["block_timestamp"])
            if row.get("block_timestamp") not in (None, "", "None")
            else None
        ),
        tx_hash=str(row["tx_hash"]),
        tx_index=(
            int(row["tx_index"])
            if row.get("tx_index") not in (None, "", "None")
            else None
        ),
        from_address=str(row["from_address"]).lower(),
        to_address=(
            str(row["to_address"]).lower()
            if row.get("to_address") not in (None, "", "None")
            else None
        ),
        value_wei=int(row["value_wei"]),
        nonce=int(row["nonce"]),
        gas=int(row["gas"]) if row.get("gas") not in (None, "", "None") else None,
        gas_price_wei=(
            int(row["gas_price_wei"])
            if row.get("gas_price_wei") not in (None, "", "None")
            else None
        ),
        is_contract_creation=str(row["is_contract_creation"]).strip().lower()
        in {"1", "true", "yes"},
    )