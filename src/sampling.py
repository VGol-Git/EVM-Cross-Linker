# Sampling module — block fetching, address extraction, EOA filtering
#
# Key features vs previous version:
#   • Concurrent cross-network fetching via ThreadPoolExecutor
#   • Raw block data saved as {network}_block{num}_raw.csv
#   • Uses MultiProviderClient (auto-fallback across RPCs)
#   • Single BLOCK_WINDOW variable instead of iterating a list

from __future__ import annotations

import csv
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .api_client import MultiProviderClient
from .config import (
    APPROX_BLOCK_TIMES,
    BLOCK_WINDOW,
    AppConfig,
    ChainConfig,
    get_window_paths,
    window_label,
)

logger = logging.getLogger(__name__)


# =====================================================================
# IO helpers
# =====================================================================

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def sanitize_address(address: str | None) -> Optional[str]:
    if not address:
        return None
    address = address.strip().lower()
    if not address.startswith("0x"):
        return None
    if len(address) != 42:
        return None
    return address


def hex_to_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.startswith("0x"):
        return int(value, 16)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# =====================================================================
# Block range planning (BLOCK-BASED windows)
# =====================================================================

def compute_block_range(
    latest_block: int,
    window_blocks: int,
) -> tuple[int, int]:
    """
    Given the latest block on-chain and a window size in blocks,
    return (start_block, end_block) inclusive.
    """
    start_block = max(0, latest_block - window_blocks + 1)
    return start_block, latest_block


def plan_blocks_to_fetch(
    start_block: int,
    end_block: int,
    max_blocks_to_fetch: int,
) -> List[int]:
    """
    Decide which blocks to actually fetch from the [start, end] range.

    - If the range is small enough (<= max_blocks_to_fetch), fetch ALL.
    - Otherwise, pick max_blocks_to_fetch evenly-spaced blocks.
    """
    total_in_range = end_block - start_block + 1

    if total_in_range <= max_blocks_to_fetch:
        return list(range(start_block, end_block + 1))

    return evenly_spaced_block_numbers(start_block, end_block, max_blocks_to_fetch)


def evenly_spaced_block_numbers(
    start_block: int,
    end_block: int,
    n_blocks: int,
) -> List[int]:
    if n_blocks <= 0:
        raise ValueError("n_blocks must be > 0")
    if start_block > end_block:
        raise ValueError("start_block must be <= end_block")

    if n_blocks == 1:
        return [end_block]

    span = end_block - start_block
    if span == 0:
        return [start_block]

    block_numbers: List[int] = []
    for i in range(n_blocks):
        ratio = i / (n_blocks - 1)
        block_num = start_block + round(span * ratio)
        block_numbers.append(block_num)

    # deduplicate, preserve order
    seen: Set[int] = set()
    out: List[int] = []
    for b in block_numbers:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


# =====================================================================
# Address dataclass (from + to extraction)
# =====================================================================

@dataclass
class AddressRecord:
    """An address observed in a sampled block with its role (sender / receiver)."""
    chain: str
    address: str
    role: str                      # "sender" or "receiver"
    source_block_number: int
    block_timestamp: Optional[int]
    tx_hash: str
    nonce: Optional[int]           # nonce from tx (only meaningful for sender)


# =====================================================================
# Block fetching & address extraction
# =====================================================================

def _block_payload_to_csv_rows(
    chain: ChainConfig,
    block_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a full block payload into flat CSV-ready rows (one per tx)."""
    block_number = hex_to_int(block_payload.get("number")) or 0
    block_ts = hex_to_int(block_payload.get("timestamp"))
    txs = block_payload.get("transactions", []) or []
    rows = []
    for tx in txs:
        rows.append({
            "chain": chain.name,
            "block_number": block_number,
            "block_timestamp": block_ts,
            "tx_hash": tx.get("hash", ""),
            "from": (tx.get("from") or "").lower(),
            "to": (tx.get("to") or "").lower(),
            "value": tx.get("value", "0x0"),
            "nonce": hex_to_int(tx.get("nonce")),
            "gas": tx.get("gas", ""),
            "gas_price": tx.get("gasPrice", ""),
            "input_prefix": (tx.get("input") or "")[:10],  # method selector
        })
    return rows


def extract_all_addresses_from_block(
    chain: ChainConfig,
    block_payload: Dict[str, Any],
) -> List[AddressRecord]:
    """
    Extract **both** from and to addresses from every transaction in a block.
    Returns one AddressRecord per (address, role) occurrence.
    """
    block_number = hex_to_int(block_payload.get("number")) or 0
    block_ts = hex_to_int(block_payload.get("timestamp"))
    txs = block_payload.get("transactions", []) or []

    records: List[AddressRecord] = []

    for tx in txs:
        tx_from = sanitize_address(tx.get("from"))
        tx_to = sanitize_address(tx.get("to"))
        tx_hash = tx.get("hash") or ""
        nonce = hex_to_int(tx.get("nonce"))

        if tx_from:
            records.append(AddressRecord(
                chain=chain.name,
                address=tx_from,
                role="sender",
                source_block_number=block_number,
                block_timestamp=block_ts,
                tx_hash=tx_hash,
                nonce=nonce,
            ))
        if tx_to:
            records.append(AddressRecord(
                chain=chain.name,
                address=tx_to,
                role="receiver",
                source_block_number=block_number,
                block_timestamp=block_ts,
                tx_hash=tx_hash,
                nonce=None,
            ))

    return records


def fetch_and_extract_blocks(
    client: MultiProviderClient,
    app_config: AppConfig,
    chain: ChainConfig,
    block_numbers: Sequence[int],
    *,
    raw_dir: Path,
    max_addresses: Optional[int] = None,
) -> List[AddressRecord]:
    """
    Fetch blocks from the API (or cache) and extract all unique
    (address, role) pairs. Saves each block as {chain}_block{num}_raw.csv.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[AddressRecord] = []
    seen: Set[str] = set()  # "address:role"

    for block_number in block_numbers:
        # Check CSV cache first (new format)
        csv_path = raw_dir / f"{chain.name}_block{block_number}_raw.csv"
        json_path = raw_dir / f"block_{block_number}.json"  # legacy

        block_payload: Optional[Dict[str, Any]] = None

        if app_config.sampling.resume_from_cache:
            if csv_path.exists():
                # CSV cache hit — we still need JSON for address extraction
                if json_path.exists():
                    block_payload = read_json(json_path)
                else:
                    # Re-fetch; CSV exists but JSON doesn't
                    pass
            elif json_path.exists():
                # Legacy JSON cache — migrate to CSV too
                block_payload = read_json(json_path)

        if block_payload is None:
            logger.info("[%s] Fetching block %s", chain.name, block_number)
            block_payload = client.get_block_by_number(
                chain=chain, block_number=block_number, full_transactions=True,
            )

        # Save JSON payload (for re-extraction)
        if app_config.sampling.save_raw_block_payloads and not json_path.exists():
            write_json(json_path, block_payload)

        # Save as {chain}_block{num}_raw.csv
        if not csv_path.exists():
            csv_rows = _block_payload_to_csv_rows(chain, block_payload)
            write_csv(csv_path, csv_rows)

        block_records = extract_all_addresses_from_block(chain, block_payload)

        for rec in block_records:
            key = f"{rec.address}:{rec.role}"
            if key in seen:
                continue
            all_records.append(rec)
            seen.add(key)

            if max_addresses and len(all_records) >= max_addresses:
                break
        if max_addresses and len(all_records) >= max_addresses:
            break

    logger.info("[%s] Extracted %s address-role pairs from %s blocks",
                chain.name, len(all_records), len(block_numbers))
    return all_records


# =====================================================================
# Active / Passive classification
# =====================================================================

def classify_active_passive(
    records: Sequence[AddressRecord],
) -> Dict[str, List[str]]:
    """
    Split sampled addresses into active and passive sets.

    Active  = address appeared as 'sender' (from) AND nonce >= 1
    Passive = address appeared only as 'receiver' (to), never as sender
    """
    sender_addresses: Set[str] = set()
    receiver_addresses: Set[str] = set()
    nonce_map: Dict[str, int] = {}

    for rec in records:
        if rec.role == "sender":
            sender_addresses.add(rec.address)
            if rec.nonce is not None:
                prev = nonce_map.get(rec.address, 0)
                nonce_map[rec.address] = max(prev, rec.nonce)
        elif rec.role == "receiver":
            receiver_addresses.add(rec.address)

    active = sorted({
        addr for addr in sender_addresses
        if nonce_map.get(addr, 0) >= 1
    })
    passive = sorted(receiver_addresses - sender_addresses)

    return {"active": active, "passive": passive}


# =====================================================================
# EOA filtering
# =====================================================================

def filter_eoa_addresses(
    client: MultiProviderClient,
    app_config: AppConfig,
    chain: ChainConfig,
    addresses: Sequence[str],
    *,
    raw_dir: Path,
    max_addresses: Optional[int] = None,
) -> List[str]:
    """
    Check eth_getCode for each address; return only EOAs (empty code).
    Caches code checks to disk.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    eoas: List[str] = []
    inspected = 0

    for address in addresses:
        if max_addresses and inspected >= max_addresses:
            break

        cache_path = raw_dir / f"{address}.json"

        if app_config.sampling.resume_from_cache and cache_path.exists():
            payload = read_json(cache_path)
            code = payload["result"]
        else:
            code = client.get_code(chain=chain, address=address)
            payload = {
                "address": address,
                "chain": chain.name,
                "chain_id": chain.chain_id,
                "result": code,
                "fetched_at": utc_now_iso(),
            }
            write_json(cache_path, payload)

        if isinstance(code, str) and code.lower() in {"0x", "0x0"}:
            eoas.append(address)

        inspected += 1

    logger.info("[%s] EOA filter: %s / %s are EOAs", chain.name, len(eoas), inspected)
    return eoas


# =====================================================================
# Pipeline for ONE chain (called inside thread)
# =====================================================================

def run_chain_pipeline(
    client: MultiProviderClient,
    app_config: AppConfig,
    chain: ChainConfig,
    window_blocks: int,
) -> Dict[str, Any]:
    """
    Full Person-1 pipeline for one (chain, window_blocks) pair:

    1. Get latest block → compute [start, end] range
    2. Decide which blocks to fetch (all if small, sampled if large)
    3. Fetch blocks → extract all from + to addresses
       → save each as {chain}_block{num}_raw.csv
    4. Classify into active / passive
    5. EOA-filter both sets
    6. Save manifests, address lists, active/passive CSVs

    Returns a summary dict.
    """
    wl = window_label(window_blocks)
    wpaths = get_window_paths(app_config.paths, window_blocks)

    # --- directories ---
    raw_blocks_dir = wpaths["raw"] / "block_samples" / chain.name
    raw_code_dir = wpaths["raw"] / "address_code" / chain.name
    interim_dir = wpaths["interim"] / chain.name
    for d in [raw_blocks_dir, raw_code_dir, interim_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Get latest block and compute range
    latest_block = client.get_latest_block_number(chain)
    latest_payload = client.get_block_by_number(chain, latest_block, full_transactions=False)
    latest_ts = hex_to_int(latest_payload.get("timestamp"))

    start_block, end_block = compute_block_range(latest_block, window_blocks)

    # 2. Decide which blocks to fetch
    blocks_to_fetch = plan_blocks_to_fetch(
        start_block, end_block, app_config.sampling.max_blocks_to_fetch,
    )

    logger.info(
        "[%s/%s] Range [%s → %s] (%s blocks in window), fetching %s blocks",
        chain.name, wl, start_block, end_block, window_blocks, len(blocks_to_fetch),
    )

    # Save manifest
    manifest = {
        "created_at": utc_now_iso(),
        "chain": chain.name,
        "chain_id": chain.chain_id,
        "window_blocks": window_blocks,
        "latest_block": latest_block,
        "latest_block_timestamp": latest_ts,
        "start_block": start_block,
        "end_block": end_block,
        "blocks_in_window": window_blocks,
        "blocks_fetched": len(blocks_to_fetch),
        "fetched_all_blocks": len(blocks_to_fetch) == window_blocks,
        "sampled_block_numbers": blocks_to_fetch,
    }
    write_json(interim_dir / "sampling_manifest.json", manifest)

    # 3. Fetch blocks and extract all addresses (from + to)
    #    Each block is also saved as {chain}_block{num}_raw.csv
    all_records = fetch_and_extract_blocks(
        client, app_config, chain, blocks_to_fetch,
        raw_dir=raw_blocks_dir,
        max_addresses=app_config.sampling.max_seed_addresses_per_chain * 2,
    )

    write_csv(
        interim_dir / "all_sampled_addresses.csv",
        [{"address": r.address, "chain": r.chain, "role": r.role,
          "block": r.source_block_number, "nonce": r.nonce} for r in all_records],
    )

    # 4. Classify active / passive (pre-EOA-filter)
    ap = classify_active_passive(all_records)

    # 5. EOA-filter both sets
    logger.info("[%s/%s] Filtering %s active + %s passive addresses for EOAs...",
                chain.name, wl, len(ap["active"]), len(ap["passive"]))

    active_eoas = filter_eoa_addresses(
        client, app_config, chain, ap["active"],
        raw_dir=raw_code_dir,
        max_addresses=app_config.sampling.max_eoa_filter_addresses,
    )
    passive_eoas = filter_eoa_addresses(
        client, app_config, chain, ap["passive"],
        raw_dir=raw_code_dir,
        max_addresses=app_config.sampling.max_eoa_filter_addresses,
    )

    # 6. Save final CSVs
    write_csv(interim_dir / "active_eoa_addresses.csv",
              [{"address": a, "chain": chain.name, "role": "active"} for a in active_eoas])
    write_csv(interim_dir / "passive_eoa_addresses.csv",
              [{"address": a, "chain": chain.name, "role": "passive"} for a in passive_eoas])
    write_csv(interim_dir / "all_eoa_addresses.csv",
              [{"address": a, "chain": chain.name, "role": "active"} for a in active_eoas] +
              [{"address": a, "chain": chain.name, "role": "passive"} for a in passive_eoas])

    summary = {
        "chain": chain.name,
        "window_blocks": window_blocks,
        "blocks_in_window": window_blocks,
        "blocks_fetched": len(blocks_to_fetch),
        "start_block": start_block,
        "end_block": end_block,
        "total_addresses_extracted": len(all_records),
        "pre_eoa_active": len(ap["active"]),
        "pre_eoa_passive": len(ap["passive"]),
        "active_eoas": len(active_eoas),
        "passive_eoas": len(passive_eoas),
        "total_eoas": len(active_eoas) + len(passive_eoas),
        "output_dir": str(interim_dir),
    }
    write_json(interim_dir / "pipeline_summary.json", summary)

    logger.info(
        "[%s/%s] Done — %s active EOAs, %s passive EOAs",
        chain.name, wl, len(active_eoas), len(passive_eoas),
    )
    return summary


# =====================================================================
# CONCURRENT pipeline — all chains in parallel
# =====================================================================

def run_concurrent_pipeline(
    client: MultiProviderClient,
    app_config: AppConfig,
    window_blocks: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Run the data-layer pipeline for ALL enabled chains concurrently.

    Uses ThreadPoolExecutor so each chain sends its API requests in
    its own thread (I/O-bound work — threads are fine here).

    Returns {chain_name: summary_dict}.
    """
    chains = [c for c in app_config.chains.values()]
    max_workers = min(len(chains), app_config.sampling.max_concurrent_chains)

    logger.info(
        "Starting concurrent pipeline: %d chains, %d threads, window=%d blocks",
        len(chains), max_workers, window_blocks,
    )

    results: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_chain = {
            pool.submit(run_chain_pipeline, client, app_config, chain, window_blocks): chain
            for chain in chains
        }

        for future in as_completed(future_to_chain):
            chain = future_to_chain[future]
            try:
                summary = future.result()
                results[chain.name] = summary
                logger.info("[%s] Pipeline completed successfully", chain.name)
            except Exception as exc:
                logger.error("[%s] Pipeline FAILED: %s", chain.name, exc, exc_info=True)
                results[chain.name] = {"chain": chain.name, "error": str(exc)}

    return results


# Legacy alias for backward compatibility
run_block_window_pipeline = run_chain_pipeline
