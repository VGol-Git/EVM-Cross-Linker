# Sampling module for wallet data

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from api_client import EtherscanClient
from config import AppConfig, ChainConfig

logger = logging.getLogger(__name__)


@dataclass
class SampledAddressRecord:
    chain: str
    address: str
    source_block_number: int
    block_timestamp: Optional[int]
    tx_hash: str
    tx_from: str
    tx_to: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def estimate_start_block(
    latest_block: int,
    latest_block_ts: int,
    observation_window_days: int,
    approx_block_time_seconds: int,
) -> int:
    seconds = observation_window_days * 24 * 60 * 60
    approx_blocks_back = math.ceil(seconds / approx_block_time_seconds)
    return max(0, latest_block - approx_blocks_back)


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

    # preserve order + deduplicate
    seen: Set[int] = set()
    out: List[int] = []
    for b in block_numbers:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def extract_from_addresses_from_block(
    chain: ChainConfig,
    block_payload: Dict[str, Any],
) -> List[SampledAddressRecord]:
    block_number = hex_to_int(block_payload.get("number")) or 0
    block_ts = hex_to_int(block_payload.get("timestamp"))
    txs = block_payload.get("transactions", []) or []

    records: List[SampledAddressRecord] = []

    for tx in txs:
        tx_from = sanitize_address(tx.get("from"))
        tx_to = sanitize_address(tx.get("to"))
        tx_hash = tx.get("hash") or ""

        if tx_from is None:
            continue

        records.append(
            SampledAddressRecord(
                chain=chain.name,
                address=tx_from,
                source_block_number=block_number,
                block_timestamp=block_ts,
                tx_hash=tx_hash,
                tx_from=tx_from,
                tx_to=tx_to,
            )
        )

    return records


def sample_addresses_from_blocks(
    client: EtherscanClient,
    app_config: AppConfig,
    chain: ChainConfig,
    block_numbers: Sequence[int],
    *,
    max_seed_addresses: Optional[int] = None,
) -> List[SampledAddressRecord]:
    raw_dir = app_config.paths.raw_dir / "block_samples" / chain.name
    interim_dir = app_config.paths.interim_dir / "seed_addresses"
    interim_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[SampledAddressRecord] = []
    seen_addresses: Set[str] = set()

    for block_number in block_numbers:
        block_path = raw_dir / f"block_{block_number}.json"

        if app_config.sampling.resume_from_cache and block_path.exists():
            logger.info("[%s] Loading cached block %s", chain.name, block_number)
            block_payload = read_json(block_path)
        else:
            logger.info("[%s] Fetching block %s", chain.name, block_number)
            block_payload = client.get_block_by_number(
                chain=chain,
                block_number=block_number,
                full_transactions=True,
            )
            if app_config.sampling.save_raw_block_payloads:
                write_json(block_path, block_payload)

        block_records = extract_from_addresses_from_block(chain, block_payload)

        for record in block_records:
            if record.address in seen_addresses:
                continue
            all_records.append(record)
            seen_addresses.add(record.address)

            if max_seed_addresses and len(all_records) >= max_seed_addresses:
                logger.info(
                    "[%s] Reached max_seed_addresses=%s",
                    chain.name,
                    max_seed_addresses,
                )
                break

        checkpoint_path = interim_dir / f"{chain.name}_sampled_addresses.csv"
        write_csv(checkpoint_path, [r.to_dict() for r in all_records])

        if max_seed_addresses and len(all_records) >= max_seed_addresses:
            break

    logger.info("[%s] Sampled %s unique seed addresses", chain.name, len(all_records))
    return all_records


def filter_eoa_addresses(
    client: EtherscanClient,
    app_config: AppConfig,
    chain: ChainConfig,
    sampled_records: Sequence[SampledAddressRecord],
    *,
    max_addresses: Optional[int] = None,
) -> List[SampledAddressRecord]:
    raw_dir = app_config.paths.raw_dir / "address_code" / chain.name
    interim_dir = app_config.paths.interim_dir / "eoa_filtered"
    interim_dir.mkdir(parents=True, exist_ok=True)

    output_records: List[SampledAddressRecord] = []
    inspected = 0

    for record in sampled_records:
        if max_addresses and inspected >= max_addresses:
            logger.info(
                "[%s] Reached max_addresses=%s during EOA filtering",
                chain.name,
                max_addresses,
            )
            break

        address = record.address
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
            output_records.append(record)

        inspected += 1

        if app_config.sampling.save_checkpoints and inspected % 50 == 0:
            checkpoint_path = interim_dir / f"{chain.name}_eoa_addresses.csv"
            write_csv(checkpoint_path, [r.to_dict() for r in output_records])

    final_path = interim_dir / f"{chain.name}_eoa_addresses.csv"
    write_csv(final_path, [r.to_dict() for r in output_records])

    logger.info(
        "[%s] EOA filter completed: %s EOAs out of %s inspected",
        chain.name,
        len(output_records),
        inspected,
    )
    return output_records


def build_sampling_manifest(
    app_config: AppConfig,
    chain: ChainConfig,
    latest_block: int,
    latest_block_ts: int,
    start_block: int,
    end_block: int,
    sampled_blocks: Sequence[int],
) -> Dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "chain": chain.name,
        "chain_id": chain.chain_id,
        "observation_window_days": app_config.sampling.observation_window_days,
        "latest_block": latest_block,
        "latest_block_timestamp": latest_block_ts,
        "start_block_estimate": start_block,
        "end_block": end_block,
        "blocks_per_chain": app_config.sampling.blocks_per_chain,
        "sampled_blocks": list(sampled_blocks),
    }


def plan_block_sample(
    client: EtherscanClient,
    app_config: AppConfig,
    chain: ChainConfig,
    *,
    approx_block_time_seconds: int,
) -> List[int]:
    latest_block = client.get_latest_block_number(chain)
    latest_block_payload = client.get_block_by_number(
        chain=chain,
        block_number=latest_block,
        full_transactions=False,
    )

    latest_block_ts = hex_to_int(latest_block_payload.get("timestamp"))
    if latest_block_ts is None:
        raise RuntimeError(f"Could not resolve latest block timestamp for {chain.name}")

    start_block = estimate_start_block(
        latest_block=latest_block,
        latest_block_ts=latest_block_ts,
        observation_window_days=app_config.sampling.observation_window_days,
        approx_block_time_seconds=approx_block_time_seconds,
    )
    end_block = latest_block

    sampled_blocks = evenly_spaced_block_numbers(
        start_block=start_block,
        end_block=end_block,
        n_blocks=app_config.sampling.blocks_per_chain,
    )

    manifest = build_sampling_manifest(
        app_config=app_config,
        chain=chain,
        latest_block=latest_block,
        latest_block_ts=latest_block_ts,
        start_block=start_block,
        end_block=end_block,
        sampled_blocks=sampled_blocks,
    )
    manifest_path = app_config.paths.interim_dir / "sampling_manifests" / f"{chain.name}.json"
    write_json(manifest_path, manifest)

    logger.info(
        "[%s] Planned %s sample blocks from %s to %s",
        chain.name,
        len(sampled_blocks),
        start_block,
        end_block,
    )
    return sampled_blocks


def run_seed_sampling_for_chain(
    client: EtherscanClient,
    app_config: AppConfig,
    chain: ChainConfig,
    *,
    approx_block_time_seconds: int,
) -> List[SampledAddressRecord]:
    sampled_blocks = plan_block_sample(
        client=client,
        app_config=app_config,
        chain=chain,
        approx_block_time_seconds=approx_block_time_seconds,
    )

    sampled_records = sample_addresses_from_blocks(
        client=client,
        app_config=app_config,
        chain=chain,
        block_numbers=sampled_blocks,
        max_seed_addresses=app_config.sampling.max_seed_addresses_per_chain,
    )

    eoa_records = filter_eoa_addresses(
        client=client,
        app_config=app_config,
        chain=chain,
        sampled_records=sampled_records,
        max_addresses=app_config.sampling.max_eoa_filter_addresses,
    )

    return eoa_records


def export_address_only_csv(
    path: Path,
    sampled_records: Sequence[SampledAddressRecord],
) -> None:
    rows = [{"address": r.address, "chain": r.chain} for r in sampled_records]
    write_csv(path, rows)