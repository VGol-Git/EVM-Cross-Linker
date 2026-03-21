# Configuration file for cross-chain wallet profiling project

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ChainConfig:
    name: str
    chain_id: int
    enabled: bool = True


@dataclass
class PathConfig:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path


@dataclass
class APIConfig:
    etherscan_api_key: str
    base_url: str = "https://api.etherscan.io/v2/api"
    requests_per_second: float = 2.8
    timeout_seconds: int = 30
    max_retries: int = 5
    backoff_base_seconds: float = 1.2
    page_size: int = 100
    user_agent: str = "cross-chain-wallet-profiler/1.0"


@dataclass
class SamplingConfig:
    observation_window_days: int = 180
    blocks_per_chain: int = 50
    max_seed_addresses_per_chain: int = 1000
    max_eoa_filter_addresses: int = 2000
    resume_from_cache: bool = True
    save_raw_block_payloads: bool = True
    save_checkpoints: bool = True


@dataclass
class AppConfig:
    paths: PathConfig
    api: APIConfig
    sampling: SamplingConfig
    chains: Dict[str, ChainConfig] = field(default_factory=dict)


def _str_to_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_paths(project_root: str | Path | None = None) -> PathConfig:
    if project_root is None:
        project_root = os.getenv("PROJECT_ROOT", Path.cwd())
    root = Path(project_root).resolve()

    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    outputs_dir = root / "outputs"
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    logs_dir = root / "logs"

    return PathConfig(
        project_root=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        outputs_dir=outputs_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
    )


def ensure_directories(paths: PathConfig) -> None:
    for path in [
        paths.project_root,
        paths.data_dir,
        paths.raw_dir,
        paths.interim_dir,
        paths.processed_dir,
        paths.outputs_dir,
        paths.figures_dir,
        paths.tables_dir,
        paths.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_default_chains() -> Dict[str, ChainConfig]:
    return {
        "ethereum": ChainConfig(name="ethereum", chain_id=1, enabled=True),
        "polygon": ChainConfig(name="polygon", chain_id=137, enabled=True),
        "arbitrum": ChainConfig(name="arbitrum", chain_id=42161, enabled=True),
        "sonic": ChainConfig(name="sonic", chain_id=146, enabled=True),
        # optional:
        "bnb": ChainConfig(
            name="bnb",
            chain_id=56,
            enabled=_str_to_bool(os.getenv("ENABLE_BNB"), default=False),
        ),
        "avalanche": ChainConfig(
            name="avalanche",
            chain_id=43114,
            enabled=_str_to_bool(os.getenv("ENABLE_AVALANCHE"), default=False),
        ),
    }


def load_config(project_root: str | Path | None = None) -> AppConfig:
    paths = build_paths(project_root=project_root)
    ensure_directories(paths)

    api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ETHERSCAN_API_KEY is not set. "
            "Set it in your environment before running the pipeline."
        )

    api = APIConfig(
        etherscan_api_key=api_key,
        base_url=os.getenv("ETHERSCAN_BASE_URL", "https://api.etherscan.io/v2/api"),
        requests_per_second=float(os.getenv("ETHERSCAN_RPS", "2.8")),
        timeout_seconds=int(os.getenv("ETHERSCAN_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("ETHERSCAN_MAX_RETRIES", "5")),
        backoff_base_seconds=float(os.getenv("ETHERSCAN_BACKOFF_BASE_SECONDS", "1.2")),
        page_size=int(os.getenv("ETHERSCAN_PAGE_SIZE", "100")),
        user_agent=os.getenv(
            "ETHERSCAN_USER_AGENT",
            "cross-chain-wallet-profiler/1.0",
        ),
    )

    sampling = SamplingConfig(
        observation_window_days=int(os.getenv("OBSERVATION_WINDOW_DAYS", "180")),
        blocks_per_chain=int(os.getenv("BLOCKS_PER_CHAIN", "50")),
        max_seed_addresses_per_chain=int(
            os.getenv("MAX_SEED_ADDRESSES_PER_CHAIN", "1000")
        ),
        max_eoa_filter_addresses=int(os.getenv("MAX_EOA_FILTER_ADDRESSES", "2000")),
        resume_from_cache=_str_to_bool(os.getenv("RESUME_FROM_CACHE"), default=True),
        save_raw_block_payloads=_str_to_bool(
            os.getenv("SAVE_RAW_BLOCK_PAYLOADS"), default=True
        ),
        save_checkpoints=_str_to_bool(os.getenv("SAVE_CHECKPOINTS"), default=True),
    )

    chains = {
        name: cfg for name, cfg in build_default_chains().items() if cfg.enabled
    }

    return AppConfig(
        paths=paths,
        api=api,
        sampling=sampling,
        chains=chains,
    )