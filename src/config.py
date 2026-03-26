# Configuration file for cross-chain wallet profiling project

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Block window — SET THIS before running the pipeline.
# This is the single control variable: how many recent blocks to sample.
# Recommended values: 1, 10, 1000, 10000
# ---------------------------------------------------------------------------
BLOCK_WINDOW: int = int(os.getenv("BLOCK_WINDOW", "1000"))

# Approximate block times (seconds) — used only for time estimates in logs
APPROX_BLOCK_TIMES: Dict[str, float] = {
    "ethereum": 12.0,
    "polygon": 2.0,
    "bnb": 3.0,
    "avalanche": 2.0,
    "arbitrum": 1.0,
    "sonic": 2.0,
}


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
    etherscan_api_key: str = ""            # may be empty when using other RPCs
    base_url: str = "https://api.etherscan.io/v2/api"
    requests_per_second: float = 2.8
    timeout_seconds: int = 30
    max_retries: int = 5
    backoff_base_seconds: float = 1.2
    page_size: int = 100
    user_agent: str = "cross-chain-wallet-profiler/1.0"


@dataclass
class SamplingConfig:
    observation_window_blocks: int = 1000
    max_blocks_to_fetch: int = 50       # if window > this, sample evenly
    max_seed_addresses_per_chain: int = 1000
    max_eoa_filter_addresses: int = 2000
    max_concurrent_chains: int = 3      # threads for parallel chain fetching
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
    """
    Default chain set aligned with professor requirements:
      - Ethereum, Polygon, BNB  → enabled by default
      - Avalanche               → optional 4th network
      - Arbitrum, Sonic         → disabled (not in study scope)
    """
    return {
        "ethereum": ChainConfig(name="ethereum", chain_id=1, enabled=True),
        "polygon": ChainConfig(name="polygon", chain_id=137, enabled=True),
        "bnb": ChainConfig(
            name="bnb",
            chain_id=56,
            enabled=_str_to_bool(os.getenv("ENABLE_BNB"), default=True),
        ),
        # optional 4th network:
        "avalanche": ChainConfig(
            name="avalanche",
            chain_id=43114,
            enabled=_str_to_bool(os.getenv("ENABLE_AVALANCHE"), default=False),
        ),
        # not in study scope, kept for future use:
        "arbitrum": ChainConfig(
            name="arbitrum",
            chain_id=42161,
            enabled=_str_to_bool(os.getenv("ENABLE_ARBITRUM"), default=False),
        ),
        "sonic": ChainConfig(
            name="sonic",
            chain_id=146,
            enabled=_str_to_bool(os.getenv("ENABLE_SONIC"), default=False),
        ),
    }


def load_config(project_root: str | Path | None = None) -> AppConfig:
    paths = build_paths(project_root=project_root)
    ensure_directories(paths)

    api = APIConfig(
        etherscan_api_key=os.getenv("ETHERSCAN_API_KEY", "").strip(),
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
        observation_window_blocks=BLOCK_WINDOW,
        max_blocks_to_fetch=int(os.getenv("MAX_BLOCKS_TO_FETCH", "50")),
        max_seed_addresses_per_chain=int(
            os.getenv("MAX_SEED_ADDRESSES_PER_CHAIN", "1000")
        ),
        max_eoa_filter_addresses=int(os.getenv("MAX_EOA_FILTER_ADDRESSES", "2000")),
        max_concurrent_chains=int(os.getenv("MAX_CONCURRENT_CHAINS", "3")),
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


# ---------------------------------------------------------------------------
# Block-window helpers
# ---------------------------------------------------------------------------

def window_label(window_blocks: int) -> str:
    """Return a short label for directory naming, e.g. '1blk', '10blk', '10000blk'."""
    return f"{window_blocks}blk"


def get_window_paths(base_paths: PathConfig, window_blocks: int) -> Dict[str, Path]:
    """
    Return window-namespaced sub-directories under the existing path layout.

    Structure:
        data/raw/{window}/          — raw block payloads & code checks
        data/interim/{window}/      — seed addresses, EOA lists, feature checkpoints
        data/processed/{window}/    — final feature tables
        outputs/tables/{window}/    — CSV exports
        outputs/figures/{window}/   — charts
    """
    wl = window_label(window_blocks)
    paths = {
        "raw": base_paths.raw_dir / wl,
        "interim": base_paths.interim_dir / wl,
        "processed": base_paths.processed_dir / wl,
        "tables": base_paths.tables_dir / wl,
        "figures": base_paths.figures_dir / wl,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_config_for_window(
    window_blocks: int,
    project_root: str | Path | None = None,
) -> Tuple[AppConfig, Dict[str, Path]]:
    """
    Load AppConfig with observation_window_blocks set to *window_blocks*
    and return the window-specific path dict alongside it.
    """
    cfg = load_config(project_root=project_root)
    cfg.sampling.observation_window_blocks = window_blocks
    wpaths = get_window_paths(cfg.paths, window_blocks)
    return cfg, wpaths
