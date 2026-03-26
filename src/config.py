# config.py
# Configuration for EVM cross-chain wallet correlation project
# Block-window version (1 / 10 / 100 blocks)

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(
    raw_value: Optional[str],
    default: Iterable[int],
) -> Tuple[int, ...]:
    if raw_value is None or not raw_value.strip():
        values = list(default)
    else:
        parts = [part.strip() for part in raw_value.split(",")]
        values = []
        for part in parts:
            if not part:
                continue
            try:
                parsed = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer in list: {part!r}. "
                    f"Expected comma-separated integers."
                ) from exc
            if parsed <= 0:
                raise ValueError(
                    f"Observation window must be > 0, got {parsed}."
                )
            values.append(parsed)

    if not values:
        raise ValueError("At least one observation window must be configured.")

    # unique + sorted for deterministic pipeline behavior
    unique_sorted = sorted(set(values))
    return tuple(unique_sorted)


def _parse_non_empty_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


# -----------------------------
# Config models
# -----------------------------


@dataclass(frozen=True)
class ChainConfig:
    name: str
    chain_id: int
    approx_block_time_seconds: float
    rpc_url: Optional[str] = None
    explorer_api_base_url: str = "https://api.etherscan.io/v2/api"
    explorer_api_key: Optional[str] = None
    enabled: bool = True

    @property
    def has_rpc(self) -> bool:
        return bool(self.rpc_url)

    @property
    def has_explorer_key(self) -> bool:
        return bool(self.explorer_api_key)


@dataclass(frozen=True)
class WindowConfig:
    block_counts: Tuple[int, ...] = (1, 10, 100)

    def validate(self) -> None:
        if not self.block_counts:
            raise ValueError("WindowConfig.block_counts cannot be empty.")
        if any(value <= 0 for value in self.block_counts):
            raise ValueError("All block windows must be positive integers.")


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

    raw_blocks_dir: Path
    raw_transactions_dir: Path
    raw_address_code_dir: Path

    interim_status_dir: Path
    interim_features_dir: Path

    processed_features_dir: Path
    processed_analysis_dir: Path
    processed_statistics_dir: Path

    def iter_all_dirs(self) -> List[Path]:
        return [
            self.project_root,
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.outputs_dir,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
            self.raw_blocks_dir,
            self.raw_transactions_dir,
            self.raw_address_code_dir,
            self.interim_status_dir,
            self.interim_features_dir,
            self.processed_features_dir,
            self.processed_analysis_dir,
            self.processed_statistics_dir,
        ]


@dataclass(frozen=True)
class APIConfig:
    # Transport / request behavior
    timeout_seconds: int = 30
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    requests_per_second: float = 4.0
    max_concurrency: int = 8
    batch_size: int = 25
    user_agent: str = "evm-cross-chain-correlation/2.0"

    # Client strategy
    use_rpc_first: bool = True
    use_explorer_fallback: bool = True


@dataclass(frozen=True)
class StorageConfig:
    table_format: str = "parquet"   # parquet | csv
    save_raw_block_payloads: bool = True
    save_raw_transaction_rows: bool = True
    save_checkpoints: bool = True
    resume_from_cache: bool = True

    def validate(self) -> None:
        if self.table_format not in {"parquet", "csv"}:
            raise ValueError(
                f"Unsupported table_format={self.table_format!r}. "
                f"Expected 'parquet' or 'csv'."
            )


@dataclass(frozen=True)
class SamplingConfig:
    windows: WindowConfig
    reference_block_tag: str = "latest"
    full_transactions: bool = True

    # Safety limits
    max_addresses_for_code_lookup: Optional[int] = None

    def validate(self) -> None:
        self.windows.validate()
        if self.reference_block_tag not in {"latest", "safe", "finalized"}:
            raise ValueError(
                "reference_block_tag must be one of: latest, safe, finalized"
            )


@dataclass
class AppConfig:
    paths: PathConfig
    api: APIConfig
    storage: StorageConfig
    sampling: SamplingConfig
    chains: Dict[str, ChainConfig] = field(default_factory=dict)

    @property
    def enabled_chains(self) -> List[ChainConfig]:
        return [chain for chain in self.chains.values() if chain.enabled]

    @property
    def enabled_chain_names(self) -> List[str]:
        return [chain.name for chain in self.enabled_chains]

    def get_chain(self, name: str) -> ChainConfig:
        try:
            return self.chains[name]
        except KeyError as exc:
            raise KeyError(f"Unknown chain name: {name}") from exc

    def validate(self) -> None:
        self.storage.validate()
        self.sampling.validate()

        if not self.enabled_chains:
            raise ValueError("No enabled chains configured.")

        for chain in self.enabled_chains:
            if chain.chain_id <= 0:
                raise ValueError(
                    f"Invalid chain_id for {chain.name}: {chain.chain_id}"
                )
            if chain.approx_block_time_seconds <= 0:
                raise ValueError(
                    f"Invalid approx_block_time_seconds for {chain.name}: "
                    f"{chain.approx_block_time_seconds}"
                )

        # At least one transport path should exist for each enabled chain.
        missing_transport = [
            chain.name
            for chain in self.enabled_chains
            if not chain.has_rpc and not chain.has_explorer_key
        ]
        if missing_transport:
            raise ValueError(
                "The following enabled chains have neither RPC URL nor explorer API key: "
                + ", ".join(missing_transport)
            )


# -----------------------------
# Path builders
# -----------------------------


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

    raw_blocks_dir = raw_dir / "blocks"
    raw_transactions_dir = raw_dir / "transactions"
    raw_address_code_dir = raw_dir / "address_code"

    interim_status_dir = interim_dir / "status"
    interim_features_dir = interim_dir / "features"

    processed_features_dir = processed_dir / "features"
    processed_analysis_dir = processed_dir / "analysis"
    processed_statistics_dir = processed_dir / "statistics"

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
        raw_blocks_dir=raw_blocks_dir,
        raw_transactions_dir=raw_transactions_dir,
        raw_address_code_dir=raw_address_code_dir,
        interim_status_dir=interim_status_dir,
        interim_features_dir=interim_features_dir,
        processed_features_dir=processed_features_dir,
        processed_analysis_dir=processed_analysis_dir,
        processed_statistics_dir=processed_statistics_dir,
    )


def ensure_directories(paths: PathConfig) -> None:
    for path in paths.iter_all_dirs():
        path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Chain builders
# -----------------------------


def _build_chain(
    *,
    name: str,
    chain_id: int,
    approx_block_time_seconds: float,
    rpc_env_name: str,
    explorer_key_env_name: str,
    enabled_default: bool,
    enabled_env_name: str,
    explorer_api_base_url: Optional[str] = None,
) -> ChainConfig:
    shared_etherscan_key = _parse_non_empty_str(os.getenv("ETHERSCAN_API_KEY"))
    explorer_api_key = _parse_non_empty_str(
        os.getenv(explorer_key_env_name)
    ) or shared_etherscan_key

    return ChainConfig(
        name=name,
        chain_id=chain_id,
        approx_block_time_seconds=approx_block_time_seconds,
        rpc_url=_parse_non_empty_str(os.getenv(rpc_env_name)),
        explorer_api_base_url=explorer_api_base_url
        or os.getenv("ETHERSCAN_BASE_URL", "https://api.etherscan.io/v2/api"),
        explorer_api_key=explorer_api_key,
        enabled=_str_to_bool(os.getenv(enabled_env_name), default=enabled_default),
    )


def build_default_chains() -> Dict[str, ChainConfig]:
    """
    Main project target according to the updated task:
    Ethereum, Polygon, BNB Smart Chain.
    Avalanche remains optional.
    """
    return {
        "ethereum": _build_chain(
            name="ethereum",
            chain_id=1,
            approx_block_time_seconds=12.0,
            rpc_env_name="ETHEREUM_RPC_URL",
            explorer_key_env_name="ETHEREUM_EXPLORER_API_KEY",
            enabled_default=True,
            enabled_env_name="ENABLE_ETHEREUM",
        ),
        "polygon": _build_chain(
            name="polygon",
            chain_id=137,
            approx_block_time_seconds=2.0,
            rpc_env_name="POLYGON_RPC_URL",
            explorer_key_env_name="POLYGON_EXPLORER_API_KEY",
            enabled_default=True,
            enabled_env_name="ENABLE_POLYGON",
        ),
        "bnb": _build_chain(
            name="bnb",
            chain_id=56,
            approx_block_time_seconds=0.75,
            rpc_env_name="BNB_RPC_URL",
            explorer_key_env_name="BNB_EXPLORER_API_KEY",
            enabled_default=True,
            enabled_env_name="ENABLE_BNB",
        ),
        "avalanche": _build_chain(
            name="avalanche",
            chain_id=43114,
            approx_block_time_seconds=2.0,
            rpc_env_name="AVALANCHE_RPC_URL",
            explorer_key_env_name="AVALANCHE_EXPLORER_API_KEY",
            enabled_default=False,
            enabled_env_name="ENABLE_AVALANCHE",
        ),
    }


# -----------------------------
# Loaders
# -----------------------------


def load_config(project_root: str | Path | None = None) -> AppConfig:
    paths = build_paths(project_root=project_root)
    ensure_directories(paths)

    windows = WindowConfig(
        block_counts=_parse_int_list(
            os.getenv("OBSERVATION_WINDOWS_BLOCKS"),
            default=(1, 10, 100),
        )
    )

    api = APIConfig(
        timeout_seconds=_env_int("API_TIMEOUT_SECONDS", 30),
        max_retries=_env_int("API_MAX_RETRIES", 5),
        backoff_base_seconds=_env_float("API_BACKOFF_BASE_SECONDS", 1.0),
        requests_per_second=_env_float("API_REQUESTS_PER_SECOND", 4.0),
        max_concurrency=_env_int("API_MAX_CONCURRENCY", 8),
        batch_size=_env_int("API_BATCH_SIZE", 25),
        user_agent=os.getenv(
            "API_USER_AGENT",
            "evm-cross-chain-correlation/2.0",
        ),
        use_rpc_first=_str_to_bool(os.getenv("USE_RPC_FIRST"), default=True),
        use_explorer_fallback=_str_to_bool(
            os.getenv("USE_EXPLORER_FALLBACK"),
            default=True,
        ),
    )

    storage = StorageConfig(
        table_format=os.getenv("TABLE_FORMAT", "parquet").strip().lower(),
        save_raw_block_payloads=_str_to_bool(
            os.getenv("SAVE_RAW_BLOCK_PAYLOADS"),
            default=True,
        ),
        save_raw_transaction_rows=_str_to_bool(
            os.getenv("SAVE_RAW_TRANSACTION_ROWS"),
            default=True,
        ),
        save_checkpoints=_str_to_bool(
            os.getenv("SAVE_CHECKPOINTS"),
            default=True,
        ),
        resume_from_cache=_str_to_bool(
            os.getenv("RESUME_FROM_CACHE"),
            default=True,
        ),
    )

    max_code_lookup_raw = _parse_non_empty_str(
        os.getenv("MAX_ADDRESSES_FOR_CODE_LOOKUP")
    )
    max_code_lookup = (
        int(max_code_lookup_raw) if max_code_lookup_raw is not None else None
    )

    sampling = SamplingConfig(
        windows=windows,
        reference_block_tag=os.getenv("REFERENCE_BLOCK_TAG", "latest").strip().lower(),
        full_transactions=_str_to_bool(
            os.getenv("FULL_TRANSACTIONS"),
            default=True,
        ),
        max_addresses_for_code_lookup=max_code_lookup,
    )

    all_chains = build_default_chains()
    chains = {name: chain for name, chain in all_chains.items() if chain.enabled}

    config = AppConfig(
        paths=paths,
        api=api,
        storage=storage,
        sampling=sampling,
        chains=chains,
    )
    config.validate()
    return config