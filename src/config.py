# config.py
# Configuration for EVM cross-chain wallet correlation project
# Block-window version (1 / 10 / 100 blocks)

from __future__ import annotations

import json
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
    if raw_value is None or not str(raw_value).strip():
        values = list(default)
    else:
        parts = [part.strip() for part in str(raw_value).split(",")]
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

    unique_sorted = sorted(set(values))
    return tuple(unique_sorted)


def _parse_non_empty_str(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def _load_json_config() -> dict:
    config_path = os.getenv("CONFIG_JSON_PATH")
    if not config_path:
        return {}

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CONFIG_JSON_PATH does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("JSON config root must be an object/dict.")

    return payload


def _json_get(config: dict, *keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _get_value(
    env_name: str,
    json_config: dict,
    json_keys: Tuple[str, ...],
    default=None,
):
    env_value = os.getenv(env_name)
    if env_value is not None and str(env_value).strip() != "":
        return env_value

    json_value = _json_get(json_config, *json_keys, default=None)
    if json_value is not None:
        return json_value

    return default


def _get_bool(
    env_name: str,
    json_config: dict,
    json_keys: Tuple[str, ...],
    default: bool,
) -> bool:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return _str_to_bool(env_value, default=default)

    json_value = _json_get(json_config, *json_keys, default=None)
    if json_value is not None:
        if isinstance(json_value, bool):
            return json_value
        return _str_to_bool(str(json_value), default=default)

    return default


def _get_int(
    env_name: str,
    json_config: dict,
    json_keys: Tuple[str, ...],
    default: int,
) -> int:
    value = _get_value(env_name, json_config, json_keys, default=default)
    return int(value)


def _get_float(
    env_name: str,
    json_config: dict,
    json_keys: Tuple[str, ...],
    default: float,
) -> float:
    value = _get_value(env_name, json_config, json_keys, default=default)
    return float(value)


def _get_block_windows(json_config: dict) -> Tuple[int, ...]:
    env_raw = os.getenv("OBSERVATION_WINDOWS_BLOCKS")
    if env_raw is not None and env_raw.strip():
        return _parse_int_list(env_raw, default=(1, 10, 100))

    json_value = _json_get(
        json_config,
        "sampling",
        "observation_windows_blocks",
        default=None,
    )
    if json_value is not None:
        if not isinstance(json_value, list):
            raise ValueError(
                "sampling.observation_windows_blocks in JSON must be a list of ints"
            )
        values = [int(x) for x in json_value]
        if not values or any(v <= 0 for v in values):
            raise ValueError(
                "sampling.observation_windows_blocks must contain positive integers"
            )
        return tuple(sorted(set(values)))

    return (1, 10, 100)


def _looks_like_explorer_api_url(value: Optional[str]) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return (
        "api.etherscan.io" in normalized
        or "etherscan.io/v2/api" in normalized
        or normalized.endswith("/v2/api")
        or "/api?module=" in normalized
    )


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
    timeout_seconds: int = 30
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    requests_per_second: float = 4.0
    max_concurrency: int = 8
    batch_size: int = 25
    user_agent: str = "evm-cross-chain-correlation/2.0"

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

            if chain.rpc_url and _looks_like_explorer_api_url(chain.rpc_url):
                raise ValueError(
                    f"{chain.name}: rpc_url looks like an explorer API endpoint, "
                    f"not a JSON-RPC URL: {chain.rpc_url}"
                )

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


def build_paths(
    project_root: str | Path | None = None,
    json_config: Optional[dict] = None,
) -> PathConfig:
    json_config = json_config or {}

    if project_root is None:
        project_root = _get_value(
            "PROJECT_ROOT",
            json_config,
            ("project_root",),
            default=Path.cwd(),
        )

    root = Path(project_root).expanduser().resolve()

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
    json_config: dict,
    explorer_base_env_name: Optional[str] = None,
    explorer_api_base_url: Optional[str] = None,
) -> ChainConfig:
    json_chain = _json_get(json_config, "chains", name, default={}) or {}

    rpc_url = _parse_non_empty_str(os.getenv(rpc_env_name))
    if rpc_url is None:
        rpc_url = _parse_non_empty_str(json_chain.get("rpc_url"))

    explorer_api_key = _parse_non_empty_str(os.getenv(explorer_key_env_name))
    if explorer_api_key is None:
        explorer_api_key = _parse_non_empty_str(json_chain.get("explorer_api_key"))
    if explorer_api_key is None:
        explorer_api_key = _parse_non_empty_str(os.getenv("ETHERSCAN_API_KEY"))

    enabled_env = os.getenv(enabled_env_name)
    if enabled_env is not None:
        enabled = _str_to_bool(enabled_env, default=enabled_default)
    else:
        enabled = bool(json_chain.get("enabled", enabled_default))

    chain_base_from_env = None
    if explorer_base_env_name:
        chain_base_from_env = _parse_non_empty_str(os.getenv(explorer_base_env_name))

    chain_base_from_json = _parse_non_empty_str(
        json_chain.get("explorer_api_base_url")
    )

    global_base_from_env = _parse_non_empty_str(os.getenv("ETHERSCAN_BASE_URL"))
    global_base_from_json = _parse_non_empty_str(
        _json_get(json_config, "api", "explorer_api_base_url", default=None)
    )

    resolved_explorer_base = (
        chain_base_from_env
        or chain_base_from_json
        or explorer_api_base_url
        or global_base_from_env
        or global_base_from_json
        or "https://api.etherscan.io/v2/api"
    )

    return ChainConfig(
        name=name,
        chain_id=chain_id,
        approx_block_time_seconds=approx_block_time_seconds,
        rpc_url=rpc_url,
        explorer_api_base_url=resolved_explorer_base,
        explorer_api_key=explorer_api_key,
        enabled=enabled,
    )


def build_default_chains(json_config: dict) -> Dict[str, ChainConfig]:
    return {
        "ethereum": _build_chain(
            name="ethereum",
            chain_id=1,
            approx_block_time_seconds=12.0,
            rpc_env_name="ETHEREUM_RPC_URL",
            explorer_key_env_name="ETHEREUM_EXPLORER_API_KEY",
            enabled_default=True,
            enabled_env_name="ENABLE_ETHEREUM",
            explorer_base_env_name="ETHEREUM_EXPLORER_BASE_URL",
            json_config=json_config,
        ),
        "polygon": _build_chain(
            name="polygon",
            chain_id=137,
            approx_block_time_seconds=2.0,
            rpc_env_name="POLYGON_RPC_URL",
            explorer_key_env_name="POLYGON_EXPLORER_API_KEY",
            enabled_default=True,
            enabled_env_name="ENABLE_POLYGON",
            explorer_base_env_name="POLYGON_EXPLORER_BASE_URL",
            json_config=json_config,
        ),
        "bnb": _build_chain(
            name="bnb",
            chain_id=56,
            approx_block_time_seconds=0.75,
            rpc_env_name="BNB_RPC_URL",
            explorer_key_env_name="BNB_EXPLORER_API_KEY",
            enabled_default=True,
            enabled_env_name="ENABLE_BNB",
            explorer_base_env_name="BNB_EXPLORER_BASE_URL",
            json_config=json_config,
        ),
        "avalanche": _build_chain(
            name="avalanche",
            chain_id=43114,
            approx_block_time_seconds=2.0,
            rpc_env_name="AVALANCHE_RPC_URL",
            explorer_key_env_name="AVALANCHE_EXPLORER_API_KEY",
            enabled_default=False,
            enabled_env_name="ENABLE_AVALANCHE",
            explorer_base_env_name="AVALANCHE_EXPLORER_BASE_URL",
            json_config=json_config,
        ),
    }


# -----------------------------
# Loaders
# -----------------------------


def load_config(project_root: str | Path | None = None) -> AppConfig:
    json_config = _load_json_config()

    paths = build_paths(project_root=project_root, json_config=json_config)
    ensure_directories(paths)

    windows = WindowConfig(
        block_counts=_get_block_windows(json_config)
    )

    api = APIConfig(
        timeout_seconds=_get_int(
            "API_TIMEOUT_SECONDS",
            json_config,
            ("api", "timeout_seconds"),
            30,
        ),
        max_retries=_get_int(
            "API_MAX_RETRIES",
            json_config,
            ("api", "max_retries"),
            5,
        ),
        backoff_base_seconds=_get_float(
            "API_BACKOFF_BASE_SECONDS",
            json_config,
            ("api", "backoff_base_seconds"),
            1.0,
        ),
        requests_per_second=_get_float(
            "API_REQUESTS_PER_SECOND",
            json_config,
            ("api", "requests_per_second"),
            4.0,
        ),
        max_concurrency=_get_int(
            "API_MAX_CONCURRENCY",
            json_config,
            ("api", "max_concurrency"),
            8,
        ),
        batch_size=_get_int(
            "API_BATCH_SIZE",
            json_config,
            ("api", "batch_size"),
            25,
        ),
        user_agent=str(
            _get_value(
                "API_USER_AGENT",
                json_config,
                ("api", "user_agent"),
                "evm-cross-chain-correlation/2.0",
            )
        ),
        use_rpc_first=_get_bool(
            "USE_RPC_FIRST",
            json_config,
            ("api", "use_rpc_first"),
            True,
        ),
        use_explorer_fallback=_get_bool(
            "USE_EXPLORER_FALLBACK",
            json_config,
            ("api", "use_explorer_fallback"),
            True,
        ),
    )

    storage = StorageConfig(
        table_format=str(
            _get_value(
                "TABLE_FORMAT",
                json_config,
                ("storage", "table_format"),
                "parquet",
            )
        ).strip().lower(),
        save_raw_block_payloads=_get_bool(
            "SAVE_RAW_BLOCK_PAYLOADS",
            json_config,
            ("storage", "save_raw_block_payloads"),
            True,
        ),
        save_raw_transaction_rows=_get_bool(
            "SAVE_RAW_TRANSACTION_ROWS",
            json_config,
            ("storage", "save_raw_transaction_rows"),
            True,
        ),
        save_checkpoints=_get_bool(
            "SAVE_CHECKPOINTS",
            json_config,
            ("storage", "save_checkpoints"),
            True,
        ),
        resume_from_cache=_get_bool(
            "RESUME_FROM_CACHE",
            json_config,
            ("storage", "resume_from_cache"),
            True,
        ),
    )

    max_code_lookup_raw = _get_value(
        "MAX_ADDRESSES_FOR_CODE_LOOKUP",
        json_config,
        ("sampling", "max_addresses_for_code_lookup"),
        default=None,
    )
    max_code_lookup = (
        int(max_code_lookup_raw)
        if max_code_lookup_raw not in (None, "", "None")
        else None
    )

    sampling = SamplingConfig(
        windows=windows,
        reference_block_tag=str(
            _get_value(
                "REFERENCE_BLOCK_TAG",
                json_config,
                ("sampling", "reference_block_tag"),
                "latest",
            )
        ).strip().lower(),
        full_transactions=_get_bool(
            "FULL_TRANSACTIONS",
            json_config,
            ("sampling", "full_transactions"),
            True,
        ),
        max_addresses_for_code_lookup=max_code_lookup,
    )

    all_chains = build_default_chains(json_config=json_config)
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