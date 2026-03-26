# api_client.py
# Senior-oriented blockchain client for block-window ingestion:
# - exact last N blocks
# - eth_blockNumber / eth_getBlockByNumber / eth_getCode
# - RPC-first, explorer fallback
# - retry + rate limiting + bounded concurrency helpers

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from .config import APIConfig, ChainConfig

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]


# ============================================================
# Exceptions
# ============================================================


class BlockchainClientError(Exception):
    """Base exception for blockchain client failures."""


class NoAvailableTransportError(BlockchainClientError):
    """Raised when neither RPC nor explorer transport is available."""


class RPCTransportError(BlockchainClientError):
    """Raised when HTTP/network layer fails for RPC."""


class RPCResponseError(BlockchainClientError):
    """Raised when RPC returns malformed or error payload."""


class RPCRateLimitError(RPCTransportError):
    """Raised when RPC endpoint appears rate-limited."""


class ExplorerTransportError(BlockchainClientError):
    """Raised when HTTP/network layer fails for explorer API."""


class ExplorerAPIError(BlockchainClientError):
    """Raised when explorer returns malformed or error payload."""


class ExplorerRateLimitError(ExplorerAPIError):
    """Raised when explorer reports a rate limit issue."""


# ============================================================
# Data classes
# ============================================================


@dataclass(frozen=True)
class RequestMeta:
    chain_name: str
    chain_id: int
    transport: str
    operation: str

    def render(self) -> str:
        return (
            f"chain={self.chain_name} chain_id={self.chain_id} "
            f"transport={self.transport} operation={self.operation}"
        )


@dataclass(frozen=True)
class AddressCodeResult:
    address: str
    code: str
    is_eoa: bool


# ============================================================
# Rate limiter
# ============================================================


class RateLimiter:
    """
    Simple thread-safe spacing limiter.

    Guarantees at least 1 / requests_per_second seconds between outgoing requests
    across all threads that share this limiter.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self.interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_ts
            sleep_for = self.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call_ts = time.monotonic()


# ============================================================
# Main client
# ============================================================


class BlockchainClient:
    """
    RPC-first blockchain client for the updated block-window task.

    Primary responsibilities:
    - resolve latest/reference block number
    - fetch exact blocks by number
    - fetch address bytecode via eth_getCode
    - provide bounded-concurrency helpers for batch block/code retrieval

    Fallback order:
    1. RPC (preferred)
    2. Explorer proxy API (optional fallback)
    """

    def __init__(self, api_config: APIConfig) -> None:
        self.api_config = api_config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": api_config.user_agent})
        self.rate_limiter = RateLimiter(api_config.requests_per_second)
        self._request_ids = itertools.count(1)

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "BlockchainClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def get_reference_block_number(
        self,
        chain: ChainConfig,
        block_tag: str = "latest",
    ) -> int:
        """
        Resolve a reference block number.

        Supported tags:
        - latest
        - safe
        - finalized
        """
        normalized_tag = str(block_tag).strip().lower()
        if normalized_tag == "latest":
            return self.get_latest_block_number(chain)
        block_payload = self.get_block_by_tag(
            chain=chain,
            block_tag=normalized_tag,
            full_transactions=False,
        )
        block_number_hex = block_payload.get("number")
        return self._parse_hex_int(block_number_hex, field_name="number")

    def get_latest_block_number(self, chain: ChainConfig) -> int:
        result = self._rpc_first_call(
            chain=chain,
            rpc_method="eth_blockNumber",
            rpc_params=[],
            explorer_module="proxy",
            explorer_action="eth_blockNumber",
            explorer_extra_params=None,
            operation_name="eth_blockNumber",
        )
        return self._parse_hex_int(result, field_name="eth_blockNumber")

    def get_block_by_tag(
        self,
        chain: ChainConfig,
        block_tag: str,
        full_transactions: bool = True,
    ) -> JsonDict:
        tag = self._normalize_block_tag(block_tag)
        result = self._rpc_first_call(
            chain=chain,
            rpc_method="eth_getBlockByNumber",
            rpc_params=[tag, full_transactions],
            explorer_module="proxy",
            explorer_action="eth_getBlockByNumber",
            explorer_extra_params={
                "tag": tag,
                "boolean": "true" if full_transactions else "false",
            },
            operation_name=f"eth_getBlockByNumber(tag={tag})",
        )
        if not isinstance(result, dict):
            raise RPCResponseError(
                f"Expected dict block payload for chain={chain.name}, got {type(result)}"
            )
        return result

    def get_block_by_number(
        self,
        chain: ChainConfig,
        block_number: int,
        full_transactions: bool = True,
    ) -> JsonDict:
        if block_number < 0:
            raise ValueError(f"block_number must be >= 0, got {block_number}")
        return self.get_block_by_tag(
            chain=chain,
            block_tag=hex(block_number),
            full_transactions=full_transactions,
        )

    def get_blocks_by_numbers(
        self,
        chain: ChainConfig,
        block_numbers: Sequence[int],
        full_transactions: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Fetch many blocks concurrently while preserving input order.
        """
        normalized_numbers = self._deduplicate_preserve_order(block_numbers)
        if not normalized_numbers:
            return []

        if max_workers is None:
            max_workers = self.api_config.max_concurrency
        max_workers = max(1, int(max_workers))

        results: Dict[int, JsonDict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_block = {
                executor.submit(
                    self.get_block_by_number,
                    chain,
                    block_number,
                    full_transactions,
                ): block_number
                for block_number in normalized_numbers
            }

            for future in as_completed(future_to_block):
                block_number = future_to_block[future]
                results[block_number] = future.result()

        return [results[block_number] for block_number in normalized_numbers]

    def get_code(
        self,
        chain: ChainConfig,
        address: str,
        block_tag: str = "latest",
    ) -> str:
        address = self._normalize_address(address)
        tag = self._normalize_block_tag(block_tag)

        result = self._rpc_first_call(
            chain=chain,
            rpc_method="eth_getCode",
            rpc_params=[address, tag],
            explorer_module="proxy",
            explorer_action="eth_getCode",
            explorer_extra_params={"address": address, "tag": tag},
            operation_name=f"eth_getCode(address={address}, tag={tag})",
        )

        if not isinstance(result, str):
            raise RPCResponseError(
                f"Expected string bytecode for address={address} on {chain.name}, "
                f"got {type(result)}"
            )
        return result

    def get_codes(
        self,
        chain: ChainConfig,
        addresses: Sequence[str],
        block_tag: str = "latest",
        max_workers: Optional[int] = None,
    ) -> List[AddressCodeResult]:
        """
        Resolve code for many addresses concurrently while preserving input order.
        """
        normalized_addresses = self._normalize_address_list(addresses)
        if not normalized_addresses:
            return []

        if max_workers is None:
            max_workers = self.api_config.max_concurrency
        max_workers = max(1, int(max_workers))

        results: Dict[str, AddressCodeResult] = {}
        tag = self._normalize_block_tag(block_tag)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_address = {
                executor.submit(self.get_code, chain, address, tag): address
                for address in normalized_addresses
            }

            for future in as_completed(future_to_address):
                address = future_to_address[future]
                code = future.result()
                results[address] = AddressCodeResult(
                    address=address,
                    code=code,
                    is_eoa=self._is_eoa_code(code),
                )

        return [results[address] for address in normalized_addresses]

    def is_eoa(
        self,
        chain: ChainConfig,
        address: str,
        block_tag: str = "latest",
    ) -> bool:
        code = self.get_code(chain=chain, address=address, block_tag=block_tag)
        return self._is_eoa_code(code)

    def classify_addresses_as_eoa(
        self,
        chain: ChainConfig,
        addresses: Sequence[str],
        block_tag: str = "latest",
        max_workers: Optional[int] = None,
    ) -> Dict[str, bool]:
        """
        Convenience helper for EOA filtering stage.
        """
        code_results = self.get_codes(
            chain=chain,
            addresses=addresses,
            block_tag=block_tag,
            max_workers=max_workers,
        )
        return {item.address: item.is_eoa for item in code_results}

    # --------------------------------------------------------
    # RPC-first transport resolution
    # --------------------------------------------------------

    def _rpc_first_call(
        self,
        *,
        chain: ChainConfig,
        rpc_method: str,
        rpc_params: Sequence[Any],
        explorer_module: str,
        explorer_action: str,
        explorer_extra_params: Optional[Dict[str, Any]],
        operation_name: str,
    ) -> Any:
        last_error: Optional[Exception] = None

        if self.api_config.use_rpc_first and chain.rpc_url:
            try:
                return self._rpc_call(chain, rpc_method, rpc_params)
            except BlockchainClientError as exc:
                last_error = exc
                logger.warning(
                    "RPC failed for %s on %s, fallback=%s, error=%s",
                    operation_name,
                    chain.name,
                    self.api_config.use_explorer_fallback,
                    exc,
                )

        if self.api_config.use_explorer_fallback and chain.explorer_api_key:
            try:
                return self._explorer_proxy_call(
                    chain=chain,
                    module=explorer_module,
                    action=explorer_action,
                    extra_params=explorer_extra_params,
                )
            except BlockchainClientError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

        raise NoAvailableTransportError(
            f"No available transport for chain={chain.name}. "
            f"rpc_url set={bool(chain.rpc_url)}, "
            f"explorer_api_key set={bool(chain.explorer_api_key)}"
        )

    # --------------------------------------------------------
    # RPC internals
    # --------------------------------------------------------

    def _rpc_call(
        self,
        chain: ChainConfig,
        method: str,
        params: Sequence[Any],
    ) -> Any:
        if not chain.rpc_url:
            raise NoAvailableTransportError(
                f"RPC URL is not configured for chain={chain.name}"
            )

        meta = RequestMeta(
            chain_name=chain.name,
            chain_id=chain.chain_id,
            transport="rpc",
            operation=method,
        )

        last_error: Optional[Exception] = None
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._request_ids),
            "method": method,
            "params": list(params),
        }

        for attempt in range(1, self.api_config.max_retries + 1):
            self.rate_limiter.wait()

            try:
                response = self.session.post(
                    chain.rpc_url,
                    json=payload,
                    timeout=self.api_config.timeout_seconds,
                    headers={"Content-Type": "application/json"},
                )
            except requests.RequestException as exc:
                last_error = RPCTransportError(
                    f"Network error for {meta.render()}: {exc}"
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 429:
                last_error = RPCRateLimitError(
                    f"HTTP 429 for {meta.render()}: {response.text[:500]}"
                )
                self._sleep_backoff(attempt, multiplier=2.0)
                continue

            if response.status_code >= 400:
                last_error = RPCTransportError(
                    f"HTTP {response.status_code} for {meta.render()}: "
                    f"{response.text[:500]}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                decoded = response.json()
            except json.JSONDecodeError:
                last_error = RPCTransportError(
                    f"Non-JSON RPC response for {meta.render()}: {response.text[:500]}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                return self._unwrap_rpc_result(decoded, meta)
            except RPCRateLimitError as exc:
                last_error = exc
                self._sleep_backoff(attempt, multiplier=2.0)
            except RPCResponseError as exc:
                last_error = exc
                self._sleep_backoff(attempt)

        raise RPCTransportError(
            f"RPC request failed after {self.api_config.max_retries} attempts "
            f"for {meta.render()}"
        ) from last_error

    def _unwrap_rpc_result(self, payload: JsonDict, meta: RequestMeta) -> Any:
        if not isinstance(payload, dict):
            raise RPCResponseError(
                f"Unexpected RPC payload type for {meta.render()}: {type(payload)}"
            )

        error = payload.get("error")
        if error is not None:
            rendered = str(error).lower()
            if "rate limit" in rendered or "too many requests" in rendered:
                raise RPCRateLimitError(
                    f"RPC rate limit for {meta.render()}: {error}"
                )
            raise RPCResponseError(
                f"RPC error for {meta.render()}: {error}"
            )

        if "result" not in payload:
            raise RPCResponseError(
                f"Missing 'result' in RPC payload for {meta.render()}: {payload}"
            )

        return payload["result"]

    # --------------------------------------------------------
    # Explorer fallback internals
    # --------------------------------------------------------

    def _explorer_proxy_call(
        self,
        *,
        chain: ChainConfig,
        module: str,
        action: str,
        extra_params: Optional[Dict[str, Any]],
    ) -> Any:
        if not chain.explorer_api_key:
            raise NoAvailableTransportError(
                f"Explorer API key is not configured for chain={chain.name}"
            )

        meta = RequestMeta(
            chain_name=chain.name,
            chain_id=chain.chain_id,
            transport="explorer",
            operation=f"{module}.{action}",
        )

        params: Dict[str, Any] = {
            "chainid": str(chain.chain_id),
            "module": module,
            "action": action,
            "apikey": chain.explorer_api_key,
        }
        if extra_params:
            params.update(extra_params)

        last_error: Optional[Exception] = None

        for attempt in range(1, self.api_config.max_retries + 1):
            self.rate_limiter.wait()

            try:
                response = self.session.get(
                    chain.explorer_api_base_url,
                    params=params,
                    timeout=self.api_config.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = ExplorerTransportError(
                    f"Network error for {meta.render()}: {exc}"
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 429:
                last_error = ExplorerRateLimitError(
                    f"HTTP 429 for {meta.render()}: {response.text[:500]}"
                )
                self._sleep_backoff(attempt, multiplier=2.0)
                continue

            if response.status_code >= 400:
                last_error = ExplorerTransportError(
                    f"HTTP {response.status_code} for {meta.render()}: "
                    f"{response.text[:500]}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                decoded = response.json()
            except json.JSONDecodeError:
                last_error = ExplorerTransportError(
                    f"Non-JSON explorer response for {meta.render()}: "
                    f"{response.text[:500]}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                return self._unwrap_explorer_result(decoded, meta)
            except ExplorerRateLimitError as exc:
                last_error = exc
                self._sleep_backoff(attempt, multiplier=2.0)
            except ExplorerAPIError as exc:
                last_error = exc
                self._sleep_backoff(attempt)

        raise ExplorerTransportError(
            f"Explorer request failed after {self.api_config.max_retries} attempts "
            f"for {meta.render()}"
        ) from last_error

    def _unwrap_explorer_result(self, payload: JsonDict, meta: RequestMeta) -> Any:
        """
        Etherscan-style proxy endpoints usually return:
        {"jsonrpc":"2.0","id":...,"result":"0x..."}
        """
        if not isinstance(payload, dict):
            raise ExplorerAPIError(
                f"Unexpected explorer payload type for {meta.render()}: {type(payload)}"
            )

        if "jsonrpc" in payload:
            if "result" not in payload:
                raise ExplorerAPIError(
                    f"Missing 'result' in explorer proxy payload for {meta.render()}: "
                    f"{payload}"
                )
            return payload["result"]

        status = str(payload.get("status", "")).strip()
        message = str(payload.get("message", "")).strip()
        result = payload.get("result")

        if status == "1":
            return result

        rendered_result = str(result).lower() if result is not None else ""
        if "rate limit" in rendered_result or "max rate limit" in rendered_result:
            raise ExplorerRateLimitError(
                f"Explorer rate limit for {meta.render()}: "
                f"message={message} result={result}"
            )

        raise ExplorerAPIError(
            f"Explorer API error for {meta.render()}: "
            f"status={status} message={message} result={result}"
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _sleep_backoff(self, attempt: int, multiplier: float = 1.0) -> None:
        seconds = (
            self.api_config.backoff_base_seconds
            * multiplier
            * (2 ** (attempt - 1))
        )
        logger.warning("Retrying after %.2f seconds", seconds)
        time.sleep(seconds)

    @staticmethod
    def _parse_hex_int(value: Any, field_name: str) -> int:
        if not isinstance(value, str):
            raise RPCResponseError(
                f"Expected hex string for {field_name}, got {type(value)}"
            )
        try:
            return int(value, 16)
        except ValueError as exc:
            raise RPCResponseError(
                f"Invalid hex integer for {field_name}: {value!r}"
            ) from exc

    @staticmethod
    def _normalize_block_tag(block_tag: str) -> str:
        tag = str(block_tag).strip().lower()
        if tag in {"latest", "safe", "finalized"}:
            return tag

        # Accept raw hex tag like 0x1234
        if tag.startswith("0x"):
            int(tag, 16)  # validation
            return tag

        raise ValueError(
            f"Unsupported block tag {block_tag!r}. "
            f"Expected latest/safe/finalized or 0x-prefixed hex block number."
        )

    @staticmethod
    def _normalize_address(address: str) -> str:
        if not isinstance(address, str):
            raise ValueError(f"Address must be a string, got {type(address)}")
        address = address.strip().lower()
        if not address:
            raise ValueError("Address cannot be empty.")
        if not address.startswith("0x"):
            raise ValueError(f"Address must start with '0x': {address!r}")
        if len(address) != 42:
            raise ValueError(
                f"Address must be 42 characters long, got {len(address)}: {address!r}"
            )
        return address

    @classmethod
    def _normalize_address_list(cls, addresses: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for address in addresses:
            normalized = cls._normalize_address(address)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    @staticmethod
    def _deduplicate_preserve_order(values: Sequence[int]) -> List[int]:
        seen = set()
        out: List[int] = []
        for value in values:
            numeric = int(value)
            if numeric in seen:
                continue
            seen.add(numeric)
            out.append(numeric)
        return out

    @staticmethod
    def _is_eoa_code(code: str) -> bool:
        return code.strip().lower() in {"0x", "0x0"}


# Backward-compatible alias, so older imports do not explode immediately.
EtherscanClient = BlockchainClient