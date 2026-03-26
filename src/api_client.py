# API client for interacting with blockchain APIs.
# All providers are JSON-RPC (no Etherscan).
# Block / code queries: any endpoint (with fallback).
# Address history queries: alchemy_getAssetTransfers (Alchemy only).

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests

from .config import APIConfig, ChainConfig
from .rpc_registry import RPCEndpoint, WorkingRPCMemory

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class EtherscanClientError(Exception):
    """Base exception (kept for backward compat)."""


class EtherscanHTTPError(EtherscanClientError):
    """HTTP-layer failure."""


class EtherscanAPIError(EtherscanClientError):
    """RPC-level error response."""


class EtherscanRateLimitError(EtherscanAPIError):
    """Rate-limit response from a provider."""


class AllProvidersFailedError(EtherscanClientError):
    """Every endpoint for a chain has been exhausted."""


# ------------------------------------------------------------------
# Per-provider rate limiter
# ------------------------------------------------------------------

class RateLimiter:
    """Thread-safe token-spacing limiter."""

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


_provider_limiters: Dict[str, RateLimiter] = {}
_provider_limiters_lock = threading.Lock()


def _get_limiter(endpoint: RPCEndpoint) -> RateLimiter:
    key = f"{endpoint.chain}:{endpoint.name}"
    with _provider_limiters_lock:
        if key not in _provider_limiters:
            _provider_limiters[key] = RateLimiter(endpoint.rps)
        return _provider_limiters[key]


# ------------------------------------------------------------------
# Main client
# ------------------------------------------------------------------

class MultiProviderClient:
    """
    Blockchain API client with automatic JSON-RPC provider fallback.

    Block / code queries: tries all endpoints in order, retrying per provider.
    Address history queries: uses alchemy_getAssetTransfers (Alchemy endpoints
    only); returns an empty list if no Alchemy endpoint is available or all fail.
    """

    def __init__(
        self,
        api_config: APIConfig,
        chain_endpoints: Dict[str, List[RPCEndpoint]],
        memory: WorkingRPCMemory,
    ) -> None:
        self.api_config = api_config
        self.chain_endpoints = chain_endpoints
        self.memory = memory
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": api_config.user_agent})

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "MultiProviderClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Low-level JSON-RPC call
    # ------------------------------------------------------------------

    def _jsonrpc_call(
        self,
        endpoint: RPCEndpoint,
        method: str,
        params: list,
        timeout: int,
    ) -> Dict[str, Any]:
        """Single JSON-RPC POST request. Raises on HTTP or RPC error."""
        limiter = _get_limiter(endpoint)
        limiter.wait()
        resp = self.session.post(
            endpoint.url,
            json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
            timeout=timeout,
        )
        if resp.status_code >= 400:
            raise EtherscanHTTPError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = str(err)
            if "rate" in msg.lower():
                raise EtherscanRateLimitError(f"Rate limit [{endpoint.name}]: {err}")
            raise EtherscanAPIError(f"RPC error [{endpoint.name}]: {err}")
        if "result" not in data:
            raise EtherscanAPIError(f"Missing 'result' in response [{endpoint.name}]: {data}")
        return data

    def _sleep_backoff(self, attempt: int, multiplier: float = 1.0) -> None:
        seconds = self.api_config.backoff_base_seconds * multiplier * (2 ** (attempt - 1))
        logger.warning("  retry backoff %.2f s (attempt %d)", seconds, attempt)
        time.sleep(seconds)

    # ------------------------------------------------------------------
    # Fallback loop for block / code queries
    # ------------------------------------------------------------------

    def _call_with_fallback(
        self,
        chain: ChainConfig,
        rpc_method: str,
        rpc_params: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Try every JSON-RPC endpoint for this chain in order.
        Each endpoint gets up to max_retries attempts.
        Raises AllProvidersFailedError if all fail.
        """
        endpoints = list(self.chain_endpoints.get(chain.name, []))
        if not endpoints:
            raise AllProvidersFailedError(f"No endpoints configured for {chain.name}")

        last_error: Optional[Exception] = None

        for ep in endpoints:
            for attempt in range(1, self.api_config.max_retries + 1):
                try:
                    result = self._jsonrpc_call(
                        ep, rpc_method, rpc_params or [],
                        self.api_config.timeout_seconds,
                    )
                    self.memory.set_working_provider(chain.name, ep.name)
                    return result

                except EtherscanRateLimitError as exc:
                    last_error = exc
                    logger.warning("[%s/%s] Rate limited (attempt %d/%d)",
                                   chain.name, ep.name, attempt, self.api_config.max_retries)
                    self._sleep_backoff(attempt, multiplier=2.0)

                except (EtherscanHTTPError, EtherscanAPIError, requests.RequestException) as exc:
                    last_error = exc
                    logger.warning("[%s/%s] Error (attempt %d/%d): %s",
                                   chain.name, ep.name, attempt, self.api_config.max_retries, exc)
                    self._sleep_backoff(attempt)

                except Exception as exc:
                    last_error = exc
                    logger.warning("[%s/%s] Unexpected error (attempt %d/%d): %s",
                                   chain.name, ep.name, attempt, self.api_config.max_retries, exc)
                    self._sleep_backoff(attempt)

            logger.warning("[%s] Provider '%s' failed after %d retries. Trying next provider..",
                           chain.name, ep.name, self.api_config.max_retries)

        raise AllProvidersFailedError(
            f"All providers failed for {chain.name}"
        ) from last_error

    # ------------------------------------------------------------------
    # Public block / code operations
    # ------------------------------------------------------------------

    def get_latest_block_number(self, chain: ChainConfig) -> int:
        data = self._call_with_fallback(chain, "eth_blockNumber", [])
        return int(data["result"], 16)

    def get_block_by_number(
        self,
        chain: ChainConfig,
        block_number: int,
        full_transactions: bool = True,
    ) -> Dict[str, Any]:
        data = self._call_with_fallback(
            chain, "eth_getBlockByNumber",
            [hex(block_number), full_transactions],
        )
        result = data.get("result")
        if result is None:
            raise EtherscanAPIError(f"Block {block_number} returned null on {chain.name}")
        return result

    def get_code(self, chain: ChainConfig, address: str, tag: str = "latest") -> str:
        data = self._call_with_fallback(
            chain, "eth_getCode", [address, tag],
        )
        code = data.get("result")
        if not isinstance(code, str):
            raise EtherscanAPIError(f"Unexpected getCode for {address} on {chain.name}")
        return code

    def is_eoa(self, chain: ChainConfig, address: str) -> bool:
        code = self.get_code(chain=chain, address=address)
        return code.lower() in {"0x", "0x0"}

    # ------------------------------------------------------------------
    # Alchemy-specific: asset transfer fetching
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_alchemy_transfer(transfer: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one alchemy_getAssetTransfers entry to Etherscan-like dict."""
        # Parse timestamp from metadata
        ts = 0
        ts_str = (transfer.get("metadata") or {}).get("blockTimestamp", "")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts = int(dt.timestamp())
            except Exception:
                pass

        # Parse block number
        block_hex = transfer.get("blockNum") or "0x0"
        try:
            block_num = int(block_hex, 16)
        except (ValueError, TypeError):
            block_num = 0

        # Parse value in wei (rawContract.value is hex)
        raw_contract = transfer.get("rawContract") or {}
        raw_val = raw_contract.get("value") or "0x0"
        try:
            if isinstance(raw_val, str) and raw_val.startswith("0x"):
                val_wei = int(raw_val, 16)
            else:
                val_wei = int(raw_val or 0)
        except (ValueError, TypeError):
            val_wei = 0

        contract_address = (raw_contract.get("address") or "").lower()

        return {
            "hash": transfer.get("hash", ""),
            "blockNumber": str(block_num),
            "timeStamp": str(ts),
            "from": (transfer.get("from") or "").lower(),
            "to": (transfer.get("to") or "").lower(),
            "value": str(val_wei),
            "contractAddress": contract_address,
            "input": "0x",
            "gasPrice": "0",
            "gasUsed": "0",
            "isError": "0",
            "asset": transfer.get("asset", ""),
            "category": transfer.get("category", ""),
        }

    def _alchemy_get_all_transfers(
        self,
        endpoint: RPCEndpoint,
        chain: ChainConfig,
        address: str,
        categories: List[str],
        startblock: int,
        endblock: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all asset transfers for an address using alchemy_getAssetTransfers.
        Calls twice (fromAddress + toAddress) and deduplicates by hash.
        Handles cursor-based pagination via pageKey.
        """
        address_lc = address.lower()
        from_block_hex = hex(startblock)
        to_block_hex = hex(endblock) if endblock < 9_999_999_999 else "latest"

        all_transfers: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()

        for direction_key in ["fromAddress", "toAddress"]:
            page_key: Optional[str] = None
            while True:
                params: Dict[str, Any] = {
                    "fromBlock": from_block_hex,
                    "toBlock": to_block_hex,
                    direction_key: address_lc,
                    "category": categories,
                    "withMetadata": True,
                    "excludeZeroValue": False,
                    "maxCount": "0x3e8",   # 1000 per page
                    "order": "asc",
                }
                if page_key:
                    params["pageKey"] = page_key

                result = self._jsonrpc_call(
                    endpoint,
                    "alchemy_getAssetTransfers",
                    [params],
                    self.api_config.timeout_seconds,
                )
                transfers_data = result.get("result") or {}
                transfers = transfers_data.get("transfers") or []

                for t in transfers:
                    h = t.get("hash", "")
                    if h and h not in seen_hashes:
                        seen_hashes.add(h)
                        all_transfers.append(self._normalize_alchemy_transfer(t))

                page_key = transfers_data.get("pageKey")
                if not page_key:
                    break

        return all_transfers

    def _fetch_via_alchemy(
        self,
        chain: ChainConfig,
        address: str,
        categories: List[str],
        startblock: int = 0,
        endblock: int = 9_999_999_999,
    ) -> List[Dict[str, Any]]:
        """
        Try all Alchemy (supports_indexer=True) endpoints for the chain.
        Returns empty list if none available or all fail — never raises.
        """
        eps = [
            e for e in self.chain_endpoints.get(chain.name, [])
            if e.supports_indexer
        ]
        if not eps:
            logger.warning(
                "[%s] No indexer endpoint available for address history. "
                "Returning empty.", chain.name
            )
            return []

        for ep in eps:
            try:
                transfers = self._alchemy_get_all_transfers(
                    ep, chain, address, categories, startblock, endblock
                )
                self.memory.set_working_provider(chain.name, ep.name)
                return transfers
            except Exception as exc:
                logger.warning(
                    "[%s/%s] alchemy_getAssetTransfers failed: %s",
                    chain.name, ep.name, exc
                )

        logger.warning(
            "[%s] All Alchemy endpoints failed for address history. Returning empty.",
            chain.name
        )
        return []

    # ------------------------------------------------------------------
    # Public address-history methods (Etherscan-compatible signatures)
    # These now use Alchemy under the hood; page/offset are ignored since
    # _alchemy_get_all_transfers handles full pagination internally.
    # ------------------------------------------------------------------

    def get_normal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9_999_999_999,
        page: int = 1,
        offset: Optional[int] = None,
        sort: str = "asc",
    ) -> list:
        return self._fetch_via_alchemy(
            chain, address, ["external"], startblock, endblock
        )

    def get_erc20_transfers(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9_999_999_999,
        page: int = 1,
        offset: Optional[int] = None,
        sort: str = "asc",
        contractaddress: Optional[str] = None,
    ) -> list:
        return self._fetch_via_alchemy(
            chain, address, ["erc20"], startblock, endblock
        )

    def get_internal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9_999_999_999,
        page: int = 1,
        offset: Optional[int] = None,
        sort: str = "asc",
    ) -> list:
        return self._fetch_via_alchemy(
            chain, address, ["internal"], startblock, endblock
        )

    # ------------------------------------------------------------------
    # Paginated wrappers (kept for API compat; now delegate to get_* above
    # which already return all pages via Alchemy cursor pagination)
    # ------------------------------------------------------------------

    def get_all_normal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9_999_999_999,
        sort: str = "asc",
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> list:
        return self.get_normal_transactions(
            chain=chain, address=address,
            startblock=startblock, endblock=endblock,
        )

    def get_all_erc20_transfers(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9_999_999_999,
        sort: str = "asc",
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        contractaddress: Optional[str] = None,
    ) -> list:
        return self.get_erc20_transfers(
            chain=chain, address=address,
            startblock=startblock, endblock=endblock,
            contractaddress=contractaddress,
        )

    def get_all_internal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9_999_999_999,
        sort: str = "asc",
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> list:
        return self.get_internal_transactions(
            chain=chain, address=address,
            startblock=startblock, endblock=endblock,
        )

    # ------------------------------------------------------------------
    # Legacy paginator (kept for any external callers)
    # ------------------------------------------------------------------

    def paginate_account_endpoint(
        self,
        *,
        fetch_page,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> list:
        """Legacy paged fetcher — kept for backward compat."""
        page_size = page_size or self.api_config.page_size
        all_rows: list = []
        page = 1
        while True:
            rows = fetch_page(page=page, offset=page_size)
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            page += 1
            if max_pages is not None and page > max_pages:
                break
        return all_rows
