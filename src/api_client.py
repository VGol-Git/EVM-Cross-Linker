# API client for interacting with blockchain APIs

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

from config import APIConfig, ChainConfig

logger = logging.getLogger(__name__)


class EtherscanClientError(Exception):
    """Base exception for Etherscan client errors."""


class EtherscanHTTPError(EtherscanClientError):
    """Raised when HTTP layer fails."""


class EtherscanAPIError(EtherscanClientError):
    """Raised when API returns an error payload."""


class EtherscanRateLimitError(EtherscanAPIError):
    """Raised when API indicates a rate limit issue."""


@dataclass(frozen=True)
class RequestMeta:
    chain_name: str
    chain_id: int
    module: str
    action: str


class RateLimiter:
    """
    Simple thread-safe token spacing limiter.
    Guarantees at least interval seconds between requests.
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


class EtherscanClient:
    def __init__(self, api_config: APIConfig) -> None:
        self.api_config = api_config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": api_config.user_agent})
        self.rate_limiter = RateLimiter(api_config.requests_per_second)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "EtherscanClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(
        self,
        *,
        chain: ChainConfig,
        module: str,
        action: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "chainid": str(chain.chain_id),
            "module": module,
            "action": action,
            "apikey": self.api_config.etherscan_api_key,
        }
        if extra_params:
            params.update(extra_params)

        meta = RequestMeta(
            chain_name=chain.name,
            chain_id=chain.chain_id,
            module=module,
            action=action,
        )

        last_error: Optional[Exception] = None

        for attempt in range(1, self.api_config.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.get(
                    self.api_config.base_url,
                    params=params,
                    timeout=self.api_config.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                last_error = EtherscanHTTPError(
                    f"HTTP {response.status_code} for {meta}: {response.text[:500]}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                last_error = EtherscanHTTPError(
                    f"Non-JSON response for {meta}: {response.text[:500]}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                self._raise_for_api_error(payload, meta)
                return payload
            except EtherscanRateLimitError as exc:
                last_error = exc
                self._sleep_backoff(attempt, multiplier=2.0)
            except EtherscanAPIError as exc:
                last_error = exc
                # For non-rate API errors, retry a couple of times anyway.
                self._sleep_backoff(attempt)

        raise EtherscanClientError(
            f"Request failed after {self.api_config.max_retries} attempts for {meta}"
        ) from last_error

    def _raise_for_api_error(self, payload: Dict[str, Any], meta: RequestMeta) -> None:
        # Proxy endpoints usually return {"jsonrpc": "...", "result": ...}
        if "jsonrpc" in payload:
            if "result" not in payload:
                raise EtherscanAPIError(f"Missing 'result' field for {meta}: {payload}")
            return

        status = str(payload.get("status", "")).strip()
        message = str(payload.get("message", "")).strip()
        result = payload.get("result")

        if status == "1":
            return

        # Some endpoints may return status=0 but a legitimate empty result.
        if self._is_empty_result(result):
            return

        rendered_result = ""
        if isinstance(result, str):
            rendered_result = result.lower()
        elif isinstance(result, list) and len(result) == 0:
            return
        else:
            rendered_result = str(result).lower()

        if "rate limit" in rendered_result or "max rate limit" in rendered_result:
            raise EtherscanRateLimitError(
                f"Rate limit response for {meta}: message={message}, result={result}"
            )

        raise EtherscanAPIError(
            f"API error for {meta}: status={status}, message={message}, result={result}"
        )

    @staticmethod
    def _is_empty_result(result: Any) -> bool:
        if result is None:
            return True
        if result == "":
            return True
        if isinstance(result, list) and len(result) == 0:
            return True
        if isinstance(result, str) and result.lower() in {
            "no transactions found",
            "no records found",
            "notok",
        }:
            # "notok" alone is not always empty, but some chains/explorers are noisy.
            return False
        return False

    def _sleep_backoff(self, attempt: int, multiplier: float = 1.0) -> None:
        seconds = self.api_config.backoff_base_seconds * multiplier * (2 ** (attempt - 1))
        logger.warning("Retrying after %.2f seconds", seconds)
        time.sleep(seconds)

    # ---------------------------
    # Proxy methods
    # ---------------------------

    def get_latest_block_number(self, chain: ChainConfig) -> int:
        payload = self._request(
            chain=chain,
            module="proxy",
            action="eth_blockNumber",
        )
        hex_number = payload["result"]
        return int(hex_number, 16)

    def get_block_by_number(
        self,
        chain: ChainConfig,
        block_number: int,
        full_transactions: bool = True,
    ) -> Dict[str, Any]:
        payload = self._request(
            chain=chain,
            module="proxy",
            action="eth_getBlockByNumber",
            extra_params={
                "tag": hex(block_number),
                "boolean": "true" if full_transactions else "false",
            },
        )
        result = payload.get("result")
        if result is None:
            raise EtherscanAPIError(
                f"Block {block_number} returned empty result on {chain.name}"
            )
        return result

    def get_code(self, chain: ChainConfig, address: str, tag: str = "latest") -> str:
        payload = self._request(
            chain=chain,
            module="proxy",
            action="eth_getCode",
            extra_params={
                "address": address,
                "tag": tag,
            },
        )
        code = payload.get("result")
        if not isinstance(code, str):
            raise EtherscanAPIError(
                f"Unexpected getCode response for {address} on {chain.name}: {payload}"
            )
        return code

    def is_eoa(self, chain: ChainConfig, address: str) -> bool:
        code = self.get_code(chain=chain, address=address)
        return code.lower() in {"0x", "0x0"}

    # ---------------------------
    # Account methods
    # ---------------------------

    def get_normal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9999999999,
        page: int = 1,
        offset: Optional[int] = None,
        sort: str = "asc",
    ) -> List[Dict[str, Any]]:
        payload = self._request(
            chain=chain,
            module="account",
            action="txlist",
            extra_params={
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offset": offset or self.api_config.page_size,
                "sort": sort,
            },
        )
        result = payload.get("result", [])
        if isinstance(result, list):
            return result
        return []

    def get_erc20_transfers(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9999999999,
        page: int = 1,
        offset: Optional[int] = None,
        sort: str = "asc",
        contractaddress: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "address": address,
            "startblock": startblock,
            "endblock": endblock,
            "page": page,
            "offset": offset or self.api_config.page_size,
            "sort": sort,
        }
        if contractaddress:
            params["contractaddress"] = contractaddress

        payload = self._request(
            chain=chain,
            module="account",
            action="tokentx",
            extra_params=params,
        )
        result = payload.get("result", [])
        if isinstance(result, list):
            return result
        return []

    def get_internal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9999999999,
        page: int = 1,
        offset: Optional[int] = None,
        sort: str = "asc",
    ) -> List[Dict[str, Any]]:
        payload = self._request(
            chain=chain,
            module="account",
            action="txlistinternal",
            extra_params={
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offset": offset or self.api_config.page_size,
                "sort": sort,
            },
        )
        result = payload.get("result", [])
        if isinstance(result, list):
            return result
        return []

    def paginate_account_endpoint(
        self,
        *,
        fetch_page,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        page_size = page_size or self.api_config.page_size
        all_rows: List[Dict[str, Any]] = []
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

    # ---------------------------
    # Convenience wrappers
    # ---------------------------

    def get_all_normal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9999999999,
        sort: str = "asc",
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.paginate_account_endpoint(
            fetch_page=lambda page, offset: self.get_normal_transactions(
                chain=chain,
                address=address,
                startblock=startblock,
                endblock=endblock,
                page=page,
                offset=offset,
                sort=sort,
            ),
            page_size=page_size,
            max_pages=max_pages,
        )

    def get_all_erc20_transfers(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9999999999,
        sort: str = "asc",
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        contractaddress: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.paginate_account_endpoint(
            fetch_page=lambda page, offset: self.get_erc20_transfers(
                chain=chain,
                address=address,
                startblock=startblock,
                endblock=endblock,
                page=page,
                offset=offset,
                sort=sort,
                contractaddress=contractaddress,
            ),
            page_size=page_size,
            max_pages=max_pages,
        )

    def get_all_internal_transactions(
        self,
        chain: ChainConfig,
        address: str,
        *,
        startblock: int = 0,
        endblock: int = 9999999999,
        sort: str = "asc",
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.paginate_account_endpoint(
            fetch_page=lambda page, offset: self.get_internal_transactions(
                chain=chain,
                address=address,
                startblock=startblock,
                endblock=endblock,
                page=page,
                offset=offset,
                sort=sort,
            ),
            page_size=page_size,
            max_pages=max_pages,
        )