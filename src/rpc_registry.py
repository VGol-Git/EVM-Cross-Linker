# RPC Provider Registry — multi-provider fallback with working-RPC memory
#
# Defines all available RPC endpoints per chain, tries them in priority
# order, and remembers which ones succeeded so re-runs skip broken providers.
# NOTE: Etherscan is NOT used. All providers are direct JSON-RPC endpoints.
# Alchemy (supports_indexer=True) is used for address history queries via
# alchemy_getAssetTransfers. Other providers handle block/code lookups only.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# =====================================================================
# Data model
# =====================================================================

@dataclass(frozen=True)
class RPCEndpoint:
    """One RPC provider endpoint for one chain."""
    name: str             # human-readable: "alchemy", "ankr", "nodereal" ...
    chain: str            # "ethereum", "bnb", "polygon"
    chain_id: int
    url: str              # full URL (with API key baked in)
    provider_type: str    # always "jsonrpc"
    rps: float = 3.0      # max requests per second for this endpoint
    supports_indexer: bool = False  # True if endpoint supports alchemy_getAssetTransfers


# =====================================================================
# All known endpoints (JSON-RPC only, no Etherscan)
# =====================================================================

_ALCHEMY_KEY = "jXwnU3n6MuiGbM_GmBQbN"
_NODEREAL_KEY = "b521ff62724e4d7c8283ab3aea9e1387"
_CHAINSTACK_BSC = "https://bsc-mainnet.core.chainstack.com/0b64b28c26b574de0df4d667a11033ac"


def build_all_endpoints() -> Dict[str, List[RPCEndpoint]]:
    """Return {chain_name: [RPCEndpoint, ...]} for every known provider."""
    registry: Dict[str, List[RPCEndpoint]] = {}

    # ---------- Ethereum ----------
    registry["ethereum"] = [
        RPCEndpoint(
            name="alchemy", chain="ethereum", chain_id=1,
            url=f"https://eth-mainnet.g.alchemy.com/v2/{_ALCHEMY_KEY}",
            provider_type="jsonrpc", rps=25.0,
            supports_indexer=True,
        ),
        RPCEndpoint(
            name="nodereal", chain="ethereum", chain_id=1,
            url=f"https://eth-mainnet.nodereal.io/v1/{_NODEREAL_KEY}",
            provider_type="jsonrpc", rps=10.0,
        ),
        RPCEndpoint(
            name="ankr", chain="ethereum", chain_id=1,
            url="https://rpc.ankr.com/eth",
            provider_type="jsonrpc", rps=30.0,
        ),
    ]

    # ---------- Polygon ----------
    registry["polygon"] = [
        RPCEndpoint(
            name="alchemy", chain="polygon", chain_id=137,
            url=f"https://polygon-mainnet.g.alchemy.com/v2/{_ALCHEMY_KEY}",
            provider_type="jsonrpc", rps=25.0,
            supports_indexer=True,
        ),
        RPCEndpoint(
            name="ankr", chain="polygon", chain_id=137,
            url="https://rpc.ankr.com/polygon",
            provider_type="jsonrpc", rps=30.0,
        ),
    ]

    # ---------- BNB Smart Chain ----------
    registry["bnb"] = [
        RPCEndpoint(
            name="alchemy", chain="bnb", chain_id=56,
            url=f"https://bnb-mainnet.g.alchemy.com/v2/{_ALCHEMY_KEY}",
            provider_type="jsonrpc", rps=25.0,
            supports_indexer=True,
        ),
        RPCEndpoint(
            name="chainstack", chain="bnb", chain_id=56,
            url=_CHAINSTACK_BSC,
            provider_type="jsonrpc", rps=15.0,
        ),
        RPCEndpoint(
            name="nodereal", chain="bnb", chain_id=56,
            url=f"https://bsc-mainnet.nodereal.io/v1/{_NODEREAL_KEY}",
            provider_type="jsonrpc", rps=10.0,
        ),
        RPCEndpoint(
            name="ankr", chain="bnb", chain_id=56,
            url="https://rpc.ankr.com/bsc",
            provider_type="jsonrpc", rps=30.0,
        ),
    ]

    return registry


# =====================================================================
# Working-RPC memory (persisted to JSON)
# =====================================================================

_WORKING_RPC_FILENAME = "working_rpcs.json"


class WorkingRPCMemory:
    """
    Persists which (chain, provider_name) last succeeded so the next
    run starts with the known-good provider.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / _WORKING_RPC_FILENAME
        self._data: Dict[str, str] = {}   # chain -> provider_name
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with self._path.open("r") as f:
                    self._data = json.load(f)
                logger.info("Loaded working-RPC memory from %s", self._path)
            except Exception:
                self._data = {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            json.dump(self._data, f, indent=2)

    def get_working_provider(self, chain: str) -> Optional[str]:
        return self._data.get(chain)

    def set_working_provider(self, chain: str, provider_name: str) -> None:
        if self._data.get(chain) == provider_name:
            return
        prev = self._data.get(chain)
        self._data[chain] = provider_name
        self.save()
        if prev is None:
            logger.info("Working RPC for %s: %s", chain, provider_name)
        else:
            logger.info("Working RPC for %s changed: %s -> %s", chain, prev, provider_name)

    def get_all(self) -> Dict[str, str]:
        return dict(self._data)


def order_endpoints_for_chain(
    chain: str,
    all_endpoints: Dict[str, List[RPCEndpoint]],
    memory: WorkingRPCMemory,
) -> List[RPCEndpoint]:
    """
    Return endpoints for *chain* ordered so the last-known-good provider
    comes first, then the rest in default order.
    """
    eps = list(all_endpoints.get(chain, []))
    if not eps:
        return eps

    last_good = memory.get_working_provider(chain)
    if last_good is None:
        return eps

    preferred = [e for e in eps if e.name == last_good]
    rest = [e for e in eps if e.name != last_good]
    return preferred + rest


# =====================================================================
# Connectivity test
# =====================================================================

def test_endpoint(ep: RPCEndpoint, timeout: int = 15) -> bool:
    """Quick smoke test — try eth_blockNumber via JSON-RPC POST."""
    try:
        resp = requests.post(
            ep.url,
            json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
            timeout=timeout,
        )
        data = resp.json()
        return "result" in data and isinstance(data["result"], str)
    except Exception as exc:
        logger.debug("Endpoint %s/%s failed smoke test: %s", ep.chain, ep.name, exc)
    return False
