# EVM Cross-Chain Wallet Correlation — Project Completion Strategy

---

## PART I — Structural Analysis & Project Plan

---

### 1. EVM Block Timings & Window Calculations

Every EVM-compatible network produces blocks at a fixed cadence. Since one API request typically fetches one block, the number of blocks in a time window directly determines how many API calls are needed per network.

#### Block times per network

| Network | Block Time | Notes |
|---|---|---|
| Ethereum | ~12 seconds | Fixed post-Merge PoS slot time |
| BNB Smart Chain | ~0.75 seconds | After Maxwell hard fork (was 3s); Fermi upgrade targeting 0.45s |
| Polygon PoS | ~2 seconds | Consistent across recent history |
| Avalanche C-Chain | ~2 seconds | Practical average; can be lower under low load |

#### Blocks per observation window

| Window | Ethereum (12s) | BSC (0.75s) | Polygon (2s) | Avalanche (2s) |
|---|---|---|---|---|
| 1 day | ~7,200 | ~115,200 | ~43,200 | ~43,200 |
| 10 days | ~72,000 | ~1,152,000 | ~432,000 | ~432,000 |
| 100 days | ~720,000 | ~11,520,000 | ~4,320,000 | ~4,320,000 |

**Key insight:** BSC generates far more blocks per day than any other listed chain because of its sub-second block time. This means if you naively request every block, BSC will cost the most API calls. The practical mitigation is to sample blocks at intervals or use batch RPC calls (`eth_getBlockByNumber` with ranges).

---

### 2. Recommended Networks — Speed of Data Acquisition

The selection criterion is: **fastest time from issuing a request to receiving usable data**, accounting for both API response latency and the total volume of data to process.

There is a counter-intuitive tradeoff here: faster block times (BSC) mean more blocks per day, which means more API calls and more processing time. Slower block times (Ethereum) mean fewer blocks to scan. "Fastest data acquisition" therefore depends on whether you optimize for API round-trips or for data completeness.

#### Recommended selection: Ethereum, Polygon, BNB Smart Chain

**Ethereum** is recommended because it has the fewest blocks per day (~7,200), meaning a full day's data can be collected with a minimal number of API calls. Every major provider (Alchemy, Infura, Ankr) has Ethereum as their primary, best-optimized endpoint. RPC response times are the lowest and most reliable for Ethereum across all providers.

**Polygon PoS** is recommended because it has a 2-second block time (43,200 blocks/day — manageable), excellent provider support on Alchemy and Infura, and PolygonScan offers a free API with 5 calls/sec. It also has a very high volume of EOA transactions, making it statistically rich for the study.

**BNB Smart Chain** is recommended because the professor explicitly mentioned it as having the most generous free-tier rate limits (10,000 requests per 5 minutes on BscScan). It also has the highest daily transaction volume of EVM chains (12M+ txs/day), giving the largest dataset. The tradeoff of more blocks per day is offset by its generous limits and the ability to batch requests efficiently.

**Avalanche C-Chain** can be added as an optional fourth network. It has the same block time as Polygon (~2s) but lower daily transaction volume, which means less data per block and faster response times.

---

### 3. Free-Tier API Limits by Provider

These are the providers recommended for this project (Etherscan excluded per instructions).

#### BscScan

Free with API key. Rate limit: **5 calls/second**, up to **100,000 calls/day**. Returns up to 10,000 records per call for transaction list endpoints. Best used for targeted address lookups (e.g., checking if an address has transactions, getting all txns for a specific address). Not ideal for block-by-block scanning — use BNB RPC for that.

Note: BscScan is being migrated — developers may be redirected to Etherscan API V2 (which uses the same API key infrastructure) or BSCTrace. Check current status at docs.bscscan.com.

#### PolygonScan

Free with API key. Rate limit: **5 calls/second**, up to **100,000 calls/day**, max **1,000 records per response**. Excellent for address-level queries on Polygon (e.g., `?module=account&action=txlist&address=0x...`). Use Alchemy or Infura for raw block fetching.

#### Alchemy

Free tier: **300 million Compute Units (CU) per month** with ~330 CU/second throughput. Representative CU costs: `eth_getBlockByNumber` (full txns) ≈ 50 CU, `eth_getCode` ≈ 26 CU. This means roughly **6 million full-block fetches/month** on free. Supports Ethereum, Polygon, BSC, Arbitrum, Avalanche, Base, and 30+ other chains from one API key. **Best provider for raw block data in this project.**

#### Infura

Free tier: **100,000 requests/day** (credit-based — heavier methods cost more credits). Supports Ethereum natively, plus Polygon, BSC, Avalanche, Linea, and others. Good as a secondary provider or fallback. Simpler pricing model than Alchemy.

#### Ankr

Public (no signup) endpoints available for 80+ chains. Soft rate limit: ~**1,000 requests per 10 seconds** (~100 req/sec) on public endpoints. With a free account (Freemium plan): **200 million API credits/month**. No API key required for basic public access, making it the easiest to start with for prototyping. Endpoint format: `https://rpc.ankr.com/eth`, `https://rpc.ankr.com/bsc`, `https://rpc.ankr.com/polygon`.

#### Summary table

| Provider | Free Limit | Key Required | Networks |
|---|---|---|---|
| BscScan | 5 req/sec, 100K/day | Yes | BSC only |
| PolygonScan | 5 req/sec, 100K/day | Yes | Polygon only |
| Alchemy | 300M CU/month | Yes | 30+ chains |
| Infura | 100K req/day | Yes | 10+ chains |
| Ankr | ~100 req/sec (public) | No (optional) | 80+ chains |

---

### 4. API Key Registration Guides

#### 4.1 Alchemy

1. Go to [alchemy.com](https://www.alchemy.com) and click **Sign Up** (top right). You can sign up via email or Google account.
2. Verify your email address via the confirmation link sent to your inbox.
3. On the welcome screen, select your use case — choose **"Infra & Tooling"** or **"Research"**. Fill in your organization/project name.
4. You will land on the **Alchemy Dashboard** at dashboard.alchemy.com.
5. Click **"+ Create new app"** in the top right.
6. Give your app a name (e.g., `evm-cross-linker`). Select the chain you want (start with **Ethereum Mainnet**). Click **Create App**.
7. In your app dashboard, click **"API Key"** (or the copy icon next to your endpoint URL). Copy the API key string — it looks like `abc123def456...`.
8. Repeat step 5-6 for each additional chain (Polygon, BSC) — you can create separate apps or use the same key with different endpoint URLs.
9. Use endpoint format: `https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY`

For Python: `pip install web3` then `Web3(Web3.HTTPProvider("https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"))`.

#### 4.2 Infura

1. Go to [infura.io](https://www.infura.io) and click **Sign Up**.
2. Enter your name, email, and a password. Click **Sign Up**.
3. Check your email and confirm your address via the verification link.
4. On the Welcome page, select your role and organization size (any option works for academic use).
5. In the **"Create your first API key"** dialog, enter a project name (e.g., `evm-research`). Click **Submit**.
6. Infura automatically creates your first key called **"My First Key"**. Go to **Dashboard → My First Key**.
7. Under the **"All Endpoints"** tab, all supported networks are enabled by default. Copy your **API Key** from the top of the page.
8. Use endpoint format: `https://mainnet.infura.io/v3/YOUR_API_KEY` for Ethereum. Polygon: `https://polygon-mainnet.infura.io/v3/YOUR_KEY`. BSC: `https://bsc-mainnet.infura.io/v3/YOUR_KEY`.

For Python: same `Web3(Web3.HTTPProvider(...))` as Alchemy.

#### 4.3 Ankr

**Option A — No signup (public RPC, recommended for prototyping):**
No registration needed. Use public endpoints directly:
- Ethereum: `https://rpc.ankr.com/eth`
- BSC: `https://rpc.ankr.com/bsc`
- Polygon: `https://rpc.ankr.com/polygon`

These work immediately with no API key. Rate limit ~100 req/sec.

**Option B — Free account for API key (recommended for sustained use):**
1. Go to [ankr.com](https://www.ankr.com) and click **Get Started** or **Log In → Sign Up**.
2. Register with email or via Google/GitHub OAuth.
3. Confirm your email.
4. In the Ankr dashboard, navigate to **RPC Service → Advanced API**.
5. Your API key (token) is shown on the page at [ankr.com/rpc/advanced-api](https://www.ankr.com/rpc/advanced-api). Copy it.
6. Authenticated endpoint format: `https://rpc.ankr.com/eth/YOUR_API_KEY`

#### 4.4 BscScan API

1. Go to [bscscan.com](https://bscscan.com) and click **Sign In → Register** in the top right.
2. Fill in username, email, password. Complete the CAPTCHA. Click **Create Account**.
3. Verify your email via the confirmation link.
4. After login, go to [bscscan.com/myaccount](https://bscscan.com/myaccount).
5. In the left sidebar, click **API Keys**.
6. Click **Add** to create a new API key. Enter an app name (e.g., `evm-research`). Click **Continue**.
7. Your API key is shown in the list. Copy it.
8. Use format: `https://api.bscscan.com/api?module=...&apikey=YOUR_KEY`

#### 4.5 PolygonScan API

1. Go to [polygonscan.com](https://polygonscan.com) and click **Sign In → Click to sign up**.
2. Register with username, email, password. Solve CAPTCHA.
3. Confirm your email.
4. After login, click your username (top right) → **My Profile → API Keys** in sidebar.
5. Click **Add** → enter app name → **Create New API Key**.
6. Copy the key from the list.
7. Use format: `https://api.polygonscan.com/api?module=...&apikey=YOUR_KEY`

Note: BscScan and PolygonScan share the same account infrastructure as Etherscan. If you already have an Etherscan account, you can use the same API key for all three explorers via the Etherscan API v2 unified endpoint.

---

### 5. Project Completion Plan — Phase by Phase

#### Phase 0 — Configuration & Storage Setup

Define `config.py` with: API keys per provider, target networks (ETH, BSC, Polygon), RPC endpoint URLs, block times per chain, observation windows in block counts, local cache directory paths. Set up a local SQLite database or Parquet-based file store to persist fetched block/transaction data so re-runs don't re-query the same blocks.

#### Phase 1 — Block Sampling & Transaction Extraction

For each network, pick a shared reference point in time (e.g., a specific Unix timestamp). Convert the observation windows (1 day / 10 days / 100 days) into block ranges using each network's block time. Fetch each block with full transactions (`eth_getBlockByNumber` with `hydrated=True`). From each transaction, extract: `from`, `to`, `value`, `nonce`, `hash`, `blockNumber`, `blockTimestamp`. Store raw data locally, cached by block number per chain. Use async batch fetching to maximize throughput within API rate limits.

#### Phase 2 — EOA Filtering

For every unique address encountered (both `from` and `to`), call `eth_getCode(address, "latest")`. If the result is `"0x"` (empty), the address is an **EOA** — keep it. If bytecode is returned, it is a **smart contract** — discard. Cache results: each address only needs to be checked once across all blocks. This is the heaviest API step; run it with async concurrency to stay within rate limits.

#### Phase 3 — Active / Passive Classification

From the EOA-filtered dataset, classify each address per network:

**Active wallet** = address appears as `from` in at least one transaction AND nonce ≥ 1. These addresses have spent gas and initiated transfers.

**Passive wallet** = address appears only as `to` and never as `from` in the sampled window (nonce = 0 observed). These are receive-only addresses.

Produce two sets per network: `active[network]`, `passive[network]`.

#### Phase 4 — Cross-Chain Address Matching

Compute set intersections for all pairwise and triple combinations:

- `active[ETH] ∩ active[BSC]`
- `active[ETH] ∩ active[POLYGON]`
- `active[BSC] ∩ active[POLYGON]`
- `active[ETH] ∩ active[BSC] ∩ active[POLYGON]`

Same for passive sets. Also compute mixed: active on Chain A, passive on Chain B.

Compute **Jaccard similarity** for each pair: `|A ∩ B| / |A ∪ B|`. Compare observed overlap to expected random overlap: if there are N₁ unique EOAs on Chain 1 and N₂ on Chain 2 out of a shared address space of ~2¹⁶⁰ possible Ethereum addresses, random co-occurrence would be near zero — any non-trivial overlap is statistically significant by default.

#### Phase 5 — Feature Extraction for Overlapping Addresses

For every address present on 2+ networks, compute per-address per-network features: total txns sent, total txns received, total value sent (in native token), total value received, first activity timestamp, last activity timestamp, number of unique counterparties, average transaction value, transaction frequency (txns/day).

#### Phase 6 — Correlation Analysis

**6a. Temporal correlation:** For addresses active on multiple chains, compute the time delta (Δ) between their first transaction on Chain A and Chain B. Plot the Δ distribution. If it clusters tightly (e.g., within hours), it suggests coordinated cross-chain behavior.

**6b. Volume correlation:** For overlapping active addresses, compute Pearson and Spearman correlation between value_sent on Chain A vs Chain B. A high positive correlation suggests users move proportional amounts across chains.

**6c. Activity frequency correlation:** Compare daily transaction counts for the same address across chains. Compute Pearson r on these time series per address, then average across all overlapping addresses.

**6d. Active/Passive ratio analysis:** What fraction of active wallets on Chain A also appear on Chain B (as active or passive)? Is this fraction higher than random chance? Compute a null model baseline and compare.

#### Phase 7 — Statistical Testing

**Chi-squared test:** Build 2×2 contingency tables (present/absent on Chain A × present/absent on Chain B). Compute χ² and p-value. A p-value < 0.05 confirms non-random co-occurrence.

**Permutation test:** Randomly shuffle address sets 1,000 times. Recompute Jaccard similarity each time to build a null distribution. Compare the observed Jaccard score to the 95th percentile of the null distribution.

**Pearson/Spearman:** Applied to the per-address feature vectors across chains for all overlapping addresses.

All tests run at each window size (1 day, 10 days, 100 days). Report how significance and effect size change with window size.

#### Phase 8 — Visualization

Produce: Venn diagrams (or UpSet plots) of address set overlaps. Heatmaps of pairwise Jaccard similarities and Pearson r values. Histogram of time-delta Δ between first cross-chain activity. Scatter plots of value_sent on Chain A vs Chain B. Bar charts of active/passive wallet counts per chain and overlap counts. Summary table of all statistical test results with p-values.

#### Phase 9 — Conclusion

If overlap is significantly above random, and behavioral features correlate across chains → hypothesis confirmed: EVM users reuse the same EOA across chains with consistent behavioral patterns, suggesting deliberate cross-chain activity. Interpret likely patterns: arbitrage, liquidity migration, obfuscation, or multi-chain yield strategies.

If overlap is negligible → hypothesis not confirmed: users do not significantly reuse addresses across EVM chains in the sampled window, suggesting chain-specific wallet management is the norm.

Either result is valid and should be reported with all supporting statistics and charts.

---

### 6. Development-Based Work Split Strategies

The project is split by **what code you write**, not by which network you handle. Three strategies are proposed.

#### Strategy A — Pipeline Layer Split (recommended)

Each person owns one horizontal layer of the entire pipeline, applied to all networks.

**Person 1 — Data Layer:** Writes `api_client.py` and `sampling.py`. Responsible for all RPC calls, rate limiting logic, async batch fetching, retry logic, local caching (SQLite/Parquet), and the EOA contract-code filtering (`eth_getCode`). Delivers clean, stored datasets of raw EOA transactions per network. This person sets up API keys and the config system.

**Person 2 — Classification & Matching Layer:** Takes the stored datasets from Person 1. Writes `classify.py` and the cross-chain matching logic. Implements the active/passive classification (nonce-based), builds the address sets per network, computes all set intersections, Jaccard similarities, and the feature extraction for overlapping addresses.

**Person 3 — Analysis & Visualization Layer:** Takes the classified and matched data from Person 2. Writes the correlation analysis (temporal, volume, frequency), all statistical tests (chi-squared, permutation), and all plots (`plots.py`). Writes the final summary tables and report-ready outputs.

These three layers have clean handoff points (local files / database tables), so people can work independently and in parallel after the first layer's output is available.

#### Strategy B — Functional Module Split

Each person owns a specific functional concern across the full stack.

**Person 1 — Infrastructure & API:** Config, API clients for all three providers (Alchemy, Ankr, BscScan/PolygonScan), block fetching, transaction extraction, storage layer. Also handles EOA vs contract filtering.

**Person 2 — Address Intelligence:** Active/passive classification, nonce analysis, address deduplication, cross-chain set operations (intersections, Jaccard), feature vector computation per address.

**Person 3 — Statistical Engine & Output:** Pearson/Spearman correlation, chi-squared, permutation tests, all matplotlib/seaborn visualizations, window-size comparison (1d/10d/100d), final report tables.

#### Strategy C — Window-Based Split (least recommended, but possible)

Each person handles the entire pipeline but for one observation window.

**Person 1 — 1-Day Window:** Runs the full pipeline (fetch → filter → classify → match → analyze) on the 1-day observation window for all three networks.

**Person 2 — 10-Day Window:** Same full pipeline for the 10-day window.

**Person 3 — 100-Day Window:** Same full pipeline for the 100-day window.

At the end, all three merge results to compare how correlations evolve across window sizes.

The downside of Strategy C is code duplication — all three people effectively write the same code. It works if the team wants maximum independence but is less efficient.

**Recommendation: Strategy A.** Clean interfaces between layers, true parallelism, no code duplication, and each person develops deep expertise in their domain.

---

## PART II — Python Code Acceleration Recommendations

The project involves fetching and processing potentially millions of transactions across three networks and three time windows. Naive sequential Python will be too slow. The following recommendations are ordered from easiest to implement to most advanced.

---

### 1. Async I/O for API Calls — `asyncio` + `aiohttp`

**Why:** API calls are I/O-bound. Sequential `requests` calls wait for a server response before issuing the next one. With `asyncio`, you can fire hundreds of requests concurrently in a single thread, limited only by the API rate limit — not by network latency.

**What to use:** `aiohttp` for async HTTP, `asyncio.Semaphore` to respect API rate limits.

```python
import asyncio, aiohttp

async def fetch_block(session, url, block_number, semaphore):
    async with semaphore:
        payload = {
            "jsonrpc": "2.0", "method": "eth_getBlockByNumber",
            "params": [hex(block_number), True], "id": block_number
        }
        async with session.post(url, json=payload) as resp:
            return await resp.json()

async def fetch_blocks(rpc_url, block_numbers, max_concurrent=50):
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_block(session, rpc_url, bn, semaphore) for bn in block_numbers]
        return await asyncio.gather(*tasks)
```

**Expected speedup:** 20–100x over sequential `requests`, depending on network latency and concurrency level. This is the single highest-impact optimization in this project.

---

### 2. JSON-RPC Batch Requests

**Why:** Most EVM RPC providers support sending multiple JSON-RPC calls in a single HTTP request (batch RPC). Instead of 100 requests, you send 1 request with a list of 100 calls. This dramatically reduces overhead per call.

```python
def build_batch(block_numbers):
    return [
        {"jsonrpc": "2.0", "method": "eth_getBlockByNumber",
         "params": [hex(bn), True], "id": bn}
        for bn in block_numbers
    ]

import requests
response = requests.post(rpc_url, json=build_batch(range(1000, 1100)))
blocks = response.json()  # list of 100 results
```

**Expected speedup:** 5–20x reduction in API call overhead. Combine with `aiohttp` for maximum throughput — send large async batches.

---

### 3. Polars Instead of Pandas for Data Processing

**Why:** Once data is collected, you'll have DataFrames with millions of rows (transactions) and need to do groupby operations, set operations, and joins. Pandas is single-threaded and Python-based. Polars is built on Rust, uses all CPU cores automatically, and its lazy evaluation engine can optimize query plans.

**Measured speedup:** Polars is typically 3–10x faster than Pandas on ETL workloads with large DataFrames. For 100-day windows with millions of transactions, this matters.

```python
import polars as pl

df = pl.read_parquet("transactions_eth.parquet")

active = (
    df.filter(pl.col("nonce") >= 1)
      .filter(pl.col("is_eoa") == True)
      .select("from_address")
      .unique()
)
```

**Recommendation:** Use Polars for all DataFrame operations. It has a Pandas-compatible API (`pl.from_pandas()`) so migrating existing Pandas code is straightforward.

---

### 4. DuckDB for Set Operations and Cross-Chain Joins

**Why:** The cross-chain address matching (Phase 4) is essentially a SQL JOIN / INTERSECT operation on large sets. DuckDB is an embedded analytical database (runs in-process, no server needed) that is extremely fast for this type of query and can read Parquet files directly.

```python
import duckdb

result = duckdb.query("""
    SELECT a.from_address
    FROM eth_active a
    INNER JOIN bsc_active b ON a.from_address = b.from_address
    INNER JOIN polygon_active p ON a.from_address = p.from_address
""").df()
```

DuckDB can process hundreds of millions of rows per second on a single laptop. For the set intersection + feature join operations in this project, it is the right tool.

---

### 5. Multiprocessing for CPU-Bound Computation

**Why:** Statistical computation (permutation tests with 1,000 iterations, computing correlation across millions of address pairs) is CPU-bound. Python's GIL prevents true multithreading, but `multiprocessing` spawns separate processes that each get a full CPU core.

```python
from multiprocessing import Pool

def compute_jaccard(pair):
    set_a, set_b = pair
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0

with Pool(processes=4) as pool:
    results = pool.map(compute_jaccard, address_set_pairs)
```

**`aiomultiprocess`** combines async I/O with multiprocessing — useful if you want to run async API calls across multiple processes simultaneously. Good for the EOA filtering step where you fetch `eth_getCode` for millions of addresses.

---

### 6. Numba for Custom Numeric Loops

**Why:** If you write custom numerical computations — e.g., computing time-delta distributions, rolling correlation over a time series, or the permutation test inner loop — plain Python loops are very slow. Numba JIT-compiles these loops to native machine code at the first call.

```python
from numba import njit
import numpy as np

@njit
def compute_deltas(timestamps_a, timestamps_b):
    result = np.empty(len(timestamps_a))
    for i in range(len(timestamps_a)):
        result[i] = timestamps_b[i] - timestamps_a[i]
    return result
```

**Best use cases here:** Permutation test inner loop (shuffling arrays and computing overlap counts thousands of times), time-delta computation across millions of address pairs, rolling frequency statistics.

**Not useful for:** Pandas/Polars operations (those are already optimized), API calls (I/O-bound, use asyncio instead).

---

### 7. cuDF / RAPIDS — GPU Acceleration (Optional, Advanced)

**Why:** If you have access to an NVIDIA GPU, NVIDIA's RAPIDS library (`cuDF`) provides a Pandas-compatible DataFrame API that runs entirely on the GPU. For the feature extraction and correlation computation steps, GPU acceleration can provide 10–50x speedup over CPU-based Pandas.

```python
import cudf  # requires NVIDIA GPU + CUDA

df = cudf.read_parquet("transactions_eth.parquet")
grouped = df.groupby("from_address")["value"].sum()
```

**Realistic assessment:** For a student project, cuDF is only practical if you have an NVIDIA GPU available (e.g., via Google Colab Pro with A100, or a local workstation). If not, Polars + DuckDB + Numba are sufficient and easier to set up.

---

### 8. Local Caching & Parquet Storage

**Why:** Re-fetching the same blocks every time you run a test is wasteful and will exhaust free-tier API limits fast. Persist all fetched data to Parquet files (compressed columnar storage) after the first fetch. All subsequent runs read from disk — which is orders of magnitude faster than API calls.

```python
import polars as pl

# Save after fetching
df.write_parquet("blocks_eth_1day.parquet", compression="zstd")

# Load on subsequent runs
df = pl.read_parquet("blocks_eth_1day.parquet")
```

Parquet with `zstd` compression is typically 5–10x smaller than CSV and 10–50x faster to read.

---

### Recommended Stack Summary

| Concern | Tool | Why |
|---|---|---|
| API fetching (I/O) | `asyncio` + `aiohttp` | 20–100x faster than sequential |
| Batch RPC | JSON-RPC batch protocol | Fewer HTTP round trips |
| DataFrame ops | Polars | 3–10x faster than Pandas |
| Cross-chain joins / set ops | DuckDB | Sub-second on millions of rows |
| CPU-bound loops (stats) | Numba `@njit` | Near-C speed for numeric loops |
| CPU parallelism | `multiprocessing` | Bypass GIL for true parallel CPU |
| Storage | Parquet (zstd) | Fast I/O, small files, no re-fetching |
| GPU (optional) | cuDF / RAPIDS | 10–50x if NVIDIA GPU available |

---

*End of document.*
