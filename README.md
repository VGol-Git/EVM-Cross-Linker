# EVM Cross-Chain Wallet Correlation

This project profiles wallet activity across EVM-compatible networks in **exact block windows** and compares how the same EOA addresses behave across chains.

The current implementation is built around a five-stage pipeline:

1. **ingest** — fetch exact last `N` blocks, extract transactions, resolve address code, and cache raw data
2. **classify** — mark addresses as `present`, `active`, or `passive` per chain/window and build cross-chain overlap tables
3. **features** — build per-address behavioral features and pairwise cross-chain alignment tables
4. **stats** — run overlap significance tests and feature correlations
5. **plots** — render charts from either in-memory outputs or artifacts already saved on disk

The project is now **block-window based**, not day-window based.

---

## Project layout

The code lives in `src/`, including the JSON run configuration file.

A typical layout is:

```text
cross_chain_wallet_profiling/
├─ src/
│  ├─ api_client.py
│  ├─ classify.py
│  ├─ config.py
│  ├─ features.py
│  ├─ normalize.py
│  ├─ pipeline.py
│  ├─ plots.py
│  ├─ sampling.py
│  ├─ stats.py
│  └─ run_config.json
├─ requirements.txt
├─ data/
├─ outputs/
└─ logs/
```

Generated artifacts are written under:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `outputs/pipeline/plots/`
- `logs/`

---

## What each module does

### `src/config.py`
Loads configuration from:

1. environment variables
2. `CONFIG_JSON_PATH`
3. built-in defaults

Priority is:

```text
env > json > defaults
```

It also validates the config and creates all required project directories.

### `src/api_client.py`
RPC-first blockchain client with explorer fallback.

Responsibilities:

- `eth_blockNumber`
- `eth_getBlockByNumber`
- `eth_getCode`
- retry and backoff
- rate limiting
- bounded concurrency
- explorer proxy fallback when RPC is unavailable

### `src/sampling.py`
Ingestion layer for exact block windows.

Responsibilities:

- build exact last-`N` block windows
- fetch blocks and cache payloads
- extract raw transactions
- collect unique addresses from `from` and `to`
- resolve `eth_getCode` for EOA filtering
- enrich transactions with sender/receiver EOA flags
- persist manifests and cache snapshots

Important implementation details:

- observation windows are block-based
- transaction reuse is validated against a window manifest
- address-code lookup runs in chunks and saves cache incrementally
- CSV is supported and parquet writes are protected from uint256-like overflow

### `src/classify.py`
Builds one address-level status row per chain and block window.

Main outputs:

- `is_present`
- `is_active`
- `is_passive`
- counts and value aggregates
- nonce summaries
- first/last seen information
- pairwise and triple overlaps across chains
- mixed active/passive overlap tables
- presence matrices used later by stats

### `src/features.py`
Builds per-address behavioral features for each chain/window and then creates cross-chain aligned feature tables for overlapping addresses.

Includes:

- sent/received counts and values
- counterparty counts
- first/last activity timestamps
- activity block ranges
- gas and gas price aggregates
- per-day JSON series
- pairwise feature alignment tables
- overlapping-address feature tables
- per-window feature summaries

### `src/stats.py`
Runs statistical tests on overlap and behavior similarity.

Includes:

- chi-square tests on presence matrices
- Fisher exact fallback for sparse 2x2 tables
- permutation tests for Jaccard overlap
- Pearson and Spearman correlations on aligned feature pairs
- daily-series correlation summaries
- per-window statistical summary tables

### `src/plots.py`
Generates charts from saved artifacts or from in-memory outputs.

Highlights:

- pairwise overlap heatmaps
- Jaccard heatmaps
- wallet-count bar charts
- first-activity delta histograms
- value/frequency scatter plots
- p-value heatmaps
- feature sample-size heatmaps
- statistical summary table figure

`plots.py` now supports **disk-backed loading**, so plots can be regenerated without rerunning the full pipeline.

### `src/pipeline.py`
Main pipeline entry point.

It orchestrates stages, logging, manifests, and plot fallback behavior.

Important behavior:

- `plots` can use in-memory outputs if earlier stages ran in the same execution
- otherwise `plots` automatically loads saved artifacts from disk

### `src/normalize.py`
Currently not part of the active pipeline. It contains older normalization ideas and is effectively optional at this stage.

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Optional Jupyter kernel registration:

```bash
python -m ipykernel install --user --name cross-chain-wallet-profiling --display-name "Python (cross-chain-wallet-profiling)"
```

Quick dependency check:

```bash
python -c "import requests, pandas, numpy, matplotlib, scipy; print('ok')"
```

---

## Configuration

### Option 1 — JSON config

The recommended way is to keep a project config file in:

```text
src/run_config.json
```

Then point `CONFIG_JSON_PATH` to it.

PowerShell:

```powershell
$env:CONFIG_JSON_PATH=".\src\run_config.json"
```

Example config:

```json
{
  "project_root": "C:/path/to/cross_chain_wallet_profiling",
  "api": {
    "requests_per_second": 10,
    "max_concurrency": 4,
    "timeout_seconds": 30,
    "max_retries": 5,
    "backoff_base_seconds": 1.0,
    "batch_size": 50,
    "user_agent": "evm-cross-chain-correlation/2.0",
    "use_rpc_first": true,
    "use_explorer_fallback": true
  },
  "storage": {
    "table_format": "csv",
    "save_raw_block_payloads": true,
    "save_raw_transaction_rows": true,
    "save_checkpoints": true,
    "resume_from_cache": true
  },
  "sampling": {
    "observation_windows_blocks": [100],
    "reference_block_tag": "latest",
    "full_transactions": true,
    "max_addresses_for_code_lookup": null
  },
  "chains": {
    "ethereum": {
      "enabled": true,
      "rpc_url": "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY",
      "explorer_api_key": "YOUR_ETHERSCAN_KEY"
    },
    "polygon": {
      "enabled": true,
      "rpc_url": "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY",
      "explorer_api_key": "YOUR_ETHERSCAN_KEY"
    },
    "bnb": {
      "enabled": true,
      "rpc_url": "https://bsc-mainnet.core.chainstack.com/YOUR_KEY",
      "explorer_api_key": "YOUR_ETHERSCAN_KEY"
    },
    "avalanche": {
      "enabled": false,
      "rpc_url": null,
      "explorer_api_key": null
    }
  }
}
```

### Option 2 — environment variables

You can also override values from the shell.

Examples:

```powershell
$env:PROJECT_ROOT="C:\path\to\cross_chain_wallet_profiling"
$env:OBSERVATION_WINDOWS_BLOCKS="1,10,100"
$env:TABLE_FORMAT="csv"
$env:API_REQUESTS_PER_SECOND="8"
$env:API_MAX_CONCURRENCY="6"
$env:ETHEREUM_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
$env:POLYGON_RPC_URL="https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"
$env:BNB_RPC_URL="https://bsc-mainnet.core.chainstack.com/YOUR_KEY"
$env:ETHERSCAN_API_KEY="YOUR_ETHERSCAN_KEY"
```

### Important config notes

- `rpc_url` must be a real **JSON-RPC** endpoint
- do **not** put explorer endpoints such as `https://api.etherscan.io/v2/api` into `rpc_url`
- if `CONFIG_JSON_PATH` is set, the file must exist
- if both JSON and env are present, env wins

---

## Recommended first run

For the first validation run, use:

- one chain
- one small block window
- CSV storage

Example:

```powershell
$env:CONFIG_JSON_PATH=".\src\run_config.json"
$env:OBSERVATION_WINDOWS_BLOCKS="1"
$env:ENABLE_POLYGON="false"
$env:ENABLE_BNB="false"
$env:TABLE_FORMAT="csv"
python -m src.pipeline --stages ingest,classify,features,stats,plots
```

This verifies the full pipeline end to end with minimal cost.

---

## Pipeline usage

### Full run

```bash
python -m src.pipeline --stages ingest,classify,features,stats,plots
```

### Run only selected stages

```bash
python -m src.pipeline --stages ingest,classify
python -m src.pipeline --stages classify,features,stats
python -m src.pipeline --stages stats
python -m src.pipeline --stages plots
python -m src.pipeline --stages stats,plots
```

Because the current pipeline supports disk-backed plot loading, `plots` can now run separately as long as the necessary artifacts already exist on disk.

### CLI arguments

Useful options:

```bash
python -m src.pipeline \
  --stages ingest,classify,features,stats,plots \
  --project-root C:/path/to/project \
  --n-permutations 1000 \
  --random-state 42 \
  --min-nonce-for-active 1
```

Additional classification flags:

```bash
--allow-unknown-eoa
--skip-require-eoa
```

---

## What gets written to disk

### After `ingest`

Examples of artifacts:

- raw block payloads
- raw transaction tables
- address observation tables
- address code cache
- address code snapshot for the current window
- enriched transaction tables
- per-window manifest

### After `classify`

Examples of artifacts:

- address status tables per chain/window
- presence matrices for present/active/passive
- pairwise overlap tables
- triple overlap tables
- mixed active/passive overlap tables

### After `features`

Examples of artifacts:

- per-chain address feature tables
- overlapping-address feature table
- pairwise cross-chain feature alignment tables
- per-window feature summary

### After `stats`

Examples of artifacts:

- presence statistics tables
- pairwise feature correlation tables
- daily-series summary tables
- window statistics summary

### After `plots`

PNG figures under:

```text
outputs/pipeline/plots/window_<N>/
```

---

## Example workflow for real runs

### 1. Build raw data once

```bash
python -m src.pipeline --stages ingest
```

### 2. Build analysis artifacts

```bash
python -m src.pipeline --stages classify,features,stats
```

### 3. Regenerate figures only

```bash
python -m src.pipeline --stages plots
```

This is useful when:

- ingestion is expensive
- you only changed plotting logic
- you only changed font sizes, titles, labels, or figure composition

---

## Performance notes

### What usually speeds things up most

- good RPC URLs for all active chains
- using cached data instead of rerunning `ingest`
- moderate chunk sizes for `eth_getCode`
- using CSV during development if parquet overflows or slows iteration

### Practical tuning fields

The most important API tuning knobs are:

- `requests_per_second`
- `max_concurrency`
- `batch_size`
- `max_retries`
- `backoff_base_seconds`

Typical interpretation:

- `requests_per_second` controls how many requests can be started per second
- `max_concurrency` controls how many requests can be in flight simultaneously
- `batch_size` controls chunk sizes for larger mass operations like address code resolution

### Explorer-only mode

If a chain has no RPC URL and relies on explorer fallback, very aggressive concurrency is often counterproductive because explorer rate limits cause retries and slowdowns.

---

## Plot customization

`src/plots.py` exposes a `PlotConfig` dataclass.

Important options include:

- figure size
- DPI
- heatmap settings
- histogram bins
- global font size
- font family

The text in charts is controlled globally through `PlotConfig`, so axis labels, titles, legends, annotations, and table text can be scaled together.

---

## Troubleshooting

### `CONFIG_JSON_PATH does not exist`
Your `CONFIG_JSON_PATH` points to a file that is missing. Fix the path or remove the variable.

### `rpc_url looks like an explorer API endpoint`
You passed an explorer URL into `rpc_url`. Replace it with a JSON-RPC endpoint.

### `The following enabled chains have neither RPC URL nor explorer API key`
A chain is enabled but has no usable transport. Add either:

- `rpc_url`, or
- `explorer_api_key`

### Explorer rate limit / too many retries
Lower one or more of:

- `requests_per_second`
- `max_concurrency`
- `batch_size`

### Parquet integer overflow
Some EVM numeric fields can exceed int64. Use `table_format="csv"` if you want the most robust development path.

### `plots` fails when run separately
Make sure the required artifacts from `classify`, `features`, and `stats` already exist on disk.

---

## Current network support

The current config builder includes:

- Ethereum
- Polygon
- BNB
- Avalanche (optional / disabled by default)

The run config can enable or disable them individually.

---

## Dependencies

Main runtime dependencies:

- `requests`
- `pandas`
- `numpy`
- `matplotlib`
- `pyarrow`
- `scipy`

See `requirements.txt` for the full list.

---

## Notes

- The project is designed for exact block windows such as `1`, `10`, and `100` blocks.
- The pipeline is file-based, so each stage hands off artifacts to the next through saved tables.
- Re-running later stages without repeating ingestion is expected and supported.
