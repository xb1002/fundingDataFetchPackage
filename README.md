
# Funding Data Fetch Package

Utilities for downloading perpetual swap data (price OHLCV, index OHLCV, premium index, and funding rates) from supported exchanges, storing it on disk, and reading it back as pandas DataFrames.

## What's inside
- `DataFetchApi`: high-level wrapper that orchestrates downloads, caching, chunked writes, and request throttling.
- `fetch_symbol_data`: programmatic equivalent of the historic CLI workflow for fetching every requested data type for a symbol across multiple exchanges.
- `dataFetcher`: adapter layer that normalizes REST calls for Binance, OKX, Bybit, Bitget, and Gate.
- `DataApi`: local reader that stitches cached CSV slices and returns filtered pandas DataFrames.

## Requirements
- Python 3.9+
- Dependencies (installed automatically):
  - `requests` for HTTP access.
  - `pandas` for CSV buffering/processing.
  - `backports.zoneinfo` (only on Python < 3.9) for timezone handling.

## Installation
From the repository root:

```bash
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate on Unix
pip install --upgrade pip
pip install -e fundingDataFetchPackage
```

## Quick start

### Programmatic download
```python
from fundingDataFetchPackage import DataFetchApi

api = DataFetchApi(base_dir="./data", timezone="UTC")
api.fetch_price_ohlcv(
    exchange="binance",
    symbol="BTC_USDT",
    start_time="2025-11-07_00:00:00",
    end_time="2025-11-08_00:00:00",
    timeframe="1m",
)
```

### Batch fetch for all exchanges
```python
from fundingDataFetchPackage import fetch_symbol_data

fetch_symbol_data(
    symbol="BTC_USDT",
    start_time="2025-11-07_00:00:00",
    end_time="2025-11-08_00:00:00",
    exchanges="binance,okx,bybit",
    data_types="price_ohlcv,funding_rate",
    timeframe="1m",
    output_dir="./data",
)
```

### Read cached data
```python
from fundingDataFetchPackage import DataApi

local_api = DataApi(base_dir="./data")
df = local_api.price_ohlcv(
    exchange="binance",
    symbol="BTC_USDT",
    start_time="2025-11-07_00:00:00",
    end_time="2025-11-08_00:00:00",
    timeframe="1m",
)
print(df.head())
```

## Package layout
```
fundingDataFetchPackage/
├── __init__.py              # Re-exports public APIs
├── dataFetchApi.py          # Download orchestration
├── dataLocalApi.py          # Local data reader
└── dataFetcher/             # Exchange adapters and DTOs
```

## Development tips
- Run `python -m py_compile fundingDataFetchPackage/*.py fundingDataFetchPackage/dataFetcher/*.py` to check syntax.
- Use the `temp/` or `data/` directories for scratch downloads; the package handles directory creation automatically.
- When adding exchanges/adapters, update `dataFetcher/__init__.py`, `dataFetcher/dataClient.py`, and mention the new dependency in `pyproject.toml`.
