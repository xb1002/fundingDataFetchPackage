from typing import List, Dict, Any, Tuple, Optional
import time

if __name__ == "__main__":
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)

from ..base_adapter import ExchangeAdapter
from ..dto import (
    OHLCVRequestParams,
    FundingRequestParams,
    CandleType,
    FundingRecordType,
)

# timeframe string and corresponding interval (milliseconds), aligned with v3 `interval`
BITGET_TIMEFRAME_MAP: Dict[str, Tuple[str, int]] = {
    "1m": ("1m", 60_000),
    "3m": ("3m", 3 * 60_000),
    "5m": ("5m", 5 * 60_000),
    "15m": ("15m", 15 * 60_000),
    "30m": ("30m", 30 * 60_000),
    "1h": ("1H", 60 * 60_000),
    "4h": ("4H", 4 * 60 * 60_000),
    "6h": ("6H", 6 * 60 * 60_000),
    "12h": ("12H", 12 * 60 * 60_000),
    "1d": ("1D", 24 * 60 * 60_000),
}

# v3 endpoints
BITGET_ENDPOINTS = {
    "price_ohlcv": "/api/v3/market/history-candles",
    "index_ohlcv": "/api/v3/market/history-candles",
    "premium_index_ohlcv": "/api/v3/market/history-candles",
    # 资费历史：改回 v2 contract 接口
    "funding_history": "/api/v2/mix/market/history-fund-rate",
}

BITGET_REQ_MAX_LIMIT = {
    "price_ohlcv": 100,
    "index_ohlcv": 100,
    "premium_index_ohlcv": 100,
    # v2 文档里 pageSize 最大 100:contentReference[oaicite:2]{index=2}
    "funding_history": 100,
}

BITGET_DEFAULT_LIMIT = {
    "price_ohlcv": 100,
    "index_ohlcv": 100,
    "premium_index_ohlcv": 100,
    # 随便给个默认，20 就行
    "funding_history": 100,
}

# map our logical key -> v3 candlestick `type`
CANDLE_TYPE_MAP: Dict[str, str] = {
    "price_ohlcv": "market",
    "index_ohlcv": "index",
    "premium_index_ohlcv": "premium",
}


class BitgetAdapter(ExchangeAdapter):
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.bitget.com"
        self.endpoint_dict = BITGET_ENDPOINTS
        self.req_max_limit = BITGET_REQ_MAX_LIMIT
        # This adapter is currently implemented for USDT perpetual futures
        self.category = "USDT-FUTURES"

    def fetch_markets(self) -> List[str]:
        """
        Use v3 Get Instruments:
        GET /api/v3/market/instruments?category=USDT-FUTURES
        """
        endpoint = "/api/v3/market/instruments"
        params = {"category": self.category}
        raw = self.make_request(
            url=f"{self.base_url}{endpoint}",
            params=params,
        )
        if not isinstance(raw, dict):
            raise ValueError("Unexpected response format from Bitget instruments endpoint")
        if raw.get("code") != "00000":
            msg = raw.get("msg", "Unknown Bitget API error")
            raise RuntimeError(f"Bitget API error {raw.get('code')}: {msg}")
        markets: List[str] = []
        for item in raw.get("data") or []:
            status = (item.get("status") or "").lower()
            # v3 status: listed / online / limit_open / offline / restrictedAPI
            if status and status not in ("online", "listed", "limit_open"):
                continue
            base = (item.get("baseCoin") or "").upper()
            quote = (item.get("quoteCoin") or "").upper()
            if not base or not quote:
                continue
            markets.append(f"{base}_{quote}")
        return sorted(set(markets))

    def _map_timeframe(self, tf: str) -> Tuple[str, int]:
        if tf not in BITGET_TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe for Bitget v3: {tf}")
        return BITGET_TIMEFRAME_MAP[tf]

    def fetch_price_ohlcv(self, req: OHLCVRequestParams) -> List[CandleType]:
        return self._fetch_candle_series("price_ohlcv", req)

    def fetch_index_ohlcv(self, req: OHLCVRequestParams) -> List[CandleType]:
        return self._fetch_candle_series("index_ohlcv", req)

    def fetch_premium_index_ohlcv(self, req: OHLCVRequestParams) -> List[CandleType]:
        # v3 /api/v3/market/history-candles with type=premium
        return self._fetch_candle_series("premium_index_ohlcv", req)

    def fetch_funding_history(self, req: FundingRequestParams) -> List[FundingRecordType]:
        endpoint = self.endpoint_dict["funding_history"]
        desired_limit = self._resolve_limit(req.limit, "funding_history")

        if req.start_time is not None:
            request_page_size = self.req_max_limit["funding_history"]
        else:
            request_page_size = max(desired_limit, BITGET_DEFAULT_LIMIT["funding_history"])
            request_page_size = min(request_page_size, self.req_max_limit["funding_history"])

        # v2: symbol + productType + pageSize
        params: Dict[str, Any] = {
            "symbol": self._map_symbol(req.symbol),
            # 对应文档里的 productType（示例是 usdt-futures）:contentReference[oaicite:4]{index=4}
            "productType": "usdt-futures",
            "pageSize": request_page_size,
        }

        if req.start_time is not None:
            records = self._fetch_funding_since(
                endpoint,
                params,
                int(req.start_time),
                desired_limit,
                request_page_size,
            )
            return records[:desired_limit]

        raw = self.make_request(
            url=f"{self.base_url}{endpoint}",
            params=params,
        )
        records = self._parse_funding_history(raw)
        return records[:desired_limit]


    def _fetch_candle_series(self, key: str, req: OHLCVRequestParams) -> List[CandleType]:
        """
        v3 Get Kline/Candlestick History:
        GET /api/v3/market/history-candles
        """
        endpoint = self.endpoint_dict[key]
        interval, interval_ms = self._map_timeframe(req.timeframe)
        limit_value = self._resolve_limit(req.limit, key)
        start_time, end_time = self._compute_time_window(req.start_time, interval_ms, limit_value)

        params: Dict[str, Any] = {
            "category": self.category,
            "symbol": self._map_symbol(req.symbol),
            "interval": interval,
            "startTime": str(start_time),
            "endTime": str(end_time),
            "limit": str(limit_value),
        }

        candle_type = CANDLE_TYPE_MAP.get(key)
        # market 是默认值，但这里显式传入更清晰
        if candle_type:
            params["type"] = candle_type

        raw = self.make_request(
            url=f"{self.base_url}{endpoint}",
            params=params,
            timeout=10.0,
        )
        return self._parse_candles(raw)

    def _fetch_funding_since(
        self,
        endpoint: str,
        base_params: Dict[str, Any],
        start_time: int,
        desired_limit: int,
        request_page_size: int,
    ) -> List[FundingRecordType]:
        page_no = 1
        collected: List[FundingRecordType] = []
        max_pages = 100

        while True:
            params = dict(base_params)
            params["pageNo"] = page_no

            raw = self.make_request(
                url=f"{self.base_url}{endpoint}",
                params=params,
            )
            page_records = self._parse_funding_history(raw)
            if not page_records:
                break

            filtered = [rec for rec in page_records if rec[0] >= start_time]
            collected.extend(filtered)
            collected.sort(key=lambda x: x[0])

            if collected and collected[0][0] <= start_time:
                break
            if len(page_records) < request_page_size:
                break
            if len(collected) >= desired_limit and collected[0][0] <= start_time:
                break

            page_no += 1
            if page_no > max_pages:
                break

        return [rec for rec in collected if rec[0] >= start_time]


    def _parse_candles(self, raw: Any) -> List[CandleType]:
        """
        v3 统一返回:
        { "code": "00000", "msg": "...", "requestTime": ..., "data": [ [ts, open, high, low, close, vol, turnover], ... ] }
        """
        if isinstance(raw, dict):
            if raw.get("code") != "00000":
                msg = raw.get("msg", "Unknown Bitget API error")
                raise RuntimeError(f"Bitget API error {raw.get('code')}: {msg}")
            data = raw.get("data") or []
        else:
            # 容错：如果上层已经把 data 提取出来，仍然兼容
            data = raw or []

        candles: List[CandleType] = []
        for item in data:
            open_time = int(item[0])
            open_price = float(item[1])
            high_price = float(item[2])
            low_price = float(item[3])
            close_price = float(item[4])
            volume = float(item[5]) if len(item) > 5 and item[5] not in ("", None) else 0.0
            candles.append(
                (
                    open_time,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                )
            )
        return candles

    def _parse_funding_history(self, raw: Any) -> List[FundingRecordType]:
        """
        v3 Funding Rate History 返回:
        {
          "code": "00000",
          "data": {
            "resultList": [
              { "symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingRateTimestamp": "..." },
              ...
            ]
          }
        }
        """
        if not isinstance(raw, dict):
            raise ValueError("Unexpected response format from Bitget funding API")
        if raw.get("code") != "00000":
            msg = raw.get("msg", "Unknown Bitget API error")
            raise RuntimeError(f"Bitget API error {raw.get('code')}: {msg}")

        data = raw.get("data") or {}
        result_list = data
        records: List[FundingRecordType] = []
        for item in result_list:
            ts_str = item.get("fundingTime")
            rate_str = item.get("fundingRate")
            if ts_str is None or rate_str is None:
                continue
            funding_time = int(ts_str)
            funding_rate = float(rate_str)
            records.append((funding_time, funding_rate))

        records.sort(key=lambda x: x[0])
        return records

    def fetch_latest_index_price(self, symbol: str) -> Tuple[int, float]:
        """
        使用 v3 Get Tickers:
        GET /api/v3/market/tickers?category=USDT-FUTURES&symbol=BTCUSDT
        从 indexPrice 字段取指数价格。
        """
        endpoint = "/api/v3/market/tickers"
        params = {
            "category": self.category,
            "symbol": self._map_symbol(symbol),
        }
        raw = self.make_request(
            url=f"{self.base_url}{endpoint}",
            params=params,
            timeout=10.0,
        )
        if not isinstance(raw, dict):
            raise ValueError("Unexpected Bitget ticker response format")
        if raw.get("code") != "00000":
            msg = raw.get("msg", "Unknown Bitget API error")
            raise RuntimeError(f"Bitget API error {raw.get('code')}: {msg}")
        data = raw.get("data") or []
        if isinstance(data, list):
            record: Optional[Dict[str, Any]] = data[0] if data else None
        else:
            record = data
        if not record or "indexPrice" not in record:
            raise ValueError(f"Bitget response missing indexPrice for {symbol}")
        ts = record.get("ts") or raw.get("requestTime")
        timestamp_ms = int(ts) if ts is not None else int(time.time() * 1000)
        return timestamp_ms, float(record["indexPrice"])

    def _compute_time_window(
        self,
        start_time: Any,
        interval_ms: int,
        limit_value: int,
    ) -> Tuple[int, int]:
        if limit_value <= 0:
            raise ValueError("Limit must be positive")
        if start_time is not None:
            start = int(start_time)
            end = start + interval_ms * limit_value
        else:
            end = int(time.time() * 1000)
            start = end - interval_ms * limit_value
            if start < 0:
                start = 0
        # v3 history-candles 最大查询区间 90 天，这里只做简单裁剪，
        # 更长区间需要上层自己拆多次请求。
        max_range_ms = 90 * 24 * 60 * 60 * 1000
        if end - start > max_range_ms:
            end = start + max_range_ms
        return start, end

    def _resolve_limit(self, limit: Any, key: str) -> int:
        default_limit = BITGET_DEFAULT_LIMIT[key]
        max_limit = self.req_max_limit[key]
        if limit is None:
            return default_limit
        if not isinstance(limit, int):
            limit = int(limit)
        if limit <= 0:
            raise ValueError("Limit must be positive")
        if limit > max_limit:
            raise ValueError(f"Limit exceeds maximum of {max_limit}")
        return limit

    def _map_symbol(self, internal_symbol: str) -> str:
        """
        内部使用 BTC_USDT，v3 使用 BTCUSDT，合约类型通过 category 区分。
        """
        return internal_symbol.replace("_", "")


if __name__ == "__main__":
    import pandas as pd
    import datetime

    start_time = int(datetime.datetime(2025, 10, 1).timestamp() * 1000)

    adapter = BitgetAdapter()
    ohlcv_req_params = OHLCVRequestParams(
        symbol="BTC_USDT",
        timeframe="1m",
        start_time=start_time,
        limit=5,
    )
    funding_req_params = FundingRequestParams(
        symbol="BTC_USDT",
        start_time=start_time,
        limit=10,
    )

    print("Markets:")
    markets = adapter.fetch_markets()
    print(markets)

    print("Price OHLCV:")
    price_ohlcv = adapter.fetch_price_ohlcv(ohlcv_req_params)
    df = pd.DataFrame(price_ohlcv, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    print(df)

    print("\nIndex OHLCV:")
    index_ohlcv = adapter.fetch_index_ohlcv(ohlcv_req_params)
    df = pd.DataFrame(index_ohlcv, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    print(df)

    print("\nPremium Index OHLCV:")
    premium_index_ohlcv = adapter.fetch_premium_index_ohlcv(ohlcv_req_params)
    df = pd.DataFrame(premium_index_ohlcv, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    print(df)

    print("\nFunding History:")
    funding_history = adapter.fetch_funding_history(funding_req_params)
    df = pd.DataFrame(funding_history, columns=["fundingTime", "fundingRate"])
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    print(df)
