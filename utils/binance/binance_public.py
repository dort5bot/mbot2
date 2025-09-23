"""utils/binance/binance_public.py
Binance Public API endpoints.

Bu modül, Binance public endpoint'lerini kapsayan asenkron bir client sağlar.
Mevcut sınıf isimleri ve bazı fonksiyon isimleri korunmuştur (kullanıcının
isteği doğrultusunda).

Beklenen yardımcı bağımlılıklar (proje içinde bulunmalı):
- BinanceHTTPClient with method: async def _request(method, path, params=None, headers=None) -> Any
- CircuitBreaker with method: async def execute(callable, *args, **kwargs) -> Any
- BinanceAPIError exception class

# Rate limiting - HTTP
# (Not: Rate limiting uygulaması CircuitBreaker veya http client tarafında yapılmalıdır.)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

# Lokal bağımlılıklar (projede olmalı)
from .binance_request import BinanceHTTPClient  # expected: async _request(...)
from .binance_circuit_breaker import CircuitBreaker  # expected: async execute(...)
from .binance_exceptions import BinanceAPIError
from .binance_types import Interval, Symbol

logger = logging.getLogger(__name__)  # bu satırı ekle

LOG = logging.getLogger("binance_public")
LOG.setLevel(logging.INFO)


class BinancePublicAPI:
    """
    Binance Public API işlemleri - Singleton.
    """

    _instance: Optional["BinancePublicAPI"] = None
    _initialized: bool = False

    def __new__(cls, http_client: BinanceHTTPClient, circuit_breaker: CircuitBreaker) -> "BinancePublicAPI":
        """
        Singleton implementasyonu.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(http_client, circuit_breaker)
        return cls._instance

    def _initialize(self, http_client: BinanceHTTPClient, circuit_breaker: CircuitBreaker) -> None:
        """Internal initialization method."""
        if not self._initialized:
            self.http: BinanceHTTPClient = http_client
            self.circuit_breaker: CircuitBreaker = circuit_breaker
            self._initialized = True
            logger.info("BinancePublicAPI initialized with provided http_client and circuit_breaker.")

    def __init__(self, http_client: BinanceHTTPClient, circuit_breaker: CircuitBreaker) -> None:
        """
        Args:
            http_client: BinanceHTTPClient örneği
            circuit_breaker: CircuitBreaker örneği
        """
        # Initialization is handled in __new__ and _initialize
        pass
        
    # -------------------------
    # Basic endpoints (mevcut)
    # -------------------------
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time.

        Returns:
            Dict with server time info, e.g. {"serverTime": 1234567890}

        Raises:
            BinanceAPIError on failure with descriptive message.
        """
        try:
            LOG.debug("Requesting server time.")
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/time")
        except Exception as e:
            LOG.exception("Error getting server time.")
            raise BinanceAPIError(f"Error getting server time: {e}")
        pass
        
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.

        Returns:
            Exchange info payload as returned by Binance.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting exchange info.")
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/exchangeInfo")
        except Exception as e:
            LOG.exception("Error getting exchange info.")
            raise BinanceAPIError(f"Error getting exchange info: {e}")
        pass
        
    async def get_symbol_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price for a symbol.

        Args:
            symbol: symbol string e.g. "BTCUSDT"

        Returns:
            Dict with price data as Binance returns, e.g. {"symbol": "BTCUSDT", "price": "12345.67"}

        Raises:
            ValueError if symbol empty.
            BinanceAPIError on HTTP/API errors.
        """
        try:
            symbol_clean = symbol.upper().strip()
            if not symbol_clean:
                raise ValueError("Symbol cannot be empty or whitespace.")
            LOG.debug("Requesting symbol price for %s", symbol_clean)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/ticker/price", {"symbol": symbol_clean}
            )
        except ValueError:
            LOG.exception("Invalid symbol provided to get_symbol_price.")
            raise
        except Exception as e:
            LOG.exception("Error getting symbol price for %s", symbol)
            raise BinanceAPIError(f"Error getting symbol price for {symbol}: {e}")

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book (depth) for a symbol.

        Args:
            symbol: e.g. "ETHUSDT"
            limit: depth limit (valid: 5, 10, 20, 50, 100, 500, 1000, 5000). Default 100.

        Returns:
            Depth payload with bids/asks.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting order book for %s limit=%s", symbol, limit)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/depth", {"symbol": symbol.upper(), "limit": limit}
            )
        except Exception as e:
            LOG.exception("Error getting order book for %s", symbol)
            raise BinanceAPIError(f"Error getting order book for {symbol}: {e}")

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades (public).

        Args:
            symbol: trading pair symbol
            limit: max number of trades (default 500)

        Returns:
            List of trade dicts.

        Notes:
            This endpoint may be limited for historical access without API key.
        """
        try:
            LOG.debug("Requesting recent trades for %s limit=%s", symbol, limit)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/trades", {"symbol": symbol.upper(), "limit": limit}
            )
        except Exception as e:
            LOG.exception("Error getting recent trades for %s", symbol)
            raise BinanceAPIError(f"Error getting recent trades for {symbol}: {e}")

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> List[List[Union[str, float, int]]]:
        """
        Get kline/candlestick data.

        Args:
            symbol: trading pair symbol
            interval: e.g. "1m", "3m", "5m", "1h", "1d", ...
            limit: number of klines to return (default 500)

        Returns:
            List of kline lists as Binance returns.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting klines for %s interval=%s limit=%s", symbol, interval, limit)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/klines", {"symbol": symbol.upper(), "interval": interval, "limit": limit}
            )
        except Exception as e:
            LOG.exception("Error getting klines for %s", symbol)
            raise BinanceAPIError(f"Error getting klines for {symbol}: {e}")

    #async def get_all_24h_tickers(self, symbol: str) -> Dict[str, Any]:
    async def get_all_24h_tickers(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get 24 hour ticker price change statistics.

        Args:
            symbol: optional trading pair symbol.
                    If None, returns ALL symbols.

        Returns:
            - Dict for single symbol
            - List of dicts for all symbols

        Raises:
            BinanceAPIError on failure.
        """
        try:
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol.upper()

            LOG.debug("Requesting 24h ticker for symbol=%s", symbol or "ALL")
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/ticker/24hr", params
            )
        except Exception as e:
            LOG.exception("Error getting 24h ticker for %s", symbol or "ALL")
            raise BinanceAPIError(f"Error getting 24h ticker for {symbol or 'ALL'}: {e}")

    async def get_all_symbols(self) -> List[str]:
        """
        Get list of all symbols from exchangeInfo.

        Returns:
            List of symbol strings.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting all symbols via exchangeInfo.")
            data = await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/exchangeInfo")
            symbols = [s["symbol"] for s in data.get("symbols", [])]
            LOG.debug("Retrieved %d symbols.", len(symbols))
            return symbols
        except Exception as e:
            LOG.exception("Error getting all symbols.")
            raise BinanceAPIError(f"Error getting all symbols: {e}")

    # ------------------------------------
    # Additional public endpoints (eklenen)
    # ------------------------------------
    async def ping(self) -> Dict[str, Any]:
        """
        Ping endpoint to check connectivity.

        Returns:
            Empty dict on success (Binance returns 200 with empty body).

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Pinging Binance API.")
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/ping")
        except Exception as e:
            LOG.exception("Ping to Binance failed.")
            raise BinanceAPIError(f"Error pinging Binance API: {e}")

    async def get_book_ticker(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get best price/qty on the order book for a symbol or all symbols.

        Args:
            symbol: optional trading pair symbol. If None, returns all book tickers.

        Returns:
            Dict for single symbol or list of dicts for all symbols.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol.upper()
            LOG.debug("Requesting bookTicker for symbol=%s", symbol)
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/ticker/bookTicker", params)
        except Exception as e:
            LOG.exception("Error getting book ticker for %s", symbol)
            raise BinanceAPIError(f"Error getting book ticker for {symbol or 'ALL'}: {e}")

    async def get_all_book_tickers(self) -> List[Dict[str, Any]]:
        """
        Convenience wrapper to fetch all book tickers.

        Returns:
            List of book ticker dicts.

        Raises:
            BinanceAPIError on failure.
        """
        LOG.debug("Requesting all book tickers (convenience).")
        return await self.get_book_ticker(None)  # type: ignore[return-value]

    async def get_avg_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current average price for a symbol.

        Args:
            symbol: trading pair symbol

        Returns:
            Dict with avgPrice value.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting avg price for %s", symbol)
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/avgPrice", {"symbol": symbol.upper()})
        except Exception as e:
            LOG.exception("Error getting avg price for %s", symbol)
            raise BinanceAPIError(f"Error getting avg price for {symbol}: {e}")

    async def get_agg_trades(
        self, symbol: str, from_id: Optional[int] = None, start_time: Optional[int] = None,
        end_time: Optional[int] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get compressed/aggregate trades. (AggTrades)

        Args:
            symbol: trading pair symbol
            from_id: id to fetch from
            start_time: timestamp in ms
            end_time: timestamp in ms
            limit: max number of results

        Returns:
            List of agg trade dicts.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            params: Dict[str, Any] = {"symbol": symbol.upper()}
            if from_id is not None:
                params["fromId"] = from_id
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time
            if limit is not None:
                params["limit"] = limit
            LOG.debug("Requesting agg trades for %s params=%s", symbol, params)
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/aggTrades", params)
        except Exception as e:
            LOG.exception("Error getting agg trades for %s", symbol)
            raise BinanceAPIError(f"Error getting agg trades for {symbol}: {e}")

    async def get_historical_trades(self, symbol: str, limit: int = 500, from_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical trades. Note: This endpoint MAY require API key depending on Binance policy.

        Args:
            symbol: trading pair symbol
            limit: number of trades (default 500)
            from_id: trade id to fetch from (optional)

        Returns:
            List of trade dicts.

        Raises:
            BinanceAPIError on failure or if API key required but not provided by http client.
        """
        try:
            params: Dict[str, Any] = {"symbol": symbol.upper(), "limit": limit}
            if from_id is not None:
                params["fromId"] = from_id
            LOG.debug("Requesting historical trades for %s params=%s", symbol, params)
            # historicalTrades often requires API key in headers; http client should include it if available.
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/historicalTrades", params)
        except Exception as e:
            LOG.exception("Error getting historical trades for %s", symbol)
            raise BinanceAPIError(
                f"Error getting historical trades for {symbol}: {e}. Note: /api/v3/historicalTrades may require API key in headers."
            )

    async def get_ui_klines(self, symbol: str, interval: str = "1m", start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None) -> Any:
        """
        Get UI klines (alternative klines endpoint sometimes used for UI).

        Args:
            symbol: trading pair symbol
            interval: kline interval
            start_time: timestamp in ms (optional)
            end_time: timestamp in ms (optional)
            limit: limit param (optional)

        Returns:
            Response from /api/v3/uiKlines

        Raises:
            BinanceAPIError on failure.
        """
        try:
            params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": interval}
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time
            if limit is not None:
                params["limit"] = limit
            LOG.debug("Requesting ui klines for %s params=%s", symbol, params)
            return await self.circuit_breaker.execute(self.http._request, "GET", "/api/v3/uiKlines", params)
        except Exception as e:
            LOG.exception("Error getting ui klines for %s", symbol)
            raise BinanceAPIError(f"Error getting ui klines for {symbol}: {e}")

    # -------------------------
    # Futures Public Endpoints (eklenen)
    # -------------------------
    async def get_futures_exchange_info(self) -> Dict[str, Any]:
        """
        Get futures exchange information.

        Returns:
            Futures exchange info payload.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting futures exchange info.")
            return await self.circuit_breaker.execute(self.http._request, "GET", "/fapi/v1/exchangeInfo", futures=True)
        except Exception as e:
            LOG.exception("Error getting futures exchange info.")
            raise BinanceAPIError(f"Error getting futures exchange info: {e}")

    async def get_futures_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get futures order book (depth) for a symbol.

        Args:
            symbol: e.g. "ETHUSDT"
            limit: depth limit (valid: 5, 10, 20, 50, 100, 500, 1000). Default 100.

        Returns:
            Depth payload with bids/asks.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting futures order book for %s limit=%s", symbol, limit)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/depth", {"symbol": symbol.upper(), "limit": limit}, futures=True
            )
        except Exception as e:
            LOG.exception("Error getting futures order book for %s", symbol)
            raise BinanceAPIError(f"Error getting futures order book for {symbol}: {e}")

    async def get_futures_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> List[List[Union[str, float, int]]]:
        """
        Get futures kline/candlestick data.

        Args:
            symbol: trading pair symbol
            interval: e.g. "1m", "3m", "5m", "1h", "1d", ...
            limit: number of klines to return (default 500)

        Returns:
            List of kline lists as Binance returns.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting futures klines for %s interval=%s limit=%s", symbol, interval, limit)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/klines", {"symbol": symbol.upper(), "interval": interval, "limit": limit}, futures=True
            )
        except Exception as e:
            LOG.exception("Error getting futures klines for %s", symbol)
            raise BinanceAPIError(f"Error getting futures klines for {symbol}: {e}")

    async def get_futures_mark_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get mark price for a futures symbol.

        Args:
            symbol: trading pair symbol

        Returns:
            Mark price data.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting futures mark price for %s", symbol)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/premiumIndex", {"symbol": symbol.upper()}, futures=True
            )
        except Exception as e:
            LOG.exception("Error getting futures mark price for %s", symbol)
            raise BinanceAPIError(f"Error getting futures mark price for {symbol}: {e}")

    async def get_futures_funding_rate_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get funding rate history for a futures symbol.

        Args:
            symbol: trading pair symbol
            limit: number of records to return (default 100)

        Returns:
            List of funding rate history records.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting futures funding rate history for %s limit=%s", symbol, limit)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/fundingRate", {"symbol": symbol.upper(), "limit": limit}, futures=True
            )
        except Exception as e:
            LOG.exception("Error getting futures funding rate history for %s", symbol)
            raise BinanceAPIError(f"Error getting futures funding rate history for {symbol}: {e}")

    async def get_futures_24hr_ticker(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get 24 hour ticker price change statistics for futures.

        Args:
            symbol: optional trading pair symbol. If None, returns all tickers.

        Returns:
            Dict for single symbol or list of dicts for all symbols.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol.upper()
            LOG.debug("Requesting futures 24hr ticker for symbol=%s", symbol)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/ticker/24hr", params, futures=True
            )
        except Exception as e:
            LOG.exception("Error getting futures 24hr ticker for %s", symbol)
            raise BinanceAPIError(f"Error getting futures 24hr ticker for {symbol or 'ALL'}: {e}")

    async def get_futures_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        Get open interest for a futures symbol.

        Args:
            symbol: trading pair symbol

        Returns:
            Open interest data.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting futures open interest for %s", symbol)
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/openInterest", {"symbol": symbol.upper()}, futures=True
            )
        except Exception as e:
            LOG.exception("Error getting futures open interest for %s", symbol)
            raise BinanceAPIError(f"Error getting futures open interest for {symbol}: {e}")

    # -------------------------
    # Convenience / helpers
    # -------------------------
    async def symbol_exists(self, symbol: str) -> bool:
        """
        Check if a symbol exists on the exchange.

        Args:
            symbol: trading pair symbol

        Returns:
            True if exists, False otherwise.

        Raises:
            BinanceAPIError on failure to fetch exchange info.
        """
        try:
            LOG.debug("Checking if symbol exists: %s", symbol)
            info = await self.get_exchange_info()
            for s in info.get("symbols", []):
                if s.get("symbol") == symbol.upper():
                    return True
            return False
        except Exception as e:
            LOG.exception("Error checking if symbol exists: %s", symbol)
            raise BinanceAPIError(f"Error checking if symbol exists {symbol}: {e}")

    # -------------------------
    # Convenience methods for futures
    # -------------------------
    async def get_all_futures_symbols(self) -> List[str]:
        """
        Get list of all futures symbols from exchangeInfo.

        Returns:
            List of symbol strings.

        Raises:
            BinanceAPIError on failure.
        """
        try:
            LOG.debug("Requesting all futures symbols via exchangeInfo.")
            data = await self.get_futures_exchange_info()
            symbols = [s["symbol"] for s in data.get("symbols", [])]
            LOG.debug("Retrieved %d futures symbols.", len(symbols))
            return symbols
        except Exception as e:
            LOG.exception("Error getting all futures symbols.")
            raise BinanceAPIError(f"Error getting all futures symbols: {e}")

    async def futures_symbol_exists(self, symbol: str) -> bool:
        """
        Check if a symbol exists on the futures exchange.

        Args:
            symbol: trading pair symbol

        Returns:
            True if exists, False otherwise.

        Raises:
            BinanceAPIError on failure to fetch exchange info.
        """
        try:
            LOG.debug("Checking if futures symbol exists: %s", symbol)
            info = await self.get_futures_exchange_info()
            for s in info.get("symbols", []):
                if s.get("symbol") == symbol.upper():
                    return True
            return False
        except Exception as e:
            LOG.exception("Error checking if futures symbol exists: %s", symbol)
            raise BinanceAPIError(f"Error checking if futures symbol exists {symbol}: {e}")
            