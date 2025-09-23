"""
utils/binance/binance_private.py
--------------------------------
Binance Private API endpoints (API key gerektiren).
Spot Account & Orders
Futures Account
Margin Trading
Staking
ListenKey (User Data Stream)
Savings
Mining
Sub-accounts

🔐 Özellikler:
- üm metodları circuit_breaker.execute ile sarmalanmış
- Sadece Private Endpoint'ler (Spot + Futures + Margin + Staking + Savings + Mining + Sub-accounts)
- Async / await uyumlu + Singleton yapı + Logging desteği + PEP8 uyumlu + Type hints + docstring
"""

import logging
from typing import Any, Dict, List, Optional

from .binance_request import BinanceHTTPClient
from .binance_circuit_breaker import CircuitBreaker
from .binance_exceptions import BinanceAPIError, BinanceAuthenticationError

logger = logging.getLogger(__name__)


class BinancePrivateAPI:
    """Binance Private API işlemleri."""

    _instance: Optional["BinancePrivateAPI"] = None

    def __new__(cls, http_client: BinanceHTTPClient, circuit_breaker: CircuitBreaker) -> "BinancePrivateAPI":
        """Singleton instance döndürür."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.http = http_client
            cls._instance.circuit_breaker = circuit_breaker
            logger.info("✅ BinancePrivateAPI singleton instance created")
        return cls._instance

    async def _require_keys(self) -> None:
        """API key kontrolü yap."""
        if not self.http.api_key or not self.http.secret_key:
            logger.error("❌ Binance API key/secret bulunamadı")
            raise BinanceAuthenticationError("API key and secret required for this endpoint")

    # ------------------------
    # Spot Account & Orders
    # ------------------------
    async def get_account_info(self) -> Dict[str, Any]:
        """Spot hesap bilgilerini getir."""
        try:
            await self._require_keys()
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/account", signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting account info")
            raise BinanceAPIError(f"Error getting account info: {e}")

    async def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """Hesap bakiyesi getir (varlık belirtilirse sadece o varlığı döndürür)."""
        try:
            await self._require_keys()
            account_info = await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/account", signed=True
            )
            if asset:
                asset = asset.upper()
                for balance in account_info.get("balances", []):
                    if balance.get("asset") == asset:
                        return balance
                return {}
            return account_info
        except Exception as e:
            logger.exception("🚨 Error getting account balance")
            raise BinanceAPIError(f"Error getting account balance: {e}")

    async def place_order(self, symbol: str, side: str, type_: str, quantity: float, price: Optional[float] = None, 
                         time_in_force: Optional[str] = None, stop_price: Optional[float] = None) -> Dict[str, Any]:
        """Yeni spot order oluştur."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"symbol": symbol.upper(), "side": side, "type": type_, "quantity": quantity}
            if price:
                params["price"] = price
            if time_in_force:
                params["timeInForce"] = time_in_force
            if stop_price:
                params["stopPrice"] = stop_price
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/api/v3/order", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error placing spot order for {symbol}")
            raise BinanceAPIError(f"Error placing spot order for {symbol}: {e}")

    async def cancel_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Mevcut spot order iptal et."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"symbol": symbol.upper()}
            if order_id:
                params["orderId"] = order_id
            if orig_client_order_id:
                params["origClientOrderId"] = orig_client_order_id
            return await self.circuit_breaker.execute(
                self.http._request, "DELETE", "/api/v3/order", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error canceling spot order for {symbol}")
            raise BinanceAPIError(f"Error canceling spot order for {symbol}: {e}")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Açık spot order'ları getir."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper()} if symbol else {}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/openOrders", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting open orders")
            raise BinanceAPIError(f"Error getting open orders: {e}")

    async def get_order_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Spot order geçmişini getir."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper(), "limit": limit}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/allOrders", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error getting spot order history for {symbol}")
            raise BinanceAPIError(f"Error getting spot order history for {symbol}: {e}")

    async def get_my_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trade history for a specific symbol.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to return (default 50, max 1000)

        Returns:
            List of trade dictionaries

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper(), "limit": limit}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/api/v3/myTrades", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error getting trade history for {symbol}")
            raise BinanceAPIError(f"Error getting trade history for {symbol}: {e}")

    # ------------------------
    # Futures Account
    # ------------------------
    async def get_futures_account_info(self) -> Dict[str, Any]:
        """Futures hesap bilgilerini getir."""
        try:
            await self._require_keys()
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v2/account", signed=True, futures=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting futures account info")
            raise BinanceAPIError(f"Error getting futures account info: {e}")

    async def get_futures_balance(self) -> List[Dict[str, Any]]:
        """Futures hesap bakiyelerini getir."""
        try:
            await self._require_keys()
            account_info = await self.get_futures_account_info()
            return account_info.get("assets", [])
        except Exception as e:
            logger.exception("🚨 Error getting futures balance")
            raise BinanceAPIError(f"Error getting futures balance: {e}")

    async def get_futures_positions(self) -> List[Dict[str, Any]]:
        """Futures açık pozisyonları getir."""
        try:
            await self._require_keys()
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v2/positionRisk", signed=True, futures=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting futures positions")
            raise BinanceAPIError(f"Error getting futures positions: {e}")

    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: Optional[str] = None,
        stop_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Yeni futures order oluştur."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {
                "symbol": symbol.upper(),
                "side": side,
                "type": type_,
                "quantity": quantity,
                "reduceOnly": reduce_only,
            }
            if price:
                params["price"] = price
            if time_in_force:
                params["timeInForce"] = time_in_force
            if stop_price:
                params["stopPrice"] = stop_price
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/fapi/v1/order", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error placing futures order for {symbol}")
            raise BinanceAPIError(f"Error placing futures order for {symbol}: {e}")

    async def cancel_futures_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Futures order iptal et."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"symbol": symbol.upper()}
            if order_id:
                params["orderId"] = order_id
            if orig_client_order_id:
                params["origClientOrderId"] = orig_client_order_id
            return await self.circuit_breaker.execute(
                self.http._request, "DELETE", "/fapi/v1/order", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error canceling futures order for {symbol}")
            raise BinanceAPIError(f"Error canceling futures order for {symbol}: {e}")

    async def get_futures_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Açık futures order'ları getir."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper()} if symbol else {}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/openOrders", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting futures open orders")
            raise BinanceAPIError(f"Error getting futures open orders: {e}")

    async def get_futures_order_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Futures order geçmişini getir."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper(), "limit": limit}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/allOrders", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error getting futures order history for {symbol}")
            raise BinanceAPIError(f"Error getting futures order history for {symbol}: {e}")

    async def get_futures_income_history(self, symbol: Optional[str] = None, income_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Futures gelir geçmişini getir."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"limit": limit}
            if symbol:
                params["symbol"] = symbol.upper()
            if income_type:
                params["incomeType"] = income_type
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/fapi/v1/income", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting futures income history")
            raise BinanceAPIError(f"Error getting futures income history: {e}")

    async def change_futures_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Futures kaldıraç oranını değiştir."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper(), "leverage": leverage}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/fapi/v1/leverage", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error changing futures leverage for {symbol}")
            raise BinanceAPIError(f"Error changing futures leverage for {symbol}: {e}")

    async def change_futures_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """Futures margin tipini değiştir (ISOLATED veya CROSSED)."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper(), "marginType": margin_type.upper()}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/fapi/v1/marginType", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error changing futures margin type for {symbol}")
            raise BinanceAPIError(f"Error changing futures margin type for {symbol}: {e}")

    async def set_futures_position_mode(self, dual_side_position: bool) -> Dict[str, Any]:
        """Futures pozisyon modunu ayarla (Hedge Mode veya One-way Mode)."""
        try:
            await self._require_keys()
            params = {"dualSidePosition": "true" if dual_side_position else "false"}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/fapi/v1/positionSide/dual", params=params, signed=True, futures=True
            )
        except Exception as e:
            logger.exception("🚨 Error setting futures position mode")
            raise BinanceAPIError(f"Error setting futures position mode: {e}")

    # ------------------------
    # Margin Trading
    # ------------------------
    async def get_margin_account_info(self) -> Dict[str, Any]:
        """Margin hesap bilgilerini getir."""
        try:
            await self._require_keys()
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/margin/account", signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting margin account info")
            raise BinanceAPIError(f"Error getting margin account info: {e}")

    async def create_margin_order(self, symbol: str, side: str, type_: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Margin order oluştur."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"symbol": symbol.upper(), "side": side, "type": type_, "quantity": quantity}
            if price:
                params["price"] = price
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/margin/order", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error creating margin order for {symbol}")
            raise BinanceAPIError(f"Error creating margin order for {symbol}: {e}")

    async def get_margin_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Margin açık order'ları getir."""
        try:
            await self._require_keys()
            params = {"symbol": symbol.upper()} if symbol else {}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/margin/openOrders", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting margin open orders")
            raise BinanceAPIError(f"Error getting margin open orders: {e}")

    # ------------------------
    # Staking
    # ------------------------
    async def get_staking_product_list(self, product: str = "STAKING", asset: Optional[str] = None) -> List[Dict[str, Any]]:
        """Staking ürün listesini getir."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"product": product}
            if asset:
                params["asset"] = asset.upper()
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/staking/productList", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting staking product list")
            raise BinanceAPIError(f"Error getting staking product list: {e}")

    async def stake_asset(self, product: str, product_id: str, amount: float) -> Dict[str, Any]:
        """Varlık stake et."""
        try:
            await self._require_keys()
            params = {"product": product, "productId": product_id, "amount": amount}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/staking/purchase", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error staking asset")
            raise BinanceAPIError(f"Error staking asset: {e}")

    async def unstake_asset(self, product: str, product_id: str, position_id: Optional[str] = None, amount: Optional[float] = None) -> Dict[str, Any]:
        """Varlık unstake et."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"product": product, "productId": product_id}
            if position_id:
                params["positionId"] = position_id
            if amount:
                params["amount"] = amount
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/staking/redeem", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error unstaking asset")
            raise BinanceAPIError(f"Error unstaking asset: {e}")

    async def get_staking_history(self, product: str, txn_type: str, asset: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Staking geçmişini getir."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"product": product, "txnType": txn_type}
            if asset:
                params["asset"] = asset.upper()
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/staking/stakingRecord", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting staking history")
            raise BinanceAPIError(f"Error getting staking history: {e}")

    # ------------------------
    # User Data Stream
    # ------------------------
    async def create_listen_key(self, futures: bool = False) -> Dict[str, Any]:
        """ListenKey oluştur (User Data Stream için)."""
        try:
            await self._require_keys()
            endpoint = "/fapi/v1/listenKey" if futures else "/api/v3/userDataStream"
            return await self.circuit_breaker.execute(
                self.http._request, "POST", endpoint, signed=True, futures=futures
            )
        except Exception as e:
            logger.exception("🚨 Error creating listen key")
            raise BinanceAPIError(f"Error creating listen key: {e}")

    async def keepalive_listen_key(self, listen_key: str, futures: bool = False) -> Dict[str, Any]:
        """ListenKey'i yenile."""
        try:
            await self._require_keys()
            endpoint = "/fapi/v1/listenKey" if futures else "/api/v3/userDataStream"
            params = {"listenKey": listen_key}
            return await self.circuit_breaker.execute(
                self.http._request, "PUT", endpoint, params=params, signed=True, futures=futures
            )
        except Exception as e:
            logger.exception("🚨 Error keeping alive listen key")
            raise BinanceAPIError(f"Error keeping alive listen key: {e}")

    async def close_listen_key(self, listen_key: str, futures: bool = False) -> Dict[str, Any]:
        """ListenKey'i kapat."""
        try:
            await self._require_keys()
            endpoint = "/fapi/v1/listenKey" if futures else "/api/v3/userDataStream"
            params = {"listenKey": listen_key}
            return await self.circuit_breaker.execute(
                self.http._request, "DELETE", endpoint, params=params, signed=True, futures=futures
            )
        except Exception as e:
            logger.exception("🚨 Error closing listen key")
            raise BinanceAPIError(f"Error closing listen key: {e}")

    # ------------------------
    # Savings
    # ------------------------
    async def get_savings_product_list(self, product_type: str = "ACTIVITY", asset: Optional[str] = None) -> List[Dict[str, Any]]:
        """Savings ürün listesini getir."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"type": product_type}
            if asset:
                params["asset"] = asset.upper()
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/lending/daily/product/list", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting savings product list")
            raise BinanceAPIError(f"Error getting savings product list: {e}")

    async def purchase_savings_product(self, product_id: str, amount: float) -> Dict[str, Any]:
        """Savings ürünü satın al."""
        try:
            await self._require_keys()
            params = {"productId": product_id, "amount": amount}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/lending/daily/purchase", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error purchasing savings product")
            raise BinanceAPIError(f"Error purchasing savings product: {e}")

    async def get_savings_balance(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """Savings bakiyesini getir."""
        try:
            await self._require_keys()
            params = {"asset": asset.upper()} if asset else {}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/lending/daily/token/position", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting savings balance")
            raise BinanceAPIError(f"Error getting savings balance: {e}")

    # ------------------------
    # Mining
    # ------------------------
    async def get_mining_earnings_list(self, algo: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Mining kazanç listesini getir."""
        try:
            await self._require_keys()
            params: Dict[str, Any] = {"algo": algo}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/mining/payment/list", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting mining earnings list")
            raise BinanceAPIError(f"Error getting mining earnings list: {e}")

    async def get_mining_account_list(self, algo: str) -> List[Dict[str, Any]]:
        """Mining hesap listesini getir."""
        try:
            await self._require_keys()
            params = {"algo": algo}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/mining/worker/list", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting mining account list")
            raise BinanceAPIError(f"Error getting mining account list: {e}")

    # ------------------------
    # Sub-accounts
    # ------------------------
    async def get_sub_account_list(self, email: Optional[str] = None) -> List[Dict[str, Any]]:
        """Sub-account listesini getir."""
        try:
            await self._require_keys()
            params = {"email": email} if email else {}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/sub-account/list", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting sub-account list")
            raise BinanceAPIError(f"Error getting sub-account list: {e}")

    async def create_sub_account(self, sub_account_string: str) -> Dict[str, Any]:
        """Yeni sub-account oluştur."""
        try:
            await self._require_keys()
            params = {"subAccountString": sub_account_string}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/sub-account/virtualSubAccount", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error creating sub-account")
            raise BinanceAPIError(f"Error creating sub-account: {e}")

    async def get_sub_account_assets(self, email: str) -> Dict[str, Any]:
        """Sub-account varlıklarını getir."""
        try:
            await self._require_keys()
            params = {"email": email}
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v3/sub-account/assets", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting sub-account assets")
            raise BinanceAPIError(f"Error getting sub-account assets: {e}")

    # ------------------------
    # Additional endpoints (eklenen)
    # ------------------------
    async def get_dust_log(self, start_time: Optional[int] = None, end_time: Optional[int] = None) -> Dict[str, Any]:
        """
        Get dust conversion history.

        Args:
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            Dust log data

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/asset/dribblet", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting dust log")
            raise BinanceAPIError(f"Error getting dust log: {e}")

    async def convert_dust(self, asset: List[str]) -> Dict[str, Any]:
        """
        Convert dust assets to BNB.

        Args:
            asset: List of assets to convert

        Returns:
            Conversion result

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {"asset": ",".join([a.upper() for a in asset])}
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/asset/dust", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error converting dust")
            raise BinanceAPIError(f"Error converting dust: {e}")

    async def get_deposit_address(self, coin: str, network: Optional[str] = None) -> Dict[str, Any]:
        """
        Get deposit address for a specific coin.

        Args:
            coin: Coin symbol
            network: Network name (optional)

        Returns:
            Deposit address information

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {"coin": coin.upper()}
            if network:
                params["network"] = network
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/capital/deposit/address", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error getting deposit address for {coin}")
            raise BinanceAPIError(f"Error getting deposit address for {coin}: {e}")

    async def get_deposit_history(self, coin: Optional[str] = None, status: Optional[int] = None, 
                                 start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get deposit history.

        Args:
            coin: Coin symbol (optional)
            status: Deposit status (0: pending, 1: success)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of deposit records

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {}
            if coin:
                params["coin"] = coin.upper()
            if status is not None:
                params["status"] = status
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/capital/deposit/hisrec", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting deposit history")
            raise BinanceAPIError(f"Error getting deposit history: {e}")

    async def get_withdraw_history(self, coin: Optional[str] = None, status: Optional[int] = None,
                                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get withdrawal history.

        Args:
            coin: Coin symbol (optional)
            status: Withdrawal status (0: Email Sent, 1: Cancelled, 2: Awaiting Approval, 
                    3: Rejected, 4: Processing, 5: Failure, 6: Completed)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of withdrawal records

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {}
            if coin:
                params["coin"] = coin.upper()
            if status is not None:
                params["status"] = status
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            return await self.circuit_breaker.execute(
                self.http._request, "GET", "/sapi/v1/capital/withdraw/history", params=params, signed=True
            )
        except Exception as e:
            logger.exception("🚨 Error getting withdrawal history")
            raise BinanceAPIError(f"Error getting withdrawal history: {e}")

    async def withdraw(self, coin: str, address: str, amount: float, network: Optional[str] = None,
                      address_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Withdraw cryptocurrency.

        Args:
            coin: Coin symbol
            address: Withdrawal address
            amount: Amount to withdraw
            network: Network name (optional)
            address_tag: Secondary address identifier (for XRP, XMR, etc.)

        Returns:
            Withdrawal result

        Raises:
            BinanceAPIError on failure
        """
        try:
            await self._require_keys()
            params = {
                "coin": coin.upper(),
                "address": address,
                "amount": amount
            }
            if network:
                params["network"] = network
            if address_tag:
                params["addressTag"] = address_tag
            return await self.circuit_breaker.execute(
                self.http._request, "POST", "/sapi/v1/capital/withdraw/apply", params=params, signed=True
            )
        except Exception as e:
            logger.exception(f"🚨 Error withdrawing {coin}")
            raise BinanceAPIError(f"Error withdrawing {coin}: {e}")