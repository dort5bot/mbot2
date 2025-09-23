"""
utils/binance/binance_types.py
Binance API type definitions and type hints.
Literal Types: OrderSide, OrderType, TimeInForce, Interval gibi alanlar için spesifik değerler tanımlandı
OrderBookEntry: Order book girdileri için ayrı bir tip eklendi
OrderResponse: Order yanıtları için özelleştirilmiş tip
FuturesBalance: Futures bakiye bilgileri için ayrı tip
WSKlineData: WebSocket kline verileri için detaylı tip
ErrorResponse: Hata yanıtları için tip
OrderRequest ve FuturesOrderRequest: Sipariş istekleri için tipler
Exchange Info Types: Borsa bilgileri için kapsamlı tipler (RateLimit, SymbolFilter, SymbolInfo, ExchangeInfo)
MarginType ve PositionSide literal tipleri eklendi
total=False kullanılarak isteğe bağlı alanlar belirtildi
"""

from typing import TypedDict, List, Union, Optional, Dict, Any, Literal
from datetime import datetime

# Literal Types
OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", 
                   "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"]
TimeInForce = Literal["GTC", "IOC", "FOK"]
Interval = Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
OrderStatus = Literal["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "PENDING_CANCEL", "REJECTED", "EXPIRED"]
PositionSide = Literal["BOTH", "LONG", "SHORT"]
MarginType = Literal["ISOLATED", "CROSSED"]

# Basic Types
Symbol = str
Asset = str

# TypedDict Definitions
class Kline(TypedDict):
    """Kline/candlestick data structure."""
    open_time: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time: int
    quote_asset_volume: str
    number_of_trades: int
    taker_buy_base_asset_volume: str
    taker_buy_quote_asset_volume: str
    ignore: str

class OrderBookEntry(TypedDict):
    """Order book entry."""
    price: str
    quantity: str

class OrderBook(TypedDict):
    """Order book data structure."""
    lastUpdateId: int
    bids: List[List[str]]
    asks: List[List[str]]

class Ticker(TypedDict):
    """24hr ticker price change statistics."""
    symbol: str
    priceChange: str
    priceChangePercent: str
    weightedAvgPrice: str
    prevClosePrice: str
    lastPrice: str
    lastQty: str
    bidPrice: str
    askPrice: str
    openPrice: str
    highPrice: str
    lowPrice: str
    volume: str
    quoteVolume: str
    openTime: int
    closeTime: int
    firstId: int
    lastId: int
    count: int

class Balance(TypedDict):
    """Account balance structure."""
    asset: str
    free: str
    locked: str

class AccountInfo(TypedDict):
    """Account information structure."""
    makerCommission: int
    takerCommission: int
    buyerCommission: int
    sellerCommission: int
    canTrade: bool
    canWithdraw: bool
    canDeposit: bool
    updateTime: int
    accountType: str
    balances: List[Balance]
    permissions: List[str]

class Order(TypedDict):
    """Order information structure."""
    symbol: str
    orderId: int
    orderListId: int
    clientOrderId: str
    price: str
    origQty: str
    executedQty: str
    cummulativeQuoteQty: str
    status: OrderStatus
    timeInForce: TimeInForce
    type: OrderType
    side: OrderSide
    stopPrice: str
    icebergQty: str
    time: int
    updateTime: int
    isWorking: bool
    origQuoteOrderQty: str

class OrderResponse(Order):
    """Order response structure."""
    transactTime: int

class Trade(TypedDict):
    """Trade information structure."""
    id: int
    price: str
    qty: str
    quoteQty: str
    time: int
    isBuyerMaker: bool
    isBestMatch: bool

# Futures Types
class Position(TypedDict):
    """Futures position information."""
    symbol: str
    positionAmt: str
    entryPrice: str
    markPrice: str
    unRealizedProfit: str
    liquidationPrice: str
    leverage: str
    maxNotionalValue: str
    marginType: MarginType
    isolatedMargin: str
    isAutoAddMargin: str
    positionSide: PositionSide
    notional: str
    isolatedWallet: str
    updateTime: int

class FuturesBalance(TypedDict):
    """Futures balance structure."""
    accountAlias: str
    asset: str
    balance: str
    crossWalletBalance: str
    crossUnPnl: str
    availableBalance: str
    maxWithdrawAmount: str

class FuturesAccount(TypedDict):
    """Futures account information."""
    assets: List[FuturesBalance]
    positions: List[Position]
    canDeposit: bool
    canTrade: bool
    canWithdraw: bool
    feeTier: int
    updateTime: int
    totalInitialMargin: str
    totalMaintMargin: str
    totalWalletBalance: str
    totalUnrealizedProfit: str
    totalMarginBalance: str
    totalPositionInitialMargin: str
    totalOpenOrderInitialMargin: str
    totalCrossWalletBalance: str
    totalCrossUnPnl: str
    availableBalance: str
    maxWithdrawAmount: str

# WebSocket Types
class WSMessage(TypedDict):
    """WebSocket message structure."""
    stream: str
    data: Dict[str, Any]

class WSKlineData(TypedDict):
    """WebSocket kline data structure."""
    t: int  # Kline start time
    T: int  # Kline close time
    s: str  # Symbol
    i: str  # Interval
    o: str  # Open price
    c: str  # Close price
    h: str  # High price
    l: str  # Low price
    v: str  # Volume
    n: int  # Number of trades
    x: bool  # Is this kline closed?
    q: str  # Quote asset volume
    V: str  # Taker buy base asset volume
    Q: str  # Taker buy quote asset volume
    B: str  # Ignore

# Response Types
class BinanceResponse(TypedDict):
    """Generic Binance API response."""
    success: bool
    data: Optional[Union[Dict[str, Any], List[Any]]]
    error: Optional[str]
    error_code: Optional[int]
    timestamp: int

class ErrorResponse(TypedDict):
    """Error response structure."""
    code: int
    msg: str



#yok
class HealthStatus(TypedDict):
    """Health status structure."""
    status: Literal["HEALTHY", "DEGRADED", "CRITICAL"]
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    timestamp: float


# Request Types
class OrderRequest(TypedDict, total=False):
    """Order request parameters."""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Optional[str]
    quoteOrderQty: Optional[str]
    price: Optional[str]
    timeInForce: Optional[TimeInForce]
    newClientOrderId: Optional[str]
    stopPrice: Optional[str]
    icebergQty: Optional[str]
    newOrderRespType: Optional[str]
    recvWindow: Optional[int]

class RequestMetrics(TypedDict):
    """Request metrics structure."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_response_time: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float

class FuturesOrderRequest(OrderRequest):
    """Futures order request parameters."""
    positionSide: Optional[PositionSide]
    reduceOnly: Optional[str]
    activationPrice: Optional[str]
    callbackRate: Optional[str]
    closePosition: Optional[str]
    workingType: Optional[str]
    priceRate: Optional[str]
    priceProtect: Optional[str]
    priceProtect: Optional[bool]

# Exchange Info Types
class RateLimit(TypedDict):
    """Rate limit information."""
    rateLimitType: str
    interval: str
    intervalNum: int
    limit: int
    weight_used: int
    weight_limit: int
    last_reset_time: float
    requests_per_second: float

class SymbolFilter(TypedDict):
    """Symbol filter information."""
    filterType: str
    minPrice: Optional[str]
    maxPrice: Optional[str]
    tickSize: Optional[str]
    minQty: Optional[str]
    maxQty: Optional[str]
    stepSize: Optional[str]
    minNotional: Optional[str]
    limit: Optional[int]
    maxNumOrders: Optional[int]
    maxNumAlgoOrders: Optional[int]

class SymbolInfo(TypedDict):
    """Symbol information."""
    symbol: str
    status: str
    baseAsset: str
    baseAssetPrecision: int
    quoteAsset: str
    quotePrecision: int
    quoteAssetPrecision: int
    baseCommissionPrecision: int
    quoteCommissionPrecision: int
    orderTypes: List[OrderType]
    icebergAllowed: bool
    ocoAllowed: bool
    quoteOrderQtyMarketAllowed: bool
    allowTrailingStop: bool
    cancelReplaceAllowed: bool
    isSpotTradingAllowed: bool
    isMarginTradingAllowed: bool
    filters: List[SymbolFilter]
    permissions: List[str]
    defaultSelfTradePreventionMode: str
    allowedSelfTradePreventionModes: List[str]

class ExchangeInfo(TypedDict):
    """Exchange information."""
    timezone: str
    serverTime: int
    rateLimits: List[RateLimit]
    exchangeFilters: List[Any]
    symbols: List[SymbolInfo]