"""
Binance API constants and configuration.
"""

from typing import Any, Dict, Final, List, Optional, Union

# API URLs
BASE_URL: Final[str] = "https://api.binance.com"
FUTURES_URL: Final[str] = "https://fapi.binance.com"
TESTNET_BASE_URL: Final[str] = "https://testnet.binance.vision"
TESTNET_FUTURES_URL: Final[str] = "https://testnet.binancefuture.com"

# API Versions
API_VERSION: Final[str] = "v3"
FUTURES_API_VERSION: Final[str] = "v1"

# Rate Limits
RATE_LIMIT_IP: Final[int] = 1200  # Requests per minute for IP
RATE_LIMIT_ORDER: Final[int] = 10  # Orders per second
RATE_LIMIT_RAW_REQUESTS: Final[int] = 5000  # Raw requests per 5 minutes

# Time Intervals
INTERVALS: Final[List[str]] = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

# Order Types
ORDER_TYPES: Final[List[str]] = [
    "LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT",
    "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"
]

# Order Sides
ORDER_SIDES: Final[List[str]] = ["BUY", "SELL"]

# Time in Force
TIME_IN_FORCE: Final[List[str]] = ["GTC", "IOC", "FOK"]

# Response Types
RESPONSE_TYPES: Final[List[str]] = ["ACK", "RESULT", "FULL"]

# Kline Fields
KLINE_FIELDS: Final[List[str]] = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]

# WebSocket Streams
WS_STREAMS: Final[Dict[str, str]] = {
    "agg_trade": "aggTrade",
    "trade": "trade",
    "kline": "kline",
    "mini_ticker": "miniTicker",
    "ticker": "ticker",
    "book_ticker": "bookTicker",
    "depth": "depth",
    "depth_level": "depth{}",
    "user_data": "userData"
}

# Error Codes
ERROR_CODES: Final[Dict[int, str]] = {
    -1000: "UNKNOWN",
    -1001: "DISCONNECTED",
    -1002: "UNAUTHORIZED",
    -1003: "TOO_MANY_REQUESTS",
    -1006: "UNEXPECTED_RESP",
    -1007: "TIMEOUT",
    -1010: "ERROR_MSG_RECEIVED",
    -1013: "INVALID_MESSAGE",
    -1014: "UNKNOWN_ORDER_COMPOSITION",
    -1015: "TOO_MANY_ORDERS",
    -1016: "SERVICE_SHUTTING_DOWN",
    -1020: "UNSUPPORTED_OPERATION",
    -1021: "INVALID_TIMESTAMP",
    -1022: "INVALID_SIGNATURE",
    -1100: "ILLEGAL_CHARS",
    -1101: "TOO_MANY_PARAMETERS",
    -1102: "MANDATORY_PARAM_EMPTY_OR_MALFORMED",
    -1103: "UNKNOWN_PARAM",
    -1104: "UNREAD_PARAMETERS",
    -1105: "PARAM_EMPTY",
    -1106: "PARAM_NOT_REQUIRED",
    -1111: "BAD_PRECISION",
    -1112: "NO_DEPTH",
    -1114: "TIF_NOT_REQUIRED",
    -1115: "INVALID_TIF",
    -1116: "INVALID_ORDER_TYPE",
    -1117: "INVALID_SIDE",
    -1118: "EMPTY_NEW_CL_ORD_ID",
    -1119: "EMPTY_ORG_CL_ORD_ID",
    -1120: "BAD_INTERVAL",
    -1121: "BAD_SYMBOL",
    -1125: "INVALID_LISTEN_KEY",
    -1127: "MORE_THAN_XX_HOURS",
    -1128: "OPTIONAL_PARAMS_BAD_COMBO",
    -1130: "INVALID_PARAMETER",
    -2008: "BAD_API_ID",
    -2010: "NEW_ORDER_REJECTED",
    -2011: "CANCEL_REJECTED",
    -2013: "NO_SUCH_ORDER",
    -2014: "BAD_API_KEY_FMT",
    -2015: "REJECTED_MBX_KEY",
}

# HTTP Status Codes
HTTP_STATUS_CODES: Final[Dict[int, str]] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    415: "Unsupported Media Type",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

# Default Configuration
DEFAULT_CONFIG: Final[Dict[str, Any]] = {
    "timeout": 30,
    "recv_window": 5000,
    "max_retries": 3,
    "retry_delay": 1,
    "debug": False,
}
