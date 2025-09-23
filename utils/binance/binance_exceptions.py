"""
Binance API custom exceptions.
"""

class BinanceError(Exception):
    """Base exception for all Binance API errors."""
    pass


class BinanceAPIError(BinanceError):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, code: int = None, response: dict = None):
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.code and self.response:
            return f"API Error {self.code}: {self.message} - Response: {self.response}"
        elif self.code:
            return f"API Error {self.code}: {self.message}"
        else:
            return f"API Error: {self.message}"


class BinanceAuthenticationError(BinanceAPIError):
    """Exception raised for authentication errors."""
    pass


class BinanceRequestError(BinanceError):
    """Exception raised for request-related errors."""
    
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.status_code:
            return f"Request Error {self.status_code}: {self.message}"
        else:
            return f"Request Error: {self.message}"


class BinanceWebSocketError(BinanceError):
    """Exception raised for WebSocket errors."""
    pass


class BinanceRateLimitError(BinanceAPIError):
    """Exception raised for rate limit errors."""
    pass


class BinanceOrderError(BinanceAPIError):
    """Exception raised for order-related errors."""
    pass


class BinanceInvalidSymbolError(BinanceAPIError):
    """Exception raised for invalid symbol errors."""
    pass


class BinanceInvalidIntervalError(BinanceAPIError):
    """Exception raised for invalid interval errors."""
    pass


class BinanceCircuitBreakerError(BinanceError):
    """Exception raised when circuit breaker is open."""
    pass


class BinanceTimeoutError(BinanceError):
    """Exception raised for timeout errors."""
    pass


class BinanceConnectionError(BinanceError):
    """Exception raised for connection errors."""
    pass


class BinanceInvalidParameterError(BinanceAPIError):
    """Exception raised for invalid parameter errors."""
    pass


class BinanceInsufficientBalanceError(BinanceAPIError):
    """Exception raised for insufficient balance errors."""
    pass