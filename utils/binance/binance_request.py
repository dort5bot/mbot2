# utils/binance/binance_request.py 
"""
binance/binance_request.py
HTTP client for Binance API requests.
Rate limiting: Weight limit sabiti (1200) hard-coded, dinamik olmalı

"""

import aiohttp
import asyncio
import time
import logging
import hashlib
import hmac
import urllib.parse
import json
import platform
from typing import Dict, List, Any, Optional, Union
from .binance_constants import BASE_URL, FUTURES_URL, DEFAULT_CONFIG
from .binance_exceptions import (
    BinanceAPIError, BinanceRequestError, BinanceRateLimitError,
    BinanceAuthenticationError, BinanceTimeoutError
)
from .binance_metrics import MetricsManager

logger = logging.getLogger(__name__)


class BinanceHTTPClient:
    """
    Async HTTP client for Binance API with retry logic and error handling.
    Implements proper rate limiting, connection pooling, and error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fapi_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize HTTP client for Binance API.
        
        Args:
            api_key: Binance API key for authenticated requests
            secret_key: Binance secret key for signing requests
            base_url: Base URL for Spot API (defaults to binance_constants.BASE_URL)
            fapi_url: Base URL for Futures API (defaults to binance_constants.FUTURES_URL)
            config: Configuration dictionary to override defaults
            session: Existing aiohttp session for connection reuse
        """
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Use provided URLs or fall back to constants
        self.base_url = base_url or BASE_URL
        self.fapi_url = fapi_url or FUTURES_URL
        
        # Merge provided config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Session management
        self._session_provided_externally = session is not None
        self._session = session
        
        # Rate limiting state
        self._last_request_time = 0
        self._min_request_interval = 1.0 / self.config.get("requests_per_second", 10)
        self._weight_used = 0
        self._weight_reset_time = time.time() + 60
        self._weight_limit = 1200  # Default Binance weight limit
        
        # Initialize metrics
        self.metrics = MetricsManager.get_instance()
        
        logger.info(
            f"✅ BinanceHTTPClient initialized - "
            f"Base URL: {self.base_url}, "
            f"FAPI URL: {self.fapi_url}, "
            f"Rate Limit: {self.config.get('requests_per_second')} req/s"
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session with proper connection pooling.
        
        Returns:
            aiohttp.ClientSession: Active session instance
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
            connector = aiohttp.TCPConnector(
                limit=self.config.get("connector_limit", 100),
                limit_per_host=self.config.get("connector_limit_per_host", 20),
                enable_cleanup_closed=True,
                force_close=False
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                # We own the session if we created it
                connector_owner=not self._session_provided_externally
            )
            
            logger.debug("Created new aiohttp ClientSession")
        
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session if we own it."""
        if (self._session and 
            not self._session.closed and 
            not self._session_provided_externally):
            await self._session.close()
            self._session = None
            logger.info("✅ BinanceHTTPClient session closed")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for private requests.
        
        Args:
            params: Request parameters to sign
            
        Returns:
            str: HMAC SHA256 signature
            
        Raises:
            BinanceAuthenticationError: If secret key is not available
        """
        if not self.secret_key:
            raise BinanceAuthenticationError("Secret key required for signed requests")
        
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _add_auth_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Add authentication headers to request.
        
        Args:
            headers: Existing headers dictionary
            
        Returns:
            Dict[str, str]: Updated headers with authentication
        """
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        return headers
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    async def _handle_rate_limit(self, response_headers: Dict[str, str]) -> None:
        """
        Handle rate limit information from response headers.
        
        Args:
            response_headers: Response headers from Binance API
        """
        # Extract rate limit information
        weight = response_headers.get('X-MBX-USED-WEIGHT', '0')
        order_count = response_headers.get('X-MBX-ORDER-COUNT-10S', '0')
        
        # Update weight-based rate limiting
        try:
            weight_used = int(weight)
            await self.metrics.record_rate_limit(weight_used)
            self._weight_used += weight_used
            
            # Update weight limit if provided
            weight_limit = response_headers.get('X-MBX-WEIGHT-LIMIT')
            if weight_limit:
                self._weight_limit = int(weight_limit)
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse rate limit headers: {e}")
        
        # Reset weight counter every minute
        if time.time() > self._weight_reset_time:
            logger.debug(f"Resetting weight counter: {self._weight_used} used")
            self._weight_used = 0
            self._weight_reset_time = time.time() + 60
            await self.metrics.reset_rate_limit()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        futures: bool = False,
        retries: Optional[int] = None
    ) -> Any:
        """
        Make HTTP request to Binance API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether request requires signature
            futures: Whether to use futures API
            retries: Number of retries (overrides config)
            
        Returns:
            Any: Parsed JSON response from API
            
        Raises:
            BinanceAPIError: For API-specific errors
            BinanceRequestError: For request-level errors
            BinanceRateLimitError: For rate limit errors
            BinanceAuthenticationError: For authentication errors
            BinanceTimeoutError: For request timeouts
        """
        retries = retries if retries is not None else self.config["max_retries"]
        params = params or {}
        
        # Prepare request URL and headers
        base_url = self.fapi_url if futures else self.base_url
        url = f"{base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'BinancePythonClient/1.0 (Python {platform.python_version()})'
        }
        
        # Add signature for authenticated requests
        if signed:
            params = params.copy()  # Don't modify original params
            params['timestamp'] = int(time.time() * 1000)
            if 'recvWindow' not in params:
                params['recvWindow'] = self.config["recv_window"]
            params['signature'] = self._generate_signature(params)
        
        headers = self._add_auth_headers(headers)
        
        # Make request with retry logic
        last_exception = None
        for attempt in range(retries + 1):
            try:
                await self._rate_limit()
                
                session = await self._get_session()
                start_time = time.time()
                
                # Prepare request based on method
                request_params = {
                    'method': method,
                    'url': url,
                    'headers': headers,
                    'params': params if method == 'GET' else None,
                }
                
                # For non-GET requests with data
                if method != 'GET':
                    if signed:
                        request_params['data'] = urllib.parse.urlencode(params)
                    else:
                        request_params['json'] = params
                
                async with session.request(**request_params) as response:
                    response_time = time.time() - start_time
                    
                    # Handle rate limits from response headers
                    await self._handle_rate_limit(response.headers)
                    
                    # Parse successful response
                    if response.status == 200:
                        data = await response.json()
                        await self.metrics.record_request(True, response_time)
                        return data
                    
                    # Handle error responses
                    error_data = await response.text()
                    await self._handle_error(response.status, error_data, response_time)
                    
            except asyncio.TimeoutError:
                error_msg = f"Request timeout after {self.config['timeout']}s"
                await self.metrics.record_request(False, self.config['timeout'], "timeout")
                last_exception = BinanceTimeoutError(error_msg)
                if attempt == retries:
                    raise last_exception
                
            except aiohttp.ClientError as e:
                error_msg = f"HTTP client error: {str(e)}"
                await self.metrics.record_request(False, 0, "connection_error")
                last_exception = BinanceRequestError(error_msg)
                if attempt == retries:
                    raise last_exception
            
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                await self.metrics.record_request(False, 0, "unexpected_error")
                last_exception = BinanceRequestError(error_msg)
                if attempt == retries:
                    raise last_exception
            
            # Exponential backoff for retries
            if attempt < retries:
                delay = self.config["retry_delay"] * (2 ** attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{retries} for {method} {endpoint} "
                    f"after {delay:.2f}s delay"
                )
                await asyncio.sleep(delay)
        
        # This should never be reached due to exception raising above
        raise last_exception or BinanceRequestError("Unknown request error")
    
    async def _handle_error(self, status_code: int, error_data: str, response_time: float) -> None:
        """
        Handle API error responses with proper error classification.
        
        Args:
            status_code: HTTP status code
            error_data: Raw error response data
            response_time: Response time in seconds
            
        Raises:
            Appropriate Binance exception based on error type
        """
        try:
            error_json = json.loads(error_data) if error_data else {}
            error_code = error_json.get('code', -1)
            error_msg = error_json.get('msg', 'Unknown error')
            
            await self.metrics.record_request(False, response_time, f"api_error_{error_code}")
            
            if status_code == 429:
                raise BinanceRateLimitError(error_msg, error_code, error_json)
            elif status_code == 401:
                raise BinanceAuthenticationError(error_msg, error_code, error_json)
            elif status_code >= 400:
                raise BinanceAPIError(error_msg, error_code, error_json)
            else:
                raise BinanceRequestError(f"HTTP {status_code}: {error_msg}")
                
        except (ValueError, json.JSONDecodeError):
            await self.metrics.record_request(False, response_time, "invalid_response")
            raise BinanceRequestError(f"HTTP {status_code}: Invalid response: {error_data}")
    
    # Public API methods
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        futures: bool = False
    ) -> Any:
        """
        Make GET request to Binance API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            signed: Whether request requires authentication
            futures: Whether to use futures API
            
        Returns:
            Any: Parsed JSON response
        """
        return await self._request('GET', endpoint, params, signed, futures)
    
    async def post(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        futures: bool = False
    ) -> Any:
        """
        Make POST request to Binance API.
        
        Args:
            endpoint: API endpoint path
            params: Request body parameters
            signed: Whether request requires authentication
            futures: Whether to use futures API
            
        Returns:
            Any: Parsed JSON response
        """
        return await self._request('POST', endpoint, params, signed, futures)
    
    async def put(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        futures: bool = False
    ) -> Any:
        """
        Make PUT request to Binance API.
        
        Args:
            endpoint: API endpoint path
            params: Request body parameters
            signed: Whether request requires authentication
            futures: Whether to use futures API
            
        Returns:
            Any: Parsed JSON response
        """
        return await self._request('PUT', endpoint, params, signed, futures)
    
    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        futures: bool = False
    ) -> Any:
        """
        Make DELETE request to Binance API.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether request requires authentication
            futures: Whether to use futures API
            
        Returns:
            Any: Parsed JSON response
        """
        return await self._request('DELETE', endpoint, params, signed, futures)
    
    def get_weight_usage(self) -> int:
        """Get current weight usage in the current minute."""
        return self._weight_used
    
    def get_weight_remaining(self) -> int:
        """Get remaining weight until reset."""
        return max(0, self._weight_limit - self._weight_used)
    
    def get_weight_limit(self) -> int:
        """Get current weight limit."""
        return self._weight_limit
    
    async def health_check(self) -> bool:
        """
        Check if Binance API is reachable.
        
        Returns:
            bool: True if API is reachable, False otherwise
        """
        try:
            await self.get('/api/v3/ping', timeout=5)
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.close()