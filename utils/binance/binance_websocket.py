
"""
WebSocket client for Binance API.

Config Yönetimi: WebSocketConfig sınıfı ile merkezi yapılandırma
"""

"""
WebSocket client for Binance API with aiogram 3.x compatibility.
"""

import asyncio
import json
import time  # bu satırı ekle
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
import websockets
from aiogram import Router, F
from aiogram.types import Message

from .binance_constants import BASE_URL, FUTURES_URL, WS_STREAMS
from .binance_exceptions import BinanceWebSocketError
from .binance_utils import generate_signature

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Enum for WebSocket stream types."""
    SPOT = "spot"
    FUTURES = "futures"
    USER_DATA = "user_data"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections."""
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    testnet: bool = False
    base_url: str = "wss://stream.binance.com:9443"
    futures_url: str = "wss://fstream.binance.com"
    reconnect_delay: int = 1
    max_reconnect_delay: int = 60
    keepalive_interval: int = 1800  # 30 minutes


class BinanceWebSocketManager:
    """
    WebSocket manager for Binance API streams with singleton pattern.
    """
    _instance: Optional['BinanceWebSocketManager'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet: bool = False,
        router: Optional[Router] = None
    ):
        """
        Initialize WebSocket manager with singleton pattern.
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key
            testnet: Whether to use testnet
            router: Aiogram router for message handling
        """
        if self._initialized:
            return

        self.config = WebSocketConfig(
            api_key=api_key,
            secret_key=secret_key,
            testnet=testnet,
            base_url="wss://testnet.binance.vision" if testnet else "wss://stream.binance.com:9443",
            futures_url="wss://stream.binancefuture.com" if testnet else "wss://fstream.binance.com"
        )

        self.connections: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.listen_keys: Dict[str, str] = {}
        
        self.router = router or Router()
        self._setup_handlers()
        
        self.running = False
        self._initialized = True
        
        logger.info("✅ BinanceWebSocketManager initialized with singleton pattern")

    def _setup_handlers(self) -> None:
        """Setup aiogram message handlers."""
        @self.router.message(F.text == "/status")
        async def cmd_status(message: Message) -> None:
            """Handle status command."""
            status = self.get_status()
            await message.answer(f"WebSocket Status:\nActive connections: {status['active_connections']}")

    async def initialize(self) -> None:
        """Initialize WebSocket manager asynchronously."""
        self.running = True
        logger.info("✅ BinanceWebSocketManager started")

    async def connect(
        self,
        streams: List[str],
        callback: Callable[[Dict[str, Any]], Any],
        futures: bool = False
    ) -> str:
        """
        Connect to WebSocket streams.
        
        Args:
            streams: List of streams to subscribe to
            callback: Callback function for messages
            futures: Whether to use futures streams
            
        Returns:
            Connection ID
            
        Raises:
            BinanceWebSocketError: If connection fails
        """
        try:
            url = self.config.futures_url if futures else self.config.base_url
            stream_param = '/'.join(streams)
            ws_url = f"{url}/stream?streams={stream_param}"
            
            connection_id = f"{'futures_' if futures else 'spot_'}{int(time.time() * 1000)}"
            
            self.connections[connection_id] = {
                'url': ws_url,
                'streams': streams,
                'callback': callback,
                'futures': futures,
                'running': True,
                'task': None
            }
            
            # Start connection task
            self.connections[connection_id]['task'] = asyncio.create_task(
                self._run_connection(connection_id)
            )
            
            logger.info(f"✅ WebSocket connection {connection_id} started for {len(streams)} streams")
            return connection_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create WebSocket connection: {e}")
            raise BinanceWebSocketError(f"Connection failed: {e}") from e

    async def _run_connection(self, connection_id: str) -> None:
        """Main WebSocket connection loop with proper error handling."""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        reconnect_delay = self.config.reconnect_delay
        
        while connection['running'] and self.running:
            try:
                async with websockets.connect(connection['url']) as ws:
                    logger.info(f"🔗 WebSocket {connection_id} connected")
                    reconnect_delay = self.config.reconnect_delay
                    
                    # Main message loop
                    while connection['running']:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = json.loads(message)
                            await connection['callback'](data)
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await ws.ping()
                            continue
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"❌ Callback error: {e}")
                            
            except websockets.ConnectionClosed:
                logger.warning(f"⚠️ WebSocket {connection_id} connection closed, reconnecting...")
            except Exception as e:
                logger.error(f"❌ WebSocket {connection_id} error: {e}")
                
            # Exponential backoff for reconnection
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, self.config.max_reconnect_delay)

    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect WebSocket connection.
        
        Args:
            connection_id: Connection ID to disconnect
            
        Raises:
            KeyError: If connection ID not found
        """
        if connection_id not in self.connections:
            raise KeyError(f"Connection {connection_id} not found")
            
        connection = self.connections[connection_id]
        connection['running'] = False
        
        if connection['task']:
            connection['task'].cancel()
            try:
                await connection['task']
            except asyncio.CancelledError:
                pass
                
        del self.connections[connection_id]
        logger.info(f"✅ WebSocket {connection_id} disconnected")

    async def subscribe_user_data(
        self, 
        callback: Callable[[Dict[str, Any]], Any], 
        futures: bool = False
    ) -> str:
        """
        Subscribe to user data stream.
        
        Args:
            callback: Callback function for messages
            futures: Whether to use futures user data
            
        Returns:
            Connection ID
            
        Raises:
            BinanceWebSocketError: If subscription fails
        """
        try:
            # Get listen key from REST API
            listen_key = await self._get_listen_key(futures)
            
            # Create user data connection
            url = self.config.futures_url if futures else self.config.base_url
            ws_url = f"{url}/ws/{listen_key}"
            
            connection_id = f"user_data_{'futures_' if futures else 'spot_'}{int(time.time() * 1000)}"
            
            self.connections[connection_id] = {
                'url': ws_url,
                'streams': ['userData'],
                'callback': callback,
                'futures': futures,
                'running': True,
                'listen_key': listen_key,
                'task': None,
                'keepalive_task': None
            }
            
            # Start keepalive task
            self.connections[connection_id]['keepalive_task'] = asyncio.create_task(
                self._keepalive_listen_key(listen_key, futures)
            )
            
            # Start connection
            self.connections[connection_id]['task'] = asyncio.create_task(
                self._run_connection(connection_id)
            )
            
            logger.info(f"✅ User data WebSocket {connection_id} started")
            return connection_id
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to user data: {e}")
            raise BinanceWebSocketError(f"User data subscription failed: {e}") from e

    async def _get_listen_key(self, futures: bool = False) -> str:
        """
        Get listen key from REST API.
        
        Args:
            futures: Whether to get futures listen key
            
        Returns:
            Listen key
            
        Raises:
            BinanceWebSocketError: If listen key retrieval fails
        """
        endpoint = '/fapi/v1/listenKey' if futures else '/api/v3/userDataStream'
        url = (FUTURES_URL if futures else BASE_URL) + endpoint
        
        headers = {}
        if self.config.api_key:
            headers['X-MBX-APIKEY'] = self.config.api_key
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['listenKey']
                    else:
                        error_text = await response.text()
                        raise BinanceWebSocketError(
                            f"Failed to get listen key: {response.status} - {error_text}"
                        )
        except Exception as e:
            logger.error(f"❌ Listen key retrieval failed: {e}")
            raise BinanceWebSocketError(f"Listen key retrieval failed: {e}") from e

    async def _keepalive_listen_key(self, listen_key: str, futures: bool = False) -> None:
        """
        Keep listen key alive.
        
        Args:
            listen_key: Listen key to keep alive
            futures: Whether it's a futures listen key
        """
        endpoint = '/fapi/v1/listenKey' if futures else '/api/v3/userDataStream'
        url = (FUTURES_URL if futures else BASE_URL) + endpoint
        
        headers = {}
        if self.config.api_key:
            headers['X-MBX-APIKEY'] = self.config.api_key
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    params = {'listenKey': listen_key}
                    async with session.put(url, headers=headers, params=params) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.warning(f"⚠️ Listen key keepalive failed: {response.status} - {error_text}")
                
                # Keepalive every 30 minutes
                await asyncio.sleep(self.config.keepalive_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Listen key keepalive error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    async def close_all(self) -> None:
        """Close all WebSocket connections gracefully."""
        self.running = False
        
        # Cancel all tasks
        for connection_id, connection in list(self.connections.items()):
            connection['running'] = False
            if connection.get('task'):
                connection['task'].cancel()
            if connection.get('keepalive_task'):
                connection['keepalive_task'].cancel()
        
        # Wait for tasks to complete
        tasks = []
        for connection in self.connections.values():
            if connection.get('task'):
                tasks.append(connection['task'])
            if connection.get('keepalive_task'):
                tasks.append(connection['keepalive_task'])
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.connections.clear()
        logger.info("✅ All WebSocket connections closed")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current WebSocket manager status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'active_connections': len(self.connections),
            'running': self.running,
            'subscriptions': {k: len(v) for k, v in self.subscriptions.items()}
        }

    # Convenience methods for common streams
    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], Any],
        futures: bool = False,
        interval: str = '1s'
    ) -> str:
        """
        Subscribe to ticker stream.
        
        Args:
            symbol: Trading symbol
            callback: Callback function
            futures: Whether to use futures
            interval: Update interval ('1s' or '3s')
            
        Returns:
            Connection ID
        """
        stream = f"{symbol.lower()}@ticker_{interval}"
        return await self.connect([stream], callback, futures)
    
    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], Any],
        futures: bool = False
    ) -> str:
        """
        Subscribe to kline stream.
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            callback: Callback function
            futures: Whether to use futures
            
        Returns:
            Connection ID
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        return await self.connect([stream], callback, futures)
    
    async def subscribe_depth(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], Any],
        futures: bool = False,
        levels: int = 20
    ) -> str:
        """
        Subscribe to depth stream.
        
        Args:
            symbol: Trading symbol
            callback: Callback function
            futures: Whether to use futures
            levels: Depth levels (5, 10, 20)
            
        Returns:
            Connection ID
        """
        stream = f"{symbol.lower()}@depth{levels}@100ms"
        return await self.connect([stream], callback, futures)
    
    async def subscribe_agg_trade(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], Any],
        futures: bool = False
    ) -> str:
        """
        Subscribe to aggregated trade stream.
        
        Args:
            symbol: Trading symbol
            callback: Callback function
            futures: Whether to use futures
            
        Returns:
            Connection ID
        """
        stream = f"{symbol.lower()}@aggTrade"
        return await self.connect([stream], callback, futures)
    
    def is_connected(self, connection_id: str) -> bool:
        """
        Check if connection is active.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            True if connected
        """
        return (connection_id in self.connections and 
                self.connections[connection_id]['running'] and
                self.running)
    
    async def __aenter__(self) -> 'BinanceWebSocketManager':
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close_all()


# Global instance for easy access
async def get_websocket_manager(
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    testnet: bool = False,
    router: Optional[Router] = None
) -> BinanceWebSocketManager:
    """
    Get or create the global WebSocket manager instance.
    
    Args:
        api_key: Binance API key
        secret_key: Binance secret key
        testnet: Whether to use testnet
        router: Aiogram router for message handling
        
    Returns:
        BinanceWebSocketManager instance
    """
    if BinanceWebSocketManager._instance is None:
        BinanceWebSocketManager._instance = BinanceWebSocketManager(
            api_key=api_key,
            secret_key=secret_key,
            testnet=testnet,
            router=router
        )
    return BinanceWebSocketManager._instance