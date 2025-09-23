"""
Utility functions for Binance API.
"""

import time
import hashlib
import hmac
import json
import asyncio  # bu satırı ekle
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .binance_constants import KLINE_FIELDS
from .binance_exceptions import BinanceInvalidParameterError


def generate_signature(secret_key: str, params: Dict[str, Any]) -> str:
    """
    Generate HMAC SHA256 signature for Binance API requests.
    
    Args:
        secret_key: Binance secret key
        params: Request parameters
        
    Returns:
        HMAC SHA256 signature
    """
    query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def validate_symbol(symbol: str) -> bool:
    """
    Validate Binance symbol format.
    
    Args:
        symbol: Trading pair symbol
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.upper()
    # Basic validation - should contain at least 3 characters and have trading pairs
    if len(symbol) < 3 or not any(char.isdigit() for char in symbol):
        return False
    
    return True


def validate_interval(interval: str) -> bool:
    """
    Validate kline interval.
    
    Args:
        interval: Kline interval string
        
    Returns:
        True if valid, False otherwise
    """
    valid_intervals = [
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ]
    return interval in valid_intervals


def convert_timestamp(timestamp: Union[int, str]) -> datetime:
    """
    Convert Binance timestamp to datetime object.
    
    Args:
        timestamp: Binance timestamp (milliseconds)
        
    Returns:
        datetime object
    """
    if isinstance(timestamp, str):
        timestamp = int(timestamp)
    return datetime.fromtimestamp(timestamp / 1000)


def convert_to_timestamp(dt: datetime) -> int:
    """
    Convert datetime object to Binance timestamp.
    
    Args:
        dt: datetime object
        
    Returns:
        Timestamp in milliseconds
    """
    return int(dt.timestamp() * 1000)


def klines_to_dataframe(klines: List[List[Any]]) -> pd.DataFrame:
    """
    Convert Binance klines data to pandas DataFrame.
    
    Args:
        klines: List of kline data from Binance API
        
    Returns:
        pandas DataFrame with kline data
    """
    if not klines:
        return pd.DataFrame(columns=KLINE_FIELDS)
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=KLINE_FIELDS)
    
    # Convert data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                      'quote_asset_volume', 'taker_buy_base_asset_volume',
                      'taker_buy_quote_asset_volume', 'ignore']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamps
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Set index
    df.set_index('open_time', inplace=True)
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
    """
    if df.empty:
        return df
    
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    return df


def format_quantity(symbol: str, quantity: float, step_size: Optional[float] = None) -> float:
    """
    Format quantity according to Binance LOT_SIZE rules.
    
    Args:
        symbol: Trading symbol
        quantity: Original quantity
        step_size: Step size from exchange info
        
    Returns:
        Formatted quantity
    """
    if step_size is None:
        # Default step sizes for common symbols
        step_sizes = {
            'BTCUSDT': 0.000001,
            'ETHUSDT': 0.0001,
            'BNBUSDT': 0.001,
            'ADAUSDT': 1,
            'DOGEUSDT': 1,
        }
        step_size = step_sizes.get(symbol.upper(), 0.001)
    
    if step_size == 0:
        return quantity
    
    # Calculate precision
    precision = 0
    while step_size < 1:
        step_size *= 10
        precision += 1
    
    # Format quantity
    return round(quantity, precision)


def format_price(symbol: str, price: float, tick_size: Optional[float] = None) -> float:
    """
    Format price according to Binance PRICE_FILTER rules.
    
    Args:
        symbol: Trading symbol
        price: Original price
        tick_size: Tick size from exchange info
        
    Returns:
        Formatted price
    """
    if tick_size is None:
        # Default tick sizes for common symbols
        tick_sizes = {
            'BTCUSDT': 0.01,
            'ETHUSDT': 0.01,
            'BNBUSDT': 0.001,
            'ADAUSDT': 0.0001,
            'DOGEUSDT': 0.0001,
        }
        tick_size = tick_sizes.get(symbol.upper(), 0.0001)
    
    if tick_size == 0:
        return price
    
    # Calculate precision
    precision = 0
    while tick_size < 1:
        tick_size *= 10
        precision += 1
    
    # Format price
    return round(price, precision)


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100


def calculate_pnl(entry_price: float, exit_price: float, quantity: float, 
                 side: str, fee: float = 0.001) -> Dict[str, float]:
    """
    Calculate profit and loss for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Trade quantity
        side: Trade side ('BUY' or 'SELL')
        fee: Trading fee percentage
        
    Returns:
        Dictionary with PnL information
    """
    if side.upper() == 'BUY':
        gross_pnl = (exit_price - entry_price) * quantity
    else:  # SELL
        gross_pnl = (entry_price - exit_price) * quantity
    
    fee_amount = (entry_price * quantity * fee) + (exit_price * quantity * fee)
    net_pnl = gross_pnl - fee_amount
    
    return {
        'gross_pnl': gross_pnl,
        'fee_amount': fee_amount,
        'net_pnl': net_pnl,
        'roe': (net_pnl / (entry_price * quantity)) * 100 if entry_price * quantity > 0 else 0
    }


def validate_order_parameters(symbol: str, side: str, type_: str, quantity: float, 
                             price: Optional[float] = None) -> None:
    """
    Validate order parameters before sending to API.
    
    Args:
        symbol: Trading symbol
        side: Order side
        type_: Order type
        quantity: Order quantity
        price: Order price (for limit orders)
        
    Raises:
        BinanceInvalidParameterError: If parameters are invalid
    """
    if not validate_symbol(symbol):
        raise BinanceInvalidParameterError(f"Invalid symbol: {symbol}")
    
    if side.upper() not in ['BUY', 'SELL']:
        raise BinanceInvalidParameterError(f"Invalid side: {side}")
    
    if type_.upper() not in ['LIMIT', 'MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT',
                           'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT', 'LIMIT_MAKER']:
        raise BinanceInvalidParameterError(f"Invalid order type: {type_}")
    
    if quantity <= 0:
        raise BinanceInvalidParameterError("Quantity must be positive")
    
    if type_.upper() in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT', 'LIMIT_MAKER']:
        if price is None or price <= 0:
            raise BinanceInvalidParameterError("Price required for limit orders")


def sleep_until(timestamp: int) -> None:
    """
    Sleep until specified timestamp (milliseconds).
    
    Args:
        timestamp: Target timestamp in milliseconds
    """
    current_time = int(time.time() * 1000)
    sleep_time = (timestamp - current_time) / 1000
    if sleep_time > 0:
        time.sleep(sleep_time)


async def async_sleep_until(timestamp: int) -> None:
    """
    Async sleep until specified timestamp (milliseconds).
    
    Args:
        timestamp: Target timestamp in milliseconds
    """
    current_time = int(time.time() * 1000)
    sleep_time = (timestamp - current_time) / 1000
    if sleep_time > 0:
        await asyncio.sleep(sleep_time)