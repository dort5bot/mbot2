"""
Circuit Breaker pattern implementation for Binance API.
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass
from .binance_exceptions import BinanceCircuitBreakerError

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state container."""
    failures: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for resilient API calls.
    
    States:
    - CLOSED: Normal operation, requests are allowed
    - OPEN: Circuit is open, requests are blocked
    - HALF_OPEN: Testing if service is recovering
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        name: str = "binance_circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds to wait before half-open state
            half_open_timeout: Time in seconds for half-open state testing
            name: Name for logging purposes
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.name = name
        self.state = CircuitBreakerState()
        self.lock = asyncio.Lock()
        logger.info(f"✅ CircuitBreaker '{name}' initialized")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BinanceCircuitBreakerError: If circuit is open
            Exception: Original function exception
        """
        async with self.lock:
            current_state = self.state.state
            
            if current_state == "OPEN":
                if time.time() - self.state.last_failure_time > self.reset_timeout:
                    self.state.state = "HALF_OPEN"
                    logger.warning(f"⚠️ CircuitBreaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise BinanceCircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry in {self.reset_timeout - (time.time() - self.state.last_failure_time):.1f}s"
                    )
            
            elif current_state == "HALF_OPEN":
                # Allow only one request in half-open state
                if self.state.failures > 0:
                    raise BinanceCircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN. Testing in progress."
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self.lock:
            if self.state.state == "HALF_OPEN":
                # Success in half-open state, reset to closed
                self.state.failures = 0
                self.state.state = "CLOSED"
                self.state.last_failure_time = 0
                logger.info(f"✅ CircuitBreaker '{self.name}' reset to CLOSED")
            else:
                # Reset failure count on success
                self.state.failures = 0
    
    async def _on_failure(self, error: Exception) -> None:
        """Handle failed execution."""
        async with self.lock:
            self.state.failures += 1
            self.state.last_failure_time = time.time()
            
            if self.state.state == "HALF_OPEN":
                # Failure in half-open state, go back to open
                self.state.state = "OPEN"
                logger.error(f"❌ CircuitBreaker '{self.name}' moved back to OPEN")
                
            elif (self.state.state == "CLOSED" and 
                  self.state.failures >= self.failure_threshold):
                # Too many failures, open the circuit
                self.state.state = "OPEN"
                logger.error(f"❌ CircuitBreaker '{self.name}' opened due to {self.state.failures} failures")
    
    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Manually record a failure. Can be used externally.
        
        Args:
            error: Optional exception for logging
        """
        await self._on_failure(error or Exception("Manual failure"))
    
    async def record_success(self) -> None:
        """
        Manually record a success. Can be used externally.
        """
        await self._on_success()

    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "state": self.state.state,
            "failures": self.state.failures,
            "last_failure_time": self.state.last_failure_time,
            "time_since_last_failure": time.time() - self.state.last_failure_time if self.state.last_failure_time else 0
        }
    
    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self.lock:
            self.state.failures = 0
            self.state.state = "CLOSED"
            self.state.last_failure_time = 0
            logger.info(f"🔄 CircuitBreaker '{self.name}' manually reset")
    
    async def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        async with self.lock:
            self.state.state = "OPEN"
            self.state.last_failure_time = time.time()
            logger.warning(f"⚠️ CircuitBreaker '{self.name}' manually forced open")
    
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state.state == "CLOSED"
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state.state == "OPEN"
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state.state == "HALF_OPEN"
