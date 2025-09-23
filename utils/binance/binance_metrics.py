"""
utils/binance/binance_metrics.py
Metrics and monitoring for Binance API.
Tamamen Async Yapı+Thread-Safe+Atomic Operations
async/await pattern'ine uygun + aiogram 3.x e uygun + Router pattern yapıda +  PEP8 + type hints + docstring + async yapı + singleton + logging olacak yapıda
Değişken isimleri ve yapı daha tutarlı

"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from threading import Lock
import statistics
import logging
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for API requests."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    errors_by_type: Dict[str, int] = field(default_factory=dict)

@dataclass
class RateLimitMetrics:
    """Metrics for rate limiting."""
    weight_used: int = 0
    weight_limit: int = 1200
    last_reset_time: float = field(default_factory=time.time)

class AdvancedMetrics:
    """
    Thread-safe advanced async metrics collection.
    """
    
    _instance: Optional["AdvancedMetrics"] = None
    _lock: Lock = Lock()
    
    def __new__(cls, window_size: int = 1000) -> "AdvancedMetrics":
        """Singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(window_size)
            return cls._instance
    
    def _initialize(self, window_size: int) -> None:
        """Initialize metrics instance."""
        self.window_size = window_size
        self.request_metrics = RequestMetrics()
        self.rate_limit_metrics = RateLimitMetrics()
        self.start_time = time.time()
        self._async_lock = asyncio.Lock()  # Async operations için lock
        logger.info("✅ AdvancedMetrics initialized with async support")

    @asynccontextmanager
    async def _atomic_operation(self):
        """Async context manager for atomic operations."""
        async with self._async_lock:
            yield
    
    async def record_request(self, success: bool, response_time: float, 
                           error_type: Optional[str] = None) -> None:
        """
        Record API request metrics asynchronously.
        
        Args:
            success: Whether the request was successful
            response_time: Response time in seconds
            error_type: Type of error if request failed
        """
        async with self._async_lock:
            self.request_metrics.total_requests += 1
            self.request_metrics.total_response_time += response_time
            self.request_metrics.response_times.append(response_time)
            
            if success:
                self.request_metrics.successful_requests += 1
            else:
                self.request_metrics.failed_requests += 1
                if error_type:
                    self.request_metrics.errors_by_type[error_type] = (
                        self.request_metrics.errors_by_type.get(error_type, 0) + 1
                    )
    
    async def record_rate_limit(self, weight_used: int = 1) -> None:
        """
        Record rate limit usage asynchronously.
        
        Args:
            weight_used: Weight used by the request
        """
        async with self._async_lock:
            self.rate_limit_metrics.weight_used += weight_used
    
    async def reset_rate_limit(self, weight_limit: Optional[int] = None) -> None:
        """
        Reset rate limit metrics asynchronously.
        
        Args:
            weight_limit: Optional new weight limit
        """
        async with self._async_lock:
            self.rate_limit_metrics.weight_used = 0
            self.rate_limit_metrics.last_reset_time = time.time()
            if weight_limit is not None:
                self.rate_limit_metrics.weight_limit = weight_limit
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics asynchronously."""
        async with self._async_lock:
            response_times = list(self.request_metrics.response_times)
            
            # Calculate statistics safely
            total_requests = self.request_metrics.total_requests
            successful_requests = self.request_metrics.successful_requests
            
            # Average response time
            avg_response_time = (
                self.request_metrics.total_response_time / total_requests 
                if total_requests > 0 else 0
            )
            
            # Percentile calculation
            p95_response_time = self._calculate_percentile(response_times, 0.95)
            p99_response_time = self._calculate_percentile(response_times, 0.99)
            
            # Time-based calculations
            uptime_seconds = time.time() - self.start_time
            uptime_minutes = uptime_seconds / 60
            
            current_rpm = total_requests / uptime_minutes if uptime_minutes > 0 else 0
            
            # Rate limit calculations
            weight_used = self.rate_limit_metrics.weight_used
            weight_limit = self.rate_limit_metrics.weight_limit
            
            return {
                "uptime_seconds": uptime_seconds,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": self.request_metrics.failed_requests,
                "success_rate": (
                    successful_requests / total_requests * 100 
                    if total_requests > 0 else 100
                ),
                "average_response_time": avg_response_time,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "current_rpm": current_rpm,
                "weight_used": weight_used,
                "weight_limit": weight_limit,
                "weight_remaining": weight_limit - weight_used,
                "weight_percentage": (
                    weight_used / weight_limit * 100 
                    if weight_limit > 0 else 0
                ),
                "errors_by_type": dict(self.request_metrics.errors_by_type),
                "last_reset_seconds_ago": time.time() - self.rate_limit_metrics.last_reset_time,
            }
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile safely."""
        if not data:
            return 0.0
        
        try:
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile)
            return sorted_data[min(index, len(sorted_data) - 1)]
        except (IndexError, ValueError, TypeError):
            return 0.0
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics asynchronously."""
        metrics = await self.get_metrics()
        
        success_rate = metrics.get("success_rate", 100.0)
        avg_response_time = metrics.get("average_response_time", 0.0)
        weight_percentage = metrics.get("weight_percentage", 0.0)
        p95_response_time = metrics.get("p95_response_time", 0.0)
        
        status = "HEALTHY"
        issues = []
        warnings = []
        
        # Critical issues
        if success_rate < 90.0:
            status = "CRITICAL"
            issues.append(f"Critical: Low success rate: {success_rate:.1f}%")
        elif success_rate < 95.0:
            status = "DEGRADED"
            warnings.append(f"Low success rate: {success_rate:.1f}%")
        
        if avg_response_time > 3.0:
            status = "CRITICAL" if status != "CRITICAL" else status
            issues.append(f"Critical: High average response time: {avg_response_time:.2f}s")
        elif avg_response_time > 1.5:
            status = "DEGRADED" if status == "HEALTHY" else status
            warnings.append(f"High average response time: {avg_response_time:.2f}s")
        
        if p95_response_time > 5.0:
            status = "CRITICAL" if status != "CRITICAL" else status
            issues.append(f"Critical: High p95 response time: {p95_response_time:.2f}s")
        elif p95_response_time > 2.5:
            status = "DEGRADED" if status == "HEALTHY" else status
            warnings.append(f"High p95 response time: {p95_response_time:.2f}s")
        
        if weight_percentage > 95.0:
            status = "CRITICAL" if status != "CRITICAL" else status
            issues.append(f"Critical: Rate limit usage: {weight_percentage:.1f}%")
        elif weight_percentage > 80.0:
            status = "DEGRADED" if status == "HEALTHY" else status
            warnings.append(f"High rate limit usage: {weight_percentage:.1f}%")
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "metrics": metrics,
            "timestamp": time.time()
        }
    
    async def reset(self) -> None:
        """Reset all metrics asynchronously."""
        async with self._async_lock:
            self.request_metrics = RequestMetrics()
            await self.reset_rate_limit()
            self.start_time = time.time()
            logger.info("🔄 Metrics reset completed")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary asynchronously."""
        metrics = await self.get_metrics()
        
        return {
            "performance_grade": self._calculate_performance_grade(metrics),
            "recommendations": self._generate_recommendations(metrics),
            "key_metrics": {
                "success_rate": metrics["success_rate"],
                "average_response_time": metrics["average_response_time"],
                "p95_response_time": metrics["p95_response_time"],
                "current_rpm": metrics["current_rpm"],
                "weight_utilization": metrics["weight_percentage"]
            },
            "detailed_metrics": metrics
        }
    
    def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate a performance grade from A to F."""
        success_rate = metrics.get("success_rate", 100.0)
        avg_response_time = metrics.get("average_response_time", 0.0)
        weight_percentage = metrics.get("weight_percentage", 0.0)
        
        score = 0
        if success_rate >= 99.0: 
            score += 2
        elif success_rate >= 95.0: 
            score += 1
        
        if avg_response_time <= 0.5: 
            score += 2
        elif avg_response_time <= 1.0: 
            score += 1
        
        if weight_percentage <= 50.0: 
            score += 1
        
        grades = {5: "A", 4: "B", 3: "C", 2: "D", 1: "E", 0: "F"}
        return grades.get(score, "F")
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        success_rate = metrics.get("success_rate", 100.0)
        avg_response_time = metrics.get("average_response_time", 0.0)
        weight_percentage = metrics.get("weight_percentage", 0.0)
        
        if success_rate < 95.0:
            recommendations.append("Investigate API failures and implement retry logic")
        
        if avg_response_time > 1.0:
            recommendations.append("Optimize request batching and reduce API call frequency")
        
        if weight_percentage > 70.0:
            recommendations.append("Monitor rate limits closely and consider distributing requests")
        
        if not recommendations:
            recommendations.append("Performance is optimal. Continue current practices")
        
        return recommendations

# Global metrics instance management
class MetricsManager:
    """Manager for global metrics instance."""
    _instance: Optional[AdvancedMetrics] = None
    _lock: Lock = Lock()
    
    @classmethod
    def get_instance(cls) -> AdvancedMetrics:
        """Get the global metrics instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = AdvancedMetrics()
            return cls._instance

# Convenience async functions for global access
async def record_request(success: bool, response_time: float, error_type: Optional[str] = None) -> None:
    """Convenience function to record request metrics."""
    metrics = MetricsManager.get_instance()
    await metrics.record_request(success, response_time, error_type)

async def record_rate_limit(weight_used: int = 1) -> None:
    """Convenience function to record rate limit usage."""
    metrics = MetricsManager.get_instance()
    await metrics.record_rate_limit(weight_used)

async def get_current_metrics() -> Dict[str, Any]:
    """Convenience function to get current metrics."""
    metrics = MetricsManager.get_instance()
    return await metrics.get_metrics()

async def get_health_status() -> Dict[str, Any]:
    """Convenience function to get health status."""
    metrics = MetricsManager.get_instance()
    return await metrics.get_health_status()

async def reset_metrics() -> None:
    """Convenience function to reset metrics."""
    metrics = MetricsManager.get_instance()
    await metrics.reset()