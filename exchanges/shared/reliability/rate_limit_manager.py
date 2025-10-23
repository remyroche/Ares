"""
Rate Limiting Management

Handles rate limiting, backoff strategies, and request throttling.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class RateLimitStrategy(Enum):
    """Rate limiting strategy enumeration"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


@dataclass
class RateLimitStatus:
    """Rate limit status"""
    is_limited: bool
    remaining_requests: int
    reset_time: datetime
    retry_after: Optional[float] = None


class RateLimitManager:
    """
    Manages rate limiting for API requests.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"RateLimitManager.{exchange_name}")
        
        # Rate limit configurations
        self.rate_limits: Dict[str, RateLimit] = {}
        
        # Request tracking
        self.request_history: List[Tuple[datetime, str]] = []  # (timestamp, endpoint)
        self.endpoint_limits: Dict[str, List[datetime]] = {}  # endpoint -> [timestamps]
        
        # Default rate limits
        self.default_rate_limit = RateLimit(
            requests_per_second=10,
            requests_per_minute=600,
            requests_per_hour=36000,
            burst_limit=20
        )
        
        # Cleanup settings
        self.max_history_size = 10000
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.now()
    
    def set_rate_limit(self, endpoint: str, rate_limit: RateLimit) -> None:
        """Set rate limit for a specific endpoint."""
        self.rate_limits[endpoint] = rate_limit
        self.logger.info(f"Set rate limit for {endpoint}: {rate_limit.requests_per_second}/s")
    
    def get_rate_limit(self, endpoint: str) -> RateLimit:
        """Get rate limit for an endpoint."""
        return self.rate_limits.get(endpoint, self.default_rate_limit)
    
    async def check_rate_limit(self, endpoint: str) -> RateLimitStatus:
        """
        Check if request is allowed under rate limits.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            RateLimitStatus object
        """
        now = datetime.now()
        rate_limit = self.get_rate_limit(endpoint)
        
        # Clean up old history
        await self._cleanup_if_needed()
        
        # Get endpoint history
        if endpoint not in self.endpoint_limits:
            self.endpoint_limits[endpoint] = []
        
        endpoint_history = self.endpoint_limits[endpoint]
        
        # Remove old requests outside the time window
        cutoff_time = now - timedelta(hours=1)
        endpoint_history[:] = [ts for ts in endpoint_history if ts > cutoff_time]
        
        # Check different time windows
        second_ago = now - timedelta(seconds=1)
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        requests_last_second = len([ts for ts in endpoint_history if ts > second_ago])
        requests_last_minute = len([ts for ts in endpoint_history if ts > minute_ago])
        requests_last_hour = len([ts for ts in endpoint_history if ts > hour_ago])
        
        # Check if any limit is exceeded
        is_limited = (
            requests_last_second >= rate_limit.requests_per_second or
            requests_last_minute >= rate_limit.requests_per_minute or
            requests_last_hour >= rate_limit.requests_per_hour
        )
        
        # Calculate remaining requests
        remaining_requests = min(
            rate_limit.requests_per_second - requests_last_second,
            rate_limit.requests_per_minute - requests_last_minute,
            rate_limit.requests_per_hour - requests_last_hour
        )
        
        # Calculate reset time
        reset_time = now + timedelta(seconds=1)  # Default to 1 second
        
        # Calculate retry after time
        retry_after = None
        if is_limited:
            if requests_last_second >= rate_limit.requests_per_second:
                retry_after = 1.0
            elif requests_last_minute >= rate_limit.requests_per_minute:
                retry_after = 60.0
            elif requests_last_hour >= rate_limit.requests_per_hour:
                retry_after = 3600.0
        
        return RateLimitStatus(
            is_limited=is_limited,
            remaining_requests=max(0, remaining_requests),
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def wait_for_rate_limit(self, endpoint: str) -> None:
        """
        Wait if rate limit is exceeded.
        
        Args:
            endpoint: API endpoint
        """
        status = await self.check_rate_limit(endpoint)
        
        if status.is_limited and status.retry_after:
            self.logger.warning(f"Rate limit exceeded for {endpoint}, waiting {status.retry_after}s")
            await asyncio.sleep(status.retry_after)
    
    async def record_request(self, endpoint: str) -> None:
        """
        Record a request for rate limiting.
        
        Args:
            endpoint: API endpoint
        """
        now = datetime.now()
        
        # Add to global history
        self.request_history.append((now, endpoint))
        
        # Add to endpoint history
        if endpoint not in self.endpoint_limits:
            self.endpoint_limits[endpoint] = []
        self.endpoint_limits[endpoint].append(now)
        
        # Clean up if needed
        await self._cleanup_if_needed()
    
    async def _cleanup_if_needed(self) -> None:
        """Clean up old request history if needed."""
        now = datetime.now()
        
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Clean up global history
        cutoff_time = now - timedelta(hours=1)
        self.request_history[:] = [
            (ts, endpoint) for ts, endpoint in self.request_history
            if ts > cutoff_time
        ]
        
        # Clean up endpoint history
        for endpoint in list(self.endpoint_limits.keys()):
            self.endpoint_limits[endpoint] = [
                ts for ts in self.endpoint_limits[endpoint]
                if ts > cutoff_time
            ]
            
            # Remove empty endpoint entries
            if not self.endpoint_limits[endpoint]:
                del self.endpoint_limits[endpoint]
        
        # Limit history size
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
        
        self.last_cleanup = now
    
    def get_rate_limit_status(self, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status for an endpoint."""
        now = datetime.now()
        rate_limit = self.get_rate_limit(endpoint)
        
        if endpoint not in self.endpoint_limits:
            return {
                "endpoint": endpoint,
                "requests_last_second": 0,
                "requests_last_minute": 0,
                "requests_last_hour": 0,
                "remaining_requests": rate_limit.requests_per_second,
                "is_limited": False
            }
        
        endpoint_history = self.endpoint_limits[endpoint]
        
        second_ago = now - timedelta(seconds=1)
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        requests_last_second = len([ts for ts in endpoint_history if ts > second_ago])
        requests_last_minute = len([ts for ts in endpoint_history if ts > minute_ago])
        requests_last_hour = len([ts for ts in endpoint_history if ts > hour_ago])
        
        remaining_requests = min(
            rate_limit.requests_per_second - requests_last_second,
            rate_limit.requests_per_minute - requests_last_minute,
            rate_limit.requests_per_hour - requests_last_hour
        )
        
        is_limited = (
            requests_last_second >= rate_limit.requests_per_second or
            requests_last_minute >= rate_limit.requests_per_minute or
            requests_last_hour >= rate_limit.requests_per_hour
        )
        
        return {
            "endpoint": endpoint,
            "requests_last_second": requests_last_second,
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "remaining_requests": max(0, remaining_requests),
            "is_limited": is_limited,
            "rate_limit": {
                "requests_per_second": rate_limit.requests_per_second,
                "requests_per_minute": rate_limit.requests_per_minute,
                "requests_per_hour": rate_limit.requests_per_hour,
                "burst_limit": rate_limit.burst_limit
            }
        }
    
    def get_all_rate_limit_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limit status for all endpoints."""
        statuses = {}
        
        for endpoint in self.endpoint_limits.keys():
            statuses[endpoint] = self.get_rate_limit_status(endpoint)
        
        return statuses
    
    def get_rate_limit_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        total_requests = len(self.request_history)
        total_endpoints = len(self.endpoint_limits)
        
        # Calculate average requests per minute
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        recent_requests = len([ts for ts, _ in self.request_history if ts > minute_ago])
        
        return {
            "total_requests": total_requests,
            "total_endpoints": total_endpoints,
            "requests_last_minute": recent_requests,
            "configured_rate_limits": len(self.rate_limits),
            "last_cleanup": self.last_cleanup.isoformat()
        }
    
    def set_default_rate_limit(self, rate_limit: RateLimit) -> None:
        """Set default rate limit for unconfigured endpoints."""
        self.default_rate_limit = rate_limit
        self.logger.info(f"Set default rate limit: {rate_limit.requests_per_second}/s")
    
    def clear_rate_limit(self, endpoint: str) -> None:
        """Clear rate limit for an endpoint."""
        if endpoint in self.rate_limits:
            del self.rate_limits[endpoint]
            self.logger.info(f"Cleared rate limit for {endpoint}")
    
    def clear_all_rate_limits(self) -> None:
        """Clear all rate limits."""
        self.rate_limits.clear()
        self.logger.info("Cleared all rate limits")
    
    def reset_request_history(self, endpoint: Optional[str] = None) -> None:
        """Reset request history."""
        if endpoint:
            self.endpoint_limits.pop(endpoint, None)
            self.logger.info(f"Reset request history for {endpoint}")
        else:
            self.request_history.clear()
            self.endpoint_limits.clear()
            self.logger.info("Reset all request history")
    
    async def execute_with_rate_limit(self, endpoint: str, func, *args, **kwargs):
        """
        Execute a function with rate limiting.
        
        Args:
            endpoint: API endpoint
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Wait for rate limit if needed
        await self.wait_for_rate_limit(endpoint)
        
        # Record the request
        await self.record_request(endpoint)
        
        # Execute the function
        return await func(*args, **kwargs)