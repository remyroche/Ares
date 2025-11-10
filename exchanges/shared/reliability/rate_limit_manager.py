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
from src.utils.tprint import tprint


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
        tprint(f"🔧 RateLimitManager.__init__ called with exchange_name={exchange_name}", "INFO")
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
        tprint(f"✅ RateLimitManager initialized for {exchange_name} with default limits: {self.default_rate_limit.requests_per_second}/s", "SUCCESS")
    
    def set_rate_limit(self, endpoint: str, rate_limit: RateLimit) -> None:
        """Set rate limit for a specific endpoint."""
        tprint(f"🔧 set_rate_limit called with endpoint={endpoint}, rate_limit={rate_limit.requests_per_second}/s", "INFO")
        self.rate_limits[endpoint] = rate_limit
        self.logger.info(f"Set rate limit for {endpoint}: {rate_limit.requests_per_second}/s")
        tprint(f"✅ Rate limit set for {endpoint}: {rate_limit.requests_per_second}/s, {rate_limit.requests_per_minute}/m, {rate_limit.requests_per_hour}/h", "SUCCESS")
    
    def get_rate_limit(self, endpoint: str) -> RateLimit:
        """Get rate limit for an endpoint."""
        tprint(f"🔧 get_rate_limit called with endpoint={endpoint}", "INFO")
        rate_limit = self.rate_limits.get(endpoint, self.default_rate_limit)
        is_default = endpoint not in self.rate_limits
        tprint(f"✅ Rate limit retrieved for {endpoint}: {rate_limit.requests_per_second}/s (default={is_default})", "SUCCESS")
        return rate_limit
    
    async def check_rate_limit(self, endpoint: str) -> RateLimitStatus:
        """
        Check if request is allowed under rate limits.

        Args:
            endpoint: API endpoint

        Returns:
            RateLimitStatus object
        """
        tprint(f"🔧 check_rate_limit called with endpoint={endpoint}", "INFO")
        now = datetime.now()
        rate_limit = self.get_rate_limit(endpoint)

        # Clean up old history
        await self._cleanup_if_needed()

        # Get endpoint history
        if endpoint not in self.endpoint_limits:
            self.endpoint_limits[endpoint] = []
            tprint(f"🆕 Created new endpoint history tracking for {endpoint}", "INFO")

        endpoint_history = self.endpoint_limits[endpoint]

        # Remove old requests outside the time window
        cutoff_time = now - timedelta(hours=1)
        old_count = len(endpoint_history)
        endpoint_history[:] = [ts for ts in endpoint_history if ts > cutoff_time]
        if old_count > len(endpoint_history):
            tprint(f"🧹 Cleaned up {old_count - len(endpoint_history)} old requests for {endpoint}", "INFO")

        # Check different time windows
        second_ago = now - timedelta(seconds=1)
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        requests_last_second = len([ts for ts in endpoint_history if ts > second_ago])
        requests_last_minute = len([ts for ts in endpoint_history if ts > minute_ago])
        requests_last_hour = len([ts for ts in endpoint_history if ts > hour_ago])

        tprint(f"📊 Rate limit check for {endpoint}: {requests_last_second}/{rate_limit.requests_per_second}s, {requests_last_minute}/{rate_limit.requests_per_minute}m, {requests_last_hour}/{rate_limit.requests_per_hour}h", "INFO")

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
            tprint(f"⚠️ Rate limit exceeded for {endpoint}, retry_after={retry_after}s", "WARNING")
        else:
            tprint(f"✅ Rate limit check passed for {endpoint}, remaining={remaining_requests}", "SUCCESS")

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
        tprint(f"🔧 wait_for_rate_limit called with endpoint={endpoint}", "INFO")
        status = await self.check_rate_limit(endpoint)

        if status.is_limited and status.retry_after:
            self.logger.warning(f"Rate limit exceeded for {endpoint}, waiting {status.retry_after}s")
            tprint(f"⏳ Waiting {status.retry_after}s for rate limit on {endpoint}", "WARNING")
            await asyncio.sleep(status.retry_after)
            tprint(f"✅ Rate limit wait completed for {endpoint}", "SUCCESS")
        else:
            tprint(f"✅ No rate limit wait needed for {endpoint}", "SUCCESS")
    
    async def record_request(self, endpoint: str) -> None:
        """
        Record a request for rate limiting.

        Args:
            endpoint: API endpoint
        """
        tprint(f"🔧 record_request called with endpoint={endpoint}", "INFO")
        now = datetime.now()

        # Add to global history
        self.request_history.append((now, endpoint))

        # Add to endpoint history
        if endpoint not in self.endpoint_limits:
            self.endpoint_limits[endpoint] = []
        self.endpoint_limits[endpoint].append(now)

        tprint(f"✅ Request recorded for {endpoint}, total_history={len(self.request_history)}, endpoint_history={len(self.endpoint_limits[endpoint])}", "SUCCESS")

        # Clean up if needed
        await self._cleanup_if_needed()
    
    async def _cleanup_if_needed(self) -> None:
        """Clean up old request history if needed."""
        now = datetime.now()

        if now - self.last_cleanup < self.cleanup_interval:
            return

        tprint(f"🔧 _cleanup_if_needed starting cleanup, last_cleanup={self.last_cleanup}", "INFO")
        old_history_size = len(self.request_history)
        old_endpoints_count = len(self.endpoint_limits)

        # Clean up global history
        cutoff_time = now - timedelta(hours=1)
        self.request_history[:] = [
            (ts, endpoint) for ts, endpoint in self.request_history
            if ts > cutoff_time
        ]

        # Clean up endpoint history
        removed_endpoints = []
        for endpoint in list(self.endpoint_limits.keys()):
            old_count = len(self.endpoint_limits[endpoint])
            self.endpoint_limits[endpoint] = [
                ts for ts in self.endpoint_limits[endpoint]
                if ts > cutoff_time
            ]

            # Remove empty endpoint entries
            if not self.endpoint_limits[endpoint]:
                del self.endpoint_limits[endpoint]
                removed_endpoints.append(endpoint)

        # Limit history size
        if len(self.request_history) > self.max_history_size:
            trimmed = len(self.request_history) - self.max_history_size
            self.request_history = self.request_history[-self.max_history_size:]
            tprint(f"⚠️ History size exceeded max, trimmed {trimmed} oldest entries", "WARNING")

        self.last_cleanup = now
        tprint(f"✅ Cleanup completed: history {old_history_size}->{len(self.request_history)}, endpoints {old_endpoints_count}->{len(self.endpoint_limits)}, removed_endpoints={len(removed_endpoints)}", "SUCCESS")
    
    def get_rate_limit_status(self, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status for an endpoint."""
        tprint(f"🔧 get_rate_limit_status called with endpoint={endpoint}", "INFO")
        now = datetime.now()
        rate_limit = self.get_rate_limit(endpoint)

        if endpoint not in self.endpoint_limits:
            tprint(f"✅ No history for {endpoint}, returning default status", "SUCCESS")
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

        tprint(f"✅ Status retrieved for {endpoint}: {requests_last_second}/{rate_limit.requests_per_second}s, limited={is_limited}", "SUCCESS")

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
        tprint(f"🔧 get_all_rate_limit_statuses called, total_endpoints={len(self.endpoint_limits)}", "INFO")
        statuses = {}

        for endpoint in self.endpoint_limits.keys():
            statuses[endpoint] = self.get_rate_limit_status(endpoint)

        tprint(f"✅ Retrieved status for {len(statuses)} endpoints", "SUCCESS")
        return statuses
    
    def get_rate_limit_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        tprint(f"🔧 get_rate_limit_statistics called", "INFO")
        total_requests = len(self.request_history)
        total_endpoints = len(self.endpoint_limits)

        # Calculate average requests per minute
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        recent_requests = len([ts for ts, _ in self.request_history if ts > minute_ago])

        stats = {
            "total_requests": total_requests,
            "total_endpoints": total_endpoints,
            "requests_last_minute": recent_requests,
            "configured_rate_limits": len(self.rate_limits),
            "last_cleanup": self.last_cleanup.isoformat()
        }
        tprint(f"✅ Statistics retrieved: total_requests={total_requests}, total_endpoints={total_endpoints}, requests_last_minute={recent_requests}", "SUCCESS")
        return stats
    
    def set_default_rate_limit(self, rate_limit: RateLimit) -> None:
        """Set default rate limit for unconfigured endpoints."""
        tprint(f"🔧 set_default_rate_limit called with rate_limit={rate_limit.requests_per_second}/s", "INFO")
        self.default_rate_limit = rate_limit
        self.logger.info(f"Set default rate limit: {rate_limit.requests_per_second}/s")
        tprint(f"✅ Default rate limit set: {rate_limit.requests_per_second}/s, {rate_limit.requests_per_minute}/m, {rate_limit.requests_per_hour}/h", "SUCCESS")
    
    def clear_rate_limit(self, endpoint: str) -> None:
        """Clear rate limit for an endpoint."""
        tprint(f"🔧 clear_rate_limit called with endpoint={endpoint}", "INFO")
        if endpoint in self.rate_limits:
            del self.rate_limits[endpoint]
            self.logger.info(f"Cleared rate limit for {endpoint}")
            tprint(f"✅ Rate limit cleared for {endpoint}", "SUCCESS")
        else:
            tprint(f"⚠️ No rate limit configured for {endpoint}, nothing to clear", "WARNING")
    
    def clear_all_rate_limits(self) -> None:
        """Clear all rate limits."""
        tprint(f"🔧 clear_all_rate_limits called, current_limits={len(self.rate_limits)}", "INFO")
        count = len(self.rate_limits)
        self.rate_limits.clear()
        self.logger.info("Cleared all rate limits")
        tprint(f"✅ Cleared {count} rate limits", "SUCCESS")
    
    def reset_request_history(self, endpoint: Optional[str] = None) -> None:
        """Reset request history."""
        tprint(f"🔧 reset_request_history called with endpoint={endpoint}", "INFO")
        if endpoint:
            had_history = endpoint in self.endpoint_limits
            self.endpoint_limits.pop(endpoint, None)
            self.logger.info(f"Reset request history for {endpoint}")
            if had_history:
                tprint(f"✅ Request history reset for {endpoint}", "SUCCESS")
            else:
                tprint(f"⚠️ No request history found for {endpoint}", "WARNING")
        else:
            history_count = len(self.request_history)
            endpoint_count = len(self.endpoint_limits)
            self.request_history.clear()
            self.endpoint_limits.clear()
            self.logger.info("Reset all request history")
            tprint(f"✅ All request history reset: cleared {history_count} requests and {endpoint_count} endpoints", "SUCCESS")
    
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
        tprint(f"🔧 execute_with_rate_limit called with endpoint={endpoint}, func={func.__name__ if hasattr(func, '__name__') else 'unknown'}", "INFO")

        # Wait for rate limit if needed
        await self.wait_for_rate_limit(endpoint)

        # Record the request
        await self.record_request(endpoint)

        # Execute the function
        try:
            tprint(f"⚡ Executing function {func.__name__ if hasattr(func, '__name__') else 'unknown'} for {endpoint}", "INFO")
            result = await func(*args, **kwargs)
            tprint(f"✅ Function executed successfully for {endpoint}", "SUCCESS")
            return result
        except Exception as e:
            tprint(f"❌ Error executing function for {endpoint}: {str(e)}", "ERROR")
            raise