"""
Unit tests for RateLimitManager.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from exchanges.shared.reliability.rate_limit_manager import (
    RateLimitManager, RateLimit, RateLimitStatus, RateLimitStrategy
)


class TestRateLimitManager:
    """Test cases for RateLimitManager."""

    @pytest.fixture
    def rate_limit_manager(self):
        """Create RateLimitManager instance for testing."""
        return RateLimitManager("test_exchange")

    @pytest.fixture
    def sample_rate_limit(self):
        """Create sample rate limit for testing."""
        return RateLimit(
            requests_per_second=10,
            requests_per_minute=600,
            requests_per_hour=36000,
            burst_limit=20,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )

    def test_initialization(self, rate_limit_manager):
        """Test RateLimitManager initialization."""
        assert rate_limit_manager.exchange_name == "test_exchange"
        assert len(rate_limit_manager.rate_limits) == 0
        assert len(rate_limit_manager.request_history) == 0
        assert len(rate_limit_manager.endpoint_limits) == 0
        assert rate_limit_manager.max_history_size == 10000
        assert rate_limit_manager.cleanup_interval == timedelta(minutes=5)

    def test_set_rate_limit(self, rate_limit_manager, sample_rate_limit):
        """Test setting rate limit for endpoint."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        assert "trading" in rate_limit_manager.rate_limits
        assert rate_limit_manager.rate_limits["trading"] == sample_rate_limit

    def test_get_rate_limit_configured(self, rate_limit_manager, sample_rate_limit):
        """Test getting configured rate limit."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        retrieved = rate_limit_manager.get_rate_limit("trading")
        
        assert retrieved == sample_rate_limit

    def test_get_rate_limit_default(self, rate_limit_manager):
        """Test getting default rate limit for unconfigured endpoint."""
        retrieved = rate_limit_manager.get_rate_limit("unknown")
        
        assert retrieved == rate_limit_manager.default_rate_limit

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_requests(self, rate_limit_manager, sample_rate_limit):
        """Test checking rate limit with no previous requests."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        status = await rate_limit_manager.check_rate_limit("trading")
        
        assert status.is_limited is False
        assert status.remaining_requests == 10  # requests_per_second
        assert status.retry_after is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_under_limit(self, rate_limit_manager, sample_rate_limit):
        """Test checking rate limit when under limit."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add some requests within limit
        now = datetime.now()
        for i in range(5):
            rate_limit_manager.endpoint_limits["trading"] = [
                now - timedelta(seconds=i) for i in range(5)
            ]
        
        status = await rate_limit_manager.check_rate_limit("trading")
        
        assert status.is_limited is False
        assert status.remaining_requests == 5  # 10 - 5

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_second(self, rate_limit_manager, sample_rate_limit):
        """Test checking rate limit when second limit exceeded."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add requests exceeding second limit
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(milliseconds=i*100) for i in range(15)  # 15 requests in last second
        ]
        
        status = await rate_limit_manager.check_rate_limit("trading")
        
        assert status.is_limited is True
        assert status.retry_after == 1.0

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_minute(self, rate_limit_manager, sample_rate_limit):
        """Test checking rate limit when minute limit exceeded."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add requests exceeding minute limit
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(seconds=i) for i in range(650)  # 650 requests in last minute
        ]
        
        status = await rate_limit_manager.check_rate_limit("trading")
        
        assert status.is_limited is True
        assert status.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_hour(self, rate_limit_manager, sample_rate_limit):
        """Test checking rate limit when hour limit exceeded."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add requests exceeding hour limit
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(minutes=i) for i in range(37000)  # 37000 requests in last hour
        ]
        
        status = await rate_limit_manager.check_rate_limit("trading")
        
        assert status.is_limited is True
        assert status.retry_after == 3600.0

    @pytest.mark.asyncio
    async def test_wait_for_rate_limit_no_wait(self, rate_limit_manager, sample_rate_limit):
        """Test waiting for rate limit when not limited."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Should not wait
        await rate_limit_manager.wait_for_rate_limit("trading")

    @pytest.mark.asyncio
    async def test_wait_for_rate_limit_with_wait(self, rate_limit_manager, sample_rate_limit):
        """Test waiting for rate limit when limited."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add requests exceeding limit
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(milliseconds=i*100) for i in range(15)
        ]
        
        start_time = datetime.now()
        await rate_limit_manager.wait_for_rate_limit("trading")
        end_time = datetime.now()
        
        # Should have waited approximately 1 second
        assert (end_time - start_time).total_seconds() >= 0.9

    @pytest.mark.asyncio
    async def test_record_request(self, rate_limit_manager):
        """Test recording a request."""
        await rate_limit_manager.record_request("trading")
        
        assert len(rate_limit_manager.request_history) == 1
        assert "trading" in rate_limit_manager.endpoint_limits
        assert len(rate_limit_manager.endpoint_limits["trading"]) == 1

    @pytest.mark.asyncio
    async def test_cleanup_if_needed_no_cleanup(self, rate_limit_manager):
        """Test cleanup when not needed."""
        rate_limit_manager.last_cleanup = datetime.now()
        
        # Add some old data
        old_time = datetime.now() - timedelta(hours=2)
        rate_limit_manager.request_history = [(old_time, "trading")]
        rate_limit_manager.endpoint_limits["trading"] = [old_time]
        
        await rate_limit_manager._cleanup_if_needed()
        
        # Should not clean up
        assert len(rate_limit_manager.request_history) == 1

    @pytest.mark.asyncio
    async def test_cleanup_if_needed_cleanup(self, rate_limit_manager):
        """Test cleanup when needed."""
        rate_limit_manager.last_cleanup = datetime.now() - timedelta(minutes=10)
        
        # Add old and new data
        old_time = datetime.now() - timedelta(hours=2)
        new_time = datetime.now()
        rate_limit_manager.request_history = [
            (old_time, "trading"),
            (new_time, "trading")
        ]
        rate_limit_manager.endpoint_limits["trading"] = [old_time, new_time]
        
        await rate_limit_manager._cleanup_if_needed()
        
        # Should clean up old data
        assert len(rate_limit_manager.request_history) == 1
        assert len(rate_limit_manager.endpoint_limits["trading"]) == 1

    @pytest.mark.asyncio
    async def test_cleanup_if_needed_limit_history_size(self, rate_limit_manager):
        """Test cleanup limiting history size."""
        rate_limit_manager.last_cleanup = datetime.now() - timedelta(minutes=10)
        rate_limit_manager.max_history_size = 5
        
        # Add more data than max size
        now = datetime.now()
        rate_limit_manager.request_history = [
            (now - timedelta(seconds=i), "trading") for i in range(10)
        ]
        
        await rate_limit_manager._cleanup_if_needed()
        
        # Should limit to max size
        assert len(rate_limit_manager.request_history) == 5

    def test_get_rate_limit_status_no_requests(self, rate_limit_manager, sample_rate_limit):
        """Test getting rate limit status with no requests."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        status = rate_limit_manager.get_rate_limit_status("trading")
        
        assert status["endpoint"] == "trading"
        assert status["requests_last_second"] == 0
        assert status["requests_last_minute"] == 0
        assert status["requests_last_hour"] == 0
        assert status["remaining_requests"] == 10
        assert status["is_limited"] is False

    def test_get_rate_limit_status_with_requests(self, rate_limit_manager, sample_rate_limit):
        """Test getting rate limit status with requests."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add some requests
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(seconds=i) for i in range(5)
        ]
        
        status = rate_limit_manager.get_rate_limit_status("trading")
        
        assert status["endpoint"] == "trading"
        assert status["requests_last_second"] == 1
        assert status["requests_last_minute"] == 5
        assert status["requests_last_hour"] == 5
        assert status["remaining_requests"] == 5
        assert status["is_limited"] is False

    def test_get_rate_limit_status_limited(self, rate_limit_manager, sample_rate_limit):
        """Test getting rate limit status when limited."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add requests exceeding limit
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(milliseconds=i*100) for i in range(15)
        ]
        
        status = rate_limit_manager.get_rate_limit_status("trading")
        
        assert status["is_limited"] is True
        assert "rate_limit" in status

    def test_get_all_rate_limit_statuses(self, rate_limit_manager, sample_rate_limit):
        """Test getting all rate limit statuses."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        rate_limit_manager.set_rate_limit("public", sample_rate_limit)
        
        # Add requests to one endpoint
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(seconds=i) for i in range(3)
        ]
        
        statuses = rate_limit_manager.get_all_rate_limit_statuses()
        
        assert "trading" in statuses
        assert "public" in statuses
        assert statuses["trading"]["requests_last_minute"] == 3
        assert statuses["public"]["requests_last_minute"] == 0

    def test_get_rate_limit_statistics(self, rate_limit_manager, sample_rate_limit):
        """Test getting rate limit statistics."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add some requests
        now = datetime.now()
        rate_limit_manager.request_history = [
            (now - timedelta(seconds=i), "trading") for i in range(5)
        ]
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(seconds=i) for i in range(5)
        ]
        
        stats = rate_limit_manager.get_rate_limit_statistics()
        
        assert stats["total_requests"] == 5
        assert stats["total_endpoints"] == 1
        assert stats["requests_last_minute"] == 5
        assert stats["configured_rate_limits"] == 1
        assert "last_cleanup" in stats

    def test_set_default_rate_limit(self, rate_limit_manager, sample_rate_limit):
        """Test setting default rate limit."""
        rate_limit_manager.set_default_rate_limit(sample_rate_limit)
        
        assert rate_limit_manager.default_rate_limit == sample_rate_limit

    def test_clear_rate_limit(self, rate_limit_manager, sample_rate_limit):
        """Test clearing rate limit for endpoint."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        rate_limit_manager.clear_rate_limit("trading")
        
        assert "trading" not in rate_limit_manager.rate_limits

    def test_clear_all_rate_limits(self, rate_limit_manager, sample_rate_limit):
        """Test clearing all rate limits."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        rate_limit_manager.set_rate_limit("public", sample_rate_limit)
        
        rate_limit_manager.clear_all_rate_limits()
        
        assert len(rate_limit_manager.rate_limits) == 0

    def test_reset_request_history_specific(self, rate_limit_manager):
        """Test resetting request history for specific endpoint."""
        rate_limit_manager.endpoint_limits["trading"] = [
            datetime.now() - timedelta(seconds=i) for i in range(5)
        ]
        rate_limit_manager.endpoint_limits["public"] = [
            datetime.now() - timedelta(seconds=i) for i in range(3)
        ]
        
        rate_limit_manager.reset_request_history("trading")
        
        assert "trading" not in rate_limit_manager.endpoint_limits
        assert "public" in rate_limit_manager.endpoint_limits

    def test_reset_request_history_all(self, rate_limit_manager):
        """Test resetting all request history."""
        rate_limit_manager.request_history = [
            (datetime.now() - timedelta(seconds=i), "trading") for i in range(5)
        ]
        rate_limit_manager.endpoint_limits["trading"] = [
            datetime.now() - timedelta(seconds=i) for i in range(5)
        ]
        
        rate_limit_manager.reset_request_history()
        
        assert len(rate_limit_manager.request_history) == 0
        assert len(rate_limit_manager.endpoint_limits) == 0

    @pytest.mark.asyncio
    async def test_execute_with_rate_limit(self, rate_limit_manager, sample_rate_limit):
        """Test executing function with rate limiting."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        async def test_function(arg1, arg2, kwarg1=None):
            return f"{arg1}_{arg2}_{kwarg1}"
        
        result = await rate_limit_manager.execute_with_rate_limit(
            "trading", test_function, "test1", "test2", kwarg1="test3"
        )
        
        assert result == "test1_test2_test3"
        assert len(rate_limit_manager.request_history) == 1

    @pytest.mark.asyncio
    async def test_execute_with_rate_limit_wait(self, rate_limit_manager, sample_rate_limit):
        """Test executing function with rate limiting that requires waiting."""
        rate_limit_manager.set_rate_limit("trading", sample_rate_limit)
        
        # Add requests exceeding limit
        now = datetime.now()
        rate_limit_manager.endpoint_limits["trading"] = [
            now - timedelta(milliseconds=i*100) for i in range(15)
        ]
        
        async def test_function():
            return "success"
        
        start_time = datetime.now()
        result = await rate_limit_manager.execute_with_rate_limit("trading", test_function)
        end_time = datetime.now()
        
        assert result == "success"
        # Should have waited
        assert (end_time - start_time).total_seconds() >= 0.9