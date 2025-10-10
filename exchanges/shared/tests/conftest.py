"""
Pytest configuration for shared exchange utilities tests.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock()


@pytest.fixture
def mock_time():
    """Create a mock time for testing."""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_exchange_config():
    """Create sample exchange configuration for testing."""
    return {
        "exchange_name": "test_exchange",
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "passphrase": "test_passphrase",
        "base_url": "https://api.test.com",
        "sandbox": True
    }