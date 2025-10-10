"""
Unit tests for AuthenticationManager.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from exchanges.shared.auth.auth_manager import AuthenticationManager, AuthConfig, APIKeyPermission


class TestAuthenticationManager:
    """Test cases for AuthenticationManager."""

    @pytest.fixture
    def auth_manager(self):
        """Create AuthenticationManager instance for testing."""
        return AuthenticationManager("test_exchange")

    @pytest.fixture
    def auth_config(self):
        """Create AuthConfig for testing."""
        return AuthConfig(
            exchange_name="test_exchange",
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            permissions={APIKeyPermission.READ, APIKeyPermission.TRADE}
        )

    @pytest.fixture
    def mock_auth_functions(self):
        """Create mock authentication functions."""
        return {
            "get_server_time": AsyncMock(return_value=1234567890000),
            "test_connection": AsyncMock(return_value=True),
            "get_account_info": AsyncMock(return_value={"account_id": "123"})
        }

    def test_initialization(self, auth_manager):
        """Test AuthenticationManager initialization."""
        assert auth_manager.exchange_name == "test_exchange"
        assert not auth_manager.is_authenticated
        assert auth_manager.current_key_id is None
        assert auth_manager.last_auth_check is None

    def test_register_auth_functions(self, auth_manager, mock_auth_functions):
        """Test registering authentication functions."""
        auth_manager.register_auth_functions(
            get_server_time=mock_auth_functions["get_server_time"],
            test_connection=mock_auth_functions["test_connection"],
            get_account_info=mock_auth_functions["get_account_info"]
        )
        
        assert "get_server_time" in auth_manager.auth_functions
        assert "test_connection" in auth_manager.auth_functions
        assert "get_account_info" in auth_manager.auth_functions

    @pytest.mark.asyncio
    async def test_successful_authentication(self, auth_manager, auth_config, mock_auth_functions):
        """Test successful authentication flow."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        
        result = await auth_manager.authenticate(auth_config)
        
        assert result is True
        assert auth_manager.is_authenticated is True
        assert auth_manager.current_key_id is not None
        assert auth_manager.last_auth_check is not None

    @pytest.mark.asyncio
    async def test_authentication_connection_failure(self, auth_manager, auth_config):
        """Test authentication with connection failure."""
        mock_functions = {
            "get_server_time": AsyncMock(return_value=1234567890000),
            "test_connection": AsyncMock(return_value=False),
            "get_account_info": AsyncMock(return_value={"account_id": "123"})
        }
        auth_manager.register_auth_functions(**mock_functions)
        
        result = await auth_manager.authenticate(auth_config)
        
        assert result is False
        assert not auth_manager.is_authenticated

    @pytest.mark.asyncio
    async def test_authentication_exception(self, auth_manager, auth_config):
        """Test authentication with exception."""
        mock_functions = {
            "get_server_time": AsyncMock(side_effect=Exception("Network error")),
            "test_connection": AsyncMock(return_value=True),
            "get_account_info": AsyncMock(return_value={"account_id": "123"})
        }
        auth_manager.register_auth_functions(**mock_functions)
        
        result = await auth_manager.authenticate(auth_config)
        
        assert result is False
        assert not auth_manager.is_authenticated

    @pytest.mark.asyncio
    async def test_reauthentication(self, auth_manager, auth_config, mock_auth_functions):
        """Test re-authentication flow."""
        # First authenticate
        auth_manager.register_auth_functions(**mock_auth_functions)
        await auth_manager.authenticate(auth_config)
        
        # Test re-authentication
        result = await auth_manager.reauthenticate()
        
        assert result is True
        assert auth_manager.is_authenticated is True

    @pytest.mark.asyncio
    async def test_reauthentication_no_key_id(self, auth_manager):
        """Test re-authentication without current key ID."""
        result = await auth_manager.reauthenticate()
        
        assert result is False

    def test_get_auth_headers_not_authenticated(self, auth_manager):
        """Test getting auth headers when not authenticated."""
        headers = auth_manager.get_auth_headers("GET", "/test")
        
        assert headers is None

    @pytest.mark.asyncio
    async def test_get_auth_headers_authenticated(self, auth_manager, auth_config, mock_auth_functions):
        """Test getting auth headers when authenticated."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        await auth_manager.authenticate(auth_config)
        
        headers = auth_manager.get_auth_headers("GET", "/test", "body")
        
        assert headers is not None
        assert "X-Timestamp" in headers

    def test_get_timestamp_for_request(self, auth_manager):
        """Test getting timestamp for request."""
        timestamp = auth_manager.get_timestamp_for_request()
        
        assert isinstance(timestamp, int)
        assert timestamp > 0

    def test_is_authenticated_and_valid_fresh(self, auth_manager):
        """Test authentication validity with fresh auth."""
        auth_manager.is_authenticated = True
        auth_manager.last_auth_check = datetime.now()
        
        # Mock time sync as synced
        with patch.object(auth_manager.time_sync_manager, 'is_time_synced', return_value=True):
            result = auth_manager.is_authenticated_and_valid()
        
        assert result is True

    def test_is_authenticated_and_valid_stale(self, auth_manager):
        """Test authentication validity with stale auth."""
        auth_manager.is_authenticated = True
        auth_manager.last_auth_check = datetime.now() - timedelta(hours=2)
        
        result = auth_manager.is_authenticated_and_valid()
        
        assert result is False

    def test_is_authenticated_and_valid_not_synced(self, auth_manager):
        """Test authentication validity when time not synced."""
        auth_manager.is_authenticated = True
        auth_manager.last_auth_check = datetime.now()
        
        with patch.object(auth_manager.time_sync_manager, 'is_time_synced', return_value=False):
            result = auth_manager.is_authenticated_and_valid()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_authentication_valid(self, auth_manager):
        """Test authentication validation when valid."""
        auth_manager.is_authenticated = True
        auth_manager.last_auth_check = datetime.now()
        
        with patch.object(auth_manager.time_sync_manager, 'is_time_synced', return_value=True):
            result = await auth_manager.validate_authentication()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_authentication_invalid(self, auth_manager, auth_config, mock_auth_functions):
        """Test authentication validation when invalid."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        await auth_manager.authenticate(auth_config)
        
        # Make auth invalid
        auth_manager.last_auth_check = datetime.now() - timedelta(hours=2)
        
        result = await auth_manager.validate_authentication()
        
        assert result is True  # Should re-authenticate successfully

    def test_get_current_permissions_no_key(self, auth_manager):
        """Test getting permissions without current key."""
        permissions = auth_manager.get_current_permissions()
        
        assert permissions == set()

    @pytest.mark.asyncio
    async def test_get_current_permissions_with_key(self, auth_manager, auth_config, mock_auth_functions):
        """Test getting permissions with current key."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        await auth_manager.authenticate(auth_config)
        
        permissions = auth_manager.get_current_permissions()
        
        assert APIKeyPermission.READ in permissions
        assert APIKeyPermission.TRADE in permissions

    def test_has_permission(self, auth_manager):
        """Test checking specific permission."""
        # Mock current permissions
        with patch.object(auth_manager, 'get_current_permissions', return_value={APIKeyPermission.READ}):
            assert auth_manager.has_permission(APIKeyPermission.READ) is True
            assert auth_manager.has_permission(APIKeyPermission.TRADE) is False

    def test_can_trade(self, auth_manager):
        """Test trade permission check."""
        with patch.object(auth_manager, 'has_permission', return_value=True):
            assert auth_manager.can_trade() is True
        
        with patch.object(auth_manager, 'has_permission', return_value=False):
            assert auth_manager.can_trade() is False

    def test_can_withdraw(self, auth_manager):
        """Test withdraw permission check."""
        with patch.object(auth_manager, 'has_permission', return_value=True):
            assert auth_manager.can_withdraw() is True
        
        with patch.object(auth_manager, 'has_permission', return_value=False):
            assert auth_manager.can_withdraw() is False

    def test_can_read(self, auth_manager):
        """Test read permission check."""
        with patch.object(auth_manager, 'has_permission', return_value=True):
            assert auth_manager.can_read() is True
        
        with patch.object(auth_manager, 'has_permission', return_value=False):
            assert auth_manager.can_read() is False

    def test_get_clock_skew(self, auth_manager):
        """Test getting clock skew."""
        with patch.object(auth_manager.time_sync_manager, 'get_clock_skew', return_value=1000):
            skew = auth_manager.get_clock_skew()
            assert skew == 1000

    def test_is_time_synced(self, auth_manager):
        """Test time sync status."""
        with patch.object(auth_manager.time_sync_manager, 'is_time_synced', return_value=True):
            assert auth_manager.is_time_synced() is True
        
        with patch.object(auth_manager.time_sync_manager, 'is_time_synced', return_value=False):
            assert auth_manager.is_time_synced() is False

    @pytest.mark.asyncio
    async def test_get_subaccount_info(self, auth_manager, mock_auth_functions):
        """Test getting subaccount information."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        
        result = await auth_manager.get_subaccount_info("sub123")
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_list_subaccounts(self, auth_manager, mock_auth_functions):
        """Test listing subaccounts."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        
        result = await auth_manager.list_subaccounts()
        
        assert isinstance(result, list)

    def test_get_authentication_status(self, auth_manager):
        """Test getting authentication status."""
        auth_manager.is_authenticated = True
        auth_manager.current_key_id = "test_key_id"
        auth_manager.last_auth_check = datetime.now()
        
        with patch.object(auth_manager, 'get_current_permissions', return_value={APIKeyPermission.READ}):
            with patch.object(auth_manager.time_sync_manager, 'is_time_synced', return_value=True):
                with patch.object(auth_manager.time_sync_manager, 'get_clock_skew', return_value=1000):
                    status = auth_manager.get_authentication_status()
        
        assert status["is_authenticated"] is True
        assert status["current_key_id"] == "test_key_id"
        assert "permissions" in status
        assert "time_synced" in status
        assert "clock_skew" in status

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys(self, auth_manager):
        """Test cleaning up expired keys."""
        with patch.object(auth_manager.api_key_manager, 'cleanup_expired_keys', return_value=5):
            result = await auth_manager.cleanup_expired_keys()
            assert result == 5

    @pytest.mark.asyncio
    async def test_stop_auto_sync(self, auth_manager):
        """Test stopping auto sync."""
        with patch.object(auth_manager.time_sync_manager, 'stop_auto_sync', return_value=None):
            await auth_manager.stop_auto_sync()

    @pytest.mark.asyncio
    async def test_force_time_sync(self, auth_manager, mock_auth_functions):
        """Test forcing time sync."""
        auth_manager.register_auth_functions(**mock_auth_functions)
        
        with patch.object(auth_manager.time_sync_manager, 'force_sync', return_value=True):
            result = await auth_manager.force_time_sync()
            assert result is True

    def test_logout(self, auth_manager):
        """Test logout functionality."""
        auth_manager.is_authenticated = True
        auth_manager.current_key_id = "test_key"
        auth_manager.last_auth_check = datetime.now()
        
        with patch.object(auth_manager, 'stop_auto_sync', return_value=None):
            auth_manager.logout()
        
        assert not auth_manager.is_authenticated
        assert auth_manager.current_key_id is None
        assert auth_manager.last_auth_check is None

    @pytest.mark.asyncio
    async def test_close(self, auth_manager):
        """Test closing authentication manager."""
        with patch.object(auth_manager, 'stop_auto_sync', return_value=None):
            await auth_manager.close()