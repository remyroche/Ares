"""
Authentication Manager

Unified authentication management for exchange APIs.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable
from dataclasses import dataclass

from .api_key_manager import APIKeyManager, APIKeyPermission
from .time_sync import TimeSyncManager
from .subaccount_manager import SubaccountManager, SubaccountInfo

from src.utils.logger import system_logger


@dataclass
class AuthConfig:
    """Authentication configuration"""
    exchange_name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    subaccount_id: Optional[str] = None
    permissions: Set[APIKeyPermission] = None
    ip_allowlist: Set[str] = None
    auto_sync_time: bool = True
    max_clock_skew_ms: int = 5000
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {APIKeyPermission.READ}


class AuthenticationManager:
    """
    Unified authentication manager that coordinates API keys, time sync, and subaccounts.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"AuthManager.{exchange_name}")
        
        # Initialize managers
        self.api_key_manager = APIKeyManager(exchange_name)
        self.time_sync_manager = TimeSyncManager(exchange_name)
        self.subaccount_manager = SubaccountManager(exchange_name)
        
        # Current authentication state
        self.current_key_id: Optional[str] = None
        self.is_authenticated: bool = False
        self.last_auth_check: Optional[datetime] = None
        self.auth_functions: Dict[str, Callable] = {}
        
    def register_auth_functions(
        self,
        get_server_time: Callable[[], Awaitable[Optional[int]]],
        test_connection: Callable[[], Awaitable[bool]],
        get_account_info: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None
    ) -> None:
        """
        Register exchange-specific authentication functions.
        
        Args:
            get_server_time: Function to get server time in milliseconds
            test_connection: Function to test API connection
            get_account_info: Optional function to get account information
        """
        self.auth_functions = {
            "get_server_time": get_server_time,
            "test_connection": test_connection,
            "get_account_info": get_account_info
        }
        
        self.logger.info("Registered authentication functions")
    
    async def authenticate(self, config: AuthConfig) -> bool:
        """
        Authenticate with the exchange using the provided configuration.
        
        Args:
            config: Authentication configuration
            
        Returns:
            True if authentication successful
        """
        try:
            # Add API key to manager
            key_id = f"{self.exchange_name}_{int(datetime.now().timestamp())}"
            key_info = self.api_key_manager.add_api_key(
                key_id=key_id,
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.passphrase,
                permissions=config.permissions,
                ip_allowlist=config.ip_allowlist
            )
            
            self.current_key_id = key_id
            
            # Test connection
            if "test_connection" in self.auth_functions:
                connection_ok = await self.auth_functions["test_connection"]()
                if not connection_ok:
                    self.logger.error("Connection test failed")
                    return False
            
            # Sync time if enabled
            if config.auto_sync_time and "get_server_time" in self.auth_functions:
                await self.time_sync_manager.sync_time(self.auth_functions["get_server_time"])
                if config.auto_sync_time:
                    await self.time_sync_manager.start_auto_sync(self.auth_functions["get_server_time"])
            
            # Set subaccount if provided
            if config.subaccount_id:
                self.subaccount_manager.set_parent_account(config.subaccount_id)
            
            self.is_authenticated = True
            self.last_auth_check = datetime.now()
            
            self.logger.info(f"Successfully authenticated with {self.exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.is_authenticated = False
            return False
    
    async def reauthenticate(self) -> bool:
        """Re-authenticate using current configuration."""
        if not self.current_key_id:
            self.logger.error("No current key ID for re-authentication")
            return False
            
        key_info = self.api_key_manager.get_api_key(self.current_key_id)
        if not key_info:
            self.logger.error("Current API key not found")
            return False
            
        # Test connection
        if "test_connection" in self.auth_functions:
            connection_ok = await self.auth_functions["test_connection"]()
            if not connection_ok:
                self.logger.error("Re-authentication failed: connection test failed")
                return False
        
        # Sync time
        if "get_server_time" in self.auth_functions:
            await self.time_sync_manager.sync_time(self.auth_functions["get_server_time"])
        
        self.is_authenticated = True
        self.last_auth_check = datetime.now()
        
        self.logger.info("Successfully re-authenticated")
        return True
    
    def get_auth_headers(
        self,
        method: str,
        endpoint: str,
        body: str = "",
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, str]]:
        """
        Get authentication headers for API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            body: Request body
            additional_headers: Additional headers to include
            
        Returns:
            Headers dictionary if authenticated, None otherwise
        """
        if not self.is_authenticated or not self.current_key_id:
            # Only log at debug level for public data access
            self.logger.debug("Not authenticated, skipping auth headers (public data access)")
            return None
        
        try:
            # Get current API key info
            key_info = self.api_key_manager.get_api_key(self.current_key_id)
            if not key_info:
                self.logger.debug("API key not found (public data access)")
                return None
            
            # Generate exchange-specific headers
            headers = self._generate_exchange_headers(
                method, endpoint, body, key_info, additional_headers
            )
            
            return headers
            
        except Exception as e:
            # Only log as error if we actually have credentials (not public data access)
            if self.is_authenticated:
                self.logger.error(f"❌ Failed to generate auth headers: {e}")
            else:
                self.logger.debug(f"Auth headers not generated (public data access): {e}")
            return None
    
    def _generate_exchange_headers(
        self,
        method: str,
        endpoint: str,
        body: str,
        key_info: 'APIKeyInfo',
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Generate exchange-specific authentication headers."""
        headers = additional_headers or {}
        
        if self.exchange_name.lower() == "binance":
            return self._generate_binance_headers(method, endpoint, body, key_info, headers)
        elif self.exchange_name.lower() == "okx":
            return self._generate_okx_headers(method, endpoint, body, key_info, headers)
        elif self.exchange_name.lower() == "mexc":
            return self._generate_mexc_headers(method, endpoint, body, key_info, headers)
        elif self.exchange_name.lower() == "bingx":
            return self._generate_bingx_headers(method, endpoint, body, key_info, headers)
        else:
            # Default headers
            headers["X-API-Key"] = key_info.api_key
            return headers
    
    def _generate_binance_headers(
        self,
        method: str,
        endpoint: str,
        body: str,
        key_info: 'APIKeyInfo',
        headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate Binance-specific authentication headers."""
        import time
        import hmac
        import hashlib
        from urllib.parse import urlencode
        
        # Check if we have valid API credentials
        if not key_info or not key_info.api_secret or not key_info.api_key:
            self.logger.debug("No API credentials available for Binance (public data access)")
            return headers
        
        # Add timestamp
        timestamp = int(time.time() * 1000)
        
        # Create query string with timestamp
        query_string = f"timestamp={timestamp}"
        if body:
            query_string += f"&{body}"
        
        # Generate signature
        signature = hmac.new(
            key_info.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers.update({
            "X-MBX-APIKEY": key_info.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        })
        
        # Add signature to query string
        if "?" in endpoint:
            endpoint += f"&{query_string}&signature={signature}"
        else:
            endpoint += f"?{query_string}&signature={signature}"
        
        return headers
    
    def _generate_okx_headers(
        self,
        method: str,
        endpoint: str,
        body: str,
        key_info: 'APIKeyInfo',
        headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate OKX-specific authentication headers."""
        import time
        import hmac
        import hashlib
        import base64
        
        # OKX uses timestamp in ISO format
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime())
        
        # Create prehash string
        prehash_string = timestamp + method.upper() + endpoint + body
        
        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                key_info.api_secret.encode('utf-8'),
                prehash_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        headers.update({
            "OK-ACCESS-KEY": key_info.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": key_info.passphrase or "",
            "Content-Type": "application/json"
        })
        
        return headers
    
    def _generate_mexc_headers(
        self,
        method: str,
        endpoint: str,
        body: str,
        key_info: 'APIKeyInfo',
        headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate MEXC-specific authentication headers."""
        import time
        import hmac
        import hashlib
        from urllib.parse import urlencode
        
        # MEXC uses timestamp in milliseconds
        timestamp = int(time.time() * 1000)
        
        # Create query string
        query_string = f"timestamp={timestamp}"
        if body:
            query_string += f"&{body}"
        
        # Generate signature
        signature = hmac.new(
            key_info.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers.update({
            "X-MEXC-APIKEY": key_info.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        })
        
        # Add signature to query string
        if "?" in endpoint:
            endpoint += f"&{query_string}&signature={signature}"
        else:
            endpoint += f"?{query_string}&signature={signature}"
        
        return headers
    
    def _generate_bingx_headers(
        self,
        method: str,
        endpoint: str,
        body: str,
        key_info: 'APIKeyInfo',
        headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate BingX-specific authentication headers."""
        import time
        import hmac
        import hashlib
        import base64
        
        # BingX uses timestamp in milliseconds
        timestamp = int(time.time() * 1000)
        
        # Create query string
        query_string = f"timestamp={timestamp}"
        if body:
            query_string += f"&{body}"
        
        # Generate signature
        signature = hmac.new(
            key_info.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers.update({
            "X-BX-APIKEY": key_info.api_key,
            "Content-Type": "application/json"
        })
        
        # Add signature to query string
        if "?" in endpoint:
            endpoint += f"&{query_string}&signature={signature}"
        else:
            endpoint += f"?{query_string}&signature={signature}"
        
        return headers
            
        # Get timestamp adjusted for clock skew
        timestamp = self.time_sync_manager.get_adjusted_timestamp()
        
        # Generate headers using API key manager
        headers = self.api_key_manager.get_headers(
            self.current_key_id,
            method,
            endpoint,
            body,
            additional_headers
        )
        
        if headers:
            # Add timestamp header
            headers["X-Timestamp"] = str(timestamp)
            
        return headers
    
    def get_timestamp_for_request(self) -> int:
        """Get timestamp for API request (adjusted for clock skew)."""
        return self.time_sync_manager.get_adjusted_timestamp()
    
    def is_authenticated_and_valid(self) -> bool:
        """Check if authentication is valid and not expired."""
        if not self.is_authenticated:
            return False
            
        # Check if auth is recent (within 1 hour)
        if self.last_auth_check:
            time_since_check = datetime.now() - self.last_auth_check
            if time_since_check > timedelta(hours=1):
                return False
                
        # Check if time is synced
        if not self.time_sync_manager.is_time_synced():
            return False
            
        return True
    
    async def validate_authentication(self) -> bool:
        """Validate current authentication status."""
        if not self.is_authenticated_and_valid():
            self.logger.warning("Authentication invalid, attempting re-authentication")
            return await self.reauthenticate()
            
        return True
    
    def get_current_permissions(self) -> Set[APIKeyPermission]:
        """Get current API key permissions."""
        if not self.current_key_id:
            return set()
            
        key_info = self.api_key_manager.get_api_key(self.current_key_id)
        if not key_info:
            return set()
            
        return key_info.permissions
    
    def has_permission(self, permission: APIKeyPermission) -> bool:
        """Check if current API key has a specific permission."""
        return permission in self.get_current_permissions()
    
    def can_trade(self) -> bool:
        """Check if current API key can trade."""
        return self.has_permission(APIKeyPermission.TRADE)
    
    def can_withdraw(self) -> bool:
        """Check if current API key can withdraw."""
        return self.has_permission(APIKeyPermission.WITHDRAW)
    
    def can_read(self) -> bool:
        """Check if current API key can read data."""
        return self.has_permission(APIKeyPermission.READ)
    
    def get_clock_skew(self) -> Optional[int]:
        """Get current clock skew in milliseconds."""
        return self.time_sync_manager.get_clock_skew()
    
    def is_time_synced(self) -> bool:
        """Check if time is synchronized."""
        return self.time_sync_manager.is_time_synced()
    
    async def get_subaccount_info(self, subaccount_id: str) -> Optional[SubaccountInfo]:
        """Get subaccount information."""
        if "get_account_info" in self.auth_functions:
            return await self.subaccount_manager.get_subaccount_info(
                subaccount_id,
                self.auth_functions["get_account_info"]
            )
        return None
    
    async def list_subaccounts(self) -> List[SubaccountInfo]:
        """List all subaccounts."""
        if "get_account_info" in self.auth_functions:
            return await self.subaccount_manager.list_subaccounts(
                self.auth_functions["get_account_info"]
            )
        return []
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """Get comprehensive authentication status."""
        return {
            "is_authenticated": self.is_authenticated,
            "current_key_id": self.current_key_id,
            "last_auth_check": self.last_auth_check.isoformat() if self.last_auth_check else None,
            "permissions": [p.value for p in self.get_current_permissions()],
            "time_synced": self.is_time_synced(),
            "clock_skew": self.get_clock_skew(),
            "api_key_stats": self.api_key_manager.get_key_statistics(),
            "time_sync_stats": self.time_sync_manager.get_sync_statistics(),
            "subaccount_stats": self.subaccount_manager.get_subaccount_statistics()
        }
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys."""
        return self.api_key_manager.cleanup_expired_keys()
    
    async def stop_auto_sync(self) -> None:
        """Stop automatic time synchronization."""
        await self.time_sync_manager.stop_auto_sync()
    
    async def force_time_sync(self) -> bool:
        """Force immediate time synchronization."""
        if "get_server_time" in self.auth_functions:
            return await self.time_sync_manager.force_sync(self.auth_functions["get_server_time"])
        return False
    
    def logout(self) -> None:
        """Logout and clear authentication state."""
        self.is_authenticated = False
        self.current_key_id = None
        self.last_auth_check = None
        
        # Stop auto sync
        asyncio.create_task(self.stop_auto_sync())
        
        self.logger.info("Logged out from authentication manager")
    
    async def close(self) -> None:
        """Close authentication manager and cleanup resources."""
        await self.stop_auto_sync()
        self.logout()
        self.logger.info("Authentication manager closed")