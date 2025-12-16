"""
Authentication manager for exchanges.

This module provides shared authentication utilities and configurations
for various cryptocurrency exchanges.
"""

from dataclasses import dataclass
from typing import Set, Optional, Dict, Any
from enum import Enum


class APIKeyPermission(Enum):
    """API key permissions for exchange access."""
    READ = "read"
    TRADE = "trade"
    WITHDRAW = "withdraw"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class AuthConfig:
    """Authentication configuration for exchange connections."""
    
    exchange_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # For exchanges like OKX
    subaccount_id: Optional[str] = None  # For exchanges like FTX
    permissions: Set[APIKeyPermission] = None
    auto_sync_time: bool = True
    testnet: bool = False
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {APIKeyPermission.READ}
        if self.additional_params is None:
            self.additional_params = {}
    
    def has_permission(self, permission: APIKeyPermission) -> bool:
        """Check if the auth config has a specific permission."""
        return permission in self.permissions
    
    def add_permission(self, permission: APIKeyPermission) -> None:
        """Add a permission to the auth config."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: APIKeyPermission) -> None:
        """Remove a permission from the auth config."""
        self.permissions.discard(permission)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert auth config to dictionary."""
        return {
            "exchange_name": self.exchange_name,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "passphrase": self.passphrase,
            "subaccount_id": self.subaccount_id,
            "permissions": [str(p) for p in self.permissions],
            "auto_sync_time": self.auto_sync_time,
            "testnet": self.testnet,
            "additional_params": self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthConfig":
        """Create auth config from dictionary."""
        permissions = {
            APIKeyPermission(p) if isinstance(p, str) else p 
            for p in data.get("permissions", [])
        }
        
        return cls(
            exchange_name=data["exchange_name"],
            api_key=data.get("api_key"),
            api_secret=data.get("api_secret"),
            passphrase=data.get("passphrase"),
            subaccount_id=data.get("subaccount_id"),
            permissions=permissions,
            auto_sync_time=data.get("auto_sync_time", True),
            testnet=data.get("testnet", False),
            additional_params=data.get("additional_params", {})
        )


class AuthManager:
    """Main authentication manager for handling multiple exchange configs."""
    
    def __init__(self):
        self._configs: Dict[str, AuthConfig] = {}
    
    def add_config(self, config: AuthConfig) -> None:
        """Add an authentication configuration."""
        self._configs[config.exchange_name] = config
    
    def get_config(self, exchange_name: str) -> Optional[AuthConfig]:
        """Get authentication configuration for an exchange."""
        return self._configs.get(exchange_name)
    
    def remove_config(self, exchange_name: str) -> bool:
        """Remove authentication configuration for an exchange."""
        if exchange_name in self._configs:
            del self._configs[exchange_name]
            return True
        return False
    
    def list_exchanges(self) -> list[str]:
        """List all configured exchanges."""
        return list(self._configs.keys())
    
    def validate_config(self, config: AuthConfig) -> bool:
        """Validate an authentication configuration."""
        if not config.exchange_name:
            return False
        
        # Check if at least one credential is provided for non-read permissions
        if any(p != APIKeyPermission.READ for p in config.permissions):
            if not config.api_key or not config.api_secret:
                return False
        
        return True


# Global auth manager instance
auth_manager = AuthManager()
