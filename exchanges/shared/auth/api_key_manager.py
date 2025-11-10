"""
API Key Management Utilities

Handles API key permissions, IP allowlist management, and key rotation.
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint


class APIKeyPermission(Enum):
    """API Key permission levels"""
    READ = "read"
    TRADE = "trade"
    WITHDRAW = "withdraw"
    ADMIN = "admin"


@dataclass
class APIKeyInfo:
    """API Key information structure"""
    key_id: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    permissions: Set[APIKeyPermission] = None
    ip_allowlist: Set[str] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()
        if self.ip_allowlist is None:
            self.ip_allowlist = set()
        if self.created_at is None:
            self.created_at = datetime.now()


class APIKeyManager:
    """
    Manages API keys, permissions, IP allowlists, and key rotation.
    """
    
    def __init__(self, exchange_name: str):
        tprint(f"Initializing APIKeyManager for exchange={exchange_name}", "INFO")
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"APIKeyManager.{exchange_name}")
        self.api_keys: Dict[str, APIKeyInfo] = {}
        self.rotation_schedule: Dict[str, datetime] = {}
        tprint(f"APIKeyManager initialized for {exchange_name}", "SUCCESS")
        
    def add_api_key(
        self,
        key_id: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        permissions: Optional[Set[APIKeyPermission]] = None,
        ip_allowlist: Optional[Set[str]] = None,
        expires_at: Optional[datetime] = None
    ) -> APIKeyInfo:
        """Add a new API key to the manager."""
        tprint(f"Adding API key key_id={key_id}, permissions={[p.value for p in (permissions or {APIKeyPermission.READ})]}", "INFO")
        if permissions is None:
            permissions = {APIKeyPermission.READ}
        if ip_allowlist is None:
            ip_allowlist = set()

        key_info = APIKeyInfo(
            key_id=key_id,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            permissions=permissions,
            ip_allowlist=ip_allowlist,
            expires_at=expires_at
        )

        self.api_keys[key_id] = key_info
        self.logger.info(f"Added API key {key_id} with permissions: {[p.value for p in permissions]}")
        tprint(f"API key {key_id} added successfully with {len(permissions)} permissions", "SUCCESS")

        return key_info
    
    def get_api_key(self, key_id: str) -> Optional[APIKeyInfo]:
        """Get API key information by ID."""
        tprint(f"Getting API key info for key_id={key_id}", "INFO")
        result = self.api_keys.get(key_id)
        if result:
            tprint(f"Found API key {key_id}", "SUCCESS")
        else:
            tprint(f"API key {key_id} not found", "WARNING")
        return result
    
    def get_active_api_keys(self) -> List[APIKeyInfo]:
        """Get all active API keys."""
        tprint(f"Getting all active API keys for {self.exchange_name}", "INFO")
        now = datetime.now()
        result = [
            key for key in self.api_keys.values()
            if key.is_active and (key.expires_at is None or key.expires_at > now)
        ]
        tprint(f"Found {len(result)} active API keys", "SUCCESS")
        return result
    
    def get_api_keys_with_permission(self, permission: APIKeyPermission) -> List[APIKeyInfo]:
        """Get all API keys with a specific permission."""
        return [
            key for key in self.get_active_api_keys()
            if permission in key.permissions
        ]
    
    def update_permissions(self, key_id: str, permissions: Set[APIKeyPermission]) -> bool:
        """Update API key permissions."""
        tprint(f"Updating permissions for API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to update permissions: API key {key_id} not found", "WARNING")
            return False

        self.api_keys[key_id].permissions = permissions
        self.logger.info(f"Updated permissions for API key {key_id}: {[p.value for p in permissions]}")
        tprint(f"Updated permissions for API key {key_id} successfully", "SUCCESS")
        return True
    
    def update_ip_allowlist(self, key_id: str, ip_allowlist: Set[str]) -> bool:
        """Update API key IP allowlist."""
        tprint(f"Updating IP allowlist for API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to update IP allowlist: API key {key_id} not found", "WARNING")
            return False

        self.api_keys[key_id].ip_allowlist = ip_allowlist
        self.logger.info(f"Updated IP allowlist for API key {key_id}: {ip_allowlist}")
        tprint(f"Updated IP allowlist for API key {key_id} with {len(ip_allowlist)} IPs", "SUCCESS")
        return True
    
    def add_ip_to_allowlist(self, key_id: str, ip_address: str) -> bool:
        """Add an IP address to the allowlist."""
        tprint(f"Adding IP {ip_address} to allowlist for API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to add IP: API key {key_id} not found", "WARNING")
            return False

        self.api_keys[key_id].ip_allowlist.add(ip_address)
        self.logger.info(f"Added IP {ip_address} to allowlist for API key {key_id}")
        tprint(f"Added IP {ip_address} to allowlist successfully", "SUCCESS")
        return True
    
    def remove_ip_from_allowlist(self, key_id: str, ip_address: str) -> bool:
        """Remove an IP address from the allowlist."""
        tprint(f"Removing IP {ip_address} from allowlist for API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to remove IP: API key {key_id} not found", "WARNING")
            return False

        self.api_keys[key_id].ip_allowlist.discard(ip_address)
        self.logger.info(f"Removed IP {ip_address} from allowlist for API key {key_id}")
        tprint(f"Removed IP {ip_address} from allowlist successfully", "SUCCESS")
        return True
    
    def validate_ip_access(self, key_id: str, ip_address: str) -> bool:
        """Validate if an IP address is allowed for the API key."""
        tprint(f"Validating IP access for {ip_address} with key {key_id}", "INFO")
        key_info = self.get_api_key(key_id)
        if not key_info:
            tprint(f"IP validation failed: API key {key_id} not found", "WARNING")
            return False

        # If no IP allowlist is set, allow all IPs
        if not key_info.ip_allowlist:
            tprint(f"IP {ip_address} allowed (no allowlist set)", "SUCCESS")
            return True

        is_allowed = ip_address in key_info.ip_allowlist
        if is_allowed:
            tprint(f"IP {ip_address} allowed (in allowlist)", "SUCCESS")
        else:
            tprint(f"IP {ip_address} not allowed (not in allowlist)", "WARNING")
        return is_allowed
    
    def deactivate_api_key(self, key_id: str) -> bool:
        """Deactivate an API key."""
        tprint(f"Deactivating API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to deactivate: API key {key_id} not found", "WARNING")
            return False

        self.api_keys[key_id].is_active = False
        self.logger.info(f"Deactivated API key {key_id}")
        tprint(f"API key {key_id} deactivated successfully", "SUCCESS")
        return True
    
    def activate_api_key(self, key_id: str) -> bool:
        """Activate an API key."""
        tprint(f"Activating API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to activate: API key {key_id} not found", "WARNING")
            return False

        self.api_keys[key_id].is_active = True
        self.logger.info(f"Activated API key {key_id}")
        tprint(f"API key {key_id} activated successfully", "SUCCESS")
        return True
    
    def schedule_key_rotation(self, key_id: str, rotation_date: datetime) -> bool:
        """Schedule key rotation for a specific date."""
        tprint(f"Scheduling key rotation for {key_id} on {rotation_date}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to schedule rotation: API key {key_id} not found", "WARNING")
            return False

        self.rotation_schedule[key_id] = rotation_date
        self.logger.info(f"Scheduled rotation for API key {key_id} on {rotation_date}")
        tprint(f"Key rotation scheduled successfully for {key_id}", "SUCCESS")
        return True
    
    def get_keys_needing_rotation(self) -> List[str]:
        """Get keys that need rotation."""
        tprint(f"Checking for keys needing rotation", "INFO")
        now = datetime.now()
        result = [
            key_id for key_id, rotation_date in self.rotation_schedule.items()
            if rotation_date <= now
        ]
        if result:
            tprint(f"Found {len(result)} keys needing rotation", "WARNING")
        else:
            tprint(f"No keys need rotation", "SUCCESS")
        return result
    
    def rotate_api_key(self, key_id: str, new_api_key: str, new_api_secret: str,
                      new_passphrase: Optional[str] = None) -> bool:
        """Rotate an API key with new credentials."""
        tprint(f"Rotating API key {key_id}", "INFO")
        if key_id not in self.api_keys:
            self.logger.warning(f"API key {key_id} not found")
            tprint(f"Failed to rotate: API key {key_id} not found", "WARNING")
            return False

        old_key = self.api_keys[key_id]

        # Create new key with same permissions and settings (don't log sensitive data)
        new_key_info = APIKeyInfo(
            key_id=f"{key_id}_rotated_{int(time.time())}",
            api_key=new_api_key,
            api_secret=new_api_secret,
            passphrase=new_passphrase or old_key.passphrase,
            permissions=old_key.permissions.copy(),
            ip_allowlist=old_key.ip_allowlist.copy(),
            created_at=datetime.now()
        )

        # Deactivate old key
        old_key.is_active = False

        # Add new key
        self.api_keys[new_key_info.key_id] = new_key_info

        # Remove from rotation schedule
        self.rotation_schedule.pop(key_id, None)

        self.logger.info(f"Rotated API key {key_id} to {new_key_info.key_id}")
        tprint(f"API key rotated successfully from {key_id} to {new_key_info.key_id}", "SUCCESS")
        return True
    
    def generate_signature(
        self,
        key_id: str,
        method: str,
        endpoint: str,
        body: str = "",
        timestamp: Optional[str] = None
    ) -> Optional[str]:
        """Generate API signature for a request."""
        tprint(f"Generating signature for {method} {endpoint} with key {key_id}", "INFO")
        key_info = self.get_api_key(key_id)
        if not key_info or not key_info.is_active:
            self.logger.warning(f"API key {key_id} not found or inactive")
            tprint(f"Failed to generate signature: API key {key_id} not found or inactive", "WARNING")
            return None

        if timestamp is None:
            timestamp = str(int(time.time() * 1000))

        # Update last used timestamp
        key_info.last_used = datetime.now()

        # Generate signature based on exchange
        if self.exchange_name.lower() == "okx":
            result = self._generate_okx_signature(key_info, timestamp, method, endpoint, body)
        elif self.exchange_name.lower() == "binance":
            result = self._generate_binance_signature(key_info, timestamp, method, endpoint, body)
        else:
            # Generic HMAC-SHA256 signature
            result = self._generate_generic_signature(key_info, timestamp, method, endpoint, body)

        if result:
            tprint(f"Signature generated successfully for {method} {endpoint}", "SUCCESS")
        else:
            tprint(f"Failed to generate signature for {method} {endpoint}", "ERROR")
        return result
    
    def _generate_okx_signature(
        self,
        key_info: APIKeyInfo,
        timestamp: str,
        method: str,
        endpoint: str,
        body: str
    ) -> str:
        """Generate OKX-specific signature."""
        import base64
        
        message = timestamp + method + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                key_info.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return signature
    
    def _generate_binance_signature(
        self,
        key_info: APIKeyInfo,
        timestamp: str,
        method: str,
        endpoint: str,
        body: str
    ) -> str:
        """Generate Binance-specific signature."""
        query_string = f"timestamp={timestamp}"
        if body:
            query_string += f"&{body}"
            
        signature = hmac.new(
            key_info.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _generate_generic_signature(
        self,
        key_info: APIKeyInfo,
        timestamp: str,
        method: str,
        endpoint: str,
        body: str
    ) -> str:
        """Generate generic HMAC-SHA256 signature."""
        message = f"{method}{endpoint}{timestamp}{body}"
        signature = hmac.new(
            key_info.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def get_headers(
        self,
        key_id: str,
        method: str,
        endpoint: str,
        body: str = "",
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, str]]:
        """Get complete headers for API request."""
        tprint(f"Getting headers for {method} {endpoint} with key {key_id}", "INFO")
        key_info = self.get_api_key(key_id)
        if not key_info or not key_info.is_active:
            tprint(f"Failed to get headers: API key {key_id} not found or inactive", "WARNING")
            return None

        timestamp = str(int(time.time() * 1000))
        signature = self.generate_signature(key_id, method, endpoint, body, timestamp)

        if not signature:
            tprint(f"Failed to get headers: signature generation failed", "ERROR")
            return None
            
        headers = {
            "Content-Type": "application/json",
            "X-Timestamp": timestamp
        }
        
        if additional_headers:
            headers.update(additional_headers)
            
        # Add exchange-specific headers
        if self.exchange_name.lower() == "okx":
            headers.update({
                "OK-ACCESS-KEY": key_info.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": key_info.passphrase or "",
            })
        elif self.exchange_name.lower() == "binance":
            headers.update({
                "X-MBX-APIKEY": key_info.api_key,
            })
        else:
            headers.update({
                "X-API-KEY": key_info.api_key,
                "X-SIGNATURE": signature,
            })

        tprint(f"Headers generated successfully for {method} {endpoint}", "SUCCESS")
        return headers
    
    def cleanup_expired_keys(self) -> int:
        """Remove expired API keys."""
        tprint(f"Cleaning up expired API keys for {self.exchange_name}", "INFO")
        now = datetime.now()
        expired_keys = [
            key_id for key_id, key_info in self.api_keys.items()
            if key_info.expires_at and key_info.expires_at <= now
        ]

        for key_id in expired_keys:
            del self.api_keys[key_id]
            self.rotation_schedule.pop(key_id, None)

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired API keys")
            tprint(f"Cleaned up {len(expired_keys)} expired API keys", "SUCCESS")
        else:
            tprint(f"No expired API keys to clean up", "INFO")

        return len(expired_keys)
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed API keys."""
        tprint(f"Calculating API key statistics for {self.exchange_name}", "INFO")
        active_keys = self.get_active_api_keys()
        total_keys = len(self.api_keys)

        permission_counts = {}
        for key in active_keys:
            for permission in key.permissions:
                permission_counts[permission.value] = permission_counts.get(permission.value, 0) + 1

        stats = {
            "total_keys": total_keys,
            "active_keys": len(active_keys),
            "inactive_keys": total_keys - len(active_keys),
            "permission_distribution": permission_counts,
            "keys_with_ip_restrictions": len([k for k in active_keys if k.ip_allowlist]),
            "scheduled_rotations": len(self.rotation_schedule)
        }
        tprint(f"API key statistics: total={total_keys}, active={len(active_keys)}", "SUCCESS")
        return stats