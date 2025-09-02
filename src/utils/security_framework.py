"""
Comprehensive Security Framework

This module provides centralized security controls including:
- Credential management and encryption
- API key security
- Data encryption / decryption
- Access control and authentication
- Audit logging and monitoring
- Security validation and compliance
"""

import json
import hashlib
import hmac
import base64
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from cryptography.fernet import Fernet
import logging
from enum import Enum
import os
import asyncio
from dataclasses import dataclass, field

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors

class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityViolation(Exception):
    """Custom exception for security violations."""
    
    def __init__(self, message: str, level: SecurityLevel = SecurityLevel.HIGH, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.level = level
        self.details = details or {}
        self.timestamp = datetime.now()

@dataclass
class SecurityEvent:
    """Represents a security event for audit logging."""
    event_type: str
    severity: SecurityLevel
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

class CredentialManager:
    """Manages API credentials and sensitive data securely."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.credentials: Dict[str, Dict[str, Any]] = {}
        self.logger = system_logger.getChild("CredentialManager")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize CredentialManager."""
        try:
            self.logger.info("🚀 Initializing CredentialManager...")
            
            # Load existing credentials if available
            await self._load_credentials()
            
            self.is_initialized = True
            self.logger.info("✅ CredentialManager initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing CredentialManager: {e}")
            return False
    
    def _generate_encryption_key(self) -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return base64.b64encode(self.cipher_suite.encrypt(data.encode())).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            return self.cipher_suite.decrypt(decoded_data).decode()
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise SecurityViolation("Failed to decrypt sensitive data", SecurityLevel.CRITICAL)
    
    async def _load_credentials(self):
        """Load credentials from secure storage."""
        # This would typically load from encrypted files or secure key management systems
        # For now, we'll use an empty dict
        pass
    
    async def _save_credentials(self):
        """Save credentials to secure storage."""
        # This would typically save to encrypted files or secure key management systems
        pass
    
    def store_credential(self, name: str, credential_type: str, **kwargs) -> bool:
        """Store a new credential securely."""
        try:
            if not self.is_initialized:
                raise SecurityViolation("CredentialManager not initialized", SecurityLevel.CRITICAL)
            
            # Encrypt sensitive fields
            encrypted_data = {}
            for key, value in kwargs.items():
                if isinstance(value, str) and key in ['password', 'api_key', 'secret', 'token']:
                    encrypted_data[key] = self._encrypt_data(value)
                else:
                    encrypted_data[key] = value
            
            self.credentials[name] = {
                "type": credential_type,
                "created_at": datetime.now().isoformat(),
                "data": encrypted_data
            }
            
            self.logger.info(f"Stored credential: {name} ({credential_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store credential {name}: {e}")
            return False
    
    def get_credential(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored credential."""
        try:
            if not self.is_initialized:
                raise SecurityViolation("CredentialManager not initialized", SecurityLevel.CRITICAL)
            
            if name not in self.credentials:
                return None
            
            credential = self.credentials[name].copy()
            
            # Decrypt sensitive fields
            for key, value in credential["data"].items():
                if key in ['password', 'api_key', 'secret', 'token'] and isinstance(value, str):
                    try:
                        credential["data"][key] = self._decrypt_data(value)
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt {key} for credential {name}: {e}")
                        credential["data"][key] = None
            
            return credential
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential {name}: {e}")
            return None
    
    def remove_credential(self, name: str) -> bool:
        """Remove a stored credential."""
        try:
            if name in self.credentials:
                del self.credentials[name]
                self.logger.info(f"Removed credential: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove credential {name}: {e}")
            return False
    
    def list_credentials(self) -> List[str]:
        """List all stored credential names."""
        return list(self.credentials.keys())

class APIKeyManager:
    """Manages API keys and their security."""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
        self.logger = system_logger.getChild("APIKeyManager")
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
    
    def generate_api_key(self, name: str, permissions: List[str], 
                        rate_limit: Optional[int] = None) -> str:
        """Generate a new API key."""
        try:
            # Generate a secure random key
            api_key = secrets.token_urlsafe(32)
            
            # Store the key securely
            self.credential_manager.store_credential(
                name=f"api_key_{name}",
                credential_type="api_key",
                key=api_key,
                permissions=permissions,
                created_at=datetime.now().isoformat()
            )
            
            # Set rate limiting if specified
            if rate_limit:
                self.rate_limits[name] = {
                    "limit": rate_limit,
                    "window": 3600,  # 1 hour window
                    "requests": []
                }
            
            self.logger.info(f"Generated API key: {name}")
            return api_key
            
        except Exception as e:
            self.logger.error(f"Failed to generate API key {name}: {e}")
            raise SecurityViolation(f"Failed to generate API key: {e}", SecurityLevel.HIGH)
    
    def validate_api_key(self, api_key: str, required_permissions: Optional[List[str]] = None) -> bool:
        """Validate an API key and check permissions."""
        try:
            # Find the credential by API key
            for name in self.credential_manager.list_credentials():
                if name.startswith("api_key_"):
                    credential = self.credential_manager.get_credential(name)
                    if credential and credential["data"].get("key") == api_key:
                        # Check if key is still valid
                        created_at = datetime.fromisoformat(credential["data"]["created_at"])
                        if datetime.now() - created_at > timedelta(days=365):  # 1 year expiry
                            self.logger.warning(f"API key expired: {name}")
                            return False
                        
                        # Check permissions if required
                        if required_permissions:
                            key_permissions = credential["data"].get("permissions", [])
                            if not all(perm in key_permissions for perm in required_permissions):
                                self.logger.warning(f"Insufficient permissions for API key: {name}")
                                return False
                        
                        # Check rate limiting
                        if not self._check_rate_limit(name):
                            self.logger.warning(f"Rate limit exceeded for API key: {name}")
                            return False
                        
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating API key: {e}")
            return False
    
    def _check_rate_limit(self, key_name: str) -> bool:
        """Check if API key is within rate limits."""
        if key_name not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[key_name]
        now = time.time()
        
        # Remove old requests outside the window
        limit_info["requests"] = [req for req in limit_info["requests"] 
                                if now - req < limit_info["window"]]
        
        # Check if limit exceeded
        if len(limit_info["requests"]) >= limit_info["limit"]:
            return False
        
        # Add current request
        limit_info["requests"].append(now)
        return True
    
    def revoke_api_key(self, name: str) -> bool:
        """Revoke an API key."""
        try:
            return self.credential_manager.remove_credential(f"api_key_{name}")
        except Exception as e:
            self.logger.error(f"Failed to revoke API key {name}: {e}")
            return False

class DataEncryption:
    """Handles data encryption and decryption."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.logger = system_logger.getChild("DataEncryption")
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data."""
        try:
            if isinstance(data, str):
                data = data.encode()
            
            encrypted = self.cipher_suite.encrypt(data)
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}")
            raise SecurityViolation("Failed to encrypt data", SecurityLevel.HIGH)
    
    def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt data."""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            return self.cipher_suite.decrypt(decoded_data)
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise SecurityViolation("Failed to decrypt data", SecurityLevel.HIGH)
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file."""
        try:
            if output_path is None:
                output_path = f"{file_path}.encrypted"
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.encrypt_data(data)
            
            with open(output_path, 'w') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"Encrypted file: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt file {file_path}: {e}")
            raise SecurityViolation(f"Failed to encrypt file: {e}", SecurityLevel.HIGH)
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt a file."""
        try:
            if output_path is None:
                output_path = encrypted_file_path.replace('.encrypted', '.decrypted')
            
            with open(encrypted_file_path, 'r') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.decrypt_data(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"Decrypted file: {encrypted_file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt file {encrypted_file_path}: {e}")
            raise SecurityViolation(f"Failed to decrypt file: {e}", SecurityLevel.HIGH)

class AccessControl:
    """Manages access control and permissions."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, List[str]] = {}
        self.permissions: Dict[str, List[str]] = {}
        self.logger = system_logger.getChild("AccessControl")
    
    def add_user(self, user_id: str, username: str, role: str, 
                password_hash: str, **kwargs) -> bool:
        """Add a new user."""
        try:
            if user_id in self.users:
                self.logger.warning(f"User {user_id} already exists")
                return False
            
            self.users[user_id] = {
                "username": username,
                "role": role,
                "password_hash": password_hash,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "active": True,
                **kwargs
            }
            
            self.logger.info(f"Added user: {username} ({user_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add user {user_id}: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return user ID if successful."""
        try:
            # Find user by username
            user_id = None
            for uid, user_data in self.users.items():
                if user_data["username"] == username:
                    user_id = uid
                    break
            
            if not user_id:
                return None
            
            # Verify password hash
            stored_hash = self.users[user_id]["password_hash"]
            if self._verify_password(password, stored_hash):
                # Update last login
                self.users[user_id]["last_login"] = datetime.now().isoformat()
                self.logger.info(f"User authenticated: {username}")
                return user_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Authentication error for {username}: {e}")
            return None
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            # This is a simplified example - in production, use proper password hashing
            # like bcrypt or Argon2
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
        except Exception:
            return False
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has a specific permission."""
        try:
            if user_id not in self.users:
                return False
            
            user_role = self.users[user_id]["role"]
            if user_role not in self.roles:
                return False
            
            role_permissions = self.roles.get(user_role, [])
            return permission in role_permissions
            
        except Exception as e:
            self.logger.error(f"Error checking permission for user {user_id}: {e}")
            return False
    
    def add_role(self, role_name: str, permissions: List[str]) -> bool:
        """Add a new role with permissions."""
        try:
            self.roles[role_name] = permissions
            self.logger.info(f"Added role: {role_name} with {len(permissions)} permissions")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add role {role_name}: {e}")
            return False

class SecurityAuditor:
    """Handles security auditing and monitoring."""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.logger = system_logger.getChild("SecurityAuditor")
        self.alert_thresholds = {
            SecurityLevel.LOW: 100,      # Alert after 100 low-severity events
            SecurityLevel.MEDIUM: 50,    # Alert after 50 medium-severity events
            SecurityLevel.HIGH: 10,      # Alert after 10 high-severity events
            SecurityLevel.CRITICAL: 1    # Alert immediately for critical events
        }
    
    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        try:
            self.events.append(event)
            
            # Log to system logger
            log_level = logging.ERROR if event.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL] else logging.WARNING
            self.logger.log(log_level, f"Security event: {event.event_type} - {event.severity.value}")
            
            # Check if we need to raise an alert
            self._check_alert_thresholds(event.severity)
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    def _check_alert_thresholds(self, severity: SecurityLevel):
        """Check if we've exceeded alert thresholds."""
        try:
            threshold = self.alert_thresholds.get(severity, float('inf'))
            recent_events = [e for e in self.events 
                           if e.severity == severity and 
                           e.timestamp > datetime.now() - timedelta(hours=1)]
            
            if len(recent_events) >= threshold:
                self.logger.warning(f"Security alert threshold exceeded: {len(recent_events)} {severity.value} events in the last hour")
                
        except Exception as e:
            self.logger.error(f"Error checking alert thresholds: {e}")
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a security report for the specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]
            
            # Group events by severity
            by_severity = {}
            by_type = {}
            by_user = {}
            
            for event in recent_events:
                # Count by severity
                severity = event.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                # Count by event type
                event_type = event.event_type
                by_type[event_type] = by_type.get(event_type, 0) + 1
                
                # Count by user
                if event.user_id:
                    by_user[event.user_id] = by_user.get(event.user_id, 0) + 1
            
            return {
                "period_hours": hours,
                "total_events": len(recent_events),
                "by_severity": by_severity,
                "by_type": by_type,
                "by_user": by_user,
                "critical_events": len([e for e in recent_events if e.severity == SecurityLevel.CRITICAL]),
                "failed_events": len([e for e in recent_events if not e.success])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating security report: {e}")
            return {}
    
    def clear_old_events(self, older_than_days: int = 30):
        """Clear old security events."""
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            original_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp >= cutoff_date]
            removed_count = original_count - len(self.events)
            
            self.logger.info(f"Cleared {removed_count} old security events")
            
        except Exception as e:
            self.logger.error(f"Error clearing old events: {e}")

class SecurityFramework:
    """Main security framework that coordinates all security components."""
    
    def __init__(self):
        self.credential_manager = CredentialManager()
        self.api_key_manager = APIKeyManager(self.credential_manager)
        self.data_encryption = DataEncryption()
        self.access_control = AccessControl()
        self.auditor = SecurityAuditor()
        self.logger = system_logger.getChild("SecurityFramework")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the security framework."""
        try:
            self.logger.info("🚀 Initializing SecurityFramework...")
            
            # Initialize all components
            await self.credential_manager.initialize()
            
            # Setup default roles and permissions
            self._setup_default_security()
            
            self.is_initialized = True
            self.logger.info("✅ SecurityFramework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error initializing SecurityFramework: {e}")
            return False
    
    def _setup_default_security(self):
        """Setup default security roles and permissions."""
        # Default roles
        self.access_control.add_role("admin", ["read", "write", "delete", "admin"])
        self.access_control.add_role("user", ["read", "write"])
        self.access_control.add_role("viewer", ["read"])
        
        # Default admin user (password should be changed in production)
        admin_password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        self.access_control.add_user(
            user_id="admin_001",
            username="admin",
            role="admin",
            password_hash=admin_password_hash
        )
    
    def log_security_event(self, event_type: str, severity: SecurityLevel, 
                          user_id: Optional[str] = None, ip_address: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None, success: bool = True):
        """Log a security event through the auditor."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            success=success
        )
        self.auditor.log_event(event)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security framework status."""
        return {
            "initialized": self.is_initialized,
            "components": {
                "credential_manager": self.credential_manager.is_initialized,
                "access_control": len(self.access_control.users) > 0,
                "auditor": len(self.auditor.events) >= 0
            },
            "security_metrics": self.auditor.get_security_report(hours=24),
            "active_users": len([u for u in self.access_control.users.values() if u.get("active", True)]),
            "total_credentials": len(self.credential_manager.list_credentials())
        }

# Convenience functions
async def create_security_framework() -> SecurityFramework:
    """Create and initialize a security framework instance."""
    framework = SecurityFramework()
    await framework.initialize()
    return framework

def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)

# Example usage
if __name__ == "__main__":
    async def main():
        # Example of setting up and using the security framework
        framework = await create_security_framework()
        
        # Create a test user
        test_password_hash = hash_password("test123")
        framework.access_control.add_user(
            user_id="user_001",
            username="testuser",
            role="user",
            password_hash=test_password_hash
        )
        
        # Generate an API key
        api_key = framework.api_key_manager.generate_api_key(
            name="test_key",
            permissions=["read", "write"],
            rate_limit=100
        )
        
        # Log some security events
        framework.log_security_event(
            event_type="user_login",
            severity=SecurityLevel.LOW,
            user_id="user_001",
            success=True
        )
        
        # Get security status
        status = framework.get_security_status()
        print("Security Framework Status:")
        print(json.dumps(status, indent=2, default=str))
    
    asyncio.run(main())