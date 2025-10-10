"""
Idempotency Management

Handles idempotency keys for order operations to prevent duplicate submissions.
"""

import hashlib
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from src.utils.logger import system_logger


@dataclass
class IdempotencyKey:
    """Idempotency key data structure"""
    key: str
    operation_type: str
    parameters_hash: str
    created_at: datetime
    expires_at: datetime
    result: Optional[Any] = None
    is_used: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IdempotencyManager:
    """
    Manages idempotency keys for order operations.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"IdempotencyManager.{exchange_name}")
        
        # Idempotency key storage
        self.idempotency_keys: Dict[str, IdempotencyKey] = {}
        
        # Settings
        self.default_ttl = timedelta(hours=24)  # 24 hours default TTL
        self.max_keys = 10000  # Maximum number of keys to store
        
    def generate_key(
        self,
        operation_type: str,
        parameters: Dict[str, Any],
        custom_key: Optional[str] = None,
        ttl: Optional[timedelta] = None
    ) -> str:
        """
        Generate an idempotency key for an operation.
        
        Args:
            operation_type: Type of operation (e.g., 'create_order', 'cancel_order')
            parameters: Operation parameters
            custom_key: Custom key (optional)
            ttl: Time to live for the key
            
        Returns:
            Generated idempotency key
        """
        if custom_key:
            key = f"{self.exchange_name}_{operation_type}_{custom_key}"
        else:
            # Generate key based on parameters
            params_str = self._serialize_parameters(parameters)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            timestamp = int(time.time())
            key = f"{self.exchange_name}_{operation_type}_{params_hash}_{timestamp}"
        
        # Create idempotency key record
        expires_at = datetime.now() + (ttl or self.default_ttl)
        idempotency_key = IdempotencyKey(
            key=key,
            operation_type=operation_type,
            parameters_hash=hashlib.sha256(self._serialize_parameters(parameters).encode()).hexdigest(),
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        # Store the key
        self.idempotency_keys[key] = idempotency_key
        
        # Cleanup if we have too many keys
        self._cleanup_if_needed()
        
        self.logger.debug(f"Generated idempotency key: {key}")
        return key
    
    def _serialize_parameters(self, parameters: Dict[str, Any]) -> str:
        """Serialize parameters for hashing."""
        # Sort parameters by key for consistent hashing
        sorted_params = sorted(parameters.items())
        
        # Convert to string representation
        param_parts = []
        for key, value in sorted_params:
            if isinstance(value, (dict, list)):
                # Convert complex types to JSON-like string
                import json
                value_str = json.dumps(value, sort_keys=True)
            else:
                value_str = str(value)
            param_parts.append(f"{key}={value_str}")
        
        return "&".join(param_parts)
    
    def check_key(self, key: str) -> Optional[IdempotencyKey]:
        """
        Check if an idempotency key exists and is valid.
        
        Args:
            key: Idempotency key to check
            
        Returns:
            IdempotencyKey if valid, None otherwise
        """
        if key not in self.idempotency_keys:
            return None
        
        idempotency_key = self.idempotency_keys[key]
        
        # Check if key has expired
        if datetime.now() > idempotency_key.expires_at:
            del self.idempotency_keys[key]
            self.logger.debug(f"Idempotency key {key} has expired")
            return None
        
        return idempotency_key
    
    def mark_key_as_used(self, key: str, result: Any = None) -> bool:
        """
        Mark an idempotency key as used.
        
        Args:
            key: Idempotency key to mark as used
            result: Result of the operation (optional)
            
        Returns:
            True if successful
        """
        if key not in self.idempotency_keys:
            return False
        
        idempotency_key = self.idempotency_keys[key]
        idempotency_key.is_used = True
        idempotency_key.result = result
        
        self.logger.debug(f"Marked idempotency key {key} as used")
        return True
    
    def get_key_result(self, key: str) -> Optional[Any]:
        """
        Get the result of a previously executed operation.
        
        Args:
            key: Idempotency key
            
        Returns:
            Result if key exists and is used, None otherwise
        """
        idempotency_key = self.check_key(key)
        if idempotency_key and idempotency_key.is_used:
            return idempotency_key.result
        
        return None
    
    def is_operation_duplicate(
        self,
        operation_type: str,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check if an operation is a duplicate based on parameters.
        
        Args:
            operation_type: Type of operation
            parameters: Operation parameters
            
        Returns:
            Existing idempotency key if duplicate, None otherwise
        """
        params_hash = hashlib.sha256(self._serialize_parameters(parameters).encode()).hexdigest()
        
        for key, idempotency_key in self.idempotency_keys.items():
            if (idempotency_key.operation_type == operation_type and
                idempotency_key.parameters_hash == params_hash and
                not idempotency_key.is_used and
                datetime.now() <= idempotency_key.expires_at):
                return key
        
        return None
    
    def create_order_key(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> str:
        """
        Create an idempotency key for order creation.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price
            client_order_id: Client order ID
            
        Returns:
            Idempotency key
        """
        parameters = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "client_order_id": client_order_id
        }
        
        return self.generate_key("create_order", parameters, custom_key=client_order_id)
    
    def create_cancel_key(self, order_id: str, symbol: str) -> str:
        """
        Create an idempotency key for order cancellation.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Idempotency key
        """
        parameters = {
            "order_id": order_id,
            "symbol": symbol
        }
        
        return self.generate_key("cancel_order", parameters, custom_key=f"cancel_{order_id}")
    
    def create_modify_key(
        self,
        order_id: str,
        symbol: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> str:
        """
        Create an idempotency key for order modification.
        
        Args:
            order_id: Order ID to modify
            symbol: Trading symbol
            new_quantity: New quantity
            new_price: New price
            
        Returns:
            Idempotency key
        """
        parameters = {
            "order_id": order_id,
            "symbol": symbol,
            "new_quantity": new_quantity,
            "new_price": new_price
        }
        
        return self.generate_key("modify_order", parameters, custom_key=f"modify_{order_id}")
    
    def validate_operation(
        self,
        operation_type: str,
        parameters: Dict[str, Any],
        key: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Validate an operation for idempotency.
        
        Args:
            operation_type: Type of operation
            parameters: Operation parameters
            key: Optional idempotency key
            
        Returns:
            (is_valid, existing_key, existing_result)
        """
        # Check if we have an existing key
        if key:
            existing_key = self.check_key(key)
            if existing_key:
                if existing_key.is_used:
                    return False, key, existing_key.result
                else:
                    return True, key, None
        
        # Check for duplicate operation
        duplicate_key = self.is_operation_duplicate(operation_type, parameters)
        if duplicate_key:
            existing_key = self.idempotency_keys[duplicate_key]
            if existing_key.is_used:
                return False, duplicate_key, existing_key.result
            else:
                return True, duplicate_key, None
        
        return True, None, None
    
    def execute_with_idempotency(
        self,
        operation_type: str,
        parameters: Dict[str, Any],
        operation_function,
        key: Optional[str] = None
    ) -> Any:
        """
        Execute an operation with idempotency protection.
        
        Args:
            operation_type: Type of operation
            parameters: Operation parameters
            operation_function: Function to execute
            key: Optional idempotency key
            
        Returns:
            Result of the operation
        """
        is_valid, existing_key, existing_result = self.validate_operation(
            operation_type, parameters, key
        )
        
        if not is_valid:
            if existing_result is not None:
                self.logger.info(f"Returning cached result for {operation_type}")
                return existing_result
            else:
                raise ValueError("Operation is not valid")
        
        if existing_key:
            # Use existing key
            idempotency_key = existing_key
        else:
            # Generate new key
            key = self.generate_key(operation_type, parameters, key)
            idempotency_key = self.idempotency_keys[key]
        
        try:
            # Execute the operation
            result = operation_function(**parameters)
            
            # Mark key as used with result
            self.mark_key_as_used(idempotency_key.key, result)
            
            return result
            
        except Exception as e:
            # Don't mark key as used if operation failed
            self.logger.error(f"Operation failed: {e}")
            raise
    
    def cleanup_expired_keys(self) -> int:
        """Clean up expired idempotency keys."""
        now = datetime.now()
        expired_keys = [
            key for key, idempotency_key in self.idempotency_keys.items()
            if now > idempotency_key.expires_at
        ]
        
        for key in expired_keys:
            del self.idempotency_keys[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired idempotency keys")
        
        return len(expired_keys)
    
    def _cleanup_if_needed(self) -> None:
        """Clean up old keys if we have too many."""
        if len(self.idempotency_keys) <= self.max_keys:
            return
        
        # Remove oldest keys
        sorted_keys = sorted(
            self.idempotency_keys.items(),
            key=lambda x: x[1].created_at
        )
        
        keys_to_remove = len(self.idempotency_keys) - self.max_keys
        for key, _ in sorted_keys[:keys_to_remove]:
            del self.idempotency_keys[key]
        
        self.logger.info(f"Cleaned up {keys_to_remove} old idempotency keys")
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """Get idempotency key statistics."""
        total_keys = len(self.idempotency_keys)
        used_keys = len([k for k in self.idempotency_keys.values() if k.is_used])
        unused_keys = total_keys - used_keys
        
        operation_counts = {}
        for idempotency_key in self.idempotency_keys.values():
            op_type = idempotency_key.operation_type
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        return {
            "total_keys": total_keys,
            "used_keys": used_keys,
            "unused_keys": unused_keys,
            "operation_distribution": operation_counts,
            "max_keys": self.max_keys
        }
    
    def get_key_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an idempotency key."""
        idempotency_key = self.check_key(key)
        if not idempotency_key:
            return None
        
        return {
            "key": idempotency_key.key,
            "operation_type": idempotency_key.operation_type,
            "parameters_hash": idempotency_key.parameters_hash,
            "created_at": idempotency_key.created_at.isoformat(),
            "expires_at": idempotency_key.expires_at.isoformat(),
            "is_used": idempotency_key.is_used,
            "has_result": idempotency_key.result is not None,
            "metadata": idempotency_key.metadata
        }
    
    def extend_key_ttl(self, key: str, additional_ttl: timedelta) -> bool:
        """Extend the TTL of an idempotency key."""
        if key not in self.idempotency_keys:
            return False
        
        idempotency_key = self.idempotency_keys[key]
        idempotency_key.expires_at += additional_ttl
        
        self.logger.debug(f"Extended TTL for key {key} by {additional_ttl}")
        return True
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an idempotency key."""
        if key not in self.idempotency_keys:
            return False
        
        del self.idempotency_keys[key]
        self.logger.debug(f"Revoked idempotency key {key}")
        return True