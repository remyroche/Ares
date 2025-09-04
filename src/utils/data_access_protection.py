"""
Data Access Protection Framework

This module provides comprehensive data access protection including:
- Data access validation and authorization
- Data integrity verification
- Access logging and audit trails
- Data encryption and security measures
- Access rate limiting and throttling
- Data sanitization and privacy protection
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np

from src.core.decorators import handles_errors, validates, log_call, traced, authenticated, requires_permission
from .logger import system_logger
from .common_operations import (
    safe_file_exists, safe_json_dump, safe_json_load, validate_dataframe_schema,
    validate_data_quality, safe_copy, generate_hash
)

class AccessLevel(Enum):
    """Data access levels."""
    READ_ONLY = 'read_only'
    READ_WRITE = 'read_write'
    ADMIN = 'admin'
    SYSTEM = 'system'

class DataSensitivity(Enum):
    """Data sensitivity levels."""
    PUBLIC = 'public'
    INTERNAL = 'internal'
    CONFIDENTIAL = 'confidential'
    RESTRICTED = 'restricted'

class DataAccessProtection:
    """Comprehensive data access protection framework."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize data access protection framework."""
        self.config = config
        self.logger = system_logger.getChild('DataAccessProtection')
        
        # Access control configuration
        self.access_policies = config.get('access_policies', {})
        self.rate_limits = config.get('rate_limits', {})
        self.encryption_enabled = config.get('encryption_enabled', False)
        self.audit_logging = config.get('audit_logging', True)
        
        # Access tracking
        self.access_log: List[Dict[str, Any]] = []
        self.access_counts: Dict[str, int] = {}
        self.blocked_attempts: Dict[str, int] = {}
        
        # Data sensitivity mapping
        self.sensitivity_mapping = {
            'klines': DataSensitivity.PUBLIC,
            'features': DataSensitivity.INTERNAL,
            'labels': DataSensitivity.CONFIDENTIAL,
            'predictions': DataSensitivity.CONFIDENTIAL,
            'models': DataSensitivity.RESTRICTED,
            'config': DataSensitivity.INTERNAL
        }
        
        # Initialize security measures
        self._initialize_security_measures()

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    def _initialize_security_measures(self) -> bool:
        """Initialize security measures and validation."""
        self.logger.info("🔒 Initializing data access protection security measures...")
        
        try:
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid security configuration")
                return False
            
            # Initialize access policies
            self._initialize_access_policies()
            
            # Initialize rate limiting
            self._initialize_rate_limiting()
            
            # Initialize encryption if enabled
            if self.encryption_enabled:
                self._initialize_encryption()
            
            self.logger.info("✅ Data access protection security measures initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize security measures: {e}")
            return False

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    def _validate_configuration(self) -> bool:
        """Validate security configuration."""
        self.logger.info("🔍 Validating security configuration...")
        
        try:
            # Check required configuration keys
            required_keys = ['access_policies', 'rate_limits']
            for key in required_keys:
                if key not in self.config:
                    self.logger.error(f"❌ Missing required configuration key: {key}")
                    return False
            
            # Validate access policies
            access_policies = self.config.get('access_policies', {})
            if not isinstance(access_policies, dict):
                self.logger.error("❌ Access policies must be a dictionary")
                return False
            
            # Validate rate limits
            rate_limits = self.config.get('rate_limits', {})
            if not isinstance(rate_limits, dict):
                self.logger.error("❌ Rate limits must be a dictionary")
                return False
            
            self.logger.info("✅ Security configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Security configuration validation failed: {e}")
            return False

    @handles_errors(Exception, fallback=None, log_level="ERROR")
    @log_call
    @traced
    def _initialize_access_policies(self) -> None:
        """Initialize access policies."""
        self.logger.info("📋 Initializing access policies...")
        
        try:
            # Default access policies
            default_policies = {
                'public_data': {
                    'allowed_operations': ['read'],
                    'required_permissions': [],
                    'sensitivity_level': DataSensitivity.PUBLIC
                },
                'internal_data': {
                    'allowed_operations': ['read', 'write'],
                    'required_permissions': ['internal_access'],
                    'sensitivity_level': DataSensitivity.INTERNAL
                },
                'confidential_data': {
                    'allowed_operations': ['read', 'write'],
                    'required_permissions': ['confidential_access'],
                    'sensitivity_level': DataSensitivity.CONFIDENTIAL
                },
                'restricted_data': {
                    'allowed_operations': ['read'],
                    'required_permissions': ['admin_access'],
                    'sensitivity_level': DataSensitivity.RESTRICTED
                }
            }
            
            # Merge with user-provided policies
            self.access_policies = {**default_policies, **self.access_policies}
            
            self.logger.info(f"✅ Initialized {len(self.access_policies)} access policies")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize access policies: {e}")

    @handles_errors(Exception, fallback=None, log_level="ERROR")
    @log_call
    @traced
    def _initialize_rate_limiting(self) -> None:
        """Initialize rate limiting configuration."""
        self.logger.info("⏱️ Initializing rate limiting...")
        
        try:
            # Default rate limits
            default_limits = {
                'read_operations': {'max_requests': 1000, 'time_window': 3600},  # 1000 requests per hour
                'write_operations': {'max_requests': 100, 'time_window': 3600},   # 100 requests per hour
                'admin_operations': {'max_requests': 50, 'time_window': 3600}     # 50 requests per hour
            }
            
            # Merge with user-provided limits
            self.rate_limits = {**default_limits, **self.rate_limits}
            
            self.logger.info(f"✅ Initialized {len(self.rate_limits)} rate limits")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize rate limiting: {e}")

    @handles_errors(Exception, fallback=None, log_level="ERROR")
    @log_call
    @traced
    def _initialize_encryption(self) -> None:
        """Initialize encryption capabilities."""
        self.logger.info("🔐 Initializing encryption...")
        
        try:
            # This would integrate with actual encryption libraries
            # For now, we'll use basic hashing for demonstration
            self.encryption_key = self.config.get('encryption_key', 'default_key')
            self.encryption_algorithm = self.config.get('encryption_algorithm', 'sha256')
            
            self.logger.info("✅ Encryption initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize encryption: {e}")

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    @authenticated
    @requires_permission('data_access')
    def validate_data_access(
        self, 
        user_id: str, 
        data_path: str, 
        operation: str,
        data_type: str = 'klines'
    ) -> Dict[str, Any]:
        """Validate data access request with comprehensive security checks."""
        self.logger.info(f"🔍 Validating data access: {user_id} -> {data_path} ({operation})")
        
        try:
            # Initialize validation result
            validation_result = {
                'allowed': False,
                'reason': '',
                'access_level': None,
                'sensitivity_level': None,
                'rate_limit_status': 'unknown',
                'audit_info': {}
            }
            
            # Check if data path exists
            if not safe_file_exists(data_path):
                validation_result['reason'] = 'Data path does not exist'
                self._log_access_attempt(user_id, data_path, operation, False, 'Data path not found')
                return validation_result
            
            # Determine data sensitivity
            sensitivity_level = self.sensitivity_mapping.get(data_type, DataSensitivity.INTERNAL)
            validation_result['sensitivity_level'] = sensitivity_level.value
            
            # Check access permissions
            permission_check = self._check_access_permissions(user_id, sensitivity_level, operation)
            if not permission_check['allowed']:
                validation_result['reason'] = permission_check['reason']
                self._log_access_attempt(user_id, data_path, operation, False, permission_check['reason'])
                return validation_result
            
            # Check rate limiting
            rate_limit_check = self._check_rate_limits(user_id, operation)
            if not rate_limit_check['allowed']:
                validation_result['reason'] = rate_limit_check['reason']
                validation_result['rate_limit_status'] = 'exceeded'
                self._log_access_attempt(user_id, data_path, operation, False, rate_limit_check['reason'])
                return validation_result
            
            # Check data integrity
            integrity_check = self._check_data_integrity(data_path)
            if not integrity_check['valid']:
                validation_result['reason'] = f"Data integrity check failed: {integrity_check['reason']}"
                self._log_access_attempt(user_id, data_path, operation, False, validation_result['reason'])
                return validation_result
            
            # All checks passed
            validation_result['allowed'] = True
            validation_result['access_level'] = permission_check['access_level']
            validation_result['rate_limit_status'] = 'within_limits'
            
            # Log successful access
            self._log_access_attempt(user_id, data_path, operation, True, 'Access granted')
            
            self.logger.info(f"✅ Data access validated for {user_id}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Data access validation failed: {e}")
            validation_result['reason'] = f"Validation error: {e}"
            return validation_result

    @handles_errors(Exception, fallback={'allowed': False, 'reason': 'Permission check failed'}, log_level="ERROR")
    @log_call
    @traced
    def _check_access_permissions(
        self, 
        user_id: str, 
        sensitivity_level: DataSensitivity, 
        operation: str
    ) -> Dict[str, Any]:
        """Check user permissions for data access."""
        self.logger.info(f"🔐 Checking access permissions for {user_id}")
        
        try:
            # This would integrate with actual user management system
            # For now, we'll use a simple permission model
            
            # Get user permissions (this would come from user management system)
            user_permissions = self._get_user_permissions(user_id)
            
            # Check if user has required permissions for sensitivity level
            required_permissions = self._get_required_permissions(sensitivity_level, operation)
            
            has_permission = all(perm in user_permissions for perm in required_permissions)
            
            if not has_permission:
                return {
                    'allowed': False,
                    'reason': f'Insufficient permissions. Required: {required_permissions}, Has: {user_permissions}'
                }
            
            # Determine access level
            access_level = self._determine_access_level(user_permissions, sensitivity_level)
            
            return {
                'allowed': True,
                'access_level': access_level,
                'reason': 'Permission check passed'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Permission check failed: {e}")
            return {'allowed': False, 'reason': f'Permission check error: {e}'}

    @handles_errors(Exception, fallback={'allowed': False, 'reason': 'Rate limit check failed'}, log_level="ERROR")
    @log_call
    @traced
    def _check_rate_limits(self, user_id: str, operation: str) -> Dict[str, Any]:
        """Check rate limits for user operations."""
        self.logger.info(f"⏱️ Checking rate limits for {user_id} ({operation})")
        
        try:
            # Get rate limit configuration for operation
            rate_limit_config = self.rate_limits.get(operation, self.rate_limits.get('read_operations'))
            
            if not rate_limit_config:
                return {'allowed': True, 'reason': 'No rate limits configured'}
            
            # Check current request count
            current_time = time.time()
            time_window = rate_limit_config['time_window']
            max_requests = rate_limit_config['max_requests']
            
            # Get user's request history
            user_requests = self._get_user_request_history(user_id, current_time - time_window)
            
            if len(user_requests) >= max_requests:
                return {
                    'allowed': False,
                    'reason': f'Rate limit exceeded: {len(user_requests)}/{max_requests} requests in {time_window}s'
                }
            
            # Record this request
            self._record_user_request(user_id, current_time)
            
            return {
                'allowed': True,
                'reason': f'Rate limit check passed: {len(user_requests) + 1}/{max_requests} requests'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Rate limit check failed: {e}")
            return {'allowed': False, 'reason': f'Rate limit check error: {e}'}

    @handles_errors(Exception, fallback={'valid': False, 'reason': 'Integrity check failed'}, log_level="ERROR")
    @log_call
    @traced
    def _check_data_integrity(self, data_path: str) -> Dict[str, Any]:
        """Check data integrity and security."""
        self.logger.info(f"🔍 Checking data integrity for {data_path}")
        
        try:
            # Check file existence and permissions
            if not safe_file_exists(data_path):
                return {'valid': False, 'reason': 'File does not exist'}
            
            # Check file size (basic security check)
            file_size = Path(data_path).stat().st_size
            max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB default
            
            if file_size > max_file_size:
                return {'valid': False, 'reason': f'File too large: {file_size} > {max_file_size}'}
            
            # Check file extension
            file_extension = Path(data_path).suffix.lower()
            allowed_extensions = self.config.get('allowed_file_extensions', ['.parquet', '.csv', '.json'])
            
            if file_extension not in allowed_extensions:
                return {'valid': False, 'reason': f'File extension not allowed: {file_extension}'}
            
            # For parquet files, check basic structure
            if file_extension == '.parquet':
                try:
                    df = pd.read_parquet(data_path, nrows=1)  # Read only first row for structure check
                    if df.empty:
                        return {'valid': False, 'reason': 'Parquet file is empty'}
                except Exception as e:
                    return {'valid': False, 'reason': f'Parquet file corrupted: {e}'}
            
            return {'valid': True, 'reason': 'Integrity check passed'}
            
        except Exception as e:
            self.logger.error(f"❌ Data integrity check failed: {e}")
            return {'valid': False, 'reason': f'Integrity check error: {e}'}

    @handles_errors(Exception, fallback=[], log_level="ERROR")
    @log_call
    @traced
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (placeholder for actual user management integration)."""
        # This would integrate with actual user management system
        # For now, return default permissions based on user_id
        if user_id.startswith('admin_'):
            return ['admin_access', 'confidential_access', 'internal_access']
        elif user_id.startswith('user_'):
            return ['internal_access']
        else:
            return []

    @handles_errors(Exception, fallback=[], log_level="ERROR")
    @log_call
    @traced
    def _get_required_permissions(self, sensitivity_level: DataSensitivity, operation: str) -> List[str]:
        """Get required permissions for sensitivity level and operation."""
        # Map sensitivity levels to required permissions
        permission_map = {
            DataSensitivity.PUBLIC: [],
            DataSensitivity.INTERNAL: ['internal_access'],
            DataSensitivity.CONFIDENTIAL: ['confidential_access'],
            DataSensitivity.RESTRICTED: ['admin_access']
        }
        
        return permission_map.get(sensitivity_level, ['admin_access'])

    @handles_errors(Exception, fallback=AccessLevel.READ_ONLY, log_level="ERROR")
    @log_call
    @traced
    def _determine_access_level(self, user_permissions: List[str], sensitivity_level: DataSensitivity) -> AccessLevel:
        """Determine user's access level based on permissions and data sensitivity."""
        if 'admin_access' in user_permissions:
            return AccessLevel.ADMIN
        elif 'confidential_access' in user_permissions and sensitivity_level in [DataSensitivity.CONFIDENTIAL, DataSensitivity.INTERNAL]:
            return AccessLevel.READ_WRITE
        elif 'internal_access' in user_permissions and sensitivity_level == DataSensitivity.INTERNAL:
            return AccessLevel.READ_WRITE
        else:
            return AccessLevel.READ_ONLY

    @handles_errors(Exception, fallback=[], log_level="ERROR")
    @log_call
    @traced
    def _get_user_request_history(self, user_id: str, since_time: float) -> List[float]:
        """Get user's request history since specified time."""
        # This would integrate with actual request tracking system
        # For now, return empty list (no rate limiting)
        return []

    @handles_errors(Exception, fallback=None, log_level="ERROR")
    @log_call
    @traced
    def _record_user_request(self, user_id: str, timestamp: float) -> None:
        """Record user request for rate limiting."""
        # This would integrate with actual request tracking system
        # For now, do nothing
        pass

    @handles_errors(Exception, fallback=None, log_level="ERROR")
    @log_call
    @traced
    def _log_access_attempt(
        self, 
        user_id: str, 
        data_path: str, 
        operation: str, 
        success: bool, 
        reason: str
    ) -> None:
        """Log data access attempt for audit trail."""
        if not self.audit_logging:
            return
        
        try:
            access_record = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'data_path': data_path,
                'operation': operation,
                'success': success,
                'reason': reason,
                'ip_address': 'unknown',  # Would be populated from request context
                'user_agent': 'unknown'   # Would be populated from request context
            }
            
            self.access_log.append(access_record)
            
            # Keep only last 10000 records to prevent memory issues
            if len(self.access_log) > 10000:
                self.access_log = self.access_log[-10000:]
            
            # Log to system logger
            log_level = 'info' if success else 'warning'
            getattr(self.logger, log_level)(
                f"Data access: {user_id} -> {data_path} ({operation}) - {'SUCCESS' if success else 'FAILED'}: {reason}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log access attempt: {e}")

    @handles_errors(Exception, fallback=pd.DataFrame(), log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    def secure_data_read(
        self, 
        user_id: str, 
        data_path: str, 
        data_type: str = 'klines',
        **kwargs
    ) -> pd.DataFrame:
        """Securely read data with comprehensive access control."""
        self.logger.info(f"📖 Secure data read: {user_id} -> {data_path}")
        
        try:
            # Validate access
            access_validation = self.validate_data_access(user_id, data_path, 'read', data_type)
            if not access_validation['allowed']:
                raise PermissionError(f"Access denied: {access_validation['reason']}")
            
            # Read data based on file type
            file_extension = Path(data_path).suffix.lower()
            
            if file_extension == '.parquet':
                data = pd.read_parquet(data_path, **kwargs)
            elif file_extension == '.csv':
                data = pd.read_csv(data_path, **kwargs)
            elif file_extension == '.json':
                data = pd.read_json(data_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Validate data quality
            quality_report = validate_data_quality(data, max_nan_ratio=0.1, check_duplicates=True)
            if not quality_report['is_valid']:
                self.logger.warning(f"⚠️ Data quality issues detected: {quality_report['issues']}")
            
            # Apply data sanitization if needed
            if access_validation['sensitivity_level'] in ['confidential', 'restricted']:
                data = self._sanitize_data(data, access_validation['access_level'])
            
            self.logger.info(f"✅ Secure data read completed: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Secure data read failed: {e}")
            return pd.DataFrame()

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    def secure_data_write(
        self, 
        user_id: str, 
        data_path: str, 
        data: pd.DataFrame, 
        data_type: str = 'klines',
        **kwargs
    ) -> bool:
        """Securely write data with comprehensive access control."""
        self.logger.info(f"📝 Secure data write: {user_id} -> {data_path}")
        
        try:
            # Validate access
            access_validation = self.validate_data_access(user_id, data_path, 'write', data_type)
            if not access_validation['allowed']:
                raise PermissionError(f"Access denied: {access_validation['reason']}")
            
            # Validate data before writing
            if data is None or data.empty:
                raise ValueError("Cannot write empty or None data")
            
            # Validate data schema
            schema_valid, schema_errors = validate_dataframe_schema(data, ['timestamp'])
            if not schema_valid:
                raise ValueError(f"Data schema validation failed: {schema_errors}")
            
            # Create backup if file exists
            if safe_file_exists(data_path):
                backup_path = f"{data_path}.backup.{int(time.time())}"
                pd.read_parquet(data_path).to_parquet(backup_path)
                self.logger.info(f"📦 Created backup: {backup_path}")
            
            # Write data based on file type
            file_extension = Path(data_path).suffix.lower()
            
            if file_extension == '.parquet':
                data.to_parquet(data_path, **kwargs)
            elif file_extension == '.csv':
                data.to_csv(data_path, **kwargs)
            elif file_extension == '.json':
                data.to_json(data_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Verify write operation
            if not safe_file_exists(data_path):
                raise RuntimeError("Data write verification failed")
            
            self.logger.info(f"✅ Secure data write completed: {data.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Secure data write failed: {e}")
            return False

    @handles_errors(Exception, fallback=data, log_level="ERROR")
    @log_call
    @traced
    def _sanitize_data(self, data: pd.DataFrame, access_level: AccessLevel) -> pd.DataFrame:
        """Sanitize data based on access level."""
        self.logger.info(f"🧹 Sanitizing data for access level: {access_level.value}")
        
        try:
            sanitized_data = safe_copy(data, deep=True)
            
            if access_level == AccessLevel.READ_ONLY:
                # Remove sensitive columns for read-only access
                sensitive_columns = ['label', 'prediction', 'confidence', 'probability']
                columns_to_remove = [col for col in sensitive_columns if col in sanitized_data.columns]
                if columns_to_remove:
                    sanitized_data = sanitized_data.drop(columns=columns_to_remove)
                    self.logger.info(f"🔒 Removed sensitive columns: {columns_to_remove}")
            
            # Round numeric values to reduce precision for privacy
            numeric_columns = sanitized_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'timestamp':  # Don't round timestamps
                    sanitized_data[col] = sanitized_data[col].round(4)
            
            return sanitized_data
            
        except Exception as e:
            self.logger.error(f"❌ Data sanitization failed: {e}")
            return data

    @handles_errors(Exception, fallback={}, log_level="ERROR")
    @log_call
    @traced
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access statistics and audit information."""
        self.logger.info("📊 Generating access statistics...")
        
        try:
            # Calculate statistics from access log
            total_attempts = len(self.access_log)
            successful_attempts = sum(1 for record in self.access_log if record['success'])
            failed_attempts = total_attempts - successful_attempts
            
            # Group by operation type
            operation_counts = {}
            for record in self.access_log:
                operation = record['operation']
                operation_counts[operation] = operation_counts.get(operation, 0) + 1
            
            # Group by user
            user_counts = {}
            for record in self.access_log:
                user_id = record['user_id']
                user_counts[user_id] = user_counts.get(user_id, 0) + 1
            
            statistics = {
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'failed_attempts': failed_attempts,
                'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
                'operation_counts': operation_counts,
                'user_counts': user_counts,
                'blocked_attempts': self.blocked_attempts,
                'access_log_size': len(self.access_log)
            }
            
            self.logger.info(f"✅ Generated access statistics: {total_attempts} total attempts")
            return statistics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate access statistics: {e}")
            return {}

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    def export_audit_log(self, output_path: str) -> bool:
        """Export audit log to file."""
        self.logger.info(f"📤 Exporting audit log to: {output_path}")
        
        try:
            audit_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(self.access_log),
                'access_log': self.access_log,
                'statistics': self.get_access_statistics()
            }
            
            safe_json_dump(audit_data, output_path, indent=2)
            
            self.logger.info(f"✅ Audit log exported successfully: {len(self.access_log)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export audit log: {e}")
            return False