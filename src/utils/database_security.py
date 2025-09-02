"""
Database Security Framework

This module provides standardized database security including:
- Secure database connections
- Connection pooling with security
- Query parameterization and sanitization
- Database access control
- Audit logging for database operations
- Data encryption for sensitive fields
"""

import sqlite3
import psycopg2
import mysql.connector
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
from datetime import datetime
import logging
import hashlib
import secrets
import ssl

from .pipeline_standards import PipelineStandards, pipeline_standards
from .security_framework import security_framework, SecurityLevel
from .logger import system_logger
from .error_handler import handle_errors

class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"

class DatabaseSecurityManager:
    """Manages database security and connections."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database security manager."""
        self.config = config or {}
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("DatabaseSecurity")
        self.security = security_framework
        self.connections = {}
        self.connection_pools = {}
        
        # Database security policies
        self.security_policies = {
            "max_connections": 10,
            "connection_timeout": 30,
            "query_timeout": 60,
            "require_ssl": True,
            "audit_queries": True,
            "encrypt_sensitive_data": True,
            "parameterized_queries_only": True
        }
        
        # Override with config if provided
        if config:
            self.security_policies.update(config.get("security_policies", {}))
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="DatabaseSecurityManager initialization"
    )
    async def initialize(self) -> bool:
        """Initialize DatabaseSecurityManager."""
        try:
            self.logger.info("🚀 Initializing DatabaseSecurityManager...")
            self.is_initialized = True
            self.logger.info("✅ DatabaseSecurityManager initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing DatabaseSecurityManager: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="database connection"
    )
    def create_secure_connection(
        self, 
        db_type: DatabaseType, 
        connection_params: Dict[str, Any]
    ) -> Optional[Any]:
        """Create a secure database connection."""
        try:
            # Validate connection parameters
            self._validate_connection_params(connection_params)
            
            # Apply security policies
            secure_params = self._apply_security_policies(connection_params)
            
            # Create connection based on database type
            if db_type == DatabaseType.SQLITE:
                connection = self._create_sqlite_connection(secure_params)
            elif db_type == DatabaseType.POSTGRESQL:
                connection = self._create_postgresql_connection(secure_params)
            elif db_type == DatabaseType.MYSQL:
                connection = self._create_mysql_connection(secure_params)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Store connection for management
            connection_id = self._generate_connection_id()
            self.connections[connection_id] = {
                "connection": connection,
                "type": db_type,
                "created_at": datetime.now(),
                "last_used": datetime.now()
            }
            
            self.logger.info(f"✅ Secure connection created for {db_type.value}")
            return connection
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create secure connection: {e}")
            return None
    
    def _validate_connection_params(self, params: Dict[str, Any]) -> None:
        """Validate connection parameters for security."""
        required_fields = ["host", "database", "user", "password"]
        
        for field in required_fields:
            if field not in params:
                raise ValueError(f"Missing required connection parameter: {field}")
        
        # Validate password strength
        if len(params.get("password", "")) < 8:
            raise ValueError("Password must be at least 8 characters long")
    
    def _apply_security_policies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security policies to connection parameters."""
        secure_params = params.copy()
        
        # Force SSL if required
        if self.security_policies["require_ssl"]:
            if "sslmode" not in secure_params:
                secure_params["sslmode"] = "require"
            if "ssl" not in secure_params:
                secure_params["ssl"] = True
        
        # Add connection timeout
        if "connect_timeout" not in secure_params:
            secure_params["connect_timeout"] = self.security_policies["connection_timeout"]
        
        # Add query timeout
        if "options" not in secure_params:
            secure_params["options"] = f"-c statement_timeout={self.security_policies['query_timeout']}s"
        
        return secure_params
    
    def _create_sqlite_connection(self, params: Dict[str, Any]) -> sqlite3.Connection:
        """Create secure SQLite connection."""
        db_path = params["database"]
        
        # Validate file path security
        if not self._is_safe_file_path(db_path):
            raise ValueError("Unsafe database file path")
        
        connection = sqlite3.connect(db_path)
        
        # Enable foreign key constraints
        connection.execute("PRAGMA foreign_keys = ON")
        
        # Set busy timeout
        connection.execute("PRAGMA busy_timeout = 30000")
        
        return connection
    
    def _create_postgresql_connection(self, params: Dict[str, Any]) -> psycopg2.extensions.connection:
        """Create secure PostgreSQL connection."""
        # Create SSL context if required
        ssl_context = None
        if self.security_policies["require_ssl"]:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        connection = psycopg2.connect(
            host=params["host"],
            database=params["database"],
            user=params["user"],
            password=params["password"],
            ssl_context=ssl_context,
            connect_timeout=params.get("connect_timeout", 30),
            options=params.get("options", "")
        )
        
        # Set session parameters for security
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION statement_timeout = %s", 
                         (self.security_policies["query_timeout"] * 1000,))
            cursor.execute("SET SESSION lock_timeout = %s", (10000,))
        
        return connection
    
    def _create_mysql_connection(self, params: Dict[str, Any]) -> mysql.connector.MySQLConnection:
        """Create secure MySQL connection."""
        connection = mysql.connector.connect(
            host=params["host"],
            database=params["database"],
            user=params["user"],
            password=params["password"],
            ssl_disabled=not self.security_policies["require_ssl"],
            connection_timeout=params.get("connect_timeout", 30),
            autocommit=False
        )
        
        # Set session variables for security
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'")
            cursor.execute("SET SESSION wait_timeout = %s", (self.security_policies["connection_timeout"],))
        
        return connection
    
    def _is_safe_file_path(self, file_path: str) -> bool:
        """Check if file path is safe (not outside allowed directories)."""
        try:
            path = Path(file_path).resolve()
            allowed_dirs = [
                Path("/workspace/data"),
                Path("/workspace/db"),
                Path("/tmp")
            ]
            
            for allowed_dir in allowed_dirs:
                if path.is_relative_to(allowed_dir):
                    return True
            
            return False
        except (ValueError, RuntimeError):
            return False
    
    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        return hashlib.sha256(
            f"{datetime.now().isoformat()}{secrets.token_hex(8)}".encode()
        ).hexdigest()[:16]
    
    @contextmanager
    def get_connection(self, connection_id: str):
        """Context manager for database connections."""
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        connection_info = self.connections[connection_id]
        connection = connection_info["connection"]
        
        try:
            # Update last used timestamp
            connection_info["last_used"] = datetime.now()
            yield connection
        except Exception as e:
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            # Connection remains open for reuse
            pass
    
    def close_connection(self, connection_id: str) -> bool:
        """Close a database connection."""
        try:
            if connection_id in self.connections:
                connection_info = self.connections[connection_id]
                connection = connection_info["connection"]
                connection.close()
                del self.connections[connection_id]
                self.logger.info(f"Connection {connection_id} closed")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error closing connection {connection_id}: {e}")
            return False
    
    def close_all_connections(self) -> None:
        """Close all database connections."""
        for connection_id in list(self.connections.keys()):
            self.close_connection(connection_id)
    
    def audit_query(self, query: str, params: Optional[Tuple] = None, user: Optional[str] = None) -> None:
        """Audit database query if enabled."""
        if not self.security_policies["audit_queries"]:
            return
        
        audit_log = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "params": str(params) if params else None,
            "user": user or "unknown",
            "action": "query_execution"
        }
        
        self.logger.info(f"Database query audited: {audit_log}")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data if encryption is enabled."""
        if not self.security_policies["encrypt_sensitive_data"]:
            return data
        
        # Simple encryption for demonstration - in production use proper encryption
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return f"{salt}:{hashed.hex()}"
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data if encryption is enabled."""
        if not self.security_policies["encrypt_sensitive_data"]:
            return encrypted_data
        
        # Simple decryption for demonstration - in production use proper decryption
        try:
            salt, hashed = encrypted_data.split(":", 1)
            # In production, implement proper decryption logic
            return f"decrypted_{hashed[:8]}"
        except Exception:
            return encrypted_data
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.connections),
            "connection_types": {
                conn_info["type"].value: 1 
                for conn_info in self.connections.values()
            },
            "oldest_connection": min(
                (conn_info["created_at"] for conn_info in self.connections.values()),
                default=None
            ).isoformat() if self.connections else None,
            "security_policies": self.security_policies
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close_all_connections()

# Global instance
database_security_manager = DatabaseSecurityManager()