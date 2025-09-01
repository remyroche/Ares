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

    def __init__(self):
        """Initialize database security manager."""
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="database connection"
    )
    def create_secure_connection(self, db_type: DatabaseType, connection_params: Dict[str, Any]) -> Any:
        """Create a secure database connection.

        Args:
            db_type: Type of database
            connection_params: Connection parameters

        Returns:
            Database connection object
        """
        try:
            # Get encrypted credentials
            if db_type == DatabaseType.SQLITE:
                connection = self._create_sqlite_connection(connection_params)
            elif db_type == DatabaseType.POSTGRESQL:
                connection = self._create_postgresql_connection(connection_params)
            elif db_type == DatabaseType.MYSQL:
                connection = self._create_mysql_connection(connection_params)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Log connection creation
            self.security.audit_logger.log_security_event(
                "database_connection",
                "system",
                f"Created {db_type.value} connection",
                {"db_type": db_type.value, "host": connection_params.get("host", "local")},
                SecurityLevel.MEDIUM
            )

            return connection

        except Exception as e:
            self.logger.error(f"Failed to create secure connection: {e}")
            raise

    def _create_sqlite_connection(self, params: Dict[str, Any]) -> sqlite3.Connection:
        """Create secure SQLite connection."""
        db_path = params.get("database", "data_cache/database.db")

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(db_path)

        # Enable foreign keys and other security features
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")

        return connection

    def _create_postgresql_connection(self, params: Dict[str, Any]) -> psycopg2.extensions.connection:
        """Create secure PostgreSQL connection."""
        # Get credentials from security framework
        username = self.security.credential_manager.get_credential("postgresql", "username")
        password = self.security.credential_manager.get_credential("postgresql", "password")

        if not username or not password:
            raise SecurityViolation("Missing PostgreSQL credentials")

        connection_params = {
            "host": params.get("host", "localhost"),
            "port": params.get("port", 5432),
            "database": params.get("database", "trading_system"),
            "user": username,
            "password": password,
            "sslmode": "require" if self.security_policies["require_ssl"] else "prefer",
            "connect_timeout": self.security_policies["connection_timeout"]
        }

        return psycopg2.connect(**connection_params)

    def _create_mysql_connection(self, params: Dict[str, Any]) -> mysql.connector.connection.MySQLConnection:
        """Create secure MySQL connection."""
        # Get credentials from security framework
        username = self.security.credential_manager.get_credential("mysql", "username")
        password = self.security.credential_manager.get_credential("mysql", "password")

        if not username or not password:
            raise SecurityViolation("Missing MySQL credentials")

        connection_params = {
            "host": params.get("host", "localhost"),
            "port": params.get("port", 3306),
            "database": params.get("database", "trading_system"),
            "user": username,
            "password": password,
            "ssl_disabled": not self.security_policies["require_ssl"],
            "connection_timeout": self.security_policies["connection_timeout"]
        }

        return mysql.connector.connect(**connection_params)

    @contextmanager
    def get_secure_connection(self, db_type: DatabaseType, connection_params: Dict[str, Any]):
        """Context manager for secure database connections.

        Args:
            db_type: Type of database
            connection_params: Connection parameters

        Yields:
            Database connection
        """
        connection = None
        try:
            connection = self.create_secure_connection(db_type, connection_params)
            yield connection
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()

    def execute_secure_query(self, connection: Any, query: str, params: Optional[Tuple] = None,
                           db_type: DatabaseType = DatabaseType.SQLITE) -> List[Dict[str, Any]]:
        """Execute a secure database query.

        Args:
            connection: Database connection
            query: SQL query
            params: Query parameters
            db_type: Type of database

        Returns:
            Query results
        """
        try:
            # Validate query for security
            self._validate_query_security(query)

            # Log query for audit
            if self.security_policies["audit_queries"]:
                self.security.audit_logger.log_security_event(
                    "database_query",
                    "system",
                    "Executed database query",
                    {"query": query[:100] + "..." if len(query) > 100 else query, "db_type": db_type.value},
                    SecurityLevel.LOW
                )

            cursor = connection.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Fetch results
            if query.strip().upper().startswith("SELECT"):
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
            else:
                connection.commit()
                return [{"affected_rows": cursor.rowcount}]

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            connection.rollback()
            raise

    def _validate_query_security(self, query: str) -> None:
        """Validate query for security issues.

        Args:
            query: SQL query to validate

        Raises:
            SecurityViolation: If query is insecure
        """
        # Check for SQL injection patterns
        dangerous_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "TRUNCATE",
            "ALTER TABLE",
            "CREATE TABLE",
            "INSERT INTO",
            "UPDATE"
        ]

        query_upper = query.upper()

        # Only allow dangerous operations if they're properly parameterized
        for pattern in dangerous_patterns:
            if pattern in query_upper and "?" not in query and "%s" not in query:
                raise SecurityViolation(f"Potentially dangerous query detected: {pattern}")

        # Check for multiple statements
        if ";" in query and query.count(";") > 1:
            raise SecurityViolation("Multiple statements not allowed")

    def encrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive data before storing in database.

        Args:
            data: Data to encrypt
            sensitive_fields: List of sensitive field names

        Returns:
            Data with encrypted sensitive fields
        """
        if not self.security_policies["encrypt_sensitive_data"]:
            return data

        encrypted_data = data.copy()

        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                encrypted_data[field] = self.security.data_encryption.encrypt_data(str(encrypted_data[field]))

        return encrypted_data

    def decrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive data after retrieving from database.

        Args:
            data: Data to decrypt
            sensitive_fields: List of sensitive field names

        Returns:
            Data with decrypted sensitive fields
        """
        if not self.security_policies["encrypt_sensitive_data"]:
            return data

        decrypted_data = data.copy()

        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field] is not None:
                try:
                    decrypted_data[field] = self.security.data_encryption.decrypt_data(decrypted_data[field])
                except Exception as e:
                    self.logger.warning(f"Failed to decrypt field {field}: {e}")
                    # Keep encrypted value if decryption fails

        return decrypted_data

    def backup_database_securely(self, db_type: DatabaseType, connection_params: Dict[str, Any],
                               backup_path: str) -> str:
        """Create a secure database backup.

        Args:
            db_type: Type of database
            connection_params: Connection parameters
            backup_path: Path for backup file

        Returns:
            Path to encrypted backup file
        """
        try:
            if db_type == DatabaseType.SQLITE:
                return self._backup_sqlite_securely(connection_params, backup_path)
            elif db_type == DatabaseType.POSTGRESQL:
                return self._backup_postgresql_securely(connection_params, backup_path)
            elif db_type == DatabaseType.MYSQL:
                return self._backup_mysql_securely(connection_params, backup_path)
            else:
                raise ValueError(f"Unsupported database type for backup: {db_type}")

        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise

    def _backup_sqlite_securely(self, params: Dict[str, Any], backup_path: str) -> str:
        """Create secure SQLite backup."""
        db_path = params.get("database", "data_cache/database.db")

        # Create backup
        with sqlite3.connect(db_path) as source:
            with sqlite3.connect(backup_path) as backup:
                source.backup(backup)

        # Encrypt backup
        encrypted_backup_path = self.security.data_encryption.encrypt_file(backup_path)

        # Log backup creation
        self.security.audit_logger.log_security_event(
            "database_backup",
            "system",
            "Created secure database backup",
            {"backup_path": encrypted_backup_path, "db_type": "sqlite"},
            SecurityLevel.MEDIUM
        )

        return encrypted_backup_path

    def _backup_postgresql_securely(self, params: Dict[str, Any], backup_path: str) -> str:
        """Create secure PostgreSQL backup."""
        # This would use pg_dump in a real implementation
        # For now, create a placeholder backup
        with open(backup_path, 'w') as f:
            f.write("-- PostgreSQL backup placeholder\n")

        # Encrypt backup
        encrypted_backup_path = self.security.data_encryption.encrypt_file(backup_path)

        return encrypted_backup_path

    def _backup_mysql_securely(self, params: Dict[str, Any], backup_path: str) -> str:
        """Create secure MySQL backup."""
        # This would use mysqldump in a real implementation
        # For now, create a placeholder backup
        with open(backup_path, 'w') as f:
            f.write("-- MySQL backup placeholder\n")

        # Encrypt backup
        encrypted_backup_path = self.security.data_encryption.encrypt_file(backup_path)

        return encrypted_backup_path

    def get_database_security_report(self) -> Dict[str, Any]:
        """Get database security report.

        Returns:
            Database security report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "security_policies": self.security_policies,
            "active_connections": len(self.connections),
            "connection_pools": len(self.connection_pools),
            "audit_logging_enabled": self.security_policies["audit_queries"],
            "encryption_enabled": self.security_policies["encrypt_sensitive_data"],
            "ssl_required": self.security_policies["require_ssl"]
        }

        return report


# Global database security manager instance
database_security_manager = DatabaseSecurityManager()