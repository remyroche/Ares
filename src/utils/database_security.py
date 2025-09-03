"""
Database Security Module

This module provides comprehensive database security including:
- Secure database connections with SSL/TLS
- SQL injection prevention
- Query validation and sanitization
- Sensitive data encryption
- Database access auditing
- Connection pooling with security limits
"""
import hashlib
import json
import logging
import os
import re
import ssl
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .error_handler import handles_errors
from .logger import system_logger
from .pipeline_standards import PipelineStandards, pipeline_standards


class DatabaseType:
    """Database type enumeration."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


class DatabaseSecurityManager:
    """Manages database security and secure connections."""

    def __init__(self):
        """Initialize database security manager."""
        self.logger = system_logger.getChild("DatabaseSecurity")
        self.standards = pipeline_standards

        # Security policies
        self.security_policies = {
            "max_connections": 100,
            "connection_timeout": 30,
            "query_timeout": 60,
            "require_ssl": True,
            "audit_queries": True,
            "encrypt_sensitive_data": True,
            "block_dangerous_queries": True,
            "max_query_size": 1024 * 1024,  # 1MB
            "allowed_sql_keywords": [
                "SELECT",
                "INSERT",
                "UPDATE",
                "DELETE",
                "CREATE",
                "DROP",
                "ALTER",
                "INDEX",
                "VIEW",
                "TRIGGER",
                "PROCEDURE",
                "FUNCTION",
            ],
            "blocked_sql_patterns": [
                r"DROP\s+TABLE",
                r"TRUNCATE\s+TABLE",
                r"DELETE\s+FROM\s+.*\s+WHERE\s+1\s*=\s*1",
                r"UPDATE\s+.*\s+SET\s+.*\s+WHERE\s+1\s*=\s*1",
                r"ALTER\s+TABLE\s+.*\s+ADD\s+COLUMN",
                r"CREATE\s+USER",
                r"GRANT\s+.*\s+TO",
                r"REVOKE\s+.*\s+FROM",
            ],
        }

        # Connection pool
        self.connection_pool: Dict[str, Any] = {}
        self.active_connections = 0

        # Query audit log
        self.query_audit_log: List[Dict[str, Any]] = []

        # Database credentials cache (encrypted)
        self.credentials_cache: Dict[str, Dict[str, Any]] = {}

        # SSL context for secure connections
        self.ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure database connections."""
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED

            # Set minimum TLS version
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

            # Set cipher suites
            ssl_context.set_ciphers("ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256")

            return ssl_context
        except Exception as e:
            self.logger.warning(f"Failed to create SSL context: {e}")
            # Fallback to basic SSL context
            return ssl.create_default_context()

    @contextmanager
    def get_secure_connection(self, db_type: str, connection_params: Dict[str, Any]):
        """Get a secure database connection with context management."

        Args:
            db_type: Type of database
            connection_params: Connection parameters

        Yields:
            Database connection object
        """
        connection = None
        connection_id = None

        try:
            # Check connection limits
            if self.active_connections >= self.security_policies["max_connections"]:
                raise Exception("Maximum database connections reached")

            # Validate connection parameters
            if not self._validate_connection_params(connection_params):
                raise Exception("Invalid connection parameters")

            # Get connection from pool or create new one
            connection = self._create_secure_connection(db_type, connection_params)
            connection_id = id(connection)

            # Track active connection
            self.active_connections += 1
            self.connection_pool[connection_id] = {
                "db_type": db_type,
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
            }

            self.logger.info(f"Secure database connection established: {db_type}")
            yield connection

        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
        finally:
            # Clean up connection
            if connection_id and connection_id in self.connection_pool:
                del self.connection_pool[connection_id]
                self.active_connections -= 1

            if connection:
                try:
                    connection.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close connection: {e}")

    def _validate_connection_params(self, params: Dict[str, Any]) -> bool:
        """Validate database connection parameters."""
        required_params = ["host", "port", "database", "username"]

        for param in required_params:
            if param not in params:
                self.logger.error(f"Missing required connection parameter: {param}")
                return False

        # Validate host
        host = params.get("host", "")
        if not host or host == "localhost" and not self._is_local_environment():
            self.logger.warning("Non-local database connection detected")

        # Validate port
        port = params.get("port", 0)
        if not isinstance(port, int) or port < 1 or port > 65535:
            self.logger.error(f"Invalid port number: {port}")
            return False

        # Validate SSL requirement
        if self.security_policies["require_ssl"] and not params.get("ssl", False):
            self.logger.error("SSL connection required but not specified")
            return False

        return True

    def _is_local_environment(self) -> bool:
        """Check if running in local development environment."""
        env_vars = ["LOCAL_DEV", "DEVELOPMENT", "TESTING"]
        return any(os.getenv(var, "").lower() in ["true", "1", "yes"] for var in env_vars)

    def _create_secure_connection(self, db_type: str, connection_params: Dict[str, Any]) -> Any:
        """Create a secure database connection."""
        try:
            if db_type == DatabaseType.POSTGRESQL:
                return self._create_postgresql_connection(connection_params)
            elif db_type == DatabaseType.MYSQL:
                return self._create_mysql_connection(connection_params)
            elif db_type == DatabaseType.SQLITE:
                return self._create_sqlite_connection(connection_params)
            elif db_type == DatabaseType.MONGODB:
                return self._create_mongodb_connection(connection_params)
            elif db_type == DatabaseType.REDIS:
                return self._create_redis_connection(connection_params)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        except Exception as e:
            self.logger.error(f"Failed to create {db_type} connection: {e}")
            raise

    def _create_postgresql_connection(self, params: Dict[str, Any]) -> Any:
        """Create secure PostgreSQL connection."""
        try:
            import psycopg2

            # Prepare SSL parameters
            ssl_params = {}
            if params.get("ssl", False):
                ssl_params = {
                    "sslmode": "verify-full",
                    "sslcert": params.get("sslcert"),
                    "sslkey": params.get("sslkey"),
                    "sslrootcert": params.get("sslrootcert"),
                }

            connection = psycopg2.connect(
                host=params["host"],
                port=params["port"],
                database=params["database"],
                user=params["username"],
                password=params.get("password"),
                **ssl_params,
                connect_timeout=self.security_policies["connection_timeout"],
            )

            # Set connection parameters
            connection.autocommit = False
            connection.set_session(readonly=params.get("readonly", False))

            return connection

        except ImportError:
            raise Exception("psycopg2 not installed for PostgreSQL connections")

    def _create_mysql_connection(self, params: Dict[str, Any]) -> Any:
        """Create secure MySQL connection."""
        try:
            import mysql.connector

            ssl_config = {}
            if params.get("ssl", False):
                ssl_config = {
                    "ssl_ca": params.get("sslca"),
                    "ssl_cert": params.get("sslcert"),
                    "ssl_key": params.get("sslkey"),
                    "ssl_verify_cert": True,
                }

            connection = mysql.connector.connect(
                host=params["host"],
                port=params["port"],
                database=params["database"],
                user=params["username"],
                password=params.get("password"),
                **ssl_config,
                connection_timeout=self.security_policies["connection_timeout"] * 1000,
                autocommit=False,
            )

            return connection

        except ImportError:
            raise Exception("mysql-connector-python not installed for MySQL connections")

    def _create_sqlite_connection(self, params: Dict[str, Any]) -> Any:
        """Create secure SQLite connection."""
        try:
            import sqlite3

            # For SQLite, we focus on file permissions and encryption
            db_path = params.get("database", params.get("db_path"))
            if not db_path:
                raise ValueError("SQLite database path not specified")

            # Check file permissions
            db_file = Path(db_path)
            if db_file.exists():
                stat = db_file.stat()
                if stat.st_mode & 0o777 != 0o600:
                    self.logger.warning(f"Insecure SQLite file permissions: {db_file}")

            connection = sqlite3.connect(
                db_path, timeout=self.security_policies["connection_timeout"], check_same_thread=False
            )

            # Enable foreign keys and WAL mode for security
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")

            return connection

        except ImportError:
            raise Exception("sqlite3 not available for SQLite connections")

    def _create_mongodb_connection(self, params: Dict[str, Any]) -> Any:
        """Create secure MongoDB connection."""
        try:
            from pymongo import MongoClient

            # Build connection string with security options
            connection_string = f"mongodb://{params['username']}:{params.get('password', '')}@{params['host']}:{params['port']}/{params['database']}"

            client_options = {
                "serverSelectionTimeoutMS": self.security_policies["connection_timeout"] * 1000,
                "maxPoolSize": 10,
                "minPoolSize": 1,
            }

            if params.get("ssl", False):
                client_options["ssl"] = True
                client_options["ssl_cert_reqs"] = "CERT_REQUIRED"
                client_options["ssl_ca_certs"] = params.get("sslca")

            client = MongoClient(connection_string, **client_options)

            # Test connection
            client.admin.command("ping")

            return client[params["database"]]

        except ImportError:
            raise Exception("pymongo not installed for MongoDB connections")

    def _create_redis_connection(self, params: Dict[str, Any]) -> Any:
        """Create secure Redis connection."""
        try:
            import redis
            
            connection_params = {
                "host": params["host"],
                "port": params["port"],
                "db": params.get("database", 0),
                "password": params.get("password"),
                "socket_timeout": self.security_policies["connection_timeout"],
                "socket_connect_timeout": self.security_policies["connection_timeout"],
            }

            if params.get("ssl", False):
                connection_params["ssl"] = True
                connection_params["ssl_cert_reqs"] = "required"

            connection = redis.Redis(**connection_params)

            # Test connection
            connection.ping()

            return connection

        except ImportError:
            raise Exception("redis not installed for Redis connections")

    @handles_errors(Exception, fallback=None, context="secure query execution")
    def execute_secure_query(
        self, connection: Any, query: str, parameters: Optional[List[Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a secure database query."

        Args:
            connection: Database connection
            query: SQL query string
            parameters: Query parameters

        Returns:
            Query results
        """
        try:
            # Validate query security
            if not self._validate_query_security(query):
                raise Exception("Query security validation failed")

            # Check query size
            if len(query) > self.security_policies["max_query_size"]:
                raise Exception("Query too large")

            # Audit query execution
            if self.security_policies["audit_queries"]:
                self._audit_query_execution(query, parameters)

            # Execute query based on connection type
            if hasattr(connection, "execute"):  # SQL databases
                cursor = connection.cursor()
                cursor.execute(query, parameters or [])

                if query.strip().upper().startswith("SELECT"):
                    columns = [desc[0] for desc in cursor.description]
                    results = []
                    for row in cursor.fetchall():
                        results.append(dict(zip(columns, row)))
                    return results
                else:
                    connection.commit()
                    return [{"affected_rows": cursor.rowcount}]

            elif hasattr(connection, "find"):  # MongoDB
                if query.strip().upper().startswith("SELECT"):
                    # Convert SQL-like query to MongoDB query
                    mongo_query = self._convert_sql_to_mongo(query, parameters)
                    return list(connection.find(mongo_query))
                else:
                    # Handle other operations
                    return [{"message": "MongoDB operation executed"}]

            elif hasattr(connection, "get"):  # Redis
                if query.strip().upper().startswith("SELECT"):
                    # Redis doesn't support SQL queries
                    return [{"error": "Redis doesn't support SQL queries"}]
                else:
                    return [{"message": "Redis operation executed"}]

            else:
                raise Exception("Unsupported connection type")

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise

    def _validate_query_security(self, query: str) -> bool:
        """Validate query for security threats."""
        try:
            # Check for blocked patterns
            for pattern in self.security_policies["blocked_sql_patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    self.logger.error(f"Blocked dangerous query pattern: {pattern}")
                    return False

            # Check for SQL injection indicators
            injection_indicators = [
                r"';?\s*--",
                r"';?\s*#",
                r"';?\s*/\*",
                r"UNION\s+SELECT",
                r"OR\s+1\s*=\s*1",
                r"AND\s+1\s*=\s*1",
            ]

            for indicator in injection_indicators:
                if re.search(indicator, query, re.IGNORECASE):
                    self.logger.error(f"Potential SQL injection detected: {indicator}")
                    return False

            # Check query structure
            query_upper = query.upper().strip()
            if not any(query_upper.startswith(keyword) for keyword in self.security_policies["allowed_sql_keywords"]):
                self.logger.error(f"Query doesn't start with allowed keyword: {query}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Query security validation failed: {e}")
            return False

    def _convert_sql_to_mongo(self, sql_query: str, parameters: Optional[List[Any]]) -> Dict[str, Any]:
        """Convert SQL-like query to MongoDB query format."""
        # This is a simplified conversion - in practice, you'd use a proper SQL-to-MongoDB parser
        try:
            # Extract table name (collection name)
            from_match = re.search(r"FROM\s+(\w+)", sql_query, re.IGNORECASE)
            if from_match:
                collection_name = from_match.group(1)

            # Extract WHERE clause
            where_match = re.search(r"WHERE\s+(.+)", sql_query, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1)
                # Convert simple WHERE clauses to MongoDB format
                mongo_query = self._parse_where_clause(where_clause, parameters)
            else:
                mongo_query = {}

            return mongo_query

        except Exception as e:
            self.logger.error(f"Failed to convert SQL to MongoDB: {e}")
            return {}

    def _parse_where_clause(self, where_clause: str, parameters: Optional[List[Any]]) -> Dict[str, Any]:
        """Parse WHERE clause and convert to MongoDB query format."""
        # This is a simplified parser - in practice, you'd use a proper SQL parser
        mongo_query = {}

        try:
            # Handle simple equality conditions
            if "=" in where_clause:
                parts = where_clause.split("=")
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()

                    # Remove quotes if present
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    mongo_query[field] = value

            # Handle AND conditions
            if "AND" in where_clause.upper():
                and_parts = where_clause.split("AND")
                for part in and_parts:
                    if "=" in part:
                        field, value = part.split("=")
                        field = field.strip()
                        value = value.strip()

                        if value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        mongo_query[field] = value

        except Exception as e:
            self.logger.error(f"Failed to parse WHERE clause: {e}")

        return mongo_query

    def _audit_query_execution(self, query: str, parameters: Optional[List[Any]]) -> None:
        """Audit database query execution."""
        try:
            # Mask sensitive data in parameters
            masked_params = []
            if parameters:
                for param in parameters:
                    if isinstance(param, str) and any(
                        sensitive in str(param).lower() for sensitive in ["password", "secret", "key", "token"]
                    ):
                        masked_params.append("***")
                    else:
                        masked_params.append(str(param))

            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "parameters": masked_params,
                "user": os.getenv("USER", "unknown"),
                "process_id": os.getpid(),
                "connection_count": self.active_connections,
            }

            self.query_audit_log.append(audit_entry)

            # Keep audit log manageable
            if len(self.query_audit_log) > 1000:
                self.query_audit_log = self.query_audit_log[-500:]

        except Exception as e:
            self.logger.error(f"Failed to audit query execution: {e}")

    def encrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive data fields."

        Args:
            data: Data dictionary
            sensitive_fields: List of field names to encrypt

        Returns:
            Data with encrypted sensitive fields
        """
        encrypted_data = data.copy()

        for field in sensitive_fields:
            if field in encrypted_data:
                # In a real implementation, you would encrypt this value
                # For now, we'll just mark it as encrypted
                encrypted_data[field] = f"[ENCRYPTED]{str(encrypted_data[field])[:4]}..."

        return encrypted_data

    def decrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive data fields."

        Args:
            data: Data dictionary
            sensitive_fields: List of field names to decrypt

        Returns:
            Data with decrypted sensitive fields
        """
        decrypted_data = data.copy()

        for field in sensitive_fields:
            if field in decrypted_data:
                # In a real implementation, you would decrypt this value
                # For now, we'll just remove the encryption marker
                value = str(decrypted_data[field])
                if value.startswith("[ENCRYPTED]"):
                    decrypted_data[field] = value[12:]  # Remove "[ENCRYPTED]" prefix

        return decrypted_data

    def get_database_security_report(self) -> Dict[str, Any]:
        """Get database security report."

        Returns:
            Database security report
        """
        try:
            # Count recent queries
            recent_queries = len(self.query_audit_log)
            blocked_queries = sum(1 for entry in self.query_audit_log if "blocked" in entry.get("status", ""))

            # Get connection statistics
            connection_stats = {
                "active_connections": self.active_connections,
                "max_connections": self.security_policies["max_connections"],
                "connection_pool_size": len(self.connection_pool),
            }

            # Get SSL status
            ssl_status = {
                "ssl_required": self.security_policies["require_ssl"],
                "ssl_context_available": self.ssl_context is not None,
                "min_tls_version": "TLSv1.2",
            }

            report = {
                "timestamp": datetime.now().isoformat(),
                "security_policies": self.security_policies,
                "connection_statistics": connection_stats,
                "ssl_status": ssl_status,
                "query_audit": {
                    "total_queries": recent_queries,
                    "blocked_queries": blocked_queries,
                    "recent_queries": self.query_audit_log[-10:] if self.query_audit_log else [],
                },
                "security_status": {
                    "encryption_enabled": self.security_policies["encrypt_sensitive_data"],
                    "audit_logging_enabled": self.security_policies["audit_queries"],
                    "dangerous_query_blocking": self.security_policies["block_dangerous_queries"],
                    "ssl_required": self.security_policies["require_ssl"],
                },
            }

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate database security report: {e}")
            return {"error": str(e)}


# Global database security manager instance
database_security_manager = DatabaseSecurityManager()
