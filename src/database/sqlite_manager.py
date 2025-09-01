# src/database/sqlite_manager.py

from collections import defaultdict
from datetime import datetime
from src.utils.logger import system_logger
from typing import Any
import asyncio
import json
import os
import time
import sqlite3

from src.config.constants import *
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.warning_symbols import (
    connection_error,
    error,
    failed,
    initialization_error,
    invalid,
    missing,
)

class ConnectionPool:
    """Async connection pool for database operations."""

    def __init__(self, max_connections: int = 10, database_path: str = "data/ares.db"):
        self.max_connections = max_connections
        self.database_path = database_path
        self.connection_pool: asyncio.Queue | None = None
        self.active_connections: int = 0
        self.total_connections_created: int = 0
        self.connection_errors: int = 0

    @handle_errors(
        exceptions=(OSError, sqlite3.Error, asyncio.TimeoutError),
        default_return=None,
    )
    @handle_errors(
        exceptions=(OSError, sqlite3.Error, PermissionError),
        default_return=None,
    )
    async def _create_connection(self) -> sqlite3.Connection | None:
        """Create a new database connection."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row

        # Enable foreign keys
        connection.execute("PRAGMA foreign_keys = ON")

        # Set journal mode to WAL for better concurrency
        connection.execute("PRAGMA journal_mode = WAL")

        return connection

    @handle_errors(
        exceptions=(asyncio.QueueEmpty, asyncio.TimeoutError, OSError),
        default_return=None,
    )
    @handle_errors(
        exceptions=(asyncio.QueueFull, sqlite3.Error, OSError),
        default_return=None,
    )
    async def return_connection(self, connection: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        if connection and self.connection_pool:
            # Reset connection state
            connection.rollback()

            # Return to pool
            try:
                self.connection_pool.put_nowait(connection)
            except asyncio.QueueFull:
                # Close connection if pool is full
                connection.close()

            self.active_connections -= 1

class SQLiteManager:
    """
    Enhanced SQLite manager with comprehensive error handling = type safety,
    async optimization = connection pooling, and automatic recovery.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize SQLite manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SQLiteManager")

        # Database state
        self.connection: sqlite3.Connection | None = None
        self.is_connected: bool = False
        self.database_path: str | None = None

        # Configuration
        self.db_config: dict[str, Any] = self.config.get("sqlite_manager", {})
        self.db_path: str = self.db_config.get("database_path", "data/ares.db")
        self.auto_backup: bool = self.db_config.get("auto_backup", True)
        self.backup_interval: int = self.db_config.get(
            "backup_interval",
            3600,
        )  # 1 hour
        self.max_connections: int = self.db_config.get("max_connections", 10)

        # Connection pooling
        self.connection_pool: ConnectionPool | None = None

        # Automatic recovery
        self.recovery_attempts: int = 0
        self.max_recovery_attempts: int = self.db_config.get("max_recovery_attempts", 3)
        self.recovery_cooldown: int = self.db_config.get(
            "recovery_cooldown",
            60,
        )  # 1 minute
        self.last_recovery_attempt: float = 0

        # Performance monitoring
        self.operation_stats: dict[str, int] = defaultdict(int)
        self.error_stats: dict[str, int] = defaultdict(int)
        self.start_time: float = time.time()

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid SQLite manager configuration"),
            AttributeError: (False, "Missing required database parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="SQLite manager initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="database configuration loading",
    )
    async def _load_database_configuration(self) -> None:
        """Load database configuration."""
        try:
            # Import constants
            DEFAULT_BACKUP_INTERVAL = DEFAULT_DATABASE_PATH
            DEFAULT_MAX_CONNECTIONS = DEFAULT_MAX_RECOVERY_ATTEMPTS
            DEFAULT_RECOVERY_COOLDOWN = DEFAULT_MAX_RECOVERY_ATTEMPTS

            # Set default database parameters
            self.db_config.setdefault("database_path", DEFAULT_DATABASE_PATH)
            self.db_config.setdefault("auto_backup", True)
            self.db_config.setdefault("backup_interval", DEFAULT_BACKUP_INTERVAL)
            self.db_config.setdefault("max_connections", DEFAULT_MAX_CONNECTIONS)
            self.db_config.setdefault("enable_foreign_keys", True)
            self.db_config.setdefault("journal_mode", "WAL")
            self.db_config.setdefault(
                "max_recovery_attempts",
                DEFAULT_MAX_RECOVERY_ATTEMPTS,
            )
            self.db_config.setdefault("recovery_cooldown", DEFAULT_RECOVERY_COOLDOWN)

            # Update configuration
            self.db_path = self.db_config["database_path"]
            self.auto_backup = self.db_config["auto_backup"]
            self.backup_interval = self.db_config["backup_interval"]
            self.max_connections = self.db_config["max_connections"]
            self.max_recovery_attempts = self.db_config["max_recovery_attempts"]
            self.recovery_cooldown = self.db_config["recovery_cooldown"]

            self.logger.info("Database configuration loaded successfully")

        except (KeyError, TypeError) as e:
            self.print(
                error(f"Error loading database configuration - Invalid config: {e}"),
            )
        except Exception as e:
            self.print(
                error(f"Error loading database configuration - Unexpected error: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate database configuration.

        Returns:
            bool: True if configuration is valid = False otherwise
        """
        try:
            # Validate database path
            if not self.db_path:
                self.print(invalid("Invalid database path"))
                return False

            # Validate backup interval
            if self.backup_interval <= 0:
                self.print(invalid("Invalid backup interval"))
                return False

            # Validate max connections
            if self.max_connections <= 0:
                self.print(invalid("Invalid max connections"))
                return False

            # Validate recovery settings
            if self.max_recovery_attempts <= 0:
                self.print(invalid("Invalid max recovery attempts"))
                return False

            if self.recovery_cooldown <= 0:
                self.print(invalid("Invalid recovery cooldown"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except (ValueError, TypeError) as e:
            self.print(error(f"Error validating configuration - Invalid value: {e}"))
            return False
        except Exception as e:
            self.print(error(f"Error validating configuration - Unexpected error: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="connection pool initialization",
    )
    @handle_file_operations(
        default_return=False,
        context="database initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="table creation",
    )
    async def _create_tables(self, connection: sqlite3.Connection) -> None:
        """Create database tables with enhanced error handling."""
        try:
            # Create trades table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    price REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create positions table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create performance table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create settings table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create documents table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    collection TEXT NOT NULL,
                    key TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (collection, key)
                )
            """)

            self.logger.info("Database tables created successfully")

        except sqlite3.Error as e:
            self.print(error(f"Error creating tables - SQLite error: {e}"))
        except Exception as e:
            self.print(error(f"Error creating tables - Unexpected error: {e}"))

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade data"),
            AttributeError: (False, "Missing trade components"),
            KeyError: (False, "Missing required trade data"),
        },
        default_return=False,
        context="trade insertion",
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position data"),
            AttributeError: (False, "Missing position components"),
            KeyError: (False, "Missing required position data"),
        },
        default_return=False,
        context="position update",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trades getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="positions getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance getting",
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid performance data"),
            AttributeError: (False, "Missing performance components"),
            KeyError: (False, "Missing required performance data"),
        },
        default_return=False,
        context="performance insertion",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="setting getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="setting setting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="document setting",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="automatic recovery",
    )
    async def _attempt_recovery(self, operation: str) -> None:
        """Attempt automatic recovery for failed operations."""
        try:
            current_time = time.time()

            # Check if we can attempt recovery
            if (
                current_time - self.last_recovery_attempt < self.recovery_cooldown
                or self.recovery_attempts >= self.max_recovery_attempts
            ):
                return

            self.logger.info(f"🔄 Attempting recovery for operation: {operation}")

            # Attempt to reinitialize connection pool
            if self.connection_pool:
                await self.connection_pool.initialize()

            self.recovery_attempts += 1
            self.last_recovery_attempt = current_time

            self.logger.info(
                f"✅ Recovery attempt {self.recovery_attempts}/{self.max_recovery_attempts} completed",
            )

        except Exception:
            self.print(error("Error during recovery attempt: {e}"))

    async def _auto_backup_task(self) -> None:
        """Background task for automatic database backup."""
        while True:
            try:
                await asyncio.sleep(self.backup_interval)
                await self.create_backup()
            except Exception:
                self.print(error("Error in auto backup task: {e}"))
                await asyncio.sleep(self.backup_interval)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="closing database connection",
    )
    async def close(self) -> None:
        """Close database connections."""
        try:
            if self.connection_pool:
                # Close all connections in pool
                while not self.connection_pool.connection_pool.empty():
                    try:
                        connection = self.connection_pool.connection_pool.get_nowait()
                        connection.close()
                    except asyncio.QueueEmpty:
                        break

                self.connection_pool = None

            self.is_connected = False
            self.logger.info("Database connections closed successfully")

        except Exception:
            self.print(connection_error("Error closing database connections: {e}"))

    @handle_file_operations(
        default_return=False,
        context="database backup",
    )
    async def create_backup(self, backup_path: str | None = None) -> bool:
        """
        Create database backup with enhanced error handling.

        Args:
            backup_path: Optional backup path

        Returns:
            bool: True if backup successful = False otherwise
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path}.backup_{timestamp}"

            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.print(failed("Failed to get connection for backup"))
                return False

            try:
                # Create backup
                backup_connection = sqlite3.connect(backup_path)
                connection.backup(backup_connection)
                backup_connection.close()

                self.logger.info(f"Database backup created successfully: {backup_path}")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception:
            self.print(error("Error creating database backup: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SQLite manager cleanup",
    )
# Global SQLite manager instance
sqlite_manager: SQLiteManager | None = None

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="SQLite manager setup",
)