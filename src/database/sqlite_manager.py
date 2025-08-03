# src/database/sqlite_manager.py

import os
import sqlite3
import asyncio
import time
from datetime import datetime
from typing import Any, Optional
import json
from collections import defaultdict

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class ConnectionPool:
    """Async connection pool for database operations."""
    
    def __init__(self, max_connections: int = 10, database_path: str = "data/ares.db"):
        self.max_connections = max_connections
        self.database_path = database_path
        self.connection_pool: Optional[asyncio.Queue] = None
        self.active_connections: int = 0
        self.total_connections_created: int = 0
        self.connection_errors: int = 0
        
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            self.connection_pool = asyncio.Queue(maxsize=self.max_connections)
            
            # Pre-populate pool with connections
            for _ in range(self.max_connections):
                connection = await self._create_connection()
                if connection:
                    await self.connection_pool.put(connection)
                    self.total_connections_created += 1
                    
        except Exception as e:
            system_logger.error(f"Error initializing connection pool: {e}")
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection."""
        try:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            connection.execute("PRAGMA foreign_keys = ON")
            
            # Set journal mode to WAL for better concurrency
            connection.execute("PRAGMA journal_mode = WAL")
            
            return connection
            
        except Exception as e:
            self.connection_errors += 1
            system_logger.error(f"Error creating connection: {e}")
            return None
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def get_connection(self) -> Optional[sqlite3.Connection]:
        """Get a connection from the pool."""
        try:
            if not self.connection_pool:
                return None
                
            # Try to get connection from pool
            try:
                connection = self.connection_pool.get_nowait()
                self.active_connections += 1
                return connection
            except asyncio.QueueEmpty:
                # Create new connection if pool is empty and under limit
                if self.active_connections < self.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        self.active_connections += 1
                        self.total_connections_created += 1
                    return connection
                else:
                    # Wait for a connection to become available
                    connection = await self.connection_pool.get()
                    self.active_connections += 1
                    return connection
                    
        except Exception as e:
            system_logger.error(f"Error getting connection from pool: {e}")
            return None
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def return_connection(self, connection: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        try:
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
                
        except Exception as e:
            system_logger.error(f"Error returning connection to pool: {e}")
            if connection:
                connection.close()
    
    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "max_connections": self.max_connections,
            "active_connections": self.active_connections,
            "pool_size": self.connection_pool.qsize() if self.connection_pool else 0,
            "total_connections_created": self.total_connections_created,
            "connection_errors": self.connection_errors,
            "utilization_rate": self.active_connections / self.max_connections if self.max_connections > 0 else 0
        }


class SQLiteManager:
    """
    Enhanced SQLite manager with comprehensive error handling, type safety,
    async optimization, connection pooling, and automatic recovery.
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
        self.connection_pool: Optional[ConnectionPool] = None
        
        # Automatic recovery
        self.recovery_attempts: int = 0
        self.max_recovery_attempts: int = self.db_config.get("max_recovery_attempts", 3)
        self.recovery_cooldown: int = self.db_config.get("recovery_cooldown", 60)  # 1 minute
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
    async def initialize(self) -> bool:
        """
        Initialize SQLite manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing SQLite Manager...")

            # Load database configuration
            await self._load_database_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for SQLite manager")
                return False

            # Initialize connection pool
            await self._initialize_connection_pool()

            # Initialize database
            await self._initialize_database()

            # Start automatic backup task
            if self.auto_backup:
                asyncio.create_task(self._auto_backup_task())

            self.logger.info("✅ SQLite Manager initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ SQLite Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="database configuration loading",
    )
    async def _load_database_configuration(self) -> None:
        """Load database configuration."""
        try:
            # Set default database parameters
            self.db_config.setdefault("database_path", "data/ares.db")
            self.db_config.setdefault("auto_backup", True)
            self.db_config.setdefault("backup_interval", 3600)
            self.db_config.setdefault("max_connections", 10)
            self.db_config.setdefault("enable_foreign_keys", True)
            self.db_config.setdefault("journal_mode", "WAL")
            self.db_config.setdefault("max_recovery_attempts", 3)
            self.db_config.setdefault("recovery_cooldown", 60)

            # Update configuration
            self.db_path = self.db_config["database_path"]
            self.auto_backup = self.db_config["auto_backup"]
            self.backup_interval = self.db_config["backup_interval"]
            self.max_connections = self.db_config["max_connections"]
            self.max_recovery_attempts = self.db_config["max_recovery_attempts"]
            self.recovery_cooldown = self.db_config["recovery_cooldown"]

            self.logger.info("Database configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading database configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate database configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate database path
            if not self.db_path:
                self.logger.error("Invalid database path")
                return False

            # Validate backup interval
            if self.backup_interval <= 0:
                self.logger.error("Invalid backup interval")
                return False

            # Validate max connections
            if self.max_connections <= 0:
                self.logger.error("Invalid max connections")
                return False

            # Validate recovery settings
            if self.max_recovery_attempts <= 0:
                self.logger.error("Invalid max recovery attempts")
                return False

            if self.recovery_cooldown <= 0:
                self.logger.error("Invalid recovery cooldown")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="connection pool initialization",
    )
    async def _initialize_connection_pool(self) -> None:
        """Initialize connection pool."""
        try:
            self.connection_pool = ConnectionPool(
                max_connections=self.max_connections,
                database_path=self.db_path
            )
            await self.connection_pool.initialize()
            
            self.logger.info(f"Connection pool initialized with {self.max_connections} connections")
            
        except Exception as e:
            self.logger.error(f"Error initializing connection pool: {e}")

    @handle_file_operations(
        default_return=False,
        context="database initialization",
    )
    async def _initialize_database(self) -> bool:
        """
        Initialize database with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection from pool")
                return False

            try:
                # Enable foreign keys
                connection.execute("PRAGMA foreign_keys = ON")

                # Set journal mode to WAL for better concurrency
                connection.execute("PRAGMA journal_mode = WAL")

                # Create tables
                await self._create_tables(connection)

                # Commit changes
                connection.commit()

                self.is_connected = True
                self.database_path = self.db_path

                self.logger.info("Database initialized successfully")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            return False

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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open'
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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'open'
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

        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade data"),
            AttributeError: (False, "Missing trade components"),
            KeyError: (False, "Missing required trade data"),
        },
        default_return=False,
        context="trade insertion",
    )
    async def insert_trade(self, trade_data: dict[str, Any]) -> bool:
        """
        Insert trade data with enhanced error handling and connection pooling.

        Args:
            trade_data: Trade data dictionary

        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            # Validate trade data
            required_fields = ["symbol", "side", "size", "price"]
            for field in required_fields:
                if field not in trade_data:
                    self.logger.error(f"Missing required trade field: {field}")
                    return False

            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for trade insertion")
                return False

            try:
                # Insert trade
                cursor = connection.execute("""
                    INSERT INTO trades (symbol, side, size, price, pnl, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trade_data["symbol"],
                    trade_data["side"],
                    trade_data["size"],
                    trade_data["price"],
                    trade_data.get("pnl", 0),
                    trade_data.get("status", "open")
                ))

                connection.commit()
                
                # Update operation stats
                self.operation_stats["trades_inserted"] += 1
                
                self.logger.info(f"Trade inserted successfully: {trade_data['symbol']}")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["trade_insertion_errors"] += 1
            self.logger.error(f"Error inserting trade: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("trade_insertion")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position data"),
            AttributeError: (False, "Missing position components"),
            KeyError: (False, "Missing required position data"),
        },
        default_return=False,
        context="position update",
    )
    async def update_position(self, position_data: dict[str, Any]) -> bool:
        """
        Update position data with enhanced error handling and connection pooling.

        Args:
            position_data: Position data dictionary

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Validate position data
            required_fields = ["symbol", "size", "entry_price", "current_price"]
            for field in required_fields:
                if field not in position_data:
                    self.logger.error(f"Missing required position field: {field}")
                    return False

            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for position update")
                return False

            try:
                # Calculate PnL
                pnl = (position_data["current_price"] - position_data["entry_price"]) * position_data["size"]

                # Update or insert position
                cursor = connection.execute("""
                    INSERT OR REPLACE INTO positions (symbol, size, entry_price, current_price, pnl, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    position_data["symbol"],
                    position_data["size"],
                    position_data["entry_price"],
                    position_data["current_price"],
                    pnl,
                    position_data.get("status", "open")
                ))

                connection.commit()
                
                # Update operation stats
                self.operation_stats["positions_updated"] += 1
                
                self.logger.info(f"Position updated successfully: {position_data['symbol']}")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["position_update_errors"] += 1
            self.logger.error(f"Error updating position: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("position_update")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trades getting",
    )
    async def get_trades(
        self,
        symbol: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get trades with enhanced error handling and connection pooling.

        Args:
            symbol: Optional symbol filter
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: List of trade records
        """
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for trades retrieval")
                return []

            try:
                # Build query
                query = "SELECT * FROM trades"
                params = []

                if symbol:
                    query += " WHERE symbol = ?"
                    params.append(symbol)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += f" LIMIT {limit}"

                # Execute query
                cursor = connection.execute(query, params)
                trades = [dict(row) for row in cursor.fetchall()]
                
                # Update operation stats
                self.operation_stats["trades_retrieved"] += 1
                
                return trades

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["trades_retrieval_errors"] += 1
            self.logger.error(f"Error getting trades: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("trades_retrieval")
            return []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="positions getting",
    )
    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Get positions with enhanced error handling and connection pooling.

        Returns:
            List[Dict[str, Any]]: List of position records
        """
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for positions retrieval")
                return []

            try:
                # Execute query
                cursor = connection.execute("SELECT * FROM positions WHERE status = 'open'")
                positions = [dict(row) for row in cursor.fetchall()]
                
                # Update operation stats
                self.operation_stats["positions_retrieved"] += 1
                
                return positions

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["positions_retrieval_errors"] += 1
            self.logger.error(f"Error getting positions: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("positions_retrieval")
            return []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance getting",
    )
    async def get_performance(self, days: int | None = None) -> list[dict[str, Any]]:
        """
        Get performance data with enhanced error handling and connection pooling.

        Args:
            days: Optional number of days to look back

        Returns:
            List[Dict[str, Any]]: List of performance records
        """
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for performance retrieval")
                return []

            try:
                # Build query
                query = "SELECT * FROM performance"
                params = []

                if days:
                    query += " WHERE timestamp >= datetime('now', '-{} days')".format(days)

                query += " ORDER BY timestamp DESC"

                # Execute query
                cursor = connection.execute(query, params)
                performance = [dict(row) for row in cursor.fetchall()]
                
                # Update operation stats
                self.operation_stats["performance_retrieved"] += 1
                
                return performance

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["performance_retrieval_errors"] += 1
            self.logger.error(f"Error getting performance: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("performance_retrieval")
            return []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid performance data"),
            AttributeError: (False, "Missing performance components"),
            KeyError: (False, "Missing required performance data"),
        },
        default_return=False,
        context="performance insertion",
    )
    async def insert_performance(self, performance_data: dict[str, Any]) -> bool:
        """
        Insert performance data with enhanced error handling and connection pooling.

        Args:
            performance_data: Performance data dictionary

        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            # Validate performance data
            required_fields = ["total_pnl", "win_rate", "sharpe_ratio", "max_drawdown"]
            for field in required_fields:
                if field not in performance_data:
                    self.logger.error(f"Missing required performance field: {field}")
                    return False

            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for performance insertion")
                return False

            try:
                # Insert performance
                cursor = connection.execute("""
                    INSERT INTO performance (total_pnl, win_rate, sharpe_ratio, max_drawdown)
                    VALUES (?, ?, ?, ?)
                """, (
                    performance_data["total_pnl"],
                    performance_data["win_rate"],
                    performance_data["sharpe_ratio"],
                    performance_data["max_drawdown"]
                ))

                connection.commit()
                
                # Update operation stats
                self.operation_stats["performance_inserted"] += 1
                
                self.logger.info("Performance data inserted successfully")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["performance_insertion_errors"] += 1
            self.logger.error(f"Error inserting performance: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("performance_insertion")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="setting getting",
    )
    async def get_setting(self, key: str) -> str | None:
        """
        Get setting with enhanced error handling and connection pooling.

        Args:
            key: Setting key

        Returns:
            Optional[str]: Setting value
        """
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for setting retrieval")
                return None

            try:
                # Execute query
                cursor = connection.execute("SELECT value FROM settings WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                # Update operation stats
                self.operation_stats["settings_retrieved"] += 1
                
                return result["value"] if result else None

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["settings_retrieval_errors"] += 1
            self.logger.error(f"Error getting setting: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("settings_retrieval")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="setting setting",
    )
    async def set_setting(self, key: str, value: str) -> bool:
        """
        Set setting with enhanced error handling and connection pooling.

        Args:
            key: Setting key
            value: Setting value

        Returns:
            bool: True if setting successful, False otherwise
        """
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for setting update")
                return False

            try:
                # Insert or update setting
                cursor = connection.execute("""
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, value))

                connection.commit()
                
                # Update operation stats
                self.operation_stats["settings_updated"] += 1
                
                self.logger.info(f"Setting updated successfully: {key}")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["settings_update_errors"] += 1
            self.logger.error(f"Error setting setting: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("settings_update")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="document setting",
    )
    async def set_document(self, collection: str, key: str, data: dict[str, Any]) -> bool:
        """
        Set document with enhanced error handling and connection pooling.

        Args:
            collection: Document collection
            key: Document key
            data: Document data

        Returns:
            bool: True if setting successful, False otherwise
        """
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for document update")
                return False

            try:
                # Convert data to JSON
                data_json = json.dumps(data)

                # Insert or update document
                cursor = connection.execute("""
                    INSERT OR REPLACE INTO documents (collection, key, data, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (collection, key, data_json))

                connection.commit()
                
                # Update operation stats
                self.operation_stats["documents_updated"] += 1
                
                self.logger.info(f"Document updated successfully: {collection}/{key}")
                return True

            finally:
                # Return connection to pool
                await self.connection_pool.return_connection(connection)

        except Exception as e:
            self.error_stats["documents_update_errors"] += 1
            self.logger.error(f"Error setting document: {e}")
            
            # Attempt recovery if needed
            await self._attempt_recovery("documents_update")
            return False

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
            if (current_time - self.last_recovery_attempt < self.recovery_cooldown or 
                self.recovery_attempts >= self.max_recovery_attempts):
                return
            
            self.logger.info(f"🔄 Attempting recovery for operation: {operation}")
            
            # Attempt to reinitialize connection pool
            if self.connection_pool:
                await self.connection_pool.initialize()
            
            self.recovery_attempts += 1
            self.last_recovery_attempt = current_time
            
            self.logger.info(f"✅ Recovery attempt {self.recovery_attempts}/{self.max_recovery_attempts} completed")
            
        except Exception as e:
            self.logger.error(f"Error during recovery attempt: {e}")

    async def _auto_backup_task(self) -> None:
        """Background task for automatic database backup."""
        while True:
            try:
                await asyncio.sleep(self.backup_interval)
                await self.create_backup()
            except Exception as e:
                self.logger.error(f"Error in auto backup task: {e}")
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
            
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

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
            bool: True if backup successful, False otherwise
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path}.backup_{timestamp}"

            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            if not connection:
                self.logger.error("Failed to get connection for backup")
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

        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            return False

    def get_database_status(self) -> dict[str, Any]:
        """
        Get comprehensive database status.

        Returns:
            Dict[str, Any]: Database status information
        """
        try:
            status = {
                "is_connected": self.is_connected,
                "database_path": self.database_path,
                "auto_backup": self.auto_backup,
                "backup_interval": self.backup_interval,
                "max_connections": self.max_connections,
                "recovery_attempts": self.recovery_attempts,
                "max_recovery_attempts": self.max_recovery_attempts,
                "uptime": time.time() - self.start_time,
                "operation_stats": dict(self.operation_stats),
                "error_stats": dict(self.error_stats)
            }
            
            # Add connection pool stats if available
            if self.connection_pool:
                status["connection_pool_stats"] = self.connection_pool.get_pool_stats()
            
            return status

        except Exception as e:
            self.logger.error(f"Error getting database status: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SQLite manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the SQLite manager."""
        self.logger.info("🛑 Stopping SQLite Manager...")

        try:
            # Close database connections
            await self.close()

            # Clear operation stats
            self.operation_stats.clear()
            self.error_stats.clear()

            self.logger.info("✅ SQLite Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping SQLite manager: {e}")


# Global SQLite manager instance
sqlite_manager: SQLiteManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="SQLite manager setup",
)
async def setup_sqlite_manager(
    config: dict[str, Any] | None = None,
) -> SQLiteManager | None:
    """
    Setup global SQLite manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[SQLiteManager]: Global SQLite manager instance
    """
    try:
        global sqlite_manager

        if config is None:
            config = {
                "sqlite_manager": {
                    "database_path": "data/ares.db",
                    "auto_backup": True,
                    "backup_interval": 3600,
                    "max_connections": 10,
                    "enable_foreign_keys": True,
                    "journal_mode": "WAL",
                    "max_recovery_attempts": 3,
                    "recovery_cooldown": 60,
                },
            }

        # Create SQLite manager
        sqlite_manager = SQLiteManager(config)

        # Initialize SQLite manager
        success = await sqlite_manager.initialize()
        if success:
            return sqlite_manager
        return None

    except Exception as e:
        print(f"Error setting up SQLite manager: {e}")
        return None
