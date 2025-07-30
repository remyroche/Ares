# src/database/sqlite_manager.py

import sqlite3
import json
import logging
import shutil
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple 

# Assuming system_logger is available
from src.utils.logger import system_logger
from src.config import CONFIG

class SQLiteManager:
    """
    Manages interactions with a local SQLite database.
    Provides methods to store and retrieve structured data, mimicking FirestoreManager.
    Enhanced with backup, migration, and persistence features.
    """

    def __init__(self):
        # Use the database path from CONFIG
        self.db_path = CONFIG.get("SQLITE_DB_PATH", "data/ares_local_db.sqlite")
        self.backup_dir = "data/backups"
        self.migration_dir = "data/migrations"
        self.conn: Optional[sqlite3.Connection] = None
        self.logger = system_logger.getChild('SQLiteManager')
        self._initialized = False
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.migration_dir, exist_ok=True)

    async def initialize(self):
        """Initializes the SQLite database connection and creates tables."""
        if self._initialized:
            self.logger.info("SQLiteManager already initialized.")
            return

        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            await self._create_tables()
            self._initialized = True
            self.logger.info(f"SQLiteManager initialized successfully at {self.db_path}.")
            
            # Create initial backup
            await self.create_backup("initial")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLiteManager at {self.db_path}: {e}", exc_info=True)
            self.conn = None

    async def _execute_query(self, query: str, params: Union[Tuple[Any, ...], List[Tuple[Any, ...]]] = ()) -> List[sqlite3.Row]:
        """Helper to execute a single query or multiple queries (for executemany)."""
        if not self.conn:
            self.logger.error("Database connection not established. Cannot execute query.")
            return []
        
        try:
            cursor = self.conn.cursor()
            if isinstance(params, list) and all(isinstance(p, tuple) for p in params):
                cursor.executemany(query, params)
            else:
                cursor.execute(query, params)
            self.conn.commit()
            return cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error executing query: {query} with params {params}. Error: {e}", exc_info=True)
            self.conn.rollback()
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {e}", exc_info=True)
            self.conn.rollback()
            return []

    async def _create_tables(self):
        """Creates necessary tables if they don't exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS ares_optimized_params (
                doc_id TEXT PRIMARY KEY,
                timestamp TEXT,
                optimization_run_id TEXT,
                performance_metrics TEXT,
                date_applied TEXT,
                params TEXT,
                is_public INTEGER
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ares_live_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metrics TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ares_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                context_data TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS detailed_trade_logs (
                TradeID TEXT PRIMARY KEY,
                Token TEXT,
                Exchange TEXT,
                Side TEXT,
                EntryTimestampUTC TEXT,
                ExitTimestampUTC TEXT,
                EntryPrice REAL,
                ExitPrice REAL,
                Quantity REAL,
                PnL REAL,
                PnLPercent REAL,
                Strategy TEXT,
                EntryReason TEXT,
                ExitReason TEXT,
                StopLoss REAL,
                TakeProfit REAL,
                Leverage INTEGER,
                FundingRate REAL,
                MarketRegime TEXT,
                SRLevel TEXT,
                TechnicalIndicators TEXT,
                RiskMetrics TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                checkpoint_path TEXT,
                timestamp TEXT,
                performance_metrics TEXT,
                model_hash TEXT,
                is_active INTEGER DEFAULT 0
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT UNIQUE,
                start_date TEXT,
                end_date TEXT,
                symbol TEXT,
                strategy TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                profit_factor REAL,
                results_data TEXT,
                created_at TEXT,
                is_migrated INTEGER DEFAULT 0
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS trading_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                timestamp TEXT,
                updated_at TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS database_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_id TEXT UNIQUE,
                source_computer TEXT,
                target_computer TEXT,
                migration_type TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT,
                file_size INTEGER,
                checksum TEXT
            );
            """
        ]
        
        for query in queries:
            await self._execute_query(query)
        
        self.logger.info("All necessary SQLite tables checked/created.")

    async def create_backup(self, backup_name: str = None) -> str:
        """Creates a backup of the database."""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.sqlite")
        
        try:
            if self.conn:
                self.conn.close()
            
            shutil.copy2(self.db_path, backup_path)
            
            if not self._initialized:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
            
            # Calculate checksum
            with open(backup_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            self.logger.info(f"Database backup created: {backup_path} (checksum: {checksum})")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}", exc_info=True)
            return ""

    async def restore_backup(self, backup_path: str) -> bool:
        """Restores database from a backup."""
        try:
            if self.conn:
                self.conn.close()
            
            shutil.copy2(backup_path, self.db_path)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            self.logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}", exc_info=True)
            return False

    async def export_for_migration(self, migration_name: str = None) -> str:
        """Exports database for migration to another computer."""
        if not migration_name:
            migration_name = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        migration_path = os.path.join(self.migration_dir, f"{migration_name}.sqlite")
        
        try:
            # Create a clean copy for migration
            shutil.copy2(self.db_path, migration_path)
            
            # Calculate checksum
            with open(migration_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            # Record migration
            migration_data = {
                'migration_id': migration_name,
                'source_computer': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'target_computer': 'pending',
                'migration_type': 'export',
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'completed_at': None,
                'file_size': os.path.getsize(migration_path),
                'checksum': checksum
            }
            
            await self.set_document('database_migrations', migration_name, migration_data)
            
            self.logger.info(f"Migration export created: {migration_path} (checksum: {checksum})")
            return migration_path
            
        except Exception as e:
            self.logger.error(f"Failed to create migration export: {e}", exc_info=True)
            return ""

    async def import_migration(self, migration_path: str) -> bool:
        """Imports database from a migration file."""
        try:
            # Verify file exists
            if not os.path.exists(migration_path):
                self.logger.error(f"Migration file not found: {migration_path}")
                return False
            
            # Calculate checksum of migration file
            with open(migration_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            # Create backup before import
            await self.create_backup("pre_migration_import")
            
            # Close current connection
            if self.conn:
                self.conn.close()
            
            # Copy migration file to database location
            shutil.copy2(migration_path, self.db_path)
            
            # Reopen connection
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Update migration record
            migration_name = os.path.basename(migration_path).replace('.sqlite', '')
            migration_data = {
                'migration_id': migration_name,
                'source_computer': 'unknown',
                'target_computer': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'migration_type': 'import',
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'file_size': os.path.getsize(migration_path),
                'checksum': checksum
            }
            
            await self.set_document('database_migrations', migration_name, migration_data)
            
            self.logger.info(f"Migration import completed: {migration_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import migration: {e}", exc_info=True)
            return False

    async def schedule_backup(self, interval_hours: int = 24):
        """Schedules regular backups."""
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
                await self.create_backup()
            except Exception as e:
                self.logger.error(f"Backup scheduling error: {e}", exc_info=True)

    async def get_backtest_results(self, backtest_id: str = None) -> List[Dict[str, Any]]:
        """Retrieves backtest results."""
        if backtest_id:
            return await self.get_document('backtest_results', backtest_id)
        else:
            return await self.get_collection('backtest_results')

    async def save_backtest_results(self, results: Dict[str, Any]) -> bool:
        """Saves backtest results to database."""
        backtest_id = results.get('backtest_id', f"backtest_{int(time.time())}")
        results['created_at'] = datetime.now().isoformat()
        return await self.set_document('backtest_results', backtest_id, results)

    async def get_trading_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves trading state by key."""
        return await self.get_document('trading_state', key)

    async def set_trading_state(self, key: str, value: Dict[str, Any]) -> bool:
        """Sets trading state by key."""
        value['updated_at'] = datetime.now().isoformat()
        return await self.set_document('trading_state', key, value)

    async def set_document(self, collection_name: str, doc_id: str, data: Dict[str, Any], is_public: bool = False):
        """
        Sets a document (inserts or updates) in a specified table.
        For simplicity, 'collection_name' maps directly to a table name.
        'is_public' is ignored for local SQLite but kept for API compatibility.
        """
        if not self.conn:
            self.logger.error(f"Database not initialized. Cannot set document in {collection_name}.")
            return False

        # Convert dicts to JSON strings for storage
        data_to_store = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in data.items()}
        
        # Add doc_id if it's not already in data (for tables where it's the PK)
        if 'doc_id' not in data_to_store:
            data_to_store['doc_id'] = doc_id

        columns = ', '.join(data_to_store.keys())
        placeholders = ', '.join(['?' for _ in data_to_store.keys()])

        query = f"""
        INSERT OR REPLACE INTO {collection_name} ({columns})
        VALUES ({placeholders});
        """
        params = tuple(data_to_store.values())

        try:
            await self._execute_query(query, params)
            self.logger.debug(f"Document {doc_id} set in {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting document {doc_id} in {collection_name}: {e}", exc_info=True)
            return False

    async def get_document(self, collection_name: str, doc_id: str, is_public: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single document by its ID.
        'is_public' is ignored for local SQLite.
        """
        if not self.conn:
            self.logger.error(f"Database not initialized. Cannot get document from {collection_name}.")
            return None

        query = f"SELECT * FROM {collection_name} WHERE doc_id = ?;"
        rows = await self._execute_query(query, (doc_id,))
        
        if rows:
            doc = dict(rows[0])
            # Convert JSON strings back to dicts/lists
            for k, v in doc.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        doc[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            return doc
        return None

    async def add_document(self, collection_name: str, data: Dict[str, Any], is_public: bool = False) -> Optional[str]:
        """
        Adds a new document to a collection (table).
        'is_public' is ignored for local SQLite.
        Returns the generated document ID.
        """
        if not self.conn:
            self.logger.error(f"Database not initialized. Cannot add document to {collection_name}.")
            return None

        # Convert dicts to JSON strings for storage
        data_to_store = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in data.items()}
        
        columns = ', '.join(data_to_store.keys())
        placeholders = ', '.join(['?' for _ in data_to_store.keys()])

        query = f"""
        INSERT INTO {collection_name} ({columns})
        VALUES ({placeholders});
        """
        params = tuple(data_to_store.values())

        try:
            await self._execute_query(query, params)
            # For tables with auto-increment, we can't easily get the ID
            # For now, return a timestamp-based ID
            doc_id = f"{collection_name}_{int(time.time())}"
            self.logger.debug(f"Document added to {collection_name} with ID: {doc_id}.")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error adding document to {collection_name}: {e}", exc_info=True)
            return None

    async def get_collection(self, collection_name: str, is_public: bool = False, query_filters: Optional[List[Tuple[str, str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all documents from a collection (table), optionally with filters.
        'is_public' is ignored for local SQLite.
        query_filters: List of (field, operator, value) tuples, e.g., [('timestamp', '>', '2023-01-01')]
        """
        if not self.conn:
            self.logger.error(f"Database not initialized. Cannot get collection from {collection_name}.")
            return []

        where_clause = ""
        params = []
        if query_filters:
            conditions = []
            for field, op, value in query_filters:
                conditions.append(f"{field} {op} ?")
                params.append(value)
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"SELECT * FROM {collection_name}{where_clause};"
        rows = await self._execute_query(query, tuple(params))
        
        results = []
        for row in rows:
            doc = dict(row)
            # Convert JSON strings back to dicts/lists
            for k, v in doc.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        doc[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            results.append(doc)
        
        self.logger.debug(f"Retrieved {len(results)} documents from {collection_name}.")
        return results

    async def delete_document(self, collection_name: str, doc_id: str, is_public: bool = False) -> bool:
        """
        Deletes a document from a collection (table).
        'is_public' is ignored for local SQLite.
        """
        if not self.conn:
            self.logger.error(f"Database not initialized. Cannot delete document from {collection_name}.")
            return False

        query = f"DELETE FROM {collection_name} WHERE doc_id = ?;"
        try:
            await self._execute_query(query, (doc_id,))
            self.logger.debug(f"Document {doc_id} deleted from {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id} from {collection_name}: {e}", exc_info=True)
            return False

    async def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self._initialized = False
            self.logger.info(f"SQLite database connection to {self.db_path} closed.")

# Global instance of SQLiteManager.
# It should be initialized asynchronously in the main application entry point.
sqlite_manager = SQLiteManager()

