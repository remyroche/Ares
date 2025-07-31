# src/database/sqlite_manager.py

import aiosqlite
import json
import os
import hashlib
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from src.utils.logger import system_logger
from src.config import CONFIG


class SQLiteManager:
    """
    Asynchronously manages interactions with a local SQLite database using aiosqlite.
    Provides methods to store and retrieve structured data, mimicking FirestoreManager.
    Enhanced with non-blocking backup, migration, and persistence features.
    """

    def __init__(self):
        """Initializes the SQLiteManager."""
        self.db_path = CONFIG.get("SQLITE_DB_PATH", "data/ares_local_db.sqlite")
        self.backup_dir = "data/backups"
        self.migration_dir = "data/migrations"
        self.conn: Optional[aiosqlite.Connection] = None
        self.logger = system_logger.getChild("SQLiteManager")
        self._initialized = False
        self._backup_task: Optional[asyncio.Task] = None

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
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
            await self._create_tables()
            self._initialized = True
            self.logger.info(
                f"SQLiteManager initialized successfully at {self.db_path}."
            )

            # Create initial backup if it doesn't exist
            if not os.path.exists(os.path.join(self.backup_dir, "initial.sqlite")):
                await self.create_backup("initial")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize SQLiteManager at {self.db_path}: {e}",
                exc_info=True,
            )
            self.conn = None

    async def __aenter__(self):
        """Async context manager for entering a 'with' block."""
        if not self.conn:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager for exiting a 'with' block."""
        await self.close()

    async def _execute_query(
        self, query: str, params: Union[Tuple[Any, ...], List[Tuple[Any, ...]]] = ()
    ) -> List[aiosqlite.Row]:
        """Helper to execute a single query or multiple queries asynchronously."""
        if not self.conn:
            self.logger.error(
                "Database connection not established. Cannot execute query."
            )
            return []

        try:
            async with self.conn.cursor() as cursor:
                if isinstance(params, list) and all(
                    isinstance(p, tuple) for p in params
                ):
                    await cursor.executemany(query, params)
                else:
                    await cursor.execute(query, params)
                await self.conn.commit()
                return await cursor.fetchall()
        except aiosqlite.Error as e:
            self.logger.error(
                f"SQLite error executing query: {query} with params {params}. Error: {e}",
                exc_info=True,
            )
            await self.conn.rollback()
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {e}", exc_info=True)
            await self.conn.rollback()
            return []

    async def _create_tables(self):
        """Creates necessary tables if they don't exist."""
        queries = [
            """CREATE TABLE IF NOT EXISTS ares_optimized_params (doc_id TEXT PRIMARY KEY, timestamp TEXT, optimization_run_id TEXT, performance_metrics TEXT, date_applied TEXT, params TEXT, is_public INTEGER);""",
            """CREATE TABLE IF NOT EXISTS ares_live_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, metrics TEXT);""",
            """CREATE TABLE IF NOT EXISTS ares_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, alert_type TEXT, severity TEXT, message TEXT, context_data TEXT);""",
            """CREATE TABLE IF NOT EXISTS detailed_trade_logs (TradeID TEXT PRIMARY KEY, Token TEXT, Exchange TEXT, Side TEXT, EntryTimestampUTC TEXT, ExitTimestampUTC TEXT, EntryPrice REAL, ExitPrice REAL, Quantity REAL, PnL REAL, PnLPercent REAL, Strategy TEXT, EntryReason TEXT, ExitReason TEXT, StopLoss REAL, TakeProfit REAL, Leverage INTEGER, FundingRate REAL, MarketRegime TEXT, SRLevel TEXT, TechnicalIndicators TEXT, RiskMetrics TEXT);""",
            """CREATE TABLE IF NOT EXISTS model_checkpoints (id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT, checkpoint_path TEXT, timestamp TEXT, performance_metrics TEXT, model_hash TEXT, is_active INTEGER DEFAULT 0);""",
            """CREATE TABLE IF NOT EXISTS backtest_results (id INTEGER PRIMARY KEY AUTOINCREMENT, backtest_id TEXT UNIQUE, start_date TEXT, end_date TEXT, symbol TEXT, strategy TEXT, total_trades INTEGER, winning_trades INTEGER, losing_trades INTEGER, win_rate REAL, total_pnl REAL, max_drawdown REAL, sharpe_ratio REAL, profit_factor REAL, results_data TEXT, created_at TEXT, is_migrated INTEGER DEFAULT 0);""",
            """CREATE TABLE IF NOT EXISTS trading_state (key TEXT PRIMARY KEY, value TEXT, timestamp TEXT, updated_at TEXT);""",
            """CREATE TABLE IF NOT EXISTS database_migrations (id INTEGER PRIMARY KEY AUTOINCREMENT, migration_id TEXT UNIQUE, source_computer TEXT, target_computer TEXT, migration_type TEXT, status TEXT, created_at TEXT, completed_at TEXT, file_size INTEGER, checksum TEXT);""",
        ]
        for query in queries:
            await self._execute_query(query)
        self.logger.info("All necessary SQLite tables checked/created.")

    async def create_backup(self, backup_name: str = None) -> str:
        """Creates a backup of the database using aiosqlite's backup protocol."""
        if not self.conn:
            self.logger.error("Database not connected. Cannot create backup.")
            return ""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.sqlite")

        try:
            async with aiosqlite.connect(backup_path) as backup_db:
                await self.conn.backup(backup_db)

            def _calculate_checksum():
                with open(backup_path, "rb") as f:
                    return hashlib.md5(f.read()).hexdigest()

            loop = asyncio.get_running_loop()
            checksum = await loop.run_in_executor(None, _calculate_checksum)

            self.logger.info(
                f"Database backup created: {backup_path} (checksum: {checksum})"
            )
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}", exc_info=True)
            return ""

    async def restore_backup(self, backup_path: str) -> bool:
        """Restores the database from a backup file."""
        if not self.conn:
            self.logger.error("Database not connected. Cannot restore backup.")
            return False
        if not os.path.exists(backup_path):
            self.logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            async with aiosqlite.connect(backup_path) as backup_db:
                await backup_db.backup(self.conn)
            self.logger.info(f"Database restored from backup: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}", exc_info=True)
            return False

    async def schedule_backup(self, interval_hours: int = 24):
        """Schedules regular backups in a background task."""
        if self._backup_task and not self._backup_task.done():
            self.logger.warning("Backup task is already running.")
            return

        async def backup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)
                    await self.create_backup()
                except asyncio.CancelledError:
                    self.logger.info("Backup task cancelled.")
                    break
                except Exception as e:
                    self.logger.error(f"Backup scheduling error: {e}", exc_info=True)

        self._backup_task = asyncio.create_task(backup_loop())
        self.logger.info(f"Backup scheduled every {interval_hours} hours.")

    async def set_document(
        self,
        collection_name: str,
        doc_id: str,
        data: Dict[str, Any],
        is_public: bool = False,
    ):
        """Sets a document (inserts or updates) in a specified table."""
        if not self.conn:
            self.logger.error(
                f"Database not initialized. Cannot set document in {collection_name}."
            )
            return False

        data_to_store = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in data.items()
        }

        # Use the primary key of the table for doc_id
        pk_col = (
            "doc_id"
            if "doc_id" in [c[1] for c in await self.get_table_schema(collection_name)]
            else "key"
        )
        data_to_store[pk_col] = doc_id

        columns = ", ".join(data_to_store.keys())
        placeholders = ", ".join(["?" for _ in data_to_store.keys()])
        query = f"INSERT OR REPLACE INTO {collection_name} ({columns}) VALUES ({placeholders});"
        params = tuple(data_to_store.values())

        try:
            await self._execute_query(query, params)
            self.logger.debug(f"Document {doc_id} set in {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(
                f"Error setting document {doc_id} in {collection_name}: {e}",
                exc_info=True,
            )
            return False

    async def get_document(
        self, collection_name: str, doc_id: str, is_public: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single document by its ID."""
        if not self.conn:
            return None

        pk_col = (
            "doc_id"
            if "doc_id" in [c[1] for c in await self.get_table_schema(collection_name)]
            else "key"
        )
        query = f"SELECT * FROM {collection_name} WHERE {pk_col} = ?;"
        rows = await self._execute_query(query, (doc_id,))

        if rows:
            doc = dict(rows[0])
            for k, v in doc.items():
                if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                    try:
                        doc[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            return doc
        return None

    async def get_collection(
        self,
        collection_name: str,
        is_public: bool = False,
        query_filters: Optional[List[Tuple[str, str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves all documents from a table, optionally with filters."""
        if not self.conn:
            return []

        where_clause = ""
        params = []
        if query_filters:
            conditions = [f"{field} {op} ?" for field, op, value in query_filters]
            params = [value for field, op, value in query_filters]
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"SELECT * FROM {collection_name}{where_clause};"
        rows = await self._execute_query(query, tuple(params))

        results = []
        for row in rows:
            doc = dict(row)
            for k, v in doc.items():
                if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                    try:
                        doc[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            results.append(doc)
        return results

    async def delete_document(
        self, collection_name: str, doc_id: str, is_public: bool = False
    ) -> bool:
        """Deletes a document from a table."""
        pk_col = (
            "doc_id"
            if "doc_id" in [c[1] for c in await self.get_table_schema(collection_name)]
            else "key"
        )
        query = f"DELETE FROM {collection_name} WHERE {pk_col} = ?;"
        await self._execute_query(query, (doc_id,))
        return True

    async def get_table_schema(self, table_name):
        """Retrieves the schema of a table."""
        query = f"PRAGMA table_info({table_name});"
        return await self._execute_query(query)

    async def close(self):
        """Closes the database connection and cancels background tasks."""
        if self._backup_task and not self._backup_task.done():
            self._backup_task.cancel()
        if self.conn:
            await self.conn.close()
            self._initialized = False
            self.logger.info(f"SQLite database connection to {self.db_path} closed.")
