# src/database/sqlite_manager.py

import sqlite3
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple 

# Assuming system_logger is available
from src.utils.logger import system_logger as logger

class SQLiteManager:
    """
    Manages interactions with a local SQLite database.
    Provides methods to store and retrieve structured data, mimicking FirestoreManager.
    """

    def __init__(self, db_path: str = "ares_local_db.sqlite"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.logger = logger.getChild('SQLiteManager')
        self._initialized = False

    async def initialize(self):
        """Initializes the SQLite database connection and creates tables."""
        if self._initialized:
            self.logger.info("SQLiteManager already initialized.")
            return

        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # Allow multiple threads to access (for async)
            self.conn.row_factory = sqlite3.Row # Access columns by name
            await self._create_tables()
            self._initialized = True
            self.logger.info(f"SQLiteManager initialized successfully at {self.db_path}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLiteManager at {self.db_path}: {e}", exc_info=True)
            self.conn = None # Ensure connection is None on failure

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
            self.conn.rollback() # Rollback on error
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
                performance_metrics TEXT, -- Stored as JSON string
                date_applied TEXT,
                params TEXT, -- Stored as JSON string
                is_public INTEGER -- 0 for false, 1 for true
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ares_live_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metrics TEXT -- Stored as JSON string
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ares_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                context_data TEXT -- Stored as JSON string
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
                TradeDurationSeconds REAL,
                NetPnLUSD REAL,
                PnLPercentage REAL,
                ExitReason TEXT,
                EntryPrice REAL,
                ExitPrice REAL,
                QuantityBaseAsset REAL,
                NotionalSizeUSD REAL,
                LeverageUsed INTEGER,
                IntendedStopLossPrice REAL,
                IntendedTakeProfitPrice REAL,
                ActualStopLossPrice REAL,
                ActualTakeProfitPrice REAL,
                OrderTypeEntry TEXT,
                OrderTypeExit TEXT,
                EntryFeesUSD REAL,
                ExitFeesUSD REAL,
                SlippageEntryPct REAL,
                SlippageExitPct REAL,
                MarketRegimeAtEntry TEXT,
                TacticianSignal TEXT,
                EnsemblePredictionAtEntry TEXT,
                EnsembleConfidenceAtEntry REAL,
                DirectionalConfidenceAtEntry REAL,
                MarketHealthScoreAtEntry REAL,
                LiquidationSafetyScoreAtEntry REAL,
                TrendStrengthAtEntry REAL,
                ADXValueAtEntry REAL,
                RSIValueAtEntry REAL,
                MACDHistogramValueAtEntry REAL,
                PriceVsVWAPRatioAtEntry REAL,
                VolumeDeltaAtEntry REAL,
                GlobalRiskMultiplierAtEntry REAL,
                AvailableAccountEquityAtEntry REAL,
                TradingEnvironment TEXT,
                IsTradingPausedAtEntry INTEGER,
                KillSwitchActiveAtEntry INTEGER,
                ModelVersionID TEXT,
                BaseModelPredictionsAtEntry TEXT, -- Stored as JSON string
                EnsembleWeightsAtEntry TEXT -- Stored as JSON string
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS daily_summaries (
                Date TEXT PRIMARY KEY,
                TotalTrades INTEGER,
                WinRate REAL,
                NetPnL REAL,
                MaxDrawdown REAL,
                EndingCapital REAL,
                AllocatedCapitalMultiplier REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS strategy_performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT,
                Regime TEXT,
                TotalTrades INTEGER,
                WinRate REAL,
                NetPnL REAL,
                AvgPnLPerTrade REAL,
                TradeDuration REAL,
                UNIQUE(Date, Regime) -- Ensure unique entry per date and regime
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ares_unhandled_exceptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                type TEXT,
                message TEXT,
                traceback TEXT,
                context TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS global_state (
                key TEXT PRIMARY KEY,
                value TEXT, -- Stored as JSON string
                timestamp TEXT
            );
            """
        ]
        for query in queries:
            await self._execute_query(query)
        self.logger.info("All necessary SQLite tables checked/created.")

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
        """Retrieves a single document by its ID from a specified table."""
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
                        pass # Not a JSON string, keep as is
            self.logger.debug(f"Document {doc_id} retrieved from {collection_name}.")
            return doc
        else:
            self.logger.debug(f"Document {doc_id} not found in {collection_name}.")
            return None

    async def add_document(self, collection_name: str, data: Dict[str, Any], is_public: bool = False) -> Optional[str]:
        """
        Adds a new document to a specified table, with an auto-incrementing ID.
        Returns the ID of the new document.
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
            # For auto-incrementing primary keys, lastrowid gives the ID
            last_id = self.conn.cursor().lastrowid
            self.logger.debug(f"Document added to {collection_name} with ID: {last_id}.")
            return str(last_id)
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
        """Deletes a document by its ID from a specified table."""
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
            self.conn = None
            self._initialized = False
            self.logger.info(f"SQLite database connection to {self.db_path} closed.")

# Global instance of SQLiteManager.
# It should be initialized asynchronously in the main application entry point.
sqlite_manager = SQLiteManager()

