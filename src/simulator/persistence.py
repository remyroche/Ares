"""
Simulator Persistence

SQLite database operations for storing simulator state, positions, and trades.
"""

import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from .position_manager import Position


class SimulatorPersistence:
    """
    Handle persistence of simulator state to SQLite database.
    
    Stores:
    - Simulator state (balance, config)
    - Positions
    - Trade history
    - Analytics
    """
    
    def __init__(self, db_path: str = "simulator_state.db"):
        """
        Initialize persistence layer.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simulator state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulator_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulator_id TEXT UNIQUE NOT NULL,
                mode TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                initial_balance REAL NOT NULL,
                current_balance REAL NOT NULL,
                direction_constraint TEXT,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Simulator positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulator_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulator_id TEXT NOT NULL,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulator_id) REFERENCES simulator_state(simulator_id)
            )
        """)
        
        # Simulator trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulator_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulator_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                direction TEXT,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL,
                slippage REAL NOT NULL,
                pnl REAL,
                is_maker INTEGER DEFAULT 0,
                fill_details_json TEXT,
                latency_ms INTEGER,
                order_type TEXT,
                trading_signal_json TEXT,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulator_id) REFERENCES simulator_state(simulator_id)
            )
        """)
        
        # Simulator analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulator_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulator_id TEXT NOT NULL,
                session_start TIMESTAMP NOT NULL,
                session_end TIMESTAMP,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                total_fees REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL,
                win_rate REAL,
                profit_factor REAL,
                metrics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulator_id) REFERENCES simulator_state(simulator_id)
            )
        """)
        
        conn.commit()
        conn.close()
        self.logger.debug(f"Database initialized at {self.db_path}")
    
    def save_simulator_state(
        self,
        simulator_id: str,
        mode: str,
        exchange: str,
        asset: str,
        initial_balance: float,
        current_balance: float,
        direction_constraint: str = None,
        config_json: str = None
    ) -> None:
        """Save or update simulator state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO simulator_state 
            (simulator_id, mode, exchange, asset, initial_balance, current_balance,
             direction_constraint, config_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (simulator_id, mode, exchange, asset, initial_balance, current_balance,
              direction_constraint, config_json))
        
        conn.commit()
        conn.close()
    
    def get_simulator_state(self, simulator_id: str) -> Optional[Dict[str, Any]]:
        """Get simulator state by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT simulator_id, mode, exchange, asset, initial_balance, current_balance,
                   direction_constraint, config_json, created_at, updated_at
            FROM simulator_state
            WHERE simulator_id = ?
        """, (simulator_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "simulator_id": row[0],
            "mode": row[1],
            "exchange": row[2],
            "asset": row[3],
            "initial_balance": row[4],
            "current_balance": row[5],
            "direction_constraint": row[6],
            "config_json": row[7],
            "created_at": row[8],
            "updated_at": row[9]
        }
    
    def save_position(
        self,
        simulator_id: str,
        position: Position
    ) -> None:
        """Save or update a position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(position.metadata) if position.metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO simulator_positions
            (simulator_id, position_id, symbol, direction, quantity, entry_price,
             entry_time, stop_loss, take_profit, status, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (simulator_id, position.id, position.symbol, position.direction,
              position.quantity, position.avg_entry_price, position.entry_time.isoformat(),
              position.stop_loss, position.take_profit, "open", metadata_json))
        
        conn.commit()
        conn.close()
    
    def get_positions(self, simulator_id: str, status: str = "open") -> List[Dict[str, Any]]:
        """Get all positions for a simulator."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status == "all":
            cursor.execute("""
                SELECT position_id, symbol, direction, quantity, entry_price,
                       entry_time, stop_loss, take_profit, status, metadata_json
                FROM simulator_positions
                WHERE simulator_id = ?
                ORDER BY entry_time DESC
            """, (simulator_id,))
        else:
            cursor.execute("""
                SELECT position_id, symbol, direction, quantity, entry_price,
                       entry_time, stop_loss, take_profit, status, metadata_json
                FROM simulator_positions
                WHERE simulator_id = ? AND status = ?
                ORDER BY entry_time DESC
            """, (simulator_id, status))
        
        rows = cursor.fetchall()
        conn.close()
        
        positions = []
        for row in rows:
            metadata = json.loads(row[9]) if row[9] else {}
            positions.append({
                "position_id": row[0],
                "symbol": row[1],
                "direction": row[2],
                "quantity": row[3],
                "entry_price": row[4],
                "entry_time": row[5],
                "stop_loss": row[6],
                "take_profit": row[7],
                "status": row[8],
                "metadata": metadata
            })
        
        return positions
    
    def close_position(self, simulator_id: str, position_id: str) -> None:
        """Mark a position as closed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE simulator_positions
            SET status = 'closed', updated_at = CURRENT_TIMESTAMP
            WHERE simulator_id = ? AND position_id = ?
        """, (simulator_id, position_id))
        
        conn.commit()
        conn.close()
    
    def save_trade(
        self,
        simulator_id: str,
        trade_data: Dict[str, Any]
    ) -> int:
        """Save a trade record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        fill_details_json = json.dumps(trade_data.get("fill_details")) if trade_data.get("fill_details") else None
        trading_signal_json = json.dumps(trade_data.get("trading_signal")) if trade_data.get("trading_signal") else None
        
        cursor.execute("""
            INSERT INTO simulator_trades
            (simulator_id, symbol, side, direction, quantity, price, fee, slippage, pnl,
             is_maker, fill_details_json, latency_ms, order_type, trading_signal_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            simulator_id, trade_data["symbol"], trade_data["side"], trade_data.get("direction"),
            trade_data["quantity"], trade_data["price"], trade_data["fee"], trade_data["slippage"],
            trade_data.get("pnl"), 1 if trade_data.get("is_maker") else 0, fill_details_json,
            trade_data.get("latency_ms"), trade_data.get("order_type"), trading_signal_json,
            trade_data["timestamp"]
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def get_trades(
        self,
        simulator_id: str,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("""
                SELECT symbol, side, direction, quantity, price, fee, slippage, pnl,
                       is_maker, fill_details_json, order_type, timestamp
                FROM simulator_trades
                WHERE simulator_id = ? AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (simulator_id, symbol, limit))
        else:
            cursor.execute("""
                SELECT symbol, side, direction, quantity, price, fee, slippage, pnl,
                       is_maker, fill_details_json, order_type, timestamp
                FROM simulator_trades
                WHERE simulator_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (simulator_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for row in rows:
            fill_details = json.loads(row[9]) if row[9] else {}
            trades.append({
                "symbol": row[0],
                "side": row[1],
                "direction": row[2],
                "quantity": row[3],
                "price": row[4],
                "fee": row[5],
                "slippage": row[6],
                "pnl": row[7],
                "is_maker": bool(row[8]),
                "fill_details": fill_details,
                "order_type": row[10],
                "timestamp": row[11]
            })
        
        return trades
    
    def close_database(self) -> None:
        """Close database connection."""
        # SQLite connection per operation, nothing to close here
        pass
