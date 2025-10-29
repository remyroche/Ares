"""Persistence utilities for monitoring data."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MonitoringPersistence:
    """Handles persistence of monitoring data."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize persistence handler.

        Args:
            base_path: Base directory for persistence files
        """
        self.base_path = Path(base_path or "monitoring_data")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.getChild('MonitoringPersistence')

    async def save_alerts(self, alerts: List[Dict[str, Any]], filename: str = "alerts.json") -> bool:
        """Save alerts to file."""
        try:
            filepath = self.base_path / filename
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'alerts': alerts
                }, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save alerts: {e}")
            return False

    async def load_alerts(self, filename: str = "alerts.json") -> List[Dict[str, Any]]:
        """Load alerts from file."""
        try:
            filepath = self.base_path / filename
            if not filepath.exists():
                return []

            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('alerts', [])
        except Exception as e:
            self.logger.error(f"Failed to load alerts: {e}")
            return []

    async def save_trades(self, trades: List[Dict[str, Any]], filename: str = "trades.json") -> bool:
        """Save trades to file."""
        try:
            filepath = self.base_path / filename
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'trades': trades
                }, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save trades: {e}")
            return False

    async def load_trades(self, filename: str = "trades.json") -> List[Dict[str, Any]]:
        """Load trades from file."""
        try:
            filepath = self.base_path / filename
            if not filepath.exists():
                return []

            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('trades', [])
        except Exception as e:
            self.logger.error(f"Failed to load trades: {e}")
            return []

    async def save_performance_metrics(self, metrics: Dict[str, Any], filename: str = "performance.json") -> bool:
        """Save performance metrics to file."""
        try:
            filepath = self.base_path / filename
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                }, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")
            return False

    async def load_performance_metrics(self, filename: str = "performance.json") -> Optional[Dict[str, Any]]:
        """Load performance metrics from file."""
        try:
            filepath = self.base_path / filename
            if not filepath.exists():
                return None

            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('metrics')
        except Exception as e:
            self.logger.error(f"Failed to load performance metrics: {e}")
            return None

    async def save_state(self, state: Dict[str, Any], component: str) -> bool:
        """Save component state."""
        try:
            filepath = self.base_path / f"{component}_state.json"
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'state': state
                }, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {component} state: {e}")
            return False

    async def load_state(self, component: str) -> Optional[Dict[str, Any]]:
        """Load component state."""
        try:
            filepath = self.base_path / f"{component}_state.json"
            if not filepath.exists():
                return None

            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('state')
        except Exception as e:
            self.logger.error(f"Failed to load {component} state: {e}")
            return None

    async def cleanup_old_files(self, days: int = 30) -> None:
        """Clean up files older than specified days."""
        try:
            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            for filepath in self.base_path.glob("*.json"):
                if filepath.stat().st_mtime < cutoff:
                    filepath.unlink()
                    self.logger.debug(f"Deleted old file: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")
