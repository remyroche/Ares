import os
from pathlib import Path
from typing import Dict, Any
import json
import datetime
from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.async_utils import AsyncFileManager


class StateManager:
    """
    Manages the operational state of the trading bot asynchronously.
    Uses a file-based flag for persistence across different system components.
    """

    def __init__(self, state_file: str = "ares_state.json", sync_init: bool = False):
        self.logger = system_logger.getChild("StateManager")
        self.state_file = state_file
        self.state_dir = Path(os.path.dirname(self.state_file) or ".")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.kill_switch_file = self.state_dir / "kill_switch.txt"
        self._state_cache = {}

        if sync_init:
            # For scripts that cannot be async, allow a sync load.
            self._state_cache = self._load_state_from_file_sync()

    @classmethod
    async def create(cls, state_file: str = "ares_state.json"):
        """Async factory to create and initialize a StateManager instance."""
        instance = cls(state_file)
        await instance._initialize()
        return instance

    async def _initialize(self):
        """Asynchronously loads the initial state from the file."""
        self._state_cache = await self._load_state_from_file()
        self.logger.info(
            f"StateManager initialized asynchronously. State file: {self.state_file}"
        )

    def _load_state_from_file_sync(self) -> Dict[str, Any]:
        """Synchronously loads state for compatibility."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                self.logger.error(
                    f"Error loading state file synchronously: {e}. Starting with empty state."
                )
        return self._get_default_state()

    async def _load_state_from_file(self) -> Dict[str, Any]:
        """Loads the state from the file asynchronously."""
        if os.path.exists(self.state_file):
            try:
                content = await AsyncFileManager.read_file(self.state_file)
                if content:
                    return json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                self.logger.error(
                    f"Error loading state file asynchronously: {e}. Starting with empty state."
                )
        return self._get_default_state()

    def _get_default_state(self) -> Dict[str, Any]:
        """Returns the default state dictionary."""
        return {
            "global_trading_status": "RUNNING",
            "is_trading_paused": False,
            "global_peak_equity": CONFIG.get("initial_equity", 10000),
            "account_equity": CONFIG.get("initial_equity", 10000),
            "global_risk_multiplier": 1.0,
            "last_retrain_timestamp": datetime.datetime.now().isoformat(),
            "production_model_run_id": None,
            "active_position": None,
            "analyst_intelligence": {},
            "strategist_params": {},
            "total_trades": 0,
            "live_profit_factor": 0.0,
            "live_sharpe_ratio": 0.0,
            "live_win_rate": 0.0,
            "leverage": 1,
            "current_position": self._get_default_position_structure(),
        }

    async def _save_state_to_file(self):
        """Saves the current state to the file asynchronously."""
        try:
            await AsyncFileManager.write_json(self.state_file, self._state_cache)
        except Exception as e:
            self.logger.error(f"Error saving state file: {e}")

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieves a state value from the in-memory cache (synchronous)."""
        return self._state_cache.get(key, default)

    async def set_state(self, key: str, value: Any):
        """Sets a state value and saves the state asynchronously."""
        self._state_cache[key] = value
        await self._save_state_to_file()
        self.logger.debug(f"State updated for '{key}'.")

    async def set_state_if_not_exists(self, key: str, value: Any):
        """Sets a state value only if it doesn't exist."""
        if key not in self._state_cache:
            await self.set_state(key, value)

    def _get_default_position_structure(self):
        return {
            "direction": None,
            "size": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "leverage": 1,
            "entry_timestamp": 0.0,
            "trade_id": None,
            "entry_context": {},
        }

    async def pause_trading(self):
        await self.set_state("is_trading_paused", True)
        self.logger.warning("Setting trading state to PAUSED.")

    async def resume_trading(self):
        await self.set_state("is_trading_paused", False)
        self.logger.info("Setting trading state to RUNNING.")

    def is_kill_switch_active(self) -> bool:
        return self.kill_switch_file.exists()

    async def activate_kill_switch(self, reason: str):
        if not self.is_kill_switch_active():
            self.logger.critical(f"!!! KILL SWITCH ACTIVATED !!! Reason: {reason}")
            await AsyncFileManager.write_file(str(self.kill_switch_file), reason)

    async def deactivate_kill_switch(self):
        if self.is_kill_switch_active():
            self.logger.warning("Deactivating kill switch.")
            try:
                os.remove(self.kill_switch_file)
            except OSError as e:
                self.logger.error(f"Error removing kill switch file: {e}")

    async def get_kill_switch_reason(self) -> str:
        if self.is_kill_switch_active():
            return await AsyncFileManager.read_file(str(self.kill_switch_file))
        return "Kill switch is not active."

    def is_running(self):
        return not self.get_state("is_trading_paused")
