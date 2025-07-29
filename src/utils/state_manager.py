# src/utils/state_manager.py
import os
import logging
from pathlib import Path
from typing import Dict, Any
import json # Import json for serializing/deserializing complex state
import datetime # Import datetime for default timestamp
from src.config import CONFIG # Assuming CONFIG is available at this path
from src.utils.logger import system_logger # Centralized logger

class StateManager:
    """
    Manages the operational state of the trading bot (e.g., RUNNING, PAUSED).
    Uses a simple file-based flag for persistence across different system components.
    Now supports more complex state objects.
    """
    def __init__(self, state_file: str = "ares_state.json"):
        self.logger = system_logger.getChild('StateManager')
        self.state_file = state_file # Allow state file to be passed
        
        # Ensure state directory exists
        self.state_dir = Path(os.path.dirname(self.state_file) or '.') # Use dirname or current if no path
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.kill_switch_file = self.state_dir / "kill_switch.txt"
        self._state_cache = {} # In-memory cache for current state

        # Load initial state from file on startup
        self._state_cache = self._load_state_from_file()
        self.logger.info(f"StateManager initialized. State file: {self.state_file}")

    def _load_state_from_file(self) -> Dict[str, Any]:
        """Loads the entire state dictionary from the state file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding state file JSON: {e}. Starting with empty state.")
            except Exception as e:
                self.logger.error(f"Error loading state file: {e}. Starting with empty state.")
        self.logger.info("State file not found or invalid. Starting with default empty state.")
        # Define default states if file doesn't exist or is invalid
        return {
            "global_trading_status": "RUNNING",
            "is_trading_paused": False,
            "global_peak_equity": CONFIG.get("initial_equity", 10000),
            "account_equity": CONFIG.get("initial_equity", 10000),
            "global_risk_multiplier": 1.0,
            "last_retrain_timestamp": datetime.datetime.now().isoformat(),
            "active_position": None, # Detailed active position
            "analyst_intelligence": {},
            "strategist_params": {},
            "total_trades": 0,
            "live_profit_factor": 0.0,
            "live_sharpe_ratio": 0.0,
            "live_win_rate": 0.0,
            "leverage": 1, # Default leverage
            "current_position": self._get_default_position_structure() # For tactician's internal use
        }

    def _save_state_to_file(self):
        """Saves the current in-memory state to the state file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self._state_cache, f, indent=4)
            # self.logger.debug(f"System state saved to {self.state_file}") # Too verbose for frequent saves
        except Exception as e:
            self.logger.error(f"Error saving state file: {e}")

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieves a specific state value by key."""
        return self._state_cache.get(key, default)

    def set_state(self, key: str, value: Any):
        """Sets a specific state value by key and saves the entire state."""
        self._state_cache[key] = value
        self._save_state_to_file()
        self.logger.debug(f"State updated for '{key}'.")

    def set_state_if_not_exists(self, key: str, value: Any):
        """Sets a state value only if it doesn't already exist."""
        if key not in self._state_cache:
            self.set_state(key, value)

    def _get_default_position_structure(self):
        """Returns the default structure for an empty position for internal use."""
        return {
            "direction": None, 
            "size": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "leverage": 1,
            "entry_timestamp": 0.0,
            "trade_id": None, # Add trade ID
            "entry_context": {} # Store all decision-making context at entry
        }


    def pause_trading(self):
        """Sets the system state to PAUSED."""
        self.set_state("is_trading_paused", True)
        self.logger.warning("Setting trading state to PAUSED.")

    def resume_trading(self):
        """Sets the system state to RUNNING."""
        self.set_state("is_trading_paused", False)
        self.logger.info("Setting trading state to RUNNING.")

    def is_kill_switch_active(self) -> bool:
        """Checks if the kill switch is currently active."""
        return self.kill_switch_file.exists()

    def activate_kill_switch(self, reason: str):
        """
        Activates the kill switch by creating a file that serves as a persistent flag.
        """
        if not self.is_kill_switch_active():
            self.logger.critical(f"!!! KILL SWITCH ACTIVATED !!! Reason: {reason}")
            with open(self.kill_switch_file, 'w') as f:
                f.write(reason)
            # You might want to trigger an immediate high-priority alert here
            # (e.g., mailer.send_alert("KILL SWITCH ACTIVATED", f"Reason: {reason}"))

    def deactivate_kill_switch(self):
        """Deactivates the kill switch by removing the flag file."""
        if self.is_kill_switch_active():
            self.logger.warning("Deactivating kill switch.")
            os.remove(self.kill_switch_file)
            # You might want to trigger an alert here
            # (e.g., mailer.send_alert("Kill Switch Deactivated", "Trading can be resumed manually."))

    def get_kill_switch_reason(self) -> str:
        if self.is_kill_switch_active():
            with open(self.kill_switch_file, 'r') as f:
                return f.read().strip()
        return "Kill switch is not active."

    def is_running(self):
        """Checks if the system state is RUNNING (not paused by local state manager)."""
        return not self.get_state("is_trading_paused")
