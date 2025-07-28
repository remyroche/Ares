# src/utils/state_manager.py
import os
from src.config import CONFIG
from src.utils.logger import system_logger

class StateManager:
    """
    Manages the operational state of the trading bot (e.g., RUNNING, PAUSED).
    Uses a simple file-based flag for persistence across different system components.
    """
    def __init__(self):
        self.logger = system_logger.getChild('StateManager')
        self.state_file = CONFIG.get("STATE_FILE", "ares_state.json")
        self.state = self._load_state()

    def _load_state(self):
        """Loads the current state from the state file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                self.logger.error(f"Error loading state file: {e}. Defaulting to RUNNING.")
                return "RUNNING"
        return "RUNNING" # Default state if file doesn't exist

    def _save_state(self):
        """Saves the current state to the state file."""
        try:
            with open(self.state_file, 'w') as f:
                f.write(self.state)
            self.logger.info(f"System state changed to: {self.state}")
        except Exception as e:
            self.logger.error(f"Error saving state file: {e}")

    def get_state(self):
        """Returns the current operational state."""
        # Always read from the file to ensure the latest state is retrieved,
        # especially if another process (like the email listener) changes it.
        return self._load_state()

    def pause_trading(self):
        """Sets the system state to PAUSED."""
        self.logger.warning("Setting trading state to PAUSED.")
        self.state = "PAUSED"
        self._save_state()

    def resume_trading(self):
        """Sets the system state to RUNNING."""
        self.logger.info("Setting trading state to RUNNING.")
        self.state = "RUNNING"
        self._save_state()

    def is_running(self):
        """Checks if the system state is RUNNING."""
        return self.get_state() == "RUNNING"
