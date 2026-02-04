import json
import os
import shutil
import tempfile
import pandas as pd
from typing import Dict, Any
from .utils import tprint

class StateManager:
    def __init__(self, filepath="state.json"):
        tprint(f"Entering function: __init__ in state.py")
        self.filepath = filepath
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        tprint(f"Entering function: _load in state.py")
        if not os.path.exists(self.filepath):
            tprint(f"State file {self.filepath} does not exist. Returning default state.")
            return {"last_ts_sig": None, "positions": {}, "run_id": None, "pending_orders": []}
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            tprint(f"Successfully loaded state from {self.filepath}. Keys: {list(data.keys())}")
            return data
        except json.JSONDecodeError:
            tprint(f"Error decoding JSON from {self.filepath}. Returning default state.")
            return {"last_ts_sig": None, "positions": {}, "run_id": None, "pending_orders": []}

    def save(self):
        # Atomic write
        tprint(f"Entering function: save in state.py")
        tprint(f"Saving state to {self.filepath}...")
        dir_name = os.path.dirname(self.filepath) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            json.dump(self.state, tf, indent=2)
            temp_name = tf.name
        shutil.move(temp_name, self.filepath)

    def get_last_ts_sig(self) -> pd.Timestamp | None:
        tprint(f"Entering function: get_last_ts_sig in state.py")
        val = self.state.get("last_ts_sig")
        if val:
            return pd.Timestamp(val)
        return None

    def set_last_ts_sig(self, ts: pd.Timestamp):
        tprint(f"Entering function: set_last_ts_sig in state.py")
        tprint(f"Setting last_ts_sig to {ts}")
        self.state["last_ts_sig"] = ts.isoformat()
        self.save()

    def get_positions(self) -> Dict[str, Any]:
        tprint(f"Entering function: get_positions in state.py")
        return self.state.get("positions", {})

    def update_positions(self, positions: Dict[str, Any]):
        tprint(f"Entering function: update_positions in state.py")
        tprint(f"Updating positions. New positions keys: {list(positions.keys())}")
        self.state["positions"] = positions
        self.save()

    def clear_position(self, symbol: str):
        tprint(f"Entering function: clear_position in state.py")
        if "positions" in self.state and symbol in self.state["positions"]:
            tprint(f"Clearing position for symbol: {symbol}")
            del self.state["positions"][symbol]
            self.save()
        else:
            tprint(f"Attempted to clear position for {symbol}, but it was not found.")

    def set_position(self, symbol: str, data: Dict[str, Any]):
        tprint(f"Entering function: set_position in state.py")
        tprint(f"Setting position for {symbol}. Data keys: {list(data.keys())}")
        if "positions" not in self.state:
            self.state["positions"] = {}
        self.state["positions"][symbol] = data
        self.save()

    def set_run_id(self, run_id: str):
        tprint(f"Entering function: set_run_id in state.py")
        tprint(f"Setting run_id to: {run_id}")
        self.state["run_id"] = run_id
        self.save()

    def get_pending_orders(self):
        tprint(f"Entering function: get_pending_orders in state.py")
        orders = self.state.get("pending_orders", [])
        tprint(f"Retrieved {len(orders)} pending orders.")
        return orders

    def set_pending_orders(self, orders: list):
        tprint(f"Entering function: set_pending_orders in state.py")
        tprint(f"Setting {len(orders)} pending orders.")
        self.state["pending_orders"] = orders
        self.save()
