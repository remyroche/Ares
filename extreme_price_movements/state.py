import json
import os
import shutil
import tempfile
import pandas as pd
from typing import Dict, Any

class StateManager:
    def __init__(self, filepath="state.json"):
        self.filepath = filepath
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.filepath):
            return {"last_ts_sig": None, "positions": {}, "run_id": None, "pending_orders": []}
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            return {"last_ts_sig": None, "positions": {}, "run_id": None, "pending_orders": []}

    def save(self):
        # Atomic write
        dir_name = os.path.dirname(self.filepath) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            json.dump(self.state, tf, indent=2)
            temp_name = tf.name
        shutil.move(temp_name, self.filepath)

    def get_last_ts_sig(self) -> pd.Timestamp | None:
        val = self.state.get("last_ts_sig")
        if val:
            return pd.Timestamp(val)
        return None

    def set_last_ts_sig(self, ts: pd.Timestamp):
        self.state["last_ts_sig"] = ts.isoformat()
        self.save()

    def get_positions(self) -> Dict[str, Any]:
        return self.state.get("positions", {})

    def update_positions(self, positions: Dict[str, Any]):
        self.state["positions"] = positions
        self.save()

    def clear_position(self, symbol: str):
        if "positions" in self.state and symbol in self.state["positions"]:
            del self.state["positions"][symbol]
            self.save()

    def set_position(self, symbol: str, data: Dict[str, Any]):
        if "positions" not in self.state:
            self.state["positions"] = {}
        self.state["positions"][symbol] = data
        self.save()

    def set_run_id(self, run_id: str):
        self.state["run_id"] = run_id
        self.save()

    def get_pending_orders(self):
        return self.state.get("pending_orders", [])

    def set_pending_orders(self, orders: list):
        self.state["pending_orders"] = orders
        self.save()
