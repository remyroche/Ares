from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = Path("extreme_price_movements/artifacts/policy_params_store.json")


def load_params_store() -> dict:
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text())


def save_params_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2, sort_keys=True))


def store_best_params(store: dict, version_key: str, bucket_id: str, params: dict, metrics: dict) -> dict:
    store.setdefault(version_key, {})
    store[version_key][bucket_id] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics,
    }
    return store


def get_initial_params(store: dict, version_key: str, bucket_id: str, defaults: dict) -> dict:
    b = store.get(version_key, {}).get(bucket_id)
    if not b:
        return defaults
    merged = dict(defaults)
    merged.update(b.get("params", {}))
    return merged
