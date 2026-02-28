from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STORE_PATH = Path("artifacts/policy_params_store.json")


def _resolve_store_path(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        # default to module-relative artifacts directory
        return Path(__file__).resolve().parent.parent / DEFAULT_STORE_PATH
    base = Path(base_dir)
    if base.is_file():
        return base
    return (base / DEFAULT_STORE_PATH).resolve()


def load_params_store(base_dir: str | Path | None = None) -> dict:
    store_path = _resolve_store_path(base_dir)
    if not store_path.exists():
        return {}
    return json.loads(store_path.read_text())


def save_params_store(store: dict, base_dir: str | Path | None = None) -> None:
    store_path = _resolve_store_path(base_dir)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2, sort_keys=True))


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
