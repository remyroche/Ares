from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:10]


def emit_run_header(tprint, run_id: str, policy_version: str, cost_model: dict, extra: dict | None = None):
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "policy_version": policy_version,
        "cost_model": cost_model,
        "cost_hash": _hash_dict(cost_model),
    }
    if extra:
        payload.update(extra)
    tprint("[RUN_HEADER] " + json.dumps(payload, sort_keys=True))


def emit_bucket_summary(tprint, run_id: str, bucket_id: str, kind: str, stats: dict):
    payload = {"run_id": run_id, "bucket_id": bucket_id, "kind": kind, **stats}
    tprint("[BUCKET_SUMMARY] " + json.dumps(payload, sort_keys=True))
