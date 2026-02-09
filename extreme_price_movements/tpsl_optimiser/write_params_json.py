from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_and_write_params(path: str | Path, bucket: str, fragments: dict[str, Any], schema_version: str = "v1") -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = {"schema_version": schema_version, "buckets": {}}

    payload.setdefault("schema_version", schema_version)
    payload.setdefault("buckets", {})
    payload["buckets"][str(bucket)] = fragments

    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
