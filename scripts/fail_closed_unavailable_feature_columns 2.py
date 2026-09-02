#!/usr/bin/env python3
"""Fail closed feature columns whose canonical raw source is unavailable.

This is deliberately narrow: it only overwrites reviewed repair keys for an
explicit symbol list, uses an atomic replace, and records exactly what changed.
It is used for canonical feature files that cannot be recomputed honestly
because the corresponding raw source no longer exists.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _feature_path(store_root: Path, symbol: str) -> Path:
    return store_root / f"symbol={symbol.replace('/', '_')}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store-root", type=Path, required=True)
    parser.add_argument("--symbols-file", type=Path, required=True)
    parser.add_argument("--keys-file", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    args = parser.parse_args()

    symbols = [
        line.strip() for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [
        line.strip() for line in args.keys_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit: dict[str, object] = {"symbols": {}, "requested_key_count": len(keys)}
    for symbol in symbols:
        path = _feature_path(args.feature_store_root, symbol)
        if not path.exists():
            raise SystemExit(f"Canonical feature file is missing: {path}")
        frame = pd.read_parquet(path)
        present = [key for key in keys if key in frame.columns]
        absent = [key for key in keys if key not in frame.columns]
        before_finite = {
            key: int(np.isfinite(pd.to_numeric(frame[key], errors="coerce")).sum())
            for key in present
        }
        for key in keys:
            frame[key] = np.nan
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        try:
            frame.to_parquet(tmp_path, index=True)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        audit["symbols"][symbol] = {
            "path": str(path),
            "cleared_existing_keys": present,
            "added_missing_nan_keys": absent,
            "cleared_key_count": len(present),
            "finite_values_removed": before_finite,
        }
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
