#!/usr/bin/env python3
"""Write the immutable key contract for an in-place portability backfill.

The canonical feature store is intentionally overwritten only for this exact
set of keys.  Keeping the set in a small committed manifest makes the backfill
reviewable and prevents a later ad-hoc runner from silently changing the
definition of only some historical columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-store-id", required=True)
    parser.add_argument("--end-ts", required=True)
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        default=None,
        help=(
            "Reuse a previously written immutable key contract. This keeps "
            "metadata-only preflight independent from the heavyweight runtime "
            "configuration import."
        ),
    )
    parser.add_argument(
        "--feature-store-root",
        type=Path,
        default=None,
        help=(
            "Optional canonical store to enumerate. When supplied, emit a "
            "stable symbol allowlist so the migration only overwrites existing "
            "canonical files."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional source-store root. Canonical symbols without a matching "
            "raw-source directory are emitted separately and excluded from the "
            "recompute allowlist so they can be fail-closed explicitly."
        ),
    )
    args = parser.parse_args()

    if args.existing_manifest is not None:
        payload = json.loads(args.existing_manifest.read_text(encoding="utf-8"))
        expected = {
            "feature_store_id": str(args.feature_store_id),
            "end_ts": str(args.end_ts),
        }
        mismatch = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if str(payload.get(key)) != value
        }
        if mismatch:
            raise SystemExit(f"Existing manifest scope mismatch: {mismatch}")
        keys = sorted(map(str, payload["keys"]))
    else:
        # Delay the project import until actually needed. The allowlist-only
        # invocation below deliberately avoids initializing optional runtime
        # modules (for example Numba) before the store migration starts.
        from extreme_price_movements.config import (
            FEATURE_PORTABILITY_REPAIR_KEYS,
            FEATURE_PORTABILITY_REPAIR_ROLLING_WINDOW_HOURS,
            FEATURE_PORTABILITY_REPAIR_SCHEMA_VERSION,
        )

        keys = sorted(map(str, FEATURE_PORTABILITY_REPAIR_KEYS))
        payload = {
            "schema": FEATURE_PORTABILITY_REPAIR_SCHEMA_VERSION,
            "feature_store_id": str(args.feature_store_id),
            "end_ts": str(args.end_ts),
            "rolling_window_hours": int(FEATURE_PORTABILITY_REPAIR_ROLLING_WINDOW_HOURS),
            "keys": keys,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload["key_contract_sha256"] = digest

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "repair_keys.txt").write_text(
        "\n".join(keys) + "\n", encoding="utf-8"
    )
    (args.output_dir / "repair_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.feature_store_root is not None:
        symbols: list[str] = []
        for meta_path in sorted(args.feature_store_root.glob("symbol=*.meta.json")):
            try:
                value = json.loads(meta_path.read_text(encoding="utf-8"))
                symbol = str(value.get("symbol", "")).strip()
            except (OSError, json.JSONDecodeError):
                symbol = ""
            if symbol:
                symbols.append(symbol)
        symbols = sorted(set(symbols))
        if not symbols:
            raise SystemExit(
                f"No canonical symbols found under {args.feature_store_root}"
            )
        source_missing: list[str] = []
        if args.source_root is not None:
            source_missing = [
                symbol
                for symbol in symbols
                if not (args.source_root / f"symbol={symbol.replace('/', '_')}").is_dir()
            ]
            symbols = [symbol for symbol in symbols if symbol not in set(source_missing)]
        (args.output_dir / "canonical_symbols.txt").write_text(
            "\n".join(symbols) + "\n", encoding="utf-8"
        )
        (args.output_dir / "unavailable_canonical_symbols.txt").write_text(
            "\n".join(source_missing) + ("\n" if source_missing else ""),
            encoding="utf-8",
        )
        payload["canonical_symbol_count"] = len(symbols)
        payload["unavailable_canonical_symbol_count"] = len(source_missing)
        (args.output_dir / "repair_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
