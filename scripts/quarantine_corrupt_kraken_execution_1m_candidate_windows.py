#!/usr/bin/env python3
"""Quarantine unreadable canonical 1m source parts intersecting candidate windows.

This is a deliberately narrow recovery tool for an append-only exact-execution
store.  It scans only the timestamp windows needed by a supplied target-free
candidate panel, validates each intersecting Parquet part, and atomically moves
only unreadable parts to the canonical quarantine tree.  The operation is
recoverable from its immutable receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import canonical_kraken_execution_1m_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _symbol_dir(root: Path, symbol: str) -> Path:
    return root / "ohlcv" / f"symbol={symbol.replace('/', '_')}"


def _part_bounds(path: Path) -> tuple[int, int] | None:
    bits = path.stem.split("-")
    try:
        return int(bits[-2]), int(bits[-1])
    except (IndexError, ValueError):
        return None


def _validate_part(record: dict[str, object], mode: str) -> tuple[dict[str, object], str | None]:
    """Return a copied record plus any integrity failure; no mutation occurs here."""
    path = Path(str(record["path"]))
    try:
        if mode == "full_parquet":
            # A narrow column read still opens and validates the full Parquet footer.
            pd.read_parquet(path, columns=["ts"])
        else:
            with path.open("rb") as handle:
                header = handle.read(4)
                handle.seek(-4, os.SEEK_END)
                footer = handle.read(4)
            if header != b"PAR1" or footer != b"PAR1":
                raise ValueError(
                    f"Parquet magic mismatch header={header!r} footer={footer!r}"
                )
    except Exception as exc:
        return record, f"{type(exc).__name__}: {exc}"
    return record, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--horizon-minutes", type=int, required=True)
    parser.add_argument("--warmup-minutes", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--validation-mode",
        choices=("full_parquet", "footer_magic"),
        default="full_parquet",
        help=(
            "full_parquet validates every intersecting footer with Arrow. "
            "footer_magic is a fast precheck for the PAR1 header/footer signature; "
            "the eventual exact-path materialiser still performs its full, "
            "fail-closed read before producing a result."
        ),
    )
    args = parser.parse_args()

    if args.horizon_minutes <= 0 or args.warmup_minutes < 0:
        raise ValueError("horizon-minutes must be positive and warmup-minutes non-negative")
    fields = pd.read_parquet(args.candidates)
    timestamp_column = "timestamp" if "timestamp" in fields else "__decision_ts__"
    symbol_column = "symbol" if "symbol" in fields else "__symbol__"
    if timestamp_column not in fields or symbol_column not in fields:
        raise ValueError("candidate panel requires timestamp/symbol columns")
    frame = fields[[timestamp_column, symbol_column]].rename(
        columns={timestamp_column: "timestamp", symbol_column: "symbol"}
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "symbol"])
    root = canonical_kraken_execution_1m_root(args.data_root).resolve()
    warmup = pd.Timedelta(minutes=args.warmup_minutes)
    horizon = pd.Timedelta(minutes=args.horizon_minutes)
    parts_to_validate: list[dict[str, object]] = []

    for symbol, group in frame.groupby("symbol", sort=True):
        start = group["timestamp"].min() - warmup
        end = group["timestamp"].max() + horizon
        start_epoch, end_epoch = int(start.timestamp()), int(end.timestamp())
        directory = _symbol_dir(root, str(symbol))
        if not directory.exists():
            continue
        for year in range(start.year, end.year + 1):
            for path in sorted((directory / f"year={year}").glob("*.parquet")):
                bounds = _part_bounds(path)
                if bounds is None or bounds[1] < start_epoch or bounds[0] > end_epoch:
                    continue
                relative = path.resolve().relative_to(root)
                record = {
                    "symbol": str(symbol),
                    "path": str(path.resolve()),
                    "relative_path": str(relative),
                    "file_start_epoch": bounds[0],
                    "file_end_epoch": bounds[1],
                }
                parts_to_validate.append(record)

    scanned: list[dict[str, object]] = []
    unreadable: list[dict[str, object]] = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record, failure in executor.map(
            lambda value: _validate_part(value, args.validation_mode), parts_to_validate
        ):
            if failure is None:
                scanned.append(record)
            else:
                record["validation_failure"] = failure
                unreadable.append(record)

    moved: list[dict[str, object]] = []
    if not args.dry_run:
        for record in unreadable:
            path = Path(str(record["path"]))
            target = root / "quarantine" / str(record["relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
            digest = _sha256(path)
            os.replace(path, target)
            moved.append(
                {
                    **record,
                    "sha256": digest,
                    "bytes": int(target.stat().st_size),
                    "quarantine_path": str(target),
                    "operation": "atomic_move_reversible",
                }
            )
    receipt = args.receipt.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError(f"receipt already exists: {receipt}")
    descriptor = os.open(receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "kraken_execution_1m_candidate_window_quarantine_v1",
                "canonical_root": str(root),
                "candidate_path": str(args.candidates.resolve()),
                "candidate_sha256": _sha256(args.candidates.resolve()),
                "horizon_minutes": args.horizon_minutes,
                "warmup_minutes": args.warmup_minutes,
                "validation_mode": args.validation_mode,
                "validation_workers": workers,
                "scanned_readable_parts": len(scanned),
                "unreadable_parts": unreadable,
                "moved_parts": moved,
                "dry_run": bool(args.dry_run),
                "restoration": (
                    "move each quarantine_path back to path only after an independent "
                    "Parquet-read validation"
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "scanned_readable_parts": len(scanned),
                "unreadable_parts": len(unreadable),
                "moved_parts": len(moved),
                "receipt": str(receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
