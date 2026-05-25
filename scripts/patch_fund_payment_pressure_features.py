#!/usr/bin/env python3
"""Rewrite funding payment pressure features without a full feature recompute."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGETS = (5, 10)


def _pct(num: int, den: int) -> float:
    return float(num) / float(max(den, 1)) * 100.0


def _rewrite_file(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    df = pd.read_parquet(path)
    total_rows = int(len(df))
    out: dict[str, Any] = {
        "path": str(path),
        "rows": total_rows,
        "updated": False,
        "missing_columns": "",
    }
    missing: list[str] = []
    if "fund_abs_z_14d" not in df.columns:
        missing.append("fund_abs_z_14d")
    for h in TARGETS:
        if f"fund_next_event_proximity_{h}h" not in df.columns:
            missing.append(f"fund_next_event_proximity_{h}h")
        if f"fund_payment_pressure_{h}h" not in df.columns:
            missing.append(f"fund_payment_pressure_{h}h")
    if missing:
        out["missing_columns"] = ",".join(missing)
        return out

    fund_abs = pd.to_numeric(df["fund_abs_z_14d"], errors="coerce").fillna(0.0)
    changed = False
    for h in TARGETS:
        pressure_col = f"fund_payment_pressure_{h}h"
        proximity_col = f"fund_next_event_proximity_{h}h"
        old = pd.to_numeric(df[pressure_col], errors="coerce")
        proximity = pd.to_numeric(df[proximity_col], errors="coerce").fillna(0.0)
        new = (fund_abs * proximity).clip(lower=0.0, upper=6.0).astype("float32")

        old_bad = int((~np.isfinite(old.to_numpy(dtype=float, copy=False))).sum())
        new_bad = int((~np.isfinite(new.to_numpy(dtype=float, copy=False))).sum())
        changed_rows = int(
            (
                old.fillna(np.float32(-999999.0)).astype("float32")
                != new.fillna(np.float32(-999999.0))
            ).sum()
        )
        out[f"{pressure_col}_old_bad_rows"] = old_bad
        out[f"{pressure_col}_old_bad_pct"] = _pct(old_bad, total_rows)
        out[f"{pressure_col}_new_bad_rows"] = new_bad
        out[f"{pressure_col}_new_bad_pct"] = _pct(new_bad, total_rows)
        out[f"{pressure_col}_changed_rows"] = changed_rows
        if changed_rows:
            df[pressure_col] = new
            changed = True

    if changed and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
    out["updated"] = bool(changed and not dry_run)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        default="data_perp/features/20260523_015947",
        help="Directory containing symbol=*.parquet feature files.",
    )
    parser.add_argument(
        "--report",
        default="data_perp/artifacts/20260523_015947/features/fund_payment_pressure_targeted_patch_report.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.feature_root)
    paths = sorted(root.glob("symbol=*.parquet"))
    if not paths:
        raise SystemExit(f"No feature parquet files found under {root}")

    rows = []
    for i, path in enumerate(paths, 1):
        rows.append(_rewrite_file(path, dry_run=bool(args.dry_run)))
        if i % 25 == 0 or i == len(paths):
            print(f"processed {i}/{len(paths)}", flush=True)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report, index=False)
    print(f"wrote report: {report}", flush=True)

    updated = sum(1 for row in rows if row.get("updated"))
    old_bad = {
        f"fund_payment_pressure_{h}h": int(
            sum(row.get(f"fund_payment_pressure_{h}h_old_bad_rows", 0) for row in rows)
        )
        for h in TARGETS
    }
    new_bad = {
        f"fund_payment_pressure_{h}h": int(
            sum(row.get(f"fund_payment_pressure_{h}h_new_bad_rows", 0) for row in rows)
        )
        for h in TARGETS
    }
    print(
        f"files={len(paths)} updated={updated} old_bad={old_bad} new_bad={new_bad}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
