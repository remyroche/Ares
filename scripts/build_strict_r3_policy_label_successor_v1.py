#!/usr/bin/env python3
"""Build an immutable rich-policy outcome-ledger successor.

The successor prepends explicitly reconstructed, target-free candidate
populations to an incumbent canonical policy ledger.  It exists for offline
strict-OOF research only: no model, calibration, portfolio, inference, or
execution state is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = [
    "candidate_id", "__decision_ts__", "__symbol__", "side_name",
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    "policy_outcome_source", "label_source_complete_1m_path",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        paths = sorted(path.rglob("*.parquet"))
    else:
        paths = [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_early(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted((root / "policy_parts").glob("month=*.parquet")):
        pieces.append(pd.read_parquet(path, columns=REQUIRED))
    if not pieces:
        raise FileNotFoundError(f"no early policy parts below {root}")
    return pd.concat(pieces, ignore_index=True)


def _normalize(frame: pd.DataFrame, origin: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise AssertionError(f"{origin}: missing required columns {missing}")
    out = frame.loc[:, REQUIRED].copy()
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    out["policy_label_available_ts"] = pd.to_datetime(out["policy_label_available_ts"], utc=True, errors="coerce")
    out["policy_path_valid"] = out["policy_path_valid"].fillna(False).astype(bool)
    if out.candidate_id.duplicated().any():
        raise AssertionError(f"{origin}: duplicated candidate IDs")
    valid = out["policy_path_valid"].to_numpy(bool)
    if valid.any():
        gross = pd.to_numeric(out.loc[valid, "policy_gross_bps"], errors="coerce").to_numpy(float)
        net = pd.to_numeric(out.loc[valid, "policy_net_bps"], errors="coerce").to_numpy(float)
        if not np.isfinite(gross).all() or not np.isfinite(net).all():
            raise AssertionError(f"{origin}: valid policy rows have non-finite economics")
        if not np.allclose(gross - net, 100.0, rtol=0.0, atol=1e-8):
            raise AssertionError(f"{origin}: policy cost is not exactly once")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-roots", required=True, help="comma-separated immutable rich-policy materializer roots")
    parser.add_argument("--incumbent-policy", type=Path, required=True)
    parser.add_argument("--cutoff", default="2025-04-01T00:00:00Z")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cutoff = pd.Timestamp(args.cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    early_roots = [Path(token.strip()) for token in args.early_roots.split(",") if token.strip()]
    if not early_roots:
        raise ValueError("--early-roots must not be empty")
    early = _normalize(pd.concat([_read_early(root) for root in early_roots], ignore_index=True), "early labels")
    incumbent = _normalize(pd.read_parquet(args.incumbent_policy, columns=REQUIRED), "incumbent labels")
    if not early["__decision_ts__"].lt(cutoff).all():
        raise AssertionError("early labels cross the declared successor cutoff")
    if not incumbent["__decision_ts__"].ge(cutoff).all():
        raise AssertionError("incumbent labels precede the declared successor cutoff")
    combined = pd.concat([early, incumbent], ignore_index=True)
    if combined.candidate_id.duplicated().any():
        raise AssertionError("successor ledger duplicates candidate IDs")
    combined = combined.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    availability = combined["policy_label_available_ts"] - combined["__decision_ts__"]
    if not availability.dropna().eq(pd.Timedelta(hours=12)).all():
        raise AssertionError("successor policy labels are not uniformly available at H12")
    args.out.mkdir(parents=True)
    target = args.out / "canonical_reconciled_policy_labels.parquet"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    combined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    monthly = combined.assign(month=combined["__decision_ts__"].dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
        rows=("candidate_id", "size"),
        valid_rows=("policy_path_valid", "sum"),
    )
    monthly["valid_fraction"] = monthly["valid_rows"] / monthly["rows"].clip(lower=1)
    monthly.to_parquet(args.out / "coverage_by_month.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_policy_label_successor_v1",
        "scope": "offline strict-OOF research outcome-ledger successor; no live/inference/model/calibration/portfolio/execution mutation",
        "cutoff": cutoff.isoformat(),
        "early_roots": [str(root) for root in early_roots],
        "early_sha256": [_sha(root) for root in early_roots],
        "incumbent_policy": str(args.incumbent_policy),
        "incumbent_sha256": _sha(args.incumbent_policy),
        "rows": int(len(combined)),
        "early_rows": int(len(early)),
        "incumbent_rows": int(len(incumbent)),
        "valid_rows": int(combined["policy_path_valid"].sum()),
        "invalid_rows": int((~combined["policy_path_valid"]).sum()),
        "candidate_contract": "early source retains every target-free path-panel candidate before outcome replay; invalid paths remain invalid labels",
        "cost_contract": "gross minus net equals 100 bps exactly for every valid row",
        "availability_contract": "policy labels resolve exactly H12 after decision timestamp",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out), "rows": int(len(combined)), "valid_rows": int(combined.policy_path_valid.sum())}, sort_keys=True))


if __name__ == "__main__":
    main()
