#!/usr/bin/env python3
"""Append immutable rich-policy label parts to an existing causal ledger.

This is deliberately an offline research-only utility.  It creates a new
ledger; neither the incumbent policy parquet nor any source label part is
modified.  Candidate identities must already have been fixed before the
extension's future paths were materialised.
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
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _normalise(frame: pd.DataFrame, *, origin: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise AssertionError(f"{origin}: missing required columns {missing}")
    out = frame.loc[:, REQUIRED].copy()
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    out["policy_label_available_ts"] = pd.to_datetime(out["policy_label_available_ts"], utc=True, errors="coerce")
    out["policy_path_valid"] = out["policy_path_valid"].fillna(False).astype(bool)
    if out.candidate_id.duplicated().any():
        raise AssertionError(f"{origin}: duplicate candidate IDs")
    valid = out["policy_path_valid"].to_numpy(bool)
    if valid.any():
        gross = pd.to_numeric(out.loc[valid, "policy_gross_bps"], errors="coerce").to_numpy(float)
        net = pd.to_numeric(out.loc[valid, "policy_net_bps"], errors="coerce").to_numpy(float)
        if not np.isfinite(gross).all() or not np.isfinite(net).all():
            raise AssertionError(f"{origin}: non-finite valid economics")
        if not np.allclose(gross - net, 100.0, rtol=0.0, atol=1e-8):
            raise AssertionError(f"{origin}: policy cost is not exactly once")
    availability = out.loc[valid, "policy_label_available_ts"] - out.loc[valid, "__decision_ts__"]
    if not availability.eq(pd.Timedelta(hours=12)).all():
        raise AssertionError(f"{origin}: labels do not resolve exactly at H12")
    return out


def _read_extension(root: Path) -> pd.DataFrame:
    paths = sorted((root / "policy_parts").rglob("policy_labels.parquet"))
    if not paths:
        raise FileNotFoundError(f"no policy label parts below {root}")
    pieces: list[pd.DataFrame] = []
    for path in paths:
        # The target-free rich-policy materialiser deliberately persists only
        # the outcome columns.  Its immutable candidate ID contains the three
        # identity fields required by the common policy ledger; reconstructing
        # them here is deterministic and does not consult price paths.
        part = pd.read_parquet(path)
        identity = part["candidate_id"].astype(str).str.rsplit("|", n=2, expand=True)
        if identity.shape[1] != 3 or identity.isna().any().any():
            raise AssertionError(f"{path}: malformed immutable candidate ID")
        part["__symbol__"] = identity.iloc[:, 0].to_numpy()
        part["side_name"] = identity.iloc[:, 1].to_numpy()
        # Candidate IDs are keyed by the completed signal-hour close; the
        # canonical policy enters at the next hourly decision open.
        part["__decision_ts__"] = pd.to_datetime(identity.iloc[:, 2], utc=True, errors="raise") + pd.Timedelta(hours=1)
        # This materialiser validates complete 15-minute H12 paths.  The
        # legacy column name is retained solely for schema compatibility; it
        # is never used to claim a one-minute source.
        part["label_source_complete_1m_path"] = part["policy_path_valid"].fillna(False).astype(bool)
        pieces.append(part)
    return _normalise(pd.concat(pieces, ignore_index=True), origin=str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-policy", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    incumbent = _normalise(pd.read_parquet(args.incumbent_policy, columns=REQUIRED), origin=str(args.incumbent_policy))
    extension = _read_extension(args.extension_root)
    incumbent_end = incumbent["__decision_ts__"].max()
    extension_start = extension["__decision_ts__"].min()
    if extension_start <= incumbent_end:
        raise AssertionError(f"extension starts {extension_start}, not after incumbent ends {incumbent_end}")
    combined = pd.concat([incumbent, extension], ignore_index=True)
    if combined.candidate_id.duplicated().any():
        raise AssertionError("combined ledger duplicates candidate IDs")
    combined = combined.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    args.out.mkdir(parents=True)
    target = args.out / "canonical_reconciled_policy_labels.parquet"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    combined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    monthly = combined.assign(month=combined["__decision_ts__"].dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
        rows=("candidate_id", "size"), valid_rows=("policy_path_valid", "sum")
    )
    monthly["valid_fraction"] = monthly["valid_rows"] / monthly["rows"].clip(lower=1)
    monthly.to_parquet(args.out / "coverage_by_month.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_policy_label_append_v1",
        "scope": "offline research-only immutable policy-ledger append; no model, live, admission, portfolio, or execution mutation",
        "incumbent_policy": str(args.incumbent_policy),
        "incumbent_sha256": _sha(args.incumbent_policy),
        "extension_root": str(args.extension_root),
        "extension_sha256": _sha(args.extension_root),
        "incumbent_rows": int(len(incumbent)),
        "extension_rows": int(len(extension)),
        "rows": int(len(combined)),
        "valid_rows": int(combined.policy_path_valid.sum()),
        "candidate_contract": "extension identities are fixed target-free before policy paths are read",
        "cost_contract": "gross minus net equals 100 bps exactly once for every valid row",
        "availability_contract": "valid policy labels resolve exactly H12 after decision timestamp",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out), "rows": int(len(combined)), "extension_rows": int(len(extension))}, sort_keys=True))


if __name__ == "__main__":
    main()
