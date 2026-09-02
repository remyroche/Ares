#!/usr/bin/env python3
"""Create one full enhanced-base policy ledger with canonical overlap priority.

The newly expanded rich-policy labels cover the complete enhanced-base OOS
population.  The earlier canonical label union is retained verbatim whenever
an identity overlaps, because it is the historic outcome source behind the
current live-stack research metrics.  New-only rows preserve the frozen local
15-minute rich-policy materialisation.  This is outcome reconciliation only;
it cannot influence candidates, features, scores, or admission maps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY = (
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for file in files:
        digest.update(str(file).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    expanded = pd.read_parquet(args.expanded).copy()
    canonical = pd.read_parquet(args.canonical, columns=["candidate_id", *POLICY]).copy()
    if expanded.candidate_id.duplicated().any() or canonical.candidate_id.duplicated().any():
        raise ValueError("policy reconciliation requires unique candidate identities")
    canonical["policy_label_available_ts"] = pd.to_datetime(canonical["policy_label_available_ts"], utc=True, errors="coerce")
    merged = expanded.merge(canonical, on="candidate_id", how="left", suffixes=("", "__canonical"), validate="one_to_one")
    matched = merged["policy_path_valid__canonical"].notna()
    gross_delta = (
        pd.to_numeric(merged.loc[matched, "policy_gross_bps"], errors="coerce")
        - pd.to_numeric(merged.loc[matched, "policy_gross_bps__canonical"], errors="coerce")
    )
    for field in POLICY:
        canonical_field = f"{field}__canonical"
        merged.loc[matched, field] = merged.loc[matched, canonical_field].to_numpy()
        merged = merged.drop(columns=canonical_field)
    merged["policy_outcome_source"] = np.where(
        matched,
        "historical_canonical_rich_policy_overlap",
        merged["policy_outcome_source"].astype(str),
    )
    valid = merged["policy_path_valid"].fillna(False).astype(bool)
    if not np.allclose(
        pd.to_numeric(merged.loc[valid, "policy_gross_bps"], errors="coerce").to_numpy(float) - 100.0,
        pd.to_numeric(merged.loc[valid, "policy_net_bps"], errors="coerce").to_numpy(float),
        rtol=0.0, atol=1e-9,
    ):
        raise AssertionError("reconciled policy ledger does not apply the cost exactly once")
    args.out.mkdir(parents=True)
    ledger = args.out / "canonical_reconciled_policy_labels.parquet"
    merged.to_parquet(ledger, index=False, compression="zstd")
    audit = pd.DataFrame([{
        "expanded_rows": int(len(expanded)), "canonical_rows": int(len(canonical)),
        "overlap_rows": int(matched.sum()), "new_only_rows": int((~matched).sum()),
        "expanded_vs_canonical_gross_max_abs_bps": float(gross_delta.abs().max()),
        "expanded_vs_canonical_gross_p999_abs_bps": float(gross_delta.abs().quantile(.999)),
        "reconciled_valid_rows": int(valid.sum()),
    }])
    audit.to_parquet(args.out / "overlap_reconciliation_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_enhanced_base_policy_overlap_reconciliation_v1",
        "scope": "offline outcome-ledger reconciliation only; no candidate, score, model, EV map, or live mutation",
        "precedence": "historical canonical rich-policy outcome wins on exact candidate_id overlap; expanded frozen rich-policy labels fill new identities only",
        "expanded": {"path": str(args.expanded), "sha256": _sha(args.expanded)},
        "canonical": {"path": str(args.canonical), "sha256": _sha(args.canonical)},
        "ledger": str(ledger), "audit": audit.iloc[0].to_dict(),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out), **audit.iloc[0].to_dict()}))


if __name__ == "__main__":
    main()
