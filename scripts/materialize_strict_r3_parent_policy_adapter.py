#!/usr/bin/env python3
"""Version a complete source-aligned parent-policy contract for new consumers.

The parent materialisation already contains every economic and availability
field.  It predates only two provenance fields required by newer consumers:
the outcome-source tag and the exactly verified 100-bps one-time cost.  This
adapter refuses to infer either when the parent gross/net identity differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = (
    "candidate_id",
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_exit_reason",
    "policy_label_available_ts",
)
EXPECTED_COST_BPS = 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-policy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    frame = pd.read_parquet(args.parent_policy, columns=list(REQUIRED)).copy()
    if frame["candidate_id"].duplicated().any():
        raise ValueError("parent policy has duplicate candidate identities")
    frame["policy_path_valid"] = frame["policy_path_valid"].fillna(False).astype(bool)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce"
    )
    valid = frame["policy_path_valid"].to_numpy(bool)
    gross = pd.to_numeric(frame["policy_gross_bps"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    cost = gross - net
    if (~np.isfinite(cost[valid])).any():
        raise ValueError("valid parent-policy rows lack finite gross/net outcomes")
    if not np.allclose(cost[valid], EXPECTED_COST_BPS, rtol=0.0, atol=1e-9):
        raise ValueError("parent policy gross/net identity is not exactly 100 bps")
    if frame.loc[valid, "policy_label_available_ts"].isna().any():
        raise ValueError("valid parent-policy row lacks a label-availability timestamp")

    frame["policy_cost_bps"] = np.where(valid, EXPECTED_COST_BPS, np.nan)
    frame["policy_outcome_source"] = np.where(
        valid,
        "source_aligned_parent_simple_policy_15m",
        "unavailable",
    )
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "canonical_policy_contract.parquet"
    frame.to_parquet(output, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_source_aligned_parent_policy_adapter_v1",
        "purpose": "provenance-only compatibility adapter; no label or economic transformation",
        "parent_policy": str(args.parent_policy),
        "parent_policy_sha256": sha256(args.parent_policy),
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "cost_identity": {
            "gross_minus_net_bps": EXPECTED_COST_BPS,
            "verified_all_valid_rows": True,
            "cost_applied_once": True,
        },
        "added_provenance_columns": {
            "policy_cost_bps": "derived only after exact parent gross-net verification",
            "policy_outcome_source": "source_aligned_parent_simple_policy_15m",
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
