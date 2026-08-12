#!/usr/bin/env python3
"""Replay a frozen strict-R3 common-bps EV bridge on a labelled OOF ledger.

The score ledger must already exist and contain only target-free scores joined
to policy outcomes *after* scoring.  This command cannot train a score model,
change a candidate identity, or consume unresolved labels as a reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    apply_strict_r3_ev_bridge,
    load_strict_r3_ev_bridge,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(frame: pd.DataFrame, *, group: str) -> pd.DataFrame:
    work = frame.copy()
    work["period"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.to_period(group).astype(str)
    valid = work["policy_path_valid"].fillna(False).astype(bool) if "policy_path_valid" in work else pd.Series(True, index=work.index)
    rows: list[dict[str, object]] = []
    for period, block in work.groupby("period", observed=True, sort=True):
        mapped = block["causal_21d_side_expected_net_bps"].notna()
        admitted = block["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
        eligible = block.loc[admitted & valid].copy()
        rows.append({
            "period": period,
            "scored_rows": int(len(block)),
            "mapped_rows": int(mapped.sum()),
            "admitted_rows": int(admitted.sum()),
            "admitted_valid_outcome_rows": int(len(eligible)),
            "admission_rate": float(admitted.mean()),
            "expected_net_bps_admitted": float(
                pd.to_numeric(eligible["causal_21d_side_expected_net_bps"], errors="coerce").mean()
            ) if len(eligible) else np.nan,
            "realised_net_bps_admitted": float(
                pd.to_numeric(eligible["policy_net_bps"], errors="coerce").mean()
            ) if len(eligible) else np.nan,
            "prior_only_rows": int(block["ev_bridge_residual_mapping_status"].eq(
                "bridge_prior_only_no_recent_residual_support",
            ).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--ev-bridge-bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable EV bridge replay output already exists: {args.out_dir}")
    bundle = load_strict_r3_ev_bridge(args.ev_bridge_bundle)
    ledger = pd.read_parquet(args.scored_label_ledger)
    mapped, audit = apply_strict_r3_ev_bridge(ledger, bundle=bundle)
    if len(mapped) != len(ledger) or mapped["candidate_id"].duplicated().any():
        raise AssertionError("EV bridge replay changed candidate identities")
    args.out_dir.mkdir(parents=True)
    mapped.to_parquet(args.out_dir / "bridge_admission_predictions.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "bridge_residual_audit.parquet", index=False, compression="zstd")
    monthly = _summary(mapped, group="M")
    weekly = _summary(mapped, group="W-MON")
    monthly.to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    weekly.to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_ev_bridge_replay_v1",
        "scored_label_ledger": str(args.scored_label_ledger),
        "scored_label_ledger_sha256": _sha(args.scored_label_ledger),
        "ev_bridge_bundle": str(args.ev_bridge_bundle),
        "ev_bridge_bundle_sha256": bundle.manifest.get("bundle_sha256"),
        "rows": int(len(mapped)),
        "mapped_rows": int(mapped["causal_21d_side_expected_net_bps"].notna().sum()),
        "admitted_rows": int(mapped["causal_21d_side_admitted_ge_50bps"].fillna(False).sum()),
        "admission_contract": (
            "frozen strict-OOF common-bps prior plus causal 21/42/84-day "
            "side-local policy-residual correction; no cross-vintage raw-score pooling"
        ),
        "score_mutation": "none; admission map only",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
