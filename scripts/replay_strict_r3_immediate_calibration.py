#!/usr/bin/env python3
"""Replay immediate exact-producer calibration on strict-R3 OOF held rows.

Each held row is mapped only by an artifact built from the matching upstream ×
conversion producer's OOS calibration reserve.  This proves first-hour
availability at a refit boundary; it does not borrow a raw score map from a
different producer.
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


KEYS = (
    "ev_score_family_id", "geometry_bundle_sha256",
    "conversion_bundle_sha256", "upstream_bundle_sha256",
    # The same serialized producer pair should normally have one activation,
    # but retain the explicit boundary in the identity.  It prevents a future
    # cache/reuse implementation from applying an older reserve map to a
    # later deployment of an otherwise identical model artifact.
    "calibration_activation_ts",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _period_metrics(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    work = frame.copy()
    work["period"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.to_period(frequency).astype(str)
    valid = work["policy_path_valid"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    for period, block in work.groupby("period", observed=True, sort=True):
        mapped = block["causal_21d_side_expected_net_bps"].notna()
        admitted = block["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
        selected = block.loc[valid & admitted].copy()
        rows.append({
            "period": period,
            "scored_rows": int(len(block)),
            "mapped_rows": int(mapped.sum()),
            "admitted_rows": int(admitted.sum()),
            "admission_rate": float(admitted.mean()),
            "admitted_net_bps_per_trade": float(
                pd.to_numeric(selected["policy_net_bps"], errors="coerce").mean()
            ) if len(selected) else np.nan,
            "admitted_expected_net_bps_per_trade": float(
                pd.to_numeric(selected["causal_21d_side_expected_net_bps"], errors="coerce").mean()
            ) if len(selected) else np.nan,
            "prior_only_rows": int(selected["ev_bridge_residual_mapping_status"].eq(
                "bridge_prior_only_no_recent_residual_support",
            ).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--immediate-calibration-index", type=Path, required=True)
    parser.add_argument(
        "--frozen-base-availability", type=Path,
        help=(
            "Decision-time raw availability sidecar produced by "
            "materialize_strict_r3_frozen_base_availability.py. When present, "
            "apply the same fail-closed executable gate as the forward scorer."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable immediate-calibration replay exists: {args.out_dir}")
    ledger = pd.read_parquet(args.scored_label_ledger)
    availability: pd.DataFrame | None = None
    if args.frozen_base_availability is not None:
        availability = pd.read_parquet(args.frozen_base_availability)
        required_availability = {
            "candidate_id", "frozen_base_feature_count",
            "frozen_base_feature_fraction", "frozen_base_contract_complete",
        }
        missing = sorted(required_availability.difference(availability.columns))
        if missing:
            raise ValueError(f"frozen base availability lacks: {missing}")
        if availability["candidate_id"].duplicated().any():
            raise ValueError("frozen base availability has duplicate identities")
        ledger = ledger.merge(
            availability.loc[:, sorted(required_availability)],
            on="candidate_id", how="left", validate="one_to_one",
        )
        if ledger["frozen_base_contract_complete"].isna().any():
            raise ValueError("frozen base availability does not cover every held identity")
    index = pd.read_parquet(args.immediate_calibration_index)
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("held score ledger requires unique candidate IDs")
    required_index = {*KEYS, "status", "ev_bridge_bundle"}
    missing = sorted(required_index.difference(index.columns))
    if missing:
        raise ValueError(f"immediate calibration index lacks: {missing}")
    fitted = index.loc[index["status"].eq("fitted_immediate_exact_producer_calibration")].copy()
    if fitted.duplicated(list(KEYS)).any():
        raise ValueError("immediate calibration index has duplicate producer mappings")
    merged = ledger.merge(
        fitted.loc[:, [*KEYS, "ev_bridge_bundle", "producer_bundle_id"]],
        on=list(KEYS), how="left", validate="many_to_one",
    )
    parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for path, group in merged.dropna(subset=["ev_bridge_bundle"]).groupby(
        "ev_bridge_bundle", observed=True, sort=True,
    ):
        bundle = load_strict_r3_ev_bridge(Path(str(path)))
        mapped, audit = apply_strict_r3_ev_bridge(group.drop(columns=["ev_bridge_bundle"]), bundle=bundle)
        parts.append(mapped)
        audit["producer_bundle_id"] = group["producer_bundle_id"].iloc[0]
        audit_parts.append(audit)
    if not parts:
        raise ValueError("no held producer has an immediate calibration artifact")
    mapped = pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if mapped["candidate_id"].duplicated().any():
        raise AssertionError("immediate calibration replay duplicated held identities")
    if availability is not None:
        # Preserve the economic map decision for audit, then apply exactly the
        # forward executable gate.  Scores remain in the ledger for coverage
        # analysis, but no candidate with a missing frozen raw base field can
        # consume an admission or portfolio slot.
        mapped["ev_map_admitted_ge_floor_before_feature_gate"] = mapped[
            "causal_21d_side_admitted_ge_50bps"
        ].fillna(False).astype(bool)
        incomplete = ~mapped["frozen_base_contract_complete"].fillna(False).astype(bool)
        mapped["admission_rejection_reason"] = np.where(
            incomplete, "frozen_base_contract_incomplete", "",
        )
        mapped.loc[incomplete, "causal_21d_side_admitted_ge_50bps"] = False
    missing_calibrator = ledger.loc[~ledger["candidate_id"].isin(mapped["candidate_id"])].copy()
    args.out_dir.mkdir(parents=True)
    mapped.to_parquet(args.out_dir / "immediate_calibration_predictions.parquet", index=False, compression="zstd")
    pd.concat(audit_parts, ignore_index=True).to_parquet(
        args.out_dir / "immediate_calibration_residual_audit.parquet", index=False,
    )
    missing_calibrator.to_parquet(
        args.out_dir / "unmapped_held_rows.parquet", index=False, compression="zstd",
    )
    _period_metrics(mapped, "M").to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    _period_metrics(mapped, "W-MON").to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_immediate_calibration_replay_v1",
        "scored_label_ledger": str(args.scored_label_ledger),
        "scored_label_ledger_sha256": _sha(args.scored_label_ledger),
        "immediate_calibration_index": str(args.immediate_calibration_index),
        "immediate_calibration_index_sha256": _sha(args.immediate_calibration_index),
        "held_rows": int(len(ledger)),
        "immediately_calibrated_rows": int(len(mapped)),
        "unmapped_no_exact_oos_reserve_calibrator_rows": int(len(missing_calibrator)),
        "admitted_rows": int(mapped["causal_21d_side_admitted_ge_50bps"].fillna(False).sum()),
        "frozen_base_availability": (
            str(args.frozen_base_availability)
            if args.frozen_base_availability is not None else None
        ),
        "frozen_base_availability_sha256": (
            _sha(args.frozen_base_availability)
            if args.frozen_base_availability is not None else None
        ),
        "feature_complete_rows": (
            int(mapped["frozen_base_contract_complete"].fillna(False).sum())
            if availability is not None else None
        ),
        "feature_incomplete_rows": (
            int((~mapped["frozen_base_contract_complete"].fillna(False).astype(bool)).sum())
            if availability is not None else None
        ),
        "contract": (
            "each scored held row uses only its matching immediate upstream x "
            "conversion calibration bundle; reserve labels were joined after "
            "target-free scoring and resolved before activation"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
