#!/usr/bin/env python3
"""Build immediate per-refit strict-R3 EV calibrators from OOS reserves.

Every input score is made by the newly fitted upstream/conversion producer on
a target-free calibration reserve that was excluded from *both* active
supervised fits.  Exact policy outcomes are joined only here, after score
generation.  The resulting artifact also freezes the complete reserve score
reference and a compact day x score-cell policy-net seed for canonical
equal-day 15%-trim admission from the producer's first live hour.  It is not a
cross-vintage raw-score bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    EVBridgeSpec,
    fit_strict_r3_ev_bridge,
    persist_strict_r3_ev_bridge,
)
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    CALIBRATION_RESERVE_DAYS,
)


GROUP_COLUMNS = (
    "ev_score_family_id", "geometry_bundle_sha256",
    "conversion_bundle_sha256", "upstream_bundle_sha256",
    "calibration_activation_ts",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_name(row: pd.Series) -> str:
    identity = "|".join(str(row[column]) for column in GROUP_COLUMNS)
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prior-bins", type=int, default=20)
    parser.add_argument("--prior-trim-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-residual-rows", type=int, default=20)
    parser.add_argument(
        "--reserve-days", type=int, default=CALIBRATION_RESERVE_DAYS,
        help="Exact same-producer calibration-reserve horizon in calendar days.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable immediate-calibration output already exists: {args.out_dir}")
    if args.prior_bins < 4:
        raise ValueError("immediate calibration needs at least four score bins")
    if not 0.0 <= args.prior_trim_fraction < 0.5:
        raise ValueError("immediate calibration trim fraction must be in [0, .5)")
    if args.reserve_days != CALIBRATION_RESERVE_DAYS:
        raise ValueError(
            f"canonical immediate calibration requires {CALIBRATION_RESERVE_DAYS} days"
        )
    reference = pd.read_parquet(args.reference_scores)
    outcome = pd.read_parquet(args.policy_outcomes)
    required_reference = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "calibration_reference_oos_to_all_active_fits", *GROUP_COLUMNS,
    }
    required_outcome = {
        "candidate_id", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts",
    }
    missing = sorted(required_reference.difference(reference.columns))
    if missing:
        raise ValueError(f"immediate calibration reference lacks: {missing}")
    missing = sorted(required_outcome.difference(outcome.columns))
    if missing:
        raise ValueError(f"immediate calibration policy outcomes lack: {missing}")
    if outcome["candidate_id"].duplicated().any():
        raise ValueError("immediate calibration requires unique policy-outcome identities")
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True, errors="raise")
    reference["calibration_activation_ts"] = pd.to_datetime(
        reference["calibration_activation_ts"], utc=True, errors="raise",
    )
    joined = reference.merge(
        outcome.loc[:, [
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_label_available_ts",
        ]],
        on="candidate_id", how="left", validate="many_to_one",
    )
    joined["policy_label_available_ts"] = pd.to_datetime(
        joined["policy_label_available_ts"], utc=True, errors="coerce",
    )
    reserve_start = joined["calibration_activation_ts"] - pd.to_timedelta(
        args.reserve_days, unit="D",
    )
    valid = (
        joined["calibration_reference_oos_to_all_active_fits"].fillna(False).astype(bool)
        & joined["__decision_ts__"].ge(reserve_start)
        & joined["__decision_ts__"].lt(joined["calibration_activation_ts"])
        & joined["policy_path_valid"].fillna(False).astype(bool)
        & joined["policy_label_available_ts"].lt(joined["calibration_activation_ts"])
    )
    joined = joined.loc[valid].copy()
    joined["stack_is_prequential"] = True
    args.out_dir.mkdir(parents=True)
    index_rows: list[dict[str, object]] = []
    for values, group in joined.groupby(list(GROUP_COLUMNS), observed=True, sort=True):
        group = group.copy()
        if group["candidate_id"].duplicated().any():
            raise ValueError("immediate calibration producer group has duplicate candidate IDs")
        record = dict(zip(GROUP_COLUMNS, values, strict=True))
        name = _bundle_name(pd.Series(record))
        directory = args.out_dir / "bundles" / f"producer={name}"
        lineage = {
            "conversion_bundle_sha256": str(record["conversion_bundle_sha256"]),
            "upstream_bundle_sha256": str(record["upstream_bundle_sha256"]),
        }
        try:
            bundle = fit_strict_r3_ev_bridge(
                group,
                fit_cutoff=record["calibration_activation_ts"],
                spec=EVBridgeSpec(
                    prior_bins=args.prior_bins,
                    prior_trim_fraction=args.prior_trim_fraction,
                    required_prior_rows_per_side=max(100, args.prior_bins * 4),
                    minimum_residual_rows=args.minimum_residual_rows,
                ),
                producer_lineage=lineage,
            )
            manifest = persist_strict_r3_ev_bridge(bundle, directory)
            status = "fitted_immediate_exact_producer_calibration"
            error = None
        except ValueError as exc:
            manifest = {}
            status = "insufficient_oos_reserve_support"
            error = str(exc)
        index_rows.append({
            **record,
            "producer_bundle_id": name,
            "reference_rows": int(len(group)),
            "reference_min_decision_ts": group["__decision_ts__"].min(),
            "reference_max_decision_ts": group["__decision_ts__"].max(),
            "reference_max_label_available_ts": group["policy_label_available_ts"].max(),
            "status": status,
            "error": error,
            "ev_bridge_bundle": None if not manifest else str(directory),
            "ev_bridge_bundle_sha256": manifest.get("bundle_sha256"),
        })
    index = pd.DataFrame(index_rows)
    index.to_parquet(args.out_dir / "immediate_calibration_index.parquet", index=False)
    manifest = {
        "schema": "strict_r3_immediate_calibration_reserve_v1",
        "reference_scores": str(args.reference_scores),
        "reference_scores_sha256": _sha(args.reference_scores),
        "policy_outcomes": str(args.policy_outcomes),
        "policy_outcomes_sha256": _sha(args.policy_outcomes),
        "candidate_rows_after_exact_oos_and_label_gate": int(len(joined)),
        "reference_window_days": int(args.reserve_days),
        "prior_bins": int(args.prior_bins),
        "prior_trim_fraction": float(args.prior_trim_fraction),
        "producer_groups": int(len(index)),
        "fitted_groups": int(index["status"].eq("fitted_immediate_exact_producer_calibration").sum()),
        "failure_groups": int(index["status"].ne("fitted_immediate_exact_producer_calibration").sum()),
        "contract": (
            "same exact upstream x conversion producer; calibration candidates "
            "were excluded from both active supervised fits; labels resolved "
            f"before producer activation; exact prior-{args.reserve_days}-calendar-day "
            "reserve; fixed reserve score cells and one equal-day "
            "policy-net seed per cell are persisted; no cross-vintage raw-score pooling"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
