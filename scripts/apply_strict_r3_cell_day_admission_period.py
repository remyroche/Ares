#!/usr/bin/env python3
"""Apply canonical exact-producer Cell-day admission over a held period."""

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

from extreme_price_movements.strict_r3_cell_day_admission import (  # noqa: E402
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
    apply_cell_day_trim15_admission_snapshot,
)
from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    load_strict_r3_ev_bridge,
)


GROUP_COLUMNS = (
    "ev_score_family_id", "geometry_bundle_sha256",
    "conversion_bundle_sha256", "upstream_bundle_sha256",
    "calibration_activation_ts",
)

# This stage maps a frozen score to causal expected policy net.  It does not
# consume the upstream feature frame, so reading wide model fields here wastes
# memory and can make a reproducible map fail operationally on long history.
# Downstream trust fitting joins this compact provenance back by candidate ID.
COMPACT_SCORED_LEDGER_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "final_score",
    "stack_is_prequential", "policy_path_valid", "policy_net_bps",
    "policy_label_available_ts", *GROUP_COLUMNS,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(values: tuple[object, ...]) -> tuple[str, ...]:
    """Use canonical string lineage keys, including UTC activation time."""
    result: list[str] = []
    for column, value in zip(GROUP_COLUMNS, values, strict=True):
        if column == "calibration_activation_ts":
            timestamp = pd.Timestamp(value)
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
            )
            result.append(timestamp.isoformat())
        else:
            result.append(str(value))
    return tuple(result)


def _bundle_index(index: pd.DataFrame) -> dict[tuple[str, ...], Path]:
    fitted = index.loc[
        index["status"].eq("fitted_immediate_exact_producer_calibration")
    ].copy()
    missing = sorted(set(GROUP_COLUMNS).difference(fitted.columns))
    if missing:
        raise ValueError(f"immediate calibration index lacks producer lineage: {missing}")
    result: dict[tuple[str, ...], Path] = {}
    for values, block in fitted.groupby(list(GROUP_COLUMNS), observed=True, sort=True):
        if len(block) != 1:
            raise ValueError("immediate calibration index has duplicate fitted producer rows")
        raw = Path(str(block.iloc[0]["ev_bridge_bundle"]))
        result[_key(tuple(values))] = raw if raw.is_absolute() else ROOT / raw
    if not result:
        raise ValueError("immediate calibration index has no fitted producer bundle")
    return result


def _fail_closed_without_exact_reserve(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain a producer population when its first reserve has no OOS labels.

    The earliest producer can legitimately have no already-resolved OOS reserve.
    It remains in the target-free scored ledger, but has no authority to admit
    a trade.  This is deliberately distinct from dropping rows or borrowing a
    neighbouring producer's calibration curve.
    """
    output = rows.copy()
    output["causal_21d_side_expected_net_bps"] = float("nan")
    output["cell_day_fixed_score_cell"] = -1
    output["cell_day_retained_day_support"] = 0
    output["causal_21d_side_mapping_status"] = (
        "insufficient_exact_producer_reserve_fail_closed"
    )
    output["causal_21d_side_admitted_ge_50bps"] = False
    output["ev_mapping_score_family_id"] = output["ev_score_family_id"].astype(str)
    output["ev_mapping_geometry_bundle_sha256"] = output[
        "geometry_bundle_sha256"
    ].astype(str)
    output["ev_mapping_conversion_vintage"] = output[
        "conversion_bundle_sha256"
    ].astype(str)
    output["ev_mapping_upstream_vintage"] = output[
        "upstream_bundle_sha256"
    ].astype(str)
    output["ev_mapping_vintage_mode"] = "insufficient_exact_producer_reserve_fail_closed"
    output["ev_bridge_bundle_identity"] = "unavailable"
    output["ev_bridge_bundle"] = None
    audit = (
        output.assign(__day__=output["__decision_ts__"].dt.normalize())
        .groupby(["__day__", "side_name"], observed=True, sort=True)
        .size()
        .rename("current_rows")
        .reset_index()
        .rename(columns={"__day__": "snapshot_utc"})
    )
    audit["seed_cell_days"] = 0
    audit["dynamic_cell_days"] = 0
    audit["reference_max_label_available_ts"] = pd.NaT
    audit["strictly_prior_resolved"] = True
    audit["mapped_curve_min_bps"] = float("nan")
    audit["mapped_curve_max_bps"] = float("nan")
    audit["admission_floor_bps"] = 50.0
    audit["ev_bridge_bundle"] = None
    audit["mapping_status"] = "insufficient_exact_producer_reserve_fail_closed"
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--immediate-calibration-index", type=Path, required=True)
    parser.add_argument(
        "--allow-multi-producer", action="store_true",
        help=(
            "Apply each exact same-producer calibration bundle to its own "
            "held score family. Without this flag the historical single-bundle "
            "contract is preserved."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    frame = pd.read_parquet(
        args.scored_label_ledger,
        columns=list(dict.fromkeys(COMPACT_SCORED_LEDGER_COLUMNS)),
    )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True,
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError("scored label ledger has duplicate candidate IDs")
    index = pd.read_parquet(args.immediate_calibration_index)
    bundle_paths = _bundle_index(index)
    if not args.allow_multi_producer and len(bundle_paths) != 1:
        raise ValueError(
            "period admission requires one fitted exact-producer calibrator; "
            "pass --allow-multi-producer for an explicitly segregated replay"
        )
    missing = sorted(set(GROUP_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"scored label ledger lacks producer lineage: {missing}")
    mapped_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    unmatched = 0
    unsupported_rows = 0
    for values, producer_rows in frame.groupby(list(GROUP_COLUMNS), observed=True, sort=True):
        key = _key(tuple(values))
        bundle_path = bundle_paths.get(key)
        if bundle_path is None:
            if not args.allow_multi_producer:
                unmatched += len(producer_rows)
                continue
            mapped, audit = _fail_closed_without_exact_reserve(producer_rows)
            unsupported_rows += len(producer_rows)
            mapped_parts.append(mapped)
            audit_parts.append(audit)
            continue
        bundle = load_strict_r3_ev_bridge(bundle_path)
        # ``apply_cell_day_trim15_admission_snapshot`` needs only the compact
        # resolved score/outcome lineage below.  Passing the full multi-vintage
        # wide ledger for every day caused repeated 1.4m-row copies and did not
        # alter the map because the function immediately filters to this exact
        # producer.  Restricting before the daily loop is algebraically
        # identical and keeps memory bounded by one producer's history.
        resolved_columns = [
            "candidate_id", "__decision_ts__", "side_name", bundle.score_column,
            "ev_score_family_id", "geometry_bundle_sha256",
            "conversion_bundle_sha256", "upstream_bundle_sha256",
            "stack_is_prequential", bundle.net_column,
            "policy_label_available_ts",
        ]
        if "policy_path_valid" in producer_rows.columns:
            resolved_columns.append("policy_path_valid")
        resolved_columns = list(dict.fromkeys(resolved_columns))
        missing_resolved = sorted(set(resolved_columns).difference(producer_rows.columns))
        if missing_resolved:
            raise ValueError(
                "scored label ledger lacks compact resolved lineage: "
                f"{missing_resolved}"
            )
        producer_resolved = producer_rows.loc[:, resolved_columns].copy()
        for day, current in producer_rows.groupby(
            producer_rows["__decision_ts__"].dt.normalize(), sort=True,
        ):
            mapped, audit = apply_cell_day_trim15_admission_snapshot(
                resolved_score_ledger=producer_resolved,
                current_scores=current.copy(),
                bundle=bundle,
            )
            mapped["ev_bridge_bundle"] = str(bundle_path)
            audit["ev_bridge_bundle"] = str(bundle_path)
            mapped_parts.append(mapped)
            audit_parts.append(audit)
    if unmatched:
        raise ValueError(
            f"{unmatched} scored rows have no matching fitted same-producer calibrator"
        )
    mapped = pd.concat(mapped_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    audit = pd.concat(audit_parts, ignore_index=True)
    if len(mapped) != len(frame) or set(mapped["candidate_id"]) != set(frame["candidate_id"]):
        raise AssertionError("Cell-day admission changed held candidate identities")
    args.out_dir.mkdir(parents=True)
    mapped.to_parquet(
        args.out_dir / "score_and_cell_day_admission_provenance.parquet",
        index=False, compression="zstd",
    )
    audit.to_parquet(args.out_dir / "cell_day_admission_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_cell_day_trim15_period_admission_v1",
        "mapping": CELL_DAY_TRIM_15_CALIBRATION_MODE,
        "scored_label_ledger": str(args.scored_label_ledger),
        "scored_label_ledger_sha256": _sha(args.scored_label_ledger),
        "immediate_calibration_index": str(args.immediate_calibration_index),
        "immediate_calibration_index_sha256": _sha(args.immediate_calibration_index),
        "ev_bridge_bundles": sorted({str(path) for path in bundle_paths.values()}),
        "producer_bundle_count": int(len(bundle_paths)),
        "allow_multi_producer": bool(args.allow_multi_producer),
        "insufficient_exact_reserve_rows_fail_closed": int(unsupported_rows),
        "rows": int(len(mapped)),
        "days": int(mapped["__decision_ts__"].dt.normalize().nunique()),
        "admitted_rows": int(mapped["causal_21d_side_admitted_ge_50bps"].sum()),
        "strictly_prior_resolved": bool(audit["strictly_prior_resolved"].all()),
        "held_outcomes_used_for_same_day_admission": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
