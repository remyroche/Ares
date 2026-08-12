#!/usr/bin/env python3
"""Prove persisted lock-step strict-R3 bundle parity from target-free inputs.

The verifier rebuilds the preceding 28-day reserve and complete 28-day held
window from the canonical target-free source, scores both with the persisted
upstream/conversion pair, and compares them to the producer's stored score
outputs.  It consumes no outcomes, so it can be used as a training/replay/
inference parity gate before a bundle is promoted.
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

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    REFERENCE_DAYS,
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    score_four_week_conversion_bundle_lockstep,
    score_monthly_upstream_bundle,
)
from scripts.run_strict_r3_canonical_lockstep_walkforward import (  # noqa: E402
    _read_targetfree_source_columns,
)


COMPONENTS = (
    "base_score",
    "base_rank42",
    "base_anchor_bps",
    "conditional_consensus_rank",
    "upstream",
    "correctness_raw",
    "correctness_rank",
    "raw_correctness_demote",
    "final_score",
)


def _sha(path: Path) -> str:
    if path.is_dir():
        manifest = path / "run_manifest.json"
        if not manifest.exists():
            raise ValueError("target-free source store lacks run_manifest.json")
        return _sha(manifest)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"]["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("strict-R3 parity verifier requires the frozen 120-field long contract")
    return fields


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument(
        "--held-candidates", type=Path,
        help=(
            "Optional point-in-time held candidate ledger. Required when the "
            "held population is a current-spread-gated subset of the historical "
            "target-free source store."
        ),
    )
    parser.add_argument(
        "--held-features", type=Path,
        help="Candidate-keyed target-free held feature panel paired with --held-candidates.",
    )
    parser.add_argument(
        "--bundle-dir", type=Path, required=True,
        help="One persisted lock-step cutoff directory containing upstream/, conversion/, and scores/.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-hours", type=int, default=72)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable parity artifact exists: {args.out_dir}")
    if args.chunk_hours < 1 or args.tolerance < 0.0:
        raise ValueError("chunk-hours must be positive and tolerance non-negative")

    fields = _fields(args.feature_contract)
    conversion = load_four_week_conversion_bundle(args.bundle_dir / "conversion")
    upstream = load_monthly_upstream_bundle(args.bundle_dir / "upstream")
    cutoff = _utc(conversion.cutoff)
    if _utc(upstream.cutoff) != cutoff or _utc(upstream.end_exclusive) != _utc(conversion.end_exclusive):
        raise ValueError("persisted lock-step upstream/conversion bundle mismatch")
    stored_reference = pd.read_parquet(args.bundle_dir / "scores" / "reserve_target_free_scores.parquet")
    stored_reference["__score_role__"] = "reference"
    stored_held = pd.read_parquet(args.bundle_dir / "scores" / "held_target_free_scores.parquet")
    stored_held["__score_role__"] = "held"
    stored_held_ts = pd.to_datetime(stored_held["__decision_ts__"], utc=True, errors="raise")
    if stored_held_ts.empty or stored_held_ts.min() < cutoff:
        raise ValueError("stored held score file begins before the bundle cutoff")
    held_end = stored_held_ts.max() + pd.Timedelta(hours=1)
    scheduled_held_end = _utc(conversion.end_exclusive)
    if held_end > scheduled_held_end:
        raise ValueError("stored held score file extends beyond the persisted bundle window")
    reference_start = cutoff - pd.Timedelta(days=REFERENCE_DAYS)
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *fields]
    raw = _read_targetfree_source_columns(
        args.source_panel,
        start=reference_start,
        end=cutoff if args.held_candidates is not None else held_end,
        fields=fields,
    )
    reference = raw.loc[raw["__decision_ts__"].lt(cutoff)].copy()
    if (args.held_candidates is None) != (args.held_features is None):
        raise ValueError("--held-candidates and --held-features must be supplied together")
    if args.held_candidates is None:
        held = raw.loc[raw["__decision_ts__"].ge(cutoff)].copy()
        held_source_contract = "canonical target-free source store"
    else:
        candidates = pd.read_parquet(args.held_candidates)
        features = pd.read_parquet(args.held_features)
        if candidates["candidate_id"].duplicated().any() or features["candidate_id"].duplicated().any():
            raise ValueError("held candidate/features require unique candidate identities")
        feature_columns = [column for column in fields if column in features]
        missing_features = sorted(set(fields).difference(feature_columns))
        if missing_features:
            raise ValueError(f"held target-free feature panel lacks: {missing_features}")
        held = candidates.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        ]].merge(
            features.loc[:, ["candidate_id", *feature_columns]],
            on="candidate_id", how="inner", validate="one_to_one",
        )
        expected_held_ids = set(stored_held["candidate_id"].astype(str))
        held = held.loc[held["candidate_id"].astype(str).isin(expected_held_ids)].copy()
        if len(held) != len(stored_held) or set(held["candidate_id"].astype(str)) != expected_held_ids:
            raise ValueError("external held target-free inputs do not exactly cover stored held scores")
        held_source_contract = "explicit point-in-time held candidate and feature ledgers"
    if reference.empty or held.empty:
        raise ValueError("target-free source does not cover the complete reserve and held block")
    if reference["candidate_id"].duplicated().any() or held["candidate_id"].duplicated().any():
        raise ValueError("target-free source has duplicate candidate identities")

    keys = ["candidate_id", "__decision_ts__", "side_name"]
    reference_upstream = score_monthly_upstream_bundle(
        upstream,
        reference,
        allow_prior_reference=True,
        prior_reference_start=reference_start,
    )
    held_upstream = score_monthly_upstream_bundle(upstream, held)
    rescored, score_audit = score_four_week_conversion_bundle_lockstep(
        conversion,
        reference=reference.merge(reference_upstream, on=keys, validate="one_to_one"),
        held=held.merge(held_upstream, on=keys, validate="one_to_one"),
        chunk_hours=args.chunk_hours,
    )

    stored = pd.concat([stored_reference, stored_held], ignore_index=True)
    for frame in (stored, rescored):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if stored["candidate_id"].duplicated().any() or rescored["candidate_id"].duplicated().any():
        raise ValueError("parity comparison encountered duplicate stored or rescored identities")
    joined = stored.merge(
        rescored,
        on=["candidate_id", "__decision_ts__", "side_name", "__score_role__"],
        suffixes=("__stored", "__rescored"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    missing = joined["_merge"].ne("both")
    if missing.any():
        details = joined.loc[missing, ["candidate_id", "__decision_ts__", "__score_role__", "_merge"]]
        raise ValueError(f"stored and rescored identities differ: {details.head(5).to_dict('records')}")
    component_rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        left = pd.to_numeric(joined[f"{component}__stored"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined[f"{component}__rescored"], errors="coerce").to_numpy(float)
        delta = np.abs(left - right)
        component_rows.append({
            "component": component,
            "rows": int(len(delta)),
            "finite_rows": int(np.isfinite(delta).sum()),
            "max_abs_delta": float(np.nanmax(delta)),
            "mean_abs_delta": float(np.nanmean(delta)),
            "within_tolerance": bool(np.nanmax(delta) <= args.tolerance),
        })
    components = pd.DataFrame(component_rows)
    parity_pass = bool(components["within_tolerance"].all())
    args.out_dir.mkdir(parents=True)
    identity_columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
    reference.loc[:, identity_columns].to_parquet(
        args.out_dir / "same_model_reference_candidates.parquet",
        index=False, compression="zstd",
    )
    reference.loc[:, ["candidate_id", *fields]].to_parquet(
        args.out_dir / "same_model_reference_features.parquet",
        index=False, compression="zstd",
    )
    components.to_parquet(args.out_dir / "component_parity.parquet", index=False)
    summary = {
        "schema": "strict_r3_lockstep_forward_parity_v1",
        "parity_pass": parity_pass,
        "tolerance": float(args.tolerance),
        "cutoff": cutoff.isoformat(),
        "held_end_exclusive": held_end.isoformat(),
        "scheduled_held_end_exclusive": scheduled_held_end.isoformat(),
        "reference_start": reference_start.isoformat(),
        "reference_rows": int(len(reference)),
        "held_rows": int(len(held)),
        "held_source_contract": held_source_contract,
        "reference_inputs_persisted_target_free": True,
        "stored_rows": int(len(stored)),
        "rescored_rows": int(len(rescored)),
        "components": components.to_dict(orient="records"),
        "score_audit": score_audit.to_dict(orient="records"),
        "contract": (
            "target-free source only; one persisted lock-step upstream/conversion pair; "
            "same preceding-28-day CDF; no outcome joins or held-window percentiles"
        ),
    }
    (args.out_dir / "parity_audit.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    manifest = {
        "schema": "strict_r3_lockstep_forward_parity_manifest_v1",
        "source_panel": str(args.source_panel),
        "source_panel_sha256": _sha(args.source_panel),
        "held_candidates": str(args.held_candidates) if args.held_candidates else None,
        "held_candidates_sha256": _sha(args.held_candidates) if args.held_candidates else None,
        "held_features": str(args.held_features) if args.held_features else None,
        "held_features_sha256": _sha(args.held_features) if args.held_features else None,
        "feature_contract": str(args.feature_contract),
        "feature_contract_sha256": _sha(args.feature_contract),
        "bundle_dir": str(args.bundle_dir),
        "upstream_bundle_sha256": upstream.manifest["bundle_sha256"],
        "conversion_bundle_sha256": conversion.manifest["bundle_sha256"],
        "geometry_bundle_sha256": conversion.geometry.bundle_sha256,
        "outcomes_consumed": [],
        "parity_pass": parity_pass,
        "same_model_reference_candidates": "same_model_reference_candidates.parquet",
        "same_model_reference_features": "same_model_reference_features.parquet",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **summary}, default=str))
    if not parity_pass:
        raise SystemExit("strict-R3 lock-step forward parity failed")


if __name__ == "__main__":
    main()
