#!/usr/bin/env python3
"""Rescore persisted strict-R3 bundles under an explicit score-domain contract.

No fitted model is changed.  This producer reuses immutable monthly upstream
and four-week conversion bundles and recreates their target-free held/reference
frames.  It reports whether the resulting score is bit-identical to a supplied
baseline, but can also materialise a deliberate causal score-domain repair:
each held upstream vintage receives a CDF reference scored by that same
upstream bundle.
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
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    score_four_week_conversion_by_upstream_vintage,
    score_monthly_upstream_bundle,
)
from scripts.run_strict_r3_canonical_walkforward import (  # noqa: E402
    _attach_scores,
    _fields,
    _month_start,
    _read_target_free,
    _utc,
)


SCHEMA = "strict_r3_frozen_geometry_rescore_v2"
REFERENCE_DAYS = 42
CORE_SCORE_COLUMNS = (
    "base_score", "base_rank42", "base_anchor_bps",
    "conditional_consensus_rank", "upstream", "correctness_raw",
    "correctness_rank", "raw_correctness_demote", "final_score",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    final = _month_start(end - pd.Timedelta(nanoseconds=1))
    return list(pd.date_range(_month_start(start), final, freq="MS", inclusive="both"))


def _core_score_comparison(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    columns = ["candidate_id", *CORE_SCORE_COLUMNS]
    old = previous.loc[:, columns].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    new = current.loc[:, columns].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not old["candidate_id"].equals(new["candidate_id"]):
        raise AssertionError("rescore changed candidate identities")
    maximum: dict[str, float] = {}
    mean_absolute: dict[str, float] = {}
    spearman: dict[str, float] = {}
    bit_identical = True
    for column in CORE_SCORE_COLUMNS:
        left = pd.to_numeric(old[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(new[column], errors="coerce").to_numpy(float)
        diff = np.abs(left - right)
        same = np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
        bit_identical = bool(bit_identical and same)
        maximum[column] = float(np.nanmax(diff)) if len(left) else 0.0
        mean_absolute[column] = float(np.nanmean(diff)) if len(left) else 0.0
        valid = np.isfinite(left) & np.isfinite(right)
        spearman[column] = (
            float(pd.Series(left[valid]).corr(pd.Series(right[valid]), method="spearman"))
            if valid.sum() >= 2 else float("nan")
        )
    return {
        "core_scores_bit_identical": bit_identical,
        "max_abs_delta": maximum,
        "mean_abs_delta": mean_absolute,
        "spearman": spearman,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--bundle-run-dir", type=Path, required=True)
    parser.add_argument("--previous-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--require-core-parity", action="store_true",
        help="Fail if a requested rescore changes any core score column.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")

    start, end = _utc(args.evaluation_start), _utc(args.evaluation_end)
    fields = _fields(args.feature_contract)
    previous = pd.read_parquet(args.previous_predictions)
    if previous["candidate_id"].duplicated().any():
        raise ValueError("previous score ledger has duplicate candidate IDs")
    # Keep the source as a parquet path.  Loading the whole 120-field
    # multi-month panel before rescoring needlessly duplicates several hundred
    # MB alongside the previous score ledger and the per-block CDF reference.
    # `_read_target_free` already applies the exact timestamp predicate through
    # parquet filters, so block reads are semantically identical and far more
    # memory-stable.
    source: Path | pd.DataFrame = args.source_panel
    upstream_root = args.bundle_run_dir / "upstream_bundles"
    conversion_root = args.bundle_run_dir / "conversion_bundles"
    monthly_scores: list[pd.DataFrame] = []
    upstream_hashes: dict[str, str] = {}
    for month in _month_range(start, end):
        month_end = month + pd.offsets.MonthBegin(1)
        directory = upstream_root / f"month={month:%Y-%m}"
        bundle = load_monthly_upstream_bundle(directory)
        if pd.Timestamp(bundle.cutoff) != month:
            raise ValueError(f"monthly bundle cutoff mismatch: {directory}")
        held = _read_target_free(source, start=month, end=month_end, fields=fields)
        monthly_scores.append(score_monthly_upstream_bundle(bundle, held))
        upstream_hashes[month.strftime("%Y-%m")] = str(bundle.manifest["bundle_sha256"])
    upstream = pd.concat(monthly_scores, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    block_rows: list[dict[str, object]] = []
    directories = sorted(conversion_root.glob("cutoff=*"))
    if not directories:
        raise FileNotFoundError(f"no persisted conversion bundles under {conversion_root}")
    geometry_hashes: set[str] = set()
    for index, directory in enumerate(directories):
        token = directory.name.removeprefix("cutoff=")
        cutoff = pd.Timestamp(token, tz="UTC")
        if not start <= cutoff < end:
            continue
        bundle = load_four_week_conversion_bundle(directory)
        if pd.Timestamp(bundle.cutoff) != cutoff:
            raise ValueError(f"conversion bundle cutoff mismatch: {directory}")
        held_end = min(pd.Timestamp(bundle.end_exclusive), end)
        reference_raw = _read_target_free(
            source, start=cutoff - pd.Timedelta(days=REFERENCE_DAYS), end=cutoff, fields=fields,
        )
        held_raw = _read_target_free(source, start=cutoff, end=held_end, fields=fields)
        held_months = sorted(held_raw["__decision_ts__"].dt.strftime("%Y-%m").unique().tolist())
        block_upstreams = {
            month: load_monthly_upstream_bundle(upstream_root / f"month={month}")
            for month in held_months
        }
        scored, audit = score_four_week_conversion_by_upstream_vintage(
            bundle,
            reference=reference_raw,
            held=held_raw,
            upstream_bundles=block_upstreams,
        )
        prediction_parts.append(
            scored.loc[scored["__score_role__"].eq("held")].drop(columns="__score_role__"),
        )
        audit_parts.append(audit.assign(block_index=index))
        geometry_hashes.add(str(bundle.geometry.bundle_sha256))
        block_rows.append({
            "block_index": index,
            "cutoff": cutoff,
            "held_end_exclusive": held_end,
            "held_rows": int(len(held_raw)),
            "reference_rows": int(len(reference_raw)),
            "conversion_bundle_sha256": str(bundle.manifest["bundle_sha256"]),
            "geometry_bundle_sha256": str(bundle.geometry.bundle_sha256),
            "geometry_refit_cadence": "never",
            "reference_upstream_bundle_sha256_by_held_month": {
                month: str(model.manifest["bundle_sha256"])
                for month, model in block_upstreams.items()
            },
            "same_upstream_bundle_for_reference_and_held_per_producer": True,
            "status": "rescored_existing_immutable_bundle",
        })
    if len(geometry_hashes) != 1:
        raise ValueError(f"rescore saw non-frozen geometry identities: {sorted(geometry_hashes)}")
    final = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if final["candidate_id"].duplicated().any():
        raise AssertionError("rescore duplicated candidate IDs")
    final["base_rank"] = pd.to_numeric(final["base_rank42"], errors="coerce")
    final["consensus_rank"] = pd.to_numeric(final["conditional_consensus_rank"], errors="coerce")
    final["stack_is_prequential"] = True
    parity = _core_score_comparison(previous, final)
    if args.require_core_parity and not parity["core_scores_bit_identical"]:
        raise AssertionError("rescore changed core scores under --require-core-parity")
    args.out_dir.mkdir(parents=True)
    final.to_parquet(args.out_dir / "walkforward_predictions.parquet", index=False, compression="zstd")
    upstream.to_parquet(args.out_dir / "monthly_upstream_predictions.parquet", index=False, compression="zstd")
    pd.concat(audit_parts, ignore_index=True).to_parquet(
        args.out_dir / "conversion_reference_audit.parquet", index=False,
    )
    pd.DataFrame(block_rows).to_parquet(args.out_dir / "conversion_block_audit.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "source_bundle_run": str(args.bundle_run_dir),
        "source_bundle_run_manifest_sha256": _sha(args.bundle_run_dir / "run_manifest.json"),
        "previous_predictions": str(args.previous_predictions),
        "previous_predictions_sha256": _sha(args.previous_predictions),
        "source_panel": str(args.source_panel),
        "source_panel_sha256": _sha(args.source_panel),
        "feature_contract": str(args.feature_contract),
        "feature_contract_sha256": _sha(args.feature_contract),
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "rows": int(len(final)),
        "geometry_bundle_sha256": next(iter(geometry_hashes)),
        "geometry_refit_cadence": "never",
        "held_percentile_operations": 0,
        "outcomes_consumed_during_scoring": [],
        "output_contract_change": (
            "same-upstream-per-producer prior-42 CDF; raw K9 memberships remain absent"
        ),
        "core_score_parity": parity,
        "same_upstream_bundle_for_reference_and_held_per_producer": True,
        "upstream_bundle_hashes": upstream_hashes,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
