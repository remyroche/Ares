#!/usr/bin/env python3
"""Fit and score the canonical LDF on each strict-R3 conversion OOF block.

The LDF is a post-admission *relative sizing* layer.  It never changes the
strict-R3 score or the causal admission decision.  Every LDF bundle uses only
the preceding three months of policy labels that were resolved before its
four-week conversion cutoff, then scores the held block target-free.
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

from extreme_price_movements.strict_r3_n5_canonical import (  # noqa: E402
    load_n5_contract,
    persist_canonical_n5_bundle,
    score_canonical_n5_bundle,
    train_canonical_n5_bundle,
)


SCHEMA = "strict_r3_ldf_v4_walkforward_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_unique(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "candidate_id" not in frame or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} must contain unique candidate_id")
    return frame


def _prepare(
    scored_labels: pd.DataFrame,
    features: pd.DataFrame,
    admission: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    contract = load_n5_contract()
    fields = tuple(map(str, contract["features"]))
    required_labels = {
        "candidate_id", "__decision_ts__", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", "final_score",
        "base_rank", "consensus_rank", "base_anchor_bps", "stack_is_prequential",
        "geometry_bundle_sha256",
    }
    missing = sorted(required_labels.difference(scored_labels.columns))
    if missing:
        raise ValueError(f"scored label ledger lacks: {missing}")
    missing = sorted(set(fields).difference(features.columns))
    if missing:
        raise ValueError(f"LDF feature sidecar lacks frozen fields: {missing}")
    required_admission = {"candidate_id", "raw_expected_bps", "mapped_ev_available"}
    missing = sorted(required_admission.difference(admission.columns))
    if missing:
        raise ValueError(f"causal admission provenance lacks: {missing}")
    # Core score fields are deliberately present in both the immutable score
    # ledger and the causal sidecar.  Keep the score-ledger values (it is the
    # authoritative producer) and verify the sidecar exactly reproduces them;
    # blindly merging both would create ``_x/_y`` columns and silently make
    # the LDF contract ambiguous.
    shared_fields = [field for field in fields if field in scored_labels.columns]
    if shared_fields:
        check = scored_labels.loc[:, ["candidate_id", *shared_fields]].merge(
            features.loc[:, ["candidate_id", *shared_fields]],
            on="candidate_id", how="left", validate="one_to_one",
            suffixes=("__score", "__sidecar"),
        )
        for field in shared_fields:
            left = pd.to_numeric(check[f"{field}__score"], errors="coerce").to_numpy(float)
            right = pd.to_numeric(check[f"{field}__sidecar"], errors="coerce").to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError(f"LDF causal sidecar differs from authoritative score ledger: {field}")
    sidecar_only = [field for field in fields if field not in shared_fields]
    work = scored_labels.merge(
        features.loc[:, ["candidate_id", *sidecar_only]],
        on="candidate_id", how="inner", validate="one_to_one",
    ).merge(
        admission.loc[:, ["candidate_id", "raw_expected_bps", "mapped_ev_available"]],
        on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(work) != len(scored_labels):
        raise ValueError("LDF sidecars do not cover every scored prediction")
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True)
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True,
    )
    if not work["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("LDF walk-forward received a non-prequential score row")
    return work.sort_values(["__decision_ts__", "candidate_id"], kind="stable"), fields


def _target_free_input(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, ["candidate_id", "final_score", "raw_expected_bps", *fields]].copy()


def _fallback_output(frame: pd.DataFrame, *, cutoff: pd.Timestamp, reason: str) -> pd.DataFrame:
    output = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    output["n5_bundle_cutoff"] = cutoff
    output["n5_available"] = False
    output["n5_unavailable_reason"] = reason
    output["trust_size_multiplier"] = np.float32(1.0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--feature-sidecar", type=Path, required=True)
    parser.add_argument("--admission-provenance", type=Path, required=True)
    parser.add_argument("--conversion-block-audit", type=Path, required=True)
    parser.add_argument(
        "--geometry-mode", choices=("frozen", "episode-isolated"), default="frozen",
        help=(
            "frozen requires one canonical geometry identity; episode-isolated "
            "fits every LDF only on rows from the held block's exact geometry hash."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")

    scored = _read_unique(args.scored_label_ledger, "scored label ledger")
    features = _read_unique(args.feature_sidecar, "LDF feature sidecar")
    admission = _read_unique(args.admission_provenance, "causal admission provenance")
    work, fields = _prepare(scored, features, admission)
    blocks = pd.read_parquet(args.conversion_block_audit).copy()
    required_blocks = {
        "cutoff", "held_end_exclusive", "conversion_bundle_sha256",
        "geometry_bundle_sha256", "geometry_refit_cadence",
    }
    missing = sorted(required_blocks.difference(blocks.columns))
    if missing:
        raise ValueError(f"conversion block audit lacks: {missing}")
    blocks["cutoff"] = pd.to_datetime(blocks["cutoff"], utc=True)
    blocks["held_end_exclusive"] = pd.to_datetime(blocks["held_end_exclusive"], utc=True)
    blocks = blocks.sort_values("cutoff", kind="stable")
    if blocks["cutoff"].duplicated().any():
        raise ValueError("conversion block audit has duplicate cutoffs")
    if blocks["geometry_bundle_sha256"].isna().any():
        raise ValueError("conversion block audit has an empty geometry/K9 identity")
    if args.geometry_mode == "frozen":
        if blocks["geometry_bundle_sha256"].astype(str).nunique() != 1:
            raise ValueError("frozen LDF walk-forward requires one geometry/K9 identity")
        if not blocks["geometry_refit_cadence"].eq("never").all():
            raise ValueError("frozen LDF walk-forward rejects refitted geometry/K9")
    else:
        required_isolation = {
            "episode_start", "episode_end_exclusive", "downstream_training_geometry_scope",
            "definition_rows_excluded_from_downstream_training",
        }
        missing_isolation = sorted(required_isolation.difference(blocks.columns))
        if missing_isolation:
            raise ValueError(
                "episode-isolated LDF requires geometry isolation provenance: "
                f"{missing_isolation}",
            )
        if not blocks["geometry_refit_cadence"].eq("episode_boundary_only").all():
            raise ValueError("episode-isolated LDF received a non-episodic geometry cadence")
        if not blocks["downstream_training_geometry_scope"].eq(
            "same_geometry_bundle_only",
        ).all():
            raise ValueError("episode-isolated LDF requires same-geometry conversion provenance")
        if not blocks["definition_rows_excluded_from_downstream_training"].fillna(False).astype(bool).all():
            raise ValueError("episode-isolated LDF received geometry-definition training rows")
        if "correctness_status" not in blocks.columns:
            raise ValueError("episode-isolated LDF requires conversion correctness status")

    args.out_dir.mkdir(parents=True)
    bundle_root = args.out_dir / "bundles"
    outputs: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for row in blocks.itertuples(index=False):
        cutoff = pd.Timestamp(row.cutoff)
        held_end = pd.Timestamp(row.held_end_exclusive)
        held = work.loc[
            work["__decision_ts__"].ge(cutoff) & work["__decision_ts__"].lt(held_end)
        ].copy()
        if held.empty:
            raise ValueError(f"no scored rows for LDF held block {cutoff}")
        bundle_dir = bundle_root / f"cutoff={cutoff:%Y%m%d}"
        geometry_hash = str(row.geometry_bundle_sha256)
        if not held["geometry_bundle_sha256"].astype(str).eq(geometry_hash).all():
            raise ValueError("held LDF rows do not match their conversion geometry identity")
        if args.geometry_mode == "episode-isolated" and str(row.correctness_status) != "complete":
            # Do not impute a conversion model at a geometry boundary.  A
            # cold representation is a real availability state, and relative
            # sizing stays exactly one until the same K9 bundle has resolved
            # enough history to fit conversion correctness.
            output = _fallback_output(
                held, cutoff=cutoff, reason="same_geometry_conversion_warmup",
            )
            outputs.append(output)
            audit_rows.append({
                "cutoff": cutoff, "held_end_exclusive": held_end,
                "geometry_bundle_sha256": geometry_hash,
                "status": "unit_size_warmup",
                "reason": "same_geometry_conversion_warmup",
                "rows": len(held),
            })
            continue
        training_work = (
            work if args.geometry_mode == "frozen"
            else work.loc[work["geometry_bundle_sha256"].astype(str).eq(geometry_hash)].copy()
        )
        if args.geometry_mode == "episode-isolated":
            training_work = training_work.loc[
                pd.to_numeric(training_work["correctness_raw"], errors="coerce").notna()
            ].copy()
            if pd.to_numeric(held["correctness_raw"], errors="coerce").isna().any():
                raise ValueError("complete episodic conversion block has missing correctness_raw")
        try:
            bundle = train_canonical_n5_bundle(training_work, cutoff=cutoff, fields=fields)
        except ValueError as exc:
            # The earliest conversion blocks can precede the first three months
            # of fully current-v4 scored/admitted history.  They are an
            # explicit unit-size warm-up, never a backfilled or in-sample LDF.
            if "insufficient resolved prior support" not in str(exc):
                raise
            scored_output = _fallback_output(held, cutoff=cutoff, reason=str(exc))
            audit_rows.append({
                "cutoff": cutoff, "held_end_exclusive": held_end,
                "conversion_bundle_sha256": row.conversion_bundle_sha256,
                "geometry_bundle_sha256": geometry_hash,
                "geometry_mode": args.geometry_mode,
                "status": "unit_size_warmup", "held_rows": int(len(held)),
                "bundle_sha256": None, "reason": str(exc),
            })
        else:
            manifest = persist_canonical_n5_bundle(
                bundle, bundle_dir,
                source_hashes={
                    "scored_label_ledger": _sha(args.scored_label_ledger),
                    "feature_sidecar": _sha(args.feature_sidecar),
                    "admission_provenance": _sha(args.admission_provenance),
                    "conversion_block_audit": _sha(args.conversion_block_audit),
                },
            )
            scored_output = score_canonical_n5_bundle(
                bundle, _target_free_input(held, fields),
            ).rename(columns={"portfolio_size_multiplier": "trust_size_multiplier"})
            scored_output = held.loc[:, ["candidate_id", "__decision_ts__"]].merge(
                scored_output, on="candidate_id", how="left", validate="one_to_one",
            )
            scored_output["n5_available"] = True
            scored_output["n5_unavailable_reason"] = None
            audit_rows.append({
                "cutoff": cutoff, "held_end_exclusive": held_end,
                "conversion_bundle_sha256": row.conversion_bundle_sha256,
                "geometry_bundle_sha256": geometry_hash,
                "geometry_mode": args.geometry_mode,
                "status": "complete", "held_rows": int(len(held)),
                "bundle_sha256": manifest["bundle_sha256"], "reason": None,
            })
        if scored_output["candidate_id"].duplicated().any() or len(scored_output) != len(held):
            raise AssertionError("LDF scoring changed held candidate identities")
        outputs.append(scored_output)
        print(json.dumps({"event": "ldf_block_complete", **audit_rows[-1]}, default=str), flush=True)

    output = pd.concat(outputs, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any() or len(output) != len(work):
        raise AssertionError("LDF OOF output does not cover the scored population exactly once")
    output.to_parquet(args.out_dir / "ldf_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audit_rows).to_parquet(args.out_dir / "ldf_block_audit.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "contract": "config/strict_r3_ldf_support_v4.json",
        "contract_sha256": _sha(ROOT / "config/strict_r3_ldf_support_v4.json"),
        "side": "long",
        "geometry_bundle_sha256": (
            str(blocks["geometry_bundle_sha256"].iloc[0])
            if args.geometry_mode == "frozen"
            else sorted(blocks["geometry_bundle_sha256"].astype(str).unique().tolist())
        ),
        "geometry_mode": args.geometry_mode,
        "geometry_refit_cadence": (
            "never" if args.geometry_mode == "frozen" else "episode_boundary_only"
        ),
        "training": (
            "three prior months, strict label availability, post-admission top-30% score gate"
            if args.geometry_mode == "frozen" else
            "three prior months from the exact held geometry identity only, strict label "
            "availability, post-admission top-30% score gate"
        ),
        "scoring": "target-free LDF fields plus causal prior-resolved expected-net map",
        "ranking_changes": False,
        "admission_changes": False,
        "warmup": "explicit unit-size where a full current-v4 three-month history is unavailable",
        "source_hashes": {
            "scored_label_ledger": _sha(args.scored_label_ledger),
            "feature_sidecar": _sha(args.feature_sidecar),
            "admission_provenance": _sha(args.admission_provenance),
            "conversion_block_audit": _sha(args.conversion_block_audit),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}), flush=True)


if __name__ == "__main__":
    main()
