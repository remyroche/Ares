#!/usr/bin/env python3
"""Replay a bounded strict Bayesian score blend through causal 21-day admission.

The input predictions are already strict-prequential.  This runner changes
only the score presented to the existing 21-day side-local EV map; it keeps the
candidate population, policy outcome, label-availability timestamps, and
admission threshold unchanged.  It is therefore the explicit test of whether
the correction is useful *as an admission-map input*, rather than as a sizing
overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec, apply_causal_21d_side_admission
from extreme_price_movements.strict_r3_canonical_current import apply_current_admission_by_geometry


TAILS = (.005, .01, .02, .05)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(mapped: pd.DataFrame, arm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = mapped.copy()
    work["year"] = work["__decision_ts__"].dt.year.astype(str)
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    work["week"] = work["__decision_ts__"].dt.strftime("%G-W%V")
    period_blocks = [("all", "all", work)]
    period_blocks.extend(("year", key, value) for key, value in work.groupby("year", sort=True))
    period_blocks.extend(("month", key, value) for key, value in work.groupby("month", sort=True))
    period_blocks.extend(("week", key, value) for key, value in work.groupby("week", sort=True))
    for period_scope, period, block in period_blocks:
        eligible = block.loc[
            block["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(block["causal_21d_side_expected_net_bps"], errors="coerce"))
        ].copy()
        valid = eligible.loc[
            eligible["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(eligible["policy_net_bps"], errors="coerce"))
        ].copy()
        def record(kind: str, selected: pd.DataFrame) -> None:
            selected_valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ]
            outcome = pd.to_numeric(selected_valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "period_scope": period_scope, "period": period, "kind": kind,
                "score_rows": len(block), "mapped_rows": int(block["causal_21d_side_expected_net_bps"].notna().sum()),
                "admitted_rows": len(eligible), "admission_rate": len(eligible) / max(1, len(block)),
                "selected_rows": len(selected), "valid_outcomes": len(selected_valid),
                "outcome_coverage": len(selected_valid) / max(1, len(selected)),
                "net_bps_per_trade": float(outcome.mean()) if len(outcome) else np.nan,
                "positive_rate": float(outcome.gt(0.0).mean()) if len(outcome) else np.nan,
            })
        record("all_admitted", eligible)
        for tail in TAILS:
            record(f"admitted_top_{tail:g}", eligible.nlargest(max(1, int(math.ceil(tail * len(eligible)))), "causal_21d_side_expected_net_bps", keep="first") if len(eligible) else eligible)
    return rows


def _matched_uplift(metrics: pd.DataFrame, *, control_arm: str, corrected_arm: str) -> pd.DataFrame:
    """Return corrected-minus-control metrics on identical score population periods.

    This evaluator intentionally permits the correction to change the causal
    EV map and therefore the admitted set.  Comparing standalone bps figures
    is misleading in that case: the decision artifact must make the matched
    changes explicit.  ``period`` and ``kind`` are shared population
    descriptors, while the selection itself may legitimately differ.
    """

    columns = [
        "period_scope", "period", "kind", "score_rows", "mapped_rows", "admitted_rows",
        "admission_rate", "selected_rows", "valid_outcomes",
        "outcome_coverage", "net_bps_per_trade", "positive_rate",
    ]
    control = metrics.loc[metrics["arm"].eq(control_arm), columns].copy()
    corrected = metrics.loc[metrics["arm"].eq(corrected_arm), columns].copy()
    merged = corrected.merge(
        control,
        on=["period_scope", "period", "kind"],
        how="inner",
        suffixes=("_corrected", "_control"),
        validate="one_to_one",
    )
    if len(merged) != len(control) or len(merged) != len(corrected):
        raise AssertionError("matched uplift lost an arm/period metric row")
    output = merged.loc[:, ["period_scope", "period", "kind", "score_rows_control"]].rename(
        columns={"score_rows_control": "score_rows"}
    )
    for metric in (
        "mapped_rows", "admitted_rows", "admission_rate", "selected_rows",
        "valid_outcomes", "outcome_coverage", "net_bps_per_trade", "positive_rate",
    ):
        output[f"control_{metric}"] = merged[f"{metric}_control"]
        output[f"corrected_{metric}"] = merged[f"{metric}_corrected"]
        output[f"uplift_{metric}"] = (
            pd.to_numeric(merged[f"{metric}_corrected"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_control"], errors="coerce")
        )
    return output


def _matched_selection_overlap(
    mapped: pd.DataFrame, *, control_arm: str, corrected_arm: str,
) -> pd.DataFrame:
    """Quantify whether an admission-map correction preserves selected sets.

    Large matched uplift accompanied by a nearly disjoint selected population
    is a fragile result.  We therefore persist exact candidate-ID overlap for
    the full admitted set and each map-ranked tail, using an explicit stable
    tie policy.  This is diagnostic only; it never affects scoring.
    """

    selected: dict[tuple[str, str, str, str], set[str]] = {}
    for arm, source in mapped.groupby("arm", sort=False):
        work = source.copy()
        work["year"] = work["__decision_ts__"].dt.year.astype(str)
        work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
        work["week"] = work["__decision_ts__"].dt.strftime("%G-W%V")
        period_blocks = [("all", "all", work)]
        period_blocks.extend(("year", key, value) for key, value in work.groupby("year", sort=True))
        period_blocks.extend(("month", key, value) for key, value in work.groupby("month", sort=True))
        period_blocks.extend(("week", key, value) for key, value in work.groupby("week", sort=True))
        for period_scope, period, block in period_blocks:
            eligible = block.loc[
                block["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(block["causal_21d_side_expected_net_bps"], errors="coerce"))
            ].sort_values(
                ["causal_21d_side_expected_net_bps", "candidate_id"],
                ascending=[False, True], kind="stable",
            )
            selected[(str(arm), period_scope, period, "all_admitted")] = set(eligible["candidate_id"].astype(str))
            for tail in TAILS:
                count = max(1, int(math.ceil(tail * len(eligible)))) if len(eligible) else 0
                selected[(str(arm), period_scope, period, f"admitted_top_{tail:g}")] = set(
                    eligible.head(count)["candidate_id"].astype(str)
                )
    rows: list[dict[str, object]] = []
    keys = sorted(
        (period_scope, period, kind)
        for arm, period_scope, period, kind in selected
        if arm == control_arm and (corrected_arm, period_scope, period, kind) in selected
    )
    for period_scope, period, kind in keys:
        control = selected[(control_arm, period_scope, period, kind)]
        corrected = selected[(corrected_arm, period_scope, period, kind)]
        union = control | corrected
        overlap = control & corrected
        rows.append({
            "period_scope": period_scope,
            "period": period,
            "kind": kind,
            "control_selected_rows": len(control),
            "corrected_selected_rows": len(corrected),
            "intersection_rows": len(overlap),
            "union_rows": len(union),
            "jaccard": len(overlap) / len(union) if union else np.nan,
            "control_retained_fraction": len(overlap) / len(control) if control else np.nan,
            "corrected_from_control_fraction": len(overlap) / len(corrected) if corrected else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--control-column", default="prequential_upstream")
    parser.add_argument("--bayes-rank-column", default="bayes_rank_traincdf")
    parser.add_argument(
        "--direct-corrected-column",
        default=None,
        help="Evaluate this already-constructed causal correction directly against --control-column.",
    )
    parser.add_argument(
        "--source-arm",
        default=None,
        help="Optional arm selector for a multi-arm correction prediction file.",
    )
    parser.add_argument("--demotion-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    # Read the source surface, rather than a narrow column projection: when
    # exact current producer lineage is available we must reuse the canonical
    # vintage-partitioned admission implementation.  Historical score files
    # without that lineage retain the generic evaluator only as a labelled
    # compatibility diagnostic.
    source = pd.read_parquet(args.predictions)
    required = {
        "candidate_id", "__decision_ts__", "side_name", "policy_path_valid",
        "policy_label_available_ts", "policy_net_bps", args.control_column,
    }
    if args.direct_corrected_column:
        required.add(args.direct_corrected_column)
    else:
        required.update((args.bayes_rank_column, "bayes_available"))
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"predictions missing {missing}")
    if args.source_arm:
        if "arm" not in source.columns:
            raise KeyError("--source-arm requires an arm column")
        source = source.loc[source["arm"].eq(args.source_arm)].copy()
        if source.empty:
            raise ValueError(f"source arm absent: {args.source_arm}")
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    source["policy_label_available_ts"] = pd.to_datetime(source["policy_label_available_ts"], utc=True, errors="raise")
    if source["candidate_id"].duplicated().any() or not source["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("candidate identity or long-only contract failed")
    if args.direct_corrected_column:
        source["bayes_blend"] = pd.to_numeric(source[args.direct_corrected_column], errors="raise")
    elif args.demotion_only:
        source["bayes_blend"] = source[args.control_column] + args.alpha * np.minimum(
            source[args.bayes_rank_column] - source[args.control_column], 0.0,
        )
    else:
        source["bayes_blend"] = (1.0 - args.alpha) * source[args.control_column] + args.alpha * source[args.bayes_rank_column]
    # The Bayesian model has an explicit warm-up.  Before then the blend is
    # exactly the control, preserving population and map provenance.
    if not args.direct_corrected_column:
        source.loc[~source["bayes_available"].fillna(False).astype(bool), "bayes_blend"] = source.loc[~source["bayes_available"].fillna(False).astype(bool), args.control_column]
    parts: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    control_arm = f"control_{args.control_column}"
    current_vintage_lineage = {
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256", "ev_score_family_id", "stack_is_prequential",
    }
    use_current_vintage = current_vintage_lineage.issubset(source.columns)
    if args.direct_corrected_column:
        # A direct correction has already encoded its own strength.  Do not
        # label it with this runner's unused --alpha default.
        identity = args.source_arm or str(args.direct_corrected_column)
        corrected_arm = f"bayes_admission_direct_{identity}"
    else:
        correction_kind = "demoter" if args.demotion_only else "blend"
        corrected_arm = f"bayes_admission_{correction_kind}_{args.alpha:.3f}"
    for arm, score in ((control_arm, args.control_column), (corrected_arm, "bayes_blend")):
        frame = source.copy()
        if use_current_vintage:
            # ``apply_current_admission_by_geometry`` accepts the canonical
            # score name.  Replace only its temporary input, preserving the
            # original score in ``score`` for transparent persisted output.
            frame["score"] = pd.to_numeric(frame[score], errors="raise")
            frame["final_score"] = frame["score"]
            mapped, audit = apply_current_admission_by_geometry(frame, geometry_mode="frozen")
            audit["mapping_contract"] = "current_exact_producer_vintage"
        else:
            frame = frame.loc[:, ["candidate_id", "__decision_ts__", "side_name", "policy_path_valid", "policy_label_available_ts", "policy_net_bps", score]].rename(columns={score: "score"})
            mapped, audit = apply_causal_21d_side_admission(
                frame, score_column="score", net_column="policy_net_bps", decision_column="__decision_ts__",
                label_available_column="policy_label_available_ts", identity_column="candidate_id",
                spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
            )
            audit["mapping_contract"] = "generic_score_domain_compatibility_only"
        if "score" not in mapped:
            mapped["score"] = pd.to_numeric(frame[score], errors="raise")
        mapped["arm"] = arm
        parts.append(mapped)
        audits.append(audit.assign(arm=arm))
        metrics.extend(_metrics(mapped, arm))
    args.out_dir.mkdir(parents=True)
    # Persist only admission decisions/expected values, never duplicate source features.
    admission = pd.concat(parts, ignore_index=True).loc[:, [
        "candidate_id", "__decision_ts__", "side_name", "arm", "score", "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "causal_21d_side_mapping_status", "causal_21d_side_reference_rows",
        "policy_path_valid", "policy_net_bps",
    ]]
    admission.to_parquet(args.out_dir / "admission_predictions.parquet", index=False, compression="zstd")
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_parquet(args.out_dir / "metrics.parquet", index=False)
    _matched_uplift(
        metrics_frame,
        control_arm=control_arm,
        corrected_arm=corrected_arm,
    ).to_parquet(args.out_dir / "matched_uplift.parquet", index=False)
    _matched_selection_overlap(
        pd.concat(parts, ignore_index=True),
        control_arm=control_arm,
        corrected_arm=corrected_arm,
    ).to_parquet(args.out_dir / "matched_selection_overlap.parquet", index=False)
    pd.concat(audits, ignore_index=True).to_parquet(args.out_dir / "admission_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_ledger_bayesian_admission_v1", "predictions": str(args.predictions), "predictions_sha256": _sha(args.predictions),
        "arms": [control_arm, corrected_arm], "alpha": args.alpha, "control_column": args.control_column, "bayes_rank_column": args.bayes_rank_column, "direct_corrected_column": args.direct_corrected_column, "source_arm": args.source_arm, "demotion_only": bool(args.demotion_only),
        "admission": (
            "current exact-producer-vintage Causal21dAdmissionSpec when lineage is available; otherwise "
            "generic compatibility mapper; prior resolved labels, common-bps EV >= 50, fail closed"
        ),
        "mapping_contract": "current_exact_producer_vintage" if use_current_vintage else "generic_score_domain_compatibility_only",
        "candidate_population": "identical long-only strict-prequential ledger", "score_change": "only bounded Bayesian blend passed into EV map", "strict_prequential": True,
        "primary_decision_artifacts": [
            "matched_uplift.parquet; corrected minus control on the same candidate population and period",
            "matched_selection_overlap.parquet; candidate-ID overlap of map-ranked admitted selections",
        ],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(admission)}))


if __name__ == "__main__":
    main()
