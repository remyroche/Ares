#!/usr/bin/env python3
"""Matched strict-R3 trust-sizing comparison: LDF versus empirical Bayes.

This is deliberately a *sizing-only* experiment.  Candidate ranking and the
causal 21-day EV admission map remain frozen.  It compares:

* the current LDF N5 support/risk shrinker on the newly selected contract;
* the historical binned empirical-Bayes B5 shrinker on its original inputs;
* the same B5 specification on original plus newly selected causal inputs.

All arms use the same three-month prequential fit / three-month held blocks.
Raw Geometry/K9 membership slots are forbidden.  The caller must supply
context rows before the evaluation interval so the first fold has genuinely
prior resolved training labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    TrustModelSpec,
    causal_size_multiplier,
    catalogue,
    fit_trust_model,
    sizing_quality,
)
from scripts import run_strict_r3_trust_sizing_ablation as legacy  # noqa: E402


SEED = 20260811
RAW_K9_PREFIX = "k09__cluster_"
IDENTITY = (
    "candidate_id", "__decision_ts__", "__symbol__", "side_name",
    "geometry_bundle_sha256", "policy_path_valid", "policy_label_available_ts",
    "policy_net_bps", "policy_gross_bps", "policy_exit_reason", "final_score",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _requested_previous(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = payload.get("source_audit", {}).get("eligible_feature_names", [])
    if len(fields) < 20:
        raise ValueError("previous Bayesian manifest lacks its frozen eligible feature contract")
    return list(dict.fromkeys(map(str, fields)))


def _requested_selected(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = payload.get("compact_fields", [])
    if len(fields) < 12:
        raise ValueError("selected feature contract is unexpectedly small")
    return list(dict.fromkeys(map(str, fields)))


def _requested_proposed(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = payload.get("mda_proposed_fields", [])
    if len(fields) < 12:
        raise ValueError("MDA proposal is unexpectedly small")
    return list(dict.fromkeys(map(str, fields)))


def _schema(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def _current_trust_overlay_fields(columns: set[str]) -> list[str]:
    """All non-posterior causal inputs used by the current trust overlay.

    This deliberately admits support/OOD, frozen K9 summaries and history,
    leaf state, recent reliability, committee agreement, and continuous
    regime/relationship-break state.  It excludes the nine raw K9 posterior
    coordinates, distances/confidences, outcomes, and identities.
    """

    prefixes = (
        "k9_", "leaf_", "cluster_", "reliability_", "residual_heads_",
        "continuous_regime__",
    )
    direct = {
        "base_score", "base_rank", "base_anchor_bps", "consensus_rank",
        "final_score", "upstream", "residual_rank", "severe200_probability",
    }
    output = [
        field for field in sorted(columns)
        if (
            field in direct or field.startswith(prefixes)
        ) and not field.startswith(RAW_K9_PREFIX)
    ]
    if len(output) < 20:
        raise ValueError("current trust-overlay surface has too few non-posterior inputs")
    return output


def _load(
    context_surface: Path,
    evaluation_surface: Path,
    requested: Sequence[str],
    *,
    context_start: pd.Timestamp | None,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    common = _schema(context_surface).intersection(_schema(evaluation_surface))
    absent = sorted(set(requested).difference(common))
    fields = [field for field in requested if field in common]
    required = list(dict.fromkeys([*IDENTITY, *fields]))
    missing = sorted(set(IDENTITY).difference(common))
    if missing:
        raise KeyError(f"surfaces lack required causal ledger columns: {missing}")
    context = pd.read_parquet(context_surface, columns=required)
    evaluation = pd.read_parquet(evaluation_surface, columns=required)
    for value in (context, evaluation):
        value["__decision_ts__"] = pd.to_datetime(value["__decision_ts__"], utc=True, errors="raise")
    if context_start is not None:
        context = context.loc[context["__decision_ts__"].ge(context_start)]
    context = context.loc[context["__decision_ts__"].lt(evaluation_start)]
    evaluation = evaluation.loc[
        evaluation["__decision_ts__"].ge(evaluation_start)
        & evaluation["__decision_ts__"].lt(evaluation_end)
    ]
    parts = [context, evaluation]
    frame = pd.concat(parts, ignore_index=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("context and evaluation candidate identities overlap")
    if frame["geometry_bundle_sha256"].nunique(dropna=False) != 1:
        raise AssertionError("comparison requires exactly one frozen Geometry/K9 identity")
    raw = [field for field in fields if field.startswith(RAW_K9_PREFIX)]
    if raw:
        raise AssertionError(f"raw K9 membership is prohibited: {raw}")
    admitted, admission = apply_causal_21d_side_admission(
        frame,
        score_column="final_score",
        net_column="policy_net_bps",
        decision_column="__decision_ts__",
        label_available_column="policy_label_available_ts",
        identity_column="candidate_id",
        spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    admitted["raw_expected_bps"] = pd.to_numeric(
        admitted["causal_21d_side_expected_net_bps"], errors="coerce",
    )
    admitted["mapped_ev_available"] = admitted["raw_expected_bps"].notna()
    return admitted, fields, {
        "combined_rows": int(len(frame)),
        "admission_audit_rows": int(len(admission)),
        "requested_fields": int(len(requested)),
        "available_fields": int(len(fields)),
        "absent_from_both_surfaces": absent,
        "geometry_bundle_sha256": str(frame["geometry_bundle_sha256"].iloc[0]),
    }


def _eligible(train: pd.DataFrame, fields: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for field in fields:
        values = pd.to_numeric(train[field], errors="coerce")
        if values.notna().mean() >= 0.90 and values.var() > 1e-12:
            selected.append(field)
    if len(selected) < 12:
        raise ValueError(f"train-only coverage/variance gate left too few fields: {len(selected)}")
    return selected


def _blocks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        held_end = min(cursor + pd.DateOffset(months=3), end)
        result.append((cursor, held_end))
        cursor = held_end
    return result


def _run_arm(
    frame: pd.DataFrame,
    requested: Sequence[str],
    spec: TrustModelSpec,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_cap: int,
    multiplier_reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for fold, (cutoff, held_end) in enumerate(_blocks(start, end)):
        train_start = cutoff - pd.DateOffset(months=3)
        train_all = frame.loc[
            frame["__decision_ts__"].ge(train_start)
            & frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["mapped_ev_available"].astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        held = frame.loc[
            frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)
        ].copy()
        if len(train_all) < 2_000 or held.empty:
            raise ValueError(f"insufficient strict support at {cutoff}: {len(train_all)}/{len(held)}")
        parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
        train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
        held["parent_expected_bps"] = parent.predict(held["final_score"])
        floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(floor)].copy()
        train = legacy._sample_equal_month(train, int(train_cap))
        fields = _eligible(train, requested)
        held["trust_gate_active"] = (
            held["mapped_ev_available"].astype(bool)
            & pd.to_numeric(held["final_score"], errors="coerce").ge(floor)
        )
        if multiplier_reference == "top30_fit":
            train_prediction, held_prediction, model_audit = fit_trust_model(train, held, fields, spec)
            reference_quality = sizing_quality(train_prediction, train, spec.sizing_mode)
        elif multiplier_reference == "all_train_population":
            # The model is still fitted only on the score-qualified top-30%
            # rows.  This additional prediction pass is label-free and gives
            # the multiplier a reference distribution that contains the
            # actionable held tail rather than artificially forcing it into
            # the top of an already-truncated calibration population.
            score_population = pd.concat([train_all, held], ignore_index=True)
            train_prediction, score_prediction, model_audit = fit_trust_model(
                train, score_population, fields, spec,
            )
            reference_prediction = score_prediction.as_frame().iloc[: len(train_all)].reset_index(drop=True)
            held_prediction = score_prediction.as_frame().iloc[len(train_all) :].reset_index(drop=True)
            # Reconstruct the lightweight prediction interface accepted by
            # sizing_quality without carrying labels into the scoring call.
            from extreme_price_movements.trust_sizing_ablation import TrustPrediction
            def restore(value: pd.DataFrame) -> TrustPrediction:
                return TrustPrediction(
                    value["posterior_expected_bps"].to_numpy(float),
                    value["shrinkage_lambda"].to_numpy(float),
                    value["posterior_predictive_sd"].to_numpy(float),
                    value["posterior_mean_sd"].to_numpy(float),
                    value["posterior_predictive_q10"].to_numpy(float),
                    value["posterior_predictive_q50"].to_numpy(float),
                    value["posterior_predictive_q90"].to_numpy(float),
                    value["p_ev_positive"].to_numpy(float),
                    value["p_adverse_tail"].to_numpy(float),
                    value["trust_effective_support"].to_numpy(float),
                )
            reference_quality = sizing_quality(restore(reference_prediction), train_all, spec.sizing_mode)
            held_prediction = restore(held_prediction)
        else:
            raise ValueError(f"unknown multiplier reference: {multiplier_reference}")
        held_quality = sizing_quality(held_prediction, held, spec.sizing_mode)
        multiplier = causal_size_multiplier(reference_quality, held_quality)
        multiplier = np.where(held["trust_gate_active"].to_numpy(bool), multiplier, 1.0)
        def train_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
            ordered = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
            if len(ordered) < 100:
                return np.full(len(values), 0.5, dtype=np.float32)
            return (np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / len(ordered)).astype(np.float32)

        output = held.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score",
            "policy_path_valid", "policy_label_available_ts", "policy_gross_bps", "policy_net_bps", "policy_exit_reason",
            "geometry_bundle_sha256", "raw_expected_bps", "parent_expected_bps", "trust_gate_active",
        ]].copy().reset_index(drop=True)
        output = pd.concat([output, held_prediction.as_frame().reset_index(drop=True)], axis=1)
        output["posterior_expected_rank_train"] = train_cdf(
            train_prediction.expected_bps, held_prediction.expected_bps,
        )
        output["posterior_adverse_rank_train"] = train_cdf(
            train_prediction.p_adverse_tail, held_prediction.p_adverse_tail,
        )
        output["trust_size_multiplier"] = multiplier.astype(np.float32)
        output["arm"] = spec.name
        parts.append(output)
        audit.append({
            "arm": spec.name, "fold": fold, "train_start": train_start,
            "train_end_exclusive": cutoff, "held_start": cutoff, "held_end_exclusive": held_end,
            "train_rows_before_top30": len(train_all), "train_rows": len(train),
            "held_rows": len(held), "train_score_floor": floor, "field_count": len(fields),
            "fields": fields, "held_active_fraction": float(held["trust_gate_active"].mean()),
            "train_geometry_bundles": int(train["geometry_bundle_sha256"].nunique()),
            "held_geometry_bundles": int(held["geometry_bundle_sha256"].nunique()),
            "raw_k9_memberships_used": False, **{k: v for k, v in model_audit.items() if k != "selected_edges"},
            "selected_edges": model_audit.get("selected_edges", []),
            "multiplier_reference": multiplier_reference,
        })
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-surface", type=Path, required=True)
    parser.add_argument("--evaluation-surface", type=Path, required=True)
    parser.add_argument("--previous-bayesian-manifest", type=Path, required=True)
    parser.add_argument("--enhanced-contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--context-start", default=None, help="Optional inclusive UTC start for causal context rows.")
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument(
        "--multiplier-reference", choices=("top30_fit", "all_train_population"),
        default="top30_fit",
        help="Causal population used to percentile-calibrate the sizing multiplier.",
    )
    parser.add_argument(
        "--include-current-trust-overlay-fields", action="store_true",
        help=(
            "Add all causal non-posterior support/OOD/K9-history/leaf/reliability/"
            "committee/continuous-regime inputs to a matched Bayesian shrinkage arm."
        ),
    )
    parser.add_argument(
        "--only-arms", default=None,
        help="Optional comma-separated TrustModelSpec names for a narrow, reproducible retest.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start = pd.Timestamp(args.eval_start, tz="UTC")
    end = pd.Timestamp(args.eval_end, tz="UTC")
    context_start = pd.Timestamp(args.context_start, tz="UTC") if args.context_start else None
    previous = _requested_previous(args.previous_bayesian_manifest)
    enhanced = _requested_selected(args.enhanced_contract)
    proposed = _requested_proposed(args.enhanced_contract)
    source_columns = _schema(args.context_surface).intersection(_schema(args.evaluation_surface))
    trust_overlay = (
        _current_trust_overlay_fields(source_columns)
        if args.include_current_trust_overlay_fields else []
    )
    requested = list(dict.fromkeys([*previous, *enhanced, *proposed, *trust_overlay]))
    frame, available, load_audit = _load(
        args.context_surface, args.evaluation_surface, requested,
        context_start=context_start, evaluation_start=start, evaluation_end=end,
    )
    previous_available = [field for field in previous if field in available]
    enhanced_available = [field for field in enhanced if field in available]
    proposed_available = [field for field in proposed if field in available]
    trust_overlay_available = [field for field in trust_overlay if field in available]
    b5 = next(spec for spec in catalogue()["bayesian"] if spec.name == "B5_stable_ranklossfp_l125_predictive")
    n5 = next(spec for spec in catalogue()["nonlinear"] if spec.name == "N5_ldf_support_l110_meanrisk")
    arms: list[tuple[TrustModelSpec, list[str], str]] = [
        (n5, enhanced_available, "current_ldf_selected_enhanced"),
        (
            TrustModelSpec(
                "B1_previous_raw_singleton_l100_mean", "bayesian", "empirical_bayes",
                "none", "uniform", 1.0, "unconditional", "mean",
            ),
            previous_available,
            "historical_b1_raw_singleton",
        ),
        (b5, previous_available, "previous_binned_bayesian"),
        (
            TrustModelSpec(
                "B5_enhanced_previous_plus_selected", "bayesian", "empirical_bayes",
                "stable_cmi", "rank_loss_false_positive", 1.25, "stable_cmi", "predictive",
            ),
            list(dict.fromkeys([*previous_available, *enhanced_available])),
            "previous_binned_bayesian_plus_enhanced",
        ),
        (
            TrustModelSpec(
                "B5_mda_proposed_previous_plus_selected", "bayesian", "empirical_bayes",
                "stable_cmi", "rank_loss_false_positive", 1.25, "stable_cmi", "predictive",
            ),
            list(dict.fromkeys([*previous_available, *proposed_available])),
            "previous_binned_bayesian_plus_mda_proposed",
        ),
    ]
    if trust_overlay_available:
        current_trust_fields = list(dict.fromkeys([*previous_available, *trust_overlay_available]))
        # Same complete non-posterior inputs, three predeclared Bayesian
        # authority/risk choices. This distinguishes a failure of the
        # empirical-Bayes representation from a failure caused only by the
        # historically selected, relatively aggressive B5 calibration.
        arms.extend([
            (
                TrustModelSpec(
                    "B1_current_trust_overlay", "bayesian", "empirical_bayes",
                    "none", "uniform", 1.0, "unconditional", "mean",
                ),
                current_trust_fields,
                "current_trust_overlay_b1_raw_singleton",
            ),
            (
                TrustModelSpec(
                    "B3_current_trust_overlay", "bayesian", "empirical_bayes",
                    "stable_cmi", "rank_loss", 1.10, "stable_cmi", "mean_risk",
                ),
                current_trust_fields,
                "current_trust_overlay_b3_meanrisk",
            ),
            (
                TrustModelSpec(
                    "B5_current_trust_overlay", "bayesian", "empirical_bayes",
                    "stable_cmi", "rank_loss_false_positive", 1.25, "stable_cmi", "predictive",
                ),
                current_trust_fields,
                "current_trust_overlay_b5_predictive",
            ),
        ])
    if args.only_arms:
        wanted = {token.strip() for token in str(args.only_arms).split(",") if token.strip()}
        available_names = {spec.name for spec, _, _ in arms}
        unknown = sorted(wanted.difference(available_names))
        if unknown:
            raise ValueError(f"unknown --only-arms entries: {unknown}; available={sorted(available_names)}")
        arms = [entry for entry in arms if entry[0].name in wanted]
    args.out_dir.mkdir(parents=True)
    output_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for spec, fields, label in arms:
        if len(fields) < 12:
            raise ValueError(f"{label} has too few shared fields: {len(fields)}")
        print(json.dumps({"event": "arm_start", "arm": spec.name, "label": label, "fields": len(fields)}), flush=True)
        output, audit = _run_arm(
            frame, fields, spec, start=start, end=end, train_cap=int(args.train_cap),
            multiplier_reference=str(args.multiplier_reference),
        )
        output_parts.append(output)
        audit_parts.append(audit.assign(arm_label=label))
        for period_kind in ("global", "month", "week"):
            metrics.append(legacy._period_tail_metrics(output, arm=spec.name, period_kind=period_kind))
        print(json.dumps({"event": "arm_complete", "arm": spec.name, "label": label}), flush=True)
    all_output = pd.concat(output_parts, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    stability = legacy._stability(all_metrics.loc[all_metrics["period_kind"].eq("month")].copy())
    selection = legacy._selection(
        all_metrics.loc[all_metrics["period_kind"].eq("global")].copy(), stability,
    )
    all_output.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    all_metrics.to_parquet(args.out_dir / "metrics.parquet", index=False)
    pd.concat(audit_parts, ignore_index=True).to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    stability.to_parquet(args.out_dir / "stability.parquet", index=False)
    selection.to_parquet(args.out_dir / "selection.parquet", index=False)
    manifest = {
        "schema": "strict_r3_matched_binned_bayesian_vs_ldf_v1",
        "evaluation": [str(start), str(end)],
        "context_surface": str(args.context_surface), "context_sha256": _sha(args.context_surface),
        "evaluation_surface": str(args.evaluation_surface), "evaluation_sha256": _sha(args.evaluation_surface),
        "previous_bayesian_manifest": str(args.previous_bayesian_manifest),
        "enhanced_contract": str(args.enhanced_contract),
        "load_audit": load_audit,
        "arms": [{"label": label, "spec": asdict(spec), "requested_fields": fields} for spec, fields, label in arms],
        "causality": "three-month strict prequential fit; policy labels resolved before held cutoff; frozen score ranking and prior-only 21-day side EV admission; trust changes sizing only",
        "multiplier_reference": str(args.multiplier_reference),
        "geometry": "one frozen Geometry/K9 bundle; raw membership columns prohibited",
        "current_trust_overlay_fields": trust_overlay_available,
        "selection": "diagnostic comparison only; no 2026 outcomes used to alter the frozen 2025 feature contract",
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.out_dir)}, default=str))


if __name__ == "__main__":
    main()
