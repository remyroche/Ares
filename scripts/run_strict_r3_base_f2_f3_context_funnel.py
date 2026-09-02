#!/usr/bin/env python3
"""Offline F2/F3 causal-context base ablation for long Strict-R3.

F2 appends multi-horizon levels, empirical CDFs and level shifts of ten
predeclared high-coverage state fields. F3 appends multi-horizon transition,
acceleration and structural-break features of eight high-coverage transition
fields. Both are generated from the source panel before model fitting and are
prefix-invariant. Each arm refits only a strict-prequential three-class base;
no residual, MC1, admission, execution or live artifact is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features import (  # noqa: E402
    STRICT_R3_F2_ROLLING_CONTEXT_SOURCE_KEYS,
    STRICT_R3_F3_TRANSITION_SOURCE_KEYS,
    strict_r3_rolling_context_features,
)
from extreme_price_movements.strict_r3_canonical_v2 import _fit_medians, _numeric_matrix  # noqa: E402
from scripts.run_strict_r3_base_recall_funnel import BASE_ROUTE_FRACTION, DEFAULT_CONTROL, DEFAULT_SOURCE, PERIODS, _utc, timestamp_route  # noqa: E402
from scripts.run_strict_r3_base_f1_session_funnel import (  # noqa: E402
    _base_params,
    _d2_weights,
    _diagnose,
    _feature_contract,
    _strict_train,
)


DEFAULT_B0_ROOT = ROOT / "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1"


def _load_source(source: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    columns = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "r3_class", "r3_label_available_ts",
        "prequential_base_rank42",
        *fields,
        *STRICT_R3_F2_ROLLING_CONTEXT_SOURCE_KEYS,
        *STRICT_R3_F3_TRANSITION_SOURCE_KEYS,
    ]))
    frame = pd.read_parquet(source, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["r3_label_available_ts"] = pd.to_datetime(frame["r3_label_available_ts"], utc=True, errors="coerce")
    derived = strict_r3_rolling_context_features(frame)
    for name, values in derived.items():
        frame[name] = values
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("source candidate IDs must be unique")
    return frame


def _gate(metrics: pd.DataFrame, arm: str) -> pd.DataFrame:
    names = ("recall_composite", "routed_policy_net_mean_bps", "rank_ic")
    wide = metrics.loc[metrics["arm"].isin(["B0", arm])].pivot(index="label", columns="arm", values=list(names))
    rows = []
    for label in ("frozen_holdout_2025q4", "frozen_oos_2026jan_jul"):
        row = wide.loc[label]
        rows.append({
            "arm": arm,
            "period": label,
            "relative_recall_gain": float(row[("recall_composite", arm)] / row[("recall_composite", "B0")] - 1.0),
            "mean_policy_net_delta_bps": float(row[("routed_policy_net_mean_bps", arm)] - row[("routed_policy_net_mean_bps", "B0")]),
            "rank_ic_delta": float(row[("rank_ic", arm)] - row[("rank_ic", "B0")]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    base_fields = _feature_contract(args.control_root)
    source = _load_source(args.source, base_fields)
    f2_fields = tuple(name for name in source.columns if name.startswith("f2_"))
    f3_fields = tuple(name for name in source.columns if name.startswith("f3_"))
    if len(f2_fields) != 100 or len(f3_fields) != 56:
        raise AssertionError(f"unexpected F2/F3 field counts: {len(f2_fields)}, {len(f3_fields)}")
    b0 = pd.read_parquet(args.b0_root / "b0_target_free_reconstruction.parquet")
    b0["__decision_ts__"] = pd.to_datetime(b0["__decision_ts__"], utc=True, errors="raise")
    source_index = source.set_index("candidate_id", drop=False)
    arms = {"F2_trailing_state": f2_fields, "F3_transition_deltas": f3_fields}
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path in sorted(args.control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        block = path.parents[1].name
        bundle = joblib.load(path)
        cutoff = _utc(bundle.cutoff)
        held_ids = b0.loc[b0["control_block"].eq(block), "candidate_id"]
        if held_ids.empty:
            continue
        held = source_index.loc[held_ids.to_numpy()].copy().reset_index(drop=True)
        train = _strict_train(source, cutoff)
        sample_weight, weight_audit = _d2_weights(train)
        output = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        for arm, extras in arms.items():
            train_coverage = float(train.loc[:, extras].notna().mean().min())
            held_coverage = float(held.loc[:, extras].notna().mean().min())
            if min(train_coverage, held_coverage) < .90:
                raise AssertionError(
                    f"{arm} fails the >=90% per-fold feature-coverage gate in {block}: "
                    f"train={train_coverage:.3f}, held={held_coverage:.3f}"
                )
            contract = (*base_fields, *extras)
            medians = _fit_medians(train, contract)
            model = lgb.LGBMClassifier(**_base_params(bundle)).fit(
                _numeric_matrix(train, contract, medians),
                train["r3_class"].astype(int),
                sample_weight=sample_weight,
            )
            proba = model.predict_proba(_numeric_matrix(held, contract, medians))
            output[f"{arm}_p_adverse"] = proba[:, 0]
            output[f"{arm}_p_weak"] = proba[:, 1]
            output[f"{arm}_p_clear"] = proba[:, 2]
            output[f"{arm}_score"] = proba[:, 2] - .5 * proba[:, 0]
        output["control_block"] = block
        rows.append(output)
        audits.append({
            "block": block, "cutoff": cutoff.isoformat(), "held_rows": int(len(held)), "train_rows": int(len(train)),
            "reserve_start": (cutoff - pd.Timedelta(days=28)).isoformat(),
            "all_labels_before_reserve": bool(train["r3_label_available_ts"].lt(cutoff - pd.Timedelta(days=28)).all()),
            "d2_weight_audit_json": json.dumps(weight_audit, sort_keys=True),
            **{
                f"{arm}_min_train_feature_coverage": float(train.loc[:, extras].notna().mean().min())
                for arm, extras in arms.items()
            },
            **{
                f"{arm}_min_held_feature_coverage": float(held.loc[:, extras].notna().mean().min())
                for arm, extras in arms.items()
            },
        })
    predictions = pd.concat(rows, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    if len(predictions) != len(b0) or predictions["candidate_id"].duplicated().any():
        raise AssertionError("F2/F3 predictions must preserve frozen B0 candidate identities")
    outcome = pd.read_parquet(args.b0_root / "outcome_joined_recall_ledger.parquet")
    outcome["__decision_ts__"] = pd.to_datetime(outcome["__decision_ts__"], utc=True, errors="raise")
    scored = outcome.merge(predictions, on=["candidate_id", "__decision_ts__", "control_block"], how="inner", validate="one_to_one")
    if len(scored) != len(b0):
        raise AssertionError("F2/F3 outcome join must preserve frozen B0 identities")
    scored["B0_route"] = timestamp_route(scored, "base_score", fraction=BASE_ROUTE_FRACTION)
    for arm in arms:
        scored[f"{arm}_route"] = timestamp_route(scored, f"{arm}_score", fraction=BASE_ROUTE_FRACTION)
    metrics_rows: list[dict[str, object]] = []
    labels: list[tuple[str, pd.DataFrame]] = []
    labels.extend(
        (
            name,
            scored.loc[
                scored["__decision_ts__"].ge(_utc(start))
                & scored["__decision_ts__"].lt(_utc(end))
            ].copy(),
        )
        for name, (start, end) in PERIODS.items()
    )
    labels.extend((str(q), group.copy()) for q, group in scored.groupby(scored["__decision_ts__"].dt.to_period("Q"), sort=True) if q >= pd.Period("2025Q4", freq="Q"))
    for label, subset in labels:
        for arm, score, route in [("B0", "base_score", "B0_route"), *[(a, f"{a}_score", f"{a}_route") for a in arms]]:
            row = _diagnose(subset, subset[route].to_numpy(bool), score, label)
            row["arm"] = arm
            metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    gates = pd.concat([_gate(metrics, arm) for arm in arms], ignore_index=True)
    quarterly = metrics.loc[metrics["label"].str.match(r"^20\d\dQ[1-4]$", na=False)].pivot(index="label", columns="arm", values="recall_policy_ge_100")
    decisions = {}
    for arm in arms:
        arm_gate = gates.loc[gates["arm"].eq(arm)]
        decisions[arm] = bool(
            arm_gate["relative_recall_gain"].ge(.02).all()
            and arm_gate["mean_policy_net_delta_bps"].ge(-5.0).all()
            and arm_gate["rank_ic_delta"].ge(-.005).all()
            and (quarterly[arm] >= quarterly["B0"]).all()
        )
    coverage = pd.DataFrame({
        "feature": [*f2_fields, *f3_fields],
        "coverage": [float(source[x].notna().mean()) for x in [*f2_fields, *f3_fields]],
    })
    coverage_rows: list[pd.DataFrame] = []
    for label, (start, end) in PERIODS.items():
        subset = source.loc[
            source["__decision_ts__"].ge(_utc(start))
            & source["__decision_ts__"].lt(_utc(end)),
            [*f2_fields, *f3_fields],
        ]
        block = pd.DataFrame({
            "period": label,
            "feature": [*f2_fields, *f3_fields],
            "coverage": [float(subset[x].notna().mean()) for x in subset.columns],
        })
        coverage_rows.append(block)
    coverage_by_period = pd.concat(coverage_rows, ignore_index=True)
    if (coverage_by_period["coverage"] < .90).any():
        failed = coverage_by_period.loc[coverage_by_period["coverage"] < .90]
        raise AssertionError(f"F2/F3 fails the >=90% period-coverage gate: {failed.to_dict('records')}")
    args.out_dir.mkdir(parents=True)
    predictions.to_parquet(args.out_dir / "f2_f3_target_free_scores.parquet", index=False, compression="zstd")
    scored.to_parquet(args.out_dir / "f2_f3_outcome_joined_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "f2_f3_block_training_audit.parquet", index=False)
    coverage.to_parquet(args.out_dir / "f2_f3_feature_coverage.parquet", index=False)
    coverage_by_period.to_parquet(args.out_dir / "f2_f3_feature_coverage_by_period.parquet", index=False)
    metrics.to_parquet(args.out_dir / "f2_f3_base_metrics.parquet", index=False)
    gates.to_parquet(args.out_dir / "f2_f3_advancement_gates.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_f2_f3_context_v2_canonical_d2",
        "scope": "offline base-only feature-block screening; no residual, MC1, portfolio or live artifact modified",
        "source": str(args.source), "b0_root": str(args.b0_root),
        "base_feature_count": len(base_fields), "f2_feature_count": len(f2_fields), "f3_feature_count": len(f3_fields),
        "causality": "F2/F3 inputs are backward-looking transforms of frozen decision-time primitive fields; fitting requires R3 label availability before the 28-day calibration reserve and uses the canonical D2 teacher weighting",
        "feature_coverage_gate": ">=90% for every train/held fold and reported period",
        "base_training_contract": "same 240k latest-label-resolved cap, train-fold median imputation, D2 teacher weighting, and fully excluded 28-day reserve as B0",
        "advance_to_downstream_rebuild": decisions,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(scored)), "advance": decisions}, sort_keys=True))


if __name__ == "__main__":
    main()
