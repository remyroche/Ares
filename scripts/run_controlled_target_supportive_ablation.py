#!/usr/bin/env python3
"""Run the controlled T0--T4 x S0--S5 target/supportive-label matrix.

The input is a prepared, exact-H12 research ledger.  This runner intentionally
does not discover or modify labels/features: root integration must pin them
first.  It then holds raw rows, folds, raw features and LightGBM capacity fixed
for every target arm.  It is not a production/promotion runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from extreme_price_movements.controlled_target_supportive_ablation import (
    AcceptanceGates,
    ContractError,
    DEFAULT_HURDLE_BPS,
    GROUPED_SUPPORT_LABELS,
    SUPPORT_LABELS,
    SUPPORT_STAGES,
    TARGET_ARMS,
    aligned_run_contract,
    derive_economic_targets,
    hurdle_decomposition_score,
    matched_support_population,
    pooled_global_top_k_metrics,
    stable_pooled_global_top_k,
    strict_oof_support_predictions,
    support_columns,
    validate_causal_raw_features,
)


SCHEMA = "controlled_target_supportive_ablation_runner_v1"
FIXED_CAPACITY = {
    "n_estimators": 250,
    "learning_rate": 0.035,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 5.0,
    "random_state": 20260801,
    "n_jobs": 1,
    "verbosity": -1,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attach_frozen_support_oof(
    frame: pd.DataFrame,
    support_oof: pd.DataFrame,
    *,
    support_labels: tuple[tuple[str, str, str, str], ...] = SUPPORT_LABELS,
) -> pd.DataFrame:
    """Attach one immutable, already-audited support-OOF surface to the ledger.

    The earlier runner regenerated support predictions in-process.  That was
    causally valid, but it made the core matrix and the standalone hurdle grid
    depend on two separate fits.  This join makes the support surface a named,
    hashable input and fails closed on candidate/fold/timestamp mismatches.
    """
    required = {"candidate_id", "oof_fold", "fold_order", "__ts__"}
    missing = required - set(support_oof.columns)
    if missing:
        raise ContractError(f"frozen support OOF is missing columns: {sorted(missing)}")
    support = support_oof.copy()
    if support["candidate_id"].duplicated().any():
        raise ContractError("frozen support OOF candidate_id join is not one-to-one")
    support_names = list(support_columns("S5", support_labels))
    missing_support = set(support_names) - set(support.columns)
    if missing_support:
        raise ContractError(f"frozen support OOF is missing support columns: {sorted(missing_support)}")
    # The support audit intentionally contains only rows after the warmup
    # fold.  It must not silently move a prediction to another candidate or
    # protocol timestamp when joined to the complete ledger.
    left = frame[["candidate_id", "oof_fold", "fold_order", "__ts__"]].copy()
    right = support[["candidate_id", "oof_fold", "fold_order", "__ts__", *support_names]].copy()
    overlap = left.merge(right, on="candidate_id", how="inner", suffixes=("_ledger", "_support"), validate="one_to_one")
    if overlap.empty:
        raise ContractError("frozen support OOF has no candidate overlap with the prepared ledger")
    for name in ("oof_fold", "fold_order"):
        if not overlap[f"{name}_ledger"].astype(str).equals(overlap[f"{name}_support"].astype(str)):
            raise ContractError(f"frozen support OOF {name} does not match the prepared ledger")
    ledger_ts = pd.to_datetime(overlap["__ts___ledger"], utc=True, errors="raise")
    support_ts = pd.to_datetime(overlap["__ts___support"], utc=True, errors="raise")
    if not (ledger_ts.to_numpy() == support_ts.to_numpy()).all():
        raise ContractError("frozen support OOF feature timestamps do not match the prepared ledger")
    support = support.drop(columns=["oof_fold", "fold_order", "__ts__"])
    attached = frame.merge(support[["candidate_id", *support_names]], on="candidate_id", how="left", validate="one_to_one")
    # Warmup rows must remain unsupported; every later fold must have the full
    # cumulative support vector.
    support_matrix = attached.loc[:, support_names].to_numpy(dtype=float)
    warmup = attached["fold_order"].to_numpy() == 0
    if np.isfinite(support_matrix[warmup]).any():
        raise ContractError("frozen support OOF contains a prediction on the warmup fold")
    if not np.isfinite(support_matrix[~warmup]).all():
        raise ContractError("frozen support OOF is incomplete on a scored fold")
    return attached


def _monthly_side_detail(prediction: pd.DataFrame, *, top_k_fraction: float) -> pd.DataFrame:
    """Partition one pooled global book by month and side for diagnostics."""
    work = prediction.copy()
    # Use the protocol timestamp for month coverage.  A decision one hour
    # after the final November feature bar can fall on 1 December; counting it
    # as a new month would falsely claim an additional untouched evaluation
    # month.
    work["month"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise").dt.strftime("%Y-%m")
    work["global_topk"] = False
    for _, group in work.groupby(["target_arm", "support_stage"], sort=False, observed=True):
        chosen = stable_pooled_global_top_k(group, "score", top_k_fraction)
        chosen_ids = set(chosen["candidate_id"].astype(str))
        work.loc[group.index, "global_topk"] = group["candidate_id"].astype(str).isin(chosen_ids).to_numpy()
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(["target_arm", "support_stage", "month", "side_name"], sort=True, observed=True):
        selected = group[group["global_topk"]]
        rows.append({
            "target_arm": keys[0], "support_stage": keys[1], "month": keys[2], "side": keys[3],
            "population_rows": int(len(group)), "global_topk_rows": int(len(selected)),
            "population_net_bps": float(group["execution_net_ev_12h"].mean() * 10_000.0),
            "population_gross_bps": float(group["execution_gross_ev_12h"].mean() * 10_000.0),
            "population_cost_bps": float(group["execution_cost_return"].mean() * 10_000.0),
            "population_positive_net_rate": float((group["execution_net_ev_12h"] > 0.0).mean()),
            "global_topk_net_bps": float(selected["execution_net_ev_12h"].mean() * 10_000.0) if len(selected) else np.nan,
            "global_topk_gross_bps": float(selected["execution_gross_ev_12h"].mean() * 10_000.0) if len(selected) else np.nan,
            "global_topk_cost_bps": float(selected["execution_cost_return"].mean() * 10_000.0) if len(selected) else np.nan,
            "global_topk_positive_net_rate": float((selected["execution_net_ev_12h"] > 0.0).mean()) if len(selected) else np.nan,
        })
    return pd.DataFrame(rows)


def _score_calibration(prediction: pd.DataFrame) -> pd.DataFrame:
    """Score-decile diagnostics on the exact realised outcomes."""
    rows: list[dict[str, object]] = []
    for keys, group in prediction.groupby(["target_arm", "support_stage"], sort=False, observed=True):
        ordered = group.assign(_rank=group["score"].rank(method="first", pct=True))
        ordered["score_decile"] = np.ceil(ordered["_rank"] * 10.0).clip(1, 10).astype(int)
        for decile, bucket in ordered.groupby("score_decile", sort=True, observed=True):
            rows.append({
                "target_arm": keys[0], "support_stage": keys[1], "score_decile": int(decile),
                "rows": int(len(bucket)), "mean_score": float(bucket["score"].mean()),
                "mean_net_bps": float(bucket["execution_net_ev_12h"].mean() * 10_000.0),
                "mean_gross_bps": float(bucket["execution_gross_ev_12h"].mean() * 10_000.0),
                "mean_cost_bps": float(bucket["execution_cost_return"].mean() * 10_000.0),
                "positive_net_rate": float((bucket["execution_net_ev_12h"] > 0.0).mean()),
            })
    return pd.DataFrame(rows)


def _correctness_checks(frame: pd.DataFrame, prediction: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Machine-readable fail-closed checks accompanying a research run."""
    checks: list[dict[str, object]] = []
    def add(name: str, passed: bool, value: object, rule: str) -> None:
        # A single Parquet column cannot safely mix integer counts and
        # timedelta/string diagnostics.  Keep the machine-readable pass bit
        # typed and serialize the human-readable value deterministically.
        checks.append({"check": name, "passed": bool(passed), "value": str(value), "rule": rule})
    add("candidate_identity_unique", frame["candidate_id"].is_unique, int(frame["candidate_id"].nunique()), "one row per candidate_id")
    accounting = np.abs(frame["execution_gross_ev_12h"] - frame["execution_cost_return"] - frame["execution_net_ev_12h"])
    add("exact_net_accounting", bool(float(accounting.max()) <= 1e-10), float(accounting.max()), "gross - cost == net")
    availability = pd.to_datetime(frame["__label_available_at__"], utc=True) - pd.to_datetime(frame["__decision_ts__"], utc=True)
    add("label_availability_h12", bool((availability == pd.Timedelta(hours=12)).all()), str(availability.min()), "label available exactly decision + 12h")
    add("feature_cutoff_causal", bool((pd.to_datetime(frame["__ts__"], utc=True) <= pd.to_datetime(frame["__decision_ts__"], utc=True)).all()), int((pd.to_datetime(frame["__ts__"], utc=True) > pd.to_datetime(frame["__decision_ts__"], utc=True)).sum()), "feature_ts <= decision_ts")
    simplex = frame[["favorable_first", "adverse_first", "timeout"]].sum(axis=1)
    add("competing_risk_simplex", bool((simplex == 1).all()), int((simplex != 1).sum()), "clean/adverse/timeout exhaustive")
    finite = np.isfinite(prediction["score"].to_numpy(dtype=float))
    add("all_oof_scores_finite", bool(finite.all()), int((~finite).sum()), "every emitted OOF score is finite")
    duplicate = int(prediction.duplicated(["candidate_id", "target_arm", "support_stage"]).sum())
    add("candidate_cell_predictions_unique", duplicate == 0, duplicate, "one score per candidate/target/support cell")
    warmup = int(prediction["fold_order"].min()) if "fold_order" in prediction.columns and len(prediction) else -1
    add("no_warmup_scores", warmup >= 1, warmup, "first protocol fold is metadata-only and never scored")
    if {"prediction_fit_end_ts", "prediction_generated_ts"}.issubset(prediction.columns):
        fit_end = pd.to_datetime(prediction["prediction_fit_end_ts"], utc=True, errors="raise")
        generated = pd.to_datetime(prediction["prediction_generated_ts"], utc=True, errors="raise")
        decision = pd.to_datetime(prediction["__decision_ts__"], utc=True, errors="raise")
        add(
            "prediction_fit_end_before_decision",
            bool((fit_end < decision).all()),
            int((fit_end >= decision).sum()),
            "every model fit resolves before the candidate decision",
        )
        add(
            "prediction_generated_at_or_before_decision",
            bool((generated <= decision).all()),
            int((generated > decision).sum()),
            "prediction timestamp is no later than the candidate decision",
        )
        add(
            "prediction_lineage_columns_complete",
            bool(prediction["prediction_model_id"].notna().all() and prediction["prediction_fold_id"].notna().all()),
            int((prediction["prediction_model_id"].isna() | prediction["prediction_fold_id"].isna()).sum()),
            "model, fold, fit-end and generated timestamps are present",
        )
    add("causal_feature_contract_validated", True, len(feature_columns), "raw feature names passed semantic checks")
    return pd.DataFrame(checks)


def _predictor(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "binary":
        model = lgb.LGBMClassifier(objective="binary", **FIXED_CAPACITY)
        model.fit(train_x, (train_y > 0.5).astype(int))
        return model.predict_proba(test_x)[:, 1]
    model = lgb.LGBMRegressor(objective="regression_l2", **FIXED_CAPACITY)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def _fit_regression(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = lgb.LGBMRegressor(objective="regression_l2", **FIXED_CAPACITY)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def _fit_probability(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    target = (np.asarray(train_y) > 0.5).astype(int)
    if target.min() == target.max():
        return np.full(len(test_x), float(target[0]))
    model = lgb.LGBMClassifier(objective="binary", **FIXED_CAPACITY)
    model.fit(train_x, target)
    return model.predict_proba(test_x)[:, 1]


def _arm_score(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    hurdle_bps: float = DEFAULT_HURDLE_BPS,
) -> np.ndarray:
    # Keep the causal missingness pattern and avoid a second 64-bit copy of
    # the full ledger for every fold.  LightGBM handles NaNs natively.
    train_x = train.loc[:, columns].to_numpy(dtype=np.float32)
    test_x = test.loc[:, columns].to_numpy(dtype=np.float32)
    if name == "T0_native24_control":
        return _fit_regression(train_x, train.target_t0_native24.to_numpy(float), test_x)
    if name == "T1_clean_opportunity":
        return _fit_probability(train_x, train.target_t1_clean_opportunity.to_numpy(float), test_x)
    if name == "T2_direct_net":
        return _fit_regression(train_x, train.target_t2_direct_net.to_numpy(float), test_x)
    if name == "T3_competing_risk_expected_net":
        components = []
        for event in (0, 1, 2):
            probability = _fit_probability(train_x, (train.target_t3_competing_class.to_numpy(int) == event), test_x)
            subset = train.target_t3_competing_class.to_numpy(int) == event
            payoff = np.full(len(test), float(train.target_t3_expected_net.mean())) if subset.sum() < 2 else _fit_regression(train_x[subset], train.target_t3_expected_net.to_numpy(float)[subset], test_x)
            components.append(probability * payoff)
        return np.sum(components, axis=0)
    if name == "T4_hurdle_decomposition":
        clear_rows = train.target_t4_clear.to_numpy(int) == 1
        fail_rows = train.target_t4_fail.to_numpy(int) == 1
        # The two events are exhaustive by construction.  Deriving fail from
        # clear preserves the probability simplex instead of allowing two
        # independently fitted binary heads to over/under-count the same row.
        clear_probability = _fit_probability(train_x, clear_rows, test_x)
        fail_probability = 1.0 - clear_probability
        clear_excess = train.target_t4_clear_excess_return.to_numpy(float)
        fail_shortfall = train.target_t4_fail_shortfall_return.to_numpy(float)
        mean_clear_excess = float(clear_excess[clear_rows].mean()) if clear_rows.any() else 0.0
        mean_fail_shortfall = float(fail_shortfall[fail_rows].mean()) if fail_rows.any() else 0.0
        if clear_rows.sum() >= 2:
            clear_excess_prediction = np.maximum(_fit_regression(train_x[clear_rows], clear_excess[clear_rows], test_x), 0.0)
        else:
            clear_excess_prediction = np.full(len(test), mean_clear_excess)
        if fail_rows.sum() >= 2:
            fail_shortfall_prediction = np.maximum(_fit_regression(train_x[fail_rows], fail_shortfall[fail_rows], test_x), 0.0)
        else:
            fail_shortfall_prediction = np.full(len(test), mean_fail_shortfall)
        return hurdle_decomposition_score(
            clear_probability,
            clear_excess_prediction,
            fail_probability,
            fail_shortfall_prediction,
            hurdle_bps=hurdle_bps,
        )
    raise ContractError(f"unknown target arm: {name}")


def run_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    fold_column: str,
    gates: AcceptanceGates,
    hurdle_bps: float = DEFAULT_HURDLE_BPS,
    frozen_support_oof: pd.DataFrame | None = None,
    support_labels: tuple[tuple[str, str, str, str], ...] = SUPPORT_LABELS,
    support_spec: str = "legacy",
    target_arms: tuple[str, ...] = TARGET_ARMS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    feature_columns = list(validate_causal_raw_features(feature_columns))
    numeric_features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    all_missing = [name for name in feature_columns if not np.isfinite(numeric_features[name].to_numpy(dtype=float)).any()]
    # A feature that is absent in every candidate cannot be learned and would
    # make the fold fits dependent on LightGBM's version-specific feature
    # pre-filter.  Exclude it deterministically and retain the evidence in the
    # run contract; partially missing columns remain native-NaN inputs.
    feature_columns = [name for name in feature_columns if name not in set(all_missing)]
    if not feature_columns:
        raise ContractError("no usable causal raw features remain after all-missing exclusion")
    targets = derive_economic_targets(frame, hurdle_bps=hurdle_bps)
    if frozen_support_oof is None:
        with_supports = strict_oof_support_predictions(
            targets,
            feature_columns=feature_columns,
            fold_column=fold_column,
            predictor=_predictor,
            support_labels=support_labels,
        )
        support_source = "regenerated_strict_oof_in_process"
    else:
        # The frozen artifact contains the same candidate rows and raw labels;
        # attach only its support columns, then continue through the exact
        # same matched-population and scoring code path.
        with_supports = _attach_frozen_support_oof(targets, frozen_support_oof, support_labels=support_labels)
        support_source = "frozen_support_oof_artifact"
    work = matched_support_population(with_supports, support_labels=support_labels)
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    fold_starts = work.groupby(fold_column, observed=True)["__ts__"].min().sort_values(kind="mergesort")
    prediction_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []
    for stage in SUPPORT_STAGES:
        columns = [*feature_columns, *support_columns(stage, support_labels)]
        for arm in target_arms:
            scored = work[[
                "candidate_id", "__ts__", "__decision_ts__", "side_name", "__symbol__",
                fold_column, "fold_order", "__label_available_at__", "execution_net_ev_12h",
                "execution_gross_ev_12h", "execution_cost_return",
            ]].copy()
            score = np.full(len(work), np.nan)
            prediction_fold_id = np.full(len(work), None, dtype=object)
            prediction_fit_end_ts = np.full(len(work), np.datetime64("NaT"), dtype="datetime64[ns]")
            prediction_generated_ts = np.full(len(work), np.datetime64("NaT"), dtype="datetime64[ns]")
            for position, fold in enumerate(fold_starts.index):
                if position == 0:
                    continue
                test = work[work[fold_column].eq(fold)]
                train = work[work[fold_column].isin(fold_starts.index[:position])]
                # Earlier OOF support features and the target are resolved before each test fold.
                test_start = fold_starts.loc[fold]
                train = train[train["__label_available_at__"] < test_start]
                if train.empty:
                    continue
                fit_end = pd.to_datetime(train["__label_available_at__"], utc=True, errors="raise").max()
                score[test.index.map(work.index.get_loc)] = _arm_score(
                    arm, train, test, columns, hurdle_bps=hurdle_bps,
                )
                positions = test.index.map(work.index.get_loc).to_numpy(dtype=int)
                prediction_fold_id[positions] = str(fold)
                prediction_fit_end_ts[positions] = fit_end.to_datetime64()
                prediction_generated_ts[positions] = pd.to_datetime(test["__ts__"], utc=True, errors="raise").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
            scored["score"] = score
            scored["target_arm"] = arm
            scored["support_stage"] = stage
            scored["prediction_model_id"] = f"{SCHEMA}:lightgbm:{arm}:{stage}:seed-{FIXED_CAPACITY['random_state']}"
            scored["prediction_fold_id"] = prediction_fold_id
            scored["prediction_fit_end_ts"] = prediction_fit_end_ts
            scored["prediction_generated_ts"] = prediction_generated_ts
            # Make the causal class explicit at candidate level.  Future
            # evaluators must never infer that a generic/blocked OOF score is
            # prequential merely from its filename or aggregate manifest.
            scored["strict_prequential_oof"] = True
            scored["diagnostic_noncausal_oof"] = False
            scored = scored[np.isfinite(scored.score)].copy()
            metrics = pooled_global_top_k_metrics(scored, "score", gates=gates)
            report_rows.append({"target_arm": arm, "support_stage": stage, "hurdle_bps": float(hurdle_bps), "model_capacity": json.dumps(FIXED_CAPACITY, sort_keys=True), **metrics})
            prediction_parts.append(scored)
    report = pd.DataFrame(report_rows)
    prediction = pd.concat(prediction_parts, ignore_index=True)
    contract = dict(aligned_run_contract(work, fold_column=fold_column, feature_columns=feature_columns, hurdle_bps=hurdle_bps, support_labels=support_labels))
    contract.update({
        "runner_schema": SCHEMA,
        "fixed_model_capacity": FIXED_CAPACITY,
        "acceptance_gates": gates.manifest(),
        "excluded_all_missing_features": all_missing,
        "oof_protocol": {
            "support_predictions": f"{support_source}; training label_available_at < test fold start",
            "target_predictions": "fit on earlier resolved protocol fold(s); first support-warmup fold is never scored",
            "candidate_oof_class": "strict_prequential_oof=true; diagnostic_noncausal_oof=false on every emitted candidate score",
            "evaluation_selection": "one pooled global top-10% over the untouched meta_oos fold",
        },
        "status": "RESEARCH_ONLY_NOT_PROMOTION",
        "support_spec": support_spec,
        "evaluated_target_arms": list(target_arms),
    })
    return prediction, report, contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True, help="Prepared exact-H12 candidate/label ledger")
    parser.add_argument("--features-json", type=Path, required=True, help="JSON list of frozen causal raw features")
    parser.add_argument("--fold-column", default="oof_fold")
    parser.add_argument("--hurdle-bps", type=float, default=DEFAULT_HURDLE_BPS, help="T4 net hurdle in basis points; default is 25.")
    parser.add_argument(
        "--support-oof",
        type=Path,
        default=None,
        help="Optional frozen supportive_head_oof_predictions.parquet; avoids refitting support heads and is hash-linked in the manifest.",
    )
    parser.add_argument("--support-spec", choices=("legacy", "grouped"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {args.output}")
    features_payload = json.loads(args.features_json.read_text())
    # The prepared-ledger materializer emits a provenance object while older
    # callers may provide a bare list.  Accept both without weakening the
    # fail-closed semantic validation in run_matrix.
    if isinstance(features_payload, dict):
        features = features_payload.get("raw_feature_columns") or features_payload.get("feature_columns")
    else:
        features = features_payload
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ContractError("features JSON must be a list or an object containing raw_feature_columns")
    frame = pd.read_parquet(args.ledger)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        support_labels = GROUPED_SUPPORT_LABELS if args.support_spec == "grouped" else SUPPORT_LABELS
        predictions, summary, contract = run_matrix(
            frame,
            feature_columns=features,
            fold_column=args.fold_column,
            gates=AcceptanceGates(),
            hurdle_bps=args.hurdle_bps,
            frozen_support_oof=pd.read_parquet(args.support_oof) if args.support_oof is not None else None,
            support_labels=support_labels,
            support_spec=args.support_spec,
        )
        predictions.to_parquet(temporary / "oof_target_supportive_predictions.parquet", index=False, compression="zstd")
        summary.to_parquet(temporary / "target_supportive_policy_summary.parquet", index=False, compression="zstd")
        lineage = (
            predictions.groupby(
                ["target_arm", "support_stage", "prediction_model_id", "prediction_fold_id", "prediction_fit_end_ts"],
                dropna=False,
                observed=True,
            )
            .agg(
                rows=("candidate_id", "size"),
                prediction_generated_min=("prediction_generated_ts", "min"),
                prediction_generated_max=("prediction_generated_ts", "max"),
                candidate_decision_min=("__decision_ts__", "min"),
                candidate_decision_max=("__decision_ts__", "max"),
            )
            .reset_index()
        )
        lineage["is_oof"] = True
        lineage["feature_count"] = len(contract["feature_columns"])
        lineage.to_parquet(temporary / "oof_prediction_manifest.parquet", index=False, compression="zstd")
        # These partitions are diagnostics of the same pooled-global policy;
        # they do not introduce timestamp, side, asset or portfolio quotas.
        _monthly_side_detail(predictions, top_k_fraction=AcceptanceGates().top_k_fraction).to_parquet(
            temporary / "target_supportive_monthly_side_metrics.parquet", index=False, compression="zstd"
        )
        _score_calibration(predictions).to_parquet(
            temporary / "target_supportive_score_calibration.parquet", index=False, compression="zstd"
        )
        _correctness_checks(frame, predictions, contract["feature_columns"]).to_parquet(
            temporary / "correctness_checks.parquet", index=False, compression="zstd"
        )
        if "chronological_protocol_folds.parquet" in {p.name for p in args.ledger.parent.iterdir()}:
            fold_source = args.ledger.parent / "chronological_protocol_folds.parquet"
            pd.read_parquet(fold_source).to_parquet(temporary / "fold_manifest.parquet", index=False, compression="zstd")
        else:
            fold_summary = frame.groupby(args.fold_column, observed=True).agg(
                rows=("candidate_id", "size"), min_ts=("__ts__", "min"), max_ts=("__ts__", "max"),
                min_label_available=("__label_available_at__", "min"), max_label_available=("__label_available_at__", "max"),
            ).reset_index()
            fold_summary.to_parquet(temporary / "fold_manifest.parquet", index=False, compression="zstd")
        manifest = {
            **contract,
            "ledger": str(args.ledger),
            "ledger_sha256": _sha256(args.ledger),
            "features_json": str(args.features_json),
            "features_json_sha256": _sha256(args.features_json),
            "artifacts": {
                "candidate_level_oof_predictions": "oof_target_supportive_predictions.parquet",
                "pooled_global_policy_summary": "target_supportive_policy_summary.parquet",
                "monthly_side_diagnostics": "target_supportive_monthly_side_metrics.parquet",
                "score_calibration": "target_supportive_score_calibration.parquet",
                "correctness_checks": "correctness_checks.parquet",
                "fold_manifest": "fold_manifest.parquet",
                "oof_prediction_manifest": "oof_prediction_manifest.parquet",
            },
        }
        prepared_manifest = args.ledger.parent / "run_manifest.json"
        if prepared_manifest.is_file():
            manifest["prepared_ledger"] = {
                "path": str(prepared_manifest),
                "sha256": _sha256(prepared_manifest),
                "native_control_horizon_caveat": json.loads(prepared_manifest.read_text()).get("native_control_horizon_caveat"),
            }
        if args.support_oof is not None:
            manifest["support_oof"] = {
                "path": str(args.support_oof),
                "sha256": _sha256(args.support_oof),
                "rule": "strict chronological OOF; fit labels available before the test fold start",
            }
        manifest["code_provenance"] = {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "core_module_path": str(Path(__file__).resolve().parents[1] / "extreme_price_movements/controlled_target_supportive_ablation.py"),
            "core_module_sha256": _sha256(Path(__file__).resolve().parents[1] / "extreme_price_movements/controlled_target_supportive_ablation.py"),
            "lightgbm_version": getattr(lgb, "__version__", "unknown"),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in temporary.iterdir()
            if path.is_file() and path.name != "run_manifest.json"
        }
        (temporary / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    except Exception:
        # Preserve a failed staging directory for forensic recovery.  It is
        # never published under the requested artifact name, but keeping any
        # already-written prediction matrix avoids silently discarding a
        # multi-hour fit when a post-processing check needs repair.
        failed = temporary.with_name(temporary.name + ".failed")
        try:
            os.replace(temporary, failed)
        except OSError:
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
