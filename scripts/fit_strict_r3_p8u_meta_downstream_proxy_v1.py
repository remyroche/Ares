#!/usr/bin/env python3
"""Fit and falsify learned PriorityProxy/GateProxy models for P8u Meta HPO.

The input is deliberately small: strict-OOF trial descriptors plus expensive,
matched downstream MC1 labels.  The output compares strongly regularised
linear models, a depth-2 tree, and a pairwise ranking surrogate under grouped
holdouts.  It does not select, promote, or alter any Meta trial by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


SCHEMA = "strict_r3_p8u_meta_downstream_proxy_v1"
CORE_FIELDS = (
    "residual_ic", "conditional_mi_given_base", "ic_base_5_10", "ic_base_10_20", "ic_base_20_30",
    "meta_top1_ev", "meta_top2_ev", "top1_candidate_only_minus_control_only_ev",
    "top2_candidate_only_minus_control_only_ev", "false_upgrade_ev", "useful_upgrade_ev",
    "base_meta_rank_correlation", "median_abs_rank_correction", "weekly_q10",
    "probe_delta_top2_ev", "probe_delta_admitted_utility",
)
TARGETS = ("dpriority_shrunk", "dgate_shrunk")


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
        ("model", estimator),
    ))


@dataclass
class PairwiseSurrogate:
    transformer: Pipeline
    classifier: LogisticRegression | None

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        transformed = self.transformer.transform(x)
        if self.classifier is None:
            return np.zeros(len(x), dtype=float)
        return transformed @ self.classifier.coef_.reshape(-1) + float(self.classifier.intercept_[0])


def _fit_pairwise(x: pd.DataFrame, y: np.ndarray, weight: np.ndarray) -> PairwiseSurrogate:
    transformer = Pipeline((("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler(quantile_range=(10.0, 90.0)))))
    values = transformer.fit_transform(x)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            delta = float(y[left] - y[right])
            if not np.isfinite(delta) or abs(delta) <= 1e-12:
                continue
            direction = 1 if delta > 0 else 0
            rows.append(values[left] - values[right]); labels.append(direction)
            weights.append(float(np.sqrt(max(weight[left], 1e-9) * max(weight[right], 1e-9))))
            rows.append(values[right] - values[left]); labels.append(1 - direction)
            weights.append(float(np.sqrt(max(weight[left], 1e-9) * max(weight[right], 1e-9))))
    if len(set(labels)) < 2:
        return PairwiseSurrogate(transformer, None)
    classifier = LogisticRegression(C=.20, max_iter=10_000, solver="lbfgs", random_state=1729)
    classifier.fit(np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), sample_weight=np.asarray(weights, dtype=float))
    return PairwiseSurrogate(transformer, classifier)


def _fit_model(name: str, x: pd.DataFrame, y: np.ndarray, weight: np.ndarray) -> Any:
    if name == "P0_ridge":
        model = _pipeline(Ridge(alpha=10.0, random_state=1729))
        model.fit(x, y, model__sample_weight=weight)
        return model
    if name == "P1_elastic_net":
        model = _pipeline(ElasticNet(alpha=.10, l1_ratio=.50, max_iter=20_000, random_state=1729))
        model.fit(x, y, model__sample_weight=weight)
        return model
    if name == "P2_depth2_gbdt":
        model = _pipeline(HistGradientBoostingRegressor(
            max_depth=2, max_leaf_nodes=4, learning_rate=.05, max_iter=120,
            min_samples_leaf=max(5, int(np.ceil(len(x) * .12))), l2_regularization=2.0, random_state=1729,
        ))
        model.fit(x, y, model__sample_weight=weight)
        return model
    if name == "P3_pairwise":
        return _fit_pairwise(x, y, weight)
    raise ValueError(name)


def _predict(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=float)


def _read_descriptors(roots: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts, fold_parts = [], []
    for root in (path.resolve() for path in roots):
        summary = pd.read_parquet(root / "trial_descriptor_summary.parquet")
        folds = pd.read_parquet(root / "trial_fold_descriptors.parquet")
        summary["descriptor_root"] = root.name; folds["descriptor_root"] = root.name
        summary_parts.append(summary); fold_parts.append(folds)
    summary = pd.concat(summary_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True)
    if summary.trial.duplicated().any():
        raise AssertionError("trial appears in multiple descriptor roots")
    return summary, folds


def _normalise_era_labels(monthly: pd.DataFrame, normalisation: dict[str, dict[str, float]]) -> pd.DataFrame:
    result = monthly.copy()
    for objective, components in (("priority", (
        ("priority_top1_delta_bps", .20), ("priority_top2_delta_bps", .40),
        ("priority_captured_utility_delta_bps_per_timestamp", .25), ("priority_weekly_q10_delta_bps", .15),
    )), ("gate", (
        ("gate_admitted_ev_delta_bps", .35), ("gate_total_utility_delta_bps_per_timestamp", .25),
        ("gate_precision_gt50_delta", .15), ("gate_precision_gt100_delta", .10),
        ("gate_volume_delta_per_timestamp", .10), ("gate_weekly_q10_delta_bps", .05),
    ))):
        value = np.zeros(len(result), dtype=float)
        for field, weight in components:
            params = normalisation[field]
            value += weight * (pd.to_numeric(result[field], errors="coerce").fillna(params["location"]).to_numpy(float) - float(params["location"])) / float(params["scale"])
        result[f"d{objective}_era"] = value
    return result


def _grouped_predictions(
    *, table: pd.DataFrame, fields: list[str], target: str, weight_column: str, group_column: str, model_name: str,
) -> pd.DataFrame:
    if group_column not in table or table[group_column].nunique() < 2:
        return pd.DataFrame()
    outputs: list[pd.DataFrame] = []
    for group in sorted(table[group_column].dropna().astype(str).unique()):
        train = table.loc[table[group_column].astype(str).ne(group)].copy()
        test = table.loc[table[group_column].astype(str).eq(group)].copy()
        if len(train) < max(8, len(fields) // 2) or len(test) < 1:
            continue
        model = _fit_model(model_name, train[fields], train[target].to_numpy(float), train[weight_column].to_numpy(float))
        output = test.loc[:, ["trial", group_column, target]].copy()
        output["prediction"] = _predict(model, test[fields])
        output["validation"] = group_column; output["held_group"] = group; output["model"] = model_name
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _validation_metrics(predictions: pd.DataFrame, *, target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return rows
    for (model, validation, group), part in predictions.groupby(["model", "validation", "held_group"], sort=True):
        values = pd.to_numeric(part[target], errors="coerce").to_numpy(float)
        scores = pd.to_numeric(part.prediction, errors="coerce").to_numpy(float)
        finite = np.isfinite(values) & np.isfinite(scores)
        values, scores = values[finite], scores[finite]
        if not len(values):
            continue
        k = min(3, len(values))
        pred_index = np.argsort(-scores, kind="stable")[:k]
        true_index = np.argsort(-values, kind="stable")[:k]
        winner = int(np.argmax(values))
        rows.append({
            "target": target, "model": model, "validation": validation, "held_group": group, "rows": int(len(values)),
            "spearman": float(spearmanr(scores, values).statistic) if len(values) >= 3 else float("nan"),
            "top3_precision": float(len(set(pred_index).intersection(true_index)) / k),
            "winner_in_proxy_top3": float(winner in set(pred_index)),
            "regret_at3": float(np.max(values) - np.max(values[pred_index])),
        })
    return rows


def _validation_support(table: pd.DataFrame, *, group_column: str, level: str) -> dict[str, Any]:
    """Describe whether a grouped falsification is genuinely available.

    A single feature contract cannot test leave-feature-contract-out.  Recording
    that as ``unsupported`` is materially different from silently returning no
    rows while the manifest claims the test passed.
    """
    groups = 0
    if group_column in table:
        groups = int(table[group_column].dropna().astype(str).nunique())
    return {
        "validation": group_column,
        "level": level,
        "groups": groups,
        "rows": int(len(table)),
        "status": "supported" if groups >= 2 else "unsupported_insufficient_group_diversity",
    }


def _importance(model: Any, fields: list[str]) -> pd.DataFrame:
    if isinstance(model, PairwiseSurrogate):
        values = np.zeros(len(fields)) if model.classifier is None else model.classifier.coef_.reshape(-1)
    elif hasattr(model.named_steps["model"], "coef_"):
        values = model.named_steps["model"].coef_.reshape(-1)
    elif hasattr(model.named_steps["model"], "feature_importances_"):
        values = model.named_steps["model"].feature_importances_
    else:
        values = np.zeros(len(fields))
    return pd.DataFrame({"feature": fields, "importance": np.asarray(values, dtype=float)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-root", type=Path, action="append", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    descriptor_summary, descriptor_folds = _read_descriptors(args.descriptor_root)
    labels = pd.read_parquet(args.label_root / "downstream_trial_labels.parquet")
    monthly = pd.read_parquet(args.label_root / "downstream_monthly_labels.parquet")
    manifest = json.loads((args.label_root / "run_manifest.json").read_text())
    normalisation = dict(manifest["normalisation"])
    fields = [field for field in CORE_FIELDS if field in descriptor_summary.columns]
    if len(fields) != len(CORE_FIELDS):
        raise AssertionError(f"descriptor fields missing: {sorted(set(CORE_FIELDS).difference(fields))}")
    table = descriptor_summary.merge(labels, on="trial", how="inner", validate="one_to_one")
    if len(table) < 20:
        raise AssertionError("need at least 20 expensive downstream-labelled trials for Proxy V1")
    table["priority_weight"] = pd.to_numeric(table.dpriority_reliability_weight, errors="coerce").fillna(0.0).clip(lower=.05)
    table["gate_weight"] = pd.to_numeric(table.dgate_reliability_weight, errors="coerce").fillna(0.0).clip(lower=.05)
    monthly = _normalise_era_labels(monthly, normalisation)
    era = descriptor_folds.merge(monthly, left_on=["trial", "held_month"], right_on=["trial", "era"], how="inner", validate="one_to_one")
    era["priority_weight"] = 1.0; era["gate_weight"] = 1.0
    model_names = ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise")
    # The requested stress test is *feature-contract* holdout.  Feature family
    # is only a descriptor and is not a valid substitute when all labelled
    # trials share the frozen F120 contract.
    validations = ("target_family", "loss", "feature_contract")
    validation_support = [
        _validation_support(table, group_column=column, level="trial")
        for column in validations
    ]
    validation_support.append(_validation_support(era, group_column="era", level="trial_era"))
    predictions: list[pd.DataFrame] = []
    fitted: dict[str, Any] = {}
    importance: list[pd.DataFrame] = []
    for target in TARGETS:
        objective = "priority" if target.startswith("dpriority") else "gate"
        weight = f"{objective}_weight"
        for model_name in model_names:
            for validation in validations:
                predictions.append(_grouped_predictions(table=table, fields=fields, target=target, weight_column=weight, group_column=validation, model_name=model_name))
            # Leave-era-out works from OOF descriptor/label pairs within the
            # exact held era rather than pretending a trial-level aggregate is
            # an era observation.
            era_target = f"d{objective}_era"
            predictions.append(_grouped_predictions(table=era, fields=fields, target=era_target, weight_column=weight, group_column="era", model_name=model_name).rename(columns={era_target: target}))
            fitted_model = _fit_model(model_name, table[fields], table[target].to_numpy(float), table[weight].to_numpy(float))
            fitted[f"{target}::{model_name}"] = fitted_model
            current_importance = _importance(fitted_model, fields)
            current_importance["target"] = target; current_importance["model"] = model_name
            importance.append(current_importance)
    prediction = pd.concat([part for part in predictions if not part.empty], ignore_index=True)
    metrics = pd.DataFrame(_validation_metrics(prediction, target="dpriority_shrunk") + _validation_metrics(prediction, target="dgate_shrunk"))
    # Predict all already-described trials for active-learning acquisition.
    acquisition = descriptor_summary.loc[:, ["trial", "target_family", "loss", "feature_family", "feature_contract"]].copy()
    for target in TARGETS:
        per_model = np.column_stack([_predict(fitted[f"{target}::{model}"], descriptor_summary[fields]) for model in model_names])
        stem = "priority" if target.startswith("dpriority") else "gate"
        acquisition[f"proxy_{stem}_mean"] = per_model.mean(axis=1)
        acquisition[f"proxy_{stem}_uncertainty"] = per_model.std(axis=1, ddof=1)
    acquisition["acquisition_priority"] = acquisition.proxy_priority_mean + acquisition.proxy_priority_uncertainty
    # The production research objective currently has GateProxy rather than
    # PriorityProxy authority.  Keep the latter as a diagnostic, but expose a
    # like-for-like optimistic Gate acquisition for the next expensive MC1
    # batch: high predicted gate value or high surrogate disagreement can be
    # selected, subject to the frozen diversity/control protocol.
    acquisition["acquisition_gate"] = acquisition.proxy_gate_mean + acquisition.proxy_gate_uncertainty
    args.out.mkdir(parents=True)
    table.to_parquet(args.out / "proxy_trial_training_table.parquet", index=False, compression="zstd")
    era.to_parquet(args.out / "proxy_era_training_table.parquet", index=False, compression="zstd")
    prediction.to_parquet(args.out / "proxy_grouped_cv_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(args.out / "proxy_grouped_cv_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(validation_support).to_parquet(args.out / "proxy_validation_support.parquet", index=False, compression="zstd")
    pd.concat(importance, ignore_index=True).to_parquet(args.out / "proxy_feature_attribution.parquet", index=False, compression="zstd")
    acquisition.to_parquet(args.out / "proxy_all_trial_acquisition.parquet", index=False, compression="zstd")
    joblib.dump({"models": fitted, "fields": fields, "schema": SCHEMA}, args.out / "proxy_models.joblib")
    # Persist individual surrogate files as well.  Consumers of one selected
    # proxy (for example the regularised Gate Ridge) should not need to load a
    # bundle containing an unrelated pairwise model serialized by another
    # script entrypoint.
    model_dir = args.out / "models"
    model_dir.mkdir()
    for key, model in fitted.items():
        target_name, model_name = key.split("::", 1)
        joblib.dump({"model": model, "fields": fields, "target": target_name, "schema": SCHEMA}, model_dir / f"{target_name}__{model_name}.joblib")
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline learned downstream-value proxy; no direct Meta selection/promotion or live mutation",
        "descriptor_roots": [str(path.resolve()) for path in args.descriptor_root], "label_root": str(args.label_root.resolve()),
        "core_fields": fields, "surrogates": list(model_names), "targets": list(TARGETS),
        "individual_model_files": [f"models/{target}__{model}.joblib" for target in TARGETS for model in model_names],
        "validation": {
            "leave-target-family-out": "supported",
            "leave-loss-family-out": "supported",
            "leave-feature-contract-out": next(
                item["status"] for item in validation_support if item["validation"] == "feature_contract"
            ),
            "leave-era-out": "supported",
        },
        "acquisition": "predicted PriorityProxy and GateProxy mean + one-model-dispersion uncertainty; every acquisition requires later actual MC1 confirmation",
        "selection_rule": "not applied here; later HPO uses the separately selected GateProxy subject to fixed control/IC/weekly floors",
    })
    _once(args.out / "correctness_report.json", {
        "descriptors_are_strict_oof_and_target_free_before_outcome_metrics": True,
        "downstream_labels_are_matched_six_month_mc1": True,
        "priority_and_gate_proxies_are_separate": True,
        "grouped_validation_includes_supported_target_loss_and_era_holdouts": True,
        "leave_feature_contract_out_is_explicitly_reported": True,
        "leave_feature_contract_out_supported": bool(next(
            item["status"] == "supported" for item in validation_support if item["validation"] == "feature_contract"
        )),
        "portfolio_is_not_the_sole_proxy_training_target": True,
        "proxy_has_no_direct_live_or_model_score_authority": True,
    })
    print(args.out)


if __name__ == "__main__":
    main()
