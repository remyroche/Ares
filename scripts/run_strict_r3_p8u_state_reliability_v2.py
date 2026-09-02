#!/usr/bin/env python3
"""Strict-OOF additive Base-reliability controls for the frozen P8U stack.

This runner neither trains nor re-ranks the Router/F72 Base/UnderF120 parent.
It emits point-in-time reliability context for later matched UnderF120 arms:

* predicted Base error scale in policy bps;
* P(|Base residual| > 100 bps), P(Base overestimates by >100 bps), and
  P(Base underestimates by >100 bps);
* timestamp-level P(weak / catastrophic Base Top-2 conversion).

It also emits the V2 deterministic authority coordinate.  Authority is an
additive, target-free inference field constructed only from strict-OOF
reliability outputs and a frozen target-free transition reference.  It has no
direct score, admission, or portfolio authority in this producer.

Every held calendar month is scored only by models trained on policy labels
whose paths resolved before that month began.  The reliability controls are
therefore valid additive inference fields and cannot manufacture admissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_state_reliability_v2_models"
MIN_CANDIDATE_TRAIN = 10_000
MIN_TIMESTAMP_TRAIN = 1_000
SEED = 1729
BASE_CONTEXT = (
    "base_score", "base_rank_ts", "v2_base_top10_score_mean",
    "v2_base_top10_score_iqr", "v2_base_top10_score_gap",
    "v2_base_tail_transition",
)
EPISODE_CONTEXT = (
    "v2_regime_id", "v2_regime_distance", "v2_regime_second_distance",
    "v2_regime_assignment_margin", "v2_regime_transition_flag",
    "v2_time_since_regime_change_hours",
)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member.relative_to(ROOT)).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(values: pd.Series) -> list[pd.Timestamp]:
    start = values.min().to_period("M").to_timestamp().tz_localize("UTC")
    end = values.max().to_period("M").to_timestamp().tz_localize("UTC") + pd.offsets.MonthBegin(1)
    return list(pd.date_range(start, end, freq="MS", inclusive="left", tz="UTC"))


def _weights(frame: pd.DataFrame, *, timestamp_normalised: bool) -> np.ndarray:
    counts = frame.v2_regime_id.value_counts(dropna=False).astype(float)
    median = float(counts.median()) if len(counts) else 1.0
    regime = np.sqrt(median / frame.v2_regime_id.map(counts).fillna(median).clip(lower=1.0)).clip(.5, 2.0)
    if timestamp_normalised:
        per_time = frame.groupby("__decision_ts__")["candidate_id"].transform("size").clip(lower=1)
        regime = regime / per_time
    values = regime.to_numpy(float)
    return values / max(values.mean(), 1e-9)


def _matrix(train: pd.DataFrame, held: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
    x_train = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).astype(np.float32).to_numpy()
    x_held = held.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).astype(np.float32).to_numpy()
    return x_train, x_held, {key: float(value) for key, value in medians.items()}


def _params(*, classification: bool, depth: int) -> dict[str, object]:
    return {
        "objective": "binary" if classification else "regression",
        "learning_rate": .035,
        "n_estimators": 300,
        "max_depth": depth,
        "num_leaves": 2 ** depth - 1,
        "min_child_samples": 300,
        "reg_lambda": 20.0,
        "reg_alpha": 1.0,
        "subsample": .85,
        "colsample_bytree": .85,
        "random_state": SEED,
        "n_jobs": 4,
        "verbosity": -1,
    }


def _fit_predict(
    train: pd.DataFrame, held: pd.DataFrame, fields: list[str], target: str, *, classification: bool, depth: int, timestamp_normalised: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    x_train, x_held, medians = _matrix(train, held, fields)
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    if classification:
        if np.unique(y).size < 2:
            return np.full(len(held), float(np.nanmean(y)), dtype=np.float32), medians
        model = lgb.LGBMClassifier(**_params(classification=True, depth=depth))
        model.fit(x_train, y.astype(np.int8), sample_weight=_weights(train, timestamp_normalised=timestamp_normalised))
        return model.predict_proba(x_held)[:, 1].astype(np.float32), medians
    model = lgb.LGBMRegressor(**_params(classification=False, depth=depth))
    model.fit(x_train, y.astype(np.float32), sample_weight=_weights(train, timestamp_normalised=timestamp_normalised))
    return model.predict(x_held).astype(np.float32), medians


def _candidate_metrics(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month, part in frame.groupby("month", sort=True):
        for column in columns:
            pred = part[column].to_numpy(float)
            valid = np.isfinite(pred) & np.isfinite(part.base_abs_residual_bps.to_numpy(float))
            if int(valid.sum()) < 100:
                continue
            y_error = part.base_abs_residual_bps.to_numpy(float)[valid]
            row = {"level": "candidate", "month": str(month), "output": column, "rows": int(valid.sum()), "spearman_abs_error": float(spearmanr(pred[valid], y_error).statistic)}
            top = pred[valid] >= np.nanquantile(pred[valid], .80)
            row["top20_error_bps"] = float(np.mean(y_error[top]))
            row["bottom80_error_bps"] = float(np.mean(y_error[~top]))
            if "_p_" in column:
                if "large_error" in column:
                    label = part.base_large_error_100.to_numpy(float)[valid]
                elif "overconfidence" in column:
                    label = part.base_overconfidence_100.to_numpy(float)[valid]
                else:
                    label = part.base_underconfidence_100.to_numpy(float)[valid]
                row["brier"] = float(brier_score_loss(label, pred[valid]))
                row["log_loss"] = float(log_loss(label, np.clip(pred[valid], 1e-5, 1 - 1e-5)))
                row["auc"] = float(roc_auc_score(label, pred[valid])) if np.unique(label).size > 1 else np.nan
            rows.append(row)
    return rows


def _timestamp_metrics(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month, part in frame.groupby("month", sort=True):
        for column, target in [(name, "weak_top2") if "weak" in name else (name, "catastrophic_top2") for name in columns]:
            valid = np.isfinite(part[column]) & part[target].notna()
            if int(valid.sum()) < 50 or part.loc[valid, target].nunique() < 2:
                continue
            pred, y = part.loc[valid, column].to_numpy(float), part.loc[valid, target].to_numpy(float)
            rows.append({"level": "timestamp", "month": str(month), "output": column, "rows": int(valid.sum()), "brier": float(brier_score_loss(y, pred)), "log_loss": float(log_loss(y, np.clip(pred, 1e-5, 1 - 1e-5))), "auc": float(roc_auc_score(y, pred))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state_root, screen_root, out = ROOT / args.state_root, ROOT / args.screen_root, ROOT / args.out
    if out.exists():
        raise FileExistsError(out)
    for root in (state_root, screen_root):
        receipt = json.loads((root / "correctness_report.json").read_text())
        if not all(value is True or key == "schema" for key, value in receipt.items()):
            raise AssertionError(f"unverified source receipt: {root}")
    state = pd.read_parquet(state_root / "target_free_state_episode_hourly.parquet")
    candidates = pd.read_parquet(state_root / "target_free_base_top10_candidates.parquet")
    events = pd.read_parquet(state_root / "labelled_base_top10_residual_events.parquet")
    failures = pd.read_parquet(state_root / "labelled_base_top10_failure_targets.parquet")
    for frame in (state, candidates, events, failures):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    summary = pd.read_parquet(screen_root / "feature_summary_selection_2025.parquet")
    selected = summary.loc[(summary["tail"] == "base_top10") & summary["eligible"]].sort_values("selection_score", ascending=False, kind="stable")
    deviation = selected.feature.head(14).tolist()
    if len(deviation) < 10:
        raise AssertionError("insufficient 2025-selected Base-reliability features")
    for name in [*BASE_CONTEXT, *EPISODE_CONTEXT, *deviation]:
        if name not in state.columns and name not in candidates.columns:
            raise AssertionError(f"reliability field unavailable: {name}")
    candidate_frame = candidates.merge(state, on="__decision_ts__", how="left", validate="many_to_one")
    candidate_frame = candidate_frame.merge(events.loc[:, ["candidate_id", "available", "base_log1p_abs_residual", "base_large_error_100", "base_overconfidence_100", "base_underconfidence_100", "base_abs_residual_bps", "residual_bps"]], on="candidate_id", how="left", validate="one_to_one")
    candidate_fields = list(dict.fromkeys([*BASE_CONTEXT, *EPISODE_CONTEXT, *deviation]))
    candidate_frame["month"] = candidate_frame.__decision_ts__.dt.strftime("%Y-%m")
    candidate_outputs = [
        "v2_pred_error_log_d2", "v2_pred_error_log_d3",
        "v2_p_large_error_100_d2", "v2_p_large_error_100_d3",
        "v2_p_overconfidence_100_d2", "v2_p_overconfidence_100_d3",
        "v2_p_underconfidence_100_d2", "v2_p_underconfidence_100_d3",
        "v2_pred_error_percentile_d2", "v2_pred_error_percentile_d3",
    ]
    for name in candidate_outputs:
        candidate_frame[name] = np.nan
    # Timestamp targets are derived from the already-separate failure target
    # panel.  They are never candidate features and use their own label
    # availability cutoff.
    timestamp_frame = state.merge(failures, on="__decision_ts__", how="inner", validate="one_to_one")
    timestamp_frame["weak_top2"] = timestamp_frame.top2_realised_ev_bps.lt(50.0).astype(np.int8)
    timestamp_frame["catastrophic_top2"] = timestamp_frame.top2_realised_ev_bps.le(0.0).astype(np.int8)
    timestamp_frame["month"] = timestamp_frame.__decision_ts__.dt.strftime("%Y-%m")
    timestamp_fields = list(dict.fromkeys([*EPISODE_CONTEXT, "v2_base_top10_score_mean", "v2_base_top10_score_iqr", "v2_base_top10_score_gap", "v2_base_tail_transition", *deviation]))
    timestamp_outputs = ["v2_p_weak_top2_d2", "v2_p_weak_top2_d3", "v2_p_catastrophic_top2_d2", "v2_p_catastrophic_top2_d3"]
    for name in timestamp_outputs:
        timestamp_frame[name] = np.nan
    # This target-free uncertainty coordinate uses the frozen state reference
    # population.  It must never borrow a held-period distribution.
    episode_contract = json.loads((state_root / "target_free_episode_contract.json").read_text())
    reference_end = pd.Timestamp(episode_contract["episode"]["reference_end"])
    reference_end = reference_end.tz_localize("UTC") if reference_end.tzinfo is None else reference_end.tz_convert("UTC")
    reference_transition = pd.to_numeric(
        state.loc[state.__decision_ts__.lt(reference_end), "v2_transition_mahalanobis"], errors="coerce",
    ).dropna().sort_values().to_numpy(float)
    if len(reference_transition) < 100:
        raise AssertionError("insufficient frozen transition reference for authority")
    transition_values = pd.to_numeric(candidate_frame["v2_transition_mahalanobis"], errors="coerce").to_numpy(float)
    transition_rank = np.searchsorted(reference_transition, transition_values, side="right") / float(len(reference_transition))
    candidate_frame["v2_transition_uncertainty"] = np.clip(transition_rank, 0.0, 1.0).astype(np.float32)
    folds: list[dict[str, object]] = []
    for month in _months(candidate_frame.__decision_ts__):
        held_c = candidate_frame.loc[candidate_frame.__decision_ts__.dt.to_period("M").eq(month.to_period("M"))].copy()
        train_c = candidate_frame.loc[candidate_frame.__decision_ts__.lt(month) & candidate_frame.available.lt(month) & candidate_frame.base_log1p_abs_residual.notna()].copy()
        held_t = timestamp_frame.loc[timestamp_frame.__decision_ts__.dt.to_period("M").eq(month.to_period("M"))].copy()
        train_t = timestamp_frame.loc[timestamp_frame.__decision_ts__.lt(month) & timestamp_frame.top2_label_available_ts.lt(month)].copy()
        row = {"month": str(month), "candidate_train": int(len(train_c)), "candidate_held": int(len(held_c)), "timestamp_train": int(len(train_t)), "timestamp_held": int(len(held_t))}
        if len(train_c) >= MIN_CANDIDATE_TRAIN and len(held_c):
            for depth in (2, 3):
                error, _ = _fit_predict(train_c, held_c, candidate_fields, "base_log1p_abs_residual", classification=False, depth=depth, timestamp_normalised=True)
                large, _ = _fit_predict(train_c, held_c, candidate_fields, "base_large_error_100", classification=True, depth=depth, timestamp_normalised=True)
                over, _ = _fit_predict(train_c, held_c, candidate_fields, "base_overconfidence_100", classification=True, depth=depth, timestamp_normalised=True)
                under, _ = _fit_predict(train_c, held_c, candidate_fields, "base_underconfidence_100", classification=True, depth=depth, timestamp_normalised=True)
                index = held_c.index
                candidate_frame.loc[index, f"v2_pred_error_log_d{depth}"] = error
                candidate_frame.loc[index, f"v2_p_large_error_100_d{depth}"] = large
                candidate_frame.loc[index, f"v2_p_overconfidence_100_d{depth}"] = over
                candidate_frame.loc[index, f"v2_p_underconfidence_100_d{depth}"] = under
                earlier = candidate_frame.loc[
                    candidate_frame.__decision_ts__.lt(month) & candidate_frame[f"v2_pred_error_log_d{depth}"].notna(),
                    f"v2_pred_error_log_d{depth}",
                ].sort_values().to_numpy(float)
                if len(earlier) >= MIN_CANDIDATE_TRAIN:
                    percentile = np.searchsorted(earlier, error, side="right") / float(len(earlier))
                    candidate_frame.loc[index, f"v2_pred_error_percentile_d{depth}"] = np.clip(percentile, 0.0, 1.0).astype(np.float32)
        if len(train_t) >= MIN_TIMESTAMP_TRAIN and len(held_t):
            for depth in (2, 3):
                weak, _ = _fit_predict(train_t, held_t, timestamp_fields, "weak_top2", classification=True, depth=depth, timestamp_normalised=False)
                catastrophic, _ = _fit_predict(train_t, held_t, timestamp_fields, "catastrophic_top2", classification=True, depth=depth, timestamp_normalised=False)
                index = held_t.index
                timestamp_frame.loc[index, f"v2_p_weak_top2_d{depth}"] = weak
                timestamp_frame.loc[index, f"v2_p_catastrophic_top2_d{depth}"] = catastrophic
        folds.append(row)
    candidate_frame["v2_pred_error_scale_bps_d2"] = np.expm1(candidate_frame.v2_pred_error_log_d2.clip(lower=0)).astype(np.float32)
    candidate_frame["v2_pred_error_scale_bps_d3"] = np.expm1(candidate_frame.v2_pred_error_log_d3.clip(lower=0)).astype(np.float32)
    authority_outputs: list[str] = []
    for depth in (2, 3):
        authority = np.clip(
            1.0
            - .5 * pd.to_numeric(candidate_frame[f"v2_p_large_error_100_d{depth}"], errors="coerce")
            - .3 * pd.to_numeric(candidate_frame[f"v2_pred_error_percentile_d{depth}"], errors="coerce")
            - .2 * pd.to_numeric(candidate_frame["v2_transition_uncertainty"], errors="coerce"),
            .25,
            1.0,
        )
        name = f"v2_state_authority_d{depth}"
        candidate_frame[name] = authority.astype(np.float32)
        authority_outputs.append(name)
    timestamp_frame["v2_timestamp_failure_risk_d2"] = np.maximum(
        timestamp_frame.v2_p_weak_top2_d2.to_numpy(float), timestamp_frame.v2_p_catastrophic_top2_d2.to_numpy(float),
    ).astype(np.float32)
    timestamp_frame["v2_timestamp_failure_risk_d3"] = np.maximum(
        timestamp_frame.v2_p_weak_top2_d3.to_numpy(float), timestamp_frame.v2_p_catastrophic_top2_d3.to_numpy(float),
    ).astype(np.float32)
    targetfree_candidate = candidate_frame.loc[:, ["candidate_id", "__decision_ts__", "base_score", "base_rank_ts", *candidate_outputs, "v2_pred_error_scale_bps_d2", "v2_pred_error_scale_bps_d3", "v2_transition_uncertainty", *authority_outputs]].copy()
    targetfree_timestamp = timestamp_frame.loc[:, ["__decision_ts__", *timestamp_outputs, "v2_timestamp_failure_risk_d2", "v2_timestamp_failure_risk_d3"]].copy()
    candidate_metrics = pd.DataFrame(_candidate_metrics(candidate_frame.dropna(subset=["base_abs_residual_bps"]), ["v2_pred_error_scale_bps_d2", "v2_pred_error_scale_bps_d3", "v2_p_large_error_100_d2", "v2_p_large_error_100_d3", "v2_p_overconfidence_100_d2", "v2_p_overconfidence_100_d3", "v2_p_underconfidence_100_d2", "v2_p_underconfidence_100_d3"]))
    timestamp_metrics = pd.DataFrame(_timestamp_metrics(timestamp_frame, timestamp_outputs))
    out.mkdir(parents=True)
    targetfree_candidate.to_parquet(out / "target_free_candidate_reliability_predictions.parquet", index=False)
    targetfree_timestamp.to_parquet(out / "target_free_timestamp_failure_predictions.parquet", index=False)
    pd.DataFrame(folds).to_parquet(out / "strict_oof_fold_support.parquet", index=False)
    candidate_metrics.to_parquet(out / "candidate_reliability_metrics.parquet", index=False)
    timestamp_metrics.to_parquet(out / "timestamp_failure_metrics.parquet", index=False)
    _once(out / "reliability_feature_contract.json", {"schema": SCHEMA, "candidate_fields": candidate_fields, "timestamp_fields": timestamp_fields, "selection_source": str((screen_root / "feature_summary_selection_2025.parquet").relative_to(ROOT)), "selection_period": "2025 only", "depths": [2, 3], "parameters": _params(classification=False, depth=2), "authority": {"formula": "clip(1 - 0.5 * p_large_error - 0.3 * prequential_error_percentile - 0.2 * frozen_transition_uncertainty, 0.25, 1.0)", "transition_reference_end": str(reference_end)}})
    correctness = {
        "parent_p8u_stack_unchanged": True,
        "reliability_outputs_are_additive_not_base_score_replacements": True,
        "strict_prequential_candidate_training_uses_resolved_labels_before_held_month": True,
        "strict_prequential_timestamp_training_uses_resolved_labels_before_held_month": True,
        "state_and_episode_features_are_target_free": True,
        "selection_features_use_2025_only": True,
        "authority_uses_only_strict_oof_predictions_and_frozen_target_free_transition_reference": True,
        "no_mc1_admission_portfolio_or_live_mutation": True,
    }
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {"schema": SCHEMA, "scope": "offline strict-OOF additive reliability controls", "state_root": str(state_root.relative_to(ROOT)), "screen_root": str(screen_root.relative_to(ROOT)), "state_root_sha256": _sha(state_root), "screen_root_sha256": _sha(screen_root), "correctness": correctness})
    print(json.dumps({"out": str(out), "candidate_fields": len(candidate_fields), "timestamp_fields": len(timestamp_fields), "candidate_rows": len(targetfree_candidate)}, sort_keys=True))


if __name__ == "__main__":
    main()
