#!/usr/bin/env python3
"""Strict-OOF, side-local hurdle/distributional execution-EV ablation.

The direct exact-net residual remains the primary score.  This runner tests
whether economically interpretable support heads improve that score only via
causal compositions: gross-over-cost incidence, conditional favourable and
adverse magnitude, actual exit-policy stop/timeout mixture, a true joint
multi-output direct-primary model, and a frozen-head two-stage residual.

It intentionally does not contain timing, wait, MAE or target-price actions.
Those belong to the separate action layer and must not silently become entry
ranking inputs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import execution_ev_metrics
from scripts.diagnose_within_july_opportunity_capture import economic_components
from scripts.run_execution_ev_mixed_period_remedies import (
    ARCHETYPE_COLUMN,
    BASELINE_COLUMN,
    DECISION_COLUMN,
    DEFAULT_WINDOWS,
    IDENTITY_COLUMNS,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    _model_features,
    apply_canonical_recent_mapping,
)
from scripts.run_strict_oof_ic_ev_conversion_ablation import (
    MIN_TRAIN_ROWS,
    PAYOFF_CAP,
    SIDES,
    _causal_ev_calibration,
    _causal_probability_calibration,
    _fit_binary,
    _fit_regression,
    _load_features,
    _strict_forward_split,
    _strict_temporal_blocks,
    prepare_conversion_frame,
)


SCHEMA = "exact_strict_oof_hurdle_distributional_ablation_v1"
ARMS = (
    "direct_net_residual",
    "gross_cost_hurdle_ev",
    "exit_policy_mixture_ev",
    "direct_exit_blend_050",
    "joint_multitask_direct_primary",
    "two_stage_stopped_gradient_residual",
)
HEADS = (
    ("gross_exceeds_cost", "target_gross_exceeds_cost", None),
    ("full_stop", "target_full_stop", None),
    ("timeout_exit", "target_timeout_exit", None),
)
EPSILON = 1e-12
FROZEN_CONTROLS = {
    "exact_policy_decomposed_hurdle_v3": ROOT
    / "data_perp/artifacts/exact_policy_decomposed_hurdle_ablation_20260727_v3",
    "historical_gross_hurdle_v2": ROOT
    / "data_perp/artifacts/historical_execution_ev_gross_hurdle_decomposition_20260729_v2",
}


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_distributional_targets(frame: pd.DataFrame) -> None:
    """Add exact, net-of-cost heads and prove the cost reconciliation once."""

    required = {"execution_gross_ev_12h", "execution_cost_return", "execution_exit_reason"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("exact-policy input lacks required distributional fields: " + ", ".join(missing))
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame["execution_cost_return"], errors="raise").to_numpy(float)
    net = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(float)
    if not (np.isfinite(gross).all() and np.isfinite(cost).all() and np.isfinite(net).all()):
        raise ValueError("gross/cost/net labels must be finite")
    if not np.allclose(gross - cost, net, rtol=0.0, atol=1e-12):
        raise ValueError("gross-minus-cost does not exactly reconcile to net target")
    positive = gross > cost
    # This deliberately records the economic predicate instead of assuming a
    # sign convention.  The equality check above makes it equivalent to net>0.
    if not np.array_equal(positive, net > 0.0):
        raise ValueError("gross-over-cost hurdle is inconsistent with exact net sign")
    reason = frame["execution_exit_reason"].astype(str).str.lower().to_numpy()
    frame["target_gross_exceeds_cost"] = positive.astype(np.int8)
    frame["target_positive_net_clipped"] = np.clip(np.maximum(net, 0.0), 0.0, PAYOFF_CAP)
    frame["target_loss_net_clipped"] = np.clip(np.maximum(-net, 0.0), 0.0, PAYOFF_CAP)
    frame["target_full_stop"] = np.isin(reason, ("full_sl", "full_stop")).astype(np.int8)
    frame["target_timeout_exit"] = (reason == "timeout").astype(np.int8)
    frame["target_other_exit"] = (
        ~((frame["target_full_stop"].to_numpy(bool)) | (frame["target_timeout_exit"].to_numpy(bool)))
    ).astype(np.int8)
    if not np.all(
        frame[["target_full_stop", "target_timeout_exit", "target_other_exit"]]
        .to_numpy(int).sum(axis=1)
        == 1
    ):
        raise ValueError("actual exit mixture is not exhaustive")
    for name, mask in (
        ("positive_payoff", positive),
        ("loss_payoff", ~positive),
        ("full_stop_payoff", frame["target_full_stop"].to_numpy(bool)),
        ("timeout_payoff", frame["target_timeout_exit"].to_numpy(bool)),
        ("other_exit_payoff", frame["target_other_exit"].to_numpy(bool)),
    ):
        if not mask.any():
            raise ValueError(f"distributional target {name} has no support")


def prepare_frame(frame: pd.DataFrame, grid: pd.DataFrame, *, grid_name: str) -> pd.DataFrame:
    work = prepare_conversion_frame(frame, grid, grid_name=grid_name)
    add_distributional_targets(work)
    return work


def _fit_joint_multitask(
    fit_x: pd.DataFrame,
    score_x: pd.DataFrame,
    fit: pd.DataFrame,
    baseline: np.ndarray,
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, str]:
    """CatBoost MultiRMSE with direct target repeated to keep it primary.

    Direct-score output is the only ranking output from this model.  The other
    outputs are training-time support tasks; they cannot enter the rank score.
    """

    direct = fit[TARGET_COLUMN].to_numpy(float) - fit[BASELINE_COLUMN].to_numpy(float)
    labels = np.column_stack(
        [
            direct,
            direct,
            direct,
            fit["target_gross_exceeds_cost"].to_numpy(float),
            fit["target_positive_net_clipped"].to_numpy(float),
            fit["target_loss_net_clipped"].to_numpy(float),
            fit["target_full_stop"].to_numpy(float),
            fit["target_timeout_exit"].to_numpy(float),
        ]
    )
    centre = labels.mean(axis=0)
    scale = np.maximum(labels.std(axis=0), 1e-4)
    try:
        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=int(iterations),
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=8.0,
            random_strength=0.5,
            random_seed=int(seed),
            thread_count=int(n_jobs),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(fit_x, (labels - centre) / scale)
        prediction = np.asarray(model.predict(score_x), dtype=float)
        if prediction.ndim != 2 or prediction.shape[1] != labels.shape[1]:
            raise RuntimeError("CatBoost MultiRMSE returned an unexpected prediction shape")
        direct_prediction = prediction[:, 0] * scale[0] + centre[0]
        return baseline + direct_prediction, "catboost_multirmse_direct_x3_plus_5_support"
    except Exception as exc:  # model availability differs across CatBoost builds
        fallback, constant = _fit_regression(
            fit_x, direct, iterations=iterations, seed=seed, n_jobs=n_jobs
        )
        prediction = fallback.predict(score_x) if fallback is not None else np.full(len(score_x), constant)
        return baseline + np.asarray(prediction, dtype=float), f"direct_fallback_multirmse_unavailable:{type(exc).__name__}"


def _fit_raw_heads(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], str]:
    fit_x = _model_features(fit, fit, feature_columns, trust_composites=False)
    score_x = _model_features(fit, score, feature_columns, trust_composites=False)
    direct_model, direct_constant = _fit_regression(
        fit_x,
        fit[TARGET_COLUMN].to_numpy(float) - fit[BASELINE_COLUMN].to_numpy(float),
        iterations=iterations,
        seed=seed,
        n_jobs=n_jobs,
    )
    direct_prediction = _predict_regression(direct_model, direct_constant, score_x)
    output: dict[str, np.ndarray] = {
        "direct_net": score[BASELINE_COLUMN].to_numpy(float) + direct_prediction,
    }
    for offset, (name, target, _) in enumerate(HEADS, start=10):
        model, constant = _fit_binary(
            fit_x, fit[target].to_numpy(np.int8), iterations=iterations, seed=seed + offset, n_jobs=n_jobs
        )
        output[f"raw_{name}"] = _predict_binary(model, constant, score_x)
    positive = fit["target_gross_exceeds_cost"].to_numpy(bool)
    loss = ~positive
    conditional_specs = (
        ("positive_magnitude", positive, "target_positive_net_clipped"),
        ("loss_magnitude", loss, "target_loss_net_clipped"),
        ("full_stop_payoff", fit["target_full_stop"].to_numpy(bool), TARGET_COLUMN),
        ("timeout_payoff", fit["target_timeout_exit"].to_numpy(bool), TARGET_COLUMN),
        ("other_exit_payoff", fit["target_other_exit"].to_numpy(bool), TARGET_COLUMN),
    )
    for offset, (name, mask, target) in enumerate(conditional_specs, start=30):
        model, constant = _fit_regression(
            fit_x.loc[mask], fit.loc[mask, target].to_numpy(float),
            iterations=iterations, seed=seed + offset, n_jobs=n_jobs,
        )
        output[name] = _predict_regression(model, constant, score_x)
    joint, joint_status = _fit_joint_multitask(
        fit_x, score_x, fit, score[BASELINE_COLUMN].to_numpy(float),
        iterations=iterations, seed=seed + 100, n_jobs=n_jobs,
    )
    output["joint_multitask_direct_primary"] = joint
    return output, joint_status


def _predict_binary(model: Any | None, constant: float, x: pd.DataFrame) -> np.ndarray:
    return np.full(len(x), constant, dtype=float) if model is None else np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _predict_regression(model: Any | None, constant: float, x: pd.DataFrame) -> np.ndarray:
    return np.full(len(x), constant, dtype=float) if model is None else np.asarray(model.predict(x), dtype=float)


def compose_scores(raw: Mapping[str, np.ndarray], probability: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    direct = np.asarray(raw["direct_net"], dtype=float)
    p_gain = np.asarray(probability["gross_exceeds_cost"], dtype=float)
    pos = np.clip(np.asarray(raw["positive_magnitude"], dtype=float), 0.0, PAYOFF_CAP)
    loss = np.clip(np.asarray(raw["loss_magnitude"], dtype=float), 0.0, PAYOFF_CAP)
    hurdle = p_gain * pos - (1.0 - p_gain) * loss
    p_stop = np.asarray(probability["full_stop"], dtype=float)
    p_timeout = np.asarray(probability["timeout_exit"], dtype=float)
    p_other = np.clip(1.0 - p_stop - p_timeout, 0.0, 1.0)
    mass = p_stop + p_timeout + p_other
    # Earliest chronological OOF rows deliberately remain wholly unscored.
    # Reject only malformed *partially scored* exit rows, not that valid warmup.
    finite_probability = np.isfinite(p_stop) & np.isfinite(p_timeout)
    if (mass[finite_probability] <= EPSILON).any():
        raise ValueError("exit-policy probability mass is invalid")
    p_stop, p_timeout, p_other = p_stop / mass, p_timeout / mass, p_other / mass
    exit_ev = (
        p_stop * np.clip(np.asarray(raw["full_stop_payoff"], dtype=float), -PAYOFF_CAP, PAYOFF_CAP)
        + p_timeout * np.clip(np.asarray(raw["timeout_payoff"], dtype=float), -PAYOFF_CAP, PAYOFF_CAP)
        + p_other * np.clip(np.asarray(raw["other_exit_payoff"], dtype=float), -PAYOFF_CAP, PAYOFF_CAP)
    )
    finite = np.isfinite(direct) & np.isfinite(hurdle) & np.isfinite(exit_ev)
    result = {
        "direct_net_residual": direct,
        "gross_cost_hurdle_ev": hurdle,
        "exit_policy_mixture_ev": exit_ev,
        "direct_exit_blend_050": 0.5 * direct + 0.5 * exit_ev,
        "joint_multitask_direct_primary": np.asarray(raw["joint_multitask_direct_primary"], dtype=float),
    }
    for name, value in result.items():
        partial = np.isfinite(value) & ~finite
        if partial.any():
            raise ValueError(f"{name} has a partial raw-head record")
    return result


def _two_stage(
    train: pd.DataFrame,
    raw_columns: Sequence[str],
    target: np.ndarray,
    availability: pd.Series,
    evaluation_raw: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Causal stopped-gradient residual learner over frozen OOF head outputs."""

    output = np.full(len(train), np.nan, dtype=float)
    final = np.full(len(evaluation_raw), np.nan, dtype=float)
    report: list[dict[str, Any]] = []
    oof_fold = train["oof_fold"].to_numpy(int)
    available = pd.to_datetime(availability, utc=True, errors="raise")
    for fold in sorted(set(oof_fold) - {0}):
        current = oof_fold == fold
        cutoff = pd.to_datetime(train.loc[current, DECISION_COLUMN], utc=True).min()
        reference = (oof_fold > 0) & (oof_fold < fold) & available.lt(cutoff).to_numpy()
        reference &= train.loc[:, list(raw_columns)].notna().all(axis=1).to_numpy()
        if int(reference.sum()) < 500:
            output[current] = train.loc[current, "direct_net_residual"].to_numpy(float)
            report.append({"fold": int(fold), "rows": int(current.sum()), "reference_rows": int(reference.sum()), "status": "direct_fallback_insufficient_prior_oof"})
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        baseline = train.loc[reference, "direct_net_residual"].to_numpy(float)
        model.fit(train.loc[reference, list(raw_columns)], target[reference] - baseline)
        output[current] = train.loc[current, "direct_net_residual"].to_numpy(float) + model.predict(train.loc[current, list(raw_columns)])
        report.append({"fold": int(fold), "rows": int(current.sum()), "reference_rows": int(reference.sum()), "status": "ridge_stopped_gradient"})
    reference = (oof_fold > 0) & available.lt(evaluation_start).to_numpy()
    reference &= train.loc[:, list(raw_columns)].notna().all(axis=1).to_numpy()
    if int(reference.sum()) < 500:
        final = evaluation_raw["direct_net_residual"].to_numpy(float)
        report.append({"fold": "evaluation", "rows": int(len(final)), "reference_rows": int(reference.sum()), "status": "direct_fallback_insufficient_prior_oof"})
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        baseline = train.loc[reference, "direct_net_residual"].to_numpy(float)
        model.fit(train.loc[reference, list(raw_columns)], target[reference] - baseline)
        final = evaluation_raw["direct_net_residual"].to_numpy(float) + model.predict(evaluation_raw.loc[:, list(raw_columns)])
        report.append({"fold": "evaluation", "rows": int(len(final)), "reference_rows": int(reference.sum()), "status": "ridge_stopped_gradient"})
    return output, final, report


def _fit_side(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    blocks = _strict_temporal_blocks(train)
    raw_names = (
        "direct_net", "raw_gross_exceeds_cost", "raw_full_stop", "raw_timeout_exit",
        "positive_magnitude", "loss_magnitude", "full_stop_payoff", "timeout_payoff",
        "other_exit_payoff", "joint_multitask_direct_primary",
    )
    raw = {name: np.full(len(train), np.nan, dtype=float) for name in raw_names}
    fold_id = np.zeros(len(train), dtype=np.int16)
    reports: dict[str, Any] = {"folds": [], "probability_calibration": {}, "joint_multitask": []}
    for number, (fit_pos, valid_pos, start) in enumerate(blocks, start=1):
        fitted, status = _fit_raw_heads(train.iloc[fit_pos], train.iloc[valid_pos], feature_columns, iterations=iterations, seed=seed + 100 * number, n_jobs=n_jobs)
        for name in raw_names:
            raw[name][valid_pos] = fitted[name]
        fold_id[valid_pos] = number
        reports["folds"].append({"fold": number, "fit_rows": int(len(fit_pos)), "validation_rows": int(len(valid_pos)), "validation_start_utc": start, "max_fit_support_label_available_utc": pd.to_datetime(train.iloc[fit_pos]["support_label_available_utc"], utc=True).max()})
        reports["joint_multitask"].append({"fold": number, "status": status})
    final, final_status = _fit_raw_heads(train, evaluation, feature_columns, iterations=iterations, seed=seed + 9_000, n_jobs=n_jobs)
    reports["joint_multitask"].append({"fold": "evaluation", "status": final_status})
    columns = [*IDENTITY_COLUMNS, DECISION_COLUMN, "support_label_available_utc", TARGET_COLUMN, "target_gross_exceeds_cost", "target_full_stop", "target_timeout_exit"]
    oof = train.loc[:, columns].copy()
    scored = evaluation.loc[:, columns].copy()
    oof["oof_fold"] = fold_id
    for name in raw_names:
        oof[name] = raw[name]
        scored[name] = final[name]
    probabilities: dict[str, np.ndarray] = {}
    final_probabilities: dict[str, np.ndarray] = {}
    evaluation_start = pd.to_datetime(evaluation[DECISION_COLUMN], utc=True).min()
    for name, target, _ in HEADS:
        calibrated, final_calibrated, calibration_report = _causal_probability_calibration(
            raw[f"raw_{name}"], train[target].to_numpy(float), fold_id,
            train["support_label_available_utc"], final[f"raw_{name}"], evaluation_start,
        )
        probabilities[name], final_probabilities[name] = calibrated, final_calibrated
        oof[f"p_{name}"] = calibrated
        scored[f"p_{name}"] = final_calibrated
        reports["probability_calibration"][name] = calibration_report
    oof_scores = compose_scores(raw, probabilities)
    final_scores = compose_scores(final, final_probabilities)
    for arm, values in oof_scores.items():
        oof[arm] = values
        scored[arm] = final_scores[arm]
    raw_columns = ("direct_net_residual", "gross_cost_hurdle_ev", "exit_policy_mixture_ev", "joint_multitask_direct_primary")
    staged_oof, staged_final, staged_report = _two_stage(
        oof, raw_columns, train[TARGET_COLUMN].to_numpy(float), train["support_label_available_utc"], scored, evaluation_start
    )
    oof["two_stage_stopped_gradient_residual"] = staged_oof
    scored["two_stage_stopped_gradient_residual"] = staged_final
    reports["two_stage"] = staged_report
    for arm in ARMS:
        mapped_oof, mapped_final = _causal_ev_calibration(
            oof[arm].to_numpy(float), train[TARGET_COLUMN].to_numpy(float), fold_id,
            train["support_label_available_utc"], scored[arm].to_numpy(float), evaluation_start,
        )
        oof[f"side_causal_oof_ev_{arm}"] = mapped_oof
        scored[f"side_causal_oof_ev_{arm}"] = mapped_final
    return oof, scored, reports


def _head_metrics(ledger: pd.DataFrame, *, window: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ("pooled_global", *SIDES):
        subset = ledger if scope == "pooled_global" else ledger.loc[ledger[SIDE_COLUMN].eq(scope)]
        for name, target, _ in HEADS:
            valid = subset[target].notna() & subset[f"p_{name}"].notna()
            y = subset.loc[valid, target].to_numpy(int)
            p = subset.loc[valid, f"p_{name}"].to_numpy(float)
            if not len(y):
                continue
            row: dict[str, Any] = {"window": window, "scope": scope, "head": name, "rows": int(len(y)), "prevalence": float(y.mean()), "brier": float(brier_score_loss(y, p)), "calibration_bias": float(p.mean() - y.mean())}
            if np.unique(y).size == 2:
                row.update(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))
            rows.append(row)
    return rows


def _select_indices(evaluation: pd.DataFrame, score: np.ndarray, fraction: float) -> np.ndarray:
    """Pooled global top-k with an explicit stable immutable-ID tie rule."""

    valid = np.flatnonzero(np.isfinite(score))
    count = max(1, int(np.ceil(float(fraction) * len(valid))))
    ranking = evaluation.iloc[valid].loc[:, list(IDENTITY_COLUMNS)].copy()
    ranking["_score"] = np.asarray(score, dtype=float)[valid]
    ranking["_position"] = valid
    # Candidate ID alone is required by the request; the remainder of the
    # immutable identity makes the rule total if a legacy candidate ID repeats.
    for column in IDENTITY_COLUMNS:
        ranking[column] = ranking[column].astype(str)
    ranking = ranking.sort_values(
        ["_score", "candidate_id", "__ts__", "__symbol__", SIDE_COLUMN],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return ranking["_position"].to_numpy(int)[:count]


def _tie_diagnostics(score: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(score, dtype=float)[np.isfinite(score)]
    if not len(finite) or not len(selected):
        return {"score_distinct_values": 0, "cutoff_tie_rows": 0, "cutoff_tie_selected_rows": 0}
    cutoff = float(np.asarray(score, dtype=float)[selected[-1]])
    tie_rows = np.flatnonzero(np.isfinite(score) & np.isclose(score, cutoff, rtol=0.0, atol=1e-14))
    return {
        "score_distinct_values": int(pd.Series(finite).nunique(dropna=True)),
        "cutoff_score": cutoff,
        "cutoff_tie_rows": int(len(tie_rows)),
        "cutoff_tie_selected_rows": int(np.isin(selected, tie_rows).sum()),
        "cutoff_tie_fraction_of_selected": float(np.isin(selected, tie_rows).mean()),
    }


def _arm_metrics(evaluation: pd.DataFrame, scores: Mapping[str, np.ndarray], *, window: str, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = evaluation[TARGET_COLUMN].to_numpy(float)
    decision = pd.to_datetime(evaluation[DECISION_COLUMN], utc=True)
    latest_month = decision.dt.to_period("M").max()
    latest_week_start = decision.max() - pd.Timedelta(days=7)
    for arm, score in scores.items():
        for fraction in (0.01, 0.05, 0.10, 0.20):
            selected = _select_indices(evaluation, np.asarray(score, float), fraction)
            chosen = evaluation.iloc[selected]
            metric = execution_ev_metrics(target, np.asarray(score, float), top_k_fraction=fraction)
            # execution_ev_metrics is retained for all-row error/correlation
            # metrics; replace its top-k result with the explicit candidate-ID
            # tie rule above.
            metric["top_k_rows"] = int(len(chosen))
            metric["top_k_mean_net_ev"] = float(chosen[TARGET_COLUMN].mean())
            metric["top_k_sum_net_ev"] = float(chosen[TARGET_COLUMN].sum())
            metric["top_k_predicted_net_ev"] = float(np.asarray(score, float)[selected].mean())
            metric["top_k_positive_ev_rate"] = float(chosen[TARGET_COLUMN].gt(0).mean())
            month_selected = chosen.loc[pd.to_datetime(chosen[DECISION_COLUMN], utc=True).dt.to_period("M").eq(latest_month)]
            week_selected = chosen.loc[pd.to_datetime(chosen[DECISION_COLUMN], utc=True).ge(latest_week_start)]
            rows.append({
                "window": window, "stage": stage, "arm": arm, "scope": "pooled_global",
                "top_fraction": fraction, "coverage_rate": float(np.isfinite(score).mean()),
                **metric,
                "selected_long_share": float(chosen[SIDE_COLUMN].eq("long").mean()),
                "selected_short_share": float(chosen[SIDE_COLUMN].eq("short").mean()),
                "selected_latest_month_rows": int(pd.to_datetime(chosen[DECISION_COLUMN], utc=True).dt.to_period("M").eq(latest_month).sum()),
                "selected_latest_week_rows": int(pd.to_datetime(chosen[DECISION_COLUMN], utc=True).ge(latest_week_start).sum()),
                "selected_latest_month_mean_net_bps": float(month_selected[TARGET_COLUMN].mean() * 1e4) if len(month_selected) else np.nan,
                "selected_latest_week_mean_net_bps": float(week_selected[TARGET_COLUMN].mean() * 1e4) if len(week_selected) else np.nan,
                "selected_latest_month_positive_rate": float(month_selected[TARGET_COLUMN].gt(0).mean()) if len(month_selected) else np.nan,
                "selected_latest_week_positive_rate": float(week_selected[TARGET_COLUMN].gt(0).mean()) if len(week_selected) else np.nan,
                **_tie_diagnostics(np.asarray(score, float), selected),
                **{f"selected_{key}": value for key, value in economic_components(chosen).items() if key != "rows"},
            })
    return rows


def _frozen_control_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bind prior negative controls; never merge their non-identical rows.

    They are comparison context only.  The new experiment's direct and two
    stage arms are compared on its own fixed strict-OOF rows, while this table
    makes the earlier probability×magnitude result explicit and prevents it
    being rediscovered or described as a novel hurdle result.
    """

    rows: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    for name, directory in FROZEN_CONTROLS.items():
        manifest = directory / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"required frozen control is missing: {manifest}")
        bindings[name] = {"path": str(manifest), "sha256": _sha256(manifest)}
        metrics = directory / "hurdle_metrics.csv"
        if metrics.exists():
            table = pd.read_csv(metrics)
            required = {"window", "stage", "arm", "scope", "top_k_mean_net_ev"}
            if required.issubset(table.columns):
                subset = table.loc[
                    table["scope"].astype(str).eq("pooled_global")
                    & table["stage"].astype(str).eq("canonical_recent_ev_mapping")
                ].copy()
                for _, item in subset.iterrows():
                    rows.append({
                        "control": name,
                        "window": str(item["window"]),
                        "arm": str(item["arm"]),
                        "mapped_top10_net_bps": float(item["top_k_mean_net_ev"]) * 1e4,
                        "comparison_status": "frozen_nonidentical_row_context_only",
                    })
    return pd.DataFrame(rows), bindings


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    frame = prepare_frame(pd.read_parquet(args.input), pd.read_parquet(args.label_grid), grid_name=args.grid_name)
    frame, feature_columns = _load_features(frame, args.feature_manifest)
    required = {BASELINE_COLUMN, ARCHETYPE_COLUMN, "execution_cost_return"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("frozen exact-policy panel missing: " + ", ".join(missing))
    oof_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    head_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for index, window in enumerate(DEFAULT_WINDOWS):
        train, evaluation, split = _strict_forward_split(frame, window, purge_hours=args.purge_hours)
        oof_sides, predicted_sides, side_report = [], [], {}
        for side_index, side in enumerate(SIDES):
            fitted = train.loc[train[SIDE_COLUMN].eq(side)].reset_index(drop=True)
            scored = evaluation.loc[evaluation[SIDE_COLUMN].eq(side)].reset_index(drop=True)
            if fitted.empty or scored.empty:
                raise ValueError(f"{window.name}: missing {side} rows")
            oof, predicted, report = _fit_side(fitted, scored, feature_columns, iterations=args.n_estimators, seed=args.random_state + 100_000 * index + 10_000 * side_index, n_jobs=args.n_jobs)
            oof[SIDE_COLUMN] = side
            predicted[SIDE_COLUMN] = side
            oof_sides.append(oof)
            predicted_sides.append(predicted)
            side_report[side] = report
        oof = pd.concat(oof_sides, ignore_index=True)
        side_predictions = pd.concat(predicted_sides, ignore_index=True)
        prediction = evaluation.loc[:, list(IDENTITY_COLUMNS)].merge(side_predictions, on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")
        if len(prediction) != len(evaluation) or prediction["direct_net_residual"].isna().any():
            raise ValueError("side-local scores do not cover the immutable evaluation population")
        map_train, map_eval = train.copy(), evaluation.copy()
        map_train[RESOLUTION_COLUMN] = map_train["support_label_available_utc"]
        map_eval[RESOLUTION_COLUMN] = map_eval["support_label_available_utc"]
        mapped_scores: dict[str, np.ndarray] = {}
        raw_scores: dict[str, np.ndarray] = {}
        for arm in ARMS:
            raw = prediction[arm].to_numpy(float)
            aligned = map_train.loc[:, list(IDENTITY_COLUMNS)].merge(
                oof.loc[:, [*IDENTITY_COLUMNS, f"side_causal_oof_ev_{arm}"]],
                on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one",
            )[f"side_causal_oof_ev_{arm}"].to_numpy(float)
            mapped, mapping_report = apply_canonical_recent_mapping(map_train, map_eval, aligned, raw)
            raw_scores[arm], mapped_scores[arm] = raw, mapped
            prediction[f"canonical_recent_ev_score_{arm}"] = mapped
            side_report.setdefault("recent_mapping", {})[arm] = mapping_report
        head_rows.extend(_head_metrics(oof, window=window.name))
        arm_rows.extend(_arm_metrics(evaluation, raw_scores, window=window.name, stage="pre_recent_mapping"))
        arm_rows.extend(_arm_metrics(evaluation, mapped_scores, window=window.name, stage="canonical_recent_ev_mapping"))
        oof["window"] = window.name
        prediction["window"] = window.name
        oof_parts.append(oof)
        prediction_parts.append(prediction)
        reports[window.name] = {"split": split, "models": side_report}
    paths = {
        "support_head_oof_ledger": staging / "support_head_oof_ledger.parquet",
        "forward_predictions": staging / "forward_predictions.parquet",
        "support_head_metrics": staging / "support_head_metrics.csv",
        "arm_metrics": staging / "arm_metrics.csv",
        "frozen_control_comparison": staging / "frozen_control_comparison.csv",
    }
    pd.concat(oof_parts, ignore_index=True).to_parquet(paths["support_head_oof_ledger"], index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(paths["forward_predictions"], index=False)
    pd.DataFrame(head_rows).to_csv(paths["support_head_metrics"], index=False)
    pd.DataFrame(arm_rows).to_csv(paths["arm_metrics"], index=False)
    control_summary, control_bindings = _frozen_control_summary()
    control_summary.to_csv(paths["frozen_control_comparison"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_nonpromotion_evidence",
        "promotion_eligible": False,
        "contract": {
            "population": "identical frozen exact-policy candidate rows and feature manifest used by strict OOF conversion control",
            "side_training": "all direct, support, joint and stopped-gradient heads fit separately for long and short",
            "availability": "every fit/calibration row requires max(execution label end, meaningful-MFE label resolution) before its fold cutoff",
            "targets": "exact net target; explicit gross>cost hurdle; net-positive and net-loss conditional magnitudes; actual full-stop/timeout/other exit mixture and conditional exact-net payoffs",
            "costs": "gross - cost exactly equals net; composed payoffs are already net and never pay costs twice",
            "joint": "CatBoost MultiRMSE with direct residual repeated three times plus five support outputs; only decoded direct output ranks",
            "two_stage": "Ridge residual on only earlier resolved outer-OOF frozen head outputs; direct raw score fallback before minimum support",
            "mapping": "common-unit canonical causal recent EV mapping after side-local causal OOF calibration",
            "ranking": "one pooled-global top 1/5/10/20% after mapping; no timestamp, side, asset or quota selection",
            "actions": "timing, wait, target-price and MAE actions excluded from ranker",
            "portfolio": "not replayed: research-only head evidence is not a promoted policy",
            "frozen_controls": "previous decomposed hurdle and historical gross-hurdle controls are bound as non-identical-row negative-control context; this experiment tests the missing direct-primary joint and stopped-gradient architecture on one identical strict-OOF population",
        },
        "inputs": {
            "data": {"path": str(args.input), "sha256": _sha256(args.input)},
            "label_grid": {"path": str(args.label_grid), "sha256": _sha256(args.label_grid), "grid": args.grid_name},
            "feature_manifest": {"path": str(args.feature_manifest), "sha256": _sha256(args.feature_manifest)},
        },
        "frozen_control_bindings": control_bindings,
        "feature_columns": feature_columns,
        "arms": list(ARMS),
        "windows": [window.__dict__ for window in DEFAULT_WINDOWS],
        "model": {"n_estimators": args.n_estimators, "n_jobs": args.n_jobs, "random_state": args.random_state},
        "folds": reports,
        "outputs": {
            name: {"path": str(args.output_dir / path.name), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = staging / "manifest.json"
    _write_json(manifest_path, manifest)
    seal = _sha256(manifest_path)
    (staging / "manifest.sha256").write_text(f"{seal}  manifest.json\n", encoding="utf-8")
    # A same-filesystem directory rename publishes either the complete sealed
    # artifact or nothing at the immutable destination.
    os.replace(staging, args.output_dir)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"))
    parser.add_argument("--feature-manifest", type=Path, default=Path("data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/regime_diagnosis_manifest.json"))
    parser.add_argument("--label-grid", type=Path, default=Path("data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet"))
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260730)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
