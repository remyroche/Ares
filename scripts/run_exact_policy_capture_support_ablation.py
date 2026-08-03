#!/usr/bin/env python3
"""Test executable capture support around the direct exact-net model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ExecutionEVModelAblationConfig,
    apply_execution_ev_causal_recent_ev_correction,
    fit_train_only_isotonic_ev_mapping,
)
from scripts.diagnose_within_july_opportunity_capture import (  # noqa: E402
    economic_components,
)
from scripts.run_execution_ev_mixed_period_remedies import (  # noqa: E402
    ARCHETYPE_COLUMN,
    BASELINE_COLUMN,
    DECISION_COLUMN,
    DEFAULT_WINDOWS,
    IDENTITY_COLUMNS,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    _model_features,
    _temporal_oof_blocks,
    apply_canonical_recent_mapping,
    build_forward_split,
)
from scripts.run_exact_policy_capture_hurdle_ablation import (  # noqa: E402
    _classifier,
    _fit_or_constant_classifier,
    _fit_or_constant_regressor,
    _metric_rows,
    _predict_classifier,
    _predict_regressor,
    _regressor,
)

SCHEMA = "exact_policy_capture_support_ablation_v2"
SIDES = ("long", "short")
FROZEN_BASE_MARGIN_SCREEN = ROOT / (
    "data_perp/artifacts/execution_ev_false_positive_feature_diagnosis_20260727_v2/"
    "frozen_screens.csv"
)
# This is deliberately a narrow, source-locked challenger.  The values below
# were selected before this interaction existed; they are *not* HPO knobs.
FROZEN_BASE_MARGIN_INTERACTION = {
    "name": "direct_capture_margin_soft_interaction",
    "feature": "base_margin_to_cutoff_z",
    "screen_sha256": "2d5e4b7c5e2e0f72ec0e638f3170b502c77eb33ac003026d47bb1384bb143eaa",
    "direction": 1.0,
    "threshold": 0.5934305191040039,
    "robust_scale": 0.7888193689286709,
    # A small, fixed rank-space perturbation.  It is intentionally not tuned
    # against any evaluation window.
    "interaction_weight": 0.25,
}
FROZEN_INTERACTION_INPUT_HASHES = {
    "data": "b736827fc941badc709cd4f3795033176959a1c275b1636b916a62d42c673799",
    "capture_labels": "3d98e87f61f8e149b2f2d18e45232b8e16e313d47a5602a5b823fbb07b261774",
    "comparison_label_grid": "c58115bcf27d8c476b5622f9e59afa626b0f5f0a403d86deba9ae7b075751e48",
    "feature_manifest": "d2d0a2e403c4c6152a9789c73edc54a483c8baa088f64a46fce16c9e2f7e6ef8",
}
STATIC_ARMS = (
    "direct_net",
    "capture_only",
    "direct_capture_blend25",
    "direct_capture_blend50",
    "direct_capture_blend75",
    "soft_net_25bps",
    "soft_net_50bps",
    "soft_net_100bps",
    "positive_net_50bps",
    "positive_net_100bps",
    "distributional_net",
    "multitask_distributional_net",
    "direct_plus_capture_residual",
    "direct_plus_full_capture_residual",
    "capture_low20_abstain",
    "severe_high20_veto",
    "bounded_capture_support",
    "direct_capture_margin_soft_interaction",
)
OOF_GATE_BASES = {
    "oof_gate_direct_net": "direct_net",
    "oof_gate_capture_only": "capture_only",
    "oof_gate_distributional_net": "distributional_net",
    "oof_gate_multitask_distributional_net": "multitask_distributional_net",
}
ARMS = (*STATIC_ARMS, *OOF_GATE_BASES)
RAW_NAMES = (
    "direct_net",
    "capture_probability",
    "severe_loss_probability",
    "favorable_order_probability",
    "capture_ratio",
    "giveback_log_bps",
    "soft_net_25bps",
    "soft_net_50bps",
    "soft_net_100bps",
    "positive_net_50bps",
    "positive_net_100bps",
    "positive_net_log_bps",
    "negative_net_log_bps",
    "multitask_distributional_net",
)
CORE_SUPPORT_NAMES = (
    "direct_net",
    "capture_probability",
    "severe_loss_probability",
    "favorable_order_probability",
    "capture_ratio",
    "giveback_log_bps",
)
MAPPING_DIAGNOSTIC_ARMS = (
    "direct_net",
    "capture_only",
    "soft_net_25bps",
    "soft_net_50bps",
    "soft_net_100bps",
    "distributional_net",
    "multitask_distributional_net",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_frozen_base_margin_interaction(
    screen: Path,
    *,
    expected_sha256: str = str(FROZEN_BASE_MARGIN_INTERACTION["screen_sha256"]),
) -> dict[str, Any]:
    """Load the predeclared margin contract and reject any changed screen.

    The screen is an input to the experiment, not an input to fitting.  Reading
    the values from the locked CSV as well as checking its digest makes an
    accidental replacement or a silently changed row fail before scoring.
    """
    observed_hash = _sha256(screen)
    if observed_hash != expected_sha256:
        raise ValueError(
            "frozen base-margin screen hash mismatch; refusing to score an "
            "interaction against an unpinned screen"
        )
    screen_frame = pd.read_csv(screen)
    feature = str(FROZEN_BASE_MARGIN_INTERACTION["feature"])
    selected = screen_frame.loc[screen_frame["feature"].eq(feature)]
    if len(selected) != 1:
        raise ValueError("expected exactly one frozen base-margin screen row")
    row = selected.iloc[0]
    contract = dict(FROZEN_BASE_MARGIN_INTERACTION)
    observed = {
        "direction": float(row["direction_tp_over_fp"]),
        "threshold": float(row["frozen_selected_book_median"]),
        "robust_scale": float(row["frozen_control_scale"]),
    }
    for name, value in observed.items():
        expected = float(contract[name])
        if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"frozen base-margin {name} mismatch: {value!r} != {expected!r}"
            )
    if contract["robust_scale"] <= 0.0 or contract["interaction_weight"] <= 0.0:
        raise ValueError("invalid frozen base-margin interaction contract")
    contract["screen_path"] = str(screen)
    contract["screen_sha256"] = observed_hash
    return contract


def assert_frozen_interaction_sources(args: argparse.Namespace) -> dict[str, str]:
    """Fail closed if this fixed challenger is run on a different source set.

    A genuinely new forward block requires an explicitly versioned successor
    contract before it can be used as decision evidence.  It must not inherit
    this diagnostic's June/July source lock by accident.
    """
    paths = {
        "data": args.input,
        "capture_labels": args.capture_labels,
        "comparison_label_grid": args.label_grid,
        "feature_manifest": args.feature_manifest,
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    mismatched = {
        name: digest
        for name, digest in observed.items()
        if digest != FROZEN_INTERACTION_INPUT_HASHES[name]
    }
    if mismatched:
        raise ValueError(
            "frozen base-margin interaction source hash mismatch for "
            + ", ".join(sorted(mismatched))
            + "; this contract may score only its pinned reused-OOS blocks"
        )
    return observed


def _standardize_from_oof(
    oof: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train = np.asarray(oof, dtype=float)
    score = np.asarray(evaluation, dtype=float)
    finite = np.isfinite(train)
    if finite.sum() < 24 or not np.isfinite(score).all():
        raise ValueError("insufficient finite OOF confidence scores")
    center = float(np.mean(train[finite]))
    scale = max(float(np.std(train[finite])), 1e-8)
    return (
        (train - center) / scale,
        (score - center) / scale,
        {"center": center, "scale": scale, "oof_rows": int(finite.sum())},
    )


def margin_capture_soft_interaction(
    direct_oof: np.ndarray,
    capture_oof: np.ndarray,
    margin_oof: np.ndarray,
    direct_evaluation: np.ndarray,
    capture_evaluation: np.ndarray,
    margin_evaluation: np.ndarray,
    *,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the locked smooth margin-by-confidence interaction.

    ``confidence`` is the positive part of the equal direct/capture OOF-z
    average.  Thus a high margin only amplifies rows on which both heads are
    relatively confident; a low margin softly discounts those same rows.  It
    never turns two low-confidence scores into a reward.  All centres/scales
    for the head scores come from the side-local temporal OOF segment.
    """
    direct_oof_z, direct_eval_z, direct_report = _standardize_from_oof(
        direct_oof, direct_evaluation
    )
    capture_oof_z, capture_eval_z, capture_report = _standardize_from_oof(
        capture_oof, capture_evaluation
    )
    margin_train = np.asarray(margin_oof, dtype=float)
    margin_score = np.asarray(margin_evaluation, dtype=float)
    if not np.isfinite(margin_train).all() or not np.isfinite(margin_score).all():
        raise ValueError("base-margin feature contains non-finite values")
    direction = float(contract["direction"])
    threshold = float(contract["threshold"])
    scale = float(contract["robust_scale"])
    weight = float(contract["interaction_weight"])
    if direction not in (-1.0, 1.0) or scale <= 0.0 or weight <= 0.0:
        raise ValueError("invalid frozen base-margin interaction parameters")

    def gate(values: np.ndarray) -> np.ndarray:
        directional_distance = (
            direction * np.asarray(values, dtype=float) - direction * threshold
        ) / scale
        return 1.0 / (1.0 + np.exp(-np.clip(directional_distance, -40.0, 40.0)))

    oof_confidence = np.maximum(0.0, 0.5 * (direct_oof_z + capture_oof_z))
    eval_confidence = np.maximum(0.0, 0.5 * (direct_eval_z + capture_eval_z))
    oof_gate = gate(margin_train)
    eval_gate = gate(margin_score)
    oof_score = direct_oof_z + weight * (2.0 * oof_gate - 1.0) * oof_confidence
    eval_score = direct_eval_z + weight * (2.0 * eval_gate - 1.0) * eval_confidence
    return oof_score, eval_score, {
        "formula": (
            "z_direct + 0.25 * (2 * sigmoid((direction * margin - "
            "direction * threshold) / robust_scale) - 1) * "
            "max(0, 0.5 * (z_direct + z_capture))"
        ),
        "direct_oof_standardization": direct_report,
        "capture_oof_standardization": capture_report,
        "margin_gate_oof_mean": float(oof_gate.mean()),
        "margin_gate_evaluation_mean": float(eval_gate.mean()),
        "oof_positive_confidence_fraction": float((oof_confidence > 0.0).mean()),
        "evaluation_positive_confidence_fraction": float(
            (eval_confidence > 0.0).mean()
        ),
    }


def add_support_targets(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    gross = work["execution_gross_ev_12h"].to_numpy(dtype=float)
    cost = work["execution_cost_return"].to_numpy(dtype=float)
    net = work[TARGET_COLUMN].to_numpy(dtype=float)
    if np.max(np.abs(gross - cost - net)) > 1e-7:
        raise ValueError("exact gross-cost-net accounting does not reconcile")
    copied_gross = work["capture_label_exact_gross"].to_numpy(dtype=float)
    copied_cost = work["capture_label_exact_cost"].to_numpy(dtype=float)
    if np.max(np.abs(gross - copied_gross)) > 1e-7:
        raise ValueError("capture-label gross does not match canonical exact gross")
    if np.max(np.abs(cost - copied_cost)) > 1e-7:
        raise ValueError("capture-label cost does not match canonical exact cost")
    favorable = work["favorable_before_adverse_at_cost"].to_numpy(dtype=bool)
    adverse = work["adverse_before_favorable_at_cost"].to_numpy(dtype=bool)
    work["target_favorable_order_soft"] = np.where(
        favorable, 1.0, np.where(adverse, 0.0, 0.5)
    )
    work["target_capture_ratio"] = np.clip(
        work["pre_exit_gross_capture_ratio"].to_numpy(dtype=float), 0.0, 1.0
    )
    gap = np.clip(
        work["pre_exit_mfe_to_gross_gap"].to_numpy(dtype=float), 0.0, 0.10
    )
    work["target_giveback_log_bps"] = np.log1p(gap * 10_000.0)
    for temperature_bps in (25, 50, 100):
        work[f"target_soft_net_{temperature_bps}bps"] = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    net / (temperature_bps / 10_000.0),
                    -40.0,
                    40.0,
                )
            )
        )
    work["target_positive_net_50bps"] = (net > 0.005).astype(np.int8)
    work["target_positive_net_100bps"] = (net > 0.010).astype(np.int8)
    work["target_positive_net_log_bps"] = np.log1p(
        np.clip(net, 0.0, 0.10) * 10_000.0
    )
    work["target_negative_net_log_bps"] = np.log1p(
        np.clip(-net, 0.0, 0.10) * 10_000.0
    )
    work["target_positive_net_contribution"] = np.clip(net, 0.0, 0.10)
    work["target_negative_net_contribution"] = np.clip(-net, 0.0, 0.10)
    return work


def support_feature_matrix(raw: Mapping[str, np.ndarray], *, full: bool) -> pd.DataFrame:
    names = (
        CORE_SUPPORT_NAMES
        if full
        else ("direct_net", "capture_probability")
    )
    return pd.DataFrame(
        {
            name: np.asarray(raw[name], dtype=float)
            for name in names
        }
    )


def apply_quantile_veto(
    score: np.ndarray,
    risk: np.ndarray,
    *,
    threshold: float,
    high_is_bad: bool,
) -> np.ndarray:
    result = np.asarray(score, dtype=float).copy()
    risk_values = np.asarray(risk, dtype=float)
    rejected = risk_values > threshold if high_is_bad else risk_values < threshold
    finite = np.isfinite(result)
    floor = float(np.nanmin(result[finite]) - max(np.nanstd(result[finite]), 0.01))
    result[rejected & finite] = floor
    return result


def standardized_capture_blends(
    direct_oof: np.ndarray,
    capture_oof: np.ndarray,
    direct_eval: np.ndarray,
    capture_eval: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    finite = np.isfinite(direct_oof) & np.isfinite(capture_oof)
    direct_mean = float(np.mean(direct_oof[finite]))
    capture_mean = float(np.mean(capture_oof[finite]))
    direct_scale = max(float(np.std(direct_oof[finite])), 1e-8)
    capture_scale = max(float(np.std(capture_oof[finite])), 1e-8)

    def z(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
        return (np.asarray(values, dtype=float) - mean) / scale

    direct_oof_z = z(direct_oof, direct_mean, direct_scale)
    capture_oof_z = z(capture_oof, capture_mean, capture_scale)
    direct_eval_z = z(direct_eval, direct_mean, direct_scale)
    capture_eval_z = z(capture_eval, capture_mean, capture_scale)
    oof = {"capture_only": np.asarray(capture_oof, dtype=float)}
    evaluation = {"capture_only": np.asarray(capture_eval, dtype=float)}
    for capture_weight in (0.25, 0.50, 0.75):
        name = f"direct_capture_blend{int(capture_weight * 100)}"
        oof[name] = (
            (1.0 - capture_weight) * direct_oof_z
            + capture_weight * capture_oof_z
        )
        evaluation[name] = (
            (1.0 - capture_weight) * direct_eval_z
            + capture_weight * capture_eval_z
        )
    return oof, evaluation


def compose_distributional_net(
    capture_probability: np.ndarray,
    positive_net_log_bps: np.ndarray,
    negative_net_log_bps: np.ndarray,
) -> np.ndarray:
    probability = np.clip(
        np.asarray(capture_probability, dtype=float), 0.0, 1.0
    )
    positive = np.expm1(np.asarray(positive_net_log_bps, dtype=float)) / 10_000.0
    negative = np.expm1(np.asarray(negative_net_log_bps, dtype=float)) / 10_000.0
    return probability * positive - (1.0 - probability) * negative


def compose_multitask_distributional_net(prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("multi-task prediction must have positive/loss columns")
    return np.clip(values[:, 0], 0.0, 0.10) - np.clip(
        values[:, 1], 0.0, 0.10
    )


def _multitask_regressor(*, iterations: int, seed: int, n_jobs: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=int(iterations),
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        random_seed=int(seed),
        thread_count=int(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )


def _fit_raw_heads(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    fit_x = _model_features(fit, fit, feature_columns, trust_composites=False)
    score_x = _model_features(fit, score, feature_columns, trust_composites=False)
    direct_model = _regressor(iterations=iterations, seed=seed, n_jobs=n_jobs)
    direct_model.fit(
        fit_x,
        fit[TARGET_COLUMN].to_numpy(dtype=float)
        - fit[BASELINE_COLUMN].to_numpy(dtype=float),
    )
    direct = score[BASELINE_COLUMN].to_numpy(dtype=float) + np.asarray(
        direct_model.predict(score_x), dtype=float
    )
    capture_model, capture_constant = _fit_or_constant_classifier(
        fit_x,
        fit["exact_net_positive"].to_numpy(dtype=np.int8),
        iterations=iterations,
        seed=seed + 11,
        n_jobs=n_jobs,
    )
    severe_model, severe_constant = _fit_or_constant_classifier(
        fit_x,
        fit["exact_net_loss_worse_two_costs"].to_numpy(dtype=np.int8),
        iterations=iterations,
        seed=seed + 22,
        n_jobs=n_jobs,
    )
    order_model, order_constant = _fit_or_constant_regressor(
        fit_x,
        fit["target_favorable_order_soft"].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed + 33,
        n_jobs=n_jobs,
    )
    ratio_model, ratio_constant = _fit_or_constant_regressor(
        fit_x,
        fit["target_capture_ratio"].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed + 44,
        n_jobs=n_jobs,
    )
    giveback_model, giveback_constant = _fit_or_constant_regressor(
        fit_x,
        fit["target_giveback_log_bps"].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed + 55,
        n_jobs=n_jobs,
    )
    output = {
        "direct_net": direct,
        "capture_probability": _predict_classifier(
            capture_model, capture_constant, score_x
        ),
        "severe_loss_probability": _predict_classifier(
            severe_model, severe_constant, score_x
        ),
        "favorable_order_probability": np.clip(
            _predict_regressor(order_model, order_constant, score_x), 0.0, 1.0
        ),
        "capture_ratio": np.clip(
            _predict_regressor(ratio_model, ratio_constant, score_x), 0.0, 1.0
        ),
        "giveback_log_bps": _predict_regressor(
            giveback_model, giveback_constant, score_x
        ),
    }
    for offset, temperature_bps in enumerate((25, 50, 100), start=1):
        model, constant = _fit_or_constant_regressor(
            fit_x,
            fit[f"target_soft_net_{temperature_bps}bps"].to_numpy(dtype=float),
            iterations=iterations,
            seed=seed + 100 + offset,
            n_jobs=n_jobs,
        )
        output[f"soft_net_{temperature_bps}bps"] = np.clip(
            _predict_regressor(model, constant, score_x), 0.0, 1.0
        )
    for offset, threshold_bps in enumerate((50, 100), start=1):
        model, constant = _fit_or_constant_classifier(
            fit_x,
            fit[f"target_positive_net_{threshold_bps}bps"].to_numpy(dtype=np.int8),
            iterations=iterations,
            seed=seed + 200 + offset,
            n_jobs=n_jobs,
        )
        output[f"positive_net_{threshold_bps}bps"] = _predict_classifier(
            model, constant, score_x
        )
    net = fit[TARGET_COLUMN].to_numpy(dtype=float)
    for offset, (name, target_column, condition) in enumerate(
        (
            (
                "positive_net_log_bps",
                "target_positive_net_log_bps",
                net > 0.0,
            ),
            (
                "negative_net_log_bps",
                "target_negative_net_log_bps",
                net <= 0.0,
            ),
        ),
        start=1,
    ):
        model, constant = _fit_or_constant_regressor(
            fit_x.loc[condition],
            fit.loc[condition, target_column].to_numpy(dtype=float),
            iterations=iterations,
            seed=seed + 300 + offset,
            n_jobs=n_jobs,
        )
        output[name] = _predict_regressor(model, constant, score_x)
    multitask = _multitask_regressor(
        iterations=iterations, seed=seed + 400, n_jobs=n_jobs
    )
    multitask.fit(
        fit_x,
        fit[
            [
                "target_positive_net_contribution",
                "target_negative_net_contribution",
            ]
        ].to_numpy(dtype=float),
    )
    output["multitask_distributional_net"] = compose_multitask_distributional_net(
        np.asarray(multitask.predict(score_x), dtype=float)
    )
    return output


def _fit_meta_residual(
    x: pd.DataFrame,
    target: np.ndarray,
    score_x: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    model = _regressor(iterations=iterations, seed=seed, n_jobs=n_jobs)
    model.set_params(depth=4, learning_rate=0.02, l2_leaf_reg=10.0)
    model.fit(x, np.asarray(target, dtype=float))
    return np.asarray(model.predict(score_x), dtype=float)


def _nested_meta_oof(
    frame: pd.DataFrame,
    raw_oof: Mapping[str, np.ndarray],
    raw_eval: Mapping[str, np.ndarray],
    *,
    full: bool,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_x = support_feature_matrix(raw_oof, full=full)
    eval_x = support_feature_matrix(raw_eval, full=full)
    available = np.isfinite(train_x.to_numpy()).all(axis=1)
    local = frame.loc[available].copy().reset_index()
    local_x = train_x.loc[available].reset_index(drop=True)
    local_target = (
        local[TARGET_COLUMN].to_numpy(dtype=float)
        - local_x["direct_net"].to_numpy(dtype=float)
    )
    meta_oof = np.full(len(frame), np.nan)
    reports = []
    for fold, (fit_pos, valid_pos) in enumerate(
        _temporal_oof_blocks(local, min_train_rows=2_000), start=1
    ):
        prediction = _fit_meta_residual(
            local_x.iloc[fit_pos],
            local_target[fit_pos],
            local_x.iloc[valid_pos],
            iterations=iterations,
            seed=seed + fold,
            n_jobs=n_jobs,
        )
        meta_oof[local.loc[valid_pos, "index"].to_numpy(dtype=int)] = prediction
        reports.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_pos)),
                "validation_rows": int(len(valid_pos)),
                "max_fit_label_resolution_utc": pd.to_datetime(
                    local.iloc[fit_pos][RESOLUTION_COLUMN], utc=True
                ).max(),
                "validation_start_utc": pd.to_datetime(
                    local.iloc[valid_pos][DECISION_COLUMN], utc=True
                ).min(),
            }
        )
    eval_residual = _fit_meta_residual(
        local_x,
        local_target,
        eval_x,
        iterations=iterations,
        seed=seed + 99,
        n_jobs=n_jobs,
    )
    return meta_oof, eval_residual, reports


def fit_capture_support_scores(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns_by_side: Mapping[str, Sequence[str]],
    margin_interaction_contract: Mapping[str, Any],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    train_scores = {arm: np.full(len(train), np.nan) for arm in ARMS}
    eval_scores = {arm: np.full(len(evaluation), np.nan) for arm in ARMS}
    eval_veto_masks = {
        arm: np.zeros(len(evaluation), dtype=bool) for arm in ARMS
    }
    eval_head_scores = {
        name: np.full(len(evaluation), np.nan) for name in RAW_NAMES
    }
    train_head_scores = {
        name: np.full(len(train), np.nan) for name in RAW_NAMES
    }
    reports: dict[str, Any] = {}
    for side_index, side in enumerate(SIDES):
        side_feature_columns = list(feature_columns_by_side[side])
        train_side = train.loc[train[SIDE_COLUMN].astype(str).eq(side)].copy()
        train_side["__global_position__"] = train_side.index
        train_side = train_side.reset_index(drop=True)
        eval_side = evaluation.loc[
            evaluation[SIDE_COLUMN].astype(str).eq(side)
        ].copy()
        eval_side["__global_position__"] = eval_side.index
        eval_side = eval_side.reset_index(drop=True)
        raw_oof = {name: np.full(len(train_side), np.nan) for name in RAW_NAMES}
        head_folds = []
        for fold, (fit_pos, valid_pos) in enumerate(
            _temporal_oof_blocks(train_side, min_train_rows=2_000), start=1
        ):
            fold_scores = _fit_raw_heads(
                train_side.iloc[fit_pos],
                train_side.iloc[valid_pos],
                side_feature_columns,
                iterations=iterations,
                seed=seed + 10_000 * side_index + 100 * fold,
                n_jobs=n_jobs,
            )
            for name, values in fold_scores.items():
                raw_oof[name][valid_pos] = values
            head_folds.append(
                {
                    "fold": fold,
                    "fit_rows": int(len(fit_pos)),
                    "validation_rows": int(len(valid_pos)),
                }
            )
        raw_eval = _fit_raw_heads(
            train_side,
            eval_side,
            side_feature_columns,
            iterations=iterations,
            seed=seed + 10_000 * side_index + 9_000,
            n_jobs=n_jobs,
        )
        capture_meta_oof, capture_meta_eval, capture_meta_folds = _nested_meta_oof(
            train_side,
            raw_oof,
            raw_eval,
            full=False,
            iterations=iterations,
            seed=seed + 20_000 * side_index + 20_000,
            n_jobs=n_jobs,
        )
        full_meta_oof, full_meta_eval, full_meta_folds = _nested_meta_oof(
            train_side,
            raw_oof,
            raw_eval,
            full=True,
            iterations=iterations,
            seed=seed + 20_000 * side_index + 30_000,
            n_jobs=n_jobs,
        )
        finite_support = np.isfinite(raw_oof["capture_probability"])
        capture_q20 = float(
            np.quantile(raw_oof["capture_probability"][finite_support], 0.20)
        )
        severe_q80 = float(
            np.quantile(raw_oof["severe_loss_probability"][finite_support], 0.80)
        )
        blend_oof, blend_eval = standardized_capture_blends(
            raw_oof["direct_net"],
            raw_oof["capture_probability"],
            raw_eval["direct_net"],
            raw_eval["capture_probability"],
        )
        margin_feature = str(margin_interaction_contract["feature"])
        interaction_oof, interaction_eval, interaction_report = (
            margin_capture_soft_interaction(
                raw_oof["direct_net"],
                raw_oof["capture_probability"],
                train_side[margin_feature].to_numpy(dtype=float),
                raw_eval["direct_net"],
                raw_eval["capture_probability"],
                eval_side[margin_feature].to_numpy(dtype=float),
                contract=margin_interaction_contract,
            )
        )
        raw_arm_oof = {
            "direct_net": raw_oof["direct_net"],
            **blend_oof,
            **{
                name: raw_oof[name]
                for name in (
                    "soft_net_25bps",
                    "soft_net_50bps",
                    "soft_net_100bps",
                    "positive_net_50bps",
                    "positive_net_100bps",
                )
            },
            "distributional_net": compose_distributional_net(
                raw_oof["capture_probability"],
                raw_oof["positive_net_log_bps"],
                raw_oof["negative_net_log_bps"],
            ),
            "multitask_distributional_net": raw_oof[
                "multitask_distributional_net"
            ],
            "direct_plus_capture_residual": raw_oof["direct_net"] + capture_meta_oof,
            "direct_plus_full_capture_residual": raw_oof["direct_net"] + full_meta_oof,
            "capture_low20_abstain": apply_quantile_veto(
                raw_oof["direct_net"],
                raw_oof["capture_probability"],
                threshold=capture_q20,
                high_is_bad=False,
            ),
            "severe_high20_veto": apply_quantile_veto(
                raw_oof["direct_net"],
                raw_oof["severe_loss_probability"],
                threshold=severe_q80,
                high_is_bad=True,
            ),
            "bounded_capture_support": apply_quantile_veto(
                apply_quantile_veto(
                    raw_oof["direct_net"] + full_meta_oof,
                    raw_oof["capture_probability"],
                    threshold=capture_q20,
                    high_is_bad=False,
                ),
                raw_oof["severe_loss_probability"],
                threshold=severe_q80,
                high_is_bad=True,
            ),
            "direct_capture_margin_soft_interaction": interaction_oof,
        }
        raw_arm_eval = {
            "direct_net": raw_eval["direct_net"],
            **blend_eval,
            **{
                name: raw_eval[name]
                for name in (
                    "soft_net_25bps",
                    "soft_net_50bps",
                    "soft_net_100bps",
                    "positive_net_50bps",
                    "positive_net_100bps",
                )
            },
            "distributional_net": compose_distributional_net(
                raw_eval["capture_probability"],
                raw_eval["positive_net_log_bps"],
                raw_eval["negative_net_log_bps"],
            ),
            "multitask_distributional_net": raw_eval[
                "multitask_distributional_net"
            ],
            "direct_plus_capture_residual": raw_eval["direct_net"] + capture_meta_eval,
            "direct_plus_full_capture_residual": raw_eval["direct_net"] + full_meta_eval,
            "capture_low20_abstain": apply_quantile_veto(
                raw_eval["direct_net"],
                raw_eval["capture_probability"],
                threshold=capture_q20,
                high_is_bad=False,
            ),
            "severe_high20_veto": apply_quantile_veto(
                raw_eval["direct_net"],
                raw_eval["severe_loss_probability"],
                threshold=severe_q80,
                high_is_bad=True,
            ),
            "bounded_capture_support": apply_quantile_veto(
                apply_quantile_veto(
                    raw_eval["direct_net"] + full_meta_eval,
                    raw_eval["capture_probability"],
                    threshold=capture_q20,
                    high_is_bad=False,
                ),
                raw_eval["severe_loss_probability"],
                threshold=severe_q80,
                high_is_bad=True,
            ),
            "direct_capture_margin_soft_interaction": interaction_eval,
        }
        side_veto_masks = {
            "capture_low20_abstain": (
                raw_eval["capture_probability"] < capture_q20
            ),
            "severe_high20_veto": (
                raw_eval["severe_loss_probability"] > severe_q80
            ),
        }
        side_veto_masks["bounded_capture_support"] = (
            side_veto_masks["capture_low20_abstain"]
            | side_veto_masks["severe_high20_veto"]
        )
        train_position = train_side["__global_position__"].to_numpy(dtype=int)
        eval_position = eval_side["__global_position__"].to_numpy(dtype=int)
        for name in RAW_NAMES:
            train_head_scores[name][train_position] = raw_oof[name]
        for name in RAW_NAMES:
            eval_head_scores[name][eval_position] = raw_eval[name]
        for arm in STATIC_ARMS:
            mapper = fit_train_only_isotonic_ev_mapping(
                raw_arm_oof[arm],
                train_side[TARGET_COLUMN].to_numpy(dtype=float),
                min_rows=24,
            )
            finite = np.isfinite(raw_arm_oof[arm])
            mapped_oof = np.full(len(train_side), np.nan)
            mapped_oof[finite] = mapper.predict(raw_arm_oof[arm][finite])
            mapped_eval = mapper.predict(raw_arm_eval[arm])
            train_scores[arm][train_position] = mapped_oof
            eval_scores[arm][eval_position] = mapped_eval
            if arm in side_veto_masks:
                eval_veto_masks[arm][eval_position] = side_veto_masks[arm]
        reports[side] = {
            "train_rows": int(len(train_side)),
            "evaluation_rows": int(len(eval_side)),
            "head_oof_rows": int(np.isfinite(raw_oof["direct_net"]).sum()),
            "head_folds": head_folds,
            "capture_meta_folds": capture_meta_folds,
            "full_meta_folds": full_meta_folds,
            "capture_low20_threshold": capture_q20,
            "severe_high20_threshold": severe_q80,
            "frozen_margin_interaction": interaction_report,
        }
    reports["oof_capture_gate_optimisation"] = {}
    for gate_arm, base_arm in OOF_GATE_BASES.items():
        quantile, thresholds, candidates, gate_report = optimize_oof_capture_gate(
            train,
            train_scores[base_arm],
            train_head_scores["capture_probability"],
        )
        eval_veto = np.zeros(len(evaluation), dtype=bool)
        for side in SIDES:
            eval_side_mask = (
                evaluation[SIDE_COLUMN].astype(str).eq(side).to_numpy()
            )
            eval_veto |= eval_side_mask & (
                eval_head_scores["capture_probability"] < thresholds[side]
            )
        train_scores[gate_arm] = train_scores[base_arm].copy()
        eval_scores[gate_arm] = eval_scores[base_arm].copy()
        eval_veto_masks[gate_arm] = eval_veto
        reports["oof_capture_gate_optimisation"][gate_arm] = {
            **gate_report,
            "base_arm": base_arm,
            "thresholds_by_side": thresholds,
            "candidates": candidates,
            "evaluation_rejected_rows": int(eval_veto.sum()),
            "selected_quantile": quantile,
        }
    for arm in ARMS:
        if not np.isfinite(eval_scores[arm]).all():
            raise ValueError(f"{arm} failed to score all evaluation rows")
    return (
        train_scores,
        eval_scores,
        eval_veto_masks,
        eval_head_scores,
        reports,
    )


def apply_final_veto(score: np.ndarray, veto_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(score, dtype=float).copy()
    rejected = np.asarray(veto_mask, dtype=bool)
    finite = np.isfinite(result)
    accepted = finite & ~rejected
    if (rejected & finite).any() and accepted.any():
        result[rejected & finite] = float(
            np.min(result[accepted]) - max(np.std(result[accepted]), 0.01)
        )
    return result


def apply_recent_mapping_frame(
    frame: pd.DataFrame,
    mapped_score: np.ndarray,
    *,
    scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if scope not in {"side", "global"}:
        raise ValueError("recent mapping scope must be side or global")
    combined = frame.copy().reset_index(drop=True)
    combined["__recent_mapping_archetype__"] = "all"
    side_column = SIDE_COLUMN
    if scope == "global":
        combined["__recent_mapping_side__"] = "all"
        side_column = "__recent_mapping_side__"
    mapped = np.asarray(mapped_score, dtype=float)
    config = ExecutionEVModelAblationConfig(
        decision_time_col=DECISION_COLUMN,
        label_end_time_col=RESOLUTION_COLUMN,
        side_col=side_column,
        catboost_archetype_col="__recent_mapping_archetype__",
        n_estimators=200,
        recent_ev_window_days=21,
        recent_ev_correction_routes=("catboost_predicted_archetype",),
    )
    corrected, report = apply_execution_ev_causal_recent_ev_correction(
        combined,
        mapped,
        combined[TARGET_COLUMN].to_numpy(dtype=float),
        {},
        route="catboost_predicted_archetype",
        config=config,
    )
    report["effective_scope"] = scope
    return corrected, report


def apply_recent_mapping_scope(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    train_oof_score: np.ndarray,
    evaluation_score: np.ndarray,
    *,
    scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    combined = pd.concat([train, evaluation], ignore_index=True)
    mapped = np.concatenate([train_oof_score, evaluation_score])
    corrected, report = apply_recent_mapping_frame(
        combined, mapped, scope=scope
    )
    return corrected[len(train) :], report


def optimize_oof_capture_gate(
    frame: pd.DataFrame,
    mapped_base_oof: np.ndarray,
    capture_oof: np.ndarray,
    *,
    candidate_quantiles: Sequence[float] = (0.0, 0.10, 0.20, 0.30, 0.40),
    latest_days: int = 7,
    min_latest_selected_rows: int = 100,
) -> tuple[float, dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    causal_score, mapping_report = apply_recent_mapping_frame(
        frame, mapped_base_oof, scope="global"
    )
    capture = np.asarray(capture_oof, dtype=float)
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    decision = pd.to_datetime(frame[DECISION_COLUMN], utc=True, errors="raise")
    valid = np.isfinite(causal_score) & np.isfinite(capture) & np.isfinite(target)
    if valid.sum() < 1_000:
        raise ValueError("insufficient temporal OOF rows for capture-gate selection")
    latest_start = decision[valid].max() - pd.Timedelta(days=int(latest_days))
    candidates = []
    threshold_by_candidate: dict[float, dict[str, float]] = {}
    for quantile in candidate_quantiles:
        thresholds = {}
        veto = np.zeros(len(frame), dtype=bool)
        for side in SIDES:
            side_mask = (
                valid
                & frame[SIDE_COLUMN].astype(str).eq(side).to_numpy()
            )
            if not side_mask.any():
                raise ValueError(f"capture-gate OOF rows missing side {side}")
            threshold = (
                -np.inf
                if float(quantile) <= 0.0
                else float(np.quantile(capture[side_mask], float(quantile)))
            )
            thresholds[side] = threshold
            veto |= (
                frame[SIDE_COLUMN].astype(str).eq(side).to_numpy()
                & (capture < threshold)
            )
        threshold_by_candidate[float(quantile)] = thresholds
        gated = apply_final_veto(causal_score, veto)
        eligible_positions = np.flatnonzero(valid)
        count = max(1, int(np.ceil(0.10 * len(eligible_positions))))
        order = np.argsort(
            -gated[eligible_positions], kind="mergesort"
        )[:count]
        selected = eligible_positions[order]
        latest_selected = selected[decision.iloc[selected].ge(latest_start)]
        overall_net = float(target[selected].mean())
        latest_net = (
            float(target[latest_selected].mean())
            if len(latest_selected)
            else -np.inf
        )
        coverage_pass = len(latest_selected) >= int(min_latest_selected_rows)
        objective = min(overall_net, latest_net) if coverage_pass else -np.inf
        candidates.append(
            {
                "quantile": float(quantile),
                "overall_selected_rows": int(len(selected)),
                "overall_top10_net_bps": overall_net * 10_000.0,
                "latest_start_utc": latest_start,
                "latest_selected_rows": int(len(latest_selected)),
                "latest_top10_net_bps": latest_net * 10_000.0,
                "latest_coverage_pass": bool(coverage_pass),
                "robust_objective_bps": objective * 10_000.0,
            }
        )
    eligible = [
        row for row in candidates if np.isfinite(row["robust_objective_bps"])
    ]
    if eligible:
        winner = max(
            eligible,
            key=lambda row: (row["robust_objective_bps"], -row["quantile"]),
        )
        status = "selected_on_temporal_oof_robust_economics"
    else:
        winner = next(row for row in candidates if row["quantile"] == 0.0)
        status = "fallback_no_gate_latest_coverage_failed"
    quantile = float(winner["quantile"])
    report = {
        "status": status,
        "selected_quantile": quantile,
        "latest_days": int(latest_days),
        "min_latest_selected_rows": int(min_latest_selected_rows),
        "mapping": mapping_report,
    }
    return quantile, threshold_by_candidate[quantile], candidates, report


def _capture_metric_rows(
    evaluation: pd.DataFrame,
    score: np.ndarray,
    *,
    window: str,
    arm: str,
    stage: str,
) -> list[dict[str, Any]]:
    rows = _metric_rows(
        evaluation, score, window=window, arm=arm, stage=stage
    )
    for row in rows:
        scope = str(row["scope"])
        mask = (
            np.ones(len(evaluation), dtype=bool)
            if scope == "pooled_global"
            else evaluation[SIDE_COLUMN]
            .astype(str)
            .eq(scope.removeprefix("side_"))
            .to_numpy()
        )
        sample = evaluation.loc[mask].reset_index(drop=True)
        prediction = np.asarray(score, dtype=float)[mask]
        count = max(1, int(np.ceil(0.10 * len(sample))))
        positions = np.argsort(-prediction, kind="mergesort")[:count]
        selected_net = sample.iloc[positions][TARGET_COLUMN].to_numpy(dtype=float)
        selected_prediction = prediction[positions]
        row.update(
            {
                "top_k_rows": count,
                "top_k_mean_net_ev": float(selected_net.mean()),
                "top_k_sum_net_ev": float(selected_net.sum()),
                "top_k_predicted_net_ev": float(selected_prediction.mean()),
                "top_k_positive_ev_rate": float((selected_net > 0.0).mean()),
            }
        )
    return rows


def _selected_capture_components(
    evaluation: pd.DataFrame, score: np.ndarray
) -> dict[str, float]:
    count = max(1, int(np.ceil(0.10 * len(evaluation))))
    selected = evaluation.iloc[
        np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:count]
    ]
    return {
        "pre_exit_mfe_bps": 10_000.0
        * float(selected["pre_exit_mfe_return"].mean()),
        "pre_exit_mfe_to_gross_gap_bps": 10_000.0
        * float(selected["pre_exit_mfe_to_gross_gap"].mean()),
        "gross_capture_ratio": float(
            selected["pre_exit_gross_capture_ratio"].mean()
        ),
        "favorable_before_adverse_at_cost_rate": float(
            selected["favorable_before_adverse_at_cost"].mean()
        ),
        "adverse_before_favorable_at_cost_rate": float(
            selected["adverse_before_favorable_at_cost"].mean()
        ),
        "exact_net_positive_rate": float(selected["exact_net_positive"].mean()),
        "severe_loss_rate": float(
            selected["exact_net_loss_worse_two_costs"].mean()
        ),
    }


def _head_metric_rows(
    evaluation: pd.DataFrame,
    heads: Mapping[str, np.ndarray],
    *,
    window: str,
) -> list[dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    specifications = {
        "capture_probability": ("exact_net_positive", 1.0, True),
        "severe_loss_probability": (
            "exact_net_loss_worse_two_costs",
            -1.0,
            True,
        ),
        "favorable_order_probability": (
            "target_favorable_order_soft",
            1.0,
            False,
        ),
        "capture_ratio": ("target_capture_ratio", 1.0, False),
        "giveback_log_bps": ("target_giveback_log_bps", -1.0, False),
        "soft_net_25bps": ("target_soft_net_25bps", 1.0, False),
        "soft_net_50bps": ("target_soft_net_50bps", 1.0, False),
        "soft_net_100bps": ("target_soft_net_100bps", 1.0, False),
        "positive_net_50bps": ("target_positive_net_50bps", 1.0, True),
        "positive_net_100bps": ("target_positive_net_100bps", 1.0, True),
        "multitask_distributional_net": (TARGET_COLUMN, 1.0, False),
    }
    rows = []
    for scope in ("pooled_global", "side_long", "side_short"):
        mask = (
            np.ones(len(evaluation), dtype=bool)
            if scope == "pooled_global"
            else evaluation[SIDE_COLUMN]
            .astype(str)
            .eq(scope.removeprefix("side_"))
            .to_numpy()
        )
        sample = evaluation.loc[mask].reset_index(drop=True)
        for head, (target_column, direction, binary) in specifications.items():
            prediction = np.asarray(heads[head], dtype=float)[mask]
            target = sample[target_column].to_numpy(dtype=float)
            rank_score = direction * prediction
            count = max(1, int(np.ceil(0.10 * len(sample))))
            selected = np.argsort(-rank_score, kind="mergesort")[:count]
            auc = np.nan
            if binary and np.unique(target).size == 2:
                auc = float(roc_auc_score(target, prediction))
            if head == "favorable_order_probability":
                resolved = (
                    sample["favorable_before_adverse_at_cost"].to_numpy(dtype=bool)
                    | sample["adverse_before_favorable_at_cost"].to_numpy(dtype=bool)
                )
                if (
                    resolved.sum() > 0
                    and np.unique(
                        sample.loc[
                            resolved, "favorable_before_adverse_at_cost"
                        ].to_numpy(dtype=np.int8)
                    ).size
                    == 2
                ):
                    auc = float(
                        roc_auc_score(
                            sample.loc[
                                resolved, "favorable_before_adverse_at_cost"
                            ].to_numpy(dtype=np.int8),
                            prediction[resolved],
                        )
                    )
            rows.append(
                {
                    "window": window,
                    "scope": scope,
                    "head": head,
                    "rows": int(len(sample)),
                    "mae": float(np.mean(np.abs(prediction - target))),
                    "spearman": float(
                        pd.Series(prediction).corr(
                            pd.Series(target), method="spearman"
                        )
                    ),
                    "auc": auc,
                    "population_target_mean": float(target.mean()),
                    "top10_target_mean": float(target[selected].mean()),
                    "top10_target_lift": float(
                        direction * (target[selected].mean() - target.mean())
                    ),
                    "top10_exact_net_bps": float(
                        sample.iloc[selected][TARGET_COLUMN].mean() * 10_000.0
                    ),
                }
            )
    return rows


def _support_replacement_rows(
    evaluation: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    *,
    window: str,
    stage: str,
) -> list[dict[str, Any]]:
    identity = list(IDENTITY_COLUMNS)
    count = max(1, int(np.ceil(0.10 * len(evaluation))))

    def selected(score: np.ndarray) -> pd.DataFrame:
        return evaluation.iloc[
            np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:count]
        ].copy()

    baseline = selected(scores["direct_net"])
    baseline_ids = pd.MultiIndex.from_frame(baseline[identity])
    baseline_components = economic_components(baseline)
    rows = []
    for arm in ARMS:
        challenger = selected(scores[arm])
        challenger_ids = pd.MultiIndex.from_frame(challenger[identity])
        components = economic_components(challenger)
        delta_mfe = (
            components["mean_path_mfe_bps"]
            - baseline_components["mean_path_mfe_bps"]
        )
        delta_gap = (
            components["mean_mfe_to_gross_gap_bps"]
            - baseline_components["mean_mfe_to_gross_gap_bps"]
        )
        delta_cost = (
            components["mean_cost_bps"] - baseline_components["mean_cost_bps"]
        )
        delta_net = (
            components["mean_net_bps"] - baseline_components["mean_net_bps"]
        )
        rows.append(
            {
                "window": window,
                "stage": stage,
                "arm": arm,
                "baseline": "direct_net",
                "selected_rows": count,
                "overlap_rows": int(challenger_ids.isin(baseline_ids).sum()),
                "delta_net_bps": delta_net,
                "delta_mfe_bps": delta_mfe,
                "delta_mfe_to_gross_gap_bps": delta_gap,
                "delta_cost_bps": delta_cost,
                "reconstructed_delta_net_bps": delta_mfe - delta_gap - delta_cost,
                "reconciliation_error_bps": (
                    delta_net - (delta_mfe - delta_gap - delta_cost)
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # This arm is intentionally limited to the already-consumed current blocks.
    # A different input has to receive a separately versioned, predeclared
    # contract before it can become forward decision evidence.
    pinned_input_hashes = assert_frozen_interaction_sources(args)
    margin_interaction_contract = load_frozen_base_margin_interaction(
        args.base_margin_screen
    )
    frame = pd.read_parquet(args.input)
    capture = pd.read_parquet(args.capture_labels)
    grid = pd.read_parquet(args.label_grid)
    grid = grid.loc[
        grid["grid_name"].eq(args.grid_name) & grid["label_valid"],
        [*IDENTITY_COLUMNS, "favorable_first", "adverse_first", "timeout"],
    ].copy()
    if grid.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("comparison label grid contains duplicate identities")
    frame = frame.merge(
        grid,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    capture = capture.rename(
        columns={
            "execution_gross_ev_12h": "capture_label_exact_gross",
            "execution_cost_return": "capture_label_exact_cost",
        }
    )
    keep_capture = [
        *IDENTITY_COLUMNS,
        "capture_label_exact_gross",
        "capture_label_exact_cost",
        "pre_exit_mfe_return",
        "pre_exit_mae_return",
        "pre_exit_mfe_to_gross_gap",
        "pre_exit_gross_capture_ratio",
        "post_peak_close_giveback_ratio",
        "giveback_after_80pct_mfe_ratio",
        "favorable_before_adverse_at_cost",
        "adverse_before_favorable_at_cost",
        "exact_net_positive",
        "exact_net_loss_worse_two_costs",
    ]
    if capture.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("capture labels contain duplicate identities")
    frame = frame.merge(
        capture.loc[:, keep_capture],
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    frame = add_support_targets(frame)
    manifest = json.loads(args.feature_manifest.read_text())
    if "feature_columns_by_side" in manifest:
        feature_columns_by_side = {
            side: list(manifest["feature_columns_by_side"][side])
            for side in SIDES
        }
    else:
        feature_columns_by_side = {
            side: list(manifest["feature_columns"]) for side in SIDES
        }
    feature_columns = list(
        dict.fromkeys(
            feature_columns_by_side["long"]
            + feature_columns_by_side["short"]
        )
    )
    for column in feature_columns:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            level = column[len(prefix) :]
            frame[column] = (
                frame[ARCHETYPE_COLUMN].astype(str).eq(level).astype("float32")
            )
    required = [
        *IDENTITY_COLUMNS,
        DECISION_COLUMN,
        RESOLUTION_COLUMN,
        TARGET_COLUMN,
        BASELINE_COLUMN,
        ARCHETYPE_COLUMN,
        str(margin_interaction_contract["feature"]),
        *feature_columns,
    ]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError("capture-support input missing columns: " + ", ".join(missing))
    frame = frame.sort_values(
        [DECISION_COLUMN, "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    metric_rows = []
    head_metric_rows = []
    replacement_rows = []
    prediction_parts = []
    head_prediction_parts = []
    fold_report: dict[str, Any] = {}
    for window_index, window in enumerate(DEFAULT_WINDOWS):
        train_pos, eval_pos, split = build_forward_split(
            frame, window, purge_hours=args.purge_hours
        )
        train = frame.iloc[train_pos].copy().reset_index(drop=True)
        evaluation = frame.iloc[eval_pos].copy().reset_index(drop=True)
        (
            train_scores,
            evaluation_scores,
            veto_masks,
            head_scores,
            reports,
        ) = (
            fit_capture_support_scores(
            train,
            evaluation,
            feature_columns_by_side,
            margin_interaction_contract,
            iterations=args.n_estimators,
            seed=args.random_state + 100_000 * window_index,
            n_jobs=args.n_jobs,
            )
        )
        head_metric_rows.extend(
            _head_metric_rows(evaluation, head_scores, window=window.name)
        )
        head_part = evaluation.loc[:, list(IDENTITY_COLUMNS)].copy()
        head_part["window"] = window.name
        for name, values in head_scores.items():
            head_part[name] = values
        head_prediction_parts.append(head_part)
        mapped_scores = {}
        for arm in ARMS:
            pre_recent_score = apply_final_veto(
                evaluation_scores[arm], veto_masks[arm]
            )
            for stage, score in (("pre_recent_mapping", pre_recent_score),):
                for row in _capture_metric_rows(
                    evaluation, score, window=window.name, arm=arm, stage=stage
                ):
                    if row["scope"] == "pooled_global":
                        row.update(
                            {
                                f"selected_{key}": value
                                for key, value in _selected_capture_components(
                                    evaluation, score
                                ).items()
                            }
                        )
                    metric_rows.append(row)
            if arm in OOF_GATE_BASES:
                mapped, mapping_report = apply_recent_mapping_scope(
                    train,
                    evaluation,
                    train_scores[arm],
                    evaluation_scores[arm],
                    scope="global",
                )
                primary_stage = "oof_selected_global_recent_mapping"
            else:
                mapped, mapping_report = apply_canonical_recent_mapping(
                    train,
                    evaluation,
                    train_scores[arm],
                    evaluation_scores[arm],
                )
                primary_stage = "canonical_recent_ev_mapping"
            mapped = apply_final_veto(mapped, veto_masks[arm])
            mapped_scores[arm] = mapped
            for row in _capture_metric_rows(
                evaluation,
                mapped,
                window=window.name,
                arm=arm,
                stage=primary_stage,
            ):
                if row["scope"] == "pooled_global":
                    row.update(
                        {
                            f"selected_{key}": value
                            for key, value in _selected_capture_components(
                                evaluation, mapped
                            ).items()
                        }
                    )
                metric_rows.append(row)
            part = evaluation.loc[:, list(IDENTITY_COLUMNS)].copy()
            part["window"] = window.name
            part["arm"] = arm
            part["raw_ev_score"] = evaluation_scores[arm]
            part["canonical_recent_ev_score"] = mapped
            part["mapping_stage"] = primary_stage
            prediction_parts.append(part)
            reports.setdefault("recent_mapping", {})[arm] = mapping_report
            if arm in MAPPING_DIAGNOSTIC_ARMS:
                mapping_variants = {
                    "archetype_recent_shrink50": (
                        evaluation_scores[arm]
                        + 0.5 * (mapped - evaluation_scores[arm]),
                        {"effective_scope": "archetype", "shrink": 0.5},
                    )
                }
                for scope in ("side", "global"):
                    scoped, scoped_report = apply_recent_mapping_scope(
                        train,
                        evaluation,
                        train_scores[arm],
                        evaluation_scores[arm],
                        scope=scope,
                    )
                    mapping_variants[f"recent_{scope}_only_mapping"] = (
                        scoped,
                        scoped_report,
                    )
                for stage, (variant_score, variant_report) in mapping_variants.items():
                    variant_score = apply_final_veto(
                        variant_score, veto_masks[arm]
                    )
                    for row in _capture_metric_rows(
                        evaluation,
                        variant_score,
                        window=window.name,
                        arm=arm,
                        stage=stage,
                    ):
                        if row["scope"] == "pooled_global":
                            row.update(
                                {
                                    f"selected_{key}": value
                                    for key, value in _selected_capture_components(
                                        evaluation, variant_score
                                    ).items()
                                }
                            )
                        metric_rows.append(row)
                    variant_part = evaluation.loc[
                        :, list(IDENTITY_COLUMNS)
                    ].copy()
                    variant_part["window"] = window.name
                    variant_part["arm"] = arm
                    variant_part["raw_ev_score"] = evaluation_scores[arm]
                    variant_part["canonical_recent_ev_score"] = variant_score
                    variant_part["mapping_stage"] = stage
                    prediction_parts.append(variant_part)
                    reports.setdefault("mapping_diagnostics", {}).setdefault(
                        arm, {}
                    )[stage] = variant_report
        replacement_rows.extend(
            _support_replacement_rows(
                evaluation,
                mapped_scores,
                window=window.name,
                stage="primary_arm_specific_recent_mapping",
            )
        )
        fold_report[window.name] = {"split": split, "models": reports}
    args.output_dir.mkdir(parents=True)
    paths = {
        "metrics": args.output_dir / "capture_support_metrics.csv",
        "replacements": args.output_dir / "capture_support_replacements.csv",
        "predictions": args.output_dir / "capture_support_predictions.parquet",
        "head_metrics": args.output_dir / "capture_head_metrics.csv",
        "head_predictions": args.output_dir / "capture_head_predictions.parquet",
    }
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(replacement_rows).to_csv(paths["replacements"], index=False)
    pd.DataFrame(head_metric_rows).to_csv(paths["head_metrics"], index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        paths["predictions"], index=False
    )
    pd.concat(head_prediction_parts, ignore_index=True).to_parquet(
        paths["head_predictions"], index=False
    )
    output_manifest = {
        "schema": SCHEMA,
        "status": "completed_research_oos_not_promotion_evidence",
        "contract": {
            "target": "canonical exact one-minute deployed-policy net EV",
            "support_labels": "decision through exact deployed-policy exit only",
            "head_training": "per-side temporal OOF fixed CatBoost; no HPO",
            "meta_training": "nested temporal OOF residual on OOF head predictions",
            "vetoes": "fixed train-OOF 20th/80th percentile; no evaluation tuning",
            "mapping": "per-side OOF isotonic EV then causal recent-EV correction",
            "ranking": "one pooled global top 10%; no timestamp or side quota",
            "promotion_gate": (
                "reused June/later-July blocks are diagnostic only; a separately "
                "predeclared contract must improve exact net on the next genuinely "
                "new forward block before promotion"
            ),
            "frozen_margin_interaction": {
                **margin_interaction_contract,
                "interpretation": (
                    "fixed soft interaction of an already-live feature and OOF head "
                    "confidence; no evaluation HPO or threshold retuning"
                ),
            },
        },
        "arms": list(ARMS),
        "windows": [window.__dict__ for window in DEFAULT_WINDOWS],
        "inputs": {
            "data": {"path": str(args.input), "sha256": _sha256(args.input)},
            "capture_labels": {
                "path": str(args.capture_labels),
                "sha256": _sha256(args.capture_labels),
            },
            "pinned_source_hashes_verified": pinned_input_hashes,
            "base_margin_screen": {
                "path": str(args.base_margin_screen),
                "sha256": _sha256(args.base_margin_screen),
            },
            "comparison_label_grid": {
                "path": str(args.label_grid),
                "sha256": _sha256(args.label_grid),
                "grid": args.grid_name,
            },
            "feature_manifest": {
                "path": str(args.feature_manifest),
                "sha256": _sha256(args.feature_manifest),
            },
        },
        "feature_columns": feature_columns,
        "feature_columns_by_side": feature_columns_by_side,
        "model": {
            "iterations": args.n_estimators,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "folds": fold_report,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", output_manifest)
    return output_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"
        ),
    )
    parser.add_argument(
        "--capture-labels",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_labels_20260727_v1/exact_policy_capture_labels.parquet"
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/regime_diagnosis_manifest.json"
        ),
    )
    parser.add_argument(
        "--label-grid",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet"
        ),
    )
    parser.add_argument(
        "--base-margin-screen",
        type=Path,
        default=FROZEN_BASE_MARGIN_SCREEN,
        help="immutable source of the predeclared smooth margin interaction",
    )
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260727)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
