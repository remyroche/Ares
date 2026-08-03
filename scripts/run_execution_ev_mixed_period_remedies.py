#!/usr/bin/env python3
"""Bounded mixed-period remedies for the frozen execution-EV winner.

Every primary result is past-to-future and uses one pooled global top decile
after the canonical 21-day causal side x archetype EV correction.  The script
does not call a timestamp-local quota.  It compares fixed, predeclared training
weights, a recent residual correction, and train-fitted trust composites.

The runs are research-selected OOS diagnostics, not untouched promotion tests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import execution_ev_metrics  # noqa: E402
from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ExecutionEVModelAblationConfig,
    apply_execution_ev_causal_recent_ev_correction,
    fit_train_only_isotonic_ev_mapping,
)


SCHEMA = "execution_ev_mixed_period_remedies_v1"
BASELINE_COLUMN = "existing_alpha_ev"
DECISION_COLUMN = "execution_decision_utc"
RESOLUTION_COLUMN = "execution_label_end_utc"
TARGET_COLUMN = "execution_net_ev_12h"
SIDE_COLUMN = "side_name"
ARCHETYPE_COLUMN = "catboost_archetype"
IDENTITY_COLUMNS = ("__ts__", "__symbol__", SIDE_COLUMN, "candidate_id")
OBSERVABLE_VOLATILITY_FEATURES = (
    "mkt_state__atr_compression_ratio__h0",
    "mkt_state__atr_pct_change__h0",
    "mkt_state__atr_slope__h0",
    "mkt_state__mkt_atr_expansion_4h__h0",
    "mkt_state__volatility_of_volatility_48__h0",
)


@dataclass(frozen=True)
class ForwardWindow:
    name: str
    train_start: str
    cutoff: str
    evaluation_end: str
    retention_role: str


DEFAULT_WINDOWS = (
    ForwardWindow(
        name="may_to_june_forward_control",
        train_start="2026-05-01T00:00:00Z",
        cutoff="2026-06-01T00:00:00Z",
        evaluation_end="2026-07-01T00:00:00Z",
        retention_role="causal_may_june_retention_control",
    ),
    ForwardWindow(
        name="later_july_forward",
        train_start="2026-05-01T00:00:00Z",
        cutoff="2026-07-12T00:00:00Z",
        evaluation_end="2026-07-20T00:00:00Z",
        retention_role="causal_later_july_research_oos",
    ),
)


@dataclass(frozen=True)
class Arm:
    name: str
    weight_mode: str = "uniform"
    train_through_june_only: bool = False
    trust_composites: bool = False
    recent_residual_shrink: float = 0.0


DEFAULT_ARMS = (
    Arm("uniform_all_available"),
    Arm("uniform_may_june_only", train_through_june_only=True),
    Arm("early_july_3x", weight_mode="early_july_3x"),
    Arm("recency_half_life_14d", weight_mode="recency_14d"),
    Arm("regime_calendar_archetype_balanced", weight_mode="regime_balanced"),
    Arm("causal_trust_composites", trust_composites=True),
    Arm("global_plus_recent_residual_050", recent_residual_shrink=0.50),
)
OBSERVABLE_STATE_ARMS = (
    Arm(
        "observable_volatility_state_balanced",
        weight_mode="observable_volatility_balanced",
    ),
)


def build_forward_split(
    frame: pd.DataFrame,
    window: ForwardWindow,
    *,
    purge_hours: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    decision = pd.to_datetime(frame[DECISION_COLUMN], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[RESOLUTION_COLUMN], utc=True, errors="raise")
    start = pd.Timestamp(window.train_start)
    cutoff = pd.Timestamp(window.cutoff)
    end = pd.Timestamp(window.evaluation_end)
    train = (
        decision.ge(start)
        & decision.lt(cutoff - pd.Timedelta(hours=float(purge_hours)))
        & resolved.lt(cutoff)
    )
    evaluation = decision.ge(cutoff) & decision.lt(end)
    train_positions = np.flatnonzero(train.to_numpy())
    evaluation_positions = np.flatnonzero(evaluation.to_numpy())
    if not len(train_positions) or not len(evaluation_positions):
        raise ValueError(f"forward window {window.name!r} has empty train/evaluation")
    if not (
        decision.iloc[train_positions].max()
        < cutoff - pd.Timedelta(hours=float(purge_hours))
        and resolved.iloc[train_positions].max() < cutoff
        and decision.iloc[evaluation_positions].min() >= cutoff
    ):
        raise RuntimeError(f"forward window {window.name!r} violates temporal safety")
    return train_positions, evaluation_positions, {
        "window": window.name,
        "evaluation_status": "past_to_future_research_oos_not_untouched",
        "promotion_eligible": False,
        "train_rows": int(len(train_positions)),
        "evaluation_rows": int(len(evaluation_positions)),
        "max_train_decision_utc": decision.iloc[train_positions].max().isoformat(),
        "max_train_label_resolution_utc": resolved.iloc[train_positions].max().isoformat(),
        "evaluation_start_utc": decision.iloc[evaluation_positions].min().isoformat(),
        "evaluation_end_utc": decision.iloc[evaluation_positions].max().isoformat(),
        "retention_role": window.retention_role,
    }


def training_weights(train: pd.DataFrame, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    decision = pd.to_datetime(train[DECISION_COLUMN], utc=True, errors="raise")
    if mode == "uniform":
        weights = np.ones(len(train), dtype=float)
    elif mode == "early_july_3x":
        weights = np.where(decision.dt.month.eq(7), 3.0, 1.0).astype(float)
    elif mode == "recency_14d":
        age_days = (decision.max() - decision).dt.total_seconds().to_numpy() / 86400.0
        weights = np.exp2(-age_days / 14.0)
    elif mode == "regime_balanced":
        # Equalize mass across calendar month x observable path archetype cells.
        # The cells and their counts are derived from authorized train inputs only.
        cells = decision.dt.strftime("%Y-%m").astype(str) + "__" + train[
            ARCHETYPE_COLUMN
        ].astype(str)
        count = cells.map(cells.value_counts()).to_numpy(dtype=float)
        weights = len(train) / np.maximum(count, 1.0)
        weights /= np.mean(weights)
        weights = np.clip(weights, 0.33, 3.0)
    elif mode == "observable_volatility_balanced":
        available = [
            column
            for column in OBSERVABLE_VOLATILITY_FEATURES
            if column in train
            and pd.to_numeric(train[column], errors="coerce").notna().mean() >= 0.95
        ]
        if len(available) < 2:
            raise ValueError(
                "observable volatility balancing requires at least two "
                "95%-covered decision-time volatility fields"
            )
        values = train.loc[:, available].apply(pd.to_numeric, errors="coerce")
        transform = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(25.0, 75.0)),
        )
        geometry = transform.fit_transform(values)
        cluster_count = min(4, max(2, len(train) // 500))
        state = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=5,
            batch_size=min(4_096, len(train)),
            max_iter=100,
            reassignment_ratio=0.0,
        ).fit_predict(geometry)
        counts = pd.Series(state).value_counts()
        weights = np.asarray([len(train) / counts.loc[item] for item in state], dtype=float)
        weights /= np.mean(weights)
        weights = np.clip(weights, 0.33, 3.0)
    else:
        raise ValueError(f"unknown weight mode {mode!r}")
    weights = np.asarray(weights, dtype=float)
    weights /= max(float(np.mean(weights)), 1e-12)
    ess = float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12))
    report = {
        "mode": mode,
        "rows": int(len(weights)),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "effective_sample_size": ess,
        "effective_sample_fraction": float(ess / max(len(weights), 1)),
        "july_weight_mass": float(weights[decision.dt.month.eq(7).to_numpy()].sum()),
    }
    if mode == "observable_volatility_balanced":
        report.update(
            {
                "observable_features": available,
                "observable_state_count": int(cluster_count),
                "observable_state_rows": {
                    str(key): int(value) for key, value in counts.sort_index().items()
                },
                "geometry_fit": "train-only median imputation, robust scaling, fixed MiniBatchKMeans K<=4",
            }
        )
    return weights, report


def join_observable_state(
    frame: pd.DataFrame,
    state_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach decision-time h0 state fields on exact immutable identities."""

    state = pd.read_parquet(state_path)
    missing = [
        column
        for column in (*IDENTITY_COLUMNS, *OBSERVABLE_VOLATILITY_FEATURES)
        if column not in state
    ]
    if missing:
        raise ValueError("observable-state input is missing columns: " + ", ".join(missing))
    if frame.duplicated(list(IDENTITY_COLUMNS)).any() or state.duplicated(
        list(IDENTITY_COLUMNS)
    ).any():
        raise ValueError("observable-state join identities must be unique")
    keep = [*IDENTITY_COLUMNS, *OBSERVABLE_VOLATILITY_FEATURES]
    if "raw_state_source_utc_h0" in state:
        keep.append("raw_state_source_utc_h0")
    joined = frame.merge(
        state.loc[:, keep],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if "raw_state_source_utc_h0" in joined:
        source = pd.to_datetime(joined["raw_state_source_utc_h0"], utc=True, errors="coerce")
        decision = pd.to_datetime(joined[DECISION_COLUMN], utc=True, errors="raise")
        invalid = source.notna() & source.gt(decision)
        if invalid.any():
            raise ValueError("observable h0 state source occurs after decision time")
    return joined, {
        "path": str(state_path),
        "input_rows": int(len(frame)),
        "state_rows": int(len(state)),
        "joined_rows": int(len(joined)),
        "complete_volatility_rows": int(
            joined.loc[:, list(OBSERVABLE_VOLATILITY_FEATURES)].notna().all(axis=1).sum()
        ),
    }


def _robust_center_scale(reference: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(reference, errors="raise").to_numpy(dtype=float)
    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), float(np.std(values)), 1e-6)
    return center, scale


def add_trust_composites(
    reference: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Add deterministic composites fitted only on the current train reference."""
    required = [
        BASELINE_COLUMN,
        "oof_clean_favorable_probability",
        "base_margin_to_cutoff_z",
        "alpha_prediction_uncertainty",
        ARCHETYPE_COLUMN,
    ]
    missing = [column for column in required if column not in reference or column not in target]
    if missing:
        raise ValueError("trust composites are missing columns: " + ", ".join(missing))
    out = target.copy()
    standardized: dict[str, np.ndarray] = {}
    for column in required[:4]:
        center, scale = _robust_center_scale(reference[column])
        standardized[column] = np.clip(
            (pd.to_numeric(target[column], errors="raise").to_numpy(dtype=float) - center)
            / scale,
            -8.0,
            8.0,
        )
    alpha = standardized[BASELINE_COLUMN]
    clean = standardized["oof_clean_favorable_probability"]
    margin = standardized["base_margin_to_cutoff_z"]
    uncertainty = np.abs(standardized["alpha_prediction_uncertainty"])
    ref_prior = reference[ARCHETYPE_COLUMN].astype(str).value_counts(normalize=True)
    rarity = -np.log(
        target[ARCHETYPE_COLUMN].astype(str).map(ref_prior).fillna(1.0 / max(len(reference), 1)).to_numpy(dtype=float)
        + 1e-9
    )
    rarity = np.clip(rarity / max(float(np.median(rarity)), 1e-6), 0.0, 8.0)
    out["trust_alpha_clean_agreement"] = -np.abs(alpha - clean)
    out["trust_alpha_clean_joint"] = alpha * clean
    out["trust_margin_after_uncertainty"] = margin / (1.0 + uncertainty)
    out["trust_clean_after_uncertainty"] = clean / (1.0 + uncertainty)
    out["trust_alpha_margin_after_ood"] = alpha * margin / (1.0 + rarity)
    out["trust_archetype_rarity"] = rarity
    return out


def _model_features(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    trust_composites: bool,
) -> pd.DataFrame:
    work = add_trust_composites(reference, target) if trust_composites else target
    names = list(feature_columns)
    if trust_composites:
        names.extend(column for column in work.columns if column.startswith("trust_"))
    numeric = work.loc[:, names].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("mixed-period model features must be finite")
    return pd.DataFrame(values, columns=names, index=target.index)


def _catboost(*, iterations: int, depth: int, seed: int, n_jobs: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MAE",
        iterations=int(iterations),
        learning_rate=0.03,
        depth=int(depth),
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        random_seed=int(seed),
        thread_count=int(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )


def _temporal_oof_blocks(frame: pd.DataFrame, *, min_train_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    decision = pd.to_datetime(frame[DECISION_COLUMN], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[RESOLUTION_COLUMN], utc=True, errors="raise")
    unique_days = pd.Index(decision.dt.floor("D").unique()).sort_values()
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for fraction in (0.40, 0.60, 0.80):
        position = min(max(int(np.floor(fraction * len(unique_days))), 1), len(unique_days) - 1)
        validation_start = pd.Timestamp(unique_days[position])
        later_position = min(position + max(int(np.ceil(0.20 * len(unique_days))), 1), len(unique_days))
        validation_end = (
            pd.Timestamp(unique_days[later_position])
            if later_position < len(unique_days)
            else decision.max() + pd.Timedelta(microseconds=1)
        )
        train = np.flatnonzero(
            (decision < validation_start - pd.Timedelta(hours=12)).to_numpy()
            & (resolved < validation_start).to_numpy()
        )
        valid = np.flatnonzero(
            (decision >= validation_start).to_numpy() & (decision < validation_end).to_numpy()
        )
        if len(train) >= int(min_train_rows) and len(valid):
            blocks.append((train, valid))
    return blocks


def fit_arm_scores(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: Sequence[str],
    arm: Arm,
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return train OOF and forward evaluation absolute-EV predictions."""
    oof = np.full(len(train), np.nan, dtype=float)
    evaluation_prediction = np.full(len(evaluation), np.nan, dtype=float)
    reports: dict[str, Any] = {}
    for side in ("long", "short"):
        train_side = train.loc[train[SIDE_COLUMN].astype(str).str.lower().eq(side)].copy().reset_index()
        eval_side = evaluation.loc[evaluation[SIDE_COLUMN].astype(str).str.lower().eq(side)].copy().reset_index()
        if train_side.empty or eval_side.empty:
            continue
        side_oof = np.full(len(train_side), np.nan, dtype=float)
        fold_reports: list[dict[str, Any]] = []
        for fold_number, (fit_pos, valid_pos) in enumerate(
            _temporal_oof_blocks(train_side, min_train_rows=2_000), start=1
        ):
            fit = train_side.iloc[fit_pos]
            valid = train_side.iloc[valid_pos]
            fit_x = _model_features(fit, fit, feature_columns, trust_composites=arm.trust_composites)
            valid_x = _model_features(fit, valid, feature_columns, trust_composites=arm.trust_composites)
            weight, weight_report = training_weights(fit, arm.weight_mode)
            model = _catboost(iterations=iterations, depth=6, seed=seed + fold_number, n_jobs=n_jobs)
            target = fit[TARGET_COLUMN].to_numpy(dtype=float) - fit[BASELINE_COLUMN].to_numpy(dtype=float)
            model.fit(fit_x, target, sample_weight=weight)
            side_oof[valid_pos] = valid[BASELINE_COLUMN].to_numpy(dtype=float) + model.predict(valid_x)
            fold_reports.append({
                "fold": fold_number,
                "fit_rows": int(len(fit)),
                "valid_rows": int(len(valid)),
                "max_fit_resolution_utc": pd.to_datetime(fit[RESOLUTION_COLUMN], utc=True).max().isoformat(),
                "validation_start_utc": pd.to_datetime(valid[DECISION_COLUMN], utc=True).min().isoformat(),
                "weights": weight_report,
            })
        final_x = _model_features(train_side, train_side, feature_columns, trust_composites=arm.trust_composites)
        eval_x = _model_features(train_side, eval_side, feature_columns, trust_composites=arm.trust_composites)
        final_weight, final_weight_report = training_weights(train_side, arm.weight_mode)
        final_model = _catboost(iterations=iterations, depth=6, seed=seed, n_jobs=n_jobs)
        final_target = train_side[TARGET_COLUMN].to_numpy(dtype=float) - train_side[BASELINE_COLUMN].to_numpy(dtype=float)
        final_model.fit(final_x, final_target, sample_weight=final_weight)
        side_eval = eval_side[BASELINE_COLUMN].to_numpy(dtype=float) + final_model.predict(eval_x)
        mapper = fit_train_only_isotonic_ev_mapping(
            side_oof,
            train_side[TARGET_COLUMN].to_numpy(dtype=float),
            min_rows=24,
        )
        side_oof_mapped = np.full(len(side_oof), np.nan, dtype=float)
        finite_oof = np.isfinite(side_oof)
        side_oof_mapped[finite_oof] = mapper.predict(side_oof[finite_oof])
        side_eval_mapped = mapper.predict(side_eval)
        oof[train_side["index"].to_numpy(dtype=int)] = side_oof_mapped
        evaluation_prediction[eval_side["index"].to_numpy(dtype=int)] = side_eval_mapped
        reports[side] = {
            "train_rows": int(len(train_side)),
            "evaluation_rows": int(len(eval_side)),
            "oof_rows": int(np.isfinite(side_oof).sum()),
            "isotonic_status": mapper.status,
            "final_weights": final_weight_report,
            "oof_folds": fold_reports,
        }
    if not np.isfinite(evaluation_prediction).all():
        raise ValueError("arm failed to score every forward evaluation row")
    return oof, evaluation_prediction, reports


def recent_residual_correction(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    train_oof_score: np.ndarray,
    evaluation_score: np.ndarray,
    feature_columns: Sequence[str],
    *,
    shrink: float,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cutoff = pd.to_datetime(evaluation[DECISION_COLUMN], utc=True).min()
    decision = pd.to_datetime(train[DECISION_COLUMN], utc=True)
    corrected_train = np.asarray(train_oof_score, dtype=float).copy()
    corrected = np.asarray(evaluation_score, dtype=float).copy()
    report: dict[str, Any] = {}
    for side in ("long", "short"):
        fit_mask = (
            train[SIDE_COLUMN].astype(str).str.lower().eq(side).to_numpy()
            & decision.ge(cutoff - pd.Timedelta(days=21)).to_numpy()
            & np.isfinite(train_oof_score)
        )
        eval_mask = evaluation[SIDE_COLUMN].astype(str).str.lower().eq(side).to_numpy()
        fit = train.loc[fit_mask].copy()
        future = evaluation.loc[eval_mask].copy()
        if len(fit) < 500 or future.empty:
            report[side] = {"status": "insufficient_support", "rows": int(len(fit))}
            continue
        fit_x = _model_features(fit, fit, feature_columns, trust_composites=False)
        eval_x = _model_features(fit, future, feature_columns, trust_composites=False)
        residual = fit[TARGET_COLUMN].to_numpy(dtype=float) - train_oof_score[fit_mask]
        weights, weight_report = training_weights(fit, "recency_14d")
        model = _catboost(iterations=100, depth=4, seed=seed + 101, n_jobs=n_jobs)
        model.fit(fit_x, residual, sample_weight=weights)
        delta = np.clip(np.asarray(model.predict(eval_x), dtype=float), -0.01, 0.01)
        corrected[eval_mask] += float(shrink) * delta
        # Produce a temporally honest correction score for the later half of
        # the recent reference.  The canonical recent-EV mapper must compare
        # like with like: corrected evaluation scores against prior corrected
        # OOF scores, never in-sample correction predictions.
        fit_decision = pd.to_datetime(fit[DECISION_COLUMN], utc=True)
        unique_days = pd.Index(fit_decision.dt.floor("D").unique()).sort_values()
        split_position = min(max(1, len(unique_days) // 2), len(unique_days) - 1)
        split_day = pd.Timestamp(unique_days[split_position])
        early = (
            fit_decision.lt(split_day)
            & pd.to_datetime(fit[RESOLUTION_COLUMN], utc=True).lt(split_day)
        ).to_numpy()
        late = fit_decision.ge(split_day).to_numpy()
        correction_oof_rows = 0
        if int(early.sum()) >= 500 and int(late.sum()) >= 100:
            early_frame = fit.loc[early]
            late_frame = fit.loc[late]
            early_x = _model_features(
                early_frame, early_frame, feature_columns, trust_composites=False
            )
            late_x = _model_features(
                early_frame, late_frame, feature_columns, trust_composites=False
            )
            early_weight, _ = training_weights(early_frame, "recency_14d")
            early_model = _catboost(
                iterations=100, depth=4, seed=seed + 202, n_jobs=n_jobs
            )
            early_model.fit(early_x, residual[early], sample_weight=early_weight)
            late_delta = np.clip(
                np.asarray(early_model.predict(late_x), dtype=float), -0.01, 0.01
            )
            original_positions = np.flatnonzero(fit_mask)[late]
            corrected_train[original_positions] += float(shrink) * late_delta
            correction_oof_rows = int(len(original_positions))
        report[side] = {
            "status": "fit_on_recent_oof_residuals",
            "rows": int(len(fit)),
            "shrink": float(shrink),
            "clip": [-0.01, 0.01],
            "mean_abs_eval_delta_before_shrink": float(np.mean(np.abs(delta))),
            "correction_oof_rows": correction_oof_rows,
            "correction_oof_split_utc": split_day.isoformat(),
            "weights": weight_report,
        }
    return corrected_train, corrected, report


def apply_canonical_recent_mapping(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    train_oof_score: np.ndarray,
    evaluation_score: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    combined = pd.concat([train, evaluation], ignore_index=True)
    mapped = np.concatenate([train_oof_score, evaluation_score])
    config = ExecutionEVModelAblationConfig(
        decision_time_col=DECISION_COLUMN,
        label_end_time_col=RESOLUTION_COLUMN,
        side_col=SIDE_COLUMN,
        catboost_archetype_col=ARCHETYPE_COLUMN,
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
    return corrected[len(train) :], report


def _metric_rows(
    window: ForwardWindow,
    arm: Arm,
    evaluation: pd.DataFrame,
    raw_score: np.ndarray,
    mapped_score: np.ndarray,
    *,
    split_report: dict[str, Any],
) -> list[dict[str, Any]]:
    target = evaluation[TARGET_COLUMN].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [("pooled", np.ones(len(evaluation), dtype=bool))]
    scopes.extend(
        (f"side_{side}", evaluation[SIDE_COLUMN].astype(str).str.lower().eq(side).to_numpy())
        for side in ("long", "short")
    )
    for stage, prediction in (("pre_recent_mapping", raw_score), ("canonical_recent_ev_mapping", mapped_score)):
        for scope, mask in scopes:
            if not mask.any():
                continue
            metrics = execution_ev_metrics(target[mask], prediction[mask], top_k_fraction=0.10)
            rows.append({
                "window": window.name,
                "retention_role": window.retention_role,
                "evaluation_status": split_report["evaluation_status"],
                "promotion_eligible": False,
                "arm": arm.name,
                "stage": stage,
                "scope": scope,
                "eligible_rows": int(mask.sum()),
                "coverage_rate": float(np.isfinite(prediction[mask]).mean()),
                **metrics,
                "top_k_mean_net_ev_bps": float(10_000.0 * metrics["top_k_mean_net_ev"]),
            })
    return rows


def run(args: argparse.Namespace) -> dict[str, Path]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(args.input)
    observable_state_audit = None
    if args.observable_state_input is not None:
        frame, observable_state_audit = join_observable_state(
            frame, args.observable_state_input
        )
    manifest = json.loads(Path(args.feature_manifest).read_text(encoding="utf-8"))
    feature_columns = list(manifest["feature_columns"])
    for column in feature_columns:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            level = column[len(prefix) :]
            frame[column] = frame[ARCHETYPE_COLUMN].astype(str).eq(level).astype("float32")
    missing = [column for column in [*feature_columns, DECISION_COLUMN, RESOLUTION_COLUMN, TARGET_COLUMN, SIDE_COLUMN, ARCHETYPE_COLUMN] if column not in frame]
    if missing:
        raise ValueError("mixed-period input is missing columns: " + ", ".join(missing))
    frame = frame.sort_values([DECISION_COLUMN, "candidate_id"], kind="stable").reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    available_arms = [
        *DEFAULT_ARMS,
        *(OBSERVABLE_STATE_ARMS if args.observable_state_input is not None else ()),
    ]
    if args.arm:
        requested = set(args.arm)
        unknown = requested - {arm.name for arm in available_arms}
        if unknown:
            raise ValueError("unknown or unavailable arms: " + ", ".join(sorted(unknown)))
        selected_arms = [arm for arm in available_arms if arm.name in requested]
    else:
        selected_arms = available_arms
    for window in DEFAULT_WINDOWS:
        train_pos, eval_pos, split_report = build_forward_split(frame, window, purge_hours=args.purge_hours)
        base_train = frame.iloc[train_pos].copy().reset_index(drop=True)
        evaluation = frame.iloc[eval_pos].copy().reset_index(drop=True)
        split_rows.append(split_report)
        window_audit: dict[str, Any] = {}
        uniform_cache: tuple[np.ndarray, np.ndarray] | None = None
        for arm in selected_arms:
            train = base_train
            # This is a valid no-July baseline only for the later-July window.
            # On May->June it is identical to the all-available baseline and is skipped.
            if arm.train_through_june_only:
                if pd.Timestamp(window.cutoff).month != 7:
                    continue
                train = train.loc[pd.to_datetime(train[DECISION_COLUMN], utc=True) < pd.Timestamp("2026-07-01T00:00:00Z")].reset_index(drop=True)
            if arm.recent_residual_shrink > 0.0:
                if uniform_cache is None:
                    raise RuntimeError("uniform arm must run before recent residual arm")
                train_oof, raw_evaluation = uniform_cache
                train_oof, raw_evaluation, residual_report = recent_residual_correction(
                    base_train,
                    evaluation,
                    train_oof,
                    raw_evaluation,
                    feature_columns,
                    shrink=arm.recent_residual_shrink,
                    seed=args.random_state,
                    n_jobs=args.n_jobs,
                )
                train_for_mapping = base_train
                arm_report = {"source": "uniform_all_available", "recent_residual": residual_report}
            else:
                train_oof, raw_evaluation, arm_report = fit_arm_scores(
                    train,
                    evaluation,
                    feature_columns,
                    arm,
                    iterations=args.n_estimators,
                    seed=args.random_state,
                    n_jobs=args.n_jobs,
                )
                train_for_mapping = train
                if arm.name == "uniform_all_available":
                    uniform_cache = (train_oof.copy(), raw_evaluation.copy())
            mapped_evaluation, mapping_report = apply_canonical_recent_mapping(
                train_for_mapping,
                evaluation,
                train_oof,
                raw_evaluation,
            )
            metric_rows.extend(
                _metric_rows(
                    window,
                    arm,
                    evaluation,
                    raw_evaluation,
                    mapped_evaluation,
                    split_report=split_report,
                )
            )
            identity = [column for column in ("__ts__", "__symbol__", SIDE_COLUMN, "candidate_id") if column in evaluation]
            prediction_parts.append(
                evaluation.loc[:, identity + [DECISION_COLUMN, RESOLUTION_COLUMN, TARGET_COLUMN]].assign(
                    window=window.name,
                    arm=arm.name,
                    prediction_pre_recent_mapping=raw_evaluation,
                    prediction_canonical_recent_ev_mapping=mapped_evaluation,
                    evaluation_status=split_report["evaluation_status"],
                    promotion_eligible=False,
                )
            )
            window_audit[arm.name] = {
                "train_rows": int(len(train_for_mapping)),
                "model": arm_report,
                "recent_mapping": mapping_report,
            }
        audit[window.name] = window_audit
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "mixed_period_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "mixed_period_splits.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(output / "mixed_period_predictions.parquet", index=False)
    manifest_out = {
        "schema": SCHEMA,
        "status": "completed",
        "input": {"path": str(args.input), "rows": int(len(frame))},
        "feature_manifest": str(args.feature_manifest),
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "architecture": "per-side CatBoost residual over frozen alpha; fixed no-HPO geometry",
        "selection_contract": "one pooled global top10 after canonical 21d causal side x predicted-archetype EV correction; never per timestamp",
        "evidence_contract": {
            "temporal": "train decisions purged 12h and train labels resolved strictly before each evaluation cutoff",
            "status": "research-selected forward OOS diagnostics; not untouched and not promotion eligible",
            "may_june_retention": "May-trained forward June evaluation, not a reverse-time replay",
            "later_july": "labels resolved before July 12 train; July 12-19 forward evaluation",
        },
        "windows": [asdict(window) for window in DEFAULT_WINDOWS],
        "arms": [asdict(arm) for arm in selected_arms],
        "observable_state_input": observable_state_audit,
        "model": {"iterations": int(args.n_estimators), "depth": 6, "random_state": int(args.random_state)},
        "audit": audit,
    }
    (output / "mixed_period_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": output / "mixed_period_metrics.csv",
        "splits": output / "mixed_period_splits.csv",
        "predictions": output / "mixed_period_predictions.parquet",
        "manifest": output / "mixed_period_manifest.json",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--observable-state-input", type=Path)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=3)
    return parser


def main() -> None:
    paths = run(_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
