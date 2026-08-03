#!/usr/bin/env python3
"""Strict-OOF calibration, EV-drift and asymmetric-uncertainty ablations.

This is deliberately a *post-model* experiment.  It consumes direct execution
EV predictions that are already outer-OOF, and uses only earlier OOF rows whose
12-hour labels have resolved before the next fold starts.  Consequently it can
also consume the longer backfill without a code change: pass a compatible OOF
prediction panel and causal context panel via ``--predictions``/``--context``.

All scores remain fractional, cost-adjusted net-return EV.  The output never
allocates a per-timestamp or per-side quota: admission is one pooled global
top-k after the relevant causal map.  Uncertainty is used asymmetrically: it
penalises predicted EV that is expected to be overestimated, never generic
absolute uncertainty.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "execution_ev_calibration_drift_uncertainty_ablation_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
SIDE = "side_name"
TARGET = "execution_net_ev_12h"
DIRECT = "direct_net_ev"
FOLD = "oof_fold"
DEFAULT_PREDICTIONS = ROOT / (
    "data_perp/artifacts/execution_ev_hierarchical_shared_multitask_compact_july19_20260726_v3/"
    "oof_predictions.parquet"
)
DEFAULT_CONTEXT = ROOT / (
    "data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v1/"
    "raw_market_state_transition_rows.parquet"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/execution_ev_calibration_drift_uncertainty_july19_20260726_v1"
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
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


@dataclass(frozen=True)
class CalibrationArm:
    name: str
    window_days: int
    half_life_days: float | None
    side_weight: float
    identity_weight: float


CALIBRATION_ARMS = (
    CalibrationArm("hierarchical_7d_uniform", 7, None, 0.75, 0.25),
    CalibrationArm("hierarchical_21d_uniform", 21, None, 0.75, 0.25),
    CalibrationArm("hierarchical_35d_uniform", 35, None, 0.75, 0.25),
    CalibrationArm("hierarchical_21d_ewm", 21, 7.0, 0.75, 0.25),
    CalibrationArm("hierarchical_21d_ewm_identity_050", 21, 7.0, 0.75, 0.50),
    CalibrationArm("hierarchical_21d_ewm_pooled", 21, 7.0, 0.35, 0.25),
)
PRIMARY_CALIBRATION = "hierarchical_21d_ewm"


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (DECISION, RESOLUTION):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    return result


def _required(frame: pd.DataFrame, columns: Sequence[str], *, source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_panel(predictions_path: Path, context_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """One-to-one join strict OOF predictions to pre-entry context only."""
    predictions = pd.read_parquet(predictions_path)
    _required(predictions, [*IDENTITY, DECISION, RESOLUTION, TARGET, DIRECT, FOLD], source="predictions")
    # The raw context panel carries many unrelated future-path columns.  Read
    # its schema first and then only the explicitly pre-entry columns; loading
    # the full historical panel is both unsafe by default and needlessly large.
    context_schema = tuple(pq.ParquetFile(context_path).schema.names)
    context_schema_set = set(context_schema)
    missing_context = sorted(set(IDENTITY).difference(context_schema_set))
    if missing_context:
        raise ValueError("context is missing required columns: " + ", ".join(missing_context))
    if predictions.duplicated(list(IDENTITY)).any():
        raise ValueError("prediction/context identities must each be unique")
    protected = set(IDENTITY) | {DECISION, RESOLUTION, TARGET, DIRECT, FOLD}
    # These sources are all pre-entry or precomputed OOF predictions.  Future
    # path fields are deliberately excluded even if a historical context panel
    # happens to contain them.
    allowed_prefixes = ("catboost_p_", "base_archetype_label__")
    allowed_names = {
        "existing_alpha_ev", "pred_peak_MFE_12h_ATR", "catboost_entropy",
        "alpha_prediction_uncertainty", "alpha_leaf_support", "base_oof_score",
        "base_margin_to_cutoff", "base_margin_to_cutoff_z",
        "oof_clean_favorable_probability", "raw_state_source_utc_h0",
    }
    context_columns = [
        c for c in context_schema
        if c not in protected
        and (
            c in allowed_names
            or c.startswith(allowed_prefixes)
            # h1/h3/h6/h12 state columns are deliberately rejected: they
            # describe later state transitions, not decision-time context.
            or (c.startswith("mkt_state__") and c.endswith("__h0"))
        )
    ]
    context = pd.read_parquet(context_path, columns=[*IDENTITY, *context_columns])
    if context.duplicated(list(IDENTITY)).any():
        raise ValueError("prediction/context identities must each be unique")
    work = predictions.merge(
        context.loc[:, [*IDENTITY, *context_columns]],
        on=list(IDENTITY), how="inner", validate="one_to_one", sort=False,
    )
    if len(work) != len(predictions):
        raise ValueError("context must cover every strict OOF prediction identity")
    work = _utc(work).sort_values([DECISION, "candidate_id"], kind="stable").reset_index(drop=True)
    if set(work[SIDE].astype(str).str.lower()) - {"long", "short"}:
        raise ValueError("side_name must be long/short")
    if "raw_state_source_utc_h0" in work:
        available = pd.to_datetime(work["raw_state_source_utc_h0"], utc=True, errors="raise")
        if (available > work[DECISION]).any():
            raise ValueError("raw h0 state has availability after decision time")
    numerical = [c for c in context_columns if c != "raw_state_source_utc_h0"]
    # Features that are sparse in the current panel are not silently invented.
    # Per-fold median imputation below is fit only on authorized prior rows.
    return work, numerical


@dataclass
class IsoMap:
    model: IsotonicRegression | None
    status: str
    rows: int

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        return raw if self.model is None else np.asarray(self.model.predict(raw), dtype=float)


def fit_weighted_isotonic(
    prediction: Sequence[float], target: Sequence[float], weights: Sequence[float], *, min_rows: int,
) -> IsoMap:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(weights) & (weights > 0.0)
    if int(valid.sum()) < max(2, int(min_rows)):
        return IsoMap(None, "identity_insufficient_prior_oof", int(valid.sum()))
    if np.unique(prediction[valid]).size < 2:
        return IsoMap(None, "identity_constant_prior_oof", int(valid.sum()))
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(prediction[valid], target[valid], sample_weight=weights[valid])
    return IsoMap(model, "weighted_isotonic_prior_oof", int(valid.sum()))


def _reference_window(
    decision: pd.Series, resolved: pd.Series, fold: np.ndarray, current_fold: int,
    validation_start: pd.Timestamp, window_days: int, *, min_rows: int,
) -> tuple[np.ndarray, str]:
    base = (fold < current_fold) & resolved.lt(validation_start).to_numpy()
    recent = base & decision.ge(validation_start - pd.Timedelta(days=int(window_days))).to_numpy()
    if int(recent.sum()) >= int(min_rows):
        return recent, "recent_window"
    return base, "all_prior_fallback"


def _weights(decision: pd.Series, mask: np.ndarray, validation_start: pd.Timestamp, half_life_days: float | None) -> np.ndarray:
    result = np.ones(int(mask.sum()), dtype=float)
    if half_life_days is not None:
        age = (validation_start - decision.loc[mask]).dt.total_seconds().to_numpy(float) / 86400.0
        result = np.power(0.5, np.maximum(age, 0.0) / float(half_life_days))
    return result


def temporal_hierarchical_calibration(
    frame: pd.DataFrame, direct: np.ndarray, target: np.ndarray, folds: np.ndarray, arm: CalibrationArm, *, min_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Side monotonic map + disjoint pooled EV anchor + identity shrinkage.

    The anchor is fitted on a later disjoint portion of prior OOF history.  It
    therefore puts both sides in common realized-EV units without using the
    current fold's outcome.  The raw direct EV is the explicit identity prior.
    """
    out = np.asarray(direct, dtype=float).copy()
    decision, resolved = frame[DECISION], frame[RESOLUTION]
    side = frame[SIDE].astype(str).str.lower().to_numpy()
    audit: list[dict[str, Any]] = []
    for current_fold in sorted(int(x) for x in np.unique(folds[np.isfinite(folds)])):
        current = (folds == current_fold) & np.isfinite(direct)
        if not current.any():
            continue
        start = decision.loc[current].min()
        reference, window_status = _reference_window(
            decision, resolved, folds, current_fold, start, arm.window_days, min_rows=min_rows,
        )
        ref_times = np.sort(decision.loc[reference].unique())
        if len(ref_times) < 2:
            audit.append({"fold": current_fold, "status": "identity_insufficient_prior_oof", "validation_rows": int(current.sum()), "reference_rows": int(reference.sum()), "validation_start_utc": start})
            continue
        split_at = pd.Timestamp(ref_times[max(1, min(len(ref_times) - 1, int(np.floor(len(ref_times) * 0.60))))])
        side_fit = reference & decision.lt(split_at).to_numpy() & resolved.lt(split_at).to_numpy()
        anchor = reference & decision.ge(split_at).to_numpy() & resolved.lt(start).to_numpy()
        if int(side_fit.sum()) < min_rows or int(anchor.sum()) < min_rows:
            audit.append({"fold": current_fold, "status": "identity_insufficient_disjoint_prior_oof", "validation_rows": int(current.sum()), "reference_rows": int(reference.sum()), "side_fit_rows": int(side_fit.sum()), "anchor_rows": int(anchor.sum()), "validation_start_utc": start})
            continue
        global_map = fit_weighted_isotonic(direct[side_fit], target[side_fit], _weights(decision, side_fit, start, arm.half_life_days), min_rows=min_rows)
        current_positions, anchor_positions = np.flatnonzero(current), np.flatnonzero(anchor)
        current_side_mapped, anchor_side_mapped = direct[current].copy(), direct[anchor].copy()
        side_report: dict[str, Any] = {}
        for side_name in ("long", "short"):
            fit = side_fit & (side == side_name)
            mapper = fit_weighted_isotonic(direct[fit], target[fit], _weights(decision, fit, start, arm.half_life_days), min_rows=min_rows)
            for positions, values in ((current_positions, current_side_mapped), (anchor_positions, anchor_side_mapped)):
                local = side[positions] == side_name
                if local.any():
                    raw = direct[positions[local]]
                    values[local] = float(arm.side_weight) * mapper.predict(raw) + (1.0 - float(arm.side_weight)) * global_map.predict(raw)
            side_report[side_name] = {"rows": int(fit.sum()), "status": mapper.status}
        anchor_map = fit_weighted_isotonic(anchor_side_mapped, target[anchor], _weights(decision, anchor, start, arm.half_life_days), min_rows=min_rows)
        anchored = anchor_map.predict(current_side_mapped)
        out[current] = (1.0 - float(arm.identity_weight)) * anchored + float(arm.identity_weight) * direct[current]
        audit.append({
            "fold": current_fold, "status": anchor_map.status, "window_status": window_status,
            "validation_rows": int(current.sum()), "reference_rows": int(reference.sum()),
            "side_fit_rows": int(side_fit.sum()), "anchor_rows": int(anchor.sum()),
            "side_fit_max_resolution_utc": resolved.loc[side_fit].max(), "anchor_start_utc": split_at,
            "anchor_max_resolution_utc": resolved.loc[anchor].max(), "validation_start_utc": start,
            "global_map": global_map.status, "side_maps": side_report,
        })
    return out, audit


def _fit_predict_regressor(train_x: pd.DataFrame, train_y: np.ndarray, eval_x: pd.DataFrame, *, seed: int) -> np.ndarray:
    medians = train_x.median(axis=0).fillna(0.0)
    x_train = train_x.fillna(medians).replace([np.inf, -np.inf], 0.0)
    x_eval = eval_x.fillna(medians).replace([np.inf, -np.inf], 0.0)
    # A regularized linear head is intentional here: it is the low-variance
    # baseline for a mapping-change correction and avoids turning one short
    # regime episode into a high-capacity post-model fit.  The runner's
    # artifact makes this explicit so a nonlinear head can be compared later
    # on identical strict-OOF inputs.
    del seed
    model = make_pipeline(StandardScaler(), Ridge(alpha=8.0)).fit(x_train, train_y)
    predicted = np.asarray(model.predict(x_eval), dtype=float)
    del model
    gc.collect()
    return predicted


def strict_prior_oof_drift_heads(
    frame: pd.DataFrame, features: Sequence[str], score: np.ndarray, target: np.ndarray, folds: np.ndarray, *, min_rows: int, seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Fit side-local mapping-change heads only on earlier resolved OOF rows."""
    outputs = {name: np.zeros(len(frame), dtype=float) for name in ("signed_residual", "overestimate_ev", "downside_ev", "absolute_mapping_error")}
    decision, resolved = frame[DECISION], frame[RESOLUTION]
    side = frame[SIDE].astype(str).str.lower().to_numpy()
    audit: list[dict[str, Any]] = []
    for current_fold in sorted(int(x) for x in np.unique(folds[np.isfinite(folds)])):
        current = (folds == current_fold) & np.isfinite(score)
        if not current.any():
            continue
        start = decision.loc[current].min()
        reference = (folds < current_fold) & resolved.lt(start).to_numpy() & np.isfinite(score) & np.isfinite(target)
        per_side: dict[str, Any] = {}
        for offset, side_name in enumerate(("long", "short")):
            train = reference & (side == side_name)
            valid = current & (side == side_name)
            if not valid.any():
                continue
            if int(train.sum()) < min_rows:
                per_side[side_name] = {"status": "zero_fallback_insufficient_prior_oof", "rows": int(train.sum())}
                continue
            train_x = frame.loc[train, list(features)].apply(pd.to_numeric, errors="coerce")
            eval_x = frame.loc[valid, list(features)].apply(pd.to_numeric, errors="coerce")
            residual = target[train] - score[train]
            targets = {
                "signed_residual": residual,
                # Only positive model optimism is a penalty.  Underestimation
                # is preserved rather than being treated as symmetric noise.
                "overestimate_ev": np.maximum(score[train] - target[train], 0.0),
                "downside_ev": np.maximum(-target[train], 0.0),
                "absolute_mapping_error": np.abs(residual),
            }
            for target_offset, (name, values) in enumerate(targets.items()):
                outputs[name][valid] = _fit_predict_regressor(train_x, values, eval_x, seed=seed + current_fold * 100 + offset * 10 + target_offset)
            per_side[side_name] = {"status": "fit_prior_resolved_outer_oof", "rows": int(train.sum()), "max_resolution_utc": resolved.loc[train].max()}
        audit.append({"fold": current_fold, "validation_start_utc": start, "per_side": per_side})
    return outputs, audit


def global_top_k_metrics(score: np.ndarray, target: np.ndarray, side: np.ndarray, mask: np.ndarray, *, top_fraction: float, eligibility: np.ndarray | None = None, sizing: np.ndarray | None = None) -> dict[str, Any]:
    valid = mask & np.isfinite(score) & np.isfinite(target)
    eligible = valid.copy() if eligibility is None else valid & eligibility
    top_count = max(1, int(np.ceil(int(valid.sum()) * float(top_fraction)))) if valid.any() else 0
    positions = np.flatnonzero(eligible)
    ranked = positions[np.argsort(-score[positions], kind="stable")[:top_count]] if len(positions) else np.empty(0, dtype=int)
    result: dict[str, Any] = {
        "rows": int(eligible.sum()),
        "mae": float(np.mean(np.abs(score[eligible] - target[eligible]))) if eligible.any() else float("nan"),
        "prediction_bias": float(np.mean(score[eligible] - target[eligible])) if eligible.any() else float("nan"),
        "ranking_scope": "one_pooled_global_top_k_after_causal_calibration",
        "global_candidate_rows": int(valid.sum()), "eligible_rows": int(eligible.sum()), "top_k_requested_rows": int(top_count), "top_k_rows": int(len(ranked)),
        "top_k_mean_net_ev": float(target[ranked].mean()) if len(ranked) else float("nan"),
        "top_k_sum_net_ev": float(target[ranked].sum()) if len(ranked) else float("nan"),
        "top_k_long_rows": int((side[ranked] == "long").sum()), "top_k_short_rows": int((side[ranked] == "short").sum()),
    }
    if sizing is not None and len(ranked):
        weights = np.clip(np.asarray(sizing, dtype=float)[ranked], 0.0, 1.0)
        result["top_k_mean_sized_net_ev_per_full_notional"] = float(np.mean(target[ranked] * weights))
        result["top_k_total_size"] = float(weights.sum())
        result["top_k_mean_size"] = float(weights.mean())
    return result


def _slices(frame: pd.DataFrame, shared: np.ndarray) -> dict[str, np.ndarray]:
    decision = frame[DECISION]
    result = {"all_oof": shared}
    month = decision.dt.year.astype(str) + "-" + decision.dt.month.astype(str).str.zfill(2)
    for value in sorted(month.loc[shared].unique()):
        result[f"month_{value}"] = shared & month.eq(value).to_numpy()
    iso = decision.dt.isocalendar()
    week = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    for value in sorted(week.loc[shared].unique()):
        result[f"week_{value}"] = shared & week.eq(value).to_numpy()
    if week.loc[shared].size:
        latest = max(week.loc[shared].unique())
        result["latest_week"] = result[f"week_{latest}"]
    return result


def drift_head_quality(
    score: np.ndarray, target: np.ndarray, heads: Mapping[str, np.ndarray], folds: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """OOF head quality by outer fold; no ranking metric is hidden here."""
    observed = {
        "signed_residual": target - score,
        "overestimate_ev": np.maximum(score - target, 0.0),
        "downside_ev": np.maximum(-target, 0.0),
        "absolute_mapping_error": np.abs(target - score),
    }
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    for fold in sorted(int(x) for x in np.unique(folds[np.isfinite(folds)])):
        mask = folds == fold
        per_head: dict[str, dict[str, float | int]] = {}
        for name, predicted in heads.items():
            valid = mask & np.isfinite(predicted) & np.isfinite(observed[name])
            if not valid.any():
                per_head[name] = {"rows": 0, "mae": float("nan"), "spearman": float("nan")}
                continue
            actual, forecast = observed[name][valid], predicted[valid]
            per_head[name] = {
                "rows": int(valid.sum()), "mae": float(np.mean(np.abs(actual - forecast))),
                "mean_observed": float(np.mean(actual)), "mean_predicted": float(np.mean(forecast)),
                "spearman": float(pd.Series(actual).corr(pd.Series(forecast), method="spearman")),
            }
        report[str(fold)] = per_head
    return report


def score_scale_coverage_audit(
    frame: pd.DataFrame, direct: np.ndarray, target: np.ndarray, folds: np.ndarray,
) -> list[dict[str, Any]]:
    """Expose cross-fold score-scale and side-coverage drift explicitly."""
    result: list[dict[str, Any]] = []
    side = frame[SIDE].astype(str).str.lower().to_numpy()
    for fold in sorted(int(x) for x in np.unique(folds[np.isfinite(folds)])):
        for side_name in ("long", "short", "all"):
            mask = folds == fold
            if side_name != "all":
                mask &= side == side_name
            values, realized = direct[mask], target[mask]
            finite = np.isfinite(values) & np.isfinite(realized)
            if not finite.any():
                continue
            values, realized = values[finite], realized[finite]
            result.append({
                "fold": fold, "side": side_name, "rows": int(len(values)),
                "direct_mean": float(np.mean(values)), "direct_std": float(np.std(values)),
                "direct_p01": float(np.quantile(values, 0.01)), "direct_p50": float(np.quantile(values, 0.50)), "direct_p99": float(np.quantile(values, 0.99)),
                "realized_mean": float(np.mean(realized)), "realized_std": float(np.std(realized)),
                "realized_p01": float(np.quantile(realized, 0.01)), "realized_p50": float(np.quantile(realized, 0.50)), "realized_p99": float(np.quantile(realized, 0.99)),
                "prediction_coverage": float(finite.mean()),
            })
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-calibration-rows", type=int, default=500)
    parser.add_argument("--min-drift-rows", type=int, default=2_000)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frame, context_features = load_panel(args.predictions, args.context)
    print(f"[calibration-drift] loaded rows={len(frame)} context_features={len(context_features)}", flush=True)
    direct = pd.to_numeric(frame[DIRECT], errors="raise").to_numpy(float)
    target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    folds = pd.to_numeric(frame[FOLD], errors="coerce").to_numpy(float)
    side = frame[SIDE].astype(str).str.lower().to_numpy()
    shared = np.isfinite(direct) & np.isfinite(target) & np.isfinite(folds)
    calibrations: dict[str, np.ndarray] = {"direct_raw": direct.copy()}
    calibration_audit: dict[str, Any] = {}
    for arm in CALIBRATION_ARMS:
        mapped, audit = temporal_hierarchical_calibration(frame, direct, target, folds, arm, min_rows=int(args.min_calibration_rows))
        calibrations[arm.name] = mapped
        calibration_audit[arm.name] = audit
    print("[calibration-drift] completed calibration variants", flush=True)
    primary = calibrations[PRIMARY_CALIBRATION]
    drift_feature_names = [DIRECT, *context_features]
    # Avoid any accidental target or resolution fields even when a caller
    # supplies an extended context panel with extra columns.
    drift_feature_names = [c for c in dict.fromkeys(drift_feature_names) if c in frame and c not in {TARGET, RESOLUTION, FOLD}]
    drift, drift_audit = strict_prior_oof_drift_heads(frame, drift_feature_names, primary, target, folds, min_rows=int(args.min_drift_rows), seed=int(args.seed))
    gc.collect()
    print("[calibration-drift] completed strict-prior drift heads", flush=True)
    arms: dict[str, np.ndarray] = {**calibrations}
    arms["primary_plus_signed_drift"] = primary + drift["signed_residual"]
    arms["primary_lcb_overestimate_050"] = primary - 0.50 * drift["overestimate_ev"]
    arms["primary_lcb_overestimate_100"] = primary - drift["overestimate_ev"]
    arms["primary_lcb_downside_050"] = primary - 0.50 * drift["downside_ev"]
    print("[calibration-drift] assembled score arms", flush=True)
    # Gate threshold is learned from strictly earlier, resolved OOF rows per
    # fold.  It only determines eligibility; the retained rows are still one
    # pooled global top-k after calibrated scoring.
    gate = np.ones(len(frame), dtype=bool)
    size = np.ones(len(frame), dtype=float)
    gate_audit: list[dict[str, Any]] = []
    for current_fold in sorted(int(x) for x in np.unique(folds[shared])):
        current = folds == current_fold
        start = frame.loc[current, DECISION].min()
        reference = (folds < current_fold) & frame[RESOLUTION].lt(start).to_numpy() & shared
        if int(reference.sum()) < int(args.min_drift_rows):
            gate_audit.append({"fold": current_fold, "status": "all_eligible_zero_size_penalty", "reference_rows": int(reference.sum())})
            continue
        threshold = float(np.quantile(drift["overestimate_ev"][reference], 0.80))
        scale = float(np.quantile(drift["overestimate_ev"][reference], 0.90))
        scale = max(scale, 1e-8)
        gate[current] = drift["overestimate_ev"][current] <= threshold
        size[current] = np.clip(1.0 - drift["overestimate_ev"][current] / scale, 0.25, 1.0)
        gate_audit.append({"fold": current_fold, "status": "prior_oof_quantile", "reference_rows": int(reference.sum()), "threshold": threshold, "scale_q90": scale, "reference_max_resolution_utc": frame.loc[reference, RESOLUTION].max(), "validation_start_utc": start})
    print("[calibration-drift] completed abstention/sizing inputs", flush=True)
    print("[calibration-drift] constructing reporting slices", flush=True)
    slices = _slices(frame, shared)
    print(f"[calibration-drift] constructed reporting slices={len(slices)}", flush=True)
    metrics: dict[str, Any] = {}
    for scope, mask in slices.items():
        print(f"[calibration-drift] metrics scope={scope}", flush=True)
        metrics[scope] = {
            name: global_top_k_metrics(values, target, side, mask, top_fraction=float(args.top_fraction))
            for name, values in arms.items()
        }
        metrics[scope]["primary_lcb_overestimate_100_abstention"] = global_top_k_metrics(
            arms["primary_lcb_overestimate_100"], target, side, mask, top_fraction=float(args.top_fraction), eligibility=gate,
        )
        metrics[scope]["primary_sizing_overestimate"] = global_top_k_metrics(
            primary, target, side, mask, top_fraction=float(args.top_fraction), sizing=size,
        )
    print("[calibration-drift] completed metrics", flush=True)
    output = frame.loc[:, [*IDENTITY, DECISION, RESOLUTION, TARGET, FOLD]].copy()
    output[DIRECT] = direct
    for name, values in arms.items():
        output[name] = values
    for name, values in drift.items():
        output[f"drift__{name}"] = values
    output["overestimate_abstention_eligible"] = gate.astype(np.int8)
    output["overestimate_size_fraction"] = size.astype(np.float32)
    output.to_parquet(args.output_dir / "oof_predictions.parquet", index=False, compression="zstd")
    print("[calibration-drift] wrote prediction panel", flush=True)
    summary = {
        "schema": SCHEMA,
        "status": "strict_prior_outer_oof_diagnostic_not_promoted",
        "contract": {
            "target": "execution_net_ev_12h, already cost-adjusted; no further cost subtraction",
            "units": "fractional net return / common EV units",
            "calibration": "side monotonic map, disjoint later pooled anchor, identity shrinkage; fit only earlier resolved outer-OOF rows",
            "drift": "side-local regularized-linear signed residual, expected positive overestimation, downside EV and absolute mapping error; strict earlier resolved OOF only",
            "uncertainty": "asymmetric: penalise expected model optimism or downside, not absolute error",
            "ranking": "one pooled global top-k after causal calibrated score; no per-timestamp and no per-side quota",
            "extended_history": "pass compatible extended strict-OOF prediction/context panels; no date-specific code path",
        },
        "rows": int(len(frame)), "shared_outer_oof_rows": int(shared.sum()), "drift_features": drift_feature_names,
        "calibration_arms": [arm.__dict__ for arm in CALIBRATION_ARMS], "primary_calibration": PRIMARY_CALIBRATION,
        "calibration_audit": calibration_audit, "drift_audit": drift_audit,
        "drift_head_quality_by_outer_fold": drift_head_quality(primary, target, drift, folds),
        "direct_score_scale_and_coverage_by_outer_fold": score_scale_coverage_audit(frame, direct, target, folds),
        "abstention_and_sizing_audit": gate_audit,
        "metrics": metrics,
        "sources": {
            "predictions": {"path": str(args.predictions), "sha256": _sha256(args.predictions)},
            "context": {"path": str(args.context), "sha256": _sha256(args.context)},
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary["metrics"].get("all_oof", {})), indent=2))


if __name__ == "__main__":
    main()
