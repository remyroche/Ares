"""Strict chronological broad-to-tail base ablations (Stage IV).

Stage IV deliberately separates *ranking* from *tail refinement*.  A broad
same-side model first emits a strict OOF score.  Only rows that would have
cleared a **prior-score-only**, global-in-time top-x handoff are eligible for a
second same-side tail model.  The meta/residual model in turn consumes only
strict OOF tail outputs.  No handoff is selected within a timestamp and no
in-sample predecessor score is substituted during a burn-in.

The module is an experiment harness, not a scheduler: callers materialise the
feature frame and choose frozen model parameters.  It intentionally writes no
artifacts and starts no large run by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)


SCHEMA = "stage_iv_broad_to_tail_strict_oof_v1"
TAIL_FRACTIONS = frozenset({0.20, 0.30, 0.40, 0.50})
_ROUTES = frozenset({"neither", "tail", "meta", "both"})
_TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


class _Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> Any: ...


ModelFitter = Callable[[pd.DataFrame, np.ndarray, np.ndarray, str, Mapping[str, Any]], _Predictor]


@dataclass(frozen=True)
class StageIVPlan:
    """One side's frozen broad -> tail -> residual contract.

    Burn-ins are measured in *prior resolved rows*, not calendar rows.  This
    makes each layer independently explicit when the candidate density varies.
    ``tail_fraction`` is a global-in-time handoff fraction (not a per-timestamp
    quota) and is intentionally limited to the four approved ablation values.
    """

    side: str
    candidate_ids: Sequence[Any]
    frame: pd.DataFrame
    base_target: Sequence[float]
    exact_net_bps: Sequence[float]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    broad_feature_names: Sequence[str]
    tail_feature_names: Sequence[str]
    meta_feature_names: Sequence[str]
    broad_params: Mapping[str, Any] = field(default_factory=dict)
    tail_params: Mapping[str, Any] = field(default_factory=dict)
    meta_params: Mapping[str, Any] = field(default_factory=dict)
    tail_fraction: float = 0.30
    broad_min_train_rows: int = 500
    tail_min_train_rows: int = 500
    meta_min_train_rows: int = 500
    min_handoff_history_rows: int = 100
    n_validation_folds: int = 4
    broad_output_route: str = "both"
    sample_weight: Sequence[float] | None = None
    meta_target: Sequence[float] | None = None
    cost_bps: float = 100.0


@dataclass(frozen=True)
class StageIVSideResult:
    side: str
    predictions: pd.DataFrame
    fold_provenance: pd.DataFrame
    plan_summary: Mapping[str, Any]


@dataclass(frozen=True)
class StageIVResult:
    """Combined sides plus pooled-global-only evaluation tables."""

    side_results: Mapping[str, StageIVSideResult]
    predictions: pd.DataFrame
    metrics_without_admission: pd.DataFrame
    metrics_with_admission: pd.DataFrame
    admitted_predictions: pd.DataFrame | None
    manifest: Mapping[str, Any]


def _utc(values: Sequence[Any], *, name: str) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} contains missing or non-UTC-convertible values")
    return result


def _vector(values: Sequence[Any], n: int, *, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).reshape(-1)
    if len(result) != n:
        raise ValueError(f"{name} must be row-aligned to frame")
    return result


def _features(values: Sequence[str], frame: pd.DataFrame, *, name: str) -> list[str]:
    names = list(dict.fromkeys(str(value) for value in values))
    if not names:
        raise ValueError(f"{name} must be non-empty")
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is absent from frame: {missing[:12]}")
    return names


def _validate_plan(plan: StageIVPlan) -> tuple[pd.DataFrame, dict[str, Any]]:
    side = str(plan.side).lower()
    if side not in {"long", "short"}:
        raise ValueError("Stage IV plans must be independently side=long or side=short")
    if float(plan.tail_fraction) not in TAIL_FRACTIONS:
        raise ValueError("tail_fraction must be one of 20%, 30%, 40%, or 50%")
    route = str(plan.broad_output_route).lower()
    if route not in _ROUTES:
        raise ValueError("broad_output_route must be neither, tail, meta, or both")
    if int(plan.n_validation_folds) < 1:
        raise ValueError("n_validation_folds must be positive")
    for name, value in {
        "broad_min_train_rows": plan.broad_min_train_rows,
        "tail_min_train_rows": plan.tail_min_train_rows,
        "meta_min_train_rows": plan.meta_min_train_rows,
        "min_handoff_history_rows": plan.min_handoff_history_rows,
    }.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if not np.isfinite(float(plan.cost_bps)):
        raise ValueError("cost_bps must be finite")
    frame = plan.frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    n = len(frame)
    if n < 2:
        raise ValueError("Stage IV requires at least two rows")
    ids = _vector(plan.candidate_ids, n, name="candidate_ids", dtype=object)
    if pd.isna(ids).any() or len(pd.unique(ids)) != n:
        raise ValueError("candidate_ids must be non-null and unique within a side")
    target = _vector(plan.base_target, n, name="base_target", dtype=np.float32)
    net = _vector(plan.exact_net_bps, n, name="exact_net_bps", dtype=np.float32)
    if not np.isfinite(target).all() or not np.isfinite(net).all():
        raise ValueError("base_target and exact_net_bps must be finite")
    decision = _utc(plan.decision_timestamps, name="decision_timestamps")
    available = _utc(plan.label_available_timestamps, name="label_available_timestamps")
    if len(decision) != n or len(available) != n or (available <= decision).any():
        raise ValueError("labels must resolve strictly after their decision timestamps")
    if plan.sample_weight is None:
        weight = np.ones(n, dtype=np.float32)
    else:
        weight = _vector(plan.sample_weight, n, name="sample_weight", dtype=np.float32)
        if not np.isfinite(weight).all() or (weight < 0.0).any() or float(weight.sum()) <= 0.0:
            raise ValueError("sample_weight must be finite, non-negative and non-empty")
    if plan.meta_target is None:
        meta_target = None
    else:
        meta_target = _vector(plan.meta_target, n, name="meta_target", dtype=np.float32)
        if not np.isfinite(meta_target).all():
            raise ValueError("meta_target must be finite")
    return frame, {
        "side": side,
        "route": route,
        "ids": ids,
        "target": target,
        "net": net,
        "decision": decision,
        "available": available,
        "weight": weight,
        "meta_target": meta_target,
        "broad_features": _features(plan.broad_feature_names, frame, name="broad_feature_names"),
        "tail_features": _features(plan.tail_feature_names, frame, name="tail_feature_names"),
        "meta_features": _features(plan.meta_feature_names, frame, name="meta_feature_names"),
    }


def _groups_at_timestamps(indices: np.ndarray, decision: pd.Series) -> list[np.ndarray]:
    if not len(indices):
        return []
    order = indices[np.argsort(decision.iloc[indices].to_numpy(dtype="datetime64[ns]"), kind="stable")]
    timestamp = decision.iloc[order].to_numpy(dtype="datetime64[ns]")
    starts = np.r_[0, np.flatnonzero(np.diff(timestamp)) + 1]
    ends = np.r_[starts[1:], len(order)]
    return [order[start:end] for start, end in zip(starts, ends)]


def _strict_blocks(
    *,
    candidate_mask: np.ndarray,
    trainable_mask: np.ndarray,
    decision: pd.Series,
    available: pd.Series,
    min_train_rows: int,
    n_folds: int,
) -> list[np.ndarray]:
    """Whole-timestamp blocks, after an independent prior-resolved burn-in."""
    candidates = np.flatnonzero(candidate_mask)
    groups = _groups_at_timestamps(candidates, decision)
    eligible_from = len(groups)
    for group_index, group in enumerate(groups):
        start = decision.iloc[group].min()
        train = trainable_mask & decision.lt(start).to_numpy() & available.lt(start).to_numpy()
        if int(train.sum()) >= int(min_train_rows):
            eligible_from = group_index
            break
    remaining = groups[eligible_from:]
    if not remaining:
        return []
    blocks = np.array_split(np.arange(len(remaining), dtype=np.int32), min(int(n_folds), len(remaining)))
    return [
        np.concatenate([remaining[int(group_index)] for group_index in block]).astype(np.int32, copy=False)
        for block in blocks if len(block)
    ]


def _predict(model: _Predictor, X: pd.DataFrame, *, layer: str) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise TypeError(f"{layer} model must expose predict")
    result = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
    if len(result) != len(X) or not np.isfinite(result).all():
        raise ValueError(f"{layer} model produced non-finite/misaligned direct scores")
    return result


def _default_lgbm_fitter(
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    layer: str,
    params: Mapping[str, Any],
) -> _Predictor:
    """Lazy default for numeric base and residual targets.

    Callers may pass another fitter (for example a multiclass-to-bps wrapper),
    but this default keeps Stage IV usable with the repository's frozen LGBM
    parameter dictionaries.
    """
    from .lgbm_pipeline import _fit_lgbm_model

    frozen = dict(params)
    frozen.setdefault("objective", "huber" if layer == "meta" else "regression")
    return _fit_lgbm_model(
        X, y, weight, classifier=False, params=frozen,
        objective_mode=f"stage_iv_{layer}",
    )


def prequential_tail_handoff(
    broad_score: Sequence[float],
    decision_timestamps: Sequence[Any],
    *,
    tail_fraction: float,
    min_history_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal global-score thresholds and top-x membership.

    Every row at a decision timestamp sees the same threshold calculated from
    **earlier timestamp groups only**.  It is deliberately not a per-bar or
    per-timestamp percentile.  ``NaN`` broad scores remain ineligible.
    """
    if float(tail_fraction) not in TAIL_FRACTIONS:
        raise ValueError("tail_fraction must be one of the four approved values")
    if int(min_history_rows) < 1:
        raise ValueError("min_history_rows must be positive")
    score = np.asarray(broad_score, dtype=np.float32).reshape(-1)
    decision = _utc(decision_timestamps, name="decision_timestamps")
    if len(score) != len(decision):
        raise ValueError("broad_score and decision_timestamps must align")
    threshold = np.full(len(score), np.nan, dtype=np.float32)
    selected = np.zeros(len(score), dtype=bool)
    history: list[float] = []
    for group in _groups_at_timestamps(np.arange(len(score), dtype=np.int32), decision):
        finite = np.isfinite(score[group])
        if len(history) >= int(min_history_rows):
            q = float(np.quantile(np.asarray(history, dtype=np.float32), 1.0 - float(tail_fraction)))
            threshold[group] = q
            selected[group[finite]] = score[group[finite]] >= q
        history.extend(score[group][finite].astype(float).tolist())
    return threshold, selected


def _design(frame: pd.DataFrame, features: Sequence[str], extra: Mapping[str, np.ndarray]) -> pd.DataFrame:
    overlap = sorted(set(features) & set(extra))
    if overlap:
        raise ValueError(f"frozen raw feature list collides with Stage IV handoff fields: {overlap}")
    out = frame.loc[:, list(features)].copy()
    for name, values in extra.items():
        out[name] = np.asarray(values, dtype=np.float32)
    return out


def _fit_stage(
    *,
    layer: str,
    design: pd.DataFrame,
    target: np.ndarray,
    weight: np.ndarray,
    candidate_mask: np.ndarray,
    trainable_mask: np.ndarray,
    decision: pd.Series,
    available: pd.Series,
    min_train_rows: int,
    n_folds: int,
    params: Mapping[str, Any],
    fitter: ModelFitter,
    output: np.ndarray,
    fold_output: np.ndarray,
    provenance: list[dict[str, Any]],
) -> None:
    blocks = _strict_blocks(
        candidate_mask=candidate_mask, trainable_mask=trainable_mask,
        decision=decision, available=available, min_train_rows=min_train_rows,
        n_folds=n_folds,
    )
    for fold_id, validation_idx in enumerate(blocks):
        start = decision.iloc[validation_idx].min()
        train_mask = trainable_mask & decision.lt(start).to_numpy() & available.lt(start).to_numpy()
        train_idx = np.flatnonzero(train_mask)
        # _strict_blocks made this true, but retaining the check prevents a
        # future refactor from silently weakening a handoff.
        if len(train_idx) < int(min_train_rows):
            raise AssertionError(f"{layer} fold bypassed its independent burn-in")
        if not available.iloc[train_idx].lt(start).all() or not decision.iloc[train_idx].lt(start).all():
            raise AssertionError(f"{layer} fold contains non-prior training information")
        model = fitter(design.iloc[train_idx], target[train_idx], weight[train_idx], layer, params)
        output[validation_idx] = _predict(model, design.iloc[validation_idx], layer=layer)
        fold_output[validation_idx] = int(fold_id)
        provenance.append({
            "side": None, "layer": layer, "fold_id": int(fold_id),
            "train_rows": int(len(train_idx)), "validation_rows": int(len(validation_idx)),
            "validation_start_ts": start,
            "validation_end_ts": decision.iloc[validation_idx].max(),
            "train_max_label_available_ts": available.iloc[train_idx].max(),
            "strict_prior_resolved": True,
            "strict_predecessor_oof": bool(layer == "broad" or trainable_mask[train_idx].all()),
        })


def generate_stage_iv_side_oof(plan: StageIVPlan, *, fitter: ModelFitter | None = None) -> StageIVSideResult:
    """Produce broad, tail and residual direct OOF scores for one side only."""
    frame, values = _validate_plan(plan)
    fit = fitter or _default_lgbm_fitter
    n = len(frame)
    decision = values["decision"]
    available = values["available"]
    assert isinstance(decision, pd.Series) and isinstance(available, pd.Series)
    side = str(values["side"])
    route = str(values["route"])
    target = values["target"]
    net = values["net"]
    weight = values["weight"]
    assert isinstance(target, np.ndarray) and isinstance(net, np.ndarray) and isinstance(weight, np.ndarray)
    broad = np.full(n, np.nan, dtype=np.float32)
    tail = np.full(n, np.nan, dtype=np.float32)
    meta = np.full(n, np.nan, dtype=np.float32)
    broad_fold = np.full(n, -1, dtype=np.int16)
    tail_fold = np.full(n, -1, dtype=np.int16)
    meta_fold = np.full(n, -1, dtype=np.int16)
    provenance: list[dict[str, Any]] = []

    broad_design = _design(frame, values["broad_features"], {})
    _fit_stage(
        layer="broad", design=broad_design, target=target, weight=weight,
        candidate_mask=np.ones(n, dtype=bool), trainable_mask=np.ones(n, dtype=bool),
        decision=decision, available=available, min_train_rows=int(plan.broad_min_train_rows),
        n_folds=int(plan.n_validation_folds), params=plan.broad_params, fitter=fit,
        output=broad, fold_output=broad_fold, provenance=provenance,
    )
    broad_scored = np.isfinite(broad)
    threshold, tail_eligible = prequential_tail_handoff(
        broad, decision, tail_fraction=float(plan.tail_fraction),
        min_history_rows=int(plan.min_handoff_history_rows),
    )
    # A handoff only exists where the broad predecessor is itself strict OOF.
    tail_eligible &= broad_scored
    tail_extra: dict[str, np.ndarray] = {}
    if route in {"tail", "both"}:
        tail_extra["__stage_iv_broad_same_side_oof_score"] = broad
    tail_design = _design(frame, values["tail_features"], tail_extra)
    _fit_stage(
        layer="tail", design=tail_design, target=target, weight=weight,
        candidate_mask=tail_eligible, trainable_mask=tail_eligible,
        decision=decision, available=available, min_train_rows=int(plan.tail_min_train_rows),
        n_folds=int(plan.n_validation_folds), params=plan.tail_params, fitter=fit,
        output=tail, fold_output=tail_fold, provenance=provenance,
    )
    tail_scored = np.isfinite(tail)
    # The residual target is deliberately per-row and same-side.  It is never
    # constructed from period aggregates or converted predecessor scores.
    supplied_meta_target = values["meta_target"]
    if supplied_meta_target is None:
        meta_target = net - tail
    else:
        assert isinstance(supplied_meta_target, np.ndarray)
        meta_target = supplied_meta_target
    meta_extra: dict[str, np.ndarray] = {"__stage_iv_tail_same_side_oof_score": tail}
    if route in {"meta", "both"}:
        meta_extra["__stage_iv_broad_same_side_oof_score"] = broad
    meta_design = _design(frame, values["meta_features"], meta_extra)
    _fit_stage(
        layer="meta", design=meta_design, target=np.asarray(meta_target, dtype=np.float32), weight=weight,
        candidate_mask=tail_scored, trainable_mask=tail_scored,
        decision=decision, available=available, min_train_rows=int(plan.meta_min_train_rows),
        n_folds=int(plan.n_validation_folds), params=plan.meta_params, fitter=fit,
        output=meta, fold_output=meta_fold, provenance=provenance,
    )
    meta_scored = np.isfinite(meta)
    final_score = tail + meta
    ids = values["ids"]
    assert isinstance(ids, np.ndarray)
    output = pd.DataFrame({
        "candidate_id": ids,
        "candidate_key": [f"{side}::{value}" for value in ids],
        "side_name": side,
        "decision_ts": decision,
        "label_available_ts": available,
        "exact_net_bps": net,
        "cost_bps": np.float32(plan.cost_bps),
        "broad_same_side_oof_score": broad,
        "broad_handoff_threshold": threshold,
        "tail_prequentially_eligible": tail_eligible,
        "tail_same_side_oof_score": tail,
        "meta_same_side_residual_oof_score": meta,
        "meta_reconstructed_expected_net_bps": final_score,
        "broad_strict_oof_available": broad_scored,
        "tail_strict_oof_available": tail_scored,
        "meta_strict_oof_available": meta_scored,
        "broad_fold_id": broad_fold,
        "tail_fold_id": tail_fold,
        "meta_fold_id": meta_fold,
    })
    for row in provenance:
        row["side"] = side
    provenance_frame = pd.DataFrame(provenance)
    if not provenance_frame.empty:
        starts = pd.to_datetime(provenance_frame.validation_start_ts, utc=True)
        maxima = pd.to_datetime(provenance_frame.train_max_label_available_ts, utc=True)
        if not (maxima < starts).all() or not provenance_frame.strict_prior_resolved.astype(bool).all():
            raise AssertionError("Stage IV provenance contains a non-causal fold")
    return StageIVSideResult(
        side=side, predictions=output, fold_provenance=provenance_frame,
        plan_summary={
            "schema": SCHEMA, "side": side, "tail_fraction": float(plan.tail_fraction),
            "broad_output_route": route,
            "burn_in_prior_resolved_rows": {
                "broad": int(plan.broad_min_train_rows), "tail": int(plan.tail_min_train_rows),
                "meta": int(plan.meta_min_train_rows), "handoff_score_history": int(plan.min_handoff_history_rows),
            },
            "handoff": "prior-score-only global-in-time top-x; never per timestamp",
            "same_side_direct_handoffs": True,
            "feature_counts": {
                "broad": len(values["broad_features"]), "tail": len(values["tail_features"]),
                "meta": len(values["meta_features"]),
            },
        },
    )


def pooled_global_stage_iv_metrics(
    ledger: pd.DataFrame,
    *,
    score_column: str,
    layer: str,
    top_fractions: Sequence[float] = _TOP_FRACTIONS,
) -> pd.DataFrame:
    """Rank once across the complete (both-side) OOF ledger.

    Monthly and side rows are only attributions of that one selected book; no
    timestamp, month, or side reranking occurs in this function.
    """
    required = {"candidate_key", "side_name", "decision_ts", "exact_net_bps", "cost_bps", score_column}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Stage IV metrics missing columns: {missing}")
    work = ledger.loc[ledger[score_column].notna()].copy()
    if work.empty or work.candidate_key.duplicated().any():
        raise ValueError("Stage IV metrics require non-empty globally unique scored rows")
    work["month"] = pd.to_datetime(work.decision_ts, utc=True).dt.strftime("%Y-%m")
    ordered = work.sort_values([score_column, "candidate_key"], ascending=[False, True], kind="stable")
    rows: list[dict[str, Any]] = []
    for fraction in top_fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("top fractions must lie in (0, 1]")
        selected = ordered.head(max(1, int(np.ceil(float(fraction) * len(ordered)))))
        common = {
            "layer": layer,
            "selection": "pooled_global_once_no_timestamp_month_or_side_rerank",
            "top_fraction": float(fraction),
            "candidate_rows": int(len(ordered)),
            "selected_global_rows": int(len(selected)),
        }
        rows.append({
            **common, "scope": "pooled_global", "month": "__all__", "side": "__all__",
            "selected_rows": int(len(selected)),
            "net_bps_per_trade": float(selected.exact_net_bps.mean()),
            "gross_bps_per_trade": float((selected.exact_net_bps + selected.cost_bps).mean()),
        })
        for (month, side), group in selected.groupby(["month", "side_name"], sort=True):
            rows.append({
                **common, "scope": "selected_contribution", "month": str(month), "side": str(side),
                "selected_rows": int(len(group)),
                "net_bps_per_trade": float(group.exact_net_bps.mean()),
                "gross_bps_per_trade": float((group.exact_net_bps + group.cost_bps).mean()),
            })
    return pd.DataFrame(rows)


def run_stage_iv_broad_to_tail_ablation(
    plans: Sequence[StageIVPlan],
    *,
    fitter: ModelFitter | None = None,
    admission_spec: Causal21dAdmissionSpec | None = None,
) -> StageIVResult:
    """Run a bounded same-contract Stage-IV cell for one or both sides.

    The function does not choose a winning ``x`` or burn-in.  A caller runs one
    immutable cell per explicitly declared configuration, preventing the
    framework from becoming an accidental broad factorial search.
    """
    if not plans:
        raise ValueError("Stage IV needs at least one side plan")
    results = [generate_stage_iv_side_oof(plan, fitter=fitter) for plan in plans]
    sides = [result.side for result in results]
    if len(set(sides)) != len(sides):
        raise ValueError("Stage IV accepts at most one plan per side in a pooled cell")
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    if predictions.candidate_key.duplicated().any():
        raise ValueError("Stage IV sides do not form a globally unique candidate population")
    metric_specs = {
        "base_broad": "broad_same_side_oof_score",
        "base_tail": "tail_same_side_oof_score",
        "meta_residual_reconstructed": "meta_reconstructed_expected_net_bps",
    }
    raw_metrics = pd.concat(
        [pooled_global_stage_iv_metrics(predictions, score_column=column, layer=layer)
         for layer, column in metric_specs.items() if predictions[column].notna().any()],
        ignore_index=True,
    )
    admitted: pd.DataFrame | None = None
    admitted_metrics = pd.DataFrame()
    if admission_spec is not None:
        admission_input = predictions.loc[predictions.meta_strict_oof_available].copy()
        if not admission_input.empty:
            admission_input = admission_input.rename(columns={"exact_net_bps": "net_bps"})
            admitted, _ = apply_causal_21d_side_admission(
                admission_input,
                score_column="meta_reconstructed_expected_net_bps", net_column="net_bps",
                decision_column="decision_ts", label_available_column="label_available_ts",
                identity_column="candidate_key", spec=admission_spec,
            )
            admitted = admitted.rename(columns={"net_bps": "exact_net_bps"})
            admitted = admitted.loc[admitted.causal_21d_side_admitted_ge_50bps.astype(bool)].copy()
            if not admitted.empty:
                admitted_metrics = pooled_global_stage_iv_metrics(
                    admitted, score_column="meta_reconstructed_expected_net_bps",
                    layer="meta_residual_reconstructed_after_causal_21d_admission",
                )
    manifest = {
        "schema": SCHEMA,
        "sides": sides,
        "rows": {"input": int(len(predictions)), "meta_strict_oof": int(predictions.meta_strict_oof_available.sum()), "admitted": int(len(admitted)) if admitted is not None else None},
        "ranking": "pooled global after same-side direct scores; never per timestamp, month, or side",
        "admission": "optional side-local prior-resolved 21-day map, applied before one pooled-global rank" if admission_spec else "not requested",
        "contracts": [dict(result.plan_summary) for result in results],
    }
    return StageIVResult(
        side_results={result.side: result for result in results}, predictions=predictions,
        metrics_without_admission=raw_metrics, metrics_with_admission=admitted_metrics,
        admitted_predictions=admitted, manifest=manifest,
    )


__all__ = [
    "SCHEMA", "TAIL_FRACTIONS", "StageIVPlan", "StageIVSideResult", "StageIVResult",
    "ModelFitter", "prequential_tail_handoff", "generate_stage_iv_side_oof",
    "pooled_global_stage_iv_metrics", "run_stage_iv_broad_to_tail_ablation",
]
