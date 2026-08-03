"""Production strict chronological OOF generation for the Stage-I stack.

The selector deliberately remains separate from this module.  Once a selected
feature list and its frozen LightGBM parameters are available, this module
creates the only admissible hand-off population:

``R3 multiclass OOF -> prior-resolved same-side bps map -> residual OOF``.

It never fills the initial burn-in rows with in-sample predictions and never
uses a shuffled split.  A caller may therefore write the resulting narrow
ledger as an immutable research artifact without reconstituting a wide feature
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .lgbm_pipeline import _fit_lgbm_model
from .prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)
from .stage_i_feature_selection import (
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    validate_stage_i_label_availability,
)


SCHEMA = "stage_i_strict_chronological_oof_v1"
_SIDES = frozenset({"long", "short"})
_BASE_HANDOFF_FEATURES = STAGE_I_META_BASE_OOF_HANDOFF_FEATURES


@dataclass(frozen=True)
class StageIStrictOOFPlan:
    """One same-side, narrow selected-feature OOF contract.

    ``decision_timestamps`` are the actual decision times, not signal-close
    times.  ``label_available_timestamps`` must be the time at which the exact
    H12 outcome is fully resolved.  Feature names are explicit so an accidental
    use of all columns cannot broaden a frozen selector contract.
    """

    side: str
    candidate_ids: Sequence[Any]
    frame: pd.DataFrame
    r3_target: Sequence[int]
    exact_net_bps: Sequence[float]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    base_feature_names: Sequence[str]
    meta_feature_names: Sequence[str]
    base_params: Mapping[str, Any]
    residual_params: Mapping[str, Any]
    exact_gross_bps: Sequence[float] | None = None
    sample_weight: Sequence[float] | None = None
    n_validation_folds: int = 4
    min_train_rows: int = 500
    value_map: PrequentialR3ValueMapConfig | None = None


@dataclass(frozen=True)
class StageIStrictOOFResult:
    """Narrow OOF ledger and immutable fold lineage for one side."""

    side: str
    predictions: pd.DataFrame
    fold_provenance: pd.DataFrame
    value_map_provenance: Mapping[str, Any]
    plan_summary: Mapping[str, Any] = field(default_factory=dict)


def _utc(values: Sequence[Any], *, name: str) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} contains missing or non-UTC-convertible values")
    return result


def _as_vector(values: Sequence[Any], n: int, *, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).reshape(-1)
    if len(result) != n:
        raise ValueError(f"{name} must be aligned to the selected-feature frame")
    return result


def _explicit_boolean_array(values: Sequence[Any], *, label: str) -> np.ndarray:
    """Reject truthy strings, NaNs and non-0/1 provenance values."""
    result: list[bool] = []
    for value in pd.Series(values).tolist():
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and float(value) in {0.0, 1.0}:
            result.append(bool(int(value)))
        else:
            raise ValueError(f"{label} must contain only explicit boolean/0/1 values")
    return np.asarray(result, dtype=bool)


def _validate_plan(plan: StageIStrictOOFPlan) -> tuple[pd.DataFrame, dict[str, np.ndarray | pd.Series]]:
    side = str(plan.side).lower()
    if side not in _SIDES:
        raise ValueError("Stage-I strict OOF must be independently side=long or side=short")
    frame = plan.frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    n = len(frame)
    if n < 2:
        raise ValueError("Stage-I strict OOF requires at least two rows")
    base_features = list(dict.fromkeys(map(str, plan.base_feature_names)))
    meta_features = list(dict.fromkeys(map(str, plan.meta_feature_names)))
    if not base_features or not meta_features:
        raise ValueError("Stage-I strict OOF requires non-empty frozen base and meta feature lists")
    # The handoff features are generated from the strict same-side base OOF
    # model below; every other selected meta feature must already be in the
    # narrow raw/context frame.
    missing = sorted(
        (set(base_features) | (set(meta_features) - set(_BASE_HANDOFF_FEATURES)))
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"selected Stage-I features are absent from narrow frame: {missing[:12]}")
    missing_handoff = [
        feature for feature in _BASE_HANDOFF_FEATURES if feature not in meta_features
    ]
    if missing_handoff:
        raise ValueError(
            "strict residual OOF requires the frozen selected meta feature contract "
            f"to include every direct same-side base handoff: {missing_handoff}"
        )
    ids = _as_vector(plan.candidate_ids, n, name="candidate_ids", dtype=object)
    if pd.isna(ids).any() or len(pd.unique(ids)) != n:
        raise ValueError("candidate_ids must be non-null and unique within a side")
    r3 = _as_vector(plan.r3_target, n, name="r3_target", dtype=np.int8)
    if not np.isin(r3, [0, 1, 2]).all():
        raise ValueError("R3 target must contain exactly classes 0=adverse, 1=weak, 2=clear")
    net = _as_vector(plan.exact_net_bps, n, name="exact_net_bps", dtype=np.float32)
    if not np.isfinite(net).all():
        raise ValueError("exact_net_bps must be finite for the strict OOF population")
    gross = (
        net + np.float32(100.0)
        if plan.exact_gross_bps is None
        else _as_vector(plan.exact_gross_bps, n, name="exact_gross_bps", dtype=np.float32)
    )
    if not np.isfinite(gross).all() or not np.allclose(gross - 100.0, net, rtol=0.0, atol=2e-3):
        raise ValueError("strict OOF requires exact gross - 100bps = exact net")
    decision = _utc(plan.decision_timestamps, name="decision_timestamps")
    available = _utc(plan.label_available_timestamps, name="label_available_timestamps")
    if len(decision) != n or len(available) != n or (available <= decision).any():
        raise ValueError("labels must resolve strictly after an aligned decision timestamp")
    # Stage I's R3/TP6-SL4 H12 outcomes are available at the executable
    # signal-close convention plus one hour of entry delay and twelve hours of
    # path resolution.  Do not let a generic strict-OOF caller silently turn
    # this into a shorter (or longer) target contract.
    validate_stage_i_label_availability(decision, available)
    if int(plan.n_validation_folds) < 1 or int(plan.min_train_rows) < 3:
        raise ValueError("strict OOF requires at least one validation fold and three training rows")
    if plan.sample_weight is None:
        weight = np.ones(n, dtype=np.float32)
    else:
        weight = _as_vector(plan.sample_weight, n, name="sample_weight", dtype=np.float32)
        if not np.isfinite(weight).all() or (weight < 0.0).any() or float(weight.sum()) <= 0.0:
            raise ValueError("sample_weight must be finite, non-negative and non-empty")
    return frame, {
        "ids": ids, "r3": r3, "gross": gross, "net": net, "decision": decision,
        "available": available, "weight": weight,
    }


def _validation_blocks(
    decision: pd.Series,
    available: pd.Series,
    *,
    n_folds: int,
    min_train_rows: int,
) -> list[np.ndarray]:
    """Create chronological whole-timestamp validation blocks.

    Candidate rows from the same decision timestamp always move together.  The
    later training mask is still resolved by label availability rather than by
    row order, which is the decisive anti-leakage rule.
    """
    ordered = np.argsort(decision.to_numpy(dtype="datetime64[ns]"), kind="stable")
    ordered_ts = decision.to_numpy(dtype="datetime64[ns]")[ordered]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_ts)) + 1]
    groups = [ordered[start:stop] for start, stop in zip(starts, np.r_[starts[1:], len(ordered)])]
    eligible_start = len(groups)
    for group_index, group in enumerate(groups):
        validation_start = decision.iloc[group].min()
        if int(available.lt(validation_start).sum()) >= int(min_train_rows):
            eligible_start = group_index
            break
    remaining = groups[eligible_start:]
    if not remaining:
        raise ValueError("no chronology remains after the configured strict OOF burn-in")
    folds = min(int(n_folds), len(remaining))
    group_blocks = np.array_split(np.arange(len(remaining), dtype=np.int32), folds)
    return [
        np.concatenate([remaining[int(group_index)] for group_index in block]).astype(
            np.int32, copy=False
        )
        for block in group_blocks
        if len(block)
    ]


def _multiclass_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("R3 strict OOF base model must expose predict_proba")
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float32)
    classes = np.asarray(getattr(model, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or set(classes.tolist()) != {0, 1, 2}:
        raise ValueError("each strict OOF R3 base fold must train all three R3 classes")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("R3 fold probabilities must be finite probability simplices")
    return probabilities


def _strict_train_mask(available: pd.Series, validation_start: pd.Timestamp) -> np.ndarray:
    return available.lt(validation_start).to_numpy()


def _runtime_model_params(
    params: Mapping[str, Any], *, layer: str
) -> dict[str, Any]:
    """Freeze the runtime objective instead of silently rewriting it.

    Legacy hand-authored plans may omit the objective; the only derived value
    is then the declared Stage-I contract default.  Any explicit disagreement
    fails before fitting, and the exact derived runtime dictionary is recorded
    in the result manifest.
    """
    runtime = dict(params)
    if layer == "base":
        if "objective" in runtime and str(runtime["objective"]).lower() != "multiclass":
            raise ValueError("Stage-I base runtime objective must be multiclass")
        if "num_class" in runtime and int(runtime["num_class"]) != 3:
            raise ValueError("Stage-I base runtime num_class must be 3")
        runtime["objective"] = "multiclass"
        runtime["num_class"] = 3
        return runtime
    if "objective" in runtime and str(runtime["objective"]).lower() != "huber":
        raise ValueError("Stage-I residual runtime objective must be huber")
    runtime["objective"] = "huber"
    return runtime


def generate_stage_i_strict_oof(plan: StageIStrictOOFPlan) -> StageIStrictOOFResult:
    """Generate one side's base-map-residual expanding chronological OOF ledger."""
    frame, values = _validate_plan(plan)
    side = str(plan.side).lower()
    ids = values["ids"]
    r3 = values["r3"]
    net = values["net"]
    gross = values["gross"]
    decision = values["decision"]
    available = values["available"]
    weight = values["weight"]
    assert isinstance(ids, np.ndarray) and isinstance(r3, np.ndarray) and isinstance(gross, np.ndarray) and isinstance(net, np.ndarray)
    assert isinstance(decision, pd.Series) and isinstance(available, pd.Series) and isinstance(weight, np.ndarray)
    base_features = list(dict.fromkeys(map(str, plan.base_feature_names)))
    meta_features = list(dict.fromkeys(map(str, plan.meta_feature_names)))
    order = np.argsort(decision.to_numpy(dtype="datetime64[ns]"), kind="stable")
    n = len(frame)
    probability = np.full((n, 3), np.nan, dtype=np.float32)
    base_fold = np.full(n, -1, dtype=np.int16)
    provenance_rows: list[dict[str, Any]] = []
    validation_blocks = _validation_blocks(
        decision, available, n_folds=plan.n_validation_folds,
        min_train_rows=plan.min_train_rows,
    )
    base_params = _runtime_model_params(plan.base_params, layer="base")
    for fold_id, validation_idx in enumerate(validation_blocks):
        validation_idx = np.asarray(validation_idx, dtype=np.int32)
        validation_start = decision.iloc[validation_idx].min()
        train_mask = _strict_train_mask(available, validation_start)
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) < int(plan.min_train_rows):
            raise ValueError(f"fold {fold_id} has insufficient prior-resolved base training rows")
        if not available.iloc[train_idx].lt(validation_start).all():
            raise AssertionError("strict base fold admitted unresolved labels")
        if set(np.unique(r3[train_idx]).tolist()) != {0, 1, 2}:
            raise ValueError(f"fold {fold_id} base training lacks an R3 class")
        model = _fit_lgbm_model(
            frame.iloc[train_idx].loc[:, base_features], r3[train_idx], weight[train_idx],
            classifier=True, params=base_params, objective_mode="stage_i_r3_multiclass",
        )
        probability[validation_idx] = _multiclass_probabilities(
            model, frame.iloc[validation_idx].loc[:, base_features]
        )
        base_fold[validation_idx] = int(fold_id)
        provenance_rows.append({
            "side": side, "layer": "base_r3", "fold_id": int(fold_id),
            "train_rows": int(len(train_idx)), "validation_rows": int(len(validation_idx)),
            "validation_start_ts": validation_start, "validation_end_ts": decision.iloc[validation_idx].max(),
            "train_max_label_available_ts": available.iloc[train_idx].max(),
            "strict_prior_resolved": True,
            "base_feature_count": int(len(base_features)), "meta_feature_count": int(len(meta_features)),
        })

    base_scored = np.isfinite(probability).all(axis=1)
    if not base_scored.any():
        raise RuntimeError("strict base OOF emitted no validation predictions")
    base_raw = np.full(n, np.nan, dtype=np.float32)
    base_raw[base_scored] = probability[base_scored, 2] - probability[base_scored, 0]
    map_config = plan.value_map or PrequentialR3ValueMapConfig(side=side)
    if str(map_config.side).lower() != side:
        raise ValueError("value map must use the same side as the base OOF plan")
    mapped_score, map_audit, map_provenance = prequential_same_side_r3_value_map(
        exact_net_bps=net[base_scored], decision_timestamps=decision.iloc[base_scored],
        label_available_timestamps=available.iloc[base_scored], side=side,
        score=base_raw[base_scored], config=map_config,
    )
    base_bps = np.full(n, np.nan, dtype=np.float32)
    base_bps[base_scored] = mapped_score
    map_columns = {column: np.full(n, np.nan, dtype=object) for column in map_audit.columns if column != "side"}
    for column in map_columns:
        map_columns[column][base_scored] = map_audit[column].to_numpy()

    residual = np.full(n, np.nan, dtype=np.float32)
    residual_fold = np.full(n, -1, dtype=np.int16)
    # The residual receives the same-side base output directly, alongside the
    # frozen meta/context subset.  Keep both the raw R3 simplex/contrast and
    # its causal bps map: the expert can learn conversion error without
    # discarding the unconverted opportunity evidence.
    raw_meta_features = [
        feature for feature in meta_features if feature not in _BASE_HANDOFF_FEATURES
    ]
    meta_design = frame.loc[:, raw_meta_features].copy()
    meta_design["r3_p_adverse"] = probability[:, 0]
    meta_design["r3_p_weak"] = probability[:, 1]
    meta_design["r3_p_clear"] = probability[:, 2]
    meta_design["r3_opportunity_score"] = base_raw
    meta_design["prequential_base_expected_net_bps"] = base_bps
    # ``meta_features`` is the winner's exact ordered selector/HPO contract.
    # Do not append a late, unselected base handoff here.
    residual_feature_names = list(meta_features)
    residual_params = _runtime_model_params(plan.residual_params, layer="meta")
    for fold_id, validation_idx in enumerate(validation_blocks):
        validation_idx = np.asarray(validation_idx, dtype=np.int32)
        validation_start = decision.iloc[validation_idx].min()
        train_mask = _strict_train_mask(available, validation_start) & base_scored
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) < int(plan.min_train_rows):
            # The first base validation block has no earlier base OOF scores by
            # construction.  It may therefore be a legitimate residual
            # burn-in; retaining an in-sample base score to fill it would be a
            # larger error than leaving it unavailable.
            provenance_rows.append({
                "side": side, "layer": "meta_residual", "fold_id": int(fold_id),
                "train_rows": int(len(train_idx)), "validation_rows": int(len(validation_idx)),
                "validation_start_ts": validation_start, "validation_end_ts": decision.iloc[validation_idx].max(),
                "train_max_label_available_ts": available.iloc[train_idx].max() if len(train_idx) else pd.NaT,
                "strict_prior_resolved": True, "skipped": True,
                "skip_reason": "insufficient_prior_base_oof_burn_in",
                "base_feature_count": int(len(base_features)), "meta_feature_count": int(len(meta_features)),
            })
            continue
        if not available.iloc[train_idx].lt(validation_start).all():
            raise AssertionError("strict residual fold admitted unresolved labels")
        target = net[train_idx] - base_bps[train_idx]
        model = _fit_lgbm_model(
            meta_design.iloc[train_idx].loc[:, residual_feature_names],
            target, weight[train_idx],
            classifier=False, params=residual_params, objective_mode="stage_i_residual",
        )
        residual[validation_idx] = np.asarray(
            model.predict(
                meta_design.iloc[validation_idx].loc[:, residual_feature_names]
            ), dtype=np.float32
        ).reshape(-1)
        residual_fold[validation_idx] = int(fold_id)
        provenance_rows.append({
            "side": side, "layer": "meta_residual", "fold_id": int(fold_id),
            "train_rows": int(len(train_idx)), "validation_rows": int(len(validation_idx)),
            "validation_start_ts": validation_start, "validation_end_ts": decision.iloc[validation_idx].max(),
            "train_max_label_available_ts": available.iloc[train_idx].max(),
            "strict_prior_resolved": True, "skipped": False, "skip_reason": "",
            "base_feature_count": int(len(base_features)), "meta_feature_count": int(len(meta_features)),
        })
    scored = base_scored & np.isfinite(residual)
    if not scored.any():
        raise RuntimeError("strict residual OOF emitted no validation predictions")
    reconstructed = base_bps + residual
    output = pd.DataFrame({
        "candidate_id": ids,
        "candidate_key": [f"{side}::{value}" for value in ids],
        "side_name": side,
        "decision_ts": decision,
        "label_available_ts": available,
        "exact_gross_bps": gross,
        "exact_net_bps": net,
        "base_strict_oof_available": base_scored,
        "strict_oof_available": scored,
        "base_fold_id": base_fold,
        "residual_fold_id": residual_fold,
        "r3_p_adverse": probability[:, 0],
        "r3_p_weak": probability[:, 1],
        "r3_p_clear": probability[:, 2],
        "r3_opportunity_score": base_raw,
        "prequential_base_expected_net_bps": base_bps,
        "residual_oof_bps": residual,
        "reconstructed_expected_net_bps": reconstructed,
    })
    for column, vector in map_columns.items():
        output[f"value_map__{column}"] = vector
    provenance = pd.DataFrame(provenance_rows)
    if not provenance["strict_prior_resolved"].all():
        raise AssertionError("Stage-I OOF provenance contains a non-causal fold")
    return StageIStrictOOFResult(
        side=side, predictions=output, fold_provenance=provenance,
        value_map_provenance=map_provenance,
        plan_summary={
            "schema": SCHEMA, "side": side, "rows": int(n), "strict_oof_rows": int(scored.sum()),
            "base_feature_names": base_features, "meta_feature_names": meta_features,
            "mandatory_base_handoff_features": list(_BASE_HANDOFF_FEATURES),
            "base_runtime_params": base_params, "residual_runtime_params": residual_params,
            # Retain legacy keys, now guaranteed to be the exact runtime
            # dictionaries rather than pre-normalisation claims.
            "base_params": base_params, "residual_params": residual_params,
        },
    )


def _pooled_metrics(ledger: pd.DataFrame, *, score_column: str, layer: str) -> pd.DataFrame:
    work = ledger.loc[ledger[score_column].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["month"] = pd.to_datetime(work["decision_ts"], utc=True).dt.strftime("%Y-%m")
    order = work.sort_values([score_column, "candidate_key"], ascending=[False, True], kind="stable")
    rows: list[dict[str, Any]] = []
    for fraction in (0.01, 0.05, 0.10, 0.20):
        selected = order.head(max(1, int(np.ceil(fraction * len(order)))) )
        common = {"layer": layer, "top_fraction": fraction, "candidate_rows": int(len(order)), "selected_global_rows": int(len(selected))}
        rows.append({**common, "scope": "pooled_global", "month": "__all__", "side": "__all__", "selected_rows": int(len(selected)), "gross_bps_per_trade": float(selected.exact_gross_bps.mean()), "net_bps_per_trade": float(selected.exact_net_bps.mean())})
        for (month, side), group in selected.groupby(["month", "side_name"], sort=True):
            rows.append({**common, "scope": "selected_contribution", "month": str(month), "side": str(side), "selected_rows": int(len(group)), "gross_bps_per_trade": float(group.exact_gross_bps.mean()), "net_bps_per_trade": float(group.exact_net_bps.mean())})
    return pd.DataFrame(rows)


def write_stage_i_strict_oof_artifact(
    results: Sequence[StageIStrictOOFResult],
    output_dir: str | Path,
    *,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
    admission_reference_results: Sequence[StageIStrictOOFResult] | None = None,
) -> Mapping[str, Any]:
    """Write immutable predictions, strict provenance, metrics and admission.

    The writer refuses an existing destination.  Metrics always rank the
    combined long/short OOF population globally; monthly/side rows only
    attribute the already-selected global book.
    """
    if not results:
        raise ValueError("Stage-I artifact writer needs at least one side result")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite immutable Stage-I artifact: {root}")
    sides = [result.side for result in results]
    if len(set(sides)) != len(sides):
        raise ValueError("Stage-I artifact writer accepts at most one result per side")
    all_predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    if all_predictions.candidate_key.duplicated().any():
        raise ValueError("side results do not form a globally unique candidate population")
    strict_oof_available = _explicit_boolean_array(
        all_predictions.strict_oof_available, label="strict_oof_available"
    )
    scored = all_predictions.loc[strict_oof_available].copy()
    if scored.empty:
        raise ValueError("cannot write a Stage-I artifact without strict OOF rows")
    metrics = pd.concat([
        _pooled_metrics(scored, score_column="prequential_base_expected_net_bps", layer="base"),
        _pooled_metrics(scored, score_column="reconstructed_expected_net_bps", layer="meta_residual"),
    ], ignore_index=True)
    reference_results = results if admission_reference_results is None else admission_reference_results
    reference_predictions = pd.concat(
        [result.predictions for result in reference_results], ignore_index=True
    )
    if reference_predictions.candidate_key.duplicated().any():
        raise ValueError("Stage-I admission reference has duplicate candidate keys")
    reference_available = _explicit_boolean_array(
        reference_predictions.strict_oof_available,
        label="admission_reference.strict_oof_available",
    )
    admission_input = reference_predictions.loc[reference_available].rename(
        columns={"exact_net_bps": "net_bps"}
    )
    admitted_reference, admission_audit = apply_causal_21d_side_admission(
        admission_input,
        score_column="reconstructed_expected_net_bps", net_column="net_bps",
        decision_column="decision_ts", label_available_column="label_available_ts",
        identity_column="candidate_key", spec=admission_spec,
    )
    evaluation_keys = set(scored.candidate_key.astype(str))
    admitted = admitted_reference.loc[
        admitted_reference.candidate_key.astype(str).isin(evaluation_keys)
    ].copy()
    if len(admitted) != len(scored) or set(admitted.candidate_key.astype(str)) != evaluation_keys:
        raise ValueError("Stage-I admission reference did not preserve every evaluation candidate")
    if admission_reference_results is not None and not admission_audit.empty:
        start = pd.to_datetime(scored.decision_ts, utc=True).min().normalize()
        end = pd.to_datetime(scored.decision_ts, utc=True).max().normalize()
        snapshot = pd.to_datetime(admission_audit.snapshot_utc, utc=True, errors="coerce")
        admission_audit = admission_audit.loc[snapshot.between(start, end, inclusive="both")].copy()
        admission_audit["used_prior_history_outside_evaluation"] = True
    admission_metrics = pooled_global_admission_comparison(
        admitted, raw_score_column="reconstructed_expected_net_bps", net_column="net_bps",
        gross_column="exact_gross_bps",
        identity_column="candidate_key", top_fractions=(0.01, 0.05, 0.10, 0.20),
    )
    provenance = pd.concat([result.fold_provenance for result in results], ignore_index=True)
    if not _explicit_boolean_array(
        provenance.strict_prior_resolved, label="fold_provenance.strict_prior_resolved"
    ).all():
        raise AssertionError("immutable artifact rejected non-causal fold provenance")
    root.mkdir(parents=True)
    all_predictions.to_parquet(root / "raw_oof_predictions.parquet", index=False, compression="zstd")
    scored.to_parquet(root / "strict_oof_predictions.parquet", index=False, compression="zstd")
    provenance.to_parquet(root / "fold_provenance.parquet", index=False, compression="zstd")
    metrics.to_parquet(root / "pooled_global_metrics.parquet", index=False, compression="zstd")
    admitted.to_parquet(root / "candidates_with_causal_21d_admission.parquet", index=False, compression="zstd")
    admission_audit.to_parquet(root / "causal_21d_admission_audit.parquet", index=False, compression="zstd")
    admission_metrics.to_parquet(root / "causal_21d_admission_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "sides": sides,
        "rows": {"input": int(len(all_predictions)), "strict_oof": int(len(scored)), "admitted": int(admitted.causal_21d_side_admitted_ge_50bps.sum())},
        "contracts": [dict(result.plan_summary) for result in results],
        "value_maps": [dict(result.value_map_provenance) for result in results],
        "ranking": "pooled global after common-bps mapping; never per timestamp or side",
        "admission": "side-local prior-resolved 21-day expected-net map >= 50 bps; optional earlier strict-OOF history supplies causal boundary support",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


__all__ = [
    "SCHEMA",
    "StageIStrictOOFPlan",
    "StageIStrictOOFResult",
    "generate_stage_i_strict_oof",
    "write_stage_i_strict_oof_artifact",
]
