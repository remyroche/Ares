"""Round-1, sequential target-family screen for the H12 candidate contract.

This module intentionally contains no distillation, GAM, ranking, certainty
weights, archetypes, portfolio rules, or target-family cross-products.  It
materialises the five *primitive* target arms and supplies a strict
prequential/nested-OOF base-plus-meta runner.  The latter is deliberately
small: it is a reusable screening tool, not a promotion training pipeline.

The soft triple-barrier arm is only valid for the reference geometry already
materialised in ``supportive_labels.parquet``.  Wider/tighter barriers require
the minute path because first-touch order cannot be reconstructed from peak,
trough, and terminal summaries.  The public API therefore fails closed for
G1/G2 rather than silently producing an invalid approximation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "sequential_target_family_screen_v1"
TARGET_ARMS = (
    "T0_reconstructed_control",
    "T1_exact_net_huber",
    "T2_soft_atr_triple_barrier",
    "T3_exact_net_multi_quantile",
    "T4_atr_normalized_net_multi_quantile",
)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
REQUIRED_LEDGER_COLUMNS = frozenset(
    {
        "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__",
        "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h",
        "__first_touch_target_soft__",
    }
)
REQUIRED_TRIPLE_COLUMNS = frozenset(
    {
        "clean_economic_favorable_first", "adverse_first", "timeout",
        "endpoint_favorable_margin_return", "endpoint_adverse_margin_return",
        "competing_risk_atr_fraction", "same_minute_favorable_adverse_conflict",
    }
)
PATH_COLUMNS = frozenset({"candidate_id", "execution_future_path", "atr_1h", "decision_price"})


class TargetFamilyScreenError(ValueError):
    """Raised when an arm cannot be materialised or scored causally."""


@dataclass(frozen=True)
class TargetFamilySpec:
    arm: str
    kind: str
    geometry: str = "G0"
    temperature_atr: float = 0.25
    quantiles: tuple[float, ...] = QUANTILES

    def manifest(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SPECS = (
    TargetFamilySpec("T0_reconstructed_control", "control"),
    TargetFamilySpec("T1_exact_net_huber", "huber_net"),
    TargetFamilySpec("T2_soft_atr_triple_barrier", "soft_triple_barrier"),
    TargetFamilySpec("T3_exact_net_multi_quantile", "raw_quantile"),
    TargetFamilySpec("T4_atr_normalized_net_multi_quantile", "atr_quantile"),
)


def _number(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise TargetFamilyScreenError(f"missing required column: {column}")
    result = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise TargetFamilyScreenError(f"non-finite values in required column: {column}")
    return result


def validate_candidate_contract(frame: pd.DataFrame) -> None:
    """Validate the common candidate, H12 availability, and cost contract."""
    missing = sorted(REQUIRED_LEDGER_COLUMNS - set(frame.columns))
    if missing:
        raise TargetFamilyScreenError(f"ledger lacks required contract columns: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise TargetFamilyScreenError("candidate_id must be unique before target materialisation")
    gross = _number(frame, "execution_gross_ev_12h")
    cost = _number(frame, "execution_cost_return")
    net = _number(frame, "execution_net_ev_12h")
    if not np.allclose(gross - cost, net, rtol=0.0, atol=1e-10):
        raise TargetFamilyScreenError("exact net contract violated: gross - row cost != net")
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    available = pd.to_datetime(frame["__label_available_at__"], utc=True, errors="raise")
    if not (available == decision + pd.Timedelta(hours=12)).all():
        raise TargetFamilyScreenError("target availability must be exactly decision + 12h")
    feature_ts = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if (feature_ts > decision).any():
        raise TargetFamilyScreenError("feature timestamp exceeds decision timestamp")


def attach_triple_barrier_context(
    ledger: pd.DataFrame,
    supportive_labels: pd.DataFrame,
) -> pd.DataFrame:
    """One-to-one attach only the path labels needed to construct T2.

    These are labels, not inference features.  The function makes that join
    explicit so callers cannot accidentally use a realised path field in the
    causal matrix.
    """
    if "candidate_id" not in supportive_labels or supportive_labels["candidate_id"].duplicated().any():
        raise TargetFamilyScreenError("supportive label candidate_id must be unique")
    missing = sorted(REQUIRED_TRIPLE_COLUMNS - set(supportive_labels.columns))
    if missing:
        raise TargetFamilyScreenError(f"supportive labels lack T2 fields: {missing}")
    # Two event-state fields are already present in the prepared ledger.  Do
    # not let pandas silently suffix/choose one copy: prove their equality to
    # the authoritative support surface, then attach only missing fields.
    common = sorted((set(ledger.columns) & REQUIRED_TRIPLE_COLUMNS) - {"candidate_id"})
    source_common = supportive_labels[["candidate_id", *common]].copy()
    compare = ledger[["candidate_id", *common]].merge(
        source_common, on="candidate_id", how="left", validate="one_to_one", suffixes=("_ledger", "_support")
    )
    for column in common:
        left = pd.to_numeric(compare[f"{column}_ledger"], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(compare[f"{column}_support"], errors="coerce").to_numpy(float)
        if not np.isfinite(right_values).all() or not np.allclose(left, right_values, rtol=0.0, atol=1e-7):
            raise TargetFamilyScreenError(f"prepared ledger and support surface disagree on {column}")
    # Both aliases derive from the same entry ATR.  The canonical column keeps
    # T4 independent from which supportive-label revision supplied it.
    additions = sorted((REQUIRED_TRIPLE_COLUMNS | {"path_auxiliary_atr_fraction"}) - set(ledger.columns))
    right = supportive_labels[["candidate_id", *additions]].copy()
    overlap = ledger.merge(right, on="candidate_id", how="left", validate="one_to_one")
    missing_rows = overlap[list(REQUIRED_TRIPLE_COLUMNS)].isna().any(axis=1)
    if bool(missing_rows.any()):
        raise TargetFamilyScreenError(f"T2 path context is missing for {int(missing_rows.sum())} candidates")
    atr = _number(overlap, "competing_risk_atr_fraction")
    if "path_auxiliary_atr_fraction" in overlap:
        path_atr = _number(overlap, "path_auxiliary_atr_fraction")
        if not np.allclose(atr, path_atr, rtol=0.0, atol=1e-8):
            raise TargetFamilyScreenError("competing-risk and path ATR fraction aliases disagree")
    overlap["entry_atr_fraction"] = atr.astype(np.float32)
    return overlap


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(shifted, -60.0, 60.0))
    return weights / weights.sum(axis=1, keepdims=True)


def soft_triple_barrier_distribution(frame: pd.DataFrame, *, temperature_atr: float = 0.25, geometry: str = "G0") -> np.ndarray:
    """Return ordered ``P(upper), P(lower), P(timeout)`` soft labels.

    Exact first-hit observations remain one-hot.  For genuine timeouts, the
    terminal barrier margins supply a smooth local label: the midpoint is
    timeout-like, while positions close to either barrier transfer mass to the
    corresponding event.  Same-minute conflicts are hard lower/adverse
    outcomes, matching the frozen source convention.
    """
    if geometry != "G0":
        raise TargetFamilyScreenError(
            f"{geometry} requires ordered H12 minute paths; endpoint summaries cannot preserve first-hit order"
        )
    if not np.isfinite(float(temperature_atr)) or float(temperature_atr) <= 0.0:
        raise TargetFamilyScreenError("temperature_atr must be strictly positive")
    missing = sorted(REQUIRED_TRIPLE_COLUMNS - set(frame.columns))
    if missing:
        raise TargetFamilyScreenError(f"soft triple barrier lacks path fields: {missing}")
    upper = _number(frame, "clean_economic_favorable_first") > 0.5
    lower = _number(frame, "adverse_first") > 0.5
    timeout = _number(frame, "timeout") > 0.5
    conflict = _number(frame, "same_minute_favorable_adverse_conflict") > 0.5
    if not np.array_equal(upper.astype(int) + lower.astype(int) + timeout.astype(int), np.ones(len(frame), dtype=int)):
        raise TargetFamilyScreenError("T2 event states must be mutually exclusive and exhaustive")
    # The frozen source gives adverse precedence on same-minute conflicts.
    if np.any(conflict & ~lower):
        raise TargetFamilyScreenError("same-minute conflict does not respect frozen adverse-first convention")
    result = np.zeros((len(frame), 3), dtype=np.float64)
    result[upper, 0] = 1.0
    result[lower, 1] = 1.0
    if timeout.any():
        atr_column = "entry_atr_fraction" if "entry_atr_fraction" in frame else "competing_risk_atr_fraction"
        atr = np.maximum(_number(frame, atr_column), 1e-8)
        fav_margin = _number(frame, "endpoint_favorable_margin_return") / atr
        adverse_margin = _number(frame, "endpoint_adverse_margin_return") / atr
        # Distance below upper and above lower is non-negative for a timeout.
        upper_distance = np.maximum(-fav_margin, 0.0)
        lower_distance = np.maximum(adverse_margin, 0.0)
        tau = float(temperature_atr)
        logits = np.column_stack(
            (-upper_distance / tau, -lower_distance / tau, -np.abs(upper_distance - lower_distance) / tau)
        )
        result[timeout] = _softmax(logits[timeout])
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-7):
        raise TargetFamilyScreenError("soft triple-barrier probabilities do not sum to one")
    return result.astype(np.float32)


def reconcile_quantiles(values: np.ndarray) -> np.ndarray:
    """Apply monotone rearrangement to quantile predictions row-wise."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(QUANTILES):
        raise TargetFamilyScreenError("quantile prediction matrix has the wrong shape")
    if not np.isfinite(array).all():
        raise TargetFamilyScreenError("quantile prediction matrix contains non-finite values")
    return np.maximum.accumulate(array, axis=1)


def quantile_expected_value(quantiles: np.ndarray) -> np.ndarray:
    """Fixed trapezoidal integration of q10..q90, with bounded tails."""
    values = reconcile_quantiles(quantiles)
    # Treat q10/q90 as constant to 0/1.  This is predeclared and avoids
    # selecting a scoring functional on final OOS evidence.
    levels = np.asarray((0.0, *QUANTILES, 1.0), dtype=float)
    extended = np.column_stack((values[:, 0], values, values[:, -1]))
    # The project currently supports NumPy releases that predate
    # ``trapezoid``.  ``trapz`` has the identical integration semantics here.
    return np.trapz(extended, levels, axis=1)


def materialize_target_family_labels(
    frame: pd.DataFrame,
    *,
    triple_geometry: str = "G0",
    triple_temperature_atr: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Materialise all T0--T4 labels in one common candidate order.

    ``T0`` is the existing reconstructed H12 control.  ``T1`` is direct net
    Huber supervision. ``T3`` and ``T4`` retain continuous labels because the
    quantile loss is selected at fit time; no future label becomes a feature.
    """
    validate_candidate_contract(frame)
    work = frame.copy()
    net = _number(work, "execution_net_ev_12h")
    atr = _number(work, "competing_risk_atr_fraction") if "competing_risk_atr_fraction" in work else None
    work["target_t0_control"] = np.clip(_number(work, "__first_touch_target_soft__"), 0.0, 1.0).astype(np.float32)
    work["target_t1_net_return"] = net.astype(np.float32)
    triple = soft_triple_barrier_distribution(work, temperature_atr=triple_temperature_atr, geometry=triple_geometry)
    work["target_t2_upper_soft"] = triple[:, 0]
    work["target_t2_lower_soft"] = triple[:, 1]
    work["target_t2_timeout_soft"] = triple[:, 2]
    work["target_t3_net_return"] = net.astype(np.float32)
    if atr is None:
        raise TargetFamilyScreenError("T4 requires entry-time competing_risk_atr_fraction")
    if (atr <= 0.0).any():
        raise TargetFamilyScreenError("entry-time ATR fraction must be strictly positive")
    work["target_t4_net_atr"] = (net / atr).astype(np.float32)
    manifest = {
        "schema": SCHEMA,
        "candidate_count": int(len(work)),
        "candidate_id_sha256": __import__("hashlib").sha256("\n".join(work.candidate_id.astype(str)).encode()).hexdigest(),
        "target_arms": [spec.manifest() for spec in DEFAULT_SPECS],
        "triple_barrier": {
            "geometry": triple_geometry,
            "temperature_atr": float(triple_temperature_atr),
            "state_order": ["upper_first", "lower_first", "timeout"],
            "hit_semantics": "one-hot exact first-hit; adverse/lower precedence on same-minute conflict",
            "timeout_semantics": "endpoint-margin softmax in ATR units; only timeout paths are softened",
        },
        "quantile_target_dictionary": {
            "T3": {"label": "exact_H12_net_return", "quantiles": list(QUANTILES), "common_unit": "bps after x10000"},
            "T4": {"label": "exact_H12_net_return / entry_ATR_return", "quantiles": list(QUANTILES), "back_conversion": "multiply predicted quantiles by entry ATR then x10000"},
        },
        "inference_guard": "only causal feature matrix is admitted; all target_* columns and path context are labels only",
    }
    return work, manifest


def nested_oof_fold_plan(frame: pd.DataFrame, *, fold_column: str = "oof_fold") -> pd.DataFrame:
    """Return the strict base/meta fold lineage required by the screen.

    This is a hook/validation artifact rather than a model fit.  For target
    fold *k*, base training is all earlier resolved folds.  Meta training is
    restricted further to earlier rows with an already-emitted base OOF output;
    therefore no meta learner can consume an in-sample upstream prediction.
    """
    if fold_column not in frame:
        raise TargetFamilyScreenError(f"missing fold column: {fold_column}")
    validate_candidate_contract(frame)
    work = frame[[fold_column, "__ts__", "__label_available_at__"]].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["__label_available_at__"] = pd.to_datetime(work["__label_available_at__"], utc=True, errors="raise")
    starts = work.groupby(fold_column, observed=True)["__ts__"].min().sort_values(kind="mergesort")
    rows: list[dict[str, object]] = []
    for position, (fold, start) in enumerate(starts.items()):
        base_folds = list(starts.index[:position])
        base_mask = work[fold_column].isin(base_folds) & work["__label_available_at__"].lt(start)
        # A meta target fold needs at least one earlier *scored* base fold;
        # the very first base fold is warmup and has no upstream OOF output.
        meta_folds = list(starts.index[1:position])
        meta_mask = work[fold_column].isin(meta_folds) & work["__label_available_at__"].lt(start)
        rows.append({
            "fold": str(fold), "fold_position": int(position), "test_start_ts": start,
            "base_train_folds": [str(x) for x in base_folds],
            "base_train_rows": int(base_mask.sum()),
            "meta_train_folds_with_base_oof": [str(x) for x in meta_folds],
            "meta_train_rows": int(meta_mask.sum()),
            "base_scored": bool(position >= 1 and base_mask.any()),
            "meta_scored": bool(position >= 2 and meta_mask.any()),
            "rule": "base labels available < test_start; meta inputs are strict earlier base OOF only",
        })
    return pd.DataFrame(rows)


def target_family_manifest(
    labels: pd.DataFrame,
    label_manifest: Mapping[str, object],
    *,
    fold_column: str = "oof_fold",
) -> dict[str, object]:
    """Return the immutable Round-1 contract for the runnable screen."""
    plan = nested_oof_fold_plan(labels, fold_column=fold_column)
    return {
        "schema": SCHEMA,
        "round": "Round 1 target screen only: no certainty/distillation/GAM/ranking/archetypes",
        "target_labels": dict(label_manifest),
        "nested_oof_protocol": plan.to_dict(orient="records"),
        "selection": "one pooled-global ranking after common bps mapping; no side/timestamp/asset quotas or portfolio constraints",
        "strict_guards": {
            "base_predictions_for_meta": "earlier chronological base OOF only",
            "meta_target": "exact_H12_net_bps - causal_base_expected_net_bps",
            "label_available": "strictly before target fold start",
            "future_path_fields": "label construction only; never inference features",
        },
        "status": "MATERIALIZED_READY_FOR_ROUND1_FIT_RESEARCH_ONLY",
    }
