"""Leakage-safe post-execution-EV entry timing/action-value meta head.

This module is intentionally downstream of the frozen execution-EV handoff. It
turns *future* executable paths into train-only counterfactual labels for a
small, explicit action grid, then fits side-local action heads using only
declared pre-entry OOF/frozen inputs.  The scoring API rejects realized path and
label fields; it needs only the serialized models and the frozen feature
contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .execution_ev_labels import (
    ExecutionLabelGeometry,
    reason_names,
    simulate_execution_ev_12h,
)
from .execution_ev_meta import ChronologicalPurgedSplit, chronological_purged_splits
from .path_archetype_labels import PATH_SHAPE_TYPES

ENTRY_TIMING_SCHEMA = "execution_entry_timing_action_value_v2"

ENTRY_TIMING_FEATURE_FAMILIES: tuple[str, ...] = (
    "execution_ev_prediction",
    "execution_ev_mapping",
    "alpha_outputs",
    "residual_outputs",
    "catboost_probabilities",
    "catboost_entropy",
    "auxiliary_heads",
    "side_archetypes",
    "uncertainty",
    "ood",
    "leaf_support",
)

# These families are themselves predictive model outputs.  They must retain
# row-level upstream OOF evidence when they are used to train or score timing
# OOF metrics.  A final-refit bundle ID is sufficient only for inference.
PREDICTIVE_ENTRY_TIMING_FEATURE_FAMILIES: frozenset[str] = frozenset(
    {
        "execution_ev_prediction",
        "execution_ev_mapping",
        "alpha_outputs",
        "residual_outputs",
        "catboost_probabilities",
        "catboost_entropy",
        "auxiliary_heads",
    }
)

# Only names with an explicit prediction/OOF/frozen prefix can contain a
# realised-looking token.  This prevents a raw output from silently becoming a
# feature because its provenance record was copied incorrectly.
FORBIDDEN_REALIZED_FEATURE_TOKENS: tuple[str, ...] = (
    "realized",
    "future",
    "label",
    "target",
    "post_fill",
    "execution_net_ev",
    "execution_gross_ev",
    "execution_mfe",
    "execution_mae",
    "fill_time",
    "missed_opportunity",
)
FORBIDDEN_SCORING_TOKENS = FORBIDDEN_REALIZED_FEATURE_TOKENS + (
    "counterfactual_path",
    "adverse_first_target",
)

ActionKind = Literal["enter_now", "wait_market", "adverse_limit"]


@dataclass(frozen=True)
class EntryAction:
    """One bounded executable entry action.

    ``wait_minutes`` is the market-entry delay for ``wait_market`` and the
    passive-order expiry for ``adverse_limit``.  Limits are side-relative and
    use ``adverse_offset_atr`` from the decision-time ATR.
    """

    kind: ActionKind
    wait_minutes: int = 0
    adverse_offset_atr: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("enter_now", "wait_market", "adverse_limit"):
            raise ValueError(f"unsupported entry action kind {self.kind!r}")
        if self.wait_minutes < 0:
            raise ValueError("action wait_minutes must be non-negative")
        if self.adverse_offset_atr < 0.0:
            raise ValueError("action adverse_offset_atr must be non-negative")
        if self.kind == "enter_now" and (
            self.wait_minutes != 0 or self.adverse_offset_atr != 0.0
        ):
            raise ValueError("enter_now must have zero delay and zero adverse offset")
        if self.kind == "wait_market" and self.adverse_offset_atr != 0.0:
            raise ValueError("wait_market must have zero adverse offset")
        if self.kind == "adverse_limit" and (
            self.wait_minutes <= 0 or self.adverse_offset_atr <= 0.0
        ):
            raise ValueError("adverse_limit requires positive wait_minutes and offset")

    @property
    def action_id(self) -> str:
        if self.kind == "enter_now":
            return "enter_now"
        if self.kind == "wait_market":
            return f"wait_market_{self.wait_minutes}m"
        return f"adverse_limit_{self.wait_minutes}m_{self.adverse_offset_atr:.4f}atr"


def default_entry_action_grid() -> tuple[EntryAction, ...]:
    """Small, stable grid; selection may only choose a subset of these actions."""

    return (
        EntryAction("enter_now"),
        EntryAction("wait_market", wait_minutes=5),
        EntryAction("wait_market", wait_minutes=10),
        EntryAction("wait_market", wait_minutes=20),
        EntryAction("adverse_limit", wait_minutes=5, adverse_offset_atr=0.25),
        EntryAction("adverse_limit", wait_minutes=10, adverse_offset_atr=0.25),
        EntryAction("adverse_limit", wait_minutes=10, adverse_offset_atr=0.50),
        EntryAction("adverse_limit", wait_minutes=20, adverse_offset_atr=0.50),
    )


@dataclass(frozen=True)
class EntryTimingTargetSpec:
    """Train-only path fields and execution accounting for action labels."""

    path_col: str = "execution_future_path"
    atr_col: str = "atr_1h"
    # The all-in route is mutually exclusive with the decomposed fee/spread
    # route.  This makes fee and spread reconciliation observable and prevents
    # subtracting either cost twice.
    cost_return_col: str | None = None
    fee_return_col: str | None = None
    entry_spread_bps_col: str | None = None
    exit_spread_bps_col: str | None = None
    allow_action_invariant_all_in_cost: bool = False
    decision_price_col: str | None = None
    horizon_hours: float = 12.0
    meaningful_mfe_atr: float = 1.5
    meaningful_mfe_return_floor: float = 0.015
    adverse_mae_atr: float = 0.25
    path_frequency_minutes: int = 1
    path_timestamp_key: str = "timestamp"
    path_open_key: str = "open"
    path_high_key: str = "high"
    path_low_key: str = "low"
    path_close_key: str = "close"
    # The canonical execution-EV geometry is part of the target definition.
    # It is intentionally serialized with the timing bundle instead of being
    # inferred from a current policy file at replay time.
    long_policy_geometry: Mapping[str, object] | None = None
    short_policy_geometry: Mapping[str, object] | None = None
    execution_ev_target_manifest_path: str | None = None
    execution_ev_target_manifest_sha256: str | None = None
    execution_ev_target_signed_manifest_sha256: str | None = None
    execution_ev_target_schema: str | None = None
    execution_ev_policy_manifest_sha256: str | None = None


@dataclass(frozen=True)
class EntryTimingFeatureProvenance:
    """Strict availability declaration for a timing-head feature."""

    family: str
    source: str
    pre_entry: bool = True
    oof_or_frozen: bool = True
    available_at_col: str | None = None
    # A training source must provide row-level OOF evidence, while a final
    # frozen source provides its immutable artifact identity.  Either is
    # required; a bare assertion that a column is "OOF" is not sufficient.
    oof_fold_col: str | None = None
    source_train_cutoff_col: str | None = None
    frozen_bundle_id: str | None = None
    cost_spread_aware: bool = False
    model_input: bool = True


@dataclass(frozen=True)
class EntryTimingTrainerConfig:
    """Bounded, deterministic side-local training and decision configuration."""

    n_splits: int = 3
    min_train_rows: int = 500
    purge_hours: float = 12.0
    embargo_hours: float = 12.0
    inner_n_splits: int = 2
    n_estimators: int = 320
    early_stopping_rounds: int = 40
    random_state: int = 42
    n_jobs: int = 1
    hpo_trials: int = 8
    hpo_timeout_seconds: float | None = None
    decision_hpo_trials: int = 8
    side_col: str = "side_name"
    archetype_col: str = "catboost_archetype"
    decision_time_col: str = "__decision_ts__"
    label_end_time_col: str | None = None
    action_grid: tuple[EntryAction, ...] = field(default_factory=default_entry_action_grid)
    missed_opportunity_penalty: float = 1.0
    adverse_first_penalty: float = 0.004
    strict_feature_families: bool = True

    def __post_init__(self) -> None:
        if self.n_splits < 1 or self.min_train_rows < 4:
            raise ValueError("n_splits must be positive and min_train_rows must be at least four")
        if self.purge_hours < 0.0 or self.embargo_hours < 0.0:
            raise ValueError("purge and embargo hours must be non-negative")
        if self.n_estimators < 8:
            raise ValueError("n_estimators must be at least eight")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if self.hpo_trials < 0 or self.decision_hpo_trials < 0:
            raise ValueError("HPO trial counts must be non-negative")
        if not 0.0 <= self.missed_opportunity_penalty <= 4.0:
            raise ValueError("missed-opportunity penalty must be in [0, 4]")
        if not 0.0 <= self.adverse_first_penalty <= 0.25:
            raise ValueError("adverse-first penalty must be in [0, .25]")
        if self.decision_time_col != "__decision_ts__":
            raise ValueError(
                "entry timing must use upstream execution-EV decision timestamp "
                "'__decision_ts__'"
            )
        _validate_action_grid(self.action_grid)


@dataclass(frozen=True)
class _ConstantClassifier:
    probability: float

    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        probability = float(np.clip(self.probability, 0.0, 1.0))
        return np.tile(
            np.asarray((1.0 - probability, probability), dtype=np.float64),
            (len(values), 1),
        )


@dataclass(frozen=True)
class _ConstantRegressor:
    value: float

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        return np.full(len(values), float(self.value), dtype=np.float64)


@dataclass(frozen=True)
class _IsotonicCalibrator:
    """Persistable identity fallback or monotonic train-OOF calibration map."""

    model: Any | None = None
    probability: bool = False
    status: str = "identity_insufficient_train_oof"

    def predict(self, values: Sequence[float]) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64)
        if self.model is not None:
            output = np.asarray(self.model.predict(output), dtype=np.float64)
        return np.clip(output, 0.0, 1.0) if self.probability else output


@dataclass
class ExecutionEntryTimingBundle:
    """Persistable final models plus immutable OOF-only audit evidence."""

    schema: str
    config: dict[str, Any]
    target_spec: EntryTimingTargetSpec
    provenance: dict[str, EntryTimingFeatureProvenance]
    feature_names: tuple[str, ...]
    execution_ev_feature: str
    decision_policy: dict[str, Any]
    models: dict[str, dict[str, dict[str, Any]]]
    report: dict[str, Any]
    input_fingerprint: str
    bundle_fingerprint: str
    oof_action_predictions: pd.DataFrame = field(repr=False)
    oof_recommendations: pd.DataFrame = field(repr=False)
    oof_provenance: pd.DataFrame = field(repr=False)


def _validate_action_grid(actions: Sequence[EntryAction]) -> tuple[EntryAction, ...]:
    grid = tuple(actions)
    if not grid or len(grid) > 24:
        raise ValueError("entry action grid must contain between one and 24 actions")
    ids = [action.action_id for action in grid]
    if len(set(ids)) != len(ids):
        raise ValueError("entry action IDs must be unique")
    if sum(action.kind == "enter_now" for action in grid) != 1:
        raise ValueError("entry action grid must contain exactly one enter_now action")
    return grid


def _utc(values: pd.Series | Sequence[Any], *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError(f"{name!r} contains null or invalid timestamps")
    return pd.Series(parsed, index=getattr(values, "index", None))


def _side_values(frame: pd.DataFrame, side_col: str) -> np.ndarray:
    if side_col not in frame.columns:
        raise ValueError(f"entry timing requires side column {side_col!r}")
    side = frame[side_col].astype("string").str.strip().str.lower().to_numpy(dtype=str)
    invalid = sorted(set(side) - {"long", "short"})
    if invalid:
        raise ValueError(f"entry timing side must be canonical long/short, got {invalid!r}")
    return side


def _numeric(frame: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"entry timing {role} is missing required column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"entry timing {role} has non-finite values in {column!r}")
    return values


def _is_realized_feature_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith(("pred_", "oof_", "frozen_", "score_", "mapped_")):
        return False
    return any(token in lowered for token in FORBIDDEN_REALIZED_FEATURE_TOKENS)


def _is_realized_scoring_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith(("pred_", "oof_", "frozen_", "score_", "mapped_")):
        return False
    return any(token in lowered for token in FORBIDDEN_SCORING_TOKENS)


def validate_entry_timing_feature_contract(
    frame: pd.DataFrame,
    provenance: Mapping[str, EntryTimingFeatureProvenance],
    *,
    config: EntryTimingTrainerConfig = EntryTimingTrainerConfig(),
    for_scoring: bool = False,
) -> tuple[list[str], str]:
    """Validate exact pre-entry OOF/frozen model inputs and their availability.

    The returned execution-EV feature is the model's frozen enter-now EV input.
    It is deliberately selected from declared provenance rather than inferred
    from similarly named realised columns.
    """

    if config.decision_time_col != "__decision_ts__":
        raise ValueError(
            "entry timing requires upstream execution-EV decision timestamp "
            "'__decision_ts__'"
        )
    if config.decision_time_col not in frame.columns:
        raise ValueError(
            f"entry timing requires decision time {config.decision_time_col!r}"
        )
    decision = _utc(frame[config.decision_time_col], name=config.decision_time_col)
    _side_values(frame, config.side_col)
    if config.archetype_col not in frame.columns:
        raise ValueError(f"entry timing requires archetype column {config.archetype_col!r}")
    names = [name for name, spec in provenance.items() if spec.model_input]
    if not names:
        raise ValueError("entry timing needs at least one declared model input")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError("entry timing features missing columns: " + ", ".join(missing))
    family_columns: dict[str, list[str]] = {family: [] for family in ENTRY_TIMING_FEATURE_FAMILIES}
    for name in names:
        spec = provenance[name]
        if spec.family not in family_columns:
            raise ValueError(
                f"unsupported entry timing feature family for {name!r}: {spec.family!r}"
            )
        if not spec.source.strip() or not spec.pre_entry or not spec.oof_or_frozen:
            raise ValueError(
                f"entry timing input {name!r} must declare source and be pre-entry OOF/frozen"
            )
        if "in-sample" in spec.source.lower() or "insample" in spec.source.lower():
            raise ValueError(f"entry timing input {name!r} is declared in-sample and is rejected")
        if _is_realized_feature_name(name):
            raise ValueError(
                f"entry timing input {name!r} appears realised; use a frozen prediction output"
            )
        if spec.available_at_col is None:
            raise ValueError(
                f"entry timing input {name!r} requires an explicit available_at_col"
            )
        if spec.available_at_col not in frame.columns:
            raise ValueError(
                f"entry timing input {name!r} references missing availability column "
                f"{spec.available_at_col!r}"
            )
        available = _utc(frame[spec.available_at_col], name=spec.available_at_col)
        if (available.to_numpy() > decision.to_numpy()).any():
            raise ValueError(
                f"entry timing input {name!r} was available after its decision timestamp"
            )
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"entry timing input {name!r} must be finite numeric")
        predictive_family = spec.family in PREDICTIVE_ENTRY_TIMING_FEATURE_FAMILIES
        if predictive_family and for_scoring:
            if not str(spec.frozen_bundle_id or "").strip():
                raise ValueError(
                    f"entry timing predictive input {name!r} requires a frozen final-refit "
                    "bundle ID for inference"
                )
        elif spec.oof_fold_col is not None:
            if spec.oof_fold_col not in frame.columns:
                raise ValueError(
                    f"entry timing input {name!r} references missing OOF fold column "
                    f"{spec.oof_fold_col!r}"
                )
            fold_values = frame[spec.oof_fold_col]
            if fold_values.isna().any() or (fold_values.astype("string").str.strip() == "").any():
                raise ValueError(f"entry timing input {name!r} has missing row-level OOF provenance")
        elif predictive_family and not for_scoring:
            raise ValueError(
                f"entry timing predictive input {name!r} requires row-level OOF fold and "
                "source train-cutoff provenance for training OOF metrics; a frozen final "
                "refit ID is inference-only"
            )
        elif not str(spec.frozen_bundle_id or "").strip():
            raise ValueError(
                f"entry timing input {name!r} must declare oof_fold_col or frozen_bundle_id; "
                "in-sample outputs are not accepted"
            )
        if predictive_family and not for_scoring:
            if spec.oof_fold_col is None or spec.source_train_cutoff_col is None:
                raise ValueError(
                    f"entry timing predictive input {name!r} requires row-level source fold "
                    "and source train cutoff for OOF timing metrics"
                )
            if spec.source_train_cutoff_col not in frame.columns:
                raise ValueError(
                    f"entry timing input {name!r} references missing source train cutoff "
                    f"column {spec.source_train_cutoff_col!r}"
                )
            source_cutoff = _utc(
                frame[spec.source_train_cutoff_col], name=spec.source_train_cutoff_col
            )
            if not (source_cutoff.to_numpy() < decision.to_numpy()).all():
                raise ValueError(
                    f"entry timing input {name!r} source train cutoff must be strictly "
                    "before every scored decision timestamp"
                )
        family_columns[spec.family].append(name)
    if config.strict_feature_families:
        minimum = {
            "execution_ev_prediction": 1,
            "execution_ev_mapping": 1,
            "alpha_outputs": 1,
            "residual_outputs": 1,
            "catboost_probabilities": len(PATH_SHAPE_TYPES),
            "catboost_entropy": 1,
            "auxiliary_heads": 5,
            "side_archetypes": 2,
        }
        absent = [
            f"{family}({count})"
            for family, count in minimum.items()
            if len(family_columns[family]) < count
        ]
        if absent:
            raise ValueError(
                "entry timing provenance lacks required frozen feature families: "
                + ", ".join(absent)
            )
    probability_columns = family_columns["catboost_probabilities"]
    if config.strict_feature_families and len(probability_columns) != len(PATH_SHAPE_TYPES):
        raise ValueError(
            "entry timing requires the complete ordered CatBoost probability vector "
            f"with exactly {len(PATH_SHAPE_TYPES)} columns"
        )
    if probability_columns:
        probabilities = frame.loc[:, probability_columns].to_numpy(dtype=np.float64, copy=False)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError("complete CatBoost probability vector must sum to one per row")
        entropy_columns = family_columns["catboost_entropy"]
        entropy = frame[entropy_columns[0]].to_numpy(dtype=np.float64)
        expected_entropy = -np.sum(
            np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
        )
        if not np.allclose(entropy, expected_entropy, atol=2e-5, rtol=2e-5):
            raise ValueError("CatBoost entropy must match the complete probability vector")
    ev_columns = family_columns["execution_ev_prediction"]
    if len(ev_columns) != 1:
        raise ValueError(
            "entry timing requires exactly one frozen execution_ev_prediction input"
        )
    if not provenance[ev_columns[0]].cost_spread_aware:
        raise ValueError(
            "entry timing requires the protected execution-EV input to declare "
            "cost_spread_aware=True"
        )
    if for_scoring:
        realized = sorted(
            name for name in frame.columns if _is_realized_scoring_name(str(name))
        )
        realized = [name for name in realized if name not in names]
        if realized:
            raise ValueError(
                "entry timing scoring frame contains realised fields: " + ", ".join(realized)
            )
    return names, ev_columns[0]


def _decode_path(value: Any, spec: EntryTimingTargetSpec, *, row: Any) -> pd.DataFrame:
    """Normalize one compact future path cell into ordered executable bars."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"counterfactual path is invalid JSON for row {row!r}") from exc
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        if "bars" in value:
            value = value["bars"]
        else:
            vector_keys = (
                spec.path_timestamp_key,
                spec.path_open_key,
                spec.path_high_key,
                spec.path_low_key,
                spec.path_close_key,
            )
            if all(key in value for key in vector_keys) and all(
                isinstance(value[key], Sequence) and not isinstance(value[key], str)
                for key in vector_keys
            ):
                lengths = {len(value[key]) for key in vector_keys}
                if len(lengths) != 1:
                    raise ValueError(f"counterfactual path vectors have unequal length for row {row!r}")
                value = [
                    {key: value[key][position] for key in vector_keys}
                    for position in range(next(iter(lengths)))
                ]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"counterfactual path must be a sequence of bars for row {row!r}")
    rows = list(value)
    if not rows:
        return pd.DataFrame(
            columns=("timestamp", "open", "high", "low", "close")
        )
    if not all(isinstance(item, Mapping) for item in rows):
        raise ValueError(f"counterfactual path bars must be objects for row {row!r}")
    aliases = {
        "timestamp": (spec.path_timestamp_key, "ts", "__ts__", "time"),
        "open": (spec.path_open_key, "price", "close"),
        "high": (spec.path_high_key, "price", "close"),
        "low": (spec.path_low_key, "price", "close"),
        "close": (spec.path_close_key, "price", "open"),
    }
    normalized: dict[str, list[Any]] = {key: [] for key in aliases}
    for bar in rows:
        for output, keys in aliases.items():
            match = next((key for key in keys if key in bar), None)
            if match is None:
                raise ValueError(
                    f"counterfactual path bar is missing {output!r} for row {row!r}"
                )
            normalized[output].append(bar[match])
    path = pd.DataFrame(normalized)
    path["timestamp"] = _utc(path["timestamp"], name="counterfactual path timestamp")
    for column in ("open", "high", "low", "close"):
        path[column] = pd.to_numeric(path[column], errors="coerce")
    numbers = path.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers <= 0.0).any():
        raise ValueError(f"counterfactual path has invalid prices for row {row!r}")
    if (path["high"] < path["low"]).any():
        raise ValueError(f"counterfactual path has high below low for row {row!r}")
    return path.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _policy_geometry(
    spec: EntryTimingTargetSpec, side_name: str
) -> ExecutionLabelGeometry:
    values = (
        spec.long_policy_geometry if side_name == "long" else spec.short_policy_geometry
    )
    if not isinstance(values, Mapping):
        raise ValueError(
            "entry timing target requires serialized long/short execution-EV policy geometry"
        )
    return ExecutionLabelGeometry.from_mapping(values)


def _mae_before_meaningful_mfe(
    post: pd.DataFrame,
    *,
    exit_bar: int,
    fill_price: float,
    side_sign: float,
    atr: float,
    exit_spread_bps: float,
    spec: EntryTimingTargetSpec,
) -> tuple[float, bool]:
    """Measure adverse-first on the executable segment resolved by the policy."""

    if exit_bar < 0 or post.empty:
        return np.nan, False
    observed = post.iloc[: int(exit_bar) + 1]
    exit_factor = 1.0 - side_sign * float(exit_spread_bps) * 1e-4
    high = side_sign * (
        observed["high"].to_numpy(dtype=float) * exit_factor - fill_price
    ) / fill_price
    low = side_sign * (
        observed["low"].to_numpy(dtype=float) * exit_factor - fill_price
    ) / fill_price
    favorable = np.maximum(high, low)
    adverse = np.minimum(high, low)
    running_mfe = np.maximum.accumulate(favorable)
    meaningful_threshold = max(
        float(spec.meaningful_mfe_return_floor),
        float(spec.meaningful_mfe_atr) * atr / fill_price,
    )
    reached = np.flatnonzero(running_mfe >= meaningful_threshold)
    cutoff = int(reached[0]) if len(reached) else len(observed) - 1
    mae_before = float(np.min(adverse[: cutoff + 1]))
    adverse_threshold = max(0.0, float(spec.adverse_mae_atr) * atr / fill_price)
    return mae_before, bool(mae_before <= -adverse_threshold)


def _geometry_aware_post_fill_outcome(
    path: pd.DataFrame,
    *,
    policy_start_position: int,
    fill_price: float,
    side_sign: float,
    atr: float,
    cost_return: float,
    exit_spread_bps: float,
    geometry: ExecutionLabelGeometry,
    spec: EntryTimingTargetSpec,
) -> dict[str, Any]:
    """Re-anchor the canonical execution-EV simulator at an action's fill.

    The 12h label simulator owns every exit decision, including stops,
    trailing, adverse exits, timeout, and MFE/MAE accounting.  We only adapt
    the action-specific executable path and apply the already-declared exit
    spread to executable exit observations.
    """

    post = path.iloc[policy_start_position:].reset_index(drop=True)
    if post.empty:
        return {
            "post_fill_gross_ev": np.nan,
            "post_fill_net_ev": np.nan,
            "post_fill_mfe": np.nan,
            "post_fill_mae": np.nan,
            "mae_before_meaningful_mfe": np.nan,
            "adverse_first": False,
            "execution_exit_reason": None,
            "execution_exit_bar": -1,
            "execution_exit_timestamp": pd.NaT,
        }
    exit_factor = 1.0 - side_sign * float(exit_spread_bps) * 1e-4
    opens = post["open"].to_numpy(dtype=np.float64, copy=True)
    # ``simulate_execution_ev_12h`` takes its entry from opens[:, 0].  Replacing
    # only that value anchors the shared policy simulator to the actual action
    # fill while retaining the observed executable path from that point onward.
    opens[0] = float(fill_price)
    highs = post["high"].to_numpy(dtype=np.float64, copy=True) * exit_factor
    lows = post["low"].to_numpy(dtype=np.float64, copy=True) * exit_factor
    closes = post["close"].to_numpy(dtype=np.float64, copy=True) * exit_factor
    result = simulate_execution_ev_12h(
        opens.reshape(1, -1),
        highs.reshape(1, -1),
        lows.reshape(1, -1),
        closes.reshape(1, -1),
        np.asarray([side_sign], dtype=np.float64),
        np.asarray([atr / fill_price], dtype=np.float64),
        np.asarray([cost_return], dtype=np.float64),
        geometry.vector(),
        geometry.vector(),
        1,
    )
    gross, net, reason, exit_bar, mfe, mae = result
    local_exit_bar = int(exit_bar[0])
    mae_before, adverse_first = _mae_before_meaningful_mfe(
        post,
        exit_bar=local_exit_bar,
        fill_price=fill_price,
        side_sign=side_sign,
        atr=atr,
        exit_spread_bps=exit_spread_bps,
        spec=spec,
    )
    return {
        "post_fill_gross_ev": float(gross[0]),
        "post_fill_net_ev": float(net[0]),
        "post_fill_mfe": float(mfe[0]),
        "post_fill_mae": float(mae[0]),
        "mae_before_meaningful_mfe": mae_before,
        "adverse_first": adverse_first,
        "execution_exit_reason": str(reason_names([int(reason[0])])[0]),
        "execution_exit_bar": local_exit_bar,
        "execution_exit_timestamp": post["timestamp"].iloc[local_exit_bar],
    }


def _simulate_action(
    path: pd.DataFrame,
    *,
    decision: pd.Timestamp,
    base_price: float,
    atr: float,
    side_sign: float,
    cost_return: float,
    entry_spread_bps: float,
    exit_spread_bps: float,
    action: EntryAction,
    geometry: ExecutionLabelGeometry,
    spec: EntryTimingTargetSpec,
) -> dict[str, Any]:
    """Simulate one future-path action without leaking it into model inputs."""

    if path.empty:
        return {"filled": False, "fill_time_minutes": np.nan, "fill_price": np.nan}
    fill_bar_is_ambiguous = False
    if action.kind == "enter_now":
        fill_position = 0
        raw_fill_price = base_price
    elif action.kind == "wait_market":
        deadline = decision + pd.Timedelta(minutes=int(action.wait_minutes))
        candidates = np.flatnonzero(path["timestamp"].to_numpy() >= deadline)
        if not len(candidates):
            return {"filled": False, "fill_time_minutes": np.nan, "fill_price": np.nan}
        fill_position = int(candidates[0])
        raw_fill_price = float(path["open"].iloc[fill_position])
    else:
        deadline = decision + pd.Timedelta(minutes=int(action.wait_minutes))
        limit_price = base_price - side_sign * float(action.adverse_offset_atr) * atr
        eligible = np.flatnonzero(path["timestamp"].to_numpy() <= deadline)
        if not len(eligible):
            return {"filled": False, "fill_time_minutes": np.nan, "fill_price": np.nan}
        if side_sign > 0.0:
            touched = path["low"].to_numpy(dtype=float)[eligible] <= limit_price
        else:
            touched = path["high"].to_numpy(dtype=float)[eligible] >= limit_price
        matched = eligible[np.flatnonzero(touched)]
        if not len(matched):
            return {"filled": False, "fill_time_minutes": np.nan, "fill_price": np.nan}
        fill_position = int(matched[0])
        # The label uses the stated limit price, never an optimistic intrabar
        # improvement inferred from OHLC ordering that is not observable.
        raw_fill_price = float(limit_price)
        # OHLC cannot order a passive touch against the same bar's favorable
        # excursion.  Exclude that whole bar and start the protected policy on
        # the next minute, retaining the fill price, rather than inventing an
        # optimistic intrabar sequence.
        fill_bar_is_ambiguous = True
    # Pay entry spread once: buys at ask, sells at bid.  This preserves the
    # changed passive-limit price while making the actual executable price
    # explicit in labels.
    fill_price = raw_fill_price * (1.0 + side_sign * float(entry_spread_bps) * 1e-4)
    fill_time = float(
        (path["timestamp"].iloc[fill_position] - decision).total_seconds() / 60.0
    )
    return {
        "filled": True,
        "fill_time_minutes": fill_time,
        "fill_price": fill_price,
        "raw_fill_price": raw_fill_price,
        "fill_bar_intrabar_ambiguity": fill_bar_is_ambiguous,
        "policy_simulation_start_position": int(
            fill_position + 1 if fill_bar_is_ambiguous else fill_position
        ),
        "policy_simulation_start_utc": path["timestamp"].iloc[
            fill_position + 1 if fill_bar_is_ambiguous else fill_position
        ]
        if (fill_position + 1 if fill_bar_is_ambiguous else fill_position) < len(path)
        else pd.NaT,
        **_geometry_aware_post_fill_outcome(
            path,
            policy_start_position=(
                fill_position + 1 if fill_bar_is_ambiguous else fill_position
            ),
            fill_price=fill_price,
            side_sign=side_sign,
            atr=atr,
            cost_return=cost_return,
            exit_spread_bps=exit_spread_bps,
            geometry=geometry,
            spec=spec,
        ),
    }


def build_counterfactual_entry_action_labels(
    frame: pd.DataFrame,
    *,
    target_spec: EntryTimingTargetSpec = EntryTimingTargetSpec(),
    action_grid: Sequence[EntryAction] | None = None,
    decision_time_col: str = "__decision_ts__",
    side_col: str = "side_name",
) -> pd.DataFrame:
    """Build train-only labels for enter-now, wait, and adverse-limit actions.

    Every path bar must begin strictly after the decision and stay within the
    bounded label horizon.  An unfilled action has zero realised action EV and
    pays its missed-opportunity label as the positive enter-now net EV.  This
    makes skipped attractive trades economically visible without rewarding a
    skip of a negative enter-now outcome.
    """

    grid = _validate_action_grid(action_grid or default_entry_action_grid())
    if decision_time_col != "__decision_ts__":
        raise ValueError(
            "counterfactual entry labels must use upstream execution-EV decision timestamp "
            "'__decision_ts__'"
        )
    if target_spec.horizon_hours <= 0.0:
        raise ValueError("entry action horizon_hours must be positive")
    if target_spec.path_frequency_minutes != 1:
        raise ValueError("counterfactual entry labels require exact one-minute paths")
    horizon_minutes = float(target_spec.horizon_hours) * 60.0
    if not horizon_minutes.is_integer():
        raise ValueError("counterfactual entry horizon must resolve to an exact number of minutes")
    expected_path_length = int(horizon_minutes)
    if expected_path_length < 1:
        raise ValueError("counterfactual entry horizon must contain at least one minute")
    if target_spec.meaningful_mfe_atr <= 0.0:
        raise ValueError("meaningful_mfe_atr must be positive")
    if target_spec.meaningful_mfe_return_floor < 0.0:
        raise ValueError("meaningful_mfe_return_floor must be non-negative")
    if target_spec.adverse_mae_atr < 0.0:
        raise ValueError("adverse_mae_atr must be non-negative")
    if target_spec.path_col not in frame.columns:
        raise ValueError(f"entry action labels require path column {target_spec.path_col!r}")
    geometries = {
        "long": _policy_geometry(target_spec, "long"),
        "short": _policy_geometry(target_spec, "short"),
    }
    decision = _utc(frame[decision_time_col], name=decision_time_col)
    sides = _side_values(frame, side_col)
    atr = _numeric(frame, target_spec.atr_col, role="decision ATR")
    if (atr <= 0.0).any():
        raise ValueError("entry action decision ATR must be strictly positive")
    decomposed_cost_columns = (
        target_spec.fee_return_col,
        target_spec.entry_spread_bps_col,
        target_spec.exit_spread_bps_col,
    )
    if target_spec.cost_return_col is not None and any(decomposed_cost_columns):
        raise ValueError(
            "entry action labels must use either all-in cost_return_col or separate "
            "fee/entry-spread/exit-spread fields, never both"
        )
    if target_spec.cost_return_col is not None:
        if not target_spec.allow_action_invariant_all_in_cost:
            raise ValueError(
                "all-in execution cost is rejected for counterfactual actions by default; "
                "provide decomposed fee_return_col, entry_spread_bps_col, and "
                "exit_spread_bps_col, or explicitly assert "
                "allow_action_invariant_all_in_cost=True"
            )
        fees = _numeric(frame, target_spec.cost_return_col, role="all-in execution cost")
        entry_spreads = np.zeros(len(frame), dtype=np.float64)
        exit_spreads = np.zeros(len(frame), dtype=np.float64)
        accounting_mode = "all_in_cost_return_once"
    elif any(decomposed_cost_columns):
        if not all(decomposed_cost_columns):
            raise ValueError(
                "separate execution accounting requires fee_return_col, entry_spread_bps_col, "
                "and exit_spread_bps_col together"
            )
        fees = _numeric(frame, str(target_spec.fee_return_col), role="execution fee")
        entry_spreads = _numeric(frame, str(target_spec.entry_spread_bps_col), role="entry spread bps")
        exit_spreads = _numeric(frame, str(target_spec.exit_spread_bps_col), role="exit spread bps")
        accounting_mode = "fee_once_entry_spread_once_exit_spread_once"
    else:
        raise ValueError(
            "counterfactual actions require decomposed fee_return_col, entry_spread_bps_col, "
            "and exit_spread_bps_col"
        )
    if (fees < 0.0).any() or (entry_spreads < 0.0).any() or (exit_spreads < 0.0).any():
        raise ValueError("entry action execution fee/spread fields must be non-negative")
    if target_spec.decision_price_col is None:
        base_prices = np.full(len(frame), np.nan, dtype=np.float64)
    else:
        base_prices = _numeric(frame, target_spec.decision_price_col, role="decision price")
        if (base_prices <= 0.0).any():
            raise ValueError("entry action decision prices must be strictly positive")
    records: list[dict[str, Any]] = []
    enter_now = next(action for action in grid if action.kind == "enter_now")
    for position, (row_index, row) in enumerate(frame.iterrows()):
        row_decision = decision.iloc[position]
        path = _decode_path(row[target_spec.path_col], target_spec, row=row_index)
        if path.empty:
            raise ValueError(f"counterfactual path is empty for row {row_index!r}")
        if int(len(path)) != expected_path_length:
            raise ValueError(
                "counterfactual labels require an exact fixed 1m horizon length of "
                f"{expected_path_length} bars for row {row_index!r}"
            )
        expected_start = row_decision + pd.Timedelta(minutes=1)
        expected_terminal = row_decision + pd.Timedelta(minutes=expected_path_length)
        if path["timestamp"].iloc[0] != expected_start:
            raise ValueError(
                "counterfactual fixed-1m path must begin exactly at the first executable "
                "minute after its decision"
            )
        if path["timestamp"].iloc[-1] != expected_terminal:
            raise ValueError(
                "counterfactual fixed-1m path must end exactly at its terminal horizon "
                f"timestamp for row {row_index!r}"
            )
        cadence = np.diff(path["timestamp"].astype("int64").to_numpy())
        if len(cadence) and not np.all(cadence == pd.Timedelta(minutes=1).value):
            raise ValueError("counterfactual path must have fixed one-minute cadence")
        base_price = (
            float(base_prices[position])
            if np.isfinite(base_prices[position])
            else (float(path["open"].iloc[0]) if not path.empty else np.nan)
        )
        if not np.isfinite(base_price) or base_price <= 0.0:
            raise ValueError(f"counterfactual path has no executable enter-now price for row {row_index!r}")
        sign = 1.0 if sides[position] == "long" else -1.0
        side_name = sides[position]
        simulations: dict[str, dict[str, Any]] = {}
        for action in grid:
            simulations[action.action_id] = _simulate_action(
                path,
                decision=row_decision,
                base_price=base_price,
                atr=float(atr[position]),
                side_sign=sign,
                cost_return=float(fees[position]),
                entry_spread_bps=float(entry_spreads[position]),
                exit_spread_bps=float(exit_spreads[position]),
                action=action,
                geometry=geometries[side_name],
                spec=target_spec,
            )
        now = simulations[enter_now.action_id]
        if not bool(now["filled"]):
            raise ValueError(f"enter_now must fill on every valid future path for row {row_index!r}")
        now_net_ev = float(now["post_fill_net_ev"])
        if not np.isfinite(now_net_ev):
            raise ValueError(
                f"enter_now has no complete geometry-aware policy outcome for row {row_index!r}"
            )
        for action_order, action in enumerate(grid):
            simulation = simulations[action.action_id]
            filled = bool(simulation["filled"])
            missed = 0.0 if filled else max(now_net_ev, 0.0)
            net_ev = float(simulation.get("post_fill_net_ev", np.nan)) if filled else np.nan
            gross_ev = float(simulation.get("post_fill_gross_ev", np.nan)) if filled else np.nan
            if filled and not (np.isfinite(net_ev) and np.isfinite(gross_ev)):
                raise ValueError(
                    f"filled action {action.action_id!r} has no complete geometry-aware "
                    f"policy outcome for row {row_index!r}"
                )
            records.append(
                {
                    "base_position": int(position),
                    "source_index": row_index,
                    "action_id": action.action_id,
                    "action_kind": action.kind,
                    "action_order": int(action_order),
                    "counterfactual_label_end_utc": path["timestamp"].iloc[-1],
                    "wait_minutes": int(action.wait_minutes),
                    "adverse_offset_atr": float(action.adverse_offset_atr),
                    "filled": filled,
                    "fill_indicator": float(filled),
                    "no_fill_indicator": float(not filled),
                    "fill_time_minutes": float(simulation.get("fill_time_minutes", np.nan)),
                    "fill_price": float(simulation.get("fill_price", np.nan)),
                    "raw_fill_price": float(simulation.get("raw_fill_price", np.nan)),
                    "fill_bar_intrabar_ambiguity": bool(
                        simulation.get("fill_bar_intrabar_ambiguity", False)
                    ),
                    "policy_simulation_start_position": int(
                        simulation.get("policy_simulation_start_position", -1)
                    ),
                    "policy_simulation_start_utc": simulation.get(
                        "policy_simulation_start_utc", pd.NaT
                    ),
                    "fee_return": float(fees[position]),
                    "entry_spread_bps": float(entry_spreads[position]),
                    "exit_spread_bps": float(exit_spreads[position]),
                    "cost_accounting_mode": accounting_mode,
                    "post_fill_gross_ev": gross_ev,
                    "post_fill_net_ev": net_ev,
                    "conditional_post_fill_executable_ev": net_ev,
                    "post_fill_mfe": float(simulation.get("post_fill_mfe", np.nan)),
                    "post_fill_mae": float(simulation.get("post_fill_mae", np.nan)),
                    "execution_exit_reason": simulation.get("execution_exit_reason"),
                    "execution_exit_bar": int(simulation.get("execution_exit_bar", -1)),
                    "execution_exit_timestamp": simulation.get(
                        "execution_exit_timestamp", pd.NaT
                    ),
                    "mae_before_meaningful_mfe": float(
                        simulation.get("mae_before_meaningful_mfe", np.nan)
                    ),
                    "adverse_first": bool(simulation.get("adverse_first", False)) if filled else False,
                    "missed_opportunity_ev": float(missed),
                    "missed_opportunity_loss": float(missed),
                    "action_realized_utility": float(net_ev) if filled else -float(missed),
                    "enter_now_net_ev": now_net_ev,
                    "filled_delta_ev_vs_now": (net_ev - now_net_ev) if filled else np.nan,
                }
            )
    labels = pd.DataFrame.from_records(records)
    for column in (
        "fill_time_minutes",
        "fill_price",
        "raw_fill_price",
        "fee_return",
        "entry_spread_bps",
        "exit_spread_bps",
        "post_fill_gross_ev",
        "post_fill_net_ev",
        "post_fill_mfe",
        "post_fill_mae",
        "mae_before_meaningful_mfe",
        "missed_opportunity_ev",
        "action_realized_utility",
        "enter_now_net_ev",
        "filled_delta_ev_vs_now",
    ):
        labels[column] = labels[column].astype("float32")
    labels["filled"] = labels["filled"].astype(bool)
    labels["fill_indicator"] = labels["fill_indicator"].astype("float32")
    labels["no_fill_indicator"] = labels["no_fill_indicator"].astype("float32")
    labels["adverse_first"] = labels["adverse_first"].astype(bool)
    labels["fill_bar_intrabar_ambiguity"] = labels[
        "fill_bar_intrabar_ambiguity"
    ].astype(bool)
    return labels


def entry_timing_realized_label_columns(
    target_spec: EntryTimingTargetSpec = EntryTimingTargetSpec(),
) -> set[str]:
    """Columns which must never be requested by the inference-only scorer."""

    return {
        target_spec.path_col,
        "filled",
        "fill_indicator",
        "no_fill_indicator",
        "fill_time_minutes",
        "fill_price",
        "raw_fill_price",
        "fill_bar_intrabar_ambiguity",
        "policy_simulation_start_position",
        "policy_simulation_start_utc",
        "post_fill_gross_ev",
        "post_fill_net_ev",
        "conditional_post_fill_executable_ev",
        "post_fill_mfe",
        "post_fill_mae",
        "execution_exit_reason",
        "execution_exit_bar",
        "execution_exit_timestamp",
        "mae_before_meaningful_mfe",
        "adverse_first",
        "missed_opportunity_ev",
        "missed_opportunity_loss",
        "action_realized_utility",
        "enter_now_net_ev",
        "filled_delta_ev_vs_now",
    }


def _default_lgbm_params(config: EntryTimingTrainerConfig, *, objective: str) -> dict[str, Any]:
    return {
        "objective": objective,
        "n_estimators": int(config.n_estimators),
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 24,
        "min_split_gain": 1e-3,
        "reg_alpha": 0.12,
        "reg_lambda": 5.0,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "max_bin": 127,
        "random_state": int(config.random_state),
        "feature_fraction_seed": int(config.random_state),
        "bagging_seed": int(config.random_state),
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": int(config.n_jobs),
        "verbosity": -1,
    }


def _fit_lgbm_classifier(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    config: EntryTimingTrainerConfig,
    params: Mapping[str, Any],
    eval_x: pd.DataFrame | None = None,
    eval_y: np.ndarray | None = None,
) -> Any:
    target = np.asarray(y, dtype=np.int8)
    if not len(target):
        return _ConstantClassifier(0.0)
    unique = np.unique(target)
    if len(unique) == 1:
        return _ConstantClassifier(float(unique[0]))
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LightGBM is required for entry timing primary heads") from exc
    model_params = _default_lgbm_params(config, objective="binary")
    model_params.update(dict(params))
    model_params["objective"] = "binary"
    model = lgb.LGBMClassifier(**model_params)
    kwargs: dict[str, Any] = {}
    callbacks: list[Any] = []
    if eval_x is not None and eval_y is not None and len(eval_x) and len(np.unique(eval_y)) > 1:
        kwargs["eval_set"] = [(eval_x, np.asarray(eval_y, dtype=np.int8))]
        callbacks = [lgb.early_stopping(int(config.early_stopping_rounds), verbose=False)]
    model.fit(x, target, callbacks=callbacks, **kwargs)
    return model


def _fit_lgbm_regressor(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    config: EntryTimingTrainerConfig,
    params: Mapping[str, Any],
    eval_x: pd.DataFrame | None = None,
    eval_y: np.ndarray | None = None,
) -> Any:
    target = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(target)
    if not finite.any():
        return _ConstantRegressor(0.0)
    x = x.iloc[np.flatnonzero(finite)]
    target = target[finite]
    if len(target) < 4 or float(np.nanstd(target)) <= 1e-12:
        return _ConstantRegressor(float(np.nanmean(target)))
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LightGBM is required for entry timing primary heads") from exc
    model_params = _default_lgbm_params(config, objective="regression_l1")
    model_params.update(dict(params))
    model_params["objective"] = "regression_l1"
    model = lgb.LGBMRegressor(**model_params)
    kwargs: dict[str, Any] = {}
    callbacks: list[Any] = []
    if eval_x is not None and eval_y is not None and len(eval_x) and np.isfinite(eval_y).sum() >= 4:
        kwargs["eval_set"] = [(eval_x, np.asarray(eval_y, dtype=np.float64))]
        callbacks = [lgb.early_stopping(int(config.early_stopping_rounds), verbose=False)]
    model.fit(x, target, callbacks=callbacks, **kwargs)
    return model


def _fit_logistic(x: pd.DataFrame, y: np.ndarray) -> Any:
    target = np.asarray(y, dtype=np.int8)
    if not len(target) or len(np.unique(target)) == 1:
        return _ConstantClassifier(float(target[0]) if len(target) else 0.0)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, solver="lbfgs", max_iter=500, random_state=42),
    )
    model.fit(x, target)
    return model


def _fit_ridge(x: pd.DataFrame, y: np.ndarray) -> Any:
    target = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(target)
    if not finite.any() or int(finite.sum()) < 4 or float(np.nanstd(target)) <= 1e-12:
        return _ConstantRegressor(float(np.nanmean(target)) if finite.any() else 0.0)
    model = make_pipeline(StandardScaler(), Ridge(alpha=8.0))
    model.fit(x.iloc[np.flatnonzero(finite)], target[finite])
    return model


def _probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    output = np.asarray(model.predict_proba(x), dtype=np.float64)
    if output.ndim != 2 or output.shape[1] < 2:
        raise ValueError("entry timing classifier did not return binary probabilities")
    return np.clip(output[:, 1], 0.0, 1.0)


def _fit_isotonic_calibrator(
    raw: Sequence[float], target: Sequence[float], *, probability: bool
) -> _IsotonicCalibrator:
    """Fit only on already OOF predictions from an authorized train window."""

    x = np.asarray(raw, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 20 or np.unique(x).size < 3:
        return _IsotonicCalibrator(probability=probability)
    if probability and np.unique(y.astype(np.int8)).size < 2:
        return _IsotonicCalibrator(probability=True)
    model = IsotonicRegression(
        y_min=0.0 if probability else None,
        y_max=1.0 if probability else None,
        out_of_bounds="clip",
        increasing=True,
    )
    model.fit(x, y)
    return _IsotonicCalibrator(model=model, probability=probability, status="isotonic_train_oof")


def _identity_isotonic() -> dict[str, _IsotonicCalibrator]:
    return {
        "fill": _IsotonicCalibrator(probability=True),
        "adverse": _IsotonicCalibrator(probability=True),
        "delta": _IsotonicCalibrator(),
        "expected_action_ev": _IsotonicCalibrator(),
    }


def _fit_train_oof_isotonic(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    config: EntryTimingTrainerConfig,
    lgbm_params: Mapping[str, Any],
    decision_policy: Mapping[str, Any],
    execution_ev_feature: str,
) -> tuple[dict[str, _IsotonicCalibrator], dict[str, Any]]:
    """Generate inner chronological OOF scores and fit maps on those only.

    The caller passes a single outer-training side/action slice.  Therefore no
    row in an outer scored fold, nor any of its path outcomes, can affect the
    calibration used to score that fold.
    """

    splits = _inner_splits(frame, config)
    if not splits:
        return _identity_isotonic(), {"status": "identity_no_authorized_inner_oof", "oof_rows": 0}
    rows: list[pd.DataFrame] = []
    for split in splits:
        state = _fit_action_state(
            x.iloc[split.train_indices].reset_index(drop=True),
            labels.iloc[split.train_indices].reset_index(drop=True),
            config=config,
            lgbm_params=lgbm_params,
        )
        valid_x = x.iloc[split.validation_indices]
        fill, adverse, delta = _prediction_from_state(state, valid_x, arm="lgbm")
        now = valid_x[execution_ev_feature].to_numpy(dtype=np.float64)
        expected, _, _ = _expected_action_utility(
            predicted_enter_now_ev=now,
            fill_probability=fill,
            adverse_probability=adverse,
            filled_delta_ev=delta,
            decision_policy=decision_policy,
        )
        target = labels.iloc[split.validation_indices].reset_index(drop=True)
        rows.append(
            pd.DataFrame(
                {
                    "raw_fill": fill,
                    "raw_adverse": adverse,
                    "raw_delta": delta,
                    "raw_expected_action_ev": expected,
                    "filled": target["filled"].to_numpy(dtype=float),
                    "adverse_first": target["adverse_first"].to_numpy(dtype=float),
                    "filled_delta_ev_vs_now": target["filled_delta_ev_vs_now"].to_numpy(dtype=float),
                    # Expected action utility includes an adverse-first cost.
                    "target_action_utility": target["action_realized_utility"].to_numpy(dtype=float)
                    - target["filled"].to_numpy(dtype=float)
                    * target["adverse_first"].to_numpy(dtype=float)
                    * float(decision_policy["adverse_first_penalty"]),
                }
            )
        )
    oof = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if oof.empty:
        return _identity_isotonic(), {"status": "identity_empty_inner_oof", "oof_rows": 0}
    filled = oof["filled"].to_numpy(dtype=bool)
    calibrators = {
        "fill": _fit_isotonic_calibrator(oof["raw_fill"], oof["filled"], probability=True),
        "adverse": _fit_isotonic_calibrator(
            oof.loc[filled, "raw_adverse"], oof.loc[filled, "adverse_first"], probability=True
        ),
        "delta": _fit_isotonic_calibrator(
            oof.loc[filled, "raw_delta"], oof.loc[filled, "filled_delta_ev_vs_now"], probability=False
        ),
        "expected_action_ev": _fit_isotonic_calibrator(
            oof["raw_expected_action_ev"], oof["target_action_utility"], probability=False
        ),
    }
    return calibrators, {
        "status": "inner_chronological_train_oof_isotonic",
        "oof_rows": int(len(oof)),
        "filled_oof_rows": int(filled.sum()),
        "component_status": {name: calibrator.status for name, calibrator in calibrators.items()},
    }


def _prediction_from_state(state: Mapping[str, Any], x: pd.DataFrame, *, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm == "lgbm":
        fill = _probability(state["lgbm_fill"], x)
        adverse = _probability(state["lgbm_adverse"], x)
        delta = np.asarray(state["lgbm_delta"].predict(x), dtype=np.float64)
    elif arm == "ridge_logistic":
        fill = _probability(state["ridge_fill"], x)
        adverse = _probability(state["ridge_adverse"], x)
        delta = np.asarray(state["ridge_delta"].predict(x), dtype=np.float64)
    elif arm == "fixed_grid":
        baseline = state["fixed_grid"]
        fill = np.full(len(x), float(baseline["fill_probability"]), dtype=np.float64)
        adverse = np.full(len(x), float(baseline["adverse_probability"]), dtype=np.float64)
        delta = np.full(len(x), float(baseline["filled_delta_ev"]), dtype=np.float64)
    else:
        raise ValueError(f"unsupported action-value arm {arm!r}")
    return (
        np.clip(fill, 0.0, 1.0),
        np.clip(adverse, 0.0, 1.0),
        np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _action_values_from_state(
    state: Mapping[str, Any],
    x: pd.DataFrame,
    *,
    action: EntryAction,
    arm: str,
    predicted_enter_now_ev: np.ndarray,
    decision_policy: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return raw and train-OOF-isotonic action components and expected EV."""

    now_ev = np.asarray(predicted_enter_now_ev, dtype=np.float64)
    if action.kind == "enter_now":
        # This layer is downstream of a cost/spread-aware execution-EV head.
        # Entering immediately is not a timing gamble: preserve that frozen EV
        # exactly and reserve fill/miss/adverse adjustments for delayed actions.
        zeros = np.zeros(len(x), dtype=np.float64)
        ones = np.ones(len(x), dtype=np.float64)
        return {
            "raw_fill": ones,
            "raw_adverse": zeros,
            "raw_delta": zeros,
            "raw_expected": now_ev,
            "raw_conditional": now_ev,
            "raw_expected_missed": zeros,
            "fill": ones,
            "adverse": zeros,
            "delta": zeros,
            "conditional": now_ev,
            "expected_missed": zeros,
            "expected": now_ev,
        }

    raw_fill, raw_adverse, raw_delta = _prediction_from_state(state, x, arm=arm)
    raw_expected, raw_missed, raw_conditional = _expected_action_utility(
        predicted_enter_now_ev=predicted_enter_now_ev,
        fill_probability=raw_fill,
        adverse_probability=raw_adverse,
        filled_delta_ev=raw_delta,
        decision_policy=decision_policy,
    )
    calibrators = state.get("isotonic", {}).get(arm, _identity_isotonic())
    fill = calibrators["fill"].predict(raw_fill)
    adverse = calibrators["adverse"].predict(raw_adverse)
    delta = calibrators["delta"].predict(raw_delta)
    expected_before_action_map, missed, conditional = _expected_action_utility(
        predicted_enter_now_ev=predicted_enter_now_ev,
        fill_probability=fill,
        adverse_probability=adverse,
        filled_delta_ev=delta,
        decision_policy=decision_policy,
    )
    expected = calibrators["expected_action_ev"].predict(expected_before_action_map)
    return {
        "raw_fill": raw_fill,
        "raw_adverse": raw_adverse,
        "raw_delta": raw_delta,
        "raw_expected": raw_expected,
        "raw_conditional": raw_conditional,
        "raw_expected_missed": raw_missed,
        "fill": fill,
        "adverse": adverse,
        "delta": delta,
        "conditional": conditional,
        "expected_missed": missed,
        "expected": expected,
    }


def _fit_action_state(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    config: EntryTimingTrainerConfig,
    lgbm_params: Mapping[str, Any],
) -> dict[str, Any]:
    fill = labels["filled"].to_numpy(dtype=np.int8)
    filled = fill == 1
    adverse = labels["adverse_first"].to_numpy(dtype=np.int8)
    delta = labels["filled_delta_ev_vs_now"].to_numpy(dtype=np.float64)
    filled_rows = np.flatnonzero(filled)
    if len(filled_rows):
        adverse_rate = float(np.mean(adverse[filled_rows]))
        delta_mean = float(np.nanmean(delta[filled_rows]))
    else:
        adverse_rate, delta_mean = 0.0, 0.0
    return {
        "lgbm_fill": _fit_lgbm_classifier(x, fill, config=config, params=lgbm_params),
        "lgbm_adverse": _fit_lgbm_classifier(
            x.iloc[filled_rows], adverse[filled_rows], config=config, params=lgbm_params
        ) if len(filled_rows) else _ConstantClassifier(0.0),
        "lgbm_delta": _fit_lgbm_regressor(
            x.iloc[filled_rows], delta[filled_rows], config=config, params=lgbm_params
        ) if len(filled_rows) else _ConstantRegressor(0.0),
        "ridge_fill": _fit_logistic(x, fill),
        "ridge_adverse": _fit_logistic(x.iloc[filled_rows], adverse[filled_rows])
        if len(filled_rows)
        else _ConstantClassifier(0.0),
        "ridge_delta": _fit_ridge(x.iloc[filled_rows], delta[filled_rows])
        if len(filled_rows)
        else _ConstantRegressor(0.0),
        "fixed_grid": {
            "fill_probability": float(np.mean(fill)) if len(fill) else 0.0,
            "adverse_probability": adverse_rate,
            "filled_delta_ev": delta_mean,
        },
        "isotonic": {
            "lgbm": _identity_isotonic(),
            "ridge_logistic": _identity_isotonic(),
            "fixed_grid": _identity_isotonic(),
        },
    }


def _action_matrix(labels: pd.DataFrame, action_grid: Sequence[EntryAction], rows: int) -> dict[str, pd.DataFrame]:
    matrix: dict[str, pd.DataFrame] = {}
    for action in action_grid:
        part = labels.loc[labels["action_id"] == action.action_id].sort_values("base_position")
        if len(part) != rows or not np.array_equal(part["base_position"].to_numpy(), np.arange(rows)):
            raise ValueError("entry timing action labels must cover every base row exactly once")
        matrix[action.action_id] = part.reset_index(drop=True)
    return matrix


def _inner_splits(
    frame: pd.DataFrame, config: EntryTimingTrainerConfig
) -> list[ChronologicalPurgedSplit]:
    minimum = max(4, min(config.min_train_rows, len(frame) // 3))
    if len(frame) < max(10, minimum + 2):
        return []
    try:
        return chronological_purged_splits(
            frame,
            n_splits=config.inner_n_splits,
            min_train_size=minimum,
            decision_time_col=config.decision_time_col,
            label_end_time_col=config.label_end_time_col,
            horizon_hours=config.purge_hours,
            embargo_hours=config.embargo_hours,
        )
    except ValueError:
        return []


def _tune_lgbm_params(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    matrix: Mapping[str, pd.DataFrame],
    *,
    reference_indices: np.ndarray,
    action_grid: Sequence[EntryAction],
    config: EntryTimingTrainerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune once on the earliest outer train window, then freeze parameters."""

    defaults = _default_lgbm_params(config, objective="binary")
    defaults.pop("objective", None)
    non_now_actions = [action for action in action_grid if action.kind != "enter_now"]
    if not non_now_actions:
        non_now_actions = [action_grid[0]]
    # Keep the HPO proxy bounded while representing both delayed market and
    # adverse-limit decisions. The same regularization is later frozen for
    # every side/action component head.
    representatives: list[EntryAction] = []
    for kind in ("wait_market", "adverse_limit"):
        candidates = [action for action in non_now_actions if action.kind == kind]
        if candidates:
            representatives.extend(
                [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
            )
    representatives = list(
        {action.action_id: action for action in representatives}.values()
    )[:4] or [non_now_actions[0]]
    local_frame = frame.iloc[reference_indices].reset_index(drop=True)
    local_x = x.iloc[reference_indices].reset_index(drop=True)
    splits = _inner_splits(local_frame, config)
    has_variable_fill = any(
        matrix[action.action_id]
        .iloc[reference_indices]["filled"]
        .astype("int8")
        .nunique()
        > 1
        for action in representatives
    )
    if config.hpo_trials <= 0 or not splits or not has_variable_fill:
        return defaults, {
            "status": "default_no_authorized_inner_hpo",
            "reference_actions": [action.action_id for action in representatives],
            "trials": 0,
        }
    try:
        import optuna
    except ImportError:  # pragma: no cover - optional dependency
        return defaults, {
            "status": "default_optuna_unavailable",
            "reference_actions": [action.action_id for action in representatives],
            "trials": 0,
        }

    def objective(trial: Any) -> float:
        params = {
            "min_child_samples": trial.suggest_int("min_child_samples", 12, 64, step=4),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 0.02, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.02, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 16.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        }
        losses: list[float] = []
        for split in splits:
            train = split.train_indices
            valid = split.validation_indices
            component_losses: list[float] = []
            for action in representatives:
                labels = (
                    matrix[action.action_id]
                    .iloc[reference_indices]
                    .reset_index(drop=True)
                )
                fill = labels["filled"].to_numpy(dtype=np.int8)
                fill_model = _fit_lgbm_classifier(
                    local_x.iloc[train],
                    fill[train],
                    config=config,
                    params=params,
                    eval_x=local_x.iloc[valid],
                    eval_y=fill[valid],
                )
                fill_probability = _probability(fill_model, local_x.iloc[valid])
                fill_loss = float(brier_score_loss(fill[valid], fill_probability))

                train_filled = train[fill[train].astype(bool)]
                valid_filled = valid[fill[valid].astype(bool)]
                adverse_loss = 0.0
                delta_loss = 0.0
                if len(train_filled) >= 8 and len(valid_filled) >= 4:
                    adverse = labels["adverse_first"].to_numpy(dtype=np.int8)
                    adverse_model = _fit_lgbm_classifier(
                        local_x.iloc[train_filled],
                        adverse[train_filled],
                        config=config,
                        params=params,
                        eval_x=local_x.iloc[valid_filled],
                        eval_y=adverse[valid_filled],
                    )
                    adverse_probability = _probability(
                        adverse_model, local_x.iloc[valid_filled]
                    )
                    adverse_loss = float(
                        brier_score_loss(
                            adverse[valid_filled], adverse_probability
                        )
                    )
                    delta = labels["filled_delta_ev_vs_now"].to_numpy(
                        dtype=np.float64
                    )
                    delta_model = _fit_lgbm_regressor(
                        local_x.iloc[train_filled],
                        delta[train_filled],
                        config=config,
                        params=params,
                        eval_x=local_x.iloc[valid_filled],
                        eval_y=delta[valid_filled],
                    )
                    delta_prediction = np.asarray(
                        delta_model.predict(local_x.iloc[valid_filled]),
                        dtype=np.float64,
                    )
                    train_delta = delta[train_filled]
                    scale = max(
                        float(
                            np.nanpercentile(train_delta, 75)
                            - np.nanpercentile(train_delta, 25)
                        ),
                        0.0025,
                    )
                    delta_loss = float(
                        np.mean(
                            np.abs(
                                delta_prediction - delta[valid_filled]
                            )
                        )
                        / scale
                    )
                component_losses.append(
                    0.35 * fill_loss
                    + 0.25 * adverse_loss
                    + 0.40 * delta_loss
                )
            losses.append(float(np.mean(component_losses)))
            trial.report(float(np.mean(losses)), step=len(losses) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(seed=int(config.random_state), multivariate=False)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=min(3, int(config.hpo_trials)), n_warmup_steps=1
        ),
    )
    study.optimize(
        objective,
        n_trials=int(config.hpo_trials),
        timeout=config.hpo_timeout_seconds,
        show_progress_bar=False,
    )
    best = {**defaults, **study.best_params}
    return best, {
        "status": "frozen_earliest_outer_train_inner_hpo",
        "reference_actions": [action.action_id for action in representatives],
        "trials": int(len(study.trials)),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }


def _allowed_actions(
    action_grid: Sequence[EntryAction], decision_policy: Mapping[str, Any]
) -> list[EntryAction]:
    max_wait = int(decision_policy["max_wait_minutes"])
    max_offset = float(decision_policy["max_adverse_offset_atr"])
    allowed = [
        action
        for action in action_grid
        if action.kind == "enter_now"
        or (
            action.kind == "wait_market" and action.wait_minutes <= max_wait
        )
        or (
            action.kind == "adverse_limit"
            and action.wait_minutes <= max_wait
            and action.adverse_offset_atr <= max_offset + 1e-12
        )
    ]
    if not any(action.kind == "enter_now" for action in allowed):
        raise ValueError("frozen decision policy cannot remove enter_now")
    return allowed


def _expected_action_utility(
    *,
    predicted_enter_now_ev: np.ndarray,
    fill_probability: np.ndarray,
    adverse_probability: np.ndarray,
    filled_delta_ev: np.ndarray,
    decision_policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return action value and its explicit fill/miss/risk components.

    ``E[action] = P(fill) * (EV_now + delta_EV)
                 - (1-P(fill)) * missed_penalty * max(EV_now, 0)
                 - P(fill) * P(adverse_first) * adverse_penalty``.
    """

    fill = np.clip(np.asarray(fill_probability, dtype=np.float64), 0.0, 1.0)
    adverse = np.clip(np.asarray(adverse_probability, dtype=np.float64), 0.0, 1.0)
    now_ev = np.asarray(predicted_enter_now_ev, dtype=np.float64)
    conditional = now_ev + np.asarray(filled_delta_ev, dtype=np.float64)
    missed_if_unfilled = np.maximum(now_ev, 0.0)
    expected_missed = (1.0 - fill) * float(decision_policy["missed_opportunity_penalty"]) * missed_if_unfilled
    adverse_cost = fill * adverse * float(decision_policy["adverse_first_penalty"])
    expected = fill * conditional - expected_missed - adverse_cost
    return expected, expected_missed, conditional


def _fixed_policy_defaults(action_grid: Sequence[EntryAction], config: EntryTimingTrainerConfig) -> dict[str, Any]:
    waits = [action.wait_minutes for action in action_grid if action.kind != "enter_now"]
    offsets = [action.adverse_offset_atr for action in action_grid if action.kind == "adverse_limit"]
    return {
        "missed_opportunity_penalty": float(config.missed_opportunity_penalty),
        "adverse_first_penalty": float(config.adverse_first_penalty),
        "max_wait_minutes": int(max(waits, default=0)),
        "max_adverse_offset_atr": float(max(offsets, default=0.0)),
    }


def _choose_action_from_values(
    matrix: Mapping[str, pd.DataFrame],
    *,
    positions: np.ndarray,
    predicted_enter_now_ev: np.ndarray,
    states: Mapping[str, Mapping[str, Any]],
    action_grid: Sequence[EntryAction],
    decision_policy: Mapping[str, Any],
    arm: str,
    x: pd.DataFrame | None,
) -> np.ndarray:
    """Choose stable max-value actions, using only current model predictions."""

    allowed = _allowed_actions(action_grid, decision_policy)
    values = np.full((len(positions), len(allowed)), -np.inf, dtype=np.float64)
    for action_position, action in enumerate(allowed):
        state = states[action.action_id]
        if x is None:
            baseline = state["fixed_grid"]
            fill = np.full(len(positions), baseline["fill_probability"], dtype=float)
            adverse = np.full(len(positions), baseline["adverse_probability"], dtype=float)
            delta = np.full(len(positions), baseline["filled_delta_ev"], dtype=float)
        else:
            fill, adverse, delta = _prediction_from_state(state, x, arm=arm)
        expected, _, _ = _expected_action_utility(
            predicted_enter_now_ev=predicted_enter_now_ev,
            fill_probability=fill,
            adverse_probability=adverse,
            filled_delta_ev=delta,
            decision_policy=decision_policy,
        )
        values[:, action_position] = expected
    selected = np.argmax(values, axis=1)
    return np.asarray([allowed[value] .action_id for value in selected], dtype=object)


def _tune_decision_policy(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    matrix: Mapping[str, pd.DataFrame],
    *,
    reference_indices: np.ndarray,
    action_grid: Sequence[EntryAction],
    execution_ev_feature: str,
    config: EntryTimingTrainerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose penalties/grid only through inner OOS fixed-grid decisions."""

    defaults = _fixed_policy_defaults(action_grid, config)
    local_frame = frame.iloc[reference_indices].reset_index(drop=True)
    local_x = x.iloc[reference_indices].reset_index(drop=True)
    local_matrix = {
        action_id: labels.iloc[reference_indices].reset_index(drop=True)
        for action_id, labels in matrix.items()
    }
    splits = _inner_splits(local_frame, config)
    if config.decision_hpo_trials <= 0 or not splits:
        return defaults, {"status": "default_no_authorized_inner_decision_hpo", "trials": 0}
    try:
        import optuna
    except ImportError:  # pragma: no cover - optional dependency
        return defaults, {"status": "default_optuna_unavailable", "trials": 0}
    waits = sorted({action.wait_minutes for action in action_grid if action.kind != "enter_now"}) or [0]
    offsets = sorted({action.adverse_offset_atr for action in action_grid if action.kind == "adverse_limit"}) or [0.0]

    def objective(trial: Any) -> float:
        policy = {
            "missed_opportunity_penalty": trial.suggest_float("missed_opportunity_penalty", 0.5, 1.5),
            "adverse_first_penalty": trial.suggest_float("adverse_first_penalty", 0.0, 0.02),
            "max_wait_minutes": trial.suggest_categorical("max_wait_minutes", waits),
            "max_adverse_offset_atr": trial.suggest_categorical("max_adverse_offset_atr", offsets),
        }
        results: list[float] = []
        allowed = _allowed_actions(action_grid, policy)
        for split in splits:
            train, valid = split.train_indices, split.validation_indices
            values = np.full((len(valid), len(allowed)), -np.inf, dtype=float)
            for action_position, action in enumerate(allowed):
                labels = local_matrix[action.action_id]
                filled = labels["filled"].to_numpy(dtype=bool)
                fill_probability = float(np.mean(filled[train]))
                filled_train = train[filled[train]]
                adverse = labels["adverse_first"].to_numpy(dtype=float)
                delta = labels["filled_delta_ev_vs_now"].to_numpy(dtype=float)
                adverse_probability = float(np.mean(adverse[filled_train])) if len(filled_train) else 0.0
                delta_ev = float(np.nanmean(delta[filled_train])) if len(filled_train) else 0.0
                expected, _, _ = _expected_action_utility(
                    predicted_enter_now_ev=local_x[execution_ev_feature].iloc[valid].to_numpy(dtype=float),
                    fill_probability=np.full(len(valid), fill_probability),
                    adverse_probability=np.full(len(valid), adverse_probability),
                    filled_delta_ev=np.full(len(valid), delta_ev),
                    decision_policy=policy,
                )
                values[:, action_position] = expected
            chosen = np.argmax(values, axis=1)
            actual = np.vstack(
                [
                    local_matrix[action.action_id]["action_realized_utility"].iloc[valid].to_numpy(dtype=float)
                    for action in allowed
                ]
            ).T
            results.append(float(np.mean(actual[np.arange(len(valid)), chosen])))
            trial.report(float(-np.mean(results)), step=len(results) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return -float(np.mean(results))

    sampler = optuna.samplers.TPESampler(seed=int(config.random_state) + 1, multivariate=False)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=min(3, int(config.decision_hpo_trials)), n_warmup_steps=1
        ),
    )
    study.optimize(objective, n_trials=int(config.decision_hpo_trials), show_progress_bar=False)
    policy = {**defaults, **study.best_params}
    policy["max_wait_minutes"] = int(policy["max_wait_minutes"])
    policy["max_adverse_offset_atr"] = float(policy["max_adverse_offset_atr"])
    return policy, {
        "status": "frozen_earliest_outer_train_inner_oos_decision_hpo",
        "trials": int(len(study.trials)),
        "best_value_negative_realized_utility": float(study.best_value),
        "best_params": dict(policy),
    }


def _oof_provenance(
    frame: pd.DataFrame,
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    decision_time_col: str,
    provenance: Mapping[str, EntryTimingFeatureProvenance],
) -> pd.DataFrame:
    decision = _utc(frame[decision_time_col], name=decision_time_col)
    result = pd.DataFrame(
        {
            "entry_timing_oof_fold": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
            "entry_timing_oof_validation_start_utc": pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
            "entry_timing_oof_train_decision_cutoff_utc": pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
        }
    )
    for split in folds:
        valid = split.validation_indices
        result.iloc[valid, result.columns.get_loc("entry_timing_oof_fold")] = int(split.fold)
        result.iloc[valid, result.columns.get_loc("entry_timing_oof_validation_start_utc")] = split.validation_start
        result.iloc[valid, result.columns.get_loc("entry_timing_oof_train_decision_cutoff_utc")] = decision.iloc[split.train_indices].max()
    for name, spec in provenance.items():
        if spec.family not in PREDICTIVE_ENTRY_TIMING_FEATURE_FAMILIES:
            continue
        assert spec.oof_fold_col is not None and spec.source_train_cutoff_col is not None
        result[f"source_{name}_oof_fold"] = frame[spec.oof_fold_col].astype("string").to_numpy()
        result[f"source_{name}_train_decision_cutoff_utc"] = _utc(
            frame[spec.source_train_cutoff_col], name=spec.source_train_cutoff_col
        ).to_numpy()
    return result


def _assert_source_oof_compatible_with_outer_folds(
    frame: pd.DataFrame,
    provenance: Mapping[str, EntryTimingFeatureProvenance],
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    decision_time_col: str,
) -> None:
    """Ensure stacked predictive inputs are no newer than each timing OOF fold."""

    decision = _utc(frame[decision_time_col], name=decision_time_col)
    predictive = [
        (name, spec)
        for name, spec in provenance.items()
        if spec.model_input and spec.family in PREDICTIVE_ENTRY_TIMING_FEATURE_FAMILIES
    ]
    for split in folds:
        outer_train_cutoff = decision.iloc[split.train_indices].max()
        outer_train_cutoff_ns = int(outer_train_cutoff.value)
        for name, spec in predictive:
            assert spec.source_train_cutoff_col is not None
            source_cutoff = _utc(
                frame[spec.source_train_cutoff_col], name=spec.source_train_cutoff_col
            )
            scored = source_cutoff.iloc[split.validation_indices]
            if (scored.astype("int64").to_numpy() > outer_train_cutoff_ns).any():
                raise ValueError(
                    f"entry timing predictive input {name!r} source train cutoff is not "
                    f"compatible with outer OOF fold {split.fold}: it must be no later than "
                    "the outer training cutoff"
                )


def _prediction_columns_for_arm(arm: str) -> dict[str, str]:
    return {
        "raw_fill": f"oof_{arm}_raw_fill_probability",
        "raw_adverse": f"oof_{arm}_raw_adverse_first_probability",
        "raw_delta": f"oof_{arm}_raw_filled_delta_ev",
        "raw_expected": f"oof_{arm}_raw_expected_action_ev",
        "fill": f"oof_{arm}_fill_probability",
        "adverse": f"oof_{arm}_adverse_first_probability",
        "delta": f"oof_{arm}_filled_delta_ev",
        "conditional": f"oof_{arm}_conditional_filled_ev",
        "expected_missed": f"oof_{arm}_expected_missed_ev",
        "expected": f"oof_{arm}_expected_action_ev",
    }


def _recommendations_from_oof(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    config: EntryTimingTrainerConfig,
    decision_policy: Mapping[str, Any],
    oof_provenance: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame(index=frame.index)
    base["side"] = _side_values(frame, config.side_col)
    base["archetype"] = frame[config.archetype_col].astype(str).to_numpy()
    timestamp = _utc(frame[config.decision_time_col], name=config.decision_time_col)
    base["month"] = timestamp.dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy()
    base["week"] = timestamp.dt.tz_localize(None).dt.to_period("W-MON").astype(str).to_numpy()
    base = base.join(oof_provenance)
    enter_now = actions.loc[actions["action_kind"].eq("enter_now")].set_index("base_position")
    for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
        columns = _prediction_columns_for_arm(arm)
        recommendations: list[dict[str, Any]] = []
        for position in range(len(frame)):
            part = actions.loc[actions["base_position"] == position]
            allowed = _allowed_actions(
                [
                    EntryAction(
                        str(row.action_kind), int(row.wait_minutes), float(row.adverse_offset_atr)
                    )
                    for row in part.sort_values("action_order").itertuples(index=False)
                ],
                decision_policy,
            )
            candidates = part.loc[part["action_id"].isin([action.action_id for action in allowed])].copy()
            expected = candidates[columns["expected"]].to_numpy(dtype=float)
            if not np.isfinite(expected).any():
                recommendations.append({})
                continue
            selected = candidates.iloc[int(np.nanargmax(expected))]
            recommendations.append(
                {
                    f"oof_{arm}_recommended_action_id": selected["action_id"],
                    f"oof_{arm}_recommended_wait_minutes": int(selected["wait_minutes"]),
                    f"oof_{arm}_recommended_adverse_offset_atr": float(selected["adverse_offset_atr"]),
                    f"oof_{arm}_raw_expected_action_ev": float(selected[columns["raw_expected"]]),
                    f"oof_{arm}_expected_action_ev": float(selected[columns["expected"]]),
                    f"oof_{arm}_fill_probability": float(selected[columns["fill"]]),
                    f"oof_{arm}_miss_probability": float(1.0 - selected[columns["fill"]]),
                    f"oof_{arm}_adverse_first_probability": float(selected[columns["adverse"]]),
                    f"oof_{arm}_expected_missed_ev": float(selected[columns["expected_missed"]]),
                    f"oof_{arm}_realized_action_utility": float(selected["action_realized_utility"]),
                    f"oof_{arm}_realized_fill": bool(selected["filled"]),
                    f"oof_{arm}_missed_profitable_trade": bool(
                        (not bool(selected["filled"])) and float(selected["missed_opportunity_ev"]) > 0.0
                    ),
                    f"oof_{arm}_realized_adverse_first": bool(selected["adverse_first"]),
                    f"oof_{arm}_realized_post_fill_mfe": float(selected["post_fill_mfe"]),
                    f"oof_{arm}_realized_post_fill_mae": float(selected["post_fill_mae"]),
                    f"oof_{arm}_enter_now_ev": float(selected["enter_now_net_ev"]),
                }
            )
        recommendation_frame = pd.DataFrame(recommendations, index=frame.index)
        base = base.join(recommendation_frame)
        oracle = actions.groupby("base_position", sort=False)["action_realized_utility"].max()
        base[f"oof_{arm}_oracle_utility"] = oracle.reindex(np.arange(len(frame))).to_numpy(dtype=float)
        base[f"oof_{arm}_regret_vs_oracle"] = (
            base[f"oof_{arm}_oracle_utility"] - base[f"oof_{arm}_realized_action_utility"]
        )
        base[f"oof_{arm}_enter_now_mfe"] = enter_now["post_fill_mfe"].reindex(np.arange(len(frame))).to_numpy(dtype=float)
        base[f"oof_{arm}_enter_now_mae"] = enter_now["post_fill_mae"].reindex(np.arange(len(frame))).to_numpy(dtype=float)
    return base


def _metric_row(part: pd.DataFrame, *, arm: str, scope: str, value: str | None = None) -> dict[str, Any]:
    prefix = f"oof_{arm}_"
    utility = part[f"{prefix}realized_action_utility"].to_numpy(dtype=float)
    now = part[f"{prefix}enter_now_ev"].to_numpy(dtype=float)
    valid = np.isfinite(utility) & np.isfinite(now)
    filled = part.loc[valid, f"{prefix}realized_fill"].to_numpy(dtype=bool)
    adverse = part.loc[valid, f"{prefix}realized_adverse_first"].to_numpy(dtype=bool)
    missed_profitable = part.loc[valid, f"{prefix}missed_profitable_trade"].to_numpy(dtype=bool)
    mfe = part.loc[valid, f"{prefix}realized_post_fill_mfe"].to_numpy(dtype=float)
    mae = part.loc[valid, f"{prefix}realized_post_fill_mae"].to_numpy(dtype=float)
    regret = part.loc[valid, f"{prefix}regret_vs_oracle"].to_numpy(dtype=float)
    expected = part.loc[valid, f"{prefix}expected_action_ev"].to_numpy(dtype=float)
    raw_expected = part.loc[valid, f"{prefix}raw_expected_action_ev"].to_numpy(dtype=float)
    now_mfe = part.loc[valid, f"{prefix}enter_now_mfe"].to_numpy(dtype=float)
    now_mae = part.loc[valid, f"{prefix}enter_now_mae"].to_numpy(dtype=float)
    mfe_retained = np.divide(mfe, now_mfe, out=np.full(len(mfe), np.nan), where=np.isfinite(now_mfe) & (now_mfe > 1e-12))
    mae_reduction = now_mae - mae
    return {
        "arm": arm,
        "scope": scope,
        "scope_value": value,
        "rows": int(valid.sum()),
        "action_ev_mean": float(np.mean(utility[valid])) if valid.any() else np.nan,
        "enter_now_ev_mean": float(np.mean(now[valid])) if valid.any() else np.nan,
        "action_ev_vs_enter_now": float(np.mean(utility[valid] - now[valid])) if valid.any() else np.nan,
        "expected_action_ev_mean": float(np.mean(expected)) if len(expected) else np.nan,
        "raw_action_ev_bias": float(np.mean(raw_expected - utility[valid])) if len(raw_expected) else np.nan,
        "isotonic_action_ev_bias": float(np.mean(expected - utility[valid])) if len(expected) else np.nan,
        "raw_action_ev_mae": float(np.mean(np.abs(raw_expected - utility[valid])) if len(raw_expected) else np.nan),
        "isotonic_action_ev_mae": float(np.mean(np.abs(expected - utility[valid])) if len(expected) else np.nan),
        "fill_rate": float(np.mean(filled)) if len(filled) else np.nan,
        "miss_rate": float(1.0 - np.mean(filled)) if len(filled) else np.nan,
        "missed_profitable_trade_rate": float(np.mean(missed_profitable)) if len(missed_profitable) else np.nan,
        "adverse_first_rate": float(np.mean(adverse[filled])) if filled.any() else np.nan,
        "post_entry_mfe_mean": float(np.nanmean(mfe[filled])) if filled.any() else np.nan,
        "post_entry_mae_mean": float(np.nanmean(mae[filled])) if filled.any() else np.nan,
        "mae_reduction": float(np.nanmean(mae_reduction)) if np.isfinite(mae_reduction).any() else np.nan,
        "mfe_retained": float(np.nanmean(mfe_retained)) if np.isfinite(mfe_retained).any() else np.nan,
        "regret_vs_oracle_mean": float(np.nanmean(regret)) if len(regret) else np.nan,
        "action_distribution": np.nan,
    }


def execution_entry_timing_metrics(recommendations: pd.DataFrame) -> pd.DataFrame:
    """OOF-only action economics overall, by side/archetype, fold, and month."""

    rows: list[dict[str, Any]] = []
    eligible = recommendations.loc[recommendations["entry_timing_oof_fold"].notna()].copy()
    for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
        rows.append(_metric_row(eligible, arm=arm, scope="overall"))
        rows.extend(
            _metric_row(part, arm=arm, scope="side", value=str(value))
            for value, part in eligible.groupby("side", observed=True, sort=True)
        )
        rows.extend(
            _metric_row(part, arm=arm, scope="archetype", value=str(value))
            for value, part in eligible.groupby("archetype", observed=True, sort=True)
        )
        rows.extend(
            _metric_row(part, arm=arm, scope="fold", value=str(int(value)))
            for value, part in eligible.groupby("entry_timing_oof_fold", observed=True, sort=True)
        )
        rows.extend(
            _metric_row(part, arm=arm, scope="month", value=str(value))
            for value, part in eligible.groupby("month", observed=True, sort=True)
        )
        rows.extend(
            _metric_row(part, arm=arm, scope="week", value=str(value))
            for value, part in eligible.groupby("week", observed=True, sort=True)
        )
        action_column = f"oof_{arm}_recommended_action_id"
        action_rows = [
            _metric_row(part, arm=arm, scope="action_distribution", value=str(value))
            for value, part in eligible.groupby(action_column, observed=True, sort=True)
        ]
        for row in action_rows:
            row["action_distribution"] = float(row["rows"] / max(len(eligible), 1))
        rows.extend(action_rows)
    return pd.DataFrame(rows)


def _worst_scope_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
        for scope in ("week", "month"):
            subset = metrics.loc[(metrics["arm"] == arm) & (metrics["scope"] == scope)]
            subset = subset.loc[np.isfinite(subset["action_ev_vs_enter_now"])]
            if subset.empty:
                report[f"{arm}_worst_{scope}"] = None
                continue
            worst = subset.sort_values("action_ev_vs_enter_now", kind="stable").iloc[0]
            report[f"{arm}_worst_{scope}"] = {
                "scope_value": str(worst["scope_value"]),
                "action_ev_vs_enter_now": float(worst["action_ev_vs_enter_now"]),
            }
    return report


def _config_payload(config: EntryTimingTrainerConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["action_grid"] = [asdict(action) for action in config.action_grid]
    return payload


def _config_from_payload(payload: Mapping[str, Any]) -> EntryTimingTrainerConfig:
    values = dict(payload)
    values["action_grid"] = tuple(EntryAction(**action) for action in values["action_grid"])
    return EntryTimingTrainerConfig(**values)


def _canonical_value(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item().__repr__()
    if isinstance(value, np.ndarray):
        return json.dumps([_canonical_value(item) for item in value.tolist()], separators=(",", ":"))
    if isinstance(value, Mapping):
        return json.dumps({str(k): _canonical_value(v) for k, v in sorted(value.items())}, sort_keys=True, separators=(",", ":"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return json.dumps([_canonical_value(item) for item in value], separators=(",", ":"))
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "<NA>"
    return repr(value)


def exact_entry_timing_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Stream an exact SHA256 over ordered training/inference contract columns."""

    digest = hashlib.sha256()
    digest.update(json.dumps(list(columns), separators=(",", ":")).encode("utf-8"))
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"fingerprint is missing contract column {column!r}")
        digest.update(str(column).encode("utf-8"))
        digest.update(str(frame[column].dtype).encode("utf-8"))
        for value in frame[column].tolist():
            digest.update(_canonical_value(value).encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _fingerprint_columns(
    config: EntryTimingTrainerConfig,
    target_spec: EntryTimingTargetSpec,
    provenance: Mapping[str, EntryTimingFeatureProvenance],
) -> list[str]:
    columns = [config.decision_time_col, config.side_col, config.archetype_col]
    if config.label_end_time_col:
        columns.append(config.label_end_time_col)
    columns.extend(name for name, spec in provenance.items() if spec.model_input)
    columns.extend(
        spec.available_at_col
        for spec in provenance.values()
        if spec.model_input and spec.available_at_col is not None
    )
    for spec in provenance.values():
        if not spec.model_input:
            continue
        if spec.oof_fold_col is not None:
            columns.append(spec.oof_fold_col)
        if spec.source_train_cutoff_col is not None:
            columns.append(spec.source_train_cutoff_col)
    columns.extend([target_spec.path_col, target_spec.atr_col])
    if target_spec.cost_return_col:
        columns.append(target_spec.cost_return_col)
    for column in (
        target_spec.fee_return_col,
        target_spec.entry_spread_bps_col,
        target_spec.exit_spread_bps_col,
    ):
        if column:
            columns.append(column)
    if target_spec.decision_price_col:
        columns.append(target_spec.decision_price_col)
    return list(dict.fromkeys(columns))


def _final_models(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    matrix: Mapping[str, pd.DataFrame],
    *,
    sides: np.ndarray,
    action_grid: Sequence[EntryAction],
    config: EntryTimingTrainerConfig,
    lgbm_params: Mapping[str, Any],
    decision_policy: Mapping[str, Any],
    execution_ev_feature: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    models: dict[str, dict[str, dict[str, Any]]] = {"long": {}, "short": {}}
    for side in ("long", "short"):
        positions = np.flatnonzero(sides == side)
        if not len(positions):
            continue
        for action in action_grid:
            local_x = x.iloc[positions].reset_index(drop=True)
            local_labels = matrix[action.action_id].iloc[positions].reset_index(drop=True)
            state = _fit_action_state(
                local_x,
                local_labels,
                config=config,
                lgbm_params=lgbm_params,
            )
            calibrators, _ = _fit_train_oof_isotonic(
                frame.iloc[positions].reset_index(drop=True),
                local_x,
                local_labels,
                config=config,
                lgbm_params=lgbm_params,
                decision_policy=decision_policy,
                execution_ev_feature=execution_ev_feature,
            )
            state["isotonic"]["lgbm"] = calibrators
            models[side][action.action_id] = state
    return models


def train_execution_entry_timing_meta(
    frame: pd.DataFrame,
    provenance: Mapping[str, EntryTimingFeatureProvenance],
    *,
    config: EntryTimingTrainerConfig = EntryTimingTrainerConfig(),
    target_spec: EntryTimingTargetSpec = EntryTimingTargetSpec(),
) -> ExecutionEntryTimingBundle:
    """Fit side-local OOF action heads and a separate final scoring bundle.

    The final refit uses all authorised rows but never writes into the OOF
    recommendation/metric tables.  Those tables consist solely of outer-fold
    predictions generated from an earlier, purged training cutoff.
    """

    action_grid = _validate_action_grid(config.action_grid)
    feature_names, execution_ev_feature = validate_entry_timing_feature_contract(
        frame, provenance, config=config
    )
    labels = build_counterfactual_entry_action_labels(
        frame,
        target_spec=target_spec,
        action_grid=action_grid,
        decision_time_col=config.decision_time_col,
        side_col=config.side_col,
    )
    matrix = _action_matrix(labels, action_grid, len(frame))
    enter_now = next(action for action in action_grid if action.kind == "enter_now")
    split_frame = frame.copy()
    split_config = config
    if config.label_end_time_col is None:
        label_end = matrix[enter_now.action_id]["counterfactual_label_end_utc"].to_numpy()
        split_frame["__entry_timing_counterfactual_label_end_utc"] = label_end
        split_config = replace(
            config,
            label_end_time_col="__entry_timing_counterfactual_label_end_utc",
        )
    folds = chronological_purged_splits(
        split_frame,
        n_splits=split_config.n_splits,
        min_train_size=split_config.min_train_rows,
        decision_time_col=split_config.decision_time_col,
        label_end_time_col=split_config.label_end_time_col,
        horizon_hours=split_config.purge_hours,
        embargo_hours=split_config.embargo_hours,
    )
    if not folds:
        raise ValueError("entry timing has no valid purged expanding OOF folds")
    _assert_source_oof_compatible_with_outer_folds(
        frame,
        provenance,
        folds,
        decision_time_col=config.decision_time_col,
    )
    x = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").astype("float32")
    if not np.isfinite(x.to_numpy(dtype=np.float32, copy=False)).all():
        raise ValueError("entry timing model matrix contains non-finite values")
    sides = _side_values(frame, config.side_col)
    lgbm_params, model_hpo = _tune_lgbm_params(
        split_frame,
        x,
        matrix,
        reference_indices=folds[0].train_indices,
        action_grid=action_grid,
        config=split_config,
    )
    decision_policy, decision_hpo = _tune_decision_policy(
        split_frame,
        x,
        matrix,
        reference_indices=folds[0].train_indices,
        action_grid=action_grid,
        execution_ev_feature=execution_ev_feature,
        config=split_config,
    )
    allowed_ids = {action.action_id for action in _allowed_actions(action_grid, decision_policy)}
    action_predictions = labels.copy()
    for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
        for column in _prediction_columns_for_arm(arm).values():
            action_predictions[column] = np.nan
    audits: list[dict[str, Any]] = []
    for fold in folds:
        for side in ("long", "short"):
            train_positions = np.asarray(
                [position for position in fold.train_indices if sides[position] == side], dtype=int
            )
            valid_positions = np.asarray(
                [position for position in fold.validation_indices if sides[position] == side], dtype=int
            )
            if not len(valid_positions):
                continue
            if len(train_positions) < 4:
                audits.append(
                    {
                        "fold": int(fold.fold),
                        "side": side,
                        "status": "insufficient_side_train_rows",
                        "train_rows": int(len(train_positions)),
                        "validation_rows": int(len(valid_positions)),
                    }
                )
                continue
            train_frame = split_frame.iloc[train_positions].reset_index(drop=True)
            states: dict[str, dict[str, Any]] = {}
            calibration_audit: dict[str, Any] = {}
            for action in action_grid:
                train_x = x.iloc[train_positions].reset_index(drop=True)
                train_labels = matrix[action.action_id].iloc[train_positions].reset_index(drop=True)
                state = _fit_action_state(
                    train_x,
                    train_labels,
                    config=split_config,
                    lgbm_params=lgbm_params,
                )
                calibrators, calibration_report = _fit_train_oof_isotonic(
                    train_frame,
                    train_x,
                    train_labels,
                    config=split_config,
                    lgbm_params=lgbm_params,
                    decision_policy=decision_policy,
                    execution_ev_feature=execution_ev_feature,
                )
                state["isotonic"]["lgbm"] = calibrators
                states[action.action_id] = state
                calibration_audit[action.action_id] = calibration_report
            validation_x = x.iloc[valid_positions]
            enter_now_ev = validation_x[execution_ev_feature].to_numpy(dtype=np.float64)
            for action in action_grid:
                local_rows = action_predictions.index[
                    (action_predictions["action_id"] == action.action_id)
                    & action_predictions["base_position"].isin(valid_positions)
                ]
                if len(local_rows) != len(valid_positions):
                    raise ValueError("entry timing action OOF rows failed exact identity alignment")
                state = states[action.action_id]
                for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
                    values = _action_values_from_state(
                        state,
                        validation_x,
                        action=action,
                        arm=arm,
                        predicted_enter_now_ev=enter_now_ev,
                        decision_policy=decision_policy,
                    )
                    columns = _prediction_columns_for_arm(arm)
                    for key in ("raw_fill", "raw_adverse", "raw_delta", "raw_expected", "fill", "adverse", "delta", "conditional", "expected_missed", "expected"):
                        action_predictions.loc[local_rows, columns[key]] = values[key]
            audits.append(
                {
                    "fold": int(fold.fold),
                    "side": side,
                    "status": "ok",
                    "train_rows": int(len(train_positions)),
                    "validation_rows": int(len(valid_positions)),
                    "isotonic": calibration_audit,
                }
            )
    oof_provenance = _oof_provenance(
        frame,
        folds,
        decision_time_col=config.decision_time_col,
        provenance=provenance,
    )
    recommendations = _recommendations_from_oof(
        split_frame,
        action_predictions,
        config=split_config,
        decision_policy=decision_policy,
        oof_provenance=oof_provenance,
    )
    diagnostics = execution_entry_timing_metrics(recommendations)
    overall_lgbm = diagnostics.loc[
        (diagnostics["arm"] == "lgbm") & (diagnostics["scope"] == "overall")
    ]
    if overall_lgbm.empty or int(overall_lgbm.iloc[0]["rows"]) == 0:
        raise ValueError("entry timing has no side-local outer-OOF predictions")
    final_models = _final_models(
        split_frame,
        x,
        matrix,
        sides=sides,
        action_grid=action_grid,
        config=split_config,
        lgbm_params=lgbm_params,
        decision_policy=decision_policy,
        execution_ev_feature=execution_ev_feature,
    )
    fingerprint_columns = _fingerprint_columns(config, target_spec, provenance)
    input_fingerprint = exact_entry_timing_fingerprint(frame, fingerprint_columns)
    frozen_payload = {
        "schema": ENTRY_TIMING_SCHEMA,
        "input_fingerprint": input_fingerprint,
        "feature_names": feature_names,
        "execution_ev_feature": execution_ev_feature,
        "config": _config_payload(config),
        "target_spec": asdict(target_spec),
        "provenance": {name: asdict(spec) for name, spec in provenance.items()},
        "decision_policy": decision_policy,
        "lgbm_params": lgbm_params,
    }
    bundle_fingerprint = hashlib.sha256(
        json.dumps(frozen_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": ENTRY_TIMING_SCHEMA,
        "input_fingerprint": input_fingerprint,
        "bundle_fingerprint": bundle_fingerprint,
        "fingerprint_columns": fingerprint_columns,
        "feature_contract": "predictive execution-EV/alpha/residual/CatBoost/auxiliary inputs require row-level source fold and source train cutoff strictly before the scored decision; final-refit IDs are inference-only; the protected cost/spread-aware execution-EV input is mandatory; realised counterfactual fields are train/report-only",
        "label_contract": "exact fixed-length 1m paths begin at the first executable minute and end at the exact horizon timestamp; every action reanchors the canonical geometry-aware execution-EV simulator at its actual fill; decomposed fee plus entry/exit spread are each applied once; adverse-limit fill bars are excluded from post-fill policy simulation and flagged",
        "oof_contract": "expanding chronological outer folds with label-path purge and embargo; every predictive source cutoff is no later than the outer training cutoff and strictly before the scored decision; final refit is excluded from all OOF metrics",
        "model_contract": "side-local shallow LGBM fill/adverse-first/conditional-filled-delta heads use train-fold inner-OOF isotonic maps for component probabilities, conditional delta and expected action EV; fixed-grid and ridge/logistic baselines share identical outer OOF rows",
        "calibration_contract": "each outer scored fold uses only its training side/action rows to generate chronological inner OOF raw scores, fit isotonic maps, and calibrate expected action EV; no scored-fold path outcome is used",
        "decision_contract": "expected action EV explicitly combines conditional filled EV, fill probability, missed-trade EV, and adverse-first penalty; grid/penalties frozen from earliest outer-train inner OOS only",
        "action_grid": [asdict(action) | {"action_id": action.action_id} for action in action_grid],
        "allowed_action_ids": sorted(allowed_ids),
        "decision_policy": decision_policy,
        "lgbm_params": lgbm_params,
        "model_hpo": model_hpo,
        "decision_hpo": decision_hpo,
        "folds": [
            {
                "fold": int(split.fold),
                "validation_start": split.validation_start.isoformat(),
                "validation_end": split.validation_end.isoformat(),
                "purge_hours": float(split.purge_hours),
                "embargo_hours": float(split.embargo_hours),
            }
            for split in folds
        ],
        "audits": audits,
        "worst_oof": _worst_scope_metrics(diagnostics),
        "diagnostics": diagnostics,
    }
    return ExecutionEntryTimingBundle(
        schema=ENTRY_TIMING_SCHEMA,
        config=_config_payload(config),
        target_spec=target_spec,
        provenance=dict(provenance),
        feature_names=tuple(feature_names),
        execution_ev_feature=execution_ev_feature,
        decision_policy=dict(decision_policy),
        models=final_models,
        report=report,
        input_fingerprint=input_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        oof_action_predictions=action_predictions,
        oof_recommendations=recommendations,
        oof_provenance=oof_provenance,
    )


def _predict_action_table(
    bundle: ExecutionEntryTimingBundle, frame: pd.DataFrame) -> pd.DataFrame:
    config = _config_from_payload(bundle.config)
    feature_names, execution_ev_feature = validate_entry_timing_feature_contract(
        frame, bundle.provenance, config=config, for_scoring=True
    )
    if tuple(feature_names) != bundle.feature_names or execution_ev_feature != bundle.execution_ev_feature:
        raise ValueError("entry timing scoring feature contract does not match the bundle")
    forbidden = entry_timing_realized_label_columns(bundle.target_spec).intersection(frame.columns)
    if forbidden:
        raise ValueError(
            "entry timing inference rejects realised action fields: " + ", ".join(sorted(forbidden))
        )
    x = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").astype("float32")
    if not np.isfinite(x.to_numpy(dtype=np.float32, copy=False)).all():
        raise ValueError("entry timing scoring features contain non-finite values")
    actions = tuple(EntryAction(**action) for action in bundle.config["action_grid"])
    rows: list[pd.DataFrame] = []
    sides = _side_values(frame, config.side_col)
    for side in ("long", "short"):
        positions = np.flatnonzero(sides == side)
        if not len(positions):
            continue
        if side not in bundle.models or not bundle.models[side]:
            raise ValueError(f"entry timing bundle does not contain side-local models for {side!r}")
        local_x = x.iloc[positions]
        now_ev = local_x[execution_ev_feature].to_numpy(dtype=np.float64)
        for action in actions:
            if action.action_id not in bundle.models[side]:
                raise ValueError(f"entry timing bundle misses action model {action.action_id!r} for {side!r}")
            values = _action_values_from_state(
                bundle.models[side][action.action_id],
                local_x,
                action=action,
                arm="lgbm",
                predicted_enter_now_ev=now_ev,
                decision_policy=bundle.decision_policy,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "base_position": positions,
                        "action_id": action.action_id,
                        "action_kind": action.kind,
                        "wait_minutes": action.wait_minutes,
                        "adverse_offset_atr": action.adverse_offset_atr,
                        "fill_probability": values["fill"],
                        "adverse_first_probability": values["adverse"],
                        "conditional_filled_ev": values["conditional"],
                        "expected_missed_ev": values["expected_missed"],
                        "raw_expected_action_ev": values["raw_expected"],
                        "expected_action_ev": values["expected"],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["base_position", "action_id"], kind="stable"
    )


def predict_execution_entry_timing_bundle(
    bundle: ExecutionEntryTimingBundle, frame: pd.DataFrame
) -> pd.DataFrame:
    """Return inference-safe action recommendation without realised inputs."""

    if not isinstance(bundle, ExecutionEntryTimingBundle) or bundle.schema != ENTRY_TIMING_SCHEMA:
        raise ValueError("not an execution entry timing action-value bundle")
    actions = _predict_action_table(bundle, frame)
    config = _config_from_payload(bundle.config)
    grid = tuple(EntryAction(**action) for action in bundle.config["action_grid"])
    allowed = _allowed_actions(grid, bundle.decision_policy)
    allowed_ids = {action.action_id for action in allowed}
    candidates = actions.loc[actions["action_id"].isin(allowed_ids)].copy()
    selected = (
        candidates.sort_values(
            ["base_position", "expected_action_ev", "action_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("base_position", sort=False, as_index=False)
        .first()
        .set_index("base_position")
        .reindex(np.arange(len(frame)))
    )
    result = pd.DataFrame(index=frame.index)
    result["entry_timing_decision"] = np.where(
        selected["action_kind"].to_numpy(dtype=object) == "enter_now", "enter_now", "wait"
    )
    result["recommended_action_id"] = selected["action_id"].to_numpy()
    result["recommended_wait_minutes"] = selected["wait_minutes"].to_numpy(dtype="int64")
    result["recommended_max_wait_minutes"] = result["recommended_wait_minutes"]
    result["recommended_adverse_offset_atr"] = selected["adverse_offset_atr"].to_numpy(dtype=float)
    result["expected_action_ev"] = selected["expected_action_ev"].to_numpy(dtype=float)
    result["fill_probability"] = selected["fill_probability"].to_numpy(dtype=float)
    result["miss_probability"] = 1.0 - result["fill_probability"]
    result["adverse_first_probability"] = selected["adverse_first_probability"].to_numpy(dtype=float)
    result["expected_missed_ev"] = selected["expected_missed_ev"].to_numpy(dtype=float)
    return result


def _atomic_joblib_dump(value: Any, path: Path) -> Path:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        if path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_execution_entry_timing_bundle(
    bundle: ExecutionEntryTimingBundle, path: str | Path
) -> Path:
    """Atomically persist the model bundle including its immutable OOF evidence."""

    if bundle.schema != ENTRY_TIMING_SCHEMA:
        raise ValueError("cannot save an unsupported entry timing bundle")
    return _atomic_joblib_dump(bundle, Path(path))


def load_execution_entry_timing_bundle(path: str | Path) -> ExecutionEntryTimingBundle:
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, ExecutionEntryTimingBundle) or bundle.schema != ENTRY_TIMING_SCHEMA:
        raise ValueError("not an execution entry timing action-value bundle")
    return bundle


def write_execution_entry_timing_artifacts(
    bundle: ExecutionEntryTimingBundle, output_dir: str | Path
) -> dict[str, Path]:
    """Atomically write bundle, exact manifests, OOF evidence, and report."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bundle_path = save_execution_entry_timing_bundle(bundle, root / "execution_entry_timing_bundle.joblib")
    diagnostics_path = _atomic_frame(
        bundle.report["diagnostics"], root / "execution_entry_timing_diagnostics.csv"
    )
    action_path = _atomic_frame(
        bundle.oof_action_predictions.join(
            bundle.oof_provenance, on="base_position", how="left", rsuffix="_base"
        ),
        root / "execution_entry_timing_oof_actions.parquet",
    )
    recommendation_path = _atomic_frame(
        bundle.oof_recommendations.reset_index(drop=True),
        root / "execution_entry_timing_oof_recommendations.parquet",
    )
    provenance_path = _atomic_json(
        root / "execution_entry_timing_provenance.json",
        {
            "schema": ENTRY_TIMING_SCHEMA,
            "input_fingerprint": bundle.input_fingerprint,
            "provenance": {name: asdict(spec) for name, spec in bundle.provenance.items()},
        },
    )
    manifest_path = _atomic_json(
        root / "execution_entry_timing_inference_manifest.json",
        {
            "schema": ENTRY_TIMING_SCHEMA,
            "bundle_fingerprint": bundle.bundle_fingerprint,
            "input_fingerprint": bundle.input_fingerprint,
            "feature_names": list(bundle.feature_names),
            "protected_execution_ev_feature": bundle.execution_ev_feature,
            "action_grid": bundle.config["action_grid"],
            "decision_policy": bundle.decision_policy,
            "lgbm_params": bundle.report["lgbm_params"],
            "target_spec": asdict(bundle.target_spec),
            "provenance_manifest": provenance_path.name,
            "bundle": bundle_path.name,
        },
    )
    payload = {key: value for key, value in bundle.report.items() if key != "diagnostics"}
    payload.update(
        {
            "bundle_path": bundle_path.name,
            "diagnostics_path": diagnostics_path.name,
            "oof_actions_path": action_path.name,
            "oof_recommendations_path": recommendation_path.name,
            "inference_manifest_path": manifest_path.name,
            "provenance_manifest_path": provenance_path.name,
        }
    )
    report_path = _atomic_json(root / "execution_entry_timing_report.json", payload)
    return {
        "bundle": bundle_path,
        "diagnostics": diagnostics_path,
        "oof_actions": action_path,
        "oof_recommendations": recommendation_path,
        "inference_manifest": manifest_path,
        "provenance": provenance_path,
        "provenance_manifest": provenance_path,
        "report": report_path,
    }
