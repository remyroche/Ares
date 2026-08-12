"""Strict chronological side-local parameter HPO for Stage-I winners.

The generic full-fit helper in :mod:`lgbm_pipeline` is binary/regression only.
Stage I therefore needs a narrow wrapper that keeps the base target genuinely
three-class throughout HPO and emits the complete probability simplex needed
by the residual handoff.  Both layers use the same prior-resolved whole-time
folds as production strict OOF; only their objective and score reconstruction
differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd

from . import lgbm_pipeline
from .stage_i_strict_oof import (
    _multiclass_probabilities,
    _strict_train_mask,
    _validation_blocks,
)
from .stage_i_ranking import RANKING_POLICY, stable_stage_i_topk_positions
from .stage_i_target_adapter import (
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_HUBER_RESIDUAL,
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
    StageITargetContract,
    fit_fold_quantile_residual3,
    recover_base_score,
    reconstruct_fold_quantile_residual3,
    training_objectives,
)
from .stage_i_base_target_ablation import training_weights as fit_target_training_weights


_LAYERS = frozenset({"base", "meta"})
_SIDES = frozenset({"long", "short"})
_MIN_OPPORTUNITY_SCORE_STD = 1e-8
_MIN_OPPORTUNITY_SCORE_UNIQUE = 2


class StageIModelHPOError(ValueError):
    """Raised when the selected Stage-I parameter search is not causal."""


class _StageIModelHPODegenerateScoreError(StageIModelHPOError):
    """A fold has no usable score ordering and cannot support economic HPO."""


class _NativeLGBMModel:
    """Small sklearn-compatible view over a Booster trained from cached bins."""

    def __init__(self, booster: Any, *, classifier: bool, classes: np.ndarray | None) -> None:
        self.booster_ = booster
        self._classifier = bool(classifier)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster_.predict(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._classifier:
            raise AttributeError("regression Booster does not expose predict_proba")
        values = np.asarray(self.booster_.predict(frame))
        if values.ndim == 1:
            return np.column_stack([1.0 - values, values])
        return values


class StageILightGBMDatasetCache:
    """Bounded cache of immutable LightGBM bin matrices for exact-repeat fits.

    A cache entry is reused only when feature order, row values, target,
    weights, objective kind and ``max_bin`` are byte-identical.  Different
    nested feature-count arms therefore remain isolated unless they truly have
    the same matrix.  This deliberately trades a small amount of memory for
    avoiding repeated pandas conversion and histogram bin construction across
    HPO trials/rungs.
    """

    def __init__(self, max_entries: int = 24) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _array_hash(values: Any) -> str:
        array = np.ascontiguousarray(np.asarray(values))
        return hashlib.sha256(array.view(np.uint8)).hexdigest()

    def _key(
        self, frame: pd.DataFrame, target: np.ndarray, weight: np.ndarray,
        *, classifier: bool, max_bin: int,
    ) -> str:
        matrix = np.ascontiguousarray(frame.to_numpy(dtype=np.float32, copy=False))
        return _stable_sha256({
            "schema": "stage_i_lgbm_dataset_cache_key_v1",
            "features": list(map(str, frame.columns)),
            "shape": list(matrix.shape),
            "matrix": self._array_hash(matrix),
            "target": self._array_hash(target),
            "weight": self._array_hash(weight),
            "classifier": bool(classifier),
            "max_bin": int(max_bin),
        })

    def fit(
        self, frame: pd.DataFrame, target: np.ndarray, weight: np.ndarray,
        *, classifier: bool, params: Mapping[str, Any],
    ) -> _NativeLGBMModel:
        import lightgbm as lgb

        effective = dict(params)
        max_bin = int(effective.get("max_bin", 255))
        key = self._key(
            frame, target, weight, classifier=classifier, max_bin=max_bin,
        )
        dataset = self._entries.get(key)
        if dataset is None:
            dataset = lgb.Dataset(
                frame,
                label=np.asarray(target),
                weight=np.asarray(weight, dtype=np.float32),
                free_raw_data=False,
                feature_name=list(map(str, frame.columns)),
                params={"max_bin": max_bin, "feature_pre_filter": False},
            )
            # Construct now so every later trial reuses the exact same bins.
            dataset.construct()
            self._entries[key] = dataset
            self.misses += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
            self.hits += 1
        native_params = dict(effective)
        rounds = int(native_params.pop("n_estimators", 100))
        booster = lgb.train(native_params, dataset, num_boost_round=rounds)
        objective = str(native_params.get("objective", "")).lower()
        classes = None
        if classifier:
            classes = (
                np.arange(int(native_params.get("num_class", 2)), dtype=np.int8)
                if objective == "multiclass"
                else np.asarray([0, 1], dtype=np.int8)
            )
        return _NativeLGBMModel(booster, classifier=classifier, classes=classes)

    def audit(self) -> dict[str, int | str]:
        return {
            "schema": "stage_i_lgbm_dataset_cache_v1",
            "entries": len(self._entries), "hits": self.hits, "misses": self.misses,
        }


@dataclass(frozen=True)
class StageIModelHPOResult:
    side: str
    layer: str
    selected_feature_names: tuple[str, ...]
    best_params: Mapping[str, Any]
    oof_score: np.ndarray
    oof_probabilities: np.ndarray | None
    requested_trials: int
    actual_trials: int
    completed_trials: int
    patience: int
    stop_reason: str
    best_trial_number: int
    best_value: float
    best_metrics: Mapping[str, float]
    hpo_cutoff_utc: str
    hpo_rows: int
    trial_audit: tuple[Mapping[str, Any], ...]
    fold_audit: tuple[Mapping[str, Any], ...]
    oof_fold_audit: tuple[Mapping[str, Any], ...]
    ranking_policy: str = RANKING_POLICY
    feasibility_contract: Mapping[str, Any] | None = None
    target_family: str = LEGACY_R3_MULTICLASS3
    target_contract: Mapping[str, Any] | None = None
    target_contract_sha256: str = ""
    hpo_schedule: Mapping[str, Any] | None = None
    hpo_schedule_sha256: str = ""
    hpo_request_sha256: str = ""


HPO_SCHEDULE_SCHEMA = "stage_i_deterministic_successive_halving_v2_all_eras"
HPO_CHECKPOINT_SCHEMA = "stage_i_halving_trial_checkpoint_v1"
# Bump this whenever the meaning of a cached rung result changes.  It is part
# of the request hash, so a code upgrade cannot silently consume old evidence.
HPO_EXECUTION_SCHEMA = "stage_i_halving_execution_v1"


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_safe(value: Any) -> Any:
    """Convert metric payloads to stable JSON without changing their values."""
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_checkpoint_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _atomic_checkpoint_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a completed checkpoint atomically; partial writes are never evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_checkpoint_safe(payload), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _frame_content_sha256(frame: pd.DataFrame) -> str:
    """Hash exact ordered HPO values, including optional weight metadata."""
    # ``hash_pandas_object`` handles the timestamp/string columns of the
    # fold-local weighting ledger as well as the numeric feature matrix.
    value_hash = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return _stable_sha256({
        "columns": list(map(str, frame.columns)), "shape": list(frame.shape),
        "values_sha256": hashlib.sha256(value_hash.tobytes()).hexdigest(),
    })


def _array_content_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _load_or_create_halving_checkpoint(
    directory: str | Path | None, *, request_sha256: str, request: Mapping[str, Any],
) -> tuple[Path | None, dict[str, Any] | None, dict[str, int]]:
    """Load only an exactly matched completed-rung cache, or create a new one.

    The cache is intentionally restricted to native Stage-I fitting.  A custom
    ``fit_model`` can capture arbitrary process-local state, so carrying its
    results across processes would not be a sound HPO shortcut.
    """
    telemetry = {"enabled": False, "hits": 0, "misses": 0, "writes": 0}
    if directory is None:
        return None, None, telemetry
    root = Path(directory)
    path = root / "halving_rungs.json"
    telemetry["enabled"] = True
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageIModelHPOError(f"invalid Stage-I HPO checkpoint: {path}") from exc
        if (
            not isinstance(state, dict)
            or state.get("schema") != HPO_CHECKPOINT_SCHEMA
            or state.get("request_sha256") != request_sha256
        ):
            raise StageIModelHPOError(
                "Stage-I HPO checkpoint lineage drift; use a distinct checkpoint directory"
            )
        if not isinstance(state.get("completed_rungs"), dict):
            raise StageIModelHPOError("Stage-I HPO checkpoint has invalid completed-rung state")
        return path, state, telemetry
    state = {
        "schema": HPO_CHECKPOINT_SCHEMA,
        "request_sha256": request_sha256,
        "request": _checkpoint_safe(request),
        "completed_rungs": {},
    }
    _atomic_checkpoint_json(path, state)
    telemetry["writes"] += 1
    return path, state, telemetry


def _persist_halving_rung(
    path: Path | None, state: dict[str, Any] | None, *, key: str,
    payload: Mapping[str, Any], telemetry: dict[str, int],
) -> None:
    if path is None or state is None:
        return
    # Each entry is written only after all chronological folds of the rung
    # have returned.  A process interrupted mid-fit leaves no reusable result.
    state["completed_rungs"][key] = _checkpoint_safe(payload)
    _atomic_checkpoint_json(path, state)
    telemetry["writes"] += 1


def _successive_halving_schedule(
    *, requested_trials: int, validation_folds: int, eta: int, enabled: bool,
) -> dict[str, Any]:
    """Return the immutable HPO budget schedule.

    Tiny searches deliberately retain the legacy full-budget path.  Every
    halving rung covers *every* chronological validation fold; cheap rungs
    save work by using time-spread row samples and fewer trees, never by
    looking only at the earliest eras.  Finalists receive the full rows,
    trees and folds before winner selection and full-OOF regeneration.
    """
    n_trials = int(requested_trials)
    n_folds = int(validation_folds)
    divisor = int(eta)
    active = bool(enabled and n_trials >= 9 and n_folds >= 2)
    fold_budgets = [n_folds]
    row_fractions = [1.0]
    tree_fractions = [1.0]
    if active:
        fold_budgets = [n_folds, n_folds, n_folds]
        row_fractions = [0.25, 0.50, 1.0]
        tree_fractions = [0.35, 0.65, 1.0]
    populations: list[int] = []
    remaining = n_trials
    for rung, _ in enumerate(fold_budgets):
        populations.append(int(remaining))
        if rung < len(fold_budgets) - 1:
            remaining = max(1, int(np.ceil(remaining / divisor)))
    return {
        "schema": HPO_SCHEDULE_SCHEMA,
        "enabled": active,
        "requested_trials": n_trials,
        "validation_folds": n_folds,
        "eta": divisor,
        "fold_budgets": fold_budgets,
        "training_row_fractions": row_fractions,
        "validation_row_fractions": row_fractions,
        "tree_fractions": tree_fractions,
        "early_rung_era_policy": "all_chronological_folds_time_spread_rows",
        "maximum_promoted_trials": populations,
        "proposal_sampler": (
            "predeclared_random_sampler" if active else "legacy_seeded_tpe"
        ),
        "promotion_order": "selection_value_desc_then_trial_number_asc",
        "final_winner_eligibility": "all_predeclared_hpo_folds_then_full_oof_regeneration",
    }


def _time_spread_positions(
    positions: np.ndarray,
    decision: pd.Series,
    *,
    fraction: float,
    minimum_rows: int,
    target: np.ndarray | None = None,
    required_classes: Sequence[int] = (),
) -> np.ndarray:
    """Deterministically retain rows spread across the complete time span."""
    raw = np.asarray(positions, dtype=np.int32)
    if not len(raw) or float(fraction) >= 1.0:
        return raw
    ordered = raw[
        np.argsort(
            decision.iloc[raw].to_numpy(dtype="datetime64[ns]"), kind="stable"
        )
    ]
    wanted = min(
        len(ordered), max(int(minimum_rows), int(np.ceil(len(ordered) * float(fraction))))
    )
    take = np.unique(
        np.linspace(0, len(ordered) - 1, wanted).round().astype(np.int32)
    )
    chosen = ordered[take]
    if target is not None and required_classes:
        chosen_classes = set(np.asarray(target)[chosen].astype(int).tolist())
        additions: list[int] = []
        for klass in required_classes:
            if int(klass) in chosen_classes:
                continue
            candidates = ordered[np.asarray(target)[ordered].astype(int) == int(klass)]
            if len(candidates):
                additions.append(int(candidates[len(candidates) // 2]))
        if additions:
            chosen = np.unique(np.r_[chosen, np.asarray(additions, dtype=np.int32)])
            chosen = chosen[
                np.argsort(
                    decision.iloc[chosen].to_numpy(dtype="datetime64[ns]"), kind="stable"
                )
            ]
    return np.asarray(chosen, dtype=np.int32)


def _utc(values: Sequence[Any], *, label: str, n: int) -> pd.Series:
    output = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(output) != n or output.isna().any():
        raise StageIModelHPOError(f"{label} must be aligned finite UTC timestamps")
    return output


def _ordered_features(frame: pd.DataFrame, values: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(map(str, values))
    if not fields or len(set(fields)) != len(fields):
        raise StageIModelHPOError("HPO requires an exact non-empty ordered selected-feature list")
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise StageIModelHPOError(f"selected HPO features are absent: {missing[:12]}")
    return fields


def _suggest_params(
    trial: optuna.Trial, *, family: str, seed: int, min_child_samples_upper: int
) -> dict[str, Any]:
    objective = training_objectives(family)[0]
    classifier = bool(objective["classifier"])
    max_depth = trial.suggest_int("max_depth", 4, 8)
    leaf_fraction = trial.suggest_categorical("leaf_fraction", [0.50, 0.75, 1.00])
    num_leaves = max(8, int(round(min(63, 2**max_depth) * float(leaf_fraction))))
    overrides = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 750, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            min(40, int(min_child_samples_upper)),
            int(min_child_samples_upper),
            log=True,
        ),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "subsample": trial.suggest_float("subsample", 0.70, 1.0),
        "subsample_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
        "random_state": int(seed),
    }
    params = lgbm_pipeline._base_lgbm_params(
        int(seed), classifier=classifier, overrides=overrides
    )
    params.update({"objective": objective["objective"]})
    if "num_class" in objective:
        params["num_class"] = int(objective["num_class"])
    return lgbm_pipeline._effective_lgbm_params(params, classifier=classifier)


def _top_tail_mean(
    score: np.ndarray,
    exact_net: np.ndarray,
    weight: np.ndarray,
    *,
    candidate_ids: Sequence[Any],
    decision_timestamps: Sequence[Any],
    side: str,
    fraction: float = 0.10,
) -> float:
    valid = np.isfinite(score) & np.isfinite(exact_net) & np.isfinite(weight) & (weight > 0.0)
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise StageIModelHPOError("HPO trial emitted no finite strict OOF scores")
    count = max(1, int(np.ceil(float(fraction) * len(positions))))
    # Rank once over the complete side, never independently at timestamps.
    # Ties are immutable identities, not dataframe/fold physical positions.
    chosen = stable_stage_i_topk_positions(
        score,
        candidate_ids=candidate_ids,
        decision_timestamps=decision_timestamps,
        side_names=str(side),
        count=count,
        valid_mask=valid,
    )
    return float(np.average(exact_net[chosen], weights=weight[chosen]))


def _rank_ic(score: np.ndarray, exact_net: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(exact_net)
    if int(valid.sum()) < 3:
        return 0.0
    left = pd.Series(score[valid]).rank(method="average").to_numpy(float)
    right = pd.Series(exact_net[valid]).rank(method="average").to_numpy(float)
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _min_child_samples_upper_bound(
    decision: pd.Series,
    available: pd.Series,
    validation_blocks: Sequence[np.ndarray],
) -> tuple[int, tuple[int, ...]]:
    """Bound leaves from the smallest required strict-prior training fold."""
    train_rows: list[int] = []
    for validation_idx_raw in validation_blocks:
        validation_idx = np.asarray(validation_idx_raw, dtype=np.int32)
        if not len(validation_idx):
            continue
        validation_start = decision.iloc[validation_idx].min()
        train_rows.append(int(_strict_train_mask(available, validation_start).sum()))
    if not train_rows or min(train_rows) < 1:
        raise StageIModelHPOError("HPO feasibility requires non-empty strict training folds")
    # At most one eighth of the smallest training fold: this leaves room for
    # several non-trivial leaves in every R3 class model, instead of allowing a
    # legal-but-one-leaf first fold.  Keep a modest lower bound for regularity.
    upper = max(16, min(400, int(min(train_rows)) // 8))
    return int(upper), tuple(train_rows)


def _require_discriminating_opportunity_score(
    score: np.ndarray, *, fold_id: int | str
) -> dict[str, float | int]:
    finite = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    unique = int(np.unique(finite).size)
    std = float(np.std(finite)) if len(finite) else 0.0
    audit: dict[str, float | int] = {
        "finite_score_rows": int(len(finite)),
        "opportunity_score_unique_count": unique,
        "opportunity_score_std": std,
    }
    if unique < _MIN_OPPORTUNITY_SCORE_UNIQUE or std <= _MIN_OPPORTUNITY_SCORE_STD:
        raise _StageIModelHPODegenerateScoreError(
            "non-discriminating Stage-I opportunity score "
            f"in fold={fold_id}: unique={unique}, std={std:.3e}"
        )
    return audit


def _evaluate_params(
    *,
    frame: pd.DataFrame,
    target: np.ndarray,
    exact_net: np.ndarray,
    weight: np.ndarray,
    decision: pd.Series,
    available: pd.Series,
    candidate_ids: np.ndarray,
    side: str,
    features: tuple[str, ...],
    layer: str,
    target_contract: StageITargetContract,
    params: Mapping[str, Any],
    validation_blocks: Sequence[np.ndarray],
    prediction_offset: np.ndarray | None,
    fit_model: Callable[..., Any],
    fold_local_weight_frame: pd.DataFrame | None = None,
    fold_local_weight_mode: str | None = None,
    fold_local_regime_column: str | None = None,
    require_discriminating_scores: bool = True,
    dataset_cache: StageILightGBMDatasetCache | None = None,
    training_row_fraction: float = 1.0,
    validation_row_fraction: float = 1.0,
) -> tuple[
    np.ndarray, np.ndarray | None, float, list[dict[str, Any]], dict[str, float]
]:
    n = len(frame)
    family = target_contract.family
    raw = np.full(n, np.nan, dtype=np.float32)
    probability_width = (
        5 if family == CUMULATIVE_ORDINAL5_O
        else 3 if family in {LEGACY_R3_MULTICLASS3, FOLD_QUANTILE_RESIDUAL3}
        else 0
    )
    probability = (
        np.full((n, probability_width), np.nan, dtype=np.float32)
        if probability_width else None
    )
    fold_rows: list[dict[str, Any]] = []
    fold_tail: list[float] = []
    use_native_cache = dataset_cache is not None and fit_model is _fit_stage_i_model

    def _fit(
        x_train: pd.DataFrame, y_train: np.ndarray, train_weight: np.ndarray,
        *, classifier: bool, params: Mapping[str, Any], objective_mode: str,
    ) -> Any:
        if use_native_cache:
            assert dataset_cache is not None
            return dataset_cache.fit(
                x_train, np.asarray(y_train), np.asarray(train_weight),
                classifier=classifier, params=params,
            )
        return fit_model(
            x_train, y_train, train_weight, classifier=classifier,
            params=params, objective_mode=objective_mode,
        )

    for fold_id, validation_idx_raw in enumerate(validation_blocks):
        validation_idx = np.asarray(validation_idx_raw, dtype=np.int32)
        validation_start = decision.iloc[validation_idx].min()
        train_idx = np.flatnonzero(_strict_train_mask(available, validation_start))
        train_idx = _time_spread_positions(
            train_idx,
            decision,
            fraction=float(training_row_fraction),
            minimum_rows=24,
            target=target,
            required_classes=(
                (0, 1, 2)
                if family == LEGACY_R3_MULTICLASS3
                else (0, 1, 2, 3, 4)
                if family == CUMULATIVE_ORDINAL5_O
                else ()
            ),
        )
        validation_idx = _time_spread_positions(
            validation_idx,
            decision,
            fraction=float(validation_row_fraction),
            minimum_rows=12,
        )
        if not len(train_idx) or not available.iloc[train_idx].lt(validation_start).all():
            raise StageIModelHPOError("HPO admitted a non-prior-resolved training label")
        if family == LEGACY_R3_MULTICLASS3 and set(np.unique(target[train_idx]).tolist()) != {0, 1, 2}:
            raise StageIModelHPOError(f"base HPO fold {fold_id} lacks an R3 class")
        fold_params = dict(params)
        fold_params["random_state"] = int(params.get("random_state", 42)) + fold_id + 1
        x_train = frame.iloc[train_idx].loc[:, list(features)]
        x_valid = frame.iloc[validation_idx].loc[:, list(features)]
        if fold_local_weight_frame is None:
            fold_train_weight = weight[train_idx]
            fold_weight_audit = {
                "training_weight_fit_scope": "provided_aligned_vector",
            }
        else:
            if fold_local_weight_mode is None or fold_local_regime_column is None:
                raise StageIModelHPOError(
                    "fold-local weighting requires an explicit mode and causal regime column"
                )
            local_weight_frame = fold_local_weight_frame.iloc[train_idx].copy()
            fold_train_weight = fit_target_training_weights(
                local_weight_frame,
                target=target[train_idx],
                mode=fold_local_weight_mode,
                regime_column=fold_local_regime_column,
            )
            if (
                len(fold_train_weight) != len(train_idx)
                or not np.isfinite(fold_train_weight).all()
                or (fold_train_weight <= 0.0).any()
            ):
                raise StageIModelHPOError("fold-local target weights are invalid")
            fold_weight_audit = {
                "training_weight_fit_scope": "strict_fold_train_only",
                "training_weight_mode": str(fold_local_weight_mode),
                "training_weight_regime_column": str(fold_local_regime_column),
                "training_weight_min": float(np.min(fold_train_weight)),
                "training_weight_max": float(np.max(fold_train_weight)),
                "training_weight_mean": float(np.mean(fold_train_weight)),
            }
        if family == CUMULATIVE_ORDINAL5_O:
            cumulative = (target[train_idx, None] > np.arange(4)[None, :]).astype(np.int8)
            survival = np.empty((len(validation_idx), 4), dtype=np.float32)
            for boundary in range(4):
                local_target = cumulative[:, boundary]
                if np.unique(local_target).size < 2:
                    survival[:, boundary] = float(local_target[0])
                    continue
                local_params = dict(fold_params)
                local_params.update({
                    "objective": "binary",
                    "random_state": int(fold_params["random_state"]) + boundary,
                })
                model = _fit(
                    x_train, local_target, fold_train_weight, classifier=True,
                    params=local_params,
                    objective_mode="stage_i_cumulative_ordinal5_hpo",
                )
                survival[:, boundary] = np.asarray(
                    model.predict_proba(x_valid), dtype=np.float32
                )[:, 1]
            fold_score, fold_probability = recover_base_score(family, survival)
            assert probability is not None and fold_probability is not None
            probability[validation_idx] = fold_probability
            raw[validation_idx] = fold_score
        elif family == FOLD_QUANTILE_RESIDUAL3:
            if prediction_offset is None:
                raise StageIModelHPOError("fold-quantile meta HPO requires a frozen base offset")
            direct_fq3 = (
                str(target_contract.metadata.get("meta_target_semantics", ""))
                == "same_side_direct_base_output_correctness_q33_v1"
            )
            if direct_fq3:
                from .stage_i_target_specific_oos import (
                    _fit_direct_correctness, _reconstruct_direct_correctness,
                )
                score_domain = tuple(target_contract.metadata.get("native_score_domain", ()))
                if score_domain not in {(-1.0, 1.0), (0.0, 1.0)}:
                    raise StageIModelHPOError("direct FQ3 HPO lacks a valid native score domain")
                fold_target, state = _fit_direct_correctness(
                    exact_net[train_idx], prediction_offset[train_idx], score_domain=score_domain,
                )
            else:
                fold_target, state = fit_fold_quantile_residual3(
                    exact_net[train_idx], prediction_offset[train_idx]
                )
            model = _fit(
                x_train, fold_target, fold_train_weight, classifier=True,
                params=fold_params,
                objective_mode="stage_i_fold_quantile_residual3_hpo",
            )
            fold_probability = _multiclass_probabilities(model, x_valid)
            if direct_fq3:
                correction, _ = _reconstruct_direct_correctness(
                    fold_probability, prediction_offset[validation_idx], state,
                )
            else:
                correction, _ = reconstruct_fold_quantile_residual3(
                    fold_probability, prediction_offset[validation_idx], state,
                )
            assert probability is not None
            probability[validation_idx] = fold_probability
            raw[validation_idx] = correction
        elif family == LEGACY_R3_MULTICLASS3:
            model = _fit(
                x_train, target[train_idx], fold_train_weight, classifier=True,
                params=fold_params, objective_mode="stage_i_r3_multiclass_hpo",
            )
            fold_probability = _multiclass_probabilities(model, x_valid)
            assert probability is not None
            probability[validation_idx] = fold_probability
            raw[validation_idx], _ = recover_base_score(family, fold_probability)
        else:
            model = _fit(
                x_train, target[train_idx], fold_train_weight, classifier=False,
                params=fold_params,
                objective_mode=(
                    "stage_i_soft_scalar_S_hpo"
                    if family == SOFT_SCALAR_S else "stage_i_residual_hpo"
                ),
            )
            prediction = np.asarray(
                model.predict(x_valid),
                dtype=np.float32,
            ).reshape(-1)
            if len(prediction) != len(validation_idx) or not np.isfinite(prediction).all():
                raise StageIModelHPOError("meta HPO fold emitted invalid residual predictions")
            raw[validation_idx] = (
                np.clip(prediction, 0.0, 1.0)
                if family == SOFT_SCALAR_S else prediction
            )
        ranking = raw[validation_idx]
        if prediction_offset is not None:
            ranking = prediction_offset[validation_idx] + ranking
        discrimination = (
            _require_discriminating_opportunity_score(ranking, fold_id=fold_id)
            if require_discriminating_scores
            else {}
        )
        tail = _top_tail_mean(
            ranking,
            exact_net[validation_idx],
            weight[validation_idx],
            candidate_ids=candidate_ids[validation_idx],
            decision_timestamps=decision.iloc[validation_idx],
            side=side,
        )
        fold_tail.append(tail)
        fold_rows.append({
            "fold_id": int(fold_id),
            "validation_start_utc": validation_start.isoformat(),
            "validation_end_utc": decision.iloc[validation_idx].max().isoformat(),
            "validation_max_label_available_utc": available.iloc[validation_idx].max().isoformat(),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "training_row_fraction": float(training_row_fraction),
            "validation_row_fraction": float(validation_row_fraction),
            "train_max_label_available_utc": available.iloc[train_idx].max().isoformat(),
            "strict_prior_resolved": True,
            "top10_exact_net_bps": tail,
            **fold_weight_audit,
            **discrimination,
        })
    ranking = raw if prediction_offset is None else prediction_offset + raw
    overall_discrimination = (
        _require_discriminating_opportunity_score(ranking, fold_id="pooled")
        if require_discriminating_scores
        else {}
    )
    pooled = _top_tail_mean(
        ranking,
        exact_net,
        weight,
        candidate_ids=candidate_ids,
        decision_timestamps=decision,
        side=side,
    )
    worst = float(np.min(fold_tail))
    median = float(np.median(fold_tail))
    ic = _rank_ic(ranking, exact_net)
    # The primary economic score is computed once over all concatenated strict
    # validation OOF rows. Fold economics only penalize lack of transport; no
    # fold or timestamp receives its own top-k allocation in the primary rank.
    robustness_penalty = (
        0.15 * max(0.0, pooled - median)
        + 0.10 * max(0.0, pooled - worst)
    )
    value = pooled - robustness_penalty + ic
    summary = {
        "pooled_global_top10_exact_net_bps": pooled,
        "median_fold_top10_exact_net_bps": median,
        "worst_fold_top10_exact_net_bps": worst,
        "rank_ic": ic,
        "robustness_penalty_bps": robustness_penalty,
        "selection_value": float(value),
        **overall_discrimination,
    }
    return raw, probability, float(value), fold_rows, summary


class _NoImprovementStop:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best = -np.inf
        self.stale = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = float(trial.value) if trial.value is not None else -np.inf
        if value > self.best + 1e-12:
            self.best, self.stale = value, 0
        else:
            self.stale += 1
        if self.stale >= self.patience:
            study.set_user_attr("stop_reason", "patience_no_improvement")
            study.stop()


def _fit_stage_i_model(*args: Any, classifier: bool, params: Mapping[str, Any], **kwargs: Any) -> Any:
    """Fit without allowing the global soft-binary switch to rewrite R3."""
    if classifier:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**dict(params))
        sample_weight = args[2] if len(args) > 2 else None
        fit_kwargs = (
            {"sample_weight": np.asarray(sample_weight, dtype=np.float32)}
            if sample_weight is not None else {}
        )
        return model.fit(args[0], args[1], **fit_kwargs)
    return lgbm_pipeline._fit_lgbm_model(
        *args, classifier=False, params=dict(params), **kwargs
    )


def run_stage_i_model_hpo(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    selected_feature_names: Sequence[str],
    candidate_ids: Sequence[Any],
    exact_net_bps: Sequence[float],
    decision_timestamps: Sequence[Any],
    label_available_timestamps: Sequence[Any],
    side: str,
    layer: str,
    target_contract: StageITargetContract | None = None,
    sample_weight: Sequence[float] | None = None,
    fold_local_weight_frame: pd.DataFrame | None = None,
    fold_local_weight_mode: str | None = None,
    fold_local_regime_column: str | None = None,
    prediction_offset_bps: Sequence[float] | None = None,
    prediction_offset_native_score: Sequence[float] | None = None,
    hpo_trials: int = 60,
    hpo_patience: int = 15,
    n_validation_folds: int = 4,
    successive_halving: bool = True,
    successive_halving_eta: int = 3,
    successive_halving_checkpoint_dir: str | Path | None = None,
    reuse_lgbm_datasets: bool = True,
    dataset_cache: StageILightGBMDatasetCache | None = None,
    min_train_rows: int = 500,
    random_state: int = 20260803,
    hpo_cutoff_utc: str = "2024-01-01T00:00:00Z",
    fit_model: Callable[..., Any] = _fit_stage_i_model,
) -> StageIModelHPOResult:
    """Tune selected fields and regenerate OOF with the frozen HPO winner."""
    side, layer = str(side).lower(), str(layer).lower()
    if side not in _SIDES or layer not in _LAYERS:
        raise StageIModelHPOError("Stage-I HPO must be one side and one base/meta layer")
    trials, patience = int(hpo_trials), int(hpo_patience)
    if trials < 1 or patience < 1:
        raise StageIModelHPOError("HPO trials and patience must be positive")
    if int(successive_halving_eta) < 2:
        raise StageIModelHPOError("successive_halving_eta must be at least two")
    work = frame.copy()
    work.columns = list(map(str, work.columns))
    features = _ordered_features(work, selected_feature_names)
    n = len(work)
    ids = np.asarray(candidate_ids, dtype=object).reshape(-1)
    if len(ids) != n or pd.isna(ids).any() or len(pd.unique(ids)) != n:
        raise StageIModelHPOError(
            "HPO requires immutable unique candidate IDs aligned to the feature frame"
        )
    if target_contract is None:
        # Compatibility is deliberately confined to the two named frozen
        # controls. Every promoted target passes a hash-bound v2 contract.
        family = LEGACY_R3_MULTICLASS3 if layer == "base" else LEGACY_HUBER_RESIDUAL
        target_contract = StageITargetContract(
            family=family, layer=layer,
            target_name=("R3_frozen_control" if layer == "base" else "Huber_residual_control"),
            geometry="legacy_frozen_TP6_SL4_H12",
            identity_sha256="0" * 64, target_sha256="0" * 64,
            economics_sha256="0" * 64, validity_sha256="0" * 64,
            weight_sha256="0" * 64, rows=n,
            target_columns=("legacy_runtime_vector",),
            metadata={"schema_v1_compatibility_only": True},
        )
    if target_contract.layer != layer or int(target_contract.rows) != n:
        raise StageIModelHPOError("HPO target adapter layer/row contract drift")
    family = target_contract.family
    target_dtype = (
        np.int8
        if family in {LEGACY_R3_MULTICLASS3, CUMULATIVE_ORDINAL5_O}
        else np.float32
    )
    y = np.asarray(target, dtype=target_dtype).reshape(-1)
    net = np.asarray(exact_net_bps, dtype=np.float32).reshape(-1)
    if len(y) != n or len(net) != n or not np.isfinite(y).all() or not np.isfinite(net).all():
        raise StageIModelHPOError("HPO target/economics must be finite and row-aligned")
    if family == LEGACY_R3_MULTICLASS3 and not np.isin(y, [0, 1, 2]).all():
        raise StageIModelHPOError("base HPO requires the exact R3 classes 0/1/2")
    if family == CUMULATIVE_ORDINAL5_O and not np.isin(y, [0, 1, 2, 3, 4]).all():
        raise StageIModelHPOError("ordinal base HPO requires exact classes 0..4")
    if family == SOFT_SCALAR_S and ((y < 0.0) | (y > 1.0)).any():
        raise StageIModelHPOError("soft scalar S target must remain in [0,1]")
    decision = _utc(decision_timestamps, label="decision_timestamps", n=n)
    available = _utc(label_available_timestamps, label="label_available_timestamps", n=n)
    if not (available - decision).eq(pd.Timedelta(hours=12)).all():
        raise StageIModelHPOError(
            "Stage-I HPO labels must resolve exactly 12h after decision"
        )
    weight = (
        np.ones(n, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    )
    if len(weight) != n or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise StageIModelHPOError("HPO sample weights must be aligned, finite and non-negative")
    fold_weight_frame: pd.DataFrame | None = None
    if fold_local_weight_frame is not None:
        if layer != "base":
            raise StageIModelHPOError("fold-local target-arm weights are base-only")
        fold_weight_frame = fold_local_weight_frame.reset_index(drop=True).copy()
        if len(fold_weight_frame) != n:
            raise StageIModelHPOError("fold-local weight frame must align to every HPO row")
        mode = str(fold_local_weight_mode or "")
        regime_column = str(fold_local_regime_column or "")
        required_weight_columns = {"decision_ts"}
        if mode in {"contract_certainty", "hybrid"}:
            required_weight_columns.add("contract_certainty")
        if mode == "hybrid":
            if not regime_column:
                raise StageIModelHPOError("hybrid weighting requires a causal regime column")
            required_weight_columns.add(regime_column)
        missing_weight = sorted(required_weight_columns.difference(fold_weight_frame.columns))
        if mode not in {"uniform", "contract_certainty", "hybrid"} or missing_weight:
            raise StageIModelHPOError(
                f"fold-local weight contract is invalid/missing {missing_weight}"
            )
        declared_decision = _utc(
            fold_weight_frame["decision_ts"], label="fold_local_weight_frame.decision_ts", n=n
        )
        if not declared_decision.equals(decision.reset_index(drop=True)):
            raise StageIModelHPOError("fold-local weight decisions drift from the HPO rows")
    elif fold_local_weight_mode is not None or fold_local_regime_column is not None:
        raise StageIModelHPOError("fold-local weighting metadata requires its aligned frame")
    offset = None
    direct_fq3 = (
        family == FOLD_QUANTILE_RESIDUAL3
        and str(target_contract.metadata.get("meta_target_semantics", ""))
        == "same_side_direct_base_output_correctness_q33_v1"
    )
    if family in {FOLD_QUANTILE_RESIDUAL3, LEGACY_HUBER_RESIDUAL}:
        if direct_fq3:
            if prediction_offset_bps is not None or prediction_offset_native_score is None:
                raise StageIModelHPOError(
                    "direct FQ3 HPO requires native-score offset and forbids a bps offset"
                )
            if "prequential_base_expected_net_bps" in features:
                raise StageIModelHPOError("direct FQ3 HPO forbids pre-mapped expected-net features")
            offset = np.asarray(prediction_offset_native_score, dtype=np.float32).reshape(-1)
            domain = tuple(target_contract.metadata.get("native_score_domain", ()))
            if domain not in {(-1.0, 1.0), (0.0, 1.0)} or (
                (offset < domain[0] - 1e-6) | (offset > domain[1] + 1e-6)
            ).any():
                raise StageIModelHPOError("direct FQ3 native-score offset/domain drift")
        else:
            if prediction_offset_bps is None or prediction_offset_native_score is not None:
                raise StageIModelHPOError("mapped-residual meta HPO requires only the bps base offset")
            offset = np.asarray(prediction_offset_bps, dtype=np.float32).reshape(-1)
        if len(offset) != n or not np.isfinite(offset).all():
            raise StageIModelHPOError("meta HPO offset must be finite and aligned")
    elif prediction_offset_bps is not None or prediction_offset_native_score is not None:
        raise StageIModelHPOError("base HPO cannot consume a residual prediction offset")

    cutoff = pd.to_datetime(hpo_cutoff_utc, utc=True, errors="raise")
    hpo_positions = np.flatnonzero(available.lt(cutoff).to_numpy())
    if len(hpo_positions) <= int(min_train_rows):
        raise StageIModelHPOError("insufficient prior-resolved pre-2024 rows for frozen HPO")
    hpo_frame = work.iloc[hpo_positions].reset_index(drop=True)
    hpo_target = y[hpo_positions]
    hpo_net = net[hpo_positions]
    hpo_weight = weight[hpo_positions]
    hpo_decision = decision.iloc[hpo_positions].reset_index(drop=True)
    hpo_available = available.iloc[hpo_positions].reset_index(drop=True)
    hpo_offset = offset[hpo_positions] if offset is not None else None
    hpo_ids = ids[hpo_positions]
    hpo_fold_weight_frame = (
        None if fold_weight_frame is None
        else fold_weight_frame.iloc[hpo_positions].reset_index(drop=True)
    )
    hpo_blocks = _validation_blocks(
        hpo_decision, hpo_available,
        n_folds=int(n_validation_folds), min_train_rows=int(min_train_rows)
    )
    full_blocks = _validation_blocks(
        decision, available,
        n_folds=int(n_validation_folds), min_train_rows=int(min_train_rows)
    )
    hpo_child_upper, hpo_fold_train_rows = _min_child_samples_upper_bound(
        hpo_decision, hpo_available, hpo_blocks
    )
    full_child_upper, full_fold_train_rows = _min_child_samples_upper_bound(
        decision, available, full_blocks
    )
    min_child_samples_upper = min(hpo_child_upper, full_child_upper)
    schedule = _successive_halving_schedule(
        requested_trials=trials,
        validation_folds=len(hpo_blocks),
        eta=int(successive_halving_eta),
        enabled=bool(successive_halving),
    )
    schedule_sha256 = _stable_sha256(schedule)
    checkpoint_request = {
        "schema": "stage_i_hpo_request_v2_full_content",
        "execution_schema": HPO_EXECUTION_SCHEMA,
        "side": side,
        "layer": layer,
        "features": list(features),
        "target_contract_sha256": target_contract.sha256,
        "candidate_ids_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(pd.Series(hpo_ids), index=False)
            .to_numpy(dtype=np.uint64).tobytes()
        ).hexdigest(),
        "frame_sha256": _frame_content_sha256(hpo_frame),
        "target_sha256": _array_content_sha256(hpo_target),
        "exact_net_bps_sha256": _array_content_sha256(hpo_net),
        "sample_weight_sha256": _array_content_sha256(hpo_weight),
        "decision_timestamps_sha256": _array_content_sha256(
            hpo_decision.to_numpy(dtype="datetime64[ns]")
        ),
        "label_available_timestamps_sha256": _array_content_sha256(
            hpo_available.to_numpy(dtype="datetime64[ns]")
        ),
        "prediction_offset_sha256": (
            None if hpo_offset is None else _array_content_sha256(hpo_offset)
        ),
        "fold_local_weight_frame_sha256": (
            None if hpo_fold_weight_frame is None
            else _frame_content_sha256(hpo_fold_weight_frame)
        ),
        "fold_local_weight_mode": fold_local_weight_mode,
        "fold_local_regime_column": fold_local_regime_column,
        "n_validation_folds": int(n_validation_folds),
        "min_train_rows": int(min_train_rows),
        "hpo_cutoff_utc": cutoff.isoformat(),
        "random_state": int(random_state),
        "schedule_sha256": schedule_sha256,
    }
    request_sha256 = _stable_sha256(checkpoint_request)
    feasibility_contract = {
        "schema": "stage_i_hpo_fold_feasibility_v2",
        "min_child_samples_upper": int(min_child_samples_upper),
        "bound_rule": "floor(smallest_required_strict_train_fold_rows / 8), clipped[16,400]",
        "hpo_fold_train_rows": list(map(int, hpo_fold_train_rows)),
        "regeneration_fold_train_rows": list(map(int, full_fold_train_rows)),
        "minimum_opportunity_score_unique_count": int(_MIN_OPPORTUNITY_SCORE_UNIQUE),
        "minimum_opportunity_score_std_exclusive": float(_MIN_OPPORTUNITY_SCORE_STD),
        "non_discriminating_score_action": "trial_pruned_or_regeneration_fail_closed",
        "hpo_schedule": schedule,
        "hpo_schedule_sha256": schedule_sha256,
        "hpo_request_sha256": request_sha256,
    }
    trial_rows: list[dict[str, Any]] = []
    active_dataset_cache = dataset_cache
    if (
        active_dataset_cache is None
        and bool(reuse_lgbm_datasets)
        and fit_model is _fit_stage_i_model
    ):
        active_dataset_cache = StageILightGBMDatasetCache()
    checkpoint_path: Path | None = None
    checkpoint_state: dict[str, Any] | None = None
    checkpoint_telemetry: dict[str, int] = {
        "enabled": False, "hits": 0, "misses": 0, "writes": 0,
    }
    # Only native fitting has a cross-process deterministic implementation
    # contract.  See _load_or_create_halving_checkpoint for why injected
    # fitters are deliberately not cached.
    if bool(schedule["enabled"]) and fit_model is _fit_stage_i_model:
        checkpoint_path, checkpoint_state, checkpoint_telemetry = (
            _load_or_create_halving_checkpoint(
                successive_halving_checkpoint_dir,
                request_sha256=request_sha256, request=checkpoint_request,
            )
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(
            trial,
            family=family,
            seed=int(random_state) + trial.number * 101,
            min_child_samples_upper=min_child_samples_upper,
        )
        try:
            _, _, value, fold_rows, metrics = _evaluate_params(
                frame=hpo_frame, target=hpo_target, exact_net=hpo_net, weight=hpo_weight,
                decision=hpo_decision, available=hpo_available, features=features,
                layer=layer, params=params, validation_blocks=hpo_blocks,
                target_contract=target_contract,
                prediction_offset=hpo_offset, candidate_ids=hpo_ids, side=side,
                fold_local_weight_frame=hpo_fold_weight_frame,
                fold_local_weight_mode=fold_local_weight_mode,
                fold_local_regime_column=fold_local_regime_column,
                fit_model=fit_model,
                dataset_cache=active_dataset_cache,
            )
        except _StageIModelHPODegenerateScoreError as exc:
            reason = str(exc)
            trial.set_user_attr("rejected_reason", reason)
            trial.set_user_attr("effective_params", params)
            trial_rows.append({
                "trial_number": int(trial.number),
                "value": None,
                "params": dict(params),
                "status": "pruned_non_discriminating_score",
                "rejected_reason": reason,
            })
            raise optuna.TrialPruned(reason) from exc
        trial.set_user_attr("effective_params", params)
        trial.set_user_attr("fold_metrics", fold_rows)
        trial.set_user_attr("economic_metrics", metrics)
        trial_rows.append({
            "trial_number": int(trial.number), "value": value,
            "params": dict(params), "economic_metrics": metrics,
            "status": "complete",
        })
        return value

    best_trial_number: int
    best_value: float
    best_metrics: Mapping[str, Any]
    hpo_fold_rows: tuple[Mapping[str, Any], ...]
    stop_reason: str
    actual_trials: int
    completed_trials: int
    if bool(schedule["enabled"]):
        # Proposals are materialised before observing any result.  Promotion
        # can therefore never change which configurations were eligible for
        # the original full budget, and reruns are deterministic.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=int(random_state)),
            study_name=f"stage_i_{layer}_{side}_strict_halving_hpo",
        )
        proposed: dict[int, tuple[optuna.Trial, dict[str, Any]]] = {}
        for _ in range(trials):
            trial = study.ask()
            params = _suggest_params(
                trial,
                family=family,
                seed=int(random_state) + trial.number * 101,
                min_child_samples_upper=min_child_samples_upper,
            )
            trial.set_user_attr("effective_params", params)
            proposed[int(trial.number)] = (trial, params)

        active = sorted(proposed)
        last_results: dict[int, tuple[float, list[dict[str, Any]], dict[str, float]]] = {}
        audit_by_trial: dict[int, dict[str, Any]] = {
            number: {
                "trial_number": number,
                "value": None,
                "params": dict(params),
                "status": "proposed",
                "rung_audit": [],
            }
            for number, (_, params) in proposed.items()
        }
        budgets = list(map(int, schedule["fold_budgets"]))
        row_fractions = list(map(float, schedule["training_row_fractions"]))
        validation_fractions = list(map(float, schedule["validation_row_fractions"]))
        tree_fractions = list(map(float, schedule["tree_fractions"]))
        for rung, fold_budget in enumerate(budgets):
            feasible: list[int] = []
            for number in active:
                trial, params = proposed[number]
                rung_params = dict(params)
                rung_params["n_estimators"] = max(
                    32,
                    int(np.ceil(int(params.get("n_estimators", 100)) * tree_fractions[rung])),
                )
                rung_child_upper = max(
                    4,
                    int(
                        np.floor(
                            min(hpo_fold_train_rows) * row_fractions[rung] / 8.0
                        )
                    ),
                )
                rung_params["min_child_samples"] = min(
                    int(rung_params.get("min_child_samples", rung_child_upper)),
                    rung_child_upper,
                )
                rung_key = f"r{int(rung):02d}_t{int(number):04d}"
                cached = (
                    None if checkpoint_state is None
                    else checkpoint_state["completed_rungs"].get(rung_key)
                )
                params_sha256 = _stable_sha256(rung_params)
                if cached is not None:
                    if cached.get("params_sha256") != params_sha256:
                        raise StageIModelHPOError(
                            "Stage-I HPO completed-rung parameter drift"
                        )
                    checkpoint_telemetry["hits"] += 1
                    if cached.get("status") == "non_discriminating":
                        reason = str(cached.get("rejected_reason", "cached non-discriminating score"))
                        audit_by_trial[number].update({
                            "status": "pruned_non_discriminating_score",
                            "rejected_reason": reason,
                        })
                        audit_by_trial[number]["rung_audit"].append({
                            **dict(cached.get("rung_audit", {})), "cache": "hit",
                        })
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        continue
                    if cached.get("status") != "complete":
                        raise StageIModelHPOError("Stage-I HPO checkpoint rung has invalid status")
                    value = float(cached["value"])
                    fold_rows = list(cached["fold_rows"])
                    metrics = dict(cached["metrics"])
                    rung_audit = {
                        **dict(cached.get("rung_audit", {})), "cache": "hit",
                    }
                else:
                    checkpoint_telemetry["misses"] += int(checkpoint_path is not None)
                    rung_audit = {
                        "rung": rung, "fold_budget": fold_budget,
                        "fold_ids": list(range(len(hpo_blocks))),
                        "training_row_fraction": row_fractions[rung],
                        "validation_row_fraction": validation_fractions[rung],
                        "tree_fraction": tree_fractions[rung],
                        "effective_n_estimators": int(rung_params["n_estimators"]),
                        "effective_min_child_samples": int(rung_params["min_child_samples"]),
                    }
                    try:
                        _, _, value, fold_rows, metrics = _evaluate_params(
                            frame=hpo_frame, target=hpo_target, exact_net=hpo_net,
                            weight=hpo_weight, decision=hpo_decision,
                            available=hpo_available, features=features, layer=layer,
                            params=rung_params, validation_blocks=hpo_blocks,
                            target_contract=target_contract,
                            prediction_offset=hpo_offset, candidate_ids=hpo_ids,
                            side=side, fold_local_weight_frame=hpo_fold_weight_frame,
                            fold_local_weight_mode=fold_local_weight_mode,
                            fold_local_regime_column=fold_local_regime_column,
                            fit_model=fit_model,
                            dataset_cache=active_dataset_cache,
                            training_row_fraction=row_fractions[rung],
                            validation_row_fraction=validation_fractions[rung],
                        )
                    except _StageIModelHPODegenerateScoreError as exc:
                        reason = str(exc)
                        rung_audit["status"] = "non_discriminating"
                        _persist_halving_rung(
                            checkpoint_path, checkpoint_state, key=rung_key,
                            payload={
                                "status": "non_discriminating",
                                "params_sha256": params_sha256,
                                "rejected_reason": reason, "rung_audit": rung_audit,
                            }, telemetry=checkpoint_telemetry,
                        )
                        audit_by_trial[number].update({
                            "status": "pruned_non_discriminating_score",
                            "rejected_reason": reason,
                        })
                        audit_by_trial[number]["rung_audit"].append(rung_audit)
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        continue
                    rung_audit.update({
                        "value": float(value), "economic_metrics": metrics,
                        "status": "complete", "cache": "miss",
                    })
                    _persist_halving_rung(
                        checkpoint_path, checkpoint_state, key=rung_key,
                        payload={
                            "status": "complete", "params_sha256": params_sha256,
                            "value": float(value), "fold_rows": fold_rows,
                            "metrics": metrics, "rung_audit": rung_audit,
                        }, telemetry=checkpoint_telemetry,
                    )
                last_results[number] = (float(value), fold_rows, metrics)
                feasible.append(number)
                audit_by_trial[number].update({
                    "value": float(value), "economic_metrics": metrics,
                    "status": "complete_full_budget" if rung == len(budgets) - 1 else "rung_complete",
                })
                audit_by_trial[number]["rung_audit"].append(rung_audit)
            if not feasible:
                raise StageIModelHPOError(
                    "Stage-I HPO produced no feasible discriminating completed trial"
                )
            ordered = sorted(
                feasible, key=lambda number: (-last_results[number][0], number)
            )
            if rung < len(budgets) - 1:
                promote_count = max(
                    1, int(np.ceil(len(ordered) / int(schedule["eta"])))
                )
                promoted = set(ordered[:promote_count])
                for number in ordered[promote_count:]:
                    trial, _ = proposed[number]
                    audit_by_trial[number]["status"] = "pruned_successive_halving"
                    audit_by_trial[number]["promotion_cutoff_rank"] = promote_count
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                active = sorted(promoted)
            else:
                for number in ordered:
                    trial, _ = proposed[number]
                    study.tell(trial, last_results[number][0])
                active = ordered
        best_trial_number = int(active[0])
        best_value, best_fold_rows_raw, best_metrics_raw = last_results[best_trial_number]
        best_params = dict(proposed[best_trial_number][1])
        best_metrics = dict(best_metrics_raw)
        hpo_fold_rows = tuple(best_fold_rows_raw)
        trial_rows = [audit_by_trial[number] for number in sorted(audit_by_trial)]
        actual_trials = trials
        completed_trials = len(active)
        stop_reason = "deterministic_successive_halving_completed"
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(random_state)),
            study_name=f"stage_i_{layer}_{side}_strict_hpo",
        )
        study.optimize(
            objective,
            n_trials=trials,
            callbacks=[_NoImprovementStop(patience)],
            show_progress_bar=False,
        )
        complete = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
        ]
        if not complete:
            raise StageIModelHPOError(
                "Stage-I HPO produced no feasible discriminating completed trial"
            )
        best_trial = study.best_trial
        best_trial_number = int(best_trial.number)
        best_value = float(study.best_value)
        best_params = dict(best_trial.user_attrs.get("effective_params") or {})
        best_metrics = dict(best_trial.user_attrs.get("economic_metrics") or {})
        hpo_fold_rows = tuple(best_trial.user_attrs.get("fold_metrics") or ())
        actual_trials = int(len(study.trials))
        completed_trials = len(complete)
        stop_reason = str(
            study.user_attrs.get("stop_reason", "requested_trials_completed")
        )
    if not best_params:
        raise StageIModelHPOError("Stage-I HPO winner has no effective parameters")
    if int(best_params.get("min_child_samples", min_child_samples_upper)) > int(
        min_child_samples_upper
    ):
        raise StageIModelHPOError(
            "HPO winner violates the frozen smallest-fold min_child_samples bound"
        )
    # Refit every strict fold with the one frozen winner. The handoff never
    # mixes per-trial or per-fold winning configurations.
    raw, probability, _, oof_fold_rows, _ = _evaluate_params(
        frame=work, target=y, exact_net=net, weight=weight,
        decision=decision, available=available, features=features,
        layer=layer, params=best_params, validation_blocks=full_blocks,
        target_contract=target_contract,
        prediction_offset=offset, candidate_ids=ids, side=side, fit_model=fit_model,
        fold_local_weight_frame=fold_weight_frame,
        fold_local_weight_mode=fold_local_weight_mode,
        fold_local_regime_column=fold_local_regime_column,
        dataset_cache=active_dataset_cache,
    )
    feasibility_contract["lgbm_dataset_cache"] = (
        active_dataset_cache.audit()
        if active_dataset_cache is not None
        else {"schema": "stage_i_lgbm_dataset_cache_v1", "enabled": False}
    )
    feasibility_contract["successive_halving_checkpoint"] = {
        "schema": HPO_CHECKPOINT_SCHEMA,
        "path": None if checkpoint_path is None else str(checkpoint_path),
        "request_sha256": request_sha256,
        **checkpoint_telemetry,
    }
    if not hpo_fold_rows or any(
        pd.Timestamp(row["validation_max_label_available_utc"]) >= cutoff
        for row in hpo_fold_rows
    ):
        raise StageIModelHPOError("HPO winner consumed an outcome at/after the frozen cutoff")
    return StageIModelHPOResult(
        side=side, layer=layer, target_family=family,
        target_contract=target_contract.to_dict(),
        target_contract_sha256=target_contract.sha256,
        selected_feature_names=features,
        best_params=best_params, oof_score=raw, oof_probabilities=probability,
        requested_trials=trials, actual_trials=actual_trials,
        completed_trials=completed_trials,
        patience=patience,
        stop_reason=stop_reason,
        best_trial_number=best_trial_number,
        best_value=best_value,
        best_metrics=best_metrics,
        hpo_cutoff_utc=cutoff.isoformat(), hpo_rows=int(len(hpo_positions)),
        trial_audit=tuple(trial_rows), fold_audit=hpo_fold_rows,
        oof_fold_audit=tuple(oof_fold_rows),
        feasibility_contract=feasibility_contract,
        hpo_schedule=schedule,
        hpo_schedule_sha256=schedule_sha256,
        hpo_request_sha256=request_sha256,
    )


__all__ = [
    "StageILightGBMDatasetCache",
    "StageIModelHPOError",
    "StageIModelHPOResult",
    "run_stage_i_model_hpo",
]
