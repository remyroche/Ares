"""Fixed-contract LightGBM factories and fold-local meta feature screening.

These hooks are intentionally narrow.  They accept only already-selected base
sets and a fold-local declared meta universe; they do not invoke Stage-I MDA,
HPO, value mapping, or the production selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_nested_feature_challenger import NestedFeatureChallengerError, NestedFeatureSet
from .stage_i_nested_stack_execution import GuardedMetaArmSpec


@dataclass(frozen=True)
class FixedLGBMContract:
    base_params: Mapping[str, Any]
    # Base-only ladders deliberately do not own or require a meta contract.
    # Meta callers still pass their explicit side-local parameter mapping.
    meta_params: Mapping[str, Any] = field(default_factory=dict)
    meta_feature_cap: int = 40
    meta_min_coverage: float = 0.80
    meta_spearman_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.meta_feature_cap < 7 or not 0.0 < self.meta_min_coverage <= 1.0 or not 0.0 < self.meta_spearman_threshold < 1.0:
            raise NestedFeatureChallengerError("invalid fixed nested LightGBM contract")


def require_side_meta_params(value: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Accept exactly the two side-local Stage-I HPO parameter objects."""
    expected = {"long", "short"}
    observed = {str(key) for key in value}
    if observed != expected:
        raise NestedFeatureChallengerError(
            "nested meta parameter contract must contain exactly long and short; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    output: dict[str, Mapping[str, Any]] = {}
    for side in sorted(expected):
        params = value[side]
        if not isinstance(params, Mapping) or not params:
            raise NestedFeatureChallengerError(f"{side}: nested meta HPO parameter object is required")
        output[side] = dict(params)
    return output


def resolve_side_meta_context_universe(
    cfg: Mapping[str, Any], *, side: str, available_columns: Sequence[str],
    direct_columns: Sequence[str],
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """Resolve a side's declared raw context universe with no cross-side escape."""
    side = str(side).lower()
    if side not in {"long", "short"}:
        raise NestedFeatureChallengerError("meta context resolution requires side=long or side=short")
    side_key = f"meta_{side}_feature_keys"
    opposite_key = f"meta_{'short' if side == 'long' else 'long'}_feature_keys"
    layer_keys = ("meta_shared_feature_keys", "meta_product_feature_keys", "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS")
    missing_layer = [key for key in layer_keys if key not in cfg]
    if missing_layer:
        raise NestedFeatureChallengerError(f"missing required layer-level meta config pools: {missing_layer}")
    side_specific_present = side_key in cfg
    root_keys = ((side_key,) if side_specific_present else ()) + layer_keys
    def expand(name: str, seen: set[str]) -> list[str]:
        if name in seen:
            return []
        value = cfg.get(name)
        if not isinstance(value, (list, tuple, set)):
            return [name]
        seen.add(name)
        output: list[str] = []
        for item in value:
            output.extend(expand(str(item), seen))
        return output
    declared: list[str] = []
    for key in root_keys:
        for item in cfg.get(key, ()) or ():
            declared.extend(expand(str(item), set()))
    declared = list(dict.fromkeys(declared))
    direct = set(map(str, direct_columns))
    prohibited = {opposite_key}
    if prohibited.intersection(declared):
        raise NestedFeatureChallengerError("opposite-side meta config key escaped into side-local universe")
    usable = [name for name in declared if name in set(available_columns) and name not in direct and not name.startswith("prequential_") and "expected_net" not in name]
    provenance = {
        "schema": "stage_i_nested_declared_meta_universe_v1", "side": side,
        "declared_key_groups": list(root_keys), "opposite_side_key_excluded": opposite_key,
        "side_specific_key_present": side_specific_present,
        "declared_feature_count": len(declared), "available_declared_feature_count": len(usable),
        "available_declared_features": usable,
        "declared_universe_sha256": sha256(json.dumps({"side": side, "keys": root_keys, "declared": declared, "available": usable}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return tuple(usable), provenance


def _lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise NestedFeatureChallengerError("LightGBM is required only when the nested execution is explicitly run") from exc
    return LGBMClassifier, LGBMRegressor


def _base_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if str(result.get("objective", "multiclass")).lower() != "multiclass" or int(result.get("num_class", 3)) != 3:
        raise NestedFeatureChallengerError("fixed nested base model must retain the completed R3 multiclass contract")
    result.update({"objective": "multiclass", "num_class": 3, "n_jobs": int(result.get("n_jobs", 1)), "verbosity": int(result.get("verbosity", -1))})
    return result


def fixed_lgbm_base_predictor(contract: FixedLGBMContract):
    """Return the actual R3 fold refitter used by the nested adapter."""
    params = _base_params(contract.base_params)

    def predict(train: pd.DataFrame, target: np.ndarray, valid: pd.DataFrame, feature_set: NestedFeatureSet) -> np.ndarray:
        if set(np.unique(target)) != {0, 1, 2}:
            raise NestedFeatureChallengerError(f"{feature_set.name}: base training fold lacks an R3 class")
        LGBMClassifier, _ = _lightgbm()
        model = LGBMClassifier(**params)
        model.fit(train, target)
        probability = np.asarray(model.predict_proba(valid), dtype=float)
        classes = np.asarray(model.classes_)
        if probability.shape != (len(valid), 3) or set(classes.tolist()) != {0, 1, 2}:
            raise NestedFeatureChallengerError("fixed nested base model did not emit the R3 simplex")
        return probability

    return predict


def fixed_lgbm_meta_predictor(contract: FixedLGBMContract):
    """Return target-aligned binary/ordinal/clipped-residual fold refitters."""
    base = dict(contract.meta_params)
    base.pop("objective", None)
    base.pop("num_class", None)
    base.update({"n_jobs": int(base.get("n_jobs", 1)), "verbosity": int(base.get("verbosity", -1))})

    def predict(train: pd.DataFrame, target: np.ndarray, weight: np.ndarray, valid: pd.DataFrame, spec: GuardedMetaArmSpec) -> np.ndarray:
        LGBMClassifier, LGBMRegressor = _lightgbm()
        if spec.family in {"reliability", "overestimate_veto"}:
            if set(np.unique(target)).difference({0.0, 1.0}) or len(np.unique(target)) < 2:
                raise NestedFeatureChallengerError(f"{spec.arm_id}: binary meta training fold has one class")
            model = LGBMClassifier(**{**base, "objective": "binary"})
            model.fit(train, target.astype(int), sample_weight=weight)
            return np.asarray(model.predict_proba(valid), dtype=float)[:, 1]
        if spec.family in {"ordinal", "quantile_ordinal_residual"}:
            classes = 4 if spec.family == "ordinal" else 3
            if set(np.unique(target)).difference(set(range(classes))) or len(np.unique(target)) < 2:
                raise NestedFeatureChallengerError(f"{spec.arm_id}: ordinal meta training fold lacks class support")
            model = LGBMClassifier(**{**base, "objective": "multiclass", "num_class": classes})
            model.fit(train, target.astype(int), sample_weight=weight)
            raw = np.asarray(model.predict_proba(valid), dtype=float)
            output = np.zeros((len(valid), classes), dtype=float)
            output[:, np.asarray(model.classes_, dtype=int)] = raw
            return output
        # This target is explicitly clipped before fitting; use L1/MAE rather
        # than inheriting the retired Huber residual objective.
        model = LGBMRegressor(**{**base, "objective": "regression_l1"})
        model.fit(train, target.astype(float), sample_weight=weight)
        return np.asarray(model.predict(valid), dtype=float)

    return predict


def fold_local_meta_feature_selector(contract: FixedLGBMContract):
    """Coverage -> nonconstant -> univariate -> Spearman pruning, per fold.

    Mandatory direct base probabilities/score/trust fields are retained exactly;
    every optional context field must be declared by the caller and selected
    from the training fold only.
    """
    def select(train: pd.DataFrame, target: np.ndarray, declared: Sequence[str], mandatory: Sequence[str], spec: GuardedMetaArmSpec) -> tuple[tuple[str, ...], Mapping[str, Any]]:
        declared = tuple(dict.fromkeys(map(str, declared)))
        mandatory = tuple(dict.fromkeys(map(str, mandatory)))
        if not set(mandatory).issubset(declared) or not set(declared).issubset(train.columns):
            raise NestedFeatureChallengerError("meta selector did not receive an exact declared/direct feature universe")
        numeric = train.loc[:, list(declared)].apply(pd.to_numeric, errors="coerce")
        coverage = numeric.notna().mean()
        nonconstant = numeric.nunique(dropna=True) > 1
        eligible = [name for name in declared if name in mandatory or (coverage[name] >= contract.meta_min_coverage and bool(nonconstant[name]))]
        if any(name not in eligible for name in mandatory):
            raise NestedFeatureChallengerError("mandatory direct base/trust feature is unavailable in a meta fold")
        association: dict[str, float | None] = {}
        y = pd.Series(np.asarray(target, dtype=float))
        for name in eligible:
            if name in mandatory:
                association[name] = None
            else:
                association[name] = abs(float(numeric[name].corr(y, method="spearman"))) if numeric[name].notna().sum() > 2 else 0.0
                if not np.isfinite(association[name]): association[name] = 0.0
        optional = sorted((name for name in eligible if name not in mandatory), key=lambda name: (-float(association[name] or 0.0), name))
        selected = list(mandatory)
        pruned: list[str] = []
        for name in optional:
            if len(selected) >= contract.meta_feature_cap:
                break
            correlated = False
            for kept in selected:
                rho = numeric[name].corr(numeric[kept], method="spearman")
                if np.isfinite(rho) and abs(float(rho)) >= contract.meta_spearman_threshold:
                    correlated = True
                    break
            if correlated:
                pruned.append(name)
            else:
                selected.append(name)
        return tuple(selected), {
            "schema": "stage_i_nested_meta_fold_selector_v1", "arm_id": spec.arm_id,
            "declared_features": list(declared), "mandatory_features": list(mandatory),
            "coverage_pass_features": eligible, "univariate_abs_spearman": association,
            "spearman_pruned_features": pruned, "selected_features": selected,
            "feature_cap": contract.meta_feature_cap, "coverage_threshold": contract.meta_min_coverage,
            "spearman_threshold": contract.meta_spearman_threshold,
        }
    return select
