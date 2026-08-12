"""Leak-safe, base-native leaf representations.

This module is intentionally narrower than the correctness-leaf pipeline.  A
base representation is allowed to use only the native R3 label and the
declared base feature universe.  The learned tree rules are frozen on an
*earlier resolved* dictionary partition and are then merely deterministic,
target-free transformations of base features on later base-training and OOF
rows.  In particular, no residual, exact-net, base-OOF, trust, or meta field
is accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .performance_regimes.correctness_leaf_regimes import (
    LeafRule,
    aggregate_membership,
    cluster_rules,
    soft_rule_membership,
)
from .performance_regimes.correctness_leaf_targets import (
    aggregate_correctness_periods,
)


EPS = 1e-8


class BaseLeafRepresentationError(ValueError):
    """Raised when a base-native representation breaks its lineage contract."""


@dataclass(frozen=True)
class BaseLeafConfig:
    """Fixed, intentionally small representation discovery contract."""

    minimum_dictionary_rows: int = 1_000
    minimum_rule_support: float = 0.01
    maximum_rules: int = 80
    maximum_clusters_per_target: int = 12
    minimum_similarity: float = 0.70
    membership_mode: str = "G1_weighted_geometric"
    tree_estimators: int = 80
    tree_learning_rate: float = 0.04
    tree_num_leaves: int = 16
    tree_max_depth: int = 4


@dataclass(frozen=True)
class FrozenBaseLeafDictionary:
    """A side/fold-specific, already-resolved tree-rule dictionary."""

    side: str
    fold_id: int
    target_name: str
    features: tuple[str, ...]
    median: MappingLike
    iqr: MappingLike
    clusters: tuple[tuple[LeafRule, ...], ...]
    rule_similarity: pd.DataFrame
    dictionary_rows: int
    dictionary_max_label_available_utc: str
    applied_from_decision_utc: str
    membership_mode: str = "G1_weighted_geometric"

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply frozen rules without reading any label/economic column.

        The method intentionally has no target argument.  Passing a frame with
        altered labels therefore cannot alter memberships; only the declared
        base feature values are consulted.
        """
        if "side_name" not in frame or not frame.side_name.astype(str).str.lower().eq(self.side).all():
            raise BaseLeafRepresentationError("frozen base leaf dictionary is side-local")
        missing = sorted(set(self.features).difference(frame.columns))
        if missing:
            raise BaseLeafRepresentationError(f"base leaf application lacks fields {missing[:8]}")
        normalized = _normalize(frame, self.features, self.median, self.iqr)
        result = frame.loc[:, [column for column in ("candidate_id", "__ts__", "__symbol__", "side_name") if column in frame]].copy()
        lineage: list[dict[str, Any]] = []
        for cluster_id, rules in enumerate(self.clusters):
            values = np.vstack([soft_rule_membership(normalized, rule) for rule in rules])
            weights = np.asarray([max(float(rule.weight), EPS) for rule in rules])
            name = (
                f"baseleaf__{self.target_name}__f{self.fold_id}__s{self.side}"
                f"__c{cluster_id:02d}__{self.membership_mode}"
            )
            result[name] = aggregate_membership(values, weights, mode=self.membership_mode)
            lineage.append({
                "side_name": self.side,
                "fold_id": self.fold_id,
                "target": self.target_name,
                "feature": name,
                "cluster": cluster_id,
                "rule_count": len(rules),
                "active_share": float((result[name].to_numpy(float) >= .60).mean()),
                "conditions_json": json.dumps([rule.conditions for rule in rules]),
                "dictionary_rows": self.dictionary_rows,
                "dictionary_max_label_available_utc": self.dictionary_max_label_available_utc,
                "applied_from_decision_utc": self.applied_from_decision_utc,
            })
        return result, pd.DataFrame(lineage)


# ``Mapping`` as an annotation imports a runtime ABC on Python 3.12.  A
# concrete alias keeps this immutable data holder concise and serialisable.
MappingLike = dict[str, float]


def strict_dictionary_split(
    history: pd.DataFrame,
    *,
    decision_column: str = "decision_ts",
    label_available_column: str = "label_available_ts",
    dictionary_fraction: float = .55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return earlier dictionary and later base-fit partitions with a label gap.

    It is not sufficient that dictionary labels predate the *outer* test.  The
    dictionary has to be fully resolved before the first later base-training
    row which receives its representations.  This guard prevents same-row or
    overlapping-horizon label selection from bleeding into the base fit.
    """
    if not 0.30 <= float(dictionary_fraction) <= 0.70:
        raise BaseLeafRepresentationError("dictionary fraction must be a conservative chronological split")
    value = history.copy()
    value[decision_column] = pd.to_datetime(value[decision_column], utc=True, errors="raise")
    value[label_available_column] = pd.to_datetime(value[label_available_column], utc=True, errors="raise")
    value = value.sort_values([decision_column, "candidate_id"], kind="stable").reset_index(drop=True)
    times = pd.Index(value[decision_column].drop_duplicates().sort_values())
    if len(times) < 8:
        raise BaseLeafRepresentationError("insufficient timestamps for a dictionary/base split")
    cut_position = min(max(1, int(np.floor(len(times) * float(dictionary_fraction)))), len(times) - 2)
    raw_cut = times[cut_position]
    dictionary = value[value[decision_column].lt(raw_cut)].copy()
    if dictionary.empty:
        raise BaseLeafRepresentationError("dictionary split contains no early rows")
    # The + one nanosecond gives the half-open availability convention an
    # unambiguous implementation without assuming any bar frequency.
    first_later = dictionary[label_available_column].max() + pd.Timedelta(nanoseconds=1)
    later = value[value[decision_column].ge(first_later)].copy()
    if later.empty or not dictionary[label_available_column].lt(later[decision_column].min()).all():
        raise BaseLeafRepresentationError("dictionary labels are not resolved before later base rows")
    return dictionary.reset_index(drop=True), later.reset_index(drop=True)


def signed_r3_target(frame: pd.DataFrame, *, target_column: str = "r3_class") -> np.ndarray:
    """Native opportunity ordering only: adverse=-1, weak=0, clear=+1."""
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    if not np.isin(target, (0.0, 1.0, 2.0)).all():
        raise BaseLeafRepresentationError("base leaf discovery requires only finite R3 classes 0/1/2")
    return (target - 1.0).astype(np.float32)


def target_values(
    dictionary: pd.DataFrame,
    *,
    horizon_hours: int | None,
    target_column: str = "r3_class",
    label_available_column: str = "label_available_ts",
) -> pd.DataFrame:
    """Build an R3-native row or equal-timestamp period discovery target."""
    out = dictionary.copy()
    out["__base_leaf_signed_r3__"] = signed_r3_target(out, target_column=target_column)
    if horizon_hours is None:
        out["base_leaf_target"] = out["__base_leaf_signed_r3__"]
        out["base_leaf_target_available_ts"] = pd.to_datetime(
            out[label_available_column], utc=True, errors="raise"
        )
        return out
    aggregated = aggregate_correctness_periods(
        out,
        target_column="__base_leaf_signed_r3__",
        horizon_hours=int(horizon_hours),
        label_available_column=label_available_column,
    )
    aggregated["base_leaf_target"] = aggregated["period_correctness_target"]
    aggregated["base_leaf_target_available_ts"] = aggregated["period_label_available_ts"]
    return aggregated


def _normalize(
    frame: pd.DataFrame,
    features: Sequence[str],
    median: MappingLike,
    iqr: MappingLike,
) -> pd.DataFrame:
    # Build the whole matrix at once.  Repeated column insertion fragments a
    # 150+ field DataFrame and can multiply memory during each fold.
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    centre = pd.Series({feature: float(median[feature]) for feature in features})
    scale = pd.Series({feature: float(iqr[feature]) for feature in features})
    return values.fillna(centre).sub(centre, axis="columns").div(scale, axis="columns").clip(-8.0, 8.0)


def _screen(frame: pd.DataFrame, features: Sequence[str], target: np.ndarray) -> list[str]:
    """Deterministic valid/nonconstant filtering and ordering, not a feature gate."""
    ranked: list[tuple[float, str]] = []
    for feature in features:
        value = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(value) & np.isfinite(target)
        if valid.sum() < 200 or np.std(value[valid]) <= 1e-10:
            continue
        association = spearmanr(value[valid], target[valid]).statistic
        if np.isfinite(association):
            ranked.append((abs(float(association)), str(feature)))
    return [feature for _, feature in sorted(ranked, key=lambda item: (-item[0], item[1]))]


def _tree_paths(node: dict[str, Any], names: Sequence[str], path: tuple[tuple[str, int, float], ...] = ()):
    if "leaf_index" in node:
        yield int(node["leaf_index"]), path, float(node.get("leaf_value", 0.0))
        return
    feature = str(names[int(node["split_feature"])])
    threshold = float(node["threshold"])
    yield from _tree_paths(node["left_child"], names, path + ((feature, -1, threshold),))
    yield from _tree_paths(node["right_child"], names, path + ((feature, 1, threshold),))


def _rules(model: Any, features: Sequence[str], reference: pd.DataFrame, config: BaseLeafConfig) -> tuple[list[LeafRule], dict[str, np.ndarray]]:
    rules: list[tuple[float, LeafRule]] = []
    memberships: dict[str, np.ndarray] = {}
    for tree in model.booster_.dump_model()["tree_info"]:
        for leaf_id, conditions, value in _tree_paths(tree["tree_structure"], features):
            if not conditions:
                continue
            rule = LeafRule(
                f"t{tree['tree_index']}_l{leaf_id}", tuple(conditions), value,
                max(abs(value), 1e-4),
            )
            membership = soft_rule_membership(reference, rule)
            support = float((membership >= .60).mean())
            if support >= float(config.minimum_rule_support):
                rules.append((support * abs(value), rule))
                memberships[rule.rule_id] = membership
    selected = [rule for _, rule in sorted(rules, key=lambda item: (-item[0], item[1].rule_id))[:config.maximum_rules]]
    return selected, {rule.rule_id: memberships[rule.rule_id] for rule in selected}


def fit_dictionary(
    dictionary: pd.DataFrame,
    *,
    side: str,
    fold_id: int,
    target_name: str,
    legal_features: Sequence[str],
    applied_from_decision: pd.Timestamp,
    config: BaseLeafConfig = BaseLeafConfig(),
) -> FrozenBaseLeafDictionary:
    """Fit one earlier-resolved shallow R3 rule dictionary."""
    if not dictionary.side_name.astype(str).str.lower().eq(side).all():
        raise BaseLeafRepresentationError("base leaf discovery is strictly side-local")
    if len(dictionary) < config.minimum_dictionary_rows:
        raise BaseLeafRepresentationError("insufficient earlier resolved rows for base leaf dictionary")
    target = pd.to_numeric(dictionary["base_leaf_target"], errors="coerce").to_numpy(float)
    available = pd.to_datetime(dictionary["base_leaf_target_available_ts"], utc=True, errors="raise")
    if not available.lt(applied_from_decision).all():
        raise BaseLeafRepresentationError("base leaf dictionary includes target unavailable at first application")
    fields = _screen(dictionary, tuple(dict.fromkeys(map(str, legal_features))), target)
    if not fields:
        raise BaseLeafRepresentationError("no usable declared base fields for leaf discovery")
    median = dictionary.loc[:, fields].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
    iqr = (dictionary.loc[:, fields].apply(pd.to_numeric, errors="coerce").quantile(.75) - dictionary.loc[:, fields].apply(pd.to_numeric, errors="coerce").quantile(.25)).replace(0.0, 1.0).fillna(1.0)
    normalized = _normalize(dictionary, fields, median.to_dict(), iqr.to_dict())
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BaseLeafRepresentationError("LightGBM is required to fit base leaf dictionaries") from exc
    model = LGBMRegressor(
        objective="regression_l2", n_estimators=config.tree_estimators,
        learning_rate=config.tree_learning_rate, num_leaves=config.tree_num_leaves,
        max_depth=config.tree_max_depth,
        min_child_samples=max(80, int(.01 * len(dictionary))), colsample_bytree=.80,
        reg_lambda=20.0, random_state=20260804 + int(fold_id), n_jobs=1, verbosity=-1,
    ).fit(normalized, target)
    reference = normalized.iloc[int(.80 * len(normalized)):].copy()
    rules, memberships = _rules(model, fields, reference, config)
    clusters, similarity = cluster_rules(rules, memberships, minimum_similarity=config.minimum_similarity)
    retained: list[tuple[LeafRule, ...]] = []
    by_id = {rule.rule_id: rule for rule in rules}
    for cluster in clusters:
        # A cluster is the portable representation.  Single fitted leaves are
        # deliberately excluded: their raw tree IDs are too fold-specific.
        if len(cluster) >= 2:
            retained.append(tuple(by_id[rule_id] for rule_id in cluster))
        if len(retained) >= config.maximum_clusters_per_target:
            break
    return FrozenBaseLeafDictionary(
        side=str(side), fold_id=int(fold_id), target_name=str(target_name),
        features=tuple(fields), median={key: float(value) for key, value in median.items()},
        iqr={key: float(value) for key, value in iqr.items()}, clusters=tuple(retained),
        rule_similarity=similarity, dictionary_rows=int(len(dictionary)),
        dictionary_max_label_available_utc=available.max().isoformat(),
        applied_from_decision_utc=pd.Timestamp(applied_from_decision).isoformat(),
    )


def support_bucket(active_share: float) -> str:
    """Predeclared support buckets keep 5--10% states eligible for selection."""
    share = float(active_share)
    if .05 <= share < .10:
        return "p05_p10"
    if .10 <= share < .20:
        return "p10_p20"
    if share >= .20:
        return "p20_plus"
    return "below_p05"


def cap_support_diverse(
    ranked: pd.DataFrame,
    *,
    feature_column: str = "feature",
    support_column: str = "active_share",
    score_column: str = "min_block_mda",
    maximum_total: int = 20,
) -> list[str]:
    """Cap a selected set while reserving room for sparse genuine contexts."""
    limits = {"p05_p10": 3, "p10_p20": 5, "p20_plus": maximum_total}
    accepted: list[str] = []
    counts = {key: 0 for key in limits}
    for row in ranked.sort_values([score_column, feature_column], ascending=[False, True], kind="stable").itertuples(index=False):
        bucket = support_bucket(float(getattr(row, support_column)))
        if bucket not in limits or counts[bucket] >= limits[bucket] or len(accepted) >= maximum_total:
            continue
        accepted.append(str(getattr(row, feature_column)))
        counts[bucket] += 1
    return accepted


__all__ = [
    "BaseLeafConfig", "BaseLeafRepresentationError", "FrozenBaseLeafDictionary",
    "cap_support_diverse", "fit_dictionary", "signed_r3_target", "strict_dictionary_split",
    "support_bucket", "target_values",
]
