"""Interpretable residual-event model arms and matched benign controls.

All estimators operate on timestamp-level, pre-entry feature matrices.  Outcome
labels are accepted only by ``fit`` and are never required by ``predict_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


EPS = 1e-8


def _weighted_resample_indices(
    weights: np.ndarray, *, max_rows: int, seed: int
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    probability = weights / max(weights.sum(), EPS)
    size = min(max_rows, max(len(weights), 400))
    return np.random.default_rng(seed).choice(
        len(weights), size=size, replace=True, p=probability
    )


@dataclass
class RobustMatrixTransform:
    medians: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RobustMatrixTransform":
        x = np.asarray(x, dtype=np.float32)
        self.medians = np.nanmedian(x, axis=0).astype(np.float32)
        q25 = np.nanquantile(x, 0.25, axis=0)
        q75 = np.nanquantile(x, 0.75, axis=0)
        self.scales = np.maximum(q75 - q25, 1e-4).astype(np.float32)
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        self.scales = np.nan_to_num(self.scales, nan=1.0, posinf=1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.medians is None or self.scales is None:
            raise RuntimeError("RobustMatrixTransform is not fitted")
        out = np.asarray(x, dtype=np.float32).copy()
        missing = ~np.isfinite(out)
        if missing.any():
            out[missing] = np.take(self.medians, np.nonzero(missing)[1])
        out = (out - self.medians) / self.scales
        return np.clip(out, -8.0, 8.0).astype(np.float32, copy=False)


def matched_benign_controls(
    x: np.ndarray,
    y: np.ndarray,
    event_blocks: np.ndarray,
    *,
    controls_per_event: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Match benign lookalikes to each adverse event block in observable space."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    blocks = np.asarray(event_blocks, dtype=np.int32)
    benign = np.flatnonzero(y == 0)
    selected = np.zeros(len(y), dtype=bool)
    report: list[dict[str, Any]] = []
    if not len(benign):
        return selected, report
    for block in np.unique(blocks[(y > 0) & (blocks >= 0)]):
        adverse = np.flatnonzero((y > 0) & (blocks == block))
        if not len(adverse):
            continue
        centre = np.nanmedian(x[adverse], axis=0)
        distance = np.nanmean((x[benign] - centre) ** 2, axis=1)
        take = benign[np.argsort(distance, kind="stable")[:controls_per_event]]
        selected[take] = True
        report.append(
            {
                "event_block": int(block),
                "adverse_rows": int(len(adverse)),
                "control_rows": int(len(take)),
                "mean_squared_distance": float(np.mean(distance[np.argsort(distance)[: len(take)]])),
            }
        )
    return selected, report


def matched_benign_period_controls(
    x: np.ndarray,
    y: np.ndarray,
    event_blocks: np.ndarray,
    days: np.ndarray,
    *,
    controls_per_event: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Match full benign windows to adverse episode blocks.

    This is intentionally a train-side sampler.  A residual event is a period
    state, so comparing its timestamp sequence against isolated benign rows
    loses onset/persistence structure.  Candidate control windows have the
    same calendar duration, contain no target-positive timestamps, and are
    ranked by their aggregate observable-state distance to the adverse block.
    """

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    blocks = np.asarray(event_blocks, dtype=np.int32)
    day_values = np.asarray(days).astype("datetime64[D]")
    selected = np.zeros(len(y), dtype=bool)
    report: list[dict[str, Any]] = []
    valid_days = np.unique(day_values[~np.isnat(day_values)])
    if not len(valid_days):
        return selected, report
    # A few full windows preserve episode shape without allowing long blocks
    # to swamp the event-balanced loss with hundreds of nearly duplicate rows.
    max_windows = min(max(int(controls_per_event), 1), 8)
    for block in np.unique(blocks[(y > 0) & (blocks >= 0)]):
        adverse = np.flatnonzero((y > 0) & (blocks == block))
        if not len(adverse):
            continue
        start = day_values[adverse].min()
        end = day_values[adverse].max()
        span_days = int((end - start).astype(int)) + 1
        centre = np.nanmedian(x[adverse], axis=0)
        candidates: list[tuple[float, np.ndarray, np.datetime64]] = []
        for candidate_start in valid_days:
            candidate_end = candidate_start + np.timedelta64(span_days - 1, "D")
            mask = (day_values >= candidate_start) & (day_values <= candidate_end)
            if not mask.any() or (y[mask] > 0).any():
                continue
            # Require a genuinely represented daily window.  Sparse candidate
            # rows are allowed, but an incomplete sequence is not a lookalike
            # episode control.
            if len(np.unique(day_values[mask])) < span_days:
                continue
            local_centre = np.nanmedian(x[mask], axis=0)
            distance = float(np.nanmean((local_centre - centre) ** 2))
            candidates.append((distance, np.flatnonzero(mask), candidate_start))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[2]))
        taken = 0
        control_rows = 0
        distances: list[float] = []
        starts: list[str] = []
        for distance, indices, candidate_start in candidates:
            # Avoid repeatedly selecting the same benign timestamp for several
            # nearly identical control windows within one target block.
            fresh = indices[~selected[indices]]
            if not len(fresh):
                continue
            selected[fresh] = True
            taken += 1
            control_rows += int(len(fresh))
            distances.append(distance)
            starts.append(str(candidate_start))
            if taken >= max_windows:
                break
        report.append(
            {
                "event_block": int(block),
                "adverse_rows": int(len(adverse)),
                "adverse_days": int(span_days),
                "control_windows": int(taken),
                "control_rows": int(control_rows),
                "mean_squared_distance": float(np.mean(distances)) if distances else np.nan,
                "control_window_starts": "|".join(starts),
                "control_mode": "matched_benign_period_window",
            }
        )
    return selected, report


@dataclass
class RuleFitArm:
    seed: int = 42
    max_rows: int = 6_000
    model: Any = None
    transform: RobustMatrixTransform = field(default_factory=RobustMatrixTransform)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_names: Sequence[str],
    ) -> "RuleFitArm":
        from imodels import RuleFitClassifier

        z = self.transform.fit(x).transform(x)
        idx = _weighted_resample_indices(weights, max_rows=self.max_rows, seed=self.seed)
        self.model = RuleFitClassifier(
            n_estimators=60,
            tree_size=3,
            max_rules=48,
            include_linear=False,
            cv=False,
            random_state=self.seed,
        )
        self.model.fit(z[idx], np.asarray(y, dtype=np.int8)[idx], feature_names=list(feature_names))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self.transform.transform(x))[:, 1], dtype=np.float32)

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for rule, coefficient in zip(self.model.rules_, self.model.coef):
            if abs(float(coefficient)) > 1e-9:
                rows.append({"rule": str(rule), "weight": float(coefficient)})
        return rows


@dataclass
class BinaryQuantileTransform:
    q20: np.ndarray | None = None
    q80: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(self, x: np.ndarray, names: Sequence[str]) -> "BinaryQuantileTransform":
        self.q20 = np.nanquantile(x, 0.20, axis=0).astype(np.float32)
        self.q80 = np.nanquantile(x, 0.80, axis=0).astype(np.float32)
        self.feature_names = [f"{name}__low20" for name in names] + [
            f"{name}__high20" for name in names
        ]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.q20 is None or self.q80 is None:
            raise RuntimeError("BinaryQuantileTransform is not fitted")
        x = np.asarray(x, dtype=np.float32)
        return np.concatenate([x <= self.q20, x >= self.q80], axis=1).astype(bool)


@dataclass
class BayesianRuleListArm:
    seed: int = 42
    max_rows: int = 2_500
    max_input_features: int = 10
    transform: BinaryQuantileTransform = field(default_factory=BinaryQuantileTransform)
    selected_indices: np.ndarray | None = None
    model: Any = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_names: Sequence[str],
    ) -> "BayesianRuleListArm":
        from imodels import BayesianRuleListClassifier

        y = np.asarray(y, dtype=np.int8)
        effect = np.zeros(x.shape[1], dtype=np.float32)
        for index in range(x.shape[1]):
            finite = np.isfinite(x[:, index])
            if finite.any() and y[finite].min() != y[finite].max():
                effect[index] = abs(
                    float(np.nanmedian(x[finite & (y > 0), index]))
                    - float(np.nanmedian(x[finite & (y == 0), index]))
                )
        self.selected_indices = np.argsort(-effect, kind="stable")[: self.max_input_features]
        local_names = [str(feature_names[index]) for index in self.selected_indices]
        local_x = np.asarray(x[:, self.selected_indices], dtype=np.float32)
        self.transform.fit(local_x, local_names)
        binary = self.transform.transform(local_x)
        idx = _weighted_resample_indices(weights, max_rows=self.max_rows, seed=self.seed)
        errors: list[str] = []
        for attempt, (iterations, chains, support) in enumerate(
            ((3_000, 2, 0.04), (8_000, 3, 0.025))
        ):
            try:
                self.model = BayesianRuleListClassifier(
                    listlengthprior=3,
                    listwidthprior=1,
                    maxcardinality=2,
                    minsupport=support,
                    n_chains=chains,
                    max_iter=iterations,
                    class1label="adverse residual event",
                    verbose=False,
                    random_state=self.seed + attempt * 101,
                )
                self.model.fit(
                    binary[idx], y[idx], feature_names=self.transform.feature_names
                )
                break
            except (ValueError, RuntimeError, IndexError) as exc:
                errors.append(f"attempt={attempt + 1}:{type(exc).__name__}:{exc}")
                self.model = None
        if self.model is None:
            raise RuntimeError("BRL posterior fitting failed; " + " | ".join(errors))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        local = np.asarray(x[:, self.selected_indices], dtype=np.float32)
        return np.asarray(self.model.predict_proba(self.transform.transform(local))[:, 1], dtype=np.float32)

    def describe(self) -> list[dict[str, Any]]:
        return [{"rule_list": repr(self.model), "weight": 1.0}]


@dataclass(frozen=True)
class _Rule:
    feature_a: int
    direction_a: int
    threshold_a: float
    feature_b: int = -1
    direction_b: int = 0
    threshold_b: float = 0.0
    probability: float = 0.5
    score: float = 0.0

    def mask(self, x: np.ndarray) -> np.ndarray:
        first = self.direction_a * x[:, self.feature_a] >= self.direction_a * self.threshold_a
        if self.feature_b < 0:
            return first
        second = self.direction_b * x[:, self.feature_b] >= self.direction_b * self.threshold_b
        return first & second


@dataclass
class ContrastiveSubgroupArm:
    max_rules: int = 16
    rules: list[_Rule] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    prevalence: float = 0.5

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_names: Sequence[str],
    ) -> "ContrastiveSubgroupArm":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        w = np.asarray(weights, dtype=np.float64)
        self.names = list(feature_names)
        self.prevalence = float(np.average(y, weights=w))
        singles: list[_Rule] = []
        for feature in range(x.shape[1]):
            finite = np.isfinite(x[:, feature])
            if finite.mean() < 0.6:
                continue
            for direction, quantile in ((1, 0.80), (1, 0.90), (-1, 0.20), (-1, 0.10)):
                threshold = float(np.nanquantile(x[:, feature], quantile))
                rule = _Rule(feature, direction, threshold)
                scored = self._score_rule(rule, x, y, w)
                if scored is not None:
                    singles.append(scored)
        singles.sort(key=lambda rule: rule.score, reverse=True)
        pool = singles[: min(24, len(singles))]
        candidates = list(pool)
        for left_index, left in enumerate(pool[:12]):
            for right in pool[left_index + 1 : 16]:
                if left.feature_a == right.feature_a:
                    continue
                combined = _Rule(
                    left.feature_a,
                    left.direction_a,
                    left.threshold_a,
                    right.feature_a,
                    right.direction_a,
                    right.threshold_a,
                )
                scored = self._score_rule(combined, x, y, w)
                if scored is not None:
                    candidates.append(scored)
        candidates.sort(key=lambda rule: rule.score, reverse=True)
        self.rules = candidates[: self.max_rules]
        return self

    def _score_rule(self, rule: _Rule, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> _Rule | None:
        mask = rule.mask(x)
        support = int(mask.sum())
        positive = int((mask & (y > 0)).sum())
        if support < 12 or positive < 3:
            return None
        probability = float(np.average(y[mask], weights=w[mask]))
        fpr = float(np.average(mask[y == 0], weights=w[y == 0])) if (y == 0).any() else 1.0
        lift = probability / max(self.prevalence, EPS)
        score = float(np.log(max(lift, EPS)) - 2.0 * fpr + 0.03 * np.log1p(positive))
        return _Rule(**{**rule.__dict__, "probability": probability, "score": score})

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = np.full(len(x), self.prevalence, dtype=np.float32)
        for rule in self.rules:
            mask = rule.mask(np.asarray(x, dtype=np.float32))
            score[mask] = np.maximum(score[mask], np.float32(rule.probability))
        return score

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for rule in self.rules:
            text = f"{self.names[rule.feature_a]} {'>=' if rule.direction_a > 0 else '<='} {rule.threshold_a:.5g}"
            if rule.feature_b >= 0:
                text += f" AND {self.names[rule.feature_b]} {'>=' if rule.direction_b > 0 else '<='} {rule.threshold_b:.5g}"
            rows.append({"rule": text, "weight": rule.probability, "discovery_score": rule.score})
        return rows


@dataclass
class _PartitionNode:
    probability: float
    feature: int = -1
    threshold: float = 0.0
    left: Any = None
    right: Any = None


@dataclass
class ModelBasedRecursivePartitionArm:
    max_depth: int = 3
    min_leaf: int = 40
    root: _PartitionNode | None = None
    names: list[str] = field(default_factory=list)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_names: Sequence[str],
    ) -> "ModelBasedRecursivePartitionArm":
        self.names = list(feature_names)
        self.root = self._grow(
            np.asarray(x, dtype=np.float32),
            np.asarray(y, dtype=np.int8),
            np.asarray(weights, dtype=np.float64),
            np.arange(len(y)),
            depth=0,
        )
        return self

    @staticmethod
    def _loss(y: np.ndarray, w: np.ndarray) -> float:
        p = np.clip(np.average(y, weights=w), 1e-5, 1.0 - 1e-5)
        return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))

    def _grow(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, idx: np.ndarray, depth: int) -> _PartitionNode:
        probability = float(np.average(y[idx], weights=w[idx]))
        node = _PartitionNode(probability=probability)
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf or y[idx].min() == y[idx].max():
            return node
        parent_loss = self._loss(y[idx], w[idx])
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(x.shape[1]):
            values = x[idx, feature]
            for threshold in np.unique(np.nanquantile(values, [0.20, 0.35, 0.50, 0.65, 0.80])):
                left = idx[values <= threshold]
                right = idx[values > threshold]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                gain = parent_loss - self._loss(y[left], w[left]) - self._loss(y[right], w[right])
                if best is None or gain > best[0]:
                    best = (float(gain), feature, float(threshold), left, right)
        if best is None or best[0] <= 0.01:
            return node
        _, node.feature, node.threshold, left, right = best
        node.left = self._grow(x, y, w, left, depth + 1)
        node.right = self._grow(x, y, w, right, depth + 1)
        return node

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        output = np.empty(len(x), dtype=np.float32)
        for row_index, row in enumerate(x):
            node = self.root
            while node is not None and node.feature >= 0:
                node = node.left if row[node.feature] <= node.threshold else node.right
            output[row_index] = 0.5 if node is None else node.probability
        return output

    def describe(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def walk(node: _PartitionNode, conditions: list[str]) -> None:
            if node.feature < 0:
                rows.append({"rule": " AND ".join(conditions) or "ALL", "weight": node.probability})
                return
            name = self.names[node.feature]
            walk(node.left, [*conditions, f"{name} <= {node.threshold:.5g}"])
            walk(node.right, [*conditions, f"{name} > {node.threshold:.5g}"])

        if self.root is not None:
            walk(self.root, [])
        return rows


@dataclass
class EpisodeLGBMArm:
    """Small nonlinear challenger for episode-versus-lookalike discrimination.

    This is intentionally constrained: the residual overlay needs a local
    state score, not another broad meta model.  Feature screening, matched
    controls, and temporal OOF selection stay outside this class so every arm
    shares one leakage contract.
    """

    seed: int = 42
    model: Any = None
    names: list[str] = field(default_factory=list)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_names: Sequence[str],
    ) -> "EpisodeLGBMArm":
        import lightgbm as lgb

        self.names = list(feature_names)
        train = lgb.Dataset(
            np.asarray(x, dtype=np.float32),
            label=np.asarray(y, dtype=np.int8),
            weight=np.asarray(weights, dtype=np.float32),
            feature_name=self.names,
        )
        self.model = lgb.train(
            {
                "objective": "binary",
                "learning_rate": 0.035,
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": 32,
                "min_gain_to_split": 0.02,
                "lambda_l1": 0.25,
                "lambda_l2": 1.50,
                "feature_fraction": 0.80,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "seed": int(self.seed),
                "feature_fraction_seed": int(self.seed),
                "bagging_seed": int(self.seed),
                "num_threads": 1,
                "verbosity": -1,
            },
            train,
            num_boost_round=180,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float32)

    def describe(self) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        gain = self.model.feature_importance(importance_type="gain")
        split = self.model.feature_importance(importance_type="split")
        rows = [
            {"rule": name, "weight": float(value), "split_count": int(count)}
            for name, value, count in zip(self.names, gain, split, strict=True)
            if float(value) > 0.0 or int(count) > 0
        ]
        return sorted(rows, key=lambda row: float(row["weight"]), reverse=True)


@dataclass
class EpisodeMLPArm:
    """Compact nonlinear representation challenger on the screened features."""

    seed: int = 42
    max_rows: int = 8_000
    model: Any = None
    names: list[str] = field(default_factory=list)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_names: Sequence[str],
    ) -> "EpisodeMLPArm":
        from sklearn.neural_network import MLPClassifier

        self.names = list(feature_names)
        idx = _weighted_resample_indices(
            np.asarray(weights, dtype=np.float64), max_rows=int(self.max_rows), seed=int(self.seed)
        )
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            alpha=0.08,
            learning_rate_init=0.001,
            batch_size=256,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=20,
            random_state=int(self.seed),
        )
        self.model.fit(
            np.asarray(x, dtype=np.float32)[idx],
            np.asarray(y, dtype=np.int8)[idx],
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict_proba(np.asarray(x, dtype=np.float32))[:, 1],
            dtype=np.float32,
        )

    def describe(self) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        first = np.asarray(self.model.coefs_[0], dtype=np.float64)
        importance = np.abs(first).mean(axis=1)
        rows = [
            {"rule": name, "weight": float(value), "representation": "mlp_input_mean_abs_weight"}
            for name, value in zip(self.names, importance, strict=True)
        ]
        return sorted(rows, key=lambda row: float(row["weight"]), reverse=True)


def build_rule_arm(name: str, *, seed: int) -> Any:
    if name == "rulefit":
        return RuleFitArm(seed=seed)
    if name == "brl":
        return BayesianRuleListArm(seed=seed)
    if name == "contrastive_subgroup":
        return ContrastiveSubgroupArm()
    if name == "model_based_recursive_partition":
        return ModelBasedRecursivePartitionArm()
    if name == "episode_lgbm":
        return EpisodeLGBMArm(seed=seed)
    if name == "episode_lgbm_contrastive":
        # The runner supplies the episode rows and matched high-rank benign
        # controls.  Reuse the same constrained estimator so the only
        # difference is the contrastive training population.
        return EpisodeLGBMArm(seed=seed)
    if name == "episode_mlp":
        return EpisodeMLPArm(seed=seed)
    raise KeyError(f"Unknown residual rule-model arm: {name}")
