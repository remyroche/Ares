"""Local Distribution Forest Proxy (LDF) support-aware position sizing.

N5 is deliberately downstream of the strict-R3 score and causal EV admission.
It never changes candidate identity, ranking, or the admission decision.  It
only maps a causal, train-referenced estimate of mean/risk/support to a bounded
relative position-size multiplier.

The module also owns the compact HPO surface used to freeze the canonical N5
contract.  Raw rolling K9 memberships are forbidden because their component
semantics are not stable across geometry bundle identities; invariant
entropy/margin/OOD/support summaries remain eligible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor

from .trust_sizing_ablation import (
    CMIEdge,
    ParentExpectation,
    RobustTransform,
    _interaction_matrix,
    assert_geometry_semantics,
    causal_size_multiplier,
    cmi_weights,
)


SCHEMA = "strict_r3_n5_forest_support_sizing_v1"
LEGACY_CANONICAL_SCHEMA = "strict_r3_n5_single_forest_support_sizing_v1"
CANONICAL_SCHEMA = "strict_r3_ldf_support_sizing_v3"
CURRENT_CANONICAL_SCHEMA = "strict_r3_ldf_support_sizing_v4"
MODEL_DISPLAY_NAME = "Local Distribution Forest Proxy (LDF)"
MODEL_FAMILY = "local_distribution_forest_proxy"
SEED = 20260810
MIN_PREDICTIVE_SD_BPS = 25.0


@dataclass(frozen=True)
class N5ForestParams:
    """Frozen LDF mean/risk forests and bounded-authority contract."""

    n_estimators: int = 64
    max_depth: int = 8
    min_samples_leaf: int = 120
    max_features: float = 0.70
    max_samples: float = 0.75
    support_prior: float = 300.0
    lambda_max: float = 1.10
    risk_aversion: float = 1.0
    mean_target: str = "policy_net"
    risk_target: str = "oob_squared"
    target_clip_bps: float = 800.0
    cmi_weighting: str = "rank_loss"
    size_floor: float = 0.25
    size_cap: float = 1.75
    seed: int = SEED

    def validate(self) -> None:
        if self.n_estimators < 32:
            raise ValueError("N5 requires at least 32 trees")
        if not 4 <= self.max_depth <= 12:
            raise ValueError("N5 max_depth must be in [4, 12]")
        if self.min_samples_leaf < 40:
            raise ValueError("N5 min_samples_leaf must be at least 40")
        if not 0.30 <= self.max_features <= 1.0:
            raise ValueError("N5 max_features must be in [0.30, 1]")
        if not 0.50 <= self.max_samples <= 1.0:
            raise ValueError("N5 max_samples must be in [0.50, 1]")
        if self.support_prior <= 0.0:
            raise ValueError("N5 support_prior must be positive")
        if not 0.75 <= self.lambda_max <= 1.50:
            raise ValueError("N5 lambda_max must be in [0.75, 1.50]")
        if self.risk_aversion <= 0.0:
            raise ValueError("N5 risk_aversion must be positive")
        if self.mean_target not in {"policy_net", "parent_residual", "winsorized_net"}:
            raise ValueError(f"unsupported N5 mean target: {self.mean_target}")
        if self.risk_target not in {"oob_squared", "oob_downside", "oob_absolute"}:
            raise ValueError(f"unsupported N5 risk target: {self.risk_target}")
        if self.target_clip_bps < 200.0:
            raise ValueError("N5 target clipping must be at least 200 bps")
        if not 0.0 < self.size_floor <= 1.0 <= self.size_cap:
            raise ValueError("N5 size bounds must straddle one")


BASELINE_N5_PARAMS = N5ForestParams()

# Selected by the matched 2025-only portable MDA/HPO funnel in
# ``strict_r3_ldf_mda_legacy_score_compact12_hpo_20260811_v1`` and confirmed
# once on the untouched 2026 replay.  This is intentionally a two-forest
# contract: its OOB mean forest and independent error forest are the exact
# implementation evaluated by the selection runner.  Keep the v3 single-tree
# proxy available for legacy-artifact loading only; do not silently substitute
# it for this current contract.
CURRENT_CANONICAL_LDF_PARAMS = N5ForestParams(
    n_estimators=96,
    max_depth=9,
    min_samples_leaf=100,
    max_features=0.75,
    max_samples=0.70,
    support_prior=300.0,
    lambda_max=1.10,
    risk_aversion=1.0,
    mean_target="policy_net",
    risk_target="oob_squared",
    target_clip_bps=800.0,
    cmi_weighting="rank_loss",
    size_floor=0.25,
    size_cap=1.75,
    seed=SEED,
)

# Frozen two-forest HPO challenger from
# data_perp/artifacts/strict_r3_n5_portable_hpo_full45_20260810_v3/winner.json.
# The executable canonical trainer verifies the checked-in contract against that
# immutable artifact before fitting, so these are not hand-selected defaults.
HPO_CHALLENGER_LDF_PARAMS = N5ForestParams(
    n_estimators=128,
    max_depth=6,
    min_samples_leaf=240,
    max_features=0.55,
    max_samples=0.65,
    support_prior=450.0,
    lambda_max=1.10,
    risk_aversion=1.0,
    mean_target="policy_net",
    risk_target="oob_squared",
    target_clip_bps=800.0,
    cmi_weighting="rank_loss",
    size_floor=0.25,
    size_cap=1.75,
    seed=SEED,
)


@dataclass(frozen=True)
class CanonicalN5Spec:
    """Exact selected N5 contract from the matched three-month funnel."""

    n_estimators: int = 64
    max_depth: int = 8
    min_samples_leaf: int = 120
    max_features: float = 0.70
    max_samples: float = 0.75
    support_prior: float = 300.0
    lambda_max: float = 1.10
    cmi_weighting: str = "rank_loss"
    sizing_mode: str = "mean_risk"
    size_floor: float = 0.25
    size_cap: float = 1.75
    train_cap: int = 60_000
    train_months: int = 3
    top_fraction: float = 0.30
    seed: int = SEED


CANONICAL_N5_SPEC = CanonicalN5Spec()


def n5_hpo_candidates(
    *,
    max_trials: int | None = None,
    seed: int = SEED,
) -> tuple[N5ForestParams, ...]:
    """Deterministic, subsampled joint mean/risk/parameter HPO surface.

    The hand-curated arms lead the search so every target family is evaluated
    before random exploration.  Further arms are generated reproducibly from
    a broad, deliberately regularised forest surface.  ``max_trials`` is a
    ceiling, not a promise to exhaust the surface: the selection runner can
    stop after its configured number of non-improving trials.
    """

    candidates = (
        BASELINE_N5_PARAMS,
        N5ForestParams(mean_target="policy_net", risk_target="oob_downside"),
        N5ForestParams(mean_target="parent_residual", risk_target="oob_squared"),
        N5ForestParams(mean_target="parent_residual", risk_target="oob_downside"),
        N5ForestParams(mean_target="winsorized_net", risk_target="oob_squared", target_clip_bps=600.0),
        N5ForestParams(mean_target="winsorized_net", risk_target="oob_downside", target_clip_bps=600.0),
        N5ForestParams(n_estimators=96, max_depth=7, min_samples_leaf=180, max_features=0.60, max_samples=0.75),
        N5ForestParams(n_estimators=96, max_depth=9, min_samples_leaf=100, max_features=0.75, max_samples=0.70),
        N5ForestParams(n_estimators=128, max_depth=6, min_samples_leaf=240, max_features=0.55, max_samples=0.65, support_prior=450.0),
        N5ForestParams(n_estimators=96, max_depth=8, min_samples_leaf=160, max_features=0.65, max_samples=0.85, support_prior=450.0, lambda_max=1.0, risk_aversion=1.25, mean_target="parent_residual", risk_target="oob_downside"),
        N5ForestParams(n_estimators=128, max_depth=8, min_samples_leaf=100, max_features=0.80, max_samples=0.80, support_prior=200.0, lambda_max=1.20, risk_aversion=1.0, mean_target="winsorized_net", risk_target="oob_squared", target_clip_bps=700.0),
        N5ForestParams(n_estimators=96, max_depth=7, min_samples_leaf=200, max_features=0.70, max_samples=0.90, support_prior=600.0, lambda_max=1.10, risk_aversion=0.80, mean_target="policy_net", risk_target="oob_absolute"),
    )
    target = int(max_trials) if max_trials is not None else len(candidates)
    if target < 1:
        raise ValueError("max_trials must be positive")

    selected: list[N5ForestParams] = list(candidates[:target])
    seen = {asdict(candidate).__repr__() for candidate in selected}
    rng = np.random.default_rng(int(seed))
    estimator_choices = (64, 80, 96, 112, 128, 160)
    mean_targets = ("policy_net", "parent_residual", "winsorized_net")
    risk_targets = ("oob_squared", "oob_downside", "oob_absolute")
    cmi_weightings = (
        "uniform", "rank", "rank_loss", "rank_false_positive",
        "rank_loss_false_positive",
    )
    while len(selected) < target:
        candidate = N5ForestParams(
            n_estimators=int(rng.choice(estimator_choices)),
            max_depth=int(rng.integers(4, 11)),
            min_samples_leaf=int(rng.integers(80, 401)),
            max_features=float(rng.uniform(0.45, 0.90)),
            max_samples=float(rng.uniform(0.55, 0.90)),
            support_prior=float(rng.uniform(150.0, 700.0)),
            lambda_max=float(rng.uniform(0.80, 1.30)),
            risk_aversion=float(rng.uniform(0.60, 1.60)),
            mean_target=str(rng.choice(mean_targets)),
            risk_target=str(rng.choice(risk_targets)),
            target_clip_bps=float(rng.choice((600.0, 700.0, 800.0, 1_000.0))),
            cmi_weighting=str(rng.choice(cmi_weightings)),
            size_floor=float(rng.choice((0.25, 0.35, 0.45))),
            size_cap=float(rng.choice((1.35, 1.50, 1.75))),
            seed=int(seed + len(selected)),
        )
        identity = asdict(candidate).__repr__()
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(candidate)
    for candidate in selected:
        candidate.validate()
    return tuple(selected)


@dataclass
class N5Prediction:
    expected_bps: np.ndarray
    predictive_sd_bps: np.ndarray
    shrinkage_lambda: np.ndarray
    effective_support: np.ndarray

    def quality(self, risk_aversion: float) -> np.ndarray:
        mean = np.maximum(np.asarray(self.expected_bps, dtype=float), 0.0)
        risk = np.asarray(self.predictive_sd_bps, dtype=float)
        return mean / np.maximum(mean**2 + float(risk_aversion) * risk**2, 625.0)

    def as_frame(self) -> pd.DataFrame:
        sd = np.maximum(np.asarray(self.predictive_sd_bps, dtype=float), MIN_PREDICTIVE_SD_BPS)
        mean = np.asarray(self.expected_bps, dtype=float)
        return pd.DataFrame(
            {
                "n5_expected_bps": mean.astype(np.float32),
                "n5_predictive_sd_bps": sd.astype(np.float32),
                "n5_shrinkage_lambda": np.asarray(self.shrinkage_lambda, dtype=np.float32),
                "n5_effective_support": np.asarray(self.effective_support, dtype=np.float32),
                "n5_p_ev_positive": (1.0 - student_t.cdf((0.0 - mean) / sd, df=5.0)).astype(np.float32),
                "n5_p_adverse_200": student_t.cdf((-200.0 - mean) / sd, df=5.0).astype(np.float32),
            }
        )


@dataclass
class FittedN5Forest:
    fields: tuple[str, ...]
    edges: tuple[CMIEdge, ...]
    transform: RobustTransform
    mean_model: RandomForestRegressor
    risk_model: RandomForestRegressor
    leaf_counts: tuple[Mapping[int, int], ...]
    params: N5ForestParams
    train_quality_reference: np.ndarray
    target_audit: Mapping[str, Any]
    training_score_floor: float | None = None
    cutoff: pd.Timestamp | None = None
    schema: str = SCHEMA

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        assert_geometry_semantics(frame, self.fields)
        # The serving frame also carries the two upstream values used after the
        # forest prediction (``final_score`` and ``raw_expected_bps``).  The
        # robust transform is fitted strictly on the declared LDF contract, so
        # select that contract explicitly rather than allowing auxiliary serving
        # columns to change the feature matrix shape or ordering.
        raw = self.transform.transform(frame.loc[:, list(self.fields)])
        return np.hstack([raw, _interaction_matrix(raw, self.fields, self.edges)]).astype(np.float32)

    def _local_mean(self, matrix: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tree = np.column_stack([estimator.predict(matrix) for estimator in self.mean_model.estimators_])
        local = tree.mean(axis=1)
        if self.params.mean_target == "parent_residual":
            local = parent + local
        return local, tree.std(axis=1)

    def _support(self, matrix: np.ndarray) -> np.ndarray:
        leaf = self.mean_model.apply(matrix)
        values = np.empty_like(leaf, dtype=float)
        for index, counts in enumerate(self.leaf_counts):
            values[:, index] = np.fromiter(
                (counts.get(int(value), 0) for value in leaf[:, index]),
                dtype=float,
                count=len(leaf),
            )
        return np.median(values, axis=1)

    def predict(self, frame: pd.DataFrame, *, batch_size: int = 50_000) -> N5Prediction:
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, len(frame), int(batch_size)):
            block = frame.iloc[start : start + int(batch_size)]
            matrix = self._matrix(block)
            parent = pd.to_numeric(block["parent_expected_bps"], errors="coerce").to_numpy(float)
            raw_expected = pd.to_numeric(block["raw_expected_bps"], errors="coerce").to_numpy(float)
            local, tree_sd = self._local_mean(matrix, parent)
            support = self._support(matrix)
            support_weight = support / (support + float(self.params.support_prior))
            mixed = support_weight * local + (1.0 - support_weight) * parent
            distance = raw_expected - parent
            lam = np.ones(len(block), dtype=float)
            stable = np.abs(distance) >= 10.0
            lam[stable] = (mixed[stable] - parent[stable]) / distance[stable]
            lam = np.clip(lam, 0.0, float(self.params.lambda_max))
            mean = parent + lam * distance
            log_risk = self.risk_model.predict(matrix)
            learned_sd = MIN_PREDICTIVE_SD_BPS * np.sqrt(np.maximum(np.expm1(log_risk), 1.0))
            predictive_sd = np.sqrt(np.maximum(learned_sd**2 + tree_sd**2, MIN_PREDICTIVE_SD_BPS**2))
            outputs.append((mean, predictive_sd, lam, support))
        if not outputs:
            empty = np.asarray([], dtype=float)
            return N5Prediction(empty, empty, empty, empty)
        return N5Prediction(*(np.concatenate([part[index] for part in outputs]) for index in range(4)))

    def size_multiplier(self, frame: pd.DataFrame) -> tuple[N5Prediction, np.ndarray]:
        prediction = self.predict(frame)
        multiplier = causal_size_multiplier(
            self.train_quality_reference,
            prediction.quality(self.params.risk_aversion),
            floor=self.params.size_floor,
            cap=self.params.size_cap,
        )
        if self.training_score_floor is not None:
            if "final_score" not in frame:
                raise ValueError("canonical LDF scoring requires final_score for its frozen train-only gate")
            active = pd.to_numeric(frame["final_score"], errors="coerce").ge(
                float(self.training_score_floor)
            ).to_numpy(bool)
            multiplier = np.where(active, multiplier, 1.0)
        return prediction, multiplier

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_display_name": MODEL_DISPLAY_NAME,
            "model_family": MODEL_FAMILY,
            "fields": list(self.fields),
            "field_count": len(self.fields),
            "edges": [asdict(edge) for edge in self.edges],
            "params": asdict(self.params),
            "target_audit": dict(self.target_audit),
            "training_score_floor": self.training_score_floor,
            "cutoff": None if self.cutoff is None else self.cutoff.isoformat(),
            "raw_k9_memberships_used": False,
            "integration": "post-admission relative sizing only; ranking and admission unchanged",
        }


@dataclass
class CanonicalN5Bundle:
    """Canonical LDF implementation of legacy arm N5_drf_support_l110_meanrisk."""

    fields: tuple[str, ...]
    edges: tuple[CMIEdge, ...]
    transform: RobustTransform
    model: RandomForestRegressor
    leaf_counts: tuple[Mapping[int, int], ...]
    parent_expectation: ParentExpectation
    residual_scale_bps: float
    train_quality_reference: np.ndarray
    training_score_floor: float
    cutoff: pd.Timestamp
    spec: CanonicalN5Spec = CANONICAL_N5_SPEC
    schema: str = CANONICAL_SCHEMA

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        assert_geometry_semantics(frame, self.fields)
        # Serving frames additionally carry final_score and raw_expected_bps for
        # the post-forest shrinkage step.  Keep the transform matrix exactly to
        # the frozen LDF feature contract.
        raw = self.transform.transform(frame.loc[:, list(self.fields)])
        return np.hstack([raw, _interaction_matrix(raw, self.fields, self.edges)]).astype(np.float32)

    def predict(self, frame: pd.DataFrame, *, batch_size: int = 50_000) -> N5Prediction:
        pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, len(frame), int(batch_size)):
            block = frame.iloc[start : start + int(batch_size)]
            matrix = self._matrix(block)
            raw_expected = pd.to_numeric(block["raw_expected_bps"], errors="coerce").to_numpy(float)
            parent = self.parent_expectation.predict(block["final_score"])
            tree = np.column_stack([estimator.predict(matrix) for estimator in self.model.estimators_])
            local = tree.mean(axis=1)
            leaf = self.model.apply(matrix)
            support_values = np.empty_like(leaf, dtype=float)
            for index, counts in enumerate(self.leaf_counts):
                support_values[:, index] = np.fromiter(
                    (counts.get(int(value), 0) for value in leaf[:, index]),
                    dtype=float,
                    count=len(leaf),
                )
            support = np.median(support_values, axis=1)
            weight = support / (support + float(self.spec.support_prior))
            mixed = weight * local + (1.0 - weight) * parent
            distance = raw_expected - parent
            lam = np.ones(len(block), dtype=float)
            stable = np.abs(distance) >= 10.0
            lam[stable] = (mixed[stable] - parent[stable]) / distance[stable]
            lam = np.clip(lam, 0.0, float(self.spec.lambda_max))
            mean = parent + lam * distance
            predictive_sd = np.sqrt(tree.std(axis=1) ** 2 + float(self.residual_scale_bps) ** 2)
            pieces.append((mean, predictive_sd, lam, support))
        if not pieces:
            empty = np.asarray([], dtype=float)
            return N5Prediction(empty, empty, empty, empty)
        return N5Prediction(*(np.concatenate([piece[index] for piece in pieces]) for index in range(4)))

    def size_multiplier(self, frame: pd.DataFrame) -> tuple[N5Prediction, np.ndarray]:
        prediction = self.predict(frame)
        multiplier = causal_size_multiplier(
            self.train_quality_reference,
            prediction.quality(1.0),
            floor=self.spec.size_floor,
            cap=self.spec.size_cap,
        )
        active = (
            pd.to_numeric(frame["final_score"], errors="coerce").ge(self.training_score_floor)
            & pd.to_numeric(frame["raw_expected_bps"], errors="coerce").notna()
        ).to_numpy(bool)
        return prediction, np.where(active, multiplier, 1.0).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_display_name": MODEL_DISPLAY_NAME,
            "model_family": MODEL_FAMILY,
            "canonical_arm_legacy_id": "N5_drf_support_l110_meanrisk",
            "cutoff": self.cutoff.isoformat(),
            "fields": list(self.fields),
            "field_count": len(self.fields),
            "edges": [asdict(edge) for edge in self.edges],
            "spec": asdict(self.spec),
            "training_score_floor": float(self.training_score_floor),
            "residual_scale_bps": float(self.residual_scale_bps),
            "raw_k9_memberships_used": False,
            "integration": "after causal admission; bounded relative sizing only",
        }


def fit_canonical_n5_bundle(
    train: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    *,
    parent_expectation: ParentExpectation,
    cutoff: object,
    training_score_floor: float,
    spec: CanonicalN5Spec = CANONICAL_N5_SPEC,
) -> CanonicalN5Bundle:
    """Fit exact original N5 support-shrinkage/mean-risk LDF implementation."""

    fields = tuple(dict.fromkeys(str(field) for field in fields))
    assert_geometry_semantics(train, fields)
    transform = RobustTransform.fit(train, fields)
    raw = transform.transform(train)
    matrix = np.hstack([raw, _interaction_matrix(raw, fields, edges)]).astype(np.float32)
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    raw_expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent = parent_expectation.predict(train["final_score"])
    model = RandomForestRegressor(
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        max_features=spec.max_features,
        bootstrap=True,
        max_samples=spec.max_samples,
        # The canonical LDF is persisted and later replayed at inference.
        # A single worker makes fitting reproducible across historical and
        # live producer environments; it does not alter the frozen forest
        # hyperparameters or its target/feature contract.
        n_jobs=1,
        random_state=spec.seed,
    ).fit(matrix, realised, sample_weight=cmi_weights(train, spec.cmi_weighting))
    tree = np.column_stack([estimator.predict(matrix) for estimator in model.estimators_])
    local = tree.mean(axis=1)
    leaf = model.apply(matrix)
    leaf_counts: tuple[Mapping[int, int], ...] = tuple(
        {int(key): int(value) for key, value in zip(*np.unique(leaf[:, index], return_counts=True))}
        for index in range(leaf.shape[1])
    )
    support_values = np.column_stack(
        [
            np.fromiter((counts.get(int(value), 0) for value in leaf[:, index]), float, len(leaf))
            for index, counts in enumerate(leaf_counts)
        ]
    )
    support = np.median(support_values, axis=1)
    weight = support / (support + spec.support_prior)
    mixed = weight * local + (1.0 - weight) * parent
    distance = raw_expected - parent
    lam = np.ones(len(train), dtype=float)
    stable = np.abs(distance) >= 10.0
    lam[stable] = (mixed[stable] - parent[stable]) / distance[stable]
    lam = np.clip(lam, 0.0, spec.lambda_max)
    mean = parent + lam * distance
    residual_scale = float(np.sqrt(np.mean((realised - local) ** 2)))
    prediction = N5Prediction(
        mean,
        np.sqrt(tree.std(axis=1) ** 2 + residual_scale**2),
        lam,
        support,
    )
    train_quality = np.sort(prediction.quality(1.0))
    return CanonicalN5Bundle(
        fields=fields,
        edges=tuple(edges),
        transform=transform,
        model=model,
        leaf_counts=leaf_counts,
        parent_expectation=parent_expectation,
        residual_scale_bps=residual_scale,
        train_quality_reference=train_quality,
        training_score_floor=float(training_score_floor),
        cutoff=pd.Timestamp(cutoff).tz_convert("UTC") if pd.Timestamp(cutoff).tzinfo else pd.Timestamp(cutoff).tz_localize("UTC"),
        spec=spec,
    )


def _forest(params: N5ForestParams, *, seed_offset: int, oob: bool) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params.n_estimators),
        max_depth=int(params.max_depth),
        min_samples_leaf=int(params.min_samples_leaf),
        max_features=float(params.max_features),
        bootstrap=True,
        max_samples=float(params.max_samples),
        oob_score=bool(oob),
        n_jobs=4,
        random_state=int(params.seed + seed_offset),
    )


def fit_n5_forest(
    train: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    *,
    params: N5ForestParams = BASELINE_N5_PARAMS,
) -> tuple[FittedN5Forest, N5Prediction]:
    """Fit mean and OOB-error forests and return an OOB train reference."""

    params.validate()
    fields = tuple(dict.fromkeys(str(field) for field in fields))
    if not fields:
        raise ValueError("N5 requires at least one feature")
    assert_geometry_semantics(train, fields)
    required = {"policy_net_bps", "raw_expected_bps", "parent_expected_bps"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"N5 training frame lacks: {missing}")
    transform = RobustTransform.fit(train, fields)
    raw = transform.transform(train)
    matrix = np.hstack([raw, _interaction_matrix(raw, fields, edges)]).astype(np.float32)
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    parent = pd.to_numeric(train["parent_expected_bps"], errors="coerce").to_numpy(float)
    raw_expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    if not (np.isfinite(realised).all() and np.isfinite(parent).all() and np.isfinite(raw_expected).all()):
        raise ValueError("N5 supervised inputs must be finite")
    if params.mean_target == "parent_residual":
        target = realised - parent
    elif params.mean_target == "winsorized_net":
        target = np.clip(realised, -params.target_clip_bps, params.target_clip_bps)
    else:
        target = realised
    weights = cmi_weights(train, params.cmi_weighting)
    mean_model = _forest(params, seed_offset=0, oob=True).fit(matrix, target, sample_weight=weights)
    oob_local = np.asarray(mean_model.oob_prediction_, dtype=float)
    if params.mean_target == "parent_residual":
        oob_local = parent + oob_local
    leaf = mean_model.apply(matrix)
    leaf_counts: tuple[Mapping[int, int], ...] = tuple(
        {int(key): int(value) for key, value in zip(*np.unique(leaf[:, index], return_counts=True))}
        for index in range(leaf.shape[1])
    )
    support = np.median(
        np.column_stack(
            [np.fromiter((counts.get(int(value), 0) for value in leaf[:, index]), float, len(leaf))
             for index, counts in enumerate(leaf_counts)]
        ),
        axis=1,
    )
    support_weight = support / (support + float(params.support_prior))
    mixed = support_weight * oob_local + (1.0 - support_weight) * parent
    distance = raw_expected - parent
    lam = np.ones(len(train), dtype=float)
    stable = np.abs(distance) >= 10.0
    lam[stable] = (mixed[stable] - parent[stable]) / distance[stable]
    lam = np.clip(lam, 0.0, float(params.lambda_max))
    oob_mean = parent + lam * distance
    error = realised - oob_mean
    if params.risk_target == "oob_downside":
        severity = np.maximum(-error, 0.0)
    elif params.risk_target == "oob_absolute":
        severity = np.abs(error)
    else:
        severity = np.sqrt(error**2)
    risk_target = np.log1p(np.maximum(severity, MIN_PREDICTIVE_SD_BPS) ** 2 / MIN_PREDICTIVE_SD_BPS**2)
    risk_model = _forest(params, seed_offset=101, oob=False).fit(matrix, risk_target, sample_weight=weights)
    log_risk_train = risk_model.predict(matrix)
    learned_sd = MIN_PREDICTIVE_SD_BPS * np.sqrt(np.maximum(np.expm1(log_risk_train), 1.0))
    tree = np.column_stack([estimator.predict(matrix) for estimator in mean_model.estimators_])
    train_sd = np.sqrt(np.maximum(learned_sd**2 + tree.std(axis=1) ** 2, MIN_PREDICTIVE_SD_BPS**2))
    train_prediction = N5Prediction(oob_mean, train_sd, lam, support)
    train_quality = np.sort(
        train_prediction.quality(params.risk_aversion)[
            np.isfinite(train_prediction.quality(params.risk_aversion))
        ]
    )
    bundle = FittedN5Forest(
        fields=fields,
        edges=tuple(edges),
        transform=transform,
        mean_model=mean_model,
        risk_model=risk_model,
        leaf_counts=leaf_counts,
        params=params,
        train_quality_reference=train_quality,
        target_audit={
            "mean_target": params.mean_target,
            "risk_target": params.risk_target,
            "oob_mean_rmse_bps": float(np.sqrt(np.mean(error**2))),
            "oob_mean_mae_bps": float(np.mean(np.abs(error))),
            "train_rows": int(len(train)),
        },
    )
    return bundle, train_prediction


@dataclass
class CurrentCanonicalLDFBundle:
    """Inference wrapper for the selected two-forest LDF contract.

    ``FittedN5Forest`` deliberately remains a general research primitive.  The
    wrapper owns the strict-R3-specific causal parent map and the predeclared
    top-30% activation floor, so training, OOF replay, and inference share one
    target-free scoring path.
    """

    forest: FittedN5Forest
    parent_expectation: ParentExpectation
    training_score_floor: float
    cutoff: pd.Timestamp
    schema: str = CURRENT_CANONICAL_SCHEMA

    @property
    def fields(self) -> tuple[str, ...]:
        return self.forest.fields

    @property
    def params(self) -> N5ForestParams:
        return self.forest.params

    def _with_parent(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"final_score", "raw_expected_bps", *self.fields}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"current canonical LDF frame lacks: {missing}")
        output = frame.copy()
        output["parent_expected_bps"] = self.parent_expectation.predict(
            output["final_score"],
        )
        return output

    def score(self, frame: pd.DataFrame) -> tuple[N5Prediction, np.ndarray]:
        prepared = self._with_parent(frame)
        prediction, multiplier = self.forest.size_multiplier(prepared)
        active = (
            pd.to_numeric(prepared["final_score"], errors="coerce")
            .ge(float(self.training_score_floor))
            .to_numpy(bool)
        )
        return prediction, np.where(active, multiplier, 1.0).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_display_name": MODEL_DISPLAY_NAME,
            "model_family": MODEL_FAMILY,
            "canonical_arm": "compact12_two_forest_meanrisk",
            "cutoff": self.cutoff.isoformat(),
            "fields": list(self.fields),
            "field_count": len(self.fields),
            "params": asdict(self.params),
            "edges": [asdict(edge) for edge in self.forest.edges],
            "target_audit": dict(self.forest.target_audit),
            "training_score_floor": float(self.training_score_floor),
            "raw_k9_memberships_used": False,
            "integration": "after causal admission; bounded relative sizing only",
        }


__all__ = [
    "BASELINE_N5_PARAMS",
    "CURRENT_CANONICAL_LDF_PARAMS",
    "CURRENT_CANONICAL_SCHEMA",
    "HPO_CHALLENGER_LDF_PARAMS",
    "CANONICAL_N5_SPEC",
    "CANONICAL_SCHEMA",
    "LEGACY_CANONICAL_SCHEMA",
    "CanonicalN5Bundle",
    "CurrentCanonicalLDFBundle",
    "CanonicalN5Spec",
    "FittedN5Forest",
    "N5ForestParams",
    "N5Prediction",
    "MODEL_DISPLAY_NAME",
    "MODEL_FAMILY",
    "SCHEMA",
    "fit_n5_forest",
    "fit_canonical_n5_bundle",
    "n5_hpo_candidates",
]
