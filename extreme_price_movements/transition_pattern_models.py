"""Train-only soft transition-morphology models for research catalogues.

This module intentionally models *transition type*, not the pooled market-state
identity.  It accepts only the pre-onset ``sequence__`` summaries materialized
by :mod:`transition_pattern_catalogue`; source/destination state labels are
kept for descriptive evaluation and are rejected as model inputs.

All transforms are fitted in ``fit`` on the supplied training rows.  A caller
performing OOF work must instantiate one model per training fold, fit it only on
that fold, and score the held-out rows with ``transform``/``predict_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from itertools import permutations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


_SEQUENCE_PREFIX = "sequence__"
_FORBIDDEN_SEQUENCE_TOKENS = (
    "target",
    "expost",
    "post_",
    "future",
    "outcome",
    "realized",
    "mfe",
    "mae",
    "timeout",
    "exit_",
    "source_state",
    "destination_state",
    "event_id",
    "available_utc",
)


@dataclass(frozen=True)
class TransitionMorphologyConfig:
    """Outcome-free representation and support parameters."""

    n_components: int = 4
    embedding_components: int = 8
    covariance_type: str = "diag"
    random_state: int = 1729
    min_component_events: int = 8
    min_component_eras: int = 1
    abstain_min_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least two")
        if self.embedding_components < 1:
            raise ValueError("embedding_components must be positive")
        if self.min_component_events < 1 or self.min_component_eras < 1:
            raise ValueError("component support thresholds must be positive")
        if not 0.0 <= self.abstain_min_probability < 1.0:
            raise ValueError("abstain_min_probability must be in [0, 1)")


@dataclass(frozen=True)
class MorphologyGateConfig:
    """Minimum support and reproducibility thresholds for a retained type."""

    min_events: int = 8
    min_eras: int = 3
    min_bootstrap_ari: float = 0.60
    min_posterior_correlation: float = 0.70


def validate_preonset_sequence_columns(
    frame: pd.DataFrame, columns: Iterable[str]
) -> list[str]:
    """Reject everything except numeric ``sequence__`` causal summaries."""

    requested = [str(column) for column in columns]
    unknown = sorted(set(requested).difference(frame.columns))
    if unknown:
        raise KeyError(f"unknown transition sequence feature(s): {unknown}")
    forbidden: list[str] = []
    for column in requested:
        lowered = column.lower()
        if not column.startswith(_SEQUENCE_PREFIX):
            forbidden.append(column)
            continue
        if any(token in lowered for token in _FORBIDDEN_SEQUENCE_TOKENS):
            forbidden.append(column)
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            forbidden.append(column)
    if forbidden:
        raise ValueError(f"non-causal or non-sequence transition feature(s): {forbidden}")
    return requested


def eligible_preonset_sequence_columns(frame: pd.DataFrame) -> list[str]:
    """Discover safe causal sequence summaries from a catalogue frame."""

    return validate_preonset_sequence_columns(
        frame,
        [
            str(column)
            for column in frame.columns
            if str(column).startswith(_SEQUENCE_PREFIX)
            and pd.api.types.is_numeric_dtype(frame[column])
        ],
    )


def component_support_table(
    component: Sequence[int | str],
    *,
    era: Sequence[object] | None = None,
    min_events: int = 8,
    min_eras: int = 3,
) -> pd.DataFrame:
    """Return recurrence/support evidence without looking at trading outcomes."""

    values = pd.Series(component, dtype="object").reset_index(drop=True)
    era_values = (
        pd.Series(era, dtype="object").reset_index(drop=True)
        if era is not None
        else pd.Series(["all_available_eras"] * len(values), dtype="object")
    )
    if len(values) != len(era_values):
        raise ValueError("component and era must have equal length")
    rows: list[dict[str, object]] = []
    for value, indices in values.groupby(values, sort=True).groups.items():
        positions = list(indices)
        local_era = era_values.iloc[positions].dropna()
        events = int(len(positions))
        eras = int(local_era.nunique())
        rows.append(
            {
                "morphology_component_id": str(value),
                "events": events,
                "eras": eras,
                "support_pass": bool(events >= min_events and eras >= min_eras),
            }
        )
    return pd.DataFrame(rows).sort_values("morphology_component_id", kind="stable").reset_index(drop=True)


def minimum_recurrence_stability_gate(
    *,
    events: int,
    eras: int,
    bootstrap_ari: float | None,
    posterior_correlation: float | None,
    config: MorphologyGateConfig = MorphologyGateConfig(),
) -> dict[str, bool]:
    """Evaluate a component only after support and unsupervised stability pass."""

    support_pass = bool(events >= config.min_events and eras >= config.min_eras)
    ari_pass = bool(bootstrap_ari is not None and np.isfinite(bootstrap_ari) and bootstrap_ari >= config.min_bootstrap_ari)
    posterior_pass = bool(
        posterior_correlation is not None
        and np.isfinite(posterior_correlation)
        and posterior_correlation >= config.min_posterior_correlation
    )
    return {
        "support_pass": support_pass,
        "bootstrap_ari_pass": ari_pass,
        "posterior_correlation_pass": posterior_pass,
        "retained": bool(support_pass and ari_pass and posterior_pass),
    }


def _deterministic_component_order(model: GaussianMixture) -> np.ndarray:
    """Map arbitrary GMM internals to a repeatable public component ordering."""

    keys: list[tuple[tuple[float, ...], float, int]] = []
    for index in range(model.n_components):
        center = tuple(np.round(np.asarray(model.means_[index], dtype=float), 12).tolist())
        keys.append((center, -float(model.weights_[index]), int(index)))
    return np.asarray([key[2] for key in sorted(keys)], dtype=int)


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0)
    return -np.sum(np.where(probability > 0.0, probability * np.log(clipped), 0.0), axis=1)


def _best_posterior_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Best column-aligned mean posterior correlation for small GMMs."""

    if reference.shape != candidate.shape or reference.shape[1] == 0:
        return float("nan")
    n_components = reference.shape[1]
    # Pattern work deliberately uses compact mixtures.  The bounded factorial
    # search avoids adding scipy as a runtime requirement to this small module.
    if n_components > 7:
        return float("nan")
    best = -np.inf
    for order in permutations(range(n_components)):
        corr: list[float] = []
        for left, right in enumerate(order):
            a, b = reference[:, left], candidate[:, right]
            if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
                corr.append(1.0 if np.allclose(a, b) else 0.0)
            else:
                corr.append(float(np.corrcoef(a, b)[0, 1]))
        best = max(best, float(np.mean(corr)))
    return float(best)


def bootstrap_morphology_stability(
    embedding: np.ndarray,
    *,
    n_components: int,
    covariance_type: str = "diag",
    random_state: int = 1729,
    bootstrap_draws: int = 12,
) -> dict[str, float | int]:
    """Refit only bootstrap resamples of training embeddings and score stability."""

    values = np.asarray(embedding, dtype=float)
    if values.ndim != 2 or len(values) < n_components:
        raise ValueError("embedding needs at least n_components two-dimensional rows")
    reference_model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=3,
    ).fit(values)
    reference_labels = reference_model.predict(values)
    reference_probability = reference_model.predict_proba(values)
    rng = np.random.default_rng(random_state + 101)
    aris: list[float] = []
    correlations: list[float] = []
    for draw in range(int(bootstrap_draws)):
        sample = rng.integers(0, len(values), size=len(values))
        try:
            fitted = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state + draw + 1,
                n_init=2,
            ).fit(values[sample])
        except ValueError:
            continue
        labels = fitted.predict(values)
        probability = fitted.predict_proba(values)
        aris.append(float(adjusted_rand_score(reference_labels, labels)))
        correlations.append(_best_posterior_correlation(reference_probability, probability))
    if not aris:
        return {
            "bootstrap_draws_completed": 0,
            "bootstrap_ari_median": float("nan"),
            "bootstrap_ari_q10": float("nan"),
            "posterior_correlation_median": float("nan"),
        }
    return {
        "bootstrap_draws_completed": int(len(aris)),
        "bootstrap_ari_median": float(np.median(aris)),
        "bootstrap_ari_q10": float(np.quantile(aris, 0.10)),
        "posterior_correlation_median": float(np.nanmedian(correlations)),
    }


@dataclass
class TransitionMorphologyEmbedder:
    """Fold-fitted pre-onset standardization, PCA embedding, and soft GMM."""

    config: TransitionMorphologyConfig = field(default_factory=TransitionMorphologyConfig)
    feature_columns: list[str] = field(default_factory=list)
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    gmm: GaussianMixture | None = None
    component_order: np.ndarray | None = None
    supported_component_ids: set[str] = field(default_factory=set)
    support_table_: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability_: dict[str, float | int] = field(default_factory=dict)

    def fit(
        self,
        train: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        era_column: str | None = None,
        bootstrap_draws: int = 0,
    ) -> "TransitionMorphologyEmbedder":
        columns = (
            eligible_preonset_sequence_columns(train)
            if feature_columns is None
            else validate_preonset_sequence_columns(train, feature_columns)
        )
        if not columns:
            raise ValueError("transition morphology needs at least one pre-onset sequence feature")
        if len(train) < self.config.n_components:
            raise ValueError("training events fewer than requested morphology components")
        self.feature_columns = list(columns)
        raw = train.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        self.imputer = SimpleImputer(strategy="median")
        standardized = self.imputer.fit_transform(raw)
        self.scaler = StandardScaler()
        standardized = self.scaler.fit_transform(standardized)
        n_embedding = min(self.config.embedding_components, standardized.shape[0], standardized.shape[1])
        self.pca = PCA(n_components=n_embedding, random_state=self.config.random_state)
        embedding = self.pca.fit_transform(standardized)
        self.gmm = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            random_state=self.config.random_state,
            n_init=5,
        ).fit(embedding)
        self.component_order = _deterministic_component_order(self.gmm)
        probability = self._probability_from_embedding(embedding)
        hard = probability.argmax(axis=1)
        component_ids = np.asarray([f"m{index:02d}" for index in hard], dtype=object)
        era = train[era_column] if era_column is not None and era_column in train.columns else None
        self.support_table_ = component_support_table(
            component_ids,
            era=era,
            min_events=self.config.min_component_events,
            min_eras=self.config.min_component_eras,
        )
        self.supported_component_ids = set(
            self.support_table_.loc[self.support_table_["support_pass"], "morphology_component_id"].astype(str)
        )
        if bootstrap_draws:
            self.stability_ = bootstrap_morphology_stability(
                embedding,
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                random_state=self.config.random_state,
                bootstrap_draws=bootstrap_draws,
            )
        return self

    def _require_fitted(self) -> None:
        if any(value is None for value in (self.imputer, self.scaler, self.pca, self.gmm, self.component_order)):
            raise RuntimeError("transition morphology embedder is not fitted")

    def _embedding(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        missing = sorted(set(self.feature_columns).difference(frame.columns))
        if missing:
            raise KeyError(f"transition morphology scoring frame misses {missing}")
        validate_preonset_sequence_columns(frame, self.feature_columns)
        raw = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        standardized = self.scaler.transform(self.imputer.transform(raw))
        return self.pca.transform(standardized)

    def _probability_from_embedding(self, embedding: np.ndarray) -> np.ndarray:
        self._require_fitted()
        raw = self.gmm.predict_proba(embedding)
        return np.asarray(raw[:, self.component_order], dtype=np.float64)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Emit soft type identity/uncertainty; never pooled state identity."""

        embedding = self._embedding(frame)
        probability = self._probability_from_embedding(embedding)
        hard = probability.argmax(axis=1)
        maximum = probability.max(axis=1)
        ordered = np.sort(probability, axis=1)
        margin = maximum - ordered[:, -2] if probability.shape[1] > 1 else maximum
        raw_ids = np.asarray([f"m{index:02d}" for index in hard], dtype=object)
        supported = np.asarray([value in self.supported_component_ids for value in raw_ids])
        accepted = supported & (maximum >= self.config.abstain_min_probability)
        result = pd.DataFrame(index=frame.index)
        for index in range(probability.shape[1]):
            result[f"morphology__posterior_m{index:02d}"] = probability[:, index].astype("float32")
        result["morphology__entropy"] = _entropy(probability).astype("float32")
        result["morphology__top2_margin"] = margin.astype("float32")
        result["morphology__raw_component_id"] = raw_ids
        result["morphology__component_id"] = np.where(accepted, raw_ids, "abstain")
        result["morphology__abstained"] = (~accepted).astype("int8")
        return result


@dataclass
class TransitionClassifierAdapter:
    """Small classifier adapter for transition-vs-stable or morphology labels."""

    random_state: int = 1729
    max_depth: int = 4
    n_estimators: int = 160
    feature_columns: list[str] = field(default_factory=list)
    imputer: SimpleImputer | None = None
    model: Any = None
    classes_: np.ndarray | None = None
    backend: str = "unfitted"

    def fit(
        self,
        train: pd.DataFrame,
        *,
        target_column: str,
        feature_columns: Sequence[str] | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> "TransitionClassifierAdapter":
        if target_column not in train:
            raise KeyError(f"transition classifier target missing: {target_column}")
        columns = (
            eligible_preonset_sequence_columns(train)
            if feature_columns is None
            else validate_preonset_sequence_columns(train, feature_columns)
        )
        if not columns:
            raise ValueError("transition classifier needs pre-onset sequence features")
        y = train[target_column]
        valid = y.notna()
        y = y.loc[valid]
        if y.nunique() < 2:
            raise ValueError("transition classifier target requires at least two classes")
        self.feature_columns = list(columns)
        self.imputer = SimpleImputer(strategy="median")
        x = pd.DataFrame(
            self.imputer.fit_transform(
                train.loc[valid, self.feature_columns].apply(pd.to_numeric, errors="coerce")
            ),
            columns=self.feature_columns,
            index=train.index[valid],
        )
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[np.flatnonzero(valid)]
        if importlib.util.find_spec("lightgbm") is not None:
            from lightgbm import LGBMClassifier

            self.model = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=self.max_depth,
                num_leaves=min(2 ** self.max_depth, 31),
                min_child_samples=8,
                reg_lambda=3.0,
                random_state=self.random_state,
                verbosity=-1,
            )
            self.backend = "lightgbm"
        else:
            self.model = HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=self.n_estimators,
                max_leaf_nodes=min(2 ** self.max_depth, 31),
                l2_regularization=3.0,
                random_state=self.random_state,
            )
            self.backend = "hist_gradient_boosting"
        self.model.fit(x, y, sample_weight=weights)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.imputer is None or self.classes_ is None:
            raise RuntimeError("transition classifier is not fitted")
        validate_preonset_sequence_columns(frame, self.feature_columns)
        x = pd.DataFrame(
            self.imputer.transform(
                frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
            ),
            columns=self.feature_columns,
            index=frame.index,
        )
        probability = np.asarray(self.model.predict_proba(x), dtype=float)
        return pd.DataFrame(
            probability,
            index=frame.index,
            columns=[f"classifier__p_{str(label)}" for label in self.classes_],
        )


@dataclass
class BayesianRuleListChallenger:
    """Binary rule-list challenger with a dependency-free MAP fallback."""

    random_state: int = 1729
    arm: Any = None
    feature_columns: list[str] = field(default_factory=list)
    status: str = "unfitted"
    backend: str = "unfitted"

    @property
    def available(self) -> bool:
        # The challenger is always executable: imodels enables an MCMC BRL,
        # otherwise a clearly labelled native Beta-Binomial MAP list is used.
        return True

    @property
    def imodels_available(self) -> bool:
        return importlib.util.find_spec("imodels") is not None

    def fit(
        self,
        train: pd.DataFrame,
        *,
        target_column: str,
        feature_columns: Sequence[str] | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> "BayesianRuleListChallenger":
        if target_column not in train:
            raise KeyError(f"BRL target missing: {target_column}")
        columns = (
            eligible_preonset_sequence_columns(train)
            if feature_columns is None
            else validate_preonset_sequence_columns(train, feature_columns)
        )
        y = pd.to_numeric(train[target_column], errors="coerce")
        valid = y.notna()
        if y.loc[valid].nunique() != 2:
            raise ValueError("Bayesian Rule List challenger requires a binary target")
        from extreme_price_movements.residual_rule_models import BayesianRuleListArm

        self.feature_columns = list(columns)
        x = train.loc[valid, self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        weights = np.ones(len(x), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)[np.flatnonzero(valid)]
        self.arm = BayesianRuleListArm(seed=self.random_state).fit(
            x,
            y.loc[valid].to_numpy(dtype=np.int8),
            weights,
            self.feature_columns,
        )
        self.status = "fitted"
        self.backend = str(self.arm.backend)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        if self.status != "fitted" or self.arm is None:
            raise RuntimeError(f"Bayesian Rule List challenger is not fitted: {self.status}")
        validate_preonset_sequence_columns(frame, self.feature_columns)
        x = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        return pd.Series(self.arm.predict_proba(x), index=frame.index, name="brl__p_transition")

    def describe(self) -> list[dict[str, Any]]:
        if self.status != "fitted" or self.arm is None:
            raise RuntimeError(f"Bayesian Rule List challenger is not fitted: {self.status}")
        return self.arm.describe()
