"""Utility routines supporting NAS and TAS regime discovery workflows.

This module contains helper functions and lightweight analytical
infrastructure that both the NAS and TAS regime discovery pipelines use.
The goal is to keep the heavy-weight component file focused on orchestration
while consolidating the feature engineering, evaluation, and scoring logic
that must remain consistent between both search strategies.

The implementation deliberately avoids heavyweight ML dependencies so it can
run inside the existing environment.  Instead, it provides deterministic,
vectorised routines implemented with NumPy/Pandas that approximate the kinds
of statistics the full production system would produce (Sharpe ratios, regime
durations, transition probabilities, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.feature_generation import FeatureCategory, generate_features_by_category, validate_feature_data


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_market_features(
    market_data: pd.DataFrame,
    categories: Optional[Sequence[FeatureCategory]] = None,
) -> pd.DataFrame:
    """Create a rich feature matrix from OHLCV market data using the shared feature bank.

    Instead of re-implementing indicator logic locally, this routine delegates to
    the unified feature generation system that powers the broader platform. The
    generated features therefore remain consistent with the rest of the
    codebase and automatically benefit from any improvements shipped in
    ``src/feature_generation``.

    Args:
        market_data: DataFrame containing at least ``open``, ``high``, ``low``,
            ``close`` and ``volume`` columns ordered by timestamp.
        categories: Optional sequence of feature categories to request from the
            feature bank. When omitted, a balanced default set emphasising
            returns, momentum, volatility, volume, oscillators, and trend is used.

    Returns:
        A DataFrame whose index matches ``market_data`` and contains engineered
        features ready for downstream clustering or modelling.
    """

    df = market_data.copy()
    if df.empty:
        return pd.DataFrame(index=df.index)

    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Market data is missing required columns for feature generation: "
            f"{sorted(missing)}"
        )

    default_categories: Sequence[FeatureCategory] = (
        FeatureCategory.RETURNS,
        FeatureCategory.MOMENTUM,
        FeatureCategory.VOLATILITY,
        FeatureCategory.VOLUME,
        FeatureCategory.OSCILLATOR,
        FeatureCategory.TREND,
        FeatureCategory.SUPPORT_RESISTANCE,
    )
    requested_categories = tuple(categories) if categories else default_categories

    validation = validate_feature_data(df, categories=list(requested_categories))
    missing_columns = validation.get("missing_columns", {})
    if missing_columns:
        formatted = {name: cols for name, cols in missing_columns.items() if cols}
        if formatted:
            raise ValueError(
                "Feature generation requirements not satisfied: "
                f"{formatted}"
            )

    feature_frame = generate_features_by_category(
        data=df,
        categories=list(requested_categories),
        lookback_optimization=False,
        target_column="close",
    )

    if feature_frame.empty:
        return feature_frame.reindex(df.index)

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]

    return feature_frame.reindex(df.index)


# ---------------------------------------------------------------------------
# Regime evaluation metrics
# ---------------------------------------------------------------------------


def _annualised_factor(index: pd.Index) -> float:
    """Best-effort estimate of the annualisation factor for the series."""

    if len(index) < 2:
        return 1.0

    # Try to infer frequency from the median delta in seconds.
    try:
        delta = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
        if delta <= 0:
            return 1.0
        periods_per_year = max(int((365 * 24 * 3600) / delta), 1)
        return float(periods_per_year)
    except Exception:
        return 252.0


def _max_drawdown(cumulative: np.ndarray) -> float:
    if cumulative.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / np.where(running_max == 0, 1.0, running_max) - 1.0
    return float(np.min(drawdowns))


def _run_lengths(assignments: Iterable[int]) -> List[int]:
    lengths: List[int] = []
    current_length = 0
    prev_label: Optional[int] = None
    for label in assignments:
        if prev_label is None or label == prev_label:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1
        prev_label = label
    if current_length > 0:
        lengths.append(current_length)
    return lengths


@dataclass
class RegimePerformance:
    """Container for multi-objective regime metrics."""

    regime_statistics: Dict[int, Dict[str, float]]
    economic_score: float
    trading_score: float
    stability_score: float
    sharpe_ratios: Dict[int, float]
    max_drawdowns: Dict[int, float]
    transition_matrix: Optional[np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        matrix = self.transition_matrix.tolist() if self.transition_matrix is not None else None
        return {
            "regime_statistics": self.regime_statistics,
            "economic_score": self.economic_score,
            "trading_score": self.trading_score,
            "stability_score": self.stability_score,
            "sharpe_ratios": self.sharpe_ratios,
            "max_drawdowns": self.max_drawdowns,
            "transition_matrix": matrix,
        }


def evaluate_regime_performance(
    assignments: List[int],
    market_data: pd.DataFrame,
    features: pd.DataFrame,
    transaction_cost: float = 0.0005,
) -> RegimePerformance:
    """Compute economic, trading, and stability metrics for a regime labelling."""

    if not assignments:
        return RegimePerformance({}, 0.0, 0.0, 0.0, {}, {}, None)

    series = pd.Series(assignments, index=features.index[: len(assignments)])
    realised_returns = features["return"].iloc[: len(assignments)].fillna(0.0)
    periods_per_year = _annualised_factor(features.index)

    statistics: Dict[int, Dict[str, float]] = {}
    sharpe_by_regime: Dict[int, float] = {}
    drawdown_by_regime: Dict[int, float] = {}

    total_turnover = 0
    prev_label = None
    transitions = np.zeros((series.nunique(), series.nunique())) if series.nunique() > 0 else None

    for regime, regime_returns in realised_returns.groupby(series.values):
        mask = series.values == regime
        regime_series = realised_returns[mask]
        regime_volume = market_data["volume"].iloc[: len(assignments)][mask]

        mean_ret = regime_series.mean()
        vol = regime_series.std(ddof=0)
        sharpe = 0.0
        if vol > 0:
            sharpe = (mean_ret * periods_per_year) / (vol * math.sqrt(periods_per_year))

        cumulative = (1 + regime_series).cumprod().values
        max_dd = _max_drawdown(cumulative)

        avg_volume = float(regime_volume.mean()) if not regime_volume.empty else 0.0
        statistics[int(regime)] = {
            "mean_return": float(mean_ret),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "avg_volume": avg_volume,
        }
        sharpe_by_regime[int(regime)] = float(sharpe)
        drawdown_by_regime[int(regime)] = float(max_dd)

    # Transaction cost / turnover estimation
    changes = series.values[:-1] != series.values[1:]
    total_turnover = float(np.sum(changes))
    net_return = realised_returns.mean() * periods_per_year - transaction_cost * total_turnover

    # Transition probabilities
    if transitions is not None:
        unique_labels = sorted(statistics.keys())
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        transitions = np.zeros((len(unique_labels), len(unique_labels)))
        for a, b in zip(series.values[:-1], series.values[1:]):
            transitions[label_to_idx[int(a)], label_to_idx[int(b)]] += 1
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transitions = transitions / row_sums

    # Regime duration / persistence
    durations = _run_lengths(series.values)
    avg_duration = float(np.mean(durations)) if durations else 0.0
    persistence = float(np.mean(np.diag(transitions))) if transitions is not None else 0.0

    # Volatility clustering proxy: autocorrelation of squared returns
    squared_returns = realised_returns.pow(2)
    if len(squared_returns) > 1:
        volatility_clustering = float(squared_returns.autocorr(lag=1))
    else:
        volatility_clustering = 0.0

    economic_score = float(np.mean(list(sharpe_by_regime.values()))) if sharpe_by_regime else 0.0
    trading_score = float(net_return)
    stability_score = float((persistence + avg_duration / max(len(assignments), 1) + (1 - abs(volatility_clustering))) / 3)

    return RegimePerformance(
        statistics,
        economic_score,
        trading_score,
        stability_score,
        sharpe_by_regime,
        drawdown_by_regime,
        transitions,
    )


def aggregate_scores(
    economic: float,
    trading: float,
    stability: float,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """Combine multi-objective metrics into a single comparable score."""

    econ, trade, stab = weights

    # Normalise the raw scores using tanh to keep them bounded in [-1, 1], then
    # map back to [0, 1] so they can be compared directly.
    norm_economic = (math.tanh(economic) + 1) / 2
    norm_trading = (math.tanh(trading) + 1) / 2
    norm_stability = (math.tanh(stability) + 1) / 2

    return float(
        econ * norm_economic
        + trade * norm_trading
        + stab * norm_stability
    )


def compute_transition_statistics(assignments: List[int]) -> Dict[str, float]:
    """Return lightweight transition statistics used by both NAS and TAS."""

    if not assignments:
        return {"avg_duration": 0.0, "persistence": 0.0}

    run_lengths = _run_lengths(assignments)
    avg_duration = float(np.mean(run_lengths)) if run_lengths else 0.0

    transitions = 0
    stays = 0
    for a, b in zip(assignments[:-1], assignments[1:]):
        if a == b:
            stays += 1
        else:
            transitions += 1
    total_pairs = max(len(assignments) - 1, 1)

    return {
        "avg_duration": avg_duration,
        "persistence": stays / total_pairs,
        "transition_rate": transitions / total_pairs,
    }


def calculate_regime_distribution(assignments: List[int]) -> Dict[int, float]:
    """Probability distribution over regimes."""

    if not assignments:
        return {}

    total = len(assignments)
    distribution: Dict[int, float] = {}
    for label in assignments:
        distribution[label] = distribution.get(label, 0) + 1
    return {k: v / total for k, v in distribution.items()}


# ---------------------------------------------------------------------------
# Simple evolutionary search helpers
# ---------------------------------------------------------------------------


def random_choice(items: List[int], rng: np.random.Generator) -> int:
    return int(items[rng.integers(0, len(items))])


def mutate_value(value: int, low: int, high: int, rng: np.random.Generator) -> int:
    if low == high:
        return low
    perturb = rng.integers(-1, 2)
    new_value = int(np.clip(value + perturb, low, high))
    return new_value


def kmeans(data: np.ndarray, n_clusters: int, iterations: int = 25, rng_seed: int = 42) -> np.ndarray:
    """A lightweight KMeans implementation for unsupervised regime discovery."""

    if data.size == 0 or n_clusters <= 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(rng_seed)
    n_samples = data.shape[0]
    n_clusters = int(np.clip(n_clusters, 1, n_samples))

    # Initialise centroids using random samples
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = data[indices]

    for _ in range(iterations):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for k in range(n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)

    return labels.astype(int)


def quantile_regimes(feature: pd.Series, n_regimes: int) -> np.ndarray:
    """Assign regimes by quantiling a single feature (used for TAS splitting)."""

    if feature.empty:
        return np.array([], dtype=int)

    unique = feature.nunique()
    if unique < n_regimes:
        # Fallback to ranking when there are too few unique values.
        ranks = feature.rank(method="dense").astype(int) - 1
        return np.clip(ranks, 0, n_regimes - 1).to_numpy()

    bins = pd.qcut(feature, q=n_regimes, labels=False, duplicates="drop")
    return bins.to_numpy().astype(int)


def adaptive_regime_count(
    base_count: int,
    volatility: float,
    consensus: float,
    min_regimes: int = 3,
    max_regimes: int = 12,
) -> int:
    """Derive a dynamic regime count using volatility and consensus cues."""

    volatility_adjustment = 0
    if volatility > 0.03:
        volatility_adjustment += 2
    elif volatility > 0.015:
        volatility_adjustment += 1

    consensus_adjustment = -1 if consensus > 0.65 else 1 if consensus < 0.35 else 0

    regime_count = base_count + volatility_adjustment + consensus_adjustment
    return int(np.clip(regime_count, min_regimes, max_regimes))


# ---------------------------------------------------------------------------
# Search classes
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    assignments: List[int]
    architecture: Dict[str, Any]
    metrics: RegimePerformance
    score: float


class NASRegimeSearch:
    """Light-weight neural architecture search for regime detection."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        config: Dict[str, Any],
    ) -> None:
        self.market_data = market_data
        self.features = features
        self.config = config
        self.population_size = max(int(config.get("population_size", 20)), 4)
        self.generations = max(int(config.get("generations", 10)), 1)
        self.weights = (
            float(config.get("economic_weight", 0.4)),
            float(config.get("trading_weight", 0.3)),
            float(config.get("stability_weight", 0.3)),
        )
        self.base_regime_count = int(config.get("n_regimes", 6))
        self.rng = np.random.default_rng(int(config.get("rng_seed", 42)))

        volatility = float(features["realized_vol"].mean()) if "realized_vol" in features else 0.0
        consensus_proxy = float(features[["return", "macd"]].corr().iloc[0, 1]) if {
            "return",
            "macd",
        }.issubset(features.columns) else 0.0
        self.dynamic_regime_count = adaptive_regime_count(
            self.base_regime_count,
            volatility,
            consensus_proxy if not math.isnan(consensus_proxy) else 0.5,
        )

    # --- candidate management -------------------------------------------------

    def _sample_architecture(self) -> Dict[str, Any]:
        architecture_type = self.rng.choice(["lstm", "transformer", "cnn", "hybrid"])
        hidden_dim = int(self.rng.integers(16, 129))
        depth = int(self.rng.integers(1, 5))
        dropout = float(self.rng.uniform(0.0, 0.5))
        attention_heads = int(self.rng.integers(1, 5)) if architecture_type in {"transformer", "hybrid"} else 0
        n_regimes = mutate_value(self.dynamic_regime_count, 3, 12, self.rng)
        return {
            "type": architecture_type,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "dropout": dropout,
            "attention_heads": attention_heads,
            "n_regimes": n_regimes,
        }

    def _prepare_representation(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Simulate neural representations based on the architecture type."""

        window = max(architecture.get("depth", 1) * 3, 3)
        feature_block = self.features.copy()

        if architecture["type"] == "lstm":
            rolling = feature_block.rolling(window=window, min_periods=1).mean()
            representation = rolling.to_numpy()
        elif architecture["type"] == "cnn":
            shifted = feature_block.diff().fillna(0.0)
            representation = np.concatenate(
                [feature_block.to_numpy(), shifted.to_numpy()], axis=1
            )
        elif architecture["type"] == "transformer":
            attention_window = max(architecture.get("attention_heads", 1) * 5, 5)
            attention_features = feature_block.rolling(attention_window, min_periods=1).apply(np.mean)
            representation = attention_features.to_numpy()
        else:  # hybrid
            rep_a = feature_block.rolling(window=window, min_periods=1).mean().to_numpy()
            rep_b = feature_block.rolling(window=window, min_periods=1).std().fillna(0.0).to_numpy()
            representation = np.concatenate([rep_a, rep_b], axis=1)

        # Normalise to zero mean / unit variance for clustering stability
        representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)
        if representation.size == 0:
            return representation
        mean = representation.mean(axis=0, keepdims=True)
        std = representation.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (representation - mean) / std

    def _evaluate_candidate(self, architecture: Dict[str, Any]) -> CandidateResult:
        representation = self._prepare_representation(architecture)
        if representation.size == 0:
            assignments = np.zeros(len(self.features), dtype=int)
        else:
            assignments = kmeans(
                representation,
                architecture.get("n_regimes", self.dynamic_regime_count),
                iterations=30,
                rng_seed=int(self.rng.integers(0, 1_000_000)),
            )

        assignments_list = assignments.tolist()
        metrics = evaluate_regime_performance(assignments_list, self.market_data, self.features)
        score = aggregate_scores(
            metrics.economic_score,
            metrics.trading_score,
            metrics.stability_score,
            self.weights,
        )
        return CandidateResult(assignments_list, architecture, metrics, score)

    def _mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        mutated = architecture.copy()
        mutated["n_regimes"] = mutate_value(mutated["n_regimes"], 3, 12, self.rng)
        mutated["hidden_dim"] = int(np.clip(mutated["hidden_dim"] + self.rng.integers(-16, 17), 16, 256))
        mutated["dropout"] = float(np.clip(mutated["dropout"] + self.rng.normal(0, 0.05), 0.0, 0.6))
        mutated["depth"] = int(np.clip(mutated["depth"] + self.rng.integers(-1, 2), 1, 6))
        if mutated["type"] in {"transformer", "hybrid"}:
            mutated["attention_heads"] = int(np.clip(mutated.get("attention_heads", 2) + self.rng.integers(-1, 2), 1, 8))
        return mutated

    def run(self) -> Dict[str, Any]:
        population = [self._sample_architecture() for _ in range(self.population_size)]
        history: List[Dict[str, Any]] = []
        best_candidate: Optional[CandidateResult] = None

        for generation in range(self.generations):
            evaluated: List[CandidateResult] = [self._evaluate_candidate(arch) for arch in population]
            evaluated.sort(key=lambda c: c.score, reverse=True)

            best_in_gen = evaluated[0]
            history.append(
                {
                    "generation": generation,
                    "score": best_in_gen.score,
                    "architecture": best_in_gen.architecture,
                }
            )

            if best_candidate is None or best_in_gen.score > best_candidate.score:
                best_candidate = best_in_gen

            elite_count = max(2, self.population_size // 3)
            elites = [candidate.architecture for candidate in evaluated[:elite_count]]

            # Generate new population via mutation + crossover style sampling
            new_population: List[Dict[str, Any]] = elites.copy()
            while len(new_population) < self.population_size:
                parent = elites[self.rng.integers(0, len(elites))]
                mutated = self._mutate_architecture(parent)
                new_population.append(mutated)

            population = new_population

        if best_candidate is None:
            best_candidate = self._evaluate_candidate(self._sample_architecture())

        return {
            "best_candidate": best_candidate.architecture,
            "assignments": best_candidate.assignments,
            "metrics": best_candidate.metrics,
            "score": best_candidate.score,
            "history": history,
        }


class TASRegimeSearch:
    """Tree-inspired statistical regime discovery search."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        config: Dict[str, Any],
        volatility_estimate: float,
    ) -> None:
        self.market_data = market_data
        self.features = features
        self.config = config
        self.population_size = max(int(config.get("population_size", 30)), 4)
        self.generations = max(int(config.get("generations", 8)), 1)
        self.weights = (
            float(config.get("economic_weight", 0.4)),
            float(config.get("trading_weight", 0.3)),
            float(config.get("stability_weight", 0.3)),
        )
        self.base_regime_count = int(config.get("n_regimes", 6))
        self.volatility_estimate = volatility_estimate
        self.rng = np.random.default_rng(int(config.get("rng_seed", 123)))

        consensus_proxy = float(features[["return", "volume_z"]].corr().iloc[0, 1]) if {
            "return",
            "volume_z",
        }.issubset(features.columns) else 0.5
        self.dynamic_regime_count = adaptive_regime_count(
            self.base_regime_count,
            self.volatility_estimate,
            consensus_proxy if not math.isnan(consensus_proxy) else 0.5,
        )

    def _sample_configuration(self) -> Dict[str, Any]:
        n_regimes = mutate_value(self.dynamic_regime_count, 3, 15, self.rng)
        depth = int(self.rng.integers(2, 6))
        min_leaf = int(self.rng.integers(5, 25))
        features_subset = self.rng.choice(self.features.columns, size=min(len(self.features.columns), depth + 2), replace=False)
        return {
            "n_regimes": n_regimes,
            "depth": depth,
            "min_leaf": min_leaf,
            "features": list(features_subset),
        }

    def _apply_tree_segmentation(self, configuration: Dict[str, Any]) -> np.ndarray:
        n_regimes = configuration["n_regimes"]
        selected_features = configuration["features"]
        assignments = np.zeros(len(self.features), dtype=int)

        # Construct a pseudo decision tree by successively quantiling selected features
        for level, feature_name in enumerate(selected_features, start=1):
            feature = self.features[feature_name]
            bins = quantile_regimes(feature, n_regimes)
            assignments = (assignments + bins * level) % n_regimes

        return assignments

    def _evaluate_configuration(self, configuration: Dict[str, Any]) -> CandidateResult:
        assignments = self._apply_tree_segmentation(configuration)
        assignments_list = assignments.tolist()
        metrics = evaluate_regime_performance(assignments_list, self.market_data, self.features)
        score = aggregate_scores(
            metrics.economic_score,
            metrics.trading_score,
            metrics.stability_score,
            self.weights,
        )
        return CandidateResult(assignments_list, configuration, metrics, score)

    def _mutate_configuration(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        mutated = configuration.copy()
        mutated["n_regimes"] = mutate_value(mutated["n_regimes"], 3, 15, self.rng)
        mutated["depth"] = int(np.clip(mutated["depth"] + self.rng.integers(-1, 2), 2, 8))
        mutated_features = set(mutated["features"])
        if self.rng.random() < 0.5 and len(mutated_features) > 1:
            mutated_features.remove(self.rng.choice(list(mutated_features)))
        if self.rng.random() < 0.5:
            mutated_features.add(self.rng.choice(self.features.columns))
        mutated["features"] = list(mutated_features)[: mutated["depth"] + 2]
        return mutated

    def run(self) -> Dict[str, Any]:
        population = [self._sample_configuration() for _ in range(self.population_size)]
        history: List[Dict[str, Any]] = []
        best_candidate: Optional[CandidateResult] = None

        for generation in range(self.generations):
            evaluated: List[CandidateResult] = [self._evaluate_configuration(config) for config in population]
            evaluated.sort(key=lambda c: c.score, reverse=True)

            best_in_gen = evaluated[0]
            history.append(
                {
                    "generation": generation,
                    "score": best_in_gen.score,
                    "configuration": best_in_gen.architecture,
                }
            )

            if best_candidate is None or best_in_gen.score > best_candidate.score:
                best_candidate = best_in_gen

            elite_count = max(2, self.population_size // 3)
            elites = [candidate.architecture for candidate in evaluated[:elite_count]]

            new_population: List[Dict[str, Any]] = elites.copy()
            while len(new_population) < self.population_size:
                parent = elites[self.rng.integers(0, len(elites))]
                mutated = self._mutate_configuration(parent)
                new_population.append(mutated)

            population = new_population

        if best_candidate is None:
            best_candidate = self._evaluate_configuration(self._sample_configuration())

        return {
            "best_candidate": best_candidate.architecture,
            "assignments": best_candidate.assignments,
            "metrics": best_candidate.metrics,
            "score": best_candidate.score,
            "history": history,
        }

