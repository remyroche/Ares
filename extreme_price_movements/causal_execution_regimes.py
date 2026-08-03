"""Decision-time, outcome-free state definitions for execution-EV candidates.

This module deliberately defines *state* rather than a profitable regime.  It
never reads outcome, calendar, or sample-weight columns while fitting.  The
only inputs are contemporaneously available candidate/head values; realised
economics are for diagnostics after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler


STATE_SCHEMA = "causal_execution_regime_state_v1"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _js(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float); right = np.asarray(right, dtype=float)
    left /= max(left.sum(), 1.0); right /= max(right.sum(), 1.0)
    mean = (left + right) / 2.0
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0.0))
    return float((kl(left, mean) + kl(right, mean)) / 2.0)


def _squared_distances(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((values[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)


def _bounded_kmeans(values: np.ndarray, k: int, seed: int, iterations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic NumPy KMeans; avoids unbounded native thread pools."""
    rng = np.random.default_rng(seed)
    centres = values[rng.choice(len(values), size=k, replace=False)].copy()
    labels = np.full(len(values), -1, dtype=int)
    for _ in range(iterations):
        new_labels = np.argmin(_squared_distances(values, centres), axis=1)
        new_centres = centres.copy()
        for idx in range(k):
            subset = values[new_labels == idx]
            new_centres[idx] = subset.mean(axis=0) if len(subset) else values[rng.integers(len(values))]
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels, centres = new_labels, new_centres
    return labels, centres


@dataclass
class CausalRegimeStateModel:
    """A KMeans state model whose geometry is entirely fit on prior rows."""

    feature_columns: tuple[str, ...]
    medians: np.ndarray
    scaler: RobustScaler
    centres: np.ndarray
    selected_k: int
    selection: dict[str, dict[str, float]]
    posterior_temperature: float
    training_state_occupancy: np.ndarray
    training_nearest_distance_median: float
    training_nearest_distance_mad_scale: float
    training_nearest_distances_sorted: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        k_values: Sequence[int] = (3, 4, 5),
        random_state: int = 41,
    ) -> "CausalRegimeStateModel":
        """Fit unsupervised states from prior decision-time feature rows only."""
        features = tuple(feature_columns)
        if not features or any(c not in frame for c in features):
            raise ValueError("all causal state features must be present")
        raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        medians = np.nanmedian(raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        raw = np.where(np.isfinite(raw), raw, medians)
        if len(raw) < 100:
            raise ValueError("at least 100 prior rows are required for a stable state model")
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(raw)
        values = scaler.transform(raw)
        # Bounded, deterministic fit sample.  It avoids a regime definition
        # becoming operationally impractical as the handoff grows; sampling is
        # feature/outcome/calendar agnostic and all later diagnostics score the
        # complete training/evaluation sets.
        if len(values) > 5_000:
            fit_values = values[np.random.default_rng(random_state).choice(len(values), size=5_000, replace=False)]
        else:
            fit_values = values
        possible = [int(k) for k in k_values if 2 <= int(k) <= max(2, len(values) // 25)]
        if not possible:
            raise ValueError("not enough rows for requested state counts")
        selection: dict[str, dict[str, float]] = {}
        winner: tuple[float, int] | None = None
        for k in possible:
            assignments = [_bounded_kmeans(fit_values, k, seed)[0] for seed in (17, 29, 43)]
            stability = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(assignments, 2)]))
            occupancy = min(float(np.bincount(a, minlength=k).min() / len(a)) for a in assignments)
            # A predeclared, outcome-free stability criterion.  The small
            # occupancy term rejects brittle tiny states without calendar use.
            objective = stability + 0.20 * occupancy
            selection[str(k)] = {"assignment_stability_ari": stability, "minimum_training_occupancy": occupancy, "objective": objective}
            candidate = (objective, -k)
            if winner is None or candidate > winner:
                winner = candidate
        assert winner is not None
        k = -winner[1]
        _, centres = _bounded_kmeans(fit_values, k, random_state, iterations=80)
        distance2 = _squared_distances(values, centres)
        temperature = float(np.median(np.min(distance2, axis=1)))
        temperature = max(temperature, 1e-6)
        states = np.argmin(distance2, axis=1)
        occupancy = np.bincount(states, minlength=k).astype(float) / len(states)
        nearest = np.min(distance2, axis=1)
        nearest_median = float(np.median(nearest))
        nearest_mad_scale = float(np.median(np.abs(nearest - nearest_median)) * 1.4826)
        return cls(features, medians, scaler, centres, k, selection, temperature, occupancy, nearest_median, max(nearest_mad_scale, 1e-6), np.sort(nearest))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Materialise causal state id, posterior and geometry/trust features."""
        raw = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        raw = np.where(np.isfinite(raw), raw, self.medians)
        values = self.scaler.transform(raw)
        distance2 = _squared_distances(values, self.centres)
        posterior = _softmax(-distance2 / self.posterior_temperature)
        order = np.sort(posterior, axis=1)
        state = np.argmax(posterior, axis=1)
        out = pd.DataFrame(index=frame.index)
        out["causal_regime_state"] = state.astype("int16")
        for idx in range(self.selected_k):
            out[f"causal_regime_posterior_{idx}"] = posterior[:, idx].astype("float32")
        out["causal_regime_entropy"] = (-np.sum(posterior * np.log(np.maximum(posterior, 1e-12)), axis=1) / np.log(self.selected_k)).astype("float32")
        out["causal_regime_top2_margin"] = (order[:, -1] - order[:, -2]).astype("float32")
        out["causal_regime_nearest_distance2"] = np.min(distance2, axis=1).astype("float32")
        # Centre/scale are fitted exclusively on prior training distance.  Do
        # not normalise with an evaluation batch: doing so would make an early
        # decision depend on later candidates in the same week.
        nearest = out["causal_regime_nearest_distance2"].to_numpy(dtype=float)
        out["causal_regime_ood_z"] = ((nearest - self.training_nearest_distance_median) / self.training_nearest_distance_mad_scale).astype("float32")
        # Robust bounded trust feature: empirical percentile of the frozen
        # training distance.  Unlike z, it stays interpretable if training
        # distances have a very small MAD.
        percentile = np.searchsorted(self.training_nearest_distances_sorted, nearest, side="right") / len(self.training_nearest_distances_sorted)
        out["causal_regime_distance_percentile"] = percentile.astype("float32")
        out["causal_regime_distance_exceedance"] = (1.0 - percentile).astype("float32")
        return out

    @property
    def predictor_feature_columns(self) -> tuple[str, ...]:
        """Single-fold inputs: posterior/geometry only, never state ID/labels."""
        return self.fold_local_posterior_columns + self.stable_geometry_feature_columns

    @property
    def fold_local_posterior_columns(self) -> tuple[str, ...]:
        """Posterior coordinates are intentionally local to this fitted model."""
        return tuple(f"causal_regime_posterior_{idx}" for idx in range(self.selected_k))

    @property
    def stable_geometry_feature_columns(self) -> tuple[str, ...]:
        """Permutation-invariant inputs permitted across refitted weekly folds."""
        return (
            "causal_regime_entropy", "causal_regime_top2_margin",
            "causal_regime_nearest_distance2", "causal_regime_distance_percentile",
            "causal_regime_distance_exceedance",
        )

    def training_drift(self, transformed_training: pd.DataFrame, transformed_eval: pd.DataFrame) -> dict[str, float]:
        train_counts = np.bincount(transformed_training["causal_regime_state"], minlength=self.selected_k)
        eval_counts = np.bincount(transformed_eval["causal_regime_state"], minlength=self.selected_k)
        return {
            "state_distribution_js": _js(train_counts, eval_counts),
            "eval_minimum_state_occupancy": float(eval_counts.min() / max(eval_counts.sum(), 1)),
            "eval_mean_ood_z": float(transformed_eval["causal_regime_ood_z"].mean()),
            "eval_p95_ood_z": float(transformed_eval["causal_regime_ood_z"].quantile(0.95)),
        }


def add_regime_transition_labels(
    frame: pd.DataFrame,
    *,
    horizon: pd.Timedelta = pd.Timedelta(hours=6),
    observed_through: pd.Timestamp | None = None,
    time_column: str = "__ts__",
) -> pd.DataFrame:
    """Add post-state labels, with their explicit time of resolution.

    ``change_within_6h`` is one iff any later candidate for the same
    symbol/side has a different assigned state in the following six hours.
    It is resolved exactly at ``decision + 6h``; rows whose horizon has not
    yet elapsed are null.  This is a training/diagnostic label, never a state
    feature at its originating decision.
    """
    required = {time_column, "__symbol__", "side_name", "causal_regime_state"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"missing transition-label columns: {sorted(missing)}")
    out = frame.copy()
    out[time_column] = pd.to_datetime(out[time_column], utc=True, errors="raise")
    known_through = pd.Timestamp(observed_through) if observed_through is not None else out[time_column].max()
    if known_through.tzinfo is None:
        known_through = known_through.tz_localize("UTC")
    resolution = out[time_column] + horizon
    out["causal_regime_change_6h_label_resolution_utc"] = resolution
    out["causal_regime_change_within_6h"] = np.nan
    order = np.lexsort((out[time_column].astype("int64").to_numpy(), out["side_name"].astype(str).to_numpy(), out["__symbol__"].astype(str).to_numpy()))
    labels = np.full(len(out), np.nan, dtype=float)
    sorted_symbols = out["__symbol__"].astype(str).to_numpy()[order]
    sorted_sides = out["side_name"].astype(str).to_numpy()[order]
    boundaries = np.r_[0, np.flatnonzero((sorted_symbols[1:] != sorted_symbols[:-1]) | (sorted_sides[1:] != sorted_sides[:-1])) + 1, len(order)]
    time_ns = out[time_column].astype("int64").to_numpy(); state_values = out["causal_regime_state"].to_numpy(dtype=int)
    resolution_ns = resolution.astype("int64").to_numpy(); known_ns = known_through.value; horizon_ns = horizon.value
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        positions = order[left:right]
        stamps = time_ns[positions]; states = state_values[positions]
        for pos, row_position in enumerate(positions):
            if resolution_ns[row_position] > known_ns:
                continue
            stop = np.searchsorted(stamps, stamps[pos] + horizon_ns, side="right")
            changed = bool(np.any(states[pos + 1:stop] != states[pos]))
            labels[row_position] = float(changed)
    out["causal_regime_change_within_6h"] = labels
    out["causal_regime_persistence_6h"] = 1.0 - out["causal_regime_change_within_6h"]
    return out


__all__ = ["STATE_SCHEMA", "CausalRegimeStateModel", "add_regime_transition_labels"]
