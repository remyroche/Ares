"""Causal trust/distribution models for strict-R3 sizing ablations.

The module deliberately keeps candidate identity, score ranking, execution,
and causal EV admission outside the model.  Every estimator receives a frozen
expected-net estimate and may only change the bounded authority assigned to
that estimate and the resulting position size.

Raw K9 membership/archetype fields are prohibited by default.  Their meanings
are bundle-local and cannot be pooled across rolling geometry definitions.
Aggregate entropy, margin, OOD, drift, and support fields are stable semantic
summaries and remain eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import SplineTransformer


SEED = 20260810
RAW_K9_PREFIX = "k09__cluster_"
RESIDUAL_BANDS_BPS = (-200.0, -50.0, 50.0, 200.0)
DEFAULT_FEATURE_BINS = 5
DEFAULT_PARENT_BINS = 10


@dataclass(frozen=True)
class TrustModelSpec:
    name: str
    pipeline: str
    model_family: str
    interactions: str = "none"
    cmi_weighting: str = "uniform"
    lambda_max: float = 1.0
    risk_mode: str = "unconditional"
    sizing_mode: str = "mean"
    probability_mode: str = "raw"
    target_mode: str = "policy_net"
    mean_weighting: str | None = None
    support_mode: str = "row_count"
    uncertainty_mode: str = "global_residual"

    def validate(self) -> None:
        if self.pipeline not in {"bayesian", "gam", "nonlinear"}:
            raise ValueError(f"unknown pipeline {self.pipeline!r}")
        if self.interactions not in {"none", "pooled_cmi", "stable_cmi"}:
            raise ValueError(f"unknown interaction mode {self.interactions!r}")
        if self.cmi_weighting not in {
            "uniform", "rank", "rank_loss", "rank_false_positive",
            "rank_loss_false_positive",
        }:
            raise ValueError(f"unknown CMI weighting {self.cmi_weighting!r}")
        if self.lambda_max not in {1.0, 1.10, 1.25}:
            raise ValueError("lambda_max must be 1.0, 1.10, or 1.25")
        if self.risk_mode not in {"unconditional", "singleton", "stable_cmi"}:
            raise ValueError(f"unknown risk mode {self.risk_mode!r}")
        if self.sizing_mode not in {"equal", "raw_ev", "mean", "mean_risk", "predictive"}:
            raise ValueError(f"unknown sizing mode {self.sizing_mode!r}")
        if self.target_mode not in {
            "policy_net", "cell_day_residual_clip300", "cell_day_residual_clip500",
        }:
            raise ValueError(f"unknown trust target mode {self.target_mode!r}")
        if self.mean_weighting not in {
            None, "uniform", "rank", "rank_loss", "rank_false_positive",
            "rank_loss_false_positive",
        }:
            raise ValueError(f"unknown mean weighting {self.mean_weighting!r}")
        if self.support_mode not in {"row_count", "independent_experience"}:
            raise ValueError(f"unknown support mode {self.support_mode!r}")
        if self.uncertainty_mode not in {"global_residual", "local_leaf"}:
            raise ValueError(f"unknown uncertainty mode {self.uncertainty_mode!r}")


@dataclass
class TrustPrediction:
    expected_bps: np.ndarray
    shrinkage_lambda: np.ndarray
    predictive_sd_bps: np.ndarray
    mean_sd_bps: np.ndarray
    q10_bps: np.ndarray
    q50_bps: np.ndarray
    q90_bps: np.ndarray
    p_ev_positive: np.ndarray
    p_adverse_tail: np.ndarray
    effective_support: np.ndarray
    residual_mean_bps: np.ndarray | None = None
    residual_q10_bps: np.ndarray | None = None
    residual_q25_bps: np.ndarray | None = None
    p_map_overestimate_50bps: np.ndarray | None = None
    p_map_overestimate_100bps: np.ndarray | None = None
    p_map_overestimate_200bps: np.ndarray | None = None

    def as_frame(self) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                "posterior_expected_bps": self.expected_bps.astype(np.float32),
                "shrinkage_lambda": self.shrinkage_lambda.astype(np.float32),
                "posterior_predictive_sd": self.predictive_sd_bps.astype(np.float32),
                "posterior_mean_sd": self.mean_sd_bps.astype(np.float32),
                "posterior_predictive_q10": self.q10_bps.astype(np.float32),
                "posterior_predictive_q50": self.q50_bps.astype(np.float32),
                "posterior_predictive_q90": self.q90_bps.astype(np.float32),
                "p_ev_positive": self.p_ev_positive.astype(np.float32),
                "p_adverse_tail": self.p_adverse_tail.astype(np.float32),
                "trust_effective_support": self.effective_support.astype(np.float32),
            }
        )
        optional = {
            "posterior_residual_mean_bps": self.residual_mean_bps,
            "posterior_residual_q10_bps": self.residual_q10_bps,
            "posterior_residual_q25_bps": self.residual_q25_bps,
            "p_map_overestimate_50bps": self.p_map_overestimate_50bps,
            "p_map_overestimate_100bps": self.p_map_overestimate_100bps,
            "p_map_overestimate_200bps": self.p_map_overestimate_200bps,
        }
        for field, value in optional.items():
            if value is not None:
                output[field] = np.asarray(value, dtype=np.float32)
        return output


@dataclass(frozen=True)
class ParentExpectation:
    edges: np.ndarray
    means: np.ndarray
    global_mean: float

    @classmethod
    def fit(
        cls,
        score: Sequence[float],
        realised: Sequence[float],
        *,
        bins: int = DEFAULT_PARENT_BINS,
        prior_strength: float = 500.0,
    ) -> "ParentExpectation":
        x = np.asarray(score, dtype=float)
        y = np.asarray(realised, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 100:
            raise ValueError("parent expectation has insufficient support")
        x, y = x[valid], y[valid]
        quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
        edges = np.unique(np.quantile(x, quantiles, method="linear"))
        code = np.digitize(x, edges, right=True)
        global_mean = float(np.mean(y))
        means = np.full(len(edges) + 1, global_mean, dtype=float)
        for idx in range(len(means)):
            local = y[code == idx]
            if len(local):
                weight = len(local) / (len(local) + float(prior_strength))
                means[idx] = weight * float(np.mean(local)) + (1.0 - weight) * global_mean
        return cls(edges.astype(float), means, global_mean)

    def predict(self, score: Sequence[float]) -> np.ndarray:
        x = np.asarray(score, dtype=float)
        code = np.digitize(np.nan_to_num(x, nan=0.5), self.edges, right=True)
        return self.means[np.clip(code, 0, len(self.means) - 1)]


@dataclass(frozen=True)
class RobustTransform:
    fields: tuple[str, ...]
    medians: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, fields: Sequence[str]) -> "RobustTransform":
        assert_geometry_semantics(frame, fields)
        x = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        q25 = np.nanquantile(x, 0.25, axis=0)
        q75 = np.nanquantile(x, 0.75, axis=0)
        scales = q75 - q25
        scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
        return cls(tuple(fields), medians.astype(float), scales.astype(float))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, list(self.fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        x = np.where(np.isfinite(x), x, self.medians)
        return np.clip((x - self.medians) / self.scales, -8.0, 8.0).astype(np.float32)


@dataclass(frozen=True)
class QuantileBins:
    fields: tuple[str, ...]
    edges: tuple[np.ndarray, ...]

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, fields: Sequence[str], *, bins: int = DEFAULT_FEATURE_BINS,
    ) -> "QuantileBins":
        assert_geometry_semantics(frame, fields)
        output: list[np.ndarray] = []
        for field in fields:
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite) < 20:
                output.append(np.asarray([], dtype=float))
                continue
            edge = np.unique(
                np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1], method="linear")
            )
            output.append(edge.astype(float))
        return cls(tuple(fields), tuple(output))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(frame), len(self.fields)), dtype=np.int16)
        for idx, (field, edges) in enumerate(zip(self.fields, self.edges)):
            value = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
            finite = np.isfinite(value)
            result[:, idx] = np.digitize(np.where(finite, value, -np.inf), edges, right=True)
            result[~finite, idx] = len(edges) + 1
        return result


@dataclass(frozen=True)
class CMIEdge:
    left: str
    right: str
    gain: float
    recurrence: float
    family_left: str
    family_right: str


def assert_geometry_semantics(frame: pd.DataFrame, fields: Sequence[str]) -> None:
    """Reject pooled raw K9/archetype semantics across bundle identities."""

    raw = [field for field in fields if str(field).startswith(RAW_K9_PREFIX)]
    if not raw:
        return
    if "geometry_bundle_sha256" not in frame:
        raise ValueError("raw K9 fields require explicit geometry bundle identity")
    identities = frame["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(identities) != 1:
        raise ValueError(
            "raw K9/archetype features may only use rows from one identical "
            "Geometry/K9 bundle"
        )


def trust_feature_family(name: str) -> str:
    text = str(name).lower()
    if "support" in text or "coverage" in text:
        return "support"
    if "ood" in text or "drift" in text:
        return "ood"
    if "cov" in text or "corr" in text or "eigen" in text:
        return "covariance"
    if "recent" in text or "correct" in text or "adverse" in text or "approx" in text:
        return "correctness"
    if "consensus" in text or "gap" in text or "base_" in text or "upstream" in text:
        return "cross_model"
    if "k9" in text or "leaf" in text or "geometry" in text:
        return "geometry"
    return "other"


def _weighted_mi(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    w = np.asarray(weight, dtype=float)
    valid = np.isfinite(w) & (w > 0)
    if valid.sum() < 20:
        return 0.0
    x, y, w = x[valid], y[valid], w[valid]
    nx = int(x.max()) + 1
    ny = int(y.max()) + 1
    joint = np.bincount(x * ny + y, weights=w, minlength=nx * ny).reshape(nx, ny)
    joint += 0.5
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    return float(np.sum(joint * np.log(joint / (px * py))))


def _weighted_cmi(
    x: np.ndarray, y: np.ndarray, baseline: np.ndarray, weight: np.ndarray,
) -> float:
    total = float(np.sum(weight))
    if total <= 0:
        return 0.0
    result = 0.0
    for state in np.unique(baseline):
        mask = baseline == state
        local = float(np.sum(weight[mask]))
        if local > 0:
            result += (local / total) * _weighted_mi(x[mask], y[mask], weight[mask])
    return float(result)


def residual_classes(realised: Sequence[float], expected: Sequence[float]) -> np.ndarray:
    residual = np.asarray(realised, dtype=float) - np.asarray(expected, dtype=float)
    return np.digitize(residual, np.asarray(RESIDUAL_BANDS_BPS), right=True).astype(np.int8)


def cmi_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    score = pd.to_numeric(frame["final_score"], errors="coerce").fillna(0.0).to_numpy(float)
    realised = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(frame["raw_expected_bps"], errors="coerce").to_numpy(float)
    rank_weight = 1.0 + 2.0 * np.clip(score, 0.0, 1.0) ** 2
    weight = np.ones(len(frame), dtype=float) if mode == "uniform" else rank_weight
    if mode in {"rank_loss", "rank_loss_false_positive"}:
        weight *= 1.0 + np.clip(-realised, 0.0, 500.0) / 500.0
    if mode in {"rank_false_positive", "rank_loss_false_positive"}:
        weight *= 1.0 + np.clip(expected - realised, 0.0, 500.0) / 250.0
    return np.clip(weight, 0.25, 6.0)


def independent_experience_support(
    frame: pd.DataFrame,
    mask: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Conservative leaf support across independent market experience."""

    mask = np.asarray(mask, dtype=bool)
    local = frame.loc[mask]
    weight = np.asarray(sample_weight, dtype=float)[mask]
    if local.empty or not np.isfinite(weight).all() or np.sum(weight) <= 0.0:
        return 0.0
    row_ess = float(np.sum(weight) ** 2 / np.sum(weight**2))
    timestamp = pd.to_datetime(local["__decision_ts__"], utc=True, errors="raise")
    days = int(timestamp.dt.floor("D").nunique())
    blocks = int(timestamp.dt.floor("12h").nunique())
    assets = int(local["__symbol__"].astype(str).nunique())
    months = int((timestamp.dt.year * 12 + timestamp.dt.month).nunique())
    components = np.asarray([
        row_ess,
        min(row_ess, 8.0 * days),
        min(row_ess, 4.0 * blocks),
        min(row_ess, 16.0 * assets),
        min(row_ess, 64.0 * months),
    ], dtype=float)
    components = np.maximum(components, 1.0)
    return float(min(row_ess, np.exp(np.mean(np.log(components)))))


def discover_cmi_edges(
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    mode: str,
    stable: bool,
    max_edges: int = 8,
    sample_cap: int = 40_000,
) -> tuple[list[CMIEdge], QuantileBins]:
    """Discover compact train-only cross-family residual-information edges."""

    assert_geometry_semantics(frame, fields)
    work = frame.copy()
    if len(work) > sample_cap:
        index = np.linspace(0, len(work) - 1, sample_cap).round().astype(int)
        work = work.iloc[index].copy()
    bins = QuantileBins.fit(work, fields)
    xbin = bins.transform(work)
    y = residual_classes(work["policy_net_bps"], work["raw_expected_bps"])
    m = pd.to_numeric(work["raw_expected_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    medges = np.unique(np.quantile(m, [0.2, 0.4, 0.6, 0.8], method="linear"))
    baseline = np.digitize(m, medges, right=True)
    weights = cmi_weights(work, mode)
    singleton = np.asarray(
        [_weighted_cmi(xbin[:, idx], y, baseline, weights) for idx in range(len(fields))]
    )
    candidate_indices: list[int] = []
    families = [trust_feature_family(field) for field in fields]
    for family in sorted(set(families)):
        local = [idx for idx, value in enumerate(families) if value == family]
        local = sorted(local, key=lambda idx: singleton[idx], reverse=True)[:4]
        candidate_indices.extend(local)
    pooled: list[tuple[float, int, int]] = []
    for offset, left in enumerate(candidate_indices):
        for right in candidate_indices[offset + 1 :]:
            if families[left] == families[right]:
                continue
            joint = xbin[:, left] * 16 + xbin[:, right]
            gain = _weighted_cmi(joint, y, baseline, weights) - max(
                singleton[left], singleton[right]
            )
            pooled.append((float(gain), left, right))
    pooled.sort(reverse=True)
    if not pooled:
        return [], bins

    recurrence: dict[tuple[int, int], int] = {}
    month = pd.to_datetime(work["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    periods = sorted(month.unique())
    if stable and len(periods) >= 2:
        for period in periods:
            mask = month.eq(period).to_numpy()
            if mask.sum() < 500:
                continue
            local_scores: list[tuple[float, int, int]] = []
            for _pooled_gain, left, right in pooled[: max(40, max_edges * 5)]:
                joint = xbin[mask, left] * 16 + xbin[mask, right]
                gain = _weighted_cmi(joint, y[mask], baseline[mask], weights[mask]) - max(
                    _weighted_cmi(xbin[mask, left], y[mask], baseline[mask], weights[mask]),
                    _weighted_cmi(xbin[mask, right], y[mask], baseline[mask], weights[mask]),
                )
                local_scores.append((gain, left, right))
            for _gain, left, right in sorted(local_scores, reverse=True)[:max_edges]:
                recurrence[(left, right)] = recurrence.get((left, right), 0) + 1
    selected: list[CMIEdge] = []
    required = max(1, math.ceil(len(periods) / 2)) if stable else 0
    for gain, left, right in pooled:
        count = recurrence.get((left, right), 0)
        if stable and count < required:
            continue
        selected.append(
            CMIEdge(
                str(fields[left]), str(fields[right]), gain,
                count / max(len(periods), 1), families[left], families[right],
            )
        )
        if len(selected) >= max_edges:
            break
    if stable and not selected:
        for gain, left, right in pooled[:max_edges]:
            selected.append(
                CMIEdge(
                    str(fields[left]), str(fields[right]), gain, 0.0,
                    families[left], families[right],
                )
            )
    return selected, bins


def _ideal_lambda(
    realised: np.ndarray, expected: np.ndarray, parent: np.ndarray, lambda_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance = expected - parent
    stable = np.abs(distance) >= 10.0
    target = np.ones(len(realised), dtype=float)
    target[stable] = (realised[stable] - parent[stable]) / distance[stable]
    target = np.clip(target, 0.0, float(lambda_max))
    authority = np.clip(np.abs(distance) / 100.0, 0.05, 4.0)
    return target, authority


def _predictive_outputs(
    mean: np.ndarray,
    lam: np.ndarray,
    sigma: np.ndarray,
    support: np.ndarray,
    *,
    mean_sd: np.ndarray | None = None,
) -> TrustPrediction:
    sigma = np.clip(np.asarray(sigma, dtype=float), 25.0, 2_000.0)
    support = np.clip(np.asarray(support, dtype=float), 1.0, 1_000_000.0)
    if mean_sd is None:
        mean_sd = sigma / np.sqrt(support)
    predictive = np.sqrt(sigma**2 + np.asarray(mean_sd, dtype=float) ** 2)
    nu = 5.0
    q10 = mean + student_t.ppf(0.10, df=nu) * predictive
    q90 = mean + student_t.ppf(0.90, df=nu) * predictive
    p_positive = 1.0 - student_t.cdf((0.0 - mean) / predictive, df=nu)
    p_adverse = student_t.cdf((-200.0 - mean) / predictive, df=nu)
    return TrustPrediction(
        np.asarray(mean, dtype=float), np.asarray(lam, dtype=float), predictive,
        np.asarray(mean_sd, dtype=float), q10, np.asarray(mean, dtype=float), q90,
        p_positive, p_adverse, support,
    )


def _cell_posterior(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    prior_mean: float,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(max(train_codes.max(initial=0), test_codes.max(initial=0))) + 1
    support = np.bincount(train_codes, weights=weight, minlength=count)
    total = np.bincount(train_codes, weights=weight * target, minlength=count)
    mean = (total + prior_strength * prior_mean) / (support + prior_strength)
    code = np.clip(test_codes, 0, count - 1)
    return mean[code], support[code]


def fit_empirical_bayes(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    bins: QuantileBins,
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    train_bin = bins.transform(train)
    score_bin = bins.transform(score)
    field_index = {field: idx for idx, field in enumerate(fields)}
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent = pd.to_numeric(train["parent_expected_bps"], errors="coerce").to_numpy(float)
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent_score = pd.to_numeric(score["parent_expected_bps"], errors="coerce").to_numpy(float)
    target_lambda, authority = _ideal_lambda(realised, expected, parent, spec.lambda_max)
    weight = cmi_weights(train, spec.cmi_weighting) * authority
    effects_train: list[np.ndarray] = []
    effects_score: list[np.ndarray] = []
    supports_train: list[np.ndarray] = []
    supports_score: list[np.ndarray] = []
    for idx in range(len(fields)):
        fit_t, support_t = _cell_posterior(
            train_bin[:, idx], train_bin[:, idx], target_lambda, weight,
            prior_mean=1.0, prior_strength=200.0,
        )
        fit_s, support_s = _cell_posterior(
            train_bin[:, idx], score_bin[:, idx], target_lambda, weight,
            prior_mean=1.0, prior_strength=200.0,
        )
        effects_train.append(fit_t - 1.0)
        effects_score.append(fit_s - 1.0)
        supports_train.append(support_t)
        supports_score.append(support_s)
    for edge in edges:
        left, right = field_index[edge.left], field_index[edge.right]
        code_train = train_bin[:, left] * 16 + train_bin[:, right]
        code_score = score_bin[:, left] * 16 + score_bin[:, right]
        fit_t, support_t = _cell_posterior(
            code_train, code_train, target_lambda, weight,
            prior_mean=1.0, prior_strength=500.0,
        )
        fit_s, support_s = _cell_posterior(
            code_train, code_score, target_lambda, weight,
            prior_mean=1.0, prior_strength=500.0,
        )
        effects_train.append(0.5 * (fit_t - 1.0))
        effects_score.append(0.5 * (fit_s - 1.0))
        supports_train.append(support_t)
        supports_score.append(support_s)
    effect_train = np.mean(np.vstack(effects_train), axis=0) if effects_train else np.zeros(len(train))
    effect_score = np.mean(np.vstack(effects_score), axis=0) if effects_score else np.zeros(len(score))
    lam_train = np.clip(1.0 + effect_train, 0.0, spec.lambda_max)
    lam_score = np.clip(1.0 + effect_score, 0.0, spec.lambda_max)
    mean_train = parent + lam_train * (expected - parent)
    mean_score = parent_score + lam_score * (expected_score - parent_score)
    residual = realised - mean_train
    global_sigma = float(np.sqrt(np.mean(np.clip(residual, -2_000, 2_000) ** 2)))
    if spec.risk_mode == "unconditional":
        sigma_train = np.full(len(train), global_sigma)
        sigma_score = np.full(len(score), global_sigma)
    else:
        risk_target = np.clip(residual, -2_000, 2_000) ** 2
        risk_effect_train: list[np.ndarray] = []
        risk_effect_score: list[np.ndarray] = []
        risk_fields: Iterable[int] = range(len(fields))
        for idx in risk_fields:
            value_t, _ = _cell_posterior(
                train_bin[:, idx], train_bin[:, idx], risk_target, weight,
                prior_mean=global_sigma**2, prior_strength=300.0,
            )
            value_s, _ = _cell_posterior(
                train_bin[:, idx], score_bin[:, idx], risk_target, weight,
                prior_mean=global_sigma**2, prior_strength=300.0,
            )
            risk_effect_train.append(np.log(np.maximum(value_t, 625.0)))
            risk_effect_score.append(np.log(np.maximum(value_s, 625.0)))
        if spec.risk_mode == "stable_cmi":
            for edge in edges:
                left, right = field_index[edge.left], field_index[edge.right]
                code_train = train_bin[:, left] * 16 + train_bin[:, right]
                code_score = score_bin[:, left] * 16 + score_bin[:, right]
                value_t, _ = _cell_posterior(
                    code_train, code_train, risk_target, weight,
                    prior_mean=global_sigma**2, prior_strength=600.0,
                )
                value_s, _ = _cell_posterior(
                    code_train, code_score, risk_target, weight,
                    prior_mean=global_sigma**2, prior_strength=600.0,
                )
                risk_effect_train.append(np.log(np.maximum(value_t, 625.0)))
                risk_effect_score.append(np.log(np.maximum(value_s, 625.0)))
        sigma_train = np.sqrt(np.exp(np.mean(np.vstack(risk_effect_train), axis=0)))
        sigma_score = np.sqrt(np.exp(np.mean(np.vstack(risk_effect_score), axis=0)))
    support_train = np.median(np.vstack(supports_train), axis=0) if supports_train else np.ones(len(train))
    support_score = np.median(np.vstack(supports_score), axis=0) if supports_score else np.ones(len(score))
    return (
        _predictive_outputs(mean_train, lam_train, sigma_train, support_train),
        _predictive_outputs(mean_score, lam_score, sigma_score, support_score),
        {"edge_count": len(edges), "global_sigma_bps": global_sigma},
    )


def _interaction_matrix(
    x: np.ndarray, fields: Sequence[str], edges: Sequence[CMIEdge],
) -> np.ndarray:
    if not edges:
        return np.empty((len(x), 0), dtype=np.float32)
    index = {field: idx for idx, field in enumerate(fields)}
    return np.column_stack(
        [x[:, index[edge.left]] * x[:, index[edge.right]] for edge in edges]
    ).astype(np.float32)


def _batch_design(
    transform: RobustTransform,
    spline: SplineTransformer,
    frame: pd.DataFrame,
    edges: Sequence[CMIEdge],
    *,
    batch_size: int = 50_000,
) -> Iterable[np.ndarray]:
    for start in range(0, len(frame), batch_size):
        block = frame.iloc[start : start + batch_size]
        raw = transform.transform(block)
        smooth = spline.transform(raw).astype(np.float32)
        interaction = _interaction_matrix(raw, transform.fields, edges)
        yield np.hstack([smooth, interaction]).astype(np.float32, copy=False)


def _fit_gam_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    transform = RobustTransform.fit(train, fields)
    x_raw = transform.transform(train)
    spline = SplineTransformer(
        n_knots=4, degree=2, include_bias=False, extrapolation="linear",
    ).fit(x_raw)
    x_train = np.hstack(
        [spline.transform(x_raw).astype(np.float32), _interaction_matrix(x_raw, fields, edges)]
    )
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent = pd.to_numeric(train["parent_expected_bps"], errors="coerce").to_numpy(float)
    target_lambda, authority = _ideal_lambda(realised, expected, parent, spec.lambda_max)
    weight = cmi_weights(train, spec.cmi_weighting) * authority
    bayesian = spec.model_family in {"bayesian_gam", "bayesian_distributional_gam"}
    mean_model: Any = BayesianRidge(
        alpha_1=1e-4, alpha_2=1e-4, lambda_1=1e-3, lambda_2=1e-3,
        max_iter=300,
    ) if bayesian else Ridge(alpha=20.0)
    mean_model.fit(x_train, target_lambda, sample_weight=weight)
    if bayesian:
        lambda_train_raw, lambda_train_sd = mean_model.predict(x_train, return_std=True)
    else:
        lambda_train_raw = mean_model.predict(x_train)
        lambda_train_sd = np.zeros(len(train), dtype=float)
    lambda_train = np.clip(lambda_train_raw, 0.0, spec.lambda_max)
    mean_train = parent + lambda_train * (expected - parent)
    residual_target = np.log(np.clip(realised - mean_train, -2_000, 2_000) ** 2 + 625.0)
    risk_model: Any | None = None
    if spec.risk_mode != "unconditional" or "distributional" in spec.model_family:
        risk_model = BayesianRidge(max_iter=300) if bayesian else Ridge(alpha=30.0)
        risk_model.fit(x_train, residual_target, sample_weight=weight)
        logvar_train = risk_model.predict(x_train)
        sigma_train = np.sqrt(np.exp(np.clip(logvar_train, np.log(625.0), np.log(4_000_000.0))))
    else:
        global_sigma = float(np.sqrt(np.mean(np.clip(realised - mean_train, -2_000, 2_000) ** 2)))
        sigma_train = np.full(len(train), global_sigma)
    mean_score_parts: list[np.ndarray] = []
    lambda_score_parts: list[np.ndarray] = []
    lambda_sd_parts: list[np.ndarray] = []
    sigma_score_parts: list[np.ndarray] = []
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent_score = pd.to_numeric(score["parent_expected_bps"], errors="coerce").to_numpy(float)
    cursor = 0
    for design in _batch_design(transform, spline, score, edges):
        count = len(design)
        if bayesian:
            raw, sd = mean_model.predict(design, return_std=True)
        else:
            raw, sd = mean_model.predict(design), np.zeros(count, dtype=float)
        lam = np.clip(raw, 0.0, spec.lambda_max)
        local_expected = expected_score[cursor : cursor + count]
        local_parent = parent_score[cursor : cursor + count]
        mean_score_parts.append(local_parent + lam * (local_expected - local_parent))
        lambda_score_parts.append(lam)
        lambda_sd_parts.append(sd)
        if risk_model is not None:
            logvar = risk_model.predict(design)
            sigma_score_parts.append(
                np.sqrt(np.exp(np.clip(logvar, np.log(625.0), np.log(4_000_000.0))))
            )
        else:
            sigma_score_parts.append(np.full(count, sigma_train[0]))
        cursor += count
    mean_score = np.concatenate(mean_score_parts)
    lambda_score = np.concatenate(lambda_score_parts)
    lambda_score_sd = np.concatenate(lambda_sd_parts)
    sigma_score = np.concatenate(sigma_score_parts)
    mean_sd_train = np.abs(expected - parent) * np.asarray(lambda_train_sd)
    mean_sd_score = np.abs(expected_score - parent_score) * lambda_score_sd
    support_train = np.full(len(train), len(train), dtype=float)
    support_score = np.full(len(score), len(train), dtype=float)
    return (
        _predictive_outputs(
            mean_train, lambda_train, sigma_train, support_train, mean_sd=mean_sd_train,
        ),
        _predictive_outputs(
            mean_score, lambda_score, sigma_score, support_score, mean_sd=mean_sd_score,
        ),
        {
            "edge_count": len(edges), "design_columns": int(x_train.shape[1]),
            "bayesian": bayesian,
        },
    )


def fit_ngboost_classifier(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    try:
        from ngboost import NGBClassifier  # type: ignore
        from ngboost.distns import k_categorical  # type: ignore
        from sklearn.tree import DecisionTreeRegressor
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("NGBoost is required for the requested NGBoost arm") from exc
    transform = RobustTransform.fit(train, fields)
    x_train_raw = transform.transform(train)
    x_score_raw = transform.transform(score)
    x_train = np.hstack([x_train_raw, _interaction_matrix(x_train_raw, fields, edges)])
    x_score = np.hstack([x_score_raw, _interaction_matrix(x_score_raw, fields, edges)])
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent = pd.to_numeric(train["parent_expected_bps"], errors="coerce").to_numpy(float)
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent_score = pd.to_numeric(score["parent_expected_bps"], errors="coerce").to_numpy(float)
    residual = realised - expected
    target = np.where(residual < -50.0, 0, np.where(residual > 50.0, 2, 1)).astype(int)
    weight = cmi_weights(train, spec.cmi_weighting)
    model = NGBClassifier(
        Dist=k_categorical(3),
        Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=SEED),
        n_estimators=180, learning_rate=0.035, minibatch_frac=0.70,
        col_sample=0.80, natural_gradient=True, verbose=False, random_state=SEED,
    )
    model.fit(x_train, target, sample_weight=weight)
    prob_train = np.asarray(model.predict_proba(x_train), dtype=float)
    prob_score = np.asarray(model.predict_proba(x_score), dtype=float)
    prior = np.bincount(target, minlength=3).astype(float)
    prior /= prior.sum()
    if spec.probability_mode == "shrunk":
        prob_train = 0.85 * prob_train + 0.15 * prior
        prob_score = 0.85 * prob_score + 0.15 * prior
    elif spec.probability_mode == "calibrated":
        # Train-only conservative temperature calibration.  Temperatures are
        # predeclared; no held outcomes choose them.
        temperature = 1.25
        prob_train = np.exp(np.log(np.clip(prob_train, 1e-8, 1.0)) / temperature)
        prob_score = np.exp(np.log(np.clip(prob_score, 1e-8, 1.0)) / temperature)
        prob_train /= prob_train.sum(axis=1, keepdims=True)
        prob_score /= prob_score.sum(axis=1, keepdims=True)
    downward, upward = (0.80, 0.15)
    lam_train = np.clip(1.0 - downward * prob_train[:, 0] + upward * prob_train[:, 2], 0.0, spec.lambda_max)
    lam_score = np.clip(1.0 - downward * prob_score[:, 0] + upward * prob_score[:, 2], 0.0, spec.lambda_max)
    mean_train = parent + lam_train * (expected - parent)
    mean_score = parent_score + lam_score * (expected_score - parent_score)
    class_second = np.asarray(
        [np.mean(residual[target == cls] ** 2) if np.any(target == cls) else np.mean(residual**2) for cls in range(3)]
    )
    sigma_train = np.sqrt(np.maximum(prob_train @ class_second, 625.0))
    sigma_score = np.sqrt(np.maximum(prob_score @ class_second, 625.0))
    entropy_train = -np.sum(prob_train * np.log(np.clip(prob_train, 1e-9, 1.0)), axis=1)
    entropy_score = -np.sum(prob_score * np.log(np.clip(prob_score, 1e-9, 1.0)), axis=1)
    support_train = len(train) / (1.0 + entropy_train)
    support_score = len(train) / (1.0 + entropy_score)
    return (
        _predictive_outputs(mean_train, lam_train, sigma_train, support_train),
        _predictive_outputs(mean_score, lam_score, sigma_score, support_score),
        {"edge_count": len(edges), "class_prevalence": prior.tolist()},
    )


def fit_local_distribution_forest_proxy(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    transform = RobustTransform.fit(train, fields)
    x_train_raw = transform.transform(train)
    x_score_raw = transform.transform(score)
    x_train = np.hstack([x_train_raw, _interaction_matrix(x_train_raw, fields, edges)])
    x_score = np.hstack([x_score_raw, _interaction_matrix(x_score_raw, fields, edges)])
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent = pd.to_numeric(train["parent_expected_bps"], errors="coerce").to_numpy(float)
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(float)
    parent_score = pd.to_numeric(score["parent_expected_bps"], errors="coerce").to_numpy(float)
    model = RandomForestRegressor(
        n_estimators=64, max_depth=8, min_samples_leaf=120, max_features=0.70,
        bootstrap=True, max_samples=0.75, n_jobs=4, random_state=SEED,
    ).fit(x_train, realised, sample_weight=cmi_weights(train, spec.cmi_weighting))

    def distribution(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree = np.column_stack([estimator.predict(x) for estimator in model.estimators_])
        return tree.mean(axis=1), tree.std(axis=1), tree

    local_train, sd_train, _ = distribution(x_train)
    local_score_parts: list[np.ndarray] = []
    sd_score_parts: list[np.ndarray] = []
    for start in range(0, len(score), 50_000):
        local, sd, _ = distribution(x_score[start : start + 50_000])
        local_score_parts.append(local)
        sd_score_parts.append(sd)
    local_score = np.concatenate(local_score_parts)
    sd_score = np.concatenate(sd_score_parts)
    # Effective local support is approximated from leaf support across trees.
    train_leaf = model.apply(x_train)
    leaf_counts = [
        dict(zip(*np.unique(train_leaf[:, idx], return_counts=True)))
        for idx in range(train_leaf.shape[1])
    ]

    def support(x: np.ndarray) -> np.ndarray:
        leaf = model.apply(x)
        values = np.empty_like(leaf, dtype=float)
        for idx, counts in enumerate(leaf_counts):
            values[:, idx] = [counts.get(value, 0) for value in leaf[:, idx]]
        return np.median(values, axis=1)

    support_train = support(x_train)
    support_score = np.concatenate(
        [support(x_score[start : start + 50_000]) for start in range(0, len(score), 50_000)]
    )
    if spec.probability_mode == "raw":
        mixed_train, mixed_score = local_train, local_score
    else:
        weight_train = support_train / (support_train + 300.0)
        weight_score = support_score / (support_score + 300.0)
        mixed_train = weight_train * local_train + (1.0 - weight_train) * parent
        mixed_score = weight_score * local_score + (1.0 - weight_score) * parent_score
    distance_train = expected - parent
    distance_score = expected_score - parent_score
    lam_train = np.ones(len(train))
    lam_score = np.ones(len(score))
    valid_train = np.abs(distance_train) >= 10.0
    valid_score = np.abs(distance_score) >= 10.0
    lam_train[valid_train] = (mixed_train[valid_train] - parent[valid_train]) / distance_train[valid_train]
    lam_score[valid_score] = (mixed_score[valid_score] - parent_score[valid_score]) / distance_score[valid_score]
    lam_train = np.clip(lam_train, 0.0, spec.lambda_max)
    lam_score = np.clip(lam_score, 0.0, spec.lambda_max)
    mean_train = parent + lam_train * distance_train
    mean_score = parent_score + lam_score * distance_score
    residual_scale = float(np.sqrt(np.mean((realised - local_train) ** 2)))
    sigma_train = np.sqrt(sd_train**2 + residual_scale**2)
    sigma_score = np.sqrt(sd_score**2 + residual_scale**2)
    return (
        _predictive_outputs(mean_train, lam_train, sigma_train, support_train),
        _predictive_outputs(mean_score, lam_score, sigma_score, support_score),
        {"edge_count": len(edges), "trees": len(model.estimators_)},
    )


def fit_cell_day_residual_forest(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    """Fit a distributional trust model directly on Cell-day mapping error.

    The fitted quantity is ``policy_net_bps - raw_expected_bps``.  The local
    residual distribution is estimated from training targets reaching the
    same forest leaves and shrunk toward the global residual distribution by
    effective leaf support.  Held outcomes never participate in these local
    distributions.  This lets downstream code require agreement between a
    negative residual quantile and an overestimation probability before the
    trust model receives demotion authority.
    """

    if spec.target_mode not in {
        "cell_day_residual_clip300", "cell_day_residual_clip500",
    }:
        raise ValueError("Cell-day residual forest requires a residual target mode")
    clip_bps = 300.0 if spec.target_mode.endswith("300") else 500.0
    transform = RobustTransform.fit(train, fields)
    x_train_raw = transform.transform(train)
    x_score_raw = transform.transform(score)
    x_train = np.hstack([x_train_raw, _interaction_matrix(x_train_raw, fields, edges)])
    x_score = np.hstack([x_score_raw, _interaction_matrix(x_score_raw, fields, edges)])
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(float)
    residual_raw = realised - expected
    residual = np.clip(residual_raw, -clip_bps, clip_bps)
    mean_weighting = spec.mean_weighting or spec.cmi_weighting
    sample_weight = cmi_weights(train, mean_weighting)
    model = RandomForestRegressor(
        n_estimators=64, max_depth=8, min_samples_leaf=120, max_features=0.70,
        bootstrap=True, max_samples=0.75, n_jobs=4, random_state=SEED,
    ).fit(x_train, residual, sample_weight=sample_weight)
    train_leaf = model.apply(x_train)

    global_mean = float(np.average(residual, weights=sample_weight))
    global_variance = float(np.average((residual - global_mean) ** 2, weights=sample_weight))
    global_q10, global_q25 = np.quantile(residual, [0.10, 0.25], method="linear")
    global_probabilities = {
        hurdle: float(np.average(residual_raw <= -hurdle, weights=sample_weight))
        for hurdle in (50.0, 100.0, 200.0)
    }
    leaf_statistics: list[dict[int, tuple[int, float, float, float, float, float]]] = []
    for tree_index in range(train_leaf.shape[1]):
        local: dict[int, tuple[int, float, float, float, float, float]] = {}
        leaves = train_leaf[:, tree_index]
        for leaf_id in np.unique(leaves):
            values = residual[leaves == leaf_id]
            raw_values = residual_raw[leaves == leaf_id]
            support = (
                independent_experience_support(train, leaves == leaf_id, sample_weight)
                if spec.support_mode == "independent_experience"
                else float(len(values))
            )
            local[int(leaf_id)] = (
                float(support),
                float(np.mean(values)),
                float(np.quantile(values, 0.10, method="linear")),
                float(np.quantile(values, 0.25, method="linear")),
                float(np.mean(raw_values <= -50.0)),
                float(np.mean(raw_values <= -100.0)),
                float(np.mean(raw_values <= -200.0)),
                float(np.var(values)),
            )
        leaf_statistics.append(local)

    def distribution(x: np.ndarray) -> dict[str, np.ndarray]:
        leaf = model.apply(x)
        rows = len(x)
        support = np.empty((rows, leaf.shape[1]), dtype=np.float32)
        mean = np.empty_like(support)
        q10 = np.empty_like(support)
        q25 = np.empty_like(support)
        p50 = np.empty_like(support)
        p100 = np.empty_like(support)
        p200 = np.empty_like(support)
        variance = np.empty_like(support)
        for tree_index, statistics in enumerate(leaf_statistics):
            values = [statistics[int(value)] for value in leaf[:, tree_index]]
            local = np.asarray(values, dtype=np.float32)
            support[:, tree_index] = local[:, 0]
            mean[:, tree_index] = local[:, 1]
            q10[:, tree_index] = local[:, 2]
            q25[:, tree_index] = local[:, 3]
            p50[:, tree_index] = local[:, 4]
            p100[:, tree_index] = local[:, 5]
            p200[:, tree_index] = local[:, 6]
            variance[:, tree_index] = local[:, 7]
        effective_support = np.median(support, axis=1)
        authority = effective_support / (effective_support + 300.0)
        return {
            "support": effective_support,
            "authority": authority,
            "mean": authority * mean.mean(axis=1) + (1.0 - authority) * global_mean,
            "mean_sd": mean.std(axis=1),
            "q10": authority * q10.mean(axis=1) + (1.0 - authority) * global_q10,
            "q25": authority * q25.mean(axis=1) + (1.0 - authority) * global_q25,
            "p50": authority * p50.mean(axis=1) + (1.0 - authority) * global_probabilities[50.0],
            "p100": authority * p100.mean(axis=1) + (1.0 - authority) * global_probabilities[100.0],
            "p200": authority * p200.mean(axis=1) + (1.0 - authority) * global_probabilities[200.0],
            "variance": authority * variance.mean(axis=1) + (1.0 - authority) * global_variance,
        }

    def batched_distribution(x: np.ndarray) -> dict[str, np.ndarray]:
        parts = [distribution(x[start : start + 25_000]) for start in range(0, len(x), 25_000)]
        return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}

    train_dist = batched_distribution(x_train)
    score_dist = batched_distribution(x_score)
    residual_noise = float(np.sqrt(np.average(
        np.clip(residual_raw - train_dist["mean"], -2_000.0, 2_000.0) ** 2,
        weights=sample_weight,
    )))

    def prediction(expected_value: np.ndarray, local: Mapping[str, np.ndarray]) -> TrustPrediction:
        mean = expected_value + local["mean"]
        aleatoric_variance = (
            local["variance"]
            if spec.uncertainty_mode == "local_leaf"
            else np.full(len(mean), residual_noise**2, dtype=float)
        )
        sigma = np.sqrt(np.maximum(local["mean_sd"] ** 2 + aleatoric_variance, 1.0))
        base = _predictive_outputs(
            mean,
            np.clip(local["authority"], 0.0, spec.lambda_max),
            sigma,
            local["support"],
        )
        base.q10_bps = expected_value + local["q10"]
        base.q50_bps = mean
        base.residual_mean_bps = local["mean"]
        base.residual_q10_bps = local["q10"]
        base.residual_q25_bps = local["q25"]
        base.p_map_overestimate_50bps = local["p50"]
        base.p_map_overestimate_100bps = local["p100"]
        base.p_map_overestimate_200bps = local["p200"]
        base.p_adverse_tail = local["p100"]
        return base

    return (
        prediction(expected, train_dist),
        prediction(expected_score, score_dist),
        {
            "edge_count": len(edges),
            "trees": len(model.estimators_),
            "target_mode": spec.target_mode,
            "residual_clip_bps": clip_bps,
            "global_residual_mean_bps": global_mean,
            "global_residual_q10_bps": float(global_q10),
            "global_residual_q25_bps": float(global_q25),
            "global_overestimate_probabilities": {
                str(int(key)): value for key, value in global_probabilities.items()
            },
            "mean_weighting": mean_weighting,
            "support_mode": spec.support_mode,
            "uncertainty_mode": spec.uncertainty_mode,
        },
    )


# Backward-compatible import for historical artifact readers. New experiment
# contracts and user-facing reports use Local Distribution Forest Proxy (LDF).
fit_distributional_forest = fit_local_distribution_forest_proxy


def fit_distributional_mlp(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the requested MLP arm") from exc
    torch.manual_seed(SEED)
    transform = RobustTransform.fit(train, fields)
    x_train_raw = transform.transform(train)
    x_score_raw = transform.transform(score)
    x_train = np.hstack([x_train_raw, _interaction_matrix(x_train_raw, fields, edges)]).astype(np.float32)
    x_score = np.hstack([x_score_raw, _interaction_matrix(x_score_raw, fields, edges)]).astype(np.float32)
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(np.float32)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(np.float32)
    parent = pd.to_numeric(train["parent_expected_bps"], errors="coerce").to_numpy(np.float32)
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(np.float32)
    parent_score = pd.to_numeric(score["parent_expected_bps"], errors="coerce").to_numpy(np.float32)

    class Net(nn.Module):
        def __init__(self, inputs: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(inputs, 32), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(32, 16), nn.SiLU(),
            )
            self.authority = nn.Linear(16, 1)
            self.scale = nn.Linear(16, 1)

        def forward(self, value: Any) -> tuple[Any, Any]:
            hidden = self.body(value)
            return self.authority(hidden).squeeze(-1), self.scale(hidden).squeeze(-1)

    device = torch.device("cpu")
    model = Net(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    rng = np.random.default_rng(SEED)
    order = np.arange(len(train))
    best, stale = float("inf"), 0
    for _epoch in range(24):
        rng.shuffle(order)
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), 2048):
            idx = order[start : start + 2048]
            xb = torch.from_numpy(x_train[idx]).to(device)
            rb = torch.from_numpy(realised[idx]).to(device)
            eb = torch.from_numpy(expected[idx]).to(device)
            pb = torch.from_numpy(parent[idx]).to(device)
            z, log_scale = model(xb)
            lam = 1.25 * torch.sigmoid(math.log(4.0) + z)
            mu = pb + lam * (eb - pb)
            scale = 25.0 + torch.nn.functional.softplus(log_scale) * 250.0
            dist = torch.distributions.StudentT(df=5.0, loc=mu, scale=scale)
            upward_penalty = 0.05 * torch.relu(lam - 1.0).pow(2).mean()
            loss = -dist.log_prob(rb).mean() + upward_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best - 1e-4:
            best, stale = epoch_loss, 0
        else:
            stale += 1
            if stale >= 4:
                break

    def predict(x: np.ndarray, expected_value: np.ndarray, parent_value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        lambdas: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 50_000):
                z, log_scale = model(torch.from_numpy(x[start : start + 50_000]).to(device))
                lambdas.append((1.25 * torch.sigmoid(math.log(4.0) + z)).cpu().numpy())
                scales.append((25.0 + torch.nn.functional.softplus(log_scale) * 250.0).cpu().numpy())
        lam = np.clip(np.concatenate(lambdas), 0.0, spec.lambda_max)
        sigma = np.concatenate(scales)
        mean = parent_value + lam * (expected_value - parent_value)
        return mean, lam, sigma

    mean_train, lam_train, sigma_train = predict(x_train, expected, parent)
    mean_score, lam_score, sigma_score = predict(x_score, expected_score, parent_score)
    support_train = np.full(len(train), len(train), dtype=float)
    support_score = np.full(len(score), len(train), dtype=float)
    return (
        _predictive_outputs(mean_train, lam_train, sigma_train, support_train),
        _predictive_outputs(mean_score, lam_score, sigma_score, support_score),
        {"edge_count": len(edges), "epochs": _epoch + 1, "best_nll": best},
    )


def fit_trust_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    spec: TrustModelSpec,
) -> tuple[TrustPrediction, TrustPrediction, dict[str, Any]]:
    spec.validate()
    assert_geometry_semantics(train, fields)
    assert_geometry_semantics(score, fields)
    stable = spec.interactions == "stable_cmi"
    if spec.interactions == "none":
        edges: list[CMIEdge] = []
        bins = QuantileBins.fit(train, fields)
    else:
        cmi_source = train.loc[
            pd.to_numeric(train["final_score"], errors="coerce").ge(
                pd.to_numeric(train["final_score"], errors="coerce").quantile(0.80)
            )
        ].copy()
        edges, bins = discover_cmi_edges(
            cmi_source, fields, mode=spec.cmi_weighting, stable=stable,
        )
        # Refit quantile bins on the complete permitted training population.
        bins = QuantileBins.fit(train, fields)
    if spec.model_family == "empirical_bayes":
        train_pred, score_pred, audit = fit_empirical_bayes(
            train, score, fields, edges, bins, spec,
        )
    elif spec.model_family in {
        "gam", "bayesian_gam", "distributional_gam", "bayesian_distributional_gam",
    }:
        train_pred, score_pred, audit = _fit_gam_model(train, score, fields, edges, spec)
    elif spec.model_family == "ngboost":
        train_pred, score_pred, audit = fit_ngboost_classifier(
            train, score, fields, edges, spec,
        )
    elif spec.model_family in {"local_distribution_forest_proxy", "distributional_forest"}:
        train_pred, score_pred, audit = fit_local_distribution_forest_proxy(
            train, score, fields, edges, spec,
        )
    elif spec.model_family == "cell_day_residual_forest":
        train_pred, score_pred, audit = fit_cell_day_residual_forest(
            train, score, fields, edges, spec,
        )
    elif spec.model_family == "distributional_mlp":
        train_pred, score_pred, audit = fit_distributional_mlp(
            train, score, fields, edges, spec,
        )
    else:
        raise ValueError(f"unknown model family {spec.model_family!r}")
    audit.update(
        {
            "spec": spec.name,
            "model_family": spec.model_family,
            "lambda_max": spec.lambda_max,
            "risk_mode": spec.risk_mode,
            "sizing_mode": spec.sizing_mode,
            "cmi_weighting": spec.cmi_weighting,
            "interaction_mode": spec.interactions,
            "target_mode": spec.target_mode,
            "selected_edges": [edge.__dict__ for edge in edges],
            "raw_k9_memberships_used": False,
        }
    )
    return train_pred, score_pred, audit


def sizing_quality(
    prediction: TrustPrediction,
    frame: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    if mode == "equal":
        return np.ones(len(frame), dtype=float)
    raw = pd.to_numeric(frame["raw_expected_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    if mode == "raw_ev":
        return np.maximum(raw, 0.0)
    mean = prediction.expected_bps
    if mode == "mean":
        return np.maximum(mean, 0.0)
    if mode == "mean_risk":
        return np.maximum(mean, 0.0) / np.maximum(
            mean**2 + prediction.predictive_sd_bps**2, 625.0,
        )
    if mode == "predictive":
        return np.maximum(mean, 0.0) * (1.0 - prediction.p_adverse_tail) / np.maximum(
            prediction.predictive_sd_bps**2, 625.0,
        )
    raise ValueError(f"unknown sizing mode {mode!r}")


def causal_size_multiplier(
    train_quality: Sequence[float], score_quality: Sequence[float], *, floor: float = 0.25, cap: float = 1.75,
) -> np.ndarray:
    reference = np.asarray(train_quality, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    value = np.asarray(score_quality, dtype=float)
    if len(reference) < 100:
        return np.ones(len(value), dtype=np.float32)
    rank = np.searchsorted(reference, np.nan_to_num(value, nan=-np.inf), side="right") / len(reference)
    return (floor + (cap - floor) * rank).astype(np.float32)


def catalogue() -> dict[str, list[TrustModelSpec]]:
    result = {
        "bayesian": [
            TrustModelSpec("B0_equal_control", "bayesian", "empirical_bayes", sizing_mode="equal"),
            TrustModelSpec("B1_raw_singleton_l100_mean", "bayesian", "empirical_bayes", lambda_max=1.0, sizing_mode="mean"),
            TrustModelSpec("B2_stable_rank_l100_meanrisk", "bayesian", "empirical_bayes", "stable_cmi", "rank", 1.0, "stable_cmi", "mean_risk"),
            TrustModelSpec("B3_stable_rankloss_l110_meanrisk", "bayesian", "empirical_bayes", "stable_cmi", "rank_loss", 1.10, "stable_cmi", "mean_risk"),
            TrustModelSpec("B4_stable_rankfp_l125_predictive", "bayesian", "empirical_bayes", "stable_cmi", "rank_false_positive", 1.25, "stable_cmi", "predictive"),
            TrustModelSpec("B5_stable_ranklossfp_l125_predictive", "bayesian", "empirical_bayes", "stable_cmi", "rank_loss_false_positive", 1.25, "stable_cmi", "predictive"),
        ],
        "gam": [
            TrustModelSpec("G0_equal_control", "gam", "gam", sizing_mode="equal"),
            TrustModelSpec("G1_gam_singleton_l100_mean", "gam", "gam", lambda_max=1.0, sizing_mode="mean"),
            TrustModelSpec("G2_bayes_gam_singleton_l110_mean", "gam", "bayesian_gam", cmi_weighting="rank", lambda_max=1.10, sizing_mode="mean"),
            TrustModelSpec("G3_bayes_gam_cmi_l110_meanrisk", "gam", "bayesian_gam", "stable_cmi", "rank", 1.10, "singleton", "mean_risk"),
            TrustModelSpec("G4_dist_gam_cmi_l110_meanrisk", "gam", "distributional_gam", "stable_cmi", "rank_loss", 1.10, "stable_cmi", "mean_risk"),
            TrustModelSpec("G5_bayes_dist_gam_cmi_l125_predictive", "gam", "bayesian_distributional_gam", "stable_cmi", "rank_false_positive", 1.25, "stable_cmi", "predictive"),
        ],
        "nonlinear": [
            TrustModelSpec("N0_equal_control", "nonlinear", "ngboost", sizing_mode="equal"),
            TrustModelSpec("N1_ngboost_raw_l100_mean", "nonlinear", "ngboost", "stable_cmi", "rank", 1.0, "singleton", "mean", "raw"),
            TrustModelSpec("N2_ngboost_cal_l110_meanrisk", "nonlinear", "ngboost", "stable_cmi", "rank_loss", 1.10, "stable_cmi", "mean_risk", "calibrated"),
            TrustModelSpec("N3_ngboost_shrunk_l125_predictive", "nonlinear", "ngboost", "stable_cmi", "rank_false_positive", 1.25, "stable_cmi", "predictive", "shrunk"),
            TrustModelSpec("N4_ldf_raw_l125_mean", "nonlinear", "local_distribution_forest_proxy", "stable_cmi", "rank", 1.25, "singleton", "mean", "raw"),
            TrustModelSpec("N5_ldf_support_l110_meanrisk", "nonlinear", "local_distribution_forest_proxy", "stable_cmi", "rank_loss", 1.10, "stable_cmi", "mean_risk", "shrunk"),
            TrustModelSpec("N6_ldf_parent_l125_predictive", "nonlinear", "local_distribution_forest_proxy", "stable_cmi", "rank_false_positive", 1.25, "stable_cmi", "predictive", "shrunk"),
            TrustModelSpec(
                "R5_cell_day_residual_clip300", "nonlinear",
                "cell_day_residual_forest", "stable_cmi", "rank_loss_false_positive",
                1.10, "stable_cmi", "mean_risk", "shrunk",
                "cell_day_residual_clip300",
            ),
            TrustModelSpec(
                "R5_cell_day_residual_clip500", "nonlinear",
                "cell_day_residual_forest", "stable_cmi", "rank_loss_false_positive",
                1.10, "stable_cmi", "mean_risk", "shrunk",
                "cell_day_residual_clip500",
            ),
            TrustModelSpec(
                "R5_cell_day_residual_clip500_neutralmean", "nonlinear",
                "cell_day_residual_forest", "stable_cmi", "rank_loss_false_positive",
                1.10, "stable_cmi", "mean_risk", "shrunk",
                "cell_day_residual_clip500", "uniform", "row_count",
            ),
            TrustModelSpec(
                "R5_cell_day_residual_clip500_neutralmean_independent", "nonlinear",
                "cell_day_residual_forest", "stable_cmi", "rank_loss_false_positive",
                1.10, "stable_cmi", "mean_risk", "shrunk",
                "cell_day_residual_clip500", "uniform", "independent_experience",
                "local_leaf",
            ),
            TrustModelSpec("N7_mlp_l100_meanrisk", "nonlinear", "distributional_mlp", "stable_cmi", "rank", 1.0, "stable_cmi", "mean_risk"),
            TrustModelSpec("N8_mlp_l110_predictive", "nonlinear", "distributional_mlp", "stable_cmi", "rank_loss", 1.10, "stable_cmi", "predictive"),
            TrustModelSpec("N9_mlp_l125_predictive", "nonlinear", "distributional_mlp", "stable_cmi", "rank_false_positive", 1.25, "stable_cmi", "predictive"),
        ],
    }
    for specs in result.values():
        for spec in specs:
            spec.validate()
    return result


__all__ = [
    "CMIEdge", "ParentExpectation", "RAW_K9_PREFIX", "RobustTransform",
    "TrustModelSpec", "TrustPrediction", "assert_geometry_semantics",
    "causal_size_multiplier", "catalogue", "cmi_weights", "discover_cmi_edges",
    "fit_cell_day_residual_forest", "fit_trust_model",
    "independent_experience_support", "residual_classes",
    "sizing_quality", "trust_feature_family",
]
