"""Small, causal Bayesian online change-point primitives for market state.

The implementation uses a Normal-Inverse-Gamma predictive model per feature.
It is deliberately dependency-light and runs on compact hourly market-state
matrices.  Inputs are expected to have been scaled from train history only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np
from numba import njit


@dataclass(frozen=True)
class BOCPDConfig:
    expected_run_hours: int = 48
    max_run_hours: int = 96
    prior_kappa: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    @property
    def hazard(self) -> float:
        return 1.0 / max(float(self.expected_run_hours), 2.0)


def robust_scale_train_oos(
    train: np.ndarray, score: np.ndarray, *, clip: float = 8.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale a univariate series with train-only median/IQR references."""

    train = np.asarray(train, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    finite = train[np.isfinite(train)]
    if finite.size < 16:
        median, scale = 0.0, 1.0
    else:
        median = float(np.median(finite))
        q25, q75 = np.quantile(finite, (0.25, 0.75))
        scale = max(float(q75 - q25), 1e-4)
    train_scaled = np.clip((np.nan_to_num(train, nan=median) - median) / scale, -clip, clip)
    score_scaled = np.clip((np.nan_to_num(score, nan=median) - median) / scale, -clip, clip)
    return (
        train_scaled.astype(np.float32),
        score_scaled.astype(np.float32),
        np.asarray(median, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


@njit(cache=True)
def _logsumexp(values: np.ndarray) -> float:
    maximum = -np.inf
    for value in values:
        if value > maximum:
            maximum = value
    if not np.isfinite(maximum):
        return maximum
    total = 0.0
    for value in values:
        total += np.exp(value - maximum)
    return maximum + log(total)


@njit(cache=True)
def _student_t_logpdf(value: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    degrees = max(2.0 * alpha, 1e-6)
    scale_sq = max(beta * (kappa + 1.0) / (alpha * kappa), 1e-8)
    squared = (value - mu) ** 2
    return (
        lgamma((degrees + 1.0) * 0.5)
        - lgamma(degrees * 0.5)
        - 0.5 * (log(degrees * pi) + log(scale_sq))
        - 0.5 * (degrees + 1.0) * log(1.0 + squared / (degrees * scale_sq))
    )


@njit(cache=True)
def _bocpd_student_t_kernel(
    values: np.ndarray,
    expected_run_hours: int,
    max_run_hours: int,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
) -> np.ndarray:
    """Compiled causal BOCPD kernel; see ``bocpd_student_t`` for contract."""

    maximum = max(max_run_hours, 2)
    size = maximum + 1
    hazard = 1.0 / max(float(expected_run_hours), 2.0)
    log_hazard = log(hazard)
    log_growth = log(1.0 - hazard)
    log_probability = np.full(size, -np.inf, dtype=np.float64)
    log_probability[0] = 0.0
    mu = np.zeros(size, dtype=np.float64)
    kappa = np.full(size, prior_kappa, dtype=np.float64)
    alpha = np.full(size, prior_alpha, dtype=np.float64)
    beta = np.full(size, prior_beta, dtype=np.float64)
    predictive = np.empty(size, dtype=np.float64)
    reset_values = np.empty(size, dtype=np.float64)
    next_log_probability = np.empty(size, dtype=np.float64)
    next_mu = np.empty(size, dtype=np.float64)
    next_kappa = np.empty(size, dtype=np.float64)
    next_alpha = np.empty(size, dtype=np.float64)
    next_beta = np.empty(size, dtype=np.float64)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)

    for index in range(values.shape[0]):
        value = values[index]
        if not np.isfinite(value):
            continue
        for run_length in range(size):
            predictive[run_length] = _student_t_logpdf(
                value, mu[run_length], kappa[run_length], alpha[run_length], beta[run_length]
            )
        prior_predictive = _student_t_logpdf(value, 0.0, prior_kappa, prior_alpha, prior_beta)
        for run_length in range(size):
            next_log_probability[run_length] = -np.inf
            reset_values[run_length] = log_probability[run_length] + log_hazard
        next_log_probability[0] = _logsumexp(reset_values) + prior_predictive
        for run_length in range(1, size):
            next_log_probability[run_length] = (
                log_probability[run_length - 1] + log_growth + predictive[run_length - 1]
            )
        continuation = log_probability[-1] + log_growth + predictive[-1]
        next_log_probability[-1] = np.logaddexp(next_log_probability[-1], continuation)
        normalizer = _logsumexp(next_log_probability)
        next_log_probability -= normalizer

        for run_length in range(size):
            next_mu[run_length] = 0.0
            next_kappa[run_length] = prior_kappa
            next_alpha[run_length] = prior_alpha
            next_beta[run_length] = prior_beta
        for run_length in range(1, size):
            previous = run_length - 1
            updated_kappa = kappa[previous] + 1.0
            next_mu[run_length] = (kappa[previous] * mu[previous] + value) / updated_kappa
            next_kappa[run_length] = updated_kappa
            next_alpha[run_length] = alpha[previous] + 0.5
            next_beta[run_length] = (
                beta[previous]
                + 0.5 * kappa[previous] * (value - mu[previous]) ** 2 / updated_kappa
            )
        # The capped state continues to update rather than losing its mass.
        updated_kappa = kappa[-1] + 1.0
        next_mu[-1] = (kappa[-1] * mu[-1] + value) / updated_kappa
        next_kappa[-1] = updated_kappa
        next_alpha[-1] = alpha[-1] + 0.5
        next_beta[-1] = beta[-1] + 0.5 * kappa[-1] * (value - mu[-1]) ** 2 / updated_kappa
        # Reset state has the prior updated with the current observation.
        reset_kappa = prior_kappa + 1.0
        next_mu[0] = value / reset_kappa
        next_kappa[0] = reset_kappa
        next_alpha[0] = prior_alpha + 0.5
        next_beta[0] = prior_beta + 0.5 * prior_kappa * value**2 / reset_kappa
        # Swap fixed buffers.  Assigning directly would alias the current and
        # next arrays, corrupting the posterior on the following observation.
        previous_log_probability = log_probability
        log_probability = next_log_probability
        next_log_probability = previous_log_probability
        previous_mu = mu
        mu = next_mu
        next_mu = previous_mu
        previous_kappa = kappa
        kappa = next_kappa
        next_kappa = previous_kappa
        previous_alpha = alpha
        alpha = next_alpha
        next_alpha = previous_alpha
        previous_beta = beta
        beta = next_beta
        next_beta = previous_beta
        output[index] = np.float32(np.exp(log_probability[0]))
    return output


def bocpd_student_t(values: np.ndarray, config: BOCPDConfig) -> np.ndarray:
    """Return causal posterior probability of a change at every observation.

    The return at index ``t`` only uses values up to and including ``t``.  A
    constant hazard is paired with prior-predictive likelihood for reset paths;
    this is essential, otherwise the changepoint posterior collapses to the
    fixed hazard and cannot react to a regime break.
    """

    return _bocpd_student_t_kernel(
        np.ascontiguousarray(values, dtype=np.float64),
        int(config.expected_run_hours),
        int(config.max_run_hours),
        float(config.prior_kappa),
        float(config.prior_alpha),
        float(config.prior_beta),
    )


@njit(cache=True)
def _bocpd_student_t_run_summary_kernel(
    values: np.ndarray,
    expected_run_hours: int,
    max_run_hours: int,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
) -> np.ndarray:
    """Causal BOCPD posterior summaries: CP, mean/q05 run length, entropy."""

    maximum = max(max_run_hours, 2)
    size = maximum + 1
    hazard = 1.0 / max(float(expected_run_hours), 2.0)
    log_hazard, log_growth = log(hazard), log(1.0 - hazard)
    log_probability = np.full(size, -np.inf, dtype=np.float64); log_probability[0] = 0.0
    mu = np.zeros(size, dtype=np.float64); kappa = np.full(size, prior_kappa, dtype=np.float64)
    alpha = np.full(size, prior_alpha, dtype=np.float64); beta = np.full(size, prior_beta, dtype=np.float64)
    predictive = np.empty(size, dtype=np.float64); reset_values = np.empty(size, dtype=np.float64)
    next_log_probability = np.empty(size, dtype=np.float64); next_mu = np.empty(size, dtype=np.float64)
    next_kappa = np.empty(size, dtype=np.float64); next_alpha = np.empty(size, dtype=np.float64); next_beta = np.empty(size, dtype=np.float64)
    output = np.full((values.shape[0], 4), np.nan, dtype=np.float32)
    for index in range(values.shape[0]):
        value = values[index]
        if not np.isfinite(value):
            continue
        for run_length in range(size):
            predictive[run_length] = _student_t_logpdf(value, mu[run_length], kappa[run_length], alpha[run_length], beta[run_length])
            next_log_probability[run_length] = -np.inf
            reset_values[run_length] = log_probability[run_length] + log_hazard
        next_log_probability[0] = _logsumexp(reset_values) + _student_t_logpdf(value, 0.0, prior_kappa, prior_alpha, prior_beta)
        for run_length in range(1, size):
            next_log_probability[run_length] = log_probability[run_length - 1] + log_growth + predictive[run_length - 1]
        next_log_probability[-1] = np.logaddexp(next_log_probability[-1], log_probability[-1] + log_growth + predictive[-1])
        next_log_probability -= _logsumexp(next_log_probability)
        for run_length in range(size):
            next_mu[run_length] = 0.0; next_kappa[run_length] = prior_kappa
            next_alpha[run_length] = prior_alpha; next_beta[run_length] = prior_beta
        for run_length in range(1, size):
            previous = run_length - 1; updated_kappa = kappa[previous] + 1.0
            next_mu[run_length] = (kappa[previous] * mu[previous] + value) / updated_kappa
            next_kappa[run_length] = updated_kappa; next_alpha[run_length] = alpha[previous] + 0.5
            next_beta[run_length] = beta[previous] + 0.5 * kappa[previous] * (value - mu[previous]) ** 2 / updated_kappa
        updated_kappa = kappa[-1] + 1.0
        next_mu[-1] = (kappa[-1] * mu[-1] + value) / updated_kappa; next_kappa[-1] = updated_kappa
        next_alpha[-1] = alpha[-1] + 0.5; next_beta[-1] = beta[-1] + 0.5 * kappa[-1] * (value - mu[-1]) ** 2 / updated_kappa
        reset_kappa = prior_kappa + 1.0
        next_mu[0] = value / reset_kappa; next_kappa[0] = reset_kappa; next_alpha[0] = prior_alpha + 0.5
        next_beta[0] = prior_beta + 0.5 * prior_kappa * value ** 2 / reset_kappa
        previous_log_probability = log_probability; log_probability = next_log_probability; next_log_probability = previous_log_probability
        previous_mu = mu; mu = next_mu; next_mu = previous_mu
        previous_kappa = kappa; kappa = next_kappa; next_kappa = previous_kappa
        previous_alpha = alpha; alpha = next_alpha; next_alpha = previous_alpha
        previous_beta = beta; beta = next_beta; next_beta = previous_beta
        mean, entropy, cumulative, q05 = 0.0, 0.0, 0.0, float(maximum)
        for run_length in range(size):
            probability = np.exp(log_probability[run_length])
            mean += run_length * probability
            if probability > 0.0:
                entropy -= probability * log(probability)
            cumulative += probability
            if cumulative >= 0.05 and q05 == float(maximum):
                q05 = float(run_length)
        output[index, 0] = np.float32(np.exp(log_probability[0]))
        output[index, 1] = np.float32(mean)
        output[index, 2] = np.float32(q05)
        output[index, 3] = np.float32(entropy / log(float(size)))
    return output


def bocpd_student_t_run_summary(values: np.ndarray, config: BOCPDConfig) -> np.ndarray:
    """Return causal ``[change_probability, run_mean, run_q05, entropy]``."""

    return _bocpd_student_t_run_summary_kernel(
        np.ascontiguousarray(values, dtype=np.float64), int(config.expected_run_hours),
        int(config.max_run_hours), float(config.prior_kappa), float(config.prior_alpha), float(config.prior_beta),
    )


def synchronized_break_score(
    train_scores: np.ndarray,
    score_scores: np.ndarray,
    *,
    individual_tail: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine independent feature breaks into count and intensity summaries.

    Per-feature cutoffs are fitted on train-only score histories.  The returned
    score is continuous, while ``count`` exposes the simultaneous-break
    mechanism directly for audit.
    """

    train_scores = np.asarray(train_scores, dtype=np.float32)
    score_scores = np.asarray(score_scores, dtype=np.float32)
    if train_scores.ndim != 2 or score_scores.ndim != 2:
        raise ValueError("Expected [time, feature] BOCPD score matrices")
    thresholds = np.nanquantile(train_scores, individual_tail, axis=0).astype(np.float32)
    denom = np.maximum(1.0 - thresholds, 1e-5)
    excess = np.maximum((score_scores - thresholds) / denom, 0.0)
    count = (score_scores > thresholds).sum(axis=1).astype(np.int16)
    intensity = excess.mean(axis=1).astype(np.float32)
    # The count carries the core hypothesis; intensity distinguishes a broad
    # near-threshold shift from a sharp multi-feature transition.
    composite = (count.astype(np.float32) + intensity).astype(np.float32)
    return composite, count, thresholds
