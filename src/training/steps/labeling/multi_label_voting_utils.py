"""Utilities for combining multiple labeling schemes with TPSL variants and voting.

Designed for direct use inside meta_labeling_hpo_sample_weighted.py:
- Generate labels across several TP/SL combinations.
- Produce quantile-based and regime-aware labels from realized returns.
- Combine labels via majority or weighted voting (weights from historical edge/confidence).
- Enhanced multi-triple-barrier with Kalman smoothing and sample weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Kalman filtering imports
# Kalman filtering imports
from numba import njit

from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    ECON_MIN_RETURN_MULTIPLE,
    compute_realized_returns,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
    create_regime_aware_quantile_labels_from_vol_scaled_returns,
)


def _compute_ewma_sigma_of_log_returns(
    close: pd.Series,
    span: int = 64,
    min_periods: int = 20,
) -> pd.Series:
    close = close.astype(float)
    log_ret = np.log(close.replace(0.0, np.nan)).diff()
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    log_ret = log_ret.fillna(0.0)
    try:
        sigma = log_ret.ewm(span=int(span), adjust=False, min_periods=int(min_periods)).std()
    except Exception:
        sigma = log_ret.ewm(span=int(span), adjust=False).std()
    sigma = sigma.replace([np.inf, -np.inf], np.nan)
    sigma = sigma.fillna(method="bfill").fillna(0.0)
    sigma = sigma.abs()
    return sigma

# Triple barrier configuration
@dataclass
class TripleBarrierConfig:
    """Configuration for a single triple-barrier setup."""
    tp_multiplier: float  # Profit take multiplier (in sigma units)
    sl_multiplier: float  # Stop loss multiplier (in sigma units)
    horizon: int         # Time horizon in bars
    name: str = ""       # Optional name for identification

    def id(self) -> str:
        base = self.name or f"tp{self.tp_multiplier:.3f}_sl{self.sl_multiplier:.3f}_h{self.horizon}"
        return base.replace(".", "p")


@dataclass(frozen=True)
class TPSLSpec:
    profit: float
    stop: float
    horizon: int
    name: str = ""

    def id(self) -> str:
        base = self.name or f"tp{self.profit:.4f}_sl{self.stop:.4f}_h{self.horizon}"
        return base.replace(".", "p")


def generate_tpsl_label_sets(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    specs: Iterable[TPSLSpec],
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    min_event_spacing: int = 2,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Run triple-barrier style labeling for multiple TP/SL/horizon combinations.

    Returns a dict keyed by spec.id() with:
        - labels: binary labels aligned to event index (signals where consensus>0)
        - returns: realized returns aligned to event index
        - diagnostics: simple hit-rate/edge stats for weighting
    """
    results: Dict[str, Dict[str, pd.Series]] = {}
    if primary_signals is None or "consensus" not in primary_signals.columns:
        raise ValueError("primary_signals must include a 'consensus' column")

    event_mask = primary_signals["consensus"] > 0

    for spec in specs:
        realized_returns, binary_labels, *_ = compute_realized_returns(
            market_data,
            primary_signals,
            profit_threshold=spec.profit,
            stop_threshold=spec.stop,
            horizon=spec.horizon,
            transaction_cost=transaction_cost,
            min_event_spacing=min_event_spacing,
        )

        labels = binary_labels.loc[event_mask].dropna().astype(int)
        rets = realized_returns.loc[event_mask].reindex(labels.index)

        # Basic diagnostics for weighting
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_ret = rets[pos_mask]
        neg_ret = rets[neg_mask]

        hit_rate = float((pos_ret > 0).mean()) if len(pos_ret) else 0.0
        edge = float(pos_ret.mean() - neg_ret.mean()) if len(pos_ret) and len(neg_ret) else 0.0

        results[spec.id()] = {
            "labels": labels,
            "returns": rets,
            "diagnostics": pd.Series(
                {"hit_rate": hit_rate, "edge": edge, "n_events": len(labels)}, dtype=float
            ),
        }

    return results


def generate_quantile_and_regime_labels(
    realized_returns: pd.Series,
    volatility: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
    q_lower: float = 0.4,
    q_upper: float = 0.6,
) -> Dict[str, pd.Series]:
    """
    Create quantile-based and regime-aware labels from realized returns.
    """
    vol_scaled = compute_vol_scaled_returns_for_events(
        realized_returns, volatility, econ_min_return_multiple=ECON_MIN_RETURN_MULTIPLE
    )

    labels_quantile = create_quantile_labels_from_vol_scaled_returns(
        vol_scaled, low_q=q_lower, high_q=q_upper
    )

    labels_regime = None
    if regimes is not None:
        try:
            labels_regime = create_regime_aware_quantile_labels_from_vol_scaled_returns(
                vol_scaled, regimes, low_q=q_lower, high_q=q_upper
            )
        except Exception:
            labels_regime = None

    out = {"quantile": labels_quantile}
    if labels_regime is not None:
        out["regime_aware"] = labels_regime
    return out


def majority_vote(method_labels: Dict[str, pd.Series], min_votes: Optional[int] = None) -> pd.Series:
    """
    Simple majority vote across label methods (binary 0/1).
    """
    if not method_labels:
        return pd.Series(dtype=float)

    all_index = sorted(set().union(*[lbl.index for lbl in method_labels.values()]))
    aligned_df = pd.DataFrame(
        {name: lbl.reindex(all_index).astype(float) for name, lbl in method_labels.items()},
        index=pd.DatetimeIndex(all_index),
    )

    n_votes = aligned_df.notna().sum(axis=1).astype(int)
    pos_votes = (aligned_df == 1).sum(axis=1).astype(int)
    neg_votes = (aligned_df == 0).sum(axis=1).astype(int)

    threshold = (
        (n_votes // 2 + 1)
        if min_votes is None
        else pd.Series(int(min_votes), index=aligned_df.index)
    )

    result = pd.Series(np.nan, index=aligned_df.index, dtype=float)
    result[(n_votes > 0) & (pos_votes >= threshold)] = 1.0
    result[(n_votes > 0) & (neg_votes >= threshold)] = 0.0
    return result


def _compute_weights_from_history(
    method_labels: Dict[str, pd.Series],
    realized_returns: pd.Series,
    floor_weight: float = 0.1,
    end_time: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if end_time is not None:
        realized_returns = realized_returns.loc[realized_returns.index < end_time]
    for name, lbl in method_labels.items():
        if end_time is not None:
            lbl = lbl.loc[lbl.index < end_time]
        aligned_returns = realized_returns.reindex(lbl.index)
        pos_mask = lbl == 1
        neg_mask = lbl == 0
        pos_ret = aligned_returns[pos_mask].dropna()
        neg_ret = aligned_returns[neg_mask].dropna()

        hit_rate = float((pos_ret > 0).mean()) if len(pos_ret) else 0.0
        edge = float(pos_ret.mean() - neg_ret.mean()) if len(pos_ret) and len(neg_ret) else 0.0

        weight = max(edge, 0.0) + 0.5 * hit_rate
        weights[name] = max(weight, floor_weight)
    return weights


def normalize_method_weights(weights: Dict[str, float], target_sum: float = 1.0) -> Dict[str, float]:
    if not weights:
        return {}

    keys = list(weights.keys())
    vals = np.array([float(weights[k]) for k in keys], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, np.inf)
    total = float(vals.sum())
    if total <= 0.0:
        uniform = float(target_sum) / float(len(keys))
        return {k: uniform for k in keys}
    scale = float(target_sum) / total
    return {k: float(v * scale) for k, v in zip(keys, vals)}


def compute_tpsl_method_weights_from_details(
    tpsl_details: Dict[str, Dict[str, pd.Series]],
    floor_weight: float = 0.1,
    end_time: Optional[pd.Timestamp] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    weights: Dict[str, float] = {}

    for name, details in (tpsl_details or {}).items():
        labels = details.get("labels", pd.Series(dtype=float))
        rets = details.get("returns", pd.Series(dtype=float))

        if end_time is not None:
            labels = labels.loc[labels.index < end_time]
            rets = rets.loc[rets.index < end_time]

        aligned_returns = rets.reindex(labels.index)
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_ret = aligned_returns[pos_mask].dropna()
        neg_ret = aligned_returns[neg_mask].dropna()

        hit_rate = float((pos_ret > 0).mean()) if len(pos_ret) else 0.0
        edge = float(pos_ret.mean() - neg_ret.mean()) if len(pos_ret) and len(neg_ret) else 0.0

        weight = max(edge, 0.0) + 0.5 * hit_rate
        weights[name] = max(weight, floor_weight)

    if normalize and weights:
        weights = normalize_method_weights(weights, target_sum=float(len(weights)))
    return weights


def weighted_vote(
    method_labels: Dict[str, pd.Series],
    realized_returns: Optional[pd.Series] = None,
    weights: Optional[Dict[str, float]] = None,
    tie_breaker: float = 0.0,
) -> pd.Series:
    """
    Weighted voting across label methods. If weights not provided, derive from historical edge.
    """
    if not method_labels:
        return pd.Series(dtype=float)

    all_index = sorted(set().union(*[lbl.index for lbl in method_labels.values()]))
    if weights is None and realized_returns is not None:
        weights = _compute_weights_from_history(method_labels, realized_returns)
    elif weights is None:
        weights = {name: 1.0 for name in method_labels.keys()}

    weighted_scores = np.zeros(len(all_index), dtype=float)
    weight_sums = np.zeros(len(all_index), dtype=float)
    for name, lbl in method_labels.items():
        aligned = lbl.reindex(all_index).astype(float)
        w = float(weights.get(name, 1.0))
        signed = (2.0 * aligned - 1.0).fillna(0.0).to_numpy(dtype=float)
        valid = aligned.notna().to_numpy(dtype=float)
        weighted_scores += w * signed
        weight_sums += w * valid
    normalized_scores = np.full(len(all_index), np.nan, dtype=float)
    nonzero = weight_sums > 0
    normalized_scores[nonzero] = weighted_scores[nonzero] / (weight_sums[nonzero] + 1e-12)

    decisions = np.full(len(all_index), np.nan, dtype=float)
    decisions[nonzero] = (normalized_scores[nonzero] + tie_breaker >= 0).astype(float)
    return pd.Series(decisions, index=pd.DatetimeIndex(all_index))


def compute_binary_consensus_stats(
    method_labels: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    if not method_labels:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    if index is None:
        index = pd.DatetimeIndex(sorted(set().union(*[lbl.index for lbl in method_labels.values()])))
    else:
        index = pd.DatetimeIndex(index)

    weights = weights or {name: 1.0 for name in method_labels.keys()}
    n_methods = float(len(method_labels))

    weighted_scores = np.zeros(len(index), dtype=float)
    weight_sums = np.zeros(len(index), dtype=float)
    n_votes = np.zeros(len(index), dtype=int)
    pos_votes = np.zeros(len(index), dtype=int)
    neg_votes = np.zeros(len(index), dtype=int)

    for name, lbl in method_labels.items():
        aligned = lbl.reindex(index).astype(float)
        valid = aligned.notna().to_numpy(dtype=bool)
        signed = (2.0 * aligned - 1.0).fillna(0.0).to_numpy(dtype=float)
        w = float(weights.get(name, 1.0))

        weighted_scores += w * signed
        weight_sums += w * valid.astype(float)
        n_votes += valid.astype(int)

        vals = aligned.to_numpy(dtype=float)
        pos_votes += (valid & (vals == 1.0)).astype(int)
        neg_votes += (valid & (vals == 0.0)).astype(int)

    score = np.full(len(index), np.nan, dtype=float)
    nonzero = weight_sums > 0
    score[nonzero] = weighted_scores[nonzero] / (weight_sums[nonzero] + 1e-12)

    coverage = np.where(n_methods > 0.0, n_votes.astype(float) / n_methods, 0.0)
    confidence = np.clip(np.abs(score) * coverage, 0.0, 1.0)

    pos_frac = np.full(len(index), np.nan, dtype=float)
    has_votes = n_votes > 0
    pos_frac[has_votes] = pos_votes[has_votes].astype(float) / n_votes[has_votes].astype(float)

    out = pd.DataFrame(index=index)
    out["score"] = score
    out["confidence"] = confidence
    out["n_votes"] = n_votes
    out["pos_frac"] = pos_frac
    out["weight_sum"] = weight_sums
    return out


def compute_base_label_agreement_stats(
    base_labels: pd.Series,
    method_labels: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if base_labels is None or base_labels.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    if not method_labels:
        out = pd.DataFrame(index=pd.DatetimeIndex(base_labels.index))
        out["agree_frac"] = np.nan
        out["support"] = np.nan
        out["support_effective"] = np.nan
        out["confidence"] = np.nan
        out["n_votes"] = 0
        out["coverage"] = 0.0
        out["weight_sum"] = 0.0
        out["weight_agree"] = 0.0
        return out

    index = pd.DatetimeIndex(base_labels.index)
    base = base_labels.reindex(index).astype(float)
    weights = weights or {name: 1.0 for name in method_labels.keys()}
    n_methods = float(len(method_labels))

    n_votes = np.zeros(len(index), dtype=int)
    weight_sums = np.zeros(len(index), dtype=float)
    weight_agree = np.zeros(len(index), dtype=float)

    for name, lbl in method_labels.items():
        aligned = lbl.reindex(index).astype(float)
        valid = aligned.notna() & base.notna()
        same = valid & (aligned == base)
        w = float(weights.get(name, 1.0))

        n_votes += valid.to_numpy(dtype=int)
        weight_sums += w * valid.to_numpy(dtype=float)
        weight_agree += w * same.to_numpy(dtype=float)

    agree_frac = np.full(len(index), np.nan, dtype=float)
    has_votes = weight_sums > 0
    agree_frac[has_votes] = weight_agree[has_votes] / (weight_sums[has_votes] + 1e-12)

    support = np.full(len(index), np.nan, dtype=float)
    support[has_votes] = (agree_frac[has_votes] - 0.5) * 2.0

    coverage = np.where(n_methods > 0.0, n_votes.astype(float) / n_methods, 0.0)
    support_effective = np.where(np.isfinite(support), support * coverage, np.nan)
    confidence = np.where(np.isfinite(support_effective), np.abs(support_effective), np.nan)

    out = pd.DataFrame(index=index)
    out["agree_frac"] = agree_frac
    out["support"] = support
    out["support_effective"] = support_effective
    out["confidence"] = confidence
    out["n_votes"] = n_votes
    out["coverage"] = coverage
    out["weight_sum"] = weight_sums
    out["weight_agree"] = weight_agree
    return out


def compute_weight_multiplier_from_agreement(
    base_labels: pd.Series,
    method_labels: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
    min_mult: float = 1.0,
    max_mult: float = 1.5,
) -> Tuple[pd.Series, pd.DataFrame]:
    stats = compute_base_label_agreement_stats(
        base_labels=base_labels,
        method_labels=method_labels,
        weights=weights,
    )
    support = stats.get("support_effective")
    if support is None:
        mult = pd.Series(1.0, index=pd.DatetimeIndex(base_labels.index), dtype=float)
        return mult, stats

    confidence_pos = support.astype(float).clip(lower=0.0, upper=1.0)
    mult = consensus_confidence_to_weight_multiplier(
        confidence=confidence_pos,
        min_mult=min_mult,
        max_mult=max_mult,
    )
    return mult, stats


def consensus_confidence_to_weight_multiplier(
    confidence: pd.Series,
    min_mult: float = 1.0,
    max_mult: float = 1.5,
) -> pd.Series:
    if confidence is None:
        return pd.Series(dtype=float)

    conf = confidence.astype(float).clip(lower=0.0, upper=1.0)
    mult = float(min_mult) + (float(max_mult) - float(min_mult)) * conf
    mult = mult.where(confidence.notna(), 1.0)
    return mult


def assemble_label_methods_for_voting(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    tpsl_specs: Iterable[TPSLSpec],
    volatility: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, pd.Series]]]:
    """
    Convenience helper to build all label methods (TPSL + quantile + regime-aware).

    Returns:
        method_labels: dict of name -> label series (aligned to event index)
        tpsl_details: full TPSL outputs (labels/returns/diagnostics) for inspection
    """
    tpsl_results = generate_tpsl_label_sets(
        market_data,
        primary_signals,
        specs=tpsl_specs,
        transaction_cost=transaction_cost,
    )

    # Use first TPSL result as base returns for quantile labels
    any_returns = None
    for res in tpsl_results.values():
        any_returns = res["returns"]
        break

    quantile_labels = {}
    if any_returns is not None:
        quantile_labels = generate_quantile_and_regime_labels(
            realized_returns=any_returns,
            volatility=volatility,
            regimes=regimes,
        )

    method_labels: Dict[str, pd.Series] = {k: v["labels"] for k, v in tpsl_results.items()}
    method_labels.update({f"quantile_{k}": v for k, v in quantile_labels.items()})

    return method_labels, tpsl_results


# ---------------------------------------------------------------------------
# Robust Consensus Labeling (3-class with weighted voting)
# ---------------------------------------------------------------------------

def _map_exit_to_trinary(exit_reasons: pd.Series, binary_labels: pd.Series) -> pd.Series:
    """Map compute_realized_returns outputs to {-1, 0, 1}."""
    trinary = pd.Series(index=binary_labels.index, data=0, dtype=int)
    trinary.loc[exit_reasons == "profit"] = 1
    trinary.loc[exit_reasons == "stop"] = -1
    # Timeouts remain 0
    return trinary


def generate_fixed_horizon_trinary_labels(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    profit: float,
    stop: float,
    horizon: int,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    name: str = "",
) -> Tuple[str, pd.Series]:
    """Fixed horizon triple-barrier labels (1 profit, -1 stop, 0 timeout)."""
    _, binary_labels, exit_reasons, *_ = compute_realized_returns(
        market_data,
        primary_signals,
        profit_threshold=profit,
        stop_threshold=stop,
        horizon=horizon,
        transaction_cost=transaction_cost,
    )
    trinary = _map_exit_to_trinary(exit_reasons, binary_labels)
    name = name or f"fixed_tp{profit:.4f}_sl{stop:.4f}_h{horizon}".replace(".", "p")
    return name, trinary


def generate_atr_adjusted_trinary_labels(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    atr_window: int = 14,
    atr_mult: float = 2.0,
    horizon: int = 24,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    name: str = "atr_dynamic",
) -> Tuple[str, pd.Series]:
    """ATR-adjusted triple-barrier labels: profit/stop = atr_mult * ATR / close."""
    if "high" not in market_data.columns or "low" not in market_data.columns:
        raise ValueError("market_data must include high/low for ATR computation")
    tr = market_data[["high", "low", "close"]].copy()
    tr["hl"] = tr["high"] - tr["low"]
    tr["hc"] = (tr["high"] - tr["close"].shift()).abs()
    tr["lc"] = (tr["low"] - tr["close"].shift()).abs()
    true_range = tr[["hl", "hc", "lc"]].max(axis=1)
    atr = true_range.rolling(atr_window).mean()
    atr_frac = atr / (market_data["close"] + 1e-8)

    profit_series = atr_mult * atr_frac
    stop_series = atr_mult * atr_frac

    _, binary_labels, exit_reasons, *_ = compute_realized_returns(
        market_data,
        primary_signals,
        profit_threshold=profit_series,
        stop_threshold=stop_series,
        horizon=horizon,
        transaction_cost=transaction_cost,
    )
    trinary = _map_exit_to_trinary(exit_reasons, binary_labels)
    return name, trinary


def generate_quantile_rank_trinary_labels(
    market_data: pd.DataFrame,
    lookahead: int = 24,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    name: str = "quantile_rank",
) -> Tuple[str, pd.Series]:
    """Quantile/ranking labels from forward returns: top -> 1, bottom -> -1, middle -> 0."""
    fwd_ret = market_data["close"].shift(-lookahead) / market_data["close"] - 1.0
    # Align to current timestamp (no label for last lookahead bars)
    fwd_ret = fwd_ret.dropna()

    q_top = fwd_ret.quantile(1 - top_pct)
    q_bot = fwd_ret.quantile(bottom_pct)

    labels = pd.Series(index=fwd_ret.index, data=0, dtype=int)
    labels[fwd_ret >= q_top] = 1
    labels[fwd_ret <= q_bot] = -1
    return name, labels


@njit
def _numba_kalman_filter(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    observation_matrices: np.ndarray,
    observation_covariances: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled Kalman Filter implementation.
    Standard KF equations for time-varying matrices.
    """
    n_timesteps, n_dim_obs = observations.shape
    n_dim_state = len(initial_state_mean)
    
    # Storage for filtered estimates
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    
    # Current state estimates
    current_state_mean = initial_state_mean.copy()
    current_state_cov = initial_state_covariance.copy()
    
    for t in range(n_timesteps):
        # 1. Predict
        # x_{t|t-1} = F * x_{t-1|t-1}
        # P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
        pred_state_mean = transition_matrix @ current_state_mean
        pred_state_cov = transition_matrix @ current_state_cov @ transition_matrix.T + transition_covariance
        
        # 2. Update
        # y_tilde = z_t - H_t * x_{t|t-1}
        # S_t = H_t * P_{t|t-1} * H_t^T + R_t
        # K_t = P_{t|t-1} * H_t^T * S_t^{-1}
        # x_{t|t} = x_{t|t-1} + K_t * y_tilde
        # P_{t|t} = (I - K_t * H_t) * P_{t|t-1}
        
        H = observation_matrices[t]
        R = observation_covariances[t]
        z = observations[t]
        
        # Skip update if observation is NaN
        if np.isnan(z).any():
            current_state_mean = pred_state_mean
            current_state_cov = pred_state_cov
        else:
            y_tilde = z - H @ pred_state_mean
            S = H @ pred_state_cov @ H.T + R
            # Invert S (2x2 or 1x1 usually)
            # For robustness, we can use pseudo-inverse logic or direct linear solve
            # Since dim is small (1 or 2), linalg.solve is fine
            K = pred_state_cov @ H.T @ np.linalg.inv(S)
            
            current_state_mean = pred_state_mean + K @ y_tilde
            current_state_cov = (np.eye(n_dim_state) - K @ H) @ pred_state_cov
            
        filtered_state_means[t] = current_state_mean
        filtered_state_covariances[t] = current_state_cov
        
    return filtered_state_means, filtered_state_covariances


def compute_volume_weighted_kalman_smoothed_price_and_volatility(
    prices: pd.Series,
    volume: Optional[pd.Series] = None,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3,
    vol_window: int = 20,
    volume_weight: float = 1.0,
    volume_adaptive: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Kalman-smoothed price with volume-based observation weighting.
    
    Args:
        prices: Price series to smooth (Close)
        volume: Volume series for adaptive noise scaling
        process_noise: Kalman process noise parameter (Q)
        measurement_noise: Kalman measurement noise parameter (R_base)
        vol_window: Rolling window for volatility estimation
        volume_weight: Base weight for volume influence (0.0=ignore, 1.0=moderate, 2.0=strong)
        volume_adaptive: If True, adapt weight based on volume anomalies
    
    Returns:
        Tuple of (kalman_smoothed_price, kalman_volatility)
    """
    # Align inputs
    df_in = pd.DataFrame({'close': prices})
    if volume is not None:
        df_in['volume'] = volume
        
    # Clean NaN
    df_clean = df_in.dropna()
    if len(df_clean) < 10:
        return prices, pd.Series(0.0, index=prices.index)
        
    n_timesteps = len(df_clean)
    
    # 1. Calculate Adaptive Measurement Noise (R_t) based on volume
    R_t_values = np.full(n_timesteps, measurement_noise)
    
    if 'volume' in df_clean.columns:
        vol_arr = df_clean['volume'].values
        
        if volume_adaptive:
            # Adaptive volume weighting: high volume = lower noise (more confidence)
            vol_median = np.nanmedian(vol_arr)
            if vol_median > 0:
                # Volume anomaly detection
                vol_zscore = (vol_arr - vol_median) / (np.nanstd(vol_arr) + 1e-12)
                
                # High volume (positive zscore) = lower noise, low volume = higher noise
                volume_scale = 1.0 / (1.0 + np.exp(-vol_zscore))  # Sigmoid scaling
                volume_scale = np.clip(volume_scale, 0.1, 3.0)  # Limit extremes
                
                # Apply volume_weight parameter
                volume_scale = 1.0 + volume_weight * (volume_scale - 1.0)
                R_t_values = measurement_noise / volume_scale
        else:
            # Simple volume scaling
            vol_safe = np.where(vol_arr < 1e-8, 1e-8, vol_arr)
            vol_median = np.nanmedian(vol_safe)
            if vol_median > 0:
                vol_ratio = vol_safe / vol_median
                vol_ratio = np.clip(vol_ratio, 0.1, 10.0)
                volume_scale = 1.0 + volume_weight * np.log1p(vol_ratio - 1.0)
                R_t_values = measurement_noise / volume_scale
            
    # 2. Setup Kalman Matrices
    # State: [price, velocity]
    initial_state_mean = np.array([df_clean['close'].iloc[0], 0.0], dtype=float)
    initial_state_covariance = np.eye(2) * 1e-3
    transition_matrix = np.array([[1.0, 1.0], [0.0, 1.0]])
    transition_covariance = np.eye(2) * process_noise
    
    # 3. Single Measurement: [Close] with volume-adaptive noise
    observation_matrices = np.array([[[1.0, 0.0]] for _ in range(n_timesteps)])
    observations = df_clean['close'].values[:, np.newaxis].astype(float)
    observation_covariances = np.zeros((n_timesteps, 1, 1))
    observation_covariances[:, 0, 0] = R_t_values

    # 4. Run Kalman Filter
    filtered_means, _ = _numba_kalman_filter(
        observations,
        transition_matrix,
        transition_covariance,
        observation_matrices,
        observation_covariances,
        initial_state_mean,
        initial_state_covariance
    )
    
    # 5. Extract smoothed price
    smoothed_price = pd.Series(
        filtered_means[:, 0],
        index=df_clean.index,
        name='kalman_price'
    )
    
    # 6. Compute Kalman volatility
    smoothed_price_full = smoothed_price.reindex(prices.index).ffill()
    
    residuals = np.log(smoothed_price_full.astype(float).abs() + 1e-12).diff()
    kalman_volatility = (
        residuals.rolling(vol_window).std().abs() * smoothed_price_full.astype(float).abs()
    ).bfill()
    kalman_volatility = kalman_volatility.reindex(prices.index).ffill()
    
    return smoothed_price_full, kalman_volatility


def compute_kalman_smoothed_price_and_volatility(
    prices: pd.Series,
    volume: Optional[pd.Series] = None,
    vwap: Optional[pd.Series] = None,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3,
    vol_window: int = 20,
    vwap_weight: float = 1.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Kalman-smoothed price and volatility using Numba optimization.
    
    Args:
        prices: Price series to smooth (Close)
        volume: Volume series for adaptive noise scaling
        vwap: VWAP series for dual measurement
        process_noise: Kalman process noise parameter (Q)
        measurement_noise: Kalman measurement noise parameter (R_base)
        vol_window: Rolling window for volatility estimation
        vwap_weight: Weight for VWAP observation (0.1=mostly ignore, 1.0=equal to close, 2.0=trust VWAP more)
    
    Returns:
        Tuple of (kalman_smoothed_price, kalman_volatility)
    """
    # Align all inputs
    df_in = pd.DataFrame({'close': prices})
    if volume is not None:
        df_in['volume'] = volume
    if vwap is not None:
        df_in['vwap'] = vwap
        
    # Clean NaN
    df_clean = df_in.dropna()
    if len(df_clean) < 10:
         # Fallback if too little data
        return prices, pd.Series(0.0, index=prices.index)
        
    n_timesteps = len(df_clean)
    
    # 1. Calculate Adaptive Measurement Noise (R_t)
    R_t_values = np.full(n_timesteps, measurement_noise)
    if 'volume' in df_clean.columns:
        vol_arr = df_clean['volume'].values
        median_vol = np.nanmedian(vol_arr)
        if median_vol > 0:
            vol_safe = np.where(vol_arr < 1e-8, 1e-8, vol_arr)
            scale_factor = median_vol / vol_safe
            scale_factor = np.clip(scale_factor, 0.1, 10.0)
            R_t_values = measurement_noise * scale_factor
            
    # 2. Setup Matrices (Consistent with Numba function signature)
    # State: [price, velocity]
    initial_state_mean = np.array([df_clean['close'].iloc[0], 0.0], dtype=float)
    initial_state_covariance = np.eye(2) * 1e-3
    transition_matrix = np.array([[1.0, 1.0], [0.0, 1.0]])
    transition_covariance = np.eye(2) * process_noise
    
    # Observation setup
    if 'vwap' in df_clean.columns:
        # Dual Measurement: [Close, VWAP]
        observation_matrices = np.array([[[1.0, 0.0], [1.0, 0.0]] for _ in range(n_timesteps)])
        observations = np.column_stack((df_clean['close'].values, df_clean['vwap'].values)).astype(float)
        
        observation_covariances = np.zeros((n_timesteps, 2, 2))
        observation_covariances[:, 0, 0] = R_t_values  # Close observation noise
        # VWAP observation noise: lower vwap_weight = higher noise = trust VWAP less
        # vwap_weight=1.0 means equal trust, vwap_weight=2.0 means trust VWAP 2x more
        vwap_noise_scale = 1.0 / max(vwap_weight, 0.1)  # Inverse relationship: higher weight = lower noise
        observation_covariances[:, 1, 1] = R_t_values * vwap_noise_scale
    else:
        # Single Measurement: [Close]
        observation_matrices = np.array([[[1.0, 0.0]] for _ in range(n_timesteps)])
        observations = df_clean['close'].values[:, np.newaxis].astype(float)
        
        observation_covariances = np.zeros((n_timesteps, 1, 1))
        # Ensure correct shape assignment
        observation_covariances[:, 0, 0] = R_t_values

    # 3. Compile & Run (First run might be slow due to JIT, subsequent are fast)
    filtered_means, _ = _numba_kalman_filter(
        observations,
        transition_matrix,
        transition_covariance,
        observation_matrices,
        observation_covariances,
        initial_state_mean,
        initial_state_covariance
    )
    
    # Extract smoothed price
    smoothed_price = pd.Series(
        filtered_means[:, 0],
        index=df_clean.index,
        name='kalman_price'
    )
    
    # Compute Kalman volatility
    smoothed_price_full = smoothed_price.reindex(prices.index).ffill()
    
    residuals = np.log(smoothed_price_full.astype(float).abs() + 1e-12).diff()
    kalman_volatility = (
        residuals.rolling(vol_window).std().abs() * smoothed_price_full.astype(float).abs()
    ).bfill()
    kalman_volatility = kalman_volatility.reindex(prices.index).ffill()
    
    return smoothed_price_full, kalman_volatility


def compute_regime_vol_scalar(
    kalman_vol: pd.Series,
    lookback: int = 100,
    low_vol_quantile: float = 0.25,
    high_vol_quantile: float = 0.75,
) -> pd.Series:
    """
    Compute regime-conditional volatility scalar for barrier adjustment.
    
    In low-vol regimes: tighten barriers (scalar < 1.0)
    In high-vol regimes: widen barriers (scalar > 1.0)
    In normal regimes: no change (scalar = 1.0)
    
    Args:
        kalman_vol: Kalman-smoothed volatility series
        lookback: Rolling window for quantile calculation
        low_vol_quantile: Quantile threshold for low-vol regime
        high_vol_quantile: Quantile threshold for high-vol regime
    
    Returns:
        Series of regime scalars
    """
    # Rolling quantiles for regime detection
    vol_low = kalman_vol.rolling(lookback, min_periods=20).quantile(low_vol_quantile)
    vol_high = kalman_vol.rolling(lookback, min_periods=20).quantile(high_vol_quantile)
    
    # Initialize with neutral scalar
    regime_scalar = pd.Series(1.0, index=kalman_vol.index)
    
    # Low-vol regime: tighten barriers (0.7x to 0.9x)
    low_vol_mask = kalman_vol < vol_low
    regime_scalar.loc[low_vol_mask] = 0.8
    
    # High-vol regime: widen barriers (1.1x to 1.3x)
    high_vol_mask = kalman_vol > vol_high
    regime_scalar.loc[high_vol_mask] = 1.2
    
    return regime_scalar


def compute_multi_triple_barrier_outcomes_vectorized(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    configs: List[TripleBarrierConfig],
    kalman_price_col: str = 'kalman_price',
    kalman_vol_col: str = 'kalman_volatility',
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    chunk_size: int = 2000,
    enable_regime_conditional: bool = True,
    direction: Optional[str] = None,
    ewma_span: int = 64,
    ewma_min_periods: int = 20,
    confidence_clip: float = 3.0,
    timeout_confidence: float = 0.25,
) -> List[Dict[str, pd.Series]]:
    """
    Compute triple-barrier outcomes for multiple configurations using vectorized operations.

    This implementation is fully vectorized and processes data in chunks following HPO patterns.

    Args:
        market_data: OHLCV data with Kalman columns
        primary_signals: Primary signals DataFrame with 'consensus' column
        configs: List of triple barrier configurations
        kalman_price_col: Column name for Kalman-smoothed price
        kalman_vol_col: Column name for Kalman volatility
        transaction_cost: Transaction cost as fraction
        chunk_size: Size of chunks to process (following HPO pattern ~2000 bars)
        enable_regime_conditional: If True, adjust barriers based on vol regime

    Returns:
        List of result dictionaries, one per config
    """
    if 'consensus' not in primary_signals.columns:
        raise ValueError("primary_signals must include a 'consensus' column")

    # Get signal events
    d = str(direction or '').lower()
    if d == 'long':
        event_mask = primary_signals['consensus'] > 0
    elif d == 'short':
        event_mask = primary_signals['consensus'] < 0
    else:
        event_mask = primary_signals['consensus'] != 0
    event_indices = event_mask[event_mask].index

    # Prepare Kalman data
    kalman_price = market_data[kalman_price_col]
    kalman_vol = market_data[kalman_vol_col]

    if "close" not in market_data.columns:
        raise ValueError("market_data must include a 'close' column")
    ewma_sigma = _compute_ewma_sigma_of_log_returns(
        close=market_data["close"],
        span=int(ewma_span),
        min_periods=int(ewma_min_periods),
    )
    
    # Compute regime-conditional scalar if enabled
    regime_scalar = None
    if enable_regime_conditional:
        regime_scalar = compute_regime_vol_scalar(kalman_vol)

    results = []

    # Process in chunks following HPO pattern
    n_chunks = max(1, len(market_data) // chunk_size)
    chunks = np.array_split(market_data.index, n_chunks)

    full_index = market_data.index

    for config in configs:
        config_results = []

        for chunk_idx in chunks:
            # Extend the chunk window with enough forward bars so that events
            # starting inside the chunk have access to their full vertical
            # barrier horizon.
            try:
                start_ts = chunk_idx[0]
                end_ts = chunk_idx[-1]
                start_pos = int(full_index.get_indexer([start_ts])[0])
                end_pos = int(full_index.get_indexer([end_ts])[0])
                if start_pos < 0 or end_pos < 0:
                    # Fallback to non-extended chunk if mapping fails.
                    window_index = chunk_idx
                    chunk_index_for_collect = chunk_idx
                else:
                    ext_end_pos = min(len(full_index) - 1, end_pos + int(config.horizon) + 1)
                    window_index = full_index[start_pos:ext_end_pos + 1]
                    chunk_index_for_collect = full_index[start_pos:end_pos + 1]
            except Exception:
                window_index = chunk_idx
                chunk_index_for_collect = chunk_idx

            chunk_data = market_data.loc[window_index]
            chunk_signals = primary_signals.loc[window_index]

            # Compute dynamic barriers for this window.
            # IMPORTANT: compute_realized_returns expects profit_threshold/stop_threshold arrays
            # aligned 1:1 with the df/signals passed in. Since we extend the chunk window
            # forward for horizon lookahead, thresholds must be computed over window_index.
            chunk_kalman_price = kalman_price.loc[window_index]
            chunk_kalman_vol = kalman_vol.loc[window_index]

            # Convert Kalman volatility (in price units) into fractional threshold space.
            # compute_realized_returns expects profit_threshold/stop_threshold as return fractions.
            vol_frac = ewma_sigma.loc[window_index].fillna(0.0)
            
            # Apply regime-conditional scaling if enabled
            if regime_scalar is not None:
                chunk_regime_scalar = regime_scalar.loc[window_index].fillna(1.0)
                vol_frac = vol_frac * chunk_regime_scalar
            
            profit_threshold = (config.tp_multiplier * vol_frac).astype(float)
            stop_threshold = (config.sl_multiplier * vol_frac).astype(float)

            # Staggered Floors Logic
            # Goal: Maintain expert diversity (Scalp vs Trend) even when volatility is near zero.
            # reducing redundancy where all experts align to the exact same econ_min.
            # We scale the floor by the strategy's aggressiveness (multiplier).
            # Baseline: assume econ_min is valid for a multiplier of ~1.0 (1 sigma).
            
            # 1. Calculate base economic floors (absolute minimums)
            base_econ_profit = float(ECON_MIN_RETURN_MULTIPLE) * float(transaction_cost)
            
            # 2. Scale floors by the config's multipliers relative to a baseline of 1.0
            # If TP mult is 2.4, floor should be 2.4x the base floor to preserve ratio.
            # We clip the scalar to be at least 1.0 to never go BELOW the hard economic floor.
            tp_floor_scalar = max(1.0, float(config.tp_multiplier))
            sl_floor_scalar = max(1.0, float(config.sl_multiplier))
            
            rr = float(config.tp_multiplier) / float(max(config.sl_multiplier, 1e-12))
            rr = float(max(rr, 1.0))

            # Ensure profit hits clear transaction costs AFTER netting fees, while preserving
            # expert diversity by scaling floors with the expert's aggressiveness.
            min_profit_floor = float(transaction_cost) * 1.05 * tp_floor_scalar
            min_stop_floor = min_profit_floor / rr

            effective_profit_floor = max(base_econ_profit * tp_floor_scalar, min_profit_floor)
            
            # For Stop Loss, we also want to preserve the Risk/Reward ratio at the floor.
            # But primarily we just want it to scale.
            # Let's use a similar scaling logic for consistency.
            # We derive an "econ_stop_base" from the profit floor and generic R:R or just use the same logic.
            # Simplest: Scale SL floor by SL multiplier.
            # To be safe, let's assume the base sl floor is slightly smaller if we want, 
            # but using the same base_econ_profit as the "unit of volatility" reference is cleaner.
            effective_stop_floor = max(base_econ_profit * sl_floor_scalar, min_stop_floor)

            # 3. Apply Staggered Clipping
            try:
                profit_threshold = profit_threshold.clip(lower=effective_profit_floor)
                stop_threshold = stop_threshold.clip(lower=effective_stop_floor)
            except Exception:
                pass

            # Compute triple barrier outcomes for this chunk
            realized_returns, _binary_labels, exit_reasons, event_durations, mfe_series, mae_series, *_ = compute_realized_returns(
                df=chunk_data,
                signals=chunk_signals,
                profit_threshold=profit_threshold,
                stop_threshold=stop_threshold,
                horizon=config.horizon,
                transaction_cost=transaction_cost,
            )

            # Confidence: margin beyond threshold at touch (0 for timeouts).
            # For profit/trailing: (MFE - PT) / PT
            # For stop: ((-MAE) - SL) / SL
            pt_thr = profit_threshold.reindex(exit_reasons.index).astype(float)
            sl_thr = stop_threshold.reindex(exit_reasons.index).astype(float)
            mfe_v = mfe_series.reindex(exit_reasons.index).astype(float)
            mae_v = mae_series.reindex(exit_reasons.index).astype(float)
            timeout_conf = float(timeout_confidence)
            if (not np.isfinite(timeout_conf)) or timeout_conf < 0.0:
                timeout_conf = 0.0
            conf = pd.Series(timeout_conf, index=exit_reasons.index, dtype=float)
            profit_mask = (exit_reasons == "profit") | (exit_reasons == "trailing")
            stop_mask = exit_reasons == "stop"

            try:
                m = (mfe_v[profit_mask] - pt_thr[profit_mask]) / (pt_thr[profit_mask] + 1e-12)
                m = np.clip(m.to_numpy(dtype=float), 0.0, float(confidence_clip))
                conf.loc[profit_mask] = 1.0 + m
            except Exception:
                pass
            try:
                m = ((-mae_v[stop_mask]) - sl_thr[stop_mask]) / (sl_thr[stop_mask] + 1e-12)
                m = np.clip(m.to_numpy(dtype=float), 0.0, float(confidence_clip))
                conf.loc[stop_mask] = 1.0 + m
            except Exception:
                pass

            # Convert to trinary labels {-1, 0, 1} using strict first-barrier-hit semantics.
            # - profit/trailing => +1
            # - stop => -1
            # - timeout/other => 0
            trinary_labels = pd.Series(0, index=exit_reasons.index, dtype=int)
            profit_mask = (exit_reasons == "profit") | (exit_reasons == "trailing")
            stop_mask = exit_reasons == "stop"
            trinary_labels.loc[profit_mask] = 1
            trinary_labels.loc[stop_mask] = -1

            config_results.append({
                'labels': trinary_labels.reindex(chunk_index_for_collect),
                'returns': realized_returns.reindex(chunk_index_for_collect),
                'exit_reasons': exit_reasons.reindex(chunk_index_for_collect),
                'durations': event_durations.reindex(chunk_index_for_collect),
                'confidence': conf.reindex(chunk_index_for_collect),
            })

        # Combine chunks for this configuration
        combined_labels = pd.concat([r['labels'] for r in config_results])
        combined_returns = pd.concat([r['returns'] for r in config_results])
        combined_exit_reasons = pd.concat([r['exit_reasons'] for r in config_results])
        combined_durations = pd.concat([r['durations'] for r in config_results])
        combined_confidence = pd.concat([r['confidence'] for r in config_results])

        # Align to event index
        aligned_labels = combined_labels.loc[event_mask].dropna().astype(int)
        aligned_returns = combined_returns.loc[event_mask].reindex(aligned_labels.index)
        aligned_exit_reasons = combined_exit_reasons.loc[event_mask].reindex(aligned_labels.index)
        aligned_durations = combined_durations.loc[event_mask].reindex(aligned_labels.index)
        aligned_confidence = combined_confidence.loc[event_mask].reindex(aligned_labels.index)

        # Compute absolute returns for weighting
        abs_returns = aligned_returns.abs()

        result = {
            'config': config,
            'labels': aligned_labels,
            'returns': aligned_returns,
            'abs_returns': abs_returns,
            'profit_threshold': profit_threshold,  # From last chunk (for reference)
            'stop_threshold': stop_threshold,  # From last chunk (for reference)
            'exit_reasons': aligned_exit_reasons,
            'event_durations': aligned_durations,
            'confidence': aligned_confidence,
        }

        results.append(result)

    return results


# Maintain backward compatibility
def compute_multi_triple_barrier_outcomes(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    configs: List[TripleBarrierConfig],
    kalman_price_col: str = 'kalman_price',
    kalman_vol_col: str = 'kalman_volatility',
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    direction: Optional[str] = None,
) -> List[Dict[str, pd.Series]]:
    """
    Legacy wrapper for backward compatibility.
    Use compute_multi_triple_barrier_outcomes_vectorized for better performance.
    """
    return compute_multi_triple_barrier_outcomes_vectorized(
        market_data=market_data,
        primary_signals=primary_signals,
        configs=configs,
        kalman_price_col=kalman_price_col,
        kalman_vol_col=kalman_vol_col,
        transaction_cost=transaction_cost,
        chunk_size=2000,
        direction=direction,
    )


def compute_kalman_multi_triple_barrier_sample_weights(
    tb_results: List[Dict[str, pd.Series]],
    kalman_volatility: pd.Series,
    economic_floor_multiplier: float = 0.25,
    normalize_weights: bool = True
) -> pd.Series:
    """
    Compute sample weights by averaging absolute returns across configurations.

    Args:
        tb_results: List of triple barrier result dictionaries
        kalman_volatility: Kalman volatility series for economic floor
        economic_floor_multiplier: Multiplier for economic floor (relative to mean volatility)
        normalize_weights: Whether to normalize weights to mean=1

    Returns:
        Series of sample weights indexed by event timestamps
    """
    if not tb_results:
        raise ValueError("No triple barrier results provided")

    # Extract absolute returns for each configuration
    abs_returns_list = []
    common_index = None

    for result in tb_results:
        abs_returns = result['abs_returns']
        conf = result.get("confidence")
        if isinstance(conf, pd.Series):
            conf_aligned = conf.reindex(abs_returns.index).astype(float)
            conf_aligned = conf_aligned.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            conf_aligned = conf_aligned.clip(lower=0.0)
            abs_returns = abs_returns.astype(float) * conf_aligned
        abs_returns_list.append(abs_returns)

        if common_index is None:
            common_index = abs_returns.index
        else:
            common_index = common_index.intersection(abs_returns.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common indices across configurations")

    # Align all absolute returns to common index
    aligned_abs_returns = []
    for abs_returns in abs_returns_list:
        aligned = abs_returns.reindex(common_index).fillna(0)
        aligned_abs_returns.append(aligned.values)

    # Average across configurations
    tb_returns_matrix = np.array(aligned_abs_returns)
    weights = np.mean(tb_returns_matrix, axis=0)

    # Apply economic floor to avoid zero-weighting small moves
    mean_volatility = kalman_volatility.loc[common_index].mean()
    economic_floor = economic_floor_multiplier * mean_volatility
    weights = np.maximum(weights, economic_floor)

    # Optional normalization
    if normalize_weights:
        weights = weights / np.mean(weights)

    return pd.Series(weights, index=common_index, name='sample_weights')


def kalman_multi_triple_barrier_labels(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    tp_multipliers: List[float] = [1.2, 1.8, 2.4],
    sl_multipliers: List[float] = [0.6, 0.9, 1.2],
    horizons: List[int] = [4, 8, 12],
    kalman_process_noise: float = 1e-5,
    kalman_measurement_noise: float = 1e-3,
    vol_window: int = 20,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    economic_floor_multiplier: float = 0.25,
    consensus_threshold: float = 0.6,
    return_detailed_results: bool = False,
    use_confidence_weighted_voting: bool = True,
    timeout_downweight_factor: float = 0.25,
) -> Union[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, Dict]]:
    """
    Generate labels using Kalman-smoothed multi-triple-barrier approach.

    Args:
        market_data: OHLCV DataFrame with 'close' column
        primary_signals: DataFrame with 'consensus' column for signal events
        tp_multipliers: List of profit-take multipliers (in sigma units)
        sl_multipliers: List of stop-loss multipliers (in sigma units)
        horizons: List of time horizons in bars
        kalman_process_noise: Kalman filter process noise
        kalman_measurement_noise: Kalman filter measurement noise
        vol_window: Rolling window for volatility estimation
        transaction_cost: Transaction cost as fraction
        economic_floor_multiplier: Multiplier for economic floor in sample weights
        consensus_threshold: Threshold for consensus labeling
        return_detailed_results: Whether to return detailed results

    Returns:
        Tuple of (consensus_labels, sample_weights) or
        (consensus_labels, sample_weights, detailed_results) if return_detailed_results=True
    """
    # Step 1: Compute Kalman-smoothed price and volatility
    if 'close' not in market_data.columns:
        raise ValueError("market_data must contain 'close' column")

    kalman_price, kalman_volatility = compute_kalman_smoothed_price_and_volatility(
        prices=market_data['close'],
        volume=market_data.get('volume', None),
        vwap=market_data.get('vwap', None),
        process_noise=kalman_process_noise,
        measurement_noise=kalman_measurement_noise,
        vol_window=vol_window
    )

    # Add to market_data for triple barrier computation
    market_data = market_data.copy()
    market_data['kalman_price'] = kalman_price
    market_data['kalman_volatility'] = kalman_volatility

    # Step 2: Define multiple triple-barrier configurations
    configs = []
    if len(tp_multipliers) == len(sl_multipliers):
        for tp_mult, sl_mult in zip(tp_multipliers, sl_multipliers):
            for horizon in horizons:
                config = TripleBarrierConfig(
                    tp_multiplier=tp_mult,
                    sl_multiplier=sl_mult,
                    horizon=horizon,
                )
                configs.append(config)
    else:
        for tp_mult in tp_multipliers:
            for sl_mult in sl_multipliers:
                for horizon in horizons:
                    config = TripleBarrierConfig(
                        tp_multiplier=tp_mult,
                        sl_multiplier=sl_mult,
                        horizon=horizon,
                    )
                    configs.append(config)

    # Bayesian optimization for hyperparameter tuning
    optimizer = BayesianTPEOptimizer(
        config=OptimizationConfig(
            n_trials=int(config.get("layer0_n_trials", config.get("stage0_n_trials", 50))),
            execution_mode=exec_mode,
            direction="minimize",  # Minimize loss, not maximize
            seed=int(config.get("random_state", 42)),
            coarse_grid_points=grid_points,
            fine_grid_points=grid_points,
        )
    )

    # Expanded search space with volume-weighted Kalman filter
    search_space = {
        "kalman_Q": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},      # Expanded range
        "kalman_R": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},      # Expanded range
        "volume_weight": {"type": "float", "low": 0.0, "high": 3.0, "log": False}, # Volume-based weighting
        "volume_adaptive": {"type": "categorical", "choices": [True, False]},       # Adaptive vs simple
    }

    # Step 3: Compute triple-barrier outcomes for each configuration
    tb_results = compute_multi_triple_barrier_outcomes(
        market_data=market_data,
        primary_signals=primary_signals,
        configs=configs,
        transaction_cost=transaction_cost
    )

    # Step 4: Compute sample weights by averaging absolute returns
    sample_weights = compute_kalman_multi_triple_barrier_sample_weights(
        tb_results=tb_results,
        kalman_volatility=kalman_volatility,
        economic_floor_multiplier=economic_floor_multiplier
    )

    # Step 5: Generate consensus labels via weighted voting
    # Align all labels to common index
    all_labels = []
    label_names = []

    for result in tb_results:
        config = result['config']
        labels = result['labels'].reindex(sample_weights.index).fillna(0).astype(int)
        all_labels.append(labels)
        label_names.append(config.id())

    # Create label matrix
    label_matrix = pd.DataFrame(
        {name: labels for name, labels in zip(label_names, all_labels)},
        index=sample_weights.index
    )

    if use_confidence_weighted_voting:
        conf_mat = []
        for result in tb_results:
            conf = result.get("confidence")
            if isinstance(conf, pd.Series):
                conf = conf.reindex(sample_weights.index).astype(float)
            else:
                conf = pd.Series(1.0, index=sample_weights.index, dtype=float)
            conf = conf.replace([np.inf, -np.inf], np.nan)
            conf = conf.fillna(1.0)
            conf = conf.clip(lower=0.0)
            conf_mat.append(conf)

        conf_matrix = pd.DataFrame(
            {name: c for name, c in zip(label_names, conf_mat)},
            index=sample_weights.index,
        )

        vote_mask = (label_matrix.values != 0).astype(float)
        conf_vals = conf_matrix.values.astype(float)
        denom = (vote_mask * conf_vals).sum(axis=1) + 1e-8
        numer = (label_matrix.values.astype(float) * conf_vals).sum(axis=1)
        scores = numer / denom
    else:
        weights = np.ones(len(configs)) / len(configs)
        scores = (label_matrix.values * weights).sum(axis=1) / (weights.sum() + 1e-8)

    # Generate consensus labels
    consensus_labels = pd.Series(0, index=sample_weights.index, dtype=int)
    consensus_labels[scores > consensus_threshold] = 1
    consensus_labels[scores < -consensus_threshold] = -1

    # Step 6: Downweight timeouts (events where no barrier was hit across the ensemble)
    try:
        timeout_factor = float(timeout_downweight_factor)
    except Exception:
        timeout_factor = 0.25
    if (not np.isfinite(timeout_factor)) or timeout_factor < 0.0:
        timeout_factor = 0.0
    if timeout_factor < 1.0:
        try:
            n_fired = (label_matrix.values != 0).sum(axis=1)
            all_timeout = n_fired <= 0
            sample_weights = sample_weights.copy()
            sample_weights.loc[all_timeout] = sample_weights.loc[all_timeout] * timeout_factor
        except Exception:
            pass

    if return_detailed_results:
        detailed_results = {
            'kalman_price': kalman_price,
            'kalman_volatility': kalman_volatility,
            'configs': configs,
            'tb_results': tb_results,
            'label_matrix': label_matrix,
            'raw_scores': pd.Series(scores, index=sample_weights.index)
        }
        return consensus_labels, sample_weights, detailed_results

    return consensus_labels, sample_weights


def robust_consensus_labels(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    scalp_config: Tuple[float, float, int] = (0.005, 0.005, 12),
    swing_config: Tuple[float, float, int] = (0.015, 0.0075, 32),
    atr_mult: float = 2.0,
    atr_window: int = 14,
    quantile_lookahead: int = 24,
    quantile_pct: float = 0.2,
    threshold: float = 0.6,
    dynamic_threshold: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the consensus labeling matrix and final label via weighted voting.

    Returns:
        label_matrix: DataFrame with individual method labels {-1,0,1}
        consensus: Series with final {-1,0,1} labels
    """
    # Phase 1: generate labels
    labels: Dict[str, pd.Series] = {}

    name, scalp_lbl = generate_fixed_horizon_trinary_labels(
        market_data, primary_signals, *scalp_config, name="scalp"
    )
    labels[name] = scalp_lbl

    name, swing_lbl = generate_fixed_horizon_trinary_labels(
        market_data, primary_signals, *swing_config, name="swing"
    )
    labels[name] = swing_lbl

    name, atr_lbl = generate_atr_adjusted_trinary_labels(
        market_data,
        primary_signals,
        atr_window=atr_window,
        atr_mult=atr_mult,
        horizon=24,
        name="volatility",
    )
    labels[name] = atr_lbl

    name, quant_lbl = generate_quantile_rank_trinary_labels(
        market_data,
        lookahead=quantile_lookahead,
        top_pct=quantile_pct,
        bottom_pct=quantile_pct,
        name="quantile",
    )
    labels[name] = quant_lbl

    # Phase 2: weighted consensus
    weights = weights or {
        "volatility": 1.5,
        "swing": 1.2,
        "quantile": 1.0,
        "scalp": 0.8,
    }

    all_index = sorted(set().union(*[s.index for s in labels.values()]))
    label_matrix = pd.DataFrame(index=pd.DatetimeIndex(all_index))
    for k, v in labels.items():
        label_matrix[k] = v.reindex(all_index).fillna(0).astype(int)

    w_vec = np.array([weights.get(col, 1.0) for col in label_matrix.columns], dtype=float)
    scores = (label_matrix.values * w_vec).sum(axis=1) / (w_vec.sum() + 1e-8)

    # Optional dynamic threshold relaxation if coverage is too low and volatility vote dominates
    adj_threshold = threshold
    if dynamic_threshold:
        high_vol_weight = weights.get("volatility", 0) > 1.4
        signal_rate = float((np.abs(scores) > threshold).mean()) if len(scores) else 0.0
        if high_vol_weight and signal_rate < 0.2:
            adj_threshold = max(0.55, threshold * 0.92)

    consensus = pd.Series(index=label_matrix.index, data=0, dtype=int)
    consensus[scores > adj_threshold] = 1
    consensus[scores < -adj_threshold] = -1

    return label_matrix, consensus


def compute_class_weights_for_consensus(consensus: pd.Series, neutral_weight: float = 1.0) -> pd.Series:
    """
    Simple class weights for {-1,0,1} consensus labels to handle imbalance.
    """
    counts = consensus.value_counts(dropna=True)
    weights = pd.Series(index=[-1, 0, 1], data=neutral_weight, dtype=float)
    for cls in [-1, 1]:
        n_cls = counts.get(cls, 0)
        n_total = counts.sum()
        if n_cls > 0:
            weights[cls] = max(neutral_weight, n_total / (2 * n_cls))
    return weights


def purge_tail_for_lookahead(index: pd.DatetimeIndex, lookahead: int) -> pd.DatetimeIndex:
    """
    Drop the last `lookahead` rows to prevent label lookahead leakage in training/validation splits.

    Use this to pre-trim data or to apply an embargo between train/val folds.
    """
    if lookahead <= 0:
        return index
    keep = len(index) - lookahead
    if keep <= 0:
        return index[0:0]
    return index[:keep]


def consensus_embargo_bars(quantile_lookahead: int, *horizons: int) -> int:
    """
    Compute a safe embargo length (in bars) given lookahead-based labels and triple-barrier horizons.
    """
    horizons = list(horizons) if horizons else [0]
    return int(max([quantile_lookahead, *horizons]))


# Example usage and testing functions
def example_kalman_multi_triple_barrier_usage():
    """
    Example demonstrating how to use the Kalman multi-triple-barrier labeling system.

    This function shows the complete pipeline:
    1. Load/generate sample market data
    2. Generate primary signals (simulated)
    3. Apply Kalman multi-triple-barrier labeling
    4. Use results for classification and sample weighting
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Create sample market data (15-min bars)
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')

    # Generate synthetic price data with trend and noise
    trend = np.linspace(100, 110, n_bars)
    noise = np.random.normal(0, 0.5, n_bars)
    price_changes = np.random.normal(0, 0.02, n_bars)
    close_prices = 100 + np.cumsum(price_changes) + trend + noise

    # Create OHLCV data
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.normal(0, 0.1, n_bars),
        'high': close_prices + np.abs(np.random.normal(0, 0.2, n_bars)),
        'low': close_prices - np.abs(np.random.normal(0, 0.2, n_bars)),
        'close': close_prices,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }).set_index('timestamp')

    # Generate synthetic primary signals
    # Simple momentum-based signals
    momentum = market_data['close'].diff(10)
    signals = pd.DataFrame(index=market_data.index)
    signals['consensus'] = 0
    signals.loc[momentum > momentum.quantile(0.8), 'consensus'] = 1   # Long signals
    signals.loc[momentum < momentum.quantile(0.2), 'consensus'] = -1  # Short signals

    try:
        # Apply Kalman multi-triple-barrier labeling
        consensus_labels, sample_weights, detailed_results = kalman_multi_triple_barrier_labels(
            market_data=market_data,
            primary_signals=signals,
            tp_multipliers=[1.0, 2.0],      # 1σ and 2σ profit targets
            sl_multipliers=[1.0, 1.5],      # 1σ and 1.5σ stop losses
            horizons=[4, 8],                # 1h and 2h horizons (15min bars)
            return_detailed_results=True
        )

        print("Kalman Multi-Triple-Barrier Labeling Results:")
        print(f"Total samples: {len(consensus_labels)}")
        print(f"Positive labels: {(consensus_labels == 1).sum()}")
        print(f"Negative labels: {(consensus_labels == -1).sum()}")
        print(f"Neutral labels: {(consensus_labels == 0).sum()}")
        print(".3f")
        print(f"Kalman volatility mean: {detailed_results['kalman_volatility'].mean():.6f}")
        print(f"Number of configurations tested: {len(detailed_results['configs'])}")

        # Show sample results
        results_df = pd.DataFrame({
            'consensus_label': consensus_labels,
            'sample_weight': sample_weights,
            'kalman_price': detailed_results['kalman_price'].loc[sample_weights.index],
            'kalman_volatility': detailed_results['kalman_volatility'].loc[sample_weights.index]
        })

        print("\nSample Results (first 10):")
        print(results_df.head(10))

        return consensus_labels, sample_weights, detailed_results

    except Exception as e:
        print(f"Error in Kalman multi-triple-barrier labeling: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Non-Leaky Committee Voting for Layer 2 (Option C)
# ---------------------------------------------------------------------------
# Uses the existing 6 TPSL experts for label diversity, but learns voting
# weights only from training data (no lookahead). The voted labels become
# the target `y` for a standard model-based Layer 2.
# ---------------------------------------------------------------------------


def compute_committee_voted_labels_for_cv(
    label_matrix: np.ndarray,
    returns_matrix: np.ndarray,
    event_index: pd.DatetimeIndex,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    floor_weight: float = 0.1,
    alpha_hit_rate: float = 0.5,
    use_return_weighted_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute committee-voted labels for a single CV fold WITHOUT lookahead.

    The voting weights are learned from TRAINING events only, then applied
    to produce voted labels for TEST events. This is non-leaky because:
    - weights come only from train_idx events
    - test_idx labels are computed using those train-derived weights

    Args:
        label_matrix: (n_events, n_experts) array of trinary labels {-1, 0, 1}
        returns_matrix: (n_events, n_experts) array of realized returns
        event_index: DatetimeIndex aligned to label_matrix rows
        train_idx: integer indices into label_matrix for training events
        test_idx: integer indices into label_matrix for test events
        floor_weight: minimum weight per expert
        alpha_hit_rate: weight = max(edge, 0) + alpha_hit_rate * hit_rate
        use_return_weighted_labels: if True, use actual returns to determine labels
            (positive return = label 1, negative = label 0). If False, use barrier
            hits (profit barrier = 1, stop barrier = 0). Default True for better
            alignment between labels and actual profitability.

    Returns:
        voted_labels_train: (len(train_idx),) binary labels for train events
        voted_labels_test: (len(test_idx),) binary labels for test events
        weights: dict of expert_idx -> weight (for diagnostics)
    """
    n_experts = label_matrix.shape[1]

    # 1. Compute weights from TRAINING data only
    weights = {}
    for k in range(n_experts):
        lbl_train = label_matrix[train_idx, k]
        ret_train = returns_matrix[train_idx, k]

        # Only consider events where expert fired (label != 0)
        fired = lbl_train != 0
        if not np.any(fired):
            weights[k] = float(floor_weight)
            continue

        lbl_fired = lbl_train[fired]
        ret_fired = ret_train[fired]

        # Binary success: label == 1 (profit hit)
        pos_mask = lbl_fired == 1
        neg_mask = lbl_fired == -1

        pos_ret = ret_fired[pos_mask]
        neg_ret = ret_fired[neg_mask]

        hit_rate = float(np.mean(pos_ret > 0)) if pos_ret.size > 0 else 0.0
        mean_pos = float(np.mean(pos_ret)) if pos_ret.size > 0 else 0.0
        mean_neg = float(np.mean(neg_ret)) if neg_ret.size > 0 else 0.0
        edge = mean_pos - mean_neg

        w = max(edge, 0.0) + float(alpha_hit_rate) * hit_rate
        weights[k] = max(w, float(floor_weight))

    # Normalize weights
    w_arr = np.array([weights[k] for k in range(n_experts)], dtype=float)
    w_sum = float(np.sum(w_arr))
    if w_sum > 0:
        w_arr = w_arr / w_sum
    else:
        w_arr = np.ones(n_experts, dtype=float) / float(n_experts)

    # 2. Apply weights to compute voted labels with RETURN-WEIGHTED smoothing
    def _vote(indices: np.ndarray, use_return_weighting: bool = True) -> np.ndarray:
        lbl_sub = label_matrix[indices, :]  # (n, n_experts)
        ret_sub = returns_matrix[indices, :]  # (n, n_experts)
        
        # fired mask: only count experts that fired
        fired = lbl_sub != 0
        
        if use_return_weighting:
            # RETURN-WEIGHTED VOTING: Use actual returns to weight votes
            # This ensures labels reflect actual profitability, not just barrier hits
            # 
            # For each event, compute weighted average of returns across experts,
            # then convert to soft label based on return sign and magnitude.
            
            # Mask invalid returns
            ret_valid = np.where(fired & np.isfinite(ret_sub), ret_sub, 0.0)
            
            # Weighted average return across experts
            weighted_ret = np.sum(ret_valid * w_arr.reshape(1, -1), axis=1)
            weight_denom = np.sum(fired.astype(float) * w_arr.reshape(1, -1), axis=1) + 1e-12
            avg_ret = weighted_ret / weight_denom
            
            # Convert to soft label using sigmoid transform
            # This maps: return=0 -> label=0.5, positive return -> label>0.5, negative -> label<0.5
            # Scale factor controls sharpness (higher = sharper transition)
            scale = 100.0  # 1% return maps to ~0.73 label, -1% to ~0.27
            z = np.clip(avg_ret * scale, -10.0, 10.0)
            soft_label = 1.0 / (1.0 + np.exp(-z))
            
            # Binary label: soft_label > 0.5 => 1, else 0
            # (equivalent to avg_ret > 0)
            voted = (soft_label > 0.5).astype(float)
        else:
            # Original barrier-hit voting (fallback)
            weighted_sum = np.sum(lbl_sub.astype(float) * w_arr.reshape(1, -1), axis=1)
            weight_denom = np.sum(fired.astype(float) * w_arr.reshape(1, -1), axis=1) + 1e-12
            score = weighted_sum / weight_denom
            voted = (score > 0).astype(float)
        
        # Mark events with no votes as NaN
        no_votes = np.sum(fired, axis=1) == 0
        voted[no_votes] = np.nan
        return voted

    voted_train = _vote(train_idx, use_return_weighting=use_return_weighted_labels)
    voted_test = _vote(test_idx, use_return_weighting=use_return_weighted_labels)

    weights_dict = {f"expert_{k}": float(w_arr[k]) for k in range(n_experts)}
    return voted_train, voted_test, weights_dict


def compute_committee_voted_labels_full(
    label_matrix: np.ndarray,
    returns_matrix: np.ndarray,
    event_index: pd.DatetimeIndex,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    floor_weight: float = 0.1,
    alpha_hit_rate: float = 0.5,
    use_return_weighted_labels: bool = True,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Compute committee-voted labels for ALL events using CV-style train-only weights.

    For each fold, weights are learned from train events and applied to test events.
    The result is a complete label series covering all events (OOF-style).

    Args:
        label_matrix: (n_events, n_experts) trinary labels
        returns_matrix: (n_events, n_experts) realized returns
        event_index: DatetimeIndex aligned to rows
        cv_splits: list of (train_idx, test_idx) tuples
        floor_weight: minimum expert weight
        alpha_hit_rate: hit_rate contribution to weight
        use_return_weighted_labels: if True, use actual returns to determine labels
            (positive return = label 1, negative = label 0). Default True.

    Returns:
        voted_labels: pd.Series of binary labels (0/1) indexed by event_index
        diagnostics: dict with per-fold weights and coverage stats
    """
    n_events = label_matrix.shape[0]
    voted = np.full(n_events, np.nan, dtype=float)
    fold_weights = []

    for fold_i, (train_idx, test_idx) in enumerate(cv_splits):
        _, voted_test, weights = compute_committee_voted_labels_for_cv(
            label_matrix=label_matrix,
            returns_matrix=returns_matrix,
            event_index=event_index,
            train_idx=train_idx,
            test_idx=test_idx,
            floor_weight=floor_weight,
            alpha_hit_rate=alpha_hit_rate,
            use_return_weighted_labels=use_return_weighted_labels,
        )
        voted[test_idx] = voted_test
        fold_weights.append({"fold": fold_i, **weights})

    voted_series = pd.Series(voted, index=event_index)

    diagnostics = {
        "n_events": n_events,
        "n_labeled": int(np.sum(~np.isnan(voted))),
        "pos_rate": float(np.nanmean(voted)) if np.any(~np.isnan(voted)) else None,
        "fold_weights": fold_weights,
    }

    return voted_series, diagnostics


if __name__ == "__main__":
    # Run example when module is executed directly
    example_kalman_multi_triple_barrier_usage()
