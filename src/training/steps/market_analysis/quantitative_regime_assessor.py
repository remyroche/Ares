from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import variation, entropy
from dataclasses import dataclass, field

@dataclass
class RegimeMetrics:
    """Detailed metrics for a single regime."""
    conditional_pnl: float
    conditional_sharpe: float
    conditional_kelly: float
    tail_risk_contribution: float  # CVaR (95%) contribution
    vol_adjusted_pnl_correlation: float
    regime_entropy: float
    velocity: float
    acceleration: float

    def to_dict(self):
        return {
            'Conditional P&L': self.conditional_pnl,
            'Conditional Sharpe': self.conditional_sharpe,
            'Conditional Kelly': self.conditional_kelly,
            'Tail Risk (CVaR)': self.tail_risk_contribution,
            'Vol-Adj P&L Corr': self.vol_adjusted_pnl_correlation,
            'Regime Entropy': self.regime_entropy,
            'Velocity': self.velocity,
            'Acceleration': self.acceleration
        }

class QuantitativeRegimeAssessor:
    """
    Advanced assessor for Quantitative Regime Engine.
    Calculates economic and statistical metrics for GMM/DPGMM/IOHMM regimes.
    """

    def calculate_metrics(self,
                         regime_probs: pd.DataFrame,
                         returns: pd.Series,
                         volatility: pd.Series) -> Dict[str, Any]:
        """
        Calculate metrics for all regimes.

        Args:
            regime_probs: DataFrame (T x K) of regime probabilities.
            returns: Series (T,) of returns aligned with regime_probs.
            volatility: Series (T,) of rolling volatility.

        Returns:
            Dictionary containing per-regime metrics and cross-regime CVs.
        """
        n_regimes = regime_probs.shape[1]
        regime_metrics = {}

        # Pre-calculate common series
        vol_adj_returns = returns / (volatility + 1e-9)

        # Calculate transition matrix
        hard_states = regime_probs.idxmax(axis=1)
        # Assuming column names are like 'REGIME_0', 'REGIME_1'... or integer index
        # Let's map to integers 0..K-1
        state_map = {col: i for i, col in enumerate(regime_probs.columns)}
        int_states = hard_states.map(state_map).fillna(-1).astype(int)

        transitions = np.zeros((n_regimes, n_regimes))
        for t in range(len(int_states) - 1):
            curr = int_states.iloc[t]
            next_s = int_states.iloc[t+1]
            if curr != -1 and next_s != -1:
                transitions[curr, next_s] += 1

        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)

        # Loop over regimes
        for col in regime_probs.columns:
            probs = regime_probs[col]

            # Weighted mean returns (Conditional P&L)
            # Weights are probabilities
            w = probs / probs.sum()
            cond_pnl = np.sum(returns * w)

            # Conditional Volatility
            cond_var = np.sum((returns - cond_pnl)**2 * w)
            cond_vol = np.sqrt(cond_var)

            # Sharpe
            cond_sharpe = cond_pnl / (cond_vol + 1e-9)

            # Kelly (Simple approximation: mu/sigma^2)
            cond_kelly = cond_pnl / (cond_var + 1e-9)

            # Tail Risk (CVaR 95%)
            # Weighted percentile is tricky. Let's use subset resampling or approximation.
            # Using soft-weighted subset:
            # Filter where prob > 0.1 for meaningful stats
            mask = probs > 0.1
            if mask.sum() > 10:
                subset_rets = returns[mask]
                subset_w = probs[mask]
                # Sort
                sorted_idx = np.argsort(subset_rets)
                sorted_rets = subset_rets.iloc[sorted_idx]
                sorted_w = subset_w.iloc[sorted_idx]
                cum_w = sorted_w.cumsum() / sorted_w.sum()

                # Find cutoff for 5%
                cvar_mask = cum_w <= 0.05
                if cvar_mask.sum() > 0:
                    tail_risk = np.average(sorted_rets[cvar_mask], weights=sorted_w[cvar_mask])
                else:
                    # Fallback to simple quantile if weights are too sparse
                    tail_risk = subset_rets.quantile(0.05)
            else:
                tail_risk = 0.0

            # Vol-adjusted P&L correlation
            # Correlation between Returns and Volatility in this regime
            # "Is this a down-vol or up-vol regime?"
            if mask.sum() > 10:
                corr = returns[mask].corr(volatility[mask])
            else:
                corr = 0.0

            # Regime-level entropy of returns
            # Histogram of returns in this regime -> Entropy
            if mask.sum() > 10:
                hist, _ = np.histogram(returns[mask], bins=20, density=True)
                reg_ent = entropy(hist + 1e-9)
            else:
                reg_ent = 0.0

            # Velocity / Acceleration of Regime Probability
            # Velocity = d(Prob)/dt
            vel = probs.diff()
            acc = vel.diff()
            avg_vel = vel.abs().mean()
            avg_acc = acc.abs().mean()

            metrics = RegimeMetrics(
                conditional_pnl=cond_pnl,
                conditional_sharpe=cond_sharpe,
                conditional_kelly=cond_kelly,
                tail_risk_contribution=tail_risk,
                vol_adjusted_pnl_correlation=corr if not np.isnan(corr) else 0.0,
                regime_entropy=reg_ent,
                velocity=avg_vel,
                acceleration=avg_acc
            )
            regime_metrics[col] = metrics

        # Calculate Coefficient of Variation (CV) between regimes
        # Collect metric arrays
        metric_keys = list(regime_metrics[list(regime_metrics.keys())[0]].to_dict().keys())
        cv_stats = {}

        for key in metric_keys:
            values = [regime_metrics[r].to_dict()[key] for r in regime_metrics]
            # CV = std / mean
            # Handle negative means by taking absolute
            mu = np.mean(values)
            sigma = np.std(values)
            if abs(mu) > 1e-9:
                cv = sigma / abs(mu)
            else:
                cv = 0.0
            cv_stats[key] = cv

        return {
            "per_regime_metrics": {k: v.to_dict() for k, v in regime_metrics.items()},
            "cross_regime_cv": cv_stats,
            "transition_matrix": trans_matrix.tolist()
        }
