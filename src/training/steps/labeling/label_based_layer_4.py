"""Layer 4 — Signal-to-Sizing bridge: Trading / Portfolio Construction.

This module implements the final step of the meta-labeling pipeline, converting
calibrated probabilities (Layer 3 output) into actionable position sizes using
a bounded, monotonic Kelly-style sizing rule.

Formula:
    size(p) = (0.35 + 0.60 * z(p)) * (Kelly ^ 1.2)

Where:
    z(p) = clip((p - p_min) / (p_max - p_min), 0, 1)
    Kelly = Theoretical Continuous Kelly based on Meta-Model's Probability
    gamma (implicit) = 1.2 (aggressive/conservative tuning parameter)

It also computes "Kelly Consistency Diagnostics":
    - Edge Monotonicity Test
    - Bet Utilization Efficiency
    - Tail Loss Amplification
    - Net Sortino, Max Drawdown, Calmar-like Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success

class Layer4PositionSizer:
    """
    Computes position sizing and performs backtesting/diagnostics on OOF predictions.
    """

    def __init__(
        self,
        oof_df: pd.DataFrame,
        p_col: str = 'meta_prob',
        target_col: str = 'target',  # Realized binary target or similar
        return_col: str = 'realized_return', # Actual return of the trade
        vol_col: str = 'volatility_1d',
        p_min: float = 0.5,
        p_max: float = 0.9,
        gamma: float = 1.2,
        transaction_cost: float = 0.000, # Returns assumed net unless specified
    ):
        """
        Initialize the Layer 4 Sizer.

        Args:
            oof_df: DataFrame containing OOF predictions and outcomes.
            p_col: Column name for calibrated probability.
            target_col: Column name for binary target (unused for sizing, used for diagnostics).
            return_col: Column name for realized return (for Kelly estimation and backtest).
            vol_col: Column name for volatility (optional, for context).
            p_min: Minimum probability for z-score scaling.
            p_max: Maximum probability for z-score scaling.
            gamma: Kelly power coefficient (1.2).
            transaction_cost: Cost to subtract if returns are gross.
        """
        self.df = oof_df.copy()
        self.p_col = p_col
        self.target_col = target_col
        self.return_col = return_col
        self.vol_col = vol_col
        self.p_min = p_min
        self.p_max = p_max
        self.gamma = gamma
        self.transaction_cost = transaction_cost

        # Validate columns
        missing = [c for c in [p_col, return_col] if c not in self.df.columns]
        if missing:
            # Try to map common aliases if missing
            aliases = {'meta_prob': 'prob', 'realized_return': 'ret'}
            for req, alias in aliases.items():
                if req in missing and alias in self.df.columns:
                    self.df[req] = self.df[alias]
                    missing.remove(req)
            if missing:
                raise ValueError(f"Missing required columns for Layer 4: {missing}")

    def _calculate_theoretical_kelly(self, p: np.ndarray) -> np.ndarray:
        """
        Calculate Theoretical Continuous Kelly fraction based on probability.

        K(p) = E[r|p] / E[r^2|p]

        We estimate AvgWin and AvgLoss globally from the dataset to construct
        the conditional moments.
        """
        # Global estimates of Win/Loss magnitude
        # Note: We use the actual distribution of returns to estimate W and L
        rets = self.df[self.return_col].values
        # Filter for non-zero outcomes to estimate W/L magnitudes
        # (Assuming 0 return is no-trade or flat, but here we analyze active signals)

        wins = rets[rets > 0]
        losses = rets[rets < 0]

        avg_win = np.mean(wins) if len(wins) > 0 else 0.01
        avg_loss = np.mean(losses) if len(losses) > 0 else -0.01

        # Ensure loss is negative
        if avg_loss > 0: avg_loss = -avg_loss

        # Expected Return: mu(p) = p * W + (1-p) * L
        mu = p * avg_win + (1 - p) * avg_loss

        # Expected Second Moment: E[r^2|p] = p * W^2 + (1-p) * L^2
        # (Variance + mu^2)
        m2 = p * (avg_win ** 2) + (1 - p) * (avg_loss ** 2)

        # Kelly Fraction = mu / m2
        # Avoid division by zero
        kelly = np.divide(mu, m2, out=np.zeros_like(mu), where=m2!=0)

        # Clip negative Kelly (we don't short based on long signal failure logic here,
        # usually low prob means size=0)
        return np.maximum(kelly, 0.0)

    def calculate_sizing(self) -> pd.Series:
        """Apply the Signal-to-Sizing formula."""
        p = self.df[self.p_col].values

        # 1. Z-score (Conviction Scaler)
        # z(p) = clip((p - p_min) / (p_max - p_min), 0, 1)
        denom = self.p_max - self.p_min
        if denom < 1e-6: denom = 1e-6
        z_p = np.clip((p - self.p_min) / denom, 0.0, 1.0)

        # 2. Kelly Baseline
        kelly_fraction = self._calculate_theoretical_kelly(p)

        # 3. Formula: size = (0.35 + 0.60 * z(p)) * (Kelly ^ 1.2)
        # We ensure Kelly is non-negative before power
        kelly_component = np.power(kelly_fraction, self.gamma)

        scaler_component = 0.35 + 0.60 * z_p

        size = scaler_component * kelly_component

        # Final clip for safety. Capping at 1.0 represents "Full Position".
        size = np.clip(size, 0.0, 1.0)

        return pd.Series(size, index=self.df.index)

    def run_backtest(self) -> Dict[str, Any]:
        """
        Executes the backtest using computed sizes and generates metrics.
        """
        tprint_info(">>> Running Layer 4 Backtest & Diagnostics...")

        # 1. Compute Sizes
        sizes = self.calculate_sizing()
        self.df['layer4_size'] = sizes

        # 2. Compute Sized Returns (PnL)
        # Assume realized_return is what we get if we bet size 1.
        raw_rets = self.df[self.return_col].values
        net_rets = raw_rets - self.transaction_cost

        # Vectorized PnL
        pnl = sizes.values * net_rets
        self.df['layer4_pnl'] = pnl

        # 3. Calculate Metrics
        metrics = {}

        # -- Performance --
        total_pnl = np.sum(pnl)
        n_trades = np.sum(sizes > 1e-4)
        avg_pnl = np.mean(pnl[sizes > 1e-4]) if n_trades > 0 else 0.0

        # Sortino
        # Sortino = Mean / DownsideStd
        downside_returns = pnl[pnl < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 1e-6
        sortino = (np.mean(pnl) / downside_std) * np.sqrt(365 * 24 * 4) if downside_std > 1e-9 else 0.0

        metrics['Net Sortino'] = float(sortino)
        metrics['Total PnL'] = float(total_pnl)
        metrics['Avg Trade PnL'] = float(avg_pnl)
        metrics['Trade Count'] = int(n_trades)

        # Max Drawdown (on cumulative PnL)
        cum_pnl = np.cumsum(pnl)
        if len(cum_pnl) > 0:
            running_max = np.maximum.accumulate(cum_pnl)
            drawdown = running_max - cum_pnl
            max_dd = np.max(drawdown)
        else:
            max_dd = 0.0
        metrics['Maximum Drawdown'] = float(max_dd)

        # Return / Drawdown (Calmar-like)
        calmar = total_pnl / max_dd if max_dd > 1e-6 else 0.0
        metrics['Return / Drawdown Ratio'] = float(calmar)

        # -- Diagnostics --

        # A. Edge Monotonicity Test
        mono_res = self._check_edge_monotonicity(pnl)
        metrics['Edge Monotonicity'] = mono_res

        # B. Bet Utilization Efficiency
        # Efficiency = Sum(PnL | Size > 0.7) / Sum(Total PnL)
        high_conviction_mask = sizes > 0.7
        high_conv_pnl = np.sum(pnl[high_conviction_mask])
        util_eff = high_conv_pnl / total_pnl if abs(total_pnl) > 1e-6 else 0.0
        metrics['Bet Utilization Efficiency'] = float(util_eff)

        # C. Tail Loss Amplification
        # Ratio: DD_Sized / DD_Flat
        flat_pnl = net_rets
        flat_cum = np.cumsum(flat_pnl)
        if len(flat_cum) > 0:
            flat_max = np.maximum.accumulate(flat_cum)
            flat_dd = np.max(flat_max - flat_cum)
        else:
            flat_dd = 0.0

        tla_ratio = max_dd / flat_dd if flat_dd > 1e-6 else 1.0
        metrics['Tail Loss Amplification'] = float(tla_ratio)

        metrics['Parameters'] = {
            'gamma': self.gamma,
            'p_min': self.p_min,
            'p_max': self.p_max
        }

        return metrics

    def _check_edge_monotonicity(self, pnl: np.ndarray) -> Dict[str, Any]:
        """
        Check if realized sharpe/return increases with probability deciles.
        """
        df_diag = pd.DataFrame({
            'prob': self.df[self.p_col],
            'pnl': pnl
        })

        # Bin by 0.1
        bins = np.arange(0.0, 1.01, 0.1)
        df_diag['bin'] = pd.cut(df_diag['prob'], bins)

        # Groupby and aggregate
        stats = df_diag.groupby('bin', observed=True)['pnl'].agg(['mean', 'std', 'count'])
        stats['sharpe'] = stats['mean'] / (stats['std'] + 1e-9)

        # Check monotonicity in the upper half (0.5+)
        # We look at bins > 0.5
        # The index contains Intervals. We extract the left bound of each interval in the index directly.
        left_bounds = np.array([i.left for i in stats.index])
        upper_stats = stats[left_bounds >= 0.5].dropna()

        sharpes = upper_stats['sharpe'].values
        # Simple check: is correlation between bin index and Sharpe positive?
        if len(sharpes) > 1:
            indices = np.arange(len(sharpes))
            corr = np.corrcoef(indices, sharpes)[0,1]
        else:
            corr = 0.0

        return {
            'correlation': float(corr) if np.isfinite(corr) else 0.0,
            'bins': stats['sharpe'].to_dict()
        }

    def save_artifacts(self, output_dir: Path):
        """Save results to disk."""
        self.df.to_csv(output_dir / "layer4_sized_events.csv", index=True)
