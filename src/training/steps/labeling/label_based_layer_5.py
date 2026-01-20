"""
Layer 5 — Grid Search & Optimization for Position Sizing.

This module implements a grid search to optimize position sizing parameters
based on Layer 4's OOF probabilities and market data.

It uses Numba-optimized backtesting to efficiently test thousands of combinations
of:
1. Threshold
2. Kelly Fraction
3. Steepness
4. SL/TP ATR Multipliers
5. Dampening (Uncertainty)

"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import time
import itertools
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success, tprint_error

try:
    from src.utils.layer5_optimized import apply_position_sizing_numba, run_atr_backtest_numba
except ImportError:
    # Fallback or local import if running as script
    try:
        from layer5_optimized import apply_position_sizing_numba, run_atr_backtest_numba
    except ImportError:
        tprint_error("❌ Could not import Numba optimized functions. Ensure src.utils.layer5_optimized exists.")
        raise

class Layer5PositionSizer:
    """
    Performs grid search optimization for Layer 5 position sizing parameters.
    Also provides position sizing calculation and backtesting capabilities.
    """

    def __init__(
        self,
        oof_df: pd.DataFrame,
        prob_col: str = 'layer4_prob',
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        atr_col: str = 'atr',
        initial_balance: float = 100000.0,
        **kwargs # Accept legacy args to prevent crash, but ignore them
    ):
        self.df = oof_df.copy()
        self.prob_col = prob_col
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.atr_col = atr_col
        self.initial_balance = initial_balance

        # Validate required columns
        required = [prob_col, price_col, high_col, low_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure ATR exists
        if atr_col not in self.df.columns:
            tprint_warning(f"⚠️ ATR column '{atr_col}' not found. Calculating ATR(14)...")
            self._calculate_atr()

        # Ensure Dampening (Uncertainty) exists
        if 'dampening' not in self.df.columns:
            tprint_info("ℹ️ 'dampening' column not found. Calculating uncertainty...")
            self._calculate_uncertainty()

        # Pre-compute Trend and Volatility regimes for breakdown metrics
        self._calculate_regimes()

        # Convert to numpy for Numba
        self.prices_close = self.df[self.price_col].to_numpy(dtype=np.float64)
        self.prices_high = self.df[self.high_col].to_numpy(dtype=np.float64)
        self.prices_low = self.df[self.low_col].to_numpy(dtype=np.float64)
        self.atr = self.df[self.atr_col].to_numpy(dtype=np.float64)
        self.probs = self.df[self.prob_col].to_numpy(dtype=np.float64)
        self.dampening = self.df['dampening'].to_numpy(dtype=np.float64)

        # Fill NaNs
        self.prices_close = np.nan_to_num(self.prices_close)
        self.prices_high = np.nan_to_num(self.prices_high)
        self.prices_low = np.nan_to_num(self.prices_low)
        self.atr = np.nan_to_num(self.atr)
        self.probs = np.nan_to_num(self.probs)
        self.dampening = np.nan_to_num(self.dampening)

    def _calculate_atr(self, window: int = 14):
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        close = self.df[self.price_col]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df[self.atr_col] = tr.rolling(window=window).mean().bfill()

    def _calculate_uncertainty(self):
        """
        Calculates a 0-1 'dampening' score where 1 = High Uncertainty.
        Priority:
        1. Entropy MA (rank normalized)
        2. Proxy Entropy
        3. Prob Std (from Layer 4 ensemble)
        4. Volatility (rank normalized)
        """
        # 1. Entropy
        entropy_cols = [c for c in self.df.columns if 'entropy_ma' in c]
        if entropy_cols:
            col = entropy_cols[0]
            # Normalize to 0-1
            self.df['dampening'] = self.df[col].rank(pct=True).fillna(0.5)
            tprint_info(f"   Using {col} for dampening (rank-normalized).")
            return

        # 2. Proxy Entropy
        if 'proxy_entropy' in self.df.columns:
            self.df['dampening'] = self.df['proxy_entropy'].rank(pct=True).fillna(0.5)
            tprint_info("   Using proxy_entropy for dampening.")
            return

        # 3. Prob Std (if Layer 4 provided it)
        if 'prob_std' in self.df.columns:
            self.df['dampening'] = self.df['prob_std'].rank(pct=True).fillna(0.5)
            tprint_info("   Using prob_std for dampening.")
            return

        # 4. Volatility fallback
        if self.atr_col in self.df.columns:
            # Volatility relative to price
            vol = self.df[self.atr_col] / (self.df[self.price_col] + 1e-9)
            self.df['dampening'] = vol.rank(pct=True).fillna(0.5)
            tprint_info("   Using ATR volatility for dampening fallback.")
        else:
            self.df['dampening'] = 0.0 # No dampening
            tprint_warning("   No uncertainty metric found. Dampening set to 0.")

    def _calculate_regimes(self):
        """
        Pre-compute Trend and Volatility percentiles for breakdown metrics.
        """
        # Trend: Price relative to MA50
        ma50 = self.df[self.price_col].rolling(50).mean()
        trend_ratio = self.df[self.price_col] / (ma50 + 1e-9)
        self.df['regime_trend'] = trend_ratio.rank(pct=True).fillna(0.5)

        # Volatility: ATR relative to Price
        vol_ratio = self.df[self.atr_col] / (self.df[self.price_col] + 1e-9)
        self.df['regime_vol'] = vol_ratio.rank(pct=True).fillna(0.5)

        # Store as numpy for fast lookup during metric calc (if needed,
        # but currently we map trades back to indices so we can just use df lookups or pre-computed arrays)
        self.regime_trend = self.df['regime_trend'].to_numpy()
        self.regime_vol = self.df['regime_vol'].to_numpy()

    def optimize(
        self,
        param_grid: Optional[Dict[str, List[float]]] = None,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Run grid search.
        """
        if param_grid is None:
            # Default sensible grid
            param_grid = {
                'threshold': [0.5, 0.55, 0.6],
                'kelly_fraction': [0.1, 0.2, 0.5], # Conservative to aggressive
                'steepness': [1.0, 2.0], # Linear vs Quadratic
                'sl_atr': [1.5, 2.0, 3.0],
                'trail_trigger': [1.0, 2.0],
                'trail_atr': [0.5, 1.0, 1.5],
                'dampening_mult': [0.0, 0.5, 1.0]
            }

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        tprint_info(f"🚀 Starting Layer 5 Grid Search: {len(combinations)} combinations...")
        t0 = time.time()

        results = []

        for i, params in enumerate(combinations):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(combinations) - i) / rate
                tprint_info(f"   Processed {i}/{len(combinations)} ({rate:.1f} iter/s). ETA: {remaining/60:.1f} min")

            # 1. Calc Sizes
            sizes = apply_position_sizing_numba(
                self.probs,
                self.dampening,
                threshold=params['threshold'],
                kelly_fraction=params['kelly_fraction'],
                steepness=params['steepness'],
                dampening_mult=params.get('dampening_mult', 0.0)
            )

            # 2. Run Backtest
            equity, trades = run_atr_backtest_numba(
                self.prices_close,
                self.prices_high,
                self.prices_low,
                self.atr,
                sizes,
                sl_atr_mult=params['sl_atr'],
                trail_trigger_mult=params['trail_trigger'],
                trail_dist_mult=params['trail_atr'],
                initial_balance=self.initial_balance
            )

            # 3. Compute Metrics
            metrics = self._compute_metrics(equity, trades, sizes, params)

            # Combine params and metrics
            row = {**params, **metrics}
            results.append(row)

        t1 = time.time()
        tprint_success(f"✅ Grid Search Complete in {t1-t0:.2f}s")

        results_df = pd.DataFrame(results)

        # Sort by Sharpe (or user preference)
        if 'Sharpe' in results_df.columns:
            results_df = results_df.sort_values('Sharpe', ascending=False)

        return results_df

    def _compute_metrics(self, equity, trades, sizes, params) -> Dict[str, float]:
        """
        Compute performance metrics.
        """
        metrics = {}

        # Basic
        total_return = (equity[-1] / equity[0]) - 1.0
        metrics['Total Return'] = total_return
        metrics['End Balance'] = equity[-1]

        n_trades = len(trades)
        metrics['Trades'] = n_trades

        if n_trades == 0:
            return {k: 0.0 for k in [
                'Win Rate', 'Sharpe', 'Sortino', 'MaxDD', 'Avg PnL', 'PnL/Day',
                'WinRate_HighConf', 'Corr_Conf_Win',
                'WinRate_HighTrend', 'WinRate_LowTrend',
                'WinRate_HighVol', 'WinRate_LowVol'
            ]}

        # PnL array from trades (col 4 is pnl)
        trade_pnls = trades[:, 4]
        wins = np.sum(trade_pnls > 0)
        metrics['Win Rate'] = wins / n_trades
        metrics['Avg PnL'] = np.mean(trade_pnls)

        # Sharpe / Sortino
        eq_series = pd.Series(equity)
        returns = eq_series.pct_change().fillna(0)

        mean_ret = returns.mean()
        std_ret = returns.std()

        ann_factor = 252 # Default fallback
        if isinstance(self.df.index, pd.DatetimeIndex):
            # Infer frequency
            span = self.df.index[-1] - self.df.index[0]
            days = span.days
            if days > 0:
                samples_per_day = len(self.df) / days
                ann_factor = np.sqrt(samples_per_day * 365)

        if std_ret > 1e-9:
            metrics['Sharpe'] = (mean_ret / std_ret) * ann_factor
        else:
            metrics['Sharpe'] = 0.0

        # Sortino
        downside = returns[returns < 0]
        std_down = downside.std()
        if std_down > 1e-9:
            metrics['Sortino'] = (mean_ret / std_down) * ann_factor
        else:
            metrics['Sortino'] = 0.0

        # MaxDD
        running_max = np.maximum.accumulate(equity)
        dd = (running_max - equity) / running_max
        metrics['MaxDD'] = np.max(dd)

        # PnL / Day
        if isinstance(self.df.index, pd.DatetimeIndex):
            span_days = (self.df.index[-1] - self.df.index[0]).days
            if span_days > 0:
                metrics['PnL/Day'] = (equity[-1] - equity[0]) / span_days
            else:
                metrics['PnL/Day'] = 0.0
        else:
            metrics['PnL/Day'] = 0.0

        # Correlation Conf & Win
        # Entry indices: trades[:, 0]
        entry_indices = trades[:, 0].astype(int)
        entry_probs = self.probs[entry_indices]
        win_binary = (trade_pnls > 0).astype(int)

        if len(entry_probs) > 1:
            metrics['Corr_Conf_Win'] = np.corrcoef(entry_probs, win_binary)[0, 1]
            if np.isnan(metrics['Corr_Conf_Win']): metrics['Corr_Conf_Win'] = 0.0
        else:
            metrics['Corr_Conf_Win'] = 0.0

        # Breakdown: Top/Down 50% by Trend & Vol
        entry_trend_ranks = self.regime_trend[entry_indices]
        entry_vol_ranks = self.regime_vol[entry_indices]

        # High Trend (> 0.5)
        mask_high_trend = entry_trend_ranks > 0.5
        if np.sum(mask_high_trend) > 0:
            metrics['WinRate_HighTrend'] = np.mean(win_binary[mask_high_trend])
        else:
            metrics['WinRate_HighTrend'] = 0.0

        # Low Trend
        mask_low_trend = entry_trend_ranks <= 0.5
        if np.sum(mask_low_trend) > 0:
            metrics['WinRate_LowTrend'] = np.mean(win_binary[mask_low_trend])
        else:
            metrics['WinRate_LowTrend'] = 0.0

        # High Vol
        mask_high_vol = entry_vol_ranks > 0.5
        if np.sum(mask_high_vol) > 0:
            metrics['WinRate_HighVol'] = np.mean(win_binary[mask_high_vol])
        else:
            metrics['WinRate_HighVol'] = 0.0

        # Low Vol
        mask_low_vol = entry_vol_ranks <= 0.5
        if np.sum(mask_low_vol) > 0:
            metrics['WinRate_LowVol'] = np.mean(win_binary[mask_low_vol])
        else:
            metrics['WinRate_LowVol'] = 0.0

        return metrics

    def run_backtest_with_params(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a single backtest with specific parameters and return detailed results.
        Useful for final evaluation of the best model.
        """
        sizes = apply_position_sizing_numba(
            self.probs,
            self.dampening,
            threshold=params['threshold'],
            kelly_fraction=params['kelly_fraction'],
            steepness=params['steepness'],
            dampening_mult=params.get('dampening_mult', 0.0)
        )

        equity, trades = run_atr_backtest_numba(
            self.prices_close,
            self.prices_high,
            self.prices_low,
            self.atr,
            sizes,
            sl_atr_mult=params['sl_atr'],
            trail_trigger_mult=params['trail_trigger'],
            trail_dist_mult=params['trail_atr'],
            initial_balance=self.initial_balance
        )

        metrics = self._compute_metrics(equity, trades, sizes, params)

        # Add detailed artifacts
        return {
            'metrics': metrics,
            'equity_curve': equity,
            'trades': trades,
            'sizes': sizes,
            'params': params
        }

if __name__ == "__main__":
    pass
