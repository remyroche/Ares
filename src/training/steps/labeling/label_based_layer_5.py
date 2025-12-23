"""Layer 5 — Signal-to-Sizing bridge: Trading / Portfolio Construction.

This module implements the final step of the meta-labeling pipeline, converting
calibrated probabilities (Layer 3 output) into actionable position sizes using
a bounded, monotonic probability-to-size mapping.

Formula:
    z(p) = clip((p - p_min) / (p_max - p_min), 0, 1)
    size(p) = z(p) ^ gamma

    Where p_max is dynamically adjusted to be min(configured_p_max, empirical_p_max)
    to ensure models with low confidence ranges can still achieve full sizing if
    they are the best available.

    Note: This implementation supersedes legacy Kelly-based formulas (e.g.
    (0.35 + 0.60 * z(p)) * (Kelly ^ 1.2)) in favor of this direct Power Law mapping.

It also computes sizing diagnostics:
    - Edge Monotonicity Test
    - Bet Utilization Efficiency
    - Tail Loss Amplification
    - Net Sortino, Max Drawdown, Calmar-like Ratio
    - Classification Metrics (AUC, PR-AUC)
    - Optimal Threshold Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import time

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
except ImportError:
    roc_auc_score = None
    average_precision_score = None
    brier_score_loss = None

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success

class Layer5PositionSizer:
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
        p_min: float = 0.1,  # Reduced from 0.5 to allow more trades
        p_max: float = 0.9,
        gamma: float = 1.2,
        transaction_cost: float = 0.000, # Returns assumed net unless specified
        gate_mode: str = 'quantile',
        gate_quantile: Optional[float] = None,
        gate_top_k: Optional[int] = None,
        gate_top_k_per_day: Optional[int] = None,
        gate_search_q_low: Optional[float] = None,
        gate_search_q_high: Optional[float] = None,
        gate_search_min_range: Optional[float] = None,
        gate_search_max_iter: Optional[int] = None,
        min_trades_reliable: int = 50,
        allow_dynamic_p_max: bool = False, # Prevent overfitting to batch max in production/small batches
    ):
        """
        Initialize the Layer 5 Position Sizer.

        Args:
            oof_df: DataFrame containing OOF predictions and outcomes.
            p_col: Column name for calibrated probability.
            target_col: Column name for binary target (unused for sizing, used for diagnostics).
            return_col: Column name for realized return (for backtest).
            vol_col: Column name for volatility (optional, for context).
            p_min: Minimum probability for z-score scaling.
            p_max: Maximum probability for z-score scaling.
            gamma: Power coefficient for probability-to-size mapping.
            transaction_cost: Cost to subtract if returns are gross.
            allow_dynamic_p_max: If True, scales sizing based on the batch's max probability (use for full history only).
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
        self.allow_dynamic_p_max = allow_dynamic_p_max

        self.gate_mode = str(gate_mode or 'p_min')
        self.gate_quantile = gate_quantile
        self.gate_top_k = gate_top_k
        self.gate_top_k_per_day = gate_top_k_per_day

        self.gate_search_q_low = gate_search_q_low
        self.gate_search_q_high = gate_search_q_high
        self.gate_search_min_range = gate_search_min_range
        self.gate_search_max_iter = gate_search_max_iter

        self.min_trades_reliable = int(max(1, min_trades_reliable)) if min_trades_reliable else 50

        self._gate_mask: Optional[np.ndarray] = None
        self._gate_threshold: Optional[float] = None
        self._gate_mode_used: Optional[str] = None

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
                raise ValueError(f"Missing required columns for Layer 5: {missing}")

        # Unit Validation
        if not self.df.empty and self.return_col in self.df.columns:
            mean_abs_ret = self.df[self.return_col].abs().mean()
            if mean_abs_ret > 1.0:
                tprint_warning(
                    f"Mean absolute return is {mean_abs_ret:.2f}, which is > 1.0. "
                    "Ensure returns are in decimal (e.g., 0.01 for 1%) and not percent or basis points."
                )


    def calculate_sizing(self) -> pd.Series:
        """Apply the Signal-to-Sizing formula."""
        if self.df.empty:
            return pd.Series(dtype=float)

        p = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)

        gate_mask, gate_threshold, gate_mode_used = self._compute_gate_mask_and_threshold(p)
        self._gate_mask = gate_mask
        self._gate_threshold = gate_threshold
        self._gate_mode_used = gate_mode_used

        # Use the computed threshold as the effective p_min for sizing.
        p_min_eff = float(gate_threshold)

        # Determine p_max_eff: fallback to empirical max if it's lower than configured p_max
        # to ensure we use the full sizing range [0, 1] even if the model is under-confident.
        p_valid = p[np.isfinite(p)]
        p_max_emp = float(np.max(p_valid)) if p_valid.size > 0 else 1.0

        if self.allow_dynamic_p_max:
            if p_valid.size > 0:
                # If configured p_max is invalid or effectively same as p_min, take empirical
                if (not np.isfinite(self.p_max)) or (self.p_max <= p_min_eff + 1e-9):
                    p_max_eff = p_max_emp
                else:
                    # Use the tighter bound to allow reaching size=1.0
                    p_max_eff = min(self.p_max, p_max_emp)
            else:
                p_max_eff = self.p_max
        else:
            # If dynamic p_max is disabled, verify configured p_max is valid.
            # If invalid (e.g. inf), fallback to empirical to prevent zero-size/error.
            if not np.isfinite(self.p_max):
                tprint_warning(f"Configured p_max is infinite/invalid and allow_dynamic_p_max=False. Falling back to empirical max: {p_max_emp}")
                p_max_eff = p_max_emp
            else:
                p_max_eff = self.p_max

        # 1) Conviction scaler (monotonic above threshold)
        denom = float(p_max_eff) - float(p_min_eff)
        if denom < 1e-6:
            denom = 1e-6

        z_p = np.clip((p - float(p_min_eff)) / denom, 0.0, 1.0)

        size = np.power(z_p, float(self.gamma))
        size = np.where(gate_mask, size, 0.0)

        # Final clip for safety. Capping at 1.0 represents "Full Position".
        size = np.clip(size, 0.0, 1.0)

        return pd.Series(size, index=self.df.index)


    def _compute_gate_mask_and_threshold(self, p: np.ndarray) -> Tuple[np.ndarray, float, str]:
        p = np.asarray(p, dtype=float)
        finite = np.isfinite(p)
        n = int(p.shape[0])
        gate = np.zeros(n, dtype=bool)

        mode = str(self.gate_mode or 'p_min').strip().lower()

        if mode == 'p_min':
            thr = float(self.p_min)
            gate = finite & (p >= thr)
            return gate, thr, 'p_min'

        if mode == 'quantile':
            q = self.gate_quantile if self.gate_quantile is not None else 0.99
            q = float(np.clip(q, 0.0, 1.0))
            p_valid = p[finite]
            thr = float(np.quantile(p_valid, q)) if p_valid.size > 0 else float(self.p_min)
            gate = finite & (p >= thr)
            return gate, thr, 'quantile'

        if mode in ('pnl_opt_quantile', 'pnl_quantile_search', 'quantile_pnl_search'):
            tprint_warning(
                "Layer 5 gate_mode=pnl_opt_quantile is disabled (return-based threshold tuning is leakage-prone). "
                "Falling back to quantile gating using probability distribution only."
            )
            q = self.gate_search_q_high if self.gate_search_q_high is not None else self.gate_quantile
            q = q if q is not None else 0.99
            q = float(np.clip(q, 0.0, 1.0))
            p_valid = p[finite]
            thr = float(np.quantile(p_valid, q)) if p_valid.size > 0 else float(self.p_min)
            gate = finite & (p >= thr)
            return gate, thr, 'quantile'

        if mode == 'top_k':
            k = self.gate_top_k if self.gate_top_k is not None else 0
            p_valid_idx = np.where(finite)[0]
            if k <= 0 or p_valid_idx.size == 0:
                thr = float(self.p_min)
                return (finite & (p >= thr)), thr, 'p_min'
            k = int(min(k, int(p_valid_idx.size)))

            # Pick top-k among finite
            # We want the threshold such that at most k items are selected.
            # Using partition is faster than full sort
            p_vals = p[p_valid_idx]
            if k < p_vals.size:
                partitioned_idx = np.argpartition(p_vals, -k)[-k:]
                thr = float(np.min(p_vals[partitioned_idx]))
            else:
                thr = float(np.min(p_vals))

            gate = finite & (p >= thr)
            return gate, thr, 'top_k'

        if mode == 'top_k_per_day':
            k = self.gate_top_k_per_day if self.gate_top_k_per_day is not None else 0
            if k <= 0:
                thr = float(self.p_min)
                return (finite & (p >= thr)), thr, 'p_min'

            if not isinstance(self.df.index, pd.DatetimeIndex):
                tprint_warning("Index is not DatetimeIndex, falling back to p_min for top_k_per_day")
                thr = float(self.p_min)
                return (finite & (p >= thr)), thr, 'p_min'

            # Vectorized top-k per day using groupby rank
            # We need to construct a Series to use groupby
            p_series = pd.Series(p, index=self.df.index)
            # Normalize index to dates
            dates = self.df.index.normalize()

            # Rank descending (method='first' breaks ties by order, 'min' gives same rank to ties)
            # We use 'first' to ensure exactly k (or fewer) are selected if we want strictly k,
            # but usually 'min' or 'dense' is fine. Let's use 'first' to be deterministic with ties.
            # However, groupby().rank() can be slow on very large frames, but faster than python loop.
            ranks = p_series.groupby(dates).rank(method='first', ascending=False)

            gate = finite & (ranks.values <= k)

            # Effective threshold is the minimum probability among selected
            if np.any(gate):
                thr = float(np.min(p[gate]))
            else:
                thr = float(self.p_min)

            return gate, thr, 'top_k_per_day'

        # Default: find probability level where win-rate is 51%+
        if 'realized_return' in self.df.columns:
            returns = pd.to_numeric(self.df['realized_return'], errors='coerce').to_numpy()
            p_valid = p[finite & np.isfinite(returns)]
            ret_valid = returns[finite & np.isfinite(returns)]
            
            if p_valid.size > 50 and ret_valid.size > 50:
                # Calculate win-rate across probability thresholds
                thresholds = np.linspace(np.min(p_valid), np.max(p_valid), 50)
                win_rates = []
                
                for thr in thresholds:
                    mask = p_valid >= thr
                    if np.sum(mask) > 10:  # Need sufficient trades
                        # FIX: Win-rate must account for transaction costs
                        net_rets_subset = ret_valid[mask] - self.transaction_cost
                        wr = np.mean(net_rets_subset > 0)
                        win_rates.append(wr)
                    else:
                        win_rates.append(0.0)
                
                # Find threshold where win-rate crosses 51%
                win_rates = np.array(win_rates)
                above_51 = thresholds[np.where(win_rates >= 0.51)[0]]
                thr = float(above_51[0]) if len(above_51) > 0 else float(self.p_min)
            else:
                thr = float(self.p_min)
        else:
            thr = float(self.p_min)
        gate = finite & (p >= thr)
        return gate, thr, 'p_min_winrate'


    def get_gate_index(self) -> pd.Index:
        if self._gate_mask is None:
            _ = self.calculate_sizing()
        if self._gate_mask is None:
            return pd.Index([])
        return self.df.index[np.asarray(self._gate_mask, dtype=bool)]

    def run_backtest(self) -> Dict[str, Any]:
        """
        Executes the backtest using computed sizes and generates metrics.
        """
        tprint_info(">>> Running Layer 5 Backtest & Diagnostics...")
        t0 = time.perf_counter()

        if self.df.empty:
            tprint_warning("Empty DataFrame provided to Layer 5 Backtest.")
            return {}

        # 1. Compute Sizes
        sizes = self.calculate_sizing()
        t_sizing = time.perf_counter()

        sizes_np = sizes.to_numpy(dtype=float, copy=False)
        self.df['layer5_size'] = sizes_np

        # 2. Compute Sized Returns (PnL)
        raw_rets = pd.to_numeric(self.df[self.return_col], errors='coerce').to_numpy(dtype=float, copy=False)
        net_rets = raw_rets - self.transaction_cost
        pnl = sizes_np * net_rets
        self.df['layer5_pnl'] = pnl

        # 3. Calculate Metrics
        trade_mask = sizes_np > 1e-4
        metrics = self._compute_metrics(pnl, sizes_np, trade_mask, net_rets)

        # Add runtime metrics
        t_metrics = time.perf_counter()

        # Edge Monotonicity
        mono_res = self._check_edge_monotonicity(pnl)
        t_mono = time.perf_counter()
        metrics['Edge Monotonicity'] = mono_res

        metrics['Runtime'] = {
            'n_rows': int(self.df.shape[0]),
            'sizing_ms': float((t_sizing - t0) * 1000.0),
            'metrics_ms': float((t_metrics - t_sizing) * 1000.0),
            'edge_monotonicity_ms': float((t_mono - t_metrics) * 1000.0),
            'total_ms': float((t_mono - t0) * 1000.0),
        }

        # 4. Generate Report and Save Artifacts
        self._save_artifacts_to_disk(metrics)

        return metrics

    def _compute_metrics(self, pnl: np.ndarray, sizes: np.ndarray, trade_mask: np.ndarray, net_rets: np.ndarray) -> Dict[str, Any]:
        """Compute all scalar metrics."""
        metrics = {}

        # --- Probability Statistics ---
        p_raw = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)
        finite_p = np.isfinite(p_raw)
        p_valid = p_raw[finite_p]

        if p_valid.size > 0:
            metrics['Prob Mean'] = float(np.mean(p_valid))
            metrics['Prob Std'] = float(np.std(p_valid))
            metrics['Prob Q50'] = float(np.quantile(p_valid, 0.50))
            metrics['Prob Q90'] = float(np.quantile(p_valid, 0.90))
            metrics['Prob Q99'] = float(np.quantile(p_valid, 0.99))
            metrics['Prob>=p_min Count'] = int(np.sum(p_valid >= float(self.p_min)))

        gate_thr = self._gate_threshold if self._gate_threshold is not None else float('nan')
        metrics['Configured p_min'] = float(self.p_min)
        metrics['Effective Gate Threshold'] = float(gate_thr)
        metrics['Gate Mode'] = str(self._gate_mode_used or self.gate_mode)

        if self._gate_mask is not None:
             metrics['Gate Count'] = int(np.sum(self._gate_mask))
             if p_valid.size > 0 and np.isfinite(gate_thr):
                 metrics['Prob>=gate_threshold Count'] = int(np.sum(p_valid >= float(gate_thr)))

        # --- PnL Statistics ---
        total_pnl = np.sum(pnl)
        n_trades = int(np.sum(trade_mask))
        avg_pnl = float(np.mean(pnl[trade_mask])) if n_trades > 0 else 0.0

        metrics['Total PnL'] = float(total_pnl)
        metrics['Avg Trade PnL'] = float(avg_pnl)
        metrics['Trade Count'] = int(n_trades)

        # Turnover: sum of absolute size changes (simplified approximation: sum of sizes * 2 for entry/exit)
        # This is strictly "Exposure Turnover", assuming each trade is independent and fully exits.
        metrics['Turnover Estimate'] = float(np.sum(sizes) * 2.0)

        metrics['Min Trades Reliable'] = self.min_trades_reliable
        metrics['Trades Reliable'] = n_trades >= self.min_trades_reliable
        if not metrics['Trades Reliable']:
            metrics['Reliability Flag'] = 'UNRELIABLE_TOO_FEW_TRADES'

        # --- Sortino / Drawdown ---
        pnl_arr = np.nan_to_num(pnl, nan=0.0)
        pnl_arr = np.clip(pnl_arr, -0.999999, None)

        # Calculate daily returns for Sortino if possible
        if isinstance(self.df.index, pd.DatetimeIndex):
            pnl_series = pd.Series(pnl_arr, index=self.df.index)
            daily_returns = pnl_series.groupby(self.df.index.normalize()).apply(
                lambda x: float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0)
            )
            daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
            # Sortino: Downside deviation relative to 0
            daily_down = np.minimum(0, daily_returns.values)
            downside_std = float(np.sqrt(np.mean(np.square(daily_down)))) if daily_down.size > 0 else 1e-6
            sortino = float(np.mean(daily_returns) / downside_std * np.sqrt(365.0)) if downside_std > 1e-9 else 0.0
        else:
            # Fallback to trade-based. Treat 0 returns (no trade) as part of the distribution.
            # Downside deviation relative to 0
            downside_returns = np.minimum(0, pnl_arr)
            downside_std = float(np.sqrt(np.mean(np.square(downside_returns)))) if downside_returns.size > 0 else 1e-6
            # No annualization for trade-based fallback unless we know frequency
            sortino = float(np.mean(pnl_arr) / downside_std) if downside_std > 1e-9 else 0.0

        metrics['Net Sortino'] = float(sortino)

        # Drawdown
        equity = np.cumprod(1.0 + pnl_arr)
        if equity.size > 0:
            running_max = np.maximum.accumulate(equity)
            dd = 1.0 - (equity / (running_max + 1e-12))
            max_dd = float(np.max(dd))
            total_return = float(equity[-1] - 1.0)
        else:
            max_dd = 0.0
            total_return = 0.0

        metrics['Maximum Drawdown'] = float(max_dd)
        metrics['Total Return'] = float(total_return)
        metrics['Return / Drawdown Ratio'] = float(total_return / max_dd) if max_dd > 1e-9 else 0.0

        # --- Bet Utilization & Tail Loss ---
        # Bet Utilization: % of profit coming from high conviction (>0.7 size) bets
        high_conviction = (sizes > 0.7) & trade_mask
        pnl_pos_total = np.sum(pnl[trade_mask & (pnl > 0)])
        pnl_pos_high = np.sum(pnl[high_conviction & (pnl > 0)])
        metrics['Bet Utilization Efficiency'] = float(pnl_pos_high / pnl_pos_total) if pnl_pos_total > 1e-9 else 0.0

        # Tail Loss Amplification: Max DD of sized vs flat strategy
        # FIX: Baseline should be "Flat on Selected Trades", not "Flat on All Trades"
        # If gate is active, we only want to compare sizing efficacy, not gate efficacy.
        flat_size_mask = self._gate_mask if self._gate_mask is not None else np.ones_like(pnl, dtype=bool)
        flat_arr = np.where(flat_size_mask, net_rets, 0.0)
        flat_arr = np.nan_to_num(flat_arr, nan=0.0)
        flat_arr = np.clip(flat_arr, -0.999999, None)

        flat_eq = np.cumprod(1.0 + flat_arr)
        if flat_eq.size > 0:
            flat_peak = np.maximum.accumulate(flat_eq)
            flat_dd = 1.0 - (flat_eq / (flat_peak + 1e-12))
            flat_max_dd = float(np.max(flat_dd))
        else:
            flat_max_dd = 0.0

        metrics['Tail Loss Amplification'] = float(max_dd / flat_max_dd) if flat_max_dd > 1e-9 else 1.0

        # --- Classification Metrics (AUC, PR-AUC) ---
        if roc_auc_score is not None:
            # Determine target: Use `target_col` if available, else derive from net_returns
            if self.target_col in self.df.columns and self.df[self.target_col].notna().any():
                y_true = pd.to_numeric(self.df[self.target_col], errors='coerce').fillna(0).values
                # Ensure binary
                y_true = (y_true > 0.5).astype(int)
            else:
                y_true = (net_rets > 0).astype(int)

            # Mask valid
            valid_clf = finite_p & np.isfinite(net_rets)
            if np.sum(valid_clf) > 10 and len(np.unique(y_true[valid_clf])) > 1:
                try:
                    metrics['Gate AUC'] = float(roc_auc_score(y_true[valid_clf], p_raw[valid_clf]))
                    metrics['Gate PR-AUC'] = float(average_precision_score(y_true[valid_clf], p_raw[valid_clf]))
                    metrics['Gate Brier Score'] = float(brier_score_loss(y_true[valid_clf], p_raw[valid_clf]))
                except Exception:
                    metrics['Gate AUC'] = 0.5
            else:
                metrics['Gate AUC'] = 0.5

        # --- Optimal Threshold Search & Baseline Comparison ---
        # Baseline (Pre-Gate): Just use p_min as threshold, no advanced gating
        # Assuming Pre-Gate means "Configured p_min with no other logic"
        base_mask = finite_p & (p_raw >= self.p_min)
        base_count = np.sum(base_mask)
        if base_count > 0:
            base_rets = net_rets[base_mask]
            metrics['Pre-Gate Trade Count'] = int(base_count)
            metrics['Pre-Gate PnL'] = float(np.sum(base_rets))
            metrics['Pre-Gate Avg PnL'] = float(np.mean(base_rets))

            # Calculate Pre-Gate Max Drawdown
            base_arr = np.nan_to_num(base_rets, nan=0.0)
            base_arr = np.clip(base_arr, -0.999999, None)
            base_eq = np.cumprod(1.0 + base_arr)
            if base_eq.size > 0:
                base_peak = np.maximum.accumulate(base_eq)
                base_dd = 1.0 - (base_eq / (base_peak + 1e-12))
                metrics['Pre-Gate Max Drawdown'] = float(np.max(base_dd))
            else:
                metrics['Pre-Gate Max Drawdown'] = 0.0

            # Simple Sharpe approximation for Pre-Gate
            if len(base_rets) > 1:
                metrics['Pre-Gate Sharpe'] = float(np.mean(base_rets) / (np.std(base_rets) + 1e-9))
            else:
                metrics['Pre-Gate Sharpe'] = 0.0
        else:
            metrics['Pre-Gate Trade Count'] = 0
            metrics['Pre-Gate PnL'] = 0.0
            metrics['Pre-Gate Max Drawdown'] = 0.0

        # Optimal Threshold Scan
        # Scan percentiles to find best Sharpe/PnL
        if np.sum(finite_p) > 20:
            best_sharpe = -999.0
            best_thr = self.p_min
            best_pnl = -999.0
            best_count = 0
            best_max_dd = 0.0

            # Scan 20 points between p_min and p_max(empirical)
            p_scan_max = np.max(p_raw[finite_p])
            thresholds = np.linspace(self.p_min, p_scan_max, 20)

            for thr in thresholds:
                mask = finite_p & (p_raw >= thr)
                if np.sum(mask) < 5: continue

                rets = net_rets[mask]
                mean_r = np.mean(rets)
                std_r = np.std(rets)
                sharpe = mean_r / (std_r + 1e-9)
                pnl_sum = np.sum(rets)

                if sharpe > best_sharpe:
                    best_sharpe = float(sharpe)
                    best_thr = float(thr)
                    best_pnl = float(pnl_sum)
                    best_count = int(np.sum(mask))

                    # Calculate Max DD for optimal threshold
                    opt_arr = np.nan_to_num(rets, nan=0.0)
                    opt_arr = np.clip(opt_arr, -0.999999, None)
                    opt_eq = np.cumprod(1.0 + opt_arr)
                    if opt_eq.size > 0:
                        opt_peak = np.maximum.accumulate(opt_eq)
                        opt_dd = 1.0 - (opt_eq / (opt_peak + 1e-12))
                        best_max_dd = float(np.max(opt_dd))
                    else:
                        best_max_dd = 0.0

            metrics['Optimal Threshold'] = best_thr
            metrics['Optimal Sharpe'] = best_sharpe
            metrics['Optimal PnL'] = best_pnl
            metrics['Optimal Trade Count'] = best_count
            metrics['Optimal Max Drawdown'] = best_max_dd

            # Improvement Stats
            current_sharpe = metrics.get('Net Sortino', 0.0) # Using Sortino as proxy or need to calc actual sharpe for 'Post-Gate'
            # Let's calc actual Post-Gate Sharpe for fair comparison (using Trade-based)
            traded_rets = net_rets[trade_mask] if np.sum(trade_mask) > 0 else []
            if len(traded_rets) > 1:
                post_gate_sharpe = float(np.mean(traded_rets) / (np.std(traded_rets) + 1e-9))
            else:
                post_gate_sharpe = 0.0

            metrics['Post-Gate Sharpe'] = post_gate_sharpe
            metrics['Sharpe Delta (Opt - Post)'] = best_sharpe - post_gate_sharpe

        # --- Trade Statistics ---
        if n_trades > 0:
            traded_pnl = pnl[trade_mask]
            traded_pnl = traded_pnl[np.isfinite(traded_pnl)]
            wins = traded_pnl[traded_pnl > 0.0]
            losses = traded_pnl[traded_pnl < 0.0]

            metrics['Win Rate'] = float(wins.size / traded_pnl.size)
            gross_profit = float(np.sum(wins))
            gross_loss = float(-np.sum(losses))
            metrics['Gross Profit'] = gross_profit
            metrics['Gross Loss'] = gross_loss
            metrics['Profit Factor'] = float(gross_profit / gross_loss) if gross_loss > 1e-9 else float('nan')

            avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
            avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
            metrics['Payoff Ratio'] = float(avg_win / -avg_loss) if (avg_win > 0 and avg_loss < 0) else float('nan')

            metrics['Expectancy'] = float(np.mean(traded_pnl))
            metrics['Exposure'] = float(np.mean(trade_mask.astype(float)))

            traded_sizes = sizes[trade_mask]
            metrics['Average Size'] = float(np.mean(traded_sizes))
            metrics['Median Size'] = float(np.median(traded_sizes))

        # --- Parameters ---
        metrics['Parameters'] = {
            'gamma': self.gamma,
            'p_min': self.p_min,
            'p_max': self.p_max,
            'gate_mode': self._gate_mode_used,
            'gate_quantile': self.gate_quantile,
            'gate_top_k': self.gate_top_k,
            'gate_top_k_per_day': self.gate_top_k_per_day,
            'allow_dynamic_p_max': self.allow_dynamic_p_max
        }

        return metrics

    def _save_artifacts_to_disk(self, metrics: Dict[str, Any]):
        """Save CSV metrics, sized events, and Markdown report."""
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        symbol = str(self.df['symbol'].iloc[0]) if 'symbol' in self.df.columns else ''
        timeframe = str(self.df['timeframe'].iloc[0]) if 'timeframe' in self.df.columns else ''
        suffix = f"{symbol}_{timeframe}_{ts}".strip('_')

        # 1. Metrics CSV
        flat = {
            'timestamp': ts,
            'symbol': symbol,
            'timeframe': timeframe,
            'p_col': self.p_col,
            'return_col': self.return_col
        }
        # Flatten metrics dictionary
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float, str, bool)) or sub_v is None:
                        flat[f"{k.lower().replace(' ', '_')}_{sub_k}"] = sub_v
            elif isinstance(v, (int, float, str, bool)) or v is None:
                flat[k] = v

        pd.DataFrame([flat]).to_csv(outcomes_dir / f"layer5_metrics_{suffix}.csv", index=False)

        # 2. Sized DataFrame
        export_cols = [c for c in [self.p_col, self.target_col, self.return_col, self.vol_col, 'layer5_size', 'layer5_pnl'] if c in self.df.columns]
        self.df[export_cols].to_csv(outcomes_dir / f"layer5_sized_events_{suffix}.csv", index=True)

        # 3. Markdown Report
        self._generate_markdown_report(metrics, outcomes_dir / f"layer5_report_{suffix}.md", symbol, timeframe, ts)

    def _generate_markdown_report(self, metrics: Dict[str, Any], filepath: Path, symbol: str, timeframe: str, ts: str):
        lines = [
            "# Layer5 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- n_rows: {len(self.df)}\n",
            f"- p_col: {self.p_col}\n",
            "\n## Summary\n"
        ]

        summary_keys = ['Trades Reliable', 'Trade Count', 'Total PnL', 'Avg Trade PnL', 'Maximum Drawdown', 'Total Return']
        for k in summary_keys:
            if k in metrics:
                lines.append(f"- {k}: {metrics[k]}\n")

        if not metrics.get('Trades Reliable', False):
             lines.append("- reliability_note: PF/Sortino/Sharpe-like are suppressed in Summary due to low trade count\n")

        lines.append("\n## Metrics\n")

        # Sort keys for consistent output
        for k, v in sorted(metrics.items()):
            if k in summary_keys or isinstance(v, dict):
                continue
            lines.append(f"- {k}: {v}\n")

        for section in ['Parameters', 'Runtime']:
            if section in metrics and isinstance(metrics[section], dict):
                 lines.append(f"\n## {section}\n")
                 for k, v in sorted(metrics[section].items()):
                     lines.append(f"- {k}: {v}\n")

        filepath.write_text(''.join(lines))


    def _check_edge_monotonicity(self, pnl: np.ndarray) -> Dict[str, Any]:
        """
        Check if realized sharpe/return increases with probability deciles.
        """
        prob = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)
        pnl = np.asarray(pnl, dtype=float)
        mask = np.isfinite(prob) & np.isfinite(pnl)

        if not np.any(mask):
            return {'correlation': 0.0}

        bins = np.arange(0.0, 1.01, 0.1)
        n_bins = len(bins) - 1

        # Digitize returns indices 1..10
        cut_idx = np.digitize(prob[mask], bins, right=True)
        # Filter out of bounds
        valid_idx = (cut_idx > 0) & (cut_idx <= n_bins)

        # 0-based bin indices for bincount
        bin_idx = cut_idx[valid_idx] - 1
        pnl_v = pnl[mask][valid_idx]

        if pnl_v.size == 0:
            return {'correlation': 0.0}

        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        sums = np.bincount(bin_idx, weights=pnl_v, minlength=n_bins).astype(float)
        sums_sq = np.bincount(bin_idx, weights=pnl_v * pnl_v, minlength=n_bins).astype(float)

        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.divide(sums, counts)
            m2 = np.divide(sums_sq, counts)
            var = m2 - mean * mean
            var = np.where(var < 0, 0.0, var) # float precision issues
            std = np.sqrt(var)

            # Sharpe calculation (simplified)
            sharpe = np.divide(mean, std)
            sharpe = np.nan_to_num(sharpe, nan=0.0)
            sharpe = np.clip(sharpe, -50.0, 50.0)

        # Min count filter
        min_count = max(10, int(0.02 * len(pnl_v)))
        valid_bins = counts >= min_count

        # Analyze upper half (prob >= 0.5)
        # Bins: 0.0-0.1 (idx0), ..., 0.5-0.6 (idx5), ..., 0.9-1.0 (idx9)
        # We look at indices 5 to 9
        upper_indices = np.where(valid_bins & (np.arange(n_bins) >= 5))[0]

        if len(upper_indices) > 1:
            sharpes = sharpe[upper_indices]
            corr = float(np.corrcoef(upper_indices, sharpes)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0

        return {
            'correlation': corr,
            'bins': {f"{bins[i]:.1f}-{bins[i+1]:.1f}": float(sharpe[i]) for i in range(n_bins) if valid_bins[i]}
        }
