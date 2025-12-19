"""Layer 5 — Signal-to-Sizing bridge: Trading / Portfolio Construction.

This module implements the final step of the meta-labeling pipeline, converting
calibrated probabilities (Layer 3 output) into actionable position sizes using
a bounded, monotonic probability-to-size mapping.

Formula:
    z(p) = clip((p - p_min) / (p_max - p_min), 0, 1)
    size(p) = z(p) ^ gamma

It also computes sizing diagnostics:
    - Edge Monotonicity Test
    - Bet Utilization Efficiency
    - Tail Loss Amplification
    - Net Sortino, Max Drawdown, Calmar-like Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import time

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
        p_min: float = 0.5,
        p_max: float = 0.9,
        gamma: float = 1.2,
        transaction_cost: float = 0.000, # Returns assumed net unless specified
        gate_mode: str = 'p_min',
        gate_quantile: Optional[float] = None,
        gate_top_k: Optional[int] = None,
        gate_top_k_per_day: Optional[int] = None,
        gate_search_q_low: Optional[float] = None,
        gate_search_q_high: Optional[float] = None,
        gate_search_min_range: Optional[float] = None,
        gate_search_max_iter: Optional[int] = None,
        min_trades_reliable: int = 50,
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

        self.gate_mode = str(gate_mode or 'p_min')
        self.gate_quantile = gate_quantile
        self.gate_top_k = gate_top_k
        self.gate_top_k_per_day = gate_top_k_per_day

        self.gate_search_q_low = gate_search_q_low
        self.gate_search_q_high = gate_search_q_high
        self.gate_search_min_range = gate_search_min_range
        self.gate_search_max_iter = gate_search_max_iter

        try:
            self.min_trades_reliable = int(min_trades_reliable)
        except Exception:
            self.min_trades_reliable = 50
        self.min_trades_reliable = int(max(1, self.min_trades_reliable))

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


    def calculate_sizing(self) -> pd.Series:
        """Apply the Signal-to-Sizing formula."""
        p = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)

        gate_mask, gate_threshold, gate_mode_used = self._compute_gate_mask_and_threshold(p)
        self._gate_mask = gate_mask
        self._gate_threshold = gate_threshold
        self._gate_mode_used = gate_mode_used

        # Use the computed threshold as the effective p_min for sizing.
        p_min_eff = float(gate_threshold)
        p_max_eff = float(self.p_max)
        try:
            p_valid = p[np.isfinite(p)]
            if p_valid.size > 0:
                p_max_emp = float(np.max(p_valid))
                if (not np.isfinite(p_max_eff)) or (p_max_eff <= p_min_eff + 1e-9):
                    p_max_eff = p_max_emp
        except Exception:
            pass

        # 1) Conviction scaler (monotonic above threshold)
        denom = float(p_max_eff) - float(p_min_eff)
        if (not np.isfinite(denom)) or denom < 1e-6:
            denom = 1e-6
        z_p = np.clip((p - float(p_min_eff)) / float(denom), 0.0, 1.0)

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

        if mode == 'quantile':
            q = self.gate_quantile
            try:
                q = float(q) if q is not None else 0.99
            except Exception:
                q = 0.99
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
            try:
                q = float(self.gate_search_q_high) if self.gate_search_q_high is not None else None
            except Exception:
                q = None
            if q is None:
                try:
                    q = float(self.gate_quantile) if self.gate_quantile is not None else 0.99
                except Exception:
                    q = 0.99
            q = float(np.clip(q, 0.0, 1.0))
            p_valid = p[finite]
            thr = float(np.quantile(p_valid, q)) if p_valid.size > 0 else float(self.p_min)
            gate = finite & (p >= thr)
            return gate, thr, 'quantile'

        if mode == 'top_k':
            k = self.gate_top_k
            try:
                k = int(k) if k is not None else 0
            except Exception:
                k = 0
            p_valid_idx = np.where(finite)[0]
            if k <= 0 or p_valid_idx.size == 0:
                thr = float(self.p_min)
                return (finite & (p >= thr)), thr, 'p_min'
            k = int(min(k, int(p_valid_idx.size)))
            # pick top-k among finite
            order = np.argsort(p[p_valid_idx])
            keep_idx = p_valid_idx[order[-k:]]
            gate = np.zeros(n, dtype=bool)
            gate[keep_idx] = True
            thr = float(np.min(p[keep_idx])) if keep_idx.size > 0 else float(self.p_min)
            return gate, thr, 'top_k'

        if mode == 'top_k_per_day':
            k = self.gate_top_k_per_day
            try:
                k = int(k) if k is not None else 0
            except Exception:
                k = 0
            if k <= 0:
                thr = float(self.p_min)
                return (finite & (p >= thr)), thr, 'p_min'

            try:
                idx = pd.DatetimeIndex(self.df.index)
            except Exception:
                thr = float(self.p_min)
                return (finite & (p >= thr)), thr, 'p_min'

            gate = np.zeros(n, dtype=bool)
            days = idx.normalize()
            for day in pd.Index(days[finite]).unique():
                day_mask = (days == day) & finite
                day_idx = np.where(day_mask)[0]
                if day_idx.size == 0:
                    continue
                kk = int(min(k, int(day_idx.size)))
                order = np.argsort(p[day_idx])
                keep_idx = day_idx[order[-kk:]]
                gate[keep_idx] = True
            thr = float(np.min(p[gate])) if np.any(gate) else float(self.p_min)
            return gate, thr, 'top_k_per_day'

        # Default: fixed p_min
        thr = float(self.p_min)
        gate = finite & (p >= thr)
        return gate, thr, 'p_min'


    def get_gate_index(self) -> pd.Index:
        try:
            if self._gate_mask is None:
                _ = self.calculate_sizing()
            if self._gate_mask is None:
                return pd.Index([])
            return self.df.index[np.asarray(self._gate_mask, dtype=bool)]
        except Exception:
            return pd.Index([])

    def run_backtest(self) -> Dict[str, Any]:
        """
        Executes the backtest using computed sizes and generates metrics.
        """
        tprint_info(">>> Running Layer 5 Backtest & Diagnostics...")

        t0 = time.perf_counter()

        # 1. Compute Sizes
        sizes = self.calculate_sizing()
        t_sizing = time.perf_counter()

        sizes_np = sizes.to_numpy(dtype=float, copy=False)
        self.df['layer5_size'] = sizes_np

        # 2. Compute Sized Returns (PnL)
        # Assume realized_return is what we get if we bet size 1.
        raw_rets = pd.to_numeric(self.df[self.return_col], errors='coerce').to_numpy(dtype=float, copy=False)
        net_rets = raw_rets - self.transaction_cost

        # Vectorized PnL
        pnl = sizes_np * net_rets
        self.df['layer5_pnl'] = pnl

        # 3. Calculate Metrics
        metrics = {}

        try:
            p_raw = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)
            p_mask = np.isfinite(p_raw)
            p_valid = p_raw[p_mask]
            metrics['Prob Mean'] = float(np.mean(p_valid)) if p_valid.size > 0 else float('nan')
            metrics['Prob Std'] = float(np.std(p_valid)) if p_valid.size > 0 else float('nan')
            metrics['Prob Q50'] = float(np.quantile(p_valid, 0.50)) if p_valid.size > 0 else float('nan')
            metrics['Prob Q90'] = float(np.quantile(p_valid, 0.90)) if p_valid.size > 0 else float('nan')
            metrics['Prob Q99'] = float(np.quantile(p_valid, 0.99)) if p_valid.size > 0 else float('nan')
            try:
                gate_mask = np.asarray(self._gate_mask, dtype=bool) if self._gate_mask is not None else None
                gate_count = int(np.sum(gate_mask)) if gate_mask is not None else 0
            except Exception:
                gate_count = 0

            gate_thr = float(self._gate_threshold) if self._gate_threshold is not None else float('nan')
            metrics['Configured p_min'] = float(self.p_min)
            metrics['Effective Gate Threshold'] = float(gate_thr) if np.isfinite(gate_thr) else float('nan')
            metrics['Prob>=p_min Count'] = int(np.sum(p_valid >= float(self.p_min))) if p_valid.size > 0 else 0
            metrics['Prob>=gate_threshold Count'] = int(np.sum(p_valid >= float(gate_thr))) if p_valid.size > 0 and np.isfinite(gate_thr) else 0
            metrics['Gate Mode'] = str(self._gate_mode_used or self.gate_mode)
            metrics['Gate Threshold'] = float(gate_thr) if np.isfinite(gate_thr) else float('nan')
            metrics['Gate Count'] = int(gate_count)
        except Exception:
            pass

        # -- Performance --
        total_pnl = np.sum(pnl)
        trade_mask = sizes_np > 1e-4
        n_trades = int(np.sum(trade_mask))
        avg_pnl = float(np.mean(pnl[trade_mask])) if n_trades > 0 else 0.0

        try:
            pnl_arr = np.asarray(pnl, dtype=float)
            pnl_arr = np.where(np.isfinite(pnl_arr), pnl_arr, 0.0)
            pnl_arr = np.clip(pnl_arr, -0.999999, None)

            idx = None
            try:
                idx = pd.DatetimeIndex(self.df.index)
            except Exception:
                idx = None

            if idx is not None and int(len(idx)) == int(len(pnl_arr)):
                pnl_series = pd.Series(pnl_arr, index=idx)
                daily_returns = pnl_series.groupby(idx.normalize()).apply(
                    lambda x: float(np.prod(1.0 + x.to_numpy(dtype=float, copy=False)) - 1.0)
                )
                daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
                daily_down = daily_returns[daily_returns < 0.0]
                downside_std = float(np.sqrt(np.mean(np.square(daily_down)))) if daily_down.size > 0 else 1e-6
                sortino = float(np.mean(daily_returns) / downside_std * np.sqrt(365.0)) if downside_std > 1e-9 else 0.0
            else:
                downside_returns = pnl_arr[pnl_arr < 0.0]
                downside_std = float(np.sqrt(np.mean(np.square(downside_returns)))) if downside_returns.size > 0 else 1e-6
                sortino = float(np.mean(pnl_arr) / downside_std) if downside_std > 1e-9 else 0.0
        except Exception:
            sortino = 0.0

        metrics['Net Sortino'] = float(sortino)
        metrics['Total PnL'] = float(total_pnl)
        metrics['Avg Trade PnL'] = float(avg_pnl)
        metrics['Trade Count'] = int(n_trades)

        try:
            min_trades = int(getattr(self, 'min_trades_reliable', 50))
        except Exception:
            min_trades = 50
        min_trades = int(max(1, min_trades))
        trades_reliable = bool(int(n_trades) >= int(min_trades))
        metrics['Min Trades Reliable'] = int(min_trades)
        metrics['Trades Reliable'] = bool(trades_reliable)
        if not trades_reliable:
            metrics['Reliability Flag'] = 'UNRELIABLE_TOO_FEW_TRADES'

        try:
            pnl_arr = np.asarray(pnl, dtype=float)
            pnl_arr = np.where(np.isfinite(pnl_arr), pnl_arr, 0.0)
            pnl_arr = np.clip(pnl_arr, -0.999999, None)
            equity = np.cumprod(1.0 + pnl_arr)
            running_max = np.maximum.accumulate(equity)
            dd = 1.0 - (equity / (running_max + 1e-12))
            max_dd = float(np.max(dd)) if dd.size > 0 else 0.0
            total_return = float(equity[-1] - 1.0) if equity.size > 0 else 0.0
        except Exception:
            max_dd = 0.0
            total_return = float('nan')

        metrics['Maximum Drawdown'] = float(max_dd)
        metrics['Total Return'] = float(total_return) if np.isfinite(total_return) else float('nan')

        calmar = float(total_return / max_dd) if np.isfinite(total_return) and max_dd > 1e-9 else 0.0
        metrics['Return / Drawdown Ratio'] = float(calmar)

        # -- Diagnostics --

        # A. Edge Monotonicity Test
        mono_res = self._check_edge_monotonicity(pnl)
        t_mono = time.perf_counter()
        metrics['Edge Monotonicity'] = mono_res

        high_conviction_mask = sizes_np > 0.7
        try:
            active = np.asarray(trade_mask, dtype=bool)
            high = np.asarray(high_conviction_mask, dtype=bool) & active
            pnl_active = np.asarray(pnl, dtype=float)[active]
            pnl_high = np.asarray(pnl, dtype=float)[high]
            pnl_active = pnl_active[np.isfinite(pnl_active)]
            pnl_high = pnl_high[np.isfinite(pnl_high)]
            gp_total = float(np.sum(pnl_active[pnl_active > 0.0])) if pnl_active.size > 0 else 0.0
            gp_high = float(np.sum(pnl_high[pnl_high > 0.0])) if pnl_high.size > 0 else 0.0
            util_eff = float(gp_high / (gp_total + 1e-12)) if gp_total > 1e-12 else 0.0
        except Exception:
            util_eff = 0.0
        metrics['Bet Utilization Efficiency'] = float(util_eff)

        try:
            flat_arr = np.asarray(net_rets, dtype=float)
            flat_arr = np.where(np.isfinite(flat_arr), flat_arr, 0.0)
            flat_arr = np.clip(flat_arr, -0.999999, None)
            flat_eq = np.cumprod(1.0 + flat_arr)
            flat_peak = np.maximum.accumulate(flat_eq)
            flat_dd = 1.0 - (flat_eq / (flat_peak + 1e-12))
            flat_max_dd = float(np.max(flat_dd)) if flat_dd.size > 0 else 0.0
        except Exception:
            flat_max_dd = 0.0

        tla_ratio = float(max_dd / flat_max_dd) if flat_max_dd > 1e-9 else 1.0
        metrics['Tail Loss Amplification'] = float(tla_ratio)

        metrics['Parameters'] = {
            'gamma': self.gamma,
            'p_min': self.p_min,
            'p_max': self.p_max
        }

        try:
            metrics['Parameters']['gate_mode'] = str(self._gate_mode_used or self.gate_mode)
            metrics['Parameters']['gate_quantile'] = self.gate_quantile
            metrics['Parameters']['gate_top_k'] = self.gate_top_k
            metrics['Parameters']['gate_top_k_per_day'] = self.gate_top_k_per_day
            metrics['Parameters']['gate_search_q_low'] = self.gate_search_q_low
            metrics['Parameters']['gate_search_q_high'] = self.gate_search_q_high
            metrics['Parameters']['gate_search_min_range'] = self.gate_search_min_range
            metrics['Parameters']['gate_search_max_iter'] = self.gate_search_max_iter
            metrics['Parameters']['gate_threshold'] = float(self._gate_threshold) if self._gate_threshold is not None else None
        except Exception:
            pass

        metrics['Runtime'] = {
            'n_rows': int(self.df.shape[0]),
            'sizing_ms': float((t_sizing - t0) * 1000.0),
            'edge_monotonicity_ms': float((t_mono - t_sizing) * 1000.0),
            'total_ms': float((t_mono - t0) * 1000.0),
        }

        try:
            pnl_arr = np.asarray(pnl, dtype=float)
            size_arr = np.asarray(sizes_np, dtype=float)
            trade_mask = np.asarray(trade_mask, dtype=bool)

            n_trades_local = int(np.sum(trade_mask))
            exposure = float(np.mean(trade_mask.astype(float))) if int(len(trade_mask)) > 0 else float('nan')

            traded_pnl = pnl_arr[trade_mask] if n_trades_local > 0 else np.asarray([], dtype=float)
            traded_pnl = traded_pnl[np.isfinite(traded_pnl)]

            wins = traded_pnl[traded_pnl > 0.0]
            losses = traded_pnl[traded_pnl < 0.0]
            gross_profit = float(np.sum(wins)) if wins.size > 0 else 0.0
            gross_loss = float(-np.sum(losses)) if losses.size > 0 else 0.0
            win_rate = float(wins.size / traded_pnl.size) if traded_pnl.size > 0 else float('nan')
            avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
            avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
            payoff_ratio = float(avg_win / (-avg_loss + 1e-12)) if (avg_win > 0.0 and avg_loss < 0.0) else float('nan')
            profit_factor = float(gross_profit / (gross_loss + 1e-12)) if gross_loss > 0.0 else float('nan')
            expectancy = float(np.mean(traded_pnl)) if traded_pnl.size > 0 else float('nan')

            pnl_mean = float(np.mean(traded_pnl)) if traded_pnl.size > 0 else float('nan')
            pnl_std = float(np.std(traded_pnl)) if traded_pnl.size > 1 else float('nan')
            sharpe_like = float(pnl_mean / (pnl_std + 1e-12) * np.sqrt(float(traded_pnl.size))) if traded_pnl.size > 1 and np.isfinite(pnl_std) else float('nan')

            size_trade = size_arr[trade_mask] if n_trades_local > 0 else np.asarray([], dtype=float)
            size_trade = size_trade[np.isfinite(size_trade)]
            avg_size = float(np.mean(size_trade)) if size_trade.size > 0 else float('nan')
            med_size = float(np.median(size_trade)) if size_trade.size > 0 else float('nan')

            q10 = float(np.quantile(traded_pnl, 0.10)) if traded_pnl.size > 0 else float('nan')
            q50 = float(np.quantile(traded_pnl, 0.50)) if traded_pnl.size > 0 else float('nan')
            q90 = float(np.quantile(traded_pnl, 0.90)) if traded_pnl.size > 0 else float('nan')

            metrics['Win Rate'] = float(win_rate)
            metrics['Gross Profit'] = float(gross_profit)
            metrics['Gross Loss'] = float(gross_loss)
            metrics['Profit Factor'] = float(profit_factor)
            metrics['Payoff Ratio'] = float(payoff_ratio)
            metrics['Expectancy'] = float(expectancy)
            metrics['Exposure'] = float(exposure)
            metrics['Average Size'] = float(avg_size)
            metrics['Median Size'] = float(med_size)
            metrics['PnL Mean (Traded)'] = float(pnl_mean)
            metrics['PnL Std (Traded)'] = float(pnl_std)
            metrics['Sharpe-like (Traded)'] = float(sharpe_like)
            metrics['PnL Q10 (Traded)'] = float(q10)
            metrics['PnL Q50 (Traded)'] = float(q50)
            metrics['PnL Q90 (Traded)'] = float(q90)
        except Exception:
            pass

        try:
            outcomes_dir = Path('outcomes')
            outcomes_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            outcomes_dir = Path('outcomes')

        try:
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        except Exception:
            ts = 'unknown'

        try:
            symbol = ''
            timeframe = ''
            if isinstance(self.df, pd.DataFrame):
                if 'symbol' in self.df.columns:
                    symbol = str(self.df['symbol'].iloc[0])
                if 'timeframe' in self.df.columns:
                    timeframe = str(self.df['timeframe'].iloc[0])
        except Exception:
            symbol = ''
            timeframe = ''

        try:
            flat = {
                'timestamp': ts,
                'symbol': symbol,
                'timeframe': timeframe,
                'p_col': str(self.p_col),
                'return_col': str(self.return_col),
                'vol_col': str(self.vol_col),
            }

            if isinstance(metrics, dict):
                # export all scalar metrics
                for k, v in metrics.items():
                    if k in {'Parameters', 'Runtime', 'Edge Monotonicity'}:
                        continue
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        flat[str(k)] = v

                params = metrics.get('Parameters')
                if isinstance(params, dict):
                    for k, v in params.items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            flat[f'param_{k}'] = v

                rt = metrics.get('Runtime')
                if isinstance(rt, dict):
                    for k, v in rt.items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            flat[f'runtime_{k}'] = v

                em = metrics.get('Edge Monotonicity')
                if isinstance(em, dict):
                    corr_v = em.get('correlation')
                    if isinstance(corr_v, (int, float)) or corr_v is None:
                        flat['edge_monotonicity_correlation'] = corr_v

            pd.DataFrame([flat]).to_csv(
                outcomes_dir / f"layer5_metrics_{symbol}_{timeframe}_{ts}.csv",
                index=False,
            )
        except Exception:
            pass

        try:
            export_cols = []
            for c in [self.p_col, self.target_col, self.return_col, self.vol_col, 'layer5_size', 'layer5_pnl']:
                if c in self.df.columns and c not in export_cols:
                    export_cols.append(c)
            sized_df = self.df[export_cols].copy() if export_cols else self.df.copy()
            sized_df.to_csv(
                outcomes_dir / f"layer5_sized_events_{symbol}_{timeframe}_{ts}.csv",
                index=True,
            )
        except Exception:
            pass

        try:
            md_path = outcomes_dir / f"layer5_report_{symbol}_{timeframe}_{ts}.md"
            lines = [
                "# Layer5 Report\n",
                f"- timestamp: {ts}\n",
                f"- symbol: {symbol}\n",
                f"- timeframe: {timeframe}\n",
                f"- n_rows: {int(self.df.shape[0])}\n",
                f"- p_col: {self.p_col}\n",
                f"- return_col: {self.return_col}\n",
                "\n## Summary\n",
            ]

            try:
                tr_cnt = int(metrics.get('Trade Count')) if isinstance(metrics, dict) else 0
            except Exception:
                tr_cnt = 0
            try:
                tr_rel = bool(metrics.get('Trades Reliable')) if isinstance(metrics, dict) else False
            except Exception:
                tr_rel = False

            if isinstance(metrics, dict):
                for k in ['Trades Reliable', 'Min Trades Reliable', 'Trade Count', 'Total PnL', 'Avg Trade PnL', 'Maximum Drawdown', 'Total Return']:
                    if k in metrics:
                        lines.append(f"- {k}: {metrics.get(k)}\n")
                if not tr_rel:
                    lines.append("- reliability_note: PF/Sortino/Sharpe-like are suppressed in Summary due to low trade count\n")

            lines.append("\n## Metrics\n")

            if isinstance(metrics, dict):
                # Scalars first
                scalar_items = []
                for k, v in metrics.items():
                    if k in {'Parameters', 'Runtime'}:
                        continue
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        scalar_items.append((str(k), v))
                for k, v in sorted(scalar_items, key=lambda kv: kv[0]):
                    if (not tr_rel) and (k in {'Profit Factor', 'Net Sortino', 'Sharpe-like (Traded)'}):
                        continue
                    lines.append(f"- {k}: {v}\n")

                params = metrics.get('Parameters')
                if isinstance(params, dict):
                    lines.append("\n## Parameters\n")
                    for k in sorted(params.keys(), key=lambda x: str(x)):
                        lines.append(f"- {k}: {params.get(k)}\n")

                rt = metrics.get('Runtime')
                if isinstance(rt, dict):
                    lines.append("\n## Runtime\n")
                    for k in sorted(rt.keys(), key=lambda x: str(x)):
                        lines.append(f"- {k}: {rt.get(k)}\n")
            md_path.write_text(''.join(lines))
        except Exception:
            pass

        return metrics

    def _check_edge_monotonicity(self, pnl: np.ndarray) -> Dict[str, Any]:
        """
        Check if realized sharpe/return increases with probability deciles.
        """
        prob = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)
        pnl = np.asarray(pnl, dtype=float)
        mask = np.isfinite(prob) & np.isfinite(pnl)

        bins = np.arange(0.0, 1.01, 0.1)
        n_bins = int(len(bins) - 1)
        cut_idx = np.digitize(prob[mask], bins, right=True)
        valid = (cut_idx > 0) & (cut_idx < len(bins))
        bin_idx = cut_idx[valid] - 1
        pnl_v = pnl[mask][valid]

        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        sums = np.bincount(bin_idx, weights=pnl_v, minlength=n_bins).astype(float)
        sums_sq = np.bincount(bin_idx, weights=pnl_v * pnl_v, minlength=n_bins).astype(float)

        mean = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
        m2 = np.divide(sums_sq, counts, out=np.full(n_bins, np.nan), where=counts > 0)
        var = m2 - mean * mean
        var = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)
        std = np.sqrt(var)

        # Guardrail: ignore tiny bins (they can explode sharpe)
        try:
            total_n = int(np.sum(counts))
        except Exception:
            total_n = 0
        min_count = 10
        if total_n > 0:
            min_count = int(max(10, round(0.02 * float(total_n))))

        sharpe = np.full(n_bins, np.nan, dtype=float)
        ok = (counts >= float(min_count)) & np.isfinite(mean) & np.isfinite(std) & (std > 1e-9)
        sharpe[ok] = mean[ok] / std[ok]
        sharpe = np.clip(sharpe, -50.0, 50.0)

        intervals = pd.IntervalIndex.from_breaks(bins, closed='right')
        stats = pd.DataFrame(
            {
                'mean': mean,
                'std': std,
                'count': counts,
                'sharpe': sharpe,
            },
            index=intervals,
        )

        # Check monotonicity in the upper half (0.5+)
        # We look at bins > 0.5
        # The index contains Intervals. We extract the left bound of each interval in the index directly.
        left_bounds = stats.index.left.to_numpy(dtype=float, copy=False)
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
        self.df.to_csv(output_dir / "layer5_sized_events.csv", index=True)
