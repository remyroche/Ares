"""Layer 5 — Signal-to-Sizing bridge: Trading / Portfolio Construction.

This module implements the final step of the meta-labeling pipeline, converting
calibrated probabilities (Layer 4 output) into actionable position sizes.

It performs sizing based on Layer 4's optimized assumptions and generates
performance metrics.

Sizing Logic (Aligned with Layer 4 Optimization):
    Size = 0 if p < 0.5
    Size = ((p - 0.5) / 0.5) ^ 2 if p >= 0.5
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
    Strictly follows Layer 4's sizing assumptions for consistency.
    """

    def __init__(
        self,
        oof_df: pd.DataFrame,
        p_col: str = 'layer4_prob', # Defaults to Layer 4 output
        target_col: str = 'target',
        return_col: str = 'realized_return',
        vol_col: str = 'volatility_1d',
        transaction_cost: float = 0.000,
        # Sizing parameters (Fixed to match Layer 4 assumptions)
        sizing_threshold: float = 0.5,
        sizing_gamma: float = 2.0,
        min_trades_reliable: int = 50,
        # Legacy/Compatibility args (ignored but accepted)
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        gamma: Optional[float] = None,
        gate_mode: Optional[str] = None,
        gate_quantile: Optional[float] = None,
        gate_top_k: Optional[int] = None,
        gate_top_k_per_day: Optional[int] = None,
        gate_search_q_low: Optional[float] = None,
        gate_search_q_high: Optional[float] = None,
        gate_search_min_range: Optional[float] = None,
        gate_search_max_iter: Optional[int] = None,
        allow_dynamic_p_max: bool = False,
    ):
        self.df = oof_df.copy()
        self.p_col = p_col
        self.target_col = target_col
        self.return_col = return_col
        self.vol_col = vol_col
        self.transaction_cost = transaction_cost

        # Sizing params - prioritize init args if consistent, otherwise default to fixed logic
        # Ideally we ignore legacy args to enforce the new logic, but if p_min/gamma are passed
        # explicitly by caller intending to tune, we might respect them IF they match our new structure.
        # Layer 4 uses threshold=0.5, gamma=2.0.
        # If caller passes p_min, treat it as sizing_threshold.

        if p_min is not None:
            self.sizing_threshold = float(p_min)
        else:
            self.sizing_threshold = sizing_threshold

        if gamma is not None:
            self.sizing_gamma = float(gamma)
        else:
            self.sizing_gamma = sizing_gamma

        self.min_trades_reliable = int(max(1, min_trades_reliable)) if min_trades_reliable else 50

        # Validate columns
        missing = [c for c in [p_col, return_col] if c not in self.df.columns]
        if missing:
             raise ValueError(f"Missing required columns for Layer 5: {missing}. Ensure LabelBasedLayer4 has run successfully.")

    def calculate_sizing(self) -> pd.Series:
        """Apply the Signal-to-Sizing formula (Quadratic Scaling)."""
        if self.df.empty:
            return pd.Series(dtype=float)

        p = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)

        # Sizing Logic:
        # Size = 0 if p < threshold
        # Size = ((p - threshold) / (1 - threshold)) ^ gamma if p >= threshold

        p_clipped = np.clip(p, 0.0, 1.0)
        denom = 1.0 - self.sizing_threshold
        if denom < 1e-6:
            denom = 1e-6

        scaled = (p_clipped - self.sizing_threshold) / denom
        scaled = np.clip(scaled, 0.0, 1.0)

        size = np.power(scaled, self.sizing_gamma)

        # Zero out below threshold explicitly
        size = np.where(p_clipped < self.sizing_threshold, 0.0, size)

        # Final safety clip
        size = np.clip(size, 0.0, 1.0)

        return pd.Series(size, index=self.df.index)

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

        metrics['Runtime'] = {
            'n_rows': int(self.df.shape[0]),
            'total_ms': float((time.perf_counter() - t0) * 1000.0),
        }

        # 4. Generate Report and Save Artifacts
        self._save_artifacts_to_disk(metrics)

        return metrics

    def _compute_metrics(self, pnl: np.ndarray, sizes: np.ndarray, trade_mask: np.ndarray, net_rets: np.ndarray) -> Dict[str, Any]:
        """Compute scalar metrics."""
        metrics = {}

        n_trades = int(np.sum(trade_mask))
        total_pnl = np.sum(pnl)
        avg_pnl = float(np.mean(pnl[trade_mask])) if n_trades > 0 else 0.0

        metrics['Total PnL'] = float(total_pnl)
        metrics['Avg Trade PnL'] = float(avg_pnl)
        metrics['Trade Count'] = int(n_trades)
        metrics['Turnover Estimate'] = float(np.sum(sizes) * 2.0)

        # Sortino
        pnl_arr = np.nan_to_num(pnl, nan=0.0)
        downside = np.minimum(0, pnl_arr)
        downside_std = float(np.sqrt(np.mean(np.square(downside)))) if downside.size > 0 else 1e-6
        sortino = float(np.mean(pnl_arr) / downside_std * np.sqrt(365.0)) if downside_std > 1e-9 else 0.0 # Heuristic annualization
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

        # Classification (on underlying probs)
        p_raw = pd.to_numeric(self.df[self.p_col], errors='coerce').to_numpy(dtype=float, copy=False)
        y_true = (net_rets > 0).astype(int)
        mask = np.isfinite(p_raw) & np.isfinite(net_rets)

        if roc_auc_score is not None and mask.sum() > 10:
             try:
                 metrics['AUC'] = float(roc_auc_score(y_true[mask], p_raw[mask]))
             except Exception:
                 metrics['AUC'] = 0.5
        else:
             metrics['AUC'] = 0.5

        return metrics

    def _save_artifacts_to_disk(self, metrics: Dict[str, Any]):
        """Save CSV metrics and report."""
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        symbol = str(self.df['symbol'].iloc[0]) if 'symbol' in self.df.columns else ''
        timeframe = str(self.df['timeframe'].iloc[0]) if 'timeframe' in self.df.columns else ''
        suffix = f"{symbol}_{timeframe}_{ts}".strip('_')

        # CSV
        flat = metrics.copy()
        if 'Runtime' in flat: del flat['Runtime']
        pd.DataFrame([flat]).to_csv(outcomes_dir / f"layer5_metrics_{suffix}.csv", index=False)

        # Report
        md_path = outcomes_dir / f"layer5_report_{suffix}.md"
        lines = [
            "# Layer5 Report\n",
            f"- timestamp: {ts}\n",
            "\n## Metrics\n"
        ]
        for k, v in metrics.items():
            if k == 'Runtime': continue
            lines.append(f"- {k}: {v}\n")

        md_path.write_text(''.join(lines))
