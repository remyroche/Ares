"""
Analyst Base Backtest Step.

This step performs a simple PnL-based backtest using analyst predictions
(ML-scored historical data) and computes true Sharpe/Sortino/max drawdown
on the resulting returns series. It saves a standalone Markdown report
in the outcomes/ directory with a filename that includes "base_analyst_".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error
from src.utils.ml_common.trading_grid_backtester import (
    run_simple_long_grid_backtest,
    run_simple_short_grid_backtest,
)


logger = logging.getLogger(__name__)


class AnalystBaseBacktestStep(BaseStep):
    """Simple PnL-based backtest for analyst base models using OOS predictions."""

    def __init__(self, step_name: str = "analyst_base_backtest"):
        super().__init__(step_name)
        self.logger = system_logger.getChild("AnalystBaseBacktest")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bars_per_year_from_timeframe(timeframe: str) -> float:
        """Approximate number of bars per year for a given timeframe string.

        Supports formats like '1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'.
        """
        tf = str(timeframe).lower().strip()

        try:
            if tf.endswith("m") and tf[:-1].isdigit():
                minutes = int(tf[:-1])
                if minutes <= 0:
                    return 365.0
                bars_per_day = (24 * 60) / minutes
                return bars_per_day * 365.0
            if tf.endswith("h") and tf[:-1].isdigit():
                hours = int(tf[:-1])
                if hours <= 0:
                    return 365.0
                bars_per_day = 24 / hours
                return bars_per_day * 365.0
            if tf.endswith("d") and tf[:-1].isdigit():
                days = int(tf[:-1])
                if days <= 0:
                    return 365.0
                bars_per_day = 1.0 / days
                return bars_per_day * 365.0
            if tf.endswith("w") and tf[:-1].isdigit():
                weeks = int(tf[:-1])
                if weeks <= 0:
                    return 52.0
                bars_per_week = 1.0 / weeks
                return bars_per_week * 52.0
        except Exception:
            # Fallback below
            pass

        # Fallback: assume daily
        return 365.0

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analyst base backtest using OOS predictions.

        Args:
            config: Configuration dictionary with at least:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe string (e.g., '15m')
                - direction: 'long', 'short', or 'both'

        Returns:
            Dict with success flag, artifacts, metrics, and optional error.
        """
        symbol = config.get("symbol", "UNKNOWN")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        execution_mode = config.get("execution_mode", "light")

        tprint(
            f"🧪 Starting analyst base backtest for {symbol} {timeframe} {direction} (mode={execution_mode})",
            "INFO",
        )

        # Ensure context matches analyst training setup so artifacts line up
        self.set_context(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model="analyst",
            execution_mode=execution_mode,
        )

        try:
            # ------------------------------------------------------------------
            # 1) Load price data via BaseStep helper (ensures consistent source)
            # ------------------------------------------------------------------
            price_data, source = self.load_market_data_or_fail(
                config,
                pipeline_state={},
                allow_config_override=False,
            )

            if price_data is None or not isinstance(price_data, pd.DataFrame) or price_data.empty:
                raise ValueError("Price data not available or empty for backtest")

            price_df = price_data.copy()
            price_df = self._normalize_datetime_index(price_df, "price data")
            price_df = price_df.sort_index()
            if "close" not in price_df.columns:
                raise ValueError("Price data is missing 'close' column")

            # ------------------------------------------------------------------
            # 2) Load ML-scored OOS data from analyst base training
            # ------------------------------------------------------------------
            artifact_name = f"ml_scored_historical_data_analyst_{direction}_oos"
            tprint_info(f"🔎 Loading ML-scored historical data: {artifact_name}")

            ml_scored = self._get_artifact(
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="predictions",
            )

            if ml_scored is None:
                raise ValueError(f"ML-scored artifact '{artifact_name}' not found")

            if not isinstance(ml_scored, pd.DataFrame) or ml_scored.empty:
                raise ValueError(f"ML-scored artifact '{artifact_name}' is empty or not a DataFrame")

            ml_df = ml_scored.copy()
            ml_df = self._normalize_datetime_index(ml_df, "ML-scored data")
            ml_df = ml_df.sort_index()

            # ------------------------------------------------------------------
            # 3) Align indices between price data and ML predictions
            # ------------------------------------------------------------------
            common_index = ml_df.index.intersection(price_df.index)
            if len(common_index) < 50:
                raise ValueError(
                    f"Insufficient overlap between ML data and price data (common samples={len(common_index)})"
                )

            ml_df = ml_df.loc[common_index]
            price_df = price_df.loc[common_index]

            close = price_df["close"].astype(float)
            raw_returns = close.pct_change().fillna(0.0)

            # ------------------------------------------------------------------
            # 4) Extract prediction & optional confidence columns
            # ------------------------------------------------------------------
            pred_candidates = [
                c
                for c in ml_df.columns
                if "pred" in c.lower()
                or "y_hat" in c.lower()
                or "target" in c.lower()
                or "forecast" in c.lower()
            ]

            if pred_candidates:
                pred_col = pred_candidates[0]
            else:
                # Fallback: first numeric column
                numeric_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric prediction column found in ML-scored data")
                pred_col = numeric_cols[0]

            predictions = ml_df[pred_col].astype(float)

            # Confidence column (optional)
            conf_candidates = [c for c in ml_df.columns if "confidence" in c.lower()]
            if conf_candidates:
                confidence = ml_df[conf_candidates[0]].astype(float)
            else:
                confidence = predictions.abs()

            # Normalize confidence to [0, 1]
            max_conf = float(confidence.max()) if len(confidence) > 0 else 0.0
            if max_conf > 1.0:
                confidence = confidence / max_conf
            confidence = confidence.clip(0.0, 1.0)

            # ------------------------------------------------------------------
            # 5) Build position signal (no lookahead: use lagged signal)
            # ------------------------------------------------------------------
            if direction == "long":
                signal = (predictions > 0).astype(float) * confidence
            elif direction == "short":
                signal = (predictions < 0).astype(float) * confidence * -1.0
            else:  # both
                signal = np.sign(predictions).astype(float) * confidence

            # Apply one-bar lag so we don't use current bar's prediction on itself
            position = signal.shift(1).fillna(0.0)

            strategy_returns = position * raw_returns

            # Optional gate overlay using gate_decision and gate_score from GateTrainingStep
            gate_mask = None
            gated_position = None
            gated_strategy_returns = None
            gate_prob = None
            prob_gated_position = None
            prob_gated_strategy_returns = None
            try:
                gate_artifact_name = f"gate_oof_predictions_{symbol}"
                tprint_info(f"🔎 Attempting to load gate decisions: {gate_artifact_name}")

                gate_oof = self._get_artifact(
                    artifact_name=gate_artifact_name,
                    artifact_type="data",
                    data_category="predictions",
                )

                if gate_oof is not None and isinstance(gate_oof, pd.DataFrame):
                    if "gate_decision" in gate_oof.columns:
                        gate_series = gate_oof["gate_decision"].astype(float)
                        gate_series = gate_series.sort_index().reindex(ml_df.index).fillna(1.0)
                        gate_mask = gate_series.clip(0.0, 1.0)

                    if "gate_score" in gate_oof.columns:
                        gate_prob_series = gate_oof["gate_score"].astype(float)
                        gate_prob_series = gate_prob_series.sort_index().reindex(ml_df.index).fillna(1.0)
                        gate_prob = gate_prob_series.clip(0.0, 1.0)
            except Exception:
                gate_mask = None
                gate_prob = None

            if gate_mask is not None:
                gated_position = position * gate_mask
                gated_strategy_returns = gated_position * raw_returns

            if gate_prob is not None:
                prob_gated_position = position * gate_prob
                prob_gated_strategy_returns = prob_gated_position * raw_returns

            # ------------------------------------------------------------------
            # 6) Compute performance metrics (true returns-based Sharpe/Sortino)
            # ------------------------------------------------------------------
            n_bars = int(len(strategy_returns))
            total_return = float((1.0 + strategy_returns).prod() - 1.0) if n_bars > 0 else 0.0

            bars_per_year = self._bars_per_year_from_timeframe(timeframe)
            mean_ret = float(strategy_returns.mean()) if n_bars > 0 else 0.0
            vol = float(strategy_returns.std()) if n_bars > 1 else 0.0

            if n_bars > 0:
                annualized_return = float((1.0 + mean_ret) ** bars_per_year - 1.0)
            else:
                annualized_return = 0.0

            annualized_vol = float(vol * np.sqrt(bars_per_year)) if vol > 0 else 0.0

            risk_free = 0.0
            sharpe = float((annualized_return - risk_free) / annualized_vol) if annualized_vol > 0 else 0.0

            downside = strategy_returns[strategy_returns < 0]
            if len(downside) > 1:
                downside_vol = float(downside.std() * np.sqrt(bars_per_year))
            else:
                downside_vol = 0.0
            sortino = float((annualized_return - risk_free) / downside_vol) if downside_vol > 0 else sharpe

            equity = (1.0 + strategy_returns).cumprod()
            running_max = equity.cummax()
            drawdown = equity / running_max - 1.0
            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            # Simple trade stats: count discrete entries (0->1 transitions)
            in_position = (position != 0.0).astype(int)
            pos_changes = in_position.diff().fillna(0)
            entries = pos_changes == 1
            approx_trades = int(entries.sum())

            positive = strategy_returns[strategy_returns > 0]
            negative = strategy_returns[strategy_returns < 0]
            n_pos = int(len(positive))
            n_neg = int(len(negative))
            n_nonzero = n_pos + n_neg
            win_rate = float(n_pos / n_nonzero) if n_nonzero > 0 else 0.0
            avg_win = float(positive.mean()) if n_pos > 0 else 0.0
            avg_loss = float(negative.mean()) if n_neg > 0 else 0.0
            gross_profit = float(positive.sum()) if n_pos > 0 else 0.0
            gross_loss = float(-negative.sum()) if n_neg > 0 else 0.0
            profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

            metrics: Dict[str, Any] = {
                "bars": n_bars,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_vol,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "approx_trades": approx_trades,
            }

            # Compute gated bar-level metrics when gate decisions are available
            if gated_strategy_returns is not None and gated_position is not None:
                g_n_bars = int(len(gated_strategy_returns))
                if g_n_bars > 0:
                    g_total_return = float((1.0 + gated_strategy_returns).prod() - 1.0)
                else:
                    g_total_return = 0.0

                g_mean_ret = float(gated_strategy_returns.mean()) if g_n_bars > 0 else 0.0
                g_vol = float(gated_strategy_returns.std()) if g_n_bars > 1 else 0.0

                if g_n_bars > 0:
                    g_annualized_return = float((1.0 + g_mean_ret) ** bars_per_year - 1.0)
                else:
                    g_annualized_return = 0.0

                g_annualized_vol = float(g_vol * np.sqrt(bars_per_year)) if g_vol > 0 else 0.0

                g_risk_free = 0.0
                g_sharpe = float((g_annualized_return - g_risk_free) / g_annualized_vol) if g_annualized_vol > 0 else 0.0

                g_downside = gated_strategy_returns[gated_strategy_returns < 0]
                if len(g_downside) > 1:
                    g_downside_vol = float(g_downside.std() * np.sqrt(bars_per_year))
                else:
                    g_downside_vol = 0.0
                g_sortino = float((g_annualized_return - g_risk_free) / g_downside_vol) if g_downside_vol > 0 else g_sharpe

                g_equity = (1.0 + gated_strategy_returns).cumprod()
                g_running_max = g_equity.cummax()
                g_drawdown = g_equity / g_running_max - 1.0
                g_max_drawdown = float(g_drawdown.min()) if len(g_drawdown) > 0 else 0.0

                g_in_position = (gated_position != 0.0).astype(int)
                g_pos_changes = g_in_position.diff().fillna(0)
                g_entries = g_pos_changes == 1
                g_approx_trades = int(g_entries.sum())

                g_positive = gated_strategy_returns[gated_strategy_returns > 0]
                g_negative = gated_strategy_returns[gated_strategy_returns < 0]
                g_n_pos = int(len(g_positive))
                g_n_neg = int(len(g_negative))
                g_n_nonzero = g_n_pos + g_n_neg
                g_win_rate_bar = float(g_n_pos / g_n_nonzero) if g_n_nonzero > 0 else 0.0
                g_avg_win = float(g_positive.mean()) if g_n_pos > 0 else 0.0
                g_avg_loss = float(g_negative.mean()) if g_n_neg > 0 else 0.0
                g_gross_profit = float(g_positive.sum()) if g_n_pos > 0 else 0.0
                g_gross_loss = float(-g_negative.sum()) if g_n_neg > 0 else 0.0
                g_profit_factor = float(g_gross_profit / g_gross_loss) if g_gross_loss > 0 else 0.0

                gate_coverage_rate = float((gated_position != 0.0).mean()) if len(gated_position) > 0 else 0.0

                metrics.update(
                    {
                        "gated_total_return": g_total_return,
                        "gated_annualized_return": g_annualized_return,
                        "gated_annualized_volatility": g_annualized_vol,
                        "gated_sharpe_ratio": g_sharpe,
                        "gated_sortino_ratio": g_sortino,
                        "gated_max_drawdown": g_max_drawdown,
                        "gated_bar_win_rate": g_win_rate_bar,
                        "gated_profit_factor": g_profit_factor,
                        "gated_avg_win": g_avg_win,
                        "gated_avg_loss": g_avg_loss,
                        "gated_approx_trades": g_approx_trades,
                        "gate_coverage_rate": gate_coverage_rate,
                    }
                )

                tprint_info(
                    f"🔐 Gate overlay: Sharpe {sharpe:.3f} → {g_sharpe:.3f}, "
                    f"Win-rate {win_rate:.3f} → {g_win_rate_bar:.3f}, "
                    f"Coverage={gate_coverage_rate:.2%}"
                )

            # Compute probability-weighted gate metrics when gate_score is available
            if prob_gated_strategy_returns is not None and prob_gated_position is not None:
                pg_n_bars = int(len(prob_gated_strategy_returns))
                if pg_n_bars > 0:
                    pg_total_return = float((1.0 + prob_gated_strategy_returns).prod() - 1.0)
                else:
                    pg_total_return = 0.0

                pg_mean_ret = float(prob_gated_strategy_returns.mean()) if pg_n_bars > 0 else 0.0
                pg_vol = float(prob_gated_strategy_returns.std()) if pg_n_bars > 1 else 0.0

                if pg_n_bars > 0:
                    pg_annualized_return = float((1.0 + pg_mean_ret) ** bars_per_year - 1.0)
                else:
                    pg_annualized_return = 0.0

                pg_annualized_vol = float(pg_vol * np.sqrt(bars_per_year)) if pg_vol > 0 else 0.0

                pg_risk_free = 0.0
                pg_sharpe = float((pg_annualized_return - pg_risk_free) / pg_annualized_vol) if pg_annualized_vol > 0 else 0.0

                pg_downside = prob_gated_strategy_returns[prob_gated_strategy_returns < 0]
                if len(pg_downside) > 1:
                    pg_downside_vol = float(pg_downside.std() * np.sqrt(bars_per_year))
                else:
                    pg_downside_vol = 0.0
                pg_sortino = float((pg_annualized_return - pg_risk_free) / pg_downside_vol) if pg_downside_vol > 0 else pg_sharpe

                pg_equity = (1.0 + prob_gated_strategy_returns).cumprod()
                pg_running_max = pg_equity.cummax()
                pg_drawdown = pg_equity / pg_running_max - 1.0
                pg_max_drawdown = float(pg_drawdown.min()) if len(pg_drawdown) > 0 else 0.0

                pg_in_position = (prob_gated_position != 0.0).astype(int)
                pg_pos_changes = pg_in_position.diff().fillna(0)
                pg_entries = pg_pos_changes == 1
                pg_approx_trades = int(pg_entries.sum())

                pg_positive = prob_gated_strategy_returns[prob_gated_strategy_returns > 0]
                pg_negative = prob_gated_strategy_returns[prob_gated_strategy_returns < 0]
                pg_n_pos = int(len(pg_positive))
                pg_n_neg = int(len(pg_negative))
                pg_n_nonzero = pg_n_pos + pg_n_neg
                pg_win_rate_bar = float(pg_n_pos / pg_n_nonzero) if pg_n_nonzero > 0 else 0.0
                pg_avg_win = float(pg_positive.mean()) if pg_n_pos > 0 else 0.0
                pg_avg_loss = float(pg_negative.mean()) if pg_n_neg > 0 else 0.0
                pg_gross_profit = float(pg_positive.sum()) if pg_n_pos > 0 else 0.0
                pg_gross_loss = float(-pg_negative.sum()) if pg_n_neg > 0 else 0.0
                pg_profit_factor = float(pg_gross_profit / pg_gross_loss) if pg_gross_loss > 0 else 0.0

                prob_gate_coverage_rate = float((prob_gated_position != 0.0).mean()) if len(prob_gated_position) > 0 else 0.0
                prob_gate_mean_score = float(gate_prob.mean()) if gate_prob is not None and len(gate_prob) > 0 else 0.0

                metrics.update(
                    {
                        "prob_gated_total_return": pg_total_return,
                        "prob_gated_annualized_return": pg_annualized_return,
                        "prob_gated_annualized_volatility": pg_annualized_vol,
                        "prob_gated_sharpe_ratio": pg_sharpe,
                        "prob_gated_sortino_ratio": pg_sortino,
                        "prob_gated_max_drawdown": pg_max_drawdown,
                        "prob_gated_bar_win_rate": pg_win_rate_bar,
                        "prob_gated_profit_factor": pg_profit_factor,
                        "prob_gated_avg_win": pg_avg_win,
                        "prob_gated_avg_loss": pg_avg_loss,
                        "prob_gated_approx_trades": pg_approx_trades,
                        "prob_gate_coverage_rate": prob_gate_coverage_rate,
                        "prob_gate_mean_score": prob_gate_mean_score,
                    }
                )

            # Optional production-like trade metrics using meta-labeling TPSL config
            try:
                va_dir = Path("versioned_artifacts") / f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
                gating_path = va_dir / "meta_gating_config.json"
                if gating_path.exists():
                    with open(gating_path, "r") as f_cfg:
                        gating_config = json.load(f_cfg)

                    meta_gating = gating_config.get("meta_gating", {})
                    triple_cfg = meta_gating.get("triple_barrier", {})

                    profit_thr = float(triple_cfg.get("profit_threshold", 0.0))
                    stop_thr = float(triple_cfg.get("stop_threshold", 0.0))
                    horizon_bars = int(triple_cfg.get("horizon_bars", 0))
                    if horizon_bars <= 0:
                        horizon_bars = 6

                    # Trailing distance in ATR multiples (if enabled during labeling)
                    trail_dist = float(triple_cfg.get("trail_distance_atr", triple_cfg.get("trail_distance", 0.0)))
                    if not np.isfinite(trail_dist):
                        trail_dist = 0.0

                    # ATR lookback window; fall back to 14 if not present
                    trail_atr_window = int(meta_gating.get("trail_atr_window", 14))
                    if trail_atr_window < 2:
                        trail_atr_window = 2

                    tx_cost = float(meta_gating.get("transaction_cost", 0.0))
                    fee_rate = tx_cost if tx_cost > 0.0 else 0.0015

                    # Use entry.prob_threshold for probabilistic gating when available
                    entry_cfg = meta_gating.get("entry", {})
                    gate_prob_threshold = float(entry_cfg.get("prob_threshold", 0.55))
                    if not np.isfinite(gate_prob_threshold) or gate_prob_threshold <= 0.0 or gate_prob_threshold >= 1.0:
                        gate_prob_threshold = 0.55

                    grid_df = None
                    grid_df_gated = None
                    grid_df_prob = None

                    if direction == "short":
                        grid_df = run_simple_short_grid_backtest(
                            close=close,
                            high=price_df["high"].astype(float) if "high" in price_df.columns else close,
                            low=price_df["low"].astype(float) if "low" in price_df.columns else close,
                            raw_returns=raw_returns,
                            predictions=predictions,
                            confidence=confidence,
                            ml_df=ml_df,
                            timeframe=timeframe,
                            fee_rate=fee_rate,
                            regime_col=None,
                            max_holding_bars=horizon_bars,
                            tp_values=[profit_thr],
                            sl_values=[stop_thr],
                            trail_distance_atr_mult=trail_dist,
                            trail_atr_lookback=trail_atr_window,
                        )

                        if gate_mask is not None:
                            grid_df_gated = run_simple_short_grid_backtest(
                                close=close,
                                high=price_df["high"].astype(float) if "high" in price_df.columns else close,
                                low=price_df["low"].astype(float) if "low" in price_df.columns else close,
                                raw_returns=raw_returns,
                                predictions=predictions,
                                confidence=confidence,
                                ml_df=ml_df,
                                timeframe=timeframe,
                                fee_rate=fee_rate,
                                regime_col=None,
                                max_holding_bars=horizon_bars,
                                tp_values=[profit_thr],
                                sl_values=[stop_thr],
                                trail_distance_atr_mult=trail_dist,
                                trail_atr_lookback=trail_atr_window,
                                gate_mask=gate_mask,
                            )

                        if gate_prob is not None:
                            grid_df_prob = run_simple_short_grid_backtest(
                                close=close,
                                high=price_df["high"].astype(float) if "high" in price_df.columns else close,
                                low=price_df["low"].astype(float) if "low" in price_df.columns else close,
                                raw_returns=raw_returns,
                                predictions=predictions,
                                confidence=confidence,
                                ml_df=ml_df,
                                timeframe=timeframe,
                                fee_rate=fee_rate,
                                regime_col=None,
                                max_holding_bars=horizon_bars,
                                tp_values=[profit_thr],
                                sl_values=[stop_thr],
                                trail_distance_atr_mult=trail_dist,
                                trail_atr_lookback=trail_atr_window,
                                gate_prob=gate_prob,
                                gate_prob_threshold=gate_prob_threshold,
                            )
                    else:
                        grid_df = run_simple_long_grid_backtest(
                            close=close,
                            high=price_df["high"].astype(float) if "high" in price_df.columns else close,
                            low=price_df["low"].astype(float) if "low" in price_df.columns else close,
                            raw_returns=raw_returns,
                            predictions=predictions,
                            confidence=confidence,
                            ml_df=ml_df,
                            timeframe=timeframe,
                            fee_rate=fee_rate,
                            regime_col=None,
                            max_holding_bars=horizon_bars,
                            tp_values=[profit_thr],
                            sl_values=[stop_thr],
                            trail_distance_atr_mult=trail_dist,
                            trail_atr_lookback=trail_atr_window,
                        )

                        if gate_mask is not None:
                            grid_df_gated = run_simple_long_grid_backtest(
                                close=close,
                                high=price_df["high"].astype(float) if "high" in price_df.columns else close,
                                low=price_df["low"].astype(float) if "low" in price_df.columns else close,
                                raw_returns=raw_returns,
                                predictions=predictions,
                                confidence=confidence,
                                ml_df=ml_df,
                                timeframe=timeframe,
                                fee_rate=fee_rate,
                                regime_col=None,
                                max_holding_bars=horizon_bars,
                                tp_values=[profit_thr],
                                sl_values=[stop_thr],
                                trail_distance_atr_mult=trail_dist,
                                trail_atr_lookback=trail_atr_window,
                                gate_mask=gate_mask,
                            )

                        if gate_prob is not None:
                            grid_df_prob = run_simple_long_grid_backtest(
                                close=close,
                                high=price_df["high"].astype(float) if "high" in price_df.columns else close,
                                low=price_df["low"].astype(float) if "low" in price_df.columns else close,
                                raw_returns=raw_returns,
                                predictions=predictions,
                                confidence=confidence,
                                ml_df=ml_df,
                                timeframe=timeframe,
                                fee_rate=fee_rate,
                                regime_col=None,
                                max_holding_bars=horizon_bars,
                                tp_values=[profit_thr],
                                sl_values=[stop_thr],
                                trail_distance_atr_mult=trail_dist,
                                trail_atr_lookback=trail_atr_window,
                                gate_prob=gate_prob,
                                gate_prob_threshold=gate_prob_threshold,
                            )

                    if grid_df is not None and not grid_df.empty:
                        best_row = grid_df.iloc[0]
                        production_total_return_with_fees = float(
                            best_row.get("strategy_total_return_with_fees_%", 0.0)
                        ) / 100.0
                        production_win_rate_with_fees = float(
                            best_row.get("win_rate_with_fees", 0.0)
                        )
                        production_n_trades = int(best_row.get("number_of_trades", 0))
                        production_max_dd_with_fees = float(
                            best_row.get("max_drawdown_with_fees", 0.0)
                        )

                        metrics.update(
                            {
                                "production_take_profit_pct": float(
                                    best_row.get("take_profit_pct", profit_thr)
                                ),
                                "production_stop_loss_pct": float(
                                    best_row.get("stop_loss_pct", stop_thr)
                                ),
                                "production_max_holding_bars": horizon_bars,
                                "production_fee_rate": fee_rate,
                                "production_trail_distance_atr": trail_dist,
                                "production_trail_atr_lookback": trail_atr_window,
                                "production_total_return_with_fees": production_total_return_with_fees,
                                "production_win_rate_with_fees": production_win_rate_with_fees,
                                "production_number_of_trades": production_n_trades,
                                "production_max_drawdown_with_fees": production_max_dd_with_fees,
                            }
                        )

                        tprint_info(
                            "Production TPSL metrics (meta_gating_config): "
                            f"TP={profit_thr:.4f}, SL={stop_thr:.4f}, horizon_bars={horizon_bars}, "
                            f"fee_rate={fee_rate:.4f}, trades={production_n_trades}, "
                            f"win_rate={production_win_rate_with_fees:.3f}"
                        )

                    if grid_df_gated is not None and not grid_df_gated.empty:
                        best_row_g = grid_df_gated.iloc[0]
                        g_total_return_with_fees = float(
                            best_row_g.get("strategy_total_return_with_fees_%", 0.0)
                        ) / 100.0
                        g_win_rate_with_fees = float(best_row_g.get("win_rate_with_fees", 0.0))
                        g_n_trades = int(best_row_g.get("number_of_trades", 0))
                        g_max_dd_with_fees = float(best_row_g.get("max_drawdown_with_fees", 0.0))

                        metrics.update(
                            {
                                "production_gated_take_profit_pct": float(
                                    best_row_g.get("take_profit_pct", profit_thr)
                                ),
                                "production_gated_stop_loss_pct": float(
                                    best_row_g.get("stop_loss_pct", stop_thr)
                                ),
                                "production_gated_max_holding_bars": horizon_bars,
                                "production_gated_fee_rate": fee_rate,
                                "production_gated_trail_distance_atr": trail_dist,
                                "production_gated_trail_atr_lookback": trail_atr_window,
                                "production_gated_total_return_with_fees": g_total_return_with_fees,
                                "production_gated_win_rate_with_fees": g_win_rate_with_fees,
                                "production_gated_number_of_trades": g_n_trades,
                                "production_gated_max_drawdown_with_fees": g_max_dd_with_fees,
                            }
                        )

                        tprint_info(
                            "Production TPSL metrics with hard gate: "
                            f"TP={profit_thr:.4f}, SL={stop_thr:.4f}, horizon_bars={horizon_bars}, "
                            f"fee_rate={fee_rate:.4f}, trades={g_n_trades}, "
                            f"win_rate={g_win_rate_with_fees:.3f}"
                        )

                    if grid_df_prob is not None and not grid_df_prob.empty:
                        best_row_pg = grid_df_prob.iloc[0]
                        pg_total_return_with_fees = float(
                            best_row_pg.get("strategy_total_return_with_fees_%", 0.0)
                        ) / 100.0
                        pg_win_rate_with_fees = float(best_row_pg.get("win_rate_with_fees", 0.0))
                        pg_n_trades = int(best_row_pg.get("number_of_trades", 0))
                        pg_max_dd_with_fees = float(best_row_pg.get("max_drawdown_with_fees", 0.0))

                        metrics.update(
                            {
                                "production_prob_gated_take_profit_pct": float(
                                    best_row_pg.get("take_profit_pct", profit_thr)
                                ),
                                "production_prob_gated_stop_loss_pct": float(
                                    best_row_pg.get("stop_loss_pct", stop_thr)
                                ),
                                "production_prob_gated_max_holding_bars": horizon_bars,
                                "production_prob_gated_fee_rate": fee_rate,
                                "production_prob_gated_trail_distance_atr": trail_dist,
                                "production_prob_gated_trail_atr_lookback": trail_atr_window,
                                "production_prob_gated_total_return_with_fees": pg_total_return_with_fees,
                                "production_prob_gated_win_rate_with_fees": pg_win_rate_with_fees,
                                "production_prob_gated_number_of_trades": pg_n_trades,
                                "production_prob_gated_max_drawdown_with_fees": pg_max_dd_with_fees,
                                "production_prob_gate_threshold": gate_prob_threshold,
                            }
                        )

                        tprint_info(
                            "Production TPSL metrics with prob gate: "
                            f"TP={profit_thr:.4f}, SL={stop_thr:.4f}, horizon_bars={horizon_bars}, "
                            f"fee_rate={fee_rate:.4f}, trades={pg_n_trades}, "
                            f"win_rate={pg_win_rate_with_fees:.3f}, prob_thr={gate_prob_threshold:.3f}"
                        )
            except Exception:
                pass

            # ------------------------------------------------------------------
            # 7) Save Markdown report under outcomes/ with "base_analyst_" prefix
            # ------------------------------------------------------------------
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"base_analyst_{symbol}_{timeframe}_{direction}_backtest_{timestamp}.md"
            filepath = outcomes_dir / filename

            tprint_info(f"Writing base analyst backtest report to {filepath}")

            with open(filepath, "w") as f:
                f.write("# Base Analyst Backtest Report\n\n")
                f.write(f"- Symbol: {symbol}\n")
                f.write(f"- Exchange: {exchange}\n")
                f.write(f"- Timeframe: {timeframe}\n")
                f.write(f"- Direction: {direction}\n")
                f.write(f"- Execution Mode: {execution_mode}\n")
                f.write(f"- Bars: {n_bars}\n")
                f.write(f"- Data Source: {source}\n")
                if len(common_index) > 0:
                    f.write(f"- Period: {common_index[0]} → {common_index[-1]}\n")
                f.write("\n## Performance Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                def pct(x: float) -> str:
                    return f"{x:.2%}"

                def num(x: float) -> str:
                    return f"{x:.4f}"

                f.write(f"| Total Return | {pct(total_return)} |\n")
                f.write(f"| Annualized Return | {pct(annualized_return)} |\n")
                f.write(f"| Annualized Volatility | {pct(annualized_vol)} |\n")
                f.write(f"| Sharpe Ratio | {num(sharpe)} |\n")
                f.write(f"| Sortino Ratio | {num(sortino)} |\n")
                f.write(f"| Max Drawdown | {pct(max_drawdown)} |\n")
                f.write(f"| Win Rate | {pct(win_rate)} |\n")
                f.write(f"| Profit Factor | {num(profit_factor)} |\n")
                f.write(f"| Avg Win per Bar | {pct(avg_win)} |\n")
                f.write(f"| Avg Loss per Bar | {pct(avg_loss)} |\n")
                f.write(f"| Approx. Trades | {approx_trades} |\n")

                if "gated_sharpe_ratio" in metrics:
                    f.write("\n## Gate-Aware Overlay Metrics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Gated Total Return | {pct(metrics['gated_total_return'])} |\n")
                    f.write(f"| Gated Annualized Return | {pct(metrics['gated_annualized_return'])} |\n")
                    f.write(f"| Gated Annualized Volatility | {pct(metrics['gated_annualized_volatility'])} |\n")
                    f.write(f"| Gated Sharpe Ratio | {num(metrics['gated_sharpe_ratio'])} |\n")
                    f.write(f"| Gated Sortino Ratio | {num(metrics['gated_sortino_ratio'])} |\n")
                    f.write(f"| Gated Max Drawdown | {pct(metrics['gated_max_drawdown'])} |\n")
                    f.write(f"| Gated Win Rate | {pct(metrics['gated_bar_win_rate'])} |\n")
                    f.write(f"| Gated Profit Factor | {num(metrics['gated_profit_factor'])} |\n")
                    f.write(f"| Gated Avg Win per Bar | {pct(metrics['gated_avg_win'])} |\n")
                    f.write(f"| Gated Avg Loss per Bar | {pct(metrics['gated_avg_loss'])} |\n")
                    f.write(f"| Gated Approx. Trades | {metrics['gated_approx_trades']} |\n")
                    f.write(f"| Gate Coverage Rate | {pct(metrics['gate_coverage_rate'])} |\n")

                if "prob_gated_sharpe_ratio" in metrics:
                    f.write("\n## Gate Probability-Weighted Overlay Metrics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Prob-Gated Total Return | {pct(metrics['prob_gated_total_return'])} |\n")
                    f.write(f"| Prob-Gated Annualized Return | {pct(metrics['prob_gated_annualized_return'])} |\n")
                    f.write(f"| Prob-Gated Annualized Volatility | {pct(metrics['prob_gated_annualized_volatility'])} |\n")
                    f.write(f"| Prob-Gated Sharpe Ratio | {num(metrics['prob_gated_sharpe_ratio'])} |\n")
                    f.write(f"| Prob-Gated Sortino Ratio | {num(metrics['prob_gated_sortino_ratio'])} |\n")
                    f.write(f"| Prob-Gated Max Drawdown | {pct(metrics['prob_gated_max_drawdown'])} |\n")
                    f.write(f"| Prob-Gated Win Rate | {pct(metrics['prob_gated_bar_win_rate'])} |\n")
                    f.write(f"| Prob-Gated Profit Factor | {num(metrics['prob_gated_profit_factor'])} |\n")
                    f.write(f"| Prob-Gated Avg Win per Bar | {pct(metrics['prob_gated_avg_win'])} |\n")
                    f.write(f"| Prob-Gated Avg Loss per Bar | {pct(metrics['prob_gated_avg_loss'])} |\n")
                    f.write(f"| Prob-Gated Approx. Trades | {metrics['prob_gated_approx_trades']} |\n")
                    f.write(f"| Prob-Gate Coverage Rate | {pct(metrics['prob_gate_coverage_rate'])} |\n")
                    f.write(f"| Prob-Gate Mean Score | {num(metrics['prob_gate_mean_score'])} |\n")

                if "production_win_rate_with_fees" in metrics:
                    f.write("\n## Production Trade Metrics (meta-labeling TPSL config)\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(
                        f"| Take Profit | {pct(metrics['production_take_profit_pct'])} |\n"
                    )
                    f.write(
                        f"| Stop Loss | {pct(metrics['production_stop_loss_pct'])} |\n"
                    )
                    f.write(
                        f"| Max Holding (bars) | {metrics['production_max_holding_bars']} |\n"
                    )
                    f.write(
                        f"| Trail Distance (ATR multiples) | {num(metrics['production_trail_distance_atr'])} |\n"
                    )
                    f.write(
                        f"| ATR Lookback (bars) | {metrics['production_trail_atr_lookback']} |\n"
                    )
                    f.write(
                        f"| Fee Rate (per trade) | {pct(metrics['production_fee_rate'])} |\n"
                    )
                    f.write(
                        f"| Trades | {metrics['production_number_of_trades']} |\n"
                    )
                    f.write(
                        f"| Total Return (with fees) | {pct(metrics['production_total_return_with_fees'])} |\n"
                    )
                    f.write(
                        f"| Max Drawdown (with fees) | {pct(metrics['production_max_drawdown_with_fees'])} |\n"
                    )
                    f.write(
                        f"| Win Rate (with fees) | {pct(metrics['production_win_rate_with_fees'])} |\n"
                    )

                if "production_gated_win_rate_with_fees" in metrics:
                    f.write("\n## Production Trade Metrics with Hard Gate\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(
                        f"| Take Profit | {pct(metrics['production_gated_take_profit_pct'])} |\n"
                    )
                    f.write(
                        f"| Stop Loss | {pct(metrics['production_gated_stop_loss_pct'])} |\n"
                    )
                    f.write(
                        f"| Max Holding (bars) | {metrics['production_gated_max_holding_bars']} |\n"
                    )
                    f.write(
                        f"| Trail Distance (ATR multiples) | {num(metrics['production_gated_trail_distance_atr'])} |\n"
                    )
                    f.write(
                        f"| ATR Lookback (bars) | {metrics['production_gated_trail_atr_lookback']} |\n"
                    )
                    f.write(
                        f"| Fee Rate (per trade) | {pct(metrics['production_gated_fee_rate'])} |\n"
                    )
                    f.write(
                        f"| Trades | {metrics['production_gated_number_of_trades']} |\n"
                    )
                    f.write(
                        f"| Total Return (with fees) | {pct(metrics['production_gated_total_return_with_fees'])} |\n"
                    )
                    f.write(
                        f"| Max Drawdown (with fees) | {pct(metrics['production_gated_max_drawdown_with_fees'])} |\n"
                    )
                    f.write(
                        f"| Win Rate (with fees) | {pct(metrics['production_gated_win_rate_with_fees'])} |\n"
                    )

                if "production_prob_gated_win_rate_with_fees" in metrics:
                    f.write("\n## Production Trade Metrics with Probabilistic Gate\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(
                        f"| Take Profit | {pct(metrics['production_prob_gated_take_profit_pct'])} |\n"
                    )
                    f.write(
                        f"| Stop Loss | {pct(metrics['production_prob_gated_stop_loss_pct'])} |\n"
                    )
                    f.write(
                        f"| Max Holding (bars) | {metrics['production_prob_gated_max_holding_bars']} |\n"
                    )
                    f.write(
                        f"| Trail Distance (ATR multiples) | {num(metrics['production_prob_gated_trail_distance_atr'])} |\n"
                    )
                    f.write(
                        f"| ATR Lookback (bars) | {metrics['production_prob_gated_trail_atr_lookback']} |\n"
                    )
                    f.write(
                        f"| Fee Rate (per trade) | {pct(metrics['production_prob_gated_fee_rate'])} |\n"
                    )
                    f.write(
                        f"| Gate Prob Threshold | {num(metrics['production_prob_gate_threshold'])} |\n"
                    )
                    f.write(
                        f"| Trades | {metrics['production_prob_gated_number_of_trades']} |\n"
                    )
                    f.write(
                        f"| Total Return (with fees) | {pct(metrics['production_prob_gated_total_return_with_fees'])} |\n"
                    )
                    f.write(
                        f"| Max Drawdown (with fees) | {pct(metrics['production_prob_gated_max_drawdown_with_fees'])} |\n"
                    )
                    f.write(
                        f"| Win Rate (with fees) | {pct(metrics['production_prob_gated_win_rate_with_fees'])} |\n"
                    )

            tprint_success(f"Base analyst backtest report saved to: {filepath}")

            return {
                "success": True,
                "artifacts": {"backtest_report_markdown": str(filepath)},
                "metrics": metrics,
            }

        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Analyst base backtest failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            return {
                "success": False,
                "artifacts": {},
                "metrics": {},
                "error": error_msg,
            }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# ----------------------------------------------------------------------
# Step registration
# ----------------------------------------------------------------------

def register_analyst_base_backtest_step() -> None:
    """Register the analyst base backtest step in the global registry."""
    from src.training.steps.base_step import step_registry

    step_registry.register("analyst_base_backtest", AnalystBaseBacktestStep)
    tprint("✅ Analyst base backtest step registered", "SUCCESS")


# Auto-register when module is imported
register_analyst_base_backtest_step()
