"""
Analyst Base Backtest Step.

This step performs a simple PnL-based backtest using analyst predictions
(ML-scored historical data) and computes true Sharpe/Sortino/max drawdown
on the resulting returns series. It saves a standalone Markdown report
in the outcomes/ directory with a filename that includes "base_analyst_".
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error


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

            price_df = price_data.copy().sort_index()
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

            ml_df = ml_scored.copy().sort_index()

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

            # ------------------------------------------------------------------
            # 7) Save Markdown report under outcomes/ with "base_analyst_" prefix
            # ------------------------------------------------------------------
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"base_analyst_{symbol}_{timeframe}_{direction}_backtest_{timestamp}.md"
            filepath = outcomes_dir / filename

            tprint_info(f"📝 Writing base analyst backtest report to {filepath}")

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

            tprint_success(f"✅ Base analyst backtest report saved to: {filepath}")

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
