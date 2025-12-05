"""
Analyst Base Backtest Step.

Simple PnL-based backtest using analyst OOS predictions to compute
true Sharpe/Sortino and related metrics, with a Markdown report
saved under outcomes/.
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
from src.utils.ml_common.trading_grid_backtester import (
    run_simple_long_grid_backtest,
    run_simple_short_grid_backtest,
)
from src.utils.ml_common.confidence_metrics import apply_risk_adjusted_confidence
from src.utils.versioned_artifacts.temporal_splits import TemporalSplitConfig


logger = logging.getLogger(__name__)


class AnalystBaseBacktestStep(BaseStep):
    """Simple PnL-based backtest for analyst base models using OOS predictions."""

    def __init__(self, step_name: str = "analyst_base_backtest"):
        super().__init__(step_name)
        self.logger = system_logger.getChild("AnalystBaseBacktest")

    @staticmethod
    def _bars_per_year_from_timeframe(timeframe: str) -> float:
        """Approximate number of bars per year for a given timeframe string."""
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
            pass

        return 365.0

    @staticmethod
    def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with a tz-naive UTC DatetimeIndex for safe alignment/sorting.

        Handles cases where the index is non-datetime, tz-aware, or mixed.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            idx = pd.to_datetime(df.index, errors="coerce")
        else:
            idx = df.index

        if isinstance(idx, pd.DatetimeIndex):
            # Convert any tz-aware index to UTC then drop tz to make it tz-naive
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)

            # Drop NaT values if any
            valid_mask = ~idx.isna()
            if not bool(valid_mask.all()):
                df = df.loc[valid_mask].copy()
                idx = idx[valid_mask]

            df = df.copy()
            df.index = idx

        return df

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        symbol = config.get("symbol", "UNKNOWN")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        execution_mode = config.get("execution_mode", "light")

        tprint(
            f"🧪 Starting analyst base backtest for {symbol} {timeframe} {direction} (mode={execution_mode})",
            "INFO",
        )

        self.set_context(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model="analyst",
            execution_mode=execution_mode,
        )

        try:
            price_data, source = self.load_market_data_or_fail(
                config,
                pipeline_state={},
                allow_config_override=False,
            )

            if price_data is None or not isinstance(price_data, pd.DataFrame) or price_data.empty:
                raise ValueError("Price data not available or empty for backtest")

            price_df = price_data.copy()
            # Trust the existing index and normalize to a clean DatetimeIndex
            price_df = self._normalize_datetime_index(price_df).sort_index()
            # Drop duplicate timestamps to ensure a unique index for reindexing
            if not price_df.index.is_unique:
                tprint_info(
                    "⚠️ Price data contains duplicate timestamps; keeping last occurrence per timestamp for backtest alignment."
                )
                price_df = price_df[~price_df.index.duplicated(keep="last")]
            required_cols = {"close", "high", "low"}
            missing = required_cols.difference(price_df.columns)
            if missing:
                raise ValueError(f"Price data is missing required columns for grid backtest: {sorted(missing)}")

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
            ml_df = self._normalize_datetime_index(ml_df).sort_index()
            # Ensure unique index for ML-scored data as well
            if not ml_df.index.is_unique:
                tprint_info(
                    "⚠️ ML-scored data contains duplicate timestamps; keeping last occurrence per timestamp for backtest alignment."
                )
                ml_df = ml_df[~ml_df.index.duplicated(keep="last")]

            # Prefer OHLC from ML-scored artifact itself to guarantee perfect
            # temporal alignment between prices and OOS predictions. Fall back
            # to externally loaded price data only when ML-scored data does not
            # provide the required OHLC columns.
            ohlc_cols = ["close", "high", "low"]
            if all(c in ml_df.columns for c in ohlc_cols):
                price_df = ml_df[ohlc_cols].copy()
                for c in ohlc_cols:
                    price_df[c] = price_df[c].astype(float)
                source = "ml_scored_historical_data"

            # ------------------------------------------------------------------
            # Derive temporal splits (train/val/test) from the actual OOS data
            # range used in this run, then restrict the backtest strictly to the
            # test period.
            # ------------------------------------------------------------------
            if not isinstance(ml_df.index, pd.DatetimeIndex) or not isinstance(price_df.index, pd.DatetimeIndex):
                raise ValueError(
                    "ML-scored data and price data must both have DatetimeIndex for temporal filtering."
                )

            # Determine the overlapping time window between ML data and price data
            data_start = max(ml_df.index.min(), price_df.index.min())
            data_end = min(ml_df.index.max(), price_df.index.max())

            if data_start >= data_end:
                # Fallback: if temporal ranges don't overlap but the ML-scored
                # artifact already contains OHLCV columns, use those directly as
                # the price data for backtesting to ensure consistent alignment
                # with the OOS predictions.
                ohlc_cols = ["close", "high", "low"]
                if all(c in ml_df.columns for c in ohlc_cols):
                    tprint_info(
                        "⚠️ No temporal overlap between external price data and ML-scored data; "
                        "using OHLC columns from ml_scored_historical_data as price series for backtest."
                    )
                    price_df = ml_df[ohlc_cols].copy()
                    for c in ohlc_cols:
                        price_df[c] = price_df[c].astype(float)
                    data_start = ml_df.index.min()
                    data_end = ml_df.index.max()
                else:
                    raise ValueError(
                        f"Insufficient temporal overlap between ML data and price data "
                        f"(ml_range={ml_df.index.min()}→{ml_df.index.max()}, "
                        f"price_range={price_df.index.min()}→{price_df.index.max()})"
                    )

            # Restrict both datasets to the common time window
            ml_df = ml_df.loc[(ml_df.index >= data_start) & (ml_df.index <= data_end)]
            price_df = price_df.loc[(price_df.index >= data_start) & (price_df.index <= data_end)]

            if len(ml_df) < 50 or len(price_df) < 50:
                raise ValueError(
                    f"Insufficient overlap between ML data and price data after window alignment "
                    f"(ml_samples={len(ml_df)}, price_samples={len(price_df)})"
                )

            # Reindex price data to ML index (ffill) so that every ML observation has a price bar
            price_df = price_df.reindex(ml_df.index, method="ffill")
            # Drop any rows where required price fields are still missing
            price_df = price_df.dropna(subset=["close", "high", "low"])
            ml_df = ml_df.loc[price_df.index]

            # Use the aligned OOS horizon to create a fresh temporal split
            temporal_config = TemporalSplitConfig.create_from_data(
                data_start=ml_df.index.min(),
                data_end=ml_df.index.max(),
                train_pct=0.6,
                val_pct=0.2,
                test_pct=0.2,
                embargo_days=1,
            )

            test_period = temporal_config.test
            test_start = test_period.start
            test_end = test_period.effective_end

            common_index = ml_df.index[(ml_df.index >= test_start) & (ml_df.index <= test_end)]

            if len(common_index) < 50:
                raise ValueError(
                    f"Insufficient overlap between ML data and price data in derived test period (common samples={len(common_index)})"
                )

            ml_df = ml_df.loc[common_index]
            price_df = price_df.loc[common_index]

            close = price_df["close"].astype(float)
            high = price_df["high"].astype(float)
            low = price_df["low"].astype(float)
            raw_returns = close.pct_change().fillna(0.0)

            volume_series = None
            if "volume" in price_df.columns:
                try:
                    volume_series = price_df["volume"].astype(float)
                except Exception:
                    volume_series = None

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
                numeric_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric prediction column found in ML-scored data")
                pred_col = numeric_cols[0]

            predictions = ml_df[pred_col].astype(float)

            # Prefer a risk-adjusted analyst confidence column when available.
            # Otherwise, fall back to raw confidence and apply risk adjustment
            # using the shared helper.
            risk_conf_col: str | None = None
            for c in ml_df.columns:
                name = c.lower()
                if "confidence" in name and "risk" in name:
                    risk_conf_col = c
                    break

            if risk_conf_col is not None:
                confidence = ml_df[risk_conf_col].astype(float).clip(0.0, 1.0)
            else:
                # Prefer an explicit model-provided confidence column when available
                # (e.g. 'analyst_confidence', 'oof_confidence', etc.). If none is
                # present in the ml_scored artifact, fall back to a derived
                # confidence based on the prediction magnitude.
                conf_col: str | None = None
                # Highest priority: explicit analyst-specific confidence
                for c in ml_df.columns:
                    name = c.lower()
                    if "analyst_confidence" in name:
                        conf_col = c
                        break
                # Next: any column that contains 'confidence'
                if conf_col is None:
                    for c in ml_df.columns:
                        if "confidence" in c.lower():
                            conf_col = c
                            break

                if conf_col is not None:
                    confidence = ml_df[conf_col].astype(float).clip(0.0, 1.0)
                else:
                    # No explicit confidence in the artifact; derive a proxy from
                    # the prediction itself.
                    confidence = predictions.clip(lower=0.0)
                    max_conf = float(confidence.max()) if len(confidence) > 0 else 0.0
                    if max_conf > 0.0:
                        confidence = confidence / max_conf
                    confidence = confidence.clip(0.0, 1.0)

                try:
                    confidence = apply_risk_adjusted_confidence(
                        confidence=confidence,
                        close=close,
                        volume=volume_series,
                    )
                except Exception:
                    pass

            # Log summary statistics of the analyst confidence distribution so
            # we can understand how different confidence thresholds interact
            # with this particular test horizon.
            try:
                conf_desc = confidence.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
                tprint_info("📊 Analyst confidence summary (test window):")
                tprint_info(f"   min={float(conf_desc['min']):.4f}, max={float(conf_desc['max']):.4f}")
                tprint_info(
                    "   p10={:.4f}, p25={:.4f}, p50={:.4f}, p75={:.4f}, p90={:.4f}".format(
                        float(conf_desc.get('10%')),
                        float(conf_desc.get('25%')),
                        float(conf_desc.get('50%')),
                        float(conf_desc.get('75%')),
                        float(conf_desc.get('90%')),
                    )
                )
                above_08 = float((confidence >= 0.8).mean()) if len(confidence) > 0 else 0.0
                above_07 = float((confidence >= 0.7).mean()) if len(confidence) > 0 else 0.0
                tprint_info(f"   share_conf>=0.7={above_07:.3f}, share_conf>=0.8={above_08:.3f}")
            except Exception:
                pass

            if direction == "long":
                signal = (predictions > 0).astype(float) * confidence
            elif direction == "short":
                signal = (predictions < 0).astype(float) * confidence * -1.0
            else:
                signal = np.sign(predictions).astype(float) * confidence

            position = signal.shift(1).fillna(0.0)

            # Optional gate overlay using gate_decision from GateTrainingStep
            gate_mask = None
            gated_position = None
            gated_strategy_returns = None
            try:
                gate_artifact_name = f"gate_oof_predictions_{symbol}"
                tprint_info(f"🔎 Attempting to load gate decisions: {gate_artifact_name}")

                gate_oof = self._get_artifact(
                    artifact_name=gate_artifact_name,
                    artifact_type="data",
                    data_category="predictions",
                )

                if (
                    gate_oof is not None
                    and isinstance(gate_oof, pd.DataFrame)
                    and "gate_decision" in gate_oof.columns
                ):
                    gate_series = gate_oof["gate_decision"].astype(float)
                    # Align to the ML-scored index; missing values default to "no gating" (1.0)
                    gate_series = gate_series.sort_index().reindex(ml_df.index).fillna(1.0)
                    gate_mask = gate_series.clip(0.0, 1.0)
            except Exception:
                gate_mask = None

            strategy_returns = position * raw_returns

            if gate_mask is not None:
                gated_position = position * gate_mask
                gated_strategy_returns = gated_position * raw_returns

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

            # Bar-level approximation: count discrete entries (0->1 in in/out position)
            in_position = (position != 0.0).astype(int)
            pos_changes = in_position.diff().fillna(0)
            entries = pos_changes == 1
            approx_trades = int(entries.sum())

            positive = strategy_returns[strategy_returns > 0]
            negative = strategy_returns[strategy_returns < 0]
            n_pos = int(len(positive))
            n_neg = int(len(negative))
            n_nonzero = n_pos + n_neg
            win_rate_bar = float(n_pos / n_nonzero) if n_nonzero > 0 else 0.0
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
                "bar_win_rate": win_rate_bar,
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
                    f"Bar win-rate {win_rate_bar:.3f} → {g_win_rate_bar:.3f}, "
                    f"Coverage={gate_coverage_rate:.2%}"
                )

            try:
                if len(confidence) >= 100:
                    ranks = confidence.rank(method="first")
                    qcut_result = pd.qcut(ranks, 10, labels=False, duplicates="drop")
                    quantiles = pd.Series(qcut_result, index=confidence.index)
                    grouped = strategy_returns.groupby(quantiles).mean()
                    mapped_vals = quantiles.map(grouped)
                    mapped = pd.Series(mapped_vals, index=confidence.index).fillna(0.0).astype(float)
                    min_val = float(mapped.min())
                    max_val = float(mapped.max())
                    if max_val > min_val:
                        mapped = (mapped - min_val) / (max_val - min_val)
                    else:
                        mapped = pd.Series(0.0, index=confidence.index)
                    confidence = mapped
            except Exception:
                pass

            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"base_analyst_{symbol}_{timeframe}_{direction}_backtest_{timestamp}.md"
            filepath = outcomes_dir / filename

            # Build multi-config grid summary (one row per TP/SL/conf combination)
            trade_win_rate: float | None = None
            grid_n_trades: int | None = None
            grid_total_return_with_fees: float | None = None

            if direction in ("long", "short"):
                regime_col = None
                try:
                    regime_candidates = [c for c in ml_df.columns if "regime" in c.lower()]
                    for c in regime_candidates:
                        nunique = ml_df[c].nunique(dropna=True)
                        if 1 < nunique <= 10:
                            regime_col = c
                            break
                except Exception:
                    regime_col = None

                if direction == "short":
                    grid_summary = run_simple_short_grid_backtest(
                        close=close,
                        high=high,
                        low=low,
                        raw_returns=raw_returns,
                        predictions=predictions,
                        confidence=confidence,
                        ml_df=ml_df,
                        timeframe=timeframe,
                        fee_rate=0.0015,
                        regime_col=regime_col,
                        max_holding_bars=6,
                    )
                else:
                    grid_summary = run_simple_long_grid_backtest(
                        close=close,
                        high=high,
                        low=low,
                        raw_returns=raw_returns,
                        predictions=predictions,
                        confidence=confidence,
                        ml_df=ml_df,
                        timeframe=timeframe,
                        fee_rate=0.0015,
                        regime_col=regime_col,
                        max_holding_bars=6,
                    )

                csv_path = filepath.with_suffix(".csv")
                grid_summary.to_csv(csv_path, index=False)
                tprint_info(f"📄 Base analyst grid backtest summary saved to CSV: {csv_path}")

                # Derive trade-level summary metrics from the best TPSL configuration
                try:
                    if not grid_summary.empty:
                        best_row = grid_summary.sort_values(
                            "strategy_total_return_with_fees_%", ascending=False
                        ).iloc[0]
                        trade_win_rate = float(best_row.get("win_rate_with_fees", np.nan))
                        grid_n_trades = int(best_row.get("number_of_trades", 0))
                        grid_total_return_with_fees = float(
                            best_row.get("strategy_total_return_with_fees_%", np.nan)
                        ) / 100.0
                except Exception:
                    trade_win_rate = None
                    grid_n_trades = None
                    grid_total_return_with_fees = None

            tprint_info(f"📝 Writing base analyst backtest report to {filepath}")

            with open(filepath, "w") as f:
                f.write("# Base Analyst Backtest Report\n\n")
                f.write(f"- Symbol: {symbol}\n")
                f.write(f"- Exchange: {exchange}\n")
                f.write(f"- Timeframe: {timeframe}\n")
                f.write(f"- Direction: {direction}\n")
                f.write(f"- Execution Mode: {execution_mode}\n")
                f.write(f"- Bars: {n_bars}\n")
                # Approximate duration in days based on timeframe
                bars_per_year = self._bars_per_year_from_timeframe(timeframe)
                bars_per_day = bars_per_year / 365.0 if bars_per_year > 0 else 0.0
                approx_days = n_bars / bars_per_day if bars_per_day > 0 else 0.0
                f.write(f"- Approx. Duration: {approx_days:.1f} days\n")
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

                f.write(f"| Total Return (bar-level) | {pct(total_return)} |\n")
                f.write(f"| Annualized Return | {pct(annualized_return)} |\n")
                f.write(f"| Annualized Volatility | {pct(annualized_vol)} |\n")
                f.write(f"| Sharpe Ratio | {num(sharpe)} |\n")
                f.write(f"| Sortino Ratio | {num(sortino)} |\n")
                f.write(f"| Max Drawdown | {pct(max_drawdown)} |\n")
                f.write(f"| Bar Win Rate | {pct(win_rate_bar)} |\n")
                if trade_win_rate is not None:
                    f.write(
                        f"| Trade Win Rate (grid TPSL best config) | {pct(trade_win_rate)} |\n"
                    )
                if grid_total_return_with_fees is not None:
                    f.write(
                        f"| Grid Total Return (with fees, best config) | {pct(grid_total_return_with_fees)} |\n"
                    )
                f.write(f"| Profit Factor | {num(profit_factor)} |\n")
                f.write(f"| Avg Win per Bar | {pct(avg_win)} |\n")
                f.write(f"| Avg Loss per Bar | {pct(avg_loss)} |\n")
                f.write(f"| Approx. Trades (position entries) | {approx_trades} |\n")

                if "gated_sharpe_ratio" in metrics:
                    f.write("\n## Gate-Aware Overlay Metrics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Gated Total Return (bar-level) | {pct(metrics['gated_total_return'])} |\n")
                    f.write(f"| Gated Annualized Return | {pct(metrics['gated_annualized_return'])} |\n")
                    f.write(f"| Gated Annualized Volatility | {pct(metrics['gated_annualized_volatility'])} |\n")
                    f.write(f"| Gated Sharpe Ratio | {num(metrics['gated_sharpe_ratio'])} |\n")
                    f.write(f"| Gated Sortino Ratio | {num(metrics['gated_sortino_ratio'])} |\n")
                    f.write(f"| Gated Max Drawdown | {pct(metrics['gated_max_drawdown'])} |\n")
                    f.write(f"| Gated Bar Win Rate | {pct(metrics['gated_bar_win_rate'])} |\n")
                    f.write(f"| Gated Profit Factor | {num(metrics['gated_profit_factor'])} |\n")
                    f.write(f"| Gated Avg Win per Bar | {pct(metrics['gated_avg_win'])} |\n")
                    f.write(f"| Gated Avg Loss per Bar | {pct(metrics['gated_avg_loss'])} |\n")
                    f.write(f"| Gated Approx. Trades | {metrics['gated_approx_trades']} |\n")
                    f.write(f"| Gate Coverage Rate | {pct(metrics['gate_coverage_rate'])} |\n")
                if grid_n_trades is not None:
                    f.write(
                        f"| Trades (grid TPSL best config) | {grid_n_trades} |\n"
                    )

            tprint_success(f"✅ Base analyst backtest report saved to: {filepath}")

            return {
                "success": True,
                "artifacts": {"backtest_report_markdown": str(filepath)},
                "metrics": metrics,
            }

        except Exception as e:
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
        return await self.execute(config)


def register_analyst_base_backtest_step() -> None:
    from src.training.steps.base_step import step_registry

    step_registry.register("analyst_base_backtest", AnalystBaseBacktestStep)
    tprint("✅ Analyst base backtest step registered", "SUCCESS")


register_analyst_base_backtest_step()
