"""
Meta-Gated Backtest Step.

This step evaluates a meta-gated strategy using the same artifacts
that will be used live:

- Labeled data from FeatureGenerationMetaLabelingStep
- meta_gating_config.json produced by that step
- Iso regressor artifact referenced in meta_gating_config

The backtest operates at the event level:
- Each labeled event corresponds to one potential trade
- The meta gate (probability + expected-return thresholds) decides
  whether the trade would be taken
- The realized_return from labeling is used as the trade PnL

This mirrors the live decision rule that gates entries on meta
probabilities and isotonic expected returns.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import pickle

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error
from src.training.steps.labeling.labeled_data_schema import (
    get_required_labeled_data_columns,
    validate_labeled_data_schema,
)


logger = logging.getLogger(__name__)


class MetaGatedBacktestStep(BaseStep):
    """Meta-gated event-level backtest using meta-labeling artifacts."""

    def __init__(self, step_name: str = "meta_gated_backtest"):
        super().__init__(step_name)
        self.logger = system_logger.getChild("MetaGatedBacktest")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a meta-gated backtest using meta-labeling artifacts.

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
            f"🧪 Starting meta-gated backtest for {symbol} {timeframe} {direction} (mode={execution_mode})",
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
            # 1) Load labeled_data artifact from meta-labeling step
            # ------------------------------------------------------------------
            artifact_name = f"labeled_data_{symbol}_{timeframe}"
            tprint_info(f"🔎 Loading labeled data artifact: {artifact_name}")

            labeled_data = self._get_artifact(
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="features",
            )

            if labeled_data is None:
                raise ValueError(f"Labeled data artifact '{artifact_name}' not found")

            if not isinstance(labeled_data, pd.DataFrame) or labeled_data.empty:
                raise ValueError(
                    f"Labeled data artifact '{artifact_name}' is empty or not a DataFrame"
                )

            df = labeled_data.copy().sort_index()

            # Validate labeled_data schema for required columns
            validate_labeled_data_schema(
                df,
                required_cols=get_required_labeled_data_columns(
                    [
                        "meta_probability",
                        "event_duration_bars",
                    ]
                ),
                context="MetaGatedBacktestStep",
            )

            realized_returns = df["realized_return"].astype(float)
            meta_prob = df["meta_probability"].astype(float)

            # Event mask: where realized_return is defined
            event_mask = ~realized_returns.isna()
            n_events = int(event_mask.sum())
            if n_events == 0:
                raise ValueError("No labeled events found in labeled_data")

            tprint_info(f"📊 Meta-gated backtest: {n_events} labeled events available")

            # ------------------------------------------------------------------
            # 2) Load meta_gating_config and iso regressor artifact
            # ------------------------------------------------------------------
            va_dir = Path("versioned_artifacts") / f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
            gating_path = va_dir / "meta_gating_config.json"

            if not gating_path.exists():
                raise FileNotFoundError(
                    f"meta_gating_config.json not found at {gating_path}; run feature_generation_meta_labeling_step first"
                )

            with open(gating_path, "r") as f_cfg:
                gating_config = json.load(f_cfg)

            meta_gating = gating_config.get("meta_gating", {})
            entry_cfg = meta_gating.get("entry", {})
            calibration_cfg = meta_gating.get("calibration", {})
            backtest_metrics_cfg = meta_gating.get("backtest_metrics", {})
            filters_cfg = meta_gating.get("filters", {})

            prob_threshold = float(entry_cfg.get("prob_threshold", 0.0))
            use_expected_return = bool(entry_cfg.get("use_expected_return", False))
            er_threshold = float(entry_cfg.get("expected_return_threshold", 0.0))

            iso_rel_path = calibration_cfg.get("iso_regressor_artifact")
            iso_model = None
            if iso_rel_path:
                iso_path = va_dir / iso_rel_path
                if iso_path.exists():
                    with open(iso_path, "rb") as f_iso:
                        iso_model = pickle.load(f_iso)
                    tprint_info(f"💾 Loaded iso regressor from {iso_path}")
                else:
                    tprint_error(
                        f"⚠️ Iso regressor artifact not found at {iso_path}; proceeding without expected-return gating"
                    )
                    use_expected_return = False

            # ------------------------------------------------------------------
            # 3) Apply meta gate to events
            # ------------------------------------------------------------------
            event_probs = meta_prob.loc[event_mask]
            event_returns = realized_returns.loc[event_mask]

            # Default: probability gate only
            gate_mask = event_probs >= prob_threshold
            expected_returns = None

            if use_expected_return and iso_model is not None:
                try:
                    prob_array = event_probs.to_numpy(dtype=float)
                    er_array = iso_model.predict(prob_array)
                    expected_returns = pd.Series(er_array, index=event_probs.index)
                    gate_mask &= expected_returns >= er_threshold
                except Exception as e:
                    tprint_error(
                        f"⚠️ Failed to apply expected-return gating ({e}); falling back to probability-only gate"
                    )
                    use_expected_return = False

            try:
                df_events = df.loc[event_probs.index]

                use_vol_filter = bool(filters_cfg.get("use_volatility_filter", True))
                vol_quantile = float(filters_cfg.get("volatility_quantile", 0.40))
                use_trend_filter = bool(filters_cfg.get("use_trend_filter", True))
                trend_window = int(filters_cfg.get("trend_window", 20))
                trend_min_abs = float(filters_cfg.get("trend_min_abs", 0.0))

                if use_vol_filter and "volatility_1d" in df_events.columns:
                    v = df_events["volatility_1d"].astype(float)
                    try:
                        v_thr = v.quantile(vol_quantile)
                    except Exception:
                        v_thr = v.quantile(0.40)
                    vol_mask = v >= v_thr
                    gate_mask &= vol_mask

                if use_trend_filter and "close" in df_events.columns:
                    close = df_events["close"].astype(float)
                    sma = close.rolling(trend_window, min_periods=trend_window // 2).mean()
                    trend = (close - sma) / sma
                    trend = trend.reindex(df_events.index)
                    trend_mask = trend.abs() >= trend_min_abs
                    gate_mask &= trend_mask
            except Exception as e:
                tprint_error(
                    f"⚠️ Candidate meta gate filters failed ({e}); falling back to prob/ER-only gate"
                )

            gated_returns = event_returns[gate_mask]
            n_trades = int(len(gated_returns))

            if n_trades == 0:
                raise ValueError(
                    "Meta gate produced zero trades; consider relaxing thresholds or verifying artifacts"
                )

            mean_ret = float(gated_returns.mean())
            std_ret = float(gated_returns.std(ddof=1)) if n_trades > 1 else 0.0
            sharpe_trade = float(mean_ret / std_ret) * np.sqrt(n_trades) if std_ret > 0 else 0.0

            # Simple trade-level equity curve (event-time, not bar-time)
            equity = (1.0 + gated_returns).cumprod()
            running_max = equity.cummax()
            drawdown = equity / running_max - 1.0
            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            tprint_info(
                f"📊 Meta-gated trades: {n_trades} | mean={mean_ret:.4f} | Sharpe(trade)={sharpe_trade:.3f} | maxDD={max_drawdown:.2%}"
            )

            # ------------------------------------------------------------------
            # 4) Write Markdown report under outcomes/
            # ------------------------------------------------------------------
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meta_gated_backtest_{symbol}_{timeframe}_{direction}_{timestamp}.md"
            filepath = outcomes_dir / filename

            tprint_info(f"📝 Writing meta-gated backtest report to {filepath}")

            with open(filepath, "w") as f:
                f.write("# Meta-Gated Backtest Report\n\n")
                f.write(f"- Symbol: {symbol}\n")
                f.write(f"- Exchange: {exchange}\n")
                f.write(f"- Timeframe: {timeframe}\n")
                f.write(f"- Direction: {direction}\n")
                f.write(f"- Execution Mode: {execution_mode}\n")
                f.write(f"- Events (labeled): {n_events}\n")
                f.write(f"- Trades (gated): {n_trades}\n")
                f.write("\n## Gating Configuration\n\n")
                f.write(f"- Probability Threshold: {prob_threshold:.3f}\n")
                f.write(f"- Use Expected Return: {use_expected_return}\n")
                if use_expected_return:
                    f.write(f"- Expected Return Threshold: {er_threshold:.4f} (fraction)\n")
                f.write("\n## Trade-Level Performance (event-time)\n\n")
                f.write(f"- Mean Return per Trade: {mean_ret:.4%}\n")
                f.write(f"- Std Dev per Trade: {std_ret:.4%}\n")
                f.write(f"- Trade-Level Sharpe (sqrt(N)): {sharpe_trade:.3f}\n")
                f.write(f"- Max Drawdown (event-time equity): {max_drawdown:.2%}\n")

                if backtest_metrics_cfg:
                    auc_oof = float(backtest_metrics_cfg.get("auc_oof", 0.0))
                    mean_return_gated_diag = float(backtest_metrics_cfg.get("mean_return_gated", 0.0))
                    sharpe_gated_diag = float(backtest_metrics_cfg.get("sharpe_gated", 0.0))
                    trades_gated_diag = int(backtest_metrics_cfg.get("trades_gated", 0))

                    avg_trades_per_day_diag = None
                    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 2:
                        start_day = df.index[0].normalize()
                        end_day = df.index[-1].normalize()
                        n_days = int((end_day - start_day).days) + 1
                        if n_days <= 0:
                            n_days = 1
                        avg_trades_per_day_diag = trades_gated_diag / float(n_days)

                    f.write("\n## Meta-Gating Diagnostics (from meta-labeling step)\n\n")
                    f.write("- These metrics are computed during the meta-labeling step for the diagnostics gate.\n")
                    f.write(f"- AUC (OOF meta-model): {auc_oof:.3f}\n")
                    f.write(f"- Mean return per gated trade (diagnostics gate): {mean_return_gated_diag:.2%}\n")
                    f.write(f"- Sharpe (diagnostics gated set): {sharpe_gated_diag:.2f}\n")
                    f.write(f"- Trades gated (diagnostics gate): {trades_gated_diag}\n")
                    if avg_trades_per_day_diag is not None:
                        f.write(f"- Approximate average trades per day (diagnostics gate): {avg_trades_per_day_diag:.2f}\n")

            tprint_success(f" Meta-gated backtest report saved to: {filepath}")

            metrics: Dict[str, Any] = {
                "n_events": n_events,
                "n_trades_gated": n_trades,
                "mean_return_gated": mean_ret,
                "std_return_gated": std_ret,
                "sharpe_trade": sharpe_trade,
                "max_drawdown_event_time": max_drawdown,
                "prob_threshold": prob_threshold,
                "use_expected_return": use_expected_return,
                "expected_return_threshold": er_threshold,
            }

            return {
                "success": True,
                "artifacts": {"meta_gated_backtest_report": str(filepath)},
                "metrics": metrics,
            }

        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Meta-gated backtest failed: {e}"
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


def register_meta_gated_backtest_step() -> None:
    """Register the meta-gated backtest step in the global registry."""
    from src.training.steps.base_step import step_registry

    step_registry.register("meta_gated_backtest", MetaGatedBacktestStep)
    tprint("✅ Meta-gated backtest step registered", "SUCCESS")


# Auto-register when module is imported
register_meta_gated_backtest_step()
