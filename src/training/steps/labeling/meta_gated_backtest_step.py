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
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning
from src.training.steps.labeling.labeled_data_schema import (
    get_required_labeled_data_columns,
    validate_labeled_data_schema,
)
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs


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

            df = labeled_data.copy()
            df = self._normalize_datetime_index(df, "labeled_data")
            df = df.sort_index()

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

            try:
                specialist_config = dict(config)
                specialist_config.setdefault("use_canonical_specialist_scalars", True)
                specialist_config.setdefault("enable_risk_hmm_specialist", False)

                specialist_df = get_specialist_models_outputs(
                    artifact_router=self.artifact_router,
                    training_index=df.index,
                    config=specialist_config,
                    logger=self.logger,
                    strict=False,
                )

                if specialist_df is not None and not specialist_df.empty:
                    prob_cols = [
                        c
                        for c in specialist_df.columns
                        if c.startswith("liquidity_regime_") and "prob_" in c
                    ]
                    if prob_cols:
                        liquidity_features = specialist_df[prob_cols].reindex(
                            df.index, method="ffill"
                        )
                        for col in liquidity_features.columns:
                            out_col = f"liquidity_{col}"
                            if out_col not in df.columns:
                                df[out_col] = liquidity_features[col]

                    scalar_cols = []
                    for col in [
                        "risk_score",
                        "path_risk_score",
                        "macro_trend_score_continuous",
                        "mr_probability_dense",
                        "mr_probability",
                        "mr_raw_score",
                        "mr_trend_state",
                        "mr_trend_is_mr",
                        "sr_labeling_xgb_prob",
                        "vol_force_scalar",
                        "smc_predicted",
                    ]:
                        if col in specialist_df.columns:
                            scalar_cols.append(col)

                    scalar_cols.extend(
                        [
                            c
                            for c in specialist_df.columns
                            if c.startswith("mr_") or c.startswith("smc_")
                        ]
                    )

                    seen = set()
                    scalar_cols_unique = []
                    for c in scalar_cols:
                        if c not in seen:
                            seen.add(c)
                            scalar_cols_unique.append(c)

                    for col in scalar_cols_unique:
                        if col not in df.columns:
                            df[col] = specialist_df[col]
            except Exception:
                pass

            realized_returns = df["realized_return"].astype(float)
            meta_prob = df["meta_probability"].astype(float)

            event_mask = ~realized_returns.isna()
            n_events_total = int(event_mask.sum())
            if n_events_total == 0:
                raise ValueError("No labeled events found in labeled_data")

            eval_mask = event_mask.copy()

            holdout_start = config.get("holdout_start")
            holdout_fraction = config.get("holdout_fraction")

            if holdout_start is None and holdout_fraction is None:
                holdout_fraction = 0.30
                tprint_info(
                    "ℹ️ No hold-out specified; defaulting to holdout_fraction=0.30 (last 30% of labeled events)",
                )
            try:
                if holdout_start and isinstance(df.index, pd.DatetimeIndex):
                    holdout_ts = pd.to_datetime(holdout_start)
                    time_mask = df.index >= holdout_ts
                    eval_mask &= time_mask
                elif holdout_fraction is not None:
                    try:
                        frac = float(holdout_fraction)
                    except Exception:
                        frac = 0.0
                    if frac > 0.0 and frac < 1.0:
                        event_idx = df.index[event_mask]
                        n_events = int(event_idx.size)
                        n_holdout = max(1, int(round(n_events * frac)))
                        holdout_idx = event_idx[-n_holdout:]
                        time_mask = df.index.isin(holdout_idx)
                        eval_mask &= time_mask
            except Exception as e_sel:
                tprint_warning(f"⚠️ Hold-out selection failed ({e_sel}); using all labeled events")
                eval_mask = event_mask.copy()

            n_events = int(eval_mask.sum())
            if n_events == 0:
                raise ValueError("Hold-out selection produced zero events; adjust holdout_start/holdout_fraction")

            tprint_info(
                f"📊 Meta-gated backtest: using {n_events} events for evaluation (total_labeled={n_events_total})"
            )

            eval_start_date = None
            eval_end_date = None
            eval_num_days = None
            if isinstance(df.index, pd.DatetimeIndex):
                eval_index = df.index[eval_mask]
                if len(eval_index) > 0:
                    eval_start_date = eval_index[0].date()
                    eval_end_date = eval_index[-1].date()
                    eval_num_days = int((eval_end_date - eval_start_date).days) + 1
                    if eval_num_days <= 0:
                        eval_num_days = 1

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
            event_probs = meta_prob.loc[eval_mask]
            event_returns = realized_returns.loc[eval_mask]

            base_n_events = int(event_returns.size)
            base_mean_ret = float(event_returns.mean()) if base_n_events > 0 else 0.0
            base_std_ret = float(event_returns.std(ddof=1)) if base_n_events > 1 else 0.0
            if base_std_ret > 0.0 and base_n_events > 0:
                base_sharpe_trade = float(base_mean_ret / base_std_ret) * float(np.sqrt(base_n_events))
            else:
                base_sharpe_trade = 0.0

            equity_base = (1.0 + event_returns).cumprod()
            running_max_base = equity_base.cummax()
            drawdown_base = equity_base / running_max_base - 1.0
            max_drawdown_base = float(drawdown_base.min()) if drawdown_base.size > 0 else 0.0

            base_hit_rate = float((event_returns > 0).mean()) if base_n_events > 0 else 0.0
            try:
                base_q05 = float(event_returns.quantile(0.05))
                base_q25 = float(event_returns.quantile(0.25))
                base_q50 = float(event_returns.quantile(0.50))
                base_q75 = float(event_returns.quantile(0.75))
                base_q95 = float(event_returns.quantile(0.95))
            except Exception:
                base_q05 = base_q25 = base_q50 = base_q75 = base_q95 = 0.0

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
                
                # New: Liquidity regime filter configuration
                use_liquidity_filter = bool(filters_cfg.get("use_liquidity_regime_filter", False))
                liquidity_regime_threshold = float(filters_cfg.get("liquidity_regime_threshold", 0.7))
                preferred_liquidity_regimes = filters_cfg.get("preferred_liquidity_regimes", [])

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

                # New: Apply liquidity regime filter if enabled
                if use_liquidity_filter:
                    # Find liquidity regime probability columns
                    liquidity_cols = [
                        c for c in df_events.columns 
                        if c.startswith('liquidity_liquidity_regime_') and 'prob_' in c
                    ]
                    
                    if liquidity_cols:
                        tprint_info(f"💧 Applying liquidity regime filter with {len(liquidity_cols)} regime columns")
                        
                        if preferred_liquidity_regimes:
                            # Filter for specific preferred regimes
                            preferred_cols = [
                                c for c in liquidity_cols 
                                if any(f"_{reg}_" in c for reg in preferred_liquidity_regimes)
                            ]
                            if preferred_cols:
                                # Create mask for any preferred regime above threshold
                                liquidity_mask = df_events[preferred_cols].fillna(0).max(axis=1) >= liquidity_regime_threshold
                                gate_mask &= liquidity_mask
                                tprint_info(f"   ↪ Using preferred regimes {preferred_liquidity_regimes} with threshold {liquidity_regime_threshold}")
                            else:
                                tprint_warning(f"   ⚠️ Preferred liquidity regimes {preferred_liquidity_regimes} not found in data")
                        else:
                            # General liquidity quality filter: require at least one regime with high probability
                            max_liquidity_prob = df_events[liquidity_cols].fillna(0).max(axis=1)
                            liquidity_mask = max_liquidity_prob >= liquidity_regime_threshold
                            gate_mask &= liquidity_mask
                            tprint_info(f"   ↪ Using general liquidity filter with threshold {liquidity_regime_threshold}")
                        
                        n_liquidity_filtered = (~liquidity_mask).sum()
                        tprint_info(f"   ↪ Liquidity filter excluded {n_liquidity_filtered} events")
                    else:
                        tprint_warning("⚠️ No liquidity regime probability columns found for filtering")
                else:
                    tprint_info("ℹ️ Liquidity regime filter disabled")

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

            gated_start_date = None
            gated_end_date = None
            gated_num_days = None
            trades_per_day = None
            if isinstance(gated_returns.index, pd.DatetimeIndex) and n_trades > 0:
                trade_index = gated_returns.index.sort_values()
                gated_start_date = trade_index[0].date()
                gated_end_date = trade_index[-1].date()
                gated_num_days = int((gated_end_date - gated_start_date).days) + 1
                if gated_num_days <= 0:
                    gated_num_days = 1
                trades_per_day = float(n_trades) / float(gated_num_days)

            mean_ret = float(gated_returns.mean())
            std_ret = float(gated_returns.std(ddof=1)) if n_trades > 1 else 0.0
            sharpe_trade = float(mean_ret / std_ret) * np.sqrt(n_trades) if std_ret > 0 else 0.0

            hit_rate = float((gated_returns > 0).mean())
            try:
                q05 = float(gated_returns.quantile(0.05))
                q25 = float(gated_returns.quantile(0.25))
                q50 = float(gated_returns.quantile(0.50))
                q75 = float(gated_returns.quantile(0.75))
                q95 = float(gated_returns.quantile(0.95))
            except Exception:
                q05 = q25 = q50 = q75 = q95 = 0.0

            # Simple trade-level equity curve (event-time, not bar-time)
            equity = (1.0 + gated_returns).cumprod()
            running_max = equity.cummax()
            drawdown = equity / running_max - 1.0
            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            tprint_info(
                f"📊 Meta-gated trades: {n_trades} | mean={mean_ret:.4f} | Sharpe(trade)={sharpe_trade:.3f} | maxDD={max_drawdown:.2%}"
            )

            if eval_start_date is not None and eval_end_date is not None and eval_num_days is not None:
                tprint_info(
                    f"📅 Evaluation period: {eval_start_date} → {eval_end_date} ({eval_num_days} days)"
                )
            if trades_per_day is not None and gated_start_date is not None and gated_end_date is not None and gated_num_days is not None:
                tprint_info(
                    f"📅 Gated trading period: {gated_start_date} → {gated_end_date} ({gated_num_days} days, ~{trades_per_day:.2f} trades/day)"
                )

            def _bootstrap_ci_mean(arr: np.ndarray, n_boot: int = 200, alpha: float = 0.05) -> tuple[float, float]:
                if arr.size == 0:
                    return float("nan"), float("nan")
                rng = np.random.default_rng(42)
                means = np.empty(n_boot, dtype=float)
                n_local = arr.size
                for i in range(n_boot):
                    idx = rng.integers(0, n_local, size=n_local)
                    means[i] = float(arr[idx].mean())
                lower = float(np.quantile(means, alpha / 2.0))
                upper = float(np.quantile(means, 1.0 - alpha / 2.0))
                return lower, upper

            mean_ci_low, mean_ci_high = _bootstrap_ci_mean(gated_returns.to_numpy(dtype=float)) if n_trades >= 20 else (float("nan"), float("nan"))

            temporal_segments = []
            try:
                n_segments = int(config.get("temporal_segments", 5))
            except Exception:
                n_segments = 5
            if n_segments > 1 and n_trades >= n_segments:
                idx_sorted = gated_returns.index.sort_values()
                seg_size = int(np.ceil(float(len(idx_sorted)) / float(n_segments)))
                for seg_idx in range(n_segments):
                    start = seg_idx * seg_size
                    if start >= len(idx_sorted):
                        break
                    end = min(len(idx_sorted), (seg_idx + 1) * seg_size)
                    seg_index = idx_sorted[start:end]
                    seg_ret = gated_returns.loc[seg_index]
                    if seg_ret.size == 0:
                        continue
                    seg_mean = float(seg_ret.mean())
                    seg_std = float(seg_ret.std(ddof=1)) if seg_ret.size > 1 else 0.0
                    if seg_std > 0.0:
                        seg_sharpe = float(seg_mean / seg_std) * float(np.sqrt(seg_ret.size))
                    else:
                        seg_sharpe = 0.0
                    temporal_segments.append(
                        {
                            "segment": seg_idx + 1,
                            "n_trades": int(seg_ret.size),
                            "mean_return": seg_mean,
                            "sharpe_trade": seg_sharpe,
                        }
                    )

            per_regime_metrics = {}
            try:
                if "hmm_regime_label_1h" in df.columns:
                    regimes_all = df.loc[event_returns.index, "hmm_regime_label_1h"]
                    regimes_trades = regimes_all[gate_mask]
                    for reg_val in pd.unique(regimes_trades.dropna()):
                        reg_mask = regimes_trades == reg_val
                        n_reg = int(reg_mask.sum())
                        if n_reg < 10:
                            continue
                        idx_reg = regimes_trades.index[reg_mask]
                        ret_reg = gated_returns.loc[idx_reg]
                        if ret_reg.size == 0:
                            continue
                        mean_reg = float(ret_reg.mean())
                        std_reg = float(ret_reg.std(ddof=1)) if ret_reg.size > 1 else 0.0
                        if std_reg > 0.0:
                            sharpe_reg = float(mean_reg / std_reg) * float(np.sqrt(ret_reg.size))
                        else:
                            sharpe_reg = 0.0
                        per_regime_metrics[str(reg_val)] = {
                            "n_trades": n_reg,
                            "mean_return": mean_reg,
                            "sharpe_trade": sharpe_reg,
                        }
            except Exception:
                per_regime_metrics = {}

            tx_cost = float(meta_gating.get("transaction_cost", 0.0))
            cost_stress = []
            if n_trades > 0 and tx_cost > 0.0:
                for mult in (1.0, 2.0, 3.0):
                    extra_cost = tx_cost * (mult - 1.0)
                    stressed = gated_returns - extra_cost
                    mean_s = float(stressed.mean())
                    std_s = float(stressed.std(ddof=1)) if stressed.size > 1 else 0.0
                    if std_s > 0.0:
                        sharpe_s = float(mean_s / std_s) * float(np.sqrt(stressed.size))
                    else:
                        sharpe_s = 0.0
                    cost_stress.append(
                        {
                            "multiplier": mult,
                            "mean_return": mean_s,
                            "sharpe_trade": sharpe_s,
                        }
                    )

            # Optional permutation test: shuffle event_returns and re-apply same gate to
            # verify that performance collapses toward noise under label randomization.
            permutation_results = []
            try:
                if bool(config.get("permutation_test", False)):
                    n_perm = int(config.get("permutation_repeats", 1) or 1)
                    if n_perm < 1:
                        n_perm = 1

                    rng = np.random.default_rng(42)
                    base_array = event_returns.to_numpy(dtype=float)
                    base_index = event_returns.index

                    for i in range(n_perm):
                        perm_idx = rng.permutation(base_array.size)
                        perm_series = pd.Series(base_array[perm_idx], index=base_index)
                        perm_gated = perm_series[gate_mask]
                        n_perm_trades = int(perm_gated.size)
                        if n_perm_trades == 0:
                            mean_perm = 0.0
                            std_perm = 0.0
                            sharpe_perm = 0.0
                            hit_perm = 0.0
                        else:
                            mean_perm = float(perm_gated.mean())
                            std_perm = float(perm_gated.std(ddof=1)) if n_perm_trades > 1 else 0.0
                            if std_perm > 0.0:
                                sharpe_perm = float(mean_perm / std_perm) * float(np.sqrt(n_perm_trades))
                            else:
                                sharpe_perm = 0.0
                            hit_perm = float((perm_gated > 0).mean())
                        permutation_results.append(
                            {
                                "run": i + 1,
                                "n_trades": n_perm_trades,
                                "mean_return": mean_perm,
                                "sharpe_trade": sharpe_perm,
                                "hit_rate": hit_perm,
                            }
                        )
            except Exception:
                permutation_results = []

            # Optional forward-walk evaluation over explicit calendar windows
            forward_walk_windows_metrics = []
            try:
                fw_cfg = config.get("forward_walk_windows")
                if fw_cfg is None:
                    try:
                        n_fw = int(config.get("forward_walk_n_windows", 0) or 0)
                    except Exception:
                        n_fw = 0
                    if n_fw > 0 and isinstance(event_returns.index, pd.DatetimeIndex):
                        idx_sorted = event_returns.index.sort_values()
                        n_idx = idx_sorted.size
                        if n_idx > 0:
                            edges = np.linspace(0, n_idx, n_fw + 1, dtype=int)
                            fw_cfg = []
                            for i in range(n_fw):
                                start_i = edges[i]
                                end_i = edges[i + 1] - 1
                                if start_i >= n_idx:
                                    continue
                                if end_i < start_i:
                                    end_i = start_i
                                if end_i >= n_idx:
                                    end_i = n_idx - 1
                                start_ts = idx_sorted[start_i]
                                end_ts = idx_sorted[end_i]
                                fw_cfg.append(
                                    {
                                        "start": str(start_ts.date()),
                                        "end": str(end_ts.date()),
                                        "label": f"FW{i + 1}",
                                    }
                                )
                if isinstance(fw_cfg, list) and fw_cfg and isinstance(event_returns.index, pd.DatetimeIndex):
                    for idx_fw, win in enumerate(fw_cfg):
                        if not isinstance(win, dict):
                            continue
                        start_str = win.get("start")
                        end_str = win.get("end")
                        if not start_str or not end_str:
                            continue
                        try:
                            start_ts = pd.to_datetime(start_str)
                            end_ts = pd.to_datetime(end_str)
                        except Exception:
                            continue

                        time_mask = (event_returns.index >= start_ts) & (event_returns.index <= end_ts)
                        if not bool(time_mask.any()):
                            continue

                        # Baseline events in this window
                        base_win = event_returns[time_mask]
                        n_events_win = int(base_win.size)
                        base_mean_win = float(base_win.mean()) if n_events_win > 0 else 0.0
                        base_std_win = float(base_win.std(ddof=1)) if n_events_win > 1 else 0.0
                        if base_std_win > 0.0 and n_events_win > 0:
                            base_sharpe_win = float(base_mean_win / base_std_win) * float(np.sqrt(n_events_win))
                        else:
                            base_sharpe_win = 0.0
                        base_hit_win = float((base_win > 0).mean()) if n_events_win > 0 else 0.0

                        # Gated trades in this window
                        gate_time_mask = gate_mask & time_mask
                        gated_win = event_returns[gate_time_mask]
                        n_trades_win = int(gated_win.size)
                        if n_trades_win > 0:
                            mean_win = float(gated_win.mean())
                            std_win = float(gated_win.std(ddof=1)) if n_trades_win > 1 else 0.0
                            if std_win > 0.0:
                                sharpe_win = float(mean_win / std_win) * float(np.sqrt(n_trades_win))
                            else:
                                sharpe_win = 0.0
                            hit_win = float((gated_win > 0).mean())
                        else:
                            mean_win = 0.0
                            std_win = 0.0
                            sharpe_win = 0.0
                            hit_win = 0.0

                        label = win.get("label") or f"window_{idx_fw + 1}"
                        forward_walk_windows_metrics.append(
                            {
                                "label": str(label),
                                "start": str(start_ts.date()),
                                "end": str(end_ts.date()),
                                "n_events": n_events_win,
                                "n_trades": n_trades_win,
                                "mean_return_gated": mean_win,
                                "sharpe_trade_gated": sharpe_win,
                                "hit_rate_gated": hit_win,
                                "base_mean_return": base_mean_win,
                                "base_sharpe_trade": base_sharpe_win,
                                "base_hit_rate": base_hit_win,
                            }
                        )
            except Exception:
                forward_walk_windows_metrics = []

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
                f.write(f"- Events (labeled, evaluation set): {n_events}\n")
                f.write(f"- Events (labeled, total): {n_events_total}\n")
                f.write(f"- Trades (gated): {n_trades}\n")
                if eval_start_date is not None and eval_end_date is not None and eval_num_days is not None:
                    f.write(f"- Evaluation period: {eval_start_date} → {eval_end_date} ({eval_num_days} days)\n")
                if trades_per_day is not None and gated_start_date is not None and gated_end_date is not None and gated_num_days is not None:
                    f.write(f"- Gated trading period: {gated_start_date} → {gated_end_date} ({gated_num_days} days, ~{trades_per_day:.2f} trades/day)\n")
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
                f.write(f"- Hit Rate (gated trades): {hit_rate:.2%}\n")
                f.write(f"- Mean Return CI (bootstrap, 95%): [{mean_ci_low:.4%}, {mean_ci_high:.4%}]\n")
                f.write("\n## Baseline (Ungated) Event Performance\n\n")
                f.write(f"- Events in evaluation set: {base_n_events}\n")
                f.write(f"- Mean Return per Event: {base_mean_ret:.4%}\n")
                f.write(f"- Std Dev per Event: {base_std_ret:.4%}\n")
                f.write(f"- Trade-Level Sharpe (sqrt(N)): {base_sharpe_trade:.3f}\n")
                f.write(f"- Max Drawdown (event-time equity): {max_drawdown_base:.2%}\n")
                f.write(f"- Hit Rate (events): {base_hit_rate:.2%}\n")
                f.write(f"- Return Quantiles (events): 5%={base_q05:.4%}, 25%={base_q25:.4%}, 50%={base_q50:.4%}, 75%={base_q75:.4%}, 95%={base_q95:.4%}\n")
                f.write("\n## Gated Return Distribution\n\n")
                f.write(f"- Return Quantiles (gated trades): 5%={q05:.4%}, 25%={q25:.4%}, 50%={q50:.4%}, 75%={q75:.4%}, 95%={q95:.4%}\n")
                if temporal_segments:
                    f.write("\n## Temporal Stability (event-time segments)\n\n")
                    f.write("| Segment | Trades | Mean Return | Sharpe (trade) |\n")
                    f.write("|---------|--------|------------|----------------|\n")
                    for seg in temporal_segments:
                        f.write(
                            f"| {seg['segment']} | {seg['n_trades']} | {seg['mean_return']:.4%} | {seg['sharpe_trade']:.3f} |\n"
                        )
                if per_regime_metrics:
                    f.write("\n## Per-Regime Performance (gated trades)\n\n")
                    f.write("| Regime | Trades | Mean Return | Sharpe (trade) |\n")
                    f.write("|--------|--------|------------|----------------|\n")
                    for reg_key, m in per_regime_metrics.items():
                        f.write(
                            f"| {reg_key} | {int(m['n_trades'])} | {float(m['mean_return']):.4%} | {float(m['sharpe_trade']):.3f} |\n"
                        )
                if forward_walk_windows_metrics:
                    f.write("\n## Forward-Walk Performance (evaluation windows)\n\n")
                    f.write("Each window is evaluated with the same meta gate and filters, restricted to the specified calendar range within the evaluation set.\n\n")
                    f.write("| Window | Start | End | Events | Trades | Mean Return (gated) | Sharpe (gated) | Hit Rate (gated) | Mean Return (base) | Sharpe (base) | Hit Rate (base) |\n")
                    f.write("|--------|-------|-----|--------|--------|----------------------|----------------|-------------------|--------------------|---------------|-----------------|\n")
                    for fw in forward_walk_windows_metrics:
                        f.write(
                            f"| {fw['label']} | {fw['start']} | {fw['end']} | {int(fw['n_events'])} | {int(fw['n_trades'])} | {float(fw['mean_return_gated']):.4%} | {float(fw['sharpe_trade_gated']):.3f} | {float(fw['hit_rate_gated']):.2%} | {float(fw['base_mean_return']):.4%} | {float(fw['base_sharpe_trade']):.3f} | {float(fw['base_hit_rate']):.2%} |\n"
                        )
                if permutation_results:
                    f.write("\n## Permutation Test (label-randomized returns)\n\n")
                    f.write("Randomly permuted realized returns with the same meta gate applied.\n\n")
                    f.write("| Run | Trades | Mean Return | Sharpe (trade) | Hit Rate |\n")
                    f.write("|-----|--------|------------|----------------|----------|\n")
                    for pr in permutation_results:
                        f.write(
                            f"| {int(pr['run'])} | {int(pr['n_trades'])} | {float(pr['mean_return']):.4%} | {float(pr['sharpe_trade']):.3f} | {float(pr['hit_rate']):.2%} |\n"
                        )
                if cost_stress:
                    f.write("\n## Transaction Cost Stress Test\n\n")
                    f.write("Multiplier refers to scaling of baseline transaction_cost used in labeling.\n\n")
                    f.write("| Cost Multiplier | Mean Return | Sharpe (trade) |\n")
                    f.write("|----------------|------------|----------------|\n")
                    for cs in cost_stress:
                        f.write(
                            f"| {cs['multiplier']:.1f} | {cs['mean_return']:.4%} | {cs['sharpe_trade']:.3f} |\n"
                        )

                if backtest_metrics_cfg:
                    auc_oof = float(backtest_metrics_cfg.get("auc_oof", 0.0))
                    mean_return_gated_diag = float(backtest_metrics_cfg.get("mean_return_gated", 0.0))
                    sharpe_gated_diag = float(backtest_metrics_cfg.get("sharpe_gated", 0.0))
                    trades_gated_diag = int(backtest_metrics_cfg.get("trades_gated", 0))

                    avg_trades_per_day_diag = None
                    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 2:
                        start_day = df.index[0].date()
                        end_day = df.index[-1].date()
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

            tprint_success(f"\x0f Meta-gated backtest report saved to: {filepath}")

            metrics: Dict[str, Any] = {
                "n_events": n_events,
                "n_events_total": n_events_total,
                "n_trades_gated": n_trades,
                "eval_start_date": str(eval_start_date) if eval_start_date is not None else None,
                "eval_end_date": str(eval_end_date) if eval_end_date is not None else None,
                "eval_num_days": eval_num_days,
                "gated_start_date": str(gated_start_date) if gated_start_date is not None else None,
                "gated_end_date": str(gated_end_date) if gated_end_date is not None else None,
                "gated_num_days": gated_num_days,
                "trades_per_day": trades_per_day,
                "mean_return_gated": mean_ret,
                "std_return_gated": std_ret,
                "sharpe_trade": sharpe_trade,
                "max_drawdown_event_time": max_drawdown,
                "hit_rate_gated": hit_rate,
                "mean_return_ci_low": mean_ci_low,
                "mean_return_ci_high": mean_ci_high,
                "base_mean_return": base_mean_ret,
                "base_std_return": base_std_ret,
                "base_sharpe_trade": base_sharpe_trade,
                "base_max_drawdown_event_time": max_drawdown_base,
                "base_hit_rate": base_hit_rate,
                "coverage_gated": float(n_trades) / float(base_n_events) if base_n_events > 0 else 0.0,
                "prob_threshold": prob_threshold,
                "use_expected_return": use_expected_return,
                "expected_return_threshold": er_threshold,
                "forward_walk_windows": forward_walk_windows_metrics,
                "permutation_results": permutation_results,
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
