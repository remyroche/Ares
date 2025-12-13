"""
Meta-Gated Backtest Step.

This step evaluates a meta-gated strategy using the same artifacts
that will be used live:

- Labeled data from FeatureGenerationMetaLabelingStep OR WeightedMetaLabelingStep
- meta_gating_config.json produced by that step
- Iso regressor artifact referenced in meta_gating_config

The backtest operates at the event level:
- Each labeled event corresponds to one potential trade
- The meta gate (probability + expected-return thresholds) decides
  whether the trade would be taken
- The realized_return from labeling is used as the trade PnL

This mirrors the live decision rule that gates entries on meta
probabilities and isotonic expected returns.

Supports both standard meta-labeling and weighted meta-labeling (sample-weighted HPO).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
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
from src.utils.ml_common.bagged_probability_aggregator import evaluate_prob_variants
from src.training.steps.labeling.snr_diagnostics import run_full
from src.utils.pipeline_standards import PipelineStandards
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

# Import post-HPO evaluation for enhanced metrics
try:
    from src.training.steps.labeling.post_hpo_model_evaluation import (
        compute_calibration_metrics,
        compute_snr_diagnostics,
        compute_backtest_metrics,
    )
    POST_HPO_EVAL_AVAILABLE = True
except ImportError:
    POST_HPO_EVAL_AVAILABLE = False


logger = logging.getLogger(__name__)


class MetaGatedBacktestStep(BaseStep):
    """Meta-gated event-level backtest using meta-labeling artifacts.
    
    Supports both standard meta-labeling and weighted meta-labeling pipelines.
    """

    def __init__(self, step_name: str = "meta_gated_backtest"):
        tprint_info("🔧 MetaGatedBacktestStep.__init__() called")
        super().__init__(step_name)
        self.logger = system_logger.getChild("MetaGatedBacktest")
    
    def _load_weighted_hpo_artifacts(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        direction: str = "long",
    ) -> Optional[Dict[str, Any]]:
        """
        Load artifacts from weighted meta-labeling HPO step.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe string
            direction: Trading direction
        
        Returns:
            Dict with HPO artifacts or None if not found
        """
        tprint_info("🔧 _load_weighted_hpo_artifacts() called")
        tprint_info(f"   symbol={symbol}, exchange={exchange}, timeframe={timeframe}, direction={direction}")
        
        # Use PipelineStandards for standardized path
        candidate_dirs = []
        try:
            base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
            candidate_dirs.append(Path(base_dir) / "post_hpo_evaluation")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to build standardized path: {e}, falling back to defaults")
        candidate_dirs.append(Path("outcomes"))
        
        # Collect matching files across all candidate directories
        patterns = [
            f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{direction}_*.json",
            f"meta_labeling_hpo_best_params_{symbol}_{exchange}_{timeframe}_{direction}_*.json",
        ]
        json_files = []
        for d in candidate_dirs:
            if not d.exists():
                continue
            for pat in patterns:
                json_files.extend(d.glob(pat))
        
        json_files = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not json_files:
            tprint_info("   ℹ️ No weighted HPO artifacts found")
            return None
        
        latest_json = json_files[0]
        tprint_info(f"   📂 Found weighted HPO artifacts: {latest_json.name}")
        
        try:
            with open(latest_json, "r") as f:
                hpo_data = json.load(f)
            
            tprint_success(f"   ✅ Loaded weighted HPO artifacts from {latest_json}")
            return {
                "best_params": hpo_data.get("best_params", {}),
                "best_score": hpo_data.get("best_score", 0),
                "best_edge": hpo_data.get("best_edge", 0),
                "diagnostics": hpo_data.get("best_config_diagnostics", {}),
                "gate_stats": hpo_data.get("gate_stats", {}),
                "source_file": str(latest_json),
            }
        except Exception as e:
            tprint_error(f"   ❌ Failed to load weighted HPO artifacts: {e}")
            return None
    
    def _compute_enhanced_backtest_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        returns: np.ndarray,
        threshold: float = 0.5,
        transaction_cost: float = 0.0,
        direction: str = "long",
    ) -> Dict[str, Any]:
        """
        Compute enhanced backtest metrics using post-HPO evaluation module.
        
        Args:
            y_true: Binary labels
            y_prob: Predicted probabilities
            returns: Realized returns
            threshold: Probability threshold
            transaction_cost: Transaction cost rate
            direction: Trading direction
        
        Returns:
            Dict with enhanced metrics
        """
        tprint_info("🔧 _compute_enhanced_backtest_metrics() called")
        
        if not POST_HPO_EVAL_AVAILABLE:
            tprint_warning("   ⚠️ Post-HPO evaluation module not available")
            return {}
        
        try:
            calibration = compute_calibration_metrics(y_true, y_prob)
            snr = compute_snr_diagnostics(y_true, y_prob, returns, threshold)
            backtest = compute_backtest_metrics(
                y_prob, returns, threshold,
                transaction_cost=transaction_cost,
                direction=direction,
            )
            
            tprint_success("   ✅ Enhanced metrics computed")
            return {
                "calibration": calibration,
                "snr": snr,
                "backtest": backtest,
            }
        except Exception as e:
            tprint_error(f"   ❌ Failed to compute enhanced metrics: {e}")
            return {}

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a meta-gated backtest using meta-labeling artifacts.

        Supports both standard meta-labeling and weighted meta-labeling (sample-weighted HPO).

        Args:
            config: Configuration dictionary with at least:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe string (e.g., '15m')
                - direction: 'long', 'short', or 'both'
                - use_weighted_meta_labeling: If True, look for weighted HPO artifacts

        Returns:
            Dict with success flag, artifacts, metrics, and optional error.
        """
        tprint_info("🔧 MetaGatedBacktestStep.execute() called")
        
        symbol = config.get("symbol", "UNKNOWN")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        execution_mode = config.get("execution_mode", "light")
        use_weighted = config.get("use_weighted_meta_labeling", True)

        tprint(
            f"🧪 Starting meta-gated backtest for {symbol} {timeframe} {direction} (mode={execution_mode})",
            "INFO",
        )
        tprint_info(f"   use_weighted_meta_labeling={use_weighted}")
        
        # Try to load weighted HPO artifacts if enabled
        weighted_hpo_artifacts = None
        if use_weighted:
            weighted_hpo_artifacts = self._load_weighted_hpo_artifacts(symbol, exchange, timeframe, direction)
            if weighted_hpo_artifacts:
                tprint_success(f"   ✅ Using weighted meta-labeling artifacts (edge={weighted_hpo_artifacts.get('best_edge', 0):.4f})")
            else:
                tprint_info("   ℹ️ No weighted artifacts found, using standard meta-labeling")

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
            # Include exchange and direction in artifact name to avoid collisions
            artifact_name = f"labeled_data_{symbol}_{exchange}_{timeframe}_{direction}"
            tprint_info(f"🔎 Loading labeled data artifact: {artifact_name}")
            
            # Try with exchange/direction first, fallback to legacy name
            labeled_data = self._get_artifact(
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="features",
            )
            
            if labeled_data is None:
                # Fallback to legacy artifact name
                legacy_name = f"labeled_data_{symbol}_{timeframe}"
                tprint_info(f"   ℹ️ Trying legacy artifact name: {legacy_name}")
                labeled_data = self._get_artifact(
                    artifact_name=legacy_name,
                    artifact_type="data",
                    data_category="features",
                )

            if labeled_data is None:
                raise ValueError(f"Labeled data artifact '{artifact_name}' (or legacy '{legacy_name}') not found")

            if not isinstance(labeled_data, pd.DataFrame) or labeled_data.empty:
                raise ValueError(
                    f"Labeled data artifact '{artifact_name}' is empty or not a DataFrame"
                )

            df = labeled_data.copy()
            df = self._normalize_datetime_index(df, "labeled_data")
            df = df.sort_index()
            
            # Data hygiene: enforce non-NaN/finite checks on critical columns
            tprint_info("🔍 Validating data quality...")
            required_cols = ["realized_return", "meta_probability"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in labeled_data")
                
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    tprint_warning(f"   ⚠️ Found {nan_count} NaN values in '{col}'")
                
                if col == "realized_return":
                    # Check for infinite values
                    inf_count = np.isinf(df[col]).sum()
                    if inf_count > 0:
                        tprint_warning(f"   ⚠️ Found {inf_count} infinite values in '{col}', replacing with NaN")
                        df.loc[np.isinf(df[col]), col] = np.nan
            
            # Align indices: ensure all critical columns share the same index
            critical_cols = ["realized_return", "meta_probability"]
            common_idx = df.index
            for col in critical_cols:
                if col in df.columns:
                    col_idx = df[col].dropna().index
                    common_idx = common_idx.intersection(col_idx)
            
            if len(common_idx) == 0:
                raise ValueError("No valid overlapping indices found in critical columns")
            
            # Filter to common index
            df = df.loc[common_idx]
            tprint_info(f"   ✅ Data validation complete: {len(df)} rows after filtering")

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
                specialist_config.setdefault("enable_mean_reversion_specialist", False)

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
            
            # Deterministic holdout selection with timezone-safe handling
            try:
                if holdout_start and isinstance(df.index, pd.DatetimeIndex):
                    # Ensure timezone-aware comparison
                    holdout_ts = pd.to_datetime(holdout_start)
                    if df.index.tz is not None and holdout_ts.tz is None:
                        holdout_ts = holdout_ts.tz_localize(df.index.tz)
                    elif df.index.tz is None and holdout_ts.tz is not None:
                        holdout_ts = holdout_ts.tz_localize(None)
                    
                    time_mask = df.index >= holdout_ts
                    eval_mask &= time_mask
                    tprint_info(f"   📅 Holdout start: {holdout_start} -> {holdout_ts}")
                elif holdout_fraction is not None:
                    try:
                        frac = float(holdout_fraction)
                    except Exception:
                        frac = 0.0
                    if frac > 0.0 and frac < 1.0:
                        # Deterministic: sort index first, then take last N%
                        event_idx = df.index[event_mask].sort_values()
                        n_events = int(event_idx.size)
                        n_holdout = max(1, int(round(n_events * frac)))
                        holdout_idx = event_idx[-n_holdout:]
                        time_mask = df.index.isin(holdout_idx)
                        eval_mask &= time_mask
                        tprint_info(f"   📅 Holdout fraction: {frac:.1%} -> {n_holdout} events (last {n_holdout} of {n_events})")
                    else:
                        tprint_warning(f"   ⚠️ Invalid holdout_fraction={holdout_fraction}, using all events")
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
            tprint_info("📂 Loading meta_gating_config...")
            
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

            prob_threshold = float(entry_cfg.get("prob_threshold", 0.6))
            use_expected_return = bool(entry_cfg.get("use_expected_return", False))
            er_threshold = float(entry_cfg.get("expected_return_threshold", 0.0))
            
            # Get transaction cost for threshold optimization
            tx_cost = float(meta_gating.get("transaction_cost", 0.0))
            optimize_thresholds = bool(config.get("optimize_thresholds", False))
            # Default to calibration on; can be disabled via config if needed
            enable_calibration = bool(config.get("enable_calibration", True))
            
            # Override with weighted HPO parameters if available
            if weighted_hpo_artifacts:
                tprint_info("   🔄 Applying weighted HPO parameters...")
                best_params = weighted_hpo_artifacts.get("best_params", {})
                
                # Override probability threshold if specified in HPO params
                if "prob_threshold" in best_params:
                    prob_threshold = float(best_params["prob_threshold"])
                    tprint_info(f"      prob_threshold overridden: {prob_threshold:.3f}")
                
                if "er_threshold" in best_params:
                    er_threshold = float(best_params["er_threshold"])
                    tprint_info(f"      er_threshold overridden: {er_threshold:.4f}")
                
                # Log weighted HPO diagnostics if available
                diagnostics = weighted_hpo_artifacts.get("diagnostics", {})
                if diagnostics:
                    tprint_info("   📊 Weighted HPO Diagnostics Summary:")
                    for key in ["auc_full", "auc_filtered", "best_fold_auc", "brier_score"]:
                        if key in diagnostics.get("filtering_diagnostics", {}):
                            tprint_info(f"      {key}: {diagnostics['filtering_diagnostics'][key]}")

            tprint_info(f"   Final gating config: prob_thr={prob_threshold:.3f}, er_thr={er_threshold:.4f}")

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
            # Get raw probabilities first (before calibration)
            raw_event_probs = meta_prob.loc[eval_mask]
            event_returns = realized_returns.loc[eval_mask]
            
            # Apply calibration if requested (before threshold optimization)
            event_probs = raw_event_probs
            calibration_applied = False
            if enable_calibration and POST_HPO_EVAL_AVAILABLE:
                try:
                    tprint_info("🔧 Applying probability calibration...")
                    # Create a simple wrapper for calibration
                    class ProbCalibrator(BaseEstimator, ClassifierMixin):
                        def __init__(self):
                            self.calibrator = None
                            self.classes_ = np.array([0, 1])
                        
                        def fit(self, X, y):
                            from sklearn.isotonic import IsotonicRegression
                            self.calibrator = IsotonicRegression(out_of_bounds='clip')
                            # X is probabilities, y is binary labels
                            self.calibrator.fit(X.reshape(-1, 1), y)
                            return self
                        
                        def predict_proba(self, X):
                            calibrated = self.calibrator.predict(X.reshape(-1, 1))
                            calibrated = np.clip(calibrated, 0, 1)
                            return np.column_stack([1 - calibrated, calibrated])
                        
                        def predict(self, X):
                            proba = self.predict_proba(X)
                            return (proba[:, 1] >= 0.5).astype(int)
                    
                    # Get binary labels for calibration
                    if "binary_label" in df.columns:
                        binary_labels_cal = df.loc[eval_mask, "binary_label"].values
                        valid_cal_mask = ~np.isnan(binary_labels_cal) & ~np.isnan(raw_event_probs.values)
                        
                        if valid_cal_mask.sum() > 50:
                            calibrator = ProbCalibrator()
                            calibrator.fit(
                                raw_event_probs.values[valid_cal_mask],
                                binary_labels_cal[valid_cal_mask]
                            )
                            calibrated_probs_array = calibrator.predict_proba(raw_event_probs.values)[:, 1]
                            event_probs = pd.Series(calibrated_probs_array, index=raw_event_probs.index)
                            calibration_applied = True
                            tprint_success("   ✅ Calibration applied")
                        else:
                            tprint_warning("   ⚠️ Insufficient data for calibration")
                    else:
                        tprint_warning("   ⚠️ binary_label column not found, skipping calibration")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Calibration failed: {e}, using uncalibrated probabilities")

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

            # ------------------------------------------------------------------
            # 3a) Compare different meta-probability variants at fixed 0.6 gate
            # ------------------------------------------------------------------
            variant_probs: Dict[str, pd.Series] = {}
            variant_comparison_csv_path = None  # Track CSV artifact path

            try:
                # Canonical meta probability used for live gating
                variant_probs["meta_probability"] = meta_prob.loc[eval_mask]

                # Ensemble OOF-based probability (if present)
                if "meta_probability_ensemble" in df.columns:
                    variant_probs["ensemble"] = df["meta_probability_ensemble"].loc[eval_mask]

                # Bagged LGBM variants from meta-labeling step
                if "meta_probability_lgbm_bag_mean" in df.columns:
                    variant_probs["lgbm_bag_mean"] = df["meta_probability_lgbm_bag_mean"].loc[eval_mask]

                if "meta_probability_lgbm_bag_lower" in df.columns:
                    variant_probs["lgbm_bag_lower"] = df["meta_probability_lgbm_bag_lower"].loc[eval_mask]

                # Optional consensus variant if/when available
                if "meta_probability_lgbm_bag_consensus" in df.columns:
                    variant_probs["lgbm_bag_consensus"] = df["meta_probability_lgbm_bag_consensus"].loc[eval_mask]

                if variant_probs:
                    # Evaluate variants at multiple thresholds if optimization is enabled
                    # Expanded sweep for better trades/day and PnL/trade optimization
                    if optimize_thresholds:
                        # Broader threshold sweep: from 0.45 to 0.85 in 0.02 increments for better granularity
                        thresholds_to_compare = [round(x, 2) for x in np.arange(0.45, 0.86, 0.02)]
                        # Always include the configured threshold
                        if prob_threshold not in thresholds_to_compare:
                            thresholds_to_compare.append(prob_threshold)
                        thresholds_to_compare = sorted(list(set(thresholds_to_compare)))
                        tprint_info(f"🔍 Evaluating meta-probability variants at {len(thresholds_to_compare)} thresholds: {thresholds_to_compare[:5]}...{thresholds_to_compare[-5:]}")
                    else:
                        # Default expanded sweep even without optimization
                        thresholds_to_compare = [round(x, 2) for x in np.arange(0.50, 0.81, 0.05)]
                        tprint_info(f"🔍 Evaluating meta-probability variants at thresholds: {thresholds_to_compare}")
                    
                    all_comparisons = []
                    for thresh in thresholds_to_compare:
                        comparison_df = evaluate_prob_variants(
                            returns=event_returns,
                            prob_variants=variant_probs,
                            threshold=thresh,
                        )
                        if not comparison_df.empty:
                            comparison_df["threshold"] = thresh
                            all_comparisons.append(comparison_df)
                    
                    if all_comparisons:
                        comparison_df = pd.concat(all_comparisons, ignore_index=True)

                    if not comparison_df.empty:
                        # Log compact summary
                        for _, row in comparison_df.iterrows():
                            tprint_info(
                                "   ↪ Variant='{}' | trades={} | mean_ret={:.4%} | Sharpe={:.3f} | maxDD={:.2%} | hit={:.2%}".format(
                                    row["variant"],
                                    int(row["n_trades"]),
                                    float(row["mean_return"]),
                                    float(row["sharpe_trade"]),
                                    float(row["max_drawdown"]),
                                    float(row["hit_rate"]),
                                )
                            )

                        # Save CSV using standardized paths
                        try:
                            try:
                                base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
                                outcomes_dir = Path(base_dir) / "meta_gated_backtest"
                                outcomes_dir.mkdir(parents=True, exist_ok=True)
                            except Exception:
                                outcomes_dir = Path("outcomes")
                                outcomes_dir.mkdir(parents=True, exist_ok=True)
                            
                            ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                            csv_name = (
                                f"meta_model_variant_comparison_"
                                f"{symbol}_{exchange}_{timeframe}_{direction}_{ts_str}.csv"
                            )
                            csv_path = outcomes_dir / csv_name
                            comparison_df.to_csv(csv_path, index=False)
                            variant_comparison_csv_path = csv_path  # Store for artifact tracking
                            tprint_success(f"✅ Saved meta-model variant comparison CSV to {csv_path}")
                        except Exception as csv_exc:
                            tprint_warning(f"⚠️ Failed to save meta-model comparison CSV: {csv_exc}")

                    else:
                        tprint_warning("⚠️ No valid rows in meta-probability variant comparison (empty result)")
            except Exception as comp_exc:
                tprint_warning(f"⚠️ Meta-probability variant comparison skipped due to error: {comp_exc}")

            # Optimize thresholds if requested (using calibrated probabilities if available)
            optimal_threshold_result = None
            if optimize_thresholds:
                tprint_info("🔍 Optimizing probability threshold...")
                try:
                    event_probs_pre = event_probs
                    event_returns_pre = event_returns

                    min_trades = 0
                    try:
                        min_trades = int(entry_cfg.get("min_trades", 0) or 0)
                    except Exception:
                        min_trades = 0

                    min_trades_per_day_required = None
                    try:
                        cfg_min_tpd = config.get("threshold_optimization_min_trades_per_day")
                        if cfg_min_tpd is None:
                            cfg_min_tpd = config.get("deployment_min_trades_per_day")
                        if cfg_min_tpd is not None:
                            min_trades_per_day_required = float(cfg_min_tpd)
                    except Exception:
                        min_trades_per_day_required = None

                    min_trades_required = min_trades
                    if min_trades_per_day_required is not None:
                        try:
                            min_trades_required = max(
                                min_trades_required,
                                int(np.ceil(float(min_trades_per_day_required) * float(eval_num_days or 1))),
                            )
                        except Exception:
                            pass

                    if min_trades_required < 10:
                        min_trades_required = 10

                    expected_returns_pre = None
                    if use_expected_return and iso_model is not None:
                        try:
                            prob_array = event_probs_pre.to_numpy(dtype=float)
                            er_array = iso_model.predict(prob_array)
                            expected_returns_pre = pd.Series(er_array, index=event_probs_pre.index)
                        except Exception:
                            expected_returns_pre = None

                    def _gate_mask_for_threshold(thresh: float) -> pd.Series:
                        mask = (event_probs_pre >= float(thresh)).copy()
                        if expected_returns_pre is not None:
                            try:
                                mask &= expected_returns_pre >= float(er_threshold)
                            except Exception:
                                pass

                        try:
                            df_events_local = df.loc[event_probs_pre.index]

                            use_vol_filter = bool(filters_cfg.get("use_volatility_filter", True))
                            vol_quantile = float(filters_cfg.get("volatility_quantile", 0.40))
                            use_trend_filter = bool(filters_cfg.get("use_trend_filter", True))
                            trend_window = int(filters_cfg.get("trend_window", 20))
                            trend_min_abs = float(filters_cfg.get("trend_min_abs", 0.0))

                            use_liquidity_filter = bool(filters_cfg.get("use_liquidity_regime_filter", False))
                            liquidity_regime_threshold = float(filters_cfg.get("liquidity_regime_threshold", 0.7))
                            preferred_liquidity_regimes = filters_cfg.get("preferred_liquidity_regimes", [])

                            if use_vol_filter and "volatility_1d" in df_events_local.columns:
                                v = df_events_local["volatility_1d"].astype(float)
                                try:
                                    v_thr = v.quantile(vol_quantile)
                                except Exception:
                                    v_thr = v.quantile(0.40)
                                mask &= v >= v_thr

                            if use_trend_filter and "close" in df_events_local.columns:
                                close = df_events_local["close"].astype(float)
                                sma = close.rolling(trend_window, min_periods=trend_window // 2).mean()
                                trend = (close - sma) / sma
                                trend = trend.reindex(df_events_local.index)
                                mask &= trend.abs() >= trend_min_abs

                            if use_liquidity_filter:
                                liquidity_cols = [
                                    c
                                    for c in df_events_local.columns
                                    if c.startswith('liquidity_liquidity_regime_') and 'prob_' in c
                                ]

                                if liquidity_cols:
                                    if preferred_liquidity_regimes:
                                        preferred_cols = [
                                            c
                                            for c in liquidity_cols
                                            if any(f"_{reg}_" in c for reg in preferred_liquidity_regimes)
                                        ]
                                        if preferred_cols:
                                            liquidity_mask = (
                                                df_events_local[preferred_cols].fillna(0).max(axis=1)
                                                >= liquidity_regime_threshold
                                            )
                                            mask &= liquidity_mask
                                    else:
                                        max_liquidity_prob = df_events_local[liquidity_cols].fillna(0).max(axis=1)
                                        mask &= max_liquidity_prob >= liquidity_regime_threshold
                        except Exception:
                            pass

                        return mask

                    if len(event_probs_pre) > 100:
                        thresholds_to_test = np.linspace(0.35, 0.85, 26)
                        best_mean_ret = -np.inf
                        best_thresh = prob_threshold
                        best_coverage = 0.0
                        best_trades_per_day = float("nan")
                        best_n_trades = 0

                        for thresh in thresholds_to_test:
                            gate_mask_test = _gate_mask_for_threshold(float(thresh))
                            try:
                                n_trades_test = int(gate_mask_test.sum())
                            except Exception:
                                n_trades_test = 0

                            if n_trades_test < min_trades_required:
                                continue

                            gated_returns_test = event_returns_pre[gate_mask_test]
                            if gated_returns_test.size == 0:
                                continue

                            mean_ret_test = float(gated_returns_test.mean())
                            coverage_test = float(n_trades_test) / float(len(event_probs_pre))

                            trades_per_day_test = None
                            try:
                                if isinstance(gated_returns_test.index, pd.DatetimeIndex) and gated_returns_test.size > 0:
                                    idx_sorted = gated_returns_test.index.sort_values()
                                    start_day = idx_sorted[0].date()
                                    end_day = idx_sorted[-1].date()
                                    n_days_local = int((end_day - start_day).days) + 1
                                    if n_days_local <= 0:
                                        n_days_local = 1
                                    trades_per_day_test = float(n_trades_test) / float(n_days_local)
                            except Exception:
                                trades_per_day_test = None

                            if min_trades_per_day_required is not None and trades_per_day_test is not None:
                                if float(trades_per_day_test) < float(min_trades_per_day_required):
                                    continue

                            # Prefer higher mean return, with coverage as tiebreaker
                            if (mean_ret_test > best_mean_ret) or (
                                np.isfinite(best_mean_ret)
                                and np.isfinite(mean_ret_test)
                                and (abs(mean_ret_test - best_mean_ret) < 1e-8)
                                and (coverage_test > best_coverage)
                            ):
                                best_mean_ret = mean_ret_test
                                best_thresh = float(thresh)
                                best_coverage = coverage_test
                                best_trades_per_day = float(trades_per_day_test) if trades_per_day_test is not None else float("nan")
                                best_n_trades = n_trades_test

                        if np.isfinite(best_mean_ret) and float(best_thresh) != float(prob_threshold):
                            optimal_threshold_result = {
                                "original_threshold": float(prob_threshold),
                                "optimal_threshold": float(best_thresh),
                                "optimal_mean_return": float(best_mean_ret),
                                "optimal_coverage": float(best_coverage),
                                "optimal_trades_per_day": float(best_trades_per_day) if np.isfinite(best_trades_per_day) else None,
                                "optimal_n_trades": int(best_n_trades),
                                "min_trades_required": int(min_trades_required),
                                "min_trades_per_day_required": float(min_trades_per_day_required)
                                if min_trades_per_day_required is not None
                                else None,
                            }
                            prob_threshold = float(best_thresh)
                            tprint_success(
                                f"   ✅ Optimal threshold: {prob_threshold:.3f} "
                                f"(Mean return: {best_mean_ret:.4%}, Coverage: {best_coverage:.2%}, "
                                f"Trades/day: {best_trades_per_day if np.isfinite(best_trades_per_day) else float('nan'):.2f})"
                            )
                        else:
                            tprint_info(f"   ℹ️ Default threshold {prob_threshold:.3f} retained (no improvement found)")
                    else:
                        tprint_warning("   ⚠️ Insufficient data for threshold optimization")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Threshold optimization failed: {e}, using default")

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

            # ------------------------------------------------------------------
            # Position sizing and fee sensitivity analysis
            # ------------------------------------------------------------------
            tprint_info("🔄 Computing position sizing and fee sensitivity...")

            # Position sizing analysis (fixed fractional sizing at different levels)
            position_sizes = [0.01, 0.02, 0.05, 0.10, 0.20]  # 1% to 20% position sizes
            sizing_results = {}

            for pos_size in position_sizes:
                # Apply position sizing to returns
                sized_returns = gated_returns * pos_size
                sized_equity = (1.0 + sized_returns).cumprod()
                sized_sharpe = (sized_returns.mean() / sized_returns.std()) * np.sqrt(len(sized_returns)) if len(sized_returns) > 1 and sized_returns.std() > 0 else 0.0
                sized_max_dd = ((sized_equity / sized_equity.cummax()) - 1.0).min() if len(sized_equity) > 0 else 0.0

                sizing_results[f"pos_{pos_size:.2f}"] = {
                    "mean_return": float(sized_returns.mean()),
                    "sharpe": float(sized_sharpe),
                    "max_drawdown": float(sized_max_dd),
                    "final_equity": float(sized_equity.iloc[-1]) if len(sized_equity) > 0 else 1.0
                }

            # Fee sensitivity analysis (different transaction cost levels)
            fee_levels = [0.0005, 0.001, 0.002, 0.005, 0.01]  # 0.05% to 1% fees
            fee_results = {}

            for fee in fee_levels:
                # Apply fees to each trade (round trip)
                fee_adjusted_returns = gated_returns - (2 * fee)  # Bid-ask spread approximation
                fee_sharpe = (fee_adjusted_returns.mean() / fee_adjusted_returns.std()) * np.sqrt(len(fee_adjusted_returns)) if len(fee_adjusted_returns) > 1 and fee_adjusted_returns.std() > 0 else 0.0
                fee_hit_rate = (fee_adjusted_returns > 0).mean()
                profitable_trades = (fee_adjusted_returns > 0).sum()

                fee_results[f"fee_{fee:.4f}"] = {
                    "mean_return": float(fee_adjusted_returns.mean()),
                    "sharpe": float(fee_sharpe),
                    "hit_rate": float(fee_hit_rate),
                    "profitable_trades": int(profitable_trades),
                    "break_even_rate": float((fee_adjusted_returns > 0).mean())
                }

            tprint_info(f"   📊 Position sizing: {len(position_sizes)} levels tested")
            tprint_info(f"   📊 Fee sensitivity: {len(fee_levels)} levels tested")

            # ------------------------------------------------------------------
            # Regime and volatility bucket performance analysis
            # ------------------------------------------------------------------
            tprint_info("🔄 Computing regime/volatility bucket performance...")

            regime_performance = {}
            volatility_performance = {}

            # Check for regime columns
            regime_cols = [col for col in df.columns if 'regime' in col.lower()]
            if regime_cols:
                tprint_info(f"   📊 Analyzing performance across {len(regime_cols)} regime types")

                for regime_col in regime_cols:
                    if regime_col not in df.columns:
                        continue

                    regime_values = df[regime_col].dropna().unique()
                    regime_performance[regime_col] = {}

                    for regime_val in regime_values:
                        # Get trades for this regime
                        regime_mask = (df[regime_col] == regime_val) & gate_mask
                        regime_trades = event_returns[regime_mask]

                        if len(regime_trades) >= 5:  # Minimum trades for meaningful stats
                            regime_mean = float(regime_trades.mean())
                            regime_std = float(regime_trades.std()) if len(regime_trades) > 1 else 0.0
                            regime_sharpe = regime_mean / regime_std * np.sqrt(len(regime_trades)) if regime_std > 0 else 0.0
                            regime_hit_rate = float((regime_trades > 0).mean())

                            # Bootstrap CI for mean return
                            if len(regime_trades) >= 10:
                                ci_low, ci_high = _bootstrap_ci_mean(regime_trades.values)
                            else:
                                ci_low, ci_high = float('nan'), float('nan')

                            regime_performance[regime_col][str(regime_val)] = {
                                "n_trades": len(regime_trades),
                                "mean_return": regime_mean,
                                "sharpe": regime_sharpe,
                                "hit_rate": regime_hit_rate,
                                "ci_low": ci_low,
                                "ci_high": ci_high
                            }

            # Volatility bucket analysis
            if 'volatility_1d' in df.columns:
                try:
                    # Create volatility buckets
                    vol_buckets = pd.qcut(df['volatility_1d'].dropna(), 3, labels=['low_vol', 'med_vol', 'high_vol'])
                    df_with_vol = df.copy()
                    df_with_vol['vol_bucket'] = vol_buckets

                    for bucket in ['low_vol', 'med_vol', 'high_vol']:
                        bucket_mask = (df_with_vol['vol_bucket'] == bucket) & gate_mask
                        bucket_trades = event_returns[bucket_mask]

                        if len(bucket_trades) >= 5:
                            bucket_mean = float(bucket_trades.mean())
                            bucket_std = float(bucket_trades.std()) if len(bucket_trades) > 1 else 0.0
                            bucket_sharpe = bucket_mean / bucket_std * np.sqrt(len(bucket_trades)) if bucket_std > 0 else 0.0
                            bucket_hit_rate = float((bucket_trades > 0).mean())

                            # Bootstrap CI
                            if len(bucket_trades) >= 10:
                                ci_low, ci_high = _bootstrap_ci_mean(bucket_trades.values)
                            else:
                                ci_low, ci_high = float('nan'), float('nan')

                            volatility_performance[bucket] = {
                                "n_trades": len(bucket_trades),
                                "mean_return": bucket_mean,
                                "sharpe": bucket_sharpe,
                                "hit_rate": bucket_hit_rate,
                                "ci_low": ci_low,
                                "ci_high": ci_high
                            }

                    tprint_info(f"   📊 Volatility bucket analysis: {len(volatility_performance)} buckets with sufficient data")

                except Exception as vol_exc:
                    tprint_warning(f"   ⚠️ Volatility bucket analysis failed: {vol_exc}")

            tprint_info(f"   📊 Regime analysis: {sum(len(regimes) for regimes in regime_performance.values())} regime categories analyzed")

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
            # 4) Write Markdown report using standardized paths
            # ------------------------------------------------------------------
            try:
                base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
                outcomes_dir = Path(base_dir) / "meta_gated_backtest"
                outcomes_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                tprint_warning(f"   ⚠️ Failed to build standardized path: {e}, using fallback")
                outcomes_dir = Path("outcomes")
                outcomes_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meta_gated_backtest_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.md"
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
                if optimal_threshold_result:
                    f.write(f"- Threshold Optimization: Original={optimal_threshold_result['original_threshold']:.3f}, Optimal={optimal_threshold_result['optimal_threshold']:.3f}\n")
                    f.write(
                        f"- Optimal Threshold Metrics: MeanReturn={optimal_threshold_result.get('optimal_mean_return', float('nan')):.4%}, "
                        f"Coverage={optimal_threshold_result['optimal_coverage']:.2%}\n"
                    )
                f.write(f"- Calibration Applied: {calibration_applied}\n")
                f.write(f"- Use Expected Return: {use_expected_return}\n")
                if use_expected_return:
                    f.write(f"- Expected Return Threshold: {er_threshold:.4f} (fraction)\n")
                f.write(f"- Transaction Cost: {tx_cost:.4f}\n")
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

            tprint_success(f"📝 Meta-gated backtest report saved to: {filepath}")

            # ------------------------------------------------------------------
            # 5) Run SNR Diagnostics
            # ------------------------------------------------------------------
            tprint_info("🔧 Running SNR Diagnostics...")
            try:
                tprint("🔬 Running SNR Diagnostics...", "INFO")
                run_full(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction,
                    model="analyst",
                    cv_splits_learn=int(config.get("snr_cv_splits_learn", 3)),
                    cv_splits_robust=int(config.get("snr_cv_splits_robust", 5)),
                    prob_thresholds=config.get("snr_prob_thresholds", [0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
                )
                tprint_success("✅ SNR Diagnostics completed successfully")
            except Exception as e_snr:
                tprint_error(f"⚠️ SNR Diagnostics failed: {e_snr}")

            # ------------------------------------------------------------------
            # 6) Enhanced Backtest Metrics (if weighted meta-labeling)
            # ------------------------------------------------------------------
            enhanced_metrics = {}
            if weighted_hpo_artifacts and POST_HPO_EVAL_AVAILABLE:
                tprint_info("📊 Computing enhanced backtest metrics (weighted meta-labeling)...")
                
                # Get binary labels for enhanced metrics
                try:
                    binary_labels = df.get("binary_label", pd.Series(index=df.index, dtype=float))
                    y_true = binary_labels.loc[eval_mask].values
                    y_prob = event_probs.values
                    returns_arr = event_returns.values
                    
                    enhanced_metrics = self._compute_enhanced_backtest_metrics(
                        y_true=y_true,
                        y_prob=y_prob,
                        returns=returns_arr,
                        threshold=prob_threshold,
                        transaction_cost=tx_cost,
                        direction=direction,
                    )
                    
                    if enhanced_metrics:
                        tprint_success("✅ Enhanced metrics computed successfully")
                        
                        # Log key enhanced metrics
                        if "snr" in enhanced_metrics:
                            snr_data = enhanced_metrics["snr"]
                            tprint_info(f"   SNR: {snr_data.get('snr_positive', 'N/A'):.4f}")
                            tprint_info(f"   IC: {snr_data.get('information_coefficient', 'N/A'):.4f}")
                        
                        if "calibration" in enhanced_metrics:
                            calib_data = enhanced_metrics["calibration"]
                            tprint_info(f"   ECE: {calib_data.get('ece', 'N/A'):.4f}")
                            tprint_info(f"   Brier: {calib_data.get('brier', 'N/A'):.4f}")
                except Exception as enh_exc:
                    tprint_warning(f"⚠️ Enhanced metrics failed: {enh_exc}")

            tprint_info("📊 Assembling final metrics...")
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
                "threshold_optimization": optimal_threshold_result,
                "calibration_applied": calibration_applied,
                "use_expected_return": use_expected_return,
                "expected_return_threshold": er_threshold,
                "transaction_cost": tx_cost,
                "position_sizing_analysis": sizing_results,
                "fee_sensitivity_analysis": fee_results,
                "regime_performance": regime_performance,
                "volatility_performance": volatility_performance,
                "forward_walk_windows": forward_walk_windows_metrics,
                "permutation_results": permutation_results,
                # Enhanced metrics from post-HPO evaluation
                "enhanced_calibration": enhanced_metrics.get("calibration", {}),
                "enhanced_snr": enhanced_metrics.get("snr", {}),
                "enhanced_backtest": enhanced_metrics.get("backtest", {}),
                # Weighted HPO info
                "weighted_hpo_used": weighted_hpo_artifacts is not None,
                "weighted_hpo_edge": weighted_hpo_artifacts.get("best_edge", 0) if weighted_hpo_artifacts else None,
            }
            
            # Save JSON metrics artifact
            json_path = None
            try:
                json_filename = f"meta_gated_backtest_metrics_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.json"
                json_path = outcomes_dir / json_filename
                with open(json_path, "w") as f_json:
                    json.dump(metrics, f_json, indent=2, default=str)
                tprint_success(f"✅ Saved metrics JSON to {json_path}")
            except Exception as json_exc:
                tprint_warning(f"⚠️ Failed to save metrics JSON: {json_exc}")
            
            # Console summary of key KPIs
            tprint_info("\n" + "=" * 70)
            tprint_info("📊 META-GATED BACKTEST SUMMARY")
            tprint_info("=" * 70)
            tprint_info(f"   Trades: {n_trades} | Coverage: {metrics['coverage_gated']:.2%}")
            tprint_info(f"   Sharpe (trade): {sharpe_trade:.3f} | Mean Return: {mean_ret:.4%}")
            tprint_info(f"   Max Drawdown: {max_drawdown:.2%} | Hit Rate: {hit_rate:.2%}")
            if enhanced_metrics.get("backtest", {}).get("cost_adjusted_sharpe"):
                cost_sharpe = enhanced_metrics["backtest"]["cost_adjusted_sharpe"]
                tprint_info(f"   Cost-Adjusted Sharpe: {cost_sharpe:.3f}")
            if optimal_threshold_result:
                tprint_info(f"   Threshold: {prob_threshold:.3f} (optimized from {optimal_threshold_result['original_threshold']:.3f})")
            if calibration_applied:
                tprint_info(f"   ✅ Calibration applied")

            # ------------------------------------------------------------------
            # Deployment Guardrails
            # ------------------------------------------------------------------
            tprint_info("🛡️  Checking deployment guardrails...")

            guardrail_failures = []
            guardrail_warnings = []

            # Guardrail 1: Minimum Sharpe ratio
            min_sharpe_required = config.get("deployment_min_sharpe", 0.5)
            if sharpe_trade < min_sharpe_required:
                guardrail_failures.append(f"Sharpe ratio {sharpe_trade:.3f} < {min_sharpe_required:.3f}")

            # Guardrail 2: Minimum PnL per trade
            min_pnl_required = config.get("deployment_min_pnl_per_trade", 0.0001)  # 0.01%
            if mean_ret < min_pnl_required:
                guardrail_failures.append(f"PnL/trade {mean_ret:.4f} < {min_pnl_required:.4f}")

            # Guardrail 3: Minimum trades per day
            min_trades_per_day_required = config.get("deployment_min_trades_per_day", 10)
            if trades_per_day is not None and trades_per_day < min_trades_per_day_required:
                guardrail_warnings.append(f"Trades/day {trades_per_day:.1f} < {min_trades_per_day_required}")

            # Guardrail 4: Maximum drawdown
            max_dd_allowed = config.get("deployment_max_drawdown", 0.50)  # 50%
            if abs(max_drawdown) > max_dd_allowed:
                guardrail_failures.append(f"Max drawdown {abs(max_drawdown):.1%} > {max_dd_allowed:.1%}")

            # Guardrail 5: Minimum hit rate
            min_hit_rate_required = config.get("deployment_min_hit_rate", 0.45)
            if hit_rate < min_hit_rate_required:
                guardrail_warnings.append(f"Hit rate {hit_rate:.1%} < {min_hit_rate_required:.1%}")

            # Report guardrail status
            if guardrail_failures:
                tprint_error("❌ DEPLOYMENT BLOCKED - Guardrail failures:")
                for failure in guardrail_failures:
                    tprint_error(f"   {failure}")
                # Add to metrics for downstream processing
                metrics["deployment_blocked"] = True
                metrics["guardrail_failures"] = guardrail_failures
            else:
                tprint_success("✅ Deployment guardrails passed")
                metrics["deployment_blocked"] = False

            if guardrail_warnings:
                tprint_warning("⚠️  Deployment warnings (monitor closely):")
                for warning in guardrail_warnings:
                    tprint_warning(f"   {warning}")
                metrics["guardrail_warnings"] = guardrail_warnings

            tprint_info("=" * 70)
            
            tprint_success(f"✅ MetaGatedBacktestStep.execute() completed successfully")
            tprint_info(f"   Trades: {n_trades}, Sharpe: {sharpe_trade:.3f}, Mean Return: {mean_ret:.4%}")

            # Collect all artifact paths
            artifacts_dict = {
                "meta_gated_backtest_report": str(filepath),
            }
            
            # Add JSON metrics artifact if saved
            if json_path is not None and json_path.exists():
                artifacts_dict["meta_gated_backtest_metrics_json"] = str(json_path)
            
            # Add variant comparison CSV if saved
            if variant_comparison_csv_path is not None and variant_comparison_csv_path.exists():
                artifacts_dict["meta_model_variant_comparison_csv"] = str(variant_comparison_csv_path)
            
            return {
                "success": True,
                "artifacts": artifacts_dict,
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
