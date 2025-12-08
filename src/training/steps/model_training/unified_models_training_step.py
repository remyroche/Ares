"""
Unified Models Training Step.

This step consolidates all analyst and tactician training (base and ensemble)
into a single unified script that calls UnifiedTrainingPipeline.
"""

import asyncio
import yaml
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# HPO imports (only those actually used)
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psutil
import gc

from src.training.steps.model_training.hpo_config import (
    HPOOrchestrator,
    ModelParameterGroups
)

# Import centralized disagreement features calculator
from src.feature_generation.categories.ensemble_disagreement import calculate_ensemble_disagreement_features

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning, tprint_data_preview, tprint_feature_counts
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs

# Import model training report generator
from src.training.steps.model_training.model_training_report_generator import create_model_training_report

# Import dynamic config calculator
from src.training.steps.model_training.dynamic_config_calculator import (
    DynamicConfigCalculator, DynamicTrainingConfig
)

# Import temporal splitting for proper train/val/test separation
from src.utils.versioned_artifacts import (
    create_temporal_split_config_for_pipeline,
    create_walkforward_split_config_for_pipeline,
    get_data_for_purpose,
    TemporalSplitConfig,
    WalkForwardSplitConfig
)

# Confidence calibration and risk-aware confidence utilities
from src.utils.ml_common.confidence_metrics import calibrate_model_confidence, apply_risk_adjusted_confidence

# Import VersionedArtifactStore
from src.utils.versioned_artifacts.store import VersionedArtifactStore

# Try to import unified training pipeline if it exists, otherwise use placeholder
try:
    from src.training.steps.models_training.unified_training_pipeline import UnifiedTrainingPipeline
    import inspect
    try:
        print(f"DEBUG: UnifiedTrainingPipeline imported from: {inspect.getfile(UnifiedTrainingPipeline)}")
    except Exception as e:
        print(f"DEBUG: Could not get file for UnifiedTrainingPipeline: {e}")
    unified_pipeline_available = True
except ImportError as e:
    print(f"DEBUG: Failed to import UnifiedTrainingPipeline: {e}")
    import traceback
    traceback.print_exc()
    unified_pipeline_available = False
    tprint_info("UnifiedTrainingPipeline not available, using placeholder")


class UnifiedModelsTrainingStep(BaseStep):
    """
    Unified Models Training Step.

    Consolidates all analyst and tactician training (base and ensemble) into a single step
    that calls UnifiedTrainingPipeline with appropriate configuration based on training type.
    """

    def __init__(self, step_name: str = "unified_models_training"):
        """Initialize the unified models training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('UnifiedModelsTraining')
        print("DEBUG: UnifiedModelsTrainingStep initialized!")
        self.unified_pipeline = None
        self.param_groups_factory = ModelParameterGroups()
        self.hpo_orchestrator = None
        self._specialist_feature_names = []

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute unified model training.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')
                - training_type: Type of training ('analyst_base', 'analyst_ensemble', 'tactician_base', 'tactician_ensemble')
                - execution_mode: Execution mode ('full', 'light', 'blank')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        try:
            # Extract configuration
            symbol = config.get('symbol')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            execution_mode = config.get('execution_mode', 'full')
            training_type = config.get('training_type', 'analyst_base')
            direction = config.get('direction', 'long') # Keep direction for logging and context

            print(f"DEBUG: UnifiedModelsTrainingStep.execute called. training_type={training_type}") # Explicit print

            # Set context for artifact manager
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                model='Analyst' if 'analyst' in training_type else 'Tactician', # Use the existing logic for 'model'
                direction=direction # Add direction to context
            )

            print(f"DEBUG: UnifiedModelsTrainingStep.execute called with training_type={training_type}")
            tprint_info(f"🚀 Starting Unified Models Training (Type: {training_type}) for {symbol} {timeframe} {direction}")

            # Check if unified pipeline is available
            if not unified_pipeline_available:
                tprint_error("UnifiedTrainingPipeline not available - cannot train models")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': "UnifiedTrainingPipeline not available",
                    'training_type': training_type
                }
            
            # Initialize unified training pipeline
            self.unified_pipeline = UnifiedTrainingPipeline(self.logger)
            
            # Load appropriate YAML configuration
            yaml_config = await self._load_training_config(training_type, config)
            
            # CRITICAL: Merge runtime config into yaml_config to ensure all passed parameters 
            # (like ensemble_features, target_data) are available to the pipeline
            yaml_config.update(config)
            
            tprint_info(f"🚀 Starting {training_type} training...")
            tprint_info("=" * 80)
            tprint_info("📥 STEP 1: RETRIEVING TRAINING DATA FROM ARTIFACTS")
            tprint_info("=" * 80)
            training_data, analyst_targets, tactician_targets = await self._retrieve_training_data(config, yaml_config)

            # Apply robust, loss-aware preprocessing to targets to reduce the impact
            # of extreme outliers on regression models and HPO.
            robust_targets_enabled = bool(config.get('enable_robust_target_processing', True))
            if robust_targets_enabled:
                if analyst_targets is not None:
                    analyst_targets = self._apply_robust_target_transform(analyst_targets, name="analyst_targets")
                if tactician_targets is not None:
                    tactician_targets = self._apply_robust_target_transform(tactician_targets, name="tactician_targets")

            # Log initial dataset size before any filtering
            if training_data is not None:
                tprint_info("=" * 80)
                tprint_info("📊 INITIAL DATASET SIZE (Before Optimization)")
                tprint_info("=" * 80)
                tprint_info(f"   Training Data: {training_data.shape[0]:,} samples × {training_data.shape[1]:,} features")
                if analyst_targets is not None:
                    tprint_info(f"   Analyst Targets: {len(analyst_targets):,} samples")
                if tactician_targets is not None:
                    tprint_info(f"   Tactician Targets: {len(tactician_targets):,} samples")
                memory_before = training_data.memory_usage(deep=True).sum() / 1024**2
                tprint_info(f"   Memory Usage (Before): {memory_before:.2f} MB")
                if len(training_data) > 0:
                    tprint_info(f"   Date Range: {training_data.index[0]} to {training_data.index[-1]}")
                else:
                    tprint_warning("⚠️ Training data is empty (0 samples) - skipping date range logging")

                # Optimize memory usage (Float64→Float32, Int64→Int32/Int16/Int8)
                from src.utils.ml_common.training_optimizations import optimize_dataframe_memory
                tprint_info("🔧 Optimizing memory usage (precision reduction)...")
                training_data = optimize_dataframe_memory(training_data)
                memory_after = training_data.memory_usage(deep=True).sum() / 1024**2
                memory_reduction = (memory_before - memory_after) / memory_before * 100
                tprint_success(f"✅ Memory optimized: {memory_before:.2f} MB → {memory_after:.2f} MB ({memory_reduction:.1f}% reduction)")
                tprint_info("=" * 80)
            else:
                tprint_error("❌ No training data retrieved!")

            # ========================================================================
            # TEMPORAL SPLITTING: Enforce train/val/test boundaries
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info("📅 STEP 2: WALK-FORWARD VALIDATION SETUP")
            tprint_info("=" * 80)
            tprint_info("🔐 TEMPORAL DATA SPLITTING - Preventing Data Leakage")
            tprint_info("=" * 80)

            # Store full datasets before filtering (needed for validation period in HPO)
            self._full_training_data = training_data.copy() if training_data is not None else None
            self._full_analyst_targets = analyst_targets.copy() if analyst_targets is not None else None
            self._full_tactician_targets = tactician_targets.copy() if tactician_targets is not None else None

            if training_data is not None and len(training_data) > 0:
                # Create or load WALK-FORWARD temporal split configuration
                tprint_info(f"📅 Creating WALK-FORWARD split configuration for {symbol} {config.get('exchange', 'binance')}")

                # Determine data boundaries from actual data
                data_start = training_data.index.min()
                data_end = training_data.index.max()

                tprint_info(f"   Data range: {data_start} to {data_end}")
                tprint_info(f"   Total samples: {len(training_data)}")
                
                # ========================================================================
                # CRITICAL VALIDATION: Check dataset size and date range
                # ========================================================================
                # Validate date range is chronological
                if data_start >= data_end:
                    error_msg = (
                        f"❌ CRITICAL: Invalid date range in training data!\n"
                        f"   Start: {data_start}\n"
                        f"   End: {data_end}\n"
                        f"   Problem: Start date is not before end date.\n"
                        f"   This indicates corrupted or improperly sorted data."
                    )
                    tprint_error(error_msg)
                    raise ValueError(error_msg)
                
                # Calculate minimum required samples for temporal split with embargos
                # For 60/20/20 split with 1-day embargo, we need at least ~7 days
                # At 15-minute intervals: 7 days × 96 samples/day = 672 samples
                # We use 1000 as absolute minimum (10.4 days) for robust validation
                MIN_SAMPLES = 500  # Reduced for testing
                RECOMMENDED_DAYS = 30  # Recommended minimum for good train/val/test split
                
                # Handle both datetime and numeric indices
                if hasattr(data_end - data_start, 'days'):
                    # DatetimeIndex case
                    total_days = (data_end - data_start).days
                else:
                    # Numeric index case: estimate days based on samples and timeframe
                    timeframe = config.get('timeframe', '15m')
                    samples_per_day = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1}.get(timeframe, 96)
                    total_days = len(training_data) // samples_per_day
                
                if len(training_data) < MIN_SAMPLES:
                    error_msg = (
                        f"❌ CRITICAL: Dataset too small for training!\n"
                        f"   Current size: {len(training_data):,} samples ({total_days} days)\n"
                        f"   Minimum required: {MIN_SAMPLES:,} samples (~10 days)\n"
                        f"   \n"
                        f"   For proper temporal splitting with 1-day embargos:\n"
                        f"   - Recommended: {RECOMMENDED_DAYS}+ days\n"
                        f"   - Absolute minimum: {MIN_SAMPLES:,} samples\n"
                        f"   \n"
                        f"   Please ensure feature generation produced sufficient data."
                    )
                    tprint_error(error_msg)
                    raise ValueError(error_msg)
                
                # Warn if dataset is small but above minimum
                if total_days < RECOMMENDED_DAYS:
                    tprint_warning(
                        f"⚠️ WARNING: Dataset smaller than recommended for optimal temporal splitting\n"
                        f"   Current: {len(training_data):,} samples ({total_days} days)\n"
                        f"   Recommended: {RECOMMENDED_DAYS}+ days\n"
                        f"   Small datasets may reduce model generalization."
                    )
                
                tprint_success(f"✅ Dataset validation passed: {len(training_data):,} samples, {total_days} days")

                # Use Walk-Forward with embargo instead of a simple fixed split
                tprint_info("🔧 ENABLING PURGED WALK-FORWARD SPLIT WITH EMBARGO")

                # Create walk-forward configuration (expanding window) and final temporal split for test
                from src.utils.versioned_artifacts.temporal_splits import (
                    create_walkforward_split_config_for_pipeline,
                    create_temporal_split_config_for_pipeline,
                    WalkForwardSplitConfig,
                    TemporalSplitConfig,
                )

                # Parameters can be overridden via config
                n_folds = int(config.get('wf_n_folds', 3))
                val_pct_per_fold = float(config.get('wf_val_pct_per_fold', 0.10))
                final_test_pct = float(config.get('wf_final_test_pct', 0.15))
                embargo_days = int(config.get('wf_embargo_days', 1))

                execution_mode_local = str(config.get('execution_mode', 'full')).lower()

                if execution_mode_local == 'blank':
                    # In BLANK mode, always regenerate temporal splits from the current
                    # data window so we effectively use the full lookback range instead
                    # of any previously cached JSON created on a shorter sample.

                    walkforward_config = WalkForwardSplitConfig.create_expanding_window(
                        data_start=data_start,
                        data_end=data_end,
                        n_folds=n_folds,
                        val_pct_per_fold=val_pct_per_fold,
                        final_test_pct=final_test_pct,
                        min_train_pct=float(config.get('wf_min_train_pct', 0.55)),
                        embargo_days=embargo_days,
                    )

                    temporal_config = TemporalSplitConfig.create_from_data(
                        data_start=data_start,
                        data_end=data_end,
                        train_pct=0.6,
                        val_pct=0.2,
                        test_pct=0.2,
                        embargo_days=1,
                    )
                else:
                    # For non-BLANK modes, reuse cached configs when available.
                    walkforward_config = create_walkforward_split_config_for_pipeline(
                        symbol=symbol,
                        exchange=config.get('exchange', 'binance'),
                        timeframe=timeframe,
                        data_start=data_start,
                        data_end=data_end,
                        n_folds=n_folds,
                        val_pct_per_fold=val_pct_per_fold,
                        final_test_pct=final_test_pct,
                        min_train_pct=float(config.get('wf_min_train_pct', 0.55)),
                        embargo_days=embargo_days,
                    )

                    temporal_config = create_temporal_split_config_for_pipeline(
                        symbol=symbol,
                        exchange=config.get('exchange', 'binance'),
                        timeframe=timeframe,
                        data_start=data_start,
                        data_end=data_end,
                    )

                # Store configs for downstream consumers (HPO, pipeline, reports)
                self._walkforward_config = walkforward_config
                self._temporal_config = temporal_config
                config['walkforward_config'] = walkforward_config
                config['temporal_config'] = temporal_config

                # Log split summary
                tprint_info("=" * 80)
                tprint_info("📊 WALK-FORWARD CONFIGURATION (expanding)")
                tprint_info("=" * 80)
                tprint_info(f"   Folds: {len(walkforward_config.folds)} | Embargo days: {embargo_days}")
                tprint_info(f"   Final test period: {walkforward_config.test.start} → {walkforward_config.test.end}")
                for fold in walkforward_config.folds:
                    tprint_info(
                        f"   Fold {fold.fold_num}: train {fold.training.start} → {fold.training.effective_end} | "
                        f"val {fold.validation.start} → {fold.validation.effective_end}"
                    )
                tprint_info("=" * 80)

                # Do NOT filter data here; keep full dataset. Downstream training/HPO should respect the configs.
                training_data_filtered = training_data.copy()

                # Basic sufficiency check
                original_len = len(training_data)
                filtered_len = len(training_data_filtered)
                n_features = len(training_data_filtered.columns)
                samples_per_feature = filtered_len / n_features if n_features > 0 else 0
                min_recommended = max(n_features * 10, 1000)

                tprint_info(f"📊 Using FULL dataset: {filtered_len} samples (no pre-filtering)")
                if filtered_len < min_recommended:
                    tprint_warning(
                        "⚠️ Dataset may be insufficient for robust training given feature count "
                        f"({filtered_len} samples, {n_features} features, {samples_per_feature:.2f} samples/feature)"
                    )
                else:
                    tprint_success(f"✅ Dataset size looks adequate ({samples_per_feature:.1f} samples/feature)")

                training_data = training_data_filtered

                # For analyst_base, capture a wide-window snapshot for empirical
                # diagnostics before additional specialist merges and light-mode
                # filtering shrink the effective training slice.
                wide_window_data = None
                wide_window_targets = None
                if training_type == 'analyst_base' and analyst_targets is not None:
                    try:
                        wide_window_data = training_data.copy()
                        wide_window_targets = analyst_targets.copy()
                        tprint_info(
                            f"🔍 [EMPIRICAL] Captured wide-window snapshot for analyst_base: "
                            f"X={wide_window_data.shape}, y={len(wide_window_targets)}"
                        )
                    except Exception as snap_exc:
                        self.logger.debug(
                            f"Failed to capture wide-window diagnostic snapshot: {snap_exc}"
                        )

                # Filter targets to match largest training period
                if analyst_targets is not None:
                    analyst_targets = analyst_targets.loc[training_data.index]
                    tprint_info(f"   ↪ Analyst targets filtered to {len(analyst_targets)} samples")

                if tactician_targets is not None:
                    training_type_local = str(config.get('training_type', 'analyst_base')).lower()
                    if 'tactician' not in training_type_local:
                        tprint_info(
                            "ℹ️ Skipping tactician target filtering because training_type does not "
                            "include tactician models (clearing tactician_targets for safety)."
                        )
                        tactician_targets = None
                    else:
                        tactician_targets = tactician_targets.loc[training_data.index]
                        tprint_info(f"   ↪ Tactician targets filtered to {len(tactician_targets)} samples")

                tprint_info("=" * 80)
            else:
                tprint_warning("⚠️ No training data available for temporal splitting")
                self._walkforward_config = None

            # --- MODIFIED: Retrieve and merge additional features for all training types (regime probs needed for all) ---
            if training_type in ['analyst_base', 'tactician_base', 'analyst_ensemble', 'tactician_ensemble']:
                tprint_info(f"Retrieving additional model outputs for {training_type}...")
                # --- FIX 5: Pass training_data for index alignment ---
                additional_outputs = await self._get_additional_model_outputs(training_type, config, training_data)

                if additional_outputs is not None:
                    # Align indices before concatenating
                    # This alignment is still necessary AFTER resampling, just in case.
                    tprint_info(
                        "🔄 Aligning primary training data with additional outputs before concatenation"
                    )
                    tprint_info(
                        f"   ↪ training_data shape={training_data.shape}, columns={len(training_data.columns)}"
                    )
                    tprint_info(
                        f"   ↪ additional_outputs shape={additional_outputs.shape}, columns={len(additional_outputs.columns)}"
                    )
                    aligned_training_data, aligned_additional_outputs = training_data.align(additional_outputs, join='inner', axis=0)                    
                    tprint_info(
                        f"   ↪ aligned_training_data shape={aligned_training_data.shape}, aligned_additional_outputs shape={aligned_additional_outputs.shape}"
                    )
                    if aligned_training_data.empty:
                        tprint_warning("Data alignment resulted in empty DataFrame. Check for index mismatches.")
                        # Fallback to original data if alignment fails
                    else:
                        merged_columns = len(aligned_training_data.columns) + len(aligned_additional_outputs.columns)
                        tprint_info(
                            f"   ↪ Concatenating columns -> expected merged column count ≈ {merged_columns}"
                        )
                        # Use safe concatenation with temporal alignment validation
                        training_data = self._safe_concat(
                            [aligned_training_data, aligned_additional_outputs],
                            axis=1,
                            operation_name="merge_training_and_additional_features",
                            validate_alignment=True
                        )
                        tprint_success(f"✅ Merged additional features. New training data shape: {training_data.shape}")
                else:
                    tprint_warning(f"No additional model outputs found for {training_type}. Proceeding with primary features only.")
            # --- END MODIFICATION ---

            # Apply light mode filtering if needed
            training_data = self._apply_light_mode_filter(training_data, config, timeframe)
            
            # Align targets to match filtered training data
            if training_data is not None and analyst_targets is not None:
                if len(analyst_targets) != len(training_data):
                    tprint_warning(f"⚠️ Aligning analyst targets from {len(analyst_targets)} to {len(training_data)} samples")
                    analyst_targets = analyst_targets.loc[training_data.index]
            if training_data is not None and tactician_targets is not None:
                if len(tactician_targets) != len(training_data):
                    tprint_warning(f"⚠️ Aligning tactician targets from {len(tactician_targets)} to {len(training_data)} samples")
                    tactician_targets = tactician_targets.loc[training_data.index]
            
            # ------------------------------------------------------------------
            # EMPIRICAL DIAGNOSTICS: Wide-window vs final-slice signal strength
            # ------------------------------------------------------------------
            if training_type == 'analyst_base' and training_data is not None and analyst_targets is not None:
                try:

                    # Both wide_window_data and training_data may be None if
                    # snapshot capture failed earlier.
                    if wide_window_data is not None and wide_window_targets is not None:
                        # Restrict to shared numeric columns for a fair comparison
                        wide_num = wide_window_data.select_dtypes(include=[np.number])
                        final_num = training_data.select_dtypes(include=[np.number])
                        shared_cols = [
                            c for c in wide_num.columns
                            if c in final_num.columns
                        ]

                        if shared_cols:
                            # Align indices for wide window
                            common_idx_wide = wide_window_targets.index.intersection(
                                wide_window_data.index
                            )
                            X_wide = wide_num.loc[common_idx_wide, shared_cols].astype(float)
                            y_wide = wide_window_targets.loc[common_idx_wide].astype(float)

                            # Final slice already has analyst_targets aligned
                            common_idx_final = analyst_targets.index.intersection(
                                training_data.index
                            )
                            X_final = final_num.loc[common_idx_final, shared_cols].astype(float)
                            y_final = analyst_targets.loc[common_idx_final].astype(float)

                            def _simple_r2(X: np.ndarray, y: np.ndarray) -> float:
                                if X.ndim != 2 or y.ndim != 1:
                                    return float('nan')
                                n, d = X.shape
                                if n < d + 2 or n < 100:
                                    # Too few samples for a stable multi-linear fit
                                    return float('nan')
                                X_aug = np.c_[np.ones(n), X]
                                try:
                                    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
                                    y_pred = X_aug @ beta
                                    ss_res = float(np.sum((y - y_pred) ** 2))
                                    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
                                    if ss_tot <= 0:
                                        return float('nan')
                                    return 1.0 - ss_res / ss_tot
                                except Exception:
                                    return float('nan')

                            r2_wide = _simple_r2(X_wide.values, y_wide.values)
                            r2_final = _simple_r2(X_final.values, y_final.values)

                            tprint_info("=" * 80)
                            tprint_info("📊 [EMPIRICAL] Wide-window vs final-slice signal (multi-linear R²)")
                            tprint_info(
                                f"   Shared numeric features: {len(shared_cols)} | "
                                f"wide_n={len(X_wide)}, final_n={len(X_final)}"
                            )
                            tprint_info(
                                f"   Wide window R² (base+shared specialists): "
                                f"{r2_wide if not np.isnan(r2_wide) else float('nan'):.4f}"
                            )
                            tprint_info(
                                f"   Final slice R² (post-intersection/filtering): "
                                f"{r2_final if not np.isnan(r2_final) else float('nan'):.4f}"
                            )
                            tprint_info("=" * 80)
                        else:
                            tprint_info(
                                "📊 [EMPIRICAL] No shared numeric feature columns between wide-window "
                                "and final-slice datasets; skipping R² comparison."
                            )
                except Exception as emp_exc:
                    self.logger.debug(
                        f"Failed to compute empirical wide-vs-final signal diagnostics: {emp_exc}"
                    )

            # Calculate COMPREHENSIVE dynamic configuration based on data and hardware
            if training_data is not None:
                non_feature_cols = [col for col in training_data.columns if 'timestamp' in col.lower() or 'datetime' in col.lower()]
                if non_feature_cols:
                    tprint_info(f"Dropping non-feature timestamp columns: {non_feature_cols}")
                    training_data = training_data.drop(columns=non_feature_cols)   
                    
                calculator = DynamicConfigCalculator()
                base_dynamic_kwargs = {
                    'total_samples': len(training_data),
                    'n_features': len(training_data.columns),
                    'timeframe': timeframe,
                    'execution_mode': config.get('execution_mode', 'full'),
                    'training_type': training_type,
                    'train_percentage': config.get('train_percentage', 0.70),
                    'validation_percentage': config.get('validation_percentage', 0.15),
                    'test_percentage': config.get('test_percentage', 0.15),
                }

                tree_dynamic_config = calculator.calculate_all_parameters(
                    model_type='lgbm',
                    **base_dynamic_kwargs,
                )

                nn_dynamic_config = calculator.calculate_all_parameters(
                    model_type='gru',
                    **base_dynamic_kwargs,
                )

                yaml_config = self._apply_dynamic_config(
                    yaml_config=yaml_config,
                    dynamic_config=tree_dynamic_config,
                    training_type=training_type,
                    nn_dynamic_config=nn_dynamic_config,
                )
                tprint_success(f"✅ Configured training with dynamic parameters (samples, epochs, batch size, memory, etc.)")
            else:
                tprint_warning("No training data available, using default configuration from YAML")
            
            # ========================================================================
            # FEATURE SET PREPARATION (A & B)
            # ========================================================================
            # Prepare Data Container with Feature Set A (Technical)
            training_data_container = {'A': training_data}

            feature_set_mode = str(
                (yaml_config.get('analyst_config', {}) or {}).get('feature_set', 'A')
            ).upper()

            # Check if any enabled model requests B explicitly
            model_configs = (yaml_config.get('analyst_config', {}) or {}).get('base_models', {})
            any_model_wants_b = False
            if isinstance(model_configs, dict):
                any_model_wants_b = any(
                    (m.get('feature_set', feature_set_mode) == 'B')
                    for m in model_configs.values()
                    if isinstance(m, dict) and m.get('enabled', True)
                )

            if training_type == 'analyst_base' and (feature_set_mode == 'B' or any_model_wants_b):
                tprint_info("=" * 80)
                tprint_info("🔄 FEATURE SET B: Building meta-gated feature set")
                tprint_info("=" * 80)
                meta_features = self._build_meta_gated_feature_set(training_data.index, config)
                if meta_features is not None and not meta_features.empty:
                    tprint_success(
                        f"✅ Constructed Feature Set B (meta-gated): "
                        f"{meta_features.shape[0]} samples × {meta_features.shape[1]} features"
                    )
                    training_data_container['B'] = meta_features

                    # If global mode is B, update main training_data pointer for HPO/Legacy use
                    if feature_set_mode == 'B':
                         training_data = meta_features
                else:
                    tprint_warning(
                        "⚠️ Requested Feature Set B (meta-gated), but failed to build a valid "
                        "meta-gated feature frame; falling back to default analyst_base features."
                    )

            # ========================================================================
            # SESSION-BASED HPO CONTROL: Prevent infinite HPO loops
            # ========================================================================
            # HPO should run ONCE per session, then move to model training/testing
            # This prevents the infinite loop where HPO keeps restarting after completion
            import os

            # Check if HPO has already been completed in this session
            hpo_already_run = os.getenv('HPO_ALREADY_RUN', 'false').lower() in ('true', '1', 'yes')

            # Check if HPO is permanently disabled
            disable_hpo_env = os.getenv('DISABLE_HPO', 'false').lower() in ('true', '1', 'yes')

            # Determine if we should run HPO
            should_run_hpo = not hpo_already_run and not disable_hpo_env

            if hpo_already_run:
                tprint_warning("✅ HPO ALREADY COMPLETED in this session - using saved optimal parameters")
                tprint_info("   Proceeding to model training with optimized hyperparameters")
            elif disable_hpo_env:
                tprint_warning("🚫 HPO PERMANENTLY DISABLED via DISABLE_HPO environment variable")
                tprint_info("   Using default parameters from config")

            # Perform hyperparameter optimization before training (only once per session)
            if should_run_hpo and training_data is not None:
                # Determine which targets to use for HPO
                hpo_targets = analyst_targets if training_type.startswith('analyst') else tactician_targets
                if hpo_targets is not None:
                    tprint_info("🔍 Performing hyperparameter optimization using custom_balanced_score...")
                    
                    # Get the appropriate model config
                    if training_type.startswith('analyst'):
                        model_config_key = 'analyst_config'
                    elif training_type.startswith('tactician'):
                        model_config_key = 'tactician_config'
                    else:
                        model_config_key = 'ensemble_config'
                    
                    # Get config file path for this training type
                    config_mapping = {
                        'analyst_base': 'src/training/steps/model_training/analyst_base_config.yaml',
                        'analyst_ensemble': 'src/training/steps/model_training/analyst_ensemble_config.yaml',
                        'tactician_base': 'src/training/steps/model_training/tactician_base_config.yaml',
                        'tactician_ensemble': 'src/training/steps/model_training/tactician_ensemble_config.yaml'
                    }
                    config_file = config_mapping.get(training_type)
                    
                    if model_config_key in yaml_config and config_file:
                        # Use new HPO system with custom_balanced_score
                        yaml_config[model_config_key] = await self._perform_hierarchical_hpo(
                            training_data=training_data,
                            targets=hpo_targets,
                            model_config=yaml_config[model_config_key],
                            config_file=config_file,
                            config=config,
                            training_type=training_type
                        )

                        # ========================================================================
                        # MARK HPO AS COMPLETED: Set session flag to prevent re-runs
                        # ========================================================================
                        # After successful HPO, mark it as completed for this session
                        # This prevents the infinite loop where HPO restarts after completion
                        os.environ['HPO_ALREADY_RUN'] = 'true'
                        tprint_success("✅ HPO completed successfully - session flag set to prevent re-runs")
                        tprint_info("   Future training steps will use these optimized parameters")
                    else:
                        tprint_warning(f"No {model_config_key} found in config or config file, skipping HPO")
                else:
                    tprint_warning("No targets available for HPO, skipping optimization")
            else:
                tprint_info("Hyperparameter optimization disabled or no training data available")
            
            # Execute training based on type
            # Pass container if available (for analyst_base), else training_data
            data_to_pass = training_data_container if training_type == 'analyst_base' else training_data

            result = await self._execute_training_by_type(
                training_type, data_to_pass, analyst_targets, tactician_targets, yaml_config, config
            )
            
            # DEBUG: Log training result structure for artifact saving diagnostics
            tprint_info("=" * 80)
            tprint_info(f"🔍 [DEBUG] TRAINING RESULT STRUCTURE for {training_type}")
            tprint_info("=" * 80)
            tprint_info(f"🔍 [DEBUG] Result keys: {list(result.keys())}")
            tprint_info(f"🔍 [DEBUG] result['success']: {result.get('success', 'NOT SET')}")
            if 'models' in result:
                tprint_info(f"🔍 [DEBUG] result['models'] keys: {list(result['models'].keys()) if result['models'] else 'None/Empty'}")
            if 'predictions' in result:
                pred = result['predictions']
                tprint_info(f"🔍 [DEBUG] result['predictions'] type: {type(pred).__name__}, shape: {getattr(pred, 'shape', 'N/A')}")
            if 'oof_predictions' in result:
                oof = result['oof_predictions']
                tprint_info(f"🔍 [DEBUG] result['oof_predictions'] type: {type(oof).__name__}, shape: {getattr(oof, 'shape', 'N/A')}")
            if 'error_message' in result:
                tprint_warning(f"🔍 [DEBUG] result['error_message']: {result['error_message']}")
            tprint_info("=" * 80)
            
            if result.get('success', False):
                tprint_success(f"✅ Unified {training_type} training completed successfully")

                # Save artifacts
                artifacts = await self._save_training_artifacts(result, training_type, config)
                result['artifacts'] = artifacts

                # Generate markdown and JSON training reports
                try:
                    tprint_info("📝 Generating training reports (Markdown + JSON)...")

                    # Prepare feature info
                    feature_info = {
                        'feature_count': training_data.shape[1] if training_data is not None else 0,
                        'feature_source': 'feature_generation_final_feature_selection_step',
                        'feature_names': list(training_data.columns) if training_data is not None else [],
                        'regime_features_included': True
                    }

                    # Generate reports
                    markdown_path, json_path = create_model_training_report(
                        training_type=training_type,
                        symbol=symbol,
                        exchange=config.get('exchange', 'binance'),
                        timeframe=timeframe,
                        direction=direction,
                        models_trained=result.get('models', {}),
                        metrics=result.get('metrics', {}),
                        hpo_results=result.get('hpo_results'),
                        regime_performance=result.get('regime_performance'),
                        training_config=config,
                        feature_info=feature_info,
                        execution_time=result.get('execution_time', 0.0),
                        outcomes_dir='outcomes'
                    )

                    if markdown_path:
                        artifacts['training_report_markdown'] = markdown_path
                        tprint_success(f"✅ Markdown report saved: {markdown_path}")

                    if json_path:
                        artifacts['training_report_json'] = json_path
                        tprint_success(f"✅ JSON metrics report saved: {json_path}")

                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate training reports: {e}")
                    self.logger.warning(f"Training report generation failed: {e}")

                # Save ML-scored historical data for backtesting (OOS only)
                if training_data is not None:
                    try:
                        model_type = 'analyst' if training_type.startswith('analyst') else 'tactician'
                        # Prefer explicit OOS keys; for ensemble training fall back to 'predictions'
                        oos_key_candidates = ['oof_predictions', 'oos_predictions', 'predictions_oos']
                        if training_type.endswith('ensemble') and 'predictions' in result:
                            oos_key_candidates.append('predictions')

                        oos_key = next((k for k in oos_key_candidates if k in result), None)
                        oos_df = None

                        if oos_key is not None and result[oos_key] is not None:
                            raw_oos = result[oos_key]

                            if isinstance(raw_oos, pd.DataFrame):
                                oos_df = raw_oos
                            else:
                                # Convert array-like predictions to a DataFrame and align to training_data index
                                try:
                                    arr = np.asarray(raw_oos)
                                    if arr.ndim == 1:
                                        arr = arr.reshape(-1, 1)

                                    if hasattr(training_data, 'index') and len(training_data.index) == arr.shape[0]:
                                        idx = training_data.index
                                    else:
                                        idx = pd.RangeIndex(arr.shape[0])

                                    cols = [
                                        f"{training_type}_pred_{i}" for i in range(arr.shape[1])
                                    ]
                                    oos_df = pd.DataFrame(arr, index=idx, columns=cols)
                                except Exception as conv_exc:
                                    tprint_warning(
                                        f"⚠️ Failed to convert OOS predictions '{oos_key}' to DataFrame: {conv_exc}"
                                    )
                                    oos_df = None

                        if oos_df is not None and not oos_df.empty:
                            tprint_info(f"📊 Saving ML-scored historical data ({model_type}) using OOS predictions: {oos_key}")

                            # Align training data to OOS index and combine with predictions
                            ml_scored_data = training_data.loc[oos_df.index].copy()
                            ml_scored_data = pd.concat([ml_scored_data, oos_df], axis=1)

                            # Attach an explicit analyst_confidence column so downstream
                            # backtests can use a well-defined confidence score.
                            analyst_confidence = None
                            try:
                                # Prefer a calibrated probability of a positive analyst outcome
                                # (e.g. target_long_fused > 0) when analyst targets are available.
                                if training_type.startswith('analyst') and analyst_targets is not None:
                                    try:
                                        # Align analyst targets to the OOS prediction index
                                        y_for_oos = analyst_targets.reindex(oos_df.index)
                                    except Exception:
                                        y_for_oos = None

                                    if y_for_oos is not None:
                                        # Binary outcome: 1 if target > 0, else 0
                                        y_binary = (y_for_oos > 0).astype(int)
                                        # Drop any rows with missing labels
                                        valid_mask = y_binary.notna()
                                        if bool(valid_mask.any()):
                                            y_binary_valid = y_binary[valid_mask].astype(int)
                                            # Aggregate OOS predictions to a single score per row
                                            base_scores = oos_df.mean(axis=1)
                                            base_scores_valid = base_scores[valid_mask]

                                            # Require at least two classes to perform calibration
                                            if y_binary_valid.nunique() >= 2:
                                                # Map raw scores to [0, 1] as an initial probability proxy
                                                s_min = float(base_scores_valid.min())
                                                s_max = float(base_scores_valid.max())
                                                if s_max > s_min:
                                                    scaled = (base_scores_valid - s_min) / (s_max - s_min)
                                                    proba_input = np.column_stack(
                                                        [1.0 - scaled.to_numpy(), scaled.to_numpy()]
                                                    )
                                                    calib_result = await calibrate_model_confidence(
                                                        y_true=y_binary_valid.to_numpy(),
                                                        y_pred_proba=proba_input,
                                                        method='isotonic_regression',
                                                        config=None,
                                                    )
                                                    if isinstance(calib_result, dict) and 'calibrated_probabilities' in calib_result:
                                                        calibrated = np.asarray(
                                                            calib_result['calibrated_probabilities'],
                                                            dtype=float,
                                                        )
                                                        if calibrated.shape[0] == base_scores_valid.shape[0]:
                                                            analyst_confidence = pd.Series(
                                                                np.nan,
                                                                index=oos_df.index,
                                                                dtype=float,
                                                            )
                                                            analyst_confidence.loc[valid_mask] = calibrated

                                # If calibration was not possible, fall back to model-provided confidence
                                if analyst_confidence is None:
                                    conf_df = result.get('confidence')
                                    if (
                                        conf_df is not None
                                        and isinstance(conf_df, pd.DataFrame)
                                        and not conf_df.empty
                                    ):
                                        # Align confidence to OOS index
                                        conf_aligned = conf_df.reindex(oos_df.index)
                                        # If multiple columns, aggregate to a single scalar per row
                                        analyst_confidence = conf_aligned.abs().mean(axis=1)
                                    else:
                                        # Fallback: derive confidence from the OOS predictions themselves
                                        analyst_confidence = oos_df.abs().mean(axis=1)
                            except Exception:
                                analyst_confidence = None

                            if analyst_confidence is not None:
                                ml_scored_data['analyst_confidence'] = analyst_confidence.astype(float)
                                # Optionally store a volatility/volume-adjusted confidence variant
                                try:
                                    close_series = None
                                    volume_series = None
                                    if 'close' in ml_scored_data.columns:
                                        close_series = ml_scored_data['close'].astype(float)
                                    if 'volume' in ml_scored_data.columns:
                                        volume_series = ml_scored_data['volume'].astype(float)
                                    if close_series is not None:
                                        risk_adj = apply_risk_adjusted_confidence(
                                            confidence=ml_scored_data['analyst_confidence'],
                                            close=close_series,
                                            volume=volume_series,
                                        )
                                        ml_scored_data['analyst_confidence_risk_adj'] = risk_adj.astype(float)
                                except Exception:
                                    # If risk adjustment fails for any reason, continue with raw confidence.
                                    pass

                            # Normalize index and ensure unique timestamps for downstream consumers
                            if isinstance(ml_scored_data.index, pd.DatetimeIndex):
                                ml_scored_data = ml_scored_data.sort_index()
                            else:
                                try:
                                    idx = pd.to_datetime(ml_scored_data.index, errors="coerce")
                                    valid_mask = ~idx.isna()
                                    if bool(valid_mask.any()):
                                        ml_scored_data = ml_scored_data.loc[valid_mask].copy()
                                        ml_scored_data.index = idx[valid_mask]
                                        ml_scored_data = ml_scored_data.sort_index()
                                except Exception:
                                    pass

                            if not ml_scored_data.index.is_unique:
                                self.logger.warning(
                                    "ML-scored OOS data has duplicate timestamps; keeping last occurrence per timestamp before saving artifact."
                                )
                                ml_scored_data = ml_scored_data[~ml_scored_data.index.duplicated(keep="last")]

                            artifact_name = f"ml_scored_historical_data_{model_type}_{direction}_oos"
                            ml_scored_path = self._save_artifact(
                                ml_scored_data,
                                artifact_name,
                                artifact_type='data',
                                data_category='predictions'
                            )
                            artifacts['ml_scored_historical_data_oos'] = ml_scored_path
                            tprint_success(f"✅ ML-scored OOS data saved: {artifact_name}")
                            tprint_info(f"   Path: {ml_scored_path} | shape={ml_scored_data.shape}")
                        else:
                            tprint_info("ℹ️ Skipping ML-scored historical data save: OOS predictions not available for this training run")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to save ML-scored data: {e}")
                        self.logger.warning(f"ML-scored data save failed: {e}")

                # Generate comprehensive training reports (markdown + JSON)
                tprint_info("📝 Generating comprehensive training reports...")
                report_paths = self._generate_training_reports(result, training_type, config)
                if report_paths:
                    artifacts.update(report_paths)
                    tprint_success(f"✅ Training reports generated: {len(report_paths)} files")

                return {
                    'success': True,
                    'artifacts': artifacts,
                    'metrics': result.get('metrics', {}),
                    'models': result.get('models', {}),
                    'training_type': training_type,
                    'execution_time': result.get('execution_time', 0.0),
                    'reports': report_paths,
                    'model_path': result.get('model_path')
                }
            else:
                tprint_error(f"❌ Unified {training_type} training failed")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': result.get('error_message', 'Training failed'),
                    'training_type': training_type
                }

        except Exception as e:
            import traceback
            error_msg = f"Unified {training_type} training failed: {str(e)}"
            traceback_str = traceback.format_exc()
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"Traceback:\n{traceback_str}")
            self.logger.error(error_msg)
            self.logger.error(traceback_str)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg,
                'traceback': traceback_str,
                'training_type': training_type
            }

    async def _load_training_config(self, training_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load appropriate YAML configuration based on training type."""
        try:
            # Map training types to config files in steps/model_training/
            config_mapping = {
                'analyst_base': 'src/training/steps/model_training/analyst_base_config.yaml',
                'analyst_ensemble': 'src/training/steps/model_training/analyst_ensemble_config.yaml',
                'tactician_base': 'src/training/steps/model_training/tactician_base_config.yaml',
                'tactician_ensemble': 'src/training/steps/model_training/tactician_ensemble_config.yaml'
            }
            
            config_file = config_mapping.get(training_type)
            if not config_file or not os.path.exists(config_file):
                # Fallback to default configuration
                tprint_info(f"Using default configuration for {training_type}")
                return self._get_default_config(training_type, config)
            
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update configuration with runtime parameters
            yaml_config = self._update_config_with_runtime_params(yaml_config, config)
            
            tprint_info(f"Loaded configuration from {config_file}")
            return yaml_config
            
        except Exception as e:
            tprint_error(f"Failed to load config for {training_type}: {e}")
            return self._get_default_config(training_type, config)

    def _get_default_config(self, training_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get default configuration when YAML file is not available."""
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'long')
        
        base_config = {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'execution_mode': config.get('execution_mode', 'light'),
            'enable_analyst': training_type.startswith('analyst'),
            'enable_tactician': training_type.startswith('tactician'),
            'enable_ensemble': training_type.endswith('ensemble'),
            # NOTE: HPO control is now handled by HPO_ALREADY_RUN environment variable
            # See session-based HPO control in execute() method
            'enable_explainability': True,
            'enable_vectorization': True
        }
        
        if training_type.startswith('analyst'):
            base_config.update({
                'analyst_config': {
                    'model_name': f"analyst_{'ensemble' if training_type.endswith('ensemble') else 'base'}",
                    'timeframe': timeframe,
                    'n_outputs': 4,
                    'output_names': ["signal_strength", "confidence", "risk_score", "regime_label"]
                }
            })
        elif training_type.startswith('tactician'):
            base_config.update({
                'tactician_config': {
                    'model_name': f"tactician_{'ensemble' if training_type.endswith('ensemble') else 'base'}",
                    'timeframe': timeframe,
                    'n_outputs': 4,
                    'output_names': ["entry_timing", "position_size", "stop_loss", "take_profit"]
                }
            })
        
        return base_config

    def _update_config_with_runtime_params(self, yaml_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Update YAML configuration with runtime parameters."""
        # Update basic parameters in the main config sections
        if 'analyst_config' in yaml_config:
            yaml_config['analyst_config']['timeframe'] = config.get('timeframe', '15m')
            yaml_config['analyst_config']['symbol'] = config.get('symbol', 'ETHUSDT')
            yaml_config['analyst_config']['direction'] = config.get('direction', 'long')
        if 'tactician_config' in yaml_config:
            yaml_config['tactician_config']['timeframe'] = config.get('timeframe', '15m')
            yaml_config['tactician_config']['symbol'] = config.get('symbol', 'ETHUSDT')
            yaml_config['tactician_config']['direction'] = config.get('direction', 'long')
        
        # Add runtime parameters to the root level
        yaml_config.update({
            'symbol': config.get('symbol', 'ETHUSDT'),
            'timeframe': config.get('timeframe', '15m'),
            'direction': config.get('direction', 'long'),
            'execution_mode': config.get('execution_mode', 'light'),
            'exchange': config.get('exchange', 'binance')
        })
        
        # Apply light mode optimizations for TCN if in light execution mode
        execution_mode = config.get('execution_mode', 'light')
        if execution_mode == 'light':
            self._apply_light_mode_tcn_optimizations(yaml_config)
        
        return yaml_config
    
    def _apply_light_mode_tcn_optimizations(self, yaml_config: Dict[str, Any]) -> None:
        """Apply aggressive model optimizations for light mode execution (10x lighter)."""
        execution_mode = yaml_config.get('execution_mode', 'light')
        
        # Check if analyst_config exists
        if 'analyst_config' in yaml_config:
            base_models = yaml_config['analyst_config'].get('base_models', {})
            
            # Optimize DepthwiseCNN (replaces TCN)
            if 'depthwise_cnn' in base_models:
                depthwise_config = base_models['depthwise_cnn']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode DepthwiseCNN optimizations (10x lighter)")

                # Drastically reduce DepthwiseCNN parameters for light mode
                depthwise_params = depthwise_config.get('params', {})
                depthwise_params['filters'] = 32  # Reduced from 64
                depthwise_params['epochs'] = 10  # Reduced from 50 (10x lighter)
                depthwise_params['batch_size'] = 128  # Increased from 64 (fewer iterations)
                depthwise_params['early_stopping_patience'] = 3  # Reduced from 7

                # REACTIVATE DepthwiseCNN HPO in light mode (user request)
                if 'hpo' in depthwise_config:
                    # Keep HPO enabled but reduce trials for light mode
                    if 'max_trials' in depthwise_config['hpo']:
                        depthwise_config['hpo']['max_trials'] = min(depthwise_config['hpo']['max_trials'], 10)
                    if 'time_budget' in depthwise_config['hpo']:
                        depthwise_config['hpo']['time_budget'] = min(depthwise_config['hpo']['time_budget'], 300)  # 5 minutes max
                    tprint_info(f"  DepthwiseCNN HPO: REACTIVATED (reduced trials for light mode)")
                else:
                    tprint_info(f"  DepthwiseCNN HPO: No HPO configuration found")

                tprint_info(f"  DepthwiseCNN epochs: 50 → 10 (10x lighter)")
            
            # Optimize CatBoost
            if 'catboost' in base_models:
                catboost_config = base_models['catboost']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode CatBoost optimizations (10x lighter)")
                
                # Reduce CatBoost iterations for light mode
                catboost_params = catboost_config.get('params', {})
                catboost_params['iterations'] = 50  # Reduced from 500 (10x lighter)
                catboost_params['depth'] = 4  # Reduced from 6
                catboost_params['early_stopping_rounds'] = 10  # Reduced from 50
                
                # REACTIVATE CatBoost HPO in light mode (user request)
                if 'hpo' in catboost_config:
                    # Keep HPO enabled but reduce trials for light mode
                    if 'max_trials' in catboost_config['hpo']:
                        catboost_config['hpo']['max_trials'] = min(catboost_config['hpo']['max_trials'], 10)
                    if 'time_budget' in catboost_config['hpo']:
                        catboost_config['hpo']['time_budget'] = min(catboost_config['hpo']['time_budget'], 300)  # 5 minutes max
                    tprint_info(f"  CatBoost HPO: REACTIVATED (reduced trials for light mode)")
                else:
                    tprint_info(f"  CatBoost HPO: No HPO configuration found")
                
                tprint_info(f"  CatBoost iterations: 500 → 50 (10x lighter)")
                tprint_info(f"  CatBoost depth: 6 → 4")
                tprint_info(f"  CatBoost HPO: REACTIVATED (reduced trials for light mode)")
            
            # Optimize LGBM
            if 'lgbm' in base_models:
                lgbm_config = base_models['lgbm']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode LGBM optimizations (10x lighter)")
                
                # Reduce LGBM estimators for light mode
                lgbm_params = lgbm_config.get('params', {})
                lgbm_params['n_estimators'] = 100  # Reduced from 1000 (10x lighter)
                lgbm_params['max_depth'] = 6  # Reduced from 8
                
                # REACTIVATE LGBM HPO in light mode (user request)
                if 'hpo' in lgbm_config:
                    # Keep HPO enabled but reduce trials for light mode
                    if 'max_trials' in lgbm_config['hpo']:
                        lgbm_config['hpo']['max_trials'] = min(lgbm_config['hpo']['max_trials'], 10)
                    if 'time_budget' in lgbm_config['hpo']:
                        lgbm_config['hpo']['time_budget'] = min(lgbm_config['hpo']['time_budget'], 300)  # 5 minutes max
                    tprint_info(f"  LGBM HPO: REACTIVATED (reduced trials for light mode)")
                else:
                    tprint_info(f"  LGBM HPO: No HPO configuration found")
                
                tprint_info(f"  LGBM n_estimators: 1000 → 100 (10x lighter)")
                tprint_info(f"  LGBM max_depth: 8 → 6")
                tprint_info(f"  LGBM HPO: REACTIVATED (reduced trials for light mode)")
        
        # Check if tactician_config has GRU model
        if 'tactician_config' in yaml_config:
            base_models = yaml_config['tactician_config'].get('base_models', [])
            for model in base_models:
                if model.get('model_name') == 'StandaloneGRU':
                    tprint_warning(f"⚡ Applying {execution_mode.upper()} mode GRU optimizations (10x lighter)")
                    params = model.get('params', {})
                    params['epochs'] = 10  # Reduce epochs (10x lighter)
                    params['batch_size'] = 128  # Increase batch size
                    if 'hpo' in model:
                        model['hpo']['enabled'] = False
                    tprint_info(f"  GRU epochs: Reduced to 10")
                    tprint_info(f"  GRU HPO: DISABLED")


    def _apply_dynamic_config(
        self,
        yaml_config: Dict[str, Any],
        dynamic_config: DynamicTrainingConfig,
        training_type: str,
        nn_dynamic_config: Optional[DynamicTrainingConfig] = None,
    ) -> Dict[str, Any]:
        """
        Apply dynamic configuration to YAML config.
        
        Args:
            yaml_config: YAML configuration dictionary
            dynamic_config: Dynamically calculated configuration
            training_type: Type of training (analyst_base, tactician_base, etc.)
            
        Returns:
            Updated YAML configuration
        """
        try:
            tprint_info("🔧 Applying dynamic configuration to YAML config...")
            nn_config = nn_dynamic_config or dynamic_config

            # Determine which config section to update
            if training_type.startswith('analyst'):
                config_key = 'analyst_config'
            elif training_type.startswith('tactician'):
                config_key = 'tactician_config'
            else:
                config_key = 'ensemble_config'
            
            # Update the appropriate config section
            if config_key in yaml_config:
                # Update training parameters
                if 'training' in yaml_config[config_key]:
                    yaml_config[config_key]['training'].update({
                        'training_samples': dynamic_config.training_samples,
                        'validation_samples': dynamic_config.validation_samples,
                        'test_samples': dynamic_config.test_samples,
                        'cv_folds': dynamic_config.cv_folds,
                        'early_stopping_patience': dynamic_config.early_stopping_patience
                    })
                
                # Update base model parameters
                if 'base_models' in yaml_config[config_key]:
                    base_models = yaml_config[config_key]['base_models']
                    
                    # Handle both list and dict formats
                    if isinstance(base_models, list):
                        # List format: iterate through list items
                        for model_item in base_models:
                            model_name = model_item.get('model_name', 'unknown')
                            model_params = model_item
                            
                            if 'params' not in model_params:
                                model_params['params'] = {}
                            
                            # Update common parameters
                            # Use model_name for matching, as model_type is for HPO
                            model_type_key = model_name.lower() # Use model_name for key
                            
                            # Neural network models
                            if any(nn in model_type_key for nn in ['gru', 'lstm', 'tcn', 'transformer', 'depthwisecnn', 'cnn']):
                                nn_params_to_update = {
                                    'batch_size': nn_config.batch_size,
                                    'epochs': nn_config.epochs if nn_config.epochs > 0 else 100,
                                    'learning_rate': nn_config.learning_rate,
                                    'early_stopping_patience': nn_config.early_stopping_patience
                                }

                                if 'gru' in model_type_key and 'training_params' in model_params['params']:
                                     model_params['params']['training_params'].update(nn_params_to_update)
                                else:
                                     model_params['params'].update(nn_params_to_update)

                                if any(ts in model_type_key for ts in ['gru', 'lstm', 'tcn', 'depthwisecnn', 'cnn']):
                                    model_params['params']['sequence_length'] = nn_config.sequence_length
                            
                            # Tree-based models
                            elif any(tree in model_type_key for tree in ['lgbm', 'catboost', 'xgboost', 'extratrees']):
                                if 'lgbm' in model_type_key:
                                    model_params['params']['n_estimators'] = dynamic_config.n_estimators
                                elif 'catboost' in model_type_key:
                                    tprint_info("Applying CatBoost GPU (Apple M1) configuration...")
                                    model_params['params']['task_type'] = 'GPU'
                                    model_params['params']['devices'] = '0' # Use '0' for Apple M1 GPU
                                    
                                    # Remove subsample if it exists, as it's not supported for GPU training
                                    if 'subsample' in model_params['params']:
                                        del model_params['params']['subsample']
                                        tprint_info("Removed 'subsample' param, not supported by CatBoost GPU.")     
                                        
                                model_params['params']['learning_rate'] = dynamic_config.learning_rate
                            
                            tprint_info(f"  Updated {model_name} with dynamic parameters")
                    else:
                        # Dict format: use items()
                        for model_name, model_params in base_models.items():
                            if 'params' not in model_params:
                                model_params['params'] = {}
                            
                            # Update common parameters
                            model_type_key = model_name.lower() # Use model_name for key
                            
                            # Neural network models
                            if any(nn in model_type_key for nn in ['gru', 'lstm', 'tcn', 'transformer', 'depthwisecnn', 'cnn']):
                                model_params['params'].update({
                                    'batch_size': nn_config.batch_size,
                                    'epochs': nn_config.epochs if nn_config.epochs > 0 else 100,
                                    'learning_rate': nn_config.learning_rate,
                                    'early_stopping_patience': nn_config.early_stopping_patience
                                })

                                if any(ts in model_type_key for ts in ['gru', 'lstm', 'tcn', 'depthwisecnn', 'cnn']):
                                    model_params['params']['sequence_length'] = nn_config.sequence_length
                            
                            # Tree-based models
                            elif any(tree in model_type_key for tree in ['lgbm', 'catboost', 'xgboost', 'extratrees']):
                                if 'lgbm' in model_type_key:
                                    model_params['params']['n_estimators'] = dynamic_config.n_estimators
                                elif 'catboost' in model_type_key:
                                    model_params['params']['iterations'] = dynamic_config.iterations
                                    tprint_info("Applying CatBoost GPU (Apple M1) configuration...")
                                    model_params['params']['task_type'] = 'GPU'
                                    model_params['params']['devices'] = '0' # Use '0' for Apple M1 GPU
                                    
                                    # Remove subsample if it exists, as it's not supported by CatBoost GPU
                                    if 'subsample' in model_params['params']:
                                        del model_params['params']['subsample']
                                        tprint_info("Removed 'subsample' param, not supported by CatBoost GPU.")
                                        
                                model_params['params']['learning_rate'] = dynamic_config.learning_rate
                            
                            tprint_info(f"  Updated {model_name} with dynamic parameters")
                
                # Update hardware settings
                if 'hardware' in yaml_config[config_key]:
                    yaml_config[config_key]['hardware'].update({
                        'memory_limit_gb': dynamic_config.memory_limit_gb,
                        'max_workers': dynamic_config.max_workers
                    })
                elif 'hardware' in yaml_config:
                    yaml_config['hardware'].update({
                        'memory_limit_gb': dynamic_config.memory_limit_gb,
                        'max_workers': dynamic_config.max_workers
                    })
            
            # Update root-level hardware settings if present
            if 'hardware' in yaml_config:
                yaml_config['hardware'].update({
                    'memory_limit_gb': dynamic_config.memory_limit_gb,
                    'max_workers': dynamic_config.max_workers
                })
            
            # Store HPO settings
            yaml_config['hpo_max_trials'] = dynamic_config.hpo_max_trials
            yaml_config['hpo_time_budget_seconds'] = dynamic_config.hpo_time_budget_seconds
            
            tprint_success("✅ Dynamic configuration applied successfully")
            return yaml_config
            
        except Exception as e:
            tprint_error(f"Failed to apply dynamic config: {e}")
            self.logger.error(f"Dynamic config application error: {e}")
            return yaml_config

    async def _perform_hierarchical_hpo(
        self,
        training_data: pd.DataFrame,
        targets: Any,
        model_config: Dict[str, Any],
        config_file: str,
        config: Dict[str, Any],
        training_type: str
    ) -> Dict[str, Any]:
        """Temporary compatibility stub for hierarchical HPO.

        Currently returns model_config unchanged and logs that HPO is skipped.
        This prevents AttributeError while keeping the HPO call-site intact.
        """
        try:
            from src.utils.tprint import tprint_warning
        except Exception:
            # If tprint is unavailable for any reason, silently skip HPO.
            return model_config
        tprint_warning(
            "⚠️ Hierarchical HPO (_perform_hierarchical_hpo) is not yet fully "
            "implemented in UnifiedModelsTrainingStep; skipping HPO and "
            "reusing YAML model configuration as-is."
        )
        return model_config

    def _apply_light_mode_filter(self, training_data: pd.DataFrame, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Apply light mode filtering to training data if needed."""
        try:
            execution_mode = config.get('execution_mode', 'light')
            
            # DEBUG: Log original data information
            if training_data is not None:
                tprint_info(f"🔍 [DEBUG] Original training data shape: {training_data.shape}")
                tprint_info(f"🔍 [DEBUG] Original data index range: {training_data.index.min()} to {training_data.index.max()}")
                tprint_info(f"🔍 [DEBUG] Timeframe: {timeframe}")
                tprint_info(f"🔍 [DEBUG] Execution mode: {execution_mode}")
                
                # Calculate expected samples based on timeframe and date range
                # Only applies when the index is datetime-like; skip for integer/other indexes
                idx = training_data.index
                if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
                    date_range = idx.max() - idx.min()
                    tprint_info(f"🔍 [DEBUG] Date range: {date_range}")
                    
                    # Estimate expected samples based on timeframe
                    if timeframe == '15m':
                        expected_samples_per_day = 24 * 4  # 24 hours * 4 quarters per hour
                        total_days = date_range.days + (date_range.seconds / (24 * 3600))
                        expected_total_samples = int(total_days * expected_samples_per_day)
                        tprint_info(f"🔍 [DEBUG] Expected samples: {expected_total_samples} (based on {total_days:.1f} days × {expected_samples_per_day} samples/day)")
                        
                        # Check if we have significantly fewer samples than expected
                        if len(training_data) < expected_total_samples * 0.5:
                            tprint_warning(f"⚠️ [DEBUG] Sample count anomaly: Have {len(training_data)} samples but expected ~{expected_total_samples}")
                            tprint_warning(f"   This suggests missing data or data quality issues")
                else:
                    tprint_info("🔍 [DEBUG] Skipping date-range based light-mode diagnostics (non-datetime index)")
            
            if execution_mode == 'light' and training_data is not None:
                # Limit to 1000 samples in light mode
                if len(training_data) > 1000:
                    tprint_info(f"Light mode: Limiting training data from {len(training_data)} to 1000 samples")
                    training_data = training_data.tail(1000)
            
            # Log final data shape after filtering
            if training_data is not None:
                tprint_info(f"🔍 [DEBUG] Final training data shape after filtering: {training_data.shape}")
                tprint_info(f"🔍 [DEBUG] Final data index range: {training_data.index.min()} to {training_data.index.max()}")
            
            return training_data
            
        except Exception as e:
            self.logger.warning(f"Error applying light mode filter: {e}")
            return training_data

    def _apply_robust_target_transform(self, targets: pd.Series, name: str = "targets") -> pd.Series:
        """Apply a robust, quantile-based clipping to target values.

        This reduces the influence of extreme outliers on regression losses and HPO
        while preserving the bulk of the label distribution.
        """
        try:
            if targets is None:
                return targets

            # Ensure Series with index preserved
            if not isinstance(targets, pd.Series):
                try:
                    targets = pd.Series(targets)
                except Exception:
                    return targets

            # For strictly binary labels, skip transformation
            unique_vals = pd.unique(targets.dropna())
            if len(unique_vals) <= 3 and set(unique_vals).issubset({0, 1, -1}):
                return targets

            finite_mask = np.isfinite(targets.astype(float))
            if finite_mask.sum() < 50:
                # Too few samples for stable quantiles
                return targets

            values = targets[finite_mask].astype(float)

            # Robust two-sided clipping around the central mass
            lower_q = 0.005
            upper_q = 0.995
            q_low = np.nanquantile(values, lower_q)
            q_high = np.nanquantile(values, upper_q)

            if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
                return targets

            pre_min, pre_max = float(np.nanmin(values)), float(np.nanmax(values))
            clipped = values.clip(q_low, q_high)

            # Log diagnostics for monitoring
            tprint_info(
                f"🔧 Robust target transform applied to {name}: "
                f"q[{lower_q:.3f}]={q_low:.5f}, q[{upper_q:.3f}]={q_high:.5f}, "
                f"pre-range=({pre_min:.5f},{pre_max:.5f}), "
                f"post-range=({float(clipped.min()):.5f},{float(clipped.max()):.5f})"
            )

            result = targets.copy()
            result.loc[finite_mask] = clipped
            return result

        except Exception as e:
            self.logger.warning(f"Robust target transform failed for {name}: {e}")
            return targets

    def _normalize_labeled_data_index(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize labeled_data index to a canonical DatetimeIndex.

        Mirrors the logic used in specialist_feature_diagnostics._prepare_labels so
        that all consumers (diagnostics, training, specialist alignment) share the
        same timestamp normalization behaviour.
        """
        if "timestamp" in labeled_df.columns:
            ts = pd.to_datetime(labeled_df["timestamp"], utc=True, errors="coerce")
            try:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                ts = ts.dt.tz_localize(None)
            valid_mask = ~ts.isna()
            labeled_df = labeled_df.loc[valid_mask].copy()
            ts = ts[valid_mask]
            labeled_df.index = ts
        elif "close_time" in labeled_df.columns:
            close_col = labeled_df["close_time"]
            try:
                if pd.api.types.is_datetime64_any_dtype(close_col):
                    ts = pd.to_datetime(close_col, utc=True, errors="coerce")
                else:
                    close_numeric = pd.to_numeric(close_col, errors="coerce")
                    ts = pd.to_datetime(close_numeric, unit="ms", utc=True, errors="coerce")
            except Exception:
                ts = pd.to_datetime(close_col, utc=True, errors="coerce")
            try:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                ts = ts.dt.tz_localize(None)
            valid_mask = ~ts.isna()
            labeled_df = labeled_df.loc[valid_mask].copy()
            ts = ts[valid_mask]
            labeled_df.index = ts
        elif isinstance(labeled_df.index, pd.DatetimeIndex):
            idx = labeled_df.index
            if idx.tz is not None:
                try:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                except Exception:
                    idx = idx.tz_localize(None)
            labeled_df = labeled_df.copy()
            labeled_df.index = idx
        else:
            raise ValueError(
                "labeled_data has neither 'timestamp'/'close_time' column nor DatetimeIndex"
            )
        return labeled_df

    async def _retrieve_training_data(self, config: Dict[str, Any], yaml_config: Dict[str, Any]) -> tuple:
        """Retrieve training data and targets from artifacts with fast-fail on missing data."""
        try:
            tprint_info("🔍 Retrieving training data from feature generation artifacts...")
            
            training_data = None
            analyst_targets = None
            tactician_targets = None

            # --- FIX: Check for pre-loaded ensemble features/targets first ---
            if 'ensemble_features' in config:
                tprint_info("🎯 Using pre-loaded ensemble features from config")
                training_data = config['ensemble_features']
                
                # Retrieve targets from config if available
                targets_df = config.get('target_data')
                
                if targets_df is not None:
                    # For analyst ensemble, we typically use the targets passed in
                    if config.get('execution_context') == 'analyst':
                         analyst_targets = targets_df
                    elif config.get('execution_context') == 'tactician':
                         tactician_targets = targets_df
                    else:
                         # Fallback to analyst targets
                         analyst_targets = targets_df
                
                tprint_success(f"✅ Loaded ensemble features: {training_data.shape}")
                return training_data, analyst_targets, tactician_targets
            # --- END FIX ---
            
            # ========================================================================
            # FEATURE LOADING FROM HDF5 VERSIONED ARTIFACTS
            # ========================================================================
            # Determine feature set size to use. For analyst_base we default to the
            # compact 40-feature final selection; for other modes we retain 60 as a
            # safe default unless explicitly overridden in config.
            training_type_local = str(config.get('training_type', 'analyst_base')).lower()
            default_feature_set_size = 40 if 'analyst_base' in training_type_local else 60
            feature_set_size = int(config.get('feature_set_size', default_feature_set_size))

            execution_mode = str(config.get('execution_mode', 'full')).lower()
            lookback_days = int(config.get('lookback_days', 0) or 0)
            is_blank_analyst_base = 'analyst_base' in training_type_local and execution_mode == 'blank'

            tprint_info("=" * 80)
            tprint_info("📦 LOADING FEATURES FROM HDF5 VERSIONED ARTIFACTS")
            tprint_info("=" * 80)
            tprint_info(f"   Source Step: feature_generation_final_feature_selection_step")
            tprint_info(f"   Target Feature Set Size: {feature_set_size} features")
            tprint_info(f"   Storage Format: HDF5 (via versioned_artifacts)")

            feature_source_name = None

            # Specialized path: analyst_base in BLANK mode should use wide ANALYST
            # generated_features_15m filtered by the BLANK-mode selected features.
            if is_blank_analyst_base:
                # For BLANK analyst_base runs, we now rely exclusively on the
                # final compact feature selection produced by
                # feature_generation_final_feature_selection_step
                # ('selected_feature_dataframe_50'). The legacy path that
                # intersected BLANK selections with ANALYST generated_features_15m
                # produced unstable feature sets and is no longer used to build
                # training_data.
                tprint_info(
                    "   ℹ️ BLANK analyst_base: using 'selected_feature_dataframe_50' from "
                    "feature_generation_final_feature_selection_step as the canonical "
                    "training feature set."
                )

            # If specialized path did not produce training_data, fall back to standard selected_feature_dataframe_* artifacts
            if training_data is None:
                # For BLANK analyst_base, first attempt to pick the longest-window
                # selected_feature_dataframe_50 directly from the BLANK/FULL
                # VersionedArtifactStores so we do not accidentally use a tiny
                # 1204-row slice via the generic artifact path when much larger
                # artifacts (e.g. 170k+ rows) are available.
                if is_blank_analyst_base:
                    try:
                        symbol_cfg = config.get('symbol', 'ETHUSDT')
                        exchange_cfg = config.get('exchange', 'binance')
                        timeframe_cfg = config.get('timeframe', '15m')
                        direction_cfg = config.get('direction', 'long')

                        # Require at least this many rows to consider a candidate
                        # a valid long-window training frame. Default ~50k rows
                        # (~520 days at 15m) ensures we have plenty of history
                        # before later trimming to the 1-year blank window.
                        min_rows_threshold = int(config.get('min_blank_training_rows', 50_000))

                        # Collect candidate (store_path, version) pairs without
                        # materializing all of them, then scan from most recent
                        # to oldest and materialize at most a small number.
                        candidate_versions: List[Tuple[str, str]] = []
                        for mode_suffix in ['blank', 'full']:
                            store_path = (
                                f"versioned_artifacts/{symbol_cfg}_{exchange_cfg}_"
                                f"{timeframe_cfg}_{direction_cfg}_{mode_suffix}"
                            )
                            try:
                                store = VersionedArtifactStore(store_path)
                                versions = store.list_versions()
                            except Exception:
                                continue

                            for v in versions:
                                if 'selected_feature_dataframe_50' in v:
                                    candidate_versions.append((store_path, v))

                        if candidate_versions:
                            # Apply execution_mode preference before scanning:
                            # - blank/light: consider all stores
                            # - full: prefer *_full store; if none, fall back to all
                            if execution_mode == "full":
                                full_only = [cv for cv in candidate_versions if cv[0].endswith("_full")]
                                if full_only:
                                    candidate_versions = full_only

                            # Sort by version string descending (newest first)
                            candidate_versions_sorted = sorted(
                                candidate_versions,
                                key=lambda x: x[1],
                                reverse=True,
                            )

                            max_versions_to_check = 5
                            checked = 0
                            for store_path, version in candidate_versions_sorted:
                                if checked >= max_versions_to_check:
                                    break
                                checked += 1
                                try:
                                    store = VersionedArtifactStore(store_path)
                                    view = store.get_view(version)
                                    df = view.materialize()
                                    if not isinstance(df, pd.DataFrame):
                                        df = pd.DataFrame(df)
                                    rows = len(df)
                                    tprint_info(
                                        f"   🔎 Evaluating candidate '{version}' from {store_path}: rows={rows}"
                                    )
                                    if rows >= min_rows_threshold:
                                        training_data = df
                                        feature_source_name = f"{store_path}:{version}"
                                        tprint_success(
                                            "✅ BLANK analyst_base: using long-window 'selected_feature_dataframe_50' from "
                                            f"{store_path} version {version} with {rows} rows as training features "
                                            f"(checked {checked} candidate(s))."
                                        )
                                        break
                                    else:
                                        tprint_info(
                                            f"   ↪ Candidate '{version}' below min_rows_threshold "
                                            f"({rows} < {min_rows_threshold}), skipping"
                                        )
                                except Exception as mat_exc:
                                    tprint_warning(
                                        f"⚠️ Failed to materialize candidate '{version}' from {store_path}: {mat_exc}"
                                    )

                            if training_data is None:
                                tprint_warning(
                                    "⚠️ BLANK analyst_base: no 'selected_feature_dataframe_50' candidate "
                                    f"with >= {min_rows_threshold} rows found after checking up to "
                                    f"{max_versions_to_check} version(s); falling back to generic artifact path."
                                )
                    except Exception as e_sel:
                        tprint_warning(
                            "⚠️ BLANK analyst_base: failed to select long-window "
                            "selected_feature_dataframe_50 from VersionedArtifactStores; "
                            f"falling back to generic artifact path: {e_sel}"
                        )

                # Prefer compact final feature selection for analyst_base; broad
                # generated_features_* matrices are no longer used for training.
                if training_data is None:
                    if 'analyst_base' in training_type_local:
                        # For analyst_base (including BLANK), require the final 50-feature
                        # DataFrame from feature_generation_final_feature_selection_step.
                        # Do not silently fall back to older selections; fail fast if
                        # this artifact is unavailable or invalid.
                        feature_artifact_names = [
                            'selected_feature_dataframe_50',
                        ]
                    else:
                    # Non-analyst_base training keeps the original behaviour
                        feature_artifact_names = [
                            f'selected_feature_dataframe_{feature_set_size}',  # Primary: selected features DataFrame (with size from config)
                            'selected_feature_dataframe_60',                   # Standard 60-feature DataFrame
                        ]

                    tprint_info(f"🔎 Attempting to load training features from HDF5 artifacts...")

                    for artifact_name in feature_artifact_names:
                        try:
                            tprint_info(f"   ↪ Trying '{artifact_name}'")
                            training_data = self._get_artifact(artifact_name, 'data')
                            if training_data is not None:
                                feature_source_name = artifact_name
                                tprint_success(
                                    f"✅ Retrieved training features from '{artifact_name}': "
                                    f"{training_data.shape if hasattr(training_data, 'shape') else type(training_data)}"
                                )
                                break
                            else:
                                tprint_warning(
                                    f"⚠️ Artifact '{artifact_name}' returned None (metadata exists but data file missing?)"
                                )
                        except Exception as e:
                            self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                            continue
            
            # FAIL FAST: If no training data DataFrame found, raise error
            if training_data is None or not isinstance(training_data, pd.DataFrame):
                if 'analyst_base' in training_type_local:
                    error_msg = (
                        "❌ CRITICAL: No training feature DataFrame found for analyst_base!\n"
                        "   Required artifact (from 'feature_generation_final_feature_selection_step'):\n"
                        "   - selected_feature_dataframe_50\n"
                        "   No fallback to older selections or generated_features_15m is permitted."
                    )
                else:
                    error_msg = (
                        "❌ CRITICAL: No training feature DataFrame found!\n"
                        f"   Allowed artifacts (from 'feature_generation_final_feature_selection_step'):\n"
                        f"   - selected_feature_dataframe_{feature_set_size}\n"
                        f"   - selected_feature_dataframe_60\n"
                        f"   No fallback to feature name lists or labeled_data reconstruction is permitted."
                    )
                tprint_error(error_msg)
                raise ValueError(error_msg)

            if training_data is not None and isinstance(training_data, pd.DataFrame):
                # ================================================================
                # ANALYST_BASE DENSE FEATURE RECONSTRUCTION
                # ================================================================
                # For analyst_base training, the selected_feature_dataframe_50 may
                # be sparse (event-only rows). We want dense OOF predictions over
                # the full 15m grid. To achieve this:
                # 1. Extract selected feature column names from the sparse frame
                # 2. Load dense analyst_combined_features from interaction step
                # 3. Apply selected columns to dense frame, preserving all rows
                #
                # This ensures analyst_base models can predict on all 15m bars.
                if 'analyst_base' in training_type_local:
                    selected_feature_cols = [
                        c for c in training_data.columns
                        if c not in {'timestamp', 'target', 'label', 'target_long', 'target_short'}
                        and not c.lower().endswith('_target')
                        and not c.lower().endswith('_label')
                    ]
                    
                    # Try to load dense analyst_combined_features
                    dense_base = None
                    try:
                        # First try from interaction generation step context
                        original_step = getattr(self.artifact_manager, '_current_step_name', None)
                        try:
                            self.artifact_manager.set_context(
                                step_name='feature_generation_interaction_generation_step',
                                symbol=config.get('symbol', 'ETHUSDT'),
                                exchange=config.get('exchange', 'binance'),
                                timeframe=config.get('timeframe', '15m'),
                                direction=config.get('direction', 'long'),
                                model='analyst',
                            )
                            dense_base = self._get_artifact('analyst_combined_features', 'data')
                        finally:
                            # Restore original context
                            if original_step:
                                self.artifact_manager.set_context(step_name=original_step)
                    except Exception as e:
                        tprint_info(f"   ↪ Could not load analyst_combined_features: {e}")
                        dense_base = None
                    
                    if dense_base is not None and isinstance(dense_base, pd.DataFrame):
                        # Find overlap between selected features and dense base
                        overlap_cols = [c for c in selected_feature_cols if c in dense_base.columns]
                        
                        if len(overlap_cols) >= len(selected_feature_cols) * 0.5:  # At least 50% overlap
                            sparse_rows = len(training_data)
                            dense_rows = len(dense_base)
                            
                            tprint_info("=" * 80)
                            tprint_info("🔄 ANALYST_BASE DENSE FEATURE RECONSTRUCTION")
                            tprint_info("=" * 80)
                            tprint_info(f"   Sparse selected_feature_dataframe: {sparse_rows} rows")
                            tprint_info(f"   Dense analyst_combined_features: {dense_rows} rows")
                            tprint_info(f"   Selected feature columns: {len(selected_feature_cols)}")
                            tprint_info(f"   Overlap columns: {len(overlap_cols)}")
                            
                            # Use dense base with selected columns
                            new_training_data = dense_base[overlap_cols].copy()
                            
                            # Preserve target columns from the sparse frame if needed
                            target_cols = [c for c in training_data.columns 
                                         if 'target' in c.lower() or 'label' in c.lower()]
                            for tc in target_cols:
                                if tc in training_data.columns:
                                    # Reindex targets to dense index
                                    try:
                                        new_training_data[tc] = training_data[tc].reindex(dense_base.index)
                                        non_null = new_training_data[tc].notna().sum()
                                        tprint_info(f"   ✅ Merged target '{tc}': {non_null} non-null values")
                                    except Exception as e:
                                        tprint_warning(f"   ⚠️ Could not merge target '{tc}': {e}")
                            
                            training_data = new_training_data
                            feature_source_name = f"dense_reconstruction:{feature_source_name}"
                            tprint_success(
                                f"✅ Reconstructed dense training data: {training_data.shape} "
                                f"({dense_rows} rows vs {sparse_rows} sparse)"
                            )
                            tprint_info("=" * 80)
                        else:
                            tprint_info(
                                f"   ℹ️ Skipping dense reconstruction: only {len(overlap_cols)}/{len(selected_feature_cols)} "
                                f"columns overlap with analyst_combined_features"
                            )
                    else:
                        tprint_info("   ℹ️ Dense analyst_combined_features not available; using sparse frame")

                # Comprehensive feature loading verification and logging
                tprint_info("=" * 80)
                tprint_info("📊 COMPREHENSIVE FEATURE LOADING VERIFICATION")
                tprint_info("=" * 80)
                tprint_success(f"✅ HDF5 Access Verified: Successfully loaded from '{feature_source_name or 'unknown_source'}'")
                tprint_info(f"📦 Feature DataFrame Shape: {training_data.shape} ({training_data.shape[0]:,} samples × {training_data.shape[1]:,} features)")
                tprint_info(f"💾 Memory Usage: {training_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                tprint_info(f"🔢 Data Types Distribution: {dict(training_data.dtypes.value_counts())}")
                tprint_info(f"📋 Column Names (first 20): {list(training_data.columns[:20])}")
                if len(training_data.columns) > 20:
                    tprint_info(f"    ... and {len(training_data.columns) - 20} more columns")

                # Explicit log: which features are loaded from selected_feature_dataframe
                exclude_cols = {'timestamp', 'target', 'label', 'target_long', 'target_short'}
                loaded_feature_names = [
                    c for c in training_data.columns
                    if c not in exclude_cols and not c.lower().endswith('_target') and not c.lower().endswith('_label')
                ]
                tprint_info("🧾 Loaded features from selected_feature_dataframe (pre-cleaning):")
                # Print in chunks to avoid overly long single lines
                chunk_size = 25
                for i in range(0, len(loaded_feature_names), chunk_size):
                    chunk = loaded_feature_names[i:i+chunk_size]
                    tprint_info(f"  - {i:03d}-{i+len(chunk)-1:03d}: {chunk}")

                # Check for missing data
                null_counts = training_data.isnull().sum()
                cols_with_nulls = null_counts[null_counts > 0]
                if len(cols_with_nulls) > 0:
                    tprint_warning(f"⚠️ Found {len(cols_with_nulls)} columns with missing values")
                    tprint_info(f"   Top 10 columns with most nulls:")
                    for col, count in cols_with_nulls.nlargest(10).items():
                        pct = (count / len(training_data)) * 100
                        tprint_info(f"      - {col}: {count:,} ({pct:.1f}%)")
                else:
                    tprint_success("✅ No missing values detected")

                # Check for constant/zero-variance columns
                numeric_cols = training_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    std_devs = training_data[numeric_cols].std()
                    zero_var_cols = std_devs[std_devs == 0].index.tolist()
                    if len(zero_var_cols) > 0:
                        tprint_warning(f"⚠️ Found {len(zero_var_cols)} zero-variance columns (will be removed during cleaning)")
                    else:
                        tprint_success("✅ All numeric columns have variance")

                tprint_info("=" * 80)

                # Use tprint_data_preview for comprehensive data visualization
                from src.utils.tprint import tprint_data_preview
                tprint_data_preview(
                    training_data,
                    name=f"Loaded Training Features ({feature_source_name})",
                    max_rows=5,
                    max_cols=10,
                    show_dtypes=True,
                    show_shape=True
                )

                self._log_feature_snapshot(training_data, feature_source_name or 'unknown_source', prefix='📥 Raw load ')
                tprint_info(
                    f"🧪 Raw feature frame -> shape={training_data.shape}, columns={len(training_data.columns)}, "
                    f"dtypes={training_data.dtypes.value_counts().to_dict()}"
                )

                # Drop duplicate columns to avoid shape mismatches later
                if training_data.columns.duplicated().any():
                    duplicate_cols = training_data.columns[training_data.columns.duplicated()].unique().tolist()
                    tprint_warning(f"🧹 Dropping duplicate columns ({len(duplicate_cols)}): {duplicate_cols}")
                    training_data = training_data.loc[:, ~training_data.columns.duplicated()].copy()

                # Drop columns that are entirely NaN or have insufficient valid data
                min_valid_threshold = 0.01  # Require at least 1% valid data
                empty_cols = []
                for col in training_data.columns:
                    valid_count = training_data[col].notna().sum()
                    valid_ratio = valid_count / len(training_data)
                    if valid_ratio < min_valid_threshold:
                        empty_cols.append(col)

                if empty_cols:
                    tprint_warning(f"⚠️ Dropping {len(empty_cols)} columns with insufficient valid data (<{min_valid_threshold*100}%): {empty_cols[:10]}{'...' if len(empty_cols) > 10 else ''}")
                    training_data = training_data.drop(columns=empty_cols)
                else:
                    tprint_info("✅ All columns have sufficient valid data")

                # Remove non-numeric columns (they break model training/HPO)
                non_numeric_cols = training_data.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
                if non_numeric_cols:
                    preview = non_numeric_cols[:10]
                    suffix = '...' if len(non_numeric_cols) > 10 else ''
                    tprint_warning(f"⚠️ Dropping {len(non_numeric_cols)} non-numeric columns: {preview}{suffix}")
                    training_data = training_data.drop(columns=non_numeric_cols)
                else:
                    tprint_info("✅ No non-numeric columns detected during cleaning")

                # Convert boolean columns to numeric floats for model compatibility
                bool_cols = training_data.select_dtypes(include=['bool']).columns.tolist()
                if bool_cols:
                    training_data[bool_cols] = training_data[bool_cols].astype(np.float32)
                    tprint_info(f"ℹ️ Converted boolean columns to float: {bool_cols}")
                else:
                    tprint_info("✅ No boolean columns required conversion")

                # Remove obvious target columns that might have slipped into the feature frame
                # BUT preserve regime probabilities (target_regime_*) which are legitimate features.
                # Treat meta-label outputs and base directional signals as pseudo-targets.
                meta_pseudo_targets = {
                    'binary_label',
                    'binary_label_long',   # NEW: Direction-specific binary label
                    'binary_label_short',  # NEW: Direction-specific binary label
                    'smoothed_label',
                    'realized_return',
                    'label_uncertainty',
                    'log_ret',
                    'primary_signal',
                }
                potential_target_cols = [
                    col
                    for col in training_data.columns
                    if (
                        col.lower() in {'target', 'label', 'target_long', 'target_short'}
                        or col.lower().endswith('_target')
                        or col.lower().endswith('_label')
                        or col.lower() in meta_pseudo_targets
                    )
                    and not col.lower().startswith('target_regime')  # Preserve regime probabilities
                ]
                if potential_target_cols:
                    tprint_warning(f"⚠️ DATA LEAKAGE PREVENTION: Dropping target-like columns from features: {potential_target_cols}")
                    training_data = training_data.drop(columns=potential_target_cols)
                else:
                    tprint_info("✅ No target-like columns detected in feature frame")

                # Remove metadata columns that are not features
                metadata_col_patterns = [
                    'labeling_method', 'labeling_timestamp', 'base_threshold',
                    'lookahead_periods', 'optimization_iteration', 'quality_acceptance_rate'
                ]
                metadata_cols_to_drop = [
                    col for col in training_data.columns
                    if any(pattern in col.lower() for pattern in metadata_col_patterns)
                ]
                if metadata_cols_to_drop:
                    tprint_warning(f"⚠️ Dropping metadata columns from features: {metadata_cols_to_drop}")
                    training_data = training_data.drop(columns=metadata_cols_to_drop)
                else:
                    tprint_info("✅ No metadata columns detected in feature frame")

                if training_data.empty:
                    raise ValueError("All feature columns were removed during cleaning; check upstream artifacts.")

                # Log data modification with comprehensive preview
                tprint_info("=" * 80)
                tprint_info("🧹 DATA MODIFICATION: After Cleaning")
                tprint_info("=" * 80)
                tprint_data_preview(
                    training_data,
                    name="Cleaned Training Features",
                    max_rows=5,
                    max_cols=10,
                    show_dtypes=True,
                    show_shape=True
                )

                self._log_feature_snapshot(training_data, feature_source_name or 'unknown_source', prefix='🧹 Cleaned ')
                tprint_info(
                    f"🧼 Post-cleaning feature frame -> shape={training_data.shape}, columns={len(training_data.columns)}, "
                    f"dtypes={training_data.dtypes.value_counts().to_dict()}"
                )

            # ========================================================================
            # LABELS/TARGETS LOADING FROM HDF5 VERSIONED ARTIFACTS
            # ========================================================================
            # Get targets from labeling integration/meta-labeling steps (direction-aware)
            direction = config.get('direction', 'long')
            tprint_info("=" * 80)
            tprint_info("🎯 LOADING LABELS/TARGETS FROM HDF5 VERSIONED ARTIFACTS")
            tprint_info("=" * 80)
            tprint_info(f"   Source Step: feature_generation_labeling_integration_step / feature_generation_meta_labeling_step")
            tprint_info(f"   Direction: {direction}")
            tprint_info(f"   Storage Format: HDF5 (via versioned_artifacts)")

            training_type = config.get('training_type', 'analyst_base')
            prefer_meta_label_targets = 'analyst_base' in str(training_type).lower()

            # Legacy label-loading path for non-analyst_base training types can be
            # reintroduced if needed. For analyst_base we always prefer the
            # meta-labeling labeled_data targets loaded below.
            if not prefer_meta_label_targets:
                pass

            # ========================================================================
            # META-LABELING TARGETS (analyst_base preferred path)
            # ========================================================================
            if prefer_meta_label_targets and analyst_targets is None:
                symbol_cfg = config.get('symbol', 'ETHUSDT')
                timeframe_cfg = config.get('timeframe', '15m')
                for artifact_name_candidate in [
                    f"labeled_data_{symbol_cfg}_{timeframe_cfg}",
                    'labeled_data',
                ]:
                    try:
                        ld = self._get_artifact(artifact_name_candidate, 'data')
                        if ld is not None and isinstance(ld, pd.DataFrame):
                            tprint_info(f"✅ Found labeled_data artifact '{artifact_name_candidate}': {ld.shape}")
                            # Prefer fused_target_long for analyst_base
                            for target_col in ['fused_target_long', 'target_long', 'smoothed_label']:
                                if target_col in ld.columns:
                                    analyst_targets = ld[target_col]
                                    tprint_success(f"✅ Using '{target_col}' from labeled_data as analyst_targets")
                                    break
                            if analyst_targets is not None:
                                break
                    except Exception as e:
                        self.logger.debug(f"Artifact '{artifact_name_candidate}' not found: {e}")
                        continue

            return training_data, analyst_targets, tactician_targets

        except Exception as e:
            self.logger.error(f"Error retrieving training data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    async def _get_regime_features(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Get regime probability features from regime steps.

        NOTE: This method is currently unused. Regime features are loaded via _get_additional_model_outputs.
        Kept for potential future use.
        """
        try:
            tprint_info("🌍 Loading regime probabilities for feature preparation...")

            # Try ensemble predictions first, then base model predictions
            regime_features = None
            for artifact_name in ['regime_ensemble_predictions', 'regime_models_predictions']:
                try:
                    regime_features = self._get_artifact(artifact_name, 'data')
                    if regime_features is not None:
                        tprint_success(f"✅ Loaded regime features from '{artifact_name}': {regime_features.shape}")
                        tprint_info(f"   Columns: {list(regime_features.columns)}")
                        break
                except Exception as e:
                    self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                    continue

            if regime_features is None:
                tprint_warning("⚠️ No regime features found - will use uniform distribution")

            return regime_features

        except Exception as e:
            self.logger.error(f"Error retrieving regime features: {e}")
            tprint_warning(f"⚠️ Failed to load regime features: {e}")
            return None

    async def _get_additional_model_outputs(self, training_type: str, config: Dict[str, Any], training_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get additional model outputs based on training type."""
        try:
            additional_features_list = []
            base_outputs_for_stats = None # Store the specific DataFrame to calculate stats on
            # --- REGIME FEATURES: Specialist models + Legacy Ensemble/Model ---
            # We treat all available regime sources as additional feature blocks rather
            # than strict fallbacks.
            have_regime_features = False

            # Common context for all regime sources
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            regime_timeframe = config.get('regime_timeframe', config.get('timeframe', '1h'))
            direction = config.get('direction', 'long')

            # 0) Specialist model outputs (ML Risk, HMM Alpha, Liquidity, SMC, etc.)
            try:
                # For analyst_base, regime specialists are critical. Use strict
                # mode so that if all specialist blocks are effectively empty
                # after alignment, the pipeline fast-fails instead of silently
                # training without these features.
                strict_specialists = training_type in ('analyst_base', 'analyst_ensemble')

                # Use canonical per-specialist scalars for unified training so
                # each specialist contributes a small, interpretable set of
                # features rather than full multi-column regime blocks.
                config_for_specialists = dict(config)
                config_for_specialists.setdefault("use_canonical_specialist_scalars", True)

                specialist_df = get_specialist_models_outputs(
                    artifact_router=self.artifact_router,
                    training_index=training_data.index,
                    config=config_for_specialists,
                    logger=self.logger,
                    strict=strict_specialists,
                )

                if specialist_df is not None and not specialist_df.empty:
                    try:
                        base_len = len(training_data)
                        spec_len = len(specialist_df)
                        common_index = training_data.index.intersection(specialist_df.index)
                        common_len = len(common_index)
                        tprint_info(
                            "📏 Base vs specialist index intersection before cleaning:"
                        )
                        tprint_info(
                            f"   Base rows={base_len}, specialist_rows={spec_len}, common_rows={common_len}"
                        )
                        if base_len > 0 and common_len < base_len:
                            dropped = base_len - common_len
                            ratio = dropped / float(base_len) * 100.0
                            tprint_warning(
                                f"   ⚠️ Potential row loss if restricted to common index: "
                                f"drop={dropped} rows ({ratio:.2f}%)"
                            )
                    except Exception as diag_exc:
                        self.logger.debug(f"Failed to log base/specialist intersection diagnostics: {diag_exc}")
                    # Drop specialist columns that are entirely NaN after alignment.
                    # These often come from misaligned or missing specialists
                    # (e.g., SMC block with no overlap) and only introduce
                    # artificial missingness + constant-filled columns downstream.
                    non_null_counts = specialist_df.notna().sum()
                    active_cols = non_null_counts[non_null_counts > 0].index.tolist()
                    dropped_cols = [c for c in specialist_df.columns if c not in active_cols]

                    if not active_cols:
                        tprint_warning(
                            "⚠️ All specialist columns are NaN after alignment; "
                            "skipping specialist block for this run."
                        )
                    else:
                        if dropped_cols:
                            tprint_warning(
                                f"⚠️ Dropping {len(dropped_cols)} all-NaN specialist columns: "
                                f"{dropped_cols[:10]}{'...' if len(dropped_cols) > 10 else ''}"
                            )
                        specialist_df = specialist_df[active_cols]

                        # Capture specialist feature names
                        self._specialist_feature_names = list(specialist_df.columns)
                        tprint_info(f"📝 Captured {len(self._specialist_feature_names)} specialist feature names for model filtering")

                        # Diagnostic: log non-null coverage for key alpha/liquidity/SMC features
                        try:
                            key_prefixes = ("alpha_", "liquidity_", "smc_", "risk_score")
                            key_cols = [
                                c for c in specialist_df.columns
                                if any(p in c.lower() for p in key_prefixes)
                            ]
                            if key_cols:
                                tprint_info("🔍 Specialist coverage diagnostics (non-null ratios for key features):")
                                for c in key_cols:
                                    nnz = int(non_null_counts.get(c, specialist_df[c].notna().sum()))
                                    ratio = nnz / float(len(specialist_df)) if len(specialist_df) > 0 else 0.0
                                    tprint_info(f"   - {c}: non-null={nnz}, ratio={ratio:.3f}")
                        except Exception as diag_exc:
                            self.logger.debug(f"Failed to log specialist coverage diagnostics: {diag_exc}")

                        additional_features_list.append(specialist_df)
                        have_regime_features = True
                        tprint_success(
                            f"✅ Added specialist model outputs: shape={specialist_df.shape}"
                        )
            except Exception as e:
                self.logger.warning(f"Failed to load specialist model outputs: {e}")
                tprint_warning(f"⚠️ Failed to load specialist model outputs: {e}")

            # 1) Legacy regime ensemble / regime model probabilities (OOF/OOS preferred)
            # NOTE: These are only needed for ensemble training. Base models should not depend on
            # regime_ensemble_predictions_* artifacts, because they are the ones producing inputs
            # for the ensemble.
            if training_type in ('analyst_ensemble', 'tactician_ensemble'):
                try:
                    tprint_info("=" * 80)
                    tprint_info("🌍 LOADING LEGACY REGIME ENSEMBLE PROBABILITIES FROM HDF5 VERSIONED ARTIFACTS")
                    tprint_info("=" * 80)
                    tprint_info("   Source Step: regime_ensemble_training / regime_models_training")
                    tprint_info("   Artifact Names: regime_ensemble_predictions[_oof/oos], regime_models_predictions[_oof/oos]")

                    regime_features = None
                    regime_candidates = [
                        'regime_ensemble_predictions_oof',
                        'regime_ensemble_predictions_oos',
                        'regime_models_predictions_oof',
                        'regime_models_predictions_oos',
                        'regime_ensemble_predictions',
                        'regime_models_predictions',
                    ]
                    for artifact_name in regime_candidates:
                        try:
                            regime_features = self._get_artifact(artifact_name, 'data')
                            if regime_features is not None:
                                tprint_success(f"✅ Found regime features in '{artifact_name}' (preferred OOF/OOS if available)")
                                if ('oof' not in artifact_name.lower()) and ('oos' not in artifact_name.lower()):
                                    try:
                                        tprint_warning("⚠️ Regime features are in-sample; creating OOS proxy via 1-step shift (ffill)")
                                        regime_features = regime_features.shift(1).fillna(method='ffill')
                                        try:
                                            saved_path = self._save_artifact(
                                                data=regime_features,
                                                artifact_name='regime_ensemble_predictions_oos',
                                                artifact_type='data',
                                            )
                                            tprint_info(f"   ↪ Saved OOS proxy regime features at: {saved_path}")
                                        except Exception as e:
                                            tprint_warning(f"   ⚠️ Failed to save OOS proxy regime features: {e}")
                                    except Exception as e:
                                        tprint_warning(f"⚠️ Failed to create OOS proxy for regime features: {e}")
                                tprint_info(f"🔍 [DEBUG] Regime features shape: {regime_features.shape}")
                                tprint_info(f"🔍 [DEBUG] Regime features columns: {list(regime_features.columns)}")
                                tprint_info(f"🔍 [DEBUG] Regime features index range: {regime_features.index.min()} to {regime_features.index.max()}")
                                tprint_info("🔍 [DEBUG] First 5 rows of regime features:")
                                tprint_info(f"{regime_features.head().to_string()}")

                                non_finite_mask = ~np.isfinite(regime_features.select_dtypes(include=[np.number]))
                                if non_finite_mask.any().any():
                                    non_finite_counts = non_finite_mask.sum()
                                    tprint_warning("🔍 [DEBUG] Non-finite values found in regime features:")
                                    for col in regime_features.columns:
                                        col_non_finite = non_finite_mask[col].sum()
                                        if col_non_finite > 0:
                                            tprint_warning(
                                                f"   - {col}: {col_non_finite} non-finite values "
                                                f"({col_non_finite/len(regime_features)*100:.1f}%)"
                                            )
                                            non_finite_indices = regime_features.index[non_finite_mask[col]]
                                            non_finite_values = regime_features.loc[non_finite_indices, col]
                                            tprint_info(f"     Sample non-finite values: {non_finite_values.head().to_dict()}")
                                break
                        except Exception as e:
                            self.logger.debug(f"   Artifact '{artifact_name}' not found: {e}")
                            continue

                    if regime_features is not None:
                        tprint_info(
                            f"   ↪ Retrieved regime ensemble/model predictions: shape={regime_features.shape}, "
                            f"columns={len(regime_features.columns)}"
                        )

                        tprint_info("=" * 80)
                        tprint_info("🌍 REGIME FEATURE ADDITION: Loading Regime Probabilities")
                        tprint_info("=" * 80)
                        tprint_data_preview(
                            regime_features,
                            name="Regime Ensemble Predictions (Before Resampling)",
                            max_rows=5,
                            max_cols=10,
                            show_dtypes=True,
                            show_shape=True,
                        )

                        if not regime_features.index.equals(training_data.index):
                            tprint_warning(
                                f"Regime features index mismatch. Resampling {len(regime_features)} rows to match {len(training_data)} rows."
                            )
                            regime_features_resampled = regime_features.reindex(training_data.index, method='ffill')
                            tprint_info(
                                f"   ↪ Resampled regime features -> shape={regime_features_resampled.shape}, "
                                f"columns={len(regime_features_resampled.columns)}"
                            )

                            tprint_info("=" * 80)
                            tprint_info("🌍 DATA MODIFICATION: Regime Features After Resampling")
                            tprint_info("=" * 80)
                            tprint_data_preview(
                                regime_features_resampled,
                                name="Regime Features (After Resampling)",
                                max_rows=5,
                                max_cols=10,
                                show_dtypes=True,
                                show_shape=True,
                            )

                            additional_features_list.append(regime_features_resampled)
                        else:
                            tprint_info("   ↪ Regime features already aligned with training index")
                            additional_features_list.append(regime_features)

                        have_regime_features = True
                        tprint_success("✅ Added legacy regime ensemble/model probability features.")
                except ValueError:
                    # Re-raise ValueError for fast-fail
                    raise
                except Exception as e:
                    error_msg = f"❌ CRITICAL: Failed to load legacy regime ensemble/model predictions: {e}"
                    tprint_error(error_msg)
                    raise ValueError(error_msg) from e

            # Final safety check: require at least one regime feature source. For ensemble
            # training this is mandatory; for base training we proceed without regime
            # features if none are available.
            if not have_regime_features:
                if training_type in ('analyst_ensemble', 'tactician_ensemble'):
                    error_msg = (
                        "❌ CRITICAL: No regime predictions artifact found!\n"
                        "   Expected at least one of the following regime sources:\n"
                        "   - specialist model outputs (ML Risk / HMM Alpha / Liquidity) via get_specialist_models_outputs\n"
                        "   - regime_ensemble_predictions / regime_models_predictions (from regime_ensemble_training / regime_models_training)\n"
                        "   \n"
                        "   This artifact is REQUIRED for ensemble model training.\n"
                        "   Format: HDF5 (versioned_artifacts)\n"
                    )
                    tprint_error(error_msg)
                    raise ValueError(error_msg)
                else:
                    tprint_warning(
                        "⚠️ No regime prediction features found; proceeding without regime blocks "
                        f"for base training type '{training_type}'."
                    )

            # --- END: Regime feature loading ---


            if training_type == 'analyst_ensemble':
                # Base models for analyst_ensemble are the analyst_base outputs.
                # To avoid leakage, restrict to OUT-OF-FOLD artifacts only.
                base_candidates = [
                    'analyst_base_outputs_oof',
                    'analyst_base_predictions_oof'
                ]
                base_outputs = None
                for name in base_candidates:
                    candidate = self._get_artifact(name, 'data')
                    if candidate is not None:
                        base_outputs = candidate
                        tprint_info(f"   ↪ Using additional features from '{name}' (OOF-only for stacking)")
                        break

                if base_outputs is None:
                    error_msg = (
                        "❌ OOF analyst base outputs not found for analyst_ensemble training.\n"
                        "   Expected one of: 'analyst_base_outputs_oof' or 'analyst_base_predictions_oof'.\n"
                        "   Please rerun analyst_base_training to generate OOF artifacts before training the ensemble."
                    )
                    tprint_error(error_msg)
                    raise ValueError(error_msg)

                # Enforce strict temporal alignment
                if not hasattr(base_outputs, 'index') or not isinstance(base_outputs.index, type(training_data.index)):
                    raise ValueError("Analyst base outputs lack proper DatetimeIndex for alignment")
                if not base_outputs.index.is_monotonic_increasing:
                    base_outputs = base_outputs.sort_index()
                if not training_data.index.is_monotonic_increasing:
                    training_data.sort_index(inplace=True)
                if not base_outputs.index.equals(training_data.index):
                    tprint_warning("Aligning 'analyst_base_outputs' to training_data index (ffill)")
                    base_outputs = base_outputs.reindex(training_data.index, method='ffill')
                    tprint_info(
                        f"   ↪ Resampled analyst_base_outputs -> shape={base_outputs.shape}, columns={len(base_outputs.columns)}"
                    )
                additional_features_list.append(base_outputs)
                base_outputs_for_stats = base_outputs  # For disagreement features
                tprint_info(
                    f"   ↪ Added analyst_base_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                )

            elif training_type == 'tactician_base':
                # Base model for tactician_base is the analyst_ensemble output
                analyst_candidates = [
                    'analyst_ensemble_outputs_oof',
                    'analyst_ensemble_outputs_oos',
                    'analyst_ensemble_outputs'
                ]
                analyst_outputs = None
                for name in analyst_candidates:
                    analyst_outputs = self._get_artifact(name, 'data')
                    if analyst_outputs is not None:
                        tprint_info(f"   ↪ Using additional features from '{name}' (preferred OOF/OOS)")
                        break
                if analyst_outputs is not None:
                    if not hasattr(analyst_outputs, 'index') or not isinstance(analyst_outputs.index, type(training_data.index)):
                        raise ValueError("Analyst ensemble outputs lack proper DatetimeIndex for alignment")
                    if not analyst_outputs.index.is_monotonic_increasing:
                        analyst_outputs = analyst_outputs.sort_index()
                    if not training_data.index.is_monotonic_increasing:
                        training_data.sort_index(inplace=True)
                    if not analyst_outputs.index.equals(training_data.index):
                        tprint_warning("Aligning 'analyst_ensemble_outputs' to training_data index (ffill/bfill)")
                        analyst_outputs = analyst_outputs.reindex(training_data.index, method='ffill').fillna(method='bfill')
                        tprint_info(
                            f"   ↪ Resampled analyst_ensemble_outputs -> shape={analyst_outputs.shape}, columns={len(analyst_outputs.columns)}"
                        )
                    additional_features_list.append(analyst_outputs)
                    tprint_info(
                        f"   ↪ Added analyst_ensemble_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )
                # No stats needed here, this is for a base model

            elif training_type == 'tactician_ensemble':
                # Base models for tactician_ensemble are the tactician_base outputs
                # Analyst_ensemble outputs are also included as features.
                analyst_candidates = [
                    'analyst_ensemble_outputs_oof',
                    'analyst_ensemble_outputs_oos',
                    'analyst_ensemble_outputs'
                ]
                tactician_candidates = [
                    'tactician_base_outputs_oof',
                    'tactician_base_outputs_oos',
                    'tactician_base_outputs'
                ]
                analyst_outputs = None
                for name in analyst_candidates:
                    analyst_outputs = self._get_artifact(name, 'data')
                    if analyst_outputs is not None:
                        tprint_info(f"   ↪ Using additional features from '{name}' (preferred OOF/OOS)")
                        break
                tactician_base_outputs = None
                for name in tactician_candidates:
                    tactician_base_outputs = self._get_artifact(name, 'data')
                    if tactician_base_outputs is not None:
                        tprint_info(f"   ↪ Using additional features from '{name}' (preferred OOF/OOS)")
                        break

                if analyst_outputs is not None:
                    if not hasattr(analyst_outputs, 'index') or not isinstance(analyst_outputs.index, type(training_data.index)):
                        raise ValueError("Analyst ensemble outputs lack proper DatetimeIndex for alignment")
                    if not analyst_outputs.index.is_monotonic_increasing:
                        analyst_outputs = analyst_outputs.sort_index()
                    if not training_data.index.is_monotonic_increasing:
                        training_data.sort_index(inplace=True)
                    if not analyst_outputs.index.equals(training_data.index):
                        tprint_warning("Aligning 'analyst_ensemble_outputs' to training_data index (ffill/bfill)")
                        analyst_outputs = analyst_outputs.reindex(training_data.index, method='ffill').fillna(method='bfill')
                        tprint_info(
                            f"   ↪ Resampled analyst_ensemble_outputs -> shape={analyst_outputs.shape}, columns={len(analyst_outputs.columns)}"
                        )
                    additional_features_list.append(analyst_outputs)
                    tprint_info(
                        f"   ↪ Added analyst_ensemble_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )

                if tactician_base_outputs is not None:
                    if not hasattr(tactician_base_outputs, 'index') or not isinstance(tactician_base_outputs.index, type(training_data.index)):
                        raise ValueError("Tactician base outputs lack proper DatetimeIndex for alignment")
                    if not tactician_base_outputs.index.is_monotonic_increasing:
                        tactician_base_outputs = tactician_base_outputs.sort_index()
                    if not training_data.index.is_monotonic_increasing:
                        training_data.sort_index(inplace=True)
                    if not tactician_base_outputs.index.equals(training_data.index):
                        tprint_warning("Aligning 'tactician_base_outputs' to training_data index (ffill)")
                        tactician_base_outputs = tactician_base_outputs.reindex(training_data.index, method='ffill')
                        tprint_info(
                            f"   ↪ Resampled tactician_base_outputs -> shape={tactician_base_outputs.shape}, columns={len(tactician_base_outputs.columns)}"
                        )
                    additional_features_list.append(tactician_base_outputs)
                    tprint_info(
                        f"   ↪ Added tactician_base_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )
                    base_outputs_for_stats = tactician_base_outputs  # For disagreement features

            # --- NEW: Calculate ensemble meta-features (disagreement features) ---
            if base_outputs_for_stats is not None and not base_outputs_for_stats.empty:
                try:
                    tprint_info("🔍 Calculating disagreement meta-features from base model outputs...")

                    # Prepare model outputs as dict for disagreement calculator
                    # Assume columns are named like: model1_prediction, model2_prediction, etc.
                    # or model1_prob_0, model1_prob_1, model2_prob_0, model2_prob_1, etc.

                    model_predictions = {}
                    model_probabilities = {}
                    model_confidences = {}

                    # Parse column names to identify model outputs
                    for col in base_outputs_for_stats.columns:
                        col_lower = col.lower()

                        # Extract model predictions (columns ending with _prediction or _pred)
                        if '_prediction' in col_lower or '_pred' in col_lower:
                            model_name = col.split('_prediction')[0].split('_pred')[0]
                            model_predictions[model_name] = base_outputs_for_stats[col].values

                        # Extract model probabilities (columns with _prob or _probability)
                        elif '_prob' in col_lower or '_probability' in col_lower:
                            # Group multi-class probabilities by model
                            parts = col.split('_')
                            for i, part in enumerate(parts):
                                if 'prob' in part.lower():
                                    model_name = '_'.join(parts[:i])
                                    if model_name not in model_probabilities:
                                        model_probabilities[model_name] = []
                                    model_probabilities[model_name].append(base_outputs_for_stats[col].values)
                                    break

                        # Extract model confidence scores
                        elif '_confidence' in col_lower or '_conf' in col_lower:
                            model_name = col.split('_confidence')[0].split('_conf')[0]
                            model_confidences[model_name] = base_outputs_for_stats[col].values

                    # Convert probability lists to arrays
                    for model_name in model_probabilities:
                        if isinstance(model_probabilities[model_name], list):
                            model_probabilities[model_name] = np.column_stack(model_probabilities[model_name])

                    tprint_info(f"   ↪ Parsed {len(model_predictions)} prediction outputs")
                    tprint_info(f"   ↪ Parsed {len(model_probabilities)} probability outputs")
                    tprint_info(f"   ↪ Parsed {len(model_confidences)} confidence outputs")

                    # Calculate disagreement features
                    if model_predictions or model_probabilities:
                        # If we only have predictions, create dummy probabilities
                        if not model_probabilities and model_predictions:
                            tprint_warning("⚠️ No probabilities found, using predictions only for disagreement features")
                            # Convert predictions to simple binary probabilities
                            for model_name, preds in model_predictions.items():
                                probs = np.column_stack([
                                    np.where(preds > 0, 0, 1),  # Prob of class 0 (negative)
                                    np.where(preds > 0, 1, 0)   # Prob of class 1 (positive)
                                ])
                                model_probabilities[model_name] = probs

                        # Calculate disagreement features using centralized calculator
                        disagreement_features_dict = calculate_ensemble_disagreement_features(
                            model_predictions=model_predictions,
                            model_probabilities=model_probabilities,
                            model_confidences=model_confidences if model_confidences else None,
                            logger=self.logger
                        )

                        # Convert dict of Series to DataFrame
                        all_meta_features = pd.DataFrame(disagreement_features_dict, index=base_outputs_for_stats.index)

                        # Filter to keep only the 7 most important disagreement features
                        # These are the most informative features for ensemble learning
                        core_features = [
                            'prediction_dispersion',    # 1. Variance of predictions across models
                            'confidence_gap',           # 2. Margin between top predictions
                            'uncertainty',              # 3. Normalized entropy (uncertainty measure)
                            'prediction_range',         # 4. Range of predictions (max - min)
                            'avg_divergence',           # 5. Average pairwise model divergence
                            'max_confidence',           # 6. Highest confidence among models
                            'disagreement_rate'         # 7. Proportion of models disagreeing on direction
                        ]

                        # Select only core features that exist
                        available_core_features = [f for f in core_features if f in all_meta_features.columns]
                        meta_features = all_meta_features[available_core_features].copy()

                        # Normalize prediction_range and avg_divergence by standard deviation
                        features_to_normalize = ['prediction_range', 'avg_divergence']
                        # DEBUG: Log disagreement features before adding
                        tprint_info("=" * 80)
                        tprint_info("🔍 [DEBUG] DISAGREEMENT FEATURES ANALYSIS")
                        tprint_info("=" * 80)
                        tprint_info(f"🔍 [DEBUG] Meta features shape: {meta_features.shape}")
                        tprint_info(f"🔍 [DEBUG] Meta features columns: {list(meta_features.columns)}")
                        tprint_info(f"🔍 [DEBUG] Meta features count: {len(meta_features.columns)}")
                        tprint_info("=" * 80)
                        for feature in features_to_normalize:
                            if feature in meta_features.columns:
                                feature_std = meta_features[feature].std()
                                if feature_std > 0:
                                    meta_features[feature] = meta_features[feature] / feature_std
                                    tprint_info(f"   ↪ Normalized '{feature}' by std={feature_std:.6f}")
                                else:
                                    tprint_warning(f"   ⚠️ Cannot normalize '{feature}' (std=0)")

                        tprint_success(f"✅ Calculated {len(meta_features.columns)} core disagreement meta-features:")
                        tprint_info(f"   Feature columns: {list(meta_features.columns)}")

                        if len(available_core_features) < len(core_features):
                            missing = set(core_features) - set(available_core_features)
                            tprint_warning(f"   ⚠️ Missing features: {missing}")

                        # Add these new features to the list
                        additional_features_list.append(meta_features)
                    else:
                        tprint_warning("⚠️ Could not parse model outputs for disagreement features, creating empty meta-features")
                        meta_features = pd.DataFrame(index=base_outputs_for_stats.index)
                        # Don't add empty DataFrame to avoid errors

                except Exception as e:
                    tprint_error(f"❌ Failed to calculate disagreement features: {e}")
                    import traceback
                    tprint_error(traceback.format_exc())
                    self.logger.error(f"Disagreement feature calculation failed: {e}")
                    # Continue without disagreement features rather than failing

            if additional_features_list:
                # DEBUG: Log before concatenation
                total_additional_cols = sum(df.shape[1] for df in additional_features_list)
                tprint_info(f"🔍 [DEBUG] Total additional features before concatenation: {total_additional_cols}")
                for i, df in enumerate(additional_features_list):
                    tprint_info(f"🔍 [DEBUG] Additional feature set {i}: shape={df.shape}, columns={list(df.columns)}")
                
                # Concatenate all features (base outputs + meta-features)
                # Use safe concatenation with temporal alignment validation
                try:
                    final_additional_features = self._safe_concat(
                        additional_features_list,
                        axis=1,
                        operation_name="concatenate_additional_features",
                        validate_alignment=True
                    )
                    
                    # DEBUG: Log after concatenation
                    tprint_info(f"🔍 [DEBUG] Final additional features shape after concatenation: {final_additional_features.shape}")
                    tprint_info(f"🔍 [DEBUG] Final additional features columns (first 20): {list(final_additional_features.columns[:20])}")
                    
                    return final_additional_features
                except ValueError as e:
                    tprint_error(f"❌ Temporal alignment error in additional features: {e}")
                    return None
            else:
                tprint_warning("🔍 [DEBUG] No additional features to add")
                return None

        except Exception as e:
            self.logger.error(f"Error retrieving additional model outputs: {e}")
            return None

    def _log_feature_snapshot(self, df: pd.DataFrame, source_name: str, prefix: str = "") -> None:
        """Log concise diagnostics about the feature dataframe."""
        try:
            n_samples, n_features = df.shape
            dtypes_summary = df.dtypes.value_counts().to_dict()
            sample_columns = df.columns[:10].tolist()
            tprint_info(f"{prefix}source={source_name}, samples={n_samples}, features={n_features}, dtypes={dtypes_summary}")
            tprint_info(f"{prefix}sample columns: {sample_columns}{'...' if n_features > len(sample_columns) else ''}")
        except Exception as exc:
            self.logger.debug(f"Failed to log feature snapshot for {source_name}: {exc}")

    def _build_meta_gated_feature_set(self, base_index: pd.Index, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Build a faithful meta-gated feature set (Feature Set B) on the 15m grid.

        This mirrors the information used by MetaGatedBacktestStep while avoiding
        explicit label/realized-return leakage. Includes:
        - Whitelisted columns from labeled_data (meta_probability, volatility_1d, etc.)
        - Canonical specialist scalar features (risk, liquidity, SMC, MR, etc.)
        """
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')

            artifact_name = f"labeled_data_{symbol}_{timeframe}"
            tprint_info(f"  Building meta-gated feature set from '{artifact_name}' for {symbol}/{exchange} [{timeframe}]")

            labeled = self._get_artifact(artifact_name, 'data')
            if labeled is None or not isinstance(labeled, pd.DataFrame) or labeled.empty:
                tprint_warning(
                    f"  Meta-gated Feature Set B: labeled data artifact '{artifact_name}' "
                    "not found or empty; cannot construct meta-gated features."
                )
                return None

            df = labeled.copy()

            # Ensure DatetimeIndex for alignment
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    idx = pd.to_datetime(df.index, errors="coerce")
                    valid_mask = ~idx.isna()
                    if not bool(valid_mask.any()):
                        tprint_warning(
                            "  Meta-gated Feature Set B: labeled_data index could not be "
                            "coerced to DatetimeIndex; skipping meta-gated features."
                        )
                        return None
                    if not bool(valid_mask.all()):
                        df = df.loc[valid_mask].copy()
                        idx = idx[valid_mask]
                    df.index = idx
                except Exception as idx_exc:
                    tprint_warning(
                        f"  Meta-gated Feature Set B: failed to normalize labeled_data index "
                        f"to DatetimeIndex ({idx_exc}); skipping."
                    )
                    return None

            df = df.sort_index()

            # Align to the training 15m grid
            df = df.reindex(base_index)

            # Whitelist meta-gated, non-leaky columns from labeled_data.
            # This list is aligned with the features actually used by
            # MetaGatedBacktestStep: meta probabilities + thresholds,
            # event/volatility context, regime labels, and specialist scalars.
            candidate_cols = [
                # Core meta-probability outputs
                "volatility_1d",
                "meta_probability",
                "meta_probability_ensemble",
                "adaptive_profit_threshold",
                "adaptive_stop_threshold",
                # Event-level context + regime labels
                "event_duration_bars",
                "hmm_regime_label_1h",
                # Specialist scalars that may be pre-merged into labeled_data
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
            ]
            # Also include any liquidity regime probability columns that
            # MetaGatedBacktestStep injects with a "liquidity_" prefix.
            liquidity_cols = [
                c for c in df.columns
                if c.startswith("liquidity_")
                and "liquidity_regime_" in c
                and "prob" in c
            ]
            candidate_cols.extend(liquidity_cols)
            
            safe_cols = [c for c in candidate_cols if c in df.columns]
            
            tprint_info(f"  Meta-gated Feature Set B: candidate columns checked: {len(candidate_cols)}")
            tprint_info(f"  Meta-gated Feature Set B: available in labeled_data: {safe_cols}")

            if not safe_cols:
                tprint_warning(
                    "  Meta-gated Feature Set B: no whitelisted meta-gated columns "
                    "available in labeled_data; skipping."
                )
                return None

            features = df[safe_cols].copy()
            tprint_info(f"  Meta-gated Feature Set B: {len(safe_cols)} columns from labeled_data")

            # ------------------------------------------------------------------
            # Add canonical specialist scalar features (same utility as
            # MetaGatedBacktestStep), aligned to the 15m grid. These are
            # regime/risk/liquidity/SMC/MR scalars and do not expose
            # realized-return labels.
            # ------------------------------------------------------------------
            try:
                specialist_config = dict(config)
                specialist_config.setdefault("use_canonical_specialist_scalars", True)
                specialist_config.setdefault("enable_risk_hmm_specialist", False)

                from src.utils.ml_common.get_specialist_models_outputs import (
                    get_specialist_models_outputs,
                )

                specialist_df = get_specialist_models_outputs(
                    artifact_router=self.artifact_router,
                    training_index=base_index,
                    config=specialist_config,
                    logger=self.logger,
                    strict=False,
                )

                if specialist_df is not None and not specialist_df.empty:
                    # Drop all-NaN columns
                    non_null_counts = specialist_df.notna().sum()
                    active_cols = non_null_counts[non_null_counts > 0].index.tolist()
                    if active_cols:
                        specialist_df = specialist_df[active_cols]
                        # Align to base_index and forward-fill
                        specialist_df = specialist_df.reindex(base_index).ffill()
                        # Avoid duplicate columns
                        new_cols = [c for c in specialist_df.columns if c not in features.columns]
                        if new_cols:
                            features = pd.concat([features, specialist_df[new_cols]], axis=1)
                            tprint_info(
                                f"  Meta-gated Feature Set B: added {len(new_cols)} specialist scalar features"
                            )
                    else:
                        tprint_warning(
                            "  Meta-gated Feature Set B: all specialist columns are NaN after alignment; "
                            "skipping specialist features."
                        )
                else:
                    tprint_info(
                        "  Meta-gated Feature Set B: no specialist outputs available; using labeled_data-only features."
                    )
            except Exception as spec_exc:
                self.logger.warning(f"Meta-gated Feature Set B: specialist feature integration failed: {spec_exc}")

            tprint_success(
                f"✅ Meta-gated Feature Set B constructed: {features.shape[0]} samples × "
                f"{features.shape[1]} features"
            )
            return features

        except Exception as e:
            self.logger.warning(f"Meta-gated Feature Set B construction failed: {e}")
            return None

    async def _execute_training_by_type(
        self, 
        training_type: str, 
        training_data, 
        analyst_targets, 
        tactician_targets, 
        yaml_config: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute training based on the specified type."""
        # Handle dict vs DataFrame input (Multi Feature Set support)
        training_data_primary = training_data
        training_data_container = None

        if isinstance(training_data, dict):
            training_data_container = training_data
            # Determine primary dataset for logging/compat
            feature_set_mode = str(
                (yaml_config.get('analyst_config', {}) or {}).get('feature_set', 'A')
            ).upper()
            primary_key = 'B' if feature_set_mode == 'B' and 'B' in training_data else 'A'
            training_data_primary = training_data.get(primary_key, training_data.get('A'))
            if training_data_primary is None:
                # Fallback to first available
                training_data_primary = next(iter(training_data.values()))

        # DEBUG: Log detailed feature information before training
        tprint_info("=" * 80)
        tprint_info("🔍 [DEBUG] FEATURE ANALYSIS BEFORE TRAINING")
        tprint_info("=" * 80)
        tprint_info(f"🔍 [DEBUG] Training type: {training_type}")
        tprint_info(f"🔍 [DEBUG] Training data shape: {training_data_primary.shape}")

        # Count features using primary dataset
        regime_features = [col for col in training_data_primary.columns if 'regime' in col.lower()]
        tprint_info(f"🔍 [DEBUG] Regime features count: {len(regime_features)}")

        # Count disagreement features
        disagreement_features = [col for col in training_data_primary.columns if any(term in col.lower() for term in ['dispersion', 'disagreement', 'uncertainty', 'confidence_gap', 'prediction_range'])]
        tprint_info(f"🔍 [DEBUG] Disagreement features count: {len(disagreement_features)}")

        # Clean leakage from all datasets if dict, or just primary
        leak_cols_names = ['label', 'target', 'future_', 'lead_']

        if training_data_container:
            for key, df in training_data_container.items():
                leak_cols = [c for c in df.columns if any(term in c.lower() for term in leak_cols_names)]
                if leak_cols:
                    tprint_warning(f"Removing {len(leak_cols)} target-like columns from Feature Set {key}")
                    training_data_container[key] = df.drop(columns=leak_cols)
            # Update primary
            if isinstance(training_data, dict):
                 feature_set_mode = str((yaml_config.get('analyst_config', {}) or {}).get('feature_set', 'A')).upper()
                 primary_key = 'B' if feature_set_mode == 'B' and 'B' in training_data else 'A'
                 training_data_primary = training_data_container.get(primary_key, training_data_container.get('A'))

        else:
            leak_cols = [c for c in training_data_primary.columns if any(term in c.lower() for term in leak_cols_names)]
            if leak_cols:
                tprint_warning(f"Removing {len(leak_cols)} target-like columns from training data")
                training_data_primary = training_data_primary.drop(columns=leak_cols)
            training_data = training_data_primary # Update reference

        self._training_features = list(training_data_primary.columns)
        self._training_feature_count = len(training_data_primary.columns)

        tprint_info("=" * 80)
        try:
            if training_type == 'analyst_base':
                # Use IncrementalAnalystTrainer for rolling OOF predictions
                tprint_info(f"🔧 Training analyst base models INCREMENTALLY with mixed feature sets...")
                if training_data_container:
                    tprint_info(f"   Available Feature Sets: {list(training_data_container.keys())}")
                
                # Import incremental trainer
                from src.utils.ml_common.incremental_analyst_trainers import IncrementalAnalystTrainer
                
                # Get execution mode from config
                execution_mode = config.get('execution_mode', 'blank')
                symbol = config.get('symbol', 'ETHUSDT')
                exchange = config.get('exchange', 'binance')
                timeframe = config.get('timeframe', '15m')
                
                # Determine data start/end from index (use primary)
                if hasattr(training_data_primary.index, 'min') and hasattr(training_data_primary.index, 'max'):
                    data_start = training_data_primary.index.min()
                    data_end = training_data_primary.index.max()
                    
                    if hasattr(data_start, 'to_pydatetime'):
                        data_start = data_start.to_pydatetime()
                    if hasattr(data_end, 'to_pydatetime'):
                        data_end = data_end.to_pydatetime()
                else:
                    from datetime import datetime, timedelta
                    data_end = datetime.now()
                    samples_per_day = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1}.get(timeframe, 96)
                    total_days = len(training_data_primary) // samples_per_day
                    data_start = data_end - timedelta(days=total_days)
                
                tprint_info(f"   Data range: {data_start} → {data_end}")
                
                # Create incremental trainer
                model_id = f"{symbol}_{exchange}_{timeframe}"
                base_models_config = yaml_config.get('analyst_config', {}).get('base_models', {})

                # Determine global default feature set from config
                # Default to 'A' if not specified, but respect config if it says 'B'
                global_feature_set = str(
                    (yaml_config.get('analyst_config', {}) or {}).get('feature_set', 'A')
                ).upper()

                incremental_trainer = IncrementalAnalystTrainer(
                    model_id=model_id,
                    execution_mode=execution_mode,
                    task_type='regression',
                    enable_incremental_hpo=True,
                    model_configs=base_models_config,
                    default_feature_set=global_feature_set
                )
                
                # Train all models incrementally (pass container or dataframe)
                # If container exists, pass it. If not, pass training_data (dataframe)
                data_arg = training_data_container if training_data_container else training_data

                incremental_results = incremental_trainer.train_all_models(
                    X=data_arg,
                    y=analyst_targets,
                    data_start=data_start,
                    data_end=data_end,
                    sample_weight=None,
                    verbose=True,
                    specialist_feature_names=self._specialist_feature_names
                )
                
                # Combine OOF predictions from all models
                combined_oof = incremental_trainer.get_combined_oof_predictions(incremental_results)
                
                # Build result dict in expected format
                models_dict = {}
                predictions_dict = {}
                
                for model_name, result in incremental_results.items():
                    models_dict[model_name] = result.final_model
                    if result.oof_predictions is not None and not result.oof_predictions.empty:
                        predictions_dict[model_name] = result.oof_predictions.get('prediction', result.oof_predictions.iloc[:, 0])
                
                tprint_success(f"✅ Incremental training complete for {len(models_dict)} models")
                tprint_info(f"   Total OOF predictions: {len(combined_oof)}")
                
                return {
                    'success': True,
                    'models': models_dict,
                    'predictions': pd.DataFrame(predictions_dict) if predictions_dict else None,
                    'oof_predictions': combined_oof,
                    'training_metadata': {
                        model_name: result.window_metadata 
                        for model_name, result in incremental_results.items()
                    },
                    'hpo_history': {
                        model_name: result.hpo_history
                        for model_name, result in incremental_results.items()
                    },
                    'best_params': {
                        model_name: result.best_params
                        for model_name, result in incremental_results.items()
                    }
                }
            elif training_type == 'analyst_ensemble':
                tprint_info(f"🔍 [DEBUG] Calling train_ensemble_models with config keys: {list(config.keys())}")
                if 'ensemble_features' in config:
                    tprint_info(f"🔍 [DEBUG] ensemble_features shape: {config['ensemble_features'].shape}")
                else:
                    tprint_warning("⚠️ [DEBUG] ensemble_features NOT in config!")
                    
                return await self.unified_pipeline.train_ensemble_models(
                    data=training_data,
                    analyst_targets=analyst_targets,
                    tactician_targets=None,
                    config=yaml_config
                )
            elif training_type == 'tactician_base':
                return await self.unified_pipeline.train_tactician_models(
                    data=training_data,
                    targets=tactician_targets,
                    config=yaml_config
                )
            elif training_type == 'tactician_ensemble':
                return await self.unified_pipeline.train_ensemble_models(
                    data=training_data,
                    analyst_targets=analyst_targets,
                    tactician_targets=tactician_targets,
                    config=yaml_config
                )
            else:
                raise ValueError(f"Unknown training type: {training_type}")
                
        except Exception as e:
            self.logger.error(f"Training execution failed for {training_type}: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }

    async def _save_training_artifacts(self, result: Dict[str, Any], training_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Save training artifacts."""
        try:
            artifacts = {}

            # Save model artifacts
            if 'models' in result:
                for model_name, model in result['models'].items():
                    # Base metadata common to all training types
                    model_metadata = {
                        'training_type': training_type,
                        'symbol': config.get('symbol'),
                        'timeframe': config.get('timeframe'),
                        'direction': config.get('direction'),
                        'created_at': datetime.now().isoformat(),
                    }

                    # For analyst_base models, persist the exact training
                    # feature list so diagnostics (e.g., Architect
                    # permutation importance) can reconstruct the true
                    # feature set used during training.
                    if training_type == 'analyst_base' and hasattr(self, '_training_features'):
                        try:
                            model_metadata['training_features'] = list(self._training_features)
                            model_metadata['training_feature_count'] = int(len(self._training_features))
                        except Exception:
                            # Non-fatal: continue without feature metadata
                            pass

                    artifact_path = self._save_artifact(
                        data=model,
                        artifact_name=f"{training_type}_{model_name}",
                        artifact_type='model',
                        metadata=model_metadata,
                    )
                    artifacts[f"{training_type}_{model_name}"] = artifact_path
                    tprint_info(
                        f"💾 Saved model artifact '{training_type}_{model_name}' to {artifact_path}"
                    )
            
            # Save predictions for ensemble training (analyst_base only)
            # CRITICAL: Accumulate predictions from ALL base models into a single DataFrame
            if training_type == 'analyst_base':
                try:
                    all_predictions = {}
                    all_confidence = {}
                    
                    # Check if we have models dict with multiple models
                    if 'models' in result and result['models']:
                        tprint_info(f"📊 Accumulating predictions from {len(result['models'])} base models...")
                        
                        # Get predictions from each model
                        for model_name, model in result['models'].items():
                            # Try to get model-specific predictions
                            if f'{model_name}_predictions' in result:
                                all_predictions[model_name] = result[f'{model_name}_predictions']
                                tprint_info(f"  ✓ Got predictions from {model_name}")
                            elif 'predictions' in result and isinstance(result['predictions'], pd.DataFrame):
                                # If predictions is a DataFrame, check if it has model-specific columns
                                if model_name in result['predictions'].columns:
                                    all_predictions[model_name] = result['predictions'][model_name]
                                    tprint_info(f"  ✓ Got predictions from {model_name} (from DataFrame)")
                            
                            # Get confidence scores
                            if f'{model_name}_confidence' in result:
                                all_confidence[model_name] = result[f'{model_name}_confidence']
                    
                    # Fallback: use result['predictions'] if it's already a multi-column DataFrame
                    if not all_predictions and 'predictions' in result and result['predictions'] is not None:
                        if isinstance(result['predictions'], pd.DataFrame) and result['predictions'].shape[1] > 1:
                            # Already a multi-column DataFrame
                            all_predictions = {col: result['predictions'][col] for col in result['predictions'].columns}
                            tprint_info(f"📊 Using existing multi-column predictions: {result['predictions'].shape}")
                        else:
                            # Single prediction - this is the problem we're fixing!
                            tprint_warning(f"⚠️ Only single prediction column found, expected multiple base models!")
                            all_predictions = {'model_0': result['predictions']}
                    
                    # Save combined predictions as DataFrame
                    if all_predictions:
                        tprint_info("=" * 80)
                        tprint_info("💾 SAVING ANALYST BASE PREDICTIONS")
                        tprint_info("=" * 80)
                        tprint_info(f"📊 Accumulated predictions from {len(all_predictions)} models:")
                        for model_name in all_predictions.keys():
                            tprint_info(f"   ✓ {model_name}")
                        
                        predictions_df = pd.DataFrame(all_predictions)
                        tprint_success(f"✅ Combined predictions DataFrame: {predictions_df.shape}")
                        tprint_info(f"   Columns: {list(predictions_df.columns)}")
                        tprint_info("=" * 80)
                        
                        # Calculate and add disagreement features
                        disagreement_features = None
                        if len(all_predictions) >= 2:
                            try:
                                tprint_info("🔍 Calculating disagreement features for base predictions...")
                                from src.feature_engineering_roadmap.disagreement_meta_features import DisagreementMetaFeatures
                                
                                disagreement_calc = DisagreementMetaFeatures(logger=self.logger)
                                
                                # Prepare model predictions dict
                                model_predictions = {name: preds.values if isinstance(preds, pd.Series) else preds 
                                                    for name, preds in all_predictions.items()}
                                
                                # Calculate disagreement features
                                disagreement_dict = disagreement_calc.calculate_all_disagreement_features(
                                    model_predictions=model_predictions,
                                    model_probabilities=None,  # We don't have probabilities for base predictions
                                    model_confidences=all_confidence if all_confidence else None
                                )
                                
                                # Convert to DataFrame
                                disagreement_features = pd.DataFrame(disagreement_dict, index=predictions_df.index)
                                
                                # Keep only core features
                                core_features = [
                                    'prediction_dispersion',
                                    'prediction_range',
                                    'prediction_std',
                                    'prediction_entropy',
                                    'pairwise_disagreement_mean',
                                    'confidence_weighted_disagreement'
                                ]
                                available_features = [f for f in core_features if f in disagreement_features.columns]
                                if available_features:
                                    disagreement_features = disagreement_features[available_features]
                                    tprint_success(f"✅ Calculated {len(available_features)} disagreement features")
                                    
                                    # Combine predictions and disagreement features
                                    combined_df = pd.concat([predictions_df, disagreement_features], axis=1)
                                    predictions_df = combined_df
                                    tprint_info(f"   Combined shape: {predictions_df.shape}")
                                    
                            except Exception as e:
                                tprint_warning(f"⚠️ Failed to calculate disagreement features: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        predictions_path = self._save_artifact(
                            data=predictions_df,
                            artifact_name='analyst_base_predictions',
                            artifact_type='data',
                            data_category='predictions'
                        )
                        artifacts['analyst_base_predictions'] = predictions_path
                        tprint_success(f"✅ Saved analyst_base_predictions: {predictions_df.shape} ({len(all_predictions)} models)")
                        tprint_info(f"   Models: {list(all_predictions.keys())}")
                        tprint_info(f"   Path: {predictions_path}")
                        if disagreement_features is not None:
                            tprint_info(f"   Disagreement features: {list(disagreement_features.columns)}")

                        # Save OOF predictions if provided by training pipeline
                        try:
                            oof = result.get('oof_predictions')
                            if oof is not None:
                                if isinstance(oof, pd.DataFrame):
                                    oof_df = oof
                                elif isinstance(oof, dict):
                                    # If dict of series, convert to DataFrame
                                    oof_df = pd.DataFrame(oof)
                                else:
                                    # If Series or ndarray-like
                                    oof_df = pd.DataFrame({'oof_pred': oof})

                                # ------------------------------------------------------------------
                                # PER-MODEL CALIBRATION: Calibrate OOF predictions before saving
                                # This ensures the ensemble receives calibrated base predictions
                                # ------------------------------------------------------------------
                                enable_per_model_calibration = bool(
                                    config.get('enable_per_model_calibration', True)
                                )

                                if enable_per_model_calibration:
                                    tprint_info("=" * 80)
                                    tprint_info("🎯 APPLYING PER-MODEL CALIBRATION TO OOF PREDICTIONS")
                                    tprint_info("=" * 80)

                                    # Get targets for calibration
                                    y_true_series = getattr(self, '_full_analyst_targets', None)

                                    if y_true_series is not None and not oof_df.empty:
                                        # Align targets to OOF predictions index
                                        y_true_aligned = y_true_series.reindex(oof_df.index)
                                        valid_mask = y_true_aligned.notna()

                                        if valid_mask.sum() >= 50:
                                            tprint_info(f"📊 Calibrating {oof_df.shape[1]} model predictions with {valid_mask.sum()} valid samples")

                                            # Calibrate each model's predictions independently
                                            calibrated_oof = oof_df.copy()
                                            calibration_metrics = {}

                                            for col_name in oof_df.columns:
                                                try:
                                                    tprint_info(f"   🔧 Calibrating {col_name}...")

                                                    # Get predictions for this model
                                                    y_pred = oof_df[col_name].values[valid_mask]
                                                    y_true = y_true_aligned.values[valid_mask]

                                                    # Convert predictions to probabilities if needed (for isotonic calibration)
                                                    # Isotonic regression expects values in [0, 1] range
                                                    # For regression targets, we'll calibrate the relationship between
                                                    # predicted and actual values
                                                    from sklearn.isotonic import IsotonicRegression

                                                    # Fit isotonic calibrator on the OOF predictions
                                                    calibrator = IsotonicRegression(
                                                        out_of_bounds='clip',
                                                        increasing='auto'  # Auto-detect monotonicity
                                                    )

                                                    # Fit calibrator: maps model predictions to actual targets
                                                    calibrator.fit(y_pred, y_true)

                                                    # Apply calibration to all predictions
                                                    calibrated_pred = calibrator.predict(oof_df[col_name].values)
                                                    calibrated_oof[col_name] = calibrated_pred

                                                    # Calculate calibration improvement metrics
                                                    from sklearn.metrics import mean_squared_error, mean_absolute_error

                                                    mse_before = mean_squared_error(y_true, y_pred)
                                                    mse_after = mean_squared_error(
                                                        y_true,
                                                        calibrator.predict(y_pred)
                                                    )

                                                    mae_before = mean_absolute_error(y_true, y_pred)
                                                    mae_after = mean_absolute_error(
                                                        y_true,
                                                        calibrator.predict(y_pred)
                                                    )

                                                    improvement = ((mse_before - mse_after) / mse_before * 100) if mse_before > 0 else 0

                                                    calibration_metrics[col_name] = {
                                                        'mse_before': float(mse_before),
                                                        'mse_after': float(mse_after),
                                                        'mse_improvement_pct': float(improvement),
                                                        'mae_before': float(mae_before),
                                                        'mae_after': float(mae_after),
                                                        'n_samples': int(len(y_pred))
                                                    }

                                                    tprint_success(
                                                        f"      ✅ {col_name}: MSE improved by {improvement:.2f}% "
                                                        f"(before={mse_before:.6f}, after={mse_after:.6f})"
                                                    )

                                                except Exception as e:
                                                    tprint_warning(f"      ⚠️ Failed to calibrate {col_name}: {e}")
                                                    # Keep original predictions if calibration fails
                                                    calibrated_oof[col_name] = oof_df[col_name]

                                            # Use calibrated predictions
                                            oof_df = calibrated_oof

                                            # Save calibration metrics
                                            if calibration_metrics:
                                                calibration_summary = {
                                                    'timestamp': datetime.now().isoformat(),
                                                    'n_models': len(calibration_metrics),
                                                    'per_model_metrics': calibration_metrics,
                                                    'avg_mse_improvement_pct': float(
                                                        np.mean([m['mse_improvement_pct'] for m in calibration_metrics.values()])
                                                    )
                                                }

                                                # Store in artifacts
                                                artifacts['calibration_metrics'] = calibration_summary

                                                tprint_info("=" * 80)
                                                tprint_success(f"✅ CALIBRATION COMPLETE: Avg MSE improvement = {calibration_summary['avg_mse_improvement_pct']:.2f}%")
                                                tprint_info("=" * 80)
                                        else:
                                            tprint_warning(
                                                f"⚠️ Insufficient valid samples for calibration ({valid_mask.sum()} < 50), "
                                                "skipping per-model calibration"
                                            )
                                    else:
                                        tprint_warning("⚠️ No targets available for calibration, skipping per-model calibration")

                                oof_path = self._save_artifact(
                                    data=oof_df,
                                    artifact_name='analyst_base_predictions_oof',
                                    artifact_type='data',
                                    data_category='predictions'
                                )
                                artifacts['analyst_base_predictions_oof'] = oof_path

                                if enable_per_model_calibration:
                                    tprint_success(f"✅ Saved calibrated analyst_base_predictions_oof: {oof_df.shape}")
                                else:
                                    tprint_success(f"✅ Saved analyst_base_predictions_oof: {oof_df.shape}")
                                tprint_info(f"   Path: {oof_path}")

                                try:
                                    y_true_series = getattr(self, '_full_analyst_targets', None)
                                    if y_true_series is not None and not oof_df.empty:
                                        y_true_aligned = y_true_series.reindex(oof_df.index)
                                        valid_mask = y_true_aligned.notna()
                                        if valid_mask.sum() >= 50:
                                            metrics_rows = []
                                            for col_name in oof_df.columns:
                                                y_pred = oof_df[col_name].values[valid_mask]
                                                y_true = y_true_aligned.values[valid_mask]
                                                if len(y_pred) == 0:
                                                    continue
                                                mse = mean_squared_error(y_true, y_pred)
                                                rmse = float(np.sqrt(mse))
                                                mae = mean_absolute_error(y_true, y_pred)
                                                try:
                                                    r2 = r2_score(y_true, y_pred)
                                                except Exception:
                                                    r2 = float('nan')
                                                try:
                                                    if np.std(y_true) > 0 and np.std(y_pred) > 0:
                                                        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
                                                    else:
                                                        corr = float('nan')
                                                except Exception:
                                                    corr = float('nan')
                                                metrics_rows.append({
                                                    'model': col_name,
                                                    'n_samples': int(len(y_true)),
                                                    'mse': float(mse),
                                                    'rmse': rmse,
                                                    'mae': float(mae),
                                                    'r2': float(r2),
                                                    'corr': corr,
                                                })
                                            if metrics_rows:
                                                metrics_df = pd.DataFrame(metrics_rows).set_index('model')
                                                oof_metrics_path = self._save_artifact(
                                                    data=metrics_df,
                                                    artifact_name='analyst_base_oof_metrics',
                                                    artifact_type='data',
                                                    data_category='predictions'
                                                )
                                                artifacts['analyst_base_oof_metrics'] = oof_metrics_path
                                                tprint_success(f"✅ Saved analyst_base_oof_metrics: {metrics_df.shape}")
                                                tprint_info(f"   Path: {oof_metrics_path}")

                                                # Additionally, export a user-facing CSV comparing all base models
                                                # for this analyst_base run so that a single training invocation
                                                # produces a ready-to-inspect model comparison table.
                                                try:
                                                    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                                                    symbol = str(config.get('symbol', 'UNKNOWN'))
                                                    exchange = str(config.get('exchange', 'binance'))
                                                    timeframe = str(config.get('timeframe', '15m'))
                                                    direction = str(config.get('direction', 'long'))

                                                    comparison_df = metrics_df.reset_index().rename(columns={'model': 'model_name'})
                                                    comparison_df.insert(0, 'symbol', symbol)
                                                    comparison_df.insert(1, 'exchange', exchange)
                                                    comparison_df.insert(2, 'timeframe', timeframe)
                                                    comparison_df.insert(3, 'direction', direction)

                                                    output_dir = Path('outcomes')
                                                    try:
                                                        output_dir.mkdir(parents=True, exist_ok=True)
                                                    except Exception:
                                                        pass

                                                    csv_name = (
                                                        f"analyst_base_model_comparison_"
                                                        f"{symbol}_{exchange}_{timeframe}_{direction}_{ts_str}.csv"
                                                    )
                                                    csv_path = output_dir / csv_name
                                                    comparison_df.to_csv(csv_path, index=False)
                                                    tprint_success(
                                                        f"✅ Exported analyst base model comparison CSV to {csv_path}"
                                                    )
                                                except Exception as comp_exc:
                                                    tprint_warning(
                                                        f"⚠️ Failed to export analyst base model comparison CSV: {comp_exc}"
                                                    )
                                        else:
                                            tprint_warning(
                                                "⚠️ Not enough samples with valid targets for OOF metrics; "
                                                "skipping OOF metrics computation"
                                            )
                                except Exception as metrics_exc:
                                    tprint_warning(f"⚠️ Failed to compute/save OOF metrics: {metrics_exc}")
                        except Exception as e:
                            tprint_warning(f"⚠️ Failed to save OOF predictions: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Save combined confidence scores (with optional calibration)
                    if all_confidence:
                        confidence_df = pd.DataFrame(all_confidence)

                        # ------------------------------------------------------------------
                        # Optional confidence calibration before persistence
                        # ------------------------------------------------------------------
                        try:
                            enable_calibration = bool(
                                config.get('enable_confidence_calibration', False)
                            )
                            if enable_calibration:
                                tprint_info("🎯 Applying confidence calibration before saving analyst_base_confidence ...")

                                # Retrieve full analyst targets captured earlier for walk-forward
                                y_true_series = getattr(self, '_full_analyst_targets', None)

                                if y_true_series is not None and not confidence_df.empty:
                                    # Align targets to confidence index
                                    y_true_aligned = y_true_series.reindex(confidence_df.index)
                                    # Drop samples without targets
                                    valid_mask = y_true_aligned.notna()

                                    if valid_mask.sum() >= 50:
                                        # Convert targets to a binary event (positive vs non-positive)
                                        y_true_values = y_true_aligned[valid_mask].values
                                        try:
                                            # If labels already look binary, keep as-is
                                            unique_vals = pd.unique(y_true_values)
                                            if set(unique_vals).issubset({0, 1}):
                                                y_true_binary = y_true_values.astype(int)
                                            else:
                                                # Generic mapping: treat "positive" outcome as 1
                                                y_true_binary = (y_true_values > 0).astype(int)
                                        except Exception:
                                            # Fallback: best-effort binary mapping
                                            y_true_binary = (y_true_values > 0).astype(int)

                                        method = config.get(
                                            'confidence_calibration_method',
                                            'isotonic_regression'
                                        )
                                        calib_config = config.get(
                                            'confidence_calibration_config',
                                            {}
                                        )

                                        calibrated_cols: Dict[str, Any] = {}
                                        for col_name in confidence_df.columns:
                                            try:
                                                col_values = confidence_df[col_name].astype(float).values
                                                # Restrict to samples with valid targets
                                                col_valid = col_values[valid_mask.values]

                                                if len(col_valid) != len(y_true_binary):
                                                    tprint_warning(
                                                        f"⚠️ Skipping calibration for column {col_name}: length mismatch"
                                                    )
                                                    calibrated_cols[col_name] = col_values
                                                    continue

                                                # Clip to [0, 1] to behave like probabilities
                                                col_valid = np.clip(col_valid, 0.0, 1.0)

                                                calib_result = await calibrate_model_confidence(
                                                    y_true=y_true_binary,
                                                    y_pred_proba=col_valid,
                                                    method=method,
                                                    config=calib_config,
                                                )

                                                if (
                                                    isinstance(calib_result, dict)
                                                    and 'calibrated_probabilities' in calib_result
                                                ):
                                                    calibrated_valid = np.asarray(
                                                        calib_result['calibrated_probabilities'],
                                                        dtype=float,
                                                    )

                                                    # Reconstruct full column, keeping original index
                                                    full_col = col_values.copy()
                                                    full_col[valid_mask.values] = calibrated_valid

                                                    # Guard against any NaNs introduced by calibration
                                                    nan_mask = ~np.isfinite(full_col)
                                                    if nan_mask.any():
                                                        full_col[nan_mask] = col_values[nan_mask]

                                                    calibrated_cols[col_name] = full_col
                                                else:
                                                    tprint_warning(
                                                        f"⚠️ Calibration result for {col_name} missing 'calibrated_probabilities'; using raw confidence"
                                                    )
                                                    calibrated_cols[col_name] = col_values
                                            except Exception as calib_exc:
                                                tprint_warning(
                                                    f"⚠️ Failed to calibrate confidence column {col_name}: {calib_exc}"
                                                )
                                                calibrated_cols[col_name] = confidence_df[col_name].values

                                        if calibrated_cols:
                                            confidence_df = pd.DataFrame(
                                                calibrated_cols,
                                                index=confidence_df.index,
                                            )
                                            tprint_success(
                                                f"✅ Confidence calibration applied using method='{method}' "
                                                f"for {len(calibrated_cols)} models"
                                            )
                                    else:
                                        tprint_warning(
                                            "⚠️ Not enough samples with valid targets for confidence calibration; "
                                            "saving raw confidence scores"
                                        )
                                else:
                                    tprint_warning(
                                        "⚠️ Analyst targets not available or confidence DataFrame empty; "
                                        "skipping confidence calibration"
                                    )
                        except Exception as e:
                            tprint_warning(
                                f"⚠️ Confidence calibration failed, falling back to raw scores: {e}"
                            )

                        confidence_path = self._save_artifact(
                            data=confidence_df,
                            artifact_name='analyst_base_confidence',
                            artifact_type='data',
                            data_category='predictions'
                        )
                        artifacts['analyst_base_confidence'] = confidence_path
                        tprint_success(
                            f"✅ Saved analyst_base_confidence: {confidence_df.shape} ({len(all_confidence)} models)"
                        )
                        tprint_info(f"   Path: {confidence_path}")
                        # Log summary statistics so calibration effects can be inspected
                        try:
                            flat = confidence_df.to_numpy().ravel()
                            if flat.size > 0:
                                mean_val = float(np.nanmean(flat))
                                std_val = float(np.nanstd(flat))
                                min_val = float(np.nanmin(flat))
                                max_val = float(np.nanmax(flat))
                                tprint_info(
                                    "📊 analyst_base_confidence summary: "
                                    f"mean={mean_val:.4f}, std={std_val:.4f}, "
                                    f"min={min_val:.4f}, max={max_val:.4f}"
                                )
                        except Exception as stats_exc:
                            self.logger.debug(f"Failed to log analyst_base_confidence summary: {stats_exc}")
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save predictions/confidence: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save predictions for tactician training (analyst_ensemble only)
            if training_type == 'analyst_ensemble':
                try:
                    # Prefer strictly OOF meta-learner predictions if provided by ensemble trainer
                    raw_oof = result.get('oof_predictions')
                    predictions_df = None

                    if raw_oof is not None:
                        arr = np.asarray(raw_oof)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)

                        # Align to available training data index when possible
                        idx = None
                        training_data = locals().get('training_data_filtered') or locals().get('training_data')
                        if training_data is not None and hasattr(training_data, 'index') and len(training_data.index) == arr.shape[0]:
                            idx = training_data.index
                        else:
                            idx = pd.RangeIndex(arr.shape[0])

                        cols = [f"analyst_ensemble_meta_oof_{i}" for i in range(arr.shape[1])]
                        predictions_df = pd.DataFrame(arr, index=idx, columns=cols)
                    elif 'predictions' in result and result['predictions'] is not None:
                        # Fallback: use whatever predictions were returned (may be IS or OOS depending on trainer)
                        predictions_df = result['predictions']
                        if not isinstance(predictions_df, pd.DataFrame):
                            predictions_df = pd.DataFrame(predictions_df)

                    if predictions_df is not None:
                        # Primary artifact: strictly OOF meta-learner outputs when available
                        oof_artifact_name = 'analyst_ensemble_outputs_oof'
                        predictions_path = self._save_artifact(
                            data=predictions_df,
                            artifact_name=oof_artifact_name,
                            artifact_type='data',
                            data_category='predictions'
                        )
                        artifacts[oof_artifact_name] = predictions_path
                        tprint_success(f"✅ Saved {oof_artifact_name}: {predictions_df.shape}")
                        tprint_info(f"   Path: {predictions_path}")

                        # Backwards-compatible alias for downstream consumers
                        alias_path = self._save_artifact(
                            data=predictions_df,
                            artifact_name='analyst_ensemble_outputs',
                            artifact_type='data',
                            data_category='predictions'
                        )
                        artifacts['analyst_ensemble_outputs'] = alias_path
                        tprint_info(f"   Alias path: {alias_path}")
                        tprint_info("ℹ️ Also saved legacy analyst_ensemble_outputs alias for compatibility")

                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save analyst_ensemble_outputs_oof: {e}")

                # Save the calibrated ensemble model
                try:
                    if 'model' in result and result['model'] is not None:
                        model_path = self._save_artifact(
                            data=result['model'],
                            artifact_name='analyst_ensemble_model_calibrated',
                            artifact_type='model',
                            data_category='models'
                        )
                        artifacts['analyst_ensemble_model_calibrated'] = model_path
                        result['model_path'] = model_path  # Add to result for downstream use
                        tprint_success(f"✅ Saved analyst_ensemble_model_calibrated: {model_path}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save analyst_ensemble_model_calibrated: {e}")

            # Save performance metrics (JSON + Markdown report)
            if 'metrics' in result:
                # Save as JSON
                metrics_path = self._save_artifact(
                    data=result['metrics'],
                    artifact_name=f"{training_type}_metrics",
                    artifact_type='metadata',
                    metadata={
                        'training_type': training_type,
                        'symbol': config.get('symbol'),
                        'timeframe': config.get('timeframe'),
                        'direction': config.get('direction'),
                        'created_at': datetime.now().isoformat()
                    }
                )
                artifacts[f"{training_type}_metrics"] = metrics_path
                tprint_success(f"✅ Saved {training_type}_metrics JSON: {metrics_path}")

                # Save as Markdown report
                try:
                    md_report_path = self._generate_metrics_markdown_report(
                        metrics=result['metrics'],
                        training_type=training_type,
                        config=config,
                        hpo_results=result.get('hpo_results'),
                        execution_time=result.get('execution_time', 0.0)
                    )
                    if md_report_path:
                        artifacts[f"{training_type}_metrics_report"] = md_report_path
                        tprint_success(f"✅ Saved metrics markdown report: {md_report_path}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save markdown report: {e}")
                    self.logger.warning(f"Markdown report generation failed: {e}")

            # Save configuration
            config_path = self._save_artifact(
                data=config,
                artifact_name=f"{training_type}_config",
                artifact_type='metadata',
                metadata={
                    'training_type': training_type,
                    'created_at': datetime.now().isoformat()
                }
            )
            artifacts[f"{training_type}_config"] = config_path
            tprint_success(f"✅ Saved {training_type}_config: {config_path}")

            return artifacts

        except Exception as e:
            self.logger.error(f"Failed to save training artifacts: {e}")
            return {}

    def _generate_metrics_markdown_report(
        self,
        metrics: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any],
        hpo_results: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0
    ) -> Optional[str]:
        """
        Generate a comprehensive markdown report for training metrics.

        Args:
            metrics: Training metrics dictionary
            training_type: Type of training (tactician_ensemble, etc.)
            config: Training configuration
            hpo_results: HPO optimization results (optional)
            execution_time: Total execution time in seconds

        Returns:
            Path to saved markdown report, or None if failed
        """
        try:
            import os
            from datetime import datetime

            # Generate report content
            report_lines = []

            # Header
            report_lines.append(f"# {training_type.replace('_', ' ').title()} Training Report")
            report_lines.append("")
            report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Symbol:** {config.get('symbol', 'UNKNOWN')}")
            report_lines.append(f"**Exchange:** {config.get('exchange', 'binance')}")
            report_lines.append(f"**Timeframe:** {config.get('timeframe', '15m')}")
            report_lines.append(f"**Direction:** {config.get('direction', 'long')}")
            report_lines.append(f"**Execution Time:** {execution_time:.2f}s")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

            # HPO Results Section
            if hpo_results:
                report_lines.append("## Hyperparameter Optimization (HPO)")
                report_lines.append("")
                report_lines.append("### Best HPO Scores")
                report_lines.append("")

                if 'best_scores' in hpo_results:
                    report_lines.append("| Model | Score | Parameters |")
                    report_lines.append("|-------|-------|------------|")
                    for model_name, score_data in hpo_results['best_scores'].items():
                        score = score_data.get('score', 'N/A') if isinstance(score_data, dict) else score_data
                        params = score_data.get('params', {}) if isinstance(score_data, dict) else {}
                        params_str = ', '.join([f"{k}={v}" for k, v in list(params.items())[:3]])
                        if len(params) > 3:
                            params_str += ', ...'
                        report_lines.append(f"| {model_name} | {score:.6f} | {params_str} |")

                report_lines.append("")
                report_lines.append("### HPO Details")
                report_lines.append("")
                report_lines.append(f"- **Optimization Rounds:** {hpo_results.get('optimization_rounds', 'N/A')}")
                report_lines.append(f"- **Total Trials:** {hpo_results.get('total_trials', 'N/A')}")
                report_lines.append(f"- **Best Overall Score:** {hpo_results.get('best_overall_score', 'N/A')}")
                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")

            # Training Metrics Section
            report_lines.append("## Training Metrics")
            report_lines.append("")

            # Accuracy metrics
            if 'accuracy' in metrics or 'train_accuracy' in metrics:
                report_lines.append("### Accuracy Metrics")
                report_lines.append("")
                report_lines.append("| Split | Accuracy |")
                report_lines.append("|-------|----------|")
                for split in ['train', 'val', 'test']:
                    key = f"{split}_accuracy"
                    if key in metrics:
                        report_lines.append(f"| {split.capitalize()} | {metrics[key]:.4f} |")
                    elif split == 'train' and 'accuracy' in metrics:
                        report_lines.append(f"| Train | {metrics['accuracy']:.4f} |")
                report_lines.append("")

            # R² metrics
            if any('r2' in k.lower() for k in metrics.keys()):
                report_lines.append("### R² Score Metrics")
                report_lines.append("")
                report_lines.append("| Split | R² Score |")
                report_lines.append("|-------|----------|")
                for split in ['train', 'val', 'test']:
                    for key in [f"{split}_r2", f"{split}_r2_score", f"r2_{split}"]:
                        if key in metrics:
                            report_lines.append(f"| {split.capitalize()} | {metrics[key]:.4f} |")
                            break
                report_lines.append("")

            # Loss metrics
            if any('loss' in k.lower() for k in metrics.keys()):
                report_lines.append("### Loss Metrics")
                report_lines.append("")
                report_lines.append("| Split | Loss |")
                report_lines.append("|-------|------|")
                for split in ['train', 'val', 'test']:
                    for key in [f"{split}_loss", f"loss_{split}"]:
                        if key in metrics:
                            report_lines.append(f"| {split.capitalize()} | {metrics[key]:.6f} |")
                            break
                report_lines.append("")

            # Other metrics
            report_lines.append("### Additional Metrics")
            report_lines.append("")

            # Filter out already-displayed metrics
            displayed_keys = set()
            for key in metrics.keys():
                if any(x in key.lower() for x in ['accuracy', 'r2', 'loss']):
                    displayed_keys.add(key)

            remaining_metrics = {k: v for k, v in metrics.items() if k not in displayed_keys}

            if remaining_metrics:
                report_lines.append("| Metric | Value |")
                report_lines.append("|--------|-------|")
                for key, value in remaining_metrics.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"| {key} | {value:.6f} |")
                    else:
                        report_lines.append(f"| {key} | {value} |")
                report_lines.append("")

            # Model Information
            report_lines.append("---")
            report_lines.append("")
            report_lines.append("## Model Information")
            report_lines.append("")
            report_lines.append(f"- **Training Type:** {training_type}")
            report_lines.append(f"- **Execution Mode:** {config.get('execution_mode', 'unknown')}")
            report_lines.append(f"- **Enable HPO:** {config.get('enable_hpo', False)}")
            report_lines.append("")

            # Save report
            report_content = '\n'.join(report_lines)

            # Determine output directory - save directly in outcomes/
            output_dir = 'outcomes'
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            filename = f"{training_type}_{symbol}_{timeframe}_{direction}_report_{timestamp}.md"
            filepath = os.path.join(output_dir, filename)

            # Write file
            with open(filepath, 'w') as f:
                f.write(report_content)

            tprint_success(f"✅ Generated markdown metrics report: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
          
    def _extract_comprehensive_metrics(
        self,
        result: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from training results.

        This method centralizes metric extraction for all model types,
        ensuring consistency and completeness across all reports.

        Args:
            result: Training result dictionary
            training_type: Type of training (analyst_base, analyst_ensemble, tactician_base, tactician_ensemble)
            config: Configuration dictionary

        Returns:
            Comprehensive metrics dictionary with all available metrics organized by category
        """
        metrics = result.get('metrics', {})
        models = result.get('models', {})

        comprehensive_metrics = {
            'model_type': training_type,
            'timestamp': datetime.now().isoformat(),
            'execution_summary': {},
            'overall_performance': {},
            'per_model_metrics': {},
            'training_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'hpo_results': {},
            'feature_importance': {},
            'data_quality': {},
            'model_complexity': {},
            'prediction_statistics': {},
            'ensemble_specific': {} if 'ensemble' in training_type else None,
            'walkforward_results': {},
            'error_analysis': {},
            'data_drift_checks': {},
            'uncertainty_calibration': {},
            'shap_explanations': {},
            'threshold_optimization': {}
        }

        # ===== EXECUTION SUMMARY =====
        comprehensive_metrics['execution_summary'] = {
            'success': result.get('success', False),
            'execution_time_seconds': result.get('execution_time', 0.0),
            'training_type': training_type,
            'models_trained_count': len(models),
            'model_names': list(models.keys()) if models else [],
            'error': result.get('error', None),
            'warnings': result.get('warnings', [])
        }

        # ===== OVERALL PERFORMANCE METRICS =====
        overall_keys = [
            'overall_accuracy', 'overall_precision', 'overall_recall', 'overall_f1_score',
            'overall_r2_score', 'overall_mse', 'overall_mae', 'overall_rmse',
            'overall_mape', 'overall_sharpe_ratio', 'overall_sortino_ratio',
            'best_model', 'best_model_score', 'model_count',
            'train_test_r2_gap', 'overfitting_ratio', 'generalization_score',
            'avg_overfitting_ratio', 'std_overfitting_ratio',
            'avg_generalization_score', 'std_generalization_score'
        ]
        for key in overall_keys:
            if key in metrics:
                comprehensive_metrics['overall_performance'][key] = metrics[key]

        # ===== PER-MODEL METRICS =====
        # Try to extract from metadata['individual_results'] first
        if 'individual_results' in result.get('metadata', {}):
            individual_results = result['metadata']['individual_results']
            for model_name, model_result in individual_results.items():
                if hasattr(model_result, 'metrics'):
                    comprehensive_metrics['per_model_metrics'][model_name] = model_result.metrics
                elif isinstance(model_result, dict) and 'metrics' in model_result:
                    comprehensive_metrics['per_model_metrics'][model_name] = model_result['metrics']
        
        # Fallback to searching in metrics dict
        for model_name in models.keys():
            if model_name in comprehensive_metrics['per_model_metrics']:
                continue  # Already extracted from individual_results
                
            model_metrics = {}

            # Standard metrics per model
            metric_types = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'r2_score', 'mse', 'mae', 'rmse', 'mape',
                'train_accuracy', 'train_r2', 'train_loss', 'train_mse',
                'val_accuracy', 'val_r2', 'val_loss', 'val_mse',
                'test_accuracy', 'test_r2', 'test_loss', 'test_mse',
                'cv_score_mean', 'cv_score_std', 'cv_scores',
                'training_time_seconds', 'prediction_time_seconds',
                'n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                'iterations', 'depth', 'l2_leaf_reg'
            ]

            for metric_type in metric_types:
                # Check both prefixed and non-prefixed versions
                for key_variant in [f"{model_name}_{metric_type}", f"{metric_type}_{model_name}"]:
                    if key_variant in metrics:
                        model_metrics[metric_type] = metrics[key_variant]
                        break

            # Add if any metrics found
            if model_metrics:
                comprehensive_metrics['per_model_metrics'][model_name] = model_metrics

        # ===== SPLIT-BASED METRICS (Train/Val/Test) =====
        for split in ['train', 'val', 'test']:
            split_metrics = {}
            split_keys = [
                f'{split}_accuracy', f'{split}_precision', f'{split}_recall', f'{split}_f1_score',
                f'{split}_r2', f'{split}_r2_score', f'{split}_mse', f'{split}_mae', f'{split}_rmse',
                f'{split}_loss', f'{split}_samples', f'{split}_time_seconds'
            ]

            for key in split_keys:
                if key in metrics:
                    metric_name = key.replace(f'{split}_', '')
                    split_metrics[metric_name] = metrics[key]

            if split_metrics:
                if split == 'train':
                    comprehensive_metrics['training_metrics'] = split_metrics
                elif split == 'val':
                    comprehensive_metrics['validation_metrics'] = split_metrics
                elif split == 'test':
                    comprehensive_metrics['test_metrics'] = split_metrics

        # ===== HPO RESULTS =====
        hpo_data = result.get('hpo_results') or metrics.get('hpo_results')
        if hpo_data:
            comprehensive_metrics['hpo_results'] = {
                'method': hpo_data.get('method', 'unknown'),
                'optimization_rounds': hpo_data.get('optimization_rounds', 0),
                'total_trials': hpo_data.get('total_trials', 0),
                'best_overall_score': hpo_data.get('best_overall_score', None),
                'best_params': hpo_data.get('best_params', {}),
                'best_scores': hpo_data.get('best_scores', {}),
                'optimization_time': hpo_data.get('optimization_time', 0),
                'per_model_trials': hpo_data.get('per_model_trials', {})
            }

        # ===== FEATURE IMPORTANCE =====
        feature_importance_data = result.get('feature_importance') or metrics.get('feature_importance')
        if feature_importance_data:
            comprehensive_metrics['feature_importance'] = feature_importance_data

        # ===== DATA QUALITY METRICS =====
        # Extract from metrics dict
        data_quality = metrics.get('data_quality', {})
        if data_quality:
            comprehensive_metrics['data_quality'] = data_quality

        # Also extract from result metadata (NEW: from model_trainer comprehensive metadata)
        result_metadata = result.get('metadata', {})
        if 'data_quality' in result_metadata:
            comprehensive_metrics['data_quality'].update(result_metadata['data_quality'])

        # Add basic data stats if available
        # Note: Top-level metrics override nested data_quality values if duplicates exist
        if 'feature_count' in metrics:
            comprehensive_metrics['data_quality']['feature_count'] = metrics['feature_count']
        if 'sample_count' in metrics:
            comprehensive_metrics['data_quality']['sample_count'] = metrics['sample_count']
        if 'missing_values_pct' in metrics:
            comprehensive_metrics['data_quality']['missing_values_pct'] = metrics['missing_values_pct']

        # ===== MODEL COMPLEXITY =====
        # Extract from metrics dict
        complexity_keys = [
            'total_parameters', 'trainable_parameters', 'model_size_mb',
            'inference_time_ms', 'memory_usage_mb'
        ]
        for key in complexity_keys:
            if key in metrics:
                comprehensive_metrics['model_complexity'][key] = metrics[key]

        # Also extract from result metadata (NEW: from model_trainer comprehensive metadata)
        if 'model_complexity' in result_metadata:
            comprehensive_metrics['model_complexity'].update(result_metadata['model_complexity'])

        # ===== PREDICTION STATISTICS =====
        # Extract from metrics dict
        pred_stats_keys = [
            'prediction_mean', 'prediction_std', 'prediction_min', 'prediction_max',
            'prediction_skewness', 'prediction_kurtosis',
            'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate',
            'confusion_matrix'
        ]
        for key in pred_stats_keys:
            if key in metrics:
                comprehensive_metrics['prediction_statistics'][key] = metrics[key]

        # Also extract from result metadata (NEW: from model_trainer comprehensive metadata)
        if 'prediction_statistics' in result_metadata:
            comprehensive_metrics['prediction_statistics'].update(result_metadata['prediction_statistics'])

        # ===== ENSEMBLE-SPECIFIC METRICS =====
        if 'ensemble' in training_type and comprehensive_metrics['ensemble_specific'] is not None:
            ensemble_keys = [
                'ensemble_diversity', 'ensemble_agreement', 'stacking_improvement',
                'base_models_count', 'meta_model_type', 'meta_model_accuracy',
                'weighted_voting_accuracy', 'simple_voting_accuracy'
            ]
            for key in ensemble_keys:
                if key in metrics:
                    comprehensive_metrics['ensemble_specific'][key] = metrics[key]

        # ===== WALK-FORWARD VALIDATION RESULTS =====
        if hasattr(self, '_walkforward_config') and self._walkforward_config:
            wf_metrics = {
                'n_folds': len(self._walkforward_config.folds),
                'strategy': self._walkforward_config.strategy,
                'embargo_days': getattr(self._walkforward_config, 'embargo_days', 0),
                'per_fold_metrics': {}
            }

            # Extract per-fold metrics if available
            for i, fold in enumerate(self._walkforward_config.folds, 1):
                fold_key = f'fold_{i}'
                fold_metrics = {}
                for metric_name in ['accuracy', 'r2', 'mse', 'mae', 'loss']:
                    key = f'{fold_key}_{metric_name}'
                    if key in metrics:
                        fold_metrics[metric_name] = metrics[key]

                if fold_metrics:
                    wf_metrics['per_fold_metrics'][fold_key] = fold_metrics

            comprehensive_metrics['walkforward_results'] = wf_metrics

        # ===== ERROR ANALYSIS =====
        error_keys = [
            'max_error', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error',
            'median_absolute_error', 'explained_variance_score',
            'directional_accuracy', 'sign_accuracy'
        ]
        for key in error_keys:
            if key in metrics:
                comprehensive_metrics['error_analysis'][key] = metrics[key]

        # Also extract from result metadata (NEW: from model_trainer comprehensive metadata)
        if 'error_analysis' in result_metadata:
            comprehensive_metrics['error_analysis'].update(result_metadata['error_analysis'])

        # ===== DATA DRIFT / DISTRIBUTION SHIFT CHECKS =====
        # Detects if train/val/test distributions differ significantly
        # Critical for ensuring models aren't learning time leakage or anomalies
        drift_keys = [
            'ks_test_train_val', 'ks_test_train_test', 'ks_test_val_test',
            'ks_statistic_train_val', 'ks_statistic_train_test', 'ks_statistic_val_test',
            'ks_pvalue_train_val', 'ks_pvalue_train_test', 'ks_pvalue_val_test',
            'psi_train_val', 'psi_train_test', 'psi_val_test',
            'psi_score', 'psi_interpretation',
            'chi_square_train_val', 'chi_square_train_test', 'chi_square_val_test',
            'chi_square_statistic', 'chi_square_pvalue',
            'wasserstein_distance_train_val', 'wasserstein_distance_train_test',
            'jensen_shannon_divergence_train_val', 'jensen_shannon_divergence_train_test',
            'drift_detected', 'drift_score', 'drift_features',
            'covariate_shift_detected', 'concept_drift_detected'
        ]
        for key in drift_keys:
            if key in metrics:
                comprehensive_metrics['data_drift_checks'][key] = metrics[key]

        # Also check per-model drift metrics
        for model_name in models.keys():
            for drift_metric in ['ks_test', 'psi', 'chi_square', 'drift_score']:
                key = f"{model_name}_{drift_metric}"
                if key in metrics:
                    if f'{model_name}_drift' not in comprehensive_metrics['data_drift_checks']:
                        comprehensive_metrics['data_drift_checks'][f'{model_name}_drift'] = {}
                    comprehensive_metrics['data_drift_checks'][f'{model_name}_drift'][drift_metric] = metrics[key]

        # ===== UNCERTAINTY / CONFIDENCE CALIBRATION =====
        # Measures how well predicted probabilities match actual outcomes
        # Critical for decision-making confidence in production
        calibration_keys = [
            'brier_score', 'brier_score_loss',
            'expected_calibration_error', 'ece', 'ece_score',
            'maximum_calibration_error', 'mce', 'mce_score',
            'calibration_curve', 'reliability_diagram',
            'calibration_slope', 'calibration_intercept',
            'log_loss', 'cross_entropy_loss',
            'prediction_confidence_mean', 'prediction_confidence_std',
            'prediction_confidence_median',
            'overconfidence_ratio', 'underconfidence_ratio',
            'confidence_histogram', 'reliability_bins',
            'sharpness', 'refinement',
            'calibration_in_the_large', 'calibration_in_the_small'
        ]
        for key in calibration_keys:
            if key in metrics:
                comprehensive_metrics['uncertainty_calibration'][key] = metrics[key]

        # Per-model calibration metrics
        for model_name in models.keys():
            for calib_metric in ['brier_score', 'ece', 'mce', 'log_loss', 'calibration_slope']:
                key = f"{model_name}_{calib_metric}"
                if key in metrics:
                    if f'{model_name}_calibration' not in comprehensive_metrics['uncertainty_calibration']:
                        comprehensive_metrics['uncertainty_calibration'][f'{model_name}_calibration'] = {}
                    comprehensive_metrics['uncertainty_calibration'][f'{model_name}_calibration'][calib_metric] = metrics[key]

        # ===== SHAPLEY-BASED EXPLANATIONS (SHAP) =====
        # Model interpretability and feature attribution
        # Note: Complex objects like plots are stored separately, only metadata here
        shap_keys = [
            'shap_values_available', 'shap_summary_plot_path', 'shap_dependence_plot_path',
            'shap_force_plot_path', 'shap_waterfall_plot_path',
            'shap_feature_importance', 'shap_interaction_values',
            'shap_top_features', 'shap_top_10_features', 'shap_top_20_features',
            'pdp_plots_path', 'ice_plots_path',
            'pdp_features', 'ice_features',
            'partial_dependence_available', 'individual_conditional_expectation_available',
            'lime_explanations_available', 'lime_top_features',
            'global_feature_importance', 'local_feature_importance'
        ]
        for key in shap_keys:
            if key in metrics:
                comprehensive_metrics['shap_explanations'][key] = metrics[key]

        # Per-model SHAP data
        for model_name in models.keys():
            for shap_metric in ['shap_values', 'shap_feature_importance', 'shap_summary_plot_path']:
                key = f"{model_name}_{shap_metric}"
                if key in metrics:
                    if f'{model_name}_shap' not in comprehensive_metrics['shap_explanations']:
                        comprehensive_metrics['shap_explanations'][f'{model_name}_shap'] = {}
                    comprehensive_metrics['shap_explanations'][f'{model_name}_shap'][shap_metric] = metrics[key]

        # ===== DECISION THRESHOLD OPTIMIZATION =====
        # Optimizing classification thresholds for business objectives
        threshold_keys = [
            'optimal_threshold', 'optimal_threshold_roc', 'optimal_threshold_pr',
            'optimal_threshold_f1', 'optimal_threshold_fbeta',
            'roc_auc_score', 'roc_curve', 'roc_curve_path',
            'pr_auc_score', 'precision_recall_curve', 'pr_curve_path',
            'f1_threshold_curve', 'fbeta_threshold_curve',
            'fbeta_score', 'fbeta_optimal', 'beta_value',
            'cost_matrix', 'cost_weighted_threshold', 'expected_cost',
            'profit_curve', 'profit_optimal_threshold',
            'youden_index', 'youden_threshold',
            'sensitivity_specificity_curve',
            'Matthews_correlation_coefficient', 'mcc', 'mcc_threshold',
            'threshold_metrics', 'threshold_analysis',
            'business_metric_optimal_threshold', 'custom_metric_threshold'
        ]
        for key in threshold_keys:
            if key in metrics:
                comprehensive_metrics['threshold_optimization'][key] = metrics[key]

        # Per-model threshold optimization
        for model_name in models.keys():
            for thresh_metric in ['optimal_threshold', 'roc_auc', 'pr_auc', 'f1_threshold', 'fbeta_optimal']:
                key = f"{model_name}_{thresh_metric}"
                if key in metrics:
                    if f'{model_name}_threshold' not in comprehensive_metrics['threshold_optimization']:
                        comprehensive_metrics['threshold_optimization'][f'{model_name}_threshold'] = {}
                    comprehensive_metrics['threshold_optimization'][f'{model_name}_threshold'][thresh_metric] = metrics[key]

        return comprehensive_metrics

    def _generate_csv_metrics_report(
        self,
        comprehensive_metrics: Dict[str, Any],
        result: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any],
        reports_dir: str
    ) -> Optional[str]:
        """
        Generate a CSV file with one line per model containing all metrics.

        This CSV excludes complex objects (plots, curves, matrices) but includes
        all numeric and string metrics for easy analysis in spreadsheets/data tools.

        Args:
            comprehensive_metrics: Extracted comprehensive metrics
            result: Training result dictionary
            training_type: Type of training
            config: Configuration dictionary
            reports_dir: Directory to save the CSV

        Returns:
            Path to the generated CSV file, or None if failed
        """
        try:
            import csv

            csv_path = os.path.join(reports_dir, f'{training_type}_metrics.csv')
            models = result.get('models', {})

            # If no models, create a single row for the training run
            if not models:
                models = {training_type: None}

            # Collect all possible column names from comprehensive metrics
            csv_columns = []
            csv_rows = []

            # Fixed columns (metadata)
            # NOTE: timestamp is ISO format (YYYY-MM-DDTHH:MM:SS) for sorting
            #       training_date is YYYY-MM-DD for daily grouping
            fixed_columns = [
                'timestamp',                # ISO format, sortable, identifies training run
                'training_date',            # Date only (YYYY-MM-DD) for grouping by day
                'model_name',
                'training_type',
                'symbol',
                'timeframe',
                'direction',
                'execution_time_seconds',
                'success',
                'models_trained_count'
            ]
            csv_columns.extend(fixed_columns)

            # Helper function to flatten nested dicts and filter out complex objects
            def flatten_metrics(metrics_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
                """Flatten nested dict, exclude complex objects."""
                flat = {}
                for key, value in metrics_dict.items():
                    full_key = f"{prefix}{key}" if prefix else key

                    # Skip complex objects that can't be represented in one CSV cell
                    if key in ['shap_values', 'shap_summary_plot_path', 'shap_dependence_plot_path',
                               'shap_force_plot_path', 'shap_waterfall_plot_path',
                               'pdp_plots_path', 'ice_plots_path',
                               'roc_curve', 'pr_curve', 'calibration_curve',
                               'reliability_diagram', 'confusion_matrix',
                               'cost_matrix', 'reliability_bins', 'confidence_histogram',
                               'cv_scores', 'threshold_metrics']:
                        continue

                    # Skip plot/curve paths (too long for CSV)
                    if 'plot_path' in key or 'curve_path' in key or 'plots_path' in key:
                        continue

                    if isinstance(value, dict):
                        # Recursively flatten nested dicts
                        flat.update(flatten_metrics(value, f"{full_key}_"))
                    elif isinstance(value, (list, tuple)):
                        # Skip lists/arrays (can't represent in single cell easily)
                        if len(value) <= 5 and all(isinstance(x, (int, float, str, bool)) for x in value):
                            # Only include short lists of simple types
                            flat[full_key] = str(value)
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        flat[full_key] = value
                    else:
                        # Skip complex types
                        continue

                return flat

            def compute_confidence_reliability(calibration_data, global_calibration):
                brier = calibration_data.get('brier_score') if isinstance(calibration_data, dict) else None
                if brier is None and isinstance(global_calibration, dict):
                    if 'overall_brier_score' in global_calibration:
                        brier = global_calibration.get('overall_brier_score')
                    elif 'brier_score' in global_calibration:
                        brier = global_calibration.get('brier_score')
                ece = None
                if isinstance(calibration_data, dict):
                    if 'expected_calibration_error' in calibration_data:
                        ece = calibration_data.get('expected_calibration_error')
                    elif 'ece' in calibration_data:
                        ece = calibration_data.get('ece')
                if ece is None and isinstance(global_calibration, dict):
                    if 'overall_ece' in global_calibration:
                        ece = global_calibration.get('overall_ece')
                    elif 'expected_calibration_error' in global_calibration:
                        ece = global_calibration.get('expected_calibration_error')
                    elif 'ece' in global_calibration:
                        ece = global_calibration.get('ece')
                components = []
                if isinstance(brier, (int, float)):
                    try:
                        brier_value = float(brier)
                        brier_norm = max(0.0, min(1.0, (0.5 - brier_value) / 0.5))
                        components.append(brier_norm)
                    except Exception:
                        pass
                if isinstance(ece, (int, float)):
                    try:
                        ece_value = float(ece)
                        ece_norm = max(0.0, min(1.0, (0.2 - ece_value) / 0.2))
                        components.append(ece_norm)
                    except Exception:
                        pass
                if not components:
                    return None, None
                score = float(sum(components) / len(components))
                if score >= 0.85:
                    level = 'very_high'
                elif score >= 0.7:
                    level = 'high'
                elif score >= 0.5:
                    level = 'medium'
                elif score >= 0.3:
                    level = 'low'
                else:
                    level = 'very_low'
                return score, level

            def compute_learnability_summary(comprehensive_metrics: Dict[str, Any]) -> Tuple[Optional[float], str, str]:
                overall = comprehensive_metrics.get('overall_performance', {}) or {}
                training_split = comprehensive_metrics.get('training_metrics', {}) or {}
                validation_split = comprehensive_metrics.get('validation_metrics', {}) or {}
                test_split = comprehensive_metrics.get('test_metrics', {}) or {}

                overfit_ratio = overall.get('overfitting_ratio')
                gen_score = overall.get('generalization_score')

                test_r2 = None
                for key in ['r2', 'r2_score']:
                    if key in test_split:
                        test_r2 = test_split.get(key)
                        break

                components = []
                if isinstance(test_r2, (int, float)):
                    try:
                        test_r2_value = float(test_r2)
                        components.append(max(0.0, min(1.0, (test_r2_value + 1.0) / 2.0)))
                    except Exception:
                        pass
                if isinstance(gen_score, (int, float)):
                    try:
                        gen_value = float(gen_score)
                        components.append(max(0.0, min(1.0, gen_value)))
                    except Exception:
                        pass
                if isinstance(overfit_ratio, (int, float)):
                    try:
                        of_value = float(overfit_ratio)
                        components.append(max(0.0, min(1.0, 1.0 - of_value)))
                    except Exception:
                        pass

                learnability_score: Optional[float] = None
                if components:
                    learnability_score = float(sum(components) / len(components))

                if learnability_score is None:
                    learnability_status = 'unknown'
                elif learnability_score >= 0.8:
                    learnability_status = 'strong'
                elif learnability_score >= 0.6:
                    learnability_status = 'moderate'
                else:
                    learnability_status = 'weak'

                overfitting_category = 'unknown'
                if isinstance(overfit_ratio, (int, float)):
                    try:
                        of_value = float(overfit_ratio)
                        if of_value < 0.1:
                            overfitting_category = 'low'
                        elif of_value < 0.2:
                            overfitting_category = 'moderate'
                        else:
                            overfitting_category = 'high'
                    except Exception:
                        overfitting_category = 'unknown'

                return learnability_score, learnability_status, overfitting_category

            # Build rows for each model
            for model_name in models.keys():
                row = {}

                # Add fixed metadata (timestamp fields first for easy sorting)
                timestamp_str = comprehensive_metrics.get('timestamp', datetime.now().isoformat())
                row['timestamp'] = timestamp_str

                # Extract date part for daily grouping (YYYY-MM-DD)
                try:
                    if 'T' in timestamp_str:
                        row['training_date'] = timestamp_str.split('T')[0]
                    else:
                        row['training_date'] = datetime.now().strftime('%Y-%m-%d')
                except Exception:
                    row['training_date'] = datetime.now().strftime('%Y-%m-%d')

                row['model_name'] = model_name
                row['training_type'] = training_type
                row['symbol'] = config.get('symbol', 'UNKNOWN')
                row['timeframe'] = config.get('timeframe', '15m')
                row['direction'] = config.get('direction', 'long')
                row['execution_time_seconds'] = comprehensive_metrics['execution_summary'].get('execution_time_seconds', 0)
                row['success'] = comprehensive_metrics['execution_summary'].get('success', False)
                row['models_trained_count'] = comprehensive_metrics['execution_summary'].get('models_trained_count', 0)

                learnability_score, learnability_status, overfitting_category = compute_learnability_summary(comprehensive_metrics)
                if learnability_score is not None:
                    row['learnability_score'] = learnability_score
                row['learnability_status'] = learnability_status
                row['overfitting_category'] = overfitting_category

                # Add flattened metrics from all categories
                categories = [
                    'overall_performance',
                    'training_metrics',
                    'validation_metrics',
                    'test_metrics',
                    'data_quality',
                    'model_complexity',
                    'prediction_statistics',
                    'error_analysis',
                    'data_drift_checks',
                    'uncertainty_calibration',
                    'threshold_optimization'
                ]

                for category in categories:
                    if category in comprehensive_metrics:
                        flat_cat = flatten_metrics(comprehensive_metrics[category], f"{category}_")
                        row.update(flat_cat)

                # Add per-model metrics if available
                if model_name in comprehensive_metrics.get('per_model_metrics', {}):
                    model_specific = flatten_metrics(
                        comprehensive_metrics['per_model_metrics'][model_name],
                        'model_specific_'
                    )
                    row.update(model_specific)

                # Add HPO results (flattened)
                if comprehensive_metrics.get('hpo_results'):
                    hpo_flat = flatten_metrics(comprehensive_metrics['hpo_results'], 'hpo_')
                    # Exclude large nested structures
                    hpo_flat = {k: v for k, v in hpo_flat.items()
                                if not k.endswith('_best_params') and not k.endswith('_best_scores')}
                    row.update(hpo_flat)

                # Add walk-forward metrics (simplified)
                if comprehensive_metrics.get('walkforward_results'):
                    wf = comprehensive_metrics['walkforward_results']
                    if 'n_folds' in wf:
                        row['walkforward_n_folds'] = wf['n_folds']
                    if 'strategy' in wf:
                        row['walkforward_strategy'] = wf['strategy']
                    if 'embargo_days' in wf:
                        row['walkforward_embargo_days'] = wf['embargo_days']

                # Add ensemble-specific metrics if applicable
                if comprehensive_metrics.get('ensemble_specific'):
                    ensemble_flat = flatten_metrics(comprehensive_metrics['ensemble_specific'], 'ensemble_')
                    row.update(ensemble_flat)

                calibration_all = comprehensive_metrics.get('uncertainty_calibration', {})
                model_calibration_key = f"{model_name}_calibration"
                model_calibration = {}
                if isinstance(calibration_all, dict) and model_calibration_key in calibration_all:
                    calib_value = calibration_all.get(model_calibration_key)
                    if isinstance(calib_value, dict):
                        model_calibration = calib_value
                reliability_score, reliability_level = compute_confidence_reliability(model_calibration, calibration_all)
                if reliability_score is not None:
                    row['confidence_reliability_score'] = reliability_score
                if reliability_level is not None:
                    row['confidence_reliability_level'] = reliability_level

                # Collect all column names
                for key in row.keys():
                    if key not in csv_columns:
                        csv_columns.append(key)

                csv_rows.append(row)

            # ========================================================================
            # 1. Write Per-Run CSV (timestamped, one file per training run)
            # ========================================================================
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()

                for row in csv_rows:
                    # Ensure all columns exist in row (fill missing with None)
                    complete_row = {col: row.get(col, None) for col in csv_columns}
                    writer.writerow(complete_row)

            tprint_success(f"✅ CSV metrics report saved: {csv_path}")

            # ========================================================================
            # 2. Append to Consolidated CSV (aggregates ALL models across ALL runs)
            # ========================================================================
            # This allows aggregation when running analyst/tactician base/ensemble separately
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')

            # Consolidated CSV path (at symbol level, not timestamped)
            # Store consolidated metrics under outcomes/, with a filename that
            # includes the training step name and a filesystem-safe timestamp
            timestamp_str = comprehensive_metrics.get('timestamp', datetime.now().isoformat())
            safe_ts = timestamp_str.replace(':', '').replace('-', '').replace('T', '_').split('.')[0]

            consolidated_dir = os.path.join('outcomes', f"{symbol}_{timeframe}_{direction}")
            os.makedirs(consolidated_dir, exist_ok=True)
            consolidated_csv_path = os.path.join(
                consolidated_dir,
                f"all_models_metrics_{training_type}_{safe_ts}.csv",
            )

            # Check if consolidated CSV exists to determine if we need headers
            file_exists = os.path.exists(consolidated_csv_path)

            # If file exists, read existing headers to ensure compatibility
            existing_columns = []
            if file_exists:
                try:
                    with open(consolidated_csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_columns = reader.fieldnames if reader.fieldnames else []
                except Exception as e:
                    self.logger.warning(f"Could not read existing consolidated CSV headers: {e}")
                    existing_columns = []

            # Merge column sets (existing + new) to ensure all columns are present
            if existing_columns:
                # Create union of column sets, preserving order
                all_columns = list(existing_columns)
                for col in csv_columns:
                    if col not in all_columns:
                        all_columns.append(col)
            else:
                all_columns = csv_columns

            # If columns were added, we need to rewrite the file with expanded headers
            if file_exists and existing_columns and set(all_columns) != set(existing_columns):
                tprint_info(f"📊 Expanding consolidated CSV with new columns...")

                # Read existing data
                existing_data = []
                try:
                    with open(consolidated_csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_data = list(reader)
                except Exception as e:
                    self.logger.error(f"Failed to read existing CSV for column expansion: {e}")
                    existing_data = []

                # Rewrite with expanded columns
                if existing_data:
                    with open(consolidated_csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
                        writer.writeheader()
                        for old_row in existing_data:
                            complete_row = {col: old_row.get(col, None) for col in all_columns}
                            writer.writerow(complete_row)
                    file_exists = True  # File has been rewritten

            # Append new rows to consolidated CSV
            try:
                # Use file locking to prevent concurrent write issues
                import fcntl
                has_fcntl = True
            except ImportError:
                # fcntl not available on Windows
                has_fcntl = False

            mode = 'a' if file_exists else 'w'
            with open(consolidated_csv_path, mode, newline='', encoding='utf-8') as csvfile:
                # Apply file lock if available (Unix-like systems)
                if has_fcntl:
                    try:
                        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
                    except Exception as e:
                        self.logger.warning(f"Could not acquire file lock: {e}")

                writer = csv.DictWriter(csvfile, fieldnames=all_columns, extrasaction='ignore')

                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()

                # Append all rows from this training run
                for row in csv_rows:
                    complete_row = {col: row.get(col, None) for col in all_columns}
                    writer.writerow(complete_row)

                # Release lock (automatic when file closes, but explicit for clarity)
                if has_fcntl:
                    try:
                        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass

            tprint_success(f"✅ Consolidated CSV updated: {consolidated_csv_path}")
            tprint_info(f"   ↪ Added {len(csv_rows)} model(s) to consolidated metrics")

            return csv_path

        except Exception as e:
            self.logger.error(f"Failed to generate CSV metrics report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _generate_training_reports(
        self,
        result: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate comprehensive markdown and JSON reports for training metrics.

        This is the centralized reporting hub that creates detailed reports
        with as many metrics as possible for each of the 4 model types:
        - Analyst Base
        - Analyst Ensemble
        - Tactician Base
        - Tactician Ensemble

        Args:
            result: Training result dictionary containing metrics and models
            training_type: Type of training (analyst_base, tactician_base, etc.)
            config: Configuration dictionary

        Returns:
            Dictionary with paths to generated reports
        """
        try:
            import json

            report_paths = {}

            # Extract comprehensive metrics using centralized extractor
            comprehensive_metrics = self._extract_comprehensive_metrics(result, training_type, config)

            # Create outcomes directory for reports
            outcomes_dir = 'outcomes'
            os.makedirs(outcomes_dir, exist_ok=True)

            # ========================================================================
            # COMPREHENSIVE MARKDOWN REPORT WITH ALL METRICS
            # ========================================================================
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            markdown_path = os.path.join(outcomes_dir, f'{training_type}_{symbol}_{timeframe}_{direction}_report_{timestamp}.md')

            with open(markdown_path, 'w') as f:
                # ===== HEADER =====
                f.write(f"# {training_type.replace('_', ' ').title()} - Comprehensive Training Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

                # ===== EXECUTION SUMMARY =====
                f.write("## 📋 Execution Summary\n\n")
                exec_summary = comprehensive_metrics['execution_summary']
                f.write(f"- **Training Type:** {exec_summary.get('training_type', 'N/A')}\n")
                f.write(f"- **Success:** {'✅ Yes' if exec_summary.get('success', False) else '❌ No'}\n")
                f.write(f"- **Execution Time:** {exec_summary.get('execution_time_seconds', 0):.2f} seconds\n")
                f.write(f"- **Models Trained:** {exec_summary.get('models_trained_count', 0)}\n")
                if exec_summary.get('model_names'):
                    f.write(f"- **Model Names:** {', '.join(exec_summary['model_names'])}\n")
                if exec_summary.get('error'):
                    f.write(f"- **Error:** {exec_summary['error']}\n")
                if exec_summary.get('warnings'):
                    f.write(f"- **Warnings:** {len(exec_summary['warnings'])} warning(s)\n")
                f.write("\n---\n\n")

                # ===== CONFIGURATION =====
                f.write("## ⚙️ Configuration\n\n")
                f.write(f"- **Symbol:** {symbol}\n")
                f.write(f"- **Exchange:** {config.get('exchange', 'binance')}\n")
                f.write(f"- **Timeframe:** {timeframe}\n")
                f.write(f"- **Direction:** {direction}\n")
                f.write(f"- **Execution Mode:** {config.get('execution_mode', 'full')}\n")
                effective_hpo_enabled = bool(comprehensive_metrics.get('hpo_results'))
                f.write(f"- **HPO Enabled:** {effective_hpo_enabled}\n")
                f.write("\n---\n\n")

                # ===== OVERALL PERFORMANCE METRICS =====
                f.write("## 📊 Overall Performance Metrics\n\n")
                overall_perf = comprehensive_metrics['overall_performance']
                if overall_perf:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(overall_perf.items()):
                        label = key.replace('overall_', '').replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No overall performance metrics available.*\n\n")
                f.write("---\n\n")

                # ===== TRAINING/VALIDATION/TEST SPLIT METRICS =====
                f.write("## 📈 Split-Based Performance Metrics\n\n")

                # Training Metrics
                f.write("### Training Set Metrics\n\n")
                train_metrics = comprehensive_metrics['training_metrics']
                if train_metrics:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(train_metrics.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No training metrics available.*\n\n")

                # Validation Metrics
                f.write("### Validation Set Metrics\n\n")
                val_metrics = comprehensive_metrics['validation_metrics']
                if val_metrics:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(val_metrics.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No validation metrics available.*\n\n")

                # Test Metrics
                f.write("### Test Set Metrics\n\n")
                test_metrics = comprehensive_metrics['test_metrics']
                if test_metrics:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(test_metrics.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No test metrics available.*\n\n")
                f.write("---\n\n")

                f.write("## 📚 Learnability & Generalization Diagnostics\n\n")
                f.write("This section summarizes how the model learns from data and how robustly it generalizes.\n\n")

                learn_train = comprehensive_metrics.get('training_metrics', {}) or {}
                learn_val = comprehensive_metrics.get('validation_metrics', {}) or {}
                learn_test = comprehensive_metrics.get('test_metrics', {}) or {}
                overall_perf = comprehensive_metrics.get('overall_performance', {}) or {}

                has_r2 = any(k in learn_train for k in ['r2', 'r2_score']) or any(k in learn_val for k in ['r2', 'r2_score']) or any(k in learn_test for k in ['r2', 'r2_score'])
                has_acc = 'accuracy' in learn_train or 'accuracy' in learn_val or 'accuracy' in learn_test

                if has_r2 or has_acc:
                    f.write("| Metric | Train | Validation | Test |\n")
                    f.write("|--------|-------|------------|------|\n")

                    def _fmt_split(split_dict, key):
                        value = split_dict.get(key)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            return f"{value:.4f}"
                        return "N/A"

                    if has_r2:
                        f.write(
                            "| R² | "
                            f"{_fmt_split(learn_train, 'r2')} | "
                            f"{_fmt_split(learn_val, 'r2')} | "
                            f"{_fmt_split(learn_test, 'r2')} |\n"
                        )
                    if has_acc:
                        f.write(
                            "| Accuracy | "
                            f"{_fmt_split(learn_train, 'accuracy')} | "
                            f"{_fmt_split(learn_val, 'accuracy')} | "
                            f"{_fmt_split(learn_test, 'accuracy')} |\n"
                        )
                    f.write("\n")
                else:
                    f.write("*No split-based R²/accuracy metrics available for learnability summary.*\n\n")

                gap = overall_perf.get('train_test_r2_gap')
                overfit_ratio = overall_perf.get('overfitting_ratio')
                gen_score = overall_perf.get('generalization_score')

                if any(v is not None for v in [gap, overfit_ratio, gen_score]):
                    f.write("### Overfitting & Generalization Indicators\n\n")

                    def _fmt_float(value):
                        try:
                            return f"{float(value):.4f}"
                        except Exception:
                            return str(value)

                    if gap is not None:
                        f.write(f"- **Train–Test R² Gap:** {_fmt_float(gap)}  \\n")
                        f.write("  Larger gaps indicate that the model fits the training data much better than unseen data.\n")
                    if overfit_ratio is not None:
                        level = "unknown"
                        try:
                            ratio_val = float(overfit_ratio)
                            if ratio_val < 0.1:
                                level = "low (good)"
                            elif ratio_val < 0.2:
                                level = "moderate"
                            else:
                                level = "high (risk of overfitting)"
                        except Exception:
                            level = "unknown"
                        f.write(f"- **Overfitting Ratio:** {_fmt_float(overfit_ratio)}  \\n")
                        f.write(f"  Approximate relative gap between train and test performance → **{level}**.\n")
                    if gen_score is not None:
                        f.write(f"- **Generalization Score:** {_fmt_float(gen_score)}  \\n")
                        f.write("  Ratio of test to train performance; values near 1.0 indicate similar train/test behaviour.\n")
                    f.write("\n")
                else:
                    f.write("*No explicit overfitting/generalization indicators were recorded for this run.*\n\n")

                f.write("---\n\n")

                regime_metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
                regime_breakdown = None
                if isinstance(regime_metadata, dict):
                    if isinstance(regime_metadata.get('regime_performance'), dict):
                        regime_breakdown = regime_metadata.get('regime_performance')
                    elif isinstance(regime_metadata.get('regime_analysis'), dict):
                        ra = regime_metadata.get('regime_analysis')
                        if isinstance(ra.get('regime_performance'), dict):
                            regime_breakdown = ra.get('regime_performance')
                        else:
                            regime_breakdown = ra
                    elif isinstance(regime_metadata.get('regime_performance'), list):
                        temp = {}
                        for entry in regime_metadata.get('regime_performance'):
                            if isinstance(entry, dict):
                                name = entry.get('regime') or entry.get('regime_name')
                                if name:
                                    temp[name] = entry
                        if temp:
                            regime_breakdown = temp

                if regime_breakdown:
                    f.write("## 🌍 Regime Breakdown\n\n")
                    f.write("| Regime | Metric | Value |\n")
                    f.write("|--------|--------|-------|\n")

                    for regime_name, stats in sorted(regime_breakdown.items()):
                        if not isinstance(stats, dict):
                            continue
                        for key, value in sorted(stats.items()):
                            if key in ('regime', 'regime_name'):
                                continue
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                try:
                                    value_str = f"{float(value):.4f}"
                                except Exception:
                                    value_str = str(value)
                            else:
                                value_str = str(value)
                            f.write(f"| {regime_name} | {label} | {value_str} |\n")

                    f.write("\n---\n\n")

                # ===== PER-MODEL DETAILED METRICS =====
                f.write("## 🤖 Per-Model Detailed Metrics\n\n")
                per_model = comprehensive_metrics['per_model_metrics']
                if per_model:
                    f.write(f"**Total Models:** {len(per_model)}\n\n")
                    for model_name, model_metrics in per_model.items():
                        f.write(f"### {model_name.upper()}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(model_metrics.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")
                else:
                    f.write("*No per-model metrics available.*\n\n")
                f.write("---\n\n")

                # ===== HPO RESULTS =====
                f.write("## 🔍 Hyperparameter Optimization (HPO) Results\n\n")
                hpo_results = comprehensive_metrics['hpo_results']
                if hpo_results and hpo_results.get('method'):
                    f.write(f"**Method:** {hpo_results.get('method', 'N/A')}\n")
                    f.write(f"**Optimization Rounds:** {hpo_results.get('optimization_rounds', 0)}\n")
                    f.write(f"**Total Trials:** {hpo_results.get('total_trials', 0)}\n")
                    f.write(f"**Best Overall Score:** {hpo_results.get('best_overall_score', 'N/A')}\n")
                    f.write(f"**Optimization Time:** {hpo_results.get('optimization_time', 0):.2f}s\n\n")

                    if hpo_results.get('best_params'):
                        f.write("### Best Parameters by Model\n\n")
                        for model_name, params in hpo_results['best_params'].items():
                            f.write(f"#### {model_name.upper()}\n\n")
                            if isinstance(params, dict):
                                f.write("```json\n")
                                f.write(json.dumps(params, indent=2))
                                f.write("\n```\n\n")

                    if hpo_results.get('best_scores'):
                        f.write("### Best Scores by Model\n\n")
                        f.write("| Model | Score |\n")
                        f.write("|-------|-------|\n")
                        for model_name, score_data in hpo_results['best_scores'].items():
                            if isinstance(score_data, dict):
                                score = score_data.get('score', 'N/A')
                            else:
                                score = score_data
                            f.write(f"| {model_name} | {score if isinstance(score, str) else f'{score:.6f}'} |\n")
                        f.write("\n")
                else:
                    f.write("*No HPO results available or HPO was disabled.*\n\n")
                f.write("---\n\n")

                # ===== WALK-FORWARD VALIDATION RESULTS =====
                f.write("## 📅 Walk-Forward Validation Results\n\n")
                wf_results = comprehensive_metrics['walkforward_results']
                if wf_results and wf_results.get('n_folds'):
                    f.write(f"**Number of Folds:** {wf_results.get('n_folds', 0)}\n")
                    f.write(f"**Strategy:** {wf_results.get('strategy', 'N/A')}\n")
                    f.write(f"**Embargo Days:** {wf_results.get('embargo_days', 0)}\n\n")

                    per_fold = wf_results.get('per_fold_metrics', {})
                    if per_fold:
                        f.write("### Per-Fold Metrics\n\n")
                        for fold_name, fold_metrics in sorted(per_fold.items()):
                            f.write(f"#### {fold_name.upper().replace('_', ' ')}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(fold_metrics.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No walk-forward validation results available.*\n\n")
                f.write("---\n\n")

                # ===== ENSEMBLE-SPECIFIC METRICS =====
                if comprehensive_metrics['ensemble_specific'] is not None:
                    f.write("## 🎯 Ensemble-Specific Metrics\n\n")
                    ensemble_metrics = comprehensive_metrics['ensemble_specific']
                    if ensemble_metrics:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(ensemble_metrics.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")

                        diversity = ensemble_metrics.get('ensemble_diversity')
                        stacking_gain = ensemble_metrics.get('stacking_improvement')
                        simple_gain = None
                        weighted_gain = None
                        best_base = None

                        try:
                            if 'base_models_count' in ensemble_metrics:
                                best_base = ensemble_metrics.get('best_base_score')
                        except Exception:
                            best_base = None

                        try:
                            if 'simple_voting_accuracy' in ensemble_metrics and best_base is not None:
                                simple_gain = float(ensemble_metrics['simple_voting_accuracy']) - float(best_base)
                            if 'weighted_voting_accuracy' in ensemble_metrics and best_base is not None:
                                weighted_gain = float(ensemble_metrics['weighted_voting_accuracy']) - float(best_base)
                        except Exception:
                            simple_gain = None
                            weighted_gain = None

                        f.write("### Diversity & Per-Member vs Ensemble Gains\n\n")
                        f.write("This section helps interpret how much the ensemble benefits from member disagreement and stacking.\n\n")

                        def _fmt_val(v):
                            try:
                                return f"{float(v):.4f}"
                            except Exception:
                                return str(v)

                        if diversity is not None:
                            f.write(f"- **Ensemble Diversity:** {_fmt_val(diversity)}  \\\n")
                            f.write("  Higher values indicate that base models make sufficiently different predictions to enable useful aggregation.\n")
                        if stacking_gain is not None:
                            f.write(f"- **Stacking Improvement:** {_fmt_val(stacking_gain)}  \\\n")
                            f.write("  Average performance gain of the stacked meta-model over the best individual base model.\n")
                        if simple_gain is not None or weighted_gain is not None:
                            if simple_gain is not None:
                                f.write(f"- **Simple Voting Gain (vs Best Base):** {_fmt_val(simple_gain)}  \\\n")
                            if weighted_gain is not None:
                                f.write(f"- **Weighted Voting Gain (vs Best Base):** {_fmt_val(weighted_gain)}  \\\n")
                            f.write("  Positive gains mean the ensemble is extracting extra signal beyond any single member.\n")
                        f.write("\n")
                    else:
                        f.write("*No ensemble-specific metrics available.*\n\n")
                    f.write("---\n\n")

                # ===== FEATURE IMPORTANCE =====
                f.write("## 📋 Feature Importance\n\n")
                feat_importance = comprehensive_metrics['feature_importance']
                if feat_importance:
                    if isinstance(feat_importance, dict):
                        f.write("### Top 20 Most Important Features\n\n")
                        # Sort by importance value (descending)
                        sorted_features = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:20]
                        f.write("| Rank | Feature | Importance |\n")
                        f.write("|------|---------|------------|\n")
                        for i, (feature, importance) in enumerate(sorted_features, 1):
                            if isinstance(importance, (int, float)):
                                f.write(f"| {i} | {feature} | {importance:.6f} |\n")
                            else:
                                f.write(f"| {i} | {feature} | {importance} |\n")
                        f.write("\n")
                else:
                    f.write("*No feature importance data available.*\n\n")
                f.write("---\n\n")

                # ===== DATA QUALITY METRICS =====
                f.write("## 📊 Data Quality Metrics\n\n")
                data_quality = comprehensive_metrics['data_quality']
                if data_quality:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(data_quality.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            formatted_value = f"{float(value):.6f}"
                        else:
                            formatted_value = value
                        f.write(f"| {label} | {formatted_value} |\n")
                    f.write("\n")
                else:
                    f.write("*No data quality metrics available.*\n\n")
                f.write("---\n\n")

                # ===== MODEL COMPLEXITY =====
                f.write("## 🧮 Model Complexity Metrics\n\n")
                complexity = comprehensive_metrics['model_complexity']
                if complexity:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(complexity.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            formatted_value = f"{float(value):.6f}"
                        else:
                            formatted_value = value
                        f.write(f"| {label} | {formatted_value} |\n")
                    f.write("\n")
                else:
                    f.write("*No model complexity metrics available.*\n\n")
                f.write("---\n\n")

                # ===== PREDICTION STATISTICS =====
                f.write("## 📊 Prediction Statistics\n\n")
                pred_stats = comprehensive_metrics['prediction_statistics']
                if pred_stats:
                    f.write("| Statistic | Value |\n")
                    f.write("|-----------|-------|\n")
                    for key, value in sorted(pred_stats.items()):
                        label = key.replace('_', ' ').title()
                        if key == 'confusion_matrix':
                            f.write(f"| {label} | See detailed analysis |\n")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No prediction statistics available.*\n\n")
                f.write("---\n\n")

                # ===== ERROR ANALYSIS =====
                f.write("## ⚠️ Error Analysis\n\n")
                error_analysis = comprehensive_metrics['error_analysis']
                if error_analysis:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(error_analysis.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No error analysis metrics available.*\n\n")
                f.write("---\n\n")

                # ===== DATA DRIFT / DISTRIBUTION SHIFT CHECKS =====
                f.write("## 📊 Data Drift & Distribution Shift Checks\n\n")
                f.write("*Detects if train/val/test distributions differ significantly (KS tests, PSI, chi-square)*\n\n")
                drift_checks = comprehensive_metrics['data_drift_checks']
                if drift_checks:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(drift_checks.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, dict):
                            # Per-model drift metrics
                            f.write(f"| **{label}** | (see details below) |\n")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")

                    # Detailed per-model drift if available
                    per_model_drift = {k: v for k, v in drift_checks.items() if isinstance(v, dict)}
                    if per_model_drift:
                        f.write("### Per-Model Drift Metrics\n\n")
                        for model_name, drift_data in per_model_drift.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(drift_data.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No data drift checks available.*\n\n")
                f.write("---\n\n")

                # ===== UNCERTAINTY / CONFIDENCE CALIBRATION =====
                f.write("## 🎯 Uncertainty & Confidence Calibration\n\n")
                f.write("*Measures how well predicted probabilities match actual outcomes (Brier Score, ECE)*\n\n")
                calibration = comprehensive_metrics['uncertainty_calibration']
                if calibration:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(calibration.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, dict):
                            # Per-model calibration metrics
                            f.write(f"| **{label}** | (see details below) |\n")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")

                    # Detailed per-model calibration if available
                    per_model_calib = {k: v for k, v in calibration.items() if isinstance(v, dict)}
                    if per_model_calib:
                        f.write("### Per-Model Calibration Metrics\n\n")
                        for model_name, calib_data in per_model_calib.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(calib_data.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No uncertainty/calibration metrics available.*\n\n")
                f.write("---\n\n")

                # ===== SHAPLEY-BASED EXPLANATIONS (SHAP) =====
                f.write("## 🔍 SHAP Explanations & Model Interpretability\n\n")
                f.write("*Shapley values, PDP/ICE curves, and feature attribution*\n\n")
                shap_exp = comprehensive_metrics['shap_explanations']
                if shap_exp:
                    # Filter out plot paths and complex objects for main table
                    simple_shap = {k: v for k, v in shap_exp.items()
                                   if not isinstance(v, dict) and 'plot' not in k.lower() and 'curve' not in k.lower()}

                    if simple_shap:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(simple_shap.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")

                    # Plot paths if available
                    plot_paths = {k: v for k, v in shap_exp.items() if 'path' in k.lower()}
                    if plot_paths:
                        f.write("### Generated Explanation Plots\n\n")
                        for key, path in sorted(plot_paths.items()):
                            label = key.replace('_', ' ').replace('path', '').strip().title()
                            f.write(f"- **{label}:** `{path}`\n")
                        f.write("\n")

                    # Per-model SHAP data
                    per_model_shap = {k: v for k, v in shap_exp.items() if isinstance(v, dict)}
                    if per_model_shap:
                        f.write("### Per-Model SHAP Data\n\n")
                        for model_name, shap_data in per_model_shap.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            for key, value in sorted(shap_data.items()):
                                label = key.replace('_', ' ').title()
                                if 'path' in key.lower():
                                    f.write(f"- **{label}:** `{value}`\n")
                                elif isinstance(value, (int, float)):
                                    f.write(f"- **{label}:** {value:.6f}\n")
                                else:
                                    f.write(f"- **{label}:** {value}\n")
                            f.write("\n")
                else:
                    f.write("*No SHAP explanations available.*\n\n")
                f.write("---\n\n")

                # ===== DECISION THRESHOLD OPTIMIZATION =====
                f.write("## ⚖️ Decision Threshold Optimization\n\n")
                f.write("*ROC/PR curves, F-beta optimization, cost-weighted thresholds*\n\n")
                threshold_opt = comprehensive_metrics['threshold_optimization']
                if threshold_opt:
                    # Filter out complex objects
                    simple_threshold = {k: v for k, v in threshold_opt.items()
                                        if not isinstance(v, dict) and 'curve' not in k.lower() and 'matrix' not in k.lower()}

                    if simple_threshold:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(simple_threshold.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")

                    # Per-model threshold optimization
                    per_model_threshold = {k: v for k, v in threshold_opt.items() if isinstance(v, dict)}
                    if per_model_threshold:
                        f.write("### Per-Model Threshold Optimization\n\n")
                        for model_name, thresh_data in per_model_threshold.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(thresh_data.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No threshold optimization metrics available.*\n\n")
                f.write("---\n\n")

                # ===== ARTIFACTS =====
                f.write("## 💾 Generated Artifacts\n\n")
                if 'artifacts' in result:
                    artifacts = result['artifacts']
                    f.write("| Artifact Name | Path |\n")
                    f.write("|---------------|------|\n")
                    for artifact_name, artifact_path in sorted(artifacts.items()):
                        f.write(f"| {artifact_name} | `{artifact_path}` |\n")
                    f.write("\n")
                else:
                    f.write("*No artifacts information available.*\n\n")

                # ===== FOOTER =====
                f.write("---\n\n")
                f.write(f"*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on {timestamp}*\n")
                f.write(f"*Training Type: {training_type.upper()} | Symbol: {symbol} | Timeframe: {timeframe} | Direction: {direction}*\n")

            report_paths['markdown'] = markdown_path
            tprint_success(f"✅ Markdown report saved: {markdown_path}")

            # ========================================================================
            # COMPREHENSIVE JSON REPORT WITH ALL METRICS
            # ========================================================================
            json_path = os.path.join(outcomes_dir, f'{training_type}_{symbol}_{timeframe}_{direction}_metrics_{timestamp}.json')

            # Build comprehensive JSON report using the extracted metrics
            json_report = {
                'report_version': '3.0',
                'metadata': {
                    'training_type': training_type,
                    'symbol': symbol,
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': config.get('execution_mode', 'full'),
                    'timestamp': timestamp,
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'comprehensive_training_metrics'
                },
                'configuration': {
                    'symbol': symbol,
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': config.get('execution_mode', 'full'),
                    # Reflect effective HPO activity rather than just the raw config flag
                    'enable_hpo': bool(comprehensive_metrics.get('hpo_results')),
                    'train_percentage': config.get('train_percentage', 0.70),
                    'validation_percentage': config.get('validation_percentage', 0.15),
                    'test_percentage': config.get('test_percentage', 0.15),
                    'walkforward_config': str(config.get('walkforward_config', 'N/A'))
                },

                # ===== COMPREHENSIVE METRICS =====
                'execution_summary': comprehensive_metrics['execution_summary'],
                'overall_performance': comprehensive_metrics['overall_performance'],
                'per_model_metrics': comprehensive_metrics['per_model_metrics'],
                'training_metrics': comprehensive_metrics['training_metrics'],
                'validation_metrics': comprehensive_metrics['validation_metrics'],
                'test_metrics': comprehensive_metrics['test_metrics'],
                'hpo_results': comprehensive_metrics['hpo_results'],
                'walkforward_results': comprehensive_metrics['walkforward_results'],
                'feature_importance': comprehensive_metrics['feature_importance'],
                'data_quality': comprehensive_metrics['data_quality'],
                'model_complexity': comprehensive_metrics['model_complexity'],
                'prediction_statistics': comprehensive_metrics['prediction_statistics'],
                'error_analysis': comprehensive_metrics['error_analysis'],
                'data_drift_checks': comprehensive_metrics['data_drift_checks'],
                'uncertainty_calibration': comprehensive_metrics['uncertainty_calibration'],
                'shap_explanations': comprehensive_metrics['shap_explanations'],
                'threshold_optimization': comprehensive_metrics['threshold_optimization'],

                # Add ensemble-specific metrics if applicable
                'ensemble_specific': comprehensive_metrics['ensemble_specific'] if comprehensive_metrics['ensemble_specific'] is not None else None,

                # ===== RAW METRICS (for backward compatibility) =====
                'raw_metrics': result.get('metrics', {}),

                # ===== ARTIFACTS =====
                'artifacts': result.get('artifacts', {}),

                # ===== MODELS INFO =====
                'models': {
                    'count': len(result.get('models', {})),
                    'names': list(result.get('models', {}).keys()),
                    'details': {
                        model_name: {
                            'type': str(type(model).__name__) if hasattr(model, '__class__') else 'unknown'
                        }
                        for model_name, model in result.get('models', {}).items()
                    }
                }
            }

            # Save JSON report
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)

            report_paths['json'] = json_path
            tprint_success(f"✅ Comprehensive JSON metrics saved: {json_path}")

            # ========================================================================
            # CSV METRICS REPORT (One line per model)
            # ========================================================================
            csv_path = self._generate_csv_metrics_report(
                comprehensive_metrics=comprehensive_metrics,
                result=result,
                training_type=training_type,
                config=config,
                reports_dir=outcomes_dir
            )
            if csv_path:
                report_paths['csv'] = csv_path

            # ========================================================================
            # SUMMARY LOG OUTPUT
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info(f"📊 TRAINING REPORT SUMMARY - {training_type.upper()}")
            tprint_info("=" * 80)
            tprint_info(f"✅ Success: {comprehensive_metrics['execution_summary']['success']}")
            tprint_info(f"⏱️  Execution Time: {comprehensive_metrics['execution_summary']['execution_time_seconds']:.2f}s")
            tprint_info(f"🤖 Models Trained: {comprehensive_metrics['execution_summary']['models_trained_count']}")

            if comprehensive_metrics['overall_performance']:
                tprint_info("📈 Overall Performance:")
                for key, value in list(comprehensive_metrics['overall_performance'].items())[:5]:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        tprint_info(f"   • {key}: {value:.4f}")
                    else:
                        tprint_info(f"   • {key}: {value}")

            tprint_info(f"📄 Markdown Report: {markdown_path}")
            tprint_info(f"📊 JSON Report: {json_path}")
            if csv_path:
                tprint_info(f"📊 CSV Report: {csv_path}")
            tprint_info("=" * 80)

            return report_paths

        except Exception as e:
            self.logger.error(f"Failed to generate training reports: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_unified_models_training_step():
    """Register the unified models training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("unified_models_training", UnifiedModelsTrainingStep)
    tprint("✅ Unified models training step registered", "SUCCESS")


# Auto-register when module is imported
register_unified_models_training_step()
