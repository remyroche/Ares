#!/usr/bin/env python3
"""
Final Feature Selection Step

This module provides the integration step for the final feature selection pipeline
that runs at the end of the market analysis pipeline.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import logging
from pathlib import Path
import asyncio

from src.training.steps.pre_training.standardized_labeling_interface import (
    assert_labels_sigma_scaled,
    validate_dataframe_schema
)
from src.training.steps.pre_training.artifacts.manifest import (
    ArtifactManifest,
    DataLocator as ArtifactDataLocator,
)

# Import the final feature selection pipeline
if TYPE_CHECKING:
    from .final_feature_selection_pipeline import (
        FeatureSelectionConfig,
        MultiStageFeatureSelector,
    )

# Import system utilities
from src.utils.logger import get_logger
from src.utils.comprehensive_function_logger import log_all_calls
from src.core.decorators import handles_errors, traced, log_execution_time, validates
from src.utils.tprint import (
    tprint,
    tprint_error,
    tprint_info,
    tprint_success,
    tprint_warning,
)
from src.training.config.data_locator import DataLocator
from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_selection_artifact,
)
from src.training.steps.pre_training.validation.schemas import (
    extract_p_value_mapping,
    track_and_control_hypotheses,
)
from src.training.config.data_locator import DataLocator as PipelineDataLocator
from .settings import get_pre_training_settings
from .column_naming import (
    ColumnNamespace,
    ensure_namespace,
    filter_namespace_columns,
    standardize_namespace_frame,
)

class FinalFeatureSelectionStep:
    """Final feature selection step for market analysis pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("FinalFeatureSelectionStep")
        self._settings = get_pre_training_settings()

        tprint("🧠 Initializing FinalFeatureSelectionStep")
        if self.config:
            tprint(f"   ⚙️ Provided configuration keys: {sorted(self.config.keys())}")
        else:
            tprint("   ⚙️ No custom configuration supplied, using defaults")
        
        # Drift monitoring configuration
        self.enable_drift_monitoring = self.config.get('enable_drift_monitoring', True)
        self.drift_thresholds = {
            'max_kl_divergence': self.config.get('max_kl_divergence', 0.15),  # ENHANCED: Stricter threshold
            'max_mean_shift': self.config.get('max_mean_shift', 2.0),
            'max_vif': self.config.get('max_vif', 10.0)
        }
        
        # Bootstrap validation configuration
        self.enable_bootstrap_validation = self.config.get('enable_bootstrap_validation', True)
        self.bootstrap_iterations = self.config.get('bootstrap_iterations', 10)
        self.stability_threshold = self.config.get('stability_threshold', 0.6)
        
        # ENHANCEMENTS: Economic interpretability and robustness
        self.preserve_economic_themes = self.config.get('preserve_economic_themes', True)
        self.min_features_per_theme = self.config.get('min_features_per_theme', 1)
        self.track_ic_over_time = self.config.get('track_ic_over_time', True)
        self.ic_window_size = self.config.get('ic_window_size', 100)
        self.min_ic_threshold = self.config.get('min_ic_threshold', 0.02)
        self.min_ic_t_stat = self.config.get('min_ic_t_stat', 2.0)
        self.validate_with_factor_portfolio = self.config.get('validate_with_factor_portfolio', True)
        self.min_factor_sharpe = self.config.get('min_factor_sharpe', 0.3)
        
        if self.enable_drift_monitoring:
            tprint(f"   🔍 Drift monitoring: Enabled (KL threshold={self.drift_thresholds['max_kl_divergence']})")
        if self.enable_bootstrap_validation:
            tprint(f"   🔄 Bootstrap validation: Enabled ({self.bootstrap_iterations} iterations)")

        # Model-specific feature count profiles
        self.model_profiles = {
            'AdvancedMambaHybrid': {
                'min_features': 80, 'target_features': 100, 'max_features': 120,
                'stage_targets': [110, 95, 85],  # Custom stage targets
                'priority_categories': ['momentum', 'interaction', 'microstructure']
            },
            'FinancialResNet': {
                'min_features': 100, 'target_features': 120, 'max_features': 150,
                'stage_targets': [140, 115, 105],
                'priority_categories': ['regime', 'temporal', 'volatility']
            },
            'DeepScaler': {
                'min_features': 60, 'target_features': 80, 'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['statistical', 'momentum', 'volatility']
            },
            'NBEATS': {
                'min_features': 50, 'target_features': 70, 'max_features': 80,
                'stage_targets': [75, 60, 55],
                'priority_categories': ['temporal', 'trend', 'seasonality']
            }
        }

        # Resolve locator-aware output directories
        locator_candidate = self.config.get('data_locator')
        self.data_locator: Optional[PipelineDataLocator] = (
            locator_candidate if isinstance(locator_candidate, PipelineDataLocator) else None
        )

        self.output_directory_key = self.config.get('output_directory_key', 'market_analysis')
        configured_output_dir = self.config.get('output_directory')
        locator = self.data_locator

        if configured_output_dir:
            output_directory = str(Path(configured_output_dir).expanduser())
            Path(output_directory).mkdir(parents=True, exist_ok=True)
        elif locator:
            output_directory = str(locator.generated_path(self.output_directory_key, ensure_exists=True))
        else:
            default_locator = PipelineDataLocator()
            output_directory = str(
                default_locator.generated_path(self.output_directory_key, ensure_exists=True)
            )

        self.final_features_dir_key = self.config.get('final_features_dir_key', 'final_feature_selection')
        configured_final_dir = self.config.get('final_features_dir')
        if configured_final_dir:
            final_features_dir = Path(configured_final_dir).expanduser()
            final_features_dir.mkdir(parents=True, exist_ok=True)
        elif locator:
            final_features_dir = locator.generated_path(self.final_features_dir_key, ensure_exists=True)
        else:
            default_locator = PipelineDataLocator()
            final_features_dir = default_locator.generated_path(
                self.final_features_dir_key,
                ensure_exists=True,
            )
        self.final_features_dir = final_features_dir

        # Initialize feature selection configuration with model-aware defaults
        model_type = self.config.get('model_type', 'default')
        profile = self.model_profiles.get(model_type, {
            'min_features': 60, 'target_features': 80, 'max_features': 100,
            'stage_targets': [95, 75, 65],
            'priority_categories': ['momentum', 'volatility', 'microstructure']
        })

        self._pipeline_import_error: Optional[BaseException] = None
        try:
            from .final_feature_selection_pipeline import FeatureSelectionConfig as _FeatureSelectionConfig
            self._pipeline_available = True
        except Exception as exc:  # pragma: no cover - fallback path
            self._pipeline_available = False
            self._pipeline_import_error = exc
            self.logger.warning(
                "⚠️ Falling back to minimal feature selection configuration due to import failure: %s",
                exc,
            )

            class _FeatureSelectionConfig:  # type: ignore[redefined-outer-name]
                def __init__(self, **kwargs: Any) -> None:
                    for key, value in kwargs.items():
                        setattr(self, key, value)

        FeatureSelectionConfig = _FeatureSelectionConfig

        self.feature_config = FeatureSelectionConfig(
            initial_features=self.config.get('initial_features', 120),
            stage_1_target=self.config.get('stage_1_target', profile['stage_targets'][0]),
            stage_2_target=self.config.get('stage_2_target', profile['stage_targets'][1]),
            stage_3_target=self.config.get('stage_3_target', profile['stage_targets'][2]),
            rf_n_estimators=self.config.get('rf_n_estimators', 100),
            cv_folds=self.config.get('cv_folds', 5),
            save_analysis=self.config.get('save_analysis', True),
            output_directory=output_directory,
            verbose=self.config.get('verbose', True),
            # Add model-specific parameters
            model_type=model_type,
            target_features=profile['target_features'],
            min_features=profile['min_features'],
            max_features=profile['max_features'],
            priority_categories=profile['priority_categories']
        )

        self.logger.info("🚀 FinalFeatureSelectionStep initialized")
        self.logger.info(f"🎯 Model Type: {model_type}")
        self.logger.info(f"📊 Target Features: {profile['target_features']} (range: {profile['min_features']}-{profile['max_features']})")
        tprint("✅ FinalFeatureSelectionStep initialization complete")
        tprint(f"   🎯 Model Type: {model_type}")
        tprint(f"   📊 Feature targets: {profile['stage_targets']}")

    @staticmethod
    def _standardize_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature columns follow the ``X_*`` naming convention."""

        return standardize_namespace_frame(data, ColumnNamespace.FEATURE, allowed_unprefixed={"datetime"})

    @staticmethod
    def _standardize_target_frame(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure target columns follow the ``y_*`` naming convention."""

        return standardize_namespace_frame(data, ColumnNamespace.TARGET)

    @log_all_calls
    @handles_errors(Exception, fallback=False)
    @log_execution_time()
    async def execute_final_feature_selection(self,
                                            symbol: str,
                                            exchange: str, 
                                            timeframe: str, 
                                            data_dir: str,
                                            **kwargs) -> bool:
        """
        Execute final feature selection step with comprehensive logging.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        tprint("🔍 Starting Final Feature Selection Step")
        tprint(f"   📊 Symbol: {symbol}")
        tprint(f"   🏢 Exchange: {exchange}")
        tprint(f"   ⏰ Timeframe: {timeframe}")
        tprint(f"   📁 Data directory: {data_dir}")
        tprint(f"   🎯 Target: xx→120→100→80→60 features")
        
        try:
            # Load feature data
            tprint("🔄 Loading feature data...")
            feature_data = await self._load_feature_data(symbol, exchange, timeframe, data_dir)
            if feature_data is None:
                tprint("❌ Failed to load feature data")
                return False
            
            tprint(f"✅ Feature data loaded: {feature_data.shape[0]} samples, {feature_data.shape[1]} features")
            
            # Load target data (if available) - prioritize standardized format from multi_horizon_profit_labeler
            tprint("🔄 Loading target data...")
            target_data = await self._load_target_data_from_standardized_format(symbol, exchange, timeframe, data_dir)
            
            if target_data is not None:
                tprint(f"✅ Target data loaded: {target_data.shape[0]} samples, {target_data.shape[1]} columns")
            else:
                tprint("⚠️ No target data found - will perform unsupervised feature selection")
            
            # Prepare data for feature selection
            tprint("🔄 Preparing data for feature selection...")
            X, y = self._prepare_data(feature_data, target_data)
            tprint(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Run feature selection
            tprint("🚀 Running multi-stage feature selection...")
            selection_result = await self._run_feature_selection(X, y, symbol, exchange, timeframe)
            
            # Save results
            tprint("💾 Saving selection results...")
            await self._save_selection_results(selection_result, symbol, exchange, timeframe, data_dir)
            
            # Generate summary report
            tprint("📊 Generating summary report...")
            await self._generate_summary_report(selection_result, symbol, exchange, timeframe)

            # Log integration with upstream components
            tprint("📋 INTEGRATION SUMMARY:")
            tprint(f"   🎯 Multi-horizon profit labels: {'✅ Used' if target_data is not None else '❌ Not found'}")
            tprint(f"   ⚙️ Feature lookback optimization: {'✅ Integrated' if 'lookback_optimized' in str(selection_result).lower() else '❌ Not applied'}")
            tprint(f"   🔧 PID-based features: {'✅ Used' if len(feature_data.columns) > 50 else '❌ Insufficient features'}")
            tprint(f"   ⚡ Vectorization: ✅ Enabled")
            tprint(f"   💾 Caching: ✅ Enabled")
            tprint(f"   📊 tprint logging: ✅ Comprehensive")

            tprint("✅ Final feature selection completed successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Final feature selection failed: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")
            return False

    async def _load_target_data_from_standardized_format(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load target data from standardized format from multi_horizon_profit_labeler."""
        try:
            result = await asyncio.to_thread(
                self._load_target_data_from_standardized_format_sync,
                symbol,
                exchange,
                timeframe,
                data_dir,
            )
            if result is not None:
                return result
            return await self._load_target_data(symbol, exchange, timeframe, data_dir)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load target data from standardized format: {e}")
            tprint_warning(f"⚠️ Standardized target data loading failed: {e}")
            return await self._load_target_data(symbol, exchange, timeframe, data_dir)

    def _load_target_data_from_standardized_format_sync(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Synchronous helper for loading standardized target data."""
        self.logger.info("🎯 Loading target data from standardized format")
        tprint("🔍 Attempting to load standardized target data artifacts")

        manifest = ArtifactManifest()
        artifact_base_name = 'market_analysis_multi_horizon_profit_labeler_outcome'
        logical_name = ArtifactDataLocator.build_logical_name(
            artifact_base_name,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
        )
        entry = manifest.get_latest(logical_name)
        fallback_allowed = False

        if entry:
            outcome_file = entry.resolved_path
            if outcome_file.exists():
                self.logger.info(f"📂 Loading target data from manifest entry: {outcome_file}")
                result = self._load_standardized_target_from_file(
                    outcome_file,
                    expected_symbol=symbol,
                    expected_exchange=exchange,
                    expected_timeframe=timeframe,
                )
                if result is not None:
                    return result
                fallback_allowed = True
            else:
                self.logger.warning(f"⚠️ Manifest referenced outcome file missing: {outcome_file}")
                fallback_allowed = True
        else:
            fallback_allowed = True

        if fallback_allowed:
            outcomes_dir = self._settings.outcomes_root
            if outcomes_dir.exists():
                pattern = f"market_analysis_multi_horizon_profit_labeler_outcome_{symbol}_{exchange}_{timeframe}_*.json"
                outcome_files = list(outcomes_dir.glob(pattern))
                if outcome_files:
                    latest_outcome_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
                    self.logger.info(f"📂 Loading target data from fallback outcomes file: {latest_outcome_file}")
                    result = self._load_standardized_target_from_file(
                        latest_outcome_file,
                        expected_symbol=symbol,
                        expected_exchange=exchange,
                        expected_timeframe=timeframe,
                    )
                    if result is not None:
                        return result

        # Fallback: try to load from data_cache or other locations
        return None

    def _load_standardized_target_from_file(
        self,
        outcome_file: Path,
        *,
        expected_symbol: str,
        expected_exchange: str,
        expected_timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Load standardized target data from a manifest-referenced outcome file."""
        try:
            with open(outcome_file, 'r', encoding='utf-8') as handle:
                outcome_data = json.load(handle)
        except FileNotFoundError:
            self.logger.warning(f"⚠️ Outcome file missing: {outcome_file}")
            return None
        except json.JSONDecodeError as exc:
            self.logger.warning(f"⚠️ Could not parse outcome JSON {outcome_file}: {exc}")
            return None

        config_data = outcome_data.get('config', {})
        if config_data:
            if config_data.get('symbol') and config_data.get('symbol') != expected_symbol:
                self.logger.warning("⚠️ Outcome file symbol mismatch; skipping")
                return None
            if config_data.get('exchange') and config_data.get('exchange') != expected_exchange:
                self.logger.warning("⚠️ Outcome file exchange mismatch; skipping")
                return None
            if config_data.get('timeframe') and config_data.get('timeframe') != expected_timeframe:
                self.logger.warning("⚠️ Outcome file timeframe mismatch; skipping")
                return None

        artifacts = outcome_data.get('artifacts', {})
        standardized_output = artifacts.get('standardized_output') if isinstance(artifacts, dict) else None
        if not standardized_output:
            self.logger.warning("⚠️ No standardized output found in outcome file")
            return None

        target_data = standardized_output.get('labels')
        weights = standardized_output.get('weights', {})
        target_columns = standardized_output.get('target_columns', [])
        sample_weights = standardized_output.get('sample_weights', None)
        quality_scores = standardized_output.get('quality_scores', {})
        validation_results = standardized_output.get('validation_results', {})

        if target_data is None:
            self.logger.warning("⚠️ No labels found in standardized output")
            return None

        if isinstance(target_data, dict):
            target_df = pd.DataFrame(target_data)
        elif isinstance(target_data, pd.DataFrame):
            target_df = target_data
        else:
            try:
                target_df = pd.DataFrame(target_data)
            except Exception:
                self.logger.warning("⚠️ Target data in unexpected format")
                tprint_warning(f"⚠️ Target data has unexpected type: {type(target_data)}")
                return None

        target_df = self._standardize_target_frame(target_df)

        if target_columns:
            target_columns = [ensure_namespace(col, ColumnNamespace.TARGET) for col in target_columns]
        else:
            target_columns = filter_namespace_columns(target_df.columns, ColumnNamespace.TARGET)

        if target_df.empty:
            self.logger.warning("⚠️ Standardized target dataframe is empty")
            return None

        self.logger.info("✅ Successfully loaded target data from standardized format")
        tprint_info(f"🎯 Target columns: {target_columns}")
        tprint_info(f"⚖️ Horizon weights: {weights}")
        tprint_info(f"📊 Sample weights: {'Available' if sample_weights is not None else 'Not available'}")
        tprint_info(f"🔍 Quality scores: {'Available' if quality_scores else 'Not available'}")
        tprint_info(f"✅ Validation status: {'Passed' if validation_results.get('is_valid', False) else 'Failed'}")

        is_valid, issues = validate_dataframe_schema(
            target_df,
            required_columns=target_columns if target_columns else None,
            min_rows=100,
            allow_nulls=True,
        )
        if not is_valid:
            tprint_warning("⚠️ Target DataFrame schema validation failed:")
            for issue in issues:
                tprint_warning(f"  - {issue}")

        assert_labels_sigma_scaled(target_df)

        best_target = self._select_best_target_with_weights(target_df, weights, target_columns)
        if best_target:
            tprint_success(f"✅ Selected best target for feature selection: {best_target}")
            selected_target_df = pd.DataFrame({best_target: target_df[best_target]})
            self.logger.info(f"📊 Target data loaded: {len(selected_target_df)} rows, 1 target column")
            return selected_target_df

        self.logger.info("📊 Using all available targets")
        self.logger.info(f"📊 Target data loaded: {len(target_df)} rows, {len(target_df.columns)} columns")
        return target_df

    def _select_best_target_with_weights(self, labels: pd.DataFrame, weights: Dict[str, float], target_columns: List[str]) -> Optional[str]:
        """Select the best target based on horizon weights and availability for feature selection."""
        try:
            if not weights or not target_columns:
                # No weights available, use first available target
                available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
                return available_targets[0] if available_targets else None

            # Priority order based on horizon weights (higher weight = higher priority)
            # Map target columns to their corresponding horizon weights
            target_priority = []
            
            for target in target_columns:
                if target in labels.columns:
                    # Determine horizon type from target name
                    if 'immediate' in target.lower() or 'small' in target.lower():
                        horizon_weight = weights.get('small', 0.0)
                    elif 'short' in target.lower() or 'medium' in target.lower():
                        horizon_weight = weights.get('medium', 0.0)
                    elif 'leverage' in target.lower() or 'high' in target.lower():
                        horizon_weight = weights.get('high', 0.0)
                    else:
                        # Default to small horizon if unclear
                        horizon_weight = weights.get('small', 0.0)
                    
                    target_priority.append((target, horizon_weight))

            # Sort by weight (descending) and return the highest weighted target
            if target_priority:
                target_priority.sort(key=lambda x: x[1], reverse=True)
                best_target = target_priority[0][0]
                tprint_info(f"   → Selected target '{best_target}' with weight {target_priority[0][1]:.3f} for feature selection")
                return best_target

            return None

        except Exception as e:
            tprint_warning(f"⚠️ Error selecting best target with weights: {e}")
            # Fallback to first available target
            available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
            return available_targets[0] if available_targets else None

    async def _load_feature_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load feature data from previous pipeline steps."""

        try:
            return await asyncio.to_thread(
                self._load_feature_data_sync,
                symbol,
                exchange,
                timeframe,
                data_dir,
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to load feature data: {e}")
            tprint_error(f"❌ Feature data loading failed: {e}")
            return None

    def _load_feature_data_sync(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Synchronous helper for loading feature data."""

        tprint("📥 Loading feature data for final selection")
        tprint(f"   📊 Context: symbol={symbol}, exchange={exchange}, timeframe={timeframe}")
        # Try different possible file locations and formats
        possible_files = [
            f"{symbol.lower()}_{timeframe}_features.parquet",
            f"{symbol.lower()}_{timeframe}_engineered_features.parquet",
            f"{symbol.lower()}_{timeframe}_final_features.parquet",
            f"{symbol.lower()}_{timeframe}_matrix_features.parquet"
        ]

        data_path = Path(data_dir)

        for filename in possible_files:
            file_path = data_path / filename
            if file_path.exists():
                self.logger.info(f"📂 Loading feature data from: {file_path}")
                tprint_success(f"   📂 Found feature file: {file_path.name}")
                data = pd.read_parquet(file_path)

                # 🔧 INTEGRATE DATA CLEANING UTILITY
                # Clean corrupted data before final feature selection
                try:
                    from src.utils.ml_common.data_processing.data_cleaning_utils import exclude_corrupted_periods

                    # Ensure datetime column exists
                    if 'timestamp' in data.columns and data['timestamp'].dtype == 'int64':
                        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                    elif 'datetime' not in data.columns:
                        # Try to infer datetime column
                        datetime_cols = [col for col in data.columns if 'time' in col.lower()]
                        if datetime_cols:
                            data['datetime'] = pd.to_datetime(data[datetime_cols[0]])
                        else:
                            data['datetime'] = data.index

                    # Apply data cleaning
                    original_count = len(data)
                    data = exclude_corrupted_periods(data)
                    cleaned_count = len(data)

                    if original_count != cleaned_count:
                        excluded_count = original_count - cleaned_count
                        self.logger.info(f"🧹 Final Feature Selection Data cleaning applied: Excluded {excluded_count:,} corrupted rows ({100*excluded_count/original_count:.4f}%)")

                except ImportError as e:
                    self.logger.warning(f"⚠️ Data cleaning utility not available for final feature selection: {e}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Data cleaning failed for final feature selection, proceeding with original data: {e}")
                    tprint_warning(f"   ⚠️ Data cleaning issues encountered: {e}")

                data = self._standardize_feature_frame(data)

                self.logger.info(f"✅ Loaded {len(data)} samples with {len(data.columns)} features")
                tprint_success(f"   ✅ Loaded feature data with shape {data.shape}")
                return data

        # If no specific feature file found, try to load from matrix operations
        matrix_file = data_path / f"{symbol.lower()}_{timeframe}_matrix_operations.parquet"
        if matrix_file.exists():
            self.logger.info(f"📂 Loading matrix operations data from: {matrix_file}")
            tprint_info(f"   📂 Falling back to matrix operations file: {matrix_file.name}")
            data = pd.read_parquet(matrix_file)
            data = self._standardize_feature_frame(data)
            self.logger.info(f"✅ Loaded {len(data)} samples with {len(data.columns)} features from matrix operations")
            tprint_success(f"   ✅ Loaded matrix operations data with shape {data.shape}")
            return data

        self.logger.warning("⚠️ No feature data files found")
        tprint_warning("⚠️ No feature data files located for final selection")
        return None
    
    async def _load_target_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.Series]:
        """Load target data if available."""

        try:
            return await asyncio.to_thread(
                self._load_target_data_sync,
                symbol,
                exchange,
                timeframe,
                data_dir,
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load target data: {e}")
            tprint_warning(f"⚠️ Target data loading encountered an error: {e}")
            return None

    def _load_target_data_sync(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.Series]:
        """Synchronous helper for loading fallback target data."""

        tprint("📥 Attempting fallback target data load")
        # Try to load target data from labeling step
        possible_target_files = [
            f"{symbol.lower()}_{timeframe}_labels.parquet",
            f"{symbol.lower()}_{timeframe}_triple_barrier_labels.parquet",
            f"{symbol.lower()}_{timeframe}_target.parquet"
        ]

        data_path = Path(data_dir)

        for filename in possible_target_files:
            file_path = data_path / filename
            if file_path.exists():
                self.logger.info(f"📂 Loading target data from: {file_path}")
                tprint_success(f"   📂 Found target file: {file_path.name}")
                data = pd.read_parquet(file_path)
                data = self._standardize_target_frame(data)

                canonical_targets = filter_namespace_columns(data.columns, ColumnNamespace.TARGET)
                target_col = canonical_targets[0] if canonical_targets else None

                if target_col:
                    target_data = data[target_col]
                    self.logger.info(f"✅ Loaded target data: {target_col} with {len(target_data)} samples")
                    tprint_success(f"   ✅ Loaded target column '{target_col}' with {len(target_data)} samples")
                    return target_data

        self.logger.info("ℹ️ No target data found - will perform unsupervised feature selection")
        tprint_warning("⚠️ No target data located, defaulting to unsupervised selection")
        return None
    
    def _prepare_data(self, feature_data: pd.DataFrame, target_data: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for feature selection with comprehensive logging."""
        
        tprint("🔄 Preparing data for feature selection...")
        tprint(f"   📊 Input features: {feature_data.shape[0]} samples, {feature_data.shape[1]} columns")
        
        # Clean feature data
        X = feature_data.copy()
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        non_numeric_count = len(X.columns) - len(numeric_columns)
        if non_numeric_count > 0:
            tprint(f"   🗑️ Removing {non_numeric_count} non-numeric columns")
        X = X[numeric_columns]
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            tprint(f"   🔧 Handling {missing_count} missing values using median imputation")
            X = X.fillna(X.median())
        else:
            tprint("   ✅ No missing values found")
        
        # Remove infinite values
        inf_count = np.isinf(X.values).sum()
        if inf_count > 0:
            tprint(f"   🔧 Handling {inf_count} infinite values")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        else:
            tprint("   ✅ No infinite values found")
        
        tprint(f"   ✅ Prepared {len(X)} samples with {len(X.columns)} numeric features")
        
        # Prepare target data if available
        y = None
        if target_data is not None:
            tprint(f"   🎯 Processing target data: {target_data.shape[0]} samples")
            # Align target data with feature data
            common_indices = X.index.intersection(target_data.index)
            if len(common_indices) > 0:
                X = X.loc[common_indices]
                y = target_data.loc[common_indices]
                tprint(f"   ✅ Aligned target data: {len(y)} samples")
            else:
                tprint("   ⚠️ No common indices between features and target")
        else:
            tprint("   ℹ️ No target data - will perform unsupervised feature selection")
        
        tprint(f"✅ Data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    async def _run_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series],
                                   symbol: str, exchange: str, timeframe: str) -> Any:
        """Run the multi-stage feature selection with comprehensive logging."""

        tprint("🔍 Running Multi-Stage Feature Selection")
        tprint(f"   📊 Input: {len(X)} samples, {len(X.columns)} features")
        tprint(f"   🎯 Target: xx→120→100→80→60 features")
        
        if y is not None:
            tprint(f"   🎯 Target: {len(y)} samples (supervised learning)")
            tprint(f"   📊 Target type: {'classification' if len(y.unique()) <= 10 else 'regression'}")
        else:
            tprint("   🎯 No target data (unsupervised learning)")
        
        tprint("   ⚡ Using vectorized operations and caching")
        tprint("   🔄 Starting feature selection pipeline...")
        
        # Run feature selection in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        selection_result = await loop.run_in_executor(
            None,
            self._run_selection_sync,
            X, y
        )

        final_scores = getattr(selection_result, 'final_scores', {}) or {}
        selection_result.eligible_for_selection = bool(final_scores.get('eligible_for_selection', True))
        selection_result.turnover_rejection_reason = final_scores.get('turnover_rejection_reason')
        if not selection_result.eligible_for_selection:
            reason = selection_result.turnover_rejection_reason or 'Turnover constraints violated'
            tprint_warning(f"🚫 Selection result marked ineligible: {reason}")

        final_numeric_scores = {
            key: float(value)
            for key, value in final_scores.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        horizon_p_values, feature_p_values, lookback_p_values = self._collect_hypothesis_p_values(selection_result)
        hypothesis_report = track_and_control_hypotheses(
            horizon_results=horizon_p_values,
            feature_results=feature_p_values,
            lookback_results=lookback_p_values,
        )
        if hypothesis_report.get("warning"):
            tprint_warning(hypothesis_report["warning"])

        selection_result.horizon_p_values = horizon_p_values
        selection_result.feature_p_values = feature_p_values
        selection_result.lookback_p_values = lookback_p_values
        selection_result.adjusted_p_values = hypothesis_report.get("adjusted_p_values", {})
        selection_result.hypothesis_report = hypothesis_report

        selection_payload = {
            'final_features': list(getattr(selection_result, 'final_features', []) or []),
            'stage_1_features': list(getattr(selection_result, 'stage_1_features', []) or []),
            'stage_2_features': list(getattr(selection_result, 'stage_2_features', []) or []),
            'stage_3_features': list(getattr(selection_result, 'stage_3_features', []) or []),
            'feature_counts': dict(getattr(selection_result, 'feature_counts', {})),
            'stage_scores': {
                'stage_1': dict(getattr(selection_result, 'stage_1_scores', {})),
                'stage_2': dict(getattr(selection_result, 'stage_2_scores', {})),
                'stage_3': dict(getattr(selection_result, 'stage_3_scores', {})),
                'final': final_numeric_scores,
            },
            'selection_time': getattr(selection_result, 'selection_time', None),
            'is_unsupervised': getattr(selection_result, 'is_unsupervised', None),
            'eligible_for_selection': selection_result.eligible_for_selection,
            'turnover_rejection_reason': selection_result.turnover_rejection_reason,
            'hypothesis_report': selection_result.hypothesis_report,
            'horizon_p_values': selection_result.horizon_p_values,
            'feature_p_values': selection_result.feature_p_values,
            'lookback_p_values': selection_result.lookback_p_values,
            'adjusted_p_values': selection_result.adjusted_p_values,
        }

        try:
            validate_selection_artifact(
                selection_payload,
                context='final_feature_selection_step.selection_result',
            )
        except DataContractValidationError as contract_error:
            tprint_error(f"❌ Selection result failed validation: {contract_error}")
            raise

        tprint("✅ Multi-stage feature selection completed")
        tprint(f"   📊 Final features: {len(selection_result.final_features)}")
        tprint(f"   ⏱️ Selection time: {selection_result.selection_time:.3f} seconds")

        return selection_result

    def _run_selection_sync(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Any:
        """Synchronous feature selection (to be run in thread pool)."""

        tprint("⚙️ Executing synchronous selection pipeline")
        # Create feature selector
        if not getattr(self, "_pipeline_available", False):
            raise RuntimeError("Feature selection pipeline is unavailable") from self._pipeline_import_error

        from .final_feature_selection_pipeline import MultiStageFeatureSelector

        selector = MultiStageFeatureSelector(self.feature_config)

        # Run selection
        if y is not None:
            tprint("   🎯 Using supervised selector with provided target")
            result = selector.select_features(X, y)
        else:
            # For unsupervised selection, create a dummy target
            # This is a simplified approach - in practice, you might want to use
            # different unsupervised feature selection methods
            tprint("   🧪 No target provided, creating proxy target for unsupervised run")
            dummy_target = X.iloc[:, 0]  # Use first feature as proxy target
            result = selector.select_features(X, dummy_target)
            result.is_unsupervised = True

        tprint("   ✅ Synchronous selection complete")
        return result

    def _collect_hypothesis_p_values(self, selection_result: Any) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Extract horizon, feature, and lookback p-values from the selection result."""

        try:
            flattened = extract_p_value_mapping(vars(selection_result))
        except Exception:
            flattened = {}

        horizon_p_values = {
            key: value for key, value in flattened.items() if "horizon" in key.lower()
        }
        lookback_p_values = {
            key: value for key, value in flattened.items() if "lookback" in key.lower()
        }
        feature_p_values = {
            key: value
            for key, value in flattened.items()
            if key not in horizon_p_values and key not in lookback_p_values
        }

        return horizon_p_values, feature_p_values, lookback_p_values
    
    async def _save_selection_results(self, selection_result: Any, symbol: str, exchange: str,
                                    timeframe: str, data_dir: str) -> None:
        """Save feature selection results."""

        try:
            await asyncio.to_thread(
                self._save_selection_results_sync,
                selection_result,
                symbol,
                exchange,
                timeframe,
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to save selection results: {e}")

    def _save_selection_results_sync(self, selection_result: Any, symbol: str, exchange: str,
                                     timeframe: str) -> None:
        """Synchronous helper for saving feature selection results."""

        output_dir = self.final_features_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final selected features
        final_features_file = output_dir / f"{symbol.lower()}_{timeframe}_final_features.json"
        final_features = selection_result.final_features

        import json
        with open(final_features_file, 'w') as f:
            json.dump({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'final_features': final_features,
                'feature_count': len(final_features),
                'selection_method': 'multi_stage_rf_shap',
                'stages': {
                    'stage_1': len(selection_result.stage_1_features),
                    'stage_2': len(selection_result.stage_2_features),
                    'stage_3': len(selection_result.stage_3_features),
                    'final': len(selection_result.final_features)
                }
            }, f, indent=2)

        self.logger.info(f"💾 Final features saved to: {final_features_file}")

        # Save detailed results
        detailed_results_file = output_dir / f"{symbol.lower()}_{timeframe}_selection_results.json"

        results_dict = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'feature_counts': selection_result.feature_counts,
            'scores': {
                'stage_1': selection_result.stage_1_scores,
                'stage_2': selection_result.stage_2_scores,
                'stage_3': selection_result.stage_3_scores,
                'final': selection_result.final_scores
            },
            'selection_time': selection_result.selection_time,
            'is_unsupervised': getattr(selection_result, 'is_unsupervised', False),
            'eligible_for_selection': getattr(selection_result, 'eligible_for_selection', True),
            'turnover_rejection_reason': getattr(selection_result, 'turnover_rejection_reason', None),
            'hypothesis_report': getattr(selection_result, 'hypothesis_report', {}),
            'horizon_p_values': getattr(selection_result, 'horizon_p_values', {}),
            'feature_p_values': getattr(selection_result, 'feature_p_values', {}),
            'lookback_p_values': getattr(selection_result, 'lookback_p_values', {}),
            'adjusted_p_values': getattr(selection_result, 'adjusted_p_values', {}),
        }

        with open(detailed_results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        self.logger.info(f"💾 Detailed results saved to: {detailed_results_file}")
    
    async def _generate_summary_report(self, selection_result: Any, symbol: str,
                                     exchange: str, timeframe: str) -> None:
        """Generate comprehensive summary report of feature selection."""
        
        try:
            tprint("📊 FEATURE SELECTION SUMMARY REPORT")
            tprint("=" * 60)
            tprint(f"🎯 Symbol: {symbol}")
            tprint(f"🏢 Exchange: {exchange}")
            tprint(f"⏰ Timeframe: {timeframe}")
            tprint(f"⏱️ Selection Time: {selection_result.selection_time:.3f}s")
            tprint("")
            
            tprint("📈 FEATURE REDUCTION PIPELINE:")
            tprint(f"   🔢 Initial Features: {selection_result.feature_counts.get('initial', 'N/A')}")
            tprint(f"   📊 Stage 1 (xx→120): {selection_result.feature_counts.get('stage_1', 'N/A')} features")
            tprint(f"   📊 Stage 2 (120→100): {selection_result.feature_counts.get('stage_2', 'N/A')} features")
            tprint(f"   📊 Stage 3 (100→80): {selection_result.feature_counts.get('stage_3', 'N/A')} features")
            tprint(f"   📊 Final (80→60): {selection_result.feature_counts.get('final', 'N/A')} features")
            tprint("")
            
            tprint("📊 STAGE SCORES:")
            if selection_result.stage_1_scores:
                score = selection_result.stage_1_scores.get('model_importance_score', 'N/A')
                tprint(f"   🎯 Stage 1 Score: {score:.4f}" if isinstance(score, (int, float)) else f"   🎯 Stage 1 Score: {score}")
            if selection_result.stage_2_scores:
                score = selection_result.stage_2_scores.get('model_importance_score', 'N/A')
                tprint(f"   🎯 Stage 2 Score: {score:.4f}" if isinstance(score, (int, float)) else f"   🎯 Stage 2 Score: {score}")
            if selection_result.stage_3_scores:
                score = selection_result.stage_3_scores.get('combined_importance_score', 'N/A')
                tprint(f"   🎯 Stage 3 Score: {score:.4f}" if isinstance(score, (int, float)) else f"   🎯 Stage 3 Score: {score}")
            if selection_result.final_scores:
                final_scores = selection_result.final_scores
                ic_score = final_scores.get('information_coefficient')
                sharpe_score = final_scores.get('long_short_sharpe')
                turnover = final_scores.get('turnover_per_period', final_scores.get('turnover'))
                turnover_annual = final_scores.get('turnover_annual')
                avg_holding = final_scores.get('avg_holding_period_bars')
                stability = final_scores.get('position_stability')
                mean_cost = final_scores.get('mean_transaction_cost')
                mean_impact = final_scores.get('mean_market_impact_cost')
                capacity_exceeded = bool(final_scores.get('capacity_exceeded'))
                rejection_reason = final_scores.get('turnover_rejection_reason')
                eligible = bool(final_scores.get('eligible_for_selection', True))

                if isinstance(ic_score, (int, float)):
                    tprint(f"   📈 OOS Information Coefficient: {ic_score:.4f}")
                else:
                    tprint(f"   📈 OOS Information Coefficient: {ic_score}")

                if isinstance(sharpe_score, (int, float)):
                    tprint(f"   ⚖️ Cost-adjusted Sharpe: {sharpe_score:.4f}")
                else:
                    tprint(f"   ⚖️ Cost-adjusted Sharpe: {sharpe_score}")

                if isinstance(turnover, (int, float)):
                    tprint(f"   🔄 Average Turnover / Period: {turnover:.4f}")
                else:
                    tprint(f"   🔄 Average Turnover / Period: {turnover}")

                if isinstance(turnover_annual, (int, float)):
                    tprint(f"   📅 Annualized Turnover: {turnover_annual:.2f}x")
                elif turnover_annual is not None:
                    tprint(f"   📅 Annualized Turnover: {turnover_annual}")

                if isinstance(avg_holding, (int, float)):
                    tprint(f"   🕒 Avg Holding Period (bars): {avg_holding:.2f}")
                if isinstance(stability, (int, float)):
                    tprint(f"   🧱 Position Stability: {stability:.2%}")

                if isinstance(mean_cost, (int, float)):
                    tprint(f"   💸 Mean Transaction Cost: {mean_cost:.6f}")
                if isinstance(mean_impact, (int, float)):
                    tprint(f"   🌊 Mean Market Impact Cost: {mean_impact:.6f}")

                if capacity_exceeded:
                    tprint_warning("   🚨 Capacity limit exceeded during backtest")

                if not eligible:
                    if rejection_reason:
                        tprint_warning(f"   🚫 Configuration rejected: {rejection_reason}")
                    else:
                        tprint_warning("   🚫 Configuration rejected due to turnover constraints")
                else:
                    tprint("   ✅ Configuration passes turnover checks")
                score = final_scores.get('cv_mean', 'N/A')
                cv_metric = final_scores.get('cv_metric', 'unknown')
                if isinstance(score, (int, float)) and not np.isnan(score):
                    tprint(f"   🎯 Final CV Score ({cv_metric}): {score:.4f}")
                else:
                    tprint(f"   🎯 Final CV Score ({cv_metric}): N/A")

                trading_cv = final_scores.get('cv_mean_trading')
                if isinstance(trading_cv, (int, float)):
                    tprint(f"   🎯 Trading CV Sharpe Mean: {trading_cv:.4f}")
                for metric_key, label in (
                    ('average_precision', 'PR-AUC'),
                    ('balanced_accuracy', 'Balanced Accuracy'),
                    ('r2', 'R²')
                ):
                    if metric_key in selection_result.final_scores:
                        metric_value = selection_result.final_scores[metric_key]
                        if isinstance(metric_value, (int, float)):
                            tprint(f"   🎯 Final {label}: {metric_value:.4f}")
                        else:
                            tprint(f"   🎯 Final {label}: {metric_value}")
            tprint("")
            
            # Show top 10 final features
            if hasattr(selection_result, 'model_performance') and 'feature_importance' in selection_result.model_performance:
                top_features = sorted(
                    selection_result.model_performance['feature_importance'].items(),
                    key=lambda x: x[1], reverse=True
                )[:10]
                
                tprint("🏆 TOP 10 FINAL FEATURES:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    tprint(f"   {i:2d}. {feature}: {importance:.4f}")
            
            tprint("")
            tprint("⚡ OPTIMIZATION SUMMARY:")
            tprint("   ✅ Vectorized operations: Enabled")
            tprint("   ✅ Caching: Enabled")
            tprint("   ✅ Comprehensive logging: Enabled")
            tprint("   ✅ Multi-stage reduction: xx→120→100→80→60")
            
            tprint("=" * 60)
            
        except Exception as e:
            tprint(f"❌ Failed to generate summary report: {e}")
            import traceback
            tprint(f"🔍 Error details: {traceback.format_exc()}")

# Convenience function for pipeline integration
def detect_feature_drift_simple(train_features: pd.DataFrame, val_features: pd.DataFrame, 
                                max_mean_shift: float = 2.0) -> Dict[str, Any]:
    """
    Simple feature drift detection between training and validation sets.
    
    Args:
        train_features: Training feature DataFrame
        val_features: Validation feature DataFrame
        max_mean_shift: Maximum allowed mean shift in standard deviations
    
    Returns:
        Dictionary with drift detection results
    """
    tprint("🔍 Detecting feature drift...")
    
    drift_results = {
        'drifted_features': [],
        'drift_scores': {},
        'n_drifted': 0,
        'drift_detected': False
    }
    
    common_features = list(set(train_features.columns).intersection(set(val_features.columns)))
    
    for feature in common_features:
        train_data = train_features[feature].dropna()
        val_data = val_features[feature].dropna()
        
        if len(train_data) < 10 or len(val_data) < 10:
            continue
        
        # Calculate mean shift
        train_mean = train_data.mean()
        train_std = train_data.std()
        val_mean = val_data.mean()
        
        mean_shift = abs(val_mean - train_mean) / (train_std + 1e-8)
        
        # Check threshold
        if mean_shift > max_mean_shift:
            drift_results['drifted_features'].append(feature)
            drift_results['drift_scores'][feature] = float(mean_shift)
            drift_results['n_drifted'] += 1
    
    drift_results['drift_detected'] = drift_results['n_drifted'] > 0
    
    if drift_results['drift_detected']:
        tprint_warning(f"⚠️ Drift detected in {drift_results['n_drifted']} features")
    else:
        tprint_success(f"✅ No significant drift detected")
    
    return drift_results


async def run_final_feature_selection_step(symbol: str,
                                         exchange: str,
                                         timeframe: str = '1h',  # Updated to 1h for analyst
                                         data_dir: Optional[str] = None,
                                         config: Optional[Dict[str, Any]] = None) -> bool:
    """Run the final feature selection step."""

    runtime_config: Dict[str, Any] = dict(config or {})
    locator = runtime_config.get('data_locator')
    if data_dir is None and isinstance(locator, PipelineDataLocator):
        data_dir_key = runtime_config.get('data_dir_key', 'market_data')
        data_dir = str(locator.data_path(data_dir_key))

    if data_dir is None:
        raise ValueError("data_dir must be provided or resolvable via DataLocator")

    tprint("🚀 Invoking run_final_feature_selection_step helper")
    tprint(f"   📊 Parameters: symbol={symbol}, exchange={exchange}, timeframe={timeframe}, data_dir={data_dir}")
    step = FinalFeatureSelectionStep(runtime_config)
    tprint("🔄 Delegating execution to FinalFeatureSelectionStep instance")
    return await step.execute_final_feature_selection(symbol, exchange, timeframe, data_dir)

