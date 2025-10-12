#!/usr/bin/env python3
"""
Final Feature Selection Step

This module provides the integration step for the final feature selection pipeline
that runs at the end of the market analysis pipeline.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Mapping
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
    tprint_debug,
)

# Import consolidated VectorBT utilities
from .utils.vectorbt_utils import (
    create_vectorbt_tools, VectorBTConfig, get_vectorbt_performance_stats,
    VECTORBT_UTILS_AVAILABLE
)
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

        # Model-specific feature count profiles - Updated for new pipeline
        self.model_profiles = {
            'AdvancedMambaHybrid': {
                'min_features': 50, 'target_features': 60, 'max_features': 80,
                'stage_targets': [60],  # Updated for mRMR/Ensemble/RFE pipeline (variable start)
                'priority_categories': ['momentum', 'interaction', 'microstructure']
            },
            'FinancialResNet': {
                'min_features': 60, 'target_features': 80, 'max_features': 100,
                'stage_targets': [80],  # Updated for mRMR/Ensemble/RFE pipeline
                'priority_categories': ['regime', 'temporal', 'volatility']
            },
            'DeepScaler': {
                'min_features': 40, 'target_features': 60, 'max_features': 80,
                'stage_targets': [60],  # Updated for mRMR/Ensemble/RFE pipeline
                'priority_categories': ['statistical', 'momentum', 'volatility']
            },
            'NBEATS': {
                'min_features': 30, 'target_features': 50, 'max_features': 70,
                'stage_targets': [50],  # Updated for mRMR/Ensemble/RFE pipeline
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
            initial_features=self.config.get('initial_features', None),  # Truly variable starting point
            stage_1_target=self.config.get('stage_1_target', profile['stage_targets'][0]),
            stage_2_target=self.config.get('stage_2_target', profile.get('stage_targets', [60])[-1]),
            stage_3_target=self.config.get('stage_3_target', profile.get('stage_targets', [60])[-1]),
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

        # Initialize VectorBT optimization tools if available
        if VECTORBT_UTILS_AVAILABLE:
            tprint("🚀 Initializing VectorBT optimization tools for final feature selection")
            vectorbt_config = VectorBTConfig(
                enable_gpu=False,  # Conservative for feature selection
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000
            )
            
            vectorbt_tools = create_vectorbt_tools(vectorbt_config)
            self.vectorbt_optimizer = vectorbt_tools['optimizer']
            self.vectorization_manager = vectorbt_tools['manager']
            self.vectorbt_enabled = vectorbt_tools['available']
            
            if self.vectorbt_enabled:
                tprint("✅ VectorBT optimization tools initialized for final feature selection")
            else:
                tprint_warning("⚠️ VectorBT optimization tools not available")
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self.vectorbt_enabled = False
            tprint("⚠️ VectorBT optimization disabled or not available")

        self.logger.info("🚀 FinalFeatureSelectionStep initialized")
        self.logger.info(f"🎯 Model Type: {model_type}")
        self.logger.info(f"📊 Target Features: {profile['target_features']} (range: {profile['min_features']}-{profile['max_features']})")
        tprint("✅ FinalFeatureSelectionStep initialization complete")
        tprint(f"   🎯 Model Type: {model_type}")
        tprint(f"   📊 Feature targets: {profile['stage_targets']}")
        tprint(f"   ⚡ VectorBT optimization: {'Enabled' if self.vectorbt_enabled else 'Disabled'}")

    @staticmethod
    def _standardize_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature columns follow the ``X_*`` naming convention."""
        tprint_debug("🔧 Standardizing feature frame columns")
        return standardize_namespace_frame(data, ColumnNamespace.FEATURE, allowed_unprefixed={"datetime"})

    @staticmethod
    def _standardize_target_frame(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure target columns follow the ``y_*`` naming convention."""
        tprint_debug("🔧 Standardizing target frame columns")
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
        tprint(f"   🎯 Target: Variable→60 pipeline (mRMR/Ensemble/RFE)")
        
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
            
            # Prepare data for feature selection with VectorBT optimization
            if self.vectorbt_enabled:
                tprint("🔄 Preparing data for feature selection with VectorBT optimization...")
                X, y = self._vectorbt_optimized_data_preparation(feature_data, target_data)
            else:
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
            tprint(f"   ⚡ VectorBT optimization: {'✅ Enabled' if self.vectorbt_enabled else '❌ Disabled'}")
            tprint(f"   💾 Caching: ✅ Enabled")
            tprint(f"   📊 tprint logging: ✅ Comprehensive")
            
            # Log comprehensive VectorBT performance statistics if available
            if self.vectorbt_enabled:
                try:
                    vectorbt_stats = self._get_enhanced_vectorbt_performance_stats()
                    tprint("📊 ENHANCED VECTORBT PERFORMANCE STATISTICS:")
                    tprint(f"   🔢 Total operations: {vectorbt_stats.get('total_operations', 0)}")
                    tprint(f"   ⚡ VectorBT rolling operations: {vectorbt_stats.get('vectorbt_rolling_operations', 0)}")
                    tprint(f"   🖥️ GPU operations: {vectorbt_stats.get('gpu_operations', 0)}")
                    tprint(f"   📦 Chunk operations: {vectorbt_stats.get('chunk_operations', 0)}")
                    tprint(f"   🧠 Memory optimizations: {vectorbt_stats.get('memory_optimizations', 0)}")
                    tprint(f"   ⏱️ Average operation time: {vectorbt_stats.get('avg_time_per_operation', 0):.4f}s")
                    tprint(f"   📈 VectorBT usage rate: {vectorbt_stats.get('vectorbt_usage_rate', 0):.2%}")
                    tprint(f"   🎯 Average speedup: {vectorbt_stats.get('average_speedup', 0):.2f}x")
                    tprint(f"   💾 Total computation time: {vectorbt_stats.get('total_computation_time', 0):.2f}s")
                    
                    # Log strategy usage if available
                    strategy_usage = vectorbt_stats.get('strategy_usage', {})
                    if strategy_usage:
                        tprint("   🎛️ Strategy usage:")
                        for strategy, count in strategy_usage.items():
                            if count > 0:
                                tprint(f"      - {strategy}: {count} operations")
                                
                except Exception as e:
                    tprint_warning(f"⚠️ Could not retrieve enhanced VectorBT performance stats: {e}")

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
        tprint("🔍 Loading target data from standardized format")
        tprint_debug(f"   📊 Context: symbol={symbol}, exchange={exchange}, timeframe={timeframe}")
        
        try:
            manifest = ArtifactManifest()
            tprint_debug("   📦 Artifact manifest initialized")
            
            # Try multiple artifact base names to support both analyst and tactician labels
            possible_base_names = [
                'pre_training_tactician_entry_labeler_outcome',      # Tactician labels (entry timing)
                'pre_training_analyst_profit_labeler_outcome',       # Analyst labels (profit targets)
                'market_analysis_multi_horizon_profit_labeler_outcome',  # Legacy format
            ]
            
            tprint_debug(f"   🔍 Checking {len(possible_base_names)} possible artifact sources")
            
            entry = None
            artifact_base_name = None
            
            for i, base_name in enumerate(possible_base_names):
                tprint_debug(f"   📂 Checking source {i+1}/{len(possible_base_names)}: {base_name}")
                try:
                    logical_name = ArtifactDataLocator.build_logical_name(
                        base_name,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                    )
                    entry = manifest.get_latest(logical_name)
                    if entry and entry.resolved_path.exists():
                        artifact_base_name = base_name
                        tprint_success(f"   ✅ Found labels from: {base_name}")
                        tprint_debug(f"   📁 File path: {entry.resolved_path}")
                        break
                    else:
                        tprint_debug(f"   ⚠️ No valid entry found for {base_name}")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Error checking {base_name}: {e}")
                    continue
            
            if entry and artifact_base_name:
                tprint_debug(f"   📂 Loading from manifest entry: {entry.resolved_path}")
                try:
                    result = self._load_standardized_target_from_file(
                        entry.resolved_path,
                        expected_symbol=symbol,
                        expected_exchange=exchange,
                        expected_timeframe=timeframe,
                    )
                    if result is not None:
                        tprint_success(f"   ✅ Successfully loaded target data: {result.shape}")
                        return result
                    else:
                        tprint_warning(f"   ⚠️ Manifest file loaded but returned None")
                except Exception as e:
                    tprint_error(f"   ❌ Error loading from manifest: {e}")
            
            # Fallback to outcomes directory
            tprint_debug("   🔄 Attempting fallback to outcomes directory")
            try:
                outcomes_dir = self._settings.outcomes_root
                if outcomes_dir.exists():
                    pattern = f"market_analysis_multi_horizon_profit_labeler_outcome_{symbol}_{exchange}_{timeframe}_*.json"
                    outcome_files = list(outcomes_dir.glob(pattern))
                    tprint_debug(f"   📂 Found {len(outcome_files)} fallback files matching pattern")
                    
                    if outcome_files:
                        latest_outcome_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
                        tprint_debug(f"   📂 Using latest file: {latest_outcome_file}")
                        result = self._load_standardized_target_from_file(
                            latest_outcome_file,
                            expected_symbol=symbol,
                            expected_exchange=exchange,
                            expected_timeframe=timeframe,
                        )
                        if result is not None:
                            tprint_success(f"   ✅ Successfully loaded from fallback file: {result.shape}")
                            return result
                        else:
                            tprint_warning(f"   ⚠️ Fallback file loaded but returned None")
                    else:
                        tprint_warning(f"   ⚠️ No fallback files found matching pattern: {pattern}")
                else:
                    tprint_warning(f"   ⚠️ Outcomes directory does not exist: {outcomes_dir}")
            except Exception as e:
                tprint_error(f"   ❌ Error in fallback loading: {e}")

            # Final fallback
            tprint_warning(f"⚠️ No standardized target data found for {symbol}/{exchange}/{timeframe}")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Critical error in target data loading: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
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
        tprint_debug(f"📂 Loading standardized target from file: {outcome_file}")
        
        try:
            # Load and parse JSON file
            tprint_debug("   📖 Reading outcome file...")
            with open(outcome_file, 'r', encoding='utf-8') as handle:
                outcome_data = json.load(handle)
            tprint_debug("   ✅ JSON file loaded successfully")
            
        except FileNotFoundError:
            tprint_error(f"❌ Outcome file not found: {outcome_file}")
            return None
        except json.JSONDecodeError as exc:
            tprint_error(f"❌ Invalid JSON in outcome file {outcome_file}: {exc}")
            return None
        except Exception as e:
            tprint_error(f"❌ Unexpected error reading file: {e}")
            return None

        # Validate configuration data
        tprint_debug("   🔍 Validating configuration data...")
        config_data = outcome_data.get('config', {})
        if config_data:
            symbol_match = config_data.get('symbol') == expected_symbol if config_data.get('symbol') else True
            exchange_match = config_data.get('exchange') == expected_exchange if config_data.get('exchange') else True
            timeframe_match = config_data.get('timeframe') == expected_timeframe if config_data.get('timeframe') else True
            
            if not symbol_match:
                tprint_warning(f"⚠️ Symbol mismatch: expected {expected_symbol}, got {config_data.get('symbol')}")
                return None
            if not exchange_match:
                tprint_warning(f"⚠️ Exchange mismatch: expected {expected_exchange}, got {config_data.get('exchange')}")
                return None
            if not timeframe_match:
                tprint_warning(f"⚠️ Timeframe mismatch: expected {expected_timeframe}, got {config_data.get('timeframe')}")
                return None
            tprint_debug("   ✅ Configuration validation passed")

        # Extract standardized output
        tprint_debug("   📦 Extracting standardized output...")
        artifacts = outcome_data.get('artifacts', {})
        standardized_output = artifacts.get('standardized_output') if isinstance(artifacts, dict) else None
        if not standardized_output:
            tprint_warning(f"⚠️ No standardized output found in outcome file: {outcome_file}")
            return None

        # Extract target data and metadata
        target_data = standardized_output.get('labels')
        weights = standardized_output.get('weights', {})
        target_columns = standardized_output.get('target_columns', [])
        sample_weights = standardized_output.get('sample_weights', None)
        quality_scores = standardized_output.get('quality_scores', {})
        validation_results = standardized_output.get('validation_results', {})

        if target_data is None:
            tprint_warning(f"⚠️ No labels found in standardized output from {outcome_file}")
            return None

        # Convert to DataFrame
        tprint_debug("   🔄 Converting target data to DataFrame...")
        try:
            if isinstance(target_data, dict):
                target_df = pd.DataFrame(target_data)
            elif isinstance(target_data, pd.DataFrame):
                target_df = target_data
            else:
                target_df = pd.DataFrame(target_data)
            tprint_debug(f"   ✅ DataFrame created: {target_df.shape}")
        except Exception as e:
            tprint_error(f"❌ Failed to convert target data to DataFrame: {e}")
            return None

        # Standardize target frame
        tprint_debug("   🔧 Standardizing target frame...")
        target_df = self._standardize_target_frame(target_df)

        # Process target columns
        if target_columns:
            target_columns = [ensure_namespace(col, ColumnNamespace.TARGET) for col in target_columns]
        else:
            target_columns = filter_namespace_columns(target_df.columns, ColumnNamespace.TARGET)

        if target_df.empty:
            tprint_error("❌ Standardized target DataFrame is empty - no data to use for feature selection")
            return None

        # Validate DataFrame schema
        tprint_debug("   🔍 Validating DataFrame schema...")
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

        # Assert sigma scaling
        try:
            assert_labels_sigma_scaled(target_df)
            tprint_debug("   ✅ Sigma scaling validation passed")
        except Exception as e:
            tprint_warning(f"⚠️ Sigma scaling validation failed: {e}")

        # Select best target
        tprint_debug("   🎯 Selecting best target...")
        best_target = self._select_best_target_with_weights(target_df, weights, target_columns)
        if best_target:
            tprint_success(f"✅ Selected best target for feature selection: {best_target}")
            selected_target_df = pd.DataFrame({best_target: target_df[best_target]})
            tprint_info(f"📊 Target data loaded: {len(selected_target_df)} rows, 1 target column")
            return selected_target_df

        tprint_info("📊 Using all available targets")
        tprint_info(f"🎯 Target columns: {target_columns}")
        tprint_info(f"⚖️ Horizon weights: {weights}")
        tprint_info(f"📊 Sample weights: {'Available' if sample_weights is not None else 'Not available'}")
        tprint_info(f"🔍 Quality scores: {'Available' if quality_scores else 'Not available'}")
        tprint_info(f"✅ Validation status: {'Passed' if validation_results.get('is_valid', False) else 'Failed'}")
        tprint_info(f"📊 Target data loaded: {len(target_df)} rows, {len(target_df.columns)} columns")
        return target_df

    def _select_best_target_with_weights(self, labels: pd.DataFrame, weights: Dict[str, float], target_columns: List[str]) -> Optional[str]:
        """Select the best target based on horizon weights and availability for feature selection."""
        tprint_debug("🎯 Selecting best target with weights")
        tprint_debug(f"   📊 Available weights: {weights}")
        tprint_debug(f"   📊 Target columns: {target_columns}")
        tprint_debug(f"   📊 Label columns: {list(labels.columns)}")
        
        try:
            if not weights or not target_columns:
                tprint_debug("   ⚠️ No weights or target columns available, using first available target")
                available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
                if available_targets:
                    tprint_info(f"   → Selected first available target: {available_targets[0]}")
                    return available_targets[0]
                else:
                    tprint_warning("   ⚠️ No available targets found")
                    return None

            # Priority order based on horizon weights (higher weight = higher priority)
            target_priority = []
            
            tprint_debug("   🔍 Mapping target columns to horizon weights...")
            for target in target_columns:
                if target in labels.columns:
                    # Determine horizon type from target name
                    if 'immediate' in target.lower() or 'small' in target.lower():
                        horizon_weight = weights.get('small', 0.0)
                        horizon_type = 'small'
                    elif 'short' in target.lower() or 'medium' in target.lower():
                        horizon_weight = weights.get('medium', 0.0)
                        horizon_type = 'medium'
                    elif 'leverage' in target.lower() or 'high' in target.lower():
                        horizon_weight = weights.get('high', 0.0)
                        horizon_type = 'high'
                    else:
                        # Default to small horizon if unclear
                        horizon_weight = weights.get('small', 0.0)
                        horizon_type = 'small (default)'
                    
                    target_priority.append((target, horizon_weight, horizon_type))
                    tprint_debug(f"   📊 {target}: {horizon_type} horizon, weight={horizon_weight:.3f}")

            # Sort by weight (descending) and return the highest weighted target
            if target_priority:
                target_priority.sort(key=lambda x: x[1], reverse=True)
                best_target = target_priority[0][0]
                best_weight = target_priority[0][1]
                best_type = target_priority[0][2]
                tprint_success(f"   ✅ Selected target '{best_target}' ({best_type} horizon) with weight {best_weight:.3f}")
                return best_target
            else:
                tprint_warning("   ⚠️ No valid target priorities found")
                return None

        except Exception as e:
            tprint_error(f"❌ Error selecting best target with weights: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            
            # Fallback to first available target
            tprint_debug("   🔄 Attempting fallback to first available target...")
            try:
                available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
                if available_targets:
                    tprint_info(f"   → Fallback selected: {available_targets[0]}")
                    return available_targets[0]
                else:
                    tprint_error("   ❌ No fallback targets available")
                    return None
            except Exception as fallback_error:
                tprint_error(f"   ❌ Fallback also failed: {fallback_error}")
                return None

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
        tprint_debug(f"   📊 Context: symbol={symbol}, exchange={exchange}, timeframe={timeframe}")
        tprint_debug(f"   📁 Data directory: {data_dir}")
        
        try:
            # Try different possible file locations and formats
            possible_files = [
                f"{symbol.lower()}_{timeframe}_features.parquet",
                f"{symbol.lower()}_{timeframe}_engineered_features.parquet",
                f"{symbol.lower()}_{timeframe}_final_features.parquet",
                f"{symbol.lower()}_{timeframe}_matrix_features.parquet"
            ]
            tprint_debug(f"   🔍 Searching for feature files: {len(possible_files)} patterns")

            data_path = Path(data_dir)
            if not data_path.exists():
                tprint_error(f"❌ Data directory does not exist: {data_path}")
                return None
            tprint_debug(f"   📂 Data path exists: {data_path.exists()}")

            # Search for feature files
            for i, filename in enumerate(possible_files):
                file_path = data_path / filename
                tprint_debug(f"   🔍 Checking file {i+1}/{len(possible_files)}: {filename}")
                
                if file_path.exists():
                    tprint_success(f"   📂 Found feature file: {file_path.name}")
                    tprint_debug(f"   📊 File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
                    
                    try:
                        # Load parquet file
                        data = pd.read_parquet(file_path)
                        tprint_debug(f"   ✅ Loaded data shape: {data.shape}")

                        # Apply data cleaning if available
                        data = self._apply_data_cleaning(data, "feature data")

                        # Standardize feature frame
                        data = self._standardize_feature_frame(data)

                        tprint_success(f"   ✅ Successfully loaded feature data: {data.shape}")
                        return data
                        
                    except Exception as e:
                        tprint_error(f"   ❌ Error loading {filename}: {e}")
                        continue

            # Fallback: try matrix operations file
            tprint_debug("   🔄 Attempting fallback to matrix operations file...")
            matrix_file = data_path / f"{symbol.lower()}_{timeframe}_matrix_operations.parquet"
            if matrix_file.exists():
                tprint_info(f"   📂 Found matrix operations file: {matrix_file.name}")
                try:
                    data = pd.read_parquet(matrix_file)
                    data = self._standardize_feature_frame(data)
                    tprint_success(f"   ✅ Loaded matrix operations data: {data.shape}")
                    return data
                except Exception as e:
                    tprint_error(f"   ❌ Error loading matrix operations file: {e}")

            tprint_warning("⚠️ No feature data files found")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Critical error loading feature data: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return None

    def _apply_data_cleaning(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply data cleaning utilities if available."""
        tprint_debug(f"   🧹 Applying data cleaning to {data_type}...")
        
        try:
            from src.utils.ml_common.data_processing.data_cleaning_utils import exclude_corrupted_periods

            # Ensure datetime column exists
            if 'timestamp' in data.columns and data['timestamp'].dtype == 'int64':
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                tprint_debug("   📅 Converted timestamp to datetime")
            elif 'datetime' not in data.columns:
                # Try to infer datetime column
                datetime_cols = [col for col in data.columns if 'time' in col.lower()]
                if datetime_cols:
                    data['datetime'] = pd.to_datetime(data[datetime_cols[0]])
                    tprint_debug(f"   📅 Inferred datetime from column: {datetime_cols[0]}")
                else:
                    data['datetime'] = data.index
                    tprint_debug("   📅 Using index as datetime")

            # Apply data cleaning
            original_count = len(data)
            data = exclude_corrupted_periods(data)
            cleaned_count = len(data)

            if original_count != cleaned_count:
                excluded_count = original_count - cleaned_count
                tprint_info(f"   🧹 Data cleaning applied: Excluded {excluded_count:,} corrupted rows ({100*excluded_count/original_count:.4f}%)")
            else:
                tprint_debug("   ✅ No corrupted data found")

        except ImportError as e:
            tprint_debug(f"   ℹ️ Data cleaning utility not available: {e}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Data cleaning failed, proceeding with original data: {e}")

        return data
    
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
        tprint_debug(f"   📊 Context: symbol={symbol}, exchange={exchange}, timeframe={timeframe}")
        tprint_debug(f"   📁 Data directory: {data_dir}")
        
        try:
            # Try to load target data from labeling step
            possible_target_files = [
                f"{symbol.lower()}_{timeframe}_labels.parquet",
                f"{symbol.lower()}_{timeframe}_triple_barrier_labels.parquet",
                f"{symbol.lower()}_{timeframe}_target.parquet"
            ]
            tprint_debug(f"   🔍 Checking {len(possible_target_files)} possible target files")

            data_path = Path(data_dir)
            if not data_path.exists():
                tprint_error(f"❌ Data directory does not exist: {data_path}")
                return None

            for i, filename in enumerate(possible_target_files):
                file_path = data_path / filename
                tprint_debug(f"   🔍 Checking target file {i+1}/{len(possible_target_files)}: {filename}")
                
                if file_path.exists():
                    tprint_success(f"   📂 Found target file: {file_path.name}")
                    try:
                        data = pd.read_parquet(file_path)
                        tprint_debug(f"   ✅ Loaded target data shape: {data.shape}")
                        
                        # Standardize target frame
                        data = self._standardize_target_frame(data)
                        tprint_debug("   🔧 Standardized target frame")

                        # Find canonical target columns
                        canonical_targets = filter_namespace_columns(data.columns, ColumnNamespace.TARGET)
                        tprint_debug(f"   🎯 Found {len(canonical_targets)} canonical target columns: {canonical_targets}")
                        
                        if canonical_targets:
                            target_col = canonical_targets[0]
                            target_data = data[target_col]
                            tprint_success(f"   ✅ Loaded target column '{target_col}' with {len(target_data)} samples")
                            return target_data
                        else:
                            tprint_warning(f"   ⚠️ No canonical target columns found in {filename}")
                            continue
                            
                    except Exception as e:
                        tprint_error(f"   ❌ Error loading {filename}: {e}")
                        continue

            tprint_warning("⚠️ No target data located, defaulting to unsupervised selection")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Critical error in fallback target data loading: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return None
    
    def _prepare_data(self, feature_data: pd.DataFrame, target_data: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for feature selection with comprehensive logging."""
        tprint("🔄 Preparing data for feature selection...")
        tprint_debug(f"   📊 Input features: {feature_data.shape[0]} samples, {feature_data.shape[1]} columns")
        
        try:
            # Clean feature data
            X = feature_data.copy()
            tprint_debug("   📋 Created feature data copy")
            
            # Remove non-numeric columns
            tprint_debug("   🔍 Identifying numeric columns...")
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            non_numeric_count = len(X.columns) - len(numeric_columns)
            
            if non_numeric_count > 0:
                tprint_info(f"   🗑️ Removing {non_numeric_count} non-numeric columns")
                non_numeric_cols = [col for col in X.columns if col not in numeric_columns]
                tprint_debug(f"   📋 Non-numeric columns: {non_numeric_cols}")
            else:
                tprint_debug("   ✅ All columns are numeric")
            
            X = X[numeric_columns]
            tprint_debug(f"   ✅ Kept {len(X.columns)} numeric columns")
            
            # Handle missing values
            tprint_debug("   🔍 Checking for missing values...")
            missing_count = X.isnull().sum().sum()
            if missing_count > 0:
                tprint_info(f"   🔧 Handling {missing_count} missing values using median imputation")
                X = X.fillna(X.median())
                tprint_debug("   ✅ Missing values imputed")
            else:
                tprint_debug("   ✅ No missing values found")
            
            # Remove infinite values
            tprint_debug("   🔍 Checking for infinite values...")
            inf_count = np.isinf(X.values).sum()
            if inf_count > 0:
                tprint_info(f"   🔧 Handling {inf_count} infinite values")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.median())
                tprint_debug("   ✅ Infinite values handled")
            else:
                tprint_debug("   ✅ No infinite values found")
            
            tprint_success(f"   ✅ Prepared {len(X)} samples with {len(X.columns)} numeric features")
            
            # Prepare target data if available
            y = None
            if target_data is not None:
                tprint_debug(f"   🎯 Processing target data: {target_data.shape[0]} samples")
                tprint_debug(f"   📊 Target data type: {type(target_data)}")
                
                # Align target data with feature data
                tprint_debug("   🔄 Aligning target data with feature data...")
                common_indices = X.index.intersection(target_data.index)
                tprint_debug(f"   📊 Common indices: {len(common_indices)}")
                
                if len(common_indices) > 0:
                    X = X.loc[common_indices]
                    y = target_data.loc[common_indices]
                    tprint_success(f"   ✅ Aligned target data: {len(y)} samples")
                else:
                    tprint_warning("   ⚠️ No common indices between features and target")
                    tprint_debug(f"   📊 Feature indices: {len(X.index)}")
                    tprint_debug(f"   📊 Target indices: {len(target_data.index)}")
            else:
                tprint_info("   ℹ️ No target data - will perform unsupervised feature selection")
            
            tprint_success(f"✅ Data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            tprint_error(f"❌ Error in data preparation: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            raise

    def _vectorbt_optimized_data_preparation(self, feature_data: pd.DataFrame, target_data: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Enhanced data preparation using VectorBT optimization for superior performance."""
        tprint("🔄 Preparing data with enhanced VectorBT optimization...")
        tprint_debug(f"   📊 Input features: {feature_data.shape[0]} samples, {feature_data.shape[1]} columns")
        tprint_debug(f"   ⚡ VectorBT enabled: {self.vectorbt_enabled}")
        tprint_debug(f"   🔧 VectorBT manager available: {self.vectorization_manager is not None}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_warning("   ⚠️ VectorBT not available, falling back to standard preparation")
            return self._prepare_data(feature_data, target_data)
        
        try:
            # Clean feature data
            tprint_debug("   📋 Creating feature data copy...")
            X = feature_data.copy()
            
            # Remove non-numeric columns
            tprint_debug("   🔍 Identifying numeric columns...")
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            non_numeric_count = len(X.columns) - len(numeric_columns)
            
            if non_numeric_count > 0:
                tprint_info(f"   🗑️ Removing {non_numeric_count} non-numeric columns")
                non_numeric_cols = [col for col in X.columns if col not in numeric_columns]
                tprint_debug(f"   📋 Non-numeric columns: {non_numeric_cols}")
            else:
                tprint_debug("   ✅ All columns are numeric")
            
            X = X[numeric_columns]
            tprint_debug(f"   ✅ Kept {len(X.columns)} numeric columns")
            
            # Use VectorBTRollingOptimizer for enhanced statistical operations
            if self.vectorbt_optimizer:
                tprint_debug("   ⚡ Using VectorBT optimizer for enhanced processing...")
                
                # Optimize data types for VectorBT processing
                tprint_debug("   🔧 Optimizing data types for VectorBT...")
                X = self._optimize_dataframe_for_vectorbt(X)
                
                # Use VectorBT for missing value imputation with rolling statistics
                tprint_debug("   🔍 Checking for missing values...")
                missing_count = X.isnull().sum().sum()
                if missing_count > 0:
                    tprint_info(f"   🔧 Handling {missing_count} missing values using VectorBT rolling operations")
                    # Use rolling median for more robust imputation
                    for col in X.columns:
                        if X[col].isnull().any():
                            tprint_debug(f"   🔧 Processing column: {col}")
                            rolling_median = self.vectorbt_optimizer.rolling_median(X[col], window=20)
                            X[col] = X[col].fillna(rolling_median)
                    tprint_debug("   ✅ Missing values imputed with VectorBT")
                else:
                    tprint_debug("   ✅ No missing values found")
                
                # Use VectorBT for outlier detection and handling
                tprint_debug("   🔍 Applying VectorBT outlier handling...")
                X = self._vectorbt_outlier_handling(X)
                
                # Use VectorBT for data normalization
                tprint_debug("   📊 Applying VectorBT normalization...")
                X = self._vectorbt_normalize_data(X)
            else:
                tprint_warning("   ⚠️ VectorBT optimizer not available, using standard methods")
                # Fallback to standard missing value handling
                missing_count = X.isnull().sum().sum()
                if missing_count > 0:
                    tprint_info(f"   🔧 Handling {missing_count} missing values using standard method")
                    X = X.fillna(X.median())
                else:
                    tprint_debug("   ✅ No missing values found")
            
            # Remove infinite values
            tprint_debug("   🔍 Checking for infinite values...")
            inf_count = np.isinf(X.values).sum()
            if inf_count > 0:
                tprint_info(f"   🔧 Handling {inf_count} infinite values")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.median())
                tprint_debug("   ✅ Infinite values handled")
            else:
                tprint_debug("   ✅ No infinite values found")
            
            # Optimize DataFrame for VectorBT processing
            tprint_debug("   ⚡ Optimizing DataFrame with VectorBT manager...")
            X_optimized = self.vectorization_manager.optimize_dataframe(X)
            tprint_success(f"   ✅ VectorBT-optimized data: {len(X_optimized)} samples, {len(X_optimized.columns)} features")
            
            # Prepare target data with VectorBT optimization
            tprint_debug("   🎯 Optimizing target data with VectorBT...")
            y = self._vectorbt_optimize_target_data(target_data, X_optimized)
            
            tprint_success(f"✅ Enhanced VectorBT data preparation completed: {X_optimized.shape[0]} samples, {X_optimized.shape[1]} features")
            return X_optimized, y
            
        except Exception as e:
            tprint_error(f"❌ Enhanced VectorBT data preparation failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            tprint_warning("   🔄 Falling back to standard data preparation...")
            return self._prepare_data(feature_data, target_data)
    
    def _optimize_dataframe_for_vectorbt(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for VectorBT processing."""
        tprint_debug("⚡ Optimizing DataFrame for VectorBT processing")
        tprint_debug(f"   📊 Input shape: {data.shape}")
        tprint_debug(f"   📊 Input dtypes: {data.dtypes.value_counts().to_dict()}")
        
        try:
            optimized_data = data.copy()
            conversions_made = 0
            
            # Convert to more memory-efficient types
            for col in optimized_data.columns:
                original_dtype = optimized_data[col].dtype
                
                if original_dtype == 'float64':
                    # Check if we can use float32
                    col_min = optimized_data[col].min()
                    col_max = optimized_data[col].max()
                    
                    if (col_min >= np.finfo(np.float32).min and 
                        col_max <= np.finfo(np.float32).max):
                        optimized_data[col] = optimized_data[col].astype(np.float32)
                        conversions_made += 1
                        tprint_debug(f"   🔧 Converted {col}: float64 -> float32")
                        
                elif original_dtype == 'int64':
                    # Check if we can use int32
                    col_min = optimized_data[col].min()
                    col_max = optimized_data[col].max()
                    
                    if (col_min >= np.iinfo(np.int32).min and 
                        col_max <= np.iinfo(np.int32).max):
                        optimized_data[col] = optimized_data[col].astype(np.int32)
                        conversions_made += 1
                        tprint_debug(f"   🔧 Converted {col}: int64 -> int32")
            
            tprint_debug(f"   ✅ Optimized {conversions_made} columns for VectorBT")
            tprint_debug(f"   📊 Output dtypes: {optimized_data.dtypes.value_counts().to_dict()}")
            return optimized_data
            
        except Exception as e:
            tprint_error(f"❌ DataFrame optimization failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return data
    
    def _vectorbt_outlier_handling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using VectorBT rolling operations."""
        tprint_debug("🔧 Handling outliers with VectorBT operations")
        tprint_debug(f"   📊 Input shape: {data.shape}")
        
        try:
            if not self.vectorbt_optimizer:
                tprint_debug("   ⚠️ VectorBT optimizer not available, skipping outlier handling")
                return data
            
            processed_data = data.copy()
            columns_processed = 0
            
            for col in processed_data.columns:
                if processed_data[col].dtype in [np.float32, np.float64]:
                    tprint_debug(f"   🔧 Processing column: {col}")
                    
                    # Use VectorBT rolling quantiles for outlier detection
                    rolling_q25 = self.vectorbt_optimizer.rolling_quantile(processed_data[col], window=50, q=0.25)
                    rolling_q75 = self.vectorbt_optimizer.rolling_quantile(processed_data[col], window=50, q=0.75)
                    rolling_iqr = rolling_q75 - rolling_q25
                    
                    # Define outlier bounds
                    lower_bound = rolling_q25 - 1.5 * rolling_iqr
                    upper_bound = rolling_q75 + 1.5 * rolling_iqr
                    
                    # Count outliers before clipping
                    outliers_before = ((processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)).sum()
                    
                    # Cap outliers instead of removing them
                    processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    # Count outliers after clipping
                    outliers_after = ((processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)).sum()
                    
                    if outliers_before > 0:
                        tprint_debug(f"   📊 {col}: clipped {outliers_before} outliers")
                        columns_processed += 1
                    else:
                        tprint_debug(f"   ✅ {col}: no outliers found")
            
            tprint_debug(f"   ✅ Processed {columns_processed} columns for outlier handling")
            return processed_data
            
        except Exception as e:
            tprint_error(f"❌ VectorBT outlier handling failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return data
    
    def _vectorbt_normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using VectorBT rolling operations."""
        tprint_debug("📊 Normalizing data with VectorBT operations")
        tprint_debug(f"   📊 Input shape: {data.shape}")
        
        try:
            if not self.vectorbt_optimizer:
                tprint_debug("   ⚠️ VectorBT optimizer not available, skipping normalization")
                return data
            
            normalized_data = data.copy()
            columns_processed = 0
            
            for col in normalized_data.columns:
                if normalized_data[col].dtype in [np.float32, np.float64]:
                    tprint_debug(f"   📊 Normalizing column: {col}")
                    
                    # Use VectorBT rolling mean and std for normalization
                    rolling_mean = self.vectorbt_optimizer.rolling_mean(normalized_data[col], window=100)
                    rolling_std = self.vectorbt_optimizer.rolling_std(normalized_data[col], window=100)
                    
                    # Avoid division by zero
                    zero_std_count = (rolling_std == 0).sum()
                    if zero_std_count > 0:
                        tprint_debug(f"   ⚠️ {col}: found {zero_std_count} zero std values, replacing with 1")
                        rolling_std = rolling_std.replace(0, 1)
                    
                    # Z-score normalization
                    normalized_data[col] = (normalized_data[col] - rolling_mean) / rolling_std
                    columns_processed += 1
                    tprint_debug(f"   ✅ {col}: normalized successfully")
            
            tprint_debug(f"   ✅ Normalized {columns_processed} columns")
            return normalized_data
            
        except Exception as e:
            tprint_error(f"❌ VectorBT normalization failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return data
    
    def _vectorbt_optimize_target_data(self, target_data: Optional[pd.Series], feature_data: pd.DataFrame) -> Optional[pd.Series]:
        """Optimize target data using VectorBT operations."""
        tprint_debug("🎯 Optimizing target data with VectorBT")
        tprint_debug(f"   📊 Feature data shape: {feature_data.shape}")
        tprint_debug(f"   📊 Target data type: {type(target_data)}")
        
        if target_data is None:
            tprint_info("   ℹ️ No target data - will perform unsupervised feature selection")
            return None
        
        try:
            tprint_debug(f"   🎯 Processing target data: {target_data.shape[0]} samples")
            tprint_debug(f"   📊 Target data dtype: {target_data.dtype}")
            
            # Align target data with feature data
            tprint_debug("   🔄 Aligning target data with feature data...")
            common_indices = feature_data.index.intersection(target_data.index)
            tprint_debug(f"   📊 Common indices: {len(common_indices)}")
            
            if len(common_indices) > 0:
                aligned_target = target_data.loc[common_indices]
                tprint_success(f"   ✅ Aligned target data: {len(aligned_target)} samples")
                
                # Use VectorBT for target data optimization if available
                if self.vectorbt_optimizer and len(aligned_target) > 50:
                    tprint_debug("   ⚡ Applying VectorBT smoothing to target data...")
                    # Apply VectorBT smoothing to target data
                    smoothed_target = self.vectorbt_optimizer.rolling_mean(aligned_target, window=5)
                    # Use a weighted combination of original and smoothed
                    optimized_target = 0.7 * aligned_target + 0.3 * smoothed_target
                    tprint_debug("   ✅ VectorBT target optimization applied")
                    return optimized_target
                else:
                    if not self.vectorbt_optimizer:
                        tprint_debug("   ⚠️ VectorBT optimizer not available for target optimization")
                    else:
                        tprint_debug("   ⚠️ Target data too small for VectorBT optimization")
                    return aligned_target
            else:
                tprint_warning("   ⚠️ No common indices between features and target")
                tprint_debug(f"   📊 Feature indices: {len(feature_data.index)}")
                tprint_debug(f"   📊 Target indices: {len(target_data.index)}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ VectorBT target optimization failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return target_data
    
    async def _run_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series],
                                   symbol: str, exchange: str, timeframe: str) -> Any:
        """Run the multi-stage feature selection with comprehensive logging."""
        tprint("🔍 Running Multi-Stage Feature Selection")
        tprint_debug(f"   📊 Input: {len(X)} samples, {len(X.columns)} features")
        tprint_debug(f"   🎯 Target: Variable→60 pipeline (mRMR/Ensemble/RFE)")
        tprint_debug(f"   📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
        
        if y is not None:
            tprint_info(f"   🎯 Target: {len(y)} samples (supervised learning)")
            tprint_debug(f"   📊 Target type: {'classification' if len(y.unique()) <= 10 else 'regression'}")
            tprint_debug(f"   📊 Target unique values: {len(y.unique())}")
        else:
            tprint_info("   🎯 No target data (unsupervised learning)")
        
        tprint_debug("   ⚡ Using vectorized operations and caching")
        tprint_debug("   🔄 Starting feature selection pipeline...")
        
        try:
            # Run feature selection in a thread pool to avoid blocking
            tprint_debug("   🔄 Executing feature selection in thread pool...")
            loop = asyncio.get_event_loop()
            selection_result = await loop.run_in_executor(
                None,
                self._run_selection_sync,
                X, y
            )
            tprint_debug("   ✅ Feature selection execution completed")

            # Process selection results
            tprint_debug("   📊 Processing selection results...")
            final_scores = getattr(selection_result, 'final_scores', {}) or {}
            selection_result.eligible_for_selection = bool(final_scores.get('eligible_for_selection', True))
            selection_result.turnover_rejection_reason = final_scores.get('turnover_rejection_reason')
            
            if not selection_result.eligible_for_selection:
                reason = selection_result.turnover_rejection_reason or 'Turnover constraints violated'
                tprint_warning(f"🚫 Selection result marked ineligible: {reason}")
            else:
                tprint_debug("   ✅ Selection result is eligible")

            # Extract numeric scores
            final_numeric_scores = {
                key: float(value)
                for key, value in final_scores.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            tprint_debug(f"   📊 Final numeric scores: {len(final_numeric_scores)} metrics")

            # Collect hypothesis p-values
            tprint_debug("   🔍 Collecting hypothesis p-values...")
            horizon_p_values, feature_p_values, lookback_p_values = self._collect_hypothesis_p_values(selection_result)
            tprint_debug(f"   📊 Horizon p-values: {len(horizon_p_values)}")
            tprint_debug(f"   📊 Feature p-values: {len(feature_p_values)}")
            tprint_debug(f"   📊 Lookback p-values: {len(lookback_p_values)}")

            # Track and control hypotheses
            tprint_debug("   🔬 Tracking and controlling hypotheses...")
            horizon_significance_metrics = {
                key: {"p_value": value}
                for key, value in horizon_p_values.items()
            }
            hypothesis_report = track_and_control_hypotheses(
                horizon_results=horizon_significance_metrics if horizon_significance_metrics else horizon_p_values,
                feature_results=feature_p_values,
                lookback_results=lookback_p_values,
            )
            
            if hypothesis_report.get("warning"):
                tprint_warning(hypothesis_report["warning"])
            else:
                tprint_debug("   ✅ No hypothesis warnings")

            # Update selection result with hypothesis data
            selection_result.horizon_p_values = horizon_p_values
            selection_result.feature_p_values = feature_p_values
            selection_result.lookback_p_values = lookback_p_values
            selection_result.adjusted_p_values = hypothesis_report.get("adjusted_p_values", {})
            selection_result.hypothesis_report = hypothesis_report

            # Create selection payload for validation
            tprint_debug("   📦 Creating selection payload...")
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

            # Validate selection artifact
            tprint_debug("   🔍 Validating selection artifact...")
            try:
                validate_selection_artifact(
                    selection_payload,
                    context='final_feature_selection_step.selection_result',
                )
                tprint_debug("   ✅ Selection artifact validation passed")
            except DataContractValidationError as contract_error:
                tprint_error(f"❌ Selection result failed validation: {contract_error}")
                raise

            tprint_success("✅ Multi-stage feature selection completed")
            tprint_info(f"   📊 Final features: {len(selection_result.final_features)}")
            tprint_info(f"   ⏱️ Selection time: {selection_result.selection_time:.3f} seconds")

            return selection_result
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            raise

    def _run_selection_sync(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Any:
        """Enhanced synchronous feature selection with VectorBT optimizations."""
        tprint("⚙️ Executing enhanced synchronous selection pipeline with VectorBT")
        tprint_debug(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        tprint_debug(f"   ⚡ VectorBT enabled: {self.vectorbt_enabled}")
        tprint_debug(f"   🎯 Target available: {y is not None}")
        
        try:
            # Use VectorBT enhanced feature selection if available
            if self.vectorbt_enabled:
                tprint_debug("   ⚡ Using VectorBT enhanced feature selection...")
                return self._vectorbt_enhanced_feature_selection(X, y)
            
            # Fallback to standard selection
            tprint_debug("   🔄 Using standard feature selection...")
            if not getattr(self, "_pipeline_available", False):
                error_msg = "Feature selection pipeline is unavailable"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from self._pipeline_import_error

            from .final_feature_selection_pipeline import MultiStageFeatureSelector

            tprint_debug("   📦 Creating MultiStageFeatureSelector...")
            selector = MultiStageFeatureSelector(self.feature_config)

            # Run selection
            if y is not None:
                tprint_info("   🎯 Using supervised selector with provided target")
                tprint_debug(f"   📊 Target shape: {y.shape}")
                result = selector.select_features(X, y)
                tprint_debug("   ✅ Supervised selection completed")
            else:
                # For unsupervised selection, create a dummy target
                tprint_info("   🧪 No target provided, creating proxy target for unsupervised run")
                dummy_target = X.iloc[:, 0]  # Use first feature as proxy target
                tprint_debug(f"   📊 Proxy target shape: {dummy_target.shape}")
                result = selector.select_features(X, dummy_target)
                result.is_unsupervised = True
                tprint_debug("   ✅ Unsupervised selection completed")

            tprint_success("   ✅ Synchronous selection complete")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Synchronous selection failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            raise
    
    def _vectorbt_enhanced_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Any:
        """Enhanced feature selection using VectorBT optimizations."""
        tprint("🚀 Running VectorBT-enhanced feature selection")
        tprint_debug(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        tprint_debug(f"   ⚡ VectorBT optimizer available: {self.vectorbt_optimizer is not None}")
        
        try:
            # Use VectorBT for feature importance calculations
            tprint_debug("   📊 Calculating VectorBT feature importance...")
            feature_importance = self._vectorbt_calculate_feature_importance(X, y)
            tprint_debug(f"   ✅ Feature importance calculated: {len(feature_importance)} features")
            
            # Use VectorBT for stability analysis
            tprint_debug("   🔍 Performing VectorBT stability analysis...")
            stability_scores = self._vectorbt_stability_analysis(X, y)
            tprint_debug(f"   ✅ Stability analysis completed: {len(stability_scores)} features")
            
            # Use VectorBT for correlation-based feature selection
            tprint_debug("   🔗 Computing VectorBT correlation analysis...")
            correlation_matrix = self._vectorbt_correlation_analysis(X)
            tprint_debug(f"   ✅ Correlation matrix computed: {correlation_matrix.shape}")
            
            # Create enhanced config with VectorBT parameters
            tprint_debug("   ⚙️ Creating enhanced configuration...")
            enhanced_config = self.feature_config
            enhanced_config.vectorbt_optimized = True
            enhanced_config.feature_importance = feature_importance
            enhanced_config.stability_scores = stability_scores
            enhanced_config.correlation_matrix = correlation_matrix
            tprint_debug("   ✅ Enhanced configuration created")
            
            # Import and use the feature selector
            if not getattr(self, "_pipeline_available", False):
                error_msg = "Feature selection pipeline is unavailable"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from self._pipeline_import_error

            from .final_feature_selection_pipeline import MultiStageFeatureSelector
            tprint_debug("   📦 Creating MultiStageFeatureSelector with VectorBT enhancements...")
            selector = MultiStageFeatureSelector(enhanced_config)
            
            if y is not None:
                tprint_info("   🎯 Using VectorBT-enhanced supervised selector")
                tprint_debug(f"   📊 Target shape: {y.shape}")
                result = selector.select_features(X, y)
                tprint_debug("   ✅ VectorBT supervised selection completed")
            else:
                tprint_info("   🧪 Using VectorBT-enhanced unsupervised selector")
                dummy_target = X.iloc[:, 0]
                tprint_debug(f"   📊 Proxy target shape: {dummy_target.shape}")
                result = selector.select_features(X, dummy_target)
                result.is_unsupervised = True
                tprint_debug("   ✅ VectorBT unsupervised selection completed")
            
            # Add VectorBT performance metrics to result
            tprint_debug("   📊 Adding VectorBT metrics to result...")
            result.vectorbt_enhanced = True
            result.feature_importance = feature_importance
            result.stability_scores = stability_scores
            result.correlation_matrix = correlation_matrix
            tprint_debug("   ✅ VectorBT metrics added to result")
            
            tprint_success("   ✅ VectorBT-enhanced selection complete")
            return result
            
        except Exception as e:
            tprint_error(f"❌ VectorBT enhanced feature selection failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            tprint_warning("   🔄 Falling back to standard feature selection...")
            return self._run_standard_feature_selection(X, y)
    
    def _run_standard_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Any:
        """Standard feature selection fallback."""
        tprint_debug("🔄 Running standard feature selection fallback")
        tprint_debug(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        try:
            if not getattr(self, "_pipeline_available", False):
                error_msg = "Feature selection pipeline is unavailable"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from self._pipeline_import_error

            from .final_feature_selection_pipeline import MultiStageFeatureSelector
            tprint_debug("   📦 Creating MultiStageFeatureSelector...")
            selector = MultiStageFeatureSelector(self.feature_config)

            if y is not None:
                tprint_debug("   🎯 Running supervised selection...")
                result = selector.select_features(X, y)
                tprint_debug("   ✅ Supervised selection completed")
            else:
                tprint_debug("   🧪 Running unsupervised selection...")
                dummy_target = X.iloc[:, 0]
                result = selector.select_features(X, dummy_target)
                result.is_unsupervised = True
                tprint_debug("   ✅ Unsupervised selection completed")

            tprint_debug("   ✅ Standard feature selection completed")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Standard feature selection failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            raise
    
    def _vectorbt_calculate_feature_importance(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate feature importance using VectorBT operations."""
        tprint_debug("📊 Calculating feature importance with VectorBT")
        tprint_debug(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        try:
            if not self.vectorbt_optimizer:
                tprint_debug("   ⚠️ VectorBT optimizer not available, returning empty importance scores")
                return {}
            
            if y is None:
                tprint_debug("   ⚠️ No target data available, returning empty importance scores")
                return {}
            
            tprint_debug(f"   📊 Target shape: {y.shape}")
            importance_scores = {}
            columns_processed = 0
            
            for col in X.columns:
                if X[col].dtype in [np.float32, np.float64]:
                    tprint_debug(f"   📊 Processing column: {col}")
                    try:
                        # Use VectorBT rolling correlation for importance
                        rolling_corr = self.vectorbt_optimizer.rolling_corr(X[col], y, window=50)
                        # Use mean absolute correlation as importance score
                        importance_score = float(abs(rolling_corr.mean()))
                        importance_scores[col] = importance_score
                        columns_processed += 1
                        tprint_debug(f"   ✅ {col}: importance = {importance_score:.4f}")
                    except Exception as col_error:
                        tprint_warning(f"   ⚠️ Error processing {col}: {col_error}")
                        continue
            
            tprint_debug(f"   ✅ Processed {columns_processed} columns for importance calculation")
            return importance_scores
            
        except Exception as e:
            tprint_error(f"❌ VectorBT feature importance calculation failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return {}
    
    def _vectorbt_stability_analysis(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Dict[str, float]:
        """Perform stability analysis using VectorBT operations."""
        tprint_debug("🔍 Performing stability analysis with VectorBT")
        tprint_debug(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        try:
            if not self.vectorbt_optimizer:
                tprint_debug("   ⚠️ VectorBT optimizer not available, returning empty stability scores")
                return {}
            
            stability_scores = {}
            columns_processed = 0
            
            for col in X.columns:
                if X[col].dtype in [np.float32, np.float64]:
                    tprint_debug(f"   🔍 Processing column: {col}")
                    try:
                        # Use VectorBT rolling std for stability measurement
                        rolling_std = self.vectorbt_optimizer.rolling_std(X[col], window=100)
                        # Lower std indicates higher stability
                        stability_score = float(1.0 / (1.0 + rolling_std.mean()))
                        stability_scores[col] = stability_score
                        columns_processed += 1
                        tprint_debug(f"   ✅ {col}: stability = {stability_score:.4f}")
                    except Exception as col_error:
                        tprint_warning(f"   ⚠️ Error processing {col}: {col_error}")
                        continue
            
            tprint_debug(f"   ✅ Processed {columns_processed} columns for stability analysis")
            return stability_scores
            
        except Exception as e:
            tprint_error(f"❌ VectorBT stability analysis failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return {}
    
    def _vectorbt_correlation_analysis(self, X: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """Perform correlation analysis using VectorBT optimization."""
        tprint_debug(f"🔗 Performing correlation analysis with VectorBT (window={window})")
        tprint_debug(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        try:
            if not self.vectorbt_optimizer:
                tprint_debug("   ⚠️ VectorBT optimizer not available, using pandas correlation")
                return X.corr()
            
            tprint_debug(f"   ⚡ Using VectorBT for correlation analysis...")
            
            # Use VectorBT for rolling correlation analysis
            try:
                correlation_matrix = self.vectorbt_optimizer.rolling_correlation_matrix(X, window=window)
                tprint_debug(f"   ✅ VectorBT rolling correlation matrix computed: {correlation_matrix.shape}")
                
                # Compute final correlation using VectorBT
                if len(X) > window:
                    tprint_debug("   ⚡ Computing final correlation with VectorBT...")
                    final_corr = self.vectorbt_optimizer.rolling_corr(X, X, window=min(window, len(X)))
                    tprint_debug(f"   ✅ Final correlation computed: {final_corr.shape}")
                    return final_corr
                else:
                    tprint_debug("   ⚠️ Data too small for VectorBT rolling correlation, using standard correlation")
                    return X.corr()
                    
            except Exception as vectorbt_error:
                tprint_warning(f"   ⚠️ VectorBT correlation failed: {vectorbt_error}, using pandas fallback")
                return X.corr()
            
        except Exception as e:
            tprint_error(f"❌ VectorBT correlation analysis failed: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            tprint_warning("   🔄 Using pandas correlation fallback...")
            return X.corr()
    
    def _get_enhanced_vectorbt_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive VectorBT performance statistics."""
        tprint_debug("📊 Getting enhanced VectorBT performance statistics")
        stats = {}
        
        try:
            # Get VectorBT rolling optimizer stats
            if self.vectorbt_optimizer:
                tprint_debug("   ⚡ Getting VectorBT optimizer stats...")
                try:
                    optimizer_stats = self.vectorbt_optimizer.get_performance_stats()
                    stats.update({
                        'vectorbt_rolling_operations': optimizer_stats.get('vectorbt_operations', 0),
                        'pandas_fallbacks': optimizer_stats.get('pandas_fallbacks', 0),
                        'gpu_operations': optimizer_stats.get('gpu_operations', 0),
                        'memory_optimizations': optimizer_stats.get('memory_optimizations', 0),
                        'chunk_operations': optimizer_stats.get('chunk_operations', 0),
                        'avg_time_per_operation': optimizer_stats.get('avg_time_per_operation', 0.0),
                        'vectorbt_usage_rate': optimizer_stats.get('vectorbt_usage_rate', 0.0)
                    })
                    tprint_debug(f"   ✅ VectorBT optimizer stats: {len(optimizer_stats)} metrics")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Error getting VectorBT optimizer stats: {e}")
            else:
                tprint_debug("   ⚠️ VectorBT optimizer not available")
            
            # Get unified vectorization manager stats
            if self.vectorization_manager:
                tprint_debug("   🔧 Getting vectorization manager stats...")
                try:
                    manager_stats = self.vectorization_manager.get_performance_stats()
                    stats.update({
                        'total_operations': manager_stats.get('total_operations', 0),
                        'strategy_usage': manager_stats.get('strategy_usage', {}),
                        'average_speedup': manager_stats.get('average_speedup', 0.0),
                        'total_computation_time': manager_stats.get('total_computation_time', 0.0)
                    })
                    tprint_debug(f"   ✅ Vectorization manager stats: {len(manager_stats)} metrics")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Error getting vectorization manager stats: {e}")
            else:
                tprint_debug("   ⚠️ Vectorization manager not available")
            
            tprint_debug(f"   ✅ Total stats collected: {len(stats)} metrics")
            return stats
            
        except Exception as e:
            tprint_error(f"❌ Error getting VectorBT performance stats: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            return {}

    def _collect_hypothesis_p_values(self, selection_result: Any) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Extract horizon, feature, and lookback p-values from the selection result."""
        tprint_debug("📊 Collecting hypothesis p-values from selection result")
        tprint_debug(f"   📊 Selection result type: {type(selection_result)}")
        
        try:
            tprint_debug("   🔍 Extracting p-value mapping...")
            flattened = extract_p_value_mapping(vars(selection_result))
            tprint_debug(f"   ✅ Extracted {len(flattened)} p-value mappings")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract p-value mapping: {e}")
            import traceback
            tprint_debug(f"   🔍 Error details: {traceback.format_exc()}")
            flattened = {}

        # Categorize p-values by type
        tprint_debug("   🔍 Categorizing p-values...")
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

        tprint_debug(f"   📊 Horizon p-values: {len(horizon_p_values)}")
        tprint_debug(f"   📊 Lookback p-values: {len(lookback_p_values)}")
        tprint_debug(f"   📊 Feature p-values: {len(feature_p_values)}")

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
            tprint_error(f"❌ Failed to save selection results: {e}")
            import traceback
            tprint_debug(f"🔍 Save error details: {traceback.format_exc()}")

    def _save_selection_results_sync(self, selection_result: Any, symbol: str, exchange: str,
                                     timeframe: str) -> None:
        """Synchronous helper for saving feature selection results."""
        tprint_debug(f"💾 Saving selection results for {symbol}/{exchange}/{timeframe}")
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
            tprint(f"   📊 Stage 1 (mRMR): {selection_result.feature_counts.get('stage_1', 'N/A')} features")
            tprint(f"   📊 Stage 2 (Ensemble): {selection_result.feature_counts.get('stage_2', 'N/A')} features")
            tprint(f"   📊 Stage 3 (RFE): {selection_result.feature_counts.get('stage_3', 'N/A')} features")
            tprint(f"   📊 Final: {selection_result.feature_counts.get('final', 'N/A')} features")
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
            tprint("⚡ VECTORBT OPTIMIZATION SUMMARY:")
            if self.vectorbt_enabled:
                tprint("   ✅ VectorBT rolling operations: Enabled")
                tprint("   ✅ VectorBT correlation analysis: Enabled")
                tprint("   ✅ VectorBT memory optimization: Enabled")
                tprint("   ✅ VectorBT matrix operations: Enabled")
                tprint("   ✅ VectorBT feature importance: Enabled")
                tprint("   ✅ VectorBT stability analysis: Enabled")
                
                # Show VectorBT performance metrics
                try:
                    vectorbt_stats = self._get_enhanced_vectorbt_performance_stats()
                    if vectorbt_stats:
                        tprint("   📊 VectorBT Performance Metrics:")
                        tprint(f"      - VectorBT operations: {vectorbt_stats.get('vectorbt_rolling_operations', 0)}")
                        tprint(f"      - GPU operations: {vectorbt_stats.get('gpu_operations', 0)}")
                        tprint(f"      - Memory optimizations: {vectorbt_stats.get('memory_optimizations', 0)}")
                        tprint(f"      - Chunk operations: {vectorbt_stats.get('chunk_operations', 0)}")
                        tprint(f"      - Usage rate: {vectorbt_stats.get('vectorbt_usage_rate', 0):.2%}")
                        tprint(f"      - Average speedup: {vectorbt_stats.get('average_speedup', 0):.2f}x")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Could not retrieve VectorBT metrics: {e}")
            else:
                tprint("   ❌ VectorBT optimization: Disabled")
            
            tprint("⚡ GENERAL OPTIMIZATION SUMMARY:")
            tprint("   ✅ Vectorized operations: Enabled")
            tprint("   ✅ Caching: Enabled")
            tprint("   ✅ Comprehensive logging: Enabled")
            tprint("   ✅ Multi-stage reduction: Variable→60 (mRMR/Ensemble/RFE)")
            
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
    tprint_debug(f"   📊 Training features: {train_features.shape}")
    tprint_debug(f"   📊 Validation features: {val_features.shape}")
    tprint_debug(f"   📊 Max mean shift threshold: {max_mean_shift}")
    
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
                                         timeframe: Optional[str] = None,
                                         data_dir: Optional[str] = None,
                                         config: Optional[Dict[str, Any]] = None) -> bool:
    """Run the final feature selection step with adaptive timeframe defaults.

    The timeframe resolution mirrors the pre-training sub-pipeline: an explicit
    ``timeframe`` argument takes precedence, then ``config['timeframe']`` or
    ``config['custom_params']`` are considered. When no explicit configuration
    is provided the helper falls back to ``15m`` by default, or ``60m`` when the
    supplied config indicates the Analyst pipeline.
    """

    runtime_config: Dict[str, Any] = dict(config or {})

    def _extract_timeframe_from_config(cfg: Mapping[str, Any]) -> Optional[str]:
        tprint_debug("🔍 Extracting timeframe from config")
        candidate = cfg.get('timeframe')
        if isinstance(candidate, str) and candidate:
            tprint_debug(f"   ✅ Found timeframe in config: {candidate}")
            return candidate

        custom_params = cfg.get('custom_params')
        if isinstance(custom_params, Mapping):
            for key in ('timeframe', 'default_timeframe'):
                candidate = custom_params.get(key)
                if isinstance(candidate, str) and candidate:
                    tprint_debug(f"   ✅ Found timeframe in custom_params: {candidate}")
                    return candidate
        tprint_debug("   ⚠️ No timeframe found in config")
        return None

    def _config_indicates_analyst(cfg: Mapping[str, Any]) -> bool:
        tprint_debug("🔍 Checking if config indicates analyst pipeline")
        analyst_truthy_strings = {'analyst', 'true', 'yes', 'enabled'}

        def _walk(value: Any, key_hint: Optional[str] = None) -> bool:
            if isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    nested_key_str = str(nested_key)
                    nested_key_lower = nested_key_str.lower()
                    if nested_key_lower in {'profile', 'mode', 'pipeline_profile', 'pipeline_mode', 'model_type'}:
                        if isinstance(nested_value, str) and nested_value.strip().lower() == 'analyst':
                            return True
                    if 'analyst' in nested_key_lower:
                        if isinstance(nested_value, bool) and nested_value:
                            return True
                        if isinstance(nested_value, str) and nested_value.strip().lower() in analyst_truthy_strings:
                            return True
                        if isinstance(nested_value, (int, float)) and nested_value == 1:
                            return True
                    if _walk(nested_value, nested_key_str):
                        return True
                return False
            if isinstance(value, (list, tuple, set)):
                return any(_walk(item, key_hint) for item in value)
            if isinstance(value, str):
                return value.strip().lower() == 'analyst'
            if isinstance(value, bool) and key_hint and 'analyst' in key_hint.lower():
                return value
            if isinstance(value, (int, float)) and key_hint and 'analyst' in key_hint.lower():
                return value == 1
            return False

        return _walk(cfg)

    resolved_timeframe: str
    timeframe_source: str
    if timeframe:
        resolved_timeframe = timeframe
        timeframe_source = 'explicit argument'
    else:
        extracted = _extract_timeframe_from_config(runtime_config)
        if extracted:
            resolved_timeframe = extracted
            timeframe_source = 'config override'
        else:
            default_timeframe = '60m' if _config_indicates_analyst(runtime_config) else '15m'
            resolved_timeframe = default_timeframe
            timeframe_source = 'analyst default' if default_timeframe == '60m' else 'global default'

    locator = runtime_config.get('data_locator')
    if data_dir is None and isinstance(locator, PipelineDataLocator):
        data_dir_key = runtime_config.get('data_dir_key', 'market_data')
        data_dir = str(locator.data_path(data_dir_key))

    if data_dir is None:
        raise ValueError("data_dir must be provided or resolvable via DataLocator")

    tprint("🚀 Invoking run_final_feature_selection_step helper")
    tprint("   📊 Parameters: symbol=%s, exchange=%s, timeframe=%s (%s), data_dir=%s"
          % (symbol, exchange, resolved_timeframe, timeframe_source, data_dir))
    step = FinalFeatureSelectionStep(runtime_config)
    tprint("🔄 Delegating execution to FinalFeatureSelectionStep instance")
    return await step.execute_final_feature_selection(symbol, exchange, resolved_timeframe, data_dir)

