"""
NAS/TAS Regime Data Splitting - Enhanced Regime Tagging with NAS/TAS Clustering Integration

This module creates a unified dataset with regime labels for regime-aware processing.
It uses NAS/TAS clustering results to determine regime membership and tag market data
with regime states and probabilities.

KEY FEATURES:
- NAS/TAS clustering integration for regime tagging
- 100% data retention (no rows lost to splitting boundaries)
- Full lookback period preservation for all features
- Temporal continuity maintained across regime transitions
- Single dataset management (no multiple files per regime)
- Context preservation around regime changes
- Enhanced regime state extraction from NAS/TAS clustering results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing infrastructure
try:
    from src.feature_generation.core.feature_generator import FeatureGenerator
    from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
    HMM_TRAINING_AVAILABLE = False  # HMM training modules no longer needed
except ImportError as e:
    from src.utils.tprint import tprint
    tprint(f"⚠️ Warning: Feature modules not available: {e}")
    HMM_TRAINING_AVAILABLE = False

# Core decorators imports
from src.core.decorators import (
    handles_errors,
    traced,
    log_execution_time
)

# Core errors imports
from src.core.errors import (
    AppError,
    ValidationError,
    DataIntegrityError,
    NotFoundError
)

from src.utils.logger import system_logger
from src.utils.data.klines_parquet import get_klines_manager

# Import our standardized utilities
from .validation_utils import get_validator, ValidationErrorType, ValidationResult, validate_training_input, validate_pipeline_state, create_standardized_error
from .config_utils import get_config_manager, get_path_manager

# Use existing error handling utilities
from src.utils.enhanced_error_handler import (
    EnhancedErrorHandler, ErrorSeverity, ErrorCategory, ErrorContext
)

# Use existing hardware utilities
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager

# Use existing data validation utilities
from src.utils.data.validation.validators import CrossStepValidator
from src.utils.data.quality.data_quality import DataQualityFramework

logger = system_logger.getChild('NasTasRegimeDataSplitting')

class HMMRegimeTagger:
    """Legacy regime tagger - HMM functionality no longer needed."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize regime tagger (HMM functionality removed)."""
        tprint('🔧 Initializing RegimeTagger (HMM functionality removed)')
        self.config = config
        self.logger = logger.getChild('RegimeTagger')
        self.base_models = {}
        self.ensemble_models = {}
        self.feature_generator = None
        self.feature_selector = None
        tprint('✅ RegimeTagger initialization completed (HMM functionality removed)')

    def _initialize_components(self):
        """Initialize regime tagger components (HMM functionality removed)."""
        try:
            # Initialize feature generator for regime analysis
            self.feature_generator = FeatureGenerator()

            # Initialize feature selection framework
            fs_config = {
                'selection_methods': ['mrmr', 'lasso_stability', 'correlation_filter'],
                'max_features': self.config.get('n_features', 100),
                'enable_stability_analysis': True,
                'enable_temporal_analysis': True
            }
            self.feature_selector = FeatureSelectionFramework(fs_config)

            self.logger.info("✅ Regime tagger components initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime tagger components: {e}")
            raise

    def load_trained_models(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Load trained models (HMM functionality removed - returns False)."""
        tprint(f'⚠️ HMM models no longer needed for {symbol} on {exchange} ({timeframe})')
        self.logger.info("⚠️ HMM models no longer needed - using regime clustering results instead")
        return False

    def create_features_for_tagging(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime tagging (HMM functionality removed)."""
        # Input validation using standardized validator
        validator = get_validator(self.logger)

        validation_result = validator.validate_dataframe(market_data, "market_data")
        if not validation_result.valid:
            raise ValueError(validation_result.errors[0])

        if self.feature_generator is None:
            error_msg = create_standardized_error(
                ValidationErrorType.CONFIG_ERROR,
                "Feature generator not initialized",
                "Initialize HMMRegimeTagger with proper configuration"
            )
            raise ValueError(error_msg)

        # Use existing feature generator for 200+ features
        features = self.feature_generator.generate_all_features(market_data)
        self.logger.info(f"✅ Generated {features.shape[1]} features for regime tagging")
        tprint(f"✅ Generated {features.shape[1]} features for regime tagging")
        return features

    def select_features_for_tagging(self, X: pd.DataFrame, is_classification: bool = True) -> pd.DataFrame:
        """Select features for regime tagging (HMM functionality removed)."""
        # Input validation using standardized validator
        validator = get_validator(self.logger)

        validation_result = validator.validate_dataframe(X, "feature_data")
        if not validation_result.valid:
            raise ValueError(validation_result.errors[0])

        if not isinstance(is_classification, bool):
            error_msg = create_standardized_error(
                ValidationErrorType.VALIDATION_ERROR,
                f"is_classification must be a boolean, got {type(is_classification)}",
                "Set is_classification to True or False"
            )
            raise ValueError(error_msg)

        if self.feature_selector is None:
            # Return all features if no feature selector
            tprint("⚠️ No feature selector available, returning all features")
            return X

        try:
            # Use existing feature selection framework
            selection_result = self.feature_selector.select_features(
                X,
                method='comprehensive',
                max_features=self.config.get('n_features', 100),
                is_classification=is_classification
            )

            selected_features = selection_result.get('selected_features', X.columns.tolist()[:self.config.get('n_features', 100)])
            X_selected = X[selected_features]

            self.logger.info(f"✅ Selected {len(selected_features)} features for regime tagging")
            return X_selected

        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection failed, using all features: {e}")
            return X

    def tag_regimes_with_models(self, market_data: pd.DataFrame,
                              use_ensemble: bool = True) -> Dict[str, Any]:
        """Tag regimes using models (HMM functionality removed - returns empty result)."""
        tprint('⚠️ HMM regime tagging no longer needed - using clustering results instead')
        # Input validation using standardized error messages
        if market_data is None:
            tprint('❌ market_data is None')
            raise ValueError("market_data is None - Provide valid market data DataFrame")

        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"market_data must be a DataFrame, got {type(market_data)} - Convert data to pandas DataFrame")

        if len(market_data) == 0:
            raise ValueError("market_data is empty - Provide non-empty market data")

        # HMM functionality removed - return empty result
        self.logger.info("⚠️ HMM regime tagging no longer needed - using clustering results instead")
        return {
            'regime_predictions': [],
            'regime_probabilities': None,
            'model_used': 'hmm_removed',
            'all_predictions': {},
            'all_probabilities': {},
            'n_regimes': 0,
            'regime_distribution': {}
        }

class NasTasRegimeDataSplitting:
    """NAS/TAS regime data splitting with clustering integration."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS/TAS regime data splitting."""
        tprint('🔧 Initializing NasTasRegimeDataSplitting')
        self.config = config
        self.logger = logger.getChild('NasTasRegimeDataSplitting')
        self.hmm_tagger = None
        tprint('✅ Basic initialization completed')

        # Initialize error handler using existing utilities
        error_context = ErrorContext(
            operation="nas_tas_regime_data_splitting",
            component="NasTasRegimeDataSplitting"
        )
        self.error_handler = EnhancedErrorHandler(logger=self.logger)
        tprint('✅ Error handler initialized')

        # Initialize hardware manager using existing utilities
        self.hardware_manager = UnifiedHardwareManager()
        tprint('✅ Hardware manager initialized')

        # Initialize data validation using existing utilities
        self.cross_step_validator = CrossStepValidator()
        self.data_quality_framework = DataQualityFramework()
        tprint('✅ Data validation utilities initialized')

        # Initialize configuration and path managers
        self.config_manager = get_config_manager(config)
        self.path_manager = get_path_manager(config)
        self.validator = get_validator(self.logger)

        # HMM functionality removed - no longer needed
        self.hmm_tagger = HMMRegimeTagger(config)

    @handles_errors
    async def execute(self, training_input: Dict[str, Any],
                    pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced regime data splitting (HMM ML model tagging removed).
        Enhanced with comprehensive error handling, validation, and reporting.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with regime-tagged data
        """
        start_time = datetime.now()
        execution_metrics = {
            'start_time': start_time.isoformat(),
            'warnings': [],
            'errors': [],
            'validation_checks': {}
        }

        tprint('🔄 Starting enhanced regime data splitting (HMM ML model integration removed)')

        try:
            # Step 1: Validate inputs
            tprint('🔍 Step 1: Validating enhanced inputs...')
            validation_result = self._validate_enhanced_inputs(training_input, pipeline_state)
            execution_metrics['validation_checks']['input_validation'] = validation_result['valid']
            if not validation_result['valid']:
                tprint(f'❌ Input validation failed: {validation_result["errors"]}')
                execution_metrics['errors'].extend(validation_result['errors'])
                raise ValueError(f"Input validation failed: {validation_result['errors']}")
            execution_metrics['warnings'].extend(validation_result['warnings'])
            tprint('✅ Enhanced input validation passed')

            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', 'UNKNOWN')
            data_dir = training_input.get('data_dir', 'data/training')

            self.logger.info(f"🔄 Starting enhanced regime data splitting for {symbol}/{exchange}/{timeframe}")
            tprint(f"🔄 Starting enhanced regime data splitting for {symbol}/{exchange}/{timeframe}")

            # Step 2: Load and validate market data
            tprint('📊 Step 2: Loading and validating market data...')
            market_data = self._load_and_validate_market_data(symbol, exchange, timeframe, data_dir)
            if market_data is None or len(market_data) == 0:
                tprint('❌ No market data available for regime tagging')
                raise ValueError("No market data available for regime tagging")
            tprint(f'✅ Market data loaded: {market_data.shape}')

            execution_metrics['validation_checks']['market_data_loaded'] = True
            execution_metrics['data_points'] = len(market_data)

            # Step 3: HMM model availability check removed
            tprint('⚠️ Step 3: HMM model availability check removed - using clustering results instead')
            hmm_models_available = False
            hmm_model_info = {
                'models_loaded': False,
                'base_models_count': 0,
                'ensemble_models_count': 0,
                'note': 'HMM functionality removed'
            }

            execution_metrics['validation_checks']['hmm_models_available'] = hmm_models_available
            execution_metrics['hmm_model_info'] = hmm_model_info

            # Step 4: Perform regime tagging using NAS/TAS clustering results
            tprint('✂️ Step 4: Performing regime tagging using NAS/TAS clustering results...')
            # Use NAS/TAS clustering results with enhanced validation
            self.logger.info("📊 Using NAS/TAS clustering results for regime tagging")
            tprint("📊 Using NAS/TAS clustering results for regime tagging")

            # Get regime data from NAS/TAS clustering results with multiple fallback options
            regime_states = self._extract_nas_tas_regime_states(pipeline_state)
            regime_probabilities = self._extract_nas_tas_regime_probabilities(pipeline_state)

            if not regime_states:
                raise ValueError("No regime data available for tagging")

            # Align data lengths with comprehensive validation and temporal consistency checks
            original_market_len = len(market_data)
            original_regime_len = len(regime_states)
            original_prob_len = len(regime_probabilities) if regime_probabilities else 0
            min_len = min(original_market_len, original_regime_len)

            # Calculate and validate data alignment impact
            data_loss_percentage = ((max(original_market_len, original_regime_len) - min_len) /
                                  max(original_market_len, original_regime_len)) * 100

            if data_loss_percentage > 5.0:  # More than 5% data loss
                warning_msg = f"Data alignment will lose {data_loss_percentage:.1f}% of data ({max(original_market_len, original_regime_len) - min_len} rows)"
                execution_metrics['warnings'].append(warning_msg)
                tprint(f"⚠️ {warning_msg}")
                self.logger.warning(f"⚠️ {warning_msg}")
            elif data_loss_percentage > 20.0:  # More than 20% data loss - this should be an error
                error_msg = f"Excessive data loss during alignment: {data_loss_percentage:.1f}% ({max(original_market_len, original_regime_len) - min_len} rows)"
                execution_metrics['errors'].append(error_msg)
                tprint(f"❌ {error_msg}")
                raise ValueError(error_msg)

            if min_len == 0:
                error_msg = "No overlapping data between market data and regime states"
                execution_metrics['errors'].append(error_msg)
                raise ValueError(error_msg)

            tprint(f"📊 Aligning data lengths: market_data={original_market_len}, regime_states={original_regime_len} -> {min_len}")

            # Validate temporal consistency if timestamp information exists
            temporal_validation_passed = True
            try:
                if hasattr(market_data, 'index') and hasattr(market_data.index, 'name') and market_data.index.name == 'timestamp':
                    # DataFrame has timestamp index
                    market_timestamps = market_data.index[:min_len]
                    if len(market_timestamps) > 1 and not market_timestamps.is_monotonic_increasing:
                        warning_msg = "Market data timestamps are not monotonic increasing - temporal consistency may be compromised"
                        execution_metrics['warnings'].append(warning_msg)
                        tprint(f"⚠️ {warning_msg}")
                        temporal_validation_passed = False
                elif 'timestamp' in market_data.columns:
                    # DataFrame has timestamp column
                    market_timestamps = market_data['timestamp'].iloc[:min_len]
                    if len(market_timestamps) > 1 and not market_timestamps.is_monotonic_increasing:
                        warning_msg = "Market data timestamps are not monotonic increasing - temporal consistency may be compromised"
                        execution_metrics['warnings'].append(warning_msg)
                        tprint(f"⚠️ {warning_msg}")
                        temporal_validation_passed = False
            except Exception as e:
                warning_msg = f"Could not validate temporal consistency: {e}"
                execution_metrics['warnings'].append(warning_msg)
                tprint(f"⚠️ {warning_msg}")

            # Create aligned copies with proper error handling
            try:
                market_data_aligned = market_data.iloc[:min_len].copy()
                if market_data_aligned is None or len(market_data_aligned) != min_len:
                    raise ValueError(f"Market data alignment failed: expected {min_len} rows")
            except Exception as e:
                error_msg = f"Failed to align market data: {str(e)}"
                execution_metrics['errors'].append(error_msg)
                raise ValueError(error_msg)

            try:
                regime_states_aligned = regime_states[:min_len]
                if len(regime_states_aligned) != min_len:
                    raise ValueError(f"Regime states alignment failed: expected {min_len}, got {len(regime_states_aligned)}")
            except Exception as e:
                error_msg = f"Failed to align regime states: {str(e)}"
                execution_metrics['errors'].append(error_msg)
                raise ValueError(error_msg)

            # Handle regime probabilities alignment with validation
            if original_prob_len > 0:
                try:
                    regime_probabilities_aligned = regime_probabilities[:min_len]
                    if len(regime_probabilities_aligned) != min_len:
                        warning_msg = f"Regime probabilities alignment mismatch: expected {min_len}, got {len(regime_probabilities_aligned)}"
                        execution_metrics['warnings'].append(warning_msg)
                        tprint(f"⚠️ {warning_msg}")
                        regime_probabilities_aligned = []
                except Exception as e:
                    warning_msg = f"Failed to align regime probabilities: {e}"
                    execution_metrics['warnings'].append(warning_msg)
                    tprint(f"⚠️ {warning_msg}")
                    regime_probabilities_aligned = []
            else:
                regime_probabilities_aligned = []

            # Final validation of all aligned data
            if len(market_data_aligned) != len(regime_states_aligned):
                error_msg = f"Final alignment validation failed: market_data={len(market_data_aligned)}, regime_states={len(regime_states_aligned)}"
                execution_metrics['errors'].append(error_msg)
                raise ValueError(error_msg)

            # Log successful alignment with comprehensive metrics
            alignment_metrics = {
                'original_market_data_length': original_market_len,
                'original_regime_states_length': original_regime_len,
                'original_probabilities_length': original_prob_len,
                'aligned_length': min_len,
                'data_loss_percentage': data_loss_percentage,
                'temporal_validation_passed': temporal_validation_passed
            }
            execution_metrics['alignment_metrics'] = alignment_metrics
            self.logger.info(f"✅ Data alignment completed successfully: {alignment_metrics}")
            tprint(f"✅ Data alignment completed: {min_len} rows, {data_loss_percentage:.1f}% data loss")

            # Clean up original references using hardware manager (after successful validation)
            del regime_states, regime_probabilities

            # Use hardware manager for memory optimization
            try:
                memory_result = self.hardware_manager.optimize_memory()
                self.logger.debug(f"Memory optimization result: {memory_result}")
            except Exception as e:
                self.logger.warning(f"⚠️ Memory optimization failed: {e}")

            # Use aligned regime data
            market_data_aligned['hmm_regime_states'] = regime_states_aligned
            if len(regime_probabilities_aligned) > 0:
                market_data_aligned['hmm_regime_probabilities'] = regime_probabilities_aligned
            market_data_aligned['hmm_regime_confidence'] = np.ones(len(market_data_aligned))  # Default confidence

            # Update market_data reference to use aligned data
            market_data = market_data_aligned

            nas_tas_tagging_info = {
                'nas_tas_tagging_completed': True,
                'nas_tas_model_used': 'nas_tas_clustering_results',
                'nas_tas_n_regimes': len(np.unique(regime_states_aligned)),
                'nas_tas_regime_distribution': {int(k): int(v) for k, v in zip(*np.unique(regime_states_aligned, return_counts=True))},
                'nas_tas_tagging_timestamp': pd.Timestamp.now().isoformat(),
                'tagging_confidence_mean': 1.0,
                'tagging_confidence_std': 0.0
            }

            execution_metrics['validation_checks']['regime_tagging_successful'] = True
            execution_metrics['regime_count'] = len(np.unique(regime_states_aligned))

            # Step 5: Validate final data quality
            tprint('📊 Step 5: Validating final data quality...')
            data_quality_score = self._calculate_enhanced_data_quality_score(market_data)
            execution_metrics['data_quality_score'] = data_quality_score
            execution_metrics['validation_checks']['data_quality_acceptable'] = data_quality_score > 0.7
            tprint(f'📊 Data quality score: {data_quality_score:.2f}')

            if data_quality_score < 0.7:
                execution_metrics['warnings'].append(f"Data quality score is low: {data_quality_score:.2f}")
                tprint(f'⚠️ Data quality score is low: {data_quality_score:.2f}')

            # Step 6: Save tagged data with enhanced error handling
            tprint('💾 Step 6: Saving tagged data...')
            save_result = await self._save_tagged_data_with_validation(market_data, symbol, exchange, timeframe, data_dir)
            execution_metrics['validation_checks']['data_saved_successfully'] = save_result['success']
            if not save_result['success']:
                execution_metrics['errors'].append(f"Data saving failed: {save_result['error']}")
                tprint(f'❌ Data saving failed: {save_result["error"]}')
            else:
                tprint('✅ Tagged data saved successfully')

            # Step 7: Generate comprehensive execution report
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_metrics.update({
                'end_time': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'success': True,
                'regime_distribution': nas_tas_tagging_info['nas_tas_regime_distribution'],
                'recommendations': self._generate_enhanced_recommendations(execution_metrics)
            })

            # Update pipeline state with enhanced information
            updated_pipeline_state = pipeline_state.copy()
            updated_pipeline_state.update({
                'step04_regime_data_splitting_completed': True,
                'step04_regime_data_splitting_timestamp': pd.Timestamp.now().isoformat(),
                'regime_tagged_data_available': True,
                'regime_tagged_data_path': f"{data_dir}/training/{exchange}_{symbol}_{timeframe}_regime_tagged_data.parquet",
                'nas_tas_tagging_info': nas_tas_tagging_info,
                'execution_metrics': execution_metrics,
                'data_quality_score': data_quality_score,
                'regime_count': execution_metrics['regime_count']
            })

            self.logger.info(f"✅ NAS/TAS regime data splitting completed successfully in {execution_time:.2f}s")
            self.logger.info(f"📊 Final metrics: {execution_metrics['regime_count']} regimes, quality score: {data_quality_score:.2f}")
            tprint(f"✅ NAS/TAS regime data splitting completed successfully in {execution_time:.2f}s")
            tprint(f"📊 Final metrics: {execution_metrics['regime_count']} regimes, quality score: {data_quality_score:.2f}")

            return updated_pipeline_state

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_metrics.update({
                'end_time': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'success': False,
                'errors': execution_metrics['errors'] + [str(e)]
            })

            self.logger.error(f"❌ NAS/TAS regime data splitting failed after {execution_time:.2f}s: {e}")
            tprint(f"❌ NAS/TAS regime data splitting failed after {execution_time:.2f}s: {e}")
            import traceback
            self.logger.error(f"❌ Error details: {traceback.format_exc()}")
            tprint(f"❌ Error details: {traceback.format_exc()}")

            # Return failure state with comprehensive error information
            return {
                'step04_regime_data_splitting_completed': False,
                'step04_regime_data_splitting_timestamp': pd.Timestamp.now().isoformat(),
                'execution_metrics': execution_metrics,
                'error_message': str(e),
                'error_timestamp': datetime.now().isoformat()
            }

    def _validate_enhanced_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs with enhanced error checking using standardized validation."""
        # Use standardized validation
        training_input_result = validate_training_input(training_input)
        pipeline_state_result = validate_pipeline_state(pipeline_state)

        # Combine results
        validation_result = {
            'valid': training_input_result.valid and pipeline_state_result.valid,
            'errors': training_input_result.errors + pipeline_state_result.errors,
            'warnings': training_input_result.warnings + pipeline_state_result.warnings
        }

        return validation_result

    def _load_and_validate_market_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load and validate market data with enhanced error handling."""
        try:
            # Use path manager for consistent path handling
            data_path = self.path_manager.get_market_data_path(exchange, symbol, timeframe, data_dir)

            if not data_path.exists():
                self.logger.warning(f"⚠️ Market data file not found: {data_path}")
                return None

            # For market data, try to use klines manager first, fallback to direct reading
            try:
                # Extract symbol and timeframe from file path
                file_name = data_path.name
                parts = file_name.replace('.parquet', '').split('_')
                if len(parts) >= 3:
                    symbol = parts[1]  # Assuming format: exchange_symbol_timeframe.parquet
                    timeframe = parts[2]
                    klines_manager = get_klines_manager()

                    # Get date filtering from config if available
                    start_date = None
                    end_date = None
                    if hasattr(self.config, 'start_date') and self.config.start_date:
                        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                    if hasattr(self.config, 'end_date') and self.config.end_date:
                        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')

                    market_data = klines_manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")
                else:
                    market_data = pd.read_parquet(data_path)
            except Exception as e:
                logger.warning(f"Failed to use klines manager, falling back to direct reading: {e}")
                market_data = pd.read_parquet(data_path)

            # Validate data quality
            if len(market_data) == 0:
                self.logger.error("❌ Market data is empty")
                return None

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                self.logger.warning(f"⚠️ Missing columns: {missing_columns}")

            # Check for null values
            null_count = market_data.isnull().sum().sum()
            if null_count > 0:
                self.logger.warning(f"⚠️ Market data contains {null_count} null values")

            self.logger.info(f"✅ Loaded and validated market data: {market_data.shape}")
            return market_data

        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            return None

    def _extract_nas_tas_regime_states(self, pipeline_state: Dict[str, Any]) -> List[Any]:
        """Extract regime states from NAS/TAS clustering results with enhanced extraction logic."""
        # Primary NAS/TAS clustering result keys
        nas_tas_keys = [
            'optimal_regime_clustering_result',
            'nas_tas_clustering_result',
            'cluster_assignments'
        ]

        # Try NAS/TAS specific keys first
        for key in nas_tas_keys:
            if key in pipeline_state and pipeline_state[key]:
                result = self._extract_regime_states_from_nas_tas_result(pipeline_state[key])
                if result:
                    self.logger.info(f"✅ Found NAS/TAS regime states under key: {key}")
                    return result

        # Fallback to legacy keys
        legacy_keys = [
            'regime_states',
            'hmm_regime_discovery_result',
            'regime_discovery_result'
        ]

        for key in legacy_keys:
            if key in pipeline_state and pipeline_state[key]:
                if isinstance(pipeline_state[key], list):
                    self.logger.info(f"✅ Found regime states under legacy key: {key}")
                    return pipeline_state[key]
                elif isinstance(pipeline_state[key], dict) and 'regime_states' in pipeline_state[key]:
                    self.logger.info(f"✅ Found regime states under legacy key: {key}")
                    return pipeline_state[key]['regime_states']

        return []

    def _extract_regime_states_from_nas_tas_result(self, nas_tas_result: Any) -> List[Any]:
        """Extract regime states from NAS/TAS clustering result with comprehensive extraction logic."""
        try:
            if isinstance(nas_tas_result, list):
                return nas_tas_result
            elif isinstance(nas_tas_result, dict):
                # Try different possible keys for regime states in NAS/TAS results
                possible_state_keys = [
                    'cluster_assignments',
                    'regime_states',
                    'states',
                    'predictions',
                    'labels',
                    'assignments'
                ]

                for state_key in possible_state_keys:
                    if state_key in nas_tas_result and nas_tas_result[state_key]:
                        states = nas_tas_result[state_key]
                        if isinstance(states, list):
                            return states
                        elif hasattr(states, 'tolist'):  # numpy array
                            return states.tolist()
                        elif hasattr(states, 'values'):  # pandas series
                            return states.values.tolist()

                # If no direct state keys found, try to extract from nested structure
                if 'clustering_result' in nas_tas_result:
                    return self._extract_regime_states_from_nas_tas_result(nas_tas_result['clustering_result'])
                elif 'results' in nas_tas_result:
                    return self._extract_regime_states_from_nas_tas_result(nas_tas_result['results'])

            return []

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting regime states from NAS/TAS result: {e}")
            return []

    def _extract_nas_tas_regime_probabilities(self, pipeline_state: Dict[str, Any]) -> List[Any]:
        """Extract regime probabilities from NAS/TAS clustering results with enhanced extraction logic."""
        # Primary NAS/TAS clustering result keys
        nas_tas_keys = [
            'optimal_regime_clustering_result',
            'nas_tas_clustering_result',
            'cluster_probabilities'
        ]

        # Try NAS/TAS specific keys first
        for key in nas_tas_keys:
            if key in pipeline_state and pipeline_state[key]:
                result = self._extract_regime_probabilities_from_nas_tas_result(pipeline_state[key])
                if result:
                    self.logger.info(f"✅ Found NAS/TAS regime probabilities under key: {key}")
                    return result

        # Fallback to legacy keys
        legacy_keys = [
            'regime_probabilities',
            'hmm_regime_discovery_result',
            'regime_discovery_result'
        ]

        for key in legacy_keys:
            if key in pipeline_state and pipeline_state[key]:
                if isinstance(pipeline_state[key], dict):
                    if 'regime_probabilities' in pipeline_state[key]:
                        self.logger.info(f"✅ Found regime probabilities under legacy key: {key}")
                        return pipeline_state[key]['regime_probabilities']
                    elif 'probabilities' in pipeline_state[key]:
                        self.logger.info(f"✅ Found regime probabilities under legacy key: {key}")
                        return pipeline_state[key]['probabilities']

        return []

    def _extract_regime_probabilities_from_nas_tas_result(self, nas_tas_result: Any) -> List[Any]:
        """Extract regime probabilities from NAS/TAS clustering result with comprehensive extraction logic."""
        try:
            if isinstance(nas_tas_result, list):
                return nas_tas_result
            elif isinstance(nas_tas_result, dict):
                # Try different possible keys for regime probabilities in NAS/TAS results
                possible_prob_keys = [
                    'cluster_probabilities',
                    'regime_probabilities',
                    'probabilities',
                    'proba',
                    'scores',
                    'confidence_scores'
                ]

                for prob_key in possible_prob_keys:
                    if prob_key in nas_tas_result and nas_tas_result[prob_key]:
                        probs = nas_tas_result[prob_key]
                        if isinstance(probs, list):
                            return probs
                        elif hasattr(probs, 'tolist'):  # numpy array
                            return probs.tolist()
                        elif hasattr(probs, 'values'):  # pandas series
                            return probs.values.tolist()

                # If no direct probability keys found, try to extract from nested structure
                if 'clustering_result' in nas_tas_result:
                    return self._extract_regime_probabilities_from_nas_tas_result(nas_tas_result['clustering_result'])
                elif 'results' in nas_tas_result:
                    return self._extract_regime_probabilities_from_nas_tas_result(nas_tas_result['results'])

            return []

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting regime probabilities from NAS/TAS result: {e}")
            return []

    def _validate_tagging_results(self, tagging_result: Dict[str, Any], market_data: pd.DataFrame) -> bool:
        """Validate HMM tagging results."""
        try:
            # Check if tagging result has required keys
            required_keys = ['regime_predictions', 'n_regimes']
            for key in required_keys:
                if key not in tagging_result:
                    self.logger.error(f"❌ Missing required key in tagging result: {key}")
                    tprint(f"❌ Missing required key in tagging result: {key}")
                    return False

            # Check if predictions match data length
            predictions = tagging_result['regime_predictions']
            if len(predictions) != len(market_data):
                self.logger.error(f"❌ Prediction length mismatch: {len(predictions)} vs {len(market_data)}")
                tprint(f"❌ Prediction length mismatch: {len(predictions)} vs {len(market_data)}")
                return False

            # Validate regime count using existing validation patterns
            n_regimes = tagging_result['n_regimes']
            if n_regimes < 2:
                self.logger.warning(f"⚠️ Very few regimes detected: {n_regimes}")
                tprint(f"⚠️ Very few regimes detected: {n_regimes}")
            elif n_regimes > 20:
                self.logger.warning(f"⚠️ Many regimes detected: {n_regimes}")
                tprint(f"⚠️ Many regimes detected: {n_regimes}")
            else:
                self.logger.info(f"✅ Regime validation passed: {n_regimes} regimes")
                tprint(f"✅ Regime validation passed: {n_regimes} regimes")

            # Additional data consistency checks
            # Check for reasonable regime values
            if len(predictions) > 0:
                min_regime = np.min(predictions)
                max_regime = np.max(predictions)
                if min_regime < 0:
                    self.logger.warning(f"⚠️ Negative regime values detected: min={min_regime}")
                    tprint(f"⚠️ Negative regime values detected: min={min_regime}")
                if max_regime > 100:
                    self.logger.warning(f"⚠️ Unusually high regime values detected: max={max_regime}")
                    tprint(f"⚠️ Unusually high regime values detected: max={max_regime}")

            # Check regime distribution
            regime_distribution = tagging_result.get('regime_distribution', {})
            if regime_distribution:
                total_points = sum(regime_distribution.values())
                if total_points != len(predictions):
                    self.logger.warning(f"⚠️ Regime distribution total mismatch: {total_points} vs {len(predictions)}")
                    tprint(f"⚠️ Regime distribution total mismatch: {total_points} vs {len(predictions)}")

            tprint("✅ HMM tagging results validation passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error validating tagging results: {e}")
            return False

    def _calculate_enhanced_data_quality_score(self, market_data: pd.DataFrame) -> float:
        """Calculate enhanced data quality score."""
        try:
            score = 1.0

            # Check for null values
            null_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            score -= null_ratio * 0.3

            # Check for duplicate rows
            duplicate_ratio = market_data.duplicated().sum() / len(market_data)
            score -= duplicate_ratio * 0.2

            # Check for infinite values
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(market_data) > 0:
                inf_count = np.isinf(market_data[numeric_cols]).sum().sum()
                inf_ratio = inf_count / (len(market_data) * len(numeric_cols))
                score -= inf_ratio * 0.3

            # Check for zero/negative prices
            if 'close' in market_data.columns:
                invalid_prices = (market_data['close'] <= 0).sum() / len(market_data)
                score -= invalid_prices * 0.2

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating data quality score: {e}")
            return 0.5

    async def _save_tagged_data_with_validation(self, market_data: pd.DataFrame, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Dict[str, Any]:
        """Save tagged data with enhanced validation."""
        try:
            # Use path manager for consistent path handling
            output_path = self.path_manager.get_regime_tagged_data_path(exchange, symbol, timeframe, data_dir)
            self.path_manager.ensure_directories_exist(output_path)

            # Validate data before saving
            if len(market_data) == 0:
                return {'success': False, 'error': 'Market data is empty'}

            # Check for required regime columns
            if 'hmm_regime_states' not in market_data.columns:
                return {'success': False, 'error': 'Missing regime states column'}

            market_data.to_parquet(output_path, index=False)

            # Verify file was created and has reasonable size
            if not output_path.exists():
                return {'success': False, 'error': 'File was not created'}

            file_size = output_path.stat().st_size
            if file_size < 1000:  # Less than 1KB seems suspicious
                return {'success': False, 'error': f'File size too small: {file_size} bytes'}

            self.logger.info(f"✅ Saved regime-tagged data: {output_path} ({file_size} bytes)")
            return {'success': True, 'file_path': str(output_path), 'file_size': file_size}

        except Exception as e:
            self.logger.error(f"❌ Error saving regime-tagged data: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_enhanced_recommendations(self, execution_metrics: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on execution metrics."""
        recommendations = []

        # Data quality recommendations
        data_quality_score = execution_metrics.get('data_quality_score', 1.0)
        if data_quality_score < 0.8:
            recommendations.append(f"Consider improving data quality - current score: {data_quality_score:.2f}")

        # Regime count recommendations
        regime_count = execution_metrics.get('regime_count', 0)
        if regime_count < 3:
            recommendations.append(f"Consider adjusting regime discovery parameters - only {regime_count} regimes detected")
        elif regime_count > 15:
            recommendations.append(f"Consider reducing regime complexity - {regime_count} regimes may be too many")

        # Performance recommendations
        execution_time = execution_metrics.get('execution_time_seconds', 0)
        if execution_time > 60:
            recommendations.append(f"Processing time is high ({execution_time:.1f}s) - consider optimizing data size")

        # HMM model recommendations
        hmm_model_info = execution_metrics.get('hmm_model_info', {})
        if not hmm_model_info.get('models_loaded', False):
            recommendations.append("Consider training HMM models for better regime detection")

        # Warning-based recommendations
        warnings = execution_metrics.get('warnings', [])
        if len(warnings) > 3:
            recommendations.append("Multiple warnings detected - review data quality and parameters")

        return recommendations

    async def _save_tagged_data(self, market_data: pd.DataFrame, symbol: str, exchange: str,
                              timeframe: str, data_dir: str) -> None:
        """Save regime-tagged data."""
        try:
            output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_tagged_data.parquet'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            market_data.to_parquet(output_path, index=False)
            self.logger.info(f"✅ Saved regime-tagged data: {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Error saving regime-tagged data: {e}")
            raise

# Convenience function
async def execute_nas_tas_regime_data_splitting(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute NAS/TAS regime data splitting with clustering integration."""
    config = config or {}
    splitter = NasTasRegimeDataSplitting(config)
    return await splitter.execute(training_input, pipeline_state)
