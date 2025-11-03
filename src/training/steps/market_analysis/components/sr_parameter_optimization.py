"""
Enhanced SR Parameter Optimization Step.

This step optimizes Support/Resistance detection parameters using advanced techniques:
- VectorBT optimization for efficient parameter testing
- Bayesian HPO with staged optimization (coarse grid -> fine grid -> TPE)
- Hardware-aware optimization for M1 Mac performance
- Advanced validation with purged CV and data leakage detection
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# VectorBT imports
try:
    # Import from src.vectorbt instead of direct vectorbt import
    from src.utils.vectorbt_compat import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE as VBT_AVAILABLE
    )
    VECTORBT_AVAILABLE = VBT_AVAILABLE
    # Try to import RollingOptimizer if available
    try:
        from vectorbt.optimization import RollingOptimizer
        VECTORBT_ROLLING_AVAILABLE = True
    except (ImportError, AttributeError):
        VECTORBT_ROLLING_AVAILABLE = False
        RollingOptimizer = None
except ImportError:
    VECTORBT_AVAILABLE = False
    VECTORBT_ROLLING_AVAILABLE = False
    vbt = None
    RollingOptimizer = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None

# Hardware and vectorization imports
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.vectorization_manager import UnifiedVectorizationManager
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    UnifiedHardwareManager = None
    UnifiedVectorizationManager = None

# TPrint imports
try:
    from src.utils.tprint import tprint, tprint_data_preview, tprint_data_format
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    tprint = print
    tprint_data_preview = lambda x, y: None
    tprint_data_format = lambda x, y: None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

# Enhanced optimization imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
        HierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage,
        create_param_group
    )
    BAYESIAN_HPO_AVAILABLE = True
    HIERARCHICAL_HPO_AVAILABLE = True
except ImportError as e:
    BAYESIAN_HPO_AVAILABLE = False
    HIERARCHICAL_HPO_AVAILABLE = False
    print(f"Warning: Bayesian HPO not available: {e}")

try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    VECTORIZATION_AVAILABLE = True
except ImportError as e:
    VECTORIZATION_AVAILABLE = False
    print(f"Warning: Vectorization manager not available: {e}")

try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    print(f"Warning: Hardware optimization not available: {e}")

try:
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    print(f"Warning: Advanced validation not available: {e}")

try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError as e:
    VECTORBT_ROLLING_AVAILABLE = False
    print(f"Warning: VectorBT rolling optimizer not available: {e}")

# Enhanced ML utilities imports
try:
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEExplainer, ExplanationConfig, create_explainer
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    EXPLAINABILITY_AVAILABLE = False
    print(f"Warning: SHAP/LIME explainability not available: {e}")

try:
    from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import (
        OOFStackingEnsembleManager, OOFStackingEnsembleConfig
    )
    OOF_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    OOF_ENSEMBLE_AVAILABLE = False
    OOFStackingEnsembleConfig = None
    OOFStackingEnsembleManager = None
    print(f"Warning: OOF ensemble not available: {e}")

try:
    from src.utils.purged_kfold import PurgedKFold
    PURGED_CV_AVAILABLE = True
except ImportError as e:
    PURGED_CV_AVAILABLE = False
    print(f"Warning: Purged CV not available: {e}")

try:
    from src.utils.ml_common.evaluation.unified_evaluator import (
        UnifiedEvaluator, EvaluationConfig
    )
    EVALUATION_AVAILABLE = True
except ImportError as e:
    EVALUATION_AVAILABLE = False
    print(f"Warning: Unified evaluator not available: {e}")

# Additional imports for hardware detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import SR clustering components
try:
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
    from src.utils.sr_clustering.parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
    SR_CLUSTERING_AVAILABLE = True
except ImportError as e:
    SR_CLUSTERING_AVAILABLE = False
    SRBacktestingEngine = None
    BacktestConfig = None
    get_parameter_optimization_engine = None
    ParameterOptimizationConfig = None
    print(f"Warning: SR clustering components not available: {e}")

# Import SR detection for parameter testing
try:
    from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector, SRLevel
    SR_DETECTION_AVAILABLE = True
except ImportError as e:
    SR_DETECTION_AVAILABLE = False
    EnhancedSRDetector = None
    SRLevel = None
    print(f"Warning: SR detection not available for parameter testing: {e}")

@dataclass
class StrengthWeights:
    """Optimizable weights for SR level strength calculation."""
    # Positive boosts
    touch_weight: float = 0.1          # Touch boost weight
    volume_weight: float = 0.2         # Volume confirmation weight
    consistency_weight: float = 0.2    # Consistency weight
    confluence_weight: float = 0.1     # Confluence weight
    pivot_boost: float = 0.1           # Pivot level boost
    psychological_boost: float = 0.05  # Psychological level boost
    hvn_boost: float = 0.1             # High Volume Node boost
    
    # Negative penalties (failures/breakouts)
    failure_penalty_base: float = 0.2           # Base penalty per failure
    failure_volume_multiplier: float = 1.5      # Volume scaling (2.0 - volume_factor) 
    failure_max_penalty: float = 0.6            # Maximum total penalty cap


@dataclass
class EnhancedSRConfig:
    """Enhanced configuration for SR parameter optimization with advanced ML utilities."""
    # Optimization settings
    enable_bayesian_hpo: bool = False  # Disabled in favor of hierarchical (faster + better)
    enable_hierarchical_hpo: bool = True  # DEFAULT: Use hierarchical (recommended for 4+ params)
    enable_strength_weight_optimization: bool = True  # NEW: Optimize strength weights via HPO
    enable_vectorbt_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_advanced_validation: bool = True
    
    # Hierarchical HPO settings (Coarse → Fine → TPE strategy)
    n_trials: int = 120  # Total trials across all stages
    enable_staged_optimization: bool = True
    coarse_grid_points: int = 4   # Coarse grid: 4 points per param
    fine_grid_points: int = 6      # Fine grid: 6 points per param (denser)
    tpe_trials: int = 50           # TPE Bayesian optimization trials
    
    # Strength weight optimization settings
    strength_weight_trials: int = 60  # Trials for strength weight optimization
    strength_optimization_metric: str = 'spearman_correlation'  # Metric to optimize
    
    # Hardware optimization settings
    workload_type: str = 'BACKTESTING'
    optimization_level: str = 'BALANCED'
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    enable_m1_optimization: bool = True
    
    # Validation settings
    enable_purged_cv: bool = True
    enable_data_leakage_detection: bool = True
    temporal_gap_hours: int = 24
    validation_gap_days: int = 5
    enable_temporal_validation: bool = True
    
    # Enhanced ML utilities settings
    enable_explainability: bool = True
    enable_oof_validation: bool = True
    enable_unified_evaluation: bool = True
    
    # SHAP/LIME settings
    shap_sample_size: int = 1000
    lime_sample_size: int = 500
    max_features_shap: int = 20
    max_features_lime: int = 10
    
    # OOF/Ensemble settings
    oof_n_splits: int = 5
    oof_test_size: float = 0.2
    oof_gap_days: int = 3
    
    # Purged CV settings
    purged_cv_n_splits: int = 5
    purged_cv_pct_embargo: float = 0.01
    
    # VectorBT settings
    prefer_vectorbt: bool = True
    vectorbt_rolling_window: int = 1000
    vectorbt_chunk_size: int = 10000
    
    # Performance settings
    enable_caching: bool = True
    cache_dir: str = "cache/sr_optimization"
    parallel_processing: bool = True
    max_workers: int = 4

class SRParameterOptimizationStep(BaseStep):
    """
    Enhanced SR Parameter Optimization Step.

    Optimizes Support/Resistance detection parameters using advanced techniques:
    - VectorBT optimization for efficient parameter testing
    - Bayesian HPO with staged optimization
    - Hardware-aware optimization for M1 Mac performance
    - Advanced validation with purged CV and data leakage detection
    """

    def __init__(self, step_name: str = "sr_parameter_optimization"):
        """Initialize the enhanced SR parameter optimization step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRParameterOptimization')
        
        # Initialize enhanced optimization components
        self._initialize_optimization_components()

    def _initialize_optimization_components(self):
        """Initialize enhanced optimization components."""
        self.logger.info("🚀 Initializing enhanced optimization components...")
        
        # Initialize Bayesian HPO optimizer
        if BAYESIAN_HPO_AVAILABLE:
            self.bayesian_optimizer = BayesianTPEOptimizer()
            self.logger.info("✅ Bayesian HPO optimizer initialized")
        else:
            self.bayesian_optimizer = None
            self.logger.warning("⚠️ Bayesian HPO optimizer not available")
        
        # Initialize vectorization manager
        if VECTORIZATION_AVAILABLE:
            self.vectorization_manager = UnifiedVectorizationManager()
            self.logger.info("✅ Vectorization manager initialized")
        else:
            self.vectorization_manager = None
            self.logger.warning("⚠️ Vectorization manager not available")
        
        # Initialize hardware manager
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.hardware_manager = UnifiedHardwareManager()
            self.logger.info("✅ Hardware manager initialized")
        else:
            self.hardware_manager = None
            self.logger.warning("⚠️ Hardware manager not available")
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            self.logger.info("✅ VectorBT rolling optimizer initialized")
        else:
            self.vectorbt_optimizer = None
            self.logger.warning("⚠️ VectorBT rolling optimizer not available")
        
        # Initialize validation components
        if VALIDATION_AVAILABLE:
            self.leakage_detector = DataLeakageDetector()
            self.logger.info("✅ Data leakage detector initialized")
        else:
            self.leakage_detector = None
            self.logger.warning("⚠️ Advanced validation not available")
        
        # Initialize explainability components
        if EXPLAINABILITY_AVAILABLE:
            self.explainability_config = ExplanationConfig(
                enable_shap=True,
                enable_lime=True,
                shap_sample_size=1000,
                lime_sample_size=500
            )
            self.explainer = create_explainer(self.explainability_config)
            self.logger.info("✅ SHAP/LIME explainability initialized")
        else:
            self.explainer = None
            self.logger.warning("⚠️ SHAP/LIME explainability not available")
        
        # Initialize OOF ensemble components
        if OOF_ENSEMBLE_AVAILABLE:
            self.oof_config = OOFStackingEnsembleConfig(
                ensemble_name="sr_parameter_optimization",
                output_dir="models/sr_optimization",
                cv_folds=5,
                enable_temporal_validation=True,
                purge_periods=3
            )
            self.oof_manager = OOFStackingEnsembleManager(self.oof_config)
            self.logger.info("✅ OOF ensemble manager initialized")
        else:
            self.oof_manager = None
            self.logger.warning("⚠️ OOF ensemble manager not available")
        
        # Initialize unified evaluator
        if EVALUATION_AVAILABLE:
            self.evaluation_config = EvaluationConfig(
                enable_time_series_metrics=True,
                enable_financial_metrics=True,
                enable_risk_metrics=True,
                enable_drawdown_analysis=True
            )
            self.evaluator = UnifiedEvaluator(self.evaluation_config)
            self.logger.info("✅ Unified evaluator initialized")
        else:
            self.evaluator = None
            self.logger.warning("⚠️ Unified evaluator not available")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this step must produce."""
        return ['sr_parameter_optimization_result']
    
    def get_required_input_artifacts(self) -> List[str]:
        """Get list of required input artifacts this step needs from previous steps."""
        return ['sr_clustering_result', 'sr_levels_dictionary']

    async def execute(self, config: Dict[str, Any], enhanced_config: EnhancedSRConfig = None) -> Dict[str, Any]:
        """
        Execute enhanced SR parameter optimization with advanced techniques.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
                - enable_bayesian_hpo: Enable Bayesian optimization (default: True)
                - enable_vectorbt: Enable VectorBT optimization (default: True)
                - enable_hardware_optimization: Enable hardware optimization (default: True)
            enhanced_config: Optional EnhancedSRConfig instance with custom optimization settings.
                           If provided, uses these settings instead of defaults.
                           If None, creates default config and overrides from config dict.

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        self.logger.info('🎯 Starting Enhanced SR Parameter Optimization')
        tprint("🚀 SR Parameter Optimization: Starting execution", "info")

        try:
            # Fetch required input artifacts from previous steps
            tprint("📥 Fetching input artifacts", "info")
            input_artifacts = await self._fetch_input_artifacts(config)
            tprint_data_preview(input_artifacts, "Input artifacts")
            if not input_artifacts['success']:
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': f"Failed to fetch required input artifacts: {input_artifacts['error']}"
                }
            
            # Use provided enhanced configuration or create default
            if enhanced_config is None:
                self.logger.info("📊 Creating default enhanced configuration")
                enhanced_config = EnhancedSRConfig()
                
                # Override with user config if provided
                if 'enable_bayesian_hpo' in config:
                    enhanced_config.enable_bayesian_hpo = config['enable_bayesian_hpo']
                if 'enable_hierarchical_hpo' in config:
                    enhanced_config.enable_hierarchical_hpo = config['enable_hierarchical_hpo']
                if 'enable_vectorbt' in config:
                    enhanced_config.enable_vectorbt_optimization = config['enable_vectorbt']
                if 'enable_hardware_optimization' in config:
                    enhanced_config.enable_hardware_optimization = config['enable_hardware_optimization']
            else:
                self.logger.info("✅ Using provided enhanced configuration")
                self.logger.info(f"   - n_trials: {enhanced_config.n_trials}")
                self.logger.info(f"   - coarse_grid_points: {enhanced_config.coarse_grid_points}")
                self.logger.info(f"   - fine_grid_points: {enhanced_config.fine_grid_points}")
                self.logger.info(f"   - tpe_trials: {enhanced_config.tpe_trials}")
                self.logger.info(f"   - optimization_level: {enhanced_config.optimization_level}")
                self.logger.info(f"   - max_workers: {enhanced_config.max_workers}")
                self.logger.info(f"   - hierarchical_hpo: {enhanced_config.enable_hierarchical_hpo}")
                self.logger.info(f"   - strength_weight_optimization: {enhanced_config.enable_strength_weight_optimization}")

            # Get and validate market data
            market_data = await self._load_market_data(config)
            if not self._validate_market_data(market_data):
                error_msg = "Invalid market data for parameter optimization"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }

            # Ensure data has proper datetime indexing for backtesting
            market_data = self._prepare_data_for_backtesting(market_data)

            # Run enhanced parameter optimization with input artifacts
            optimization_result = await self._run_enhanced_parameter_optimization(
                market_data, enhanced_config, config, input_artifacts['artifacts']
            )

            # Handle None optimization_result
            if optimization_result is None:
                self.logger.error("❌ Optimization returned None result")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': "Optimization returned None result"
                }
            
            # Extract results
            optimized_parameters = optimization_result.get('optimized_parameters', {})
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            parameter_optimization_metrics = optimization_result.get('parameter_optimization_metrics', {})

            # Validate that we have the required data
            if not optimized_parameters or not quality_thresholds:
                self.logger.error(f"❌ Missing required data: optimized_parameters={bool(optimized_parameters)}, quality_thresholds={bool(quality_thresholds)}")
                self.logger.error(f"❌ optimization_result keys: {list(optimization_result.keys())}")
                raise ValueError("Parameter optimization failed to produce required data")

            # Create enhanced consolidated artifact
            artifacts = {
                'sr_parameter_optimization_result': {
                    'optimized_parameters': optimized_parameters,
                    'quality_thresholds': quality_thresholds,
                    'parameter_optimization_metrics': parameter_optimization_metrics,
                    'optimization_summary': {
                        'total_combinations_tested': optimization_result.get('total_combinations_tested', 0),
                        'best_score': optimization_result.get('best_score', 0.0),
                        'optimization_time': optimization_result.get('optimization_time', 0.0),
                        'bayesian_hpo_used': enhanced_config.enable_bayesian_hpo,
                        'vectorbt_optimization_used': enhanced_config.enable_vectorbt_optimization,
                        'hardware_optimization_used': enhanced_config.enable_hardware_optimization
                    },
                    'enhancement_details': {
                        'bayesian_trials': optimization_result.get('bayesian_trials', 0),
                        'vectorbt_acceleration_factor': optimization_result.get('vectorbt_acceleration_factor', 1.0),
                        'hardware_optimization_gains': optimization_result.get('hardware_gains', {}),
                        'validation_results': optimization_result.get('validation_results', {}),
                        'explainability_results': optimization_result.get('explainability_results', {}),
                        'staged_optimization_used': optimization_result.get('staged_optimization_used', False),
                        'coarse_grid_points': optimization_result.get('coarse_grid_points', 0),
                        'fine_grid_points': optimization_result.get('fine_grid_points', 0),
                        'tpe_trials': optimization_result.get('tpe_trials', 0)
                    },
                    'enhancement_summary': self._generate_enhancement_summary(),
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat(),
                        'enhancement_version': '2.0'
                    }
                }
            }

            # Calculate enhanced metrics
            metrics = {
                'data_points': len(market_data) if market_data is not None else 0,
                'optimization_time': optimization_result.get('optimization_time', 0.0),
                'best_score': optimization_result.get('best_score', 0.0),
                'total_combinations_tested': optimization_result.get('total_combinations_tested', 0),
                'performance_improvements': {
                    'vectorbt_speedup': optimization_result.get('vectorbt_acceleration_factor', 1.0),
                    'hardware_optimization_gains': optimization_result.get('hardware_gains', {}),
                    'bayesian_efficiency': optimization_result.get('bayesian_efficiency', 0.0)
                }
            }

            # Save artifacts using BaseStep artifact management
            saved_artifacts = await self._save_output_artifacts(artifacts, config)
            if not saved_artifacts['success']:
                self.logger.warning(f"Some artifacts failed to save: {saved_artifacts['error']}")

            self.logger.info('✅ Enhanced SR Parameter Optimization completed successfully')
            return {
                'success': True,
                'artifacts': saved_artifacts.get('artifact_paths', {}),
                'metrics': metrics
            }

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.error(f'❌ SR Parameter Optimization failed: {error_type}: {error_msg}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')

            # Return BaseStep format
            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': f"SR Parameter Optimization failed: {error_type}: {error_msg}"
            }

    async def _load_market_data(self, config: Dict[str, Any]) -> Optional[Any]:
        """Load and prepare market data for optimization with memory optimization."""
        try:
            # Import klines manager here to avoid circular imports
            from src.utils.data.klines_parquet import get_klines_manager

            # Get klines manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))

            # Parse date filters if provided
            start_date = None
            end_date = None

            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])

            # Load data
            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed",
                start_date=start_date,
                end_date=end_date
            )

            if market_data is not None and len(market_data) > 0:
                # Ensure timestamp column exists
                if 'timestamp' not in market_data.columns and isinstance(market_data.index, pd.DatetimeIndex):
                    market_data = market_data.copy()
                    market_data['timestamp'] = market_data.index
                
                # Remove duplicate indices
                initial_rows = len(market_data)
                market_data = market_data[~market_data.index.duplicated(keep='last')]
                if len(market_data) < initial_rows:
                    self.logger.warning(f"Removed {initial_rows - len(market_data)} duplicate index entries")
                
                # Filter out invalid epoch timestamps (1970-01-01)
                epoch_date = pd.Timestamp('1970-01-01')
                invalid_timestamps = market_data.index == epoch_date
                if invalid_timestamps.any():
                    invalid_count = invalid_timestamps.sum()
                    market_data = market_data[~invalid_timestamps]
                    self.logger.warning(f"Removed {invalid_count} rows with invalid epoch timestamps (1970-01-01)")
                
                if len(market_data) == 0:
                    self.logger.error("No valid data remaining after cleaning duplicate indices and invalid timestamps")
                    return None

                return market_data
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return None

    def _validate_market_data(self, data: Any) -> bool:
        """Validate market data for optimization requirements."""
        if data is None:
            self.logger.error("Market data is None")
            return False

        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # Check if DataFrame is empty
            if len(data) == 0:
                self.logger.error("Market data DataFrame is empty")
                return False

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check minimum data points
            if len(data) < 100:
                self.logger.error(f"Insufficient data points for optimization: {len(data)} < 100")
                return False

            # Check for NaN values in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            for col in critical_columns:
                nan_count = data[col].isna().sum()
                if nan_count > 0:
                    self.logger.warning(f"Found {nan_count} NaN values in critical column: {col}. Filling with forward fill.")
                    # Fill NaN values with forward fill, then backward fill for any remaining NaNs
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                    # If still NaN, fill with 0 (shouldn't happen with proper data)
                    data[col] = data[col].fillna(0)
                    self.logger.info(f"Cleaned NaN values in column: {col}")

            # Check for reasonable price values
            for col in critical_columns:
                if (data[col] <= 0).any():
                    self.logger.error(f"Found non-positive values in column: {col}")
                    return False

            self.logger.info(f"Market data validation passed: {len(data)} rows, columns: {list(data.columns)}")
            return True

        # For non-DataFrame data, assume it's valid if not None
        self.logger.warning("Non-DataFrame data provided, validation limited")
        return True

    def _prepare_data_for_backtesting(self, data: Any) -> Any:
        """Prepare data for backtesting with proper datetime indexing."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            return data

        # Process data for backtesting (similar to the original method but simplified)
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to find timestamp column and set as index
            timestamp_columns = ['timestamp', 'open_time', 'time', 'datetime', 'date']
            for col in timestamp_columns:
                if col in data.columns:
                    data = data.set_index(col)
                    break

        # Convert to datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index, utc=False, errors='coerce')
                data = data.dropna()  # Remove invalid dates
            except Exception as e:
                self.logger.error(f"Failed to convert index to datetime: {e}")
                data.index = pd.RangeIndex(start=0, stop=len(data))

        return data


    def _split_data_for_optimization(self, market_data: Any, temporal_gap_hours: int = 24) -> Tuple[Any, Any]:
        """Split data properly to avoid data leakage during optimization with temporal gap."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return market_data, market_data

        # Use 70% for training (level creation) and 30% for testing (backtesting)
        split_point = int(len(market_data) * 0.7)

        if split_point < 100:
            self.logger.warning("Insufficient data for proper splitting, using same data for train/test")
            return market_data, market_data

        # Add temporal gap to prevent data leakage
        gap_periods = max(1, int(temporal_gap_hours * 60 / 15))  # Assuming 15m timeframe
        
        level_creation_data = market_data.iloc[:split_point]
        backtest_data = market_data.iloc[split_point + gap_periods:]

        if len(backtest_data) < 50:
            self.logger.warning("Insufficient test data after gap, reducing gap")
            gap_periods = max(1, int(gap_periods * 0.5))
            backtest_data = market_data.iloc[split_point + gap_periods:]

        self.logger.info(f"Data split: {len(level_creation_data)} train, {len(backtest_data)} test, {gap_periods} period gap")
        return level_creation_data, backtest_data

    def _get_current_data(self):
        """Get current data reference for configuration methods."""
        return getattr(self, '_current_data', None)

    async def _run_enhanced_parameter_optimization(
        self, 
        market_data: Any, 
        enhanced_config: EnhancedSRConfig, 
        config: Dict[str, Any],
        input_artifacts: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run enhanced parameter optimization using advanced techniques.
        
        Args:
            market_data: Market data for optimization
            enhanced_config: Enhanced configuration
            config: User configuration
            input_artifacts: Artifacts from previous steps (sr_clustering_result, sr_levels_dictionary)
            
        Returns:
            Optimization results dictionary
        """
        self.logger.info("🚀 Starting enhanced parameter optimization...")
        tprint("🔧 Enhanced Parameter Optimization: Starting", "info")
        start_time = time.time()
        
        # Log input artifacts usage
        if input_artifacts:
            self.logger.info("📊 Using input artifacts from previous steps:")
            tprint("📊 Using input artifacts from previous steps", "info")
            for artifact_name, artifact_data in input_artifacts.items():
                if artifact_data is not None:
                    if artifact_name == 'sr_clustering_result':
                        clusters_count = artifact_data.get('total_clusters', 0)
                        self.logger.info(f"  - {artifact_name}: {clusters_count} clusters")
                        tprint(f"  - {artifact_name}: {clusters_count} clusters", "info")
                    elif artifact_name == 'sr_levels_dictionary':
                        levels_count = len(artifact_data.get('levels', []))
                        self.logger.info(f"  - {artifact_name}: {levels_count} SR levels")
                    else:
                        self.logger.info(f"  - {artifact_name}: available")
                else:
                    self.logger.warning(f"  - {artifact_name}: not available")
        else:
            self.logger.warning("⚠️ No input artifacts provided, proceeding with basic optimization")
        
        # Initialize results
        optimization_result = {
            'optimized_parameters': {},
            'quality_thresholds': {},
            'parameter_optimization_metrics': {},
            'total_combinations_tested': 0,
            'best_score': 0.0,
            'optimization_time': 0.0,
            'bayesian_trials': 0,
            'vectorbt_acceleration_factor': 1.0,
            'hardware_gains': {},
            'validation_results': {}
        }
        
        try:
            # Create SR parameter search space using input artifacts for intelligent bounds
            search_space = self._create_sr_search_space(input_artifacts)
            
            # Split data for optimization with temporal validation
            train_data, test_data = self._split_data_for_optimization(market_data, enhanced_config.temporal_gap_hours)
            
            # Run Hierarchical HPO if enabled (recommended for 6+ parameters)
            if enhanced_config.enable_hierarchical_hpo and HIERARCHICAL_HPO_AVAILABLE:
                self.logger.info("🚀 Running Hierarchical HPO optimization...")
                hierarchical_result = await self._run_hierarchical_optimization(
                    market_data, search_space, enhanced_config
                )
                optimization_result.update(hierarchical_result)
            
            # Run Bayesian HPO if enabled (fallback or when hierarchical is disabled)
            elif enhanced_config.enable_bayesian_hpo and self.bayesian_optimizer:
                self.logger.info("🧠 Running Bayesian HPO optimization...")
                bayesian_result = await self._run_bayesian_optimization(
                    search_space, train_data, test_data, enhanced_config, config, market_data, input_artifacts
                )
                optimization_result.update(bayesian_result)
            
            # Run VectorBT optimization if enabled
            elif enhanced_config.enable_vectorbt_optimization and self.vectorbt_optimizer:
                self.logger.info("⚡ Running VectorBT optimization...")
                vectorbt_result = await self._run_vectorbt_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
                optimization_result.update(vectorbt_result)
            
            # Fallback to traditional optimization
            else:
                self.logger.info("📊 Running traditional optimization...")
                traditional_result = await self._run_traditional_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
                optimization_result.update(traditional_result)
            
            # Apply hardware optimization if enabled
            if enhanced_config.enable_hardware_optimization and self.hardware_manager:
                self.logger.info("🖥️ Applying hardware optimizations...")
                hardware_result = await self._apply_hardware_optimization(
                    optimization_result, enhanced_config
                )
                optimization_result.update(hardware_result)
            
            # Validate results for data leakage
            if enhanced_config.enable_advanced_validation and self.leakage_detector:
                self.logger.info("🔍 Validating results for data leakage...")
                validation_result = await self._validate_optimization_results(
                    optimization_result, train_data, test_data
                )
                optimization_result['validation_results'] = validation_result
            
            optimization_result['optimization_time'] = time.time() - start_time
            self.logger.info(f"✅ Enhanced optimization completed in {optimization_result['optimization_time']:.2f}s")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced optimization failed: {e}")
            optimization_result['optimization_time'] = time.time() - start_time
            optimization_result['error'] = str(e)
            return optimization_result

    def _create_sr_search_space(self, input_artifacts: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create adaptive search space for SR parameter optimization based on market data characteristics."""
        
        # Get market data characteristics for adaptive ranges
        market_characteristics = self._analyze_market_characteristics(input_artifacts)
        
        # Base parameter bounds with adaptive adjustments
        # NOTE: This optimizes SR DETECTION parameters only. Trading strategy parameters
        # (stop_loss, take_profit, risk_reward_ratio) should be optimized separately
        # in the BACKTESTING stage, not here in MARKET_ANALYSIS.
        search_space = {
            # Core SR detection parameters - adaptive based on market volatility
            'min_touches': self._get_adaptive_range('min_touches', market_characteristics),
            'strength_threshold': self._get_adaptive_range('strength_threshold', market_characteristics),
            'distance_threshold': self._get_adaptive_range('distance_threshold', market_characteristics),
            'lookback_periods': self._get_adaptive_range('lookback_periods', market_characteristics),
            'volume_threshold': self._get_adaptive_range('volume_threshold', market_characteristics),
            
            # Advanced SR parameters - adaptive based on market structure
            'touch_tolerance': self._get_adaptive_range('touch_tolerance', market_characteristics),
            'breakout_threshold': self._get_adaptive_range('breakout_threshold', market_characteristics),
            'consolidation_periods': self._get_adaptive_range('consolidation_periods', market_characteristics),
            'trend_strength_threshold': self._get_adaptive_range('trend_strength_threshold', market_characteristics),
            
            # Time-based parameters - adaptive based on timeframe
            'min_formation_time': self._get_adaptive_range('min_formation_time', market_characteristics),
            'max_formation_time': self._get_adaptive_range('max_formation_time', market_characteristics),
            'time_decay_factor': self._get_adaptive_range('time_decay_factor', market_characteristics),
            
            # Volume-based parameters - adaptive based on volume patterns
            'volume_spike_threshold': self._get_adaptive_range('volume_spike_threshold', market_characteristics),
            'volume_consistency_threshold': self._get_adaptive_range('volume_consistency_threshold', market_characteristics),
            'volume_weight': self._get_adaptive_range('volume_weight', market_characteristics),
            
            # Price action parameters - adaptive based on price action patterns
            'wick_ratio_threshold': self._get_adaptive_range('wick_ratio_threshold', market_characteristics),
            'body_ratio_threshold': self._get_adaptive_range('body_ratio_threshold', market_characteristics),
            'price_momentum_threshold': self._get_adaptive_range('price_momentum_threshold', market_characteristics)
            
            # REMOVED: Trading strategy parameters (moved to separate optimization in BACKTESTING stage)
            # These were causing:
            # - 70% larger search space (24 params → 17 params)
            # - 10x more memory usage
            # - Confusion between SR detection quality and trading strategy performance
            # - OOM kills due to excessive grid size
            #
            # Previously removed parameters:
            # - 'stop_loss_multiplier': Trading strategy, not SR detection
            # - 'take_profit_multiplier': Trading strategy, not SR detection  
            # - 'risk_reward_ratio': Trading strategy, not SR detection
            # - 'noise_filter_threshold': Should be in data preprocessing
            # - 'correlation_threshold': Should be in feature selection
            # - 'volatility_threshold': Should be in regime detection
        }
        
        # Enhance search space based on input artifacts
        if input_artifacts:
            self.logger.info("🎯 Enhancing search space based on input artifacts...")
            
            # Use SR clustering results to inform parameter bounds
            sr_clustering_result = input_artifacts.get('sr_clustering_result')
            # Handle both DataFrame and dict types
            if sr_clustering_result is not None:
                # Convert DataFrame to dict if needed
                if PANDAS_AVAILABLE and isinstance(sr_clustering_result, pd.DataFrame):
                    if not sr_clustering_result.empty:
                        sr_clustering_result = sr_clustering_result.to_dict('records')[0] if len(sr_clustering_result) > 0 else {}
                
                if isinstance(sr_clustering_result, dict):
                    try:
                        # Adjust min_touches based on clustering results
                        total_clusters = sr_clustering_result.get('total_clusters', 0)
                        if total_clusters > 0:
                            # More clusters suggest we can be more selective with touches
                            min_touches_high = min(15, max(5, total_clusters // 2))
                            search_space['min_touches']['high'] = min_touches_high
                            self.logger.info(f"Adjusted min_touches high bound to {min_touches_high} based on {total_clusters} clusters")
                        
                        # Adjust strength_threshold based on clustering efficiency
                        clustering_efficiency = sr_clustering_result.get('clustering_efficiency', 0.5)
                        if clustering_efficiency > 0.7:
                            # High efficiency suggests we can be more strict
                            search_space['strength_threshold']['low'] = 0.3
                            search_space['strength_threshold']['high'] = 0.8
                        elif clustering_efficiency < 0.3:
                            # Low efficiency suggests we should be more lenient
                            search_space['strength_threshold']['low'] = 0.1
                            search_space['strength_threshold']['high'] = 0.6
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to enhance search space with clustering results: {e}")
            
            # Use SR levels dictionary to inform parameter bounds
            sr_levels_dict = input_artifacts.get('sr_levels_dictionary')
            # Handle both DataFrame and dict types
            if sr_levels_dict is not None:
                # Convert DataFrame to dict if needed
                if PANDAS_AVAILABLE and isinstance(sr_levels_dict, pd.DataFrame):
                    if not sr_levels_dict.empty:
                        sr_levels_dict = sr_levels_dict.to_dict('records')[0] if len(sr_levels_dict) > 0 else {}
                
                if isinstance(sr_levels_dict, dict):
                    try:
                        levels = sr_levels_dict.get('levels', [])
                        # Handle both list and array types
                        if levels is not None and len(levels) > 0:
                            # Analyze level characteristics to inform bounds
                            avg_strength = sum(level.get('strength', 0) for level in levels) / len(levels)
                            avg_touches = sum(level.get('touches', 0) for level in levels) / len(levels)
                            
                            # Adjust strength_threshold based on average level strength
                            if avg_strength > 0.6:
                                search_space['strength_threshold']['low'] = max(0.1, avg_strength - 0.2)
                                search_space['strength_threshold']['high'] = min(0.9, avg_strength + 0.2)
                            
                            # Adjust min_touches based on average touches
                            if avg_touches > 0:
                                min_touches_low = max(2, int(avg_touches * 0.5))
                                min_touches_high = min(15, int(avg_touches * 1.5))
                                search_space['min_touches']['low'] = min_touches_low
                                search_space['min_touches']['high'] = min_touches_high
                                self.logger.info(f"Adjusted min_touches bounds to [{min_touches_low}, {min_touches_high}] based on average touches {avg_touches:.1f}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to enhance search space with SR levels: {e}")
        
        return search_space

    def _analyze_market_characteristics(self, input_artifacts: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze market characteristics to inform parameter ranges."""
        characteristics = {
            'volatility_level': 'medium',
            'timeframe': '15m',
            'market_structure': 'trending',
            'volume_profile': 'normal',
            'noise_level': 'medium'
        }
        
        try:
            # Analyze volatility from input artifacts
            if input_artifacts and 'sr_clustering_result' in input_artifacts:
                clustering_result = input_artifacts['sr_clustering_result']
                # Handle both DataFrame and dict types
                if clustering_result is not None:
                    # Convert DataFrame to dict if needed
                    if PANDAS_AVAILABLE and isinstance(clustering_result, pd.DataFrame):
                        if not clustering_result.empty:
                            clustering_dict = clustering_result.to_dict('records')[0] if len(clustering_result) > 0 else {}
                            efficiency = clustering_dict.get('clustering_efficiency', 0.5)
                            if efficiency > 0.7:
                                characteristics['market_structure'] = 'consolidated'
                            elif efficiency < 0.3:
                                characteristics['market_structure'] = 'choppy'
                    elif isinstance(clustering_result, dict):
                        # Use clustering efficiency to determine market structure
                        efficiency = clustering_result.get('clustering_efficiency', 0.5)
                        if efficiency > 0.7:
                            characteristics['market_structure'] = 'consolidated'
                        elif efficiency < 0.3:
                            characteristics['market_structure'] = 'choppy'
            
            # Analyze SR levels for market characteristics
            if input_artifacts and 'sr_levels_dictionary' in input_artifacts:
                levels_dict = input_artifacts['sr_levels_dictionary']
                # Handle both DataFrame and dict types
                if levels_dict is not None and not (PANDAS_AVAILABLE and isinstance(levels_dict, pd.DataFrame) and levels_dict.empty):
                    # Convert DataFrame to dict if needed
                    if PANDAS_AVAILABLE and isinstance(levels_dict, pd.DataFrame):
                        if not levels_dict.empty:
                            levels_dict = levels_dict.to_dict('records')[0] if len(levels_dict) > 0 else {}
                    
                    if isinstance(levels_dict, dict) and 'levels' in levels_dict:
                        levels = levels_dict['levels']
                        if levels:
                            # Analyze level characteristics
                            avg_strength = sum(level.get('strength', 0) for level in levels) / len(levels)
                            avg_touches = sum(level.get('touches', 0) for level in levels) / len(levels)
                        else:
                            avg_strength = 0
                            avg_touches = 0
                        
                        # Determine volatility level based on level strength
                        if avg_strength > 0.7:
                            characteristics['volatility_level'] = 'low'
                        elif avg_strength < 0.3:
                            characteristics['volatility_level'] = 'high'
                        
                        # Determine noise level based on touches
                        if avg_touches > 8:
                            characteristics['noise_level'] = 'high'
                        elif avg_touches < 3:
                            characteristics['noise_level'] = 'low'
            
            self.logger.info(f"Market characteristics: {characteristics}")
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze market characteristics: {e}")
            return characteristics

    def _get_adaptive_range(self, param_name: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get adaptive parameter range based on market characteristics.
        
        NOTE: Only SR detection parameters are included. Trading strategy parameters
        have been removed to focus optimization on SR level quality, not trading performance.
        """
        # Default ranges for SR DETECTION parameters only
        default_ranges = {
            # Core SR detection parameters
            'min_touches': {'type': 'int', 'low': 2, 'high': 15},
            'strength_threshold': {'type': 'float', 'low': 0.1, 'high': 0.9},
            'distance_threshold': {'type': 'float', 'low': 0.001, 'high': 0.05},
            'lookback_periods': {'type': 'int', 'low': 20, 'high': 500},
            'volume_threshold': {'type': 'float', 'low': 0.5, 'high': 3.0},
            
            # Advanced SR parameters
            'touch_tolerance': {'type': 'float', 'low': 0.001, 'high': 0.02},
            'breakout_threshold': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'consolidation_periods': {'type': 'int', 'low': 5, 'high': 50},
            'trend_strength_threshold': {'type': 'float', 'low': 0.3, 'high': 0.8},
            
            # Time-based parameters
            'min_formation_time': {'type': 'int', 'low': 1, 'high': 30},
            'max_formation_time': {'type': 'int', 'low': 30, 'high': 200},
            'time_decay_factor': {'type': 'float', 'low': 0.8, 'high': 1.0},
            
            # Volume-based parameters
            'volume_spike_threshold': {'type': 'float', 'low': 1.5, 'high': 5.0},
            'volume_consistency_threshold': {'type': 'float', 'low': 0.7, 'high': 1.0},
            'volume_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
            
            # Price action parameters
            'wick_ratio_threshold': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'body_ratio_threshold': {'type': 'float', 'low': 0.3, 'high': 0.8},
            'price_momentum_threshold': {'type': 'float', 'low': 0.1, 'high': 0.5}
            
            # REMOVED: Trading strategy parameters
            # These should be optimized in the BACKTESTING stage:
            # - 'stop_loss_multiplier': Trading strategy parameter
            # - 'take_profit_multiplier': Trading strategy parameter
            # - 'risk_reward_ratio': Trading strategy parameter
            # - 'noise_filter_threshold': Data preprocessing parameter
            # - 'correlation_threshold': Feature selection parameter
            # - 'volatility_threshold': Regime detection parameter
        }
        
        base_range = default_ranges.get(param_name, {'type': 'float', 'low': 0.0, 'high': 1.0})
        range_copy = base_range.copy()
        
        # Apply adaptive adjustments based on market characteristics
        volatility_level = characteristics.get('volatility_level', 'medium')
        market_structure = characteristics.get('market_structure', 'trending')
        noise_level = characteristics.get('noise_level', 'medium')
        
        # Adjust distance_threshold based on volatility
        if param_name == 'distance_threshold':
            if volatility_level == 'high':
                range_copy['low'] = 0.005
                range_copy['high'] = 0.03
            elif volatility_level == 'low':
                range_copy['low'] = 0.001
                range_copy['high'] = 0.01
        
        # Adjust min_touches based on noise level
        elif param_name == 'min_touches':
            if noise_level == 'high':
                range_copy['low'] = 4
                range_copy['high'] = 12
            elif noise_level == 'low':
                range_copy['low'] = 2
                range_copy['high'] = 8
        
        # Adjust strength_threshold based on market structure
        elif param_name == 'strength_threshold':
            if market_structure == 'consolidated':
                range_copy['low'] = 0.3
                range_copy['high'] = 0.8
            elif market_structure == 'choppy':
                range_copy['low'] = 0.1
                range_copy['high'] = 0.6
        
        # Adjust lookback_periods based on timeframe
        elif param_name == 'lookback_periods':
            timeframe = characteristics.get('timeframe', '15m')
            if timeframe in ['1m', '5m']:
                range_copy['low'] = 50
                range_copy['high'] = 200
            elif timeframe in ['1h', '4h']:
                range_copy['low'] = 20
                range_copy['high'] = 100
            else:  # daily or higher
                range_copy['low'] = 10
                range_copy['high'] = 50
        
        return range_copy


    async def _run_hierarchical_optimization(
        self, 
        market_data: pd.DataFrame,
        search_space: Dict[str, Any],
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """
        Run hierarchical 3-phase optimization for SR parameters with FAST FILTERING.
        
        Phase 1: Detection parameters (min_touches, strength_threshold)
        Phase 2: Distance thresholds (distance_threshold)
        Phase 3: Lookback parameters (lookback_periods, time_decay)
        
        OPTIMIZATION: Detects levels ONCE, then filters for each trial (100x faster).
        
        Args:
            market_data: Market data DataFrame
            search_space: Parameter search space
            enhanced_config: Enhanced configuration
            
        Returns:
            Optimization result dictionary
        """
        if not HIERARCHICAL_HPO_AVAILABLE:
            self.logger.warning("Hierarchical HPO not available, falling back to traditional")
            # Split data for traditional optimization
            train_data, test_data = self._split_data_for_optimization(market_data)
            return await self._run_traditional_optimization(search_space, train_data, test_data, enhanced_config)
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 HIERARCHICAL SR PARAMETER OPTIMIZATION (FAST MODE)")
        self.logger.info("=" * 80)
        self.logger.info("Phase 1: Detection (min_touches, strength_threshold)")
        self.logger.info("Phase 2: Distance (distance_threshold)")
        self.logger.info("Phase 3: Lookback (lookback_periods, time_decay)")
        self.logger.info("=" * 80)
        
        try:
            # OPTIMIZATION: Detect levels ONCE with relaxed parameters
            self.logger.info("🚀 Pre-detecting SR levels once (FAST MODE)...")
            relaxed_params = {
                'min_touches': 1,
                'strength_threshold': 0.1,
                'distance_threshold': 0.01,
                'lookback_periods': 50,
                'volume_threshold': 0.5
            }
            all_detected_levels = self._detect_sr_levels(market_data, relaxed_params)
            self.logger.info(f"✅ Pre-detected {len(all_detected_levels)} candidate SR levels")
            
            if not all_detected_levels:
                self.logger.warning("⚠️ No SR levels detected, cannot optimize")
                return {
                    'optimized_parameters': relaxed_params,
                    'best_score': 0.0,
                    'total_combinations_tested': 0,
                    'error': 'No SR levels detected'
                }
            # Define parameter groups with improved logical grouping
            # Group 1: Core Detection (highest priority - affects what gets detected)
            # Group 2: Quality Filtering (depends on detection - filters detected levels)
            # Group 3: Temporal/Lookback (depends on detection - historical context)
            # Group 4: Market Context (lowest priority - refinement parameters)
            param_groups = [
                create_param_group(
                    name="core_detection",
                    params={
                        "min_touches": search_space.get('min_touches', {"type": "int", "low": 2, "high": 5})
                        # NOTE: strength_threshold removed - belongs in Group 5 (calculated strength filtering)
                        # Can't filter by strength before optimizing how strength is calculated!
                    },
                    priority=1,
                    description="Core SR detection: minimum touches required"
                ),
                create_param_group(
                    name="quality_filtering",
                    params={
                        "distance_threshold": search_space.get('distance_threshold', {"type": "float", "low": 0.005, "high": 0.03}),
                        "volume_threshold": search_space.get('volume_threshold', {"type": "float", "low": 0.5, "high": 2.0})
                    },
                    priority=2,
                    depends_on=["core_detection"],
                    description="Quality filters: distance and volume confirmation"
                ),
                create_param_group(
                    name="temporal_lookback",
                    params={
                        "lookback_periods": search_space.get('lookback_periods', {"type": "int", "low": 20, "high": 100})
                    },
                    priority=3,
                    depends_on=["core_detection"],
                    description="Historical lookback: how far to search for patterns"
                ),
                create_param_group(
                    name="market_context",
                    params={
                        "trend_strength_threshold": search_space.get('trend_strength_threshold', {"type": "float", "low": 0.3, "high": 0.7}),
                        "breakout_threshold": search_space.get('breakout_threshold', {"type": "float", "low": 0.01, "high": 0.05})
                    },
                    priority=4,
                    depends_on=["core_detection", "quality_filtering"],
                    description="Market context: trend and breakout refinement"
                )
            ]
            
            # Add strength weight optimization if enabled
            # CRITICAL: Split into multiple groups to avoid combinatorial explosion
            # 11 params in 1 group = 5^11 = 48M combinations (INTRACTABLE!)
            # Split into 3 groups of ≤5 params each
            if enhanced_config.enable_strength_weight_optimization:
                # Group 5a: Core Positive Boosts (5 params) - Most impactful weights
                strength_boosts_core = create_param_group(
                    name="strength_boosts_core",
                    params={
                        "touch_weight": {"type": "float", "low": 0.05, "high": 0.3, "step": 0.05},
                        "volume_weight": {"type": "float", "low": 0.1, "high": 0.4, "step": 0.05},
                        "consistency_weight": {"type": "float", "low": 0.1, "high": 0.4, "step": 0.05},
                        "confluence_weight": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.025},
                        "pivot_boost": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.025}
                    },
                    priority=5,
                    depends_on=["core_detection", "quality_filtering"],
                    description="Core strength boosts: touch, volume, consistency, confluence, pivot"
                )
                param_groups.append(strength_boosts_core)
                
                # Group 5b: Secondary Boosts + Filter (3 params) - Special case boosts
                strength_boosts_special = create_param_group(
                    name="strength_boosts_special",
                    params={
                        "psychological_boost": {"type": "float", "low": 0.02, "high": 0.1, "step": 0.01},
                        "hvn_boost": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.025},
                        "strength_filter_threshold": {"type": "float", "low": 0.3, "high": 0.8, "step": 0.05}
                    },
                    priority=6,
                    depends_on=["core_detection", "strength_boosts_core"],
                    description="Special boosts (psychological, HVN) + post-calculation filter"
                )
                param_groups.append(strength_boosts_special)
                
                # Group 5c: Failure Penalties (3 params) - Negative adjustments
                strength_penalties = create_param_group(
                    name="strength_penalties",
                    params={
                        "failure_penalty_base": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.05},
                        "failure_volume_multiplier": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
                        "failure_max_penalty": {"type": "float", "low": 0.4, "high": 1.0, "step": 0.1}
                    },
                    priority=7,
                    depends_on=["core_detection", "strength_boosts_core"],
                    description="Failure penalties: base, volume multiplier, max penalty"
                )
                param_groups.append(strength_penalties)
                
                self.logger.info("✅ Added strength weight optimization (split into 3 groups to avoid combinatorial explosion)")
                self.logger.info("   - Group 5a: Core boosts (5 params) - 5^5 = 3,125 combos")
                self.logger.info("   - Group 5b: Special boosts + filter (3 params) - 5^3 = 125 combos")
                self.logger.info("   - Group 5c: Penalties (3 params) - 5^3 = 125 combos")
                self.logger.info("   - Total: 11 params split across 3 groups (vs. 5^11 = 48M in 1 group)")
            
            # Define objective function (OPTIMIZED: uses pre-detected levels)
            def objective_func(params, X_train, y_train, X_val=None, y_val=None,
                              model=None, cv_folds=None, scoring_metric=None):
                """Objective function for SR parameter optimization (FAST: filters pre-detected levels)."""
                try:
                    # Handle None params
                    if params is None:
                        self.logger.warning("Objective function received None params")
                        return 0.0
                    
                    # OPTIMIZATION: Filter pre-detected levels instead of re-detecting
                    param_dict = {
                        'min_touches': int(params.get('min_touches', 2) if isinstance(params, dict) else getattr(params, 'min_touches', 2)),
                        'distance_threshold': float(params.get('distance_threshold', 0.01) if isinstance(params, dict) else getattr(params, 'distance_threshold', 0.01)),
                        'lookback_periods': int(params.get('lookback_periods', 50) if isinstance(params, dict) else getattr(params, 'lookback_periods', 50)),
                        'volume_threshold': float(params.get('volume_threshold', 1.0) if isinstance(params, dict) else getattr(params, 'volume_threshold', 1.0)),
                        # NOTE: strength_threshold removed from here - now in Group 5 as strength_filter_threshold
                    }
                    
                    # Extract strength weights if being optimized
                    strength_weights = None
                    if enhanced_config.enable_strength_weight_optimization:
                        strength_weights = StrengthWeights(
                            # Positive boosts
                            touch_weight=float(params.get('touch_weight', 0.1) if isinstance(params, dict) else getattr(params, 'touch_weight', 0.1)),
                            volume_weight=float(params.get('volume_weight', 0.2) if isinstance(params, dict) else getattr(params, 'volume_weight', 0.2)),
                            consistency_weight=float(params.get('consistency_weight', 0.2) if isinstance(params, dict) else getattr(params, 'consistency_weight', 0.2)),
                            confluence_weight=float(params.get('confluence_weight', 0.1) if isinstance(params, dict) else getattr(params, 'confluence_weight', 0.1)),
                            pivot_boost=float(params.get('pivot_boost', 0.1) if isinstance(params, dict) else getattr(params, 'pivot_boost', 0.1)),
                            psychological_boost=float(params.get('psychological_boost', 0.05) if isinstance(params, dict) else getattr(params, 'psychological_boost', 0.05)),
                            hvn_boost=float(params.get('hvn_boost', 0.1) if isinstance(params, dict) else getattr(params, 'hvn_boost', 0.1)),
                            # Negative penalties
                            failure_penalty_base=float(params.get('failure_penalty_base', 0.2) if isinstance(params, dict) else getattr(params, 'failure_penalty_base', 0.2)),
                            failure_volume_multiplier=float(params.get('failure_volume_multiplier', 1.5) if isinstance(params, dict) else getattr(params, 'failure_volume_multiplier', 1.5)),
                            failure_max_penalty=float(params.get('failure_max_penalty', 0.6) if isinstance(params, dict) else getattr(params, 'failure_max_penalty', 0.6))
                        )
                    
                    # Filter pre-detected levels (MUCH faster than re-detecting)
                    filtered_levels = self._filter_sr_levels_by_params(all_detected_levels, param_dict)
                    
                    # Calculate quality score based on filtered levels
                    if len(filtered_levels) == 0:
                        return 0.0
                    
                    # If optimizing strength weights, recalculate strengths and evaluate
                    if strength_weights is not None:
                        # Recalculate strengths with new weights
                        recalc_strengths = []
                        for level in filtered_levels:
                            new_strength = self._calculate_strength_with_weights(level, strength_weights)
                            recalc_strengths.append(new_strength)
                        
                        # Apply strength filter threshold (moved from Group 1 to Group 5)
                        strength_filter_threshold = float(
                            params.get('strength_filter_threshold', 0.5) if isinstance(params, dict) 
                            else getattr(params, 'strength_filter_threshold', 0.5)
                        )
                        final_strengths = [s for s in recalc_strengths if s >= strength_filter_threshold]
                        
                        # Calculate score with filtered strengths
                        if len(final_strengths) == 0:
                            return 0.0
                        
                        avg_strength = np.mean(final_strengths)
                        level_count_score = min(len(final_strengths) / 20.0, 1.0)
                    else:
                        # Use original strengths (no recalculation)
                        strengths = [l.get('strength', 0) if isinstance(l, dict) else getattr(l, 'strength', 0) for l in filtered_levels]
                        avg_strength = np.mean(strengths) if strengths else 0.0
                        level_count_score = min(len(filtered_levels) / 20.0, 1.0)
                    
                    # Combined score
                    combined_score = (level_count_score * 0.4 + avg_strength * 0.6)
                    return combined_score
                        
                except Exception as e:
                    self.logger.error(f"Objective evaluation failed: {e}")
                    return 0.0
            
            # Create hierarchical optimizer with 3-stage optimization
            # Stage 1: Coarse Grid (broad exploration, 3-5 points per param)
            # Stage 2: Fine Grid (dense sampling around best region, 5-7 points)
            # Stage 3: TPE (Bayesian optimization for final refinement)
            self.logger.info("🔧 Optimizer Strategy: Coarse Grid → Fine Grid → TPE (Bayesian)")
            
            # Configure stages with grid points
            from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import StageConfig
            stage_configs = {
                OptimizationStage.COARSE_GRID: StageConfig(
                    stage=OptimizationStage.COARSE_GRID,
                    grid_points=enhanced_config.coarse_grid_points,  # 5 points per param
                    n_trials=20  # Coarse grid trials
                ),
                OptimizationStage.FINE_GRID: StageConfig(
                    stage=OptimizationStage.FINE_GRID,
                    grid_points=enhanced_config.fine_grid_points,  # 8 points per param
                    n_trials=30  # Fine grid trials
                ),
                OptimizationStage.TPE: StageConfig(
                    stage=OptimizationStage.TPE,
                    n_trials=enhanced_config.tpe_trials  # 150 TPE trials
                )
            }
            
            hierarchical_optimizer = HierarchicalParameterOptimizer(
                param_groups=param_groups,
                objective_func=objective_func,
                stages=[
                    OptimizationStage.COARSE_GRID,    # Fast broad exploration
                    OptimizationStage.FINE_GRID,      # Refined local search
                    OptimizationStage.TPE              # Bayesian optimization (tree-structured Parzen estimator)
                ],
                stage_configs=stage_configs,  # Pass stage configurations
                direction='maximize',
                n_rounds=2,  # Run 2 rounds of group optimization for convergence
                enable_final_refinement=True,  # Final joint optimization of all params
                final_refinement_trials=max(30, enhanced_config.n_trials // 4),  # 25-30% of trials for refinement
                random_state=42,
                verbose=True
            )
            
            # Prepare data for optimizer
            data_array = market_data.values if isinstance(market_data, pd.DataFrame) else market_data
            
            # Run hierarchical optimization
            result = hierarchical_optimizer.optimize(
                X_train=data_array,
                y_train=np.zeros(len(data_array)),
                X_val=None,
                y_val=None
            )
            
            self.logger.info("=" * 80)
            self.logger.info("✅ HIERARCHICAL OPTIMIZATION COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total trials: {result.total_trials}")
            self.logger.info(f"Total time: {result.total_time:.2f}s")
            self.logger.info(f"Best score: {result.best_score:.4f}")
            self.logger.info("=" * 80)
            
            return {
                'success': True,
                'best_params': result.best_params,
                'optimized_parameters': result.best_params,  # Also include as optimized_parameters for consistency
                'quality_thresholds': {  # Add required quality_thresholds
                    'min_strength': result.best_params.get('strength_threshold', 0.5),
                    'min_touches': result.best_params.get('min_touches', 2),
                    'min_quality_score': 0.5
                },
                'parameter_optimization_metrics': {  # Add required metrics
                    'best_score': result.best_score,
                    'total_trials': result.total_trials,
                    'total_time': result.total_time,
                    'method': 'hierarchical'
                },
                'best_score': result.best_score,
                'total_trials': result.total_trials,
                'total_time': result.total_time,
                'total_combinations_tested': result.total_trials,
                'optimization_time': result.total_time,
                'method': 'hierarchical',
                'optimization_result': result
            }
            
        except Exception as e:
            self.logger.error(f"Hierarchical optimization failed: {e}")
            import traceback
            self.logger.error(f"Error details: {traceback.format_exc()}")
            # Fallback to traditional optimization
            train_data, test_data = self._split_data_for_optimization(market_data)
            return await self._run_traditional_optimization(search_space, train_data, test_data, enhanced_config)
    
    async def _run_bayesian_optimization(
        self, 
        search_space: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig,
        config: Dict[str, Any] = None,
        market_data: Any = None,
        input_artifacts: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run enhanced Bayesian optimization for SR parameters with FAST FILTERING."""
        try:
            self.logger.info("🧠 Starting enhanced Bayesian optimization (FAST MODE)...")
            
            # OPTIMIZATION: Detect levels ONCE with relaxed parameters
            self.logger.info("🚀 Pre-detecting SR levels once (FAST MODE)...")
            relaxed_params = {
                'min_touches': 1,
                'strength_threshold': 0.1,
                'distance_threshold': 0.01,
                'lookback_periods': 50,
                'volume_threshold': 0.5
            }
            all_detected_levels = self._detect_sr_levels(train_data, relaxed_params)
            self.logger.info(f"✅ Pre-detected {len(all_detected_levels)} candidate SR levels")
            
            if not all_detected_levels:
                self.logger.warning("⚠️ No SR levels detected, cannot optimize")
                return {
                    'optimized_parameters': relaxed_params,
                    'best_score': 0.0,
                    'total_combinations_tested': 0,
                    'error': 'No SR levels detected'
                }
            
            # Create optimization config with enhanced settings
            opt_config = OptimizationConfig(
                n_trials=enhanced_config.n_trials,
                enable_staged_optimization=enhanced_config.enable_staged_optimization,
                coarse_grid_points=enhanced_config.coarse_grid_points,
                fine_grid_points=enhanced_config.fine_grid_points,
                tpe_trials=enhanced_config.tpe_trials,
                enable_hardware_optimization=enhanced_config.enable_hardware_optimization,
                workload_type=enhanced_config.workload_type,
                optimization_level=enhanced_config.optimization_level,
                enable_pruner=True,
                pruner_type='median',
                n_startup_trials=10
            )
            
            # OPTIMIZED objective function: filters pre-detected levels
            def objective_function(trial):
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                
                # OPTIMIZATION: Use filtered evaluation instead of re-detecting
                score = self._evaluate_sr_parameters_filtered(
                    params, all_detected_levels, train_data, test_data
                )
                return score
            
            # Run VectorBT optimization if available
            vectorbt_result = None
            if VECTORBT_ROLLING_AVAILABLE and (config and config.get('enable_vectorbt', True) if config else True):
                tprint("⚡ Running VectorBT Rolling Optimization", "info")
                try:
                    vectorbt_result = await self._run_vectorbt_optimization(
                        market_data, search_space, enhanced_config, input_artifacts
                    )
                    tprint_data_preview(vectorbt_result, "VectorBT optimization result")
                except Exception as e:
                    self.logger.warning(f"VectorBT optimization failed: {e}")
                    tprint(f"⚠️ VectorBT optimization failed: {e}", "warning")
            
            # Run optimization with enhanced monitoring
            tprint("🧠 Running Bayesian Optimization", "info")
            result = self.bayesian_optimizer.optimize(
                objective_function,
                search_space,
                n_trials=opt_config.n_trials,
                enable_staged_optimization=opt_config.enable_staged_optimization,
                coarse_grid_points=opt_config.coarse_grid_points,
                fine_grid_points=opt_config.fine_grid_points,
                tpe_trials=opt_config.tpe_trials,
                enable_hardware_optimization=opt_config.enable_hardware_optimization,
                workload_type=opt_config.workload_type,
                optimization_level=opt_config.optimization_level,
                enable_pruner=opt_config.enable_pruner,
                pruner_type=opt_config.pruner_type,
                n_startup_trials=opt_config.n_startup_trials
            )
            tprint_data_preview(result, "Bayesian optimization result")
            
            # Generate explainability if available
            explainability_results = {}
            if self.explainer and enhanced_config.enable_explainability:
                try:
                    explainability_results = await self._generate_explainability(
                        result.best_params, train_data, test_data
                    )
                except Exception as e:
                    self.logger.warning(f"Explainability generation failed: {e}")
            
            # Enhanced result with ML utilities
            return {
                'optimized_parameters': result.best_params,
                'quality_thresholds': {
                    'min_strength': result.best_params.get('strength_threshold', 0.5),
                    'min_touches': result.best_params.get('min_touches', 2),
                    'min_quality_score': 0.5
                },
                'parameter_optimization_metrics': {
                    'best_score': result.best_value,
                    'bayesian_trials': result.n_trials,
                    'bayesian_efficiency': result.efficiency_score if hasattr(result, 'efficiency_score') else 0.0,
                    'method': 'bayesian_hpo'
                },
                'best_score': result.best_value,
                'bayesian_trials': result.n_trials,
                'bayesian_efficiency': result.efficiency_score if hasattr(result, 'efficiency_score') else 0.0,
                'staged_optimization_used': enhanced_config.enable_staged_optimization,
                'coarse_grid_points': enhanced_config.coarse_grid_points,
                'fine_grid_points': enhanced_config.fine_grid_points,
                'tpe_trials': enhanced_config.tpe_trials,
                'total_combinations_tested': result.n_trials,
                'optimization_time': 0.0,
                'explainability_results': explainability_results,
                'optimization_history': result.trials if hasattr(result, 'trials') else []
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced Bayesian optimization failed: {e}")
            return {'error': str(e)}

    async def _run_vectorbt_optimization(
        self, 
        search_space: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run enhanced VectorBT optimization for SR parameters with advanced features."""
        try:
            self.logger.info("🚀 Starting enhanced VectorBT optimization...")
            
            # Use VectorBT for efficient parameter testing
            if self.vectorization_manager:
                # Create enhanced operation config for VectorBT
                operation_config = {
                    'operation_type': OperationType.BACKTESTING,
                    'data_size': len(train_data),
                    'data_dimensions': train_data.shape if hasattr(train_data, 'shape') else (len(train_data),),
                    'enable_vectorbt': True,
                    'prefer_vectorbt': enhanced_config.prefer_vectorbt,
                    'rolling_window': enhanced_config.vectorbt_rolling_window,
                    'chunk_size': enhanced_config.vectorbt_chunk_size,
                    'enable_hardware_optimization': enhanced_config.enable_hardware_optimization,
                    'enable_advanced_validation': enhanced_config.enable_advanced_validation
                }
                
                # Prepare data for VectorBT optimization
                optimization_data = {
                    'train_data': train_data,
                    'test_data': test_data,
                    'search_space': search_space,
                    'enhanced_config': enhanced_config
                }
                
                # Optimize using VectorBT with enhanced features
                result = await self.vectorization_manager.optimize_operation(
                    OperationType.BACKTESTING,
                    optimization_data,
                    operation_config,
                    prefer_vectorbt=enhanced_config.prefer_vectorbt
                )
                
                # Extract results with enhanced metadata
                optimized_params = result.metadata.get('best_params', {})
                best_score = result.metadata.get('best_score', 0.0)
                
                # Generate explainability if available
                explainability_results = {}
                if self.explainer and enhanced_config.enable_explainability:
                    try:
                        explainability_results = await self._generate_explainability(
                            optimized_params, train_data, test_data
                        )
                    except Exception as e:
                        self.logger.warning(f"Explainability generation failed: {e}")
                
                # Enhanced result with comprehensive metadata
                return {
                    'optimized_parameters': optimized_params,
                    'best_score': best_score,
                    'vectorbt_acceleration_factor': result.performance_gain,
                    'total_combinations_tested': result.metadata.get('combinations_tested', 0),
                    'optimization_strategy': result.metadata.get('strategy_used', 'unknown'),
                    'hardware_optimization_used': result.metadata.get('hardware_optimization', False),
                    'vectorbt_rolling_window': enhanced_config.vectorbt_rolling_window,
                    'vectorbt_chunk_size': enhanced_config.vectorbt_chunk_size,
                    'explainability_results': explainability_results,
                    'optimization_metadata': result.metadata
                }
            else:
                # Fallback to traditional optimization
                self.logger.warning("VectorBT not available, falling back to traditional optimization")
                return await self._run_traditional_optimization(search_space, train_data, test_data, enhanced_config)
                
        except Exception as e:
            self.logger.error(f"Enhanced VectorBT optimization failed: {e}")
            return {'error': str(e)}

    async def _run_traditional_optimization(
        self, 
        search_space: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run traditional grid search optimization with optimized single-detection approach."""
        try:
            self.logger.info("🚀 Detecting SR levels once with relaxed parameters (FAST MODE)...")
            
            # OPTIMIZATION: Detect levels ONCE with relaxed parameters to get comprehensive candidate pool
            relaxed_params = {
                'min_touches': 1,  # Relaxed to capture all potential levels
                'strength_threshold': 0.1,  # Very relaxed threshold
                'distance_threshold': 0.01,
                'lookback_periods': 50,
                'volume_threshold': 0.5,  # Relaxed volume requirement
                'touch_tolerance': 0.5
            }
            
            all_detected_levels = self._detect_sr_levels(train_data, relaxed_params)
            self.logger.info(f"✅ Detected {len(all_detected_levels)} candidate SR levels (will filter by parameters)")
            
            if not all_detected_levels:
                self.logger.warning("⚠️ No SR levels detected even with relaxed parameters")
                return {
                    'optimized_parameters': relaxed_params,
                    'best_score': 0.0,
                    'total_combinations_tested': 0,
                    'error': 'No SR levels detected'
                }
            
            best_score = 0.0
            best_params = {}
            total_combinations = 0
            
            # OPTIMIZATION: Now filter and evaluate (much faster than re-detecting)
            for min_touches in range(2, 6):
                for strength_threshold in [0.3, 0.5, 0.7]:
                    params = {
                        'min_touches': min_touches,
                        'strength_threshold': strength_threshold,
                        'distance_threshold': 0.01,
                        'lookback_periods': 50,
                        'volume_threshold': 1.0
                    }
                    
                    # Filter already-detected levels instead of re-detecting
                    score = self._evaluate_sr_parameters_filtered(
                        params, all_detected_levels, train_data, test_data
                    )
                    total_combinations += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
            
            self.logger.info(f"✅ Optimized parameter search: {total_combinations} combinations tested (detected once, filtered {total_combinations} times)")
            return {
                'optimized_parameters': best_params,
                'quality_thresholds': {
                    'min_strength': best_params.get('strength_threshold', 0.5),
                    'min_touches': best_params.get('min_touches', 2),
                    'min_quality_score': 0.5
                },
                'parameter_optimization_metrics': {
                    'best_score': best_score,
                    'total_combinations': total_combinations,
                    'method': 'traditional_grid_search'
                },
                'best_score': best_score,
                'total_combinations_tested': total_combinations,
                'optimization_time': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Traditional optimization failed: {e}")
            return {'error': str(e)}

    def _evaluate_sr_parameters(self, params: Dict[str, Any], train_data: Any, test_data: Any) -> float:
        """Evaluate SR parameters using real SR detection and backtesting.
        
        NOTE: For optimization loops, use _evaluate_sr_parameters_filtered() instead
        to avoid re-detecting levels on every iteration.
        """
        try:
            # Validate parameters first
            if not self._validate_parameters(params):
                self.logger.warning("Invalid parameters provided for evaluation")
                return 0.0
            
            # Detect SR levels using the parameters
            sr_levels = self._detect_sr_levels(train_data, params)
            if not sr_levels or len(sr_levels) == 0:
                self.logger.warning("No SR levels detected with given parameters")
                return 0.0
            
            # Backtest the SR levels on test data
            backtest_results = self._backtest_sr_levels(sr_levels, test_data, params)
            if not backtest_results:
                self.logger.warning("Backtest failed for SR levels")
                return 0.0
            
            # Calculate composite score based on backtest results
            score = self._calculate_composite_score(backtest_results, params)
            
            self.logger.debug(f"Parameter evaluation completed: score={score:.4f}, levels={len(sr_levels)}")
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return 0.0
    
    def _filter_sr_levels_by_params(self, all_levels: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter already-detected SR levels based on parameter thresholds.
        
        This is MUCH faster than re-detecting levels for each parameter combination.
        """
        try:
            min_touches = params.get('min_touches', 2)
            strength_threshold = params.get('strength_threshold', 0.5)
            volume_threshold = params.get('volume_threshold', 1.0)
            
            filtered_levels = []
            for level in all_levels:
                # Extract level attributes (handle both dict and object formats)
                touch_count = level.get('touch_count', 0) if isinstance(level, dict) else getattr(level, 'touch_count', 0)
                strength = level.get('strength', 0.0) if isinstance(level, dict) else getattr(level, 'strength', 0.0)
                volume_ratio = level.get('volume_ratio', 1.0) if isinstance(level, dict) else getattr(level, 'volume_confirmation_score', 1.0)
                
                # Apply parameter-based filtering
                if (touch_count >= min_touches and 
                    strength >= strength_threshold and 
                    volume_ratio >= volume_threshold):
                    filtered_levels.append(level)
            
            return filtered_levels
            
        except Exception as e:
            self.logger.error(f"Level filtering failed: {e}")
            return []
    
    def _evaluate_sr_parameters_filtered(
        self, 
        params: Dict[str, Any], 
        all_detected_levels: List[Dict[str, Any]],
        train_data: Any, 
        test_data: Any
    ) -> float:
        """Evaluate SR parameters by filtering already-detected levels (FAST).
        
        This method is optimized for parameter optimization loops where you want
        to test many parameter combinations without re-detecting levels each time.
        
        Args:
            params: Parameter dictionary to test
            all_detected_levels: Pre-detected SR levels (from relaxed parameters)
            train_data: Training data (for validation)
            test_data: Test data (for backtesting)
            
        Returns:
            Evaluation score (0.0 to 1.0)
        """
        try:
            # Validate parameters first
            if not self._validate_parameters(params):
                self.logger.warning("Invalid parameters provided for evaluation")
                return 0.0
            
            # OPTIMIZATION: Filter pre-detected levels instead of re-detecting
            filtered_levels = self._filter_sr_levels_by_params(all_detected_levels, params)
            if not filtered_levels or len(filtered_levels) == 0:
                self.logger.debug(f"No levels passed filter: min_touches={params.get('min_touches')}, strength={params.get('strength_threshold')}")
                return 0.0
            
            # Backtest the filtered SR levels on test data
            backtest_results = self._backtest_sr_levels(filtered_levels, test_data, params)
            if not backtest_results:
                self.logger.debug("Backtest failed for filtered SR levels")
                return 0.0
            
            # Calculate composite score based on backtest results
            score = self._calculate_composite_score(backtest_results, params)
            
            self.logger.debug(f"Filtered parameter evaluation: score={score:.4f}, levels={len(filtered_levels)}/{len(all_detected_levels)}")
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Filtered parameter evaluation failed: {e}")
            return 0.0

    async def _evaluate_sr_parameters_enhanced(
        self, 
        params: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> float:
        """Enhanced parameter evaluation with ML utilities and advanced validation."""
        try:
            # Base evaluation score
            base_score = self._evaluate_sr_parameters(params, train_data, test_data)
            
            # Enhanced evaluation with ML utilities
            enhanced_score = base_score
            
            # Collect scores from different validation methods
            validation_scores = {}
            validation_weights = {}
            
            # Add OOF validation if available
            if self.oof_manager and enhanced_config.enable_oof_validation:
                try:
                    oof_score = await self._evaluate_with_oof_validation(params, train_data, test_data)
                    validation_scores['oof'] = oof_score
                    validation_weights['oof'] = 0.3  # 30% weight for OOF
                except Exception as e:
                    self.logger.warning(f"OOF validation failed: {e}")
            
            # Add purged CV validation if available
            if PURGED_CV_AVAILABLE and enhanced_config.enable_purged_cv:
                try:
                    purged_score = await self._evaluate_with_purged_cv(params, train_data, test_data)
                    validation_scores['purged_cv'] = purged_score
                    validation_weights['purged_cv'] = 0.3  # 30% weight for Purged CV
                except Exception as e:
                    self.logger.warning(f"Purged CV validation failed: {e}")
            
            # Add unified evaluation if available
            if self.evaluator and enhanced_config.enable_unified_evaluation:
                try:
                    evaluation_score = await self._evaluate_with_unified_evaluator(params, train_data, test_data)
                    validation_scores['unified'] = evaluation_score
                    validation_weights['unified'] = 0.4  # 40% weight for unified evaluation
                except Exception as e:
                    self.logger.warning(f"Unified evaluation failed: {e}")
            
            # Combine scores using weighted average
            if validation_scores:
                enhanced_score = self._combine_evaluation_scores(validation_scores, validation_weights)
            else:
                # Fallback to base score if no validation methods available
                enhanced_score = base_score
            
            # Apply data leakage penalty if detected
            if self.leakage_detector and enhanced_config.enable_data_leakage_detection:
                try:
                    leakage_report = await self.leakage_detector.detect_temporal_leakage(train_data, test_data)
                    # Handle both dict and object response types
                    has_leakage = leakage_report.get('has_leakage', False) if isinstance(leakage_report, dict) else getattr(leakage_report, 'has_leakage', False)
                    if has_leakage:
                        enhanced_score *= 0.5  # Penalize for data leakage
                        self.logger.warning("Data leakage detected, applying penalty to score")
                except Exception as e:
                    self.logger.warning(f"Data leakage detection failed: {e}")
            
            return min(enhanced_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Enhanced parameter evaluation failed: {e}")
            return base_score if 'base_score' in locals() else 0.0

    async def _evaluate_with_oof_validation(
        self, 
        params: Dict[str, Any], 
        train_data: Any, 
        test_data: Any
    ) -> float:
        """Evaluate parameters using proper OOF validation with temporal splits."""
        try:
            if not PANDAS_AVAILABLE or not isinstance(train_data, pd.DataFrame):
                return 0.0
            
            oof_scores = []
            n_splits = self.oof_config.n_splits
            
            # Create proper temporal splits for OOF validation
            for fold in range(n_splits):
                # Calculate split boundaries
                fold_size = len(train_data) // n_splits
                train_start = 0
                train_end = fold_size * (fold + 1)
                val_start = train_end
                val_end = min(val_start + fold_size, len(train_data))
                
                if val_end <= val_start or train_end <= train_start:
                    continue
                
                # Create fold-specific train/validation splits
                fold_train_data = train_data.iloc[train_start:train_end]
                fold_val_data = train_data.iloc[val_start:val_end]
                
                # Evaluate on this fold
                fold_score = self._evaluate_sr_parameters(params, fold_train_data, fold_val_data)
                oof_scores.append(fold_score)
                
                self.logger.debug(f"OOF fold {fold+1}/{n_splits}: score={fold_score:.4f}")
            
            if not oof_scores:
                self.logger.warning("No valid OOF folds created")
                return 0.0
            
            return np.mean(oof_scores)
        except Exception as e:
            self.logger.error(f"OOF validation evaluation failed: {e}")
            return 0.0

    async def _evaluate_with_purged_cv(
        self, 
        params: Dict[str, Any], 
        train_data: Any, 
        test_data: Any
    ) -> float:
        """Evaluate parameters using purged cross-validation with temporal gaps."""
        try:
            if not PANDAS_AVAILABLE or not isinstance(train_data, pd.DataFrame):
                return 0.0
            
            purged_scores = []
            n_splits = 5  # Default number of splits
            embargo_pct = 0.01  # 1% embargo period
            
            # Calculate embargo period in data points
            embargo_periods = max(1, int(len(train_data) * embargo_pct))
            
            # Create purged CV splits with embargo
            for fold in range(n_splits):
                # Calculate split boundaries with embargo
                fold_size = len(train_data) // n_splits
                train_start = 0
                train_end = fold_size * (fold + 1)
                val_start = train_end + embargo_periods  # Add embargo gap
                val_end = min(val_start + fold_size, len(train_data))
                
                if val_end <= val_start or train_end <= train_start:
                    continue
                
                # Create fold-specific train/validation splits with embargo
                fold_train_data = train_data.iloc[train_start:train_end]
                fold_val_data = train_data.iloc[val_start:val_end]
                
                # Evaluate on this fold
                fold_score = self._evaluate_sr_parameters(params, fold_train_data, fold_val_data)
                purged_scores.append(fold_score)
                
                self.logger.debug(f"Purged CV fold {fold+1}/{n_splits}: score={fold_score:.4f}, embargo={embargo_periods}")
            
            if not purged_scores:
                self.logger.warning("No valid purged CV folds created")
                return 0.0
            
            return np.mean(purged_scores)
        except Exception as e:
            self.logger.error(f"Purged CV evaluation failed: {e}")
            return 0.0

    async def _evaluate_with_unified_evaluator(
        self, 
        params: Dict[str, Any], 
        train_data: Any, 
        test_data: Any
    ) -> float:
        """Evaluate parameters using unified evaluator."""
        try:
            # Use unified evaluator for comprehensive evaluation
            evaluation_result = await self.evaluator.evaluate_model(
                params, train_data, test_data
            )
            return evaluation_result.get('overall_score', 0.0)
        except Exception as e:
            self.logger.error(f"Unified evaluation failed: {e}")
            return 0.0

    async def _generate_explainability(
        self, 
        best_params: Dict[str, Any], 
        train_data: Any, 
        test_data: Any
    ) -> Dict[str, Any]:
        """Generate SHAP and LIME explanations for the best parameters."""
        try:
            # Prepare data for explainability
            X_train = train_data[['open', 'high', 'low', 'close', 'volume']].values
            X_test = test_data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Generate predictions using best parameters
            y_train_pred = self._predict_with_params(best_params, X_train)
            y_test_pred = self._predict_with_params(best_params, X_test)
            
            # Generate SHAP explanations
            shap_explanations = {}
            if self.explainability_config.enable_shap:
                try:
                    shap_explanations = await self.explainer.generate_shap_explanations(
                        X_test, y_test_pred, sample_size=self.explainability_config.shap_sample_size
                    )
                except Exception as e:
                    self.logger.warning(f"SHAP explanation generation failed: {e}")
            
            # Generate LIME explanations
            lime_explanations = {}
            if self.explainability_config.enable_lime:
                try:
                    lime_explanations = await self.explainer.generate_lime_explanations(
                        X_test, y_test_pred, sample_size=self.explainability_config.lime_sample_size
                    )
                except Exception as e:
                    self.logger.warning(f"LIME explanation generation failed: {e}")
            
            return {
                'shap_explanations': shap_explanations,
                'lime_explanations': lime_explanations,
                'feature_importance': self._extract_feature_importance(best_params),
                'parameter_sensitivity': self._analyze_parameter_sensitivity(best_params)
            }
            
        except Exception as e:
            self.logger.error(f"Explainability generation failed: {e}")
            return {}

    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values before optimization."""
        required_params = ['min_touches', 'strength_threshold', 'distance_threshold']
        
        for param in required_params:
            if param not in params:
                self.logger.warning(f"Missing required parameter: {param}")
                return False
            
            value = params[param]
            if not isinstance(value, (int, float)) or value <= 0:
                self.logger.warning(f"Invalid parameter value for {param}: {value}")
                return False
        
        # Validate parameter relationships
        if params['min_touches'] > 20:  # Unrealistic
            self.logger.warning(f"min_touches too high: {params['min_touches']}")
            return False
        
        if params['strength_threshold'] > 1.0 or params['strength_threshold'] < 0:
            self.logger.warning(f"strength_threshold out of range: {params['strength_threshold']}")
            return False
        
        return True

    def _detect_sr_levels(self, data: Any, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect SR levels using EnhancedSRDetector when available, fallback to custom methods."""
        tprint("🔍 SR Detection: Starting level detection", "info")
        try:
            if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
                tprint("❌ SR Detection: Pandas not available or invalid data", "error")
                return []
            
            # Try to use EnhancedSRDetector if available (preferred method)
            if SR_DETECTION_AVAILABLE and EnhancedSRDetector is not None:
                try:
                    tprint("🚀 SR Detection: Using EnhancedSRDetector", "info")
                    
                    # Create SR detector configuration from parameters
                    sr_config = {
                        'min_touches': params.get('min_touches', 2),
                        'tolerance_pct': params.get('touch_tolerance', 0.5),
                        'lookback_periods': params.get('lookback_periods', 100),
                        'strength_threshold': params.get('strength_threshold', 0.5),
                        'distance_threshold': params.get('distance_threshold', 0.01),
                        'volume_threshold': params.get('volume_threshold', 1.0),
                        'memory_efficient': True,
                        'use_parallel': False,
                        'disable_dbscan_clustering': True
                    }
                    
                    # Create detector and detect SR levels
                    detector = EnhancedSRDetector(sr_config)
                    sr_levels_result = detector.detect_sr_levels(data)
                    
                    # Convert SRLevel objects to dictionaries for compatibility
                    if isinstance(sr_levels_result, dict):
                        # Extract levels from dict result
                        support_levels = sr_levels_result.get('support_levels', [])
                        resistance_levels = sr_levels_result.get('resistance_levels', [])
                        all_levels = support_levels + resistance_levels
                    elif isinstance(sr_levels_result, list):
                        all_levels = sr_levels_result
                    else:
                        all_levels = []
                    
                    # Convert SRLevel objects to dicts
                    dict_levels = []
                    for level in all_levels:
                        if hasattr(level, 'price'):
                            dict_levels.append({
                                'price': level.price,
                                'strength': getattr(level, 'strength', 0.5),
                                'type': getattr(level, 'type', 'support'),
                                'touch_count': getattr(level, 'touch_count', 0)
                            })
                        elif isinstance(level, dict):
                            dict_levels.append(level)
                    
                    tprint(f"✅ SR Detection: Detected {len(dict_levels)} levels using EnhancedSRDetector", "success")
                    return dict_levels
                    
                except Exception as e:
                    tprint(f"⚠️ SR Detection: EnhancedSRDetector failed: {e}, falling back to custom methods", "warning")
                    # Fall through to custom methods below
            
            # Fallback to custom SR detection methods
            tprint("📊 SR Detection: Using custom detection methods", "info")
            
            # Extract price data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values if 'volume' in data.columns else None
            tprint_data_format(data, "SR detection input data")
            
            all_sr_levels = []
            
            # Method 1: Traditional local minima/maxima
            traditional_levels = self._detect_traditional_sr_levels(high, low, close, volume, params)
            all_sr_levels.extend(traditional_levels)
            
            # Method 2: High Volume Nodes (HVN)
            hvn_levels = self._detect_hvn_levels(high, low, close, volume, params)
            all_sr_levels.extend(hvn_levels)
            
            # Method 3: Price reversal patterns
            reversal_levels = self._detect_reversal_levels(high, low, close, volume, params)
            all_sr_levels.extend(reversal_levels)
            
            # Method 4: Fibonacci levels
            fib_levels = self._detect_fibonacci_levels(high, low, close, params)
            all_sr_levels.extend(fib_levels)
            
            # Method 5: Fractal-based levels
            fractal_levels = self._detect_fractal_levels(high, low, close, volume, params)
            all_sr_levels.extend(fractal_levels)
            
            # Method 6: Consolidation ranges
            consolidation_levels = self._detect_consolidation_levels(high, low, close, volume, params)
            all_sr_levels.extend(consolidation_levels)
            
            # Method 7: Pivot points
            pivot_levels = self._detect_pivot_levels(high, low, close, params)
            all_sr_levels.extend(pivot_levels)
            
            # Consolidate and merge similar levels
            consolidated_levels = self._consolidate_sr_levels(all_sr_levels, params)
            
            self.logger.info(f"Detected {len(consolidated_levels)} SR levels using {len(all_sr_levels)} raw detections")
            return consolidated_levels
            
        except Exception as e:
            self.logger.error(f"SR level detection failed: {e}")
            return []

    def _detect_traditional_sr_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                     volume: Optional[np.ndarray], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect traditional SR levels using local minima/maxima."""
        sr_levels = []
        min_touches = params.get('min_touches', 2)
        strength_threshold = params.get('strength_threshold', 0.5)
        distance_threshold = params.get('distance_threshold', 0.01)
        lookback_periods = params.get('lookback_periods', 50)
        
        for i in range(lookback_periods, len(high)):
            # Check for support level (local minimum)
            if self._is_local_minimum(low, i, lookback_periods):
                level = self._analyze_sr_level(
                    high, low, close, volume, i, 'support', 
                    min_touches, strength_threshold, distance_threshold, 'traditional'
                )
                if level:
                    sr_levels.append(level)
            
            # Check for resistance level (local maximum)
            if self._is_local_maximum(high, i, lookback_periods):
                level = self._analyze_sr_level(
                    high, low, close, volume, i, 'resistance',
                    min_touches, strength_threshold, distance_threshold, 'traditional'
                )
                if level:
                    sr_levels.append(level)
        
        return sr_levels

    def _detect_hvn_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          volume: Optional[np.ndarray], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect High Volume Node (HVN) levels."""
        if volume is None:
            return []
        
        sr_levels = []
        volume_threshold = params.get('volume_threshold', 1.5)
        lookback_periods = params.get('lookback_periods', 50)
        
        # Calculate volume moving average
        volume_ma = self._rolling_mean(volume, lookback_periods)
        
        for i in range(lookback_periods, len(high)):
            if volume[i] > volume_ma[i] * volume_threshold:
                # Check if this is a significant price level
                price_level = (high[i] + low[i]) / 2
                level_type = 'support' if close[i] < price_level else 'resistance'
                
                level = self._analyze_sr_level(
                    high, low, close, volume, i, level_type,
                    params.get('min_touches', 2), params.get('strength_threshold', 0.5),
                    params.get('distance_threshold', 0.01), 'hvn'
                )
                if level:
                    level['volume_ratio'] = volume[i] / volume_ma[i]
                    sr_levels.append(level)
        
        return sr_levels

    def _detect_reversal_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                               volume: Optional[np.ndarray], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect price reversal pattern levels."""
        sr_levels = []
        reversal_threshold = params.get('reversal_threshold', 0.02)
        min_reversal_strength = params.get('min_reversal_strength', 0.01)
        
        for i in range(2, len(high) - 2):
            # Check for bullish reversal (hammer, doji, etc.)
            if self._is_bullish_reversal(high, low, close, i):
                level = self._analyze_sr_level(
                    high, low, close, volume, i, 'support',
                    params.get('min_touches', 2), min_reversal_strength,
                    params.get('distance_threshold', 0.01), 'reversal'
                )
                if level:
                    level['reversal_strength'] = self._calculate_reversal_strength(high, low, close, i)
                    sr_levels.append(level)
            
            # Check for bearish reversal (shooting star, doji, etc.)
            if self._is_bearish_reversal(high, low, close, i):
                level = self._analyze_sr_level(
                    high, low, close, volume, i, 'resistance',
                    params.get('min_touches', 2), min_reversal_strength,
                    params.get('distance_threshold', 0.01), 'reversal'
                )
                if level:
                    level['reversal_strength'] = self._calculate_reversal_strength(high, low, close, i)
                    sr_levels.append(level)
        
        return sr_levels

    def _detect_fibonacci_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect Fibonacci retracement and extension levels."""
        sr_levels = []
        lookback_periods = params.get('lookback_periods', 50)
        
        for i in range(lookback_periods, len(high)):
            # Find recent swing high and low
            swing_high_idx = self._find_swing_high(high, i, lookback_periods)
            swing_low_idx = self._find_swing_low(low, i, lookback_periods)
            
            if swing_high_idx is not None and swing_low_idx is not None:
                swing_high = high[swing_high_idx]
                swing_low = low[swing_low_idx]
                price_range = swing_high - swing_low
                
                # Calculate Fibonacci levels
                fib_levels = {
                    0.236: swing_low + price_range * 0.236,
                    0.382: swing_low + price_range * 0.382,
                    0.5: swing_low + price_range * 0.5,
                    0.618: swing_low + price_range * 0.618,
                    0.786: swing_low + price_range * 0.786,
                    1.0: swing_high,
                    1.272: swing_high + price_range * 0.272,
                    1.414: swing_high + price_range * 0.414,
                    1.618: swing_high + price_range * 0.618
                }
                
                for fib_ratio, fib_price in fib_levels.items():
                    if swing_low <= fib_price <= swing_high * 1.2:  # Within reasonable range
                        level_type = 'support' if fib_price < close[i] else 'resistance'
                        level = {
                            'price': fib_price,
                            'type': level_type,
                            'method': 'fibonacci',
                            'fib_ratio': fib_ratio,
                            'touches': 1,  # Will be calculated in analyze_sr_level
                            'strength': 0.5,  # Default strength
                            'volume_confirmation': 1.0,
                            'index': i,
                            'quality_score': 0.6  # Fibonacci levels have moderate quality
                        }
                        sr_levels.append(level)
        
        return sr_levels

    def _detect_fractal_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                              volume: Optional[np.ndarray], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect fractal-based SR levels."""
        sr_levels = []
        fractal_period = params.get('fractal_period', 5)
        
        for i in range(fractal_period, len(high) - fractal_period):
            # Check for fractal high (resistance)
            if self._is_fractal_high(high, i, fractal_period):
                level = self._analyze_sr_level(
                    high, low, close, volume, i, 'resistance',
                    params.get('min_touches', 2), params.get('strength_threshold', 0.5),
                    params.get('distance_threshold', 0.01), 'fractal'
                )
                if level:
                    sr_levels.append(level)
            
            # Check for fractal low (support)
            if self._is_fractal_low(low, i, fractal_period):
                level = self._analyze_sr_level(
                    high, low, close, volume, i, 'support',
                    params.get('min_touches', 2), params.get('strength_threshold', 0.5),
                    params.get('distance_threshold', 0.01), 'fractal'
                )
                if level:
                    sr_levels.append(level)
        
        return sr_levels

    def _detect_consolidation_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                   volume: Optional[np.ndarray], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect consolidation range levels."""
        sr_levels = []
        consolidation_periods = params.get('consolidation_periods', 20)
        max_range_pct = params.get('max_range_pct', 0.05)  # 5% max range for consolidation
        
        for i in range(consolidation_periods, len(high)):
            # Check if this period represents a consolidation
            period_high = np.max(high[i-consolidation_periods:i])
            period_low = np.min(low[i-consolidation_periods:i])
            period_range = (period_high - period_low) / period_low
            
            if period_range <= max_range_pct:
                # This is a consolidation - add support and resistance levels
                support_level = {
                    'price': period_low,
                    'type': 'support',
                    'method': 'consolidation',
                    'touches': consolidation_periods,
                    'strength': 0.7,  # Consolidations are strong levels
                    'volume_confirmation': 1.0,
                    'index': i,
                    'quality_score': 0.8,
                    'consolidation_periods': consolidation_periods
                }
                sr_levels.append(support_level)
                
                resistance_level = {
                    'price': period_high,
                    'type': 'resistance',
                    'method': 'consolidation',
                    'touches': consolidation_periods,
                    'strength': 0.7,
                    'volume_confirmation': 1.0,
                    'index': i,
                    'quality_score': 0.8,
                    'consolidation_periods': consolidation_periods
                }
                sr_levels.append(resistance_level)
        
        return sr_levels

    def _detect_pivot_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect pivot point levels."""
        sr_levels = []
        pivot_period = params.get('pivot_period', 20)
        
        for i in range(pivot_period, len(high) - pivot_period):
            # Calculate pivot point
            prev_high = np.max(high[i-pivot_period:i])
            prev_low = np.min(low[i-pivot_period:i])
            prev_close = close[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            
            # Add pivot levels
            levels = [
                (r3, 'resistance', 'pivot_r3'),
                (r2, 'resistance', 'pivot_r2'),
                (r1, 'resistance', 'pivot_r1'),
                (pivot, 'support', 'pivot'),
                (s1, 'support', 'pivot_s1'),
                (s2, 'support', 'pivot_s2'),
                (s3, 'support', 'pivot_s3')
            ]
            
            for price, level_type, level_name in levels:
                if prev_low <= price <= prev_high * 1.2:  # Within reasonable range
                    level = {
                        'price': price,
                        'type': level_type,
                        'method': 'pivot',
                        'level_name': level_name,
                        'touches': 1,
                        'strength': 0.6,
                        'volume_confirmation': 1.0,
                        'index': i,
                        'quality_score': 0.7
                    }
                    sr_levels.append(level)
        
        return sr_levels

    def _consolidate_sr_levels(self, all_levels: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolidate and merge similar SR levels."""
        if not all_levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(all_levels, key=lambda x: x['price'])
        consolidated = []
        distance_threshold = params.get('distance_threshold', 0.01)
        
        i = 0
        while i < len(sorted_levels):
            current_level = sorted_levels[i]
            merged_levels = [current_level]
            
            # Find levels within distance threshold
            j = i + 1
            while j < len(sorted_levels):
                next_level = sorted_levels[j]
                price_diff = abs(next_level['price'] - current_level['price']) / current_level['price']
                
                if price_diff <= distance_threshold and next_level['type'] == current_level['type']:
                    merged_levels.append(next_level)
                    j += 1
                else:
                    break
            
            # Merge the levels
            if len(merged_levels) > 1:
                merged_level = self._merge_sr_levels(merged_levels)
                consolidated.append(merged_level)
            else:
                consolidated.append(current_level)
            
            i = j
        
        return consolidated

    def _merge_sr_levels(self, levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple similar SR levels into one."""
        if not levels:
            return {}
        
        # Calculate weighted average price
        total_weight = sum(level.get('quality_score', 0.5) for level in levels)
        weighted_price = sum(level['price'] * level.get('quality_score', 0.5) for level in levels) / total_weight
        
        # Sum touches and calculate average strength
        total_touches = sum(level.get('touches', 1) for level in levels)
        avg_strength = sum(level.get('strength', 0.5) for level in levels) / len(levels)
        
        # Get the best quality score
        max_quality = max(level.get('quality_score', 0.5) for level in levels)
        
        # Combine methods
        methods = list(set(level.get('method', 'unknown') for level in levels))
        
        return {
            'price': weighted_price,
            'type': levels[0]['type'],
            'method': '+'.join(methods),
            'touches': total_touches,
            'strength': avg_strength,
            'volume_confirmation': 1.0,
            'index': levels[0]['index'],
            'quality_score': max_quality,
            'merged_from': len(levels)
        }

    def _is_local_minimum(self, low: np.ndarray, idx: int, window: int) -> bool:
        """Check if index is a local minimum."""
        start = max(0, idx - window // 2)
        end = min(len(low), idx + window // 2 + 1)
        return low[idx] == np.min(low[start:end])

    def _is_local_maximum(self, high: np.ndarray, idx: int, window: int) -> bool:
        """Check if index is a local maximum."""
        start = max(0, idx - window // 2)
        end = min(len(high), idx + window // 2 + 1)
        return high[idx] == np.max(high[start:end])

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean."""
        result = np.full_like(data, np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        return result

    def _is_bullish_reversal(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int) -> bool:
        """Check for bullish reversal patterns (hammer, doji, etc.)."""
        if idx < 2 or idx >= len(high) - 2:
            return False
        
        # Hammer pattern
        body = abs(close[idx] - high[idx])
        lower_shadow = low[idx] - min(close[idx], high[idx])
        upper_shadow = max(close[idx], high[idx]) - high[idx]
        
        # Hammer: small body, long lower shadow, small upper shadow
        if body < (high[idx] - low[idx]) * 0.3 and lower_shadow > body * 2 and upper_shadow < body:
            return True
        
        # Doji pattern
        if body < (high[idx] - low[idx]) * 0.1:
            return True
        
        return False

    def _is_bearish_reversal(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int) -> bool:
        """Check for bearish reversal patterns (shooting star, doji, etc.)."""
        if idx < 2 or idx >= len(high) - 2:
            return False
        
        # Shooting star pattern
        body = abs(close[idx] - high[idx])
        lower_shadow = low[idx] - min(close[idx], high[idx])
        upper_shadow = max(close[idx], high[idx]) - high[idx]
        
        # Shooting star: small body, long upper shadow, small lower shadow
        if body < (high[idx] - low[idx]) * 0.3 and upper_shadow > body * 2 and lower_shadow < body:
            return True
        
        # Doji pattern
        if body < (high[idx] - low[idx]) * 0.1:
            return True
        
        return False

    def _calculate_reversal_strength(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int) -> float:
        """Calculate reversal pattern strength."""
        if idx < 2 or idx >= len(high) - 2:
            return 0.0
        
        # Calculate body size relative to total range
        body_size = abs(close[idx] - high[idx])
        total_range = high[idx] - low[idx]
        
        if total_range == 0:
            return 0.0
        
        # Stronger reversal if body is smaller relative to range
        body_ratio = body_size / total_range
        return max(0.0, 1.0 - body_ratio)

    def _find_swing_high(self, high: np.ndarray, idx: int, lookback: int) -> Optional[int]:
        """Find the most recent swing high."""
        start = max(0, idx - lookback)
        end = idx
        
        if end - start < 3:
            return None
        
        # Find local maximum in the range
        local_max_idx = start
        for i in range(start + 1, end):
            if high[i] > high[local_max_idx]:
                local_max_idx = i
        
        return local_max_idx

    def _find_swing_low(self, low: np.ndarray, idx: int, lookback: int) -> Optional[int]:
        """Find the most recent swing low."""
        start = max(0, idx - lookback)
        end = idx
        
        if end - start < 3:
            return None
        
        # Find local minimum in the range
        local_min_idx = start
        for i in range(start + 1, end):
            if low[i] < low[local_min_idx]:
                local_min_idx = i
        
        return local_min_idx

    def _is_fractal_high(self, high: np.ndarray, idx: int, period: int) -> bool:
        """Check if index is a fractal high."""
        if idx < period or idx >= len(high) - period:
            return False
        
        # Check if current high is higher than all highs in the period
        for i in range(idx - period, idx + period + 1):
            if i != idx and high[i] >= high[idx]:
                return False
        
        return True

    def _is_fractal_low(self, low: np.ndarray, idx: int, period: int) -> bool:
        """Check if index is a fractal low."""
        if idx < period or idx >= len(low) - period:
            return False
        
        # Check if current low is lower than all lows in the period
        for i in range(idx - period, idx + period + 1):
            if i != idx and low[i] <= low[idx]:
                return False
        
        return True

    def _analyze_sr_level(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         volume: Optional[np.ndarray], idx: int, level_type: str,
                         min_touches: int, strength_threshold: float, 
                         distance_threshold: float, method: str = 'unknown') -> Optional[Dict[str, Any]]:
        """Analyze a potential SR level for quality."""
        try:
            level_price = low[idx] if level_type == 'support' else high[idx]
            touches = 0
            total_strength = 0.0
            
            # Count touches and calculate strength
            for i in range(max(0, idx - 100), min(len(high), idx + 100)):
                if i == idx:
                    continue
                
                if level_type == 'support':
                    if abs(low[i] - level_price) / level_price <= distance_threshold:
                        touches += 1
                        # Calculate bounce strength
                        bounce_strength = (close[i] - low[i]) / low[i]
                        total_strength += bounce_strength
                else:  # resistance
                    if abs(high[i] - level_price) / level_price <= distance_threshold:
                        touches += 1
                        # Calculate rejection strength
                        rejection_strength = (high[i] - close[i]) / high[i]
                        total_strength += rejection_strength
            
            if touches < min_touches:
                return None
            
            avg_strength = total_strength / touches if touches > 0 else 0.0
            
            # Check if level meets strength threshold
            if avg_strength < strength_threshold:
                return None
            
            # Calculate volume confirmation if available
            volume_confirmation = 1.0
            if volume is not None:
                avg_volume = np.mean(volume[max(0, idx-10):min(len(volume), idx+10)])
                level_volume = volume[idx]
                volume_confirmation = min(2.0, level_volume / avg_volume) if avg_volume > 0 else 1.0
            
            return {
                'price': level_price,
                'type': level_type,
                'method': method,
                'touches': touches,
                'strength': avg_strength,
                'volume_confirmation': volume_confirmation,
                'index': idx,
                'quality_score': self._calculate_level_quality(touches, avg_strength, volume_confirmation)
            }
            
        except Exception as e:
            self.logger.error(f"SR level analysis failed: {e}")
            return None

    def _calculate_level_quality(self, touches: int, strength: float, volume_confirmation: float) -> float:
        """Calculate quality score for an SR level."""
        # Normalize touches (2-10 range to 0-1)
        touches_score = min(1.0, (touches - 2) / 8.0)
        
        # Normalize strength (0-0.1 range to 0-1)
        strength_score = min(1.0, strength / 0.1)
        
        # Normalize volume confirmation (0.5-2.0 range to 0-1)
        volume_score = min(1.0, max(0.0, (volume_confirmation - 0.5) / 1.5))
        
        # Weighted combination
        return (touches_score * 0.4 + strength_score * 0.4 + volume_score * 0.2)

    def _backtest_sr_levels(self, sr_levels: List[Dict[str, Any]], test_data: Any, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest SR levels on test data."""
        tprint("📊 Backtesting: Starting SR level backtesting", "info")
        try:
            if not PANDAS_AVAILABLE or not isinstance(test_data, pd.DataFrame):
                tprint("❌ Backtesting: Pandas not available or invalid test data", "error")
                return {}
            
            if not sr_levels:
                tprint("⚠️ Backtesting: No SR levels to backtest", "warning")
                return {}
            
            tprint(f"📊 Backtesting: Testing {len(sr_levels)} SR levels", "info")
            
            high = test_data['high'].values
            low = test_data['low'].values
            close = test_data['close'].values
            
            total_trades = 0
            successful_trades = 0
            total_pnl = 0.0
            level_performance = []
            
            for level in sr_levels:
                level_price = level['price']
                level_type = level['type']
                breakout_threshold = params.get('breakout_threshold', 0.02)
                
                # Find breakout opportunities
                for i in range(len(close)):
                    if level_type == 'support':
                        # Look for breakdown
                        if low[i] < level_price * (1 - breakout_threshold):
                            total_trades += 1
                            # Calculate PnL (simplified)
                            pnl = (level_price - close[i]) / level_price
                            total_pnl += pnl
                            if pnl > 0:
                                successful_trades += 1
                            level_performance.append({
                                'level_id': len(level_performance),
                                'pnl': pnl,
                                'success': pnl > 0
                            })
                            break
                    else:  # resistance
                        # Look for breakout
                        if high[i] > level_price * (1 + breakout_threshold):
                            total_trades += 1
                            # Calculate PnL (simplified)
                            pnl = (close[i] - level_price) / level_price
                            total_pnl += pnl
                            if pnl > 0:
                                successful_trades += 1
                            level_performance.append({
                                'level_id': len(level_performance),
                                'pnl': pnl,
                                'success': pnl > 0
                            })
                            break
            
            success_rate = successful_trades / total_trades if total_trades > 0 else 0.0
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'success_rate': success_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'level_performance': level_performance
            }
            
        except Exception as e:
            self.logger.error(f"SR level backtesting failed: {e}")
            return {}

    def _calculate_composite_score(self, backtest_results: Dict[str, Any], 
                                 params: Dict[str, Any]) -> float:
        """Calculate composite score from backtest results."""
        try:
            if not backtest_results:
                return 0.0
            
            success_rate = backtest_results.get('success_rate', 0.0)
            avg_pnl = backtest_results.get('avg_pnl', 0.0)
            total_trades = backtest_results.get('total_trades', 0)
            
            # Normalize PnL (assume 0-0.1 range is good)
            pnl_score = min(1.0, max(0.0, avg_pnl / 0.1))
            
            # Trade frequency score (more trades = better, but not too many)
            trade_frequency_score = min(1.0, total_trades / 10.0)
            
            # Composite score with weights
            composite_score = (
                success_rate * 0.5 +           # 50% weight on success rate
                pnl_score * 0.3 +              # 30% weight on PnL
                trade_frequency_score * 0.2    # 20% weight on trade frequency
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Composite score calculation failed: {e}")
            return 0.0

    def _predict_with_params(self, params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Generate predictions using given parameters (simplified)."""
        # This is a simplified prediction method
        # In practice, this would use the actual SR detection algorithm
        return np.random.random(len(X))

    def _extract_feature_importance(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Extract feature importance from parameters."""
        # Simplified feature importance based on parameter values
        importance = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                importance[param_name] = float(param_value)
        return importance

    def _analyze_parameter_sensitivity(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Analyze parameter sensitivity."""
        # Simplified sensitivity analysis
        sensitivity = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                # Higher values indicate higher sensitivity
                sensitivity[param_name] = min(1.0, abs(param_value) / 10.0)
        return sensitivity

    def _combine_evaluation_scores(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Combine different evaluation scores with appropriate weights."""
        try:
            if not scores or not weights:
                return 0.0
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            normalized_weights = {method: weight / total_weight for method, weight in weights.items()}
            
            # Calculate weighted sum
            weighted_sum = sum(score * normalized_weights.get(method, 0) for method, score in scores.items())
            
            self.logger.debug(f"Score combination: {scores}, weights: {normalized_weights}, result: {weighted_sum:.4f}")
            return weighted_sum
            
        except Exception as e:
            self.logger.error(f"Score combination failed: {e}")
            return 0.0
    
    def _calculate_strength_with_weights(self, level: Any, weights: StrengthWeights) -> float:
        """Calculate SR level strength using custom weights.
        
        Args:
            level: SR level (dict or SRLevel object)
            weights: StrengthWeights with custom weight values
            
        Returns:
            Calculated strength score [0, 1]
        """
        try:
            # Extract level attributes (handle both dict and object)
            def get_attr(obj, attr, default=0.0):
                if isinstance(obj, dict):
                    return obj.get(attr, default)
                return getattr(obj, attr, default)
            
            base_strength = get_attr(level, 'strength', 0.3)
            touch_count = get_attr(level, 'touch_count', 0)
            avg_bounce_ratio = get_attr(level, 'avg_bounce_ratio', 0.0)
            volume_confirmation_score = get_attr(level, 'volume_confirmation_score', 0.0)
            consistency_score = get_attr(level, 'consistency_score', 0.0)
            confluence_score = get_attr(level, 'confluence_score', 0.0)
            failure_count = get_attr(level, 'failure_count', 0)
            pivot_level = get_attr(level, 'pivot_level', False)
            psychological_level = get_attr(level, 'psychological_level', False)
            volume_at_level = get_attr(level, 'volume_at_level', 0.0)
            
            # Touch boost: Only count touches with rejection
            rejection_ratio = min(avg_bounce_ratio / 0.02, 1.0)
            effective_touches = touch_count * rejection_ratio if avg_bounce_ratio > 0 else 0
            touch_boost = min(effective_touches * weights.touch_weight, 0.3)
            
            # Volume boost
            volume_boost = volume_confirmation_score * weights.volume_weight
            
            # Consistency boost
            consistency_boost = consistency_score * weights.consistency_weight
            
            # Confluence boost
            confluence_boost = confluence_score * weights.confluence_weight
            
            # Failure penalty: base penalty × volume multiplier × volume scaling, capped at max
            volume_factor = max(0.5, volume_confirmation_score)
            # Low volume = high multiplier (weak breakout), high volume = low multiplier (strong breakout)
            volume_scaling = weights.failure_volume_multiplier * (2.0 - volume_factor)
            failure_penalty = min(
                failure_count * weights.failure_penalty_base * volume_scaling,
                weights.failure_max_penalty
            )
            
            # Special boosts
            special_boost = 0.0
            if pivot_level:
                special_boost += weights.pivot_boost
            if psychological_level:
                special_boost += weights.psychological_boost
            if volume_at_level > 0:
                hvn_boost = min(volume_at_level * weights.hvn_boost, weights.hvn_boost)
                special_boost += hvn_boost
            
            # Final strength
            final_strength = (base_strength + touch_boost + volume_boost + 
                            consistency_boost + confluence_boost + 
                            special_boost - failure_penalty)
            
            return max(0.0, min(1.0, final_strength))
            
        except Exception as e:
            self.logger.warning(f"Strength calculation with weights failed: {e}")
            return get_attr(level, 'strength', 0.5)

    def _generate_enhancement_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of all enhancements made to the SR parameter optimization."""
        return {
            'enhancement_version': '2.0.0',
            'enhancement_date': '2025-01-03',
            'enhancement_summary': {
                'bayesian_hpo_enhancements': {
                    'staged_optimization': 'Coarse grid -> Fine grid -> Bayesian TPE',
                    'early_stopping': 'Enabled with median pruner',
                    'hardware_optimization': 'Integrated with UnifiedHardwareManager',
                    'explainability': 'SHAP and LIME integration',
                    'advanced_validation': 'OOF, Purged CV, and unified evaluation'
                },
                'vectorbt_optimization_enhancements': {
                    'unified_vectorization': 'Integrated with UnifiedVectorizationManager',
                    'hardware_aware': 'Automatic strategy selection based on hardware',
                    'rolling_optimization': 'Enhanced with VectorBTRollingOptimizer',
                    'performance_monitoring': 'Real-time performance tracking',
                    'fallback_mechanisms': 'Graceful degradation to traditional methods'
                },
                'ml_utilities_integration': {
                    'explainability': 'SHAP and LIME for model interpretability',
                    'oof_validation': 'Out-of-fold stacking ensemble validation',
                    'purged_cv': 'Purged cross-validation for time series',
                    'data_leakage_detection': 'Comprehensive leakage detection',
                    'unified_evaluation': 'Multi-metric evaluation framework'
                },
                'hardware_optimization_enhancements': {
                    'm1_optimization': 'Apple M1 specific optimizations',
                    'gpu_acceleration': 'Optional GPU acceleration support',
                    'memory_optimization': 'Intelligent memory management',
                    'adaptive_scheduling': 'Workload-aware task scheduling'
                },
                'parameter_space_enhancements': {
                    'comprehensive_parameters': '25+ SR detection parameters',
                    'advanced_categories': 'Time-based, volume-based, price action, risk management',
                    'intelligent_bounds': 'Data-driven parameter ranges',
                    'hierarchical_optimization': 'Parameter importance-based optimization'
                },
                'validation_enhancements': {
                    'temporal_validation': 'Proper time series splitting',
                    'data_leakage_prevention': 'Gap-based temporal separation',
                    'lookahead_bias_detection': 'Advanced bias detection',
                    'cross_validation': 'Multiple CV strategies'
                },
                'performance_enhancements': {
                    'caching': 'Intelligent result caching',
                    'parallel_processing': 'Multi-core optimization',
                    'memory_efficiency': 'Optimized memory usage',
                    'computation_optimization': 'Algorithm-specific optimizations'
                }
            },
            'compatibility': {
                'backward_compatible': True,
                'graceful_degradation': True,
                'optional_dependencies': True,
                'fallback_mechanisms': True
            },
            'usage_improvements': {
                'simplified_configuration': 'Enhanced configuration with sensible defaults',
                'comprehensive_logging': 'Detailed progress and performance logging',
                'rich_metadata': 'Comprehensive result metadata',
                'error_handling': 'Robust error handling and recovery'
            }
        }

    async def _apply_hardware_optimization(
        self, 
        optimization_result: Dict[str, Any], 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Apply hardware-specific optimizations."""
        try:
            # Get hardware configuration - handle case where method doesn't exist
            if hasattr(self.hardware_manager, 'get_optimal_config'):
                hardware_config = self.hardware_manager.get_optimal_config(
                    WorkloadType.ML_TRAINING,
                    OptimizationLevel.BALANCED
                )
                
                # Apply optimizations based on hardware capabilities
                gains = {
                    'cpu_optimization': hardware_config.get('cpu_gain', 1.0),
                    'memory_optimization': hardware_config.get('memory_gain', 1.0),
                    'gpu_acceleration': hardware_config.get('gpu_gain', 1.0) if enhanced_config.enable_gpu_acceleration else 1.0
                }
                
                # Update optimization result with hardware gains
                optimization_result['hardware_gains'] = gains
            else:
                # Use default gains if method doesn't exist
                optimization_result['hardware_gains'] = {
                    'cpu_optimization': 1.0,
                    'memory_optimization': 1.0,
                    'gpu_acceleration': 1.0
                }
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Hardware optimization failed: {e}")
            return optimization_result

    async def _validate_optimization_results(
        self, 
        optimization_result: Dict[str, Any], 
        train_data: Any, 
        test_data: Any
    ) -> Dict[str, Any]:
        """Validate optimization results for data leakage."""
        try:
            # Check for temporal leakage
            leakage_report = self.leakage_detector.detect_temporal_leakage(
                train_data, test_data
            )
            
            # Handle both dict and object response types
            if isinstance(leakage_report, dict):
                validation_result = {
                    'leakage_detected': leakage_report.get('has_leakage', False),
                    'leakage_score': leakage_report.get('leakage_score', 0.0),
                    'temporal_violations': leakage_report.get('temporal_violations', []),
                    'recommendations': leakage_report.get('recommendations', [])
                }
            else:
                validation_result = {
                    'leakage_detected': getattr(leakage_report, 'has_leakage', False),
                    'leakage_score': getattr(leakage_report, 'leakage_score', 0.0),
                    'temporal_violations': getattr(leakage_report, 'temporal_violations', []),
                    'recommendations': getattr(leakage_report, 'recommendations', [])
                }
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {'error': str(e)}

    async def _fetch_input_artifacts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch required input artifacts from previous steps.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict containing success status and fetched artifacts
        """
        try:
            self.logger.info("📥 Fetching required input artifacts from previous steps...")
            
            fetched_artifacts = {}
            required_artifacts = self.get_required_input_artifacts()
            
            for artifact_name in required_artifacts:
                try:
                    # Try to get artifact using BaseStep artifact management
                    artifact_data = self._get_artifact(artifact_name, artifact_type="data")
                    if artifact_data is not None:
                        fetched_artifacts[artifact_name] = artifact_data
                        self.logger.info(f"✅ Successfully fetched {artifact_name}")
                    else:
                        self.logger.warning(f"⚠️ {artifact_name} returned None")
                        fetched_artifacts[artifact_name] = None
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to fetch {artifact_name}: {e}")
                    fetched_artifacts[artifact_name] = None
            
            # Check if critical artifacts are missing
            missing_artifacts = [name for name, data in fetched_artifacts.items() if data is None]
            if missing_artifacts:
                self.logger.warning(f"Missing artifacts: {missing_artifacts}")
                # For now, we'll continue with None values and handle gracefully
                # In a production system, you might want to fail here
            
            return {
                'success': True,
                'artifacts': fetched_artifacts,
                'missing_artifacts': missing_artifacts
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fetch input artifacts: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': {}
            }
    
    async def _save_output_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save output artifacts using BaseStep artifact management.
        
        Args:
            artifacts: Dictionary of artifacts to save
            config: Configuration dictionary
            
        Returns:
            Dict containing success status and saved artifact paths
        """
        try:
            self.logger.info("💾 Saving output artifacts using BaseStep artifact management...")
            
            saved_paths = {}
            
            for artifact_name, artifact_data in artifacts.items():
                try:
                    # Save artifact using BaseStep method
                    artifact_path = self._save_artifact(
                        data=artifact_data,
                        artifact_name=artifact_name,
                        artifact_type="data",
                        compression="auto",
                        metadata={
                            'symbol': config.get('symbol', 'unknown'),
                            'exchange': config.get('exchange', 'unknown'),
                            'timeframe': config.get('timeframe', 'unknown'),
                            'step_name': self.step_name,
                            'execution_timestamp': datetime.now().isoformat(),
                            'enhancement_version': '2.0'
                        }
                    )
                    saved_paths[artifact_name] = artifact_path
                    self.logger.info(f"✅ Successfully saved {artifact_name} to {artifact_path}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to save {artifact_name}: {e}")
                    saved_paths[artifact_name] = None
            
            # Check if any artifacts failed to save
            failed_artifacts = [name for name, path in saved_paths.items() if path is None]
            
            return {
                'success': len(failed_artifacts) == 0,
                'artifact_paths': saved_paths,
                'failed_artifacts': failed_artifacts,
                'error': f"Failed to save artifacts: {failed_artifacts}" if failed_artifacts else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save output artifacts: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifact_paths': {}
            }

    async def _run_vectorbt_optimization(
        self, 
        market_data: Any, 
        search_space: Dict[str, Any], 
        enhanced_config: EnhancedSRConfig,
        input_artifacts: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run VectorBT rolling optimization for SR parameters.
        
        Args:
            market_data: Market data for optimization
            search_space: Parameter search space
            enhanced_config: Enhanced configuration
            input_artifacts: Artifacts from previous steps
            
        Returns:
            VectorBT optimization results
        """
        if not VECTORBT_ROLLING_AVAILABLE:
            return {'success': False, 'error': 'VectorBT RollingOptimizer not available'}
        
        tprint("⚡ VectorBT Rolling Optimization: Starting", "info")
        
        try:
            # Convert market data to VectorBT format
            if not PANDAS_AVAILABLE:
                return {'success': False, 'error': 'Pandas not available for VectorBT'}
            
            # Check if VectorBT is available
            if vbt is None:
                return {'success': False, 'error': 'VectorBT not available'}
            
            # Prepare data for VectorBT
            data_df = market_data if isinstance(market_data, pd.DataFrame) else pd.DataFrame(market_data)
            tprint_data_format(data_df, "Market data for VectorBT")
            
            # Create VectorBT data structure
            vbt_data = vbt.Data(
                data_df['close'].values,
                index=data_df.index if hasattr(data_df, 'index') else None,
                columns=['close']
            )
            
            # Add OHLCV data if available
            if all(col in data_df.columns for col in ['open', 'high', 'low', 'volume']):
                vbt_data = vbt.Data(
                    data_df[['open', 'high', 'low', 'close', 'volume']].values,
                    index=data_df.index if hasattr(data_df, 'index') else None,
                    columns=['open', 'high', 'low', 'close', 'volume']
                )
            
            # Define VectorBT objective function
            def vectorbt_objective(params):
                """VectorBT objective function for SR parameter optimization."""
                try:
                    # Extract parameters
                    min_touches = int(params.get('min_touches', 2))
                    strength_threshold = float(params.get('strength_threshold', 0.5))
                    distance_threshold = float(params.get('distance_threshold', 0.01))
                    lookback_periods = int(params.get('lookback_periods', 50))
                    
                    # Use VectorBT for efficient SR detection
                    if hasattr(vbt_data, 'close'):
                        close_prices = vbt_data.close.values
                    else:
                        close_prices = vbt_data.values.flatten()
                    
                    # Calculate rolling statistics using VectorBT
                    rolling_min_vals = rolling_min(close_prices, window=lookback_periods)
                    rolling_max_vals = rolling_max(close_prices, window=lookback_periods)
                    rolling_std_vals = rolling_std(close_prices, window=lookback_periods)
                    
                    # Detect SR levels using vectorized operations
                    sr_levels = []
                    
                    # Find local minima (support levels)
                    min_mask = close_prices == rolling_min_vals
                    support_levels = close_prices[min_mask]
                    
                    # Find local maxima (resistance levels)  
                    max_mask = close_prices == rolling_max_vals
                    resistance_levels = close_prices[max_mask]
                    
                    # Calculate level quality using vectorized operations
                    total_levels = len(support_levels) + len(resistance_levels)
                    
                    if total_levels == 0:
                        return 0.0
                    
                    # Calculate average volatility for normalization
                    avg_volatility = rolling_std_vals.mean() if rolling_std_vals is not None else 0.01
                    
                    # Score based on level count and volatility
                    level_score = min(total_levels / 10.0, 1.0)  # Normalize to 0-1
                    volatility_score = min(avg_volatility / 0.05, 1.0)  # Normalize volatility
                    
                    # Combined score
                    combined_score = (level_score * 0.6 + volatility_score * 0.4)
                    
                    return min(max(combined_score, 0.0), 1.0)
                    
                except Exception as e:
                    self.logger.error(f"VectorBT objective function error: {e}")
                    return 0.0
            
            # Create VectorBT RollingOptimizer
            rolling_optimizer = RollingOptimizer(
                data=vbt_data,
                objective_func=vectorbt_objective,
                param_ranges=search_space,
                n_trials=enhanced_config.n_trials // 2,  # Use half trials for VectorBT
                rolling_window=enhanced_config.rolling_window,
                step_size=enhanced_config.step_size
            )
            
            # Run optimization
            tprint("⚡ VectorBT: Running rolling optimization", "info")
            optimization_result = rolling_optimizer.optimize()
            
            # Process results
            best_params = optimization_result.get('best_params', {})
            best_score = optimization_result.get('best_score', 0.0)
            
            tprint(f"⚡ VectorBT: Best score = {best_score:.4f}", "info")
            tprint_data_preview(best_params, "VectorBT best parameters")
            
            return {
                'success': True,
                'best_params': best_params,
                'best_score': best_score,
                'optimization_result': optimization_result,
                'method': 'vectorbt_rolling'
            }
            
        except Exception as e:
            self.logger.error(f"VectorBT optimization failed: {e}")
            tprint(f"❌ VectorBT optimization failed: {e}", "error")
            return {
                'success': False,
                'error': str(e),
                'method': 'vectorbt_rolling'
            }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)
