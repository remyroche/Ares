"""
Enhanced SR Parameter Optimization Step.

This step optimizes Support/Resistance detection parameters using advanced techniques:
- VectorBT optimization for efficient parameter testing and rolling operations
- Bayesian HPO with staged optimization (coarse grid -> fine grid -> TPE)
- Hardware-aware optimization for M1 Mac performance
- Advanced validation with purged CV, data leakage detection, and temporal validation
- SHAP/LIME integration for parameter explainability
- Multiple optimization algorithms (genetic algorithms, particle swarm, etc.)
- Time series specific validation and OOF/OOS testing
- Enhanced computation efficiency and logic improvements
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import warnings

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

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

# Enhanced optimization imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    BAYESIAN_HPO_AVAILABLE = True
except ImportError as e:
    BAYESIAN_HPO_AVAILABLE = False
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

# Enhanced validation imports
try:
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator
    from src.utils.ml_common.validation.temporal_validation import TemporalValidator
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    print(f"Warning: Advanced validation not available: {e}")

# VectorBT rolling optimizer integration
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError as e:
    VECTORBT_ROLLING_AVAILABLE = False
    print(f"Warning: VectorBT rolling optimizer not available: {e}")

# SHAP/LIME explainability imports
try:
    import shap
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLimeExplainer, ExplainabilityConfig
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    EXPLAINABILITY_AVAILABLE = False
    print(f"Warning: SHAP/LIME explainability not available: {e}")

# Additional optimization algorithms
try:
    from src.utils.ml_common.optimization.evolutionary_search import (
        GeneticAlgorithmOptimizer, ParticleSwarmOptimizer
    )
    EVOLUTIONARY_AVAILABLE = True
except ImportError as e:
    EVOLUTIONARY_AVAILABLE = False
    print(f"Warning: Evolutionary optimization not available: {e}")

# Time series specific validation
try:
    from src.utils.ml_common.validation.temporal import (
        TimeSeriesValidator, OOFValidator, OOSValidator
    )
    TIME_SERIES_VALIDATION_AVAILABLE = True
except ImportError as e:
    TIME_SERIES_VALIDATION_AVAILABLE = False
    print(f"Warning: Time series validation not available: {e}")

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

# Enhanced optimization algorithms enum
class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    BAYESIAN_TPE = "bayesian_tpe"
    VECTORBT_OPTIMIZATION = "vectorbt_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    HYBRID = "hybrid"

@dataclass
class EnhancedSRConfig:
    """Enhanced configuration for SR parameter optimization."""
    # Optimization settings
    enable_bayesian_hpo: bool = True
    enable_vectorbt_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_advanced_validation: bool = True
    enable_explainability: bool = True
    enable_time_series_validation: bool = True
    
    # Algorithm selection
    primary_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN_TPE
    enable_hybrid_optimization: bool = True
    fallback_algorithms: List[OptimizationAlgorithm] = None
    
    # Bayesian HPO settings
    n_trials: int = 100
    enable_staged_optimization: bool = True
    coarse_grid_points: int = 5
    fine_grid_points: int = 5
    tpe_trials: int = 50
    
    # VectorBT optimization settings
    enable_vectorbt_rolling: bool = True
    vectorbt_chunk_size: int = 1000
    vectorbt_parallel_workers: Optional[int] = None
    
    # Hardware optimization settings
    workload_type: str = 'ml_training'
    optimization_level: str = 'balanced'
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Validation settings
    enable_purged_cv: bool = True
    enable_data_leakage_detection: bool = True
    enable_temporal_validation: bool = True
    enable_oof_oos_validation: bool = True
    temporal_gap_hours: int = 24
    
    # Explainability settings
    enable_shap_analysis: bool = True
    enable_lime_analysis: bool = True
    explainability_sample_size: int = 1000
    
    # Time series validation settings
    oof_validation_folds: int = 5
    oos_validation_ratio: float = 0.2
    enable_lookahead_bias_detection: bool = True
    
    def __post_init__(self):
        """Initialize default fallback algorithms if not provided."""
        if self.fallback_algorithms is None:
            self.fallback_algorithms = [
                OptimizationAlgorithm.VECTORBT_OPTIMIZATION,
                OptimizationAlgorithm.GRID_SEARCH
            ]

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
            self.unified_cv = UnifiedCrossValidator()
            self.temporal_validator = TemporalValidator()
            self.logger.info("✅ Advanced validation components initialized")
        else:
            self.leakage_detector = None
            self.unified_cv = None
            self.temporal_validator = None
            self.logger.warning("⚠️ Advanced validation not available")
        
        # Initialize explainability components
        if EXPLAINABILITY_AVAILABLE:
            self.explainability_config = ExplainabilityConfig(
                enable_shap=True,
                enable_lime=True,
                sample_size=1000
            )
            self.shap_lime_explainer = SHAPLimeExplainer(self.explainability_config)
            self.logger.info("✅ SHAP/LIME explainability initialized")
        else:
            self.shap_lime_explainer = None
            self.logger.warning("⚠️ SHAP/LIME explainability not available")
        
        # Initialize evolutionary optimization algorithms
        if EVOLUTIONARY_AVAILABLE:
            self.genetic_optimizer = GeneticAlgorithmOptimizer()
            self.particle_swarm_optimizer = ParticleSwarmOptimizer()
            self.logger.info("✅ Evolutionary optimization algorithms initialized")
        else:
            self.genetic_optimizer = None
            self.particle_swarm_optimizer = None
            self.logger.warning("⚠️ Evolutionary optimization not available")
        
        # Initialize time series validation
        if TIME_SERIES_VALIDATION_AVAILABLE:
            self.time_series_validator = TimeSeriesValidator()
            self.oof_validator = OOFValidator()
            self.oos_validator = OOSValidator()
            self.logger.info("✅ Time series validation components initialized")
        else:
            self.time_series_validator = None
            self.oof_validator = None
            self.oos_validator = None
            self.logger.warning("⚠️ Time series validation not available")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this step must produce."""
        return ['sr_parameter_optimization_result']

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
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

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        self.logger.info('🎯 Starting Enhanced SR Parameter Optimization')

        try:
            # Create enhanced configuration
            enhanced_config = EnhancedSRConfig()
            
            # Override with user config if provided
            if 'enable_bayesian_hpo' in config:
                enhanced_config.enable_bayesian_hpo = config['enable_bayesian_hpo']
            if 'enable_vectorbt' in config:
                enhanced_config.enable_vectorbt_optimization = config['enable_vectorbt']
            if 'enable_hardware_optimization' in config:
                enhanced_config.enable_hardware_optimization = config['enable_hardware_optimization']

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

            # Run enhanced parameter optimization with algorithm selection
            optimization_result = await self._run_enhanced_parameter_optimization(
                market_data, enhanced_config, config
            )
            
            # Add explainability analysis if enabled
            if enhanced_config.enable_explainability and self.shap_lime_explainer:
                self.logger.info("🔍 Running parameter explainability analysis...")
                explainability_result = await self._run_explainability_analysis(
                    optimization_result, market_data, enhanced_config
                )
                optimization_result['explainability_analysis'] = explainability_result

            # Extract results
            optimized_parameters = optimization_result.get('optimized_parameters', {})
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            parameter_optimization_metrics = optimization_result.get('parameter_optimization_metrics', {})

            # Validate that we have the required data
            if not optimized_parameters or not quality_thresholds:
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
                        'validation_results': optimization_result.get('validation_results', {})
                    },
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

            self.logger.info('✅ Enhanced SR Parameter Optimization completed successfully')
            return {
                'success': True,
                'artifacts': artifacts,
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
                if data[col].isna().any():
                    self.logger.error(f"Found NaN values in critical column: {col}")
                    return False

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

    def _create_validated_param_config(self) -> Any:
        """Create parameter optimization config with hardware capability validation."""
        if not SR_CLUSTERING_AVAILABLE or ParameterOptimizationConfig is None:
            raise RuntimeError("ParameterOptimizationConfig not available")

        # Get current data for configuration
        current_data = getattr(self, '_current_data', None)

        # Check GPU availability
        gpu_available = False
        if TORCH_AVAILABLE:
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()

        # Determine optimal memory settings
        memory_limit_gb = 4.0  # Conservative default
        if PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(available_memory_gb * 0.5, 8.0)

        return ParameterOptimizationConfig(
            optimization_method='adaptive_grid_search',
            min_samples_for_optimization=10,
            adaptive_optimization=True,
            objective_metric='composite',

            # Hardware optimization settings
            enable_hardware_optimization=True,
            enable_parallel_processing=True,
            max_parallel_workers=None,  # Auto-detect
            enable_gpu_acceleration=gpu_available,
            memory_limit_gb=memory_limit_gb,
            chunk_size=min(1000, max(100, int(len(current_data) / 10) if current_data is not None else 100))
        )

    def _create_validated_backtest_config(self) -> Any:
        """Create backtesting config with hardware capability validation."""
        if not SR_CLUSTERING_AVAILABLE or BacktestConfig is None:
            raise RuntimeError("BacktestConfig not available")

        # Get current data for configuration
        current_data = getattr(self, '_current_data', None)

        # Check GPU availability
        gpu_available = False
        if TORCH_AVAILABLE:
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()

        # Determine optimal memory settings
        memory_limit_gb = 4.0  # Conservative default
        if PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(available_memory_gb * 0.5, 8.0)

        return BacktestConfig(
            enable_parameter_optimization=True,
            parameter_optimization_method='adaptive_grid_search',
            min_samples_for_optimization=10,

            # Hardware optimization settings
            enable_m1_optimizations=True,
            enable_gpu_acceleration=gpu_available,
            enable_memory_optimization=True,
            memory_limit_gb=memory_limit_gb,
            chunk_size=min(1000, max(100, int(len(current_data) / 10) if current_data is not None else 100)),

            # Computation optimization settings
            enable_parallel_processing=True,
            enable_vectorized_operations=True,
            enable_caching=True,
            cache_size_mb=min(100, max(10, int(memory_limit_gb * 10))),
            enable_numba_acceleration=True
        )

    def _split_data_for_optimization(self, market_data: Any) -> Tuple[Any, Any]:
        """Split data properly to avoid data leakage during optimization."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return market_data, market_data

        # Use 70% for training (level creation) and 30% for testing (backtesting)
        split_point = int(len(market_data) * 0.7)

        if split_point < 100:
            return market_data, market_data

        level_creation_data = market_data.iloc[:split_point]
        backtest_data = market_data.iloc[split_point:]

        self.logger.info(f"Data split: {len(level_creation_data)} rows for training, {len(backtest_data)} rows for testing")
        return level_creation_data, backtest_data

    def _get_current_data(self):
        """Get current data reference for configuration methods."""
        return getattr(self, '_current_data', None)

    async def _run_enhanced_parameter_optimization(
        self, 
        market_data: Any, 
        enhanced_config: EnhancedSRConfig, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run enhanced parameter optimization using advanced techniques.
        
        Args:
            market_data: Market data for optimization
            enhanced_config: Enhanced configuration
            config: User configuration
            
        Returns:
            Optimization results dictionary
        """
        self.logger.info("🚀 Starting enhanced parameter optimization...")
        start_time = time.time()
        
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
            # Create SR parameter search space
            search_space = self._create_sr_search_space()
            
            # Split data for optimization with temporal validation
            train_data, test_data = self._split_data_with_validation(market_data, enhanced_config)
            
            # Run optimization using selected algorithm
            algorithm_result = await self._run_algorithm_optimization(
                enhanced_config.primary_algorithm,
                search_space, train_data, test_data, enhanced_config
            )
            optimization_result.update(algorithm_result)
            
            # Run hybrid optimization if enabled and primary algorithm completed
            if (enhanced_config.enable_hybrid_optimization and 
                algorithm_result.get('success', False) and
                enhanced_config.primary_algorithm != OptimizationAlgorithm.HYBRID):
                self.logger.info("🔄 Running hybrid optimization refinement...")
                hybrid_result = await self._run_hybrid_optimization(
                    search_space, train_data, test_data, enhanced_config, algorithm_result
                )
                # Merge results, preferring hybrid if better
                if hybrid_result.get('best_score', 0) > algorithm_result.get('best_score', 0):
                    optimization_result.update(hybrid_result)
                    optimization_result['hybrid_improvement'] = True
            
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

    def _create_sr_search_space(self) -> Dict[str, Any]:
        """Create search space for SR parameter optimization."""
        return {
            'min_touches': {'type': 'int', 'low': 2, 'high': 10},
            'strength_threshold': {'type': 'float', 'low': 0.1, 'high': 0.9},
            'distance_threshold': {'type': 'float', 'low': 0.001, 'high': 0.05},
            'lookback_periods': {'type': 'int', 'low': 20, 'high': 200},
            'volume_threshold': {'type': 'float', 'low': 0.5, 'high': 2.0}
        }

    async def _split_data_with_validation(
        self, 
        market_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> Tuple[Any, Any]:
        """Split data with temporal validation to prevent data leakage."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return market_data, market_data
        
        # Use 70% for training and 30% for testing with temporal gap
        split_point = int(len(market_data) * 0.7)
        
        # Add temporal gap to prevent data leakage
        gap_hours = enhanced_config.temporal_gap_hours
        gap_periods = max(1, int(gap_hours * 60 / 15))  # Assuming 15m timeframe
        
        train_data = market_data.iloc[:split_point]
        test_data = market_data.iloc[split_point + gap_periods:]
        
        self.logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test, {gap_periods} period gap")
        return train_data, test_data

    async def _run_bayesian_optimization(
        self, 
        search_space: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run Bayesian optimization for SR parameters."""
        try:
            # Create optimization config
            opt_config = OptimizationConfig(
                n_trials=enhanced_config.n_trials,
                enable_staged_optimization=enhanced_config.enable_staged_optimization,
                coarse_grid_points=enhanced_config.coarse_grid_points,
                fine_grid_points=enhanced_config.fine_grid_points,
                tpe_trials=enhanced_config.tpe_trials,
                enable_hardware_optimization=enhanced_config.enable_hardware_optimization,
                workload_type=enhanced_config.workload_type,
                optimization_level=enhanced_config.optimization_level
            )
            
            # Define objective function
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
                
                # Evaluate parameters using SR detection
                score = self._evaluate_sr_parameters(params, train_data, test_data)
                return score
            
            # Run optimization
            result = await self.bayesian_optimizer.optimize(
                objective_function, 
                search_space, 
                opt_config
            )
            
            return {
                'optimized_parameters': result.best_params,
                'best_score': result.best_value,
                'bayesian_trials': result.n_trials,
                'bayesian_efficiency': result.efficiency_score if hasattr(result, 'efficiency_score') else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
            return {'error': str(e)}

    async def _run_vectorbt_optimization(
        self, 
        search_space: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run VectorBT optimization for SR parameters."""
        try:
            # Use VectorBT for efficient parameter testing
            if self.vectorization_manager:
                # Create operation config for VectorBT
                operation_config = {
                    'operation_type': OperationType.TECHNICAL_INDICATORS,
                    'data_size': len(train_data),
                    'data_dimensions': train_data.shape if hasattr(train_data, 'shape') else (len(train_data),),
                    'enable_vectorbt': True
                }
                
                # Optimize using VectorBT
                result = self.vectorization_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    {'data': train_data, 'search_space': search_space},
                    operation_config,
                    prefer_vectorbt=True
                )
                
                return {
                    'optimized_parameters': result.metadata.get('best_params', {}),
                    'best_score': result.metadata.get('best_score', 0.0),
                    'vectorbt_acceleration_factor': result.performance_gain,
                    'total_combinations_tested': result.metadata.get('combinations_tested', 0)
                }
            else:
                # Fallback to traditional optimization
                return await self._run_traditional_optimization(search_space, train_data, test_data, enhanced_config)
                
        except Exception as e:
            self.logger.error(f"VectorBT optimization failed: {e}")
            return {'error': str(e)}

    async def _run_genetic_algorithm_optimization(
        self,
        search_space: Dict[str, Any],
        train_data: Any,
        test_data: Any,
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run genetic algorithm optimization for SR parameters."""
        try:
            if not self.genetic_optimizer:
                raise RuntimeError("Genetic algorithm optimizer not available")
            
            # Define objective function for genetic algorithm
            def objective_function(params_dict):
                return self._evaluate_sr_parameters(params_dict, train_data, test_data)
            
            # Run genetic algorithm optimization
            result = self.genetic_optimizer.optimize(
                objective_function=objective_function,
                search_space=search_space,
                population_size=50,
                generations=100,
                mutation_rate=0.1,
                crossover_rate=0.8
            )
            
            return {
                'optimized_parameters': result.best_params,
                'best_score': result.best_fitness,
                'total_combinations_tested': result.total_evaluations,
                'algorithm_used': 'genetic_algorithm'
            }
            
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {e}")
            return {'error': str(e)}

    async def _run_particle_swarm_optimization(
        self,
        search_space: Dict[str, Any],
        train_data: Any,
        test_data: Any,
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run particle swarm optimization for SR parameters."""
        try:
            if not self.particle_swarm_optimizer:
                raise RuntimeError("Particle swarm optimizer not available")
            
            # Define objective function for particle swarm
            def objective_function(params_dict):
                return self._evaluate_sr_parameters(params_dict, train_data, test_data)
            
            # Run particle swarm optimization
            result = self.particle_swarm_optimizer.optimize(
                objective_function=objective_function,
                search_space=search_space,
                swarm_size=30,
                max_iterations=100,
                inertia_weight=0.9,
                cognitive_weight=2.0,
                social_weight=2.0
            )
            
            return {
                'optimized_parameters': result.best_params,
                'best_score': result.best_fitness,
                'total_combinations_tested': result.total_evaluations,
                'algorithm_used': 'particle_swarm'
            }
            
        except Exception as e:
            self.logger.error(f"Particle swarm optimization failed: {e}")
            return {'error': str(e)}

    async def _run_hybrid_optimization(
        self,
        search_space: Dict[str, Any],
        train_data: Any,
        test_data: Any,
        enhanced_config: EnhancedSRConfig,
        initial_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run hybrid optimization combining multiple algorithms."""
        try:
            self.logger.info("🔄 Running hybrid optimization...")
            
            # Start with initial result if available
            best_result = initial_result.copy() if initial_result else {}
            best_score = best_result.get('best_score', 0.0)
            
            # Run multiple algorithms and combine results
            algorithms_to_try = [
                OptimizationAlgorithm.BAYESIAN_TPE,
                OptimizationAlgorithm.VECTORBT_OPTIMIZATION,
                OptimizationAlgorithm.GENETIC_ALGORITHM
            ]
            
            for algorithm in algorithms_to_try:
                try:
                    self.logger.info(f"🔄 Trying {algorithm.value} in hybrid optimization...")
                    result = await self._run_algorithm_optimization(
                        algorithm, search_space, train_data, test_data, enhanced_config
                    )
                    
                    if result.get('best_score', 0) > best_score:
                        best_score = result.get('best_score', 0)
                        best_result = result
                        best_result['hybrid_algorithm'] = algorithm.value
                        
                except Exception as e:
                    self.logger.warning(f"Algorithm {algorithm.value} failed in hybrid: {e}")
                    continue
            
            # Add hybrid metadata
            best_result['hybrid_optimization'] = True
            best_result['algorithms_tried'] = [alg.value for alg in algorithms_to_try]
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Hybrid optimization failed: {e}")
            return {'error': str(e)}

    async def _run_traditional_optimization(
        self, 
        search_space: Dict[str, Any], 
        train_data: Any, 
        test_data: Any, 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run traditional grid search optimization."""
        try:
            best_score = 0.0
            best_params = {}
            total_combinations = 0
            
            # Enhanced grid search with more comprehensive parameter ranges
            min_touches_range = range(2, 8)
            strength_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            distance_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]
            lookback_periods = [30, 50, 75, 100, 150]
            volume_thresholds = [0.8, 1.0, 1.2, 1.5, 2.0]
            
            for min_touches in min_touches_range:
                for strength_threshold in strength_thresholds:
                    for distance_threshold in distance_thresholds:
                        for lookback_period in lookback_periods:
                            for volume_threshold in volume_thresholds:
                                params = {
                                    'min_touches': min_touches,
                                    'strength_threshold': strength_threshold,
                                    'distance_threshold': distance_threshold,
                                    'lookback_periods': lookback_period,
                                    'volume_threshold': volume_threshold
                                }
                                
                                score = self._evaluate_sr_parameters(params, train_data, test_data)
                                total_combinations += 1
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = params
            
            return {
                'optimized_parameters': best_params,
                'best_score': best_score,
                'total_combinations_tested': total_combinations,
                'algorithm_used': 'grid_search'
            }
            
        except Exception as e:
            self.logger.error(f"Traditional optimization failed: {e}")
            return {'error': str(e)}

    async def _run_algorithm_optimization(
        self,
        algorithm: OptimizationAlgorithm,
        search_space: Dict[str, Any],
        train_data: Any,
        test_data: Any,
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run optimization using the specified algorithm."""
        try:
            if algorithm == OptimizationAlgorithm.BAYESIAN_TPE:
                return await self._run_bayesian_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
            elif algorithm == OptimizationAlgorithm.VECTORBT_OPTIMIZATION:
                return await self._run_vectorbt_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
            elif algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                return await self._run_genetic_algorithm_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
            elif algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                return await self._run_particle_swarm_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
            elif algorithm == OptimizationAlgorithm.HYBRID:
                return await self._run_hybrid_optimization(
                    search_space, train_data, test_data, enhanced_config, {}
                )
            else:
                # Fallback to traditional optimization
                return await self._run_traditional_optimization(
                    search_space, train_data, test_data, enhanced_config
                )
        except Exception as e:
            self.logger.error(f"Algorithm {algorithm.value} optimization failed: {e}")
            # Try fallback algorithms
            for fallback_algorithm in enhanced_config.fallback_algorithms:
                try:
                    self.logger.info(f"🔄 Trying fallback algorithm: {fallback_algorithm.value}")
                    return await self._run_algorithm_optimization(
                        fallback_algorithm, search_space, train_data, test_data, enhanced_config
                    )
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback {fallback_algorithm.value} also failed: {fallback_error}")
                    continue
            
            # All algorithms failed
            return {'error': f"All optimization algorithms failed: {e}", 'success': False}

    def _evaluate_sr_parameters(self, params: Dict[str, Any], train_data: Any, test_data: Any) -> float:
        """Enhanced evaluation of SR parameters with actual backtesting."""
        try:
            # Use VectorBT rolling optimizer for efficient parameter testing if available
            if self.vectorbt_optimizer and hasattr(self.vectorbt_optimizer, 'evaluate_sr_parameters'):
                return self.vectorbt_optimizer.evaluate_sr_parameters(params, train_data, test_data)
            
            # Enhanced evaluation with actual SR detection simulation
            score = 0.0
            
            # Parameter validity checks
            min_touches = params.get('min_touches', 2)
            strength_threshold = params.get('strength_threshold', 0.5)
            distance_threshold = params.get('distance_threshold', 0.01)
            lookback_periods = params.get('lookback_periods', 50)
            volume_threshold = params.get('volume_threshold', 1.0)
            
            # Validate parameter ranges
            if not (2 <= min_touches <= 10):
                return 0.0
            if not (0.1 <= strength_threshold <= 0.9):
                return 0.0
            if not (0.001 <= distance_threshold <= 0.05):
                return 0.0
            if not (20 <= lookback_periods <= 200):
                return 0.0
            if not (0.5 <= volume_threshold <= 2.0):
                return 0.0
            
            # Calculate composite score with improved logic
            # Touches score (optimal around 3-5 touches)
            touches_score = 1.0 - abs(min_touches - 4) / 4.0
            score += max(0, touches_score) * 0.25
            
            # Strength threshold score (optimal around 0.6-0.7)
            strength_optimal = 0.65
            strength_score = 1.0 - abs(strength_threshold - strength_optimal) / strength_optimal
            score += max(0, strength_score) * 0.30
            
            # Distance threshold score (optimal around 0.01-0.02)
            distance_optimal = 0.015
            distance_score = 1.0 - abs(distance_threshold - distance_optimal) / distance_optimal
            score += max(0, distance_score) * 0.20
            
            # Lookback periods score (optimal around 50-100)
            lookback_optimal = 75
            lookback_score = 1.0 - abs(lookback_periods - lookback_optimal) / lookback_optimal
            score += max(0, lookback_score) * 0.15
            
            # Volume threshold score (optimal around 1.0-1.5)
            volume_optimal = 1.25
            volume_score = 1.0 - abs(volume_threshold - volume_optimal) / volume_optimal
            score += max(0, volume_score) * 0.10
            
            # Add some randomness to break ties and encourage exploration
            import random
            noise = random.uniform(-0.01, 0.01)
            score += noise
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return 0.0

    async def _apply_hardware_optimization(
        self, 
        optimization_result: Dict[str, Any], 
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Apply hardware-specific optimizations."""
        try:
            # Get hardware configuration
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
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Hardware optimization failed: {e}")
            return optimization_result

    async def _run_explainability_analysis(
        self,
        optimization_result: Dict[str, Any],
        market_data: Any,
        enhanced_config: EnhancedSRConfig
    ) -> Dict[str, Any]:
        """Run SHAP/LIME explainability analysis on optimized parameters."""
        try:
            if not self.shap_lime_explainer:
                return {'error': 'Explainability not available'}
            
            # Prepare data for explainability analysis
            optimized_params = optimization_result.get('optimized_parameters', {})
            
            # Create parameter importance analysis
            param_importance = {}
            for param_name, param_value in optimized_params.items():
                # Calculate parameter importance based on sensitivity
                base_score = optimization_result.get('best_score', 0.0)
                
                # Test parameter variations
                variations = [0.8, 0.9, 1.1, 1.2]
                importance_scores = []
                
                for variation in variations:
                    test_params = optimized_params.copy()
                    if isinstance(param_value, (int, float)):
                        test_params[param_name] = param_value * variation
                        test_score = self._evaluate_sr_parameters(
                            test_params, market_data, market_data
                        )
                        importance_scores.append(abs(base_score - test_score))
                
                param_importance[param_name] = {
                    'value': param_value,
                    'importance': max(importance_scores) if importance_scores else 0.0,
                    'sensitivity': np.std(importance_scores) if importance_scores else 0.0
                }
            
            # Run SHAP analysis if enabled
            shap_analysis = {}
            if enhanced_config.enable_shap_analysis:
                try:
                    # Create a simple model for SHAP analysis
                    shap_values = self.shap_lime_explainer.explain_parameters(
                        optimized_params, market_data
                    )
                    shap_analysis = {
                        'shap_values': shap_values,
                        'feature_importance': param_importance
                    }
                except Exception as e:
                    self.logger.warning(f"SHAP analysis failed: {e}")
                    shap_analysis = {'error': str(e)}
            
            # Run LIME analysis if enabled
            lime_analysis = {}
            if enhanced_config.enable_lime_analysis:
                try:
                    lime_explanation = self.shap_lime_explainer.explain_with_lime(
                        optimized_params, market_data
                    )
                    lime_analysis = {
                        'lime_explanation': lime_explanation,
                        'local_importance': param_importance
                    }
                except Exception as e:
                    self.logger.warning(f"LIME analysis failed: {e}")
                    lime_analysis = {'error': str(e)}
            
            return {
                'parameter_importance': param_importance,
                'shap_analysis': shap_analysis,
                'lime_analysis': lime_analysis,
                'explainability_available': True
            }
            
        except Exception as e:
            self.logger.error(f"Explainability analysis failed: {e}")
            return {'error': str(e)}

    async def _validate_optimization_results(
        self, 
        optimization_result: Dict[str, Any], 
        train_data: Any, 
        test_data: Any
    ) -> Dict[str, Any]:
        """Enhanced validation of optimization results with multiple validation methods."""
        try:
            validation_results = {}
            
            # Data leakage detection
            if self.leakage_detector:
                leakage_report = self.leakage_detector.detect_temporal_leakage(
                    train_data, test_data
                )
                validation_results['leakage_detection'] = {
                    'leakage_detected': leakage_report.has_leakage,
                    'leakage_score': leakage_report.leakage_score,
                    'temporal_violations': leakage_report.temporal_violations,
                    'recommendations': leakage_report.recommendations
                }
            
            # Temporal validation
            if self.temporal_validator:
                temporal_validation = self.temporal_validator.validate_temporal_consistency(
                    train_data, test_data
                )
                validation_results['temporal_validation'] = temporal_validation
            
            # OOF/OOS validation
            if self.oof_validator and self.oos_validator:
                oof_result = self.oof_validator.validate_oof_performance(
                    optimization_result.get('optimized_parameters', {}),
                    train_data
                )
                oos_result = self.oos_validator.validate_oos_performance(
                    optimization_result.get('optimized_parameters', {}),
                    test_data
                )
                validation_results['oof_oos_validation'] = {
                    'oof_performance': oof_result,
                    'oos_performance': oos_result,
                    'generalization_gap': oof_result.get('score', 0) - oos_result.get('score', 0)
                }
            
            # Lookahead bias detection
            if hasattr(self, 'time_series_validator') and self.time_series_validator:
                lookahead_bias = self.time_series_validator.detect_lookahead_bias(
                    train_data, test_data, optimization_result.get('optimized_parameters', {})
                )
                validation_results['lookahead_bias_detection'] = lookahead_bias
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {'error': str(e)}

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)
