"""
Regime Data Splitting Component.

This component tags data by regimes discovered in previous stages.
Enhanced with comprehensive error handling, validation, and reporting.
Refactored to use common utilities for better maintainability and performance.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Handle optional dependencies with explicit error reporting
IMPORT_ERRORS = []

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    NUMPY_AVAILABLE = False
    np = None
    IMPORT_ERRORS.append(f"numpy: {e}")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    PANDAS_AVAILABLE = False
    pd = None
    IMPORT_ERRORS.append(f"pandas: {e}")

from ..components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Import common utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_to_parquet, safe_read_parquet, validate_dataframe_schema,
    optimize_dataframe_dtypes, safe_fillna, safe_float, safe_int,
    validate_finite, validate_positive, validate_range, safe_divide,
    safe_log, safe_sqrt, safe_power, safe_mean, safe_std, safe_percentage_change,
    safe_kelly_calculation, safe_weighted_average, safe_correlation,
    safe_covariance, safe_percentile, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, timed_operation, format_bytes,
    chunked_iterable, parallel_map, get_m1_gpu_manager, get_m1_memory_optimizer,
    get_m1_cpu_optimizer, cleanup_m1_optimizers, integrate_with_m1_optimizers,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage
)

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, safe_kelly_calculation as math_safe_kelly,
    safe_weighted_average as math_safe_weighted_avg, safe_percentage_change as math_safe_pct_change,
    safe_correlation as math_safe_corr, safe_covariance as math_safe_cov,
    safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile as math_safe_percentile, validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inv, math_safe as math_safe_func,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.data.klines_parquet import (
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from src.utils.matrix_operations.unified_operations import (
    safe_matrix_operation, validate_matrix, optimize_matrix_operations
)

from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager as get_gpu_manager, is_m1_available, is_mps_available,
    optimize_dataframe_for_m1, create_m1_optimized_array, m1_backtesting_simulate,
    m1_monte_carlo_simulate
)

from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer as get_memory_optimizer, optimize_dataframe_memory,
    start_m1_memory_monitoring, stop_m1_memory_monitoring, optimize_memory as mem_optimize
)

from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer as get_cpu_optimizer, optimize_function_for_m1,
    parallel_map_m1, create_m1_optimized_thread_pool, run_cpu_intensive_task
)

from src.utils.ml_common.common_operations import (
    safe_ml_operation, validate_ml_data, optimize_ml_performance
)


class RegimeSplittingStatus(Enum):
    """Status enumeration for regime splitting operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class RegimeSplittingMetrics:
    """Comprehensive metrics for regime splitting operations."""
    total_data_points: int = 0
    regime_count: int = 0
    regime_distribution: Dict[int, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    validation_checks_passed: int = 0
    validation_checks_failed: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    data_quality_score: float = 0.0
    regime_continuity_score: float = 0.0


@dataclass
class RegimeSplittingReport:
    """Comprehensive report for regime splitting operations."""
    status: RegimeSplittingStatus
    metrics: RegimeSplittingMetrics
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RegimeDataSplittingComponent(BaseMarketAnalysisComponent):
    """
    Regime Data Splitting Component.
    
    Tags data by regimes discovered in previous stages.
    Enhanced with comprehensive error handling, validation, and reporting.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the regime data splitting component."""
        super().__init__(config)
        self.logger = system_logger.getChild('RegimeDataSplitting')
        
        # Validate dependencies and fail fast if missing
        self._validate_dependencies()
        
        # Initialize metrics tracking
        self.metrics = RegimeSplittingMetrics()
        self.start_time: Optional[datetime] = None
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        # Initialize common utilities
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        self.gpu_manager = get_gpu_manager()
        self.memory_optimizer = get_memory_optimizer()
        self.cpu_optimizer = get_cpu_optimizer()
        
    def _validate_dependencies(self) -> None:
        """Validate required dependencies and fail fast if missing."""
        try:
            missing_deps = []
            
            if not NUMPY_AVAILABLE:
                missing_deps.append("numpy")
            if not PANDAS_AVAILABLE:
                missing_deps.append("pandas")
                
            if missing_deps:
                error_msg = f"Critical dependencies missing: {', '.join(missing_deps)}"
                self.logger.error(f"❌ {error_msg}")
                raise ImportError(error_msg)
                
            self.logger.info("✅ All required dependencies available")
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in dependency validation: {e}")
            raise
    
    def _initialize_hardware_optimizations(self) -> None:
        """Initialize hardware optimizations for M1."""
        try:
            # Check if M1 hardware is available
            if is_m1_available():
                self.logger.info("🧠 M1 hardware detected, initializing optimizations")
                
                # Start memory monitoring
                start_m1_memory_monitoring()
                
                # Optimize numpy for M1
                self.cpu_optimizer.optimize_numpy_operations()
                
                # Log hardware info
                gpu_info = self.gpu_manager.get_gpu_info()
                cpu_info = self.cpu_optimizer.get_cpu_info()
                
                self.logger.info(f"🧠 Hardware Info - GPU: {gpu_info.get('gpu_name', 'N/A')}, "
                               f"CPU Cores: {cpu_info.get('total_cores', 'N/A')}")
            else:
                self.logger.info("💻 Non-M1 hardware detected, using standard optimizations")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            # Continue without optimizations
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return [
            'regime_data_splitting_result',
            'regime_splitting_report',
            'regime_validation_results'
        ]
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime data splitting with comprehensive error handling and reporting.
        
        Args:
            data: Market data for regime tagging
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with regime data splitting results
        """
        self.start_time = datetime.now()
        self.logger.info('✂️ Starting Enhanced Regime Data Splitting')
        tprint('✂️ Starting Enhanced Regime Data Splitting')
        
        # Initialize report
        report = RegimeSplittingReport(status=RegimeSplittingStatus.IN_PROGRESS)
        tprint(f'📊 Initialized report with status: {report.status.value}')
        
        try:
            # Fast fail validation for critical inputs with standardized error messages
            if data is None:
                error_msg = "CRITICAL_ERROR: Input data is None. Action required: Provide valid market data for regime splitting."
                self.logger.error(f"❌ {error_msg}")
                tprint(f"❌ {error_msg}")
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append(error_msg)
                return self._create_failure_result(report, error_msg)
            
            if not isinstance(pipeline_state, dict):
                error_msg = "CRITICAL_ERROR: Pipeline state must be a dictionary. Action required: Ensure pipeline_state is properly initialized as a dict."
                self.logger.error(f"❌ {error_msg}")
                tprint(f"❌ {error_msg}")
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append(error_msg)
                return self._create_failure_result(report, error_msg)
            
            if not self.config.symbol or not self.config.exchange:
                error_msg = "CRITICAL_ERROR: Symbol and exchange must be configured. Action required: Set config.symbol and config.exchange before execution."
                self.logger.error(f"❌ {error_msg}")
                tprint(f"❌ {error_msg}")
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append(error_msg)
                return self._create_failure_result(report, error_msg)
        
        try:
            # Step 1: Validate inputs
            tprint('🔍 Step 1: Validating inputs...')
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result['valid']:
                tprint(f'❌ Input validation failed: {validation_result["errors"]}')
                report.status = RegimeSplittingStatus.VALIDATION_FAILED
                report.errors.extend(validation_result['errors'])
                return self._create_failure_result(report, "Input validation failed")
            tprint('✅ Input validation passed')
            
            # Step 2: Load and prepare data
            tprint('📊 Step 2: Loading and preparing market data...')
            market_data = await self._load_and_prepare_data(data)
            if market_data is None:
                tprint('❌ Failed to load market data')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append("Failed to load market data")
                return self._create_failure_result(report, "Data loading failed")
            tprint(f'✅ Market data loaded: {market_data.shape}')
            
            # Step 3: Get regime discovery results
            tprint('🔍 Step 3: Retrieving regime discovery results...')
            regime_discovery = await self._get_regime_discovery_results(pipeline_state)
            if not regime_discovery:
                tprint('❌ No regime discovery results available')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append("No regime discovery results available")
                return self._create_failure_result(report, "Missing regime discovery results")
            tprint('✅ Regime discovery results retrieved')
            
            # Step 4: Perform regime data splitting
            tprint('✂️ Step 4: Performing regime data splitting...')
            splitting_result = await self._perform_regime_splitting(
                market_data, regime_discovery, report
            )
            
            if not splitting_result['success']:
                tprint(f'❌ Regime splitting failed: {splitting_result["errors"]}')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.extend(splitting_result['errors'])
                return self._create_failure_result(report, "Regime splitting failed")
            tprint('✅ Regime data splitting completed')
            
            # Step 5: Validate results
            tprint('🔍 Step 5: Validating splitting results...')
            validation_result = await self._validate_splitting_results(splitting_result, report)
            if not validation_result['valid']:
                tprint(f'❌ Result validation failed: {validation_result["errors"]}')
                report.status = RegimeSplittingStatus.VALIDATION_FAILED
                report.errors.extend(validation_result['errors'])
                return self._create_failure_result(report, "Result validation failed")
            tprint('✅ Result validation passed')
            
            # Step 6: Generate comprehensive report
            tprint('📊 Step 6: Generating comprehensive report...')
            report = await self._generate_comprehensive_report(
                splitting_result, market_data, report
            )
            tprint('✅ Comprehensive report generated')
            
            # Step 7: Create artifacts
            tprint('💾 Step 7: Creating artifacts...')
            artifacts = await self._create_artifacts(splitting_result, report)
            tprint('✅ Artifacts created')
            
            # Update metrics
            tprint('📈 Updating metrics...')
            self._update_metrics(report, splitting_result)
            
            report.status = RegimeSplittingStatus.COMPLETED
            self.logger.info(f'✅ Enhanced Regime Data Splitting completed: {self.metrics.regime_count} regimes processed')
            tprint(f'✅ Enhanced Regime Data Splitting completed: {self.metrics.regime_count} regimes processed')
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'regime_count': self.metrics.regime_count,
                    'execution_time': self.metrics.processing_time_seconds,
                    'data_quality_score': self.metrics.data_quality_score
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ Enhanced Regime Data Splitting failed: {e}')
            tprint(f'❌ Enhanced Regime Data Splitting failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            tprint(f'❌ Error details: {traceback.format_exc()}')
            
            report.status = RegimeSplittingStatus.FAILED
            report.errors.append(f"Unexpected error: {str(e)}")
            
            return self._create_failure_result(report, str(e))
    
    async def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and pipeline state."""
        self.logger.info("🔍 Validating inputs...")
        tprint("🔍 Validating inputs...")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check data availability with standardized error messages
        if data is None:
            validation_result['valid'] = False
            validation_result['errors'].append("VALIDATION_ERROR: Input data is None. Action required: Provide valid market data.")
        
        # Check pipeline state
        if not isinstance(pipeline_state, dict):
            validation_result['valid'] = False
            validation_result['errors'].append("VALIDATION_ERROR: Pipeline state must be a dictionary. Action required: Initialize pipeline_state as dict.")
        
        # Check for required regime discovery results
        required_keys = ['hmm_regime_discovery_result']
        for key in required_keys:
            if key not in pipeline_state:
                validation_result['warnings'].append(f"WARNING: Missing pipeline state key '{key}'. Action suggested: Ensure previous steps completed successfully.")
        
        # Check configuration
        if not self.config.symbol:
            validation_result['valid'] = False
            validation_result['errors'].append("CONFIG_ERROR: Symbol not configured. Action required: Set config.symbol.")
        
        if not self.config.exchange:
            validation_result['valid'] = False
            validation_result['errors'].append("CONFIG_ERROR: Exchange not configured. Action required: Set config.exchange.")
        
        if validation_result['valid']:
            self.logger.info("✅ Input validation passed")
            tprint("✅ Input validation passed")
        else:
            self.logger.error(f"❌ Input validation failed: {validation_result['errors']}")
            tprint(f"❌ Input validation failed: {validation_result['errors']}")
        
        return validation_result
    
    async def _load_and_prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime splitting using common utilities."""
        self.logger.info("📊 Loading and preparing market data...")
        tprint("📊 Loading and preparing market data...")
        
        try:
            if data is None:
                self.logger.error("❌ No data provided")
                return None
            
            # Handle different data types with memory optimization
            if isinstance(data, pd.DataFrame):
                # Use view instead of copy when possible to save memory
                market_data = data.copy() if data.is_copy is None else data
            elif isinstance(data, dict) and 'data' in data:
                market_data = data['data']
            else:
                self.logger.error(f"❌ Unsupported data type: {type(data)}")
                tprint(f"❌ Unsupported data type: {type(data)}")
                return None
            
            # Validate DataFrame structure using common utilities
            if not isinstance(market_data, pd.DataFrame):
                self.logger.error("❌ Data is not a DataFrame")
                return None
            
            if market_data.empty:
                self.logger.error("❌ Market data is empty")
                return None
            
            # Check for required columns using common utilities
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(market_data, required_columns):
                self.logger.warning("⚠️ Missing required columns, creating fallback columns")
                # Create fallback columns with validation
                for col in required_columns:
                    if col not in market_data.columns:
                        if col == 'volume':
                            market_data[col] = 1000.0  # Default volume
                        else:
                            market_data[col] = market_data.get('close', 100.0)  # Use close price as fallback
            
            # Optimize DataFrame for M1 if available
            if is_m1_available():
                market_data = optimize_dataframe_for_m1(market_data)
            
            # Apply memory optimization
            market_data = self.memory_optimizer.optimize_dataframe_memory(market_data)
            
            # Validate and clean data using common utilities
            market_data = safe_fillna(market_data, method='ffill')
            
            # Optimize data types
            market_data = optimize_dataframe_dtypes(market_data)
            
            # Validate data quality using common utilities
            quality_metrics = calculate_data_quality_metrics(market_data)
            if quality_metrics.get('missing_percentage', 0) > 10:
                self.logger.warning(f"⚠️ High missing data percentage: {quality_metrics['missing_percentage']:.2f}%")
            
            # Create data quality report
            quality_report = create_data_quality_report(market_data)
            if quality_report.get('issues'):
                self.logger.warning(f"⚠️ Data quality issues detected: {quality_report['issues']}")
            
            self.logger.info(f"✅ Market data loaded: {market_data.shape}")
            tprint(f"✅ Market data loaded: {market_data.shape}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            tprint(f"❌ Error loading market data: {e}")
            return None
    
    async def _get_regime_discovery_results(self, pipeline_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get regime discovery results from pipeline state."""
        self.logger.info("🔍 Retrieving regime discovery results...")
        
        try:
            # Try different possible keys for regime discovery results
            possible_keys = [
                'hmm_regime_discovery_result',
                'regime_discovery_result',
                'hmm_clustering_result',
                'regime_states',
                'regime_probabilities'
            ]
            
            regime_discovery = None
            for key in possible_keys:
                if key in pipeline_state and pipeline_state[key]:
                    regime_discovery = pipeline_state[key]
                    self.logger.info(f"✅ Found regime discovery results under key: {key}")
                    break
            
            if regime_discovery is None:
                self.logger.error("❌ No regime discovery results found in pipeline state")
                return None
            
            # Validate regime discovery results
            if isinstance(regime_discovery, dict):
                if not regime_discovery:
                    self.logger.error("❌ Regime discovery results are empty")
                    return None
            elif isinstance(regime_discovery, list):
                if len(regime_discovery) == 0:
                    self.logger.error("❌ Regime discovery results list is empty")
                    return None
            
            return regime_discovery
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving regime discovery results: {e}")
            return None
    
    async def _perform_regime_splitting(
        self, 
        market_data: pd.DataFrame, 
        regime_discovery: Dict[str, Any],
        report: RegimeSplittingReport
    ) -> Dict[str, Any]:
        """Perform the actual regime data splitting process using common utilities and hardware optimizations."""
        self.logger.info("✂️ Performing regime data splitting...")
        
        # Use memory checkpoint for M1 optimization
        with memory_checkpoint("regime_splitting"):
            try:
                # Extract regime states and probabilities using safe operations
                regime_states = self._extract_regime_states(regime_discovery)
                regime_probabilities = self._extract_regime_probabilities(regime_discovery)
                
                if regime_states is None:
                    return {
                        'success': False,
                        'errors': ['Failed to extract regime states'],
                        'data': None
                    }
                
                # Align data lengths with memory optimization
                min_len = min(len(market_data), len(regime_states))
                tprint(f"📊 Aligning data lengths: {len(market_data)} -> {min_len}")
                
                # Use safe DataFrame operations
                market_data_aligned = safe_dataframe_operation(
                    market_data, lambda df: df.iloc[:min_len].copy()
                )
                regime_states_aligned = regime_states[:min_len]
                
                if regime_probabilities is not None:
                    regime_probabilities_aligned = regime_probabilities[:min_len]
                else:
                    regime_probabilities_aligned = None
                
                # Clean up original data references to free memory
                del market_data
                mem_optimize()  # Use common memory optimization
                
                # Add regime information to market data using safe operations
                market_data_aligned['regime_state'] = regime_states_aligned
                if regime_probabilities_aligned is not None:
                    market_data_aligned['regime_probability'] = regime_probabilities_aligned
                    # Use safe math operations for confidence calculation
                    market_data_aligned['regime_confidence'] = safe_apply_function(
                        market_data_aligned, 
                        lambda row: math_safe_func(np.max, regime_probabilities_aligned[row.name], default=1.0)
                    )
                else:
                    market_data_aligned['regime_confidence'] = 1.0
                
                # Calculate regime statistics using common utilities
                regime_stats = self._calculate_regime_statistics_optimized(market_data_aligned)
                
                # Create regime data dictionary
                regime_data = {
                    'market_data': market_data_aligned,
                    'regime_states': regime_states_aligned,
                    'regime_probabilities': regime_probabilities_aligned,
                    'regime_statistics': regime_stats
                }
                
                # Optimize final DataFrame for M1 if available
                if is_m1_available():
                    regime_data['market_data'] = optimize_dataframe_for_m1(regime_data['market_data'])
                
                self.logger.info(f"✅ Regime splitting completed: {len(np.unique(regime_states_aligned))} regimes")
                
                return {
                    'success': True,
                    'data': regime_data,
                    'regime_stats': regime_stats,
                    'errors': []
                }
                
            except Exception as e:
                self.logger.error(f"❌ Error in regime splitting: {e}")
                return {
                    'success': False,
                    'errors': [f"Regime splitting failed: {str(e)}"],
                    'data': None
                }
    
    def _extract_regime_states(self, regime_discovery: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract regime states from regime discovery results."""
        try:
            # Try different possible structures
            if 'regime_states' in regime_discovery:
                states = regime_discovery['regime_states']
            elif 'states' in regime_discovery:
                states = regime_discovery['states']
            elif 'predictions' in regime_discovery:
                states = regime_discovery['predictions']
            elif isinstance(regime_discovery, list):
                states = regime_discovery
            else:
                self.logger.error("❌ Cannot extract regime states from discovery results")
                return None
            
            # Convert to numpy array if needed
            if not isinstance(states, np.ndarray):
                states = np.array(states)
            
            return states
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting regime states: {e}")
            return None
    
    def _extract_regime_probabilities(self, regime_discovery: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract regime probabilities from regime discovery results."""
        try:
            # Try different possible structures
            if 'regime_probabilities' in regime_discovery:
                probs = regime_discovery['regime_probabilities']
            elif 'probabilities' in regime_discovery:
                probs = regime_discovery['probabilities']
            elif 'proba' in regime_discovery:
                probs = regime_discovery['proba']
            else:
                return None  # Probabilities are optional
            
            # Convert to numpy array if needed
            if not isinstance(probs, np.ndarray):
                probs = np.array(probs)
            
            return probs
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting regime probabilities: {e}")
            return None
    
    def _calculate_regime_statistics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive regime statistics using common utilities."""
        try:
            regime_stats = {}
            
            # Basic regime distribution using safe operations
            regime_counts = safe_dataframe_operation(
                market_data, 
                lambda df: df['regime_state'].value_counts().to_dict()
            )
            regime_stats['regime_distribution'] = regime_counts
            regime_stats['total_regimes'] = len(regime_counts)
            regime_stats['total_data_points'] = len(market_data)
            
            # Calculate statistics per regime using safe operations
            regime_details = {}
            unique_regimes = safe_dataframe_operation(
                market_data, 
                lambda df: df['regime_state'].unique()
            )
            
            for regime_id in unique_regimes:
                regime_data = safe_filter_dataframe(
                    market_data, 
                    f"regime_state == {regime_id}"
                )
                
                # Use safe math operations for calculations
                count = len(regime_data)
                percentage = safe_divide(count, len(market_data), 0.0) * 100
                
                regime_details[regime_id] = {
                    'count': count,
                    'percentage': percentage,
                    'volatility_std': safe_std(regime_data['close']) if 'close' in regime_data.columns else 0.0,
                    'mean_volume': safe_mean(regime_data['volume']) if 'volume' in regime_data.columns else 0.0,
                    'mean_price': safe_mean(regime_data['close']) if 'close' in regime_data.columns else 0.0,
                    'price_range': {
                        'min': safe_float(regime_data['close'].min()) if 'close' in regime_data.columns else 0.0,
                        'max': safe_float(regime_data['close'].max()) if 'close' in regime_data.columns else 0.0
                    }
                }
            
            regime_stats['regime_details'] = regime_details
            
            return regime_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_statistics_optimized(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime statistics with M1 optimizations."""
        try:
            # Use M1-optimized operations if available
            if is_m1_available():
                return self._calculate_regime_statistics_m1_optimized(market_data)
            else:
                return self._calculate_regime_statistics(market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error in optimized regime statistics: {e}")
            return self._calculate_regime_statistics(market_data)
    
    def _calculate_regime_statistics_m1_optimized(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime statistics optimized for M1 hardware."""
        try:
            regime_stats = {}
            
            # Use M1-optimized array operations
            regime_states = create_m1_optimized_array(market_data['regime_state'].values)
            
            # Basic regime distribution using M1 optimizations
            unique_regimes, counts = np.unique(regime_states, return_counts=True)
            regime_counts = dict(zip(unique_regimes, counts))
            
            regime_stats['regime_distribution'] = regime_counts
            regime_stats['total_regimes'] = len(unique_regimes)
            regime_stats['total_data_points'] = len(market_data)
            
            # Calculate statistics per regime using M1-optimized operations
            regime_details = {}
            for regime_id in unique_regimes:
                mask = regime_states == regime_id
                regime_data = market_data[mask]
                
                # Use M1-optimized math operations
                count = int(np.sum(mask))
                percentage = safe_divide(count, len(market_data), 0.0) * 100
                
                regime_details[regime_id] = {
                    'count': count,
                    'percentage': percentage,
                    'volatility_std': math_safe_std(regime_data['close'].values) if 'close' in regime_data.columns else 0.0,
                    'mean_volume': math_safe_mean(regime_data['volume'].values) if 'volume' in regime_data.columns else 0.0,
                    'mean_price': math_safe_mean(regime_data['close'].values) if 'close' in regime_data.columns else 0.0,
                    'price_range': {
                        'min': safe_float(regime_data['close'].min()) if 'close' in regime_data.columns else 0.0,
                        'max': safe_float(regime_data['close'].max()) if 'close' in regime_data.columns else 0.0
                    }
                }
            
            regime_stats['regime_details'] = regime_details
            
            return regime_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error in M1-optimized regime statistics: {e}")
            return self._calculate_regime_statistics(market_data)
    
    async def _validate_splitting_results(
        self, 
        splitting_result: Dict[str, Any], 
        report: RegimeSplittingReport
    ) -> Dict[str, Any]:
        """Validate the results of regime splitting."""
        self.logger.info("🔍 Validating splitting results...")
        tprint("🔍 Validating splitting results...")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not splitting_result['success']:
                validation_result['valid'] = False
                validation_result['errors'].append("Splitting operation failed")
                return validation_result
            
            regime_data = splitting_result['data']
            if regime_data is None:
                validation_result['valid'] = False
                validation_result['errors'].append("No regime data produced")
                return validation_result
            
            # Validate market data
            market_data = regime_data['market_data']
            if market_data is None or market_data.empty:
                validation_result['valid'] = False
                validation_result['errors'].append("Market data is empty")
                return validation_result
            
            # Validate regime states
            regime_states = regime_data['regime_states']
            if regime_states is None or len(regime_states) == 0:
                validation_result['valid'] = False
                validation_result['errors'].append("No regime states found")
                return validation_result
            
            # Check regime diversity
            unique_regimes = len(np.unique(regime_states))
            if unique_regimes < 2:
                validation_result['warnings'].append(f"Only {unique_regimes} regime(s) found - may indicate poor regime discovery")
            
            # Check data alignment
            if len(market_data) != len(regime_states):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Data length mismatch: market_data={len(market_data)}, regime_states={len(regime_states)}")
            
            # Validate regime statistics
            regime_stats = regime_data['regime_statistics']
            if not regime_stats or 'total_regimes' not in regime_stats:
                validation_result['warnings'].append("Incomplete regime statistics")
            
            # Additional data consistency checks
            # Check for regime state consistency
            if 'regime_state' in market_data.columns:
                regime_states_in_data = market_data['regime_state'].values
                if not np.array_equal(regime_states, regime_states_in_data):
                    validation_result['warnings'].append("Regime states in data don't match extracted regime states")
            
            # Check for regime probability consistency
            if 'regime_probability' in market_data.columns and regime_data['regime_probabilities'] is not None:
                regime_probs_in_data = market_data['regime_probability'].values
                if len(regime_probs_in_data) != len(regime_data['regime_probabilities']):
                    validation_result['warnings'].append("Regime probabilities length mismatch")
            
            # Check for data type consistency
            if not isinstance(regime_states, np.ndarray):
                validation_result['warnings'].append("Regime states should be numpy array")
            
            # Check for reasonable regime values
            if len(regime_states) > 0:
                min_regime = np.min(regime_states)
                max_regime = np.max(regime_states)
                if min_regime < 0:
                    validation_result['warnings'].append(f"Negative regime values detected: min={min_regime}")
                if max_regime > 100:
                    validation_result['warnings'].append(f"Unusually high regime values detected: max={max_regime}")
            
            if validation_result['valid']:
                self.logger.info("✅ Splitting results validation passed")
                tprint("✅ Splitting results validation passed")
            else:
                self.logger.error(f"❌ Splitting results validation failed: {validation_result['errors']}")
                tprint(f"❌ Splitting results validation failed: {validation_result['errors']}")
            
            if validation_result['warnings']:
                tprint(f"⚠️ Validation warnings: {validation_result['warnings']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating splitting results: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    async def _generate_comprehensive_report(
        self,
        splitting_result: Dict[str, Any],
        market_data: pd.DataFrame,
        report: RegimeSplittingReport
    ) -> RegimeSplittingReport:
        """Generate comprehensive execution report."""
        self.logger.info("📊 Generating comprehensive report...")
        
        try:
            # Update metrics
            regime_data = splitting_result['data']
            regime_stats = regime_data['regime_statistics']
            
            self.metrics.total_data_points = len(market_data)
            self.metrics.regime_count = regime_stats.get('total_regimes', 0)
            self.metrics.regime_distribution = regime_stats.get('regime_distribution', {})
            self.metrics.processing_time_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate data quality score
            self.metrics.data_quality_score = self._calculate_data_quality_score(market_data)
            
            # Calculate regime continuity score
            self.metrics.regime_continuity_score = self._calculate_regime_continuity_score(
                regime_data['regime_states']
            )
            
            # Generate execution summary
            report.execution_summary = {
                'total_data_points': self.metrics.total_data_points,
                'regime_count': self.metrics.regime_count,
                'processing_time_seconds': self.metrics.processing_time_seconds,
                'data_quality_score': self.metrics.data_quality_score,
                'regime_continuity_score': self.metrics.regime_continuity_score,
                'memory_usage_mb': self.metrics.memory_usage_mb
            }
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            self.logger.info("✅ Comprehensive report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            report.errors.append(f"Report generation failed: {str(e)}")
            return report
    
    def _calculate_data_quality_score(self, market_data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1) using common utilities."""
        try:
            # Use common data quality metrics
            quality_metrics = calculate_data_quality_metrics(market_data)
            
            score = 1.0
            
            # Check for null values using safe operations
            null_ratio = safe_divide(
                quality_metrics.get('missing_values', 0),
                len(market_data) * len(market_data.columns),
                0.0
            )
            score -= null_ratio * 0.3
            
            # Check for duplicate rows using safe operations
            duplicate_ratio = safe_divide(
                quality_metrics.get('duplicate_rows', 0),
                len(market_data),
                0.0
            )
            score -= duplicate_ratio * 0.2
            
            # Check for infinite values using safe operations
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(market_data) > 0:
                inf_count = safe_dataframe_operation(
                    market_data,
                    lambda df: np.isinf(df[numeric_cols]).sum().sum()
                )
                inf_ratio = safe_divide(inf_count, len(market_data) * len(numeric_cols), 0.0)
                score -= inf_ratio * 0.3
            
            # Check for zero/negative prices using safe operations
            if 'close' in market_data.columns:
                invalid_prices = safe_divide(
                    (market_data['close'] <= 0).sum(),
                    len(market_data),
                    0.0
                )
                score -= invalid_prices * 0.2
            
            # Use safe math operations for final score
            return math_validate_range(score, 0.0, 1.0, "data_quality_score")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating data quality score: {e}")
            return 0.5  # Default score
    
    def _calculate_regime_continuity_score(self, regime_states: np.ndarray) -> float:
        """Calculate regime continuity score (0-1) using safe math operations."""
        try:
            if len(regime_states) < 2:
                return 1.0
            
            # Count regime transitions using safe operations
            transitions = math_safe_func(
                np.sum, 
                regime_states[1:] != regime_states[:-1],
                default=0
            )
            transition_ratio = safe_divide(transitions, len(regime_states) - 1, 0.0)
            
            # Higher continuity = fewer transitions (score closer to 1)
            continuity_score = 1.0 - min(1.0, transition_ratio * 2)
            
            # Validate the score is in valid range
            return math_validate_range(continuity_score, 0.0, 1.0, "continuity_score")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime continuity score: {e}")
            return 0.5  # Default score
    
    def _generate_recommendations(self, report: RegimeSplittingReport) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []
        
        # Data quality recommendations
        if self.metrics.data_quality_score < 0.8:
            recommendations.append("Consider improving data quality - current score is below 0.8")
        
        # Regime diversity recommendations
        if self.metrics.regime_count < 3:
            recommendations.append("Consider adjusting regime discovery parameters - only few regimes detected")
        
        # Continuity recommendations
        if self.metrics.regime_continuity_score < 0.7:
            recommendations.append("Regime transitions are frequent - consider smoothing parameters")
        
        # Performance recommendations
        if self.metrics.processing_time_seconds > 60:
            recommendations.append("Processing time is high - consider optimizing data size or parameters")
        
        # Memory recommendations
        if self.metrics.memory_usage_mb > 1000:
            recommendations.append("High memory usage detected - consider streaming processing")
        
        return recommendations
    
    async def _create_artifacts(
        self, 
        splitting_result: Dict[str, Any], 
        report: RegimeSplittingReport
    ) -> Dict[str, Any]:
        """Create comprehensive artifacts using common serialization utilities."""
        self.logger.info("💾 Creating artifacts...")
        
        try:
            # Create artifacts with enhanced metadata
            artifacts = {
                'regime_data_splitting_result': {
                    'regime_data': splitting_result['data'],
                    'regime_stats': splitting_result['regime_stats'],
                    'processing_metrics': {
                        'total_data_points': self.metrics.total_data_points,
                        'regime_count': self.metrics.regime_count,
                        'processing_time_seconds': self.metrics.processing_time_seconds,
                        'data_quality_score': self.metrics.data_quality_score,
                        'regime_continuity_score': self.metrics.regime_continuity_score,
                        'memory_usage_mb': get_memory_usage().get('used_memory', 0) / (1024 * 1024),
                        'hardware_optimized': is_m1_available()
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'execution_timestamp': datetime.now().isoformat(),
                        'component_version': 'enhanced_v2.0_common_utils',
                        'hardware_info': {
                            'is_m1': is_m1_available(),
                            'mps_available': is_mps_available(),
                            'cpu_cores': self.cpu_optimizer.get_cpu_info().get('total_cores', 'unknown')
                        }
                    }
                },
                'regime_splitting_report': {
                    'status': report.status.value,
                    'metrics': {
                        'total_data_points': self.metrics.total_data_points,
                        'regime_count': self.metrics.regime_count,
                        'regime_distribution': self.metrics.regime_distribution,
                        'processing_time_seconds': self.metrics.processing_time_seconds,
                        'data_quality_score': self.metrics.data_quality_score,
                        'regime_continuity_score': self.metrics.regime_continuity_score
                    },
                    'execution_summary': report.execution_summary,
                    'warnings': report.warnings,
                    'errors': report.errors,
                    'recommendations': report.recommendations,
                    'timestamp': report.timestamp
                },
                'regime_validation_results': {
                    'validation_checks_passed': self.metrics.validation_checks_passed,
                    'validation_checks_failed': self.metrics.validation_checks_failed,
                    'data_quality_validation': self.metrics.data_quality_score > 0.7,
                    'regime_diversity_validation': self.metrics.regime_count >= 2,
                    'continuity_validation': self.metrics.regime_continuity_score > 0.5,
                    'overall_validation_passed': (
                        self.metrics.data_quality_score > 0.7 and
                        self.metrics.regime_count >= 2 and
                        self.metrics.regime_continuity_score > 0.5
                    )
                }
            }
            
            # Save artifacts using common serialization utilities
            await self._save_artifacts_to_files(artifacts)
            
            self.logger.info("✅ Artifacts created successfully")
            return artifacts
            
        except Exception as e:
            self.logger.error(f"❌ Error creating artifacts: {e}")
            return {}
    
    async def _save_artifacts_to_files(self, artifacts: Dict[str, Any]) -> None:
        """Save artifacts to files using common serialization utilities."""
        try:
            # Create artifacts directory
            artifacts_dir = Path("artifacts/regime_data_splitting")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime data splitting result as parquet
            if 'regime_data' in artifacts['regime_data_splitting_result']:
                regime_data = artifacts['regime_data_splitting_result']['regime_data']
                if 'market_data' in regime_data:
                    parquet_path = artifacts_dir / "regime_market_data.parquet"
                    safe_to_parquet(regime_data['market_data'], parquet_path)
                    self.logger.info(f"💾 Saved regime market data to {parquet_path}")
            
            # Save report as JSON
            report_path = artifacts_dir / "regime_splitting_report.json"
            self.serializer.save(artifacts['regime_splitting_report'], str(report_path))
            self.logger.info(f"💾 Saved regime splitting report to {report_path}")
            
            # Save validation results as JSON
            validation_path = artifacts_dir / "regime_validation_results.json"
            self.serializer.save(artifacts['regime_validation_results'], str(validation_path))
            self.logger.info(f"💾 Saved validation results to {validation_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saving artifacts to files: {e}")
            # Continue without file saving
    
    def _update_metrics(self, report: RegimeSplittingReport, splitting_result: Dict[str, Any]) -> None:
        """Update metrics based on execution results."""
        try:
            self.metrics.validation_checks_passed = len([r for r in report.validation_results.values() if r])
            self.metrics.validation_checks_failed = len([r for r in report.validation_results.values() if not r])
            self.metrics.warnings_count = len(report.warnings)
            self.metrics.errors_count = len(report.errors)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating metrics: {e}")
    
    def _create_failure_result(self, report: RegimeSplittingReport, error_message: str) -> ComponentResult:
        """Create a failure result with comprehensive error information."""
        return ComponentResult(
            success=False,
            artifacts={
                'regime_splitting_report': {
                    'status': report.status.value,
                    'errors': report.errors,
                    'warnings': report.warnings,
                    'timestamp': report.timestamp
                }
            },
            error_message=error_message,
            metadata={
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'failure_timestamp': datetime.now().isoformat()
            }
        )
    
    def cleanup(self) -> None:
        """Clean up resources and stop hardware optimizations."""
        try:
            # Stop memory monitoring
            stop_m1_memory_monitoring()
            
            # Clean up M1 optimizers
            cleanup_m1_optimizers()
            
            # Clear any cached data
            mem_optimize()
            
            self.logger.info("🧹 Cleanup completed successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup