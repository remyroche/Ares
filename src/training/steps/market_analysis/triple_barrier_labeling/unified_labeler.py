"""
Unified Triple Barrier Labeling Implementation

This module provides a streamlined, robust implementation of triple barrier labeling
with comprehensive error handling, validation, and reporting.

Key Features:
- Unified configuration and execution
- Explicit error handling (no silent failures)
- Comprehensive validation framework
- Enhanced reporting and metrics
- Performance optimization with proper fallbacks
- Regime-aware labeling support

Transaction cost semantics:
- We assume 0.08% per trade (round-trip) via transaction_cost=0.0008
- The transaction cost is applied consistently to all exits:
  - Profit-take exit: net = +pt_mult - transaction_cost
  - Stop-loss exit:  net = -sl_mult - transaction_cost
  - Time-barrier exit (no barrier touched): net = -transaction_cost
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from src.utils.logger import get_logger

# Hardware optimization imports with proper error handling
try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimizations not available: {e}")
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False

# Numba acceleration with proper error handling
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False

# Constants for risk management
DEFAULT_PROFIT_TAKE_MULTIPLIER = 0.002  # 0.2% - conservative
DEFAULT_STOP_LOSS_MULTIPLIER = 0.001    # 0.1% - conservative
DEFAULT_TRANSACTION_COST = 0.0008       # 0.08% transaction cost
MIN_BARRIER_MULTIPLIER = 0.0005         # Minimum 0.05% barrier
MAX_BARRIER_MULTIPLIER = 0.05           # Maximum 5% barrier
EPSILON = 1e-10                         # Numerical stability constant

# Custom Exception Classes
class TripleBarrierError(Exception):
    """Base exception for triple barrier labeling errors."""
    pass

class ValidationError(TripleBarrierError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(TripleBarrierError):
    """Raised when configuration is invalid."""
    pass

class HardwareOptimizationError(TripleBarrierError):
    """Raised when hardware optimization fails critically."""
    pass

class DataQualityError(TripleBarrierError):
    """Raised when data quality is insufficient."""
    pass

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning."""
        self.warnings.append(warning)

@dataclass
class TripleBarrierConfig:
    """Unified configuration for triple barrier labeling."""
    
    # Core parameters
    profit_take_multiplier: float = DEFAULT_PROFIT_TAKE_MULTIPLIER
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    
    # Behavior flags
    binary_classification: bool = True
    regime_aware: bool = True
    regime_column: str = 'hmm_regime'
    
    # Error handling behavior
    fail_on_validation_error: bool = True
    fail_on_hardware_optimization_error: bool = False
    fail_on_data_quality_error: bool = True
    
    # Performance settings
    enable_numba_acceleration: bool = True
    enable_hardware_optimizations: bool = True
    memory_limit_gb: float = 8.0
    
    # Validation settings
    min_data_points: int = 100
    max_missing_data_ratio: float = 0.1
    min_label_distribution_ratio: float = 0.05
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate profit take multiplier
        if not (MIN_BARRIER_MULTIPLIER <= self.profit_take_multiplier <= MAX_BARRIER_MULTIPLIER):
            errors.append(f"Profit take multiplier {self.profit_take_multiplier:.4f} outside valid range [{MIN_BARRIER_MULTIPLIER:.4f}, {MAX_BARRIER_MULTIPLIER:.4f}]")
        
        # Validate stop loss multiplier
        if not (MIN_BARRIER_MULTIPLIER <= self.stop_loss_multiplier <= MAX_BARRIER_MULTIPLIER):
            errors.append(f"Stop loss multiplier {self.stop_loss_multiplier:.4f} outside valid range [{MIN_BARRIER_MULTIPLIER:.4f}, {MAX_BARRIER_MULTIPLIER:.4f}]")
        
        # Check risk-reward ratio
        if self.profit_take_multiplier <= 0 or self.stop_loss_multiplier <= 0:
            errors.append("Profit take and stop loss multipliers must be positive")
        else:
            risk_reward_ratio = self.profit_take_multiplier / self.stop_loss_multiplier
            if risk_reward_ratio < 0.5:
                errors.append(f"Risk-reward ratio {risk_reward_ratio:.2f} < 0.5 - very unprofitable")
            elif risk_reward_ratio < 1.0:
                # This is a warning, not an error - some strategies may use this
                pass
        
        # Check if barriers are too close
        barrier_diff = abs(self.profit_take_multiplier - self.stop_loss_multiplier)
        if barrier_diff < 0.0005:
            errors.append(f"Profit take and stop loss too close (diff: {barrier_diff:.4f} < 0.05%)")
        
        # Validate transaction cost
        if self.transaction_cost < 0:
            errors.append("Transaction cost cannot be negative")
        
        # Validate time parameters
        if self.time_barrier_minutes <= 0:
            errors.append("Time barrier must be positive")
        
        if self.max_lookahead <= 0:
            errors.append("Max lookahead must be positive")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

@dataclass
class TripleBarrierResult:
    """Result of triple barrier labeling execution."""
    
    # Core results
    success: bool
    labeled_data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    
    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    execution_duration: float = 0.0
    
    # Data statistics
    input_data_shape: Tuple[int, int] = (0, 0)
    output_data_shape: Tuple[int, int] = (0, 0)
    data_quality_score: float = 0.0
    
    # Labeling results
    total_labels_generated: int = 0
    label_distribution: Dict[int, int] = field(default_factory=dict)
    labeling_method_used: str = "unknown"
    
    # Performance metrics
    numba_acceleration_used: bool = False
    hardware_optimizations_used: List[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    
    # Validation results
    validation_passed: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    # Error tracking
    non_critical_failures: List[str] = field(default_factory=list)
    fallback_methods_used: List[str] = field(default_factory=list)
    
    # Quality metrics
    label_quality_score: float = 0.0
    regime_coverage: Dict[int, int] = field(default_factory=dict)
    barrier_hit_statistics: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime objects to strings
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result
    
    def generate_summary(self) -> str:
        """Generate human-readable summary."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return f"""
Triple Barrier Labeling Execution Summary
=========================================
Status: {status}
Duration: {self.execution_duration:.2f}s
Input: {self.input_data_shape[0]:,} rows × {self.input_data_shape[1]} columns
Output: {self.output_data_shape[0]:,} rows × {self.output_data_shape[1]} columns
Labels Generated: {self.total_labels_generated:,}
Method: {self.labeling_method_used}
Quality Score: {self.data_quality_score:.2%}
Validation: {'✅ PASSED' if self.validation_passed else '❌ FAILED'}
Performance: {'⚡ Optimized' if self.numba_acceleration_used else '🐍 Standard'}
"""

class MetricsCollector:
    """Collects and tracks execution metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[f"{operation}_duration"] = duration
            del self.start_times[operation]
            return duration
        return 0.0
        
    def record_metric(self, name: str, value: Any):
        """Record a metric value."""
        self.metrics[name] = value
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "total_operations": len([k for k in self.metrics.keys() if k.endswith("_duration")]),
            "total_duration": sum(v for k, v in self.metrics.items() if k.endswith("_duration")),
            "metrics": self.metrics
        }

class ProgressReporter:
    """Reports progress during execution."""
    
    def __init__(self, logger, total_steps: int):
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def start_step(self, step_name: str):
        """Start a new step."""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        tprint(f"🔄 Step {self.current_step}/{self.total_steps} ({progress:.1f}%): {step_name}")
        self.logger.info(f"🔄 Step {self.current_step}/{self.total_steps} ({progress:.1f}%): {step_name}")
        
    def complete_step(self, step_name: str, metrics: Dict[str, Any] = None):
        """Complete a step with optional metrics."""
        elapsed = time.time() - self.start_time
        tprint(f"✅ Completed: {step_name} in {elapsed:.2f}s")
        self.logger.info(f"✅ Completed: {step_name} in {elapsed:.2f}s")
        if metrics:
            for key, value in metrics.items():
                tprint(f"   {key}: {value}")
                self.logger.info(f"   {key}: {value}")
                
    def report_failure(self, step_name: str, error: str):
        """Report a step failure."""
        tprint(f"❌ Failed: {step_name} - {error}")
        self.logger.error(f"❌ Failed: {step_name} - {error}")

class DataValidator:
    """Comprehensive data validation for triple barrier labeling."""
    
    def __init__(self, config: TripleBarrierConfig):
        self.config = config
        self.logger = get_logger('DataValidator')
    
    def validate_ohlc_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate OHLC data integrity."""
        result = ValidationResult()
        
        # Check if data is empty
        if data is None or data.empty:
            result.add_error("Input data is None or empty")
            return result
        
        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in data.columns]
        if missing:
            result.add_error(f"Missing required columns: {missing}")
            return result
        
        # Check minimum data points
        if len(data) < self.config.min_data_points:
            result.add_error(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
        
        # Check OHLC consistency
        invalid_high = data['high'] < np.maximum(data['open'], data['close'])
        invalid_low = data['low'] > np.minimum(data['open'], data['close'])
        
        if invalid_high.any():
            invalid_count = invalid_high.sum()
            result.add_error(f"Found {invalid_count} rows with high < max(open, close)")
        
        if invalid_low.any():
            invalid_count = invalid_low.sum()
            result.add_error(f"Found {invalid_count} rows with low > min(open, close)")
        
        # Check for missing data
        for col in required:
            missing_ratio = data[col].isna().sum() / len(data)
            if missing_ratio > self.config.max_missing_data_ratio:
                result.add_error(f"Column '{col}' has {missing_ratio:.1%} missing data (max: {self.config.max_missing_data_ratio:.1%})")
            elif missing_ratio > 0:
                result.add_warning(f"Column '{col}' has {missing_ratio:.1%} missing data")
        
        # Check for non-positive prices
        for col in required:
            non_positive = (data[col] <= 0).sum()
            if non_positive > 0:
                result.add_error(f"Column '{col}' has {non_positive} non-positive values")
        
        return result
    
    def validate_regime_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate regime data if regime-aware labeling is enabled."""
        result = ValidationResult()
        
        if not self.config.regime_aware:
            return result
        
        if self.config.regime_column not in data.columns:
            result.add_error(f"Regime column '{self.config.regime_column}' not found for regime-aware labeling")
            return result
        
        regime_data = data[self.config.regime_column]
        
        # Check for missing regime data
        missing_regime_ratio = regime_data.isna().sum() / len(regime_data)
        if missing_regime_ratio > self.config.max_missing_data_ratio:
            result.add_error(f"Regime column has {missing_regime_ratio:.1%} missing data")
        elif missing_regime_ratio > 0:
            result.add_warning(f"Regime column has {missing_regime_ratio:.1%} missing data")
        
        # Check regime distribution
        regime_counts = regime_data.value_counts()
        if len(regime_counts) < 2:
            result.add_warning(f"Only {len(regime_counts)} regime(s) detected")
        
        # Check for severely imbalanced regimes
        if len(regime_counts) > 1:
            max_count = regime_counts.max()
            min_count = regime_counts.min()
            balance_ratio = max_count / min_count
            if balance_ratio > 10:
                result.add_warning(f"Severely imbalanced regimes (ratio: {balance_ratio:.1f})")
        
        return result

class HardwareManager:
    """Unified hardware optimization manager."""
    
    def __init__(self, config: TripleBarrierConfig):
        self.config = config
        self.logger = get_logger('HardwareManager')
        self.optimizations = {}
        self.failures = []
        
    def initialize_optimizations(self):
        """Initialize all hardware optimizations with proper error handling."""
        if not HARDWARE_OPTIMIZATIONS_AVAILABLE:
            self.logger.warning("Hardware optimizations not available")
            return
        
        optimizations_to_try = [
            ('m1_cpu', self._init_m1_cpu),
            ('m1_memory', self._init_m1_memory),
            ('m1_gpu', self._init_m1_gpu)
        ]
        
        for name, init_func in optimizations_to_try:
            try:
                self.optimizations[name] = init_func()
                self.logger.info(f"✅ {name} optimization initialized")
            except Exception as e:
                self.failures.append(f"{name}: {e}")
                if self.config.fail_on_hardware_optimization_error:
                    raise HardwareOptimizationError(f"Critical {name} optimization failed: {e}")
                else:
                    self.logger.warning(f"⚠️ {name} optimization failed: {e}")
    
    def _init_m1_cpu(self):
        """Initialize M1 CPU optimizer."""
        return get_m1_cpu_optimizer()
    
    def _init_m1_memory(self):
        """Initialize M1 memory optimizer."""
        return get_m1_memory_optimizer(self.config.memory_limit_gb)
    
    def _init_m1_gpu(self):
        """Initialize M1 GPU manager."""
        return get_m1_gpu_manager()
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply hardware optimizations to dataframe."""
        if 'm1_memory' in self.optimizations:
            try:
                return self.optimizations['m1_memory'].optimize_dataframe_memory(data)
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
        
        return data

# Numba-accelerated triple barrier labeling function
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def _numba_triple_barrier_labels(
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        pt_mult: float, 
        sl_mult: float, 
        end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated triple barrier labeling with profit tracking and transaction costs."""
        labels = np.zeros(close.shape[0], dtype=np.int8)
        profit_pcts = np.zeros(close.shape[0], dtype=np.float64)
        transaction_costs = np.zeros(close.shape[0], dtype=np.float64)
        n = close.shape[0]
        
        for i in range(n - 1):
            entry_price = close[i]
            
            # Numerical stability check
            if entry_price <= EPSILON:
                labels[i] = 0
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
                continue
                
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i:
                labels[i] = 0
                # Time-barrier exit: apply transaction cost per trade
                profit_pcts[i] = -transaction_cost
                transaction_costs[i] = transaction_cost
                continue
                
            # Default to time-barrier outcome unless a barrier is hit
            lab = 0
            profit_pct = -transaction_cost
            tx_cost = transaction_cost
            
            for j in range(i + 1, end_idx):
                if high[j] >= profit_barrier:
                    lab = 1
                    gross_profit = pt_mult
                    tx_cost = transaction_cost
                    profit_pct = gross_profit - tx_cost
                    break
                    
                if low[j] <= stop_barrier:
                    lab = -1
                    gross_loss = -sl_mult
                    tx_cost = transaction_cost
                    profit_pct = gross_loss - tx_cost
                    break
                    
            labels[i] = lab
            profit_pcts[i] = profit_pct
            transaction_costs[i] = tx_cost
            
        return (labels, profit_pcts, transaction_costs)

class UnifiedTripleBarrierLabeler:
    """Unified triple barrier labeling implementation with comprehensive error handling and reporting."""
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """Initialize the unified triple barrier labeler."""
        self.config = config or TripleBarrierConfig()
        self.logger = get_logger('UnifiedTripleBarrierLabeler')
        self.metrics = MetricsCollector()
        self.progress = None
        self.validator = DataValidator(self.config)
        self.hardware_manager = HardwareManager(self.config)
        
        # Initialize components
        self._initialize_components()
        
        # Log initialization
        self._log_initialization()
    
    def _initialize_components(self):
        """Initialize all components with proper error handling."""
        try:
            # Initialize hardware optimizations
            if self.config.enable_hardware_optimizations:
                self.hardware_manager.initialize_optimizations()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            if self.config.fail_on_hardware_optimization_error:
                raise
    
    def _log_initialization(self):
        """Log initialization parameters."""
        tprint('🚀 Initializing Unified Triple Barrier Labeler')
        tprint(f'📋 Configuration:')
        tprint(f'   → Profit take: {self.config.profit_take_multiplier:.4f} ({self.config.profit_take_multiplier*100:.2f}%)')
        tprint(f'   → Stop loss: {self.config.stop_loss_multiplier:.4f} ({self.config.stop_loss_multiplier*100:.2f}%)')
        tprint(f'   → Transaction cost: {self.config.transaction_cost:.4f} ({self.config.transaction_cost*100:.2f}%)')
        tprint(f'   → Time barrier: {self.config.time_barrier_minutes} minutes')
        tprint(f'   → Max lookahead: {self.config.max_lookahead}')
        tprint(f'   → Binary classification: {self.config.binary_classification}')
        tprint(f'   → Regime aware: {self.config.regime_aware}')
        tprint(f'   → Numba acceleration: {NUMBA_AVAILABLE}')
        tprint(f'   → Hardware optimizations: {HARDWARE_OPTIMIZATIONS_AVAILABLE}')
        
        # Also log to logger for consistency
        self.logger.info('🚀 Initializing Unified Triple Barrier Labeler')
        self.logger.info(f'📋 Configuration:')
        self.logger.info(f'   → Profit take: {self.config.profit_take_multiplier:.4f} ({self.config.profit_take_multiplier*100:.2f}%)')
        self.logger.info(f'   → Stop loss: {self.config.stop_loss_multiplier:.4f} ({self.config.stop_loss_multiplier*100:.2f}%)')
        self.logger.info(f'   → Transaction cost: {self.config.transaction_cost:.4f} ({self.config.transaction_cost*100:.2f}%)')
        self.logger.info(f'   → Time barrier: {self.config.time_barrier_minutes} minutes')
        self.logger.info(f'   → Max lookahead: {self.config.max_lookahead}')
        self.logger.info(f'   → Binary classification: {self.config.binary_classification}')
        self.logger.info(f'   → Regime aware: {self.config.regime_aware}')
        self.logger.info(f'   → Numba acceleration: {NUMBA_AVAILABLE}')
        self.logger.info(f'   → Hardware optimizations: {HARDWARE_OPTIMIZATIONS_AVAILABLE}')
    
    def apply_labeling(self, data: pd.DataFrame) -> TripleBarrierResult:
        """Main entry point for triple barrier labeling."""
        tprint('🏷️ Starting Triple Barrier Labeling Process')
        start_time = datetime.now()
        result = TripleBarrierResult(
            success=False,  # Will be set to True if successful
            start_time=start_time,
            input_data_shape=data.shape if data is not None else (0, 0)
        )
        
        try:
            # Setup progress reporting
            total_steps = 6
            self.progress = ProgressReporter(self.logger, total_steps)
            
            # Step 1: Validate input data
            self.progress.start_step("Data Validation")
            validation_result = self._validate_input_data(data)
            if not validation_result.is_valid and self.config.fail_on_validation_error:
                raise ValidationError(f"Data validation failed: {'; '.join(validation_result.errors)}")
            
            result.validation_passed = validation_result.is_valid
            result.validation_errors = validation_result.errors
            result.validation_warnings = validation_result.warnings
            
            self.progress.complete_step("Data Validation", {
                "valid": validation_result.is_valid,
                "errors": len(validation_result.errors),
                "warnings": len(validation_result.warnings)
            })
            
            # Step 2: Prepare data
            self.progress.start_step("Data Preparation")
            prepared_data = self._prepare_data(data)
            self.progress.complete_step("Data Preparation", {"rows": len(prepared_data)})
            
            # Step 3: Apply hardware optimizations
            self.progress.start_step("Hardware Optimization")
            optimized_data = self._optimize_data(prepared_data)
            self.progress.complete_step("Hardware Optimization", {
                "optimizations": len(self.hardware_manager.optimizations)
            })
            
            # Step 4: Apply triple barrier labeling
            self.progress.start_step("Triple Barrier Labeling")
            labeled_data = self._apply_triple_barrier_labeling(optimized_data)
            self.progress.complete_step("Triple Barrier Labeling", {
                "labels_generated": len(labeled_data[labeled_data['label'].notna()])
            })
            
            # Step 5: Post-process results
            self.progress.start_step("Post-processing")
            final_data = self._post_process_results(labeled_data)
            self.progress.complete_step("Post-processing", {"final_rows": len(final_data)})
            
            # Step 6: Generate final metrics
            self.progress.start_step("Metrics Generation")
            self._populate_result_metrics(result, final_data)
            self.progress.complete_step("Metrics Generation")
            
            # Mark as successful
            result.success = True
            result.labeled_data = final_data
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            # Log final summary
            tprint("✅ Triple barrier labeling completed successfully")
            tprint(result.generate_summary())
            self.logger.info("✅ Triple barrier labeling completed successfully")
            self.logger.info(result.generate_summary())
            
            return result
            
        except Exception as e:
            # Handle any unexpected errors
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            tprint(f"❌ Triple barrier labeling failed: {e}")
            self.logger.error(f"❌ Triple barrier labeling failed: {e}")
            if self.progress:
                self.progress.report_failure("Triple Barrier Labeling", str(e))
            
            return result
    
    def _validate_input_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate input data comprehensively."""
        try:
            tprint("🔍 Validating input data...")
            
            # Validate OHLC data
            ohlc_result = self.validator.validate_ohlc_data(data)
            
            # Validate regime data if needed
            regime_result = self.validator.validate_regime_data(data)
            
            # Combine results
            combined_result = ValidationResult()
            combined_result.errors.extend(ohlc_result.errors)
            combined_result.errors.extend(regime_result.errors)
            combined_result.warnings.extend(ohlc_result.warnings)
            combined_result.warnings.extend(regime_result.warnings)
            combined_result.is_valid = len(combined_result.errors) == 0
            
            if combined_result.is_valid:
                tprint(f"✅ Data validation passed with {len(combined_result.warnings)} warnings")
            else:
                tprint(f"❌ Data validation failed with {len(combined_result.errors)} errors")
                
            return combined_result
            
        except Exception as e:
            tprint(f"❌ Data validation error: {e}")
            self.logger.error(f"Data validation error: {e}")
            result = ValidationResult()
            result.add_error(f"Validation process failed: {e}")
            return result
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for labeling."""
        try:
            tprint("📝 Preparing data for labeling...")
            
            # Create working copy
            prepared_data = data.copy()
            
            # Standardize column names
            rename_map = self._get_column_rename_map(prepared_data)
            if rename_map:
                prepared_data = prepared_data.rename(columns=rename_map)
                tprint(f"📝 Renamed columns: {rename_map}")
                self.logger.info(f"📝 Renamed columns: {rename_map}")
            
            # Ensure required columns exist
            required_columns = ['close', 'high', 'low']
            missing_columns = [col for col in required_columns if col not in prepared_data.columns]
            if missing_columns:
                error_msg = f"Missing required columns after preparation: {missing_columns}"
                tprint(f"❌ {error_msg}")
                raise DataQualityError(error_msg)

            # Synthesize 'open' if missing to satisfy validator consistency checks
            if 'open' not in prepared_data.columns:
                # Use previous close as open; for first row fallback to close
                synthesized_open = prepared_data['close'].shift(1)
                synthesized_open.iloc[0] = prepared_data['close'].iloc[0]
                prepared_data['open'] = synthesized_open
                self.logger.info("🧪 Synthesized 'open' column from previous 'close' to satisfy validation")
            
            tprint(f"✅ Data preparation completed: {len(prepared_data)} rows")
            return prepared_data
            
        except Exception as e:
            tprint(f"❌ Data preparation failed: {e}")
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def _get_column_rename_map(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get column rename mapping for standardization."""
        rename_map = {}
        canonical_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume'
        }
        
        for original, canonical in canonical_map.items():
            if original in data.columns and canonical not in data.columns:
                rename_map[original] = canonical
                
        return rename_map
    
    def _optimize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply hardware optimizations to data."""
        if not self.config.enable_hardware_optimizations:
            return data
        
        try:
            optimized_data = self.hardware_manager.optimize_dataframe(data)
            self.logger.debug("✅ Data optimized for hardware")
            return optimized_data
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization failed: {e}")
            return data
    
    def _apply_triple_barrier_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier labeling to data."""
        try:
            tprint("🏷️ Applying triple barrier labeling...")
            
            n = len(data)
            close = data['close'].to_numpy()
            high = data['high'].to_numpy()
            low = data['low'].to_numpy()
            idx = data.index
            
            # Calculate end indices
            end_idx_arr = self._calculate_end_indices(n, idx)
            
            # Apply barrier logic
            labels, profit_pcts, transaction_costs = self._apply_barrier_logic(
                close, high, low, end_idx_arr
            )
            
            # Add results to dataframe
            result_data = data.copy()
            result_data['label'] = labels
            result_data['potential_profit_pct'] = profit_pcts
            result_data['transaction_cost'] = transaction_costs
            result_data['net_profit_pct'] = profit_pcts  # Net profit after transaction costs
            result_data['labeling_method'] = 'unified'
            
            tprint(f"✅ Triple barrier labeling completed: {len(result_data)} rows processed")
            return result_data
            
        except Exception as e:
            tprint(f"❌ Triple barrier labeling failed: {e}")
            self.logger.error(f"Triple barrier labeling failed: {e}")
            raise
    
    def _calculate_end_indices(self, n: int, idx: pd.Index) -> np.ndarray:
        """Calculate end indices for barrier evaluation."""
        try:
            tprint("📊 Calculating end indices for barrier evaluation...")
            
            arange_n = np.arange(n, dtype=np.int64)
            end_by_lookahead = np.minimum(arange_n + int(self.config.max_lookahead), n)
            
            if isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing:
                try:
                    # Use non-deprecated conversion to nanoseconds since epoch
                    idx_ns = idx.asi8  # int64 nanoseconds
                    delta_ns = np.int64(self.config.time_barrier_minutes) * np.int64(60000000000)
                    end_times = idx_ns + delta_ns
                    end_by_time = np.searchsorted(idx_ns, end_times, side='right')
                    tprint(f"✅ Time barrier calculation completed: {self.config.time_barrier_minutes} minutes")
                except Exception as e:
                    tprint(f"⚠️ Time barrier calculation failed: {e}, using lookahead only")
                    self.logger.warning(f"⚠️ Time barrier calculation failed: {e}, using lookahead only")
                    end_by_time = end_by_lookahead
            else:
                tprint("📊 Using lookahead-based end indices (no datetime index)")
                end_by_time = end_by_lookahead
                
            result = np.minimum(end_by_lookahead, end_by_time).astype(np.int64)
            tprint(f"✅ End indices calculated: {len(result)} indices")
            return result
            
        except Exception as e:
            tprint(f"❌ End indices calculation failed: {e}")
            self.logger.error(f"End indices calculation failed: {e}")
            raise
    
    def _apply_barrier_logic(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        end_idx_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply barrier logic with performance optimization."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        transaction_costs = np.zeros(n, dtype=np.float64)
        
        pt_mult = self.config.profit_take_multiplier
        sl_mult = self.config.stop_loss_multiplier
        tx_cost = self.config.transaction_cost
        
        # Use Numba acceleration if available and data is large enough
        use_numba = (NUMBA_AVAILABLE and 
                    self.config.enable_numba_acceleration and 
                    callable(globals().get('_numba_triple_barrier_labels')) and 
                    n >= 512)
        
        if use_numba:
            tprint('⚡ Using Numba-accelerated triple barrier labeling')
            self.logger.info('⚡ Using Numba-accelerated triple barrier labeling')
            labels, profit_pcts, transaction_costs = _numba_triple_barrier_labels(
                close.astype(np.float64), 
                high.astype(np.float64), 
                low.astype(np.float64), 
                pt_mult, 
                sl_mult, 
                end_idx_arr.astype(np.int64),
                tx_cost
            )
        else:
            tprint('🐍 Using Python triple barrier labeling')
            self.logger.info('🐍 Using Python triple barrier labeling')
            labels, profit_pcts, transaction_costs = self._apply_barrier_logic_python(
                close, high, low, end_idx_arr, pt_mult, sl_mult, tx_cost
            )
            
        return labels, profit_pcts, transaction_costs
    
    def _apply_barrier_logic_python(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        end_idx_arr: np.ndarray,
        pt_mult: float,
        sl_mult: float,
        tx_cost: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply barrier logic in Python."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        transaction_costs = np.zeros(n, dtype=np.float64)
        
        for i in range(n - 1):
            entry_price = close[i]
            
            # Numerical stability check
            if entry_price <= EPSILON:
                labels[i] = 0
                # Time-barrier implied because we cannot trade; no transaction occurs
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
                continue
                
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i:
                labels[i] = 0
                # Time-barrier exit at the same timestamp: apply trade cost
                profit_pcts[i] = -tx_cost
                transaction_costs[i] = tx_cost
                continue
                
            # Get window data
            win_high = high[i + 1:end_idx]
            win_low = low[i + 1:end_idx]
            
            # Find barrier hits
            profit_hits = np.where(win_high >= profit_barrier)[0]
            stop_hits = np.where(win_low <= stop_barrier)[0]
            
            # Determine label and profit
            if profit_hits.size == 0 and stop_hits.size == 0:
                # No barriers hit - time barrier
                labels[i] = 0
                profit_pcts[i] = -tx_cost
                transaction_costs[i] = tx_cost
            elif profit_hits.size == 0:
                # Only stop loss hit
                labels[i] = -1
                profit_pcts[i] = -sl_mult - tx_cost
                transaction_costs[i] = tx_cost
            elif stop_hits.size == 0:
                # Only profit take hit
                labels[i] = 1
                profit_pcts[i] = pt_mult - tx_cost
                transaction_costs[i] = tx_cost
            else:
                # Both hit - use first one chronologically
                first_profit_hit = profit_hits[0]
                first_stop_hit = stop_hits[0]
                if first_profit_hit <= first_stop_hit:
                    labels[i] = 1
                    profit_pcts[i] = pt_mult - tx_cost
                    transaction_costs[i] = tx_cost
                else:
                    labels[i] = -1
                    profit_pcts[i] = -sl_mult - tx_cost
                    transaction_costs[i] = tx_cost
                    
        return labels, profit_pcts, transaction_costs
    
    def _post_process_results(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing and filtering."""
        try:
            tprint("🔧 Post-processing results...")
            original_count = len(labeled_data)
            
            # Filter out HOLD samples if binary classification
            if self.config.binary_classification:
                hold_samples = (labeled_data['label'] == 0).sum()
                labeled_data = labeled_data[labeled_data['label'] != 0].copy()
                tprint(f'📊 Filtered {hold_samples} HOLD samples for binary classification')
                self.logger.info(f'📊 Filtered {hold_samples} HOLD samples for binary classification')
            
            tprint(f"✅ Post-processing completed: {len(labeled_data)} rows remaining")
            return labeled_data
            
        except Exception as e:
            tprint(f"❌ Post-processing failed: {e}")
            self.logger.error(f"Post-processing failed: {e}")
            raise
    
    def _populate_result_metrics(self, result: TripleBarrierResult, final_data: pd.DataFrame):
        """Populate result metrics from execution."""
        result.output_data_shape = final_data.shape
        result.total_labels_generated = len(final_data[final_data['label'].notna()])
        
        # Label distribution
        if 'label' in final_data.columns:
            result.label_distribution = final_data['label'].value_counts().to_dict()
        
        # Performance metrics
        result.numba_acceleration_used = NUMBA_AVAILABLE and self.config.enable_numba_acceleration
        result.hardware_optimizations_used = list(self.hardware_manager.optimizations.keys())
        
        # Quality metrics
        result.data_quality_score = self._calculate_data_quality_score(final_data)
        result.label_quality_score = self._calculate_label_quality_score(final_data)
        
        # Barrier hit statistics
        result.barrier_hit_statistics = self._calculate_barrier_hit_statistics(final_data)

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        score = 1.0
        
        # Penalize missing data
        for col in ['close', 'high', 'low']:
            if col in data.columns:
                missing_ratio = data[col].isna().sum() / len(data)
                score -= missing_ratio * 0.5
        
        # Penalize invalid OHLC relationships
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = (data['high'] < np.maximum(data['open'], data['close'])).sum()
            invalid_low = (data['low'] > np.minimum(data['open'], data['close'])).sum()
            invalid_ratio = (invalid_high + invalid_low) / len(data)
            score -= invalid_ratio * 0.3
        
        return max(0.0, score)
    
    def _calculate_label_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate label quality score."""
        if 'label' not in data.columns:
            return 0.0
        
        labels = data['label'].dropna()
        if len(labels) == 0:
            return 0.0
        
        # Check label distribution balance
        label_counts = labels.value_counts()
        if len(label_counts) < 2:
            return 0.5  # Only one class
        
        # Calculate balance score
        max_count = label_counts.max()
        min_count = label_counts.min()
        balance_ratio = min_count / max_count
        balance_score = min(1.0, balance_ratio * 2)  # Scale to 0-1
        
        return balance_score
    
    def _calculate_barrier_hit_statistics(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate barrier hit statistics."""
        if 'label' not in data.columns:
            return {}
        
        labels = data['label'].dropna()
        return {
            'profit_take_hits': int((labels == 1).sum()),
            'stop_loss_hits': int((labels == -1).sum()),
            'time_barrier_hits': int((labels == 0).sum()),
            'total_hits': int(len(labels))
        }

# Convenience functions for easy integration
def create_triple_barrier_labeler(
    profit_take_multiplier: float = DEFAULT_PROFIT_TAKE_MULTIPLIER,
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    binary_classification: bool = True,
    regime_aware: bool = True,
    regime_column: str = 'hmm_regime',
    fail_on_validation_error: bool = True,
    fail_on_hardware_optimization_error: bool = False
) -> UnifiedTripleBarrierLabeler:
    """Create a triple barrier labeler with specified parameters."""
    config = TripleBarrierConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        binary_classification=binary_classification,
        regime_aware=regime_aware,
        regime_column=regime_column,
        fail_on_validation_error=fail_on_validation_error,
        fail_on_hardware_optimization_error=fail_on_hardware_optimization_error
    )
    
    return UnifiedTripleBarrierLabeler(config)

def apply_triple_barrier_labeling(
    data: pd.DataFrame,
    profit_take_multiplier: float = DEFAULT_PROFIT_TAKE_MULTIPLIER,
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    binary_classification: bool = True,
    regime_aware: bool = True,
    regime_column: str = 'hmm_regime'
) -> TripleBarrierResult:
    """Apply triple barrier labeling to data."""
    labeler = create_triple_barrier_labeler(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        binary_classification=binary_classification,
        regime_aware=regime_aware,
        regime_column=regime_column
    )
    
    return labeler.apply_labeling(data)

if __name__ == '__main__':
    # Test the implementation
    tprint('🧪 Testing Unified Triple Barrier Labeling')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'hmm_regime': np.random.choice([0, 1, 2], 1000)  # Add regime data
    }, index=dates)
    
    # Test unified labeling
    tprint('\n📊 Testing unified triple barrier labeling...')
    result = apply_triple_barrier_labeling(data)
    
    if result.success:
        tprint(f'✅ Labeling completed successfully')
        tprint(f'   Labels generated: {result.total_labels_generated}')
        tprint(f'   Label distribution: {result.label_distribution}')
        tprint(f'   Quality score: {result.data_quality_score:.2%}')
        tprint(f'   Execution time: {result.execution_duration:.2f}s')
    else:
        tprint(f'❌ Labeling failed: {result.error_message}')
    
    tprint('✅ Unified Triple Barrier Labeling test completed!')