"""
Enhanced Optimized Triple Barrier Labeler with Advanced Optimizations

This module integrates Optuna-based optimization with matrix operations,
coarse grid search, hardware acceleration, and comprehensive math validation
for maximum performance and accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations.hardware_integration import (
        HardwareOptimizedMatrixProcessor, MatrixOperationConfig
    )
    from src.utils.matrix_operations.vectorized_core import (
        VectorizedProcessor, BatchProcessor, ChunkedProcessor
    )
    from src.utils.matrix_operations.batch_operations import (
        BatchMatrixOperations, BatchConfig
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPERATIONS_AVAILABLE = False

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimization not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import math validation
try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, validate_positive, validate_range,
        validate_finite, validate_probability, safe_power
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Math validation not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

# Import existing optimization components
try:
    from src.feature_engineering.step06_labeling_components.regime_specific_triple_barrier_optimizer import (
        RegimeSpecificTripleBarrierOptimizer
    )
    from src.training.steps.market_analysis.regime_aware_triple_barrier_optimizer import (
        RegimeAwareTripleBarrierOptimizer,
        RegimeBarrierParams,
        RegimePerformanceMetrics
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Optimization components not available: {e}")
    OPTIMIZATION_AVAILABLE = False

# Import our core labeling components
from .core import TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
from .regime_aware import RegimeAwareTripleBarrierLabeler, RegimeAwareConfig
from .quality_assessment import LabelQualityAssessor, LabelQualityMetrics
from .utils import LabelingUtils

logger = logging.getLogger(__name__)

@dataclass
class CoarseGridConfig:
    """Configuration for coarse grid search (first stage)."""
    pt_mult_range: Tuple[float, float] = (0.0005, 0.02)
    sl_mult_range: Tuple[float, float] = (0.0005, 0.01)
    time_barrier_range: Tuple[int, int] = (15, 120)
    lookahead_range: Tuple[int, int] = (50, 300)
    grid_size: int = 15  # Number of points per dimension (coarse)
    top_k_candidates: int = 8  # Top candidates to pass to fine grid

@dataclass
class FineGridConfig:
    """Configuration for fine grid search (second stage)."""
    refinement_factor: float = 0.3  # How much to narrow the search space around coarse results
    grid_size: int = 10  # Number of points per dimension (fine)
    top_k_candidates: int = 3  # Top candidates to pass to Bayesian optimization
    min_range_size: float = 0.001  # Minimum range size to prevent over-narrowing

@dataclass
class BayesianConfig:
    """Configuration for Bayesian optimization (third stage)."""
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    early_stopping_patience: int = 20
    objective_function: str = "sharpe_ratio"  # or "profit_factor", "win_rate", "combined"
    acquisition_function: str = "EI"  # Expected Improvement
    random_state: Optional[int] = None
    
@dataclass
class HardwareOptimizationConfig:
    """Configuration for hardware optimizations."""
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    batch_size: int = 1000
    max_memory_usage: float = 0.8  # 80% of available memory
    parallel_workers: int = 4
    use_vectorized_operations: bool = True
    enable_chunked_processing: bool = True

@dataclass
class OptimizedBarrierParams:
    """Optimized barrier parameters for a specific regime."""
    regime_id: Union[int, str]
    regime_name: str
    pt_mult: float
    sl_mult: float
    time_barrier_minutes: int
    max_lookahead: int
    transaction_cost: float = 0.0008  # 0.08% fee per trade
    optimization_score: float = 0.0
    optimization_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_triple_barrier_config(self) -> TripleBarrierConfig:
        """Convert to TripleBarrierConfig."""
        return TripleBarrierConfig(
            pt_mult=self.pt_mult,
            sl_mult=self.sl_mult,
            min_holding_period=1,
            max_holding_period=self.max_lookahead,
            transaction_cost=self.transaction_cost
        )

@dataclass
class RegimeTradingMetrics:
    """Trading metrics for a specific regime."""
    regime_id: Union[int, str]
    regime_name: str
    total_trades: int
    trades_per_100_bars: float
    long_trades: int
    short_trades: int
    long_short_ratio: float
    win_rate: float
    avg_profit_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    avg_holding_period: float
    pt_mult: float
    sl_mult: float
    optimization_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime_id': self.regime_id,
            'regime_name': self.regime_name,
            'total_trades': self.total_trades,
            'trades_per_100_bars': self.trades_per_100_bars,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_short_ratio': self.long_short_ratio,
            'win_rate': self.win_rate,
            'avg_profit_pct': self.avg_profit_pct,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'profit_factor': self.profit_factor,
            'avg_holding_period': self.avg_holding_period,
            'pt_mult': self.pt_mult,
            'sl_mult': self.sl_mult,
            'optimization_score': self.optimization_score
        }

class EnhancedOptimizedTripleBarrierLabeler:
    """
    Enhanced optimized triple barrier labeler with advanced optimizations.
    
    This class integrates Optuna-based optimization with matrix operations,
    coarse grid search, hardware acceleration, and comprehensive math validation
    for maximum performance and accuracy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced optimized triple barrier labeler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize configuration
        self.coarse_grid_config = CoarseGridConfig()
        self.fine_grid_config = FineGridConfig()
        self.bayesian_config = BayesianConfig()
        self.hardware_config = HardwareOptimizationConfig()
        
        # Initialize components
        self.utils = LabelingUtils()
        self.quality_assessor = LabelQualityAssessor()
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        # Initialize matrix operations
        self._initialize_matrix_operations()
        
        # Initialize optimization components if available
        if OPTIMIZATION_AVAILABLE:
            self.regime_optimizer = RegimeAwareTripleBarrierOptimizer(config)
            self.regime_specific_optimizer = RegimeSpecificTripleBarrierOptimizer(config)
        else:
            self.regime_optimizer = None
            self.regime_specific_optimizer = None
            self.logger.warning("⚠️ Optimization components not available - using default parameters")
        
        # Storage for optimized parameters and metrics
        self.optimized_params: Dict[Union[int, str], OptimizedBarrierParams] = {}
        self.regime_metrics: Dict[Union[int, str], RegimeTradingMetrics] = {}
        self.optimization_results: Dict[str, Any] = {}
        self.coarse_grid_results: Dict[str, Any] = {}
        
        self._log_initialization()
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                if self.hardware_config.enable_gpu_acceleration:
                    self.gpu_manager = get_m1_gpu_manager()
                    if self.gpu_manager and self.gpu_manager.is_m1:
                        self.logger.info("✅ M1 GPU acceleration enabled")
                    else:
                        self.logger.warning("⚠️ M1 GPU not available")
                
                if self.hardware_config.enable_memory_optimization:
                    self.memory_optimizer = get_m1_memory_optimizer()
                    if self.memory_optimizer:
                        self.logger.info("✅ M1 Memory optimization enabled")
                
                if self.hardware_config.enable_cpu_optimization:
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    if self.cpu_optimizer:
                        self.logger.info("✅ M1 CPU optimization enabled")
                        
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
        else:
            self.logger.warning("⚠️ Hardware optimization not available")
    
    def _initialize_matrix_operations(self):
        """Initialize matrix operations components."""
        self.matrix_processor = None
        self.vectorized_processor = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                # Initialize matrix processor with hardware integration
                matrix_config = MatrixOperationConfig(
                    enable_gpu=self.hardware_config.enable_gpu_acceleration,
                    enable_memory_opt=self.hardware_config.enable_memory_optimization,
                    batch_size=self.hardware_config.batch_size
                )
                
                self.matrix_processor = HardwareOptimizedMatrixProcessor(matrix_config)
                self.vectorized_processor = VectorizedProcessor()
                self.batch_processor = BatchProcessor()
                
                self.logger.info("✅ Matrix operations initialized")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations initialization failed: {e}")
        else:
            self.logger.warning("⚠️ Matrix operations not available")
    
    def _log_initialization(self):
        """Log initialization parameters."""
        self.logger.info("🚀 Initializing Enhanced Optimized Triple Barrier Labeler")
        self.logger.info(f"📋 Optimization available: {OPTIMIZATION_AVAILABLE}")
        self.logger.info(f"🔧 Matrix operations: {MATRIX_OPERATIONS_AVAILABLE}")
        self.logger.info(f"⚡ Hardware optimization: {HARDWARE_OPTIMIZATION_AVAILABLE}")
        self.logger.info(f"🧮 Math validation: {MATH_VALIDATION_AVAILABLE}")
        
        if OPTIMIZATION_AVAILABLE:
            self.logger.info("✅ Optuna-based optimization integrated")
        else:
            self.logger.warning("⚠️ Using default parameters (no optimization)")
    
    def _coarse_grid_search(self, regime_data: pd.DataFrame, regime_name: str) -> List[Dict[str, Any]]:
        """Perform coarse grid search to find promising parameter combinations.
        
        Args:
            regime_data: Market data for the specific regime
            regime_name: Name of the regime
            
        Returns:
            List of top parameter combinations
        """
        self.logger.info(f"🔍 Starting coarse grid search for regime: {regime_name}")
        
        # Create parameter grids
        pt_mults = np.linspace(
            self.coarse_grid_config.pt_mult_range[0],
            self.coarse_grid_config.pt_mult_range[1],
            self.coarse_grid_config.grid_size
        )
        sl_mults = np.linspace(
            self.coarse_grid_config.sl_mult_range[0],
            self.coarse_grid_config.sl_mult_range[1],
            self.coarse_grid_config.grid_size
        )
        time_barriers = np.linspace(
            self.coarse_grid_config.time_barrier_range[0],
            self.coarse_grid_config.time_barrier_range[1],
            self.coarse_grid_config.grid_size,
            dtype=int
        )
        lookaheads = np.linspace(
            self.coarse_grid_config.lookahead_range[0],
            self.coarse_grid_config.lookahead_range[1],
            self.coarse_grid_config.grid_size,
            dtype=int
        )
        
        # Generate all combinations
        param_combinations = []
        for pt_mult in pt_mults:
            for sl_mult in sl_mults:
                for time_barrier in time_barriers:
                    for lookahead in lookaheads:
                        # Validate parameters
                        if MATH_VALIDATION_AVAILABLE:
                            if not (validate_positive(pt_mult) and validate_positive(sl_mult) and 
                                   validate_range(pt_mult, 0.0001, 0.1) and 
                                   validate_range(sl_mult, 0.0001, 0.1)):
                                continue
                        
                        param_combinations.append({
                            'pt_mult': pt_mult,
                            'sl_mult': sl_mult,
                            'time_barrier': int(time_barrier),
                            'lookahead': int(lookahead)
                        })
        
        self.logger.info(f"📊 Testing {len(param_combinations)} parameter combinations")
        
        # Evaluate combinations in parallel
        if self.hardware_config.parallel_workers > 1:
            candidates = self._evaluate_combinations_parallel(regime_data, param_combinations)
        else:
            candidates = self._evaluate_combinations_sequential(regime_data, param_combinations)
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:self.coarse_grid_config.top_k_candidates]
        
        self.logger.info(f"✅ Coarse grid search completed. Top score: {top_candidates[0]['score']:.4f}")
        
        return top_candidates
    
    def _fine_grid_search(self, regime_data: pd.DataFrame, regime_name: str, coarse_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform fine grid search around promising coarse grid results.
        
        Args:
            regime_data: Market data for the specific regime
            regime_name: Name of the regime
            coarse_candidates: Top candidates from coarse grid search
            
        Returns:
            List of top parameter combinations from fine grid
        """
        self.logger.info(f"🔍 Starting fine grid search for regime: {regime_name}")
        
        if not coarse_candidates:
            self.logger.warning(f"⚠️ No coarse candidates for regime {regime_name}")
            return []
        
        # Find the best coarse candidate to center the fine grid around
        best_coarse = max(coarse_candidates, key=lambda x: x['score'])
        
        # Create refined ranges around the best candidate
        pt_center = best_coarse['pt_mult']
        sl_center = best_coarse['sl_mult']
        time_center = best_coarse['time_barrier']
        lookahead_center = best_coarse['lookahead']
        
        # Calculate refined ranges
        pt_range = self._calculate_refined_range(
            pt_center, 
            self.coarse_grid_config.pt_mult_range,
            self.fine_grid_config.refinement_factor,
            self.fine_grid_config.min_range_size
        )
        
        sl_range = self._calculate_refined_range(
            sl_center,
            self.coarse_grid_config.sl_mult_range,
            self.fine_grid_config.refinement_factor,
            self.fine_grid_config.min_range_size
        )
        
        time_range = self._calculate_refined_range(
            time_center,
            self.coarse_grid_config.time_barrier_range,
            self.fine_grid_config.refinement_factor,
            max(1, self.fine_grid_config.min_range_size * 100)  # Minimum 1 for integers
        )
        
        lookahead_range = self._calculate_refined_range(
            lookahead_center,
            self.coarse_grid_config.lookahead_range,
            self.fine_grid_config.refinement_factor,
            max(1, self.fine_grid_config.min_range_size * 100)  # Minimum 1 for integers
        )
        
        self.logger.info(f"📊 Fine grid ranges:")
        self.logger.info(f"   PT: {pt_range[0]:.6f} - {pt_range[1]:.6f}")
        self.logger.info(f"   SL: {sl_range[0]:.6f} - {sl_range[1]:.6f}")
        self.logger.info(f"   Time: {time_range[0]} - {time_range[1]}")
        self.logger.info(f"   Lookahead: {lookahead_range[0]} - {lookahead_range[1]}")
        
        # Create fine parameter grids
        pt_mults = np.linspace(pt_range[0], pt_range[1], self.fine_grid_config.grid_size)
        sl_mults = np.linspace(sl_range[0], sl_range[1], self.fine_grid_config.grid_size)
        time_barriers = np.linspace(time_range[0], time_range[1], self.fine_grid_config.grid_size, dtype=int)
        lookaheads = np.linspace(lookahead_range[0], lookahead_range[1], self.fine_grid_config.grid_size, dtype=int)
        
        # Generate all combinations
        param_combinations = []
        for pt_mult in pt_mults:
            for sl_mult in sl_mults:
                for time_barrier in time_barriers:
                    for lookahead in lookaheads:
                        # Validate parameters
                        if MATH_VALIDATION_AVAILABLE:
                            if not (validate_positive(pt_mult) and validate_positive(sl_mult) and 
                                   validate_range(pt_mult, 0.0001, 0.1) and 
                                   validate_range(sl_mult, 0.0001, 0.1)):
                                continue
                        
                        param_combinations.append({
                            'pt_mult': pt_mult,
                            'sl_mult': sl_mult,
                            'time_barrier': int(time_barrier),
                            'lookahead': int(lookahead)
                        })
        
        self.logger.info(f"📊 Testing {len(param_combinations)} fine parameter combinations")
        
        # Evaluate combinations in parallel
        if self.hardware_config.parallel_workers > 1:
            candidates = self._evaluate_combinations_parallel(regime_data, param_combinations)
        else:
            candidates = self._evaluate_combinations_sequential(regime_data, param_combinations)
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:self.fine_grid_config.top_k_candidates]
        
        if top_candidates:
            self.logger.info(f"✅ Fine grid search completed. Top score: {top_candidates[0]['score']:.4f}")
        else:
            self.logger.warning(f"⚠️ No valid candidates found in fine grid search for {regime_name}")
        
        return top_candidates
    
    def _calculate_refined_range(self, center: float, original_range: Tuple[float, float], 
                                refinement_factor: float, min_range_size: float) -> Tuple[float, float]:
        """Calculate refined range around a center point."""
        original_size = original_range[1] - original_range[0]
        refined_size = max(original_size * refinement_factor, min_range_size)
        
        new_min = max(center - refined_size / 2, original_range[0])
        new_max = min(center + refined_size / 2, original_range[1])
        
        # Ensure we don't exceed original bounds
        if new_max - new_min < min_range_size:
            # If range is too small, expand around center
            new_min = max(center - min_range_size / 2, original_range[0])
            new_max = min(center + min_range_size / 2, original_range[1])
        
        return (new_min, new_max)
    
    def _evaluate_combinations_parallel(self, regime_data: pd.DataFrame, param_combinations: List[Dict]) -> List[Dict]:
        """Evaluate parameter combinations in parallel."""
        candidates = []
        
        with ThreadPoolExecutor(max_workers=self.hardware_config.parallel_workers) as executor:
            futures = []
            for params in param_combinations:
                future = executor.submit(self._evaluate_single_combination, regime_data, params)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        candidates.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ Parameter evaluation failed: {e}")
        
        return candidates
    
    def _bayesian_optimization(self, regime_data: pd.DataFrame, regime_name: str, fine_candidates: List[Dict[str, Any]]) -> Optional[OptimizedBarrierParams]:
        """Perform Bayesian optimization using Optuna on fine grid candidates.
        
        Args:
            regime_data: Market data for the specific regime
            regime_name: Name of the regime
            fine_candidates: Top candidates from fine grid search
            
        Returns:
            Optimized barrier parameters or None if optimization fails
        """
        self.logger.info(f"🔍 Starting Bayesian optimization for regime: {regime_name}")
        
        if not OPTIMIZATION_AVAILABLE:
            self.logger.warning("⚠️ Optuna not available - using best fine grid candidate")
            if fine_candidates:
                best_candidate = max(fine_candidates, key=lambda x: x['score'])
                return OptimizedBarrierParams(
                    regime_id=regime_name,
                    regime_name=regime_name,
                    pt_mult=best_candidate['pt_mult'],
                    sl_mult=best_candidate['sl_mult'],
                    time_barrier_minutes=best_candidate['time_barrier'],
                    max_lookahead=best_candidate['lookahead'],
                    transaction_cost=0.0008,
                    optimization_score=best_candidate['score']
                )
            return None
        
        if not fine_candidates:
            self.logger.warning(f"⚠️ No fine candidates for regime {regime_name}")
            return None
        
        try:
            import optuna
            
            # Create study
            study_name = f"triple_barrier_{regime_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    seed=self.bayesian_config.random_state
                )
            )
            
            # Define objective function
            def objective(trial):
                # Get parameter ranges from fine candidates
                best_fine = max(fine_candidates, key=lambda x: x['score'])
                
                # Create refined ranges around best fine candidate
                pt_range = self._calculate_refined_range(
                    best_fine['pt_mult'],
                    self.coarse_grid_config.pt_mult_range,
                    self.fine_grid_config.refinement_factor * 0.5,  # Even more refined
                    self.fine_grid_config.min_range_size * 0.5
                )
                
                sl_range = self._calculate_refined_range(
                    best_fine['sl_mult'],
                    self.coarse_grid_config.sl_mult_range,
                    self.fine_grid_config.refinement_factor * 0.5,
                    self.fine_grid_config.min_range_size * 0.5
                )
                
                time_range = self._calculate_refined_range(
                    best_fine['time_barrier'],
                    self.coarse_grid_config.time_barrier_range,
                    self.fine_grid_config.refinement_factor * 0.5,
                    max(1, self.fine_grid_config.min_range_size * 50)
                )
                
                lookahead_range = self._calculate_refined_range(
                    best_fine['lookahead'],
                    self.coarse_grid_config.lookahead_range,
                    self.fine_grid_config.refinement_factor * 0.5,
                    max(1, self.fine_grid_config.min_range_size * 50)
                )
                
                # Suggest parameters
                pt_mult = trial.suggest_float('pt_mult', pt_range[0], pt_range[1])
                sl_mult = trial.suggest_float('sl_mult', sl_range[0], sl_range[1])
                time_barrier = trial.suggest_int('time_barrier', int(time_range[0]), int(time_range[1]))
                lookahead = trial.suggest_int('lookahead', int(lookahead_range[0]), int(lookahead_range[1]))
                
                # Create config
                config = TripleBarrierConfig(
                    pt_mult=pt_mult,
                    sl_mult=sl_mult,
                    min_holding_period=1,
                    max_holding_period=lookahead,
                    transaction_cost=0.0008
                )
                
                # Evaluate parameters
                score = self._evaluate_parameters_enhanced(regime_data, config)
                
                # Add intermediate values for pruning
                trial.set_user_attr('pt_mult', pt_mult)
                trial.set_user_attr('sl_mult', sl_mult)
                trial.set_user_attr('time_barrier', time_barrier)
                trial.set_user_attr('lookahead', lookahead)
                
                return score
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.bayesian_config.n_trials,
                timeout=self.bayesian_config.timeout,
                show_progress_bar=False
            )
            
            # Get best parameters
            best_trial = study.best_trial
            best_params = best_trial.params
            
            # Create optimized parameters object
            optimized_params = OptimizedBarrierParams(
                regime_id=regime_name,
                regime_name=regime_name,
                pt_mult=best_params['pt_mult'],
                sl_mult=best_params['sl_mult'],
                time_barrier_minutes=best_params['time_barrier'],
                max_lookahead=best_params['lookahead'],
                transaction_cost=0.0008,
                optimization_score=best_trial.value
            )
            
            self.logger.info(f"✅ Bayesian optimization completed for {regime_name}")
            self.logger.info(f"   Best score: {best_trial.value:.4f}")
            self.logger.info(f"   Best params: PT={best_params['pt_mult']:.6f}, SL={best_params['sl_mult']:.6f}")
            self.logger.info(f"   Trials: {len(study.trials)}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed for {regime_name}: {e}")
            # Fallback to best fine candidate
            if fine_candidates:
                best_candidate = max(fine_candidates, key=lambda x: x['score'])
                return OptimizedBarrierParams(
                    regime_id=regime_name,
                    regime_name=regime_name,
                    pt_mult=best_candidate['pt_mult'],
                    sl_mult=best_candidate['sl_mult'],
                    time_barrier_minutes=best_candidate['time_barrier'],
                    max_lookahead=best_candidate['lookahead'],
                    transaction_cost=0.0008,
                    optimization_score=best_candidate['score']
                )
            return None
    
    def _evaluate_combinations_sequential(self, regime_data: pd.DataFrame, param_combinations: List[Dict]) -> List[Dict]:
        """Evaluate parameter combinations sequentially."""
        candidates = []
        
        for params in param_combinations:
            try:
                result = self._evaluate_single_combination(regime_data, params)
                if result is not None:
                    candidates.append(result)
            except Exception as e:
                self.logger.warning(f"⚠️ Parameter evaluation failed: {e}")
        
        return candidates
    
    def _evaluate_single_combination(self, regime_data: pd.DataFrame, params: Dict) -> Optional[Dict]:
        """Evaluate a single parameter combination."""
        try:
            # Create config
            config = TripleBarrierConfig(
                pt_mult=params['pt_mult'],
                sl_mult=params['sl_mult'],
                min_holding_period=1,
                max_holding_period=params['lookahead'],
                transaction_cost=0.0008
            )
            
            # Evaluate parameters
            score = self._evaluate_parameters_enhanced(regime_data, config)
            
            if MATH_VALIDATION_AVAILABLE:
                if not validate_finite(score):
                    return None
            
            return {
                'pt_mult': params['pt_mult'],
                'sl_mult': params['sl_mult'],
                'time_barrier': params['time_barrier'],
                'lookahead': params['lookahead'],
                'score': score
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Single combination evaluation failed: {e}")
            return None
    
    def _evaluate_parameters_enhanced(self, data: pd.DataFrame, config: TripleBarrierConfig) -> float:
        """Enhanced parameter evaluation with matrix operations and hardware acceleration."""
        try:
            # Use vectorized operations if available
            if self.vectorized_processor and self.hardware_config.use_vectorized_operations:
                return self._evaluate_parameters_vectorized(data, config)
            else:
                return self._evaluate_parameters_standard(data, config)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced parameter evaluation failed: {e}")
            return self._evaluate_parameters_standard(data, config)
    
    def _evaluate_parameters_vectorized(self, data: pd.DataFrame, config: TripleBarrierConfig) -> float:
        """Vectorized parameter evaluation using matrix operations."""
        try:
            # Convert to numpy arrays for vectorized operations
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Use matrix operations for barrier calculations
            if self.matrix_processor:
                # Calculate barrier prices vectorized
                pt_prices = close * (1 + config.pt_mult)
                sl_prices = close * (1 - config.sl_mult)
                
                # Use matrix operations for barrier hit detection
                barrier_hits = self.matrix_processor.vectorized_barrier_detection(
                    close, high, low, pt_prices, sl_prices, config.max_holding_period
                )
            else:
                # Fallback to standard evaluation
                return self._evaluate_parameters_standard(data, config)
            
            # Calculate metrics vectorized
            if len(barrier_hits) == 0:
                return -np.inf
            
            # Extract labels and profits
            labels = barrier_hits['labels']
            profits = barrier_hits['profits']
            
            # Calculate Sharpe ratio with math validation
            if MATH_VALIDATION_AVAILABLE:
                mean_profit = validate_finite(profits.mean())
                std_profit = validate_finite(profits.std())
                if std_profit > 0:
                    sharpe_ratio = safe_divide(mean_profit, std_profit) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            else:
                if profits.std() > 0:
                    sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            
            # Calculate win rate
            win_rate = (labels > 0).mean()
            
            # Calculate profit factor with math validation
            if MATH_VALIDATION_AVAILABLE:
                gross_profit = profits[profits > 0].sum()
                gross_loss = abs(profits[profits < 0].sum())
                profit_factor = safe_divide(gross_profit, gross_loss) if gross_loss > 0 else 0
            else:
                gross_profit = profits[profits > 0].sum()
                gross_loss = abs(profits[profits < 0].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Combined score with math validation
            if MATH_VALIDATION_AVAILABLE:
                score = (
                    validate_finite(sharpe_ratio) * 0.4 +
                    validate_probability(win_rate) * 0.3 +
                    validate_finite(min(profit_factor, 5.0)) * 0.2 +
                    validate_finite(min(len(labels) / 100, 1.0)) * 0.1
                )
            else:
                score = (
                    sharpe_ratio * 0.4 +
                    win_rate * 0.3 +
                    min(profit_factor, 5.0) * 0.2 +
                    min(len(labels) / 100, 1.0) * 0.1
                )
            
            return validate_finite(score) if MATH_VALIDATION_AVAILABLE else score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized evaluation failed: {e}")
            return self._evaluate_parameters_standard(data, config)
    
    def _evaluate_parameters_standard(self, data: pd.DataFrame, config: TripleBarrierConfig) -> float:
        """Standard parameter evaluation (fallback)."""
        try:
            # Create labeler and generate labels
            labeler = TripleBarrierLabeler(config)
            result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
            
            if result is None or 'label' not in result.columns:
                return -np.inf
            
            # Calculate metrics
            labels = result['label'].dropna()
            profits = result['profit_pct'].dropna()
            
            if len(labels) == 0 or len(profits) == 0:
                return -np.inf
            
            # Calculate Sharpe ratio
            if profits.std() > 0:
                sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate win rate
            win_rate = (labels > 0).mean()
            
            # Calculate profit factor
            gross_profit = profits[profits > 0].sum()
            gross_loss = abs(profits[profits < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Combined score
            score = (
                sharpe_ratio * 0.4 +
                win_rate * 0.3 +
                min(profit_factor, 5.0) * 0.2 +
                min(len(labels) / 100, 1.0) * 0.1
            )
            
            return score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Standard parameter evaluation failed: {e}")
            return -np.inf
    
    def optimize_regime_parameters(
        self, 
        data: pd.DataFrame, 
        regime_data: Optional[pd.DataFrame] = None,
        n_trials: int = None
    ) -> Dict[str, Any]:
        """Optimize triple barrier parameters using three-stage optimization.
        
        Args:
            data: Market data with OHLC columns
            regime_data: HMM regime data (if None, will be detected)
            n_trials: Number of Bayesian trials (overrides bayesian_config.n_trials)
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTIMIZATION_AVAILABLE:
            self.logger.warning("⚠️ Optimization not available - using default parameters")
            return self._create_default_parameters(data, regime_data)
        
        # Override Bayesian trials if provided
        if n_trials is not None:
            self.bayesian_config.n_trials = n_trials
        
        self.logger.info(f"🔧 Starting three-stage regime parameter optimization")
        self.logger.info(f"   Coarse grid: {self.coarse_grid_config.grid_size}³ combinations")
        self.logger.info(f"   Fine grid: {self.fine_grid_config.grid_size}³ combinations")
        self.logger.info(f"   Bayesian: {self.bayesian_config.n_trials} trials")
        
        start_time = time.time()
        
        try:
            # Prepare regime data
            if regime_data is None:
                regime_data = self._prepare_regime_data(data)
            
            # Get unique regimes
            unique_regimes = regime_data['regime'].unique()
            self.logger.info(f"📊 Found {len(unique_regimes)} regimes: {unique_regimes}")
            
            # Storage for results from each stage
            coarse_results = {}
            fine_results = {}
            optimization_results = {}
            
            # Optimize parameters for each regime using three-stage process
            for regime in unique_regimes:
                self.logger.info(f"\n🎯 Optimizing parameters for regime: {regime}")
                
                # Get regime-specific data
                regime_mask = regime_data['regime'] == regime
                regime_data_subset = data[regime_mask].copy()
                
                if len(regime_data_subset) < 100:
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime} ({len(regime_data_subset)} samples)")
                    continue
                
                # Stage 1: Coarse Grid Search
                self.logger.info(f"📊 Stage 1: Coarse Grid Search")
                coarse_start = time.time()
                coarse_candidates = self._coarse_grid_search(regime_data_subset, regime)
                coarse_time = time.time() - coarse_start
                coarse_results[regime] = {
                    'candidates': coarse_candidates,
                    'time': coarse_time
                }
                
                if not coarse_candidates:
                    self.logger.warning(f"⚠️ No coarse candidates for regime {regime}")
                    continue
                
                # Stage 2: Fine Grid Search
                self.logger.info(f"📊 Stage 2: Fine Grid Search")
                fine_start = time.time()
                fine_candidates = self._fine_grid_search(regime_data_subset, regime, coarse_candidates)
                fine_time = time.time() - fine_start
                fine_results[regime] = {
                    'candidates': fine_candidates,
                    'time': fine_time
                }
                
                if not fine_candidates:
                    self.logger.warning(f"⚠️ No fine candidates for regime {regime}")
                    # Use best coarse candidate
                    best_coarse = max(coarse_candidates, key=lambda x: x['score'])
                    regime_params = OptimizedBarrierParams(
                        regime_id=regime,
                        regime_name=regime,
                        pt_mult=best_coarse['pt_mult'],
                        sl_mult=best_coarse['sl_mult'],
                        time_barrier_minutes=best_coarse['time_barrier'],
                        max_lookahead=best_coarse['lookahead'],
                        transaction_cost=0.0008,
                        optimization_score=best_coarse['score']
                    )
                else:
                    # Stage 3: Bayesian Optimization
                    self.logger.info(f"📊 Stage 3: Bayesian Optimization")
                    bayesian_start = time.time()
                    regime_params = self._bayesian_optimization(regime_data_subset, regime, fine_candidates)
                    bayesian_time = time.time() - bayesian_start
                    
                    if regime_params is None:
                        # Fallback to best fine candidate
                        best_fine = max(fine_candidates, key=lambda x: x['score'])
                        regime_params = OptimizedBarrierParams(
                            regime_id=regime,
                            regime_name=regime,
                            pt_mult=best_fine['pt_mult'],
                            sl_mult=best_fine['sl_mult'],
                            time_barrier_minutes=best_fine['time_barrier'],
                            max_lookahead=best_fine['lookahead'],
                            transaction_cost=0.0008,
                            optimization_score=best_fine['score']
                        )
                    
                    fine_results[regime]['bayesian_time'] = bayesian_time
                
                if regime_params:
                    self.optimized_params[regime] = regime_params
                    optimization_results[regime] = regime_params.to_dict()
                    
                    # Log timing for this regime
                    total_regime_time = coarse_time + fine_time + fine_results[regime].get('bayesian_time', 0)
                    self.logger.info(f"✅ Regime {regime} completed in {total_regime_time:.2f}s")
                    self.logger.info(f"   Coarse: {coarse_time:.2f}s, Fine: {fine_time:.2f}s, Bayesian: {fine_results[regime].get('bayesian_time', 0):.2f}s")
            
            # Calculate regime metrics
            self._calculate_regime_metrics(data, regime_data)
            
            total_time = time.time() - start_time
            self.logger.info(f"\n✅ Three-stage optimization completed in {total_time:.2f}s")
            
            # Calculate timing breakdown
            total_coarse_time = sum(r['time'] for r in coarse_results.values())
            total_fine_time = sum(r['time'] for r in fine_results.values())
            total_bayesian_time = sum(r.get('bayesian_time', 0) for r in fine_results.values())
            
            self.optimization_results = {
                'optimization_time': total_time,
                'coarse_time': total_coarse_time,
                'fine_time': total_fine_time,
                'bayesian_time': total_bayesian_time,
                'n_trials': self.bayesian_config.n_trials,
                'regimes_optimized': len(optimization_results),
                'regime_parameters': optimization_results,
                'regime_metrics': {k: v.to_dict() for k, v in self.regime_metrics.items()},
                'stage_results': {
                    'coarse': coarse_results,
                    'fine': fine_results
                }
            }
            
            return self.optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Three-stage optimization failed: {e}")
            return self._create_default_parameters(data, regime_data)
    
    def _optimize_single_regime(
        self, 
        regime_data: pd.DataFrame, 
        regime_name: str, 
        n_trials: int
    ) -> Optional[OptimizedBarrierParams]:
        """Optimize parameters for a single regime."""
        try:
            # Define optimization ranges
            pt_range = (0.0005, 0.02)  # 0.05% to 2%
            sl_range = (0.0005, 0.01)  # 0.05% to 1%
            time_range = (15, 120)     # 15 to 120 minutes
            lookahead_range = (50, 300) # 50 to 300 points
            
            best_score = -np.inf
            best_params = None
            
            # Simple grid search optimization (can be replaced with Optuna)
            for trial in range(n_trials):
                # Sample parameters
                pt_mult = np.random.uniform(*pt_range)
                sl_mult = np.random.uniform(*sl_range)
                time_barrier = int(np.random.uniform(*time_range))
                max_lookahead = int(np.random.uniform(*lookahead_range))
                
                # Create config
                config = TripleBarrierConfig(
                    pt_mult=pt_mult,
                    sl_mult=sl_mult,
                    min_holding_period=1,
                    max_holding_period=max_lookahead,
                    transaction_cost=0.0008
                )
                
                # Test parameters
                score = self._evaluate_parameters(regime_data, config)
                
                if score > best_score:
                    best_score = score
                    best_params = OptimizedBarrierParams(
                        regime_id=regime_name,
                        regime_name=regime_name,
                        pt_mult=pt_mult,
                        sl_mult=sl_mult,
                        time_barrier_minutes=time_barrier,
                        max_lookahead=max_lookahead,
                        transaction_cost=0.0008,
                        optimization_score=score
                    )
            
            if best_params:
                self.logger.info(f"✅ Optimized {regime_name}: PT={best_params.pt_mult:.4f}, SL={best_params.sl_mult:.4f}, Score={best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize regime {regime_name}: {e}")
            return None
    
    def _evaluate_parameters(self, data: pd.DataFrame, config: TripleBarrierConfig) -> float:
        """Evaluate parameters using Sharpe ratio and other metrics."""
        try:
            # Create labeler and generate labels
            labeler = TripleBarrierLabeler(config)
            result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
            
            if result is None or 'label' not in result.columns:
                return -np.inf
            
            # Calculate metrics
            labels = result['label'].dropna()
            profits = result['profit_pct'].dropna()
            
            if len(labels) == 0 or len(profits) == 0:
                return -np.inf
            
            # Calculate Sharpe ratio
            if profits.std() > 0:
                sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate win rate
            win_rate = (labels > 0).mean()
            
            # Calculate profit factor
            gross_profit = profits[profits > 0].sum()
            gross_loss = abs(profits[profits < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Combined score (weighted)
            score = (
                sharpe_ratio * 0.4 +
                win_rate * 0.3 +
                min(profit_factor, 5.0) * 0.2 +  # Cap profit factor
                min(len(labels) / 100, 1.0) * 0.1  # Sample size bonus
            )
            
            return score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter evaluation failed: {e}")
            return -np.inf
    
    def _prepare_regime_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare regime data from HMM states."""
        # Check for existing HMM regime columns
        hmm_columns = ['hmm_regime', 'composite_cluster_id', 'regime']
        existing_regime_col = None
        
        for col in hmm_columns:
            if col in data.columns:
                existing_regime_col = col
                break
        
        if existing_regime_col:
            return pd.DataFrame({'regime': data[existing_regime_col]}, index=data.index)
        
        # Create default regimes if none found
        self.logger.warning("⚠️ No HMM regime data found - creating default regimes")
        regimes = ['bull' if i % 200 < 100 else 'bear' for i in range(len(data))]
        return pd.DataFrame({'regime': regimes}, index=data.index)
    
    def _create_default_parameters(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Create default parameters when optimization is not available."""
        if regime_data is None:
            regime_data = self._prepare_regime_data(data)
        
        unique_regimes = regime_data['regime'].unique()
        default_params = {}
        
        for regime in unique_regimes:
            # Default parameters based on regime type
            if 'bull' in str(regime).lower():
                pt_mult, sl_mult = 0.015, 0.007
            elif 'bear' in str(regime).lower():
                pt_mult, sl_mult = 0.008, 0.004
            else:
                pt_mult, sl_mult = 0.01, 0.005
            
            default_params[regime] = {
                'regime_id': regime,
                'regime_name': str(regime),
                'pt_mult': pt_mult,
                'sl_mult': sl_mult,
                'time_barrier_minutes': 30,
                'max_lookahead': 100,
                'transaction_cost': 0.0008,
                'optimization_score': 0.0
            }
        
        return {
            'optimization_time': 0.0,
            'n_trials': 0,
            'regimes_optimized': len(default_params),
            'regime_parameters': default_params,
            'regime_metrics': {}
        }
    
    def _calculate_regime_metrics(self, data: pd.DataFrame, regime_data: pd.DataFrame):
        """Calculate comprehensive metrics for each regime."""
        unique_regimes = regime_data['regime'].unique()
        
        for regime in unique_regimes:
            if regime not in self.optimized_params:
                continue
            
            regime_mask = regime_data['regime'] == regime
            regime_data_subset = data[regime_mask].copy()
            
            if len(regime_data_subset) < 10:
                continue
            
            # Get optimized parameters
            params = self.optimized_params[regime]
            config = params.to_triple_barrier_config()
            
            # Generate labels
            labeler = TripleBarrierLabeler(config)
            result = labeler.create_labels(regime_data_subset, method=LabelingMethod.TRIPLE_BARRIER)
            
            if result is None or 'label' not in result.columns:
                continue
            
            # Calculate metrics
            labels = result['label'].dropna()
            profits = result['profit_pct'].dropna()
            
            if len(labels) == 0:
                continue
            
            # Basic metrics
            total_trades = len(labels)
            trades_per_100_bars = (total_trades / len(regime_data_subset)) * 100
            
            long_trades = (labels > 0).sum()
            short_trades = (labels < 0).sum()
            long_short_ratio = long_trades / short_trades if short_trades > 0 else np.inf
            
            win_rate = (labels > 0).mean()
            avg_profit_pct = profits.mean()
            total_return_pct = profits.sum()
            
            # Risk metrics
            if profits.std() > 0:
                sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Drawdown calculation
            cumulative_returns = (1 + profits).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown_pct = abs(drawdown.min())
            
            # Profit factor
            gross_profit = profits[profits > 0].sum()
            gross_loss = abs(profits[profits < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Average holding period (simplified)
            avg_holding_period = len(regime_data_subset) / total_trades if total_trades > 0 else 0
            
            # Create metrics object
            metrics = RegimeTradingMetrics(
                regime_id=regime,
                regime_name=str(regime),
                total_trades=total_trades,
                trades_per_100_bars=trades_per_100_bars,
                long_trades=long_trades,
                short_trades=short_trades,
                long_short_ratio=long_short_ratio,
                win_rate=win_rate,
                avg_profit_pct=avg_profit_pct,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_pct=max_drawdown_pct,
                profit_factor=profit_factor,
                avg_holding_period=avg_holding_period,
                pt_mult=params.pt_mult,
                sl_mult=params.sl_mult,
                optimization_score=params.optimization_score
            )
            
            self.regime_metrics[regime] = metrics
    
    def create_optimized_labels(
        self, 
        data: pd.DataFrame, 
        regime_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Create labels using optimized parameters for each regime.
        
        Args:
            data: Market data with OHLC columns
            regime_data: HMM regime data
            
        Returns:
            DataFrame with optimized labels
        """
        if not self.optimized_params:
            self.logger.warning("⚠️ No optimized parameters found - running optimization first")
            self.optimize_regime_parameters(data, regime_data)
        
        if regime_data is None:
            regime_data = self._prepare_regime_data(data)
        
        # Create regime-aware labeler with optimized parameters
        regime_params = {}
        for regime, params in self.optimized_params.items():
            regime_params[regime] = params.to_triple_barrier_config()
        
        regime_config = RegimeAwareConfig(
            regime_detection_method="hmm",
            regime_params=regime_params
        )
        
        # Create labels
        regime_labeler = RegimeAwareTripleBarrierLabeler(regime_config=regime_config)
        labels = regime_labeler.create_regime_aware_labels(data, regime_data)
        
        return labels
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'optimization_summary': self.optimization_results,
            'regime_parameters': {k: v.to_dict() for k, v in self.optimized_params.items()},
            'regime_metrics': {k: v.to_dict() for k, v in self.regime_metrics.items()},
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def print_optimization_report(self):
        """Print a formatted optimization report."""
        print("\n" + "="*80)
        print("🎯 OPTIMIZED TRIPLE BARRIER LABELING REPORT")
        print("="*80)
        
        if not self.optimized_params:
            print("⚠️ No optimization results available")
            return
        
        print(f"\n📊 THREE-STAGE OPTIMIZATION SUMMARY")
        print(f"   Regimes optimized: {len(self.optimized_params)}")
        print(f"   Total time: {self.optimization_results.get('optimization_time', 0):.2f}s")
        print(f"   Coarse grid time: {self.optimization_results.get('coarse_time', 0):.2f}s")
        print(f"   Fine grid time: {self.optimization_results.get('fine_time', 0):.2f}s")
        print(f"   Bayesian time: {self.optimization_results.get('bayesian_time', 0):.2f}s")
        print(f"   Bayesian trials: {self.optimization_results.get('n_trials', 0)}")
        
        print(f"\n🎯 REGIME PARAMETERS")
        for regime, params in self.optimized_params.items():
            print(f"\n   {regime.upper()}:")
            print(f"      Profit Target: {params.pt_mult:.4f} ({params.pt_mult*100:.2f}%)")
            print(f"      Stop Loss: {params.sl_mult:.4f} ({params.sl_mult*100:.2f}%)")
            print(f"      Time Barrier: {params.time_barrier_minutes} minutes")
            print(f"      Max Lookahead: {params.max_lookahead} bars")
            print(f"      Transaction Cost: {params.transaction_cost:.4f} ({params.transaction_cost*100:.2f}%)")
            print(f"      Optimization Score: {params.optimization_score:.4f}")
        
        if self.regime_metrics:
            print(f"\n📈 REGIME TRADING METRICS")
            for regime, metrics in self.regime_metrics.items():
                print(f"\n   {regime.upper()}:")
                print(f"      Total Trades: {metrics.total_trades}")
                print(f"      Trades per 100 bars: {metrics.trades_per_100_bars:.2f}")
                print(f"      Long/Short Ratio: {metrics.long_short_ratio:.2f}")
                print(f"      Win Rate: {metrics.win_rate:.2%}")
                print(f"      Avg Profit: {metrics.avg_profit_pct:.4f} ({metrics.avg_profit_pct*100:.2f}%)")
                print(f"      Total Return: {metrics.total_return_pct:.4f} ({metrics.total_return_pct*100:.2f}%)")
                print(f"      Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
                print(f"      Max Drawdown: {metrics.max_drawdown_pct:.4f} ({metrics.max_drawdown_pct*100:.2f}%)")
                print(f"      Profit Factor: {metrics.profit_factor:.4f}")
                print(f"      Avg Holding Period: {metrics.avg_holding_period:.1f} bars")
        
        print("\n" + "="*80)

# Backward compatibility alias
OptimizedTripleBarrierLabeler = EnhancedOptimizedTripleBarrierLabeler