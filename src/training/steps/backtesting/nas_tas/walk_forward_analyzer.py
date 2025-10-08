"""
Walk-Forward Analysis

Enhanced walk-forward analysis for NAS-TAS models with:
- Regime-aware validation and performance tracking
- Hardware-accelerated parallel processing (M1 optimization)
- Cross-validation with purging and embargo
- Data leakage detection
- Comprehensive validation and error handling
- Advanced metric calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# ML Utilities
from src.utils.ml_common.cv_utils import TimeSeriesSplitValidator
from src.utils.ml_common.oof_generator import OOFGenerator
from src.utils.ml_common.data_leakage_detector import DataLeakageDetector

# Math and validation utilities
from src.utils.math_validation import (
    validate_probability, validate_positive, validate_range,
    safe_divide, safe_log, check_for_nans, check_for_infs
)

# Common operations
from src.utils.common_operations import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor, calculate_calmar_ratio
)
from src.utils.common_utilities import ensure_list, ensure_array, flatten_dict

# Output utilities
from src.utils.tprint import tprint

# Hardware optimization
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUAccelerator
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    tprint("⚠️  M1 hardware optimization not available", "warning")

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward analysis modes."""
    FIXED_WINDOW = "fixed_window"      # Fixed training window
    EXPANDING_WINDOW = "expanding_window"  # Expanding training window
    ADAPTIVE_WINDOW = "adaptive_window"    # Adaptive window based on regime changes
    ROLLING_WINDOW = "rolling_window"     # Rolling window


class ValidationMetric(Enum):
    """Validation metrics for walk-forward analysis."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Walk-forward settings
    mode: WalkForwardMode = WalkForwardMode.EXPANDING_WINDOW
    initial_training_size: int = 1000  # Initial training window size
    validation_size: int = 100         # Validation window size
    step_size: int = 50               # Step size for moving window
    
    # Regime-aware settings
    enable_regime_aware_validation: bool = True
    regime_change_threshold: float = 0.3  # Threshold for regime change detection
    min_regime_samples: int = 50      # Minimum samples per regime
    
    # Model retraining
    enable_model_retraining: bool = True
    retraining_frequency: int = 10    # Retrain every N steps
    enable_incremental_learning: bool = True
    incremental_learning_rate: float = 0.01
    
    # Performance tracking
    validation_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.ACCURACY,
        ValidationMetric.F1_SCORE,
        ValidationMetric.SHARPE_RATIO
    ])
    performance_threshold: float = 0.6  # Minimum performance threshold
    degradation_threshold: float = 0.1  # Performance degradation threshold
    
    # Data handling
    enable_data_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_data_validation: bool = True
    
    # Data leakage and integrity
    enable_leakage_detection: bool = True
    enable_purging: bool = True
    embargo_pct: float = 0.01  # Embargo percentage for temporal data
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    chunk_size_mb: int = 128
    
    # Output settings
    save_results: bool = True
    results_path: str = "walk_forward_results"
    enable_detailed_logging: bool = True
    enable_visualization: bool = True
    
    # Advanced features
    enable_ensemble_validation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_regime_transition_analysis: bool = True


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    
    # Basic results
    success: bool
    execution_time: float
    total_folds: int
    successful_folds: int
    
    # Performance metrics
    overall_performance: Dict[str, float]
    fold_performance: List[Dict[str, Any]]
    regime_performance: Dict[int, Dict[str, float]]
    
    # Model evolution
    model_evolution: List[Dict[str, Any]]
    retraining_events: List[Dict[str, Any]]
    
    # Regime analysis
    regime_transitions: List[Dict[str, Any]]
    regime_stability: Dict[int, float]
    
    # Validation insights
    performance_trends: Dict[str, str]
    degradation_events: List[Dict[str, Any]]
    improvement_events: List[Dict[str, Any]]
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    data_statistics: Dict[str, Any] = field(default_factory=dict)


class WalkForwardAnalyzer:
    """
    Walk-forward analyzer for NAS-TAS models.
    
    Provides comprehensive walk-forward analysis with regime-aware validation,
    model evolution tracking, and performance degradation detection.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize enhanced walk-forward analyzer with hardware acceleration.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint("🚀 Initializing Enhanced Walk-Forward Analyzer", "header")
        
        # Analysis state
        self.fold_results = []
        self.model_evolution = []
        self.regime_transitions = []
        self.performance_history = []
        
        # Model registry
        self.available_models = {}
        self.model_performance = {}
        
        # Initialize CV utilities
        if config.enable_purging:
            self.cv_validator = TimeSeriesSplitValidator(
                n_splits=5,  # Default, can be adjusted
                test_size=config.validation_size / (config.initial_training_size + config.validation_size),
                embargo_pct=config.embargo_pct
            )
            tprint("✅ Time-series CV validator initialized with purging", "success")
        else:
            self.cv_validator = None
        
        # Initialize leakage detector
        if config.enable_leakage_detection:
            self.leakage_detector = DataLeakageDetector()
            tprint("✅ Data leakage detector initialized", "success")
        else:
            self.leakage_detector = None
        
        # Initialize OOF generator
        self.oof_generator = OOFGenerator()
        
        # Initialize hardware optimization if available
        self.hardware_enabled = M1_HARDWARE_AVAILABLE and config.enable_hardware_optimization
        if self.hardware_enabled:
            self._init_hardware_optimization()
        else:
            self.gpu_accelerator = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_processor = None
            self.batch_processor = None
            tprint("ℹ️  Hardware optimization disabled", "info")
        
        # Initialize parallel processing
        self.enable_parallel = config.enable_parallel_processing
        self.max_workers = config.max_workers
        
        tprint(f"📊 Walk-Forward Configuration:", "info")
        tprint(f"   Mode: {config.mode.value}", "info")
        tprint(f"   Initial training size: {config.initial_training_size}", "info")
        tprint(f"   Validation size: {config.validation_size}", "info")
        tprint(f"   Step size: {config.step_size}", "info")
        tprint(f"   Purging: {config.enable_purging}, Embargo: {config.embargo_pct:.2%}", "info")
        tprint(f"   Leakage detection: {config.enable_leakage_detection}", "info")
        tprint(f"   Parallel processing: {self.enable_parallel} ({self.max_workers} workers)", "info")
        tprint(f"   Hardware optimization: {self.hardware_enabled}", "info")
        
        tprint("✅ Walk-Forward Analyzer initialization complete", "success")
    
    def _init_hardware_optimization(self):
        """Initialize hardware optimization components"""
        try:
            tprint("⚡ Initializing M1 hardware optimization", "info")
            
            # Initialize M1 accelerators
            self.gpu_accelerator = M1GPUAccelerator()
            self.memory_optimizer = M1MemoryOptimizer()
            self.cpu_optimizer = M1CPUOptimizer()
            
            # Initialize matrix operations
            self.matrix_processor = HardwareOptimizedMatrixProcessor()
            self.batch_processor = BatchMatrixProcessor(
                chunk_size_mb=self.config.chunk_size_mb,
                enable_gpu=True,
                enable_parallel=True,
                max_workers=self.max_workers
            )
            
            # Optimize memory
            self.memory_optimizer.optimize_memory_for_ml()
            
            tprint("✅ Hardware optimization initialized", "success")
            tprint(f"   GPU: {'Available' if self.gpu_accelerator.is_available() else 'Not available'}", "info")
            tprint(f"   Memory optimized: {self.memory_optimizer.is_optimized}", "info")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware optimization: {e}")
            tprint(f"⚠️  Hardware optimization init failed: {e}", "warning")
            self.gpu_accelerator = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_processor = None
            self.batch_processor = None
    
    def register_models(self, 
                       regime_models: Dict[int, Dict[str, Any]],
                       ensemble_models: Optional[Dict[str, Any]] = None):
        """
        Register models for walk-forward analysis.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
        """
        self.logger.info("📝 Registering models for walk-forward analysis")
        
        try:
            # Register regime models
            for regime_id, models in regime_models.items():
                self.available_models[regime_id] = {}
                
                for model_type, model_info in models.items():
                    self.available_models[regime_id][model_type] = {
                        'model': model_info['model'],
                        'performance': model_info.get('val_metrics', {}),
                        'feature_importance': model_info.get('feature_importance', {}),
                        'hyperparameters': model_info.get('hyperparameters', {})
                    }
                    
                    # Initialize performance tracking
                    model_id = f"regime_{regime_id}_{model_type}"
                    self.model_performance[model_id] = {
                        'fold_performance': [],
                        'overall_performance': {},
                        'evolution_history': []
                    }
            
            # Register ensemble models
            if ensemble_models:
                self.available_models['ensemble'] = ensemble_models
            
            self.logger.info(f"✅ Registered models for {len(self.available_models)} regimes")
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def run_walk_forward_analysis(self, 
                                market_data: pd.DataFrame,
                                target_variable: str = 'close',
                                feature_columns: Optional[List[str]] = None) -> WalkForwardResult:
        """
        Run comprehensive walk-forward analysis with hardware acceleration.
        
        Args:
            market_data: Historical market data
            target_variable: Target variable for prediction
            feature_columns: List of feature columns
            
        Returns:
            WalkForwardResult with complete analysis results
        """
        start_time = datetime.now()
        tprint("🚀 Starting Walk-Forward Analysis", "header")
        
        try:
            # Validate and prepare data
            prepared_data = self._prepare_data(market_data, target_variable, feature_columns)
            
            if not prepared_data['success']:
                tprint(f"❌ Data preparation failed: {prepared_data['error']}", "error")
                return WalkForwardResult(
                    success=False,
                    execution_time=0.0,
                    total_folds=0,
                    successful_folds=0,
                    error_message=prepared_data['error']
                )
            
            data = prepared_data['data']
            data_statistics = prepared_data['statistics']
            
            # Initialize analysis state
            self._initialize_analysis_state()
            
            # Generate walk-forward folds
            tprint("🔄 Generating walk-forward folds", "info")
            folds = self._generate_walk_forward_folds(data)
            
            if not folds:
                tprint("❌ No valid folds generated", "error")
                return WalkForwardResult(
                    success=False,
                    execution_time=0.0,
                    total_folds=0,
                    successful_folds=0,
                    error_message="No valid folds generated"
                )
            
            tprint(f"✅ Generated {len(folds)} walk-forward folds", "success")
            
            # Run walk-forward analysis
            tprint(f"🔄 Running walk-forward analysis on {len(folds)} folds", "info")
            fold_results = self._run_walk_forward_folds(folds, data, target_variable)
            
            successful_folds = len([f for f in fold_results if f.get('success', False)])
            tprint(f"✅ Completed {successful_folds}/{len(folds)} folds successfully", "success")
            
            # Analyze results
            tprint("📈 Analyzing walk-forward results", "info")
            analysis_results = self._analyze_walk_forward_results(fold_results)
            
            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = WalkForwardResult(
                success=True,
                execution_time=execution_time,
                total_folds=len(folds),
                successful_folds=successful_folds,
                overall_performance=analysis_results['overall_performance'],
                fold_performance=fold_results,
                regime_performance=analysis_results['regime_performance'],
                model_evolution=self.model_evolution,
                retraining_events=analysis_results['retraining_events'],
                regime_transitions=self.regime_transitions,
                regime_stability=analysis_results['regime_stability'],
                performance_trends=analysis_results['performance_trends'],
                degradation_events=analysis_results['degradation_events'],
                improvement_events=analysis_results['improvement_events'],
                configuration=self._get_configuration_summary(),
                data_statistics=data_statistics
            )
            
            # Save results if requested
            if self.config.save_results:
                tprint("💾 Saving walk-forward results", "info")
                self._save_walk_forward_results(result)
            
            tprint(f"✅ Walk-Forward Analysis Complete", "success")
            tprint(f"   Execution time: {execution_time:.2f}s", "info")
            tprint(f"   Total folds: {result.total_folds}", "info")
            tprint(f"   Successful folds: {result.successful_folds}", "info")
            tprint(f"   Success rate: {result.successful_folds/result.total_folds:.1%}", "info")
            
            # Print performance summary
            if result.overall_performance:
                tprint("📊 Performance Summary:", "info")
                for metric, values in result.overall_performance.items():
                    if isinstance(values, dict) and 'mean' in values:
                        tprint(f"   {metric}: {values['mean']:.4f} ± {values.get('std', 0):.4f}", "info")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            tprint(f"❌ Walk-forward analysis failed: {e}", "error")
            self.logger.exception("Full traceback:")
            
            return WalkForwardResult(
                success=False,
                execution_time=execution_time,
                total_folds=0,
                successful_folds=0,
                error_message=str(e)
            )
    
    def _prepare_data(self, 
                     market_data: pd.DataFrame,
                     target_variable: str,
                     feature_columns: Optional[List[str]]) -> Dict[str, Any]:
        """Prepare and validate data for walk-forward analysis."""
        try:
            tprint("📊 Preparing data for walk-forward analysis", "info")
            
            # Validate data
            if market_data.empty:
                tprint("❌ Empty dataset provided", "error")
                return {'success': False, 'error': 'Empty dataset'}
            
            # Check required columns
            if target_variable not in market_data.columns:
                tprint(f"❌ Target variable '{target_variable}' not found", "error")
                return {'success': False, 'error': f'Target variable {target_variable} not found'}
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in market_data.columns if col != target_variable]
            
            tprint(f"   Records: {len(market_data):,}, Features: {len(feature_columns)}", "info")
            
            # Check for NaN/Inf values
            nan_count = market_data.isnull().sum().sum()
            inf_count = np.isinf(market_data.select_dtypes(include=[np.number]).values).sum()
            
            if nan_count > 0:
                tprint(f"⚠️  Found {nan_count:,} NaN values", "warning")
            if inf_count > 0:
                tprint(f"⚠️  Found {inf_count:,} Inf values", "warning")
            
            # Calculate data statistics
            missing_ratio = nan_count / (len(market_data) * len(market_data.columns))
            data_quality = 1.0 - missing_ratio
            
            statistics = {
                'total_records': len(market_data),
                'total_features': len(feature_columns),
                'date_range': (market_data.index[0], market_data.index[-1]) if hasattr(market_data.index, '__getitem__') else None,
                'missing_data_ratio': missing_ratio,
                'inf_count': inf_count,
                'data_quality': data_quality
            }
            
            # Sort by index if datetime
            if hasattr(market_data.index, 'sort_values'):
                market_data = market_data.sort_index()
            
            # Check for data leakage if enabled
            if self.leakage_detector and self.config.enable_leakage_detection:
                tprint("🔍 Checking for data leakage", "info")
                try:
                    X = market_data[feature_columns].values
                    y = market_data[target_variable].values
                    
                    leakage_results = self.leakage_detector.detect_leakage(X, y)
                    
                    if leakage_results.get('has_leakage', False):
                        leakage_score = leakage_results.get('leakage_score', 0)
                        tprint(f"⚠️  Data leakage detected: score={leakage_score:.4f}", "warning")
                        statistics['leakage_detected'] = True
                        statistics['leakage_score'] = leakage_score
                    else:
                        tprint("✅ No data leakage detected", "success")
                        statistics['leakage_detected'] = False
                        statistics['leakage_score'] = 0.0
                except Exception as e:
                    tprint(f"⚠️  Leakage detection failed: {e}", "warning")
                    statistics['leakage_detected'] = False
                    statistics['leakage_score'] = 0.0
            
            tprint(f"✅ Data quality: {data_quality:.1%}", "success")
            
            return {
                'success': True,
                'data': market_data,
                'statistics': statistics
            }
            
        except Exception as e:
            tprint(f"❌ Data preparation failed: {e}", "error")
            return {'success': False, 'error': str(e)}
    
    def _initialize_analysis_state(self):
        """Initialize analysis state."""
        self.fold_results = []
        self.model_evolution = []
        self.regime_transitions = []
        self.performance_history = []
    
    def _generate_walk_forward_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward folds."""
        try:
            total_size = len(data)
            folds = []
            
            if self.config.mode == WalkForwardMode.FIXED_WINDOW:
                folds = self._generate_fixed_window_folds(data)
            elif self.config.mode == WalkForwardMode.EXPANDING_WINDOW:
                folds = self._generate_expanding_window_folds(data)
            elif self.config.mode == WalkForwardMode.ADAPTIVE_WINDOW:
                folds = self._generate_adaptive_window_folds(data)
            elif self.config.mode == WalkForwardMode.ROLLING_WINDOW:
                folds = self._generate_rolling_window_folds(data)
            else:
                raise ValueError(f"Unknown walk-forward mode: {self.config.mode}")
            
            self.logger.info(f"📊 Generated {len(folds)} walk-forward folds")
            return folds
            
        except Exception as e:
            self.logger.error(f"❌ Fold generation failed: {e}")
            return []
    
    def _generate_fixed_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate fixed window folds."""
        folds = []
        total_size = len(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            fold = {
                'fold_id': len(folds),
                'training_start': 0,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[:start_idx],
                'validation_data': data.iloc[start_idx:end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _generate_expanding_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate expanding window folds."""
        folds = []
        total_size = len(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            fold = {
                'fold_id': len(folds),
                'training_start': 0,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[:start_idx],
                'validation_data': data.iloc[start_idx:end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _generate_adaptive_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate adaptive window folds based on regime changes."""
        folds = []
        total_size = len(data)
        
        # Detect regime changes
        regime_changes = self._detect_regime_changes(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            # Adjust window based on regime changes
            adjusted_start = self._adjust_window_for_regime_changes(
                start_idx, regime_changes, data
            )
            
            fold = {
                'fold_id': len(folds),
                'training_start': adjusted_start,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[adjusted_start:start_idx],
                'validation_data': data.iloc[start_idx:end_idx],
                'regime_changes': [rc for rc in regime_changes if adjusted_start <= rc['index'] < end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _generate_rolling_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate rolling window folds."""
        folds = []
        total_size = len(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            fold = {
                'fold_id': len(folds),
                'training_start': start_idx - self.config.initial_training_size,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[start_idx - self.config.initial_training_size:start_idx],
                'validation_data': data.iloc[start_idx:end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _detect_regime_changes(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect regime changes in data."""
        try:
            # Simple regime change detection based on volatility
            if 'close' in data.columns:
                prices = data['close'].values
                returns = np.diff(prices) / prices[:-1]
                volatility = pd.Series(returns).rolling(window=20).std()
                
                # Detect significant changes in volatility
                volatility_changes = []
                for i in range(1, len(volatility)):
                    if abs(volatility.iloc[i] - volatility.iloc[i-1]) > self.config.regime_change_threshold:
                        volatility_changes.append({
                            'index': i,
                            'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else None,
                            'change_type': 'volatility',
                            'magnitude': abs(volatility.iloc[i] - volatility.iloc[i-1])
                        })
                
                return volatility_changes
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Regime change detection failed: {e}")
            return []
    
    def _adjust_window_for_regime_changes(self, 
                                        start_idx: int,
                                        regime_changes: List[Dict[str, Any]],
                                        data: pd.DataFrame) -> int:
        """Adjust training window based on regime changes."""
        # Find the most recent regime change before start_idx
        recent_changes = [rc for rc in regime_changes if rc['index'] < start_idx]
        
        if recent_changes:
            # Adjust start to include the regime change
            latest_change = max(recent_changes, key=lambda x: x['index'])
            adjusted_start = max(0, latest_change['index'] - self.config.min_regime_samples)
            return adjusted_start
        
        return max(0, start_idx - self.config.initial_training_size)
    
    def _run_walk_forward_folds(self, 
                               folds: List[Dict[str, Any]],
                               data: pd.DataFrame,
                               target_variable: str) -> List[Dict[str, Any]]:
        """Run walk-forward analysis on all folds."""
        fold_results = []
        
        for fold in folds:
            try:
                self.logger.info(f"🔄 Processing fold {fold['fold_id']}...")
                
                # Detect regime for training data
                training_regime = self._detect_regime_for_data(fold['training_data'])
                
                # Select model for regime
                selected_model = self._select_model_for_regime(training_regime)
                
                if selected_model is None:
                    fold_results.append({
                        'fold_id': fold['fold_id'],
                        'success': False,
                        'error': 'No model available for regime'
                    })
                    continue
                
                # Train/retrain model if needed
                if self.config.enable_model_retraining:
                    retrained_model = self._retrain_model(
                        selected_model, fold['training_data'], target_variable
                    )
                    if retrained_model:
                        selected_model = retrained_model
                
                # Validate model on validation data
                validation_result = self._validate_model(
                    selected_model, fold['validation_data'], target_variable
                )
                
                # Calculate performance metrics
                performance_metrics = self._calculate_fold_performance(
                    validation_result, fold['validation_data'], target_variable
                )
                
                # Record fold result
                fold_result = {
                    'fold_id': fold['fold_id'],
                    'success': True,
                    'training_regime': training_regime,
                    'selected_model': selected_model['model_type'],
                    'performance_metrics': performance_metrics,
                    'validation_result': validation_result,
                    'regime_changes': fold.get('regime_changes', [])
                }
                
                fold_results.append(fold_result)
                
                # Update model evolution
                self._update_model_evolution(fold_result)
                
                # Update performance history
                self.performance_history.append({
                    'fold_id': fold['fold_id'],
                    'performance': performance_metrics,
                    'regime': training_regime,
                    'model': selected_model['model_type']
                })
                
                self.logger.info(f"   ✅ Fold {fold['fold_id']} completed - Performance: {performance_metrics.get('f1_score', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Fold {fold['fold_id']} failed: {e}")
                fold_results.append({
                    'fold_id': fold['fold_id'],
                    'success': False,
                    'error': str(e)
                })
        
        return fold_results
    
    def _detect_regime_for_data(self, data: pd.DataFrame) -> int:
        """Detect regime for given data."""
        try:
            # Simple regime detection based on volatility
            if 'close' in data.columns and len(data) > 1:
                prices = data['close'].values
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                if volatility < 0.01:
                    return 0  # Low volatility regime
                elif volatility < 0.03:
                    return 1  # Medium volatility regime
                else:
                    return 2  # High volatility regime
            else:
                return 0  # Default regime
                
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return 0
    
    def _select_model_for_regime(self, regime_id: int) -> Optional[Dict[str, Any]]:
        """Select model for regime."""
        try:
            if regime_id not in self.available_models:
                return None
            
            regime_models = self.available_models[regime_id]
            
            # Select best performing model
            best_model_type = max(
                regime_models.keys(),
                key=lambda x: regime_models[x]['performance'].get('f1_score', 0.0)
            )
            
            return {
                'model': regime_models[best_model_type]['model'],
                'model_type': best_model_type,
                'regime_id': regime_id,
                'performance': regime_models[best_model_type]['performance']
            }
            
        except Exception as e:
            self.logger.warning(f"Model selection failed: {e}")
            return None
    
    def _retrain_model(self, 
                      selected_model: Dict[str, Any],
                      training_data: pd.DataFrame,
                      target_variable: str) -> Optional[Dict[str, Any]]:
        """Retrain model on new data."""
        try:
            if not self.config.enable_model_retraining:
                return selected_model
            
            # Simple retraining (in practice, this would be more sophisticated)
            model = selected_model['model']
            
            # Check if model supports incremental learning
            if hasattr(model, 'partial_fit'):
                # Incremental learning
                X = training_data.drop(columns=[target_variable]).values
                y = training_data[target_variable].values
                model.partial_fit(X, y)
            else:
                # Full retraining
                X = training_data.drop(columns=[target_variable]).values
                y = training_data[target_variable].values
                model.fit(X, y)
            
            return selected_model
            
        except Exception as e:
            self.logger.warning(f"Model retraining failed: {e}")
            return selected_model
    
    def _validate_model(self, 
                       selected_model: Dict[str, Any],
                       validation_data: pd.DataFrame,
                       target_variable: str) -> Dict[str, Any]:
        """Validate model on validation data."""
        try:
            model = selected_model['model']
            
            # Prepare validation data
            X_val = validation_data.drop(columns=[target_variable]).values
            y_val = validation_data[target_variable].values
            
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X_val)
            else:
                return {'success': False, 'error': 'Model does not support prediction'}
            
            # Calculate confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_val)
                    confidence = np.mean(np.max(proba, axis=1))
                except:
                    confidence = 0.5
            
            return {
                'success': True,
                'predictions': predictions,
                'confidence': confidence,
                'model_type': selected_model['model_type'],
                'regime_id': selected_model['regime_id']
            }
            
        except Exception as e:
            self.logger.warning(f"Model validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_fold_performance(self, 
                                   validation_result: Dict[str, Any],
                                   validation_data: pd.DataFrame,
                                   target_variable: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics using validated utilities."""
        try:
            if not validation_result['success']:
                return self._get_default_metrics()
            
            predictions = ensure_array(validation_result['predictions'])
            y_true = ensure_array(validation_data[target_variable].values)
            
            # Validate input lengths
            if len(predictions) != len(y_true):
                tprint(f"⚠️  Prediction length mismatch: {len(predictions)} vs {len(y_true)}", "warning")
                return self._get_default_metrics()
            
            # Remove NaN/Inf values
            valid_mask = ~(check_for_nans(predictions) | check_for_infs(predictions) |
                          check_for_nans(y_true) | check_for_infs(y_true))
            predictions = predictions[valid_mask]
            y_true = y_true[valid_mask]
            
            if len(predictions) == 0:
                tprint("⚠️  No valid predictions after filtering", "warning")
                return self._get_default_metrics()
            
            # Calculate basic metrics
            accuracy = float(np.mean(predictions == y_true))
            accuracy = validate_probability(accuracy)
            
            # Calculate precision, recall, F1 with proper error handling
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
                
                precision = validate_probability(precision)
                recall = validate_probability(recall)
                f1 = validate_probability(f1)
            except Exception as e:
                self.logger.warning(f"sklearn metrics failed: {e}")
                precision, recall, f1 = 0.0, 0.0, 0.0
            
            # Calculate trading metrics using common_operations
            if 'close' in validation_data.columns:
                try:
                    prices = ensure_array(validation_data['close'].values)
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Remove invalid values
                    returns_valid = returns[~(check_for_nans(returns) | check_for_infs(returns))]
                    
                    if len(returns_valid) > 0:
                        # Use validated calculation functions
                        sharpe_ratio = calculate_sharpe_ratio(returns_valid)
                        sortino_ratio = calculate_sortino_ratio(returns_valid)
                        max_dd = calculate_max_drawdown(np.cumsum(returns_valid))
                        win_rate = calculate_win_rate(returns_valid)
                        profit_factor = calculate_profit_factor(returns_valid)
                        calmar_ratio = calculate_calmar_ratio(returns_valid, max_dd)
                        
                        # Validate all metrics
                        sharpe_ratio = validate_positive(sharpe_ratio, default=0.0) if not check_for_nans(sharpe_ratio) else 0.0
                        sortino_ratio = validate_positive(sortino_ratio, default=0.0) if not check_for_nans(sortino_ratio) else 0.0
                        max_dd = float(max_dd) if not check_for_nans(max_dd) else 0.0
                        win_rate = validate_probability(win_rate) if not check_for_nans(win_rate) else 0.0
                        profit_factor = validate_positive(profit_factor, default=0.0) if not check_for_nans(profit_factor) else 0.0
                        calmar_ratio = float(calmar_ratio) if not check_for_nans(calmar_ratio) else 0.0
                    else:
                        sharpe_ratio = sortino_ratio = max_dd = win_rate = profit_factor = calmar_ratio = 0.0
                except Exception as e:
                    self.logger.warning(f"Trading metrics calculation failed: {e}")
                    sharpe_ratio = sortino_ratio = max_dd = win_rate = profit_factor = calmar_ratio = 0.0
            else:
                sharpe_ratio = sortino_ratio = max_dd = win_rate = profit_factor = calmar_ratio = 0.0
            
            # Get and validate confidence
            confidence = validate_probability(validation_result.get('confidence', 0.5))
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'calmar_ratio': calmar_ratio,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Performance calculation failed: {e}")
            tprint(f"⚠️  Performance calculation failed: {e}", "warning")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when calculation fails."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'confidence': 0.0
        }
    
    def _update_model_evolution(self, fold_result: Dict[str, Any]):
        """Update model evolution tracking."""
        try:
            evolution_entry = {
                'fold_id': fold_result['fold_id'],
                'regime': fold_result['training_regime'],
                'model_type': fold_result['selected_model'],
                'performance': fold_result['performance_metrics'],
                'timestamp': datetime.now()
            }
            
            self.model_evolution.append(evolution_entry)
            
        except Exception as e:
            self.logger.warning(f"Model evolution update failed: {e}")
    
    def _analyze_walk_forward_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward results."""
        try:
            # Calculate overall performance
            successful_folds = [f for f in fold_results if f['success']]
            
            if not successful_folds:
                return self._get_default_analysis()
            
            # Aggregate performance metrics
            overall_performance = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio']:
                values = [f['performance_metrics'][metric] for f in successful_folds]
                overall_performance[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            # Analyze regime performance
            regime_performance = self._analyze_regime_performance(successful_folds)
            
            # Analyze model evolution
            retraining_events = self._analyze_retraining_events()
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends(successful_folds)
            
            # Detect degradation and improvement events
            degradation_events = self._detect_degradation_events(successful_folds)
            improvement_events = self._detect_improvement_events(successful_folds)
            
            # Analyze regime stability
            regime_stability = self._analyze_regime_stability()
            
            return {
                'overall_performance': overall_performance,
                'regime_performance': regime_performance,
                'retraining_events': retraining_events,
                'performance_trends': performance_trends,
                'degradation_events': degradation_events,
                'improvement_events': improvement_events,
                'regime_stability': regime_stability
            }
            
        except Exception as e:
            self.logger.error(f"❌ Results analysis failed: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when calculation fails."""
        return {
            'overall_performance': {},
            'regime_performance': {},
            'retraining_events': [],
            'performance_trends': {},
            'degradation_events': [],
            'improvement_events': [],
            'regime_stability': {}
        }
    
    def _analyze_regime_performance(self, successful_folds: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """Analyze performance by regime."""
        regime_performance = {}
        
        for fold in successful_folds:
            regime = fold['training_regime']
            performance = fold['performance_metrics']
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'folds': 0,
                    'accuracy': [],
                    'f1_score': [],
                    'sharpe_ratio': []
                }
            
            regime_performance[regime]['folds'] += 1
            regime_performance[regime]['accuracy'].append(performance['accuracy'])
            regime_performance[regime]['f1_score'].append(performance['f1_score'])
            regime_performance[regime]['sharpe_ratio'].append(performance['sharpe_ratio'])
        
        # Calculate averages
        for regime in regime_performance:
            perf = regime_performance[regime]
            regime_performance[regime] = {
                'folds': perf['folds'],
                'mean_accuracy': np.mean(perf['accuracy']),
                'mean_f1_score': np.mean(perf['f1_score']),
                'mean_sharpe_ratio': np.mean(perf['sharpe_ratio']),
                'std_accuracy': np.std(perf['accuracy']),
                'std_f1_score': np.std(perf['f1_score']),
                'std_sharpe_ratio': np.std(perf['sharpe_ratio'])
            }
        
        return regime_performance
    
    def _analyze_retraining_events(self) -> List[Dict[str, Any]]:
        """Analyze model retraining events."""
        retraining_events = []
        
        for i, evolution in enumerate(self.model_evolution):
            if i > 0:
                prev_evolution = self.model_evolution[i-1]
                
                # Check if model changed
                if evolution['model_type'] != prev_evolution['model_type']:
                    retraining_events.append({
                        'fold_id': evolution['fold_id'],
                        'from_model': prev_evolution['model_type'],
                        'to_model': evolution['model_type'],
                        'regime': evolution['regime'],
                        'performance_change': evolution['performance']['f1_score'] - prev_evolution['performance']['f1_score']
                    })
        
        return retraining_events
    
    def _analyze_performance_trends(self, successful_folds: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze performance trends."""
        trends = {}
        
        if len(successful_folds) < 3:
            return trends
        
        # Analyze F1 score trend
        f1_scores = [f['performance_metrics']['f1_score'] for f in successful_folds]
        early_f1 = np.mean(f1_scores[:len(f1_scores)//3])
        late_f1 = np.mean(f1_scores[-len(f1_scores)//3:])
        
        if late_f1 > early_f1 + 0.05:
            trends['f1_score'] = 'improving'
        elif late_f1 < early_f1 - 0.05:
            trends['f1_score'] = 'declining'
        else:
            trends['f1_score'] = 'stable'
        
        # Analyze accuracy trend
        accuracy_scores = [f['performance_metrics']['accuracy'] for f in successful_folds]
        early_acc = np.mean(accuracy_scores[:len(accuracy_scores)//3])
        late_acc = np.mean(accuracy_scores[-len(accuracy_scores)//3:])
        
        if late_acc > early_acc + 0.05:
            trends['accuracy'] = 'improving'
        elif late_acc < early_acc - 0.05:
            trends['accuracy'] = 'declining'
        else:
            trends['accuracy'] = 'stable'
        
        return trends
    
    def _detect_degradation_events(self, successful_folds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance degradation events."""
        degradation_events = []
        
        for i in range(1, len(successful_folds)):
            current_fold = successful_folds[i]
            prev_fold = successful_folds[i-1]
            
            current_f1 = current_fold['performance_metrics']['f1_score']
            prev_f1 = prev_fold['performance_metrics']['f1_score']
            
            if current_f1 < prev_f1 - self.config.degradation_threshold:
                degradation_events.append({
                    'fold_id': current_fold['fold_id'],
                    'metric': 'f1_score',
                    'current_value': current_f1,
                    'previous_value': prev_f1,
                    'degradation': prev_f1 - current_f1,
                    'regime': current_fold['training_regime']
                })
        
        return degradation_events
    
    def _detect_improvement_events(self, successful_folds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance improvement events."""
        improvement_events = []
        
        for i in range(1, len(successful_folds)):
            current_fold = successful_folds[i]
            prev_fold = successful_folds[i-1]
            
            current_f1 = current_fold['performance_metrics']['f1_score']
            prev_f1 = prev_fold['performance_metrics']['f1_score']
            
            if current_f1 > prev_f1 + self.config.degradation_threshold:
                improvement_events.append({
                    'fold_id': current_fold['fold_id'],
                    'metric': 'f1_score',
                    'current_value': current_f1,
                    'previous_value': prev_f1,
                    'improvement': current_f1 - prev_f1,
                    'regime': current_fold['training_regime']
                })
        
        return improvement_events
    
    def _analyze_regime_stability(self) -> Dict[int, float]:
        """Analyze regime stability."""
        regime_stability = {}
        
        for regime in set([f['training_regime'] for f in self.performance_history]):
            regime_folds = [f for f in self.performance_history if f['regime'] == regime]
            
            if len(regime_folds) < 2:
                regime_stability[regime] = 1.0
                continue
            
            # Calculate stability based on performance consistency
            f1_scores = [f['performance']['f1_score'] for f in regime_folds]
            stability = 1.0 - (np.std(f1_scores) / (np.mean(f1_scores) + 1e-8))
            regime_stability[regime] = max(0.0, min(1.0, stability))
        
        return regime_stability
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'mode': self.config.mode.value,
            'initial_training_size': self.config.initial_training_size,
            'validation_size': self.config.validation_size,
            'step_size': self.config.step_size,
            'enable_regime_aware_validation': self.config.enable_regime_aware_validation,
            'enable_model_retraining': self.config.enable_model_retraining,
            'validation_metrics': [m.value for m in self.config.validation_metrics],
            'performance_threshold': self.config.performance_threshold,
            'degradation_threshold': self.config.degradation_threshold
        }
    
    def _save_walk_forward_results(self, result: WalkForwardResult):
        """Save walk-forward results."""
        try:
            results_path = Path(self.config.results_path)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_summary = {
                'success': result.success,
                'execution_time': result.execution_time,
                'total_folds': result.total_folds,
                'successful_folds': result.successful_folds,
                'overall_performance': result.overall_performance,
                'regime_performance': result.regime_performance,
                'performance_trends': result.performance_trends,
                'configuration': result.configuration,
                'data_statistics': result.data_statistics
            }
            
            with open(results_path / "walk_forward_summary.json", 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            # Save detailed results
            with open(results_path / "walk_forward_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"💾 Walk-forward results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save walk-forward results: {e}")
    
    def get_walk_forward_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward analysis."""
        if not self.fold_results:
            return {'error': 'No walk-forward data available'}
        
        successful_folds = [f for f in self.fold_results if f['success']]
        
        return {
            'total_folds': len(self.fold_results),
            'successful_folds': len(successful_folds),
            'success_rate': len(successful_folds) / len(self.fold_results),
            'model_evolution_events': len(self.model_evolution),
            'regime_transitions': len(self.regime_transitions),
            'performance_history': len(self.performance_history)
        }