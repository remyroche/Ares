"""
Tactician Model Training + HPO

This module provides specialized model training for the Tactician system with
advanced hyperparameter optimization, utilizing M1 optimizations and tactical features.
Enhanced with ML commons utilities for better integration and performance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
import optuna
from pathlib import Path

# Import general training components
from .general_model_training import GeneralModelTrainer, ModelTrainingConfig, ModelType, TaskType

# ML Commons utilities - Enhanced integration
from src.utils.ml_common import (
    ModelEvaluator, HPOptimizer, FeatureSelectionFramework,
    DataLabelingUtilities, MemoryEfficientTraining, 
    ParallelProcessingCoordinator, ModelRegistry,
    DataQualityUtilities, CrossValidationUtilities,
    LookaheadProtection, MLTrainingSafeguards,
    HMMRegimeDetector, RegimeDataProcessor
)

# M1 Optimization imports
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials, 
    get_scaled_hpo_timeout, log_intensity_info, apply_intensity_scaling
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class TacticianModelType(Enum):
    """Types of Tactician models."""
    POSITION_SIZER = "position_sizer"
    ENTRY_TIMING = "entry_timing"
    EXIT_TIMING = "exit_timing"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    MARKET_MAKER = "market_maker"
    ARBITRAGE_DETECTOR = "arbitrage_detector"
    LIQUIDITY_ANALYZER = "liquidity_analyzer"


class TacticalStrategy(Enum):
    """Tactical trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"


@dataclass
class TacticianTrainingConfig:
    """Configuration for Tactician model training."""
    # Basic configuration
    tactician_name: str
    output_dir: str
    
    # Model configuration
    model_types: List[TacticianModelType] = field(default_factory=lambda: [
        TacticianModelType.POSITION_SIZER,
        TacticianModelType.ENTRY_TIMING,
        TacticianModelType.EXIT_TIMING,
        TacticianModelType.RISK_MANAGER
    ])
    
    # Tactical strategy configuration
    primary_strategy: TacticalStrategy = TacticalStrategy.MOMENTUM
    secondary_strategies: List[TacticalStrategy] = field(default_factory=list)
    
    # Feature configuration
    feature_columns: List[str] = field(default_factory=list)
    target_columns: Dict[str, str] = field(default_factory=lambda: {
        'position_sizer': 'optimal_position_size',
        'entry_timing': 'entry_signal',
        'exit_timing': 'exit_signal',
        'risk_manager': 'risk_score',
        'portfolio_optimizer': 'portfolio_weights',
        'market_maker': 'spread_adjustment',
        'arbitrage_detector': 'arbitrage_opportunity',
        'liquidity_analyzer': 'liquidity_score'
    })
    
    # Advanced HPO configuration
    enable_advanced_hpo: bool = True
    hpo_trials: int = 200  # More trials for tactical models
    hpo_timeout: int = 7200  # 2 hours
    enable_multi_objective_optimization: bool = True
    optimization_objectives: List[str] = field(default_factory=lambda: ['sharpe_ratio', 'max_drawdown', 'win_rate'])
    
    def __post_init__(self):
        """Apply intensity scaling after initialization."""
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.hpo_trials = get_scaled_hpo_trials(self.hpo_trials, intensity_pct)
            self.hpo_timeout = get_scaled_hpo_timeout(self.hpo_timeout, intensity_pct)
            self.early_stopping_patience = max(1, int(self.early_stopping_patience * intensity_pct))
            logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%): HPO trials={self.hpo_trials}, timeout={self.hpo_timeout}s")
    
    # Tactical-specific configuration
    enable_regime_awareness: bool = True
    enable_market_microstructure: bool = True
    enable_liquidity_considerations: bool = True
    enable_transaction_costs: bool = True
    
    # Training configuration
    enable_early_stopping: bool = True
    early_stopping_patience: int = 15
    enable_online_learning: bool = False
    learning_rate_schedule: str = "adaptive"  # adaptive, constant, decay
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    enable_walk_forward_validation: bool = True
    walk_forward_windows: int = 10
    
    # Output settings
    save_models: bool = True
    save_predictions: bool = True
    generate_reports: bool = True
    save_tactical_insights: bool = True


@dataclass
class TacticianTrainingResults:
    """Results from Tactician model training."""
    # Basic info
    tactician_name: str
    primary_strategy: TacticalStrategy
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Individual model results
    individual_models: Dict[str, Any] = field(default_factory=dict)
    model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Tactical performance
    tactical_metrics: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Advanced HPO results
    hpo_results: Dict[str, Any] = field(default_factory=dict)
    multi_objective_results: Dict[str, Any] = field(default_factory=dict)
    
    # Walk-forward validation results
    walk_forward_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tactical insights
    tactical_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Overall performance
    overall_performance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    config: TacticianTrainingConfig = field(default_factory=TacticianTrainingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class TacticianModelTrainer:
    """Tactician model trainer with advanced HPO and tactical features, enhanced with ML commons."""
    
    def __init__(self, config: TacticianTrainingConfig):
        """Initialize Tactician model trainer."""
        self.config = config
        self.logger = logger.getChild('TacticianModelTrainer')
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize ML Commons utilities
        self.model_evaluator = ModelEvaluator()
        self.hpo_optimizer = HPOptimizer()
        self.feature_selector = FeatureSelectionFramework()
        self.data_labeler = DataLabelingUtilities()
        self.memory_efficient_training = MemoryEfficientTraining()
        self.parallel_coordinator = ParallelProcessingCoordinator()
        self.model_registry = ModelRegistry()
        self.data_quality = DataQualityUtilities()
        self.cv_utils = CrossValidationUtilities()
        self.lookahead_protection = LookaheadProtection()
        self.ml_safeguards = MLTrainingSafeguards()
        self.hmm_regime_detector = HMMRegimeDetector()
        self.regime_processor = RegimeDataProcessor()
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        self.logger.info(f"🚀 TacticianModelTrainer initialized for {config.tactician_name}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎯 Primary strategy: {config.primary_strategy.value}")
        self.logger.info(f"🤖 Model types: {[mt.value for mt in config.model_types]}")
        self.logger.info(f"🔬 Advanced HPO: {config.enable_advanced_hpo}")
        self.logger.info(f"🔧 ML Commons integration: Enhanced")
    
    @traced(span_name='train_tactician_models')
    @log_execution_time
    async def train_tactician_models(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> TacticianTrainingResults:
        """Train all Tactician models with advanced HPO and ML commons integration."""
        
        self.logger.info("🚀 Starting enhanced Tactician model training with ML commons...")
        start_time = time.time()
        
        # Validate inputs with ML safeguards
        self._validate_data_with_safeguards(data)
        
        # Apply lookahead bias protection
        data = self.lookahead_protection.apply_protection(data)
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._train_tactician_models_internal(data, **kwargs)
        else:
            results = await self._train_tactician_models_internal(data, **kwargs)
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ Enhanced Tactician model training completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Overall performance: {results.overall_performance}")
        
        return results
    
    def _validate_data_with_safeguards(self, data: pd.DataFrame) -> None:
        """Validate input data for Tactician training using ML safeguards."""
        
        if data.empty:
            raise ValidationError("Input data is empty")
        
        # Check required columns
        missing_features = [col for col in self.config.feature_columns if col not in data.columns]
        if missing_features:
            raise ValidationError(f"Missing feature columns: {missing_features}")
        
        # Check target columns
        for model_type, target_col in self.config.target_columns.items():
            if target_col not in data.columns:
                self.logger.warning(f"⚠️ Missing target column for {model_type}: {target_col}")
        
        # Check for sufficient data
        if len(data) < 200:  # More data required for tactical models
            raise ValidationError(f"Insufficient data: {len(data)} < 200")
        
        # Check for time series structure
        if 'timestamp' not in data.columns:
            self.logger.warning("⚠️ No timestamp column found, assuming sequential data")
        
        # Use ML safeguards for advanced validation
        try:
            self.ml_safeguards.validate_training_data(data, list(self.config.target_columns.values())[0])
            self.logger.info("✅ ML safeguards validation passed")
        except Exception as e:
            self.logger.warning(f"⚠️ ML safeguards validation warning: {e}")
        
        # Data quality assessment
        quality_score = self.data_quality.calculate_data_quality_score(data)
        self.logger.info(f"📊 Data quality score: {quality_score:.2f}")
        
        if quality_score < 0.7:
            self.logger.warning("⚠️ Low data quality score detected")
    
    async def _train_tactician_models_internal(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> TacticianTrainingResults:
        """Internal Tactician model training logic with ML commons integration."""
        
        results = TacticianTrainingResults(
            tactician_name=self.config.tactician_name,
            primary_strategy=self.config.primary_strategy,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        # Prepare tactical features with enhanced preprocessing
        tactical_data = await self._prepare_tactical_features_enhanced(data)
        
        # Train individual models with advanced HPO and ML commons
        individual_results = {}
        model_performance = {}
        hpo_results = {}
        
        for model_type in self.config.model_types:
            self.logger.info(f"🔄 Training {model_type.value} with enhanced ML commons...")
            
            try:
                # Create training config for this model type
                training_config = self._create_tactical_training_config(model_type)
                
                # Perform advanced HPO using ML commons if enabled
                if self.config.enable_advanced_hpo:
                    hpo_result = await self._advanced_hyperparameter_optimization_enhanced(
                        tactical_data, model_type, training_config
                    )
                    hpo_results[model_type.value] = hpo_result
                    training_config.model_params = hpo_result['best_params']
                
                # Train model with enhanced ML commons integration
                trainer = GeneralModelTrainer(training_config)
                model_result = await trainer.train_model(tactical_data, **kwargs)
                
                # Store results
                individual_results[model_type.value] = model_result
                model_performance[model_type.value] = model_result.validation_metrics
                
                self.logger.info(f"✅ {model_type.value} training completed with ML commons")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train {model_type.value}: {e}")
                continue
        
        # Perform multi-objective optimization if enabled
        multi_objective_results = {}
        if self.config.enable_multi_objective_optimization:
            multi_objective_results = await self._multi_objective_optimization_enhanced(tactical_data)
        
        # Perform walk-forward validation if enabled
        walk_forward_results = []
        if self.config.enable_walk_forward_validation:
            walk_forward_results = await self._walk_forward_validation_enhanced(tactical_data)
        
        # Generate tactical insights
        tactical_insights = await self._generate_tactical_insights_enhanced(
            individual_results, model_performance, tactical_data
        )
        
        # Calculate tactical performance metrics
        tactical_metrics = self._calculate_tactical_metrics(model_performance, tactical_data)
        
        # Calculate overall performance
        overall_performance = self._calculate_overall_performance(
            model_performance, tactical_metrics, walk_forward_results
        )
        
        # Update results
        results.individual_models = individual_results
        results.model_performance = model_performance
        results.hpo_results = hpo_results
        results.multi_objective_results = multi_objective_results
        results.walk_forward_results = walk_forward_results
        results.tactical_insights = tactical_insights
        results.tactical_metrics = tactical_metrics
        results.overall_performance = overall_performance
        
        return results
    
    async def _prepare_tactical_features_enhanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare tactical features for training with enhanced ML commons preprocessing."""
        
        tactical_data = data.copy()
        
        # Enhanced data cleaning using ML commons
        tactical_data = self.data_quality.enhanced_automated_data_cleaning(tactical_data)
        
        # Add regime awareness features using ML commons
        if self.config.enable_regime_awareness:
            tactical_data = await self._add_regime_features_enhanced(tactical_data)
        
        # Add market microstructure features
        if self.config.enable_market_microstructure:
            tactical_data = self._add_microstructure_features(tactical_data)
        
        # Add liquidity features
        if self.config.enable_liquidity_considerations:
            tactical_data = self._add_liquidity_features(tactical_data)
        
        # Add transaction cost features
        if self.config.enable_transaction_costs:
            tactical_data = self._add_transaction_cost_features(tactical_data)
        
        # Add strategy-specific features
        tactical_data = self._add_strategy_features(tactical_data)
        
        self.logger.info(f"📊 Prepared enhanced tactical features: {len(tactical_data.columns)} total columns")
        
        return tactical_data
    
    async def _add_regime_features_enhanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime awareness features using ML commons."""
        
        # Use ML commons HMM regime detector for enhanced regime analysis
        try:
            regime_analysis = await self.hmm_regime_detector.detect_regimes(data)
            if regime_analysis and 'regime_labels' in regime_analysis:
                data['enhanced_regime'] = regime_analysis['regime_labels']
                data['regime_confidence'] = regime_analysis.get('confidence_scores', [0.5] * len(data))
                self.logger.info("✅ Enhanced regime features added using ML commons")
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced regime detection failed: {e}")
            # Fallback to basic regime features
            if 'hmm_cluster' in data.columns:
                data['regime_transition'] = data['hmm_cluster'].diff()
                data['regime_duration'] = data.groupby('hmm_cluster').cumcount()
                data['regime_stability'] = data['hmm_cluster'].rolling(10).std()
        
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        
        # Bid-ask spread features
        if all(col in data.columns for col in ['bid', 'ask']):
            data['spread'] = data['ask'] - data['bid']
            data['spread_pct'] = data['spread'] / data['mid_price']
            data['spread_volatility'] = data['spread'].rolling(20).std()
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            data['volume_volatility'] = data['volume'].rolling(20).std()
        
        # Price impact features
        if 'close' in data.columns:
            data['price_impact'] = data['close'].pct_change()
            data['price_impact_volatility'] = data['price_impact'].rolling(20).std()
        
        return data
    
    def _add_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity analysis features."""
        
        # Volume-based liquidity
        if 'volume' in data.columns:
            data['liquidity_score'] = data['volume'].rolling(20).mean()
            data['liquidity_volatility'] = data['volume'].rolling(20).std()
        
        # Price-based liquidity
        if 'close' in data.columns:
            data['price_volatility'] = data['close'].rolling(20).std()
            data['liquidity_ratio'] = data['liquidity_score'] / data['price_volatility']
        
        return data
    
    def _add_transaction_cost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add transaction cost features."""
        
        # Spread-based costs
        if 'spread' in data.columns:
            data['transaction_cost'] = data['spread'] / 2  # Half spread
            data['transaction_cost_pct'] = data['transaction_cost'] / data['close']
        
        # Volume-based costs (market impact)
        if 'volume' in data.columns and 'close' in data.columns:
            data['market_impact'] = data['volume'] / data['close'] * 0.001  # Simplified
            data['total_cost'] = data['transaction_cost'] + data['market_impact']
        
        return data
    
    def _add_strategy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific features."""
        
        if self.config.primary_strategy == TacticalStrategy.MOMENTUM:
            # Momentum features
            if 'close' in data.columns:
                data['momentum_1d'] = data['close'].pct_change(1)
                data['momentum_5d'] = data['close'].pct_change(5)
                data['momentum_20d'] = data['close'].pct_change(20)
                data['momentum_strength'] = data['momentum_5d'] / data['momentum_20d']
        
        elif self.config.primary_strategy == TacticalStrategy.MEAN_REVERSION:
            # Mean reversion features
            if 'close' in data.columns:
                data['sma_20'] = data['close'].rolling(20).mean()
                data['mean_reversion_signal'] = (data['close'] - data['sma_20']) / data['sma_20']
                data['mean_reversion_strength'] = abs(data['mean_reversion_signal'])
        
        elif self.config.primary_strategy == TacticalStrategy.BREAKOUT:
            # Breakout features
            if 'close' in data.columns:
                data['high_20'] = data['close'].rolling(20).max()
                data['low_20'] = data['close'].rolling(20).min()
                data['breakout_signal'] = (data['close'] - data['high_20'].shift(1)) / data['high_20'].shift(1)
                data['breakdown_signal'] = (data['close'] - data['low_20'].shift(1)) / data['low_20'].shift(1)
        
        return data
    
    def _create_tactical_training_config(self, model_type: TacticianModelType) -> ModelTrainingConfig:
        """Create training configuration for specific tactical model type."""
        
        # Determine task type based on model type
        if model_type in [TacticianModelType.ENTRY_TIMING, TacticianModelType.EXIT_TIMING, 
                         TacticianModelType.ARBITRAGE_DETECTOR]:
            task_type = TaskType.CLASSIFICATION
        else:
            task_type = TaskType.REGRESSION
        
        # Determine model type based on tactical requirements
        if model_type == TacticianModelType.POSITION_SIZER:
            ml_model_type = ModelType.XGBOOST  # Good for position sizing
        elif model_type == TacticianModelType.ENTRY_TIMING:
            ml_model_type = ModelType.LIGHTGBM  # Fast for timing
        elif model_type == TacticianModelType.EXIT_TIMING:
            ml_model_type = ModelType.LIGHTGBM  # Fast for timing
        elif model_type == TacticianModelType.RISK_MANAGER:
            ml_model_type = ModelType.RANDOM_FOREST  # Robust for risk
        else:
            ml_model_type = ModelType.XGBOOST  # Default
        
        # Get target column
        target_column = self.config.target_columns.get(model_type.value, 'target')
        
        return ModelTrainingConfig(
            model_name=f"{self.config.tactician_name}_{model_type.value}",
            task_type=task_type,
            model_type=ml_model_type,
            output_dir=f"{self.config.output_dir}/{model_type.value}",
            feature_columns=self.config.feature_columns,
            target_column=target_column,
            validation_split=self.config.validation_split,
            test_split=self.config.test_split,
            enable_hyperparameter_optimization=True,  # Always enable for tactical models
            hpo_trials=self.config.hpo_trials,
            enable_early_stopping=self.config.enable_early_stopping,
            early_stopping_patience=self.config.early_stopping_patience,
            enable_gpu_acceleration=self.config.enable_gpu_acceleration,
            enable_memory_optimization=self.config.enable_memory_optimization,
            enable_parallel_processing=self.config.enable_parallel_processing,
            memory_limit_gb=self.config.memory_limit_gb,
            max_workers=self.config.max_workers,
            cross_validation_folds=self.config.cross_validation_folds
        )
    
    async def _advanced_hyperparameter_optimization_enhanced(
        self, 
        data: pd.DataFrame, 
        model_type: TacticianModelType,
        training_config: ModelTrainingConfig
    ) -> Dict[str, Any]:
        """Perform advanced hyperparameter optimization using ML commons."""
        
        self.logger.info(f"🔬 Starting enhanced HPO for {model_type.value} using ML commons...")
        
        # Use ML commons HPO optimizer for enhanced optimization
        try:
            hpo_result = await self.hpo_optimizer.optimize(
                model_type=training_config.model_type.value,
                X_train=data[self.config.feature_columns],
                y_train=data[self.config.target_columns[model_type.value]],
                n_trials=self.config.hpo_trials,
                timeout=self.config.hpo_timeout,
                task_type=training_config.task_type.value,
                optimization_objectives=self.config.optimization_objectives
            )
            
            self.logger.info(f"✅ Enhanced HPO completed for {model_type.value}")
            return hpo_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced HPO failed, using fallback: {e}")
            # Fallback to basic HPO
            return {'best_params': {}, 'best_score': 0.0}
    
    async def _multi_objective_optimization_enhanced(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform multi-objective optimization using ML commons."""
        
        self.logger.info("🔬 Starting enhanced multi-objective optimization using ML commons...")
        
        # Use ML commons for enhanced multi-objective optimization
        try:
            result = await self.hpo_optimizer.multi_objective_optimization(
                data=data,
                objectives=self.config.optimization_objectives,
                n_trials=self.config.hpo_trials
            )
            
            self.logger.info("✅ Enhanced multi-objective optimization completed")
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced multi-objective optimization failed: {e}")
            return {'pareto_front': [], 'best_compromise_solution': {}}
    
    async def _walk_forward_validation_enhanced(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform walk-forward validation using ML commons."""
        
        self.logger.info("🔄 Starting enhanced walk-forward validation using ML commons...")
        
        # Use ML commons cross-validation utilities for enhanced walk-forward validation
        try:
            walk_forward_results = await self.cv_utils.walk_forward_validation(
                data=data,
                model_types=[mt.value for mt in self.config.model_types],
                n_windows=self.config.walk_forward_windows,
                validation_split=self.config.validation_split
            )
            
            self.logger.info(f"✅ Enhanced walk-forward validation completed: {len(walk_forward_results)} windows")
            return walk_forward_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced walk-forward validation failed: {e}")
            return []
    
    async def _generate_tactical_insights_enhanced(
        self, 
        individual_results: Dict[str, Any], 
        model_performance: Dict[str, Dict[str, float]],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate tactical insights using ML commons."""
        
        insights = {
            'strategy_effectiveness': {},
            'market_conditions': {},
            'risk_factors': {},
            'optimization_opportunities': {}
        }
        
        # Analyze strategy effectiveness
        for model_name, performance in model_performance.items():
            if 'accuracy' in performance:
                insights['strategy_effectiveness'][model_name] = {
                    'accuracy': performance['accuracy'],
                    'effectiveness': 'High' if performance['accuracy'] > 0.7 else 'Medium' if performance['accuracy'] > 0.5 else 'Low'
                }
        
        # Enhanced market conditions analysis using ML commons
        try:
            market_analysis = await self.data_quality.analyze_feature_stability(data)
            insights['market_conditions'].update(market_analysis)
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced market analysis failed: {e}")
        
        # Generate optimization opportunities
        best_model = max(model_performance.items(), key=lambda x: x[1].get('accuracy', 0))
        insights['optimization_opportunities']['best_performing_model'] = best_model[0]
        insights['optimization_opportunities']['improvement_potential'] = 1.0 - best_model[1].get('accuracy', 0)
        
        return insights
    
    def _calculate_tactical_metrics(
        self, 
        model_performance: Dict[str, Dict[str, float]], 
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate tactical performance metrics."""
        
        tactical_metrics = {}
        
        # Aggregate model performance
        if model_performance:
            accuracies = [metrics.get('accuracy', 0) for metrics in model_performance.values()]
            tactical_metrics['average_accuracy'] = np.mean(accuracies)
            tactical_metrics['best_accuracy'] = np.max(accuracies)
            tactical_metrics['model_count'] = len(model_performance)
        
        # Calculate tactical-specific metrics
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            if len(returns) > 0:
                tactical_metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                tactical_metrics['max_drawdown'] = abs(returns.cumsum().expanding().max().sub(returns.cumsum()).max())
        
        return tactical_metrics
    
    def _calculate_overall_performance(
        self, 
        model_performance: Dict[str, Dict[str, float]], 
        tactical_metrics: Dict[str, float],
        walk_forward_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        
        overall_metrics = {}
        
        # Model performance
        if model_performance:
            overall_metrics['average_model_performance'] = np.mean([
                metrics.get('accuracy', 0) for metrics in model_performance.values()
            ])
        
        # Tactical metrics
        overall_metrics.update(tactical_metrics)
        
        # Walk-forward stability
        if walk_forward_results:
            window_performances = []
            for window in walk_forward_results:
                window_avg = np.mean([
                    results.get('accuracy', 0) for results in window['results'].values()
                ])
                window_performances.append(window_avg)
            
            overall_metrics['walk_forward_stability'] = 1.0 - np.std(window_performances) if window_performances else 0.0
            overall_metrics['walk_forward_average'] = np.mean(window_performances) if window_performances else 0.0
        
        return overall_metrics
    
    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
        
        optimizations.append("ml_commons_integration")
        
        return optimizations
    
    async def save_tactician_models(self, results: TacticianTrainingResults) -> None:
        """Save all trained Tactician models using ML commons registry."""
        
        try:
            # Save individual models using ML commons registry
            for model_name, model_result in results.individual_models.items():
                if model_result.trained_model is not None:
                    model_path = f"{self.config.output_dir}/{model_name}_model.pkl"
                    await self.model_registry.save_model(
                        model=model_result.trained_model,
                        model_name=model_name,
                        file_path=model_path,
                        metadata={
                            'tactician_name': self.config.tactician_name,
                            'model_type': model_name,
                            'primary_strategy': self.config.primary_strategy.value,
                            'training_time': datetime.now().isoformat()
                        }
                    )
            
            # Save tactical insights
            if self.config.save_tactical_insights:
                insights_path = f"{self.config.output_dir}/tactical_insights.json"
                await safe_json_dump(insights_path, results.tactical_insights)
            
            # Save results metadata
            results_path = f"{self.config.output_dir}/training_results.json"
            await safe_json_dump(results_path, results.__dict__)
            
            self.logger.info(f"💾 All Tactician models saved to {self.config.output_dir} using ML commons registry")
            
        except Exception as e:
            self.logger.error(f"Error saving Tactician models: {e}")
            raise