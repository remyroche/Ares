"""
Training Orchestrator

Orchestrates the complete model training pipeline including regime detection,
model training, selection, and management for the NAS-TAS system.
"""

import asyncio
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
import warnings
warnings.filterwarnings('ignore')

# Import tprint for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    # Fallback function if tprint is not available
    def tprint(message: str, color: str = "white", **kwargs):
        print(f"[TRAINING_ORCHESTRATOR] {message}")
    def tprint_debug(message: str, **kwargs):
        print(f"[DEBUG] {message}")
    def tprint_info(message: str, **kwargs):
        print(f"[INFO] {message}")
    def tprint_warning(message: str, **kwargs):
        print(f"[WARNING] {message}")
    def tprint_error(message: str, **kwargs):
        print(f"[ERROR] {message}")
    def tprint_success(message: str, **kwargs):
        print(f"[SUCCESS] {message}")
    def tprint_progress(message: str, **kwargs):
        print(f"[PROGRESS] {message}")
    def tprint_performance(message: str, **kwargs):
        print(f"[PERFORMANCE] {message}")
    def tprint_timer(message: str, **kwargs):
        print(f"[TIMER] {message}")
    TPRINT_AVAILABLE = False

# Import unified NAS/TAS tools
try:
    from src.nas_tas.data.data_processor import UnifiedDataProcessor, DataProcessingConfig
    from src.nas_tas.config.base_config import UnifiedArchitectureConfig, create_comprehensive_config
    from src.nas_tas.evaluation.unified_evaluator import UnifiedEvaluator, EvaluationConfig
    from src.nas_tas.unified_pipeline import UnifiedPipelineConfig, create_nas_pipeline, create_tas_pipeline
    UNIFIED_TOOLS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Unified NAS/TAS tools not available: {e}")
    UNIFIED_TOOLS_AVAILABLE = False

# Import components
from .regime_aware_trainer import RegimeAwareTrainer, RegimeAwareTrainingConfig, RegimeTrainingResult
from .model_selector import ModelSelector, ModelSelectionConfig, ModelSelectionResult
from .model_manager import ModelManager, ModelManagerConfig
from .performance_tracker import PerformanceTracker, PerformanceConfig

# Import market analysis modules for enhanced compatibility
try:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeConfig
    HYBRID_REGIME_AVAILABLE = True
except ImportError:
    HYBRID_REGIME_AVAILABLE = False

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Orchestration modes."""
    FULL_PIPELINE = "full_pipeline"  # Complete pipeline from data to deployment
    TRAINING_ONLY = "training_only"   # Only model training
    SELECTION_ONLY = "selection_only" # Only model selection
    EVALUATION_ONLY = "evaluation_only" # Only evaluation


@dataclass
class OrchestratorConfig:
    """Configuration for training orchestrator using unified system."""
    
    # Unified configuration - primary config source
    unified_config: Optional[UnifiedArchitectureConfig] = None
    
    # Orchestration mode
    mode: OrchestrationMode = OrchestrationMode.FULL_PIPELINE
    
    # Component configurations (legacy support)
    training_config: RegimeAwareTrainingConfig = field(default_factory=RegimeAwareTrainingConfig)
    selection_config: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    manager_config: ModelManagerConfig = field(default_factory=ModelManagerConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Pipeline settings (delegated to unified config when available)
    enable_regime_detection: bool = True
    enable_model_training: bool = True
    enable_model_selection: bool = True
    enable_model_management: bool = True
    enable_performance_tracking: bool = True
    
    # Data settings (delegated to unified data processor)
    data_validation: bool = True
    feature_engineering: bool = True
    data_preprocessing: bool = True
    
    # Training settings (delegated to unified config)
    enable_hyperparameter_optimization: bool = True
    enable_cross_validation: bool = True
    enable_ensemble_training: bool = True
    
    # Evaluation settings (delegated to unified evaluator)
    enable_backtesting: bool = True
    enable_walk_forward_analysis: bool = True
    enable_performance_attribution: bool = True
    
    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_directory: str = "orchestrator_results"
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Advanced settings (delegated to unified config)
    enable_parallel_processing: bool = False
    max_workers: int = 4
    enable_caching: bool = True
    cache_directory: str = "orchestrator_cache"

    # Hybrid regime detection
    enable_hybrid_regime_detection: bool = True
    
    def __post_init__(self):
        """Initialize unified configuration if not provided."""
        if self.unified_config is None and UNIFIED_TOOLS_AVAILABLE:
            # Create comprehensive config as base
            self.unified_config = create_comprehensive_config()
            
            # Apply orchestrator-specific settings
            self.unified_config.architecture_type = ArchitectureType.NEURAL_ONLY
            self.unified_config.optimization_mode = OptimizationMode.REGIME_AWARE
            self.unified_config.n_regimes = 8
            self.unified_config.population_size = 50
            self.unified_config.generations = 100
            
            # Apply pipeline settings
            self.unified_config.enable_parallel_processing = self.enable_parallel_processing
            self.unified_config.max_workers = self.max_workers
            self.unified_config.output_dir = self.output_directory
            self.unified_config.verbose = self.enable_logging
            self.unified_config.save_intermediate_results = self.save_results
            self.unified_config.save_best_models = self.save_models
            
            # Apply data settings
            self.unified_config.enable_feature_engineering = self.feature_engineering
            self.unified_config.enable_data_preprocessing = self.data_preprocessing
            self.unified_config.enable_data_validation = self.data_validation
    
    def get_unified_config(self) -> UnifiedArchitectureConfig:
        """Get or create unified configuration."""
        if self.unified_config is None:
            if UNIFIED_TOOLS_AVAILABLE:
                self.__post_init__()  # Initialize if not done
            else:
                raise RuntimeError("Unified tools not available and no unified config provided")
        
        return self.unified_config
    hybrid_regime_weight_tas: float = 0.4
    hybrid_regime_weight_nas: float = 0.6


@dataclass
class OrchestrationResult:
    """Result from orchestration process."""
    
    # Overall results
    success: bool
    execution_time: float
    mode: OrchestrationMode
    
    # Component results
    training_result: Optional[RegimeTrainingResult] = None
    selection_result: Optional[ModelSelectionResult] = None
    management_result: Optional[Dict[str, Any]] = None
    performance_result: Optional[Dict[str, Any]] = None
    
    # Pipeline metrics
    n_regimes_detected: int = 0
    n_models_trained: int = 0
    n_models_selected: int = 0
    overall_performance: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    configuration: Optional[Dict[str, Any]] = None


class TrainingOrchestrator:
    """
    Training orchestrator for the complete NAS-TAS model training pipeline.
    
    Orchestrates regime detection, model training, selection, and management
    to provide a complete end-to-end solution.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize training orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        tprint("🎯 Initializing Training Orchestrator", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint(f"📊 Config: mode={config.mode.value}, regime_detection={config.enable_regime_detection}", color="cyan")
        
        # Initialize unified tools first
        tprint("🔧 Initializing unified tools", color="yellow")
        self._initialize_unified_tools()
        
        # Set up logging
        tprint("📝 Setting up logging", color="yellow")
        if config.enable_logging:
            self._setup_logging()
        
        # Initialize components
        tprint("🔧 Initializing components", color="yellow")
        self._initialize_components()
        
        # Orchestration state
        tprint("📊 Initializing orchestration state", color="yellow")
        self.current_pipeline_state = {}
        self.execution_history = []
        self.performance_cache = {}
        
        self.logger.info("✅ Training Orchestrator initialized")
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Components enabled:")
        self.logger.info(f"     - Regime detection: {config.enable_regime_detection}")
        self.logger.info(f"     - Hybrid regime detection: {config.enable_hybrid_regime_detection}")
        self.logger.info(f"     - Model training: {config.enable_model_training}")
        self.logger.info(f"     - Model selection: {config.enable_model_selection}")
        self.logger.info(f"   Unified tools available: {UNIFIED_TOOLS_AVAILABLE}")
        self.logger.info(f"     - Model management: {config.enable_model_management}")
        self.logger.info(f"     - Performance tracking: {config.enable_performance_tracking}")
        
        tprint("✅ Training Orchestrator initialization complete", color="green")
        tprint(f"🎯 Mode: {config.mode.value}, Components: regime={config.enable_regime_detection}, training={config.enable_model_training}, selection={config.enable_model_selection}", color="cyan")
    
    def _initialize_unified_tools(self):
        """Initialize unified NAS/TAS tools."""
        if not UNIFIED_TOOLS_AVAILABLE:
            tprint_warning("Unified NAS/TAS tools not available")
            self.unified_data_processor = None
            self.unified_evaluator = None
            self.unified_pipeline = None
            return
        
        try:
            # Get unified configuration
            unified_config = self.config.get_unified_config()
            
            # Initialize unified data processor
            data_config = DataProcessingConfig(
                handle_missing_values=True,
                missing_value_strategy="median",
                handle_outliers=True,
                outlier_method="iqr",
                enable_scaling=True,
                scaling_method="standard",
                enable_feature_engineering=unified_config.enable_feature_engineering,
                create_time_features=True,
                validate_data=unified_config.enable_data_validation,
                min_data_quality_score=0.8
            )
            self.unified_data_processor = UnifiedDataProcessor(data_config)
            tprint_success("Unified data processor initialized")
            
            # Initialize unified evaluator
            eval_config = EvaluationConfig(
                evaluation_type="comprehensive",
                calculate_performance_metrics=True,
                calculate_financial_metrics=True,
                calculate_regime_metrics=True,
                calculate_risk_metrics=True,
                financial_validation=True,
                enable_parallel_evaluation=unified_config.enable_parallel_processing,
                max_workers=unified_config.max_workers
            )
            self.unified_evaluator = UnifiedEvaluator(eval_config)
            tprint_success("Unified evaluator initialized")
            
            # Initialize unified pipeline based on architecture type
            if unified_config.architecture_type == ArchitectureType.NEURAL_ONLY:
                self.unified_pipeline = create_nas_pipeline()
            elif unified_config.architecture_type == ArchitectureType.TREE_ONLY:
                self.unified_pipeline = create_tas_pipeline()
            else:  # HYBRID_NEURAL_TREE
                from src.nas_tas.unified_pipeline import create_hybrid_pipeline
                self.unified_pipeline = create_hybrid_pipeline()
            
            tprint_success("Unified pipeline initialized")
            
        except Exception as e:
            tprint_error(f"Unified tools initialization failed: {e}")
            self.unified_data_processor = None
            self.unified_evaluator = None
            self.unified_pipeline = None
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{self.config.output_directory}/orchestrator.log")
            ]
        )
    
    def _initialize_components(self):
        """Initialize orchestration components."""
        tprint("🔧 Starting component initialization", color="yellow")
        try:
            # Initialize trainer
            if self.config.enable_model_training:
                tprint("🎓 Creating regime-aware trainer", color="yellow")
                self.trainer = RegimeAwareTrainer(self.config.training_config)
                self.logger.info("✅ Regime-aware trainer initialized")
                tprint("✅ Regime-aware trainer created", color="green")
            else:
                self.trainer = None
                tprint("⏭️ Model training disabled, skipping trainer", color="cyan")
            
            # Initialize selector
            if self.config.enable_model_selection:
                tprint("🎯 Creating model selector", color="yellow")
                self.selector = ModelSelector(self.config.selection_config)
                self.logger.info("✅ Model selector initialized")
                tprint("✅ Model selector created", color="green")
            else:
                self.selector = None
                tprint("⏭️ Model selection disabled, skipping selector", color="cyan")
            
            # Initialize manager
            if self.config.enable_model_management:
                tprint("📁 Creating model manager", color="yellow")
                self.manager = ModelManager(self.config.manager_config)
                self.logger.info("✅ Model manager initialized")
                tprint("✅ Model manager created", color="green")
            else:
                self.manager = None
                tprint("⏭️ Model management disabled, skipping manager", color="cyan")
            
            # Initialize performance tracker
            if self.config.enable_performance_tracking:
                tprint("📊 Creating performance tracker", color="yellow")
                self.performance_tracker = PerformanceTracker(self.config.performance_config)
                self.logger.info("✅ Performance tracker initialized")
                tprint("✅ Performance tracker created", color="green")
            else:
                self.performance_tracker = None
                tprint("⏭️ Performance tracking disabled, skipping tracker", color="cyan")
            
            tprint("✅ All components initialized successfully", color="green")
            
        except Exception as e:
            tprint(f"❌ Component initialization failed: {e}", color="red")
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def orchestrate(self,
                   market_data: pd.DataFrame,
                   target_variable: str,
                   feature_columns: Optional[List[str]] = None,
                   timestamps: Optional[pd.Series] = None,
                   context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """Synchronously orchestrate the complete training pipeline.

        This method is the legacy entry point used throughout the codebase. It
        internally drives the asynchronous unified pipeline execution using
        ``asyncio.run`` so callers can continue to invoke it from synchronous
        contexts. When running inside an existing event loop, use
        :meth:`orchestrate_async` instead and ``await`` the result.

        Args:
            market_data: Market data for training.
            target_variable: Name of target variable.
            feature_columns: List of feature columns (``None`` for all except target).
            timestamps: Optional timestamps aligned with the market data.
            context: Additional context for orchestration.

        Returns:
            OrchestrationResult with complete pipeline results.
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            raise RuntimeError(
                "TrainingOrchestrator.orchestrate() cannot be called while an event loop is running. "
                "Use `await orchestrator.orchestrate_async(...)` instead."
            )

        return asyncio.run(
            self.orchestrate_async(
                market_data=market_data,
                target_variable=target_variable,
                feature_columns=feature_columns,
                timestamps=timestamps,
                context=context,
            )
        )

    async def orchestrate_async(self,
                                market_data: pd.DataFrame,
                                target_variable: str,
                                feature_columns: Optional[List[str]] = None,
                                timestamps: Optional[pd.Series] = None,
                                context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """Asynchronously orchestrate the complete training pipeline.

        This mirrors :meth:`orchestrate` but is designed for callers that
        already operate inside an event loop. Legacy synchronous workflows
        should continue using :meth:`orchestrate`, while new asynchronous
        workflows can ``await`` this coroutine directly.

        Args:
            market_data: Market data for training
            target_variable: Name of target variable
            feature_columns: List of feature columns (None for all except target)
            timestamps: Optional timestamps
            context: Additional context

        Returns:
            OrchestrationResult with complete pipeline results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting orchestration pipeline")

        try:
            # Initialize result
            result = OrchestrationResult(
                success=False,
                execution_time=0.0,
                mode=self.config.mode,
                start_time=start_time
            )
            
            # Use unified pipeline if available and appropriate
            if UNIFIED_TOOLS_AVAILABLE and self.unified_pipeline and self.config.mode == OrchestrationMode.FULL_PIPELINE:
                self.logger.info("🔧 Using unified pipeline for orchestration")
                return await self._orchestrate_with_unified_pipeline(
                    market_data, target_variable, feature_columns, timestamps, context, start_time
                )
            
            # Fallback to legacy orchestration
            self.logger.info("🔧 Using legacy orchestration pipeline")
            
            # Step 1: Data validation and preprocessing
            if self.config.data_validation or self.config.data_preprocessing:
                self.logger.info("📊 Validating and preprocessing data...")
                processed_data = self._validate_and_preprocess_data(
                    market_data, target_variable, feature_columns, timestamps
                )
            else:
                processed_data = market_data
            
            # Step 2: Feature engineering
            if self.config.feature_engineering:
                self.logger.info("🔧 Performing feature engineering...")
                processed_data = self._perform_feature_engineering(processed_data, target_variable)
            
            # Step 3: Model training
            training_result = None
            if self.config.enable_model_training and self.trainer:
                self.logger.info("🤖 Training regime-aware models...")
                training_result = self._orchestrate_training(
                    processed_data, target_variable, feature_columns, timestamps
                )
                result.training_result = training_result
                
                if not training_result.success:
                    result.error_message = f"Training failed: {training_result.error_message}"
                    return result
                
                result.n_regimes_detected = training_result.n_regimes_detected
                result.n_models_trained = len(training_result.models_trained)
            
            # Step 4: Model selection setup
            if self.config.enable_model_selection and self.selector and training_result:
                self.logger.info("🎯 Setting up model selection...")
                self._setup_model_selection(training_result)
            
            # Step 5: Model management
            management_result = None
            if self.config.enable_model_management and self.manager and training_result:
                self.logger.info("📦 Managing trained models...")
                management_result = self._orchestrate_model_management(training_result)
                result.management_result = management_result
            
            # Step 6: Performance tracking
            performance_result = None
            if self.config.enable_performance_tracking and self.performance_tracker:
                self.logger.info("📈 Setting up performance tracking...")
                performance_result = self._orchestrate_performance_tracking(training_result)
                result.performance_result = performance_result
            
            # Step 7: Evaluation and backtesting
            if self.config.enable_backtesting and training_result:
                self.logger.info("🧪 Performing backtesting...")
                backtest_result = self._orchestrate_backtesting(
                    processed_data, training_result, timestamps
                )
                result.overall_performance.update(backtest_result)
            
            # Step 8: Save results
            if self.config.save_results:
                self.logger.info("💾 Saving orchestration results...")
                self._save_orchestration_results(result)
            
            # Complete result
            end_time = datetime.now()
            result.end_time = end_time
            result.execution_time = (end_time - start_time).total_seconds()
            result.success = True
            result.configuration = self._get_configuration_summary()
            
            self.logger.info(f"✅ Orchestration completed in {result.execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {result.n_regimes_detected}")
            self.logger.info(f"   Models trained: {result.n_models_trained}")
            self.logger.info(f"   Overall performance: {result.overall_performance}")
            
            # Update execution history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Orchestration failed: {e}")
            
            return OrchestrationResult(
                success=False,
                execution_time=execution_time,
                mode=self.config.mode,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def _orchestrate_with_unified_pipeline(self, 
                                               market_data: pd.DataFrame,
                                               target_variable: str,
                                               feature_columns: Optional[List[str]],
                                               timestamps: Optional[pd.Series],
                                               context: Optional[Dict[str, Any]],
                                               start_time: datetime) -> OrchestrationResult:
        """Orchestrate using unified pipeline."""
        try:
            # Get unified configuration
            unified_config = self.config.get_unified_config()
            
            # Separate features and target
            if feature_columns is None:
                feature_columns = [col for col in market_data.columns if col != target_variable]
            
            X = market_data[feature_columns]
            y = market_data[target_variable]
            
            # Process data using unified data processor
            if self.unified_data_processor:
                tprint_info("Processing data with unified data processor")
                processed_X, processed_y, validation_result = self.unified_data_processor.process_data(X, y, fit=True)
                
                if not validation_result.validation_passed:
                    tprint_warning(f"Data validation failed: {validation_result.validation_score:.3f}")
                    if validation_result.validation_score < 0.5:
                        raise ValueError(f"Data quality too low: {validation_result.validation_score:.3f}")
            else:
                processed_X, processed_y = X, y
            
            # Execute unified pipeline
            tprint_info("Executing unified pipeline")
            pipeline_result = await self.unified_pipeline.execute_pipeline(
                processed_X, processed_y, unified_config
            )
            
            # Convert unified result to orchestrator result
            result = OrchestrationResult(
                success=pipeline_result.execution_info.status == "completed",
                execution_time=(datetime.now() - start_time).total_seconds(),
                mode=self.config.mode,
                start_time=start_time,
                end_time=datetime.now()
            )
            
            if pipeline_result.execution_info.status == "completed":
                result.n_regimes_detected = getattr(pipeline_result, 'n_regimes', 0)
                result.n_models_trained = len(getattr(pipeline_result, 'models', []))
                result.best_model = getattr(pipeline_result, 'best_model', None)
                result.performance_metrics = getattr(pipeline_result, 'performance_metrics', {})
                
                tprint_success("Unified pipeline execution completed successfully")
            else:
                result.error_message = f"Pipeline failed: {pipeline_result.execution_info.error_message}"
                tprint_error(f"Unified pipeline failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            tprint_error(f"Unified pipeline orchestration failed: {e}")
            return OrchestrationResult(
                success=False,
                execution_time=(datetime.now() - start_time).total_seconds(),
                mode=self.config.mode,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
    
    def _validate_and_preprocess_data(self, 
                                    market_data: pd.DataFrame,
                                    target_variable: str,
                                    feature_columns: Optional[List[str]],
                                    timestamps: Optional[pd.Series]) -> pd.DataFrame:
        """Validate and preprocess market data using unified data processor."""
        try:
            # Check if target variable exists
            if target_variable not in market_data.columns:
                raise ValueError(f"Target variable '{target_variable}' not found in data")
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in market_data.columns if col != target_variable]
            
            # Use unified data processor if available
            if UNIFIED_TOOLS_AVAILABLE:
                tprint_info("Using unified data processor for validation and preprocessing")
                
                # Initialize unified data processor
                data_config = DataProcessingConfig(
                    handle_missing_values=True,
                    missing_value_strategy="median",
                    handle_outliers=True,
                    outlier_method="iqr",
                    enable_scaling=True,
                    scaling_method="standard",
                    enable_feature_engineering=True,
                    create_time_features=True,
                    validate_data=True,
                    min_data_quality_score=0.7
                )
                
                processor = UnifiedDataProcessor(data_config)
                
                # Separate features and target
                X = market_data[feature_columns]
                y = market_data[target_variable]
                
                # Process data
                processed_X, processed_y, validation_result = processor.process_data(X, y, fit=True)
                
                # Check validation results
                if not validation_result.validation_passed:
                    tprint_warning(f"Data validation failed: {validation_result.validation_score:.3f}")
                    if validation_result.validation_score < 0.5:
                        raise ValueError(f"Data quality too low: {validation_result.validation_score:.3f}")
                
                # Reconstruct DataFrame
                processed_data = pd.DataFrame(processed_X, columns=feature_columns, index=market_data.index)
                processed_data[target_variable] = processed_y
                
                tprint_success(f"Unified data processing completed - Shape: {processed_data.shape}")
                tprint_info(f"Data quality score: {validation_result.validation_score:.3f}")
                
                return processed_data
            
            else:
                # Fallback to original implementation
                tprint_warning("Unified tools not available, using fallback data processing")
                return self._fallback_data_validation(market_data, target_variable, feature_columns)
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _fallback_data_validation(self, 
                                market_data: pd.DataFrame,
                                target_variable: str,
                                feature_columns: Optional[List[str]]) -> pd.DataFrame:
        """Fallback data validation when unified tools are not available."""
        # Original implementation as fallback
        missing_values = market_data.isnull().sum()
        if missing_values.any():
            missing_summary = missing_values[missing_values > 0].to_dict()
            self.logger.warning(f"⚠️ Found missing values in {len(missing_summary)} columns: {missing_summary}")
            try:
                market_data = market_data.ffill().bfill()
                self.logger.info(f"✅ Filled missing values using forward/backward fill")
            except Exception as e:
                self.logger.error(f"❌ Failed to fill missing values: {e}")
                raise
        
        # Check for infinite values
        inf_values = np.isinf(market_data.select_dtypes(include=[np.number])).sum()
        if inf_values.any():
            inf_summary = inf_values[inf_values > 0].to_dict()
            self.logger.warning(f"⚠️ Found infinite values in {len(inf_summary)} columns: {inf_summary}")
            try:
                market_data = market_data.replace([np.inf, -np.inf], np.nan)
                market_data = market_data.ffill().bfill()
                self.logger.info(f"✅ Replaced infinite values and filled using forward/backward fill")
            except Exception as e:
                self.logger.error(f"❌ Failed to handle infinite values: {e}")
                raise
        
        # Check data types
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        non_numeric_features = [col for col in feature_columns if col not in numeric_columns]
        if non_numeric_features:
            self.logger.warning(f"⚠️ Non-numeric feature columns detected: {non_numeric_features}")
            self.logger.info(f"   Total features: {len(feature_columns)}, Numeric features: {len(numeric_columns)}")
        
        self.logger.info(f"✅ Fallback data validation completed - Shape: {market_data.shape}")
        return market_data
    
    def _perform_feature_engineering(self, 
                                   market_data: pd.DataFrame,
                                   target_variable: str) -> pd.DataFrame:
        """Perform feature engineering on market data."""
        try:
            # Create a copy to avoid modifying original data
            data = market_data.copy()
            
            # Technical indicators
            if 'close' in data.columns:
                # Price-based features
                data['price_change'] = data['close'].pct_change()
                data['price_volatility'] = data['price_change'].rolling(window=20).std()
                data['price_momentum'] = data['close'] / data['close'].shift(20)
                
                # Moving averages
                data['ma_5'] = data['close'].rolling(window=5).mean()
                data['ma_20'] = data['close'].rolling(window=20).mean()
                data['ma_50'] = data['close'].rolling(window=50).mean()
                
                # Price position
                data['price_position_20'] = (data['close'] - data['close'].rolling(window=20).min()) / (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
            
            if 'volume' in data.columns:
                # Volume-based features
                data['volume_change'] = data['volume'].pct_change()
                data['volume_ma'] = data['volume'].rolling(window=20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            if 'high' in data.columns and 'low' in data.columns:
                # Range-based features
                data['price_range'] = (data['high'] - data['low']) / data['close']
                data['range_volatility'] = data['price_range'].rolling(window=20).std()
            
            # Time-based features
            if data.index.dtype == 'datetime64[ns]':
                data['hour'] = data.index.hour
                data['day_of_week'] = data.index.dayofweek
                data['month'] = data.index.month
            
            # Remove rows with NaN values created by rolling operations
            data = data.dropna()
            
            self.logger.info(f"✅ Feature engineering completed - New shape: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            self.logger.warning("⚠️ Returning original data - feature engineering will be skipped, which may impact model performance")
            # Add warning to result warnings if we have access to result object
            if 'result' in locals():
                result.warnings.append(f"Feature engineering failed: {e}")
            return market_data  # Return original data if engineering fails
    
    def _orchestrate_training(self, 
                            market_data: pd.DataFrame,
                            target_variable: str,
                            feature_columns: Optional[List[str]],
                            timestamps: Optional[pd.Series]) -> RegimeTrainingResult:
        """Orchestrate model training."""
        try:
            # Train models
            training_result = self.trainer.train_models(
                market_data=market_data,
                target_variable=target_variable,
                feature_columns=feature_columns,
                timestamps=timestamps
            )
            
            if training_result.success:
                self.logger.info(f"✅ Training completed - {training_result.n_regimes_detected} regimes, {len(training_result.models_trained)} models")
            else:
                self.logger.error(f"❌ Training failed: {training_result.error_message}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"❌ Training orchestration failed: {e}")
            return RegimeTrainingResult(
                success=False,
                training_time=0.0,
                n_regimes_detected=0,
                models_trained={},
                error_message=str(e)
            )
    
    def _setup_model_selection(self, training_result: RegimeTrainingResult):
        """Setup model selection with trained models."""
        try:
            # Register models with selector
            self.selector.register_models(
                regime_models=training_result.models_trained,
                ensemble_models=training_result.ensemble_models
            )
            
            self.logger.info("✅ Model selection setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Model selection setup failed: {e}")
            raise
    
    def _orchestrate_model_management(self, training_result: RegimeTrainingResult) -> Dict[str, Any]:
        """Orchestrate model management."""
        try:
            # Register models with manager
            management_result = self.manager.register_models(training_result.models_trained)
            
            # Deploy models
            deployment_result = self.manager.deploy_models()
            
            # Setup monitoring
            monitoring_result = self.manager.setup_monitoring()
            
            result = {
                'registration': management_result,
                'deployment': deployment_result,
                'monitoring': monitoring_result
            }
            
            self.logger.info("✅ Model management orchestration completed")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model management orchestration failed: {e}")
            error_result = {'error': str(e), 'success': False}
            self.logger.warning("⚠️ Model management failed - models will not be registered or deployed")
            return error_result
    
    def _orchestrate_performance_tracking(self, training_result: RegimeTrainingResult) -> Dict[str, Any]:
        """Orchestrate performance tracking setup."""
        try:
            # Setup performance tracking for all models
            tracking_result = {}
            
            for regime_id, models in training_result.models_trained.items():
                for model_type, model_info in models.items():
                    model_id = f"regime_{regime_id}_{model_type}"
                    
                    # Setup tracking for this model
                    tracking_result[model_id] = self.performance_tracker.setup_model_tracking(
                        model_id=model_id,
                        model_info=model_info
                    )
            
            self.logger.info("✅ Performance tracking orchestration completed")
            return tracking_result
            
        except Exception as e:
            self.logger.error(f"❌ Performance tracking orchestration failed: {e}")
            return {'error': str(e)}
    
    def _orchestrate_backtesting(self, 
                               market_data: pd.DataFrame,
                               training_result: RegimeTrainingResult,
                               timestamps: Optional[pd.Series]) -> Dict[str, float]:
        """Orchestrate comprehensive backtesting evaluation."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            import numpy as np
            
            tprint("🔄 Starting comprehensive backtesting evaluation", color="yellow")
            backtest_results = {}
            
            # Backtesting configuration
            backtest_config = {
                'n_splits': 5,  # Number of time series splits
                'test_size': 0.2,  # 20% for testing
                'min_train_size': 100,  # Minimum training samples
                'step_size': 1,  # Step size for rolling window
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'confidence_level': 0.95
            }
            
            # Initialize time series cross-validation
            tscv = TimeSeriesSplit(
                n_splits=backtest_config['n_splits'],
                test_size=int(len(market_data) * backtest_config['test_size'])
            )
            
            # Prepare market data for backtesting
            if timestamps is not None:
                # Sort by timestamps
                sorted_indices = timestamps.argsort()
                market_data_sorted = market_data.iloc[sorted_indices]
            else:
                market_data_sorted = market_data
            
            # Extract features and targets (assuming last column is target)
            X = market_data_sorted.iloc[:, :-1].values
            y = market_data_sorted.iloc[:, -1].values
            
            # Test each regime's models
            for regime_id, models in training_result.models_trained.items():
                tprint(f"📊 Backtesting regime {regime_id}", color="blue")
                regime_performance = {}
                
                for model_type, model_info in models.items():
                    if not isinstance(model_info, dict):
                        self.logger.warning(f"⚠️ Invalid model_info for {model_type}: {model_info}")
                        continue

                    model = model_info.get('model')
                    if model is None:
                        self.logger.warning(f"⚠️ No model found for {model_type} in regime {regime_id}")
                        continue
                    
                    # Perform time series cross-validation
                    cv_scores = []
                    cv_metrics = {metric: [] for metric in backtest_config['metrics']}
                    
                    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                        try:
                            # Split data
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            # Skip if insufficient data
                            if len(X_train) < backtest_config['min_train_size']:
                                continue
                            
                            # Train model on this fold
                            model_copy = self._clone_model(model)
                            model_copy.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model_copy.predict(X_test)
                            
                            # Calculate metrics
                            fold_metrics = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            }
                            
                            # Store metrics
                            for metric, value in fold_metrics.items():
                                cv_metrics[metric].append(value)
                            
                            cv_scores.append(fold_metrics['f1_score'])
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ Fold {fold} failed for {model_type}: {e}")
                            continue
                    
                    # Calculate average performance across folds
                    if cv_scores:
                        regime_performance[model_type] = {
                            'mean_f1': np.mean(cv_scores),
                            'std_f1': np.std(cv_scores),
                            'mean_accuracy': np.mean(cv_metrics['accuracy']),
                            'mean_precision': np.mean(cv_metrics['precision']),
                            'mean_recall': np.mean(cv_metrics['recall']),
                            'n_folds': len(cv_scores),
                            'confidence_interval': self._calculate_confidence_interval(cv_scores, backtest_config['confidence_level'])
                        }
                        
                        tprint(f"   ✅ {model_type}: F1={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}", color="green")
                    else:
                        regime_performance[model_type] = {
                            'mean_f1': 0.0,
                            'std_f1': 0.0,
                            'mean_accuracy': 0.0,
                            'mean_precision': 0.0,
                            'mean_recall': 0.0,
                            'n_folds': 0,
                            'confidence_interval': (0.0, 0.0)
                        }
                
                # Calculate regime-level performance
                if regime_performance:
                    regime_f1_scores = [perf['mean_f1'] for perf in regime_performance.values()]
                    backtest_results[f'regime_{regime_id}'] = {
                        'mean_f1': np.mean(regime_f1_scores),
                        'std_f1': np.std(regime_f1_scores),
                        'model_count': len(regime_performance),
                        'model_performance': regime_performance
                    }
                    
                    tprint(f"   📊 Regime {regime_id} average F1: {np.mean(regime_f1_scores):.3f}±{np.std(regime_f1_scores):.3f}", color="cyan")
            
            # Calculate overall backtest performance
            if backtest_results:
                overall_f1_scores = [result['mean_f1'] for result in backtest_results.values()]
                backtest_results['overall'] = {
                    'mean_f1': np.mean(overall_f1_scores),
                    'std_f1': np.std(overall_f1_scores),
                    'regime_count': len(backtest_results),
                    'total_models': sum(result['model_count'] for result in backtest_results.values())
                }
                
                tprint(f"🎯 Overall backtest F1: {np.mean(overall_f1_scores):.3f}±{np.std(overall_f1_scores):.3f}", color="green")
            
            self.logger.info(f"✅ Comprehensive backtesting completed")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting orchestration failed: {e}")
            return {'overall': {'mean_f1': 0.0, 'std_f1': 0.0, 'regime_count': 0, 'total_models': 0}}
    
    def _clone_model(self, model):
        """Clone a model for cross-validation."""
        try:
            from sklearn.base import clone
            return clone(model)
        except Exception:
            # Fallback: return the original model (not ideal but functional)
            return model
    
    def _calculate_confidence_interval(self, scores, confidence_level):
        """Calculate confidence interval for scores."""
        try:
            import scipy.stats as stats
            n = len(scores)
            mean = np.mean(scores)
            std = np.std(scores)
            se = std / np.sqrt(n)
            
            # Calculate t-statistic
            alpha = 1 - confidence_level
            t_val = stats.t.ppf(1 - alpha/2, n-1)
            
            # Calculate confidence interval
            margin_error = t_val * se
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            
            return (ci_lower, ci_upper)
        except Exception:
            # Fallback: simple standard error
            mean = np.mean(scores)
            std = np.std(scores)
            return (mean - std, mean + std)
    
    def _save_orchestration_results(self, result: OrchestrationResult):
        """Save orchestration results."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_summary = {
                'success': result.success,
                'execution_time': result.execution_time,
                'mode': result.mode.value,
                'n_regimes_detected': result.n_regimes_detected,
                'n_models_trained': result.n_models_trained,
                'overall_performance': result.overall_performance,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'error_message': result.error_message,
                'warnings': result.warnings
            }
            
            with open(output_dir / "orchestration_result.json", 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            # Save detailed results if available
            if result.training_result:
                with open(output_dir / "training_result.pkl", 'wb') as f:
                    pickle.dump(result.training_result, f)
            
            if result.selection_result:
                with open(output_dir / "selection_result.pkl", 'wb') as f:
                    pickle.dump(result.selection_result, f)
            
            self.logger.info(f"✅ Orchestration results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save orchestration results: {e}")
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'mode': self.config.mode.value,
            'training_strategy': self.config.training_config.training_strategy.value,
            'selection_strategy': self.config.selection_config.selection_strategy.value,
            'routing_method': self.config.selection_config.routing_method.value,
            'enable_regime_detection': self.config.enable_regime_detection,
            'enable_hybrid_regime_detection': self.config.enable_hybrid_regime_detection,
            'hybrid_regime_weight_tas': self.config.hybrid_regime_weight_tas,
            'hybrid_regime_weight_nas': self.config.hybrid_regime_weight_nas,
            'enable_model_training': self.config.enable_model_training,
            'enable_model_selection': self.config.enable_model_selection,
            'enable_model_management': self.config.enable_model_management,
            'enable_performance_tracking': self.config.enable_performance_tracking,
            'enable_backtesting': self.config.enable_backtesting
        }
    
    def select_model_for_prediction(self, 
                                  market_data: pd.DataFrame,
                                  context: Optional[Dict[str, Any]] = None) -> ModelSelectionResult:
        """
        Select model for making predictions.
        
        Args:
            market_data: Current market data
            context: Additional context
            
        Returns:
            ModelSelectionResult with selected model
        """
        if not self.selector:
            raise ValueError("Model selector not initialized")
        
        return self.selector.select_model(market_data, context=context)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        return {
            'components_initialized': {
                'trainer': self.trainer is not None,
                'selector': self.selector is not None,
                'manager': self.manager is not None,
                'performance_tracker': self.performance_tracker is not None
            },
            'execution_history': len(self.execution_history),
            'last_execution': self.execution_history[-1].start_time.isoformat() if self.execution_history else None,
            'configuration': self._get_configuration_summary()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executions."""
        if not self.execution_history:
            return {}
        
        successful_executions = [r for r in self.execution_history if r.success]
        
        if not successful_executions:
            return {'error': 'No successful executions found'}
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'average_execution_time': np.mean([r.execution_time for r in successful_executions]),
            'average_regimes_detected': np.mean([r.n_regimes_detected for r in successful_executions]),
            'average_models_trained': np.mean([r.n_models_trained for r in successful_executions]),
            'latest_performance': successful_executions[-1].overall_performance
        }