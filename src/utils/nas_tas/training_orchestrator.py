"""
Training Orchestrator

Orchestrates the complete model training pipeline including regime detection,
model training, selection, and management for the NAS-TAS system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from src.utils.nas_tas.shared_logging import (
    TPRINT_AVAILABLE,
    tprint,
    tprint_debug,
    tprint_error,
    tprint_info,
    tprint_performance,
    tprint_progress,
    tprint_success,
    tprint_timer,
    tprint_warning,
)
from src.utils.nas_tas.shared_serialization import JSONSerializer, PickleSerializer
from src.utils.nas_tas.shared_services import (
    DataValidationResult,
    FeatureEngineeringResult,
    SharedOrchestrationServices,
    engineer_core_features,
    validate_market_data,
)

# Import components
from .regime_aware_trainer import RegimeAwareTrainer, RegimeAwareTrainingConfig, RegimeTrainingResult
from .model_selector import ModelSelector, ModelSelectionConfig, ModelSelectionResult
from .model_manager import ModelManager, ModelManagerConfig
from .performance_tracker import PerformanceTracker, PerformanceConfig

# Import advanced overfitting detection
from src.utils.nas_tas.advanced_overfitting_detection import (
    EnhancedOverfittingDetectorWithLearningCurves,
    OverfittingConfig,
    OverfittingReport
)

# Import market analysis modules for enhanced compatibility
try:
    from src.utils.nas_tas.core.hybrid_regime_detector import (
        HybridNASTASRegimeDetector,
        HybridRegimeConfig,
    )
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
    """Configuration for training orchestrator."""
    
    # Orchestration mode
    mode: OrchestrationMode = OrchestrationMode.FULL_PIPELINE
    
    # Component configurations
    training_config: RegimeAwareTrainingConfig = field(default_factory=RegimeAwareTrainingConfig)
    selection_config: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    manager_config: ModelManagerConfig = field(default_factory=ModelManagerConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Pipeline settings
    enable_regime_detection: bool = True
    enable_model_training: bool = True
    enable_model_selection: bool = True
    enable_model_management: bool = True
    enable_performance_tracking: bool = True
    
    # Data settings
    data_validation: bool = True
    feature_engineering: bool = True
    data_preprocessing: bool = True
    
    # Training settings
    enable_hyperparameter_optimization: bool = True
    enable_cross_validation: bool = True
    enable_ensemble_training: bool = True
    
    # Evaluation settings
    enable_backtesting: bool = True
    enable_walk_forward_analysis: bool = True
    enable_performance_attribution: bool = True
    
    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_directory: str = "orchestrator_results"
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Advanced settings
    enable_parallel_processing: bool = False
    max_workers: int = 4
    enable_caching: bool = True
    cache_directory: str = "orchestrator_cache"

    # Hybrid regime detection
    enable_hybrid_regime_detection: bool = True
    hybrid_regime_weight_tas: float = 0.4
    hybrid_regime_weight_nas: float = 0.6
    
    # Overfitting detection
    enable_overfitting_detection: bool = True
    overfitting_config: Optional[OverfittingConfig] = None


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
    overfitting_results: Optional[Dict[str, Any]] = None
    
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
        self.services = SharedOrchestrationServices()
        self.last_data_validation: Optional[DataValidationResult] = None
        self.last_feature_engineering: Optional[FeatureEngineeringResult] = None
        tprint(f"📊 Config: mode={config.mode.value}, regime_detection={config.enable_regime_detection}", color="cyan")

        self._validate_configuration()

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
        
        # Initialize overfitting detection
        if config.enable_overfitting_detection:
            tprint("🔍 Initializing overfitting detection", color="yellow")
            self.overfitting_detector = EnhancedOverfittingDetectorWithLearningCurves(config.overfitting_config)
            self.logger.info("✅ Overfitting detector initialized")
            tprint("✅ Overfitting detector created", color="green")
        else:
            self.overfitting_detector = None
            tprint("⏭️ Overfitting detection disabled", color="cyan")
        
        self.logger.info("✅ Training Orchestrator initialized")
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Components enabled:")
        self.logger.info(f"     - Regime detection: {config.enable_regime_detection}")
        self.logger.info(f"     - Hybrid regime detection: {config.enable_hybrid_regime_detection}")
        self.logger.info(f"     - Model training: {config.enable_model_training}")
        self.logger.info(f"     - Model selection: {config.enable_model_selection}")
        self.logger.info(f"     - Model management: {config.enable_model_management}")
        self.logger.info(f"     - Performance tracking: {config.enable_performance_tracking}")
        
        tprint("✅ Training Orchestrator initialization complete", color="green")
        tprint(f"🎯 Mode: {config.mode.value}, Components: regime={config.enable_regime_detection}, training={config.enable_model_training}, selection={config.enable_model_selection}", color="cyan")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(output_dir / "orchestrator.log")
        file_handler.setFormatter(formatter)

        logger = self.logger
        logger.setLevel(log_level)
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    def _initialize_components(self):
        """Initialize orchestration components."""
        tprint("🔧 Starting component initialization", color="yellow")
        try:
            services = SharedOrchestrationServices()
            # Initialize trainer
            if self.config.enable_model_training:
                tprint("🎓 Creating regime-aware trainer", color="yellow")
                trainer = RegimeAwareTrainer(self.config.training_config)
                services = services.with_updates(trainer=trainer)
                self.logger.info("✅ Regime-aware trainer initialized")
                tprint("✅ Regime-aware trainer created", color="green")
            else:
                tprint("⏭️ Model training disabled, skipping trainer", color="cyan")

            # Initialize selector
            if self.config.enable_model_selection:
                tprint("🎯 Creating model selector", color="yellow")
                selector = ModelSelector(self.config.selection_config)
                services = services.with_updates(selector=selector)
                self.logger.info("✅ Model selector initialized")
                tprint("✅ Model selector created", color="green")
            else:
                tprint("⏭️ Model selection disabled, skipping selector", color="cyan")

            # Initialize manager
            if self.config.enable_model_management:
                tprint("📁 Creating model manager", color="yellow")
                manager = ModelManager(self.config.manager_config)
                services = services.with_updates(manager=manager)
                self.logger.info("✅ Model manager initialized")
                tprint("✅ Model manager created", color="green")
            else:
                tprint("⏭️ Model management disabled, skipping manager", color="cyan")

            # Initialize performance tracker
            if self.config.enable_performance_tracking:
                tprint("📊 Creating performance tracker", color="yellow")
                performance_tracker = PerformanceTracker(self.config.performance_config)
                services = services.with_updates(performance_tracker=performance_tracker)
                self.logger.info("✅ Performance tracker initialized")
                tprint("✅ Performance tracker created", color="green")
            else:
                tprint("⏭️ Performance tracking disabled, skipping tracker", color="cyan")

            self.services = services
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
        """
        Orchestrate the complete training pipeline.
        
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

        active_feature_columns = list(feature_columns) if feature_columns else None

        try:
            # Initialize result
            result = OrchestrationResult(
                success=False,
                execution_time=0.0,
                mode=self.config.mode,
                start_time=start_time
            )
            
            # Step 1: Data validation and preprocessing
            if self.config.data_validation or self.config.data_preprocessing:
                self.logger.info("📊 Validating and preprocessing data...")
                processed_data = self._validate_and_preprocess_data(
                    market_data, target_variable, feature_columns, timestamps
                )
                if self.last_data_validation:
                    active_feature_columns = self.last_data_validation.feature_columns
            else:
                processed_data = market_data

            # Step 2: Feature engineering
            if self.config.feature_engineering:
                self.logger.info("🔧 Performing feature engineering...")
                processed_data = self._perform_feature_engineering(processed_data, target_variable)
                if self.last_feature_engineering:
                    engineered = self.last_feature_engineering.added_features
                    if active_feature_columns is None:
                        active_feature_columns = [
                            col for col in processed_data.columns if col != target_variable
                        ]
                    else:
                        merged = list(dict.fromkeys(active_feature_columns + engineered))
                        active_feature_columns = [
                            col
                            for col in merged
                            if col in processed_data.columns and col != target_variable
                        ]

            # Step 3: Model training
            training_result = None
            if self.config.enable_model_training and self.services.trainer:
                self.logger.info("🤖 Training regime-aware models...")
                training_result = self._orchestrate_training(
                    processed_data, target_variable, active_feature_columns, timestamps
                )
                result.training_result = training_result
                
                if not training_result.success:
                    result.error_message = f"Training failed: {training_result.error_message}"
                    return result
                
                result.n_regimes_detected = training_result.n_regimes_detected
                result.n_models_trained = len(training_result.models_trained)
                
                # Step 3.5: Overfitting detection
                if self.config.enable_overfitting_detection and self.overfitting_detector:
                    self.logger.info("🔍 Performing overfitting detection...")
                    overfitting_results = self._detect_overfitting_in_training_result(
                        training_result,
                        processed_data,
                        target_variable,
                        active_feature_columns,
                    )
                    result.overfitting_results = overfitting_results
            
            # Step 4: Model selection setup
            if self.config.enable_model_selection and self.services.selector and training_result:
                self.logger.info("🎯 Setting up model selection...")
                self._setup_model_selection(training_result)
            
            # Step 5: Model management
            management_result = None
            if self.config.enable_model_management and self.services.manager and training_result:
                self.logger.info("📦 Managing trained models...")
                management_result = self._orchestrate_model_management(training_result)
                result.management_result = management_result
            
            # Step 6: Performance tracking
            performance_result = None
            if self.config.enable_performance_tracking and self.services.performance_tracker:
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

    def _validate_configuration(self) -> None:
        """Ensure incompatible configuration options are rejected early."""

        errors: List[str] = []

        if self.config.enable_model_selection and not self.config.enable_model_training:
            errors.append("Model selection requires model training to be enabled")
        if self.config.enable_model_management and not self.config.enable_model_training:
            errors.append("Model management requires model training to be enabled")
        if self.config.enable_performance_tracking and not self.config.enable_model_training:
            errors.append("Performance tracking requires model training to be enabled")
        if self.config.enable_backtesting and not self.config.enable_model_training:
            errors.append("Backtesting requires model training to be enabled")
        if (
            self.config.enable_hybrid_regime_detection
            and not self.config.enable_regime_detection
        ):
            errors.append(
                "Hybrid regime detection cannot be enabled when regime detection is disabled"
            )

        if errors:
            message = "Invalid orchestrator configuration:\n - " + "\n - ".join(errors)
            raise ValueError(message)

    def _validate_and_preprocess_data(self,
                                    market_data: pd.DataFrame,
                                    target_variable: str,
                                    feature_columns: Optional[List[str]],
                                    timestamps: Optional[pd.Series]) -> pd.DataFrame:
        """Validate and preprocess market data."""
        try:
            validation_result = validate_market_data(
                market_data,
                target_variable,
                feature_columns,
                logger=self.logger,
            )
            self.last_data_validation = validation_result
            return validation_result.data
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _perform_feature_engineering(self, 
                                   market_data: pd.DataFrame,
                                   target_variable: str) -> pd.DataFrame:
        """Perform feature engineering on market data."""
        try:
            engineering_result = engineer_core_features(
                market_data,
                logger=self.logger,
            )
            self.last_feature_engineering = engineering_result
            return engineering_result.data
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            self.logger.warning(
                "⚠️ Returning original data - feature engineering will be skipped, which may impact model performance"
            )
            return market_data
    
    def _detect_overfitting_in_training_result(self, 
                                             training_result: RegimeTrainingResult,
                                             market_data: pd.DataFrame,
                                             target_variable: str,
                                             feature_columns: Optional[List[str]]) -> Dict[str, Any]:
        """Detect overfitting in training results."""
        try:
            overfitting_results = {}
            
            # Prepare data for overfitting detection
            if feature_columns is None:
                feature_columns = [
                    col for col in market_data.columns if col != target_variable
                ]
            else:
                feature_columns = [
                    col
                    for col in feature_columns
                    if col in market_data.columns and col != target_variable
                ]

            if not feature_columns:
                raise ValueError("No feature columns available for overfitting detection")

            X = market_data[feature_columns].to_numpy()
            y = market_data[target_variable].to_numpy()

            n_samples = len(market_data)
            split_issue: Optional[str] = None
            X_train = X_val = y_train = y_val = None  # type: ignore[assignment]

            if n_samples < 5:
                split_issue = "insufficient_samples"
                self.logger.warning(
                    "⚠️ Insufficient samples (%d) for overfitting detection", n_samples
                )
            else:
                split_index = max(int(n_samples * 0.8), 1)
                if split_index >= n_samples:
                    split_index = n_samples - 1

                if split_index <= 0 or split_index >= n_samples:
                    split_issue = "split_failed"
                    self.logger.warning(
                        "⚠️ Unable to determine a stable validation split for overfitting detection"
                    )
                else:
                    X_train, X_val = X[:split_index], X[split_index:]
                    y_train, y_val = y[:split_index], y[split_index:]
                    if len(X_val) == 0 or len(X_train) == 0:
                        split_issue = "split_failed"
                        self.logger.warning(
                            "⚠️ Validation split produced empty training or validation data"
                        )

            # Check each regime's models for overfitting
            for regime_id, models in training_result.models_trained.items():
                regime_overfitting = {}

                for model_type, model_info in models.items():
                    if not isinstance(model_info, dict) or 'model' not in model_info:
                        continue

                    model = model_info['model']

                    # Perform overfitting detection
                    try:
                        if split_issue:
                            raise RuntimeError(split_issue)

                        overfitting_report = self.overfitting_detector.detect_overfitting_with_learning_curves(
                            model=model,
                            X_train=X_train,
                            X_val=X_val,
                            y_train=y_train,
                            y_val=y_val,
                            model_name=f"regime_{regime_id}_{model_type}",
                            model_type=model_type,
                            fold_number=regime_id
                        )

                        regime_overfitting[model_type] = {
                            'overfitting_detected': overfitting_report.overfitting_detected,
                            'severity': overfitting_report.severity,
                            'indicators': overfitting_report.indicators,
                            'warnings': overfitting_report.warnings,
                            'recommendations': overfitting_report.recommendations
                        }

                    except Exception as e:
                        if split_issue and isinstance(e, RuntimeError):
                            if split_issue == "insufficient_samples":
                                message = (
                                    "Not enough samples to perform hold-out overfitting detection"
                                )
                            elif split_issue == "split_failed":
                                message = (
                                    "Unable to produce a reliable validation split for overfitting detection"
                                )
                            else:
                                message = (
                                    "Hold-out split unavailable; overfitting detection skipped"
                                )
                        else:
                            message = str(e)
                        self.logger.warning(
                            "⚠️ Overfitting detection failed for %s in regime %s: %s",
                            model_type,
                            regime_id,
                            message,
                        )
                        regime_overfitting[model_type] = {
                            'overfitting_detected': False,
                            'severity': 'unknown',
                            'error': message,
                        }

                overfitting_results[f'regime_{regime_id}'] = regime_overfitting

            self.logger.info(f"✅ Overfitting detection completed for {len(overfitting_results)} regimes")
            return overfitting_results
            
        except Exception as e:
            self.logger.error(f"❌ Overfitting detection failed: {e}")
            return {'error': str(e)}
    
    def _orchestrate_training(self, 
                            market_data: pd.DataFrame,
                            target_variable: str,
                            feature_columns: Optional[List[str]],
                            timestamps: Optional[pd.Series]) -> RegimeTrainingResult:
        """Orchestrate model training."""
        try:
            # Train models
            training_result = self.services.trainer.train_models(
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
            self.services.selector.register_models(
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
            management_result = self.services.manager.register_models(training_result.models_trained)
            
            # Deploy models
            deployment_result = self.services.manager.deploy_models()
            
            # Setup monitoring
            monitoring_result = self.services.manager.setup_monitoring()
            
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
                    tracking_result[model_id] = self.services.performance_tracker.setup_model_tracking(
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
        """Orchestrate backtesting evaluation."""
        try:
            # Simple backtesting implementation
            backtest_results = {}
            
            # Test each regime's models
            for regime_id, models in training_result.models_trained.items():
                regime_performance = {}
                
                for model_type, model_info in models.items():
                    if not isinstance(model_info, dict):
                        self.logger.warning(f"⚠️ Invalid model_info for {model_type}: {model_info}")
                        continue

                    model = model_info.get('model')
                    if model is None:
                        self.logger.warning(f"⚠️ No model found for {model_type} in regime {regime_id}")
                        continue

                    # Get test performance
                    test_metrics = model_info.get('test_metrics', {})
                    if not isinstance(test_metrics, dict):
                        test_metrics = {}

                    regime_performance[model_type] = test_metrics.get('f1_score', 0.0)
                
                # Average performance for regime
                if regime_performance:
                    backtest_results[f'regime_{regime_id}'] = np.mean(list(regime_performance.values()))
            
            # Overall backtest performance
            if backtest_results:
                backtest_results['overall'] = np.mean(list(backtest_results.values()))
            
            self.logger.info(f"✅ Backtesting completed - Overall performance: {backtest_results.get('overall', 0):.3f}")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting orchestration failed: {e}")
            return {'overall': 0.0}
    
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
            
            summary_path = output_dir / "orchestration_result.json"
            if not JSONSerializer.save(result_summary, summary_path):
                self.logger.warning(f"⚠️ Failed to persist orchestration summary to {summary_path}")

            # Save detailed results if available
            if result.training_result:
                training_path = output_dir / "training_result.pkl"
                if not PickleSerializer.save(result.training_result, training_path):
                    self.logger.warning(f"⚠️ Failed to persist training results to {training_path}")

            if result.selection_result:
                selection_path = output_dir / "selection_result.pkl"
                if not PickleSerializer.save(result.selection_result, selection_path):
                    self.logger.warning(f"⚠️ Failed to persist selection results to {selection_path}")
            
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
        if not self.services.selector:
            raise ValueError("Model selector not initialized")

        return self.services.selector.select_model(market_data, context=context)

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        return {
            'components_initialized': self.services.summary(),
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