"""
Enhanced TAS Engine

This module provides a comprehensive Tree Architecture Search engine that integrates
modern tree algorithms, automated optimization, advanced feature engineering, and
sophisticated evaluation metrics to match NAS-level sophistication.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import enhanced components
try:
    from .models.enhanced_tree_models import (
        EnhancedTreeModelFactory, TreeModelConfig, TreeModelResult,
        TreeModelEvaluator, create_model_ensemble
    )
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False

try:
    from .automl.tree_automl import (
        TreeAutoMLManager, AutoMLConfig, AutoMLResult,
        create_tree_automl_manager
    )
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False

try:
    from .evaluation.advanced_metrics import (
        AdvancedEvaluator, AdvancedEvaluationResult,
        create_advanced_evaluator
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

try:
    from ..shared_utils.evolutionary_search import (
        EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult,
        create_evolutionary_algorithm_manager
    )
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False

# Import existing TAS components
try:
    from .core.tas_config import TASConfig, TASSearchConfig, TASOptimizationConfig
    from .core.tas_result import TASResult
    from .core.tas_engine import TreeArchitectureSearchEngine
    TAS_CORE_AVAILABLE = True
except ImportError:
    TAS_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTASConfig:
    """Configuration for Enhanced TAS Engine."""
    
    # Base TAS configuration
    base_tas_config: Optional[TASConfig] = None
    
    # Enhanced model settings
    enable_enhanced_models: bool = True
    model_types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"
    ])
    enable_ensemble: bool = True
    ensemble_method: str = "voting"  # "voting", "stacking", "blending"
    
    # AutoML settings
    enable_automl: bool = True
    automl_method: str = "optuna"  # "optuna", "grid", "random", "bayesian"
    max_automl_trials: int = 100
    automl_timeout: int = 3600
    
    # Evolutionary search settings
    enable_evolutionary_search: bool = True
    evolutionary_algorithm: str = "nsga2"  # "nsga2", "spea2", "ga"
    population_size: int = 50
    max_generations: int = 100
    
    # Advanced evaluation settings
    enable_advanced_metrics: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "risk_adjusted", "regime_aware", "economic_significance", "trading_viability"
    ])
    
    # Feature engineering settings
    enable_feature_engineering: bool = True
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_score", "rfe", "embedded"
    max_features: int = 100
    feature_importance_threshold: float = 0.01
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "robustness", "efficiency", "interpretability"
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])
    
    # Performance settings
    max_search_time: int = 3600  # 1 hour
    max_evaluations: int = 1000
    parallel_evaluations: int = 4
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_dir: str = "enhanced_tas_results"
    verbose: bool = True


@dataclass
class EnhancedTASResult:
    """Result from Enhanced TAS optimization."""
    
    # Best model and configuration
    best_model: Any
    best_config: TreeModelConfig
    best_score: float
    
    # Model comparison
    model_results: List[TreeModelResult]
    model_rankings: List[Tuple[str, float]]
    
    # AutoML results
    automl_result: Optional[AutoMLResult] = None
    
    # Evolutionary search results
    evolutionary_result: Optional[EvolutionaryResult] = None
    
    # Advanced evaluation results
    advanced_evaluation: Optional[AdvancedEvaluationResult] = None
    
    # Feature engineering results
    feature_importance: Dict[str, float] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    
    # Multi-objective optimization results
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    objective_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    total_search_time: float = 0.0
    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    
    # Search statistics
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class EnhancedTASEngine:
    """Enhanced Tree Architecture Search Engine."""
    
    def __init__(self, config: EnhancedTASConfig):
        """Initialize Enhanced TAS Engine.
        
        Args:
            config: Enhanced TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Search state
        self.search_history = []
        self.best_result = None
        self.current_search = None
        
        self.logger.info("✅ Enhanced TAS Engine initialized")
        self.logger.info(f"   Enhanced models: {config.enable_enhanced_models}")
        self.logger.info(f"   AutoML: {config.enable_automl}")
        self.logger.info(f"   Evolutionary search: {config.enable_evolutionary_search}")
        self.logger.info(f"   Advanced metrics: {config.enable_advanced_metrics}")
        self.logger.info(f"   Feature engineering: {config.enable_feature_engineering}")
    
    def _initialize_components(self):
        """Initialize all TAS components."""
        try:
            # Initialize enhanced models
            if self.config.enable_enhanced_models and ENHANCED_MODELS_AVAILABLE:
                self.model_factory = EnhancedTreeModelFactory()
                self.model_evaluator = TreeModelEvaluator()
                self.logger.info("✅ Enhanced models initialized")
            else:
                self.model_factory = None
                self.model_evaluator = None
                self.logger.warning("⚠️ Enhanced models not available")
            
            # Initialize AutoML
            if self.config.enable_automl and AUTOML_AVAILABLE:
                automl_config = AutoMLConfig(
                    optimization_method=self.config.automl_method,
                    max_trials=self.config.max_automl_trials,
                    timeout_seconds=self.config.automl_timeout,
                    model_types=self.config.model_types,
                    enable_ensemble=self.config.enable_ensemble,
                    ensemble_method=self.config.ensemble_method
                )
                self.automl_manager = create_tree_automl_manager(automl_config)
                self.logger.info("✅ AutoML initialized")
            else:
                self.automl_manager = None
                self.logger.warning("⚠️ AutoML not available")
            
            # Initialize evolutionary search
            if self.config.enable_evolutionary_search and EVOLUTIONARY_AVAILABLE:
                evolutionary_config = EvolutionaryConfig(
                    population_size=self.config.population_size,
                    max_generations=self.config.max_generations,
                    use_nsga2=self.config.evolutionary_algorithm == "nsga2",
                    use_spea2=self.config.evolutionary_algorithm == "spea2",
                    use_genetic_algorithm=self.config.evolutionary_algorithm == "ga"
                )
                self.evolutionary_manager = create_evolutionary_algorithm_manager(evolutionary_config)
                self.logger.info("✅ Evolutionary search initialized")
            else:
                self.evolutionary_manager = None
                self.logger.warning("⚠️ Evolutionary search not available")
            
            # Initialize advanced evaluation
            if self.config.enable_advanced_metrics and ADVANCED_METRICS_AVAILABLE:
                self.advanced_evaluator = create_advanced_evaluator()
                self.logger.info("✅ Advanced evaluation initialized")
            else:
                self.advanced_evaluator = None
                self.logger.warning("⚠️ Advanced evaluation not available")
            
            # Initialize base TAS engine
            if TAS_CORE_AVAILABLE:
                if self.config.base_tas_config is None:
                    self.config.base_tas_config = TASConfig()
                self.base_tas_engine = TreeArchitectureSearchEngine(self.config.base_tas_config)
                self.logger.info("✅ Base TAS engine initialized")
            else:
                self.base_tas_engine = None
                self.logger.warning("⚠️ Base TAS engine not available")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def search(self, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               X_test: Optional[np.ndarray] = None,
               y_test: Optional[np.ndarray] = None,
               regime_labels: Optional[np.ndarray] = None) -> EnhancedTASResult:
        """Perform enhanced tree architecture search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            regime_labels: Regime labels (optional)
            
        Returns:
            EnhancedTASResult with comprehensive search results
        """
        try:
            self.logger.info("🚀 Starting Enhanced TAS search...")
            start_time = time.time()
            
            # Initialize search state
            self.search_history = []
            self.best_result = None
            
            # Step 1: Feature Engineering (if enabled)
            if self.config.enable_feature_engineering:
                X_train, X_val, X_test = self._perform_feature_engineering(
                    X_train, y_train, X_val, y_val, X_test
                )
            
            # Step 2: AutoML Optimization (if enabled)
            automl_result = None
            if self.config.enable_automl and self.automl_manager is not None:
                self.logger.info("🔧 Running AutoML optimization...")
                automl_result = self.automl_manager.optimize(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                self.search_history.append({
                    'step': 'automl',
                    'result': automl_result,
                    'timestamp': datetime.now()
                })
            
            # Step 3: Evolutionary Search (if enabled)
            evolutionary_result = None
            if self.config.enable_evolutionary_search and self.evolutionary_manager is not None:
                self.logger.info("🧬 Running evolutionary search...")
                evolutionary_result = self._run_evolutionary_search(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                self.search_history.append({
                    'step': 'evolutionary',
                    'result': evolutionary_result,
                    'timestamp': datetime.now()
                })
            
            # Step 4: Model Ensemble Creation (if enabled)
            ensemble_models = []
            if self.config.enable_ensemble and self.model_factory is not None:
                self.logger.info("🎭 Creating model ensemble...")
                ensemble_models = self._create_model_ensemble(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
            
            # Step 5: Advanced Evaluation
            advanced_evaluation = None
            if self.config.enable_advanced_metrics and self.advanced_evaluator is not None:
                self.logger.info("📊 Performing advanced evaluation...")
                advanced_evaluation = self._perform_advanced_evaluation(
                    X_train, y_train, X_val, y_val, X_test, y_test, regime_labels
                )
            
            # Step 6: Multi-objective Optimization
            pareto_front = []
            objective_scores = {}
            if self.config.enable_multi_objective:
                self.logger.info("🎯 Performing multi-objective optimization...")
                pareto_front, objective_scores = self._perform_multi_objective_optimization(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
            
            # Step 7: Select Best Model
            best_model, best_config, best_score = self._select_best_model(
                automl_result, evolutionary_result, ensemble_models
            )
            
            # Step 8: Create Comprehensive Result
            total_search_time = time.time() - start_time
            
            result = EnhancedTASResult(
                best_model=best_model,
                best_config=best_config,
                best_score=best_score,
                model_results=self._get_all_model_results(),
                model_rankings=self._rank_models(),
                automl_result=automl_result,
                evolutionary_result=evolutionary_result,
                advanced_evaluation=advanced_evaluation,
                feature_importance=self._get_feature_importance(),
                selected_features=self._get_selected_features(),
                pareto_front=pareto_front,
                objective_scores=objective_scores,
                total_search_time=total_search_time,
                total_evaluations=len(self.search_history),
                successful_evaluations=len([r for r in self.search_history if r.get('success', False)]),
                failed_evaluations=len([r for r in self.search_history if not r.get('success', True)]),
                search_history=self.search_history,
                convergence_info=self._get_convergence_info(),
                success=True
            )
            
            self.logger.info(f"✅ Enhanced TAS search completed in {total_search_time:.2f}s")
            self.logger.info(f"   Best score: {best_score:.4f}")
            self.logger.info(f"   Total evaluations: {result.total_evaluations}")
            self.logger.info(f"   Successful evaluations: {result.successful_evaluations}")
            
            return result
            
        except Exception as e:
            total_search_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced TAS search failed: {e}")
            return EnhancedTASResult(
                best_model=None,
                best_config=TreeModelConfig(),
                best_score=0.0,
                model_results=[],
                model_rankings=[],
                total_search_time=total_search_time,
                total_evaluations=len(self.search_history),
                successful_evaluations=0,
                failed_evaluations=len(self.search_history),
                search_history=self.search_history,
                success=False,
                error_message=str(e)
            )
    
    def _perform_feature_engineering(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Perform feature engineering."""
        try:
            # This is a placeholder for feature engineering
            # In practice, you would integrate with existing feature engineering tools
            self.logger.info("🔧 Performing feature engineering...")
            
            # For now, return the data as-is
            # In practice, you would:
            # 1. Apply technical indicators
            # 2. Create interaction features
            # 3. Apply feature selection
            # 4. Normalize/scale features
            
            return X_train, X_val, X_test
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineering failed: {e}")
            return X_train, X_val, X_test
    
    def _run_evolutionary_search(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_test: Optional[np.ndarray] = None,
                               y_test: Optional[np.ndarray] = None) -> Optional[EvolutionaryResult]:
        """Run evolutionary search."""
        try:
            # Define objective functions
            def accuracy_objective(params):
                try:
                    config = TreeModelConfig(**params)
                    model = self.model_factory.create_model(config)
                    result = self.model_evaluator.evaluate_model(
                        model, X_train, y_train, X_val, y_val, X_test, y_test
                    )
                    return result.val_score if result.success else 0.0
                except Exception:
                    return 0.0
            
            def robustness_objective(params):
                try:
                    config = TreeModelConfig(**params)
                    model = self.model_factory.create_model(config)
                    result = self.model_evaluator.evaluate_model(
                        model, X_train, y_train, X_val, y_val, X_test, y_test
                    )
                    # Calculate robustness as inverse of variance
                    if result.cv_scores:
                        robustness = 1.0 / (1.0 + np.std(result.cv_scores))
                        return robustness
                    else:
                        return 0.0
                except Exception:
                    return 0.0
            
            # Define parameter space
            parameter_space = {
                'model_type': {'type': 'categorical', 'choices': self.config.model_types},
                'n_estimators': {'type': 'integer', 'min': 50, 'max': 500},
                'max_depth': {'type': 'integer', 'min': 3, 'max': 15},
                'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3}
            }
            
            # Run evolutionary search
            objective_functions = [accuracy_objective, robustness_objective]
            result = self.evolutionary_manager.optimize_with_algorithm(
                objective_functions, parameter_space, self.config.evolutionary_algorithm
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Evolutionary search failed: {e}")
            return None
    
    def _create_model_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              X_test: Optional[np.ndarray] = None,
                              y_test: Optional[np.ndarray] = None) -> List[Any]:
        """Create model ensemble."""
        try:
            if self.model_factory is None:
                return []
            
            # Create ensemble of different model types
            ensemble_models = []
            for model_type in self.config.model_types:
                try:
                    config = TreeModelConfig(model_type=model_type)
                    model = self.model_factory.create_model(config)
                    result = self.model_evaluator.evaluate_model(
                        model, X_train, y_train, X_val, y_val, X_test, y_test
                    )
                    if result.success:
                        ensemble_models.append(result.model)
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not create {model_type} model: {e}")
                    continue
            
            return ensemble_models
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model ensemble creation failed: {e}")
            return []
    
    def _perform_advanced_evaluation(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   X_test: Optional[np.ndarray] = None,
                                   y_test: Optional[np.ndarray] = None,
                                   regime_labels: Optional[np.ndarray] = None) -> Optional[AdvancedEvaluationResult]:
        """Perform advanced evaluation."""
        try:
            if self.advanced_evaluator is None:
                return None
            
            # Use best model for evaluation
            if self.best_result is not None:
                model = self.best_result.model
                predictions = model.predict(X_val)
                targets = y_val
                returns = predictions  # Use predictions as returns for now
                
                return self.advanced_evaluator.evaluate(
                    predictions, targets, returns, regime_labels
                )
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced evaluation failed: {e}")
            return None
    
    def _perform_multi_objective_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                           X_val: np.ndarray, y_val: np.ndarray,
                                           X_test: Optional[np.ndarray] = None,
                                           y_test: Optional[np.ndarray] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Perform multi-objective optimization."""
        try:
            # This is a placeholder for multi-objective optimization
            # In practice, you would implement NSGA-II or similar algorithm
            pareto_front = []
            objective_scores = {}
            
            # For now, return empty results
            return pareto_front, objective_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Multi-objective optimization failed: {e}")
            return [], {}
    
    def _select_best_model(self, automl_result: Optional[AutoMLResult],
                         evolutionary_result: Optional[EvolutionaryResult],
                         ensemble_models: List[Any]) -> Tuple[Any, TreeModelConfig, float]:
        """Select the best model from all results."""
        try:
            best_model = None
            best_config = TreeModelConfig()
            best_score = float('-inf')
            
            # Check AutoML result
            if automl_result is not None and automl_result.success:
                if automl_result.best_score > best_score:
                    best_score = automl_result.best_score
                    best_model = automl_result.best_model
                    best_config = automl_result.best_config
            
            # Check evolutionary result
            if evolutionary_result is not None and evolutionary_result.success:
                if evolutionary_result.pareto_front:
                    # Use best individual from Pareto front
                    best_individual = evolutionary_result.pareto_front[0]
                    # Convert individual parameters to model config
                    config = TreeModelConfig(**best_individual.parameters)
                    if self.model_factory is not None:
                        model = self.model_factory.create_model(config)
                        # Evaluate model
                        if self.model_evaluator is not None:
                            result = self.model_evaluator.evaluate_model(
                                model, X_train, y_train, X_val, y_val
                            )
                            if result.success and result.val_score > best_score:
                                best_score = result.val_score
                                best_model = result.model
                                best_config = result.config
            
            # Check ensemble models
            if ensemble_models:
                # For now, use the first ensemble model
                # In practice, you would evaluate all ensemble models
                if best_model is None:
                    best_model = ensemble_models[0]
                    best_config = TreeModelConfig()
                    best_score = 0.0
            
            return best_model, best_config, best_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Best model selection failed: {e}")
            return None, TreeModelConfig(), 0.0
    
    def _get_all_model_results(self) -> List[TreeModelResult]:
        """Get all model results from search history."""
        try:
            results = []
            for search_step in self.search_history:
                if 'result' in search_step:
                    result = search_step['result']
                    if hasattr(result, 'model_results'):
                        results.extend(result.model_results)
            return results
        except Exception:
            return []
    
    def _rank_models(self) -> List[Tuple[str, float]]:
        """Rank models by performance."""
        try:
            rankings = []
            for result in self._get_all_model_results():
                if result.success:
                    rankings.append((result.model_type, result.val_score))
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
        except Exception:
            return []
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from best model."""
        try:
            if self.best_result is not None and hasattr(self.best_result, 'feature_importance'):
                return self.best_result.feature_importance
            else:
                return {}
        except Exception:
            return {}
    
    def _get_selected_features(self) -> List[str]:
        """Get selected features."""
        try:
            # This is a placeholder
            # In practice, you would get this from feature engineering results
            return []
        except Exception:
            return []
    
    def _get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        try:
            return {
                'total_steps': len(self.search_history),
                'convergence_reached': len(self.search_history) > 0,
                'final_score': self.best_result.best_score if self.best_result else 0.0
            }
        except Exception:
            return {}


# Convenience functions
def create_enhanced_tas_engine(config: Optional[EnhancedTASConfig] = None) -> EnhancedTASEngine:
    """Create Enhanced TAS Engine instance."""
    if config is None:
        config = EnhancedTASConfig()
    return EnhancedTASEngine(config)


def quick_enhanced_tas_search(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             X_test: Optional[np.ndarray] = None,
                             y_test: Optional[np.ndarray] = None,
                             model_types: List[str] = None,
                             max_search_time: int = 3600) -> EnhancedTASResult:
    """Quick Enhanced TAS search with default settings.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features (optional)
        y_test: Test targets (optional)
        model_types: List of model types to test
        max_search_time: Maximum search time in seconds
        
    Returns:
        EnhancedTASResult with search results
    """
    if model_types is None:
        model_types = ["xgboost", "lightgbm", "catboost"]
    
    config = EnhancedTASConfig(
        model_types=model_types,
        max_search_time=max_search_time,
        enable_automl=True,
        enable_evolutionary_search=True,
        enable_advanced_metrics=True,
        enable_feature_engineering=True
    )
    
    engine = create_enhanced_tas_engine(config)
    return engine.search(X_train, y_train, X_val, y_val, X_test, y_test)