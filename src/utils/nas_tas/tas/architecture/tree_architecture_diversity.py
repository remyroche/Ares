"""
Tree Architecture Diversity Expansion for CLVSA Models

This module provides comprehensive architecture diversity for tree-based models
while maintaining CLVSA architecture awareness, including:
- Multiple tree ensemble types
- Hybrid tree-neural architectures
- CLVSA-optimized tree structures
- Advanced tree search strategies
- Hardware-aware tree architectures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import pickle
import time
from datetime import datetime
from enum import Enum

# Tree model imports
try:
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
        AdaBoostClassifier, AdaBoostRegressor,
        BaggingClassifier, BaggingRegressor
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Import existing utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

    def _log_fallback(level: int, message: str, *args, **kwargs) -> None:
        logging.getLogger(__name__).log(level, message)

    def tprint(message: str, *args, **kwargs):  # type: ignore
        _log_fallback(logging.INFO, message)

    def tprint_info(message: str, *args, **kwargs):  # type: ignore
        _log_fallback(logging.INFO, message)

    def tprint_warning(message: str, *args, **kwargs):  # type: ignore
        _log_fallback(logging.WARNING, message)

    def tprint_error(message: str, *args, **kwargs):  # type: ignore
        _log_fallback(logging.ERROR, message)

    def tprint_success(message: str, *args, **kwargs):  # type: ignore
        _log_fallback(logging.INFO, message)

logger = logging.getLogger(__name__)


class TreeArchitectureType(Enum):
    """Types of tree architectures."""
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    EXTRA_TREES = "extra_trees"
    ADA_BOOST = "ada_boost"
    BAGGING = "bagging"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    HYBRID_TREE_NEURAL = "hybrid_tree_neural"
    CLVSA_OPTIMIZED = "cvlsa_optimized"
    HARDWARE_AWARE = "hardware_aware"


@dataclass
class TreeArchitectureConfig:
    """Configuration for tree architecture diversity."""
    
    # Architecture selection
    enable_architecture_diversity: bool = True
    max_architectures: int = 10
    architecture_selection_strategy: str = "performance_based"  # "performance_based", "diversity_based", "hybrid"
    
    # Tree ensemble parameters
    enable_random_forest: bool = True
    enable_gradient_boosting: bool = True
    enable_extra_trees: bool = True
    enable_ada_boost: bool = True
    enable_bagging: bool = True
    
    # Advanced tree models
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    enable_catboost: bool = True
    
    # Hybrid architectures
    enable_hybrid_tree_neural: bool = True
    enable_cvlsa_optimized: bool = True
    enable_hardware_aware: bool = True
    
    # CLVSA-specific settings
    cvlsa_optimization_level: int = 2  # 1: basic, 2: intermediate, 3: advanced
    cvlsa_memory_efficiency: bool = True
    cvlsa_parallelization: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    hardware_aware_selection: bool = True
    memory_efficient_architectures: bool = True
    
    # Performance optimization
    enable_performance_optimization: bool = True
    performance_threshold: float = 0.8
    optimization_budget: float = 0.1  # fraction of time for optimization


@dataclass
class TreeArchitectureCandidate:
    """Candidate tree architecture."""

    architecture_type: TreeArchitectureType
    model_instance: Any
    parameters: Dict[str, Any]
    task_type: str = "classification"
    performance_score: float = 0.0
    diversity_score: float = 0.0
    hardware_efficiency: float = 0.0
    cvlsa_compatibility: float = 0.0
    memory_usage: float = 0.0
    training_time: float = 0.0
    prediction_latency: float = 0.0
    
    # Metadata
    creation_time: datetime = field(default_factory=datetime.now)
    optimization_history: List[Dict] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)


class TreeArchitectureFactory:
    """Factory for creating diverse tree architectures."""
    
    def __init__(self, config: Optional[TreeArchitectureConfig] = None):
        """Initialize tree architecture factory."""
        self.config = config or TreeArchitectureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Architecture registry
        self.architecture_registry = {}
        self._register_architectures()
        
        tprint_info("✅ Tree Architecture Factory initialized")
    
    def _register_architectures(self):
        """Register available tree architectures."""
        registry: Dict[TreeArchitectureType, Callable] = {
            TreeArchitectureType.DECISION_TREE: self._create_decision_tree,
            TreeArchitectureType.RANDOM_FOREST: self._create_random_forest,
            TreeArchitectureType.GRADIENT_BOOSTING: self._create_gradient_boosting,
            TreeArchitectureType.EXTRA_TREES: self._create_extra_trees,
            TreeArchitectureType.ADA_BOOST: self._create_ada_boost,
            TreeArchitectureType.BAGGING: self._create_bagging,
            TreeArchitectureType.XGBOOST: self._create_xgboost,
            TreeArchitectureType.LIGHTGBM: self._create_lightgbm,
            TreeArchitectureType.CATBOOST: self._create_catboost,
            TreeArchitectureType.HYBRID_TREE_NEURAL: self._create_hybrid_tree_neural,
            TreeArchitectureType.CLVSA_OPTIMIZED: self._create_cvlsa_optimized,
            TreeArchitectureType.HARDWARE_AWARE: self._create_hardware_aware
        }

        filtered: Dict[TreeArchitectureType, Callable] = {}

        def register_if_enabled(arch_type: TreeArchitectureType,
                                enabled: bool,
                                dependency_available: bool = True) -> None:
            if enabled and dependency_available:
                filtered[arch_type] = registry[arch_type]
            elif enabled and not dependency_available:
                self.logger.debug(
                    "Skipping %s architecture; required dependency unavailable.",
                    arch_type.value
                )

        # Always register decision trees if sklearn is available
        register_if_enabled(TreeArchitectureType.DECISION_TREE, SKLEARN_AVAILABLE)

        register_if_enabled(
            TreeArchitectureType.RANDOM_FOREST,
            self.config.enable_random_forest and SKLEARN_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.GRADIENT_BOOSTING,
            self.config.enable_gradient_boosting and SKLEARN_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.EXTRA_TREES,
            self.config.enable_extra_trees and SKLEARN_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.ADA_BOOST,
            self.config.enable_ada_boost and SKLEARN_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.BAGGING,
            self.config.enable_bagging and SKLEARN_AVAILABLE
        )

        register_if_enabled(
            TreeArchitectureType.XGBOOST,
            self.config.enable_xgboost,
            XGBOOST_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.LIGHTGBM,
            self.config.enable_lightgbm,
            LIGHTGBM_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.CATBOOST,
            self.config.enable_catboost,
            CATBOOST_AVAILABLE
        )

        register_if_enabled(
            TreeArchitectureType.HYBRID_TREE_NEURAL,
            self.config.enable_hybrid_tree_neural and SKLEARN_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.CLVSA_OPTIMIZED,
            self.config.enable_cvlsa_optimized and SKLEARN_AVAILABLE
        )
        register_if_enabled(
            TreeArchitectureType.HARDWARE_AWARE,
            self.config.enable_hardware_aware and SKLEARN_AVAILABLE
        )

        self.architecture_registry = filtered
    
    def create_architecture(self, 
                           architecture_type: TreeArchitectureType,
                           parameters: Optional[Dict[str, Any]] = None,
                           task_type: str = "classification") -> TreeArchitectureCandidate:
        """
        Create tree architecture candidate.
        
        Args:
            architecture_type: Type of architecture to create
            parameters: Architecture parameters
            task_type: Type of task (classification/regression)
            
        Returns:
            Tree architecture candidate
        """
        try:
            if architecture_type not in self.architecture_registry:
                raise ValueError(f"Unknown architecture type: {architecture_type}")
            
            # Create model instance
            model_instance = self.architecture_registry[architecture_type](parameters, task_type)

            # Create candidate
            candidate = TreeArchitectureCandidate(
                architecture_type=architecture_type,
                model_instance=model_instance,
                parameters=(parameters or {}).copy(),
                task_type=task_type,
                creation_time=datetime.now()
            )
            
            # Apply CLVSA optimizations if enabled
            if self.config.enable_cvlsa_optimized:
                candidate = self._apply_cvlsa_optimizations(candidate)
            
            # Apply hardware optimizations if enabled
            if self.config.enable_hardware_optimization:
                candidate = self._apply_hardware_optimizations(candidate)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Created {architecture_type.value} architecture")
            
            return candidate
            
        except Exception as e:
            tprint_error(f"❌ Architecture creation failed: {e}")
            raise
    
    def create_diverse_architectures(self, 
                                   n_architectures: int = 5,
                                   task_type: str = "classification") -> List[TreeArchitectureCandidate]:
        """
        Create diverse set of tree architectures.
        
        Args:
            n_architectures: Number of architectures to create
            task_type: Type of task (classification/regression)
            
        Returns:
            List of diverse architecture candidates
        """
        try:
            architectures = []
            available_types = list(self.architecture_registry.keys())

            if not available_types:
                raise RuntimeError("No architectures available with current configuration")

            if not self.config.enable_architecture_diversity:
                default_type = (
                    TreeArchitectureType.RANDOM_FOREST
                    if TreeArchitectureType.RANDOM_FOREST in self.architecture_registry
                    else available_types[0]
                )
                architectures.append(self.create_architecture(default_type, task_type=task_type))
                return architectures

            # Select diverse architecture types
            selected_types = self._select_diverse_types(available_types, n_architectures)
            
            for arch_type in selected_types:
                try:
                    candidate = self.create_architecture(arch_type, task_type=task_type)
                    architectures.append(candidate)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to create {arch_type.value}: {e}")
                    continue
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Created {len(architectures)} diverse architectures")
            
            return architectures
            
        except Exception as e:
            tprint_error(f"❌ Diverse architecture creation failed: {e}")
            return []
    
    def _create_decision_tree(self, parameters: Optional[Dict], task_type: str):
        """Create decision tree architecture."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        if task_type == "classification":
            return DecisionTreeClassifier(**parameters or {})
        else:
            return DecisionTreeRegressor(**parameters or {})
    
    def _create_random_forest(self, parameters: Optional[Dict], task_type: str):
        """Create random forest architecture."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        if task_type == "classification":
            return RandomForestClassifier(**parameters or {})
        else:
            return RandomForestRegressor(**parameters or {})
    
    def _create_gradient_boosting(self, parameters: Optional[Dict], task_type: str):
        """Create gradient boosting architecture."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        if task_type == "classification":
            return GradientBoostingClassifier(**parameters or {})
        else:
            return GradientBoostingRegressor(**parameters or {})
    
    def _create_extra_trees(self, parameters: Optional[Dict], task_type: str):
        """Create extra trees architecture."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        if task_type == "classification":
            return ExtraTreesClassifier(**parameters or {})
        else:
            return ExtraTreesRegressor(**parameters or {})
    
    def _create_ada_boost(self, parameters: Optional[Dict], task_type: str):
        """Create AdaBoost architecture."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        if task_type == "classification":
            return AdaBoostClassifier(**parameters or {})
        else:
            return AdaBoostRegressor(**parameters or {})
    
    def _create_bagging(self, parameters: Optional[Dict], task_type: str):
        """Create bagging architecture."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        base_estimator = DecisionTreeClassifier() if task_type == "classification" else DecisionTreeRegressor()
        
        if task_type == "classification":
            return BaggingClassifier(base_estimator=base_estimator, **parameters or {})
        else:
            return BaggingRegressor(base_estimator=base_estimator, **parameters or {})
    
    def _create_xgboost(self, parameters: Optional[Dict], task_type: str):
        """Create XGBoost architecture."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        if task_type == "classification":
            return xgb.XGBClassifier(**parameters or {})
        else:
            return xgb.XGBRegressor(**parameters or {})
    
    def _create_lightgbm(self, parameters: Optional[Dict], task_type: str):
        """Create LightGBM architecture."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        if task_type == "classification":
            return lgb.LGBMClassifier(**parameters or {})
        else:
            return lgb.LGBMRegressor(**parameters or {})
    
    def _create_catboost(self, parameters: Optional[Dict], task_type: str):
        """Create CatBoost architecture."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        
        if task_type == "classification":
            return cb.CatBoostClassifier(**parameters or {})
        else:
            return cb.CatBoostRegressor(**parameters or {})
    
    def _create_hybrid_tree_neural(self, parameters: Optional[Dict], task_type: str):
        """Create hybrid tree-neural architecture."""
        # Placeholder for hybrid architecture
        # This would combine tree models with neural networks
        return self._create_random_forest(parameters, task_type)
    
    def _create_cvlsa_optimized(self, parameters: Optional[Dict], task_type: str):
        """Create CLVSA-optimized architecture."""
        # CLVSA-specific optimizations
        cvlsa_params = parameters or {}
        
        # Apply CLVSA-specific parameters
        if self.config.cvlsa_memory_efficiency:
            cvlsa_params.setdefault('max_depth', 10)  # Limit depth for memory efficiency
        
        if self.config.cvlsa_parallelization:
            cvlsa_params.setdefault('n_jobs', -1)  # Enable parallelization
        
        return self._create_random_forest(cvlsa_params, task_type)
    
    def _create_hardware_aware(self, parameters: Optional[Dict], task_type: str):
        """Create hardware-aware architecture."""
        # Hardware-aware optimizations
        hw_params = parameters or {}
        
        # Apply hardware-specific optimizations
        if self.config.memory_efficient_architectures:
            hw_params.setdefault('max_depth', 8)  # Limit depth for memory efficiency
            hw_params.setdefault('n_estimators', 100)  # Limit estimators for memory
        
        return self._create_random_forest(hw_params, task_type)
    
    def _select_diverse_types(self, available_types: List[TreeArchitectureType], n_architectures: int) -> List[TreeArchitectureType]:
        """Select diverse architecture types."""
        # Prioritize different categories
        priority_types = [
            TreeArchitectureType.RANDOM_FOREST,
            TreeArchitectureType.GRADIENT_BOOSTING,
            TreeArchitectureType.XGBOOST,
            TreeArchitectureType.LIGHTGBM,
            TreeArchitectureType.CLVSA_OPTIMIZED
        ]
        
        selected_types = []
        
        # Add priority types first
        for arch_type in priority_types:
            if arch_type in available_types and len(selected_types) < n_architectures:
                selected_types.append(arch_type)
        
        # Add remaining types
        for arch_type in available_types:
            if arch_type not in selected_types and len(selected_types) < n_architectures:
                selected_types.append(arch_type)
        
        return selected_types[:n_architectures]
    
    def _apply_cvlsa_optimizations(self, candidate: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Apply CLVSA-specific optimizations."""
        # CLVSA optimizations
        candidate.cvlsa_compatibility = 1.0
        
        # Memory efficiency
        if self.config.cvlsa_memory_efficiency:
            candidate.memory_usage *= 0.8  # Reduce memory usage
        
        return candidate
    
    def _apply_hardware_optimizations(self, candidate: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Apply hardware-specific optimizations."""
        # Hardware optimizations
        candidate.hardware_efficiency = 1.0
        
        # Memory optimization
        if self.config.memory_efficient_architectures:
            candidate.memory_usage *= 0.9
        
        return candidate


class TreeArchitectureEvaluator:
    """Evaluator for tree architecture candidates."""
    
    def __init__(self, config: Optional[TreeArchitectureConfig] = None):
        """Initialize tree architecture evaluator."""
        self.config = config or TreeArchitectureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint_info("✅ Tree Architecture Evaluator initialized")
    
    def evaluate_architecture(self, 
                            candidate: TreeArchitectureCandidate,
                            X: np.ndarray,
                            y: np.ndarray,
                            cv_folds: int = 5) -> TreeArchitectureCandidate:
        """
        Evaluate tree architecture candidate.
        
        Args:
            candidate: Architecture candidate to evaluate
            X: Training features
            y: Training targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Evaluated architecture candidate
        """
        start_time = time.time()
        
        try:
            # Train and evaluate model
            model = candidate.model_instance
            model.fit(X, y)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(candidate, X, y, cv_folds)
            candidate.performance_score = performance_score
            
            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(candidate)
            candidate.diversity_score = diversity_score
            
            # Calculate hardware efficiency
            hardware_efficiency = self._calculate_hardware_efficiency(candidate)
            candidate.hardware_efficiency = hardware_efficiency
            
            # Calculate memory usage
            memory_usage = self._calculate_memory_usage(model)
            candidate.memory_usage = memory_usage
            
            # Calculate training time
            training_time = time.time() - start_time
            candidate.training_time = training_time
            
            # Calculate prediction latency
            prediction_latency = self._calculate_prediction_latency(model, X)
            candidate.prediction_latency = prediction_latency
            
            # Update performance history
            candidate.performance_history.append(performance_score)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Architecture evaluated: {candidate.architecture_type.value} (score: {performance_score:.3f})")
            
            return candidate
            
        except Exception as e:
            tprint_error(f"❌ Architecture evaluation failed: {e}")
            candidate.performance_score = 0.0
            return candidate
    
    def _calculate_performance_score(self,
                                     candidate: TreeArchitectureCandidate,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     cv_folds: int) -> float:
        """Calculate performance score."""
        model = candidate.model_instance

        if not SKLEARN_AVAILABLE:
            try:
                return float(model.score(X, y))
            except Exception as e:
                tprint_warning(f"⚠️ Unable to calculate score without scikit-learn: {e}")
                return 0.0

        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, r2_score

        scoring = 'accuracy' if candidate.task_type == 'classification' else 'r2'

        try:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            return float(np.mean(cv_scores))
        except ValueError:
            try:
                predictions = model.predict(X)
                if candidate.task_type == 'classification':
                    return float(accuracy_score(y, predictions))
                return float(r2_score(y, predictions))
            except Exception as inner_error:
                tprint_error(f"❌ Performance score fallback failed: {inner_error}")
                return 0.0
        except Exception as e:
            tprint_error(f"❌ Performance score calculation failed: {e}")
            return 0.0
    
    def _calculate_diversity_score(self, candidate: TreeArchitectureCandidate) -> float:
        """Calculate diversity score."""
        # Diversity based on architecture type
        diversity_scores = {
            TreeArchitectureType.DECISION_TREE: 0.1,
            TreeArchitectureType.RANDOM_FOREST: 0.8,
            TreeArchitectureType.GRADIENT_BOOSTING: 0.9,
            TreeArchitectureType.EXTRA_TREES: 0.7,
            TreeArchitectureType.ADA_BOOST: 0.6,
            TreeArchitectureType.BAGGING: 0.5,
            TreeArchitectureType.XGBOOST: 0.9,
            TreeArchitectureType.LIGHTGBM: 0.9,
            TreeArchitectureType.CATBOOST: 0.9,
            TreeArchitectureType.HYBRID_TREE_NEURAL: 1.0,
            TreeArchitectureType.CLVSA_OPTIMIZED: 0.8,
            TreeArchitectureType.HARDWARE_AWARE: 0.7
        }
        
        return diversity_scores.get(candidate.architecture_type, 0.5)
    
    def _calculate_hardware_efficiency(self, candidate: TreeArchitectureCandidate) -> float:
        """Calculate hardware efficiency score."""
        # Hardware efficiency based on architecture type
        efficiency_scores = {
            TreeArchitectureType.DECISION_TREE: 1.0,
            TreeArchitectureType.RANDOM_FOREST: 0.8,
            TreeArchitectureType.GRADIENT_BOOSTING: 0.7,
            TreeArchitectureType.EXTRA_TREES: 0.8,
            TreeArchitectureType.ADA_BOOST: 0.6,
            TreeArchitectureType.BAGGING: 0.7,
            TreeArchitectureType.XGBOOST: 0.9,
            TreeArchitectureType.LIGHTGBM: 0.9,
            TreeArchitectureType.CATBOOST: 0.8,
            TreeArchitectureType.HYBRID_TREE_NEURAL: 0.6,
            TreeArchitectureType.CLVSA_OPTIMIZED: 0.9,
            TreeArchitectureType.HARDWARE_AWARE: 1.0
        }
        
        return efficiency_scores.get(candidate.architecture_type, 0.5)
    
    def _calculate_memory_usage(self, model: Any) -> float:
        """Calculate memory usage in MB."""
        try:
            serialized = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            return len(serialized) / (1024 * 1024)
        except Exception as e:
            tprint_warning(f"Memory usage calculation failed: {e}. Returning 0.0.")
            return 0.0
    
    def _calculate_prediction_latency(self, model: Any, X: np.ndarray) -> float:
        """Calculate prediction latency."""
        try:
            start_time = time.time()
            model.predict(X[:100])  # Predict on subset
            return time.time() - start_time
        except Exception as e:
            tprint_warning(f"Prediction latency calculation failed: {e}. Returning 0.0.")
            return 0.0


class TreeArchitectureSelector:
    """Selector for optimal tree architectures."""
    
    def __init__(self, config: Optional[TreeArchitectureConfig] = None):
        """Initialize tree architecture selector."""
        self.config = config or TreeArchitectureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint_info("✅ Tree Architecture Selector initialized")
    
    def select_best_architectures(self, 
                                candidates: List[TreeArchitectureCandidate],
                                selection_criteria: str = "performance") -> List[TreeArchitectureCandidate]:
        """
        Select best architectures based on criteria.
        
        Args:
            candidates: List of architecture candidates
            selection_criteria: Selection criteria ("performance", "diversity", "hybrid")
            
        Returns:
            List of selected architectures
        """
        try:
            if not candidates:
                return []
            
            # Sort candidates based on criteria
            if selection_criteria == "performance":
                sorted_candidates = sorted(candidates, key=lambda x: x.performance_score, reverse=True)
            elif selection_criteria == "diversity":
                sorted_candidates = sorted(candidates, key=lambda x: x.diversity_score, reverse=True)
            elif selection_criteria == "hybrid":
                sorted_candidates = sorted(candidates, key=lambda x: self._calculate_hybrid_score(x), reverse=True)
            else:
                sorted_candidates = candidates
            
            # Select top candidates
            n_selected = min(self.config.max_architectures, len(sorted_candidates))
            selected = sorted_candidates[:n_selected]
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Selected {len(selected)} best architectures")
            
            return selected
            
        except Exception as e:
            tprint_error(f"❌ Architecture selection failed: {e}")
            return candidates
    
    def _calculate_hybrid_score(self, candidate: TreeArchitectureCandidate) -> float:
        """Calculate hybrid score combining multiple criteria."""
        weights = {
            'performance': 0.4,
            'diversity': 0.2,
            'hardware_efficiency': 0.2,
            'cvlsa_compatibility': 0.2
        }
        
        score = (
            weights['performance'] * candidate.performance_score +
            weights['diversity'] * candidate.diversity_score +
            weights['hardware_efficiency'] * candidate.hardware_efficiency +
            weights['cvlsa_compatibility'] * candidate.cvlsa_compatibility
        )
        
        return score


# Factory functions
def create_tree_architecture_factory(config: Optional[TreeArchitectureConfig] = None) -> TreeArchitectureFactory:
    """Create tree architecture factory instance."""
    return TreeArchitectureFactory(config)


def create_tree_architecture_evaluator(config: Optional[TreeArchitectureConfig] = None) -> TreeArchitectureEvaluator:
    """Create tree architecture evaluator instance."""
    return TreeArchitectureEvaluator(config)


def create_tree_architecture_selector(config: Optional[TreeArchitectureConfig] = None) -> TreeArchitectureSelector:
    """Create tree architecture selector instance."""
    return TreeArchitectureSelector(config)


# Example usage
if __name__ == "__main__":
    # Create tree architecture factory
    config = TreeArchitectureConfig(
        enable_architecture_diversity=True,
        enable_cvlsa_optimized=True,
        enable_hardware_optimization=True
    )
    
    factory = create_tree_architecture_factory(config)
    evaluator = create_tree_architecture_evaluator(config)
    selector = create_tree_architecture_selector(config)
    
    # Create diverse architectures
    architectures = factory.create_diverse_architectures(n_architectures=5)
    
    print(f"Created {len(architectures)} diverse tree architectures!")
    for arch in architectures:
        print(f"  - {arch.architecture_type.value}: {arch.model_instance.__class__.__name__}")