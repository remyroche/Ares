"""
TAS Meta-Learning - Tree Architecture Search Meta-Learning System

Meta-learning system for tree-based architectures that learns to adapt
tree structures and parameters based on historical performance and
market regime changes.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# Import shared utilities
try:
    from src.utils.common_operations import (
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
        safe_json_dump, safe_json_load, ensure_directory
    )
    from src.utils.math_validation import MathValidation
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)

# Import tree-specific libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    TREE_LIBS_AVAILABLE = True
except ImportError:
    TREE_LIBS_AVAILABLE = False
    tprint_warning("Tree libraries not available, using fallback implementations")

logger = logging.getLogger(__name__)

@dataclass
class TASMetaLearningConfig:
    """Configuration for TAS meta-learning."""
    
    # Meta-learning parameters
    meta_learning_algorithm: str = "tree_structure_adaptation"  # tree_structure_adaptation, parameter_adaptation
    adaptation_strategy: str = "performance_based"  # performance_based, regime_based, hybrid
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.05
    max_adaptations: int = 10
    
    # Tree structure adaptation
    enable_structure_adaptation: bool = True
    structure_adaptation_weight: float = 0.5
    depth_adaptation_range: Tuple[int, int] = (3, 15)
    leaf_adaptation_range: Tuple[int, int] = (10, 1000)
    
    # Parameter adaptation
    enable_parameter_adaptation: bool = True
    parameter_adaptation_weight: float = 0.5
    n_estimators_range: Tuple[int, int] = (50, 500)
    max_features_range: Tuple[float, float] = (0.1, 1.0)
    
    # Historical learning
    enable_historical_learning: bool = True
    history_window: int = 10
    history_weight: float = 0.3
    performance_weight: float = 0.7
    
    # Regime-based adaptation
    enable_regime_adaptation: bool = True
    regime_detection_threshold: float = 0.1
    regime_adaptation_factor: float = 1.2
    
    # Performance monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_meta_learning_results: bool = True
    
    # Output settings
    output_dir: str = "tas_meta_learning_results"
    results_format: str = "json"

@dataclass
class TASMetaLearningResult:
    """Result from TAS meta-learning."""
    
    # Meta-learning results
    success: bool
    adapted_model: Optional[Any] = None
    adaptation_score: float = 0.0
    original_score: float = 0.0
    improvement: float = 0.0
    
    # Adaptation details
    adaptations_applied: List[Dict[str, Any]] = field(default_factory=list)
    structure_changes: Optional[Dict[str, Any]] = None
    parameter_changes: Optional[Dict[str, Any]] = None
    
    # Historical learning
    historical_insights: Optional[Dict[str, Any]] = None
    performance_trends: Optional[Dict[str, Any]] = None
    
    # Regime analysis
    regime_detected: bool = False
    regime_adaptation_applied: bool = False
    regime_confidence: float = 0.0
    
    # Performance metrics
    meta_learning_time: float = 0.0
    memory_usage_mb: float = 0.0
    convergence_achieved: bool = False
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class TASMetaLearning:
    """
    Tree Architecture Search Meta-Learning System.
    
    Meta-learning system for tree-based architectures that learns to adapt
    tree structures and parameters based on historical performance and
    market regime changes.
    """
    
    def __init__(self, config: Optional[TASMetaLearningConfig] = None):
        """Initialize TAS meta-learning system."""
        self.config = config or TASMetaLearningConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utilities
        self._init_utilities()
        
        # Meta-learning state
        self.adaptation_history = []
        self.performance_history = []
        self.regime_history = []
        
        tprint_success("🚀 TAS Meta-Learning initialized")
        tprint_info(f"   → Algorithm: {self.config.meta_learning_algorithm}")
        tprint_info(f"   → Strategy: {self.config.adaptation_strategy}")
        tprint_info(f"   → Historical learning: {'enabled' if self.config.enable_historical_learning else 'disabled'}")
        tprint_info(f"   → Regime adaptation: {'enabled' if self.config.enable_regime_adaptation else 'disabled'}")
    
    def _init_utilities(self):
        """Initialize utility components."""
        if SHARED_UTILS_AVAILABLE:
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
        else:
            self.math_validator = None
            self.serializer = None
    
    def adapt_model(self, 
                   model: Any,
                   X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series],
                   historical_performance: Optional[List[float]] = None,
                   regime_info: Optional[Dict[str, Any]] = None) -> TASMetaLearningResult:
        """
        Adapt a tree model using meta-learning.
        
        Args:
            model: Base tree model to adapt
            X: Current features
            y: Current targets
            historical_performance: Historical performance scores
            regime_info: Current regime information
            
        Returns:
            TASMetaLearningResult with adaptation results
        """
        start_time = time.time()
        tprint_info("🧠 Starting TAS meta-learning adaptation")
        
        try:
            # Validate inputs
            self._validate_inputs(model, X, y)
            
            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Evaluate original model
            original_score = self._evaluate_model(model, X_processed, y_processed)
            
            # Initialize adaptation
            adapted_model = self._clone_model(model)
            adaptations_applied = []
            
            # Apply meta-learning adaptations
            if self.config.meta_learning_algorithm == "tree_structure_adaptation":
                adapted_model, structure_changes = self._adapt_tree_structure(
                    adapted_model, X_processed, y_processed, historical_performance
                )
                adaptations_applied.append({
                    'type': 'structure_adaptation',
                    'changes': structure_changes,
                    'timestamp': time.time()
                })
            
            elif self.config.meta_learning_algorithm == "parameter_adaptation":
                adapted_model, parameter_changes = self._adapt_parameters(
                    adapted_model, X_processed, y_processed, historical_performance
                )
                adaptations_applied.append({
                    'type': 'parameter_adaptation',
                    'changes': parameter_changes,
                    'timestamp': time.time()
                })
            
            # Apply regime-based adaptations if enabled
            regime_adaptation_applied = False
            regime_confidence = 0.0
            if self.config.enable_regime_adaptation and regime_info:
                adapted_model, regime_changes = self._apply_regime_adaptation(
                    adapted_model, regime_info, historical_performance
                )
                if regime_changes:
                    adaptations_applied.append({
                        'type': 'regime_adaptation',
                        'changes': regime_changes,
                        'timestamp': time.time()
                    })
                    regime_adaptation_applied = True
                    regime_confidence = regime_info.get('confidence', 0.0)
            
            # Evaluate adapted model
            adapted_score = self._evaluate_model(adapted_model, X_processed, y_processed)
            improvement = adapted_score - original_score
            
            # Apply historical learning if enabled
            historical_insights = None
            performance_trends = None
            if self.config.enable_historical_learning and historical_performance:
                historical_insights = self._analyze_historical_performance(historical_performance)
                performance_trends = self._analyze_performance_trends(historical_performance)
            
            # Update meta-learning state
            self._update_meta_learning_state(adaptations_applied, adapted_score, regime_info)
            
            # Get performance metrics
            memory_usage = self._get_memory_usage()
            
            # Create result
            result = TASMetaLearningResult(
                success=True,
                adapted_model=adapted_model,
                adaptation_score=adapted_score,
                original_score=original_score,
                improvement=improvement,
                adaptations_applied=adaptations_applied,
                structure_changes=adaptations_applied[0].get('changes') if adaptations_applied else None,
                parameter_changes=adaptations_applied[0].get('changes') if adaptations_applied else None,
                historical_insights=historical_insights,
                performance_trends=performance_trends,
                regime_detected=regime_info is not None,
                regime_adaptation_applied=regime_adaptation_applied,
                regime_confidence=regime_confidence,
                meta_learning_time=time.time() - start_time,
                memory_usage_mb=memory_usage,
                convergence_achieved=self._check_convergence(adaptations_applied)
            )
            
            # Save results if configured
            if self.config.save_meta_learning_results:
                self._save_meta_learning_results(result)
            
            tprint_success(f"✅ TAS meta-learning completed in {result.meta_learning_time:.2f}s")
            tprint_info(f"   → Original score: {result.original_score:.4f}")
            tprint_info(f"   → Adapted score: {result.adaptation_score:.4f}")
            tprint_info(f"   → Improvement: {result.improvement:.4f}")
            tprint_info(f"   → Adaptations applied: {len(result.adaptations_applied)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ TAS meta-learning failed: {e}")
            
            return TASMetaLearningResult(
                success=False,
                meta_learning_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_inputs(self, model, X, y):
        """Validate input data."""
        if not TREE_LIBS_AVAILABLE:
            raise ImportError("Tree libraries not available")
        
        if model is None:
            raise ValueError("Model cannot be None")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
    
    def _preprocess_data(self, X, y):
        """Preprocess input data."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Handle missing values
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_array, y_array
    
    def _clone_model(self, model):
        """Clone a model for adaptation."""
        try:
            # Create a new instance of the same model type
            model_type = type(model)
            
            # Get model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
            else:
                params = {}
            
            # Create new model with same parameters
            cloned_model = model_type(**params)
            
            # If model is already fitted, copy the fitted state
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                # For tree models, we need to retrain
                pass
            
            return cloned_model
            
        except Exception as e:
            tprint_warning(f"⚠️ Model cloning failed: {e}")
            return model
    
    def _adapt_tree_structure(self, model, X, y, historical_performance):
        """Adapt tree structure based on performance."""
        try:
            changes = {}
            
            # Analyze current performance
            current_score = self._evaluate_model(model, X, y)
            
            # Determine adaptation strategy
            if historical_performance and len(historical_performance) > 1:
                performance_trend = np.mean(historical_performance[-3:]) - np.mean(historical_performance[:-3])
                
                if performance_trend < -self.config.adaptation_threshold:
                    # Performance declining, increase complexity
                    changes['action'] = 'increase_complexity'
                    changes['max_depth'] = min(
                        self.config.depth_adaptation_range[1],
                        getattr(model, 'max_depth', 10) + 2
                    )
                    changes['min_samples_leaf'] = max(
                        1,
                        getattr(model, 'min_samples_leaf', 1) - 1
                    )
                elif performance_trend > self.config.adaptation_threshold:
                    # Performance improving, maintain or slightly reduce complexity
                    changes['action'] = 'maintain_complexity'
                else:
                    # Stable performance, optimize structure
                    changes['action'] = 'optimize_structure'
                    changes['max_depth'] = getattr(model, 'max_depth', 10)
                    changes['min_samples_leaf'] = getattr(model, 'min_samples_leaf', 1)
            else:
                # No historical data, use default adaptation
                changes['action'] = 'default_adaptation'
                changes['max_depth'] = self.config.depth_adaptation_range[0]
                changes['min_samples_leaf'] = 1
            
            # Apply changes to model
            if hasattr(model, 'set_params'):
                model.set_params(**{k: v for k, v in changes.items() if k != 'action'})
            
            # Retrain model with new structure
            model.fit(X, y)
            
            return model, changes
            
        except Exception as e:
            tprint_warning(f"⚠️ Tree structure adaptation failed: {e}")
            return model, {}
    
    def _adapt_parameters(self, model, X, y, historical_performance):
        """Adapt model parameters based on performance."""
        try:
            changes = {}
            
            # Analyze current performance
            current_score = self._evaluate_model(model, X, y)
            
            # Determine parameter adaptation
            if historical_performance and len(historical_performance) > 1:
                performance_trend = np.mean(historical_performance[-3:]) - np.mean(historical_performance[:-3])
                
                if performance_trend < -self.config.adaptation_threshold:
                    # Performance declining, increase model capacity
                    changes['action'] = 'increase_capacity'
                    if hasattr(model, 'n_estimators'):
                        changes['n_estimators'] = min(
                            self.config.n_estimators_range[1],
                            getattr(model, 'n_estimators', 100) + 50
                        )
                    if hasattr(model, 'max_features'):
                        changes['max_features'] = min(
                            self.config.max_features_range[1],
                            getattr(model, 'max_features', 0.5) + 0.1
                        )
                elif performance_trend > self.config.adaptation_threshold:
                    # Performance improving, maintain parameters
                    changes['action'] = 'maintain_parameters'
                else:
                    # Stable performance, optimize parameters
                    changes['action'] = 'optimize_parameters'
                    changes['n_estimators'] = getattr(model, 'n_estimators', 100)
                    changes['max_features'] = getattr(model, 'max_features', 0.5)
            else:
                # No historical data, use default parameters
                changes['action'] = 'default_parameters'
                changes['n_estimators'] = self.config.n_estimators_range[0]
                changes['max_features'] = self.config.max_features_range[0]
            
            # Apply changes to model
            if hasattr(model, 'set_params'):
                model.set_params(**{k: v for k, v in changes.items() if k != 'action'})
            
            # Retrain model with new parameters
            model.fit(X, y)
            
            return model, changes
            
        except Exception as e:
            tprint_warning(f"⚠️ Parameter adaptation failed: {e}")
            return model, {}
    
    def _apply_regime_adaptation(self, model, regime_info, historical_performance):
        """Apply regime-based adaptations."""
        try:
            changes = {}
            
            # Analyze regime information
            regime_type = regime_info.get('type', 'unknown')
            regime_confidence = regime_info.get('confidence', 0.0)
            regime_stability = regime_info.get('stability', 0.0)
            
            # Determine regime-based adaptation
            if regime_confidence > self.config.regime_detection_threshold:
                if regime_type == 'volatile':
                    # Volatile regime - increase model robustness
                    changes['action'] = 'increase_robustness'
                    changes['max_depth'] = min(
                        self.config.depth_adaptation_range[1],
                        getattr(model, 'max_depth', 10) + 3
                    )
                    changes['min_samples_leaf'] = max(
                        5,
                        getattr(model, 'min_samples_leaf', 1) + 2
                    )
                elif regime_type == 'stable':
                    # Stable regime - optimize for efficiency
                    changes['action'] = 'optimize_efficiency'
                    changes['max_depth'] = max(
                        self.config.depth_adaptation_range[0],
                        getattr(model, 'max_depth', 10) - 2
                    )
                    changes['min_samples_leaf'] = getattr(model, 'min_samples_leaf', 1)
                else:
                    # Unknown regime - maintain current settings
                    changes['action'] = 'maintain_settings'
            else:
                # Low regime confidence - use historical performance
                if historical_performance and len(historical_performance) > 1:
                    performance_trend = np.mean(historical_performance[-3:]) - np.mean(historical_performance[:-3])
                    if performance_trend < -self.config.adaptation_threshold:
                        changes['action'] = 'historical_adaptation'
                        changes['max_depth'] = getattr(model, 'max_depth', 10) + 1
                    else:
                        changes['action'] = 'no_adaptation'
                else:
                    changes['action'] = 'no_adaptation'
            
            # Apply changes to model
            if changes.get('action') != 'no_adaptation' and hasattr(model, 'set_params'):
                model.set_params(**{k: v for k, v in changes.items() if k != 'action'})
                model.fit(X, y)
            
            return model, changes
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime adaptation failed: {e}")
            return model, {}
    
    def _analyze_historical_performance(self, historical_performance):
        """Analyze historical performance patterns."""
        try:
            if not historical_performance or len(historical_performance) < 2:
                return None
            
            performance_array = np.array(historical_performance)
            
            insights = {
                'mean_performance': float(np.mean(performance_array)),
                'std_performance': float(np.std(performance_array)),
                'trend': float(np.polyfit(range(len(performance_array)), performance_array, 1)[0]),
                'volatility': float(np.std(np.diff(performance_array))),
                'best_performance': float(np.max(performance_array)),
                'worst_performance': float(np.min(performance_array)),
                'performance_range': float(np.max(performance_array) - np.min(performance_array))
            }
            
            return insights
            
        except Exception as e:
            tprint_warning(f"⚠️ Historical performance analysis failed: {e}")
            return None
    
    def _analyze_performance_trends(self, historical_performance):
        """Analyze performance trends."""
        try:
            if not historical_performance or len(historical_performance) < 3:
                return None
            
            performance_array = np.array(historical_performance)
            
            # Calculate trend components
            recent_performance = performance_array[-3:]
            older_performance = performance_array[:-3] if len(performance_array) > 3 else performance_array
            
            trends = {
                'recent_mean': float(np.mean(recent_performance)),
                'older_mean': float(np.mean(older_performance)),
                'trend_direction': 'improving' if np.mean(recent_performance) > np.mean(older_performance) else 'declining',
                'trend_strength': float(abs(np.mean(recent_performance) - older_performance))),
                'consistency': float(1.0 - np.std(recent_performance) / np.mean(recent_performance)) if np.mean(recent_performance) > 0 else 0.0
            }
            
            return trends
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance trend analysis failed: {e}")
            return None
    
    def _evaluate_model(self, model, X, y):
        """Evaluate model performance."""
        try:
            y_pred = model.predict(X)
            
            # Use appropriate metric based on problem type
            if hasattr(model, 'predict_proba'):
                # Classification
                return accuracy_score(y, y_pred)
            else:
                # Regression
                return r2_score(y, y_pred)
                
        except Exception as e:
            tprint_warning(f"⚠️ Model evaluation failed: {e}")
            return 0.0
    
    def _check_convergence(self, adaptations_applied):
        """Check if meta-learning has converged."""
        if len(adaptations_applied) < 2:
            return False
        
        # Check if recent adaptations are minimal
        recent_adaptations = adaptations_applied[-2:]
        return all(adaptation.get('action') in ['maintain_complexity', 'maintain_parameters', 'no_adaptation'] 
                  for adaptation in recent_adaptations)
    
    def _update_meta_learning_state(self, adaptations_applied, score, regime_info):
        """Update meta-learning state."""
        self.adaptation_history.extend(adaptations_applied)
        self.performance_history.append(score)
        
        if regime_info:
            self.regime_history.append(regime_info)
        
        # Keep only recent history
        if len(self.adaptation_history) > self.config.max_adaptations:
            self.adaptation_history = self.adaptation_history[-self.config.max_adaptations:]
        
        if len(self.performance_history) > self.config.history_window:
            self.performance_history = self.performance_history[-self.config.history_window:]
    
    def _get_memory_usage(self):
        """Get memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def _save_meta_learning_results(self, result):
        """Save meta-learning results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tas_meta_learning_{timestamp}.{self.config.results_format}"
            filepath = Path(self.config.output_dir) / filename
            
            ensure_directory(filepath.parent)
            
            # Prepare data for serialization
            result_data = {
                'success': result.success,
                'adaptation_score': result.adaptation_score,
                'original_score': result.original_score,
                'improvement': result.improvement,
                'adaptations_applied': result.adaptations_applied,
                'structure_changes': result.structure_changes,
                'parameter_changes': result.parameter_changes,
                'historical_insights': result.historical_insights,
                'performance_trends': result.performance_trends,
                'regime_detected': result.regime_detected,
                'regime_adaptation_applied': result.regime_adaptation_applied,
                'regime_confidence': result.regime_confidence,
                'meta_learning_time': result.meta_learning_time,
                'memory_usage_mb': result.memory_usage_mb,
                'convergence_achieved': result.convergence_achieved,
                'error_message': result.error_message,
                'warnings': result.warnings
            }
            
            if self.config.results_format == 'json':
                safe_json_dump(result_data, filepath)
            elif self.config.results_format == 'pickle':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(result_data, f)
            
            tprint_success(f"💾 Meta-learning results saved to {filepath}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save meta-learning results: {e}")
    
    def get_meta_learning_summary(self):
        """Get meta-learning summary."""
        return {
            'total_adaptations': len(self.adaptation_history),
            'performance_history_length': len(self.performance_history),
            'regime_history_length': len(self.regime_history),
            'config': self.config.__dict__,
            'recent_performance': self.performance_history[-5:] if self.performance_history else [],
            'recent_adaptations': self.adaptation_history[-3:] if self.adaptation_history else []
        }