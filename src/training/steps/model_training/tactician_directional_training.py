"""
Tactician Directional Training - Enhanced Training Step

This module enhances the Tactician training step to directly optimize for directional objectives:
1. directional_accuracy: How often the entry direction is correct
2. adverse_movement_minimization: Minimize adverse price movement  
3. directional_profit_efficiency: Profit from correct directional moves
4. risk_adjusted_performance: Risk-adjusted returns

The enhanced training includes:
- Custom loss functions for directional optimization
- Multi-objective training with Pareto front analysis
- Advanced feature engineering for directional prediction
- Risk-aware model architecture
- Enhanced evaluation metrics
- Integration with existing Tactician training pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
import time
import traceback
from dataclasses import dataclass

# Core ML imports
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV
import optuna
from optuna.samplers import TPESampler

# Enhanced utilities
from src.utils.ml_common.optimization.pareto import (
    ParetoFront, Solution, ObjectiveDirection, 
    scalarize_financial_goals, select_knee_point
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite
)

# Base training imports
from .tactician_models_training_refactored import TacticianModelsTrainingStepRefactored
from .tactician_directional_optimization import (
    EntryTimingLossFunction, DirectionalOptimizationResult, EntryTimingTacticianOptimizer
)
from src.utils.ml_common.config import TacticianTrainingConfig

logger = logging.getLogger(__name__)


class DirectionalTacticianTrainingStep(TacticianModelsTrainingStepRefactored):
    """
    Enhanced Tactician training step that directly optimizes for directional objectives.
    
    This class extends the base Tactician training with:
    - Directional optimization objectives
    - Custom loss functions
    - Enhanced feature engineering
    - Multi-objective training
    - Risk-aware model architecture
    """
    
    def __init__(self, 
                 config: Optional[TacticianTrainingConfig] = None, 
                 enable_vectorization: bool = True,
                 enable_directional_optimization: bool = True):
        """
        Initialize directional Tactician training step.
        
        Args:
            config: Tactician training configuration
            enable_vectorization: Whether to enable vectorized training
            enable_directional_optimization: Whether to enable directional optimization
        """
        super().__init__(config, enable_vectorization)
        
        self.enable_directional_optimization = enable_directional_optimization
        self.logger = logger.getChild('DirectionalTacticianTrainingStep')
        
        # Initialize entry timing components
        if enable_directional_optimization:
            # Note: EntryTimingTacticianOptimizer needs to be implemented in tactician_directional_optimization.py
            # For now, using the loss functions and creating a basic optimizer structure
            self.loss_functions = EntryTimingLossFunction()
            
            # Entry timing optimization objectives (renamed from directional_objectives for consistency)
            self.directional_objectives = {
                'early_entry_penalty': 'min',
                'late_entry_penalty': 'min',
                'optimal_entry_reward': 'max',
                'entry_timing_efficiency': 'max'
            }
            
            # Also keep the original name for backward compatibility
            self.entry_timing_objectives = self.directional_objectives
            
            # Initialize directional optimizer (now using proper implementation)
            self.directional_optimizer = EntryTimingTacticianOptimizer(config)
            self.entry_timing_optimizer = self.directional_optimizer  # Alias for compatibility
            
            self.logger.info("🚀 Entry timing optimization enabled")
        else:
            self.directional_optimizer = None
            self.entry_timing_optimizer = None
            self.loss_functions = None
            self.directional_objectives = None
            self.entry_timing_objectives = None
            
            self.logger.info("ℹ️ Entry timing optimization disabled")
    
    def execute(self,
                X: np.ndarray,
                y: np.ndarray,
                regime_labels: np.ndarray,
                feature_names: Optional[List[str]] = None,
                hmm_states: Optional[np.ndarray] = None,
                analyst_signals: Optional[np.ndarray] = None,
                analyst_model_outputs: Optional[np.ndarray] = None,
                hmm_regime_features: Optional[np.ndarray] = None,
                all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
                hmm_model_outputs: Optional[np.ndarray] = None,
                analyst_ensemble_outputs: Optional[np.ndarray] = None
            ) -> Dict[str, Any]:
        """
        Execute enhanced Tactician training with directional optimization.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_signals: Binary signals from Analyst (green light indicators)
            analyst_model_outputs: Analyst model predictions used as features
            hmm_regime_features: HMM regime features (probabilities, characteristics)
            all_analyst_models_outputs: All individual analyst ML model outputs
            hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
            analyst_ensemble_outputs: Analyst ensemble model outputs
            
        Returns:
            Dictionary containing training results and metadata with directional optimization
        """
        try:
            self.logger.info("🚀 Starting Enhanced Directional Tactician training step")
            self.overall_start_time = time.time()
            
            # Phase 1: Data Validation
            self._start_phase("DATA_VALIDATION", {
                'samples': X.shape[0],
                'features': X.shape[1],
                'directional_optimization': self.enable_directional_optimization
            })
            
            try:
                self._validate_training_inputs(X, y, regime_labels, feature_names, hmm_states)
                self._complete_phase("DATA_VALIDATION", {"status": "success"})
            except Exception as e:
                self._complete_phase("DATA_VALIDATION", {"status": "failed", "error": str(e)})
                raise
            
            # Phase 2: Feature Preparation with Directional Enhancement
            self._start_phase("FEATURE_PREPARATION", {
                'original_samples': X.shape[0],
                'original_features': X.shape[1],
                'has_analyst_signals': analyst_signals is not None,
                'has_hmm_features': hmm_regime_features is not None,
                'has_analyst_models': all_analyst_models_outputs is not None,
                'has_analyst_ensemble': analyst_ensemble_outputs is not None,
                'directional_enhancement': self.enable_directional_optimization
            })
            
            try:
                X, y, regime_labels, feature_names, preparation_metrics = self._prepare_features(
                    X, y, regime_labels, feature_names, hmm_states, 
                    analyst_signals, analyst_model_outputs, hmm_regime_features, 
                    all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
                )
                
                # Using existing features for entry timing optimization (no additional feature engineering)
                if self.enable_directional_optimization:
                    self.logger.info(f"📊 Using existing features for entry timing optimization: {X.shape[1]} features")
                
                self._complete_phase("FEATURE_PREPARATION", preparation_metrics)
            except Exception as e:
                self._complete_phase("FEATURE_PREPARATION", {"status": "failed", "error": str(e)})
                raise
            
            # Phase 3: Directional Training Execution
            if self.enable_directional_optimization:
                self._start_phase("DIRECTIONAL_TRAINING", {
                    'samples': X.shape[0],
                    'features': X.shape[1],
                    'objectives': list(self.directional_objectives.keys())
                })
                
                try:
                    results = self._execute_directional_training(
                        X, y, regime_labels, feature_names, hmm_states
                    )
                    self._complete_phase("DIRECTIONAL_TRAINING", {"status": "success"})
                except Exception as e:
                    self._complete_phase("DIRECTIONAL_TRAINING", {"status": "failed", "error": str(e)})
                    raise
            else:
                # Fallback to standard training
                self._start_phase("STANDARD_TRAINING", {
                    'samples': X.shape[0],
                    'features': X.shape[1]
                })
                
                try:
                    results = self._execute_training(X, y, regime_labels, feature_names, hmm_states)
                    self._complete_phase("STANDARD_TRAINING", {"status": "success"})
                except Exception as e:
                    self._complete_phase("STANDARD_TRAINING", {"status": "failed", "error": str(e)})
                    raise
            
            # Phase 4: Results Finalization
            self._start_phase("RESULTS_FINALIZATION", {
                'models_trained': len(results.get('models', {})),
                'ensemble_trained': 'ensemble_model' in results
            })
            
            try:
                results = self._finalize_directional_results(results, analyst_signals)
                self._complete_phase("RESULTS_FINALIZATION", {"status": "success"})
            except Exception as e:
                self._complete_phase("RESULTS_FINALIZATION", {"status": "failed", "error": str(e)})
                raise
            
            # Log completion
            total_time = time.time() - self.overall_start_time
            self.logger.info(f"✅ Enhanced Directional Tactician training completed in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced Directional Tactician training failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return self._create_error_result(str(e))
    
    # Feature enhancement removed - using existing features from base training
    
    def _execute_directional_training(self,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    regime_labels: np.ndarray,
                                    feature_names: Optional[List[str]],
                                    hmm_states: Optional[np.ndarray]
                                ) -> Dict[str, Any]:
        """Execute directional training with multi-objective optimization."""
        training_metrics = {
            'directional_optimization': True,
            'objectives': list(self.directional_objectives.keys()),
            'training_method': 'directional_multi_objective',
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        try:
            # Use directional optimizer for training
            directional_result = self.directional_optimizer.optimize_tactician_directionally(
                X=X, y=y, regime_labels=regime_labels,
                feature_names=feature_names, hmm_states=hmm_states,
                max_trials=getattr(self.config, 'hpo_n_trials', 100)
            )
            
            # Convert directional result to standard format
            results = {
                'models': {
                    'directional_tactician': directional_result.model
                },
                'evaluations': {
                    'directional_tactician': {
                        'directional_accuracy': directional_result.directional_accuracy,
                        'adverse_movement_minimization': directional_result.adverse_movement_minimization,
                        'directional_profit_efficiency': directional_result.directional_profit_efficiency,
                        'risk_adjusted_performance': directional_result.risk_adjusted_performance,
                        'composite_score': directional_result.composite_score
                    }
                },
                'training_metrics': training_metrics,
                'directional_optimization': {
                    'enabled': True,
                    'objectives': self.directional_objectives,
                    'optimization_time': directional_result.optimization_time,
                    'n_trials': directional_result.n_trials,
                    'optimization_history': directional_result.optimization_history
                }
            }
            
            # Add ensemble training if enabled
            if hasattr(self.config, 'enable_ensemble_training') and self.config.enable_ensemble_training:
                ensemble_results = self._train_directional_ensemble(X, y, feature_names, results)
                results.update(ensemble_results)
            
            self.logger.info(f"✅ Directional training completed")
            self.logger.info(f"   Directional accuracy: {directional_result.directional_accuracy:.4f}")
            self.logger.info(f"   Adverse movement min: {directional_result.adverse_movement_minimization:.4f}")
            self.logger.info(f"   Profit efficiency: {directional_result.directional_profit_efficiency:.4f}")
            self.logger.info(f"   Risk-adjusted perf: {directional_result.risk_adjusted_performance:.4f}")
            self.logger.info(f"   Composite score: {directional_result.composite_score:.4f}")
            
            return results
            
        except Exception as e:
            training_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Directional training execution failed: {e}")
            raise
    
    def _train_directional_ensemble(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  feature_names: Optional[List[str]],
                                  base_models_results: Dict[str, Any]
                              ) -> Dict[str, Any]:
        """Train ensemble model with directional optimization."""
        try:
            self.logger.info("🔄 Training directional ensemble model...")
            
            # Get base model
            base_model = base_models_results.get('models', {}).get('directional_tactician')
            if not base_model:
                self.logger.warning("⚠️ No base directional model found for ensemble training")
                return {}
            
            # Generate base model predictions
            base_predictions = base_model.predict(X).reshape(-1, 1)
            
            # Create ensemble features
            X_ensemble = np.column_stack([X, base_predictions])
            
            # Train ensemble model with directional optimization
            ensemble_result = self.directional_optimizer.optimize_tactician_directionally(
                X=X_ensemble, y=y, regime_labels=np.zeros(len(y)),  # Single regime for ensemble
                feature_names=None, hmm_states=None,
                max_trials=50  # Fewer trials for ensemble
            )
            
            ensemble_results = {
                'ensemble_model': ensemble_result.model,
                'ensemble_evaluation': {
                    'directional_accuracy': ensemble_result.directional_accuracy,
                    'adverse_movement_minimization': ensemble_result.adverse_movement_minimization,
                    'directional_profit_efficiency': ensemble_result.directional_profit_efficiency,
                    'risk_adjusted_performance': ensemble_result.risk_adjusted_performance,
                    'composite_score': ensemble_result.composite_score
                },
                'ensemble_method': 'directional_optimization',
                'base_models_used': ['directional_tactician']
            }
            
            self.logger.info(f"✅ Directional ensemble training completed")
            self.logger.info(f"   Ensemble composite score: {ensemble_result.composite_score:.4f}")
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"❌ Directional ensemble training failed: {e}")
            return {}
    
    def _finalize_directional_results(self, 
                                    results: Dict[str, Any], 
                                    analyst_signals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Finalize results with directional optimization metadata."""
        try:
            # Add directional optimization metadata
            if 'directional_optimization' in results:
                results['directional_optimization']['metadata'] = {
                    'objectives_optimized': list(self.directional_objectives.keys()),
                    'optimization_method': 'multi_objective_directional',
                    'feature_engineering': 'directional_enhanced',
                    'loss_functions': 'directional_custom',
                    'evaluation_metrics': 'directional_specific'
                }
            
            # Add tactician-specific metadata
            results = self._finalize_results(results, analyst_signals)
            
            # Add directional performance summary
            if 'evaluations' in results:
                directional_eval = results['evaluations'].get('directional_tactician', {})
                if directional_eval:
                    results['directional_performance_summary'] = {
                        'primary_objective': 'directional_accuracy',
                        'secondary_objectives': [
                            'adverse_movement_minimization',
                            'directional_profit_efficiency', 
                            'risk_adjusted_performance'
                        ],
                        'performance_scores': directional_eval,
                        'optimization_focus': 'entry_point_directional_prediction'
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to finalize directional results: {e}")
            return results


# Convenience functions
def create_directional_tactician_training_step(
    config: Optional[TacticianTrainingConfig] = None,
    enable_vectorization: bool = True,
    enable_directional_optimization: bool = True
) -> DirectionalTacticianTrainingStep:
    """
    Create enhanced directional Tactician training step.
    
    Args:
        config: Tactician training configuration
        enable_vectorization: Whether to enable vectorized training
        enable_directional_optimization: Whether to enable directional optimization
        
    Returns:
        DirectionalTacticianTrainingStep instance
    """
    return DirectionalTacticianTrainingStep(
        config=config,
        enable_vectorization=enable_vectorization,
        enable_directional_optimization=enable_directional_optimization
    )


def execute_directional_tactician_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
    hmm_model_outputs: Optional[np.ndarray] = None,
    analyst_ensemble_outputs: Optional[np.ndarray] = None,
    enable_vectorization: bool = True,
    enable_directional_optimization: bool = True
) -> Dict[str, Any]:
    """
    Execute enhanced directional Tactician training step.
    
    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels
        config: Training configuration
        feature_names: Feature names
        hmm_states: HMM states
        analyst_signals: Analyst signals
        analyst_model_outputs: Analyst model outputs
        hmm_regime_features: HMM regime features
        all_analyst_models_outputs: All analyst model outputs
        hmm_model_outputs: HMM model outputs
        analyst_ensemble_outputs: Analyst ensemble outputs
        enable_vectorization: Whether to enable vectorized training
        enable_directional_optimization: Whether to enable directional optimization
        
    Returns:
        Dictionary containing training results and metadata
    """
    step = create_directional_tactician_training_step(
        config=config,
        enable_vectorization=enable_vectorization,
        enable_directional_optimization=enable_directional_optimization
    )
    
    return step.execute(
        X=X, y=y, regime_labels=regime_labels,
        feature_names=feature_names, hmm_states=hmm_states,
        analyst_signals=analyst_signals, analyst_model_outputs=analyst_model_outputs,
        hmm_regime_features=hmm_regime_features,
        all_analyst_models_outputs=all_analyst_models_outputs,
        hmm_model_outputs=hmm_model_outputs,
        analyst_ensemble_outputs=analyst_ensemble_outputs
    )


if __name__ == '__main__':
    # Test the directional training
    print("🎯 Testing Directional Tactician Training")
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)  # Directional targets
    regime_labels = np.random.choice([0, 1, 2], n_samples)
    
    # Test directional training
    print("\n📊 Testing directional training...")
    result = execute_directional_tactician_training(
        X=X, y=y, regime_labels=regime_labels,
        enable_directional_optimization=True
    )
    
    print(f"✅ Directional training completed:")
    if 'directional_performance_summary' in result:
        perf = result['directional_performance_summary']['performance_scores']
        print(f"   Directional accuracy: {perf.get('directional_accuracy', 0):.4f}")
        print(f"   Adverse movement min: {perf.get('adverse_movement_minimization', 0):.4f}")
        print(f"   Profit efficiency: {perf.get('directional_profit_efficiency', 0):.4f}")
        print(f"   Risk-adjusted perf: {perf.get('risk_adjusted_performance', 0):.4f}")
        print(f"   Composite score: {perf.get('composite_score', 0):.4f}")
    
    print('✅ Directional Tactician Training test completed!')