"""
HMM-Specific Hyperparameter Optimization Configuration

This module provides specialized HPO configurations for HMM model training,
leveraging the common HPO utilities to reduce code duplication in market_analysis/.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .hpo_utils import HyperparameterOptimization
from ..config.base_training_config import HMMTrainingConfig

class HMMHyperparameterOptimizer:
    """
    Specialized HPO for HMM model training with regime-aware configurations.
    """

    def __init__(self, config: Optional[HMMTrainingConfig] = None):
        """
        Initialize HMM HPO with specialized search spaces.

        Args:
            config: HMM training configuration
        """
        self.config = config or HMMTrainingConfig()

    def get_hmm_state_recognition_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """
        Get HPO search spaces optimized for HMM state recognition.

        Returns:
            Dictionary of search spaces for each model type
        """
        return {
            'logistic_regression': {
                'C': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']},
                'max_iter': {'type': 'int', 'low': 500, 'high': 2000}
            },
            'lightgbm': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'max_depth': {'type': 'int', 'low': 4, 'high': 10},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0}
            },
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 5, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'max_depth': {'type': 'int', 'low': 4, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'catboost': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'depth': {'type': 'int', 'low': 4, 'high': 10},
                'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0}
            }
        }

    def get_hmm_ensemble_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """
        Get HPO search spaces for HMM ensemble models with stacking and meta-learning.

        Returns:
            Dictionary of search spaces for ensemble model types
        """
        return {
            'stacking': {
                'meta_learner': {'type': 'categorical', 'choices': ['logistic_regression', 'xgboost', 'lightgbm']},
                'cv_folds': {'type': 'int', 'low': 3, 'high': 10},
                'stack_method': {'type': 'categorical', 'choices': ['auto', 'predict_proba', 'decision_function']}
            },
            'voting': {
                'voting': {'type': 'categorical', 'choices': ['hard', 'soft']},
                'weights': {'type': 'categorical', 'choices': ['uniform', 'accuracy_based']},
                'n_estimators': {'type': 'int', 'low': 3, 'high': 10}
            },
            'bagging': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
                'max_samples': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'max_features': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            }
        }

    def get_regime_specific_search_spaces(self, regime_id: int) -> Dict[str, Dict[str, Any]]:
        """
        Get regime-specific search spaces with adaptive parameters.

        Args:
            regime_id: Regime identifier

        Returns:
            Dictionary of search spaces adapted for specific regime
        """
        base_spaces = self.get_hmm_state_recognition_search_spaces()

        # Adapt parameters based on regime characteristics
        regime_modifiers = {
            'high_volatility': {
                'lightgbm': {'learning_rate': {'low': 0.005, 'high': 0.1}},
                'xgboost': {'learning_rate': {'low': 0.005, 'high': 0.1}},
                'catboost': {'learning_rate': {'low': 0.005, 'high': 0.1}}
            },
            'low_volatility': {
                'random_forest': {'n_estimators': {'low': 200, 'high': 2000}},
                'logistic_regression': {'C': {'low': 0.01, 'high': 100.0}}
            },
            'trending': {
                'xgboost': {'colsample_bytree': {'low': 0.8, 'high': 1.0}},
                'lightgbm': {'colsample_bytree': {'low': 0.8, 'high': 1.0}}
            },
            'mean_reverting': {
                'random_forest': {'max_features': {'choices': ['sqrt', 'log2']}},
                'xgboost': {'reg_alpha': {'low': 0.1, 'high': 2.0}},
                'lightgbm': {'reg_alpha': {'low': 0.1, 'high': 2.0}}
            }
        }

        # Apply regime-specific modifications
        if regime_id < 3:  # First few regimes are typically high volatility
            modifier = regime_modifiers['high_volatility']
            for model_type, modifications in modifier.items():
                if model_type in base_spaces:
                    base_spaces[model_type].update(modifications)

        return base_spaces

    def optimize_multi_objective_hmm(
        self,
        objective_function: callable,
        model_types: List[str],
        n_trials: int = 100,
        objectives: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-objective optimization for HMM models.

        Args:
            objective_function: Function to optimize
            model_types: List of model types to optimize
            n_trials: Number of optimization trials
            objectives: List of objectives (accuracy, f1_score, regime_stability)

        Returns:
            Multi-objective optimization results
        """
        if objectives is None:
            objectives = ["accuracy", "f1_score", "regime_stability"]

        # Create parameter space combining all model types
        combined_param_space = {}
        for model_type in model_types:
            spaces = self.get_hmm_state_recognition_search_spaces()
            if model_type in spaces:
                # Add model_type indicator
                combined_param_space.update({
                    f"{model_type}_{param}": config
                    for param, config in spaces[model_type].items()
                })
                # Add model type choice
                combined_param_space['model_type'] = {
                    'type': 'categorical',
                    'choices': model_types
                }

        # Lightweight random-search multi-objective routine (previously legacy adapter)
        trials: List[Dict[str, Any]] = []
        rng = np.random.default_rng(42)

        def _sample_params(space: Dict[str, Any]) -> Dict[str, Any]:
            params: Dict[str, Any] = {}
            for name, cfg in space.items():
                if isinstance(cfg, dict):
                    typ = cfg.get('type', 'float')
                    if typ == 'int':
                        low, high = int(cfg.get('low', 0)), int(cfg.get('high', 100))
                        params[name] = int(rng.integers(low, max(low + 1, high + 1)))
                    elif typ == 'float':
                        low, high = float(cfg.get('low', 0.0)), float(cfg.get('high', 1.0))
                        if cfg.get('log', False) and low > 0 and high > low:
                            params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
                        else:
                            params[name] = float(rng.uniform(low, high))
                    elif typ == 'categorical':
                        choices = cfg.get('choices', [])
                        if choices:
                            params[name] = choices[int(rng.integers(0, len(choices)))]
                elif isinstance(cfg, list) and cfg:
                    params[name] = cfg[int(rng.integers(0, len(cfg)))]
            return params

        for i in range(n_trials):
            params = _sample_params(combined_param_space)
            try:
                scores = objective_function(**params)
                if not isinstance(scores, (list, tuple)):
                    scores = [scores]
                trials.append({'params': params, 'objectives': scores, 'trial_number': i, 'success': True})
            except Exception as e:
                trials.append({'params': params, 'objectives': [float('-inf')] * len(objectives), 'trial_number': i, 'success': False, 'error': str(e)})

        # Pareto front
        successful = [t for t in trials if t['success']]
        pareto: List[Dict[str, Any]] = []
        for t in successful:
            dominated = False
            for o in successful:
                if t is o:
                    continue
                dom = True
                strictly_better = False
                for a, b in zip(o['objectives'], t['objectives']):
                    if a < b:
                        dom = False
                        break
                    if a > b:
                        strictly_better = True
                if dom and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(t)

        pareto.sort(key=lambda x: x['objectives'][0] if x['objectives'] else float('-inf'), reverse=True)

        return {
            'trials': trials,
            'pareto_front': pareto,
            'best_params': pareto[0]['params'] if pareto else None,
            'n_trials': n_trials,
            'n_objectives': len(objectives),
            'direction': 'maximize',
            'success': len(pareto) > 0
        }

    def get_hmm_model_types(self) -> List[str]:
        """Get recommended HMM model types."""
        return [
            "logistic_regression",  # Interpretable linear model
            "lightgbm",             # Fast, efficient gradient boosting
            "random_forest",        # Robust ensemble tree model
            "xgboost",              # XGBoost gradient boosting
            "catboost"              # CatBoost gradient boosting
        ]

    def get_hmm_objectives(self) -> List[str]:
        """Get recommended objectives for HMM training."""
        return [
            "accuracy",
            "f1_score",
            "regime_stability",
            "temporal_consistency",
            "feature_importance_stability"
        ]

    def get_hmm_objective_weights(self) -> List[float]:
        """Get recommended objective weights."""
        return [0.4, 0.3, 0.15, 0.1, 0.05]

# Global instance
_hmm_hpo_instance = None

def get_hmm_hyperparameter_optimizer(config: Optional[HMMTrainingConfig] = None) -> HMMHyperparameterOptimizer:
    """Get global HMM hyperparameter optimizer instance."""
    global _hmm_hpo_instance
    if _hmm_hpo_instance is None:
        _hmm_hpo_instance = HMMHyperparameterOptimizer(config)
    return _hmm_hpo_instance

# Export key classes and functions
__all__ = ['HMMHyperparameterOptimizer', 'get_hmm_hyperparameter_optimizer']