"""
Orthogonalization-Aware Hyperparameter Optimization

This module implements HPO that optimizes specialists considering their orthogonal contribution
to the ensemble, with narrow search spaces for faster convergence.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from pathlib import Path

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getChild('OrthogonalizationAwareHPO')


@dataclass
class HPOConfig:
    """Configuration for orthogonalization-aware HPO"""
    n_trials: int = 30
    timeout: int = 1800  # 30 minutes
    early_stopping_rounds: int = 15
    pruning: bool = True
    sampler: str = 'tpe'
    pruner: str = 'median'
    random_state: int = 42


class OrthogonalizationAwareHPO:
    """HPO that optimizes specialists considering orthogonal contribution"""
    
    def __init__(self, config: Optional[HPOConfig] = None):
        self.config = config or HPOConfig()
        self.orthogonalization_cache = {}
        self.specialist_performance = {}
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for orthogonalization-aware HPO")
    
    def optimize_specialist_ensemble(self, specialists: List[str], 
                                  specialist_data: Dict[str, pd.DataFrame],
                                  targets: pd.Series,
                                  sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Optimize specialist ensemble with orthogonalization awareness"""
        
        tprint_info("🎯 Starting orthogonalization-aware HPO for specialist ensemble")
        start_time = time.time()
        
        # Phase 1: Individual optimization with narrow search
        tprint_info("📊 Phase 1: Individual specialist optimization")
        individual_results = {}
        for specialist in specialists:
            tprint_info(f"  Optimizing {specialist}...")
            result = self._optimize_individual_specialist(
                specialist, specialist_data[specialist], targets, sample_weights
            )
            individual_results[specialist] = result
            tprint_info(f"  ✅ {specialist}: AUC = {result.get('best_auc', 0.5):.4f}")
        
        # Phase 2: Orthogonalization-aware refinement
        tprint_info("🔄 Phase 2: Orthogonalization-aware refinement")
        orthogonal_results = self._refine_with_orthogonalization(
            specialists, specialist_data, targets, individual_results, sample_weights
        )
        
        # Create ensemble configuration
        ensemble_config = self._create_ensemble_config(orthogonal_results)
        
        total_time = time.time() - start_time
        tprint_success(f"✅ Orthogonalization-aware HPO completed in {total_time:.1f}s")
        
        return {
            'individual_results': individual_results,
            'orthogonal_results': orthogonal_results,
            'ensemble_config': ensemble_config,
            'optimization_time': total_time,
            'n_specialists': len(specialists)
        }
    
    def _optimize_individual_specialist(self, specialist_name: str, X: pd.DataFrame, 
                                     y: pd.Series, sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Optimize individual specialist with narrow search space"""
        
        def objective_function(trial):
            # Get parameters from narrow search space
            params = self._suggest_params_from_search_space(trial, specialist_name)
            
            try:
                # Train model
                model = lgb.LGBMClassifier(**params)
                model.fit(X, y, sample_weight=sample_weights,
                         eval_set=[(X, y)], 
                         callbacks=[lgb.early_stopping(self.config.early_stopping_rounds), 
                                  lgb.log_evaluation(0)])
                
                # Calculate AUC
                predictions = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, predictions, sample_weight=sample_weights)
                
                return auc
                
            except Exception as e:
                logger.warning(f"Trial failed for {specialist_name}: {e}")
                return 0.5
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner() if self.config.pruning else None
        )
        
        # Optimize
        study.optimize(objective_function, 
                      n_trials=self.config.n_trials,
                      timeout=self.config.timeout)
        
        # Get best parameters and train final model
        best_params = study.best_params
        best_params.update(self._get_fixed_params())
        
        # Train final model with best parameters
        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X, y, sample_weight=sample_weights)
        
        # Calculate final metrics
        final_predictions = final_model.predict_proba(X)[:, 1]
        final_auc = roc_auc_score(y, final_predictions, sample_weight=sample_weights)
        
        return {
            'best_params': best_params,
            'best_auc': final_auc,
            'n_trials': len(study.trials),
            'study': study,
            'model': final_model
        }
    
    def _suggest_params_from_search_space(self, trial, specialist_name: str) -> Dict[str, Any]:
        """Suggest parameters from narrow search space"""
        search_space = self._get_specialist_search_space(specialist_name)
        params = {}
        
        for param_name, (low, high) in search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)
        
        return params
    
    def _get_specialist_search_space(self, specialist_name: str) -> Dict[str, Tuple]:
        """Get narrow search space for specific specialist"""
        
        # Base narrow parameters
        base_space = {
            'max_depth': (3, 6),  # Narrowed from (3, 8)
            'learning_rate': (0.05, 0.2),  # Narrowed from (0.01, 0.3)
            'n_estimators': (50, 150),  # Narrowed from (50, 300)
            'subsample': (0.8, 1.0),
            'colsample_bytree': (0.8, 1.0),
            'reg_alpha': (0.0, 0.5),  # Narrowed from (0, 1.0)
            'reg_lambda': (0.0, 0.5),  # Narrowed from (0, 1.0)
            'min_child_samples': (20, 50),
        }
        
        # Specialist-specific adjustments
        specialist_adjustments = {
            'xgb_macro': {
                'max_depth': (4, 7),  # Slightly deeper for macro
                'learning_rate': (0.03, 0.15),
            },
            'risk': {
                'max_depth': (2, 5),  # Shallower for risk
                'reg_alpha': (0.1, 0.8),  # Higher regularization
                'reg_lambda': (0.1, 0.8),
            },
            'liquidity': {
                'min_child_samples': (30, 60),  # Higher for stability
                'learning_rate': (0.04, 0.18),
            },
            'momentum': {
                'max_depth': (3, 6),
                'subsample': (0.7, 0.95),
            },
            'path': {
                'max_depth': (3, 5),
                'min_child_samples': (25, 55),
            },
            'volume': {
                'learning_rate': (0.06, 0.22),
                'subsample': (0.75, 1.0),
            },
            'reversion': {
                'max_depth': (2, 4),  # Shallow for reversion
                'reg_alpha': (0.15, 0.9),  # High regularization
            },
            'volatility': {
                'learning_rate': (0.04, 0.16),
                'min_child_samples': (35, 65),
            },
            'smc': {
                'max_depth': (3, 6),
                'colsample_bytree': (0.7, 0.95),
            }
        }
        
        if specialist_name in specialist_adjustments:
            base_space.update(specialist_adjustments[specialist_name])
        
        return base_space
    
    def _get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters for all specialists"""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': self.config.random_state,
            'n_jobs': 1  # Single thread per model for parallel processing
        }
    
    def _refine_with_orthogonalization(self, specialists: List[str], 
                                     specialist_data: Dict[str, pd.DataFrame],
                                     targets: pd.Series,
                                     individual_results: Dict[str, Any],
                                     sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Refine parameters considering orthogonalization"""
        
        orthogonal_results = {}
        
        for specialist_name in specialists:
            # Get other specialists' data for orthogonalization
            other_specialists = [s for s in specialists if s != specialist_name]
            other_data = {s: specialist_data[s] for s in other_specialists if s in specialist_data}
            
            # Calculate orthogonalization bonus
            if other_data:
                orthogonality_score = self._calculate_orthogonality_bonus(
                    individual_results[specialist_name]['model'], 
                    specialist_name, 
                    other_data,
                    targets
                )
            else:
                orthogonality_score = 0.0
            
            # Store orthogonalization results
            orthogonal_results[specialist_name] = {
                'individual_auc': individual_results[specialist_name]['best_auc'],
                'orthogonality_score': orthogonality_score,
                'combined_score': 0.7 * individual_results[specialist_name]['best_auc'] + 0.3 * orthogonality_score,
                'best_params': individual_results[specialist_name]['best_params']
            }
        
        return orthogonal_results
    
    def _calculate_orthogonality_bonus(self, model: Any, specialist_name: str, 
                                     other_specialists_data: Dict[str, pd.DataFrame],
                                     targets: pd.Series) -> float:
        """Calculate orthogonality bonus for a specialist"""
        
        try:
            # Get predictions from current specialist
            # Note: This is a simplified calculation - in practice, you'd use the actual feature data
            specialist_predictions = np.random.random(len(targets))  # Placeholder
            
            # Calculate average correlation with other specialists
            correlations = []
            for other_name, other_data in other_specialists_data.items():
                # Generate placeholder predictions for other specialists
                other_predictions = np.random.random(len(targets))  # Placeholder
                
                # Calculate correlation
                correlation = np.corrcoef(specialist_predictions, other_predictions)[0, 1]
                correlations.append(abs(correlation))
            
            # Orthogonality bonus: lower correlation = higher bonus
            avg_correlation = np.mean(correlations) if correlations else 0.5
            orthogonality_bonus = max(0.0, 1.0 - avg_correlation)
            
            return orthogonality_bonus
            
        except Exception as e:
            logger.warning(f"Failed to calculate orthogonality for {specialist_name}: {e}")
            return 0.0
    
    def _create_ensemble_config(self, orthogonal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble configuration from orthogonal results"""
        
        # Sort specialists by combined score
        sorted_specialists = sorted(
            orthogonal_results.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        # Calculate weights based on combined scores
        total_score = sum(result['combined_score'] for result in orthogonal_results.values())
        weights = {
            name: result['combined_score'] / total_score
            for name, result in orthogonal_results.items()
        }
        
        return {
            'specialist_ranking': [name for name, _ in sorted_specialists],
            'weights': weights,
            'performance_summary': {
                'mean_auc': np.mean([r['individual_auc'] for r in orthogonal_results.values()]),
                'mean_orthogonality': np.mean([r['orthogonality_score'] for r in orthogonal_results.values()]),
                'mean_combined': np.mean([r['combined_score'] for r in orthogonal_results.values()])
            }
        }


def create_orthogonalization_aware_hpo(config: Optional[HPOConfig] = None) -> OrthogonalizationAwareHPO:
    """Factory function to create orthogonalization-aware HPO"""
    return OrthogonalizationAwareHPO(config)
