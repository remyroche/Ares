"""
Random Survival Forest for Tactician Models - Multi-Horizon Framework Integration

This module provides Random Survival Forest implementation specifically designed for
tactician timing prediction with full integration into the multi-horizon framework.

Features:
- Multi-horizon survival analysis for entry timing prediction
- Integration with analyst signals and HMM regime probabilities
- Optimized for 1m timeframe with 2s latency constraint
- Support for censored data (no entry events)
- Feature importance for timing-relevant features
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass
from pathlib import Path

# Core imports
from src.utils.logger import get_system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

# Survival analysis imports
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.datasets import load_veterans_lung_cancer
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    SURVIVAL_ANALYSIS_AVAILABLE = True
except ImportError:
    SURVIVAL_ANALYSIS_AVAILABLE = False
    tprint_warning("⚠️ Survival analysis libraries not available. Install with: pip install scikit-survival")

# Multi-horizon framework imports
try:
    from src.utils.ml_common.training.multi_horizon_training import MultiHorizonTrainingManager
    from src.utils.ml_common.training.horizon_config import HorizonConfig
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False
    tprint_warning("⚠️ Multi-horizon framework not available")

logger = get_system_logger().getChild('RandomSurvivalForestTactician')


@dataclass
class SurvivalAnalysisConfig:
    """Configuration for Random Survival Forest tactician model."""
    
    # Model parameters
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    bootstrap: bool = True
    max_samples: float = 0.8
    
    # Multi-horizon settings
    horizons: List[float] = None  # Time horizons in minutes [1, 2, 5, 10]
    horizon_weights: List[float] = None  # Weights for each horizon
    
    # Timing optimization
    entry_timing_range: float = 0.005  # 0.5% range for entry timing
    expected_movement: float = 0.01  # Expected 1% movement
    latency_constraint: float = 2.0  # 2 second latency constraint
    
    # Feature engineering
    enable_timing_features: bool = True
    enable_regime_features: bool = True
    enable_analyst_features: bool = True
    enable_microstructure_features: bool = True
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [1, 2, 5, 10]  # 1m to 10m horizons (removed 15m and 30m)
        if self.horizon_weights is None:
            self.horizon_weights = [0.4, 0.3, 0.2, 0.1]  # Weighted towards shorter horizons


class RandomSurvivalForestTactician:
    """
    Random Survival Forest implementation for tactician timing prediction.
    
    This model predicts the probability of optimal entry timing within different time horizons
    using survival analysis techniques. The "entry" refers to the best time to enter a trade
    for minimal loss and highest gain, based on the analyst's green light signal.
    
    The model is fully integrated with the ML pipeline including HPO, validation, and
    multi-horizon framework support.
    
    Other ML models specifically designed for optimal entry timing:
    1. Point Process Models (Hawkes Process, Cox Process) - Model self-exciting events
    2. Survival Analysis Models (Cox Regression, AFT) - Model time-to-event
    3. Reinforcement Learning (DQN, Actor-Critic) - Learn optimal timing through rewards
    4. Sequence Models (LSTM, Transformer) - Capture temporal patterns in timing
    5. Multi-Armed Bandits - Explore vs exploit for timing decisions
    6. Bayesian Optimization - Optimize entry timing parameters
    7. Gaussian Process Regression - Model timing uncertainty
    8. Kalman Filters - Track optimal entry states
    9. Hidden Markov Models - Model regime-dependent timing
    10. Neural ODEs - Model continuous-time dynamics for timing
    """
    
    def __init__(self, config: Optional[SurvivalAnalysisConfig] = None):
        """
        Initialize Random Survival Forest tactician model.
        
        Args:
            config: Configuration for survival analysis model
        """
        self.logger = logger.getChild('RandomSurvivalForestTactician')
        
        if not SURVIVAL_ANALYSIS_AVAILABLE:
            raise ImportError("Survival analysis libraries not available. Install with: pip install scikit-survival")
        
        self.config = config or SurvivalAnalysisConfig()
        self.model = None
        self.feature_names = None
        self.horizon_models = {}  # Models for different horizons
        self.training_metrics = {}
        
        tprint_info("🚀 Initialized Random Survival Forest Tactician")
        tprint_info(f"📊 Horizons: {self.config.horizons} minutes")
        tprint_info(f"📊 Horizon weights: {self.config.horizon_weights}")
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            analyst_signals: Optional[np.ndarray] = None,
            hmm_regime_probs: Optional[np.ndarray] = None,
            multi_horizon_data: Optional[Dict[str, Any]] = None,
            enable_hpo: bool = True,
            hpo_trials: int = 100,
            cv_folds: int = 5,
            enable_entry_timing_optimization: bool = True,
            entry_timing_trials: int = 50) -> Dict[str, Any]:
        """
        Fit Random Survival Forest model for multi-horizon timing prediction.
        
        Args:
            X: Input features (market data, technical indicators, etc.)
            y: Target values (entry timing in minutes, or censored)
            feature_names: Names of input features
            analyst_signals: Analyst model outputs
            hmm_regime_probs: HMM regime probabilities
            multi_horizon_data: Multi-horizon training data
            
        Returns:
            Training results and metrics
        """
        tprint_info("🔄 Training Random Survival Forest for tactician timing prediction...")
        
        # Validate inputs
        self._validate_inputs(X, y, feature_names)
        
        # Prepare features
        X_enhanced, enhanced_feature_names = self._prepare_features(
            X, feature_names, analyst_signals, hmm_regime_probs
        )
        
        # Prepare survival data
        survival_data = self._prepare_survival_data(y, multi_horizon_data)
        
        # Train horizon-specific models with HPO if enabled
        horizon_results = {}
        for i, horizon in enumerate(self.config.horizons):
            tprint_info(f"🔄 Training model for {horizon} minute horizon...")
            
            # Prepare data for this horizon
            X_horizon, y_horizon = self._prepare_horizon_data(
                X_enhanced, survival_data, horizon
            )
            
            # Train model for this horizon with HPO if enabled
            if enable_hpo:
                horizon_model = self._train_horizon_model_with_hpo(
                    X_horizon, y_horizon, horizon, hpo_trials, cv_folds
                )
            else:
                horizon_model = self._train_horizon_model(
                    X_horizon, y_horizon, horizon
                )
            
            self.horizon_models[horizon] = horizon_model
            horizon_results[horizon] = self._evaluate_horizon_model(
                horizon_model, X_horizon, y_horizon, horizon
            )
        
        # Train ensemble model for all horizons
        ensemble_model = self._train_ensemble_model(X_enhanced, survival_data)
        self.model = ensemble_model
        
        # Optimize entry timing parameters if enabled
        if enable_entry_timing_optimization:
            tprint_info("🎯 Optimizing entry timing parameters")
            try:
                from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
                    optimize_entry_timing, EntryTimingConfig
                )
                
                # Configure entry timing optimization
                entry_config = EntryTimingConfig(
                    n_trials=entry_timing_trials,
                    timeout_minutes=30,
                    random_state=42
                )
                
                # Optimize entry timing parameters
                entry_timing_result = optimize_entry_timing(
                    model=self,  # Use self as the model
                    X=X_enhanced,
                    y=y,
                    analyst_signals=analyst_signals,
                    hmm_regime_probs=hmm_regime_probs,
                    model_name="RandomSurvivalForestTactician",
                    config=entry_config,
                    optimization_method="optuna"
                )
                
                # Store optimization results
                self.entry_timing_optimization = entry_timing_result
                tprint_success(f"✅ Entry timing optimization completed")
                tprint_info(f"📊 Best profit: {entry_timing_result.profit:.4f}")
                tprint_info(f"📊 Best Sharpe: {entry_timing_result.sharpe_ratio:.4f}")
                tprint_info(f"📊 Win rate: {entry_timing_result.win_rate:.4f}")
                
            except ImportError as e:
                tprint_warning(f"⚠️ Entry timing optimization not available: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Entry timing optimization failed: {e}")
        
        # Calculate training metrics
        self.training_metrics = self._calculate_training_metrics(
            horizon_results, X_enhanced, survival_data
        )
        
        tprint_success("✅ Random Survival Forest training completed")
        tprint_info(f"📊 Trained models for {len(self.config.horizons)} horizons")
        
        return {
            'horizon_models': self.horizon_models,
            'ensemble_model': self.model,
            'training_metrics': self.training_metrics,
            'feature_names': enhanced_feature_names,
            'horizons': self.config.horizons,
            'horizon_weights': self.config.horizon_weights,
            'entry_timing_optimization': getattr(self, 'entry_timing_optimization', None)
        }
    
    def predict(self, 
                X: np.ndarray, 
                analyst_signals: Optional[np.ndarray] = None,
                hmm_regime_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Predict entry timing probabilities for different horizons.
        
        Args:
            X: Input features
            analyst_signals: Analyst model outputs
            hmm_regime_probs: HMM regime probabilities
            
        Returns:
            Dictionary with predictions for each horizon
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare features
        X_enhanced, _ = self._prepare_features(
            X, self.feature_names, analyst_signals, hmm_regime_probs
        )
        
        # Get predictions for each horizon
        predictions = {}
        for horizon, model in self.horizon_models.items():
            # Predict survival function
            survival_func = model.predict_survival_function(X_enhanced)
            
            # Calculate entry probability within horizon
            entry_prob = self._calculate_entry_probability(survival_func, horizon)
            
            predictions[f'horizon_{horizon}m'] = {
                'entry_probability': entry_prob,
                'survival_function': survival_func,
                'median_time': self._calculate_median_time(survival_func),
                'confidence': self._calculate_confidence(survival_func)
            }
        
        # Ensemble prediction
        ensemble_prediction = self._ensemble_predict(predictions)
        
        return {
            'horizon_predictions': predictions,
            'ensemble_prediction': ensemble_prediction,
            'recommended_horizon': self._recommend_horizon(predictions),
            'timing_confidence': self._calculate_timing_confidence(predictions)
        }
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]]):
        """Validate input data."""
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        
        if feature_names and len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature names length {len(feature_names)} != features {X.shape[1]}")
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
    
    def _prepare_features(self, 
                          X: np.ndarray, 
                          feature_names: Optional[List[str]],
                          analyst_signals: Optional[np.ndarray],
                          hmm_regime_probs: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Prepare enhanced features for survival analysis."""
        enhanced_features = [X]
        enhanced_names = list(feature_names) if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Add analyst signals
        if analyst_signals is not None:
            enhanced_features.append(analyst_signals)
            enhanced_names.extend([f'analyst_signal_{i}' for i in range(analyst_signals.shape[1])])
        
        # Add HMM regime probabilities
        if hmm_regime_probs is not None:
            enhanced_features.append(hmm_regime_probs)
            enhanced_names.extend([f'hmm_regime_{i}' for i in range(hmm_regime_probs.shape[1])])
        
        # Add timing-specific features
        if self.config.enable_timing_features:
            timing_features = self._create_timing_features(X)
            enhanced_features.append(timing_features)
            enhanced_names.extend([f'timing_feature_{i}' for i in range(timing_features.shape[1])])
        
        return np.hstack(enhanced_features), enhanced_names
    
    def _create_timing_features(self, X: np.ndarray) -> np.ndarray:
        """Create timing-specific features."""
        timing_features = []
        
        # Price momentum features
        if X.shape[1] > 0:  # Assuming first column is price-related
            price_momentum = np.diff(X[:, 0], prepend=X[0, 0])
            timing_features.append(price_momentum.reshape(-1, 1))
        
        # Volatility features
        if X.shape[1] > 1:  # Assuming second column is volatility-related
            volatility = X[:, 1]
            timing_features.append(volatility.reshape(-1, 1))
        
        # Time-based features
        time_features = np.arange(len(X)).reshape(-1, 1)
        timing_features.append(time_features)
        
        return np.hstack(timing_features) if timing_features else np.zeros((X.shape[0], 1))
    
    def _prepare_survival_data(self, y: np.ndarray, multi_horizon_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare survival data for training."""
        # Convert y to survival format
        # Assuming y contains timing information (positive = entry time, negative = censored)
        event_observed = y > 0
        duration = np.abs(y)
        
        # Create survival array
        survival_array = np.array([(event_observed[i], duration[i]) for i in range(len(y))], 
                                dtype=[('event', bool), ('time', float)])
        
        return {
            'survival_array': survival_array,
            'event_observed': event_observed,
            'duration': duration,
            'multi_horizon_data': multi_horizon_data
        }
    
    def _prepare_horizon_data(self, X: np.ndarray, survival_data: Dict[str, Any], horizon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for specific horizon."""
        # Filter data for this horizon
        horizon_mask = survival_data['duration'] <= horizon
        
        X_horizon = X[horizon_mask]
        y_horizon = survival_data['survival_array'][horizon_mask]
        
        return X_horizon, y_horizon
    
    def _train_horizon_model(self, X: np.ndarray, y: np.ndarray, horizon: float) -> RandomSurvivalForest:
        """Train Random Survival Forest for specific horizon."""
        model = RandomSurvivalForest(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            max_samples=self.config.max_samples,
            random_state=42
        )
        
        model.fit(X, y)
        return model
    
    def _train_horizon_model_with_hpo(self, X: np.ndarray, y: np.ndarray, horizon: float, hpo_trials: int, cv_folds: int) -> RandomSurvivalForest:
        """Train Random Survival Forest for specific horizon using existing HPO tools."""
        try:
            from src.utils.ml_common.validation.hpo_overfitting_prevention import (
                HPOWithOverfittingPrevention, 
                HPOOverfittingPreventionConfig
            )
            from src.utils.ml_common.validation.universal_temporal_validation import (
                UniversalTimeSeriesSplit
            )
            
            # Configure HPO with existing tools and staged optimization
            hpo_config = HPOOverfittingPreventionConfig(
                n_trials=hpo_trials,
                timeout_minutes=30,  # 30 minutes per horizon
                enable_pruning=True,
                pruner_type="median",
                sampler_type="tpe",  # Bayesian TPE
                enable_nested_cv=True,
                outer_cv_folds=cv_folds,
                inner_cv_folds=3,
                regularization_methods=["l2", "dropout"],
                overfitting_detection=True,
                temporal_validation=True,
                # Enhanced staged HPO
                enable_staged_hpo=True,
                coarse_strategy="grid",
                coarse_grid_points=3,
                fine_grid_points=5,
                coarse_n_samples=50,
                bayes_n_trials=30,
                finalize_refine=True
            )
            
            # Create HPO optimizer
            hpo_optimizer = HPOWithOverfittingPrevention(hpo_config)
            
            # Define parameter space for Random Survival Forest
            param_space = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.3, 0.5]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]},
                'max_samples': {'type': 'float', 'low': 0.5, 'high': 1.0}
            }
            
            # Custom objective function for survival analysis
            def survival_objective(trial, X, y):
                # Sample parameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                    'random_state': 42
                }
                
                # Create model
                model = RandomSurvivalForest(**params)
                
                # Use temporal cross-validation
                tscv = UniversalTimeSeriesSplit(
                    n_splits=cv_folds,
                    test_size=0.2,
                    gap_size=1,
                    min_train_size=0.3
                )
                
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate with concordance index
                    predictions = model.predict(X_val)
                    c_index = concordance_index_censored(
                        y_val['event'], y_val['time'], predictions
                    )[0]
                    scores.append(c_index)
                
                return np.mean(scores)
            
            # Run staged HPO optimization (Grid Search + Bayesian TPE)
            optimization_result = hpo_optimizer.optimize_with_staged_hpo(
                model_class=RandomSurvivalForest,
                X=X,
                y=y,
                model_name=f"RandomSurvivalForest_{horizon}m",
                model_type="random_survival_forest",
                param_space=param_space,
                is_classification=False  # Survival analysis
            )
            
            # Get best parameters
            best_params = optimization_result.best_params
            
            # Train final model with best parameters
            final_model = RandomSurvivalForest(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                bootstrap=best_params['bootstrap'],
                max_samples=best_params['max_samples'],
                random_state=42
            )
            
            final_model.fit(X, y)
            
            tprint_success(f"✅ HPO completed for {horizon}m horizon using existing tools")
            tprint_info(f"📊 Best score: {optimization_result.best_score:.4f}")
            tprint_info(f"📊 Overfitting risk: {optimization_result.final_overfitting_risk}")
            
            return final_model
            
        except ImportError as e:
            tprint_warning(f"⚠️ HPO tools not available: {e}, using default parameters")
            return self._train_horizon_model(X, y, horizon)
        except Exception as e:
            tprint_warning(f"⚠️ HPO failed: {e}, using default parameters")
            return self._train_horizon_model(X, y, horizon)
    
    def _evaluate_horizon_model(self, model: RandomSurvivalForest, X: np.ndarray, y: np.ndarray, horizon: float) -> Dict[str, Any]:
        """Evaluate model performance for specific horizon."""
        # Predict survival function
        survival_func = model.predict_survival_function(X)
        
        # Calculate concordance index
        c_index = concordance_index_censored(
            y['event'], y['time'], 
            model.predict(X)
        )[0]
        
        return {
            'horizon': horizon,
            'concordance_index': c_index,
            'n_samples': len(X),
            'model_params': model.get_params()
        }
    
    def _train_ensemble_model(self, X: np.ndarray, survival_data: Dict[str, Any]) -> RandomSurvivalForest:
        """Train ensemble model for all horizons."""
        # Use all data for ensemble model
        model = RandomSurvivalForest(
            n_estimators=self.config.n_estimators * 2,  # More estimators for ensemble
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            max_samples=self.config.max_samples,
            random_state=42
        )
        
        model.fit(X, survival_data['survival_array'])
        return model
    
    def _calculate_entry_probability(self, survival_func, horizon: float) -> np.ndarray:
        """Calculate probability of entry within horizon."""
        # Get survival probability at horizon
        survival_prob = survival_func(horizon)
        
        # Entry probability = 1 - survival probability
        entry_prob = 1 - survival_prob
        
        return entry_prob
    
    def _calculate_median_time(self, survival_func) -> np.ndarray:
        """Calculate median time to entry."""
        # Find time where survival probability = 0.5
        times = np.linspace(0, 60, 1000)  # 0 to 60 minutes
        survival_probs = survival_func(times)
        
        median_times = []
        for i in range(len(survival_probs)):
            # Find where survival probability crosses 0.5
            survival_curve = survival_probs[i]
            median_idx = np.argmin(np.abs(survival_curve - 0.5))
            median_times.append(times[median_idx])
        
        return np.array(median_times)
    
    def _calculate_confidence(self, survival_func) -> np.ndarray:
        """Calculate confidence in timing prediction."""
        # Confidence based on how steep the survival curve is
        times = np.linspace(0, 30, 100)  # 0 to 30 minutes
        survival_probs = survival_func(times)
        
        confidences = []
        for i in range(len(survival_probs)):
            survival_curve = survival_probs[i]
            # Calculate slope of survival curve
            slope = np.gradient(survival_curve)
            # Confidence is based on maximum slope (steepest decline)
            confidence = np.max(np.abs(slope))
            confidences.append(confidence)
        
        return np.array(confidences)
    
    def _ensemble_predict(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble prediction from all horizons."""
        # Weighted average of entry probabilities
        weighted_probs = []
        for horizon, pred in predictions.items():
            horizon_minutes = float(horizon.split('_')[1].replace('m', ''))
            weight_idx = self.config.horizons.index(horizon_minutes)
            weight = self.config.horizon_weights[weight_idx]
            
            weighted_probs.append(weight * pred['entry_probability'])
        
        ensemble_prob = np.sum(weighted_probs, axis=0)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean([pred['confidence'] for pred in predictions.values()], axis=0)
        
        return {
            'entry_probability': ensemble_prob,
            'confidence': ensemble_confidence,
            'recommended_action': self._recommend_action(ensemble_prob, ensemble_confidence)
        }
    
    def _recommend_horizon(self, predictions: Dict[str, Any]) -> str:
        """Recommend optimal horizon based on predictions."""
        best_horizon = None
        best_score = -np.inf
        
        for horizon, pred in predictions.items():
            # Score based on entry probability and confidence
            score = pred['entry_probability'].mean() * pred['confidence'].mean()
            
            if score > best_score:
                best_score = score
                best_horizon = horizon
        
        return best_horizon
    
    def _calculate_timing_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall timing confidence."""
        all_confidences = []
        for pred in predictions.values():
            all_confidences.extend(pred['confidence'])
        
        return np.mean(all_confidences)
    
    def _recommend_action(self, entry_prob: np.ndarray, confidence: np.ndarray) -> List[str]:
        """Recommend trading action based on predictions."""
        actions = []
        
        for i in range(len(entry_prob)):
            prob = entry_prob[i]
            conf = confidence[i]
            
            if prob > 0.7 and conf > 0.5:
                actions.append('STRONG_ENTRY')
            elif prob > 0.5 and conf > 0.3:
                actions.append('MODERATE_ENTRY')
            elif prob > 0.3 and conf > 0.2:
                actions.append('WEAK_ENTRY')
            else:
                actions.append('NO_ENTRY')
        
        return actions
    
    def _calculate_training_metrics(self, horizon_results: Dict[str, Any], X: np.ndarray, survival_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive training metrics."""
        metrics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_horizons': len(self.config.horizons),
            'horizon_results': horizon_results,
            'overall_concordance': np.mean([result['concordance_index'] for result in horizon_results.values()]),
            'training_time': time.time() - getattr(self, '_training_start_time', time.time())
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from ensemble model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.feature_importances_
        
        return {
            'feature_names': self.feature_names,
            'importance_scores': importance,
            'top_features': sorted(zip(self.feature_names, importance), 
                                  key=lambda x: x[1], reverse=True)[:10]
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        import joblib
        
        model_data = {
            'horizon_models': self.horizon_models,
            'ensemble_model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        tprint_success(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        
        model_data = joblib.load(filepath)
        
        self.horizon_models = model_data['horizon_models']
        self.model = model_data['ensemble_model']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        
        tprint_success(f"✅ Model loaded from {filepath}")


# Multi-horizon integration
class MultiHorizonRandomSurvivalForest:
    """
    Multi-horizon Random Survival Forest for tactician models.
    
    Integrates with the multi-horizon framework to provide comprehensive
    timing predictions across different time horizons.
    """
    
    def __init__(self, config: Optional[SurvivalAnalysisConfig] = None):
        """Initialize multi-horizon Random Survival Forest."""
        self.config = config or SurvivalAnalysisConfig()
        self.horizon_models = {}
        self.ensemble_model = None
        
        if MULTI_HORIZON_AVAILABLE:
            self.multi_horizon_manager = MultiHorizonTrainingManager()
        else:
            self.multi_horizon_manager = None
            tprint_warning("⚠️ Multi-horizon framework not available")
    
    def fit_multi_horizon(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          horizon_config: Optional[HorizonConfig] = None) -> Dict[str, Any]:
        """Fit multi-horizon Random Survival Forest model."""
        if self.multi_horizon_manager is None:
            raise ValueError("Multi-horizon framework not available")
        
        # Use multi-horizon manager to prepare data
        horizon_data = self.multi_horizon_manager.prepare_multi_horizon_data(
            X, y, horizon_config
        )
        
        # Train Random Survival Forest for each horizon
        results = {}
        for horizon, data in horizon_data.items():
            rsf_model = RandomSurvivalForestTactician(self.config)
            horizon_result = rsf_model.fit(
                data['X'], data['y'], 
                data.get('feature_names'),
                data.get('analyst_signals'),
                data.get('hmm_regime_probs')
            )
            
            self.horizon_models[horizon] = rsf_model
            results[horizon] = horizon_result
        
        return results
    
    def predict_multi_horizon(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict across all horizons."""
        predictions = {}
        
        for horizon, model in self.horizon_models.items():
            horizon_pred = model.predict(X)
            predictions[horizon] = horizon_pred
        
        return predictions