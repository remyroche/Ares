"""
SR Performance Predictor

Multi-output LightGBM model for predicting SR level performance metrics.
Predicts bounce_strength, hold_strength, and trade_profit for tested SR levels.
Includes SHAP integration for interpretability.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import shap

# Import HPO utilities
try:
    from src.utils.ml_common.optimization.hpo_utils import optimize_hyperparameters
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    optimize_hyperparameters = None

logger = logging.getLogger(__name__)


class SRPerformancePredictor:
    """Multi-output LightGBM model for SR performance prediction.
    
    Predicts three key metrics for SR levels that get tested:
    - bounce_strength: How strongly price bounces (0-1)
    - hold_strength: Whether level holds or breaks (0-1)
    - trade_profit: Simulated trade profitability (-1 to 1)
    
    Uses SHAP for model interpretability and feature importance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize SR performance predictor.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.models = {}  # Dict of {target_name: lgb.Booster}
        self.shap_explainers = {}  # Dict of {target_name: shap.TreeExplainer}
        self.feature_names = None
        self.training_metrics = {}
        self.hpo_results = {}  # Store HPO results for each target
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define prediction targets
        self.targets = ['bounce_strength', 'hold_strength', 'trade_profit']
        
    def _get_default_config(self) -> Dict:
        """Default LightGBM configuration for SR performance prediction.
        
        Anti-overfitting configuration:
        - Strong regularization to prevent overfitting
        - Shallow trees for generalization
        - High min_data_in_leaf for robustness
        """
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # Tree complexity
            'num_leaves': 15,
            'max_depth': 4,
            
            # Regularization
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_data_in_leaf': 50,
            
            # Learning
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            
            'verbose': -1,
            'seed': 42,
            'force_col_wise': True
        }
    
    def train(self, 
              training_data: pd.DataFrame,
              n_folds: int = 5,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 50,
              filter_untested: bool = True) -> Dict:
        """Train multi-output model with time series cross-validation.
        
        Args:
            training_data: DataFrame with features + target columns
            n_folds: Number of CV folds
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping patience
            filter_untested: Filter out untested levels (recommended)
            
        Returns:
            Dictionary with CV scores and metrics for all targets
        """
        self.logger.info(f"🤖 Training SR Performance Predictor")
        self.logger.info(f"   Training samples (raw): {len(training_data)}")
        
        # Filter out untested levels (hit_rate == 0)
        if filter_untested and 'hit_rate' in training_data.columns:
            original_len = len(training_data)
            training_data = training_data[training_data['hit_rate'] > 0].copy()
            filtered_count = original_len - len(training_data)
            self.logger.info(f"   📊 Filtered out {filtered_count} untested levels")
        
        # Additional quality filtering
        if 'feature_strength' in training_data.columns:
            original_len = len(training_data)
            training_data = training_data[training_data['feature_strength'] >= 0.3].copy()
            filtered_count = original_len - len(training_data)
            self.logger.info(f"   📊 Filtered out {filtered_count} weak levels (strength < 0.3)")
        
        self.logger.info(f"   ✅ Training samples (filtered): {len(training_data)}")
        
        if len(training_data) < 100:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples (need ≥100)")
        
        # Identify feature columns
        exclude_cols = {'date', 'symbol', 'exchange', 'timeframe', 'quality_score',
                       'hit_rate', 'bounce_strength', 'hold_strength', 'trade_profit',
                       'sample_weight'}
        
        feature_cols = [c for c in training_data.columns 
                       if c.startswith('feature_') or c not in exclude_cols]
        
        self.feature_names = feature_cols
        self.logger.info(f"   Using {len(self.feature_names)} features")
        
        # Prepare features
        X = training_data[self.feature_names].fillna(0.0)
        
        # Get sample weights if available
        sample_weight = None
        if 'sample_weight' in training_data.columns:
            sample_weight = training_data['sample_weight'].values
            self.logger.info(f"   Using sample weights")
        
        # Train each target separately
        all_results = {}
        
        for target in self.targets:
            self.logger.info(f"\n📈 Training model for: {target}")
            
            if target not in training_data.columns:
                self.logger.warning(f"   ⚠️ Target '{target}' not found in data, skipping")
                continue
            
            y = training_data[target].values
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_folds)
            cv_scores = []
            cv_predictions = []
            cv_actuals = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply sample weights if available
                train_weight = sample_weight[train_idx] if sample_weight is not None else None
                val_weight = sample_weight[val_idx] if sample_weight is not None else None
                
                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train, weight=train_weight)
                val_data = lgb.Dataset(X_val, label=y_val, weight=val_weight, reference=train_data)
                
                # Train
                model = lgb.train(
                    self.config,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'val'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                # Evaluate
                val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                mae = mean_absolute_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)
                
                cv_scores.append({'rmse': rmse, 'mae': mae, 'r2': r2})
                cv_predictions.extend(val_pred)
                cv_actuals.extend(y_val)
                
                self.logger.info(f"   Fold {fold}/{n_folds}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            # Train final model on all data
            self.logger.info(f"   Training final model on all data...")
            
            train_data = lgb.Dataset(X, label=y, weight=sample_weight)
            final_model = lgb.train(
                self.config,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            self.models[target] = final_model
            
            # Create SHAP explainer
            self.logger.info(f"   Creating SHAP explainer...")
            self.shap_explainers[target] = shap.TreeExplainer(final_model)
            
            # Compute CV statistics
            cv_metrics = {
                'rmse_mean': np.mean([s['rmse'] for s in cv_scores]),
                'rmse_std': np.std([s['rmse'] for s in cv_scores]),
                'mae_mean': np.mean([s['mae'] for s in cv_scores]),
                'mae_std': np.std([s['mae'] for s in cv_scores]),
                'r2_mean': np.mean([s['r2'] for s in cv_scores]),
                'r2_std': np.std([s['r2'] for s in cv_scores]),
            }
            
            # Compute ranking correlation
            spearman_corr, _ = spearmanr(cv_actuals, cv_predictions)
            cv_metrics['spearman_correlation'] = spearman_corr
            
            all_results[target] = cv_metrics
            
            self.logger.info(f"   ✅ Final metrics for {target}:")
            self.logger.info(f"      RMSE: {cv_metrics['rmse_mean']:.4f} ± {cv_metrics['rmse_std']:.4f}")
            self.logger.info(f"      MAE:  {cv_metrics['mae_mean']:.4f} ± {cv_metrics['mae_std']:.4f}")
            self.logger.info(f"      R²:   {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
            self.logger.info(f"      Spearman: {spearman_corr:.4f}")
        
        self.training_metrics = all_results
        
        # Log feature importance
        self._log_feature_importance()
    
        # Compute and cache SHAP values
        self.logger.info("Caching SHAP values from training data...")
        self.compute_shap_values(training_data=training_data)

    return all_results
    
    def train_with_hpo(self,
                      training_data: pd.DataFrame,
                      n_trials: int = 50,
                      hpo_method: str = 'bayesian',
                      n_folds: int = 5,
                      num_boost_round: int = 1000,
                      early_stopping_rounds: int = 50,
                      filter_untested: bool = True) -> Dict:
        """Train multi-output model with hyperparameter optimization.
        
        Args:
            training_data: DataFrame with features + target columns
            n_trials: Number of HPO trials per target
            hpo_method: HPO method ('bayesian', 'staged', 'multi_objective')
            n_folds: Number of CV folds
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping patience
            filter_untested: Filter out untested levels (recommended)
            
        Returns:
            Dictionary with optimized parameters and metrics for all targets
        """
        if not HPO_AVAILABLE:
            self.logger.warning("⚠️ HPO utilities not available, falling back to default config")
            return self.train(training_data, n_folds, num_boost_round, early_stopping_rounds, filter_untested)
        
        self.logger.info(f"🤖 Training SR Performance Predictor with HPO")
        self.logger.info(f"   HPO Method: {hpo_method}, Trials: {n_trials}")
        self.logger.info(f"   Training samples (raw): {len(training_data)}")
        
        # Filter data (same as regular train)
        if filter_untested and 'hit_rate' in training_data.columns:
            original_len = len(training_data)
            training_data = training_data[training_data['hit_rate'] > 0].copy()
            filtered_count = original_len - len(training_data)
            self.logger.info(f"   📊 Filtered out {filtered_count} untested levels")
        
        if 'feature_strength' in training_data.columns:
            original_len = len(training_data)
            training_data = training_data[training_data['feature_strength'] >= 0.3].copy()
            filtered_count = original_len - len(training_data)
            self.logger.info(f"   📊 Filtered out {filtered_count} weak levels (strength < 0.3)")
        
        self.logger.info(f"   ✅ Training samples (filtered): {len(training_data)}")
        
        if len(training_data) < 50:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples (need ≥50 for HPO)")
        
        # Identify feature columns
        exclude_cols = {'date', 'symbol', 'exchange', 'timeframe', 'quality_score',
                       'hit_rate', 'bounce_strength', 'hold_strength', 'trade_profit',
                       'sample_weight'}
        
        feature_cols = [c for c in training_data.columns 
                       if c.startswith('feature_') or c not in exclude_cols]
        
        self.feature_names = feature_cols
        self.logger.info(f"   Using {len(self.feature_names)} features")
        
        # Prepare features
        X = training_data[self.feature_names].fillna(0.0).values
        
        # Get sample weights if available
        sample_weight = None
        if 'sample_weight' in training_data.columns:
            sample_weight = training_data['sample_weight'].values
            self.logger.info(f"   Using sample weights")
        
        # Define LightGBM search space for regression
        search_space = {
            'num_leaves': {'type': 'int', 'low': 15, 'high': 63},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 0.9},
            'bagging_fraction': {'type': 'float', 'low': 0.5, 'high': 0.9},
            'bagging_freq': {'type': 'int', 'low': 1, 'high': 10},
            'min_data_in_leaf': {'type': 'int', 'low': 20, 'high': 100},
            'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 2.0},
            'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 2.0},
        }
        
        # Train each target with HPO
        all_results = {}
        
        for target in self.targets:
            self.logger.info(f"\n📈 Optimizing hyperparameters for: {target}")
            
            if target not in training_data.columns:
                self.logger.warning(f"   ⚠️ Target '{target}' not found in data, skipping")
                continue
            
            y = training_data[target].values
            
            # Define model factory for LightGBM
            def lgb_factory(**params):
                # Merge with base config
                model_params = self.config.copy()
                model_params.update(params)
                return lgb.LGBMRegressor(**model_params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_folds)
            
            # Run HPO
            self.logger.info(f"   🔍 Running {hpo_method} optimization with {n_trials} trials...")
            
            # Configure HPO to disable non-linear transformations
            # (they rename parameters like learning_rate -> raw_learning_rate which breaks LightGBM)
            hpo_config = {
                'use_nonlinear_optimization': False,  # Disable parameter renaming
                'enable_parallel': True,
                'max_workers': 4,
                'enable_monitoring': True
            }
            
            hpo_result = optimize_hyperparameters(
                model_factory=lgb_factory,
                X=X,
                y=y,
                search_space=search_space,
                n_trials=n_trials,
                method=hpo_method,
                scoring='neg_mean_squared_error',  # Use MSE for regression
                cv=tscv,
                config=hpo_config
            )
            
            # Store HPO results
            self.hpo_results[target] = hpo_result
            
            if 'best_params' in hpo_result:
                best_params = hpo_result['best_params']
                best_score = hpo_result.get('best_score', 0)
                
                self.logger.info(f"   ✅ Best params found:")
                for param, value in best_params.items():
                    self.logger.info(f"      {param}: {value}")
                self.logger.info(f"   Best CV score: {-best_score:.4f} RMSE")
                
                # Train final model with best params
                self.logger.info(f"   Training final model with optimized parameters...")
                
                final_config = self.config.copy()
                final_config.update(best_params)
                
                train_data = lgb.Dataset(X, label=y, weight=sample_weight)
                final_model = lgb.train(
                    final_config,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data],
                    valid_names=['train'],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                
                self.models[target] = final_model
                
                # Create SHAP explainer
                self.logger.info(f"   Creating SHAP explainer...")
                self.shap_explainers[target] = shap.TreeExplainer(final_model)
                
                # Store metrics
                all_results[target] = {
                    'best_params': best_params,
                    'best_cv_score': -best_score,
                    'hpo_method': hpo_method,
                    'n_trials': n_trials,
                }
                
            else:
                self.logger.warning(f"   ⚠️ HPO failed for {target}, using default config")
                # Fall back to default config
                train_data = lgb.Dataset(X, label=y, weight=sample_weight)
                final_model = lgb.train(
                    self.config,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data],
                    valid_names=['train'],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                self.models[target] = final_model
                self.shap_explainers[target] = shap.TreeExplainer(final_model)
        
        self.training_metrics = all_results
        
        # Log feature importance
        self._log_feature_importance()
        
        # Compute and cache SHAP values
        self.logger.info("Caching SHAP values from training data...")
        self.compute_shap_values(training_data=training_data)
    
        self.logger.info(f"\n✅ HPO training complete for all targets!")

    def compute_shap_values(self, training_data: pd.DataFrame):
        """
        Computes and caches SHAP values for all models using the provided training data.
        This is useful after loading a model to avoid re-computing on every call.
    
        Args:
            training_data: The training DataFrame (or compatible data) to use for SHAP.
        """
        self.logger.info("Calculating and caching SHAP values for all targets...")
        if self.feature_names is None:
            raise ValueError("Model feature names are not set. Load or train a model first.")
    
        try:
            X = training_data[self.feature_names].fillna(0.0)
        except KeyError as e:
            missing = set(self.feature_names) - set(training_data.columns)
            self.logger.error(f"Missing features for SHAP computation: {missing}")
            raise ValueError(f"Missing features: {missing}")
    
        self.X_train_for_shap = X.copy()
        self.shap_values.clear()  # Clear any old values
    
        for target in self.targets:
            if target in self.shap_explainers:
                self.logger.info(f"   Calculating for {target}...")
                self.shap_values[target] = self.shap_explainers[target].shap_values(self.X_train_for_shap)
            else:
                self.logger.warning(f"   No SHAP explainer found for {target}, skipping.")
        self.logger.info("SHAP values cached.")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict all performance metrics.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Dictionary of {target_name: predictions_array}
        """
        if not self.models:
            raise ValueError("Models not trained! Call train() first or load() a trained model.")
        
        # Ensure feature order matches training
        try:
            X = features[self.feature_names].fillna(0.0)
        except KeyError as e:
            missing = set(self.feature_names) - set(features.columns)
            self.logger.error(f"Missing features: {missing}")
            raise ValueError(f"Missing features: {missing}")
        
        # Predict each target
        predictions = {}
        for target, model in self.models.items():
            pred = model.predict(X)
            
            # Clip to valid ranges
            if target == 'trade_profit':
                pred = np.clip(pred, -1, 1)
            else:  # bounce_strength, hold_strength
                pred = np.clip(pred, 0, 1)
            
            predictions[target] = pred
        
        return predictions
    
    def predict_single(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        """Predict performance for a single SR level.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            Dictionary of {target_name: prediction}
        """
        features_df = pd.DataFrame([features_dict])
        predictions = self.predict(features_df)
        return {target: float(pred[0]) for target, pred in predictions.items()}
    
    def explain_prediction(self, 
                          features: pd.DataFrame,
                          target: str,
                          sample_idx: int = 0) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction.
        
        Args:
            features: DataFrame with features
            target: Target to explain ('bounce_strength', 'hold_strength', 'trade_profit')
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with prediction, base_value, and SHAP values
        """
        if target not in self.shap_explainers:
            raise ValueError(f"No explainer for target '{target}'")
        
        X = features[self.feature_names].fillna(0.0)
        explainer = self.shap_explainers[target]
        
        # Get SHAP values
        shap_values = explainer.shap_values(X.iloc[sample_idx:sample_idx+1])
        
        # Get prediction
        prediction = self.models[target].predict(X.iloc[sample_idx:sample_idx+1])[0]
        
        return {
            'prediction': float(prediction),
            'base_value': float(explainer.expected_value),
            'shap_values': {
                feature: float(shap_val)
                for feature, shap_val in zip(self.feature_names, shap_values[0])
            },
            'feature_values': features.iloc[sample_idx][self.feature_names].to_dict()
        }
    
    def get_feature_importance(self, 
                               target: str,
                               method: str = 'shap',
                               top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a target.
        
        Args:
            target: Target name
            method: 'shap' or 'gain'
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and importance scores
        """
        if target not in self.models:
            raise ValueError(f"No model for target '{target}'")
            
        if method == 'shap':
                if target not in self.shap_values or self.X_train_for_shap is None:
                    self.logger.warning(f"SHAP values not cached for '{target}'.")
                    self.logger.warning("Call compute_shap_values(training_data) or use 'gain' method.")
                    # Fallback to gain as before
                    self.logger.warning("Falling back to 'gain' method.")
                    method = 'gain'
                else:
                    # Use cached values
                    self.logger.info(f"Using cached SHAP values for {target}.")
                    shap_sum = np.abs(self.shap_values[target]).mean(axis=0)
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': shap_sum
                    }).sort_values('importance', ascending=False)
        
                    return importance_df.head(top_n)
                                   
                            
        if method == 'shap':
            # Use mean absolute SHAP values
            if target not in self.shap_explainers:
                raise ValueError(f"No SHAP explainer for '{target}'")
            
            # This requires calling on training data - placeholder for now
            # In practice, you'd pass training data or cache SHAP values
            self.logger.warning("SHAP importance requires training data. Use 'gain' method instead.")
            method = 'gain'
        
        if method == 'gain':
            # Use LightGBM's built-in feature importance
            importance = self.models[target].feature_importance(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        
        raise ValueError(f"Unknown method: {method}")
    
    def plot_shap_summary(self, 
                         target: str,
                         save_path: Optional[Path] = None):
        """Generate SHAP summary plot from cached training data.
    
        Args:
            target: Target to explain
            save_path: Optional path to save plot
        """
        if target not in self.shap_explainers:
            raise ValueError(f"No explainer for target '{target}'")
    
        if target not in self.shap_values or self.X_train_for_shap is None:
            raise ValueError(
                f"SHAP values or X_train not cached for target '{target}'. "
                f"Run train() or call compute_shap_values(training_data) after load()."
            )
    
        X = self.X_train_for_shap
        shap_values = self.shap_values[target]
    
        # Compute SHAP values
        self.logger.info(f"Generating SHAP summary plot for {target} from cache...")
    
        # Create summary plot
        plt.figure(figsize=(10, 8))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved SHAP plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, save_dir: Path):
        """Save models, explainers, and metadata.
        
        Args:
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for target, model in self.models.items():
            model_path = save_dir / f'model_{target}.txt'
            model.save_model(str(model_path))
            self.logger.info(f"Saved {target} model to {model_path}")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'targets': self.targets,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'hpo_results': self.hpo_results if hasattr(self, 'hpo_results') else {},
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Saved all models to {save_dir}")
    
    def load(self, load_dir: Path):
        """Load models and metadata.
        
        Args:
            load_dir: Directory containing saved models
        """
        load_dir = Path(load_dir)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Load directory not found: {load_dir}")
        
        # Load metadata
        metadata_path = load_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.targets = metadata['targets']
        self.config = metadata['config']
        self.training_metrics = metadata.get('training_metrics', {})
        
        # Load each model
        for target in self.targets:
            model_path = load_dir / f'model_{target}.txt'
            if model_path.exists():
                self.models[target] = lgb.Booster(model_file=str(model_path))
                # Recreate SHAP explainer
                self.shap_explainers[target] = shap.TreeExplainer(self.models[target])
                self.logger.info(f"Loaded {target} model from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")

        # Clear any cached SHAP values from a previous instance
        self.shap_values = {}
        self.X_train_for_shap = None
        self.logger.info(f"✅ Loaded models from {load_dir}")
        self.logger.info("Run compute_shap_values(training_data) to cache SHAP values for loaded models.")
        
        self.logger.info(f"✅ Loaded models from {load_dir}")
    
    def _log_feature_importance(self):
        """Log top features for each target."""
        for target, model in self.models.items():
            importance = model.feature_importance(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"\n📊 Top 15 features for {target}:")
            for idx, row in importance_df.head(15).iterrows():
                self.logger.info(f"   {row['feature']:<40} {row['importance']:>10.0f}")

