"""
Enhanced Model Training Module

This module provides shallow/regularized/monotone GBDT and calibrated logistic regression
for improved model robustness and comparison.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings

# Import LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    # Fallback implementation if tprint not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class EnhancedModelTrainer:
    """
    Enhanced model trainer supporting shallow/regularized/monotone GBDT and calibrated logistic.
    """
    
    def __init__(
        self,
        model_type: str = "shallow_gbdt",
        calibration_method: str = "isotonic",
        random_state: int = 42,
        n_splits: int = 5,
        n_bags: int = 10
    ):
        """
        Initialize enhanced model trainer.
        
        Args:
            model_type: Type of model ('shallow_gbdt', 'regularized_gbdt', 'monotone_gbdt', 'calibrated_logistic')
            calibration_method: Calibration method ('isotonic', 'sigmoid', 'none')
            random_state: Random state for reproducibility
            n_splits: Number of CV splits
            n_bags: Number of bagged estimators
        """
        self.model_type = model_type
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.n_splits = n_splits
        self.n_bags = n_bags
        
        self.models = []
        self.oof_predictions = None
        self.training_metrics = {}
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters based on model type."""
        if self.model_type == "shallow_gbdt":
            return {
                'n_estimators': 100,  # Fewer trees for shallow model
                'max_depth': 3,        # Shallower depth
                'num_leaves': 7,       # Fewer leaves
                'learning_rate': 0.1,  # Higher learning rate
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,      # Stronger L1 regularization
                'reg_lambda': 0.5,     # Stronger L2 regularization
                'min_child_samples': 20,  # Higher minimum samples
                'min_split_gain': 0.1,   # Higher gain threshold
                'n_jobs': -1,
                'verbose': -1,
                'random_state': self.random_state,
            }
        
        elif self.model_type == "regularized_gbdt":
            return {
                'n_estimators': 200,
                'max_depth': 4,
                'num_leaves': 15,
                'learning_rate': 0.05,
                'subsample': 0.7,      # More aggressive subsampling
                'colsample_bytree': 0.7,
                'reg_alpha': 1.0,      # Strong L1 regularization
                'reg_lambda': 1.0,     # Strong L2 regularization
                'min_child_samples': 30,
                'min_split_gain': 0.05,
                'feature_fraction': 0.8,  # Feature subsampling
                'bagging_fraction': 0.8,   # Bagging fraction
                'bagging_freq': 5,         # Bagging frequency
                'n_jobs': -1,
                'verbose': -1,
                'random_state': self.random_state,
            }
        
        elif self.model_type == "monotone_gbdt":
            return {
                'n_estimators': 150,
                'max_depth': 5,
                'num_leaves': 20,
                'learning_rate': 0.08,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'min_child_samples': 25,
                'min_split_gain': 0.01,
                'n_jobs': -1,
                'verbose': -1,
                'random_state': self.random_state,
                # Monotone constraints will be set per feature
            }
        
        elif self.model_type == "calibrated_logistic":
            return {
                'penalty': 'l2',
                'C': 1.0,              # Inverse regularization strength
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': self.random_state,
                'class_weight': 'balanced',
            }
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_monotone_constraints(self, X: pd.DataFrame) -> Optional[Dict[str, int]]:
        """
        Determine monotone constraints for features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary of monotone constraints or None
        """
        if self.model_type != "monotone_gbdt":
            return None
        
        constraints = {}
        
        # Example monotone constraints based on feature names
        for col in X.columns:
            col_lower = col.lower()
            
            # Price-based features: positive monotonicity
            if any(term in col_lower for term in ['price', 'close', 'sma', 'ema', 'trend']):
                constraints[col] = 1
            
            # Volatility features: positive monotonicity
            elif any(term in col_lower for term in ['vol', 'std', 'atr', 'range']):
                constraints[col] = 1
            
            # Momentum features: positive monotonicity
            elif any(term in col_lower for term in ['momentum', 'rsi', 'macd']):
                constraints[col] = 1
            
            # Volume features: positive monotonicity
            elif any(term in col_lower for term in ['volume', 'liquidity']):
                constraints[col] = 1
            
            # Microstructure features: negative monotonicity (higher spread = lower probability)
            elif any(term in col_lower for term in ['spread', 'noise', 'gap']):
                constraints[col] = -1
            
            # Default: no constraint
            else:
                constraints[col] = 0
        
        return constraints
    
    def train_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train a single model."""
        params = self.get_model_params()
        
        if self.model_type in ["shallow_gbdt", "regularized_gbdt", "monotone_gbdt"]:
            if not LGB_AVAILABLE:
                raise ImportError("LightGBM is required for GBDT models")
            
            # Add monotone constraints if applicable
            if self.model_type == "monotone_gbdt":
                monotone_constraints = self.get_monotone_constraints(X_train)
                if monotone_constraints:
                    params['monotone_constraints'] = [monotone_constraints.get(col, 0) for col in X_train.columns]
            
            model = lgb.LGBMClassifier(**params)
            
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            return model
        
        elif self.model_type == "calibrated_logistic":
            # Base logistic regression
            base_model = LogisticRegression(**params)
            
            # Apply calibration if requested
            if self.calibration_method != "none":
                calibrated_model = CalibratedClassifierCV(
                    base_model,
                    method=self.calibration_method,
                    cv=3  # Internal CV for calibration
                )
            else:
                calibrated_model = base_model
            
            if sample_weights is not None:
                calibrated_model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                calibrated_model.fit(X_train, y_train)
            
            return calibrated_model
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, List[Any]]:
        """
        Train model with time series cross-validation and bagging.
        
        Args:
            X: Feature matrix
            y: Target series
            sample_weights: Sample weights
            
        Returns:
            Tuple of (OOF predictions DataFrame, trained models)
        """
        tprint_info(f"Training {self.model_type} model with {self.n_splits} CV splits and {self.n_bags} bags")
        
        # Prepare output
        oof_probs = np.full(len(y), np.nan, dtype=float)
        self.models = []
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None
            
            # Skip if insufficient class variety
            if len(np.unique(y_train.dropna())) < 2:
                tprint_warning(f"Fold {fold_idx}: Insufficient class variety, skipping")
                continue
            
            # Train bagged ensemble for this fold
            fold_probs = []
            for bag_idx in range(self.n_bags):
                try:
                    # Bootstrap sample
                    rng = np.random.RandomState(self.random_state + fold_idx * 100 + bag_idx)
                    n_train = len(X_train)
                    boot_idx = rng.choice(n_train, size=n_train, replace=True)
                    
                    X_boot = X_train.iloc[boot_idx]
                    y_boot = y_train.iloc[boot_idx]
                    w_boot = w_train[boot_idx] if w_train is not None else None
                    
                    # Train model
                    model = self.train_single_model(X_boot, y_boot, w_boot)
                    probs = model.predict_proba(X_val)[:, 1]
                    
                    fold_probs.append(probs)
                    self.models.append(model)
                    
                except Exception as e:
                    tprint_warning(f"Bag {bag_idx} fold {fold_idx} failed: {e}")
                    continue
            
            # Average predictions across bags
            if fold_probs:
                oof_probs[val_idx] = np.mean(fold_probs, axis=0)
        
        # Store OOF predictions
        self.oof_predictions = oof_probs
        
        # Calculate training metrics
        valid_mask = ~np.isnan(oof_probs)
        if valid_mask.sum() > 0:
            y_valid = y[valid_mask]
            oof_valid = oof_probs[valid_mask]
            
            self.training_metrics = {
                'auc': float(roc_auc_score(y_valid, oof_valid)),
                'brier_score': float(brier_score_loss(y_valid, oof_valid)),
                'n_valid_samples': int(valid_mask.sum()),
                'mean_prediction': float(np.mean(oof_valid)),
                'std_prediction': float(np.std(oof_valid)),
            }
            
            tprint_success(f"Training completed - AUC: {self.training_metrics['auc']:.4f}, "
                          f"Brier: {self.training_metrics['brier_score']:.4f}")
        
        # Build output DataFrame
        oof_df = pd.DataFrame({
            f'{self.model_type}_mean': oof_probs,
            f'{self.model_type}_lower': oof_probs * 0.9,
            f'{self.model_type}_upper': oof_probs * 1.1,
        }, index=X.index)
        
        return oof_df, self.models
    
    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple model types.
        
        Args:
            X: Feature matrix
            y: Target series
            sample_weights: Sample weights
            model_types: List of model types to compare
            
        Returns:
            Dictionary with comparison results
        """
        if model_types is None:
            model_types = ["shallow_gbdt", "regularized_gbdt", "monotone_gbdt", "calibrated_logistic"]
        
        results = {}
        
        for model_type in model_types:
            tprint_info(f"Training {model_type} for comparison...")
            
            # Create trainer for this model type
            trainer = EnhancedModelTrainer(
                model_type=model_type,
                calibration_method=self.calibration_method,
                random_state=self.random_state,
                n_splits=self.n_splits,
                n_bags=self.n_bags
            )
            
            try:
                # Train model
                oof_df, models = trainer.train(X, y, sample_weights)
                
                # Store results
                results[model_type] = {
                    'oof_predictions': oof_df,
                    'models': models,
                    'metrics': trainer.training_metrics,
                    'trainer': trainer
                }
                
                tprint_success(f"{model_type} - AUC: {trainer.training_metrics['auc']:.4f}")
                
            except Exception as e:
                tprint_error(f"{model_type} training failed: {e}")
                results[model_type] = {
                    'error': str(e),
                    'metrics': {}
                }
        
        # Create comparison summary
        tprint_info("\n=== Model Comparison Summary ===")
        for model_type, result in results.items():
            if 'metrics' in result and result['metrics']:
                metrics = result['metrics']
                print(f"{model_type:20s}: AUC={metrics['auc']:.4f}, Brier={metrics['brier_score']:.4f}")
            else:
                print(f"{model_type:20s}: FAILED")
        
        return results


def train_enhanced_models(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train enhanced models with shallow/regularized/monotone GBDT and calibrated logistic.
    
    Args:
        X: Feature matrix
        y: Target series
        sample_weights: Sample weights
        config: Configuration dictionary
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with training results
    """
    if config is None:
        config = {}
    
    # Initialize trainer
    model_type = config.get('enhanced_model_type', 'shallow_gbdt')
    calibration_method = config.get('calibration_method', 'isotonic')
    
    trainer = EnhancedModelTrainer(
        model_type=model_type,
        calibration_method=calibration_method,
        random_state=config.get('random_state', 42),
        n_splits=config.get('cv_splits', 5),
        n_bags=config.get('n_bags', 10)
    )
    
    # Train model
    oof_df, models = trainer.train(X, y, sample_weights)
    
    # Compare models if requested
    comparison_results = {}
    if config.get('compare_models', True):
        comparison_results = trainer.compare_models(X, y, sample_weights)
    
    results = {
        'primary_model': {
            'type': model_type,
            'oof_predictions': oof_df,
            'models': models,
            'metrics': trainer.training_metrics,
            'trainer': trainer
        },
        'comparison': comparison_results,
        'config': config
    }
    
    if verbose:
        tprint_success(f"Enhanced model training completed for {model_type}")
        if trainer.training_metrics:
            metrics = trainer.training_metrics
            tprint_info(f"  AUC: {metrics['auc']:.4f}")
            tprint_info(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    return results
