"""
SR Quality Model - LightGBM

Pure ML-based SR level quality prediction.
Replaces hand-crafted weighted scoring with data-driven predictions.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)


class SRQualityModel:
    """LightGBM model for predicting SR level quality.
    
    PURE ML APPROACH:
    - Trained on historical SR performance data
    - Predicts quality_score (0-1) from features
    - Replaces weighted composite scoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize SR quality model.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _get_default_config(self) -> Dict:
        """Default LightGBM configuration optimized for SR quality prediction."""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42,
            'force_col_wise': True
        }
    
    def train(self, training_data: pd.DataFrame, 
             target_column: str = 'quality_score',
             n_folds: int = 5,
             num_boost_round: int = 1000,
             early_stopping_rounds: int = 50) -> Dict:
        """Train LightGBM model with time series cross-validation.
        
        Args:
            training_data: DataFrame with features + quality_score
            target_column: Target column to predict
            n_folds: Number of CV folds
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with CV scores and metrics
        """
        
        self.logger.info(f"🤖 Training SR Quality Model")
        self.logger.info(f"   Training samples: {len(training_data)}")
        
        # Separate features from targets/metadata
        exclude_cols = ['date', 'symbol', 'exchange', 'timeframe', 
                       'quality_score', 'hit_rate', 'bounce_strength', 
                       'hold_strength', 'trade_profit']
        
        feature_cols = [c for c in training_data.columns 
                       if c not in exclude_cols and not pd.isna(training_data[c]).all()]
        
        X = training_data[feature_cols]
        y = training_data[target_column]
        
        self.feature_names = feature_cols
        
        self.logger.info(f"   Features: {len(feature_cols)}")
        self.logger.info(f"   Target: {target_column}")
        self.logger.info(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")
        self.logger.info(f"   Target mean: {y.mean():.3f} ± {y.std():.3f}")
        
        # Handle missing values
        X = X.fillna(0.0)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        cv_scores = []
        fold_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"\n  📈 Training Fold {fold_idx + 1}/{n_folds}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.logger.info(f"     Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.config,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            fold_scores = {
                'fold': fold_idx,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'val_mae': mean_absolute_error(y_val, y_pred_val),
                'num_boost_rounds': model.current_iteration()
            }
            
            cv_scores.append(fold_scores)
            fold_models.append(model)
            
            self.logger.info(f"     ✓ Val RMSE: {fold_scores['val_rmse']:.4f} | R²: {fold_scores['val_r2']:.4f} | MAE: {fold_scores['val_mae']:.4f}")
        
        # Select best model (lowest validation RMSE)
        best_idx = np.argmin([s['val_rmse'] for s in cv_scores])
        self.model = fold_models[best_idx]
        
        self.logger.info(f"\n🎯 Best Model: Fold {best_idx + 1}")
        self.logger.info(f"   Val RMSE: {cv_scores[best_idx]['val_rmse']:.4f}")
        self.logger.info(f"   Val R²: {cv_scores[best_idx]['val_r2']:.4f}")
        self.logger.info(f"   Val MAE: {cv_scores[best_idx]['val_mae']:.4f}")
        
        # Average CV scores
        avg_metrics = {
            'avg_val_rmse': np.mean([s['val_rmse'] for s in cv_scores]),
            'avg_val_r2': np.mean([s['val_r2'] for s in cv_scores]),
            'avg_val_mae': np.mean([s['val_mae'] for s in cv_scores]),
            'std_val_rmse': np.std([s['val_rmse'] for s in cv_scores]),
            'std_val_r2': np.std([s['val_r2'] for s in cv_scores])
        }
        
        self.logger.info(f"\n📊 Cross-Validation Summary:")
        self.logger.info(f"   Avg Val RMSE: {avg_metrics['avg_val_rmse']:.4f} ± {avg_metrics['std_val_rmse']:.4f}")
        self.logger.info(f"   Avg Val R²: {avg_metrics['avg_val_r2']:.4f} ± {avg_metrics['std_val_r2']:.4f}")
        self.logger.info(f"   Avg Val MAE: {avg_metrics['avg_val_mae']:.4f}")
        
        # Feature importance
        self._log_feature_importance()
        
        # Store metrics
        self.training_metrics = {
            'cv_scores': cv_scores,
            'best_fold': best_idx,
            'avg_metrics': avg_metrics,
            'config': self.config
        }
        
        return self.training_metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict quality scores for SR levels.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of quality scores (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first or load() a trained model.")
        
        # Ensure feature order matches training
        try:
            X = features[self.feature_names]
        except KeyError as e:
            missing = set(self.feature_names) - set(features.columns)
            self.logger.error(f"Missing features: {missing}")
            raise ValueError(f"Missing features: {missing}")
        
        # Fill NaN values
        X = X.fillna(0.0)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Clip to [0, 1] range (quality scores)
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def predict_single(self, features_dict: Dict[str, float]) -> float:
        """Predict quality for a single SR level.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            Quality score (0-1)
        """
        features_df = pd.DataFrame([features_dict])
        predictions = self.predict(features_df)
        return float(predictions[0])
    
    def _log_feature_importance(self):
        """Log top 20 most important features."""
        if self.model is None:
            return
        
        importance = self.model.feature_importance(importance_type='gain')
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        self.logger.info("\n🏆 Top 20 Feature Importance:")
        for idx, row in feature_importance_df.head(20).iterrows():
            self.logger.info(f"   {row['feature']:<35} {row['importance']:>8.0f} ({row['importance_pct']:>5.1f}%)")
        
        # Return for analysis
        return feature_importance_df
    
    def save(self, path: str):
        """Save trained model and metadata.
        
        Args:
            path: Path to save model (e.g., 'models/sr_quality_model.lgb')
        """
        if self.model is None:
            raise ValueError("No model to save! Train first.")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'config': self.config,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = str(model_path) + '.metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"✅ Model saved to {model_path}")
        self.logger.info(f"✅ Metadata saved to {metadata_path}")
    
    def load(self, path: str):
        """Load trained model and metadata.
        
        Args:
            path: Path to model file
        """
        model_path = Path(path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = str(model_path) + '.metadata.json'
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata.get('training_metrics', {})
            self.config = metadata.get('config', self._get_default_config())
        else:
            self.logger.warning(f"⚠️ Metadata not found, using default config")
        
        self.logger.info(f"✅ Model loaded from {path}")
        self.logger.info(f"   Features: {len(self.feature_names)}")
        if 'avg_metrics' in self.training_metrics:
            avg_r2 = self.training_metrics['avg_metrics'].get('avg_val_r2', 'N/A')
            self.logger.info(f"   Avg Val R²: {avg_r2}")


# Convenience functions
def train_sr_quality_model(training_data_path: str, 
                          output_model_path: str = 'models/sr_quality_model.lgb') -> SRQualityModel:
    """Train and save SR quality model.
    
    Args:
        training_data_path: Path to training data parquet
        output_model_path: Where to save trained model
        
    Returns:
        Trained model
    """
    logger.info(f"🚀 Training SR Quality Model")
    logger.info(f"   Training data: {training_data_path}")
    logger.info(f"   Output model: {output_model_path}")
    
    # Load training data
    training_df = pd.read_parquet(training_data_path)
    logger.info(f"   Loaded {len(training_df)} training samples")
    
    # Create and train model
    model = SRQualityModel()
    metrics = model.train(training_df)
    
    # Save model
    model.save(output_model_path)
    
    logger.info(f"✅ Model training complete!")
    
    return model


def load_sr_quality_model(model_path: str = 'models/sr_quality_model.lgb') -> SRQualityModel:
    """Load trained SR quality model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model ready for predictions
    """
    model = SRQualityModel()
    model.load(model_path)
    return model

