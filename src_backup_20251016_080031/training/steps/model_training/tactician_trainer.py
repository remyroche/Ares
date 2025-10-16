"""
Tactician Trainer

This module provides training utilities for tactician models.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

# Import ML libraries with fallback support
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

class TacticianTrainer:
    """
    Trainer for tactician models with support for multiple algorithms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.training_results = {}
        
    async def train_base_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_type: str,
        signal_type: str = 'long'
    ) -> Dict[str, Any]:
        """
        Train a base model for the given signal type.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            signal_type: Type of signal ('long' or 'short')
            
        Returns:
            Dictionary with training results
        """
        try:
            tprint(f"🔍 [TACTICIAN_TRAINER] Training {model_type} model for {signal_type} signals...", color="blue")
            
            # Handle missing values
            X_clean = X.fillna(X.median())
            y_clean = y.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
            
            # Train model based on type
            if model_type == 'random_forest':
                model = self._train_random_forest(X_train, y_train)
            elif model_type == 'xgboost':
                model = self._train_xgboost(X_train, y_train)
            elif model_type == 'lightgbm':
                model = self._train_lightgbm(X_train, y_train)
            elif model_type == 'catboost':
                model = self._train_catboost(X_train, y_train)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model
            model_key = f"{model_type}_{signal_type}"
            self.models[model_key] = model
            
            tprint(f"✅ [TACTICIAN_TRAINER] {model_type} model trained with accuracy: {accuracy:.3f}", color="green")
            
            return {
                'success': True,
                'model': model,
                'accuracy': accuracy,
                'model_type': model_type,
                'signal_type': signal_type,
                'model_key': model_key
            }
            
        except Exception as e:
            tprint(f"❌ [TACTICIAN_TRAINER] Error training {model_type} model: {e}", color="red")
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type,
                'signal_type': signal_type
            }
    
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series):
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _train_catboost(self, X: pd.DataFrame, y: pd.Series):
        """Train CatBoost model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            random_state=42,
            verbose=False
        )
        model.fit(X, y)
        return model
    
    def save_models(self, output_dir: str) -> Dict[str, Any]:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary with save results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_models = {}
            for model_key, model in self.models.items():
                model_path = output_path / f"{model_key}.joblib"
                joblib.dump(model, model_path)
                saved_models[model_key] = str(model_path)
            
            tprint(f"✅ [TACTICIAN_TRAINER] Saved {len(saved_models)} models to {output_dir}", color="green")
            
            return {
                'success': True,
                'saved_models': saved_models,
                'output_dir': output_dir
            }
            
        except Exception as e:
            tprint(f"❌ [TACTICIAN_TRAINER] Error saving models: {e}", color="red")
            return {
                'success': False,
                'error': str(e)
            }
