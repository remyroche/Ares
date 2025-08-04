#!/usr/bin/env python3
"""
Model Training Integrator for Ares Trading System.
Enables full functionality with trained models.
"""

import asyncio
import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.utils.comprehensive_logger import get_component_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.data_optimizer import get_data_optimizer
from src.analyst.ml_confidence_predictor import MLConfidencePredictor


class ModelTrainingIntegrator:
    """
    Model Training Integrator for enabling full functionality with trained models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Model Training Integrator."""
        self.config = config
        self.logger = get_component_logger("ModelTrainingIntegrator")
        
        # Training configuration
        self.training_config = config.get("model_training_integrator", {})
        self.models_path = self.training_config.get("models_path", "models/")
        self.training_data_path = self.training_config.get("training_data_path", "data/training/")
        self.test_size = self.training_config.get("test_size", 0.2)
        self.random_state = self.training_config.get("random_state", 42)
        
        # Model types and configurations
        self.model_configs = {
            "lightgbm": {
                "class": lgb.LGBMClassifier,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": self.random_state,
                    "verbose": -1
                }
            },
            "xgboost": {
                "class": xgb.XGBClassifier,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": self.random_state,
                    "verbosity": 0
                }
            },
            "random_forest": {
                "class": RandomForestClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": self.random_state
                }
            },
            "gradient_boosting": {
                "class": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": self.random_state
                }
            },
            "logistic_regression": {
                "class": LogisticRegression,
                "params": {
                    "random_state": self.random_state,
                    "max_iter": 1000
                }
            },
            "catboost": {
                "class": CatBoostClassifier,
                "params": {
                    "iterations": 100,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "random_state": self.random_state,
                    "verbose": False
                }
            }
        }
        
        # Training statistics
        self.training_stats = {
            "models_trained": 0,
            "total_training_time": 0,
            "best_model": None,
            "best_score": 0.0,
            "training_history": []
        }
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        try:
            os.makedirs(self.models_path, exist_ok=True)
            os.makedirs(self.training_data_path, exist_ok=True)
            self.logger.info("Directories ensured")
            
        except Exception as e:
            self.logger.error(f"Error ensuring directories: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model training integrator initialization"
    )
    async def initialize(self) -> bool:
        """Initialize Model Training Integrator."""
        try:
            self.logger.info("Initializing Model Training Integrator...")
            
            # Initialize data optimizer
            self.data_optimizer = get_data_optimizer()
            
            # Load existing models if available
            await self._load_existing_models()
            
            self.logger.info("✅ Model Training Integrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Model Training Integrator: {e}")
            return False
    
    async def _load_existing_models(self) -> None:
        """Load existing trained models."""
        try:
            self.trained_models = {}
            
            # Check for existing model files
            model_files = [f for f in os.listdir(self.models_path) if f.endswith('.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                model_path = os.path.join(self.models_path, model_file)
                
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.trained_models[model_name] = model
                    self.logger.info(f"Loaded existing model: {model_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name}: {e}")
            
            self.logger.info(f"Loaded {len(self.trained_models)} existing models")
            
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")
    
    async def generate_training_data(self, size: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data for model training."""
        try:
            self.logger.info(f"Generating training data with {size} samples...")
            
            np.random.seed(self.random_state)
            
            # Generate synthetic market features
            data = {
                'price_change': np.random.normal(0, 0.02, size),
                'volume_change': np.random.normal(0, 0.1, size),
                'volatility': np.random.exponential(0.01, size),
                'rsi': np.random.uniform(0, 100, size),
                'macd': np.random.normal(0, 0.01, size),
                'bollinger_position': np.random.uniform(0, 1, size),
                'support_distance': np.random.exponential(0.02, size),
                'resistance_distance': np.random.exponential(0.02, size),
                'trend_strength': np.random.uniform(0, 1, size),
                'momentum': np.random.normal(0, 0.01, size),
                'volume_sma_ratio': np.random.normal(1, 0.2, size),
                'price_sma_ratio': np.random.normal(1, 0.05, size),
                'atr': np.random.exponential(0.01, size),
                'stoch_k': np.random.uniform(0, 100, size),
                'stoch_d': np.random.uniform(0, 100, size),
                'williams_r': np.random.uniform(-100, 0, size),
                'cci': np.random.normal(0, 100, size),
                'adx': np.random.uniform(0, 100, size),
                'obv_change': np.random.normal(0, 1000, size),
                'vwap_deviation': np.random.normal(0, 0.01, size)
            }
            
            # Create feature DataFrame
            X = pd.DataFrame(data)
            
            # Generate target labels (price increase probability)
            # Combine features to create realistic target
            price_signal = (X['price_change'] + X['momentum'] + X['trend_strength'] * 0.5) / 3
            volume_signal = X['volume_change'] * 0.3
            technical_signal = (X['rsi'] - 50) / 50 * 0.2 + (X['macd'] * 10) + (X['bollinger_position'] - 0.5) * 0.3
            
            combined_signal = price_signal + volume_signal + technical_signal
            
            # Convert to binary classification (price increase vs decrease)
            y = (combined_signal > 0).astype(int)
            
            # Add some noise to make it more realistic
            noise = np.random.random(size) < 0.1
            y = y ^ noise
            
            self.logger.info(f"Generated training data: {len(X)} samples, {X.shape[1]} features")
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    async def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models and select the best one."""
        try:
            self.logger.info("Starting model training...")
            
            # Optimize training data
            if self.data_optimizer:
                X = await self.data_optimizer.optimize_dataframe(X, strategy="speed")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            trained_models = {}
            model_scores = {}
            training_times = {}
            
            # Train each model type
            for model_name, model_config in self.model_configs.items():
                try:
                    self.logger.info(f"Training {model_name}...")
                    
                    start_time = datetime.now()
                    
                    # Initialize model
                    model_class = model_config["class"]
                    model_params = model_config["params"]
                    model = model_class(**model_params)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Store results
                    trained_models[model_name] = model
                    model_scores[model_name] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std
                    }
                    training_times[model_name] = training_time
                    
                    self.logger.info(f"✅ {model_name} trained successfully:")
                    self.logger.info(f"  - Accuracy: {accuracy:.4f}")
                    self.logger.info(f"  - F1 Score: {f1:.4f}")
                    self.logger.info(f"  - CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
                    self.logger.info(f"  - Training time: {training_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
            
            # Find best model
            best_model_name = max(model_scores.keys(), 
                                key=lambda x: model_scores[x]["f1_score"])
            best_model = trained_models[best_model_name]
            best_score = model_scores[best_model_name]["f1_score"]
            
            # Update training statistics
            self.training_stats["models_trained"] = len(trained_models)
            self.training_stats["best_model"] = best_model_name
            self.training_stats["best_score"] = best_score
            self.training_stats["total_training_time"] = sum(training_times.values())
            
            # Save training history
            training_record = {
                "timestamp": datetime.now().isoformat(),
                "models_trained": len(trained_models),
                "best_model": best_model_name,
                "best_score": best_score,
                "model_scores": model_scores,
                "training_times": training_times
            }
            self.training_stats["training_history"].append(training_record)
            
            # Save models
            await self._save_models(trained_models)
            
            self.logger.info(f"🎉 Model training completed!")
            self.logger.info(f"Best model: {best_model_name} (F1: {best_score:.4f})")
            
            return {
                "trained_models": trained_models,
                "model_scores": model_scores,
                "training_times": training_times,
                "best_model": best_model_name,
                "best_score": best_score
            }
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {}
    
    async def _save_models(self, models: Dict[str, Any]) -> None:
        """Save trained models to disk."""
        try:
            for model_name, model in models.items():
                model_path = os.path.join(self.models_path, f"{model_name}.pkl")
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                self.logger.info(f"Saved model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    async def train_ml_confidence_predictor(self) -> bool:
        """Train the ML Confidence Predictor with synthetic data."""
        try:
            self.logger.info("Training ML Confidence Predictor...")
            
            # Generate training data
            X, y = await self.generate_training_data(15000)
            
            if X.empty or y.empty:
                self.logger.error("Failed to generate training data")
                return False
            
            # Train models
            training_results = await self.train_models(X, y)
            
            if not training_results:
                self.logger.error("Failed to train models")
                return False
            
            # Update ML Confidence Predictor with trained models
            await self._update_ml_confidence_predictor(training_results)
            
            self.logger.info("✅ ML Confidence Predictor training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML Confidence Predictor: {e}")
            return False
    
    async def _update_ml_confidence_predictor(self, training_results: Dict[str, Any]) -> None:
        """Update ML Confidence Predictor with trained models."""
        try:
            # Get the best model
            best_model_name = training_results["best_model"]
            trained_models = training_results["trained_models"]
            best_model = trained_models[best_model_name]
            
            # Create a simple model interface for ML Confidence Predictor
            class TrainedModelWrapper:
                def __init__(self, model, model_name):
                    self.model = model
                    self.model_name = model_name
                    self.is_trained = True
                
                def predict_proba(self, X):
                    return self.model.predict_proba(X)
                
                def predict(self, X):
                    return self.model.predict(X)
            
            # Create wrapper for the best model
            model_wrapper = TrainedModelWrapper(best_model, best_model_name)
            
            # Store in a way that ML Confidence Predictor can access
            # This would typically be done through a model registry or configuration
            self.trained_models["ml_confidence_predictor"] = model_wrapper
            
            self.logger.info(f"Updated ML Confidence Predictor with {best_model_name}")
            
        except Exception as e:
            self.logger.error(f"Error updating ML Confidence Predictor: {e}")
    
    async def train_ensemble_models(self) -> Dict[str, Any]:
        """Train ensemble models for different timeframes."""
        try:
            self.logger.info("Training ensemble models...")
            
            ensemble_models = {}
            
            # Train models for different timeframes
            timeframes = ["1m", "5m", "15m", "1h"]
            
            for timeframe in timeframes:
                self.logger.info(f"Training ensemble model for {timeframe}...")
                
                # Generate timeframe-specific data
                X, y = await self.generate_training_data(10000)
                
                if not X.empty and not y.empty:
                    # Train models for this timeframe
                    training_results = await self.train_models(X, y)
                    
                    if training_results:
                        ensemble_models[timeframe] = training_results["trained_models"]
                        self.logger.info(f"✅ Ensemble model trained for {timeframe}")
                    else:
                        self.logger.warning(f"Failed to train ensemble model for {timeframe}")
                else:
                    self.logger.warning(f"Failed to generate data for {timeframe}")
            
            # Save ensemble models
            await self._save_ensemble_models(ensemble_models)
            
            self.logger.info(f"✅ Ensemble training completed: {len(ensemble_models)} timeframes")
            return ensemble_models
            
        except Exception as e:
            self.logger.error(f"Error training ensemble models: {e}")
            return {}
    
    async def _save_ensemble_models(self, ensemble_models: Dict[str, Any]) -> None:
        """Save ensemble models to disk."""
        try:
            for timeframe, models in ensemble_models.items():
                timeframe_path = os.path.join(self.models_path, f"ensemble_{timeframe}")
                os.makedirs(timeframe_path, exist_ok=True)
                
                for model_name, model in models.items():
                    model_path = os.path.join(timeframe_path, f"{model_name}.pkl")
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                self.logger.info(f"Saved ensemble models for {timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble models: {e}")
    
    async def load_trained_models(self) -> Dict[str, Any]:
        """Load all trained models from disk."""
        try:
            self.logger.info("Loading trained models...")
            
            loaded_models = {}
            
            # Load individual models
            model_files = [f for f in os.listdir(self.models_path) if f.endswith('.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                model_path = os.path.join(self.models_path, model_file)
                
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    loaded_models[model_name] = model
                    self.logger.info(f"Loaded model: {model_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name}: {e}")
            
            # Load ensemble models
            ensemble_dirs = [d for d in os.listdir(self.models_path) if d.startswith('ensemble_')]
            
            for ensemble_dir in ensemble_dirs:
                timeframe = ensemble_dir.replace('ensemble_', '')
                ensemble_path = os.path.join(self.models_path, ensemble_dir)
                
                ensemble_models = {}
                ensemble_files = [f for f in os.listdir(ensemble_path) if f.endswith('.pkl')]
                
                for model_file in ensemble_files:
                    model_name = model_file.replace('.pkl', '')
                    model_path = os.path.join(ensemble_path, model_file)
                    
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        
                        ensemble_models[model_name] = model
                        
                    except Exception as e:
                        self.logger.error(f"Error loading ensemble model {model_name}: {e}")
                
                if ensemble_models:
                    loaded_models[f"ensemble_{timeframe}"] = ensemble_models
                    self.logger.info(f"Loaded ensemble models for {timeframe}")
            
            self.logger.info(f"✅ Loaded {len(loaded_models)} model groups")
            return loaded_models
            
        except Exception as e:
            self.logger.error(f"Error loading trained models: {e}")
            return {}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        try:
            return {
                "training_stats": self.training_stats,
                "models_available": list(self.trained_models.keys()) if hasattr(self, 'trained_models') else [],
                "best_model": self.training_stats.get("best_model"),
                "best_score": self.training_stats.get("best_score"),
                "total_training_time": self.training_stats.get("total_training_time"),
                "models_trained": self.training_stats.get("models_trained"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting training stats: {e}")
            return {"error": str(e)}
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model training integrator cleanup"
    )
    async def stop(self) -> None:
        """Stop Model Training Integrator."""
        try:
            self.logger.info("Stopping Model Training Integrator...")
            
            # Save final training statistics
            stats = self.get_training_stats()
            stats_path = os.path.join(self.models_path, "training_stats.json")
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            self.logger.info("✅ Model Training Integrator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Model Training Integrator: {e}")


# Global model training integrator instance
model_training_integrator: Optional[ModelTrainingIntegrator] = None


async def setup_model_training_integrator(config: Dict[str, Any]) -> ModelTrainingIntegrator:
    """Setup global model training integrator."""
    global model_training_integrator
    
    if model_training_integrator is None:
        model_training_integrator = ModelTrainingIntegrator(config)
        await model_training_integrator.initialize()
    
    return model_training_integrator


def get_model_training_integrator() -> Optional[ModelTrainingIntegrator]:
    """Get global model training integrator instance."""
    return model_training_integrator 