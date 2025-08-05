# src/training/steps/step5_analyst_specialist_training.py

import asyncio
import json
import os
import pandas as pd
import pickle
import numpy as np
from typing import Any, Dict, Optional, List
from datetime import datetime

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AnalystSpecialistTrainingStep:
    """Step 5: Analyst Specialist Models Training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}
        
    async def initialize(self) -> None:
        """Initialize the analyst specialist training step."""
        try:
            self.logger.info("Initializing Analyst Specialist Training Step...")
            self.logger.info("Analyst Specialist Training Step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Analyst Specialist Training Step: {e}")
            raise
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyst specialist models training.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dict containing training results
        """
        try:
            self.logger.info("🔄 Executing Analyst Specialist Training...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            
            # Load labeled data
            labeled_data_dir = f"{data_dir}/labeled_data"
            labeled_data = {}
            
            # Load all labeled data files
            for file in os.listdir(labeled_data_dir):
                if file.startswith(f"{exchange}_{symbol}_") and file.endswith("_labeled.pkl"):
                    regime_name = file.replace(f"{exchange}_{symbol}_", "").replace("_labeled.pkl", "")
                    labeled_file = os.path.join(labeled_data_dir, file)
                    
                    with open(labeled_file, 'rb') as f:
                        labeled_data[regime_name] = pickle.load(f)
            
            if not labeled_data:
                raise ValueError(f"No labeled data found in {labeled_data_dir}")
            
            # Train specialist models for each regime
            training_results = {}
            
            for regime_name, regime_data in labeled_data.items():
                self.logger.info(f"Training specialist models for regime: {regime_name}")
                
                # Train models for this regime
                regime_models = await self._train_regime_models(regime_data, regime_name)
                training_results[regime_name] = regime_models
            
            # Save training results
            models_dir = f"{data_dir}/analyst_models"
            os.makedirs(models_dir, exist_ok=True)
            
            for regime_name, models in training_results.items():
                regime_models_dir = f"{models_dir}/{regime_name}"
                os.makedirs(regime_models_dir, exist_ok=True)
                
                for model_name, model_data in models.items():
                    model_file = f"{regime_models_dir}/{model_name}.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_data, f)
            
            # Save training summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_analyst_training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            self.logger.info(f"✅ Analyst specialist training completed. Results saved to {models_dir}")
            
            # Update pipeline state
            pipeline_state["analyst_models"] = training_results
            
            return {
                "analyst_models": training_results,
                "models_dir": models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in Analyst Specialist Training: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "duration": 0.0
            }
    
    async def _train_regime_models(self, data: pd.DataFrame, regime_name: str) -> Dict[str, Any]:
        """
        Train specialist models for a specific regime.
        
        Args:
            data: Labeled data for the regime
            regime_name: Name of the regime
            
        Returns:
            Dict containing trained models
        """
        try:
            self.logger.info(f"Training specialist models for regime: {regime_name}")
            
            # Prepare data
            feature_columns = [col for col in data.columns if col not in ['label', 'regime']]
            X = data[feature_columns]
            y = data['label']
            
            # Split data for training and validation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train different model types
            models = {}
            
            # 1. Random Forest
            models['random_forest'] = await self._train_random_forest(X_train, X_test, y_train, y_test, regime_name)
            
            # 2. LightGBM (as a substitute for some advanced models)
            models['lightgbm'] = await self._train_lightgbm(X_train, X_test, y_train, y_test, regime_name)
            
            # 3. XGBoost (as a substitute for some advanced models)
            models['xgboost'] = await self._train_xgboost(X_train, X_test, y_train, y_test, regime_name)
            
            # 4. Neural Network (as a substitute for TCN/Transformer)
            models['neural_network'] = await self._train_neural_network(X_train, X_test, y_train, y_test, regime_name)
            
            # 5. Support Vector Machine
            models['svm'] = await self._train_svm(X_train, X_test, y_train, y_test, regime_name)
            
            self.logger.info(f"Trained {len(models)} models for regime {regime_name}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error training models for regime {regime_name}: {e}")
            raise
    
    async def _train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                 y_train: pd.Series, y_test: pd.Series, regime_name: str) -> Dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "RandomForest",
                "regime": regime_name,
                "training_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest for {regime_name}: {e}")
            raise
    
    async def _train_lightgbm(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series, regime_name: str) -> Dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "LightGBM",
                "regime": regime_name,
                "training_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM for {regime_name}: {e}")
            raise
    
    async def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            y_train: pd.Series, y_test: pd.Series, regime_name: str) -> Dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "regime": regime_name,
                "training_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost for {regime_name}: {e}")
            raise
    
    async def _train_neural_network(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  y_train: pd.Series, y_test: pd.Series, regime_name: str) -> Dict[str, Any]:
        """Train Neural Network model."""
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": {},  # Neural networks don't have direct feature importance
                "model_type": "NeuralNetwork",
                "regime": regime_name,
                "training_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training Neural Network for {regime_name}: {e}")
            raise
    
    async def _train_svm(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series, regime_name: str) -> Dict[str, Any]:
        """Train Support Vector Machine model."""
        try:
            from sklearn.svm import SVC
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model
            model = SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": {},  # SVMs don't have direct feature importance
                "model_type": "SVM",
                "regime": regime_name,
                "training_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training SVM for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs
) -> bool:
    """
    Run the analyst specialist training step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = AnalystSpecialistTrainingStep(config)
        await step.initialize()
        
        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        return result.get("status") == "SUCCESS"
        
    except Exception as e:
        print(f"❌ Analyst specialist training failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")
    
    asyncio.run(test())
