# src/training/steps/step9_tactician_specialist_training.py

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


class TacticianSpecialistTrainingStep:
    """Step 9: Tactician Specialist Models Training (LightGBM + Calibrated Logistic Regression)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}
        
    async def initialize(self) -> None:
        """Initialize the tactician specialist training step."""
        try:
            self.logger.info("Initializing Tactician Specialist Training Step...")
            self.logger.info("Tactician Specialist Training Step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Tactician Specialist Training Step: {e}")
            raise
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tactician specialist models training.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dict containing training results
        """
        try:
            self.logger.info("🔄 Executing Tactician Specialist Training...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            
            # Load tactician labeled data
            labeled_data_dir = f"{data_dir}/tactician_labeled_data"
            labeled_file = f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            
            if not os.path.exists(labeled_file):
                raise FileNotFoundError(f"Tactician labeled data not found: {labeled_file}")
            
            # Load labeled data
            with open(labeled_file, 'rb') as f:
                labeled_data = pickle.load(f)
            
            # Convert to DataFrame if needed
            if not isinstance(labeled_data, pd.DataFrame):
                labeled_data = pd.DataFrame(labeled_data)
            
            # Train tactician specialist models
            training_results = await self._train_tactician_models(labeled_data, symbol, exchange)
            
            # Save training results
            models_dir = f"{data_dir}/tactician_models"
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name, model_data in training_results.items():
                model_file = f"{models_dir}/{model_name}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
            
            # Save training summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_tactician_training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            self.logger.info(f"✅ Tactician specialist training completed. Results saved to {models_dir}")
            
            # Update pipeline state
            pipeline_state["tactician_models"] = training_results
            
            return {
                "tactician_models": training_results,
                "models_dir": models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in Tactician Specialist Training: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "duration": 0.0
            }
    
    async def _train_tactician_models(self, data: pd.DataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Train tactician specialist models.
        
        Args:
            data: Labeled data for tactician
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Dict containing trained models
        """
        try:
            self.logger.info(f"Training tactician specialist models for {symbol} on {exchange}...")
            
            # Prepare data
            feature_columns = [col for col in data.columns if col not in ['tactician_label', 'regime']]
            X = data[feature_columns]
            y = data['tactician_label']
            
            # Split data for training and validation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train different model types
            models = {}
            
            # 1. LightGBM
            models['lightgbm'] = await self._train_lightgbm(X_train, X_test, y_train, y_test, symbol, exchange)
            
            # 2. Calibrated Logistic Regression
            models['calibrated_logistic'] = await self._train_calibrated_logistic(X_train, X_test, y_train, y_test, symbol, exchange)
            
            # 3. XGBoost (additional model)
            models['xgboost'] = await self._train_xgboost(X_train, X_test, y_train, y_test, symbol, exchange)
            
            # 4. Random Forest (additional model)
            models['random_forest'] = await self._train_random_forest(X_train, X_test, y_train, y_test, symbol, exchange)
            
            self.logger.info(f"Trained {len(models)} tactician models")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error training tactician models: {e}")
            raise
    
    async def _train_lightgbm(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> Dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model with enhanced regularization
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1,  # L2 regularization
                random_state=42,
                verbose=-1,
                early_stopping_rounds=50
            )
            
            # Train with validation set
            eval_set = [(X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, eval_metric='logloss')
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "LightGBM",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM: {e}")
            raise
    
    async def _train_calibrated_logistic(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                        y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> Dict[str, Any]:
        """Train Calibrated Logistic Regression model."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import classification_report, accuracy_score
            
            # Base logistic regression
            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
            
            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_model,
                cv=5,
                method='isotonic'
            )
            
            # Train model
            calibrated_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = calibrated_model.predict(X_test)
            y_pred_proba = calibrated_model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                "model": calibrated_model,
                "accuracy": accuracy,
                "feature_importance": {},  # Logistic regression doesn't have direct feature importance
                "model_type": "CalibratedLogisticRegression",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "C": 1.0,
                    "max_iter": 1000,
                    "calibration_method": "isotonic",
                    "cv_folds": 5
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training Calibrated Logistic Regression: {e}")
            raise
    
    async def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> Dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model with enhanced regularization
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1,  # L2 regularization
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=50
            )
            
            # Train with validation set
            eval_set = [(X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")
            raise
    
    async def _train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                 y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> Dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, accuracy_score
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "RandomForest",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs
) -> bool:
    """
    Run the tactician specialist training step.
    
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
        step = TacticianSpecialistTrainingStep(config)
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
        print(f"❌ Tactician specialist training failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")
    
    asyncio.run(test())
