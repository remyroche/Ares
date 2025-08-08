# src/training/model_trainer.py

import pickle
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import ray
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
from src.utils.mlflow_utils import log_training_metadata_to_mlflow

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.training.feature_engineering import FeatureGenerator
from src.training.data_cleaning import handle_missing_data
from src.training.steps.step12_final_parameters_optimization.optimized_optuna_optimization import AdvancedOptunaManager
import shap
import matplotlib.pyplot as plt
import tempfile


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str
    timeframe: str
    features: List[str]
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10


@dataclass
class TrainingData:
    """Container for training data."""
    features: pd.DataFrame
    labels: pd.Series
    timeframe: str
    model_type: str
    data_info: Dict[str, Any]


class RayModelTrainer:
    """
    Ray-based model trainer for distributed model training and data processing.
    Handles both analyst and tactician models with parallel processing capabilities.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Ray model trainer.

        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("RayModelTrainer")
        
        # Ray configuration
        self.ray_config: Dict[str, Any] = self.config.get("ray", {})
        self.num_cpus: int = self.ray_config.get("num_cpus", 4)
        self.num_gpus: int = self.ray_config.get("num_gpus", 0)
        
        # Model trainer state
        self.is_training: bool = False
        self.trained_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        
        # Configuration
        self.model_trainer_config: Dict[str, Any] = self.config.get("model_trainer", {})
        self.enable_analyst_models: bool = self.model_trainer_config.get("enable_analyst_models", True)
        self.enable_tactician_models: bool = self.model_trainer_config.get("enable_tactician_models", True)
        
        # Model configurations
        self.analyst_models_config: Dict[str, Any] = self.model_trainer_config.get("analyst_models", {})
        self.tactician_models_config: Dict[str, Any] = self.model_trainer_config.get("tactician_models", {})
        
        # Initialize Ray
        self._initialize_ray()

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid Ray configuration"),
            RuntimeError: (False, "Ray initialization failed"),
        },
        default_return=False,
        context="Ray initialization",
    )
    def _initialize_ray(self) -> bool:
        """
        Initialize Ray cluster.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not ray.is_initialized():
                ray.init(
                    num_cpus=self.num_cpus,
                    num_gpus=self.num_gpus,
                    ignore_reinit_error=True,
                    logging_level=self.ray_config.get("logging_level", "info")
                )
                self.logger.info(f"✅ Ray initialized with {self.num_cpus} CPUs, {self.num_gpus} GPUs")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ray initialization failed: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid model trainer configuration"),
            AttributeError: (False, "Missing required model trainer parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="model trainer initialization",
    )
    def initialize(self) -> bool:
        """
        Initialize model trainer.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Ray Model Trainer...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for model trainer")
                return False
            
            # Initialize model storage
            self._initialize_model_storage()
            
            self.logger.info("✅ Ray Model Trainer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ray Model Trainer initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate model trainer configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate model trainer specific settings
            if not self.enable_analyst_models and not self.enable_tactician_models:
                self.logger.error("At least one model type must be enabled")
                return False
            
            # Validate analyst models configuration
            if self.enable_analyst_models:
                if not self.analyst_models_config:
                    self.logger.error("Analyst models enabled but no configuration provided")
                    return False
            
            # Validate tactician models configuration
            if self.enable_tactician_models:
                if not self.tactician_models_config:
                    self.logger.error("Tactician models enabled but no configuration provided")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model storage initialization",
    )
    def _initialize_model_storage(self) -> None:
        """Initialize model storage and metadata."""
        try:
            # Create model storage directory if it doesn't exist
            model_dir = self.model_trainer_config.get("model_directory", "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Load existing model metadata
            metadata_file = os.path.join(model_dir, "model_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    self.model_metadata = json.load(f)
            
            self.logger.info(f"✅ Model storage initialized: {model_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model storage: {e}")
            raise

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training parameters"),
            AttributeError: (False, "Missing training components"),
            KeyError: (False, "Missing required training data"),
        },
        default_return=False,
        context="model training",
    )
    def train_models(
        self,
        training_input: Dict[str, Any],
        use_hpo: bool = True,
        hpo_trials: int = 50,
        hpo_model_type: str = "random_forest",
    ) -> Optional[Dict[str, Any]]:
        """
        Train all required models based on configuration using Ray.
        If use_hpo is True, run Optuna HPO before final model training.
        Logs all training runs to MLflow.
        """
        try:
            self.logger.info("🚀 Starting Ray-based model training...")
            self.is_training = True
            if not self._validate_training_input(training_input):
                return None
            training_data = self._prepare_training_data(training_input)
            if training_data is None:
                return None
            best_params = None
            hpo_result = None
            with mlflow.start_run() as run:
                # Log training metadata
                log_training_metadata_to_mlflow(
                    symbol=training_input.get("symbol", "ETHUSDT"),
                    timeframe="1m",
                    model_type=hpo_model_type,
                    run_id=run.info.run_id
                )
                if use_hpo:
                    self.logger.info("🔍 Running Optuna HPO before model training...")
                    tactician_data = training_data.get("tactician_1m")
                    if tactician_data is None:
                        self.logger.error("No tactician_1m data for HPO.")
                        return None
                    X = tactician_data.features
                    y = tactician_data.labels
                    hpo_manager = AdvancedOptunaManager()
                    hpo_result = hpo_manager.optimize(
                        model_type=hpo_model_type,
                        X=X,
                        y=y,
                        n_trials=hpo_trials,
                        cv_folds=5,
                        early_stopping_patience=10
                    )
                    best_params = hpo_result.get("best_params")
                    mlflow.log_params(best_params)
                    self.logger.info(f"Optuna HPO best params: {best_params}")
                training_results = self._train_models_with_ray(training_data, training_input, best_params=best_params)
                self._store_trained_models(training_results)
                # Log model metrics and artifacts
                tactician_models = training_results.get("tactician_models", {})
                for tf, result in tactician_models.items():
                    if result["training_status"] == "completed":
                        mlflow.log_metrics(result["model_metrics"], step=0)
                        if "model_path" in result:
                            mlflow.log_artifact(result["model_path"])
                        if "scaler_path" in result:
                            mlflow.log_artifact(result["scaler_path"])
                        # SHAP explainability integration
                        try:
                            model = joblib.load(result["model_path"])
                            scaler = joblib.load(result["scaler_path"])
                            X_sample = training_data["tactician_1m"].features.iloc[:200]
                            X_sample_scaled = scaler.transform(X_sample)
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_sample_scaled)
                            plt.figure()
                            shap.summary_plot(shap_values, X_sample, show=False)
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                plt.savefig(tmpfile.name)
                                mlflow.log_artifact(tmpfile.name, artifact_path="shap")
                            plt.close()
                        except Exception as e:
                            self.logger.warning(f"SHAP explainability failed: {e}")
                self.is_training = False
                self.logger.info("✅ Ray-based model training completed successfully")
                return training_results
        except Exception as e:
            self.logger.error(f"❌ Ray-based model training failed: {e}")
            self.is_training = False
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="training input validation",
    )
    def _validate_training_input(self, training_input: Dict[str, Any]) -> bool:
        """
        Validate training input parameters.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "lookback_days"]
            
            for field in required_fields:
                if field not in training_input:
                    self.logger.error(f"Missing required training input field: {field}")
                    return False
            
            # Validate specific field values
            if training_input.get("lookback_days", 0) <= 0:
                self.logger.error("Invalid lookback_days value")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Training input validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training data preparation",
    )
    def _prepare_training_data(
        self,
        training_input: Dict[str, Any],
    ) -> Optional[Dict[str, TrainingData]]:
        """
        Prepare training data for model training.
        Loads the labeled/enhanced feature file produced by the previous pipeline step (step 4),
        not the raw data from step 1.
        """
        try:
            self.logger.info("📊 Preparing training data from labeled/enhanced pipeline output...")
            prepared_data = {}
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            labeled_path = f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet"
            import pandas as pd
            import os
            if os.path.exists(labeled_path):
                data = pd.read_parquet(labeled_path)
                self.logger.info(f"Loaded labeled data from {labeled_path}")
            else:
                # Fallback to CSV if Parquet is not available
                labeled_csv = labeled_path.replace('.parquet', '.csv')
                if os.path.exists(labeled_csv):
                    data = pd.read_csv(labeled_csv, parse_dates=["timestamp"])
                    self.logger.info(f"Loaded labeled data from {labeled_csv}")
                else:
                    self.logger.error(f"Labeled/enhanced data file not found: {labeled_path} or {labeled_csv}")
                    return None
            data = handle_missing_data(data)
            feature_generator = FeatureGenerator()
            # Use all columns except label as features, and 'label' as target
            feature_cols = [col for col in data.columns if col not in ("label", "tactician_label", "target")]
            label_col = "label" if "label" in data.columns else ("tactician_label" if "tactician_label" in data.columns else "target")
            features = data[feature_cols]
            labels = data[label_col]
            prepared_data["tactician_1m"] = TrainingData(
                features=features,
                labels=labels,
                timeframe="1m",
                model_type="tactician",
                data_info={
                    "rows": len(data),
                    "columns": len(features.columns),
                    "timeframe": "1m",
                }
            )
            self.logger.info("✅ Training data prepared successfully from labeled/enhanced pipeline output")
            return prepared_data
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare training data: {e}")
            return None

    def _train_models_with_ray(
        self,
        training_data: Dict[str, TrainingData],
        training_input: Dict[str, Any],
        best_params: dict = None,
    ) -> Dict[str, Any]:
        """
        Train models using Ray for distributed processing.
        Accepts best_params from HPO for model instantiation.
        """
        try:
            self.logger.info("🧠 Starting Ray-based model training...")
            @ray.remote
            def train_single_model(model_config: ModelConfig, training_data: TrainingData, best_params: dict = None) -> Dict[str, Any]:
                return self._train_single_model_remote(model_config, training_data, best_params=best_params)
            model_configs = []
            if self.enable_tactician_models:
                data_key = "tactician_1m"
                if data_key in training_data:
                    config = ModelConfig(
                        model_type="tactician",
                        timeframe="1m",
                        features=list(training_data[data_key].features.columns),
                        target_column="target"
                    )
                    model_configs.append((config, training_data[data_key]))
            training_futures = []
            for config, data in model_configs:
                future = train_single_model.remote(config, data, best_params)
                training_futures.append(future)
            training_results = ray.get(training_futures)
            tactician_results = {}
            for result in training_results:
                tactician_results[result["timeframe"]] = result
            return {
                "tactician_models": tactician_results,
                "training_input": training_input,
                "training_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"❌ Ray-based model training failed: {e}")
            return {}

    def _train_single_model_remote(
        self,
        model_config: ModelConfig,
        training_data: TrainingData,
        best_params: dict = None,
    ) -> Dict[str, Any]:
        """
        Train a single model (Ray remote function).
        Accepts best_params from HPO for model instantiation.
        """
        try:
            X = training_data.features
            y = training_data.labels
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=model_config.test_size, random_state=model_config.random_state
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # Use best_params if provided
            if best_params:
                model = RandomForestClassifier(**best_params)
            else:
                model = RandomForestClassifier(
                    n_estimators=model_config.n_estimators,
                    max_depth=model_config.max_depth,
                    random_state=model_config.random_state
                )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            result = {
                "timeframe": model_config.timeframe,
                "model_type": model_config.model_type,
                "training_status": "completed",
                "model_metrics": metrics,
                "feature_importance": feature_importance,
                "model_path": f"models/{model_config.model_type}_{model_config.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "scaler_path": f"models/{model_config.model_type}_{model_config.timeframe}_scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            }
            self._store_model_remote(result, model, scaler)
            return result
        except Exception as e:
            self.logger.error(f"❌ Failed to train {model_config.model_type} model for {model_config.timeframe}: {e}")
            return {
                "timeframe": model_config.timeframe,
                "model_type": model_config.model_type,
                "training_status": "failed",
                "error": str(e),
            }

    def _store_model_remote(
        self,
        result: Dict[str, Any],
        model: Any,
        scaler: StandardScaler,
    ) -> None:
        """
        Store model and scaler (Ray remote function).

        Args:
            result: Model result
            model: Trained model
            scaler: Fitted scaler
        """
        try:
            # Create model directory
            model_dir = self.model_trainer_config.get("model_directory", "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, os.path.basename(result["model_path"]))
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, os.path.basename(result["scaler_path"]))
            joblib.dump(scaler, scaler_path)
            
            # Update result paths
            result["model_path"] = model_path
            result["scaler_path"] = scaler_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store model: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trained models storage",
    )
    def _store_trained_models(self, training_results: Dict[str, Any]) -> None:
        """
        Store all trained models metadata.

        Args:
            training_results: Complete training results
        """
        try:
            self.logger.info("📁 Storing trained models metadata...")
            
            # Store analyst models metadata
            if training_results.get("analyst_models"):
                for timeframe, model_result in training_results["analyst_models"].items():
                    if model_result["training_status"] == "completed":
                        self._store_model_metadata(model_result)
            
            # Store tactician models metadata
            if training_results.get("tactician_models"):
                for timeframe, model_result in training_results["tactician_models"].items():
                    if model_result["training_status"] == "completed":
                        self._store_model_metadata(model_result)
            
            # Save metadata file
            model_dir = self.model_trainer_config.get("model_directory", "models")
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.model_metadata, f, indent=2)
            
            self.logger.info("✅ All trained models metadata stored successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store trained models metadata: {e}")

    def _store_model_metadata(self, model_result: Dict[str, Any]) -> None:
        """
        Store model metadata.

        Args:
            model_result: Model training result
        """
        try:
            model_key = f"{model_result['model_type']}_{model_result['timeframe']}"
            self.model_metadata[model_key] = {
                "path": model_result["model_path"],
                "scaler_path": model_result.get("scaler_path"),
                "training_timestamp": datetime.now().isoformat(),
                "metrics": model_result["model_metrics"],
                "feature_importance": model_result.get("feature_importance", {}),
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store model metadata: {e}")

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.

        Returns:
            dict: Training status information
        """
        return {
            "is_training": self.is_training,
            "trained_models_count": len(self.trained_models),
            "analyst_models_enabled": self.enable_analyst_models,
            "tactician_models_enabled": self.enable_tactician_models,
            "ray_cluster_info": {
                "num_cpus": self.num_cpus,
                "num_gpus": self.num_gpus,
                "is_initialized": ray.is_initialized(),
            }
        }

    def get_trained_models(self) -> Dict[str, Any]:
        """
        Get all trained models.

        Returns:
            dict: Trained models information
        """
        return self.trained_models.copy()

    def load_model(self, model_type: str, timeframe: str) -> Optional[Tuple[Any, StandardScaler]]:
        """
        Load a trained model and its scaler.

        Args:
            model_type: Type of model (analyst/tactician)
            timeframe: Model timeframe

        Returns:
            tuple: (model, scaler) or None if not found
        """
        try:
            model_key = f"{model_type}_{timeframe}"
            if model_key in self.model_metadata:
                metadata = self.model_metadata[model_key]
                
                # Load model
                model = joblib.load(metadata["path"])
                
                # Load scaler
                scaler = None
                if "scaler_path" in metadata:
                    scaler = joblib.load(metadata["scaler_path"])
                
                return model, scaler
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model {model_type}_{timeframe}: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model trainer cleanup",
    )
    def stop(self) -> None:
        """Stop the model trainer and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Ray Model Trainer...")
            self.is_training = False
            
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
                self.logger.info("✅ Ray cluster shutdown")
            
            self.logger.info("✅ Ray Model Trainer stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to stop Ray Model Trainer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="model trainer setup",
)
def setup_model_trainer(
    config: Optional[Dict[str, Any]] = None,
) -> Optional[RayModelTrainer]:
    """
    Setup and return a configured RayModelTrainer instance.

    Args:
        config: Configuration dictionary

    Returns:
        RayModelTrainer: Configured model trainer instance
    """
    try:
        trainer = RayModelTrainer(config or {})
        if trainer.initialize():
            return trainer
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup Ray model trainer: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        "ray": {
            "num_cpus": 4,
            "num_gpus": 0,
            "logging_level": "info"
        },
        "model_trainer": {
            "enable_analyst_models": True,
            "enable_tactician_models": True,
            "model_directory": "models",
            "analyst_models": {
                "timeframes": ["1h", "15m", "5m", "1m"]
            },
            "tactician_models": {
                "timeframes": ["1m"]
            }
        }
    }
    
    # Setup trainer
    trainer = setup_model_trainer(config)
    
    if trainer:
        # Example training input
        training_input = {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "timeframe": "1m",
            "lookback_days": 30,
            "data_dir": "data/training" # Added data_dir for the new _prepare_training_data
        }
        
        # Train models
        results = trainer.train_models(training_input)
        
        if results:
            print("✅ Training completed successfully!")
            print(f"Analyst models: {len(results.get('analyst_models', {}))}")
            print(f"Tactician models: {len(results.get('tactician_models', {}))}")
        else:
            print("❌ Training failed!")
        
        # Cleanup
        trainer.stop()
    else:
        print("❌ Failed to setup model trainer!") 
