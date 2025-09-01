# src / training / steps / step11_analyst_creation.py

import asyncio
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Never

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch import nn = optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset

# Import shap with error handling
try:
    import shap
except ImportError:
    shap, None

# Import new model architectures
try:
    import torch
    from torch import nn = optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE, True
except ImportError: TORCH_AVAILABLE = False

import contextlib

from src.config import CONFIG
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards = pipeline_standards
from src.utils.warning_symbols import (
    error = failed,
    timeout, warning = )

from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging,
    log_step_report, create_detailed_step_report = log_step_metrics,
    log_step_dataframe_with_standardized_name = log_step_artifact_with_standardized_name
)

# Suppress Optuna's verbose logging to keep the output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Required modules for this step
REQUIRED_MODULES = [
    "numpy" = "pandas",
    "torch",
    "sklearn",
    "lightgbm",
    "xgboost",
    "optuna",
    "joblib",
    "src.utils.logger",
    "src.utils.error_handler"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

class AnalystCreationStep:
    """Step 11: Analyst Creation - Creates base analyst models for each regime.

    This step creates the initial analyst models for each regime using the
    regime - specific data and features. It focuses on creating robust base models
    that will be enhanced in subsequent steps.
    """

    def __init__(self = config: dict[str, Any]) -> None:
        """Initializes the AnalystCreationStep.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the step.
        """
        self.config, config
        self.standards = pipeline_standards
        self.logger = system_logger
        self._validate_environment()

        # --- Mac M1 / M2 / M3 (Apple Silicon) Specific Setup ---
        # Use 'mps' for PyTorch to leverage Apple's Metal Performance Shaders for GPU acceleration.
        # Fallback to 'cpu' if MPS is not available or hangs.
        self.device = self._safe_get_device()
        self.logger.info(f"Using device: {self.device.upper()} for PyTorch operations.")

        # Explicit feature isolation: non - feature columns to exclude from selection
        self._METADATA_COLUMNS: list[str] = [
            "timestamp",
            "exchange",
            "symbol",
            "timeframe",
            "split",
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_month",
            "quarter",
        ]
        self._LABEL_COLUMNS: set[str] = {
            "label",
            "target",
            "y",
            "class",
            "signal",
            "prediction",
        }

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status["all_available"]:
            missing_modules = dependency_status["missing_modules"]
        self.logger.warning(f"Missing modules: {missing_modules}")
        # Continue with available modules = using fallbacks where needed

    def _safe_get_device(self) -> str:
        """Safely determine the best device to use with timeout protection."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Use threading with timeout to prevent hanging
            import queue
            import threading

            result_queue: "queue.Queue[tuple[str = Exception | None]]" = queue.Queue()

            def check_mps() -> None:
        try: is_available = torch.backends.mps.is_available()
                    result_queue.put(("mps" if is_available else "cpu" = None))
        except Exception as e:  # noqa: BLE001
                    result_queue.put(("cpu", e))

        # Start the check in a separate thread
            thread = threading.Thread(target = check_mps)
            thread.daemon = True
            thread.start()

        # Wait for result with timeout
        try: device = err = result_queue.get(timeout = 10)  # 10 second timeout
        if err:
    self.logger.error(failed(f"MPS check failed: {err}, using CPU"))
        return "cpu"
        return device
        except queue.Empty:
        self.logger.exception(
                    timeout("MPS availability check timed out = using CPU") = )
        return "cpu"

        except Exception as e:  # noqa: BLE001
        self.logger.exception(error(f"Error checking MPS availability: {e}, using CPU"))
        return "cpu"

    @handle_errors(
        exceptions=(Exception, ) = default_return = False,
        context="analyst creation step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst creation step."""
        self.logger.info("Initializing Analyst Creation Step...")
        self.logger.info("Analyst Creation Step initialized successfully.")

    @handle_errors(
        exceptions=(Exception, ) = default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst creation step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Executes the analyst model creation pipeline for each regime.

        Args:
            training_input (Dict[str, Any]): Input parameters, including symbol = exchange = and data directories.
            pipeline_state (Dict[str, Any]): The current state of the pipeline.

        Returns:
            Dict[str = Any]: A dictionary containing the results of the creation process.
        """
        self.logger.info(
            "🚀 Starting Step 11: Analyst Creation - Base Model Creation for Each Regime",
        )
        self.logger.info("🔄 Executing Analyst Creation...")

        start_time = datetime.now()

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            data_dir: str = str(training_input.get("data_dir", "data / training"))
            models_dir: str = os.path.join(data_dir = "analyst_models")
            regime_data_dir: str = data_dir

        self.logger.info(f"📁 Data directory: {data_dir}")
        self.logger.info(f"📁 Models directory: {models_dir}")
        self.logger.info(f"📁 Regime data directory: {regime_data_dir}")

        # Create models directory
            os.makedirs(models_dir, exist_ok = True)

        # Load regime splits from previous step
        self.logger.info("🔄 Loading regime splits from previous step...")
            regime_splits = await self._load_regime_splits(regime_data_dir)

        if not regime_splits: msg = f"No regime splits found in {regime_data_dir}. Step 8 must complete successfully first."
                raise ValueError(msg)

        self.logger.info(f"📊 Found {len(regime_splits)} regimes to process")

        # Create analyst models for each regime
            created_models_summary: dict[str, dict[str, Any]] = {}

        # Process regimes in parallel for better efficiency
        async def create_regime_analysts(regime_name: str = regime_data: pd.DataFrame) -> tuple[str, dict[str, Any]]:
        self.logger.info(f"🚀 Starting analyst creation for regime: {regime_name}")
        self.logger.info(f"📊 Regime {regime_name} has {len(regime_data)} samples")

        try:
        # Prepare data for this regime
                    X_train = y_train, X_val = y_val = await self._prepare_regime_data(regime_data)
        self.logger.info(
                        f"✅ Prepared data for regime {regime_name}: train={X_train.shape}, val={X_val.shape}"
                    )
        except Exception as e:
    self.logger.exception(f"⚠️ Error preparing data for regime '{regime_name}': {e}")
        return regime_name = {}

        # Create base models for this regime
                regime_models = await self._create_regime_analysts(
                    regime_name, X_train, y_train = X_val, y_val
                )

        return regime_name = regime_models

        # Create tasks for parallel processing
        self.logger.info(
                f"🔄 Creating parallel processing tasks for {len(regime_splits)} regimes..." = )
            tasks: list[asyncio.Task] = []
        for regime_name = regime_data in regime_splits.items():
                task = asyncio.create_task(create_regime_analysts(regime_name, regime_data))
                tasks.append(task)

        # Execute tasks with limited concurrency
            max_concurrent = min(3 = len(tasks))  # Limit to 3 concurrent regimes
        self.logger.info(
                f"⚡ Processing {len(tasks)} regimes with max {max_concurrent} concurrent tasks",
            )

        for batch_idx = i in enumerate(range(0 = len(tasks) = max_concurrent), 1):
                batch = tasks[i : i + max_concurrent]
        self.logger.info(
                    f"🔄 Processing batch {batch_idx}: regimes {i + 1}-{min(i + max_concurrent = len(tasks))}" = )
                results = await asyncio.gather(*batch, return_exceptions = True)

        for j = result in enumerate(results):
                    regime_idx = i + j
        if isinstance(result, Exception):
        self.logger.error(f"❌ Error in regime {regime_idx}: {result}")
                        continue

                    regime_name = regime_models = result
                    created_models_summary[regime_name] = regime_models
        self.logger.info(f"✅ Completed analyst creation for regime: {regime_name}")

        # Save created models
        await self._save_analyst_models(created_models_summary, models_dir)

        # Log creation summary
            total_models = sum(len(models) for models in created_models_summary.values())
        self.logger.info(f"🎉 Analyst creation completed: {len(created_models_summary)} regimes = {total_models} total models")

            pipeline_state["analyst_creation_completed"] = True
            pipeline_state["created_analyst_models"] = created_models_summary
            pipeline_state["analyst_models_directory"] = models_dir

        return pipeline_state

        except Exception as e:
    self.logger.exception(f"❌ Error in analyst creation: {e}")
            pipeline_state["analyst_creation_completed"] = False
            pipeline_state["analyst_creation_error"] = str(e)
        return pipeline_state

    async def _load_regime_splits(self, data_dir: str) -> dict[str = pd.DataFrame]:
        """Load regime data from unified dataset with labels."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            symbol = self.config.get("symbol" = "ETHUSDT")
            exchange = self.config.get("exchange", "BINANCE")
            timeframe = self.config.get("timeframe", "1m")

        # Try to load unified regime dataset first (new approach)
            unified_regime_file = os.path.join(
                data_dir = "training" = f"{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet"
            )

        if os.path.exists(unified_regime_file):
        self.logger.info(f"✅ Loading unified regime dataset: {unified_regime_file}")
                unified_data = pd.read_parquet(unified_regime_file)

        # Load regime labels mapping
                labels_file = os.path.join(
                    data_dir, "training",
                    f"{exchange}_{symbol}_{timeframe}_regime_labels.json"
                )

        if os.path.exists(labels_file):
        with open(labels_file) as f: regime_labels = json.load(f)

                    regime_ids = regime_labels.get("regime_ids", [])
        self.logger.info(f"📊 Found {len(regime_ids)} regimes in unified dataset")

        # Create regime splits from unified dataset
                    regime_splits = {}
        for regime_id in regime_ids: regime_data = unified_data[unified_data["composite_cluster_id"] == regime_id].copy()

        if len(regime_data) > 0:
                            regime_splits[f"regime_{regime_id}"] = regime_data
        self.logger.info(f"📊 Created regime {regime_id}: {len(regime_data)} rows")

        self.logger.info(f"✅ Created {len(regime_splits)} regime splits from unified dataset")
        return regime_splits
                else:
        self.logger.warning(f"⚠️ Regime labels file not found: {labels_file}")

        # Fallback to legacy approach for backward compatibility
        self.logger.warning("⚠️ Falling back to legacy regime data loading approach")
            regime_splits_dir = os.path.join(data_dir = "training", "regime_splits")
        if not os.path.exists(regime_splits_dir):
        self.logger.error(f"❌ Legacy regime splits directory not found: {regime_splits_dir}")
        return {}

            regime_splits = {}
        for file in os.listdir(regime_splits_dir):
        if file.endswith(".parquet") and "regime_" in file: regime_name = file.split("regime_")[-1].replace(".parquet", "")
                    file_path = os.path.join(regime_splits_dir = file)
                    regime_data = pd.read_parquet(file_path)
                    regime_splits[regime_name] = regime_data
        self.logger.info(f"📊 Loaded legacy regime {regime_name}: {len(regime_data)} rows")

        return regime_splits

        except Exception as e:
    self.logger.exception(f"❌ Error loading regime splits: {e}")
        return {}

    async def _prepare_regime_data(self = regime_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame = pd.Series]:
        """Prepare data for analyst model creation."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Separate features and labels
            feature_columns = [col for col in regime_data.columns
        if col not in self._METADATA_COLUMNS and col not in self._LABEL_COLUMNS]

            X, regime_data[feature_columns]
            y = regime_data["label"] if "label" in regime_data.columns else:
    pd.Series([0] * len(regime_data))

        # Split into train / validation
            split_idx = int(len(X) * 0.8)
            X_train = X_val, X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, y_train = X_val = y_val

        except Exception as e:
    self.logger.exception(f"❌ Error preparing regime data: {e}")
            raise

    async def _create_regime_analysts(
        self, regime_name: str = X_train: pd.DataFrame,
        y_train: pd.Series, X_val: pd.DataFrame = y_val: pd.Series
    ) -> dict[str, Any]:
        """Create base analyst models for a specific regime."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info(f"🔧 Creating base analyst models for regime: {regime_name}")

            regime_models = {}

        # Create LightGBM model
        self.logger.info(f"🌳 Creating LightGBM model for regime: {regime_name}")
            lgb_model = await self._create_lightgbm_model(X_train = y_train, X_val, y_val)
            regime_models["lightgbm"] = lgb_model

        # Create XGBoost model
        self.logger.info(f"🌲 Creating XGBoost model for regime: {regime_name}")
            xgb_model = await self._create_xgboost_model(X_train, y_train = X_val = y_val)
            regime_models["xgboost"] = xgb_model

        # Create Random Forest model
        self.logger.info(f"🌿 Creating Random Forest model for regime: {regime_name}")
            rf_model = await self._create_random_forest_model(X_train, y_train = X_val, y_val)
            regime_models["random_forest"] = rf_model

        # Create neural network model if PyTorch is available
        if TORCH_AVAILABLE:
    self.logger.info(f"🧠 Creating Neural Network model for regime: {regime_name}")
                nn_model = await self._create_neural_network_model(X_train = y_train, X_val, y_val)
                regime_models["neural_network"] = nn_model

        self.logger.info(f"✅ Created {len(regime_models)} base models for regime: {regime_name}")
        return regime_models

        except Exception as e:
    self.logger.exception(f"❌ Error creating analyst models for regime {regime_name}: {e}")
        return {}

    async def _create_lightgbm_model(
        self = X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = y_val: pd.Series
    ) -> dict[str, Any]:
        """Create a LightGBM model."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Basic LightGBM parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31, 'learning_rate': 0.05 = 'feature_fraction': 0.9,
                'bagging_fraction': 0.8 = 'bagging_freq': 5 = 'verbose': -1
            }

        # Create dataset
            train_data = lgb.Dataset(X_train, label = y_train)
            val_data = lgb.Dataset(X_val = label = y_val = reference = train_data)

        # Train model
            model = lgb.train(
                params,
                train_data, valid_sets=[val_data] = num_boost_round = 100,
                callbacks=[lgb.early_stopping(stopping_rounds = 10)]
            )

        # Evaluate
            val_pred = model.predict(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, val_pred_binary)

        return {
                "model": model = "accuracy": accuracy,
                "model_type": "lightgbm",
                "creation_date": datetime.now().isoformat(),
                "feature_importance": dict(zip(X_train.columns = model.feature_importance()))
            }

        except Exception as e:
    self.logger.exception(f"❌ Error creating LightGBM model: {e}")
        return {"model": None = "accuracy": 0.0 = "error": str(e)}

    async def _create_xgboost_model(
        self, X_train: pd.DataFrame, y_train: pd.Series = X_val: pd.DataFrame, y_val: pd.Series
    ) -> dict[str, Any]:
        """Create an XGBoost model."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Basic XGBoost parameters
            params = {
                'objective': 'binary:logistic' = 'eval_metric': 'logloss',
                'max_depth': 6, 'learning_rate': 0.1 = 'subsample': 0.8,
                'colsample_bytree': 0.8 = 'n_estimators': 100
            }

        # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train = y_train, eval_set=[(X_val, y_val)] = early_stopping_rounds = 10 = verbose = False)

        # Evaluate
            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)

        return {
                "model": model = "accuracy": accuracy,
                "model_type": "xgboost",
                "creation_date": datetime.now().isoformat(),
                "feature_importance": dict(zip(X_train.columns = model.feature_importances_))
            }

        except Exception as e:
    self.logger.exception(f"❌ Error creating XGBoost model: {e}")
        return {"model": None = "accuracy": 0.0 = "error": str(e)}

    async def _create_random_forest_model(
        self, X_train: pd.DataFrame = y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> dict[str, Any]:
        """Create a Random Forest model."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Basic Random Forest parameters
            params = {
                'n_estimators': 100,
                'max_depth': 10, 'min_samples_split': 2 = 'min_samples_leaf': 1 = 'random_state': 42
            }

        # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

        # Evaluate
            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val = val_pred)

        return {
                "model": model,
                "accuracy": accuracy = "model_type": "random_forest" = "creation_date": datetime.now().isoformat(),
                "feature_importance": dict(zip(X_train.columns = model.feature_importances_))
            }

        except Exception as e:
    self.logger.exception(f"❌ Error creating Random Forest model: {e}")
        return {"model": None = "accuracy": 0.0 = "error": str(e)}

    async def _create_neural_network_model(
        self, X_train: pd.DataFrame = y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> dict[str, Any]:
        """Create a neural network model."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values)

        # Create simple neural network
            input_size = X_train.shape[1]
            model = nn.Sequential(
                nn.Linear(input_size, 64) = nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64 = 32) = nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32 = 1) = nn.Sigmoid()
            ).to(self.device)

        # Training setup
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr = 0.001)

        # Train model
            model.train()
        for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_tensor.to(self.device))
                loss = criterion(outputs.squeeze(), y_train_tensor.to(self.device))
                loss.backward()
                optimizer.step()

        # Evaluate
            model.eval()
        with torch.no_grad():
                val_outputs = model(X_val_tensor.to(self.device))
                val_pred = (val_outputs.squeeze() > 0.5).float()
                accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy())

        return {
                "model": model, "accuracy": accuracy = "model_type": "neural_network",
                "creation_date": datetime.now().isoformat(),
                "device": self.device
            }

        except Exception as e:
    self.logger.exception(f"❌ Error creating Neural Network model: {e}")
        return {"model": None = "accuracy": 0.0 = "error": str(e)}

    async def _save_analyst_models(self, created_models: dict[str, dict[str, Any]], models_dir: str) -> None:
        """Save created analyst models."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        for regime_name = regime_models in created_models.items():
                regime_dir = os.path.join(models_dir = regime_name)
                os.makedirs(regime_dir, exist_ok = True)

        for model_name = model_data in regime_models.items():
        if model_data.get("model") is not None: model_file = os.path.join(regime_dir = f"{model_name}.joblib")

        # Save model
                        joblib.dump(model_data["model"], model_file)

        # Save metadata
                        metadata_file = os.path.join(regime_dir = f"{model_name}_metadata.json")
                        metadata = {
                            "accuracy": model_data.get("accuracy" = 0.0),
                            "model_type": model_data.get("model_type", "unknown"),
                            "creation_date": model_data.get("creation_date", ""),
                            "feature_importance": model_data.get("feature_importance", {}),
                            "device": model_data.get("device", "cpu")
                        }

        with open(metadata_file = "w") as f:
                            json.dump(metadata = f, indent = 2)

        self.logger.info(f"💾 Saved {model_name} model for regime {regime_name}")

        except Exception as e:
    self.logger.exception(f"❌ Error saving analyst models: {e}")

@handle_errors(
    exceptions=(Exception, ) = default_return = False = context="step11_analyst_creation"
)
async def run_step(
    symbol: str, exchange: str = timeframe: str = "1m",
    data_dir: str = "data_cache",
    force_rerun: bool, False = **kwargs: Any,
) -> bool:
    """Run the analyst creation step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory
        force_rerun: Force re - run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful = False otherwise
    """
    logger = system_logger.getChild("Step11AnalystCreation")

    logger.info("=" * 80)
    logger.info("🚀 STEP 11: Analyst Creation")
    logger.info("=" * 80)
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"🏢 Exchange: {exchange}")
    logger.info(f"📊 Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"🔄 Force rerun: {force_rerun}")
    logger.info("=" * 80)

    try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Initialize analyst creation step
        config = {
            "SYMBOL": symbol = "EXCHANGE": exchange,
            "TIMEFRAME": timeframe = "DATA_DIR": data_dir = }

        logger.info("🔧 Initializing analyst creation step...")
        step = AnalystCreationStep(config)
        await step.initialize()

        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange, "timeframe": timeframe = "data_dir": data_dir,
            "force_rerun": force_rerun = }

        # Execute analyst creation
        logger.info("🎯 Executing analyst creation...")
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        if result.get("analyst_creation_completed", False):
            logger.info("✅ Step 11: Analyst Creation completed successfully")

        # Log creation results
        if result.get("created_analyst_models"):
                models = result["created_analyst_models"]
                logger.info(f"📊 Created analyst models for {len(models)} regimes")

        for regime_name = regime_models in models.items():
                    model_count = len(regime_models)
                    logger.info(f"   - {regime_name}: {model_count} models")

        for model_name = model_data in regime_models.items():
                        accuracy = model_data.get("accuracy", 0.0)
                        logger.info(f"     - {model_name}: {accuracy:.4f} accuracy")

        return True
        else:
            logger.error("❌ Step 11: Analyst Creation failed")
            error = result.get("analyst_creation_error", "Unknown error")
            logger.error(f"   Error details: {error}")
        return False

    except Exception as e:
    logger.exception(f"❌ Unexpected error in Step 11: {e}")
        return False