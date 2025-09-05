from typing import Dict, List, Optional, Union, Any, Tuple
"""Step 11: Analyst Creation - Creates base analyst models for each regime.

This step creates the initial analyst models for each regime using the
regime-specific data and features. It focuses on creating robust base models
that will be enhanced in subsequent steps.
"""
import asyncio
import json
import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import numpy as np
import joblib
import optuna
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
optuna.logging.set_verbosity(optuna.logging.WARNING)

def handles_errors(exceptions: List[Any]=(Exception,), default_return: Any=None, context: Any=None) -> None:
    """Fallback error handling decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logging.error(f'Error in {func.__name__}: {e}')
                return default_return
        return wrapper
    return decorator

def traced(span_name: Any=None) -> None:
    """Fallback tracing decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validates(min_quality_score: int=None, max_correlation: int=None, required_grade: Any=None) -> None:
    """Fallback validation decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def error(message: str) -> None:
    return f'ERROR: {message}'

def failed(message: str) -> None:
    return f'FAILED: {message}'

def timeout(message: str) -> None:
    return f'TIMEOUT: {message}'

def warning(message: str) -> None:
    return f'WARNING: {message}'

def log_step_report(*args, **kwargs) -> None:
    return 'fallback_report'

def create_detailed_step_report(*args, **kwargs) -> Any:
    return {}

def log_step_metrics(*args, **kwargs) -> None:
    return None

def log_step_dataframe_with_standardized_name(*args, **kwargs) -> None:
    return 'fallback_dataframe'

def log_step_artifact_with_standardized_name(*args, **kwargs) -> None:
    return 'fallback_artifact'
system_logger = logging.getLogger(__name__)
system_logger.setLevel(logging.INFO)
if not system_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    system_logger.addHandler(handler)

class PipelineStandards:

    @staticmethod
    def validate_environment_dependencies(modules: List[Any]) -> bool:
        result = {module: True for module in modules}
        result['all_available'] = True
        result['missing_modules'] = []
        return result

class pipeline_standards:

    @staticmethod
    def build_path(path_type: Any, exchange: str, symbol: str) -> Any:
        return f'data/{path_type}/{exchange}/{symbol}'
REQUIRED_MODULES = ['numpy', 'pandas', 'torch', 'sklearn', 'lightgbm', 'xgboost', 'optuna', 'joblib']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

def validate_mandatory_dependencies() -> bool:
    """Validate that all mandatory ML dependencies are available."""
    missing_deps = []
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    try:
        import sklearn
    except ImportError:
        missing_deps.append('sklearn')
    try:
        import lightgbm
    except ImportError:
        missing_deps.append('lightgbm')
    try:
        import xgboost
    except ImportError:
        missing_deps.append('xgboost')
    try:
        import optuna
    except ImportError:
        missing_deps.append('optuna')
    try:
        import joblib
    except ImportError:
        missing_deps.append('joblib')
    if missing_deps:
        raise ImportError(f"Missing mandatory dependencies for Step 11: {', '.join(missing_deps)}. Please install them using: pip install {' '.join(missing_deps)}")
    return True

class AnalystCreationStep:
    """Step 11: Analyst Creation - Creates base analyst models for each regime.

    This step creates the initial analyst models for each regime using the
    regime-specific data and features. It focuses on creating robust base models
    that will be enhanced in subsequent steps.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initializes the AnalystCreationStep.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the step.
        """
        validate_mandatory_dependencies()
        self.config = config
        self.standards = pipeline_standards
        self.logger = system_logger
        self._validate_environment()
        self.device = self._safe_get_device()
        self.logger.info(f'Using device: {self.device.upper()} for PyTorch operations.')
        self._METADATA_COLUMNS: list[str] = ['timestamp', 'exchange', 'symbol', 'timeframe', 'split', 'year', 'month', 'day', 'day_of_week', 'day_of_month', 'quarter']
        self._LABEL_COLUMNS: set[str] = {'label', 'target', 'y', 'class', 'signal', 'prediction'}

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status['all_available']:
            missing_modules = dependency_status['missing_modules']
            self.logger.warning(f'Missing modules: {missing_modules}')

    def _safe_get_device(self) -> str:
        """Safely determine the best device to use with timeout protection."""
        try:
            import queue
            import threading
            result_queue: 'queue.Queue[tuple[str, Exception | None]]' = queue.Queue()

            def check_mps() -> None:
                try:
                    is_available = torch.backends.mps.is_available()
                    result_queue.put(('mps' if is_available else 'cpu', None))
                except Exception as e:
                    result_queue.put(('cpu', e))
            thread = threading.Thread(target=check_mps)
            thread.daemon = True
            thread.start()
            try:
                device, err = result_queue.get(timeout=10)
                if err:
                    self.logger.error(failed(f'MPS check failed: {err}, using CPU'))
                    return 'cpu'
                return device
            except queue.Empty:
                self.logger.exception(timeout('MPS availability check timed out, using CPU'))
                return 'cpu'
        except Exception as e:
            self.logger.exception(error(f'Error checking MPS availability: {e}, using CPU'))
            return 'cpu'

    @handles_errors(exceptions=(Exception,), default_return=False, context='analyst creation step initialization')
    async def initialize(self) -> None:
        """Initialize the analyst creation step."""
        self.logger.info('Initializing Analyst Creation Step...')
        self.logger.info('Analyst Creation Step initialized successfully.')

    @handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Execution failed'}, context='analyst creation step execution')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Executes the analyst model creation pipeline for each regime.

        Args:
            training_input (Dict[str, Any]): Input parameters, including symbol, exchange, and data directories.
            pipeline_state (Dict[str, Any]): The current state of the pipeline.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the creation process.
        """
        self.logger.info('🚀 Starting Step 11: Analyst Creation - Base Model Creation for Each Regime')
        self.logger.info('🔄 Executing Analyst Creation...')
        start_time = datetime.now()
        try:
            data_dir: str = str(training_input.get('data_dir', 'data/training'))
            models_dir: str = os.path.join(data_dir, 'analyst_models')
            regime_data_dir: str = data_dir
            self.logger.info(f'📁 Data directory: {data_dir}')
            self.logger.info(f'📁 Models directory: {models_dir}')
            self.logger.info(f'📁 Regime data directory: {regime_data_dir}')
            os.makedirs(models_dir, exist_ok=True)
            self.logger.info('🔄 Loading regime splits from previous step...')
            regime_splits = await self._load_regime_splits(regime_data_dir)
            if not regime_splits:
                msg = f'No regime splits found in {regime_data_dir}. Step 8 must complete successfully first.'
                raise ValueError(msg)
            self.logger.info(f'📊 Found {len(regime_splits)} regimes to process')
            created_models_summary: dict[str, dict[str, Any]] = {}

            async def create_regime_analysts(regime_name: str, regime_data: pd.DataFrame) -> tuple[str, dict[str, Any]]:
                self.logger.info(f'🚀 Starting analyst creation for regime: {regime_name}')
                self.logger.info(f'📊 Regime {regime_name} has {len(regime_data)} samples')
                try:
                    X_train, y_train, X_val, y_val = await self._prepare_regime_data(regime_data)
                    self.logger.info(f'✅ Prepared data for regime {regime_name}: train={X_train.shape}, val={X_val.shape}')
                except Exception as e:
                    self.logger.exception(f"⚠️ Error preparing data for regime '{regime_name}': {e}")
                    return (regime_name, {})
                regime_models = await self._create_regime_analysts(regime_name, X_train, y_train, X_val, y_val)
                return (regime_name, regime_models)
            self.logger.info(f'🔄 Creating parallel processing tasks for {len(regime_splits)} regimes...')
            tasks: list[asyncio.Task] = []
            for regime_name, regime_data in regime_splits.items():
                task = asyncio.create_task(create_regime_analysts(regime_name, regime_data))
                tasks.append(task)
            max_concurrent = min(3, len(tasks))
            self.logger.info(f'⚡ Processing {len(tasks)} regimes with max {max_concurrent} concurrent tasks')
            for batch_idx, i in enumerate(range(0, len(tasks), max_concurrent), 1):
                batch = tasks[i:i + max_concurrent]
                self.logger.info(f'🔄 Processing batch {batch_idx}: regimes {i + 1}-{min(i + max_concurrent, len(tasks))}')
                results = await asyncio.gather(*batch, return_exceptions=True)
                for j, result in enumerate(results):
                    regime_idx = i + j
                    if isinstance(result, Exception):
                        self.logger.error(f'❌ Error in regime {regime_idx}: {result}')
                        continue
                    regime_name, regime_models = result
                    created_models_summary[regime_name] = regime_models
                    self.logger.info(f'✅ Completed analyst creation for regime: {regime_name}')
            await self._save_analyst_models(created_models_summary, models_dir)
            total_models = sum((len(models) for models in created_models_summary.values()))
            self.logger.info(f'🎉 Analyst creation completed: {len(created_models_summary)} regimes, {total_models} total models')
            pipeline_state['analyst_creation_completed'] = True
            pipeline_state['created_analyst_models'] = created_models_summary
            pipeline_state['analyst_models_directory'] = models_dir
            return pipeline_state
        except Exception as e:
            self.logger.exception(f'❌ Error in analyst creation: {e}')
            pipeline_state['analyst_creation_completed'] = False
            pipeline_state['analyst_creation_error'] = str(e)
            return pipeline_state

    async def _load_regime_splits(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Load regime data from unified dataset with labels."""
        try:
            symbol = self.config.get('symbol', 'ETHUSDT')
            exchange = self.config.get('exchange', 'BINANCE')
            timeframe = self.config.get('timeframe', '1m')
            unified_regime_file = os.path.join(data_dir, 'training', f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet')
            if os.path.exists(unified_regime_file):
                self.logger.info(f'✅ Loading unified regime dataset: {unified_regime_file}')
                unified_data = pd.read_parquet(unified_regime_file)
                labels_file = os.path.join(data_dir, 'training', f'{exchange}_{symbol}_{timeframe}_regime_labels.json')
                if os.path.exists(labels_file):
                    with open(labels_file) as f:
                        regime_labels = json.load(f)
                    regime_ids = regime_labels.get('regime_ids', [])
                    self.logger.info(f'📊 Found {len(regime_ids)} regimes in unified dataset')
                    regime_splits = {}
                    for regime_id in regime_ids:
                        regime_data = unified_data[unified_data['composite_cluster_id'] == regime_id].copy()
                        if len(regime_data) > 0:
                            regime_splits[f'regime_{regime_id}'] = regime_data
                            self.logger.info(f'📊 Created regime {regime_id}: {len(regime_data)} rows')
                    self.logger.info(f'✅ Created {len(regime_splits)} regime splits from unified dataset')
                    return regime_splits
                else:
                    self.logger.warning(f'⚠️ Regime labels file not found: {labels_file}')
            self.logger.warning('⚠️ Falling back to legacy regime data loading approach')
            regime_splits_dir = os.path.join(data_dir, 'training', 'regime_splits')
            if not os.path.exists(regime_splits_dir):
                self.logger.error(f'❌ Legacy regime splits directory not found: {regime_splits_dir}')
                return {}
            regime_splits = {}
            for file in os.listdir(regime_splits_dir):
                if file.endswith('.parquet') and 'regime_' in file:
                    regime_name = file.split('regime_')[-1].replace('.parquet', '')
                    file_path = os.path.join(regime_splits_dir, file)
                    regime_data = pd.read_parquet(file_path)
                    regime_splits[regime_name] = regime_data
                    self.logger.info(f'📊 Loaded legacy regime {regime_name}: {len(regime_data)} rows')
            return regime_splits
        except Exception as e:
            self.logger.exception(f'❌ Error loading regime splits: {e}')
            return {}

    async def _prepare_regime_data(self, regime_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for analyst model creation."""
        try:
            feature_columns = [col for col in regime_data.columns if col not in self._METADATA_COLUMNS and col not in self._LABEL_COLUMNS]
            X = regime_data[feature_columns]
            y = regime_data['label'] if 'label' in regime_data.columns else pd.Series([0] * len(regime_data))
            split_idx = int(len(X) * 0.8)
            X_train, X_val = (X.iloc[:split_idx], X.iloc[split_idx:])
            y_train, y_val = (y.iloc[:split_idx], y.iloc[split_idx:])
            return (X_train, y_train, X_val, y_val)
        except Exception as e:
            self.logger.exception(f'❌ Error preparing regime data: {e}')
            raise

    async def _create_regime_analysts(self, regime_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create base analyst models for a specific regime."""
        try:
            self.logger.info(f'🔧 Creating base analyst models for regime: {regime_name}')
            regime_models = {}
            self.logger.info(f'🌳 Creating LightGBM model for regime: {regime_name}')
            lgb_model = await self._create_lightgbm_model(X_train, y_train, X_val, y_val)
            regime_models['lightgbm'] = lgb_model
            self.logger.info(f'🌲 Creating XGBoost model for regime: {regime_name}')
            xgb_model = await self._create_xgboost_model(X_train, y_train, X_val, y_val)
            regime_models['xgboost'] = xgb_model
            self.logger.info(f'🌿 Creating Random Forest model for regime: {regime_name}')
            rf_model = await self._create_random_forest_model(X_train, y_train, X_val, y_val)
            regime_models['random_forest'] = rf_model
            if TORCH_AVAILABLE:
                self.logger.info(f'🧠 Creating Neural Network model for regime: {regime_name}')
                nn_model = await self._create_neural_network_model(X_train, y_train, X_val, y_val)
                regime_models['neural_network'] = nn_model
            self.logger.info(f'✅ Created {len(regime_models)} base models for regime: {regime_name}')
            return regime_models
        except Exception as e:
            self.logger.exception(f'❌ Error creating analyst models for regime {regime_name}: {e}')
            return {}

    async def _create_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create a LightGBM model."""
        try:
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1}
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=100, callbacks=[lgb.early_stopping(stopping_rounds=10)])
            val_pred = model.predict(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, val_pred_binary)
            return {'model': model, 'accuracy': accuracy, 'model_type': 'lightgbm', 'creation_date': datetime.now().isoformat(), 'feature_importance': dict(zip(X_train.columns, model.feature_importance()))}
        except Exception as e:
            self.logger.exception(f'❌ Error creating LightGBM model: {e}')
            raise RuntimeError(f'Failed to create LightGBM model: {e}')

    async def _create_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create an XGBoost model."""
        try:
            params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 100}
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            return {'model': model, 'accuracy': accuracy, 'model_type': 'xgboost', 'creation_date': datetime.now().isoformat(), 'feature_importance': dict(zip(X_train.columns, model.feature_importances_))}
        except Exception as e:
            self.logger.exception(f'❌ Error creating XGBoost model: {e}')
            raise RuntimeError(f'Failed to create XGBoost model: {e}')

    async def _create_random_forest_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create a Random Forest model."""
        try:
            params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42}
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            return {'model': model, 'accuracy': accuracy, 'model_type': 'random_forest', 'creation_date': datetime.now().isoformat(), 'feature_importance': dict(zip(X_train.columns, model.feature_importances_))}
        except Exception as e:
            self.logger.exception(f'❌ Error creating Random Forest model: {e}')
            raise RuntimeError(f'Failed to create Random Forest model: {e}')

    async def _create_neural_network_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create a neural network model."""
        try:
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values)
            input_size = X_train.shape[1]
            model = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()).to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_tensor.to(self.device))
                loss = criterion(outputs.squeeze(), y_train_tensor.to(self.device))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(self.device))
                val_pred = (val_outputs.squeeze() > 0.5).float()
                accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy())
            return {'model': model, 'accuracy': accuracy, 'model_type': 'neural_network', 'creation_date': datetime.now().isoformat(), 'device': self.device}
        except Exception as e:
            self.logger.exception(f'❌ Error creating Neural Network model: {e}')
            raise RuntimeError(f'Failed to create Neural Network model: {e}')

    async def _save_analyst_models(self, created_models: dict[str, dict[str, Any]], models_dir: str) -> None:
        """Save created analyst models."""
        try:
            for regime_name, regime_models in created_models.items():
                regime_dir = os.path.join(models_dir, regime_name)
                os.makedirs(regime_dir, exist_ok=True)
                for model_name, model_data in regime_models.items():
                    if model_data.get('model') is not None:
                        model_file = os.path.join(regime_dir, f'{model_name}.joblib')
                        joblib.dump(model_data['model'], model_file)
                        metadata_file = os.path.join(regime_dir, f'{model_name}_metadata.json')
                        metadata = {'accuracy': model_data.get('accuracy', 0.0), 'model_type': model_data.get('model_type', 'unknown'), 'creation_date': model_data.get('creation_date', ''), 'feature_importance': model_data.get('feature_importance', {}), 'device': model_data.get('device', 'cpu')}
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        self.logger.info(f'💾 Saved {model_name} model for regime {regime_name}')
        except Exception as e:
            self.logger.exception(f'❌ Error saving analyst models: {e}')
            raise RuntimeError(f'Failed to save analyst models: {e}')

@handles_errors(exceptions=(Exception,), default_return=False, context='step11_analyst_creation')
async def run_step(symbol: str, exchange: str, timeframe: str='1m', data_dir: str='data_cache', force_rerun: bool=False, **kwargs: Any) -> bool:
    """Run the analyst creation step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    logger = system_logger.getChild('Step11AnalystCreation')
    logger.info('=' * 80)
    logger.info('🚀 STEP 11: Analyst Creation')
    logger.info('=' * 80)
    logger.info(f'🎯 Symbol: {symbol}')
    logger.info(f'🏢 Exchange: {exchange}')
    logger.info(f'📊 Timeframe: {timeframe}')
    logger.info(f'📁 Data directory: {data_dir}')
    logger.info(f'🔄 Force rerun: {force_rerun}')
    logger.info('=' * 80)
    try:
        config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir}
        logger.info('🔧 Initializing analyst creation step...')
        step = AnalystCreationStep(config)
        await step.initialize()
        training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'force_rerun': force_rerun}
        logger.info('🎯 Executing analyst creation...')
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        if result.get('analyst_creation_completed', False):
            logger.info('✅ Step 11: Analyst Creation completed successfully')
            if result.get('created_analyst_models'):
                models = result['created_analyst_models']
                logger.info(f'📊 Created analyst models for {len(models)} regimes')
                for regime_name, regime_models in models.items():
                    model_count = len(regime_models)
                    logger.info(f'   - {regime_name}: {model_count} models')
                    for model_name, model_data in regime_models.items():
                        accuracy = model_data.get('accuracy', 0.0)
                        logger.info(f'     - {model_name}: {accuracy:.4f} accuracy')
            return True
        else:
            logger.error('❌ Step 11: Analyst Creation failed')
            error = result.get('analyst_creation_error', 'Unknown error')
            logger.error(f'   Error details: {error}')
            return False
    except Exception as e:
        logger.exception(f'❌ Unexpected error in Step 11: {e}')
        return False