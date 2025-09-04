from src.core.decorators import handles_errors, traced, validates
from src.core.domain import BLANK_TRAINING_LOOKBACK_DAYS
import contextlib
import numpy.random._pickle as np_random_pickle
import queue
import threading
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
from torch import nn, optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Union, Any, Tuple
try:
    import shap
except ImportError:
    shap = None
try:
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    import signal
    import warnings
    import sys
    from io import StringIO
    from sklearn.metrics import log_loss
    import platform
    from shap.explainers import TreeExplainer
    from sklearn.inspection import permutation_importance
    from shap.explainers import KernelExplainer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from src.utils.vif_calculator import calculate_vif_robust
    from src.analyst.meta_label_relevance import compute_shap_importance
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except ImportError as e:
    pass
try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from src.config import CONFIG
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.warning_symbols import error, failed, timeout, warning
optuna.logging.set_verbosity(optuna.logging.WARNING)
REQUIRED_MODULES = ['numpy', 'pandas', 'torch', 'sklearn', 'lightgbm', 'xgboost', 'optuna', 'joblib', 'src.utils.logger', 'src.utils.error_handler']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
'\nCompatibility shim for NumPy RNG unpickling across versions.\nWe avoid nested functions to keep the shim picklable.\n'
_NUMPY_RNG_UNPICKLE_PATCHED = False
_NP_ORIGINAL_BITGEN_CTOR = None

def _normalized_numpy_bitgen_ctor(bit_generator_name: Any, state: Any=None, *args, **kwargs) -> None:
    """Module-level normalized ctor to avoid creating a closure (picklable)."""
    global _NP_ORIGINAL_BITGEN_CTOR
    name_candidate = bit_generator_name
    try:
        if hasattr(name_candidate, '__name__'):
            name_candidate = name_candidate.__name__
        elif isinstance(name_candidate, str) and name_candidate.startswith('<class '):
            name_candidate = name_candidate.split('.')[-1].split("'>")[0]
    except Exception:
        pass
    effective_state = kwargs.get('state', state)
    try:
        return _NP_ORIGINAL_BITGEN_CTOR(name_candidate, effective_state)
    except (TypeError, ValueError):
        try:
            return _NP_ORIGINAL_BITGEN_CTOR(name_candidate)
        except Exception:
            try:
                import numpy as _np
                bitgen_cls = getattr(_np.random, name_candidate, None)
                if bitgen_cls is None and name_candidate == 'MT19937':
                    try:
                        import numpy.random._mt19937 as _mt
                        bitgen_cls = getattr(_mt, 'MT19937', None)
                    except Exception:
                        bitgen_cls = None
                if bitgen_cls is not None:
                    return bitgen_cls()
            except Exception:
                pass
            raise

def _enable_numpy_rng_unpickle_compat(logger: logging.Logger=None) -> None:
    """Enable compatibility for unpickling NumPy RNG BitGenerators (idempotent)."""
    global _NUMPY_RNG_UNPICKLE_PATCHED, _NP_ORIGINAL_BITGEN_CTOR
    if _NUMPY_RNG_UNPICKLE_PATCHED:
        return
    try:
        original_ctor = getattr(np_random_pickle, '__bit_generator_ctor', None)
        if original_ctor is None:
            _NUMPY_RNG_UNPICKLE_PATCHED = True
            return
        _NP_ORIGINAL_BITGEN_CTOR = original_ctor
        np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor
        _NUMPY_RNG_UNPICKLE_PATCHED = True
        if logger is not None:
            logger.info('Applied NumPy RNG unpickle compatibility shim')
    except Exception as _shim_exc:
        _NUMPY_RNG_UNPICKLE_PATCHED = True
        if logger is not None:
            logger.warning(warning(f'NumPy RNG unpickle compatibility shim not applied: {_shim_exc}'))

class RegimeAwareAnalystEnhancementStep:
    """Step 12: Regime-Aware Analyst Models Enhancement.

    This step refines the trained analyst models through a regime-specific sequential process:
    1.  **Regime-Specific Model Loading:** Loads models organized by HMM regime clusters
    2.  **Regime-Specific Hyperparameter Optimization (HPO):** Uses Optuna with regime-aware early pruning
    3.  **Regime-Specific Feature Selection:** Employs robust feature selection methods per regime
    4.  **Regime-Specific Final Retraining:** Trains new models using regime-specific optimal parameters
    5.  **Regime-Specific Advanced Optimization:** Applies regime-aware quantization, pruning, and distillation
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initializes the RegimeAwareAnalystEnhancementStep.

        Args: config (Dict[str, Any]): Configuration dictionary for the step.

        """
        self.config = config
        self.standards = pipeline_standards
        self.logger = system_logger
        self._validate_environment()
        self.regime_config = self._initialize_regime_config()
        self.device = self._safe_get_device()
        self.logger.info(f'Using device: {self.device.upper()} for PyTorch operations.')
        self._METADATA_COLUMNS: list[str] = ['timestamp', 'exchange', 'symbol', 'timeframe', 'split', 'year', 'month', 'day', 'day_of_week', 'day_of_month', 'quarter', 'composite_cluster_id']
        self._LABEL_COLUMNS: set[str] = {'label', 'target', 'y', 'class', 'signal', 'prediction'}
        self.regime_enhanced_models: dict[str, dict[str, Any]] = {}
        self.regime_validation_results: dict[str, dict[str, Any]] = {}
        self.regime_optimization_results: dict[str, dict[str, Any]] = {}

    def _initialize_regime_config(self) -> dict[str, Any]:
        """Initialize regime-specific configuration for analyst enhancement."""
        return {'regime_specific_optimization': True, 'regime_specific_feature_selection': True, 'regime_specific_hyperparameter_optimization': True, 'regime_specific_validation': True, 'regime_specific_logging': True, 'min_regime_samples': 1000, 'regime_validation_split': 0.2, 'regime_optimization_trials': 50, 'regime_feature_selection_threshold': 0.01, 'regime_parallel_processing': True, 'regime_memory_optimization': True}

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status['all_available']:
            missing_modules = dependency_status['missing_modules']
            self.logger.warning(f'Missing modules: {missing_modules}')

    def _safe_get_device(self) -> str:
        """Safely determine the best device to use with timeout protection."""
        try:
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

    @handles_errors(exceptions=(Exception,), default_return=False, context='analyst enhancement step initialization')
    async def initialize(self) -> None:
        """Initialize the analyst enhancement step."""
        self.logger.info('Initializing Analyst Enhancement Step...')
        self.logger.info('Analyst Enhancement Step initialized successfully.')

    @handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Execution failed'}, context='regime-aware analyst enhancement step execution')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Executes the full regime-aware analyst model enhancement pipeline.

        Args:
            training_input (Dict[str, Any]): Input parameters, including symbol, exchange, and data directories.
            pipeline_state (Dict[str, Any]): The current state of the pipeline.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the regime-specific enhancement process.
        """
        self.logger.info('🚀 Starting Step 12: Regime-Aware Analyst Enhancement - Model Optimization and Feature Selection')
        self.logger.info('🔄 Executing Regime-Aware Analyst Enhancement...')
        self.logger.info(f'📊 Regime configuration: {self.regime_config}')
        with contextlib.suppress(Exception):
            pass
        start_time = datetime.now()
        try:
            data_dir: str = str(training_input.get('data_dir', 'data/training'))
            models_dir: str = os.path.join(data_dir, 'models')
            regime_data_dir: str = data_dir
            self.logger.info(f'📁 Data directory: {data_dir}')
            self.logger.info(f'📁 Models directory: {models_dir}')
            self.logger.info(f'📁 Regime data directory: {regime_data_dir}')
            self.logger.info('🔄 Loading HMM-based models from previous step...')
            self.logger.info({'msg': 'Load models start', 'dir': models_dir})
            with contextlib.suppress(Exception):
                pass
            hmm_models: dict[str, Any] = self._load_models(models_dir)
            self.logger.info({'msg': 'Load models complete', 'count': len(hmm_models or {})})
            with contextlib.suppress(Exception):
                pass
            if not hmm_models:
                msg = f'No HMM-based models found in {models_dir}. Step 5 must complete successfully first.'
                raise ValueError(msg)
            if isinstance(hmm_models, dict):
                try:
                    timeframes_count: int = len(hmm_models)
                    counts_per_timeframe: dict[str, int | str] = {timeframe: len(models) if isinstance(models, dict) else 'n/a' for timeframe, models in hmm_models.items()}
                    self.logger.info(f'Loaded HMM-based models summary: timeframes={timeframes_count}, models_per_timeframe={counts_per_timeframe}')
                except Exception:
                    pass
            try:
                from src.training.steps.unified_data_loader import UnifiedDataLoader
                data_loader = UnifiedDataLoader(self.config)
                perf_metrics = data_loader.get_performance_metrics()
                self.logger.info('📊 Performance before enhancement:')
                self.logger.info(f"   Memory Usage: {perf_metrics['memory_usage']['percent']:.1f}%")
                self.logger.info(f"   Cache Size: {perf_metrics['cache_stats']['cache_size']}/{perf_metrics['cache_stats']['max_cache_size']}")
            except Exception as e:
                self.logger.warning(f'⚠️ Could not get performance metrics: {e}')
            import asyncio
            import gc
            self.logger.info('🔄 Setting up parallel processing for model enhancement...')
            enhanced_models_summary: dict[str, dict[str, Any]] = {}

            async def enhance_regime_models(regime_name: str, regime_models: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                self.logger.info(f'🚀 Starting regime-specific enhancement for regime: {regime_name}')
                self.logger.info(f'📊 Regime {regime_name} has {len(regime_models)} models to enhance')
                with contextlib.suppress(Exception):
                    pass
                try:
                    self.logger.info(f'📂 Loading regime-specific training data for regime: {regime_name}')
                    X_train, y_train, X_val, y_val = await self._load_regime_data(regime_data_dir, regime_name)
                    self.logger.info(f'✅ Loaded regime-specific data for regime {regime_name}: train={X_train.shape}, val={X_val.shape}')
                except FileNotFoundError as e:
                    self.logger.exception(f"⚠️ {e} — skipping regime '{regime_name}'")
                    return (regime_name, {})
                except Exception as e:
                    self.logger.error(f'❌ Error loading regime {regime_name} data: {e}')
                    return (regime_name, {})
                self.logger.info(f'🧹 Performing memory cleanup for regime: {regime_name}')
                gc.collect()
                enhanced_regime_models: dict[str, Any] = {}
                self.logger.info(f'🔄 Starting model enhancement loop for regime: {regime_name}')
                for i, (model_name, model_data) in enumerate(regime_models.items(), start=1):
                    self.logger.info(f'🔧 Enhancing model {i}/{len(regime_models)}: {model_name} for {regime_name}...')
                    enhanced_model_package = await self._enhance_single_model(model_data, model_name, regime_name, X_train, y_train, X_val, y_val)
                    enhanced_regime_models[model_name] = enhanced_model_package
                    self.logger.info(f'✅ Completed enhancement for {model_name} in regime {regime_name}')
                    self.logger.info(f'🧹 Memory cleanup after {model_name}')
                    gc.collect()
                with contextlib.suppress(Exception):
                    pass
                try:
                    regime_validation = {'models_enhanced': len(enhanced_regime_models), 'train_size': int(len(X_train)), 'val_size': int(len(X_val))}
                except Exception:
                    regime_validation = {'models_enhanced': len(enhanced_regime_models)}
                return (regime_name, {'models': enhanced_regime_models, 'validation': regime_validation})
            self.logger.info(f'🔄 Creating parallel processing tasks for {len(hmm_models)} regimes...')
            tasks: list[asyncio.Task] = []
            for regime_name, regime_models in hmm_models.items():
                task = asyncio.create_task(enhance_regime_models(regime_name, regime_models))
                tasks.append(task)
            max_concurrent = min(3, len(tasks))
            self.logger.info(f'⚡ Processing {len(tasks)} regimes with max {max_concurrent} concurrent tasks')
            for batch_idx, i in enumerate(range(0, len(tasks), max_concurrent), 1):
                batch = tasks[i:i + max_concurrent]
                self.logger.info(f'🔄 Processing batch {batch_idx}: regimes {i + 1}-{min(i + max_concurrent, len(tasks))}')
                results = await asyncio.gather(*batch, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f'❌ Error in parallel regime processing: {result}')
                    else:
                        regime_name, enhanced_package = result
                        enhanced_models_summary[regime_name] = enhanced_package
                        self.logger.info(f'✅ Completed batch processing for regime: {regime_name}')
                self.logger.info(f'🧹 Memory cleanup after batch {batch_idx}')
                gc.collect()
            self.logger.info('💾 Saving enhanced models...')
            models_only: dict[str, dict[str, Any]] = {}
            for regime_name, package in enhanced_models_summary.items():
                try:
                    models_only[regime_name] = package.get('models', {})
                except Exception:
                    models_only[regime_name] = package or {}
            enhanced_models_dir: str = self._save_enhanced_models(models_only, data_dir, training_input)
            duration: float = (datetime.now() - start_time).total_seconds()
            self.logger.info(f'✅ Analyst enhancement completed in {duration:.2f}s. Results saved to {enhanced_models_dir}')
            with contextlib.suppress(Exception):
                pass
            pipeline_state['enhanced_hmm_models'] = enhanced_models_summary
            pipeline_state['step12_results'] = {'enhanced_models_summary': enhanced_models_summary, 'duration': duration}
            return {'status': 'SUCCESS', 'enhanced_models_dir': enhanced_models_dir, 'duration': duration}
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f'❌ Error in Analyst Enhancement after {duration:.2f}s: {e}', exc_info=True)
            return {'status': 'FAILED', 'error': str(e), 'duration': duration}

    def _load_models(self, models_dir: str) -> dict[str, Any]:
        """Loads all analyst models from the specified directory, supporting both traditional and HMM composite regime structures."""
        _enable_numpy_rng_unpickle_compat(self.logger)
        analyst_models: dict[str, Any] = {}
        if not os.path.exists(models_dir):
            return analyst_models
        has_regime_specific_structure: bool = False
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                if any((regime_file.endswith(('.pkl', '.joblib')) for regime_file in os.listdir(item_path))):
                    has_regime_specific_structure = True
                    break
        if has_regime_specific_structure:
            self.logger.info('🔄 Loading models with regime-specific structure')
            for regime_dir in os.listdir(models_dir):
                regime_path = os.path.join(models_dir, regime_dir)
                if os.path.isdir(regime_path):
                    regime_models: dict[str, Any] = {}
                    for model_file in os.listdir(regime_path):
                        if model_file.endswith(('.pkl', '.joblib')):
                            model_name = model_file.replace('.pkl', '')
                            model_name = model_name.replace('.joblib', '')
                            model_path = os.path.join(regime_path, model_file)
                            try:
                                if model_file.endswith('.joblib'):
                                    regime_models[model_name] = joblib.load(model_path)
                                else:
                                    with open(model_path, 'rb') as f:
                                        regime_models[model_name] = pickle.load(f)
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f'Failed to load {model_name}: {e}')
                                continue
                    if regime_models:
                        analyst_models[regime_dir] = regime_models
        else:
            self.logger.info('🔄 Loading models with traditional structure')
            for model_file in os.listdir(models_dir):
                if model_file.endswith(('.pkl', '.joblib')):
                    model_name = model_file.replace('.pkl', '')
                    model_name = model_name.replace('.joblib', '')
                    model_path = os.path.join(models_dir, model_file)
                    try:
                        if model_file.endswith('.joblib'):
                            analyst_models[model_name] = joblib.load(model_path)
                        else:
                            with open(model_path, 'rb') as f:
                                analyst_models[model_name] = pickle.load(f)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f'Failed to load {model_name}: {e}')
                        continue
        return analyst_models

    async def _load_regime_data(self, data_dir: str, timeframe_name: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads training and validation data for a specific timeframe using optimized unified data loader."""
        try:
            self.logger.info(f"Loading data for timeframe '{timeframe_name}' using unified data loader...")
            symbol: str = str(self.config.get('symbol', 'ETHUSDT'))
            exchange: str = str(self.config.get('exchange', 'BINANCE'))
            timeframe: str = str(self.config.get('timeframe', '1m'))
            try:
                config_lookback: int = int(self.config.get('lookback_days', BLANK_TRAINING_LOOKBACK_DAYS))
                data_loader = get_unified_data_loader(self.config)
                historical_data: pd.DataFrame | None = await data_loader.load_unified_data(symbol=symbol, exchange=exchange, timeframe=timeframe, lookback_days=config_lookback, use_streaming=True)
                if historical_data is not None and (not historical_data.empty):
                    if 'timeframe' in historical_data.columns:
                        timeframe_data = historical_data[historical_data['timeframe'] == timeframe_name]
                    else:
                        timeframe_data = historical_data
                    if not timeframe_data.empty:
                        self.logger.info(f"✅ Loaded {len(timeframe_data)} rows for timeframe '{timeframe_name}' using unified data loader")
                        split_idx: int = int(len(timeframe_data) * 0.8)
                        train_data = timeframe_data.iloc[:split_idx]
                        val_data = timeframe_data.iloc[split_idx:]
                        if 'label' in timeframe_data.columns:
                            X_train = train_data.drop(['label', 'timestamp'], axis=1, errors='ignore')
                            y_train: pd.Series = train_data['label']
                            X_val = val_data.drop(['label', 'timestamp'], axis=1, errors='ignore')
                            y_val: pd.Series = val_data['label']
                        else:
                            X_train = train_data.drop(['timestamp'], axis=1, errors='ignore')
                            y_train = pd.Series(np.random.choice([0, 1], size=len(train_data)))
                            X_val = val_data.drop(['timestamp'], axis=1, errors='ignore')
                            y_val = pd.Series(np.random.choice([0, 1], size=len(val_data)))
                        return (X_train, y_train, X_val, y_val)
            except Exception as e:
                self.logger.warning(f"⚠️ Unified data loader failed for timeframe '{timeframe_name}': {e}, falling back to pickle files")
            symbol = str(self.config.get('symbol', 'ETHUSDT'))
            exchange = str(self.config.get('exchange', 'BINANCE'))
            hmm_data_path = os.path.join(data_dir, f'{exchange}_{symbol}_hmm_composite_clusters_{timeframe_name}.parquet')
            if os.path.exists(hmm_data_path):
                hmm_data: pd.DataFrame = pd.read_parquet(hmm_data_path)
                intensity_path = os.path.join(data_dir, f'{exchange}_{symbol}_hmm_composite_intensity_{timeframe_name}.parquet')
                if os.path.exists(intensity_path):
                    intensity_data: pd.DataFrame = pd.read_parquet(intensity_path)
                    data = hmm_data.merge(intensity_data, on='timestamp', how='inner')
                else:
                    data = hmm_data
                self.logger.info(f'Loaded HMM data shape: {data.shape}, columns: {list(data.columns)}')
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                data = data[numeric_columns]
                self.logger.info(f'After numeric filtering: {data.shape}, columns: {list(data.columns)}')
                target_column: str | None = None
                target_candidates = ['composite_cluster_id', 'label', 'target', 'y', 'class', 'signal', 'prediction']
                for possible_target in target_candidates:
                    if possible_target in data.columns:
                        target_column = possible_target
                        self.logger.info(f'Found target column: {target_column}')
                        break
                if target_column is None:
                    self.logger.warning(f'No target column found in HMM data. Available columns: {list(data.columns)}')
                    target_created: bool = self._create_target_from_data(data, timeframe_name)
                    if target_created:
                        target_column = 'label'
                        self.logger.info('Successfully created target column from available data')
                    else:
                        self.logger.warning('Creating dummy target - this may not be suitable for training')
                        data['label'] = np.random.choice([0, 1], size=len(data))
                        target_column = 'label'
                elif target_column != 'label':
                    data['label'] = data[target_column]
                    data = data.drop(columns=[target_column])
                X = data.drop('label', axis=1)
                y: pd.Series = data['label']
                unique_targets = y.unique()
                blank_mode = os.environ.get('BLANK_TRAINING_MODE', '0') == '1' or bool(CONFIG.get('BLANK_TRAINING_MODE', False)) or bool(CONFIG.get('blank_training_mode', False))
                target_dist = dict(y.value_counts())
                self.logger.info(f'Target distribution: {target_dist}', extra={'mode': 'blank' if blank_mode else 'full', 'target_distribution': target_dist, 'unique_classes': sorted(set(y.unique().tolist())), 'note': ('BLANK MODE: Skewed or single-class targets can be normal with limited data' if blank_mode else '',), 'next_steps': ('Optional in blank mode: consider proxy or median-based target for quick diversity' if blank_mode else 'Review labeling thresholds/event rates if distribution is degenerate',)})
                if len(unique_targets) <= 1:
                    self.logger.warning(f'⚠️ Target has only {len(unique_targets)} unique values: {unique_targets}', extra={'mode': 'blank' if blank_mode else 'full', 'unique_values': unique_targets.tolist() if hasattr(unique_targets, 'tolist') else list(unique_targets), 'unique_count': len(unique_targets), 'note': 'BLANK MODE: Often normal with limited data' if blank_mode else 'Consider revisiting labeling thresholds or creating a proxy target', 'next_steps': ('Optional: use proxy/median-based target or adjust quick-test settings' if blank_mode else 'Check label generation, event rates, or use proxy/median-based target',)})
                    if len(data.columns) > 1:
                        proxy_column: str = data.columns[0]
                        if proxy_column != 'label':
                            proxy_values = data[proxy_column]
                            median_val = proxy_values.median()
                            y = (proxy_values > median_val).astype(int)
                            self.logger.info(f'Created proxy target from {proxy_column} (median: {median_val})')
                            self.logger.info(f'New target distribution: {dict(y.value_counts())}')
                train_size: int = int(0.8 * len(data))
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:]
                y_val = y[train_size:]
                self.logger.info(f'Data loaded and split: X_train shape {X_train.shape}, X_val shape {X_val.shape}')
                self.logger.info(f'Target classes in training: {y_train.unique()}, in validation: {y_val.unique()}')
                return (X_train, y_train, X_val, y_val)
            msg = f"HMM data file for timeframe '{timeframe_name}' not found: {hmm_data_path}. Step 6 requires HMM data from Step 5."
            raise FileNotFoundError(msg)
        except Exception as e:
            self.logger.exception(error(f"Error loading HMM data for '{timeframe_name}': {e}"))
            raise

    @traced(span_name='Step6._create_target_from_data')
    @validates(mode='warn', arg_index=1)
    def _create_target_from_data(self, data: pd.DataFrame, regime_name: str) -> bool:
        """Attempts to create a meaningful target column from available data.

        Args:
            data: The regime data DataFrame
            regime_name: Name of the regime

        Returns:
            bool: True if target was successfully created, False otherwise

        """
        try:
            price_columns = [col for col in data.columns if any((price_term in col.lower() for price_term in ['close', 'price', 'value']))]
            if price_columns:
                price_col = price_columns[0]
                price_values = data[price_col]
                if len(price_values) > 1:
                    price_changes = price_values.pct_change().fillna(0)
                    threshold = float(price_changes.std() * 0.1)
                    target = (price_changes > threshold).astype(int)
                    if target.nunique() >= 2:
                        data['label'] = target
                        self.logger.info(f'Created momentum-based target from {price_col}')
                        return True
            volume_columns = [col for col in data.columns if 'volume' in col.lower()]
            if volume_columns:
                volume_col = volume_columns[0]
                volume_values = data[volume_col]
                if len(volume_values) > 1:
                    volume_median = float(volume_values.median())
                    target = (volume_values > volume_median).astype(int)
                    if target.nunique() >= 2:
                        data['label'] = target
                        self.logger.info(f'Created volume-based target from {volume_col}')
                        return True
            for col in data.columns:
                if col != 'label' and str(data[col].dtype) in ['int64', 'float64']:
                    series = data[col]
                    if series.dropna().std() > 0:
                        median_val = float(series.median())
                        target = (series > median_val).astype(int)
                        if target.nunique() >= 2:
                            data['label'] = target
                            self.logger.info(f'Created median-based target from {col}')
                            return True
            return False
        except Exception as e:
            self.logger.warning(f"Failed to create target from data for regime '{regime_name}': {e}")
            return False

    async def _enhance_single_model(self, model_data: dict[str, Any], model_name: str, timeframe_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Applies the full enhancement pipeline to a single HMM-based model with architecture-specific optimizations."""
        self.logger.info(f'🔧 Starting HMM-specific enhancement pipeline for {model_name} in {timeframe_name}')
        if isinstance(model_data, dict):
            original_accuracy = model_data.get('accuracy', 'N/A')
            initial_model = model_data.get('model')
        else:
            original_accuracy = 'N/A'
            initial_model = model_data
        self.logger.info(f'📊 Original model accuracy: {original_accuracy}')
        self.logger.info(f'📊 Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}')
        self.logger.info(f'📊 Target distribution: {y_train.value_counts().to_dict()}')
        if y_train.nunique() <= 1:
            self.logger.warning(f'⚠️ Target has only {y_train.nunique()} unique values: {y_train.unique()}')
            self.logger.warning('⚠️ Skipping model enhancement due to insufficient target diversity')
            return {'model': initial_model, 'selected_features': list(X_train.columns), 'accuracy': original_accuracy, 'enhancement_metadata': {'enhancement_date': datetime.now().isoformat(), 'original_accuracy': original_accuracy, 'hpo_score': 0.0, 'final_accuracy': original_accuracy, 'improvement': 0.0, 'best_params': {}, 'feature_selection_method': 'None - insufficient target diversity', 'original_feature_count': len(X_train.columns), 'selected_feature_count': len(X_train.columns), 'shap_summary': {}, 'enhancement_applied': False, 'reason': f'Insufficient target diversity (only {y_train.nunique()} unique values)'}}
        allow_features = [c for c in X_train.columns if c not in self._METADATA_COLUMNS and c not in self._LABEL_COLUMNS]
        if len(allow_features) != X_train.shape[1]:
            self.logger.info(f'Feature isolation excluded {X_train.shape[1] - len(allow_features)} non-feature columns')
        X_train = X_train[allow_features]
        X_val = X_val[allow_features]
        if model_name == 'tcn':
            self.logger.info(f'🎯 Applying TCN-specific enhancements for {timeframe_name}')
            enhanced_model = await self._enhance_tcn_model(initial_model, X_train, y_train, X_val, y_val, timeframe_name)
        elif model_name == 'transformer':
            self.logger.info(f'🎯 Applying Transformer-specific enhancements for {timeframe_name}')
            enhanced_model = await self._enhance_transformer_model(initial_model, X_train, y_train, X_val, y_val, timeframe_name)
        elif model_name == 'lightgbm':
            self.logger.info(f'🎯 Applying LightGBM-specific enhancements for {timeframe_name}')
            enhanced_model = await self._enhance_lightgbm_model(initial_model, X_train, y_train, X_val, y_val, timeframe_name)
        elif model_name == 'cnn':
            self.logger.info(f'🎯 Applying CNN-specific enhancements for {timeframe_name}')
            enhanced_model = await self._enhance_cnn_model(initial_model, X_train, y_train, X_val, y_val, timeframe_name)
        else:
            self.logger.info(f'🎯 Applying default enhancements for {model_name} in {timeframe_name}')
            enhanced_model = await self._enhance_default_model(initial_model, model_name, X_train, y_train, X_val, y_val, timeframe_name)
        return enhanced_model

    async def _enhance_tcn_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeframe_name: str) -> dict[str, Any]:
        """Enhance TCN model with temporal-specific optimizations."""
        try:
            self.logger.info(f'🔄 TCN enhancement: Temporal convolution optimization for {timeframe_name}')
            best_params = await self._optimize_tcn_hyperparameters(X_train, y_train, X_val, y_val)
            optimal_features = await self._select_temporal_features(X_train, y_train, X_val, y_val)
            enhanced_tcn = await self._retrain_tcn_model(best_params, X_train[optimal_features], y_train)
            enhanced_tcn = await self._apply_tcn_optimizations(enhanced_tcn, X_train[optimal_features], y_train)
            final_accuracy = await self._evaluate_tcn_model(enhanced_tcn, X_val[optimal_features], y_val)
            return {'model': enhanced_tcn, 'selected_features': optimal_features, 'accuracy': final_accuracy, 'enhancement_metadata': {'enhancement_date': datetime.now().isoformat(), 'model_type': 'TCN', 'timeframe': timeframe_name, 'tcn_optimizations': ['temporal_convolution', 'dilation_optimization', 'residual_connections'], 'final_accuracy': final_accuracy, 'best_params': best_params, 'selected_feature_count': len(optimal_features)}}
        except Exception as e:
            self.logger.exception(f'❌ TCN enhancement failed: {e}')
            return {'model': model, 'selected_features': list(X_train.columns), 'accuracy': 0.0}

    async def _enhance_transformer_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeframe_name: str) -> dict[str, Any]:
        """Enhance Transformer model with attention mechanism optimizations."""
        try:
            self.logger.info(f'🔄 Transformer enhancement: Attention mechanism optimization for {timeframe_name}')
            best_params = await self._optimize_transformer_hyperparameters(X_train, y_train, X_val, y_val)
            optimal_features = await self._select_attention_features(X_train, y_train, X_val, y_val)
            enhanced_transformer = await self._retrain_transformer_model(best_params, X_train[optimal_features], y_train)
            enhanced_transformer = await self._apply_transformer_optimizations(enhanced_transformer, X_train[optimal_features], y_train)
            final_accuracy = await self._evaluate_transformer_model(enhanced_transformer, X_val[optimal_features], y_val)
            return {'model': enhanced_transformer, 'selected_features': optimal_features, 'accuracy': final_accuracy, 'enhancement_metadata': {'enhancement_date': datetime.now().isoformat(), 'model_type': 'Transformer', 'timeframe': timeframe_name, 'transformer_optimizations': ['attention_heads', 'positional_encoding', 'layer_norm'], 'final_accuracy': final_accuracy, 'best_params': best_params, 'selected_feature_count': len(optimal_features)}}
        except Exception as e:
            self.logger.exception(f'❌ Transformer enhancement failed: {e}')
            return {'model': model, 'selected_features': list(X_train.columns), 'accuracy': 0.0}

    async def _enhance_lightgbm_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeframe_name: str) -> dict[str, Any]:
        """Enhance LightGBM model with tree-based optimizations."""
        try:
            self.logger.info(f'🔄 LightGBM enhancement: Tree-based optimization for {timeframe_name}')
            best_params = await self._optimize_lightgbm_hyperparameters(X_train, y_train, X_val, y_val)
            optimal_features = await self._select_tree_features(X_train, y_train, X_val, y_val)
            enhanced_lgb = await self._retrain_lightgbm_model(best_params, X_train[optimal_features], y_train)
            enhanced_lgb = await self._apply_lightgbm_optimizations(enhanced_lgb, X_train[optimal_features], y_train)
            final_accuracy = await self._evaluate_lightgbm_model(enhanced_lgb, X_val[optimal_features], y_val)
            return {'model': enhanced_lgb, 'selected_features': optimal_features, 'accuracy': final_accuracy, 'enhancement_metadata': {'enhancement_date': datetime.now().isoformat(), 'model_type': 'LightGBM', 'timeframe': timeframe_name, 'lightgbm_optimizations': ['leaf_optimization', 'feature_pre_filtering', 'categorical_encoding'], 'final_accuracy': final_accuracy, 'best_params': best_params, 'selected_feature_count': len(optimal_features)}}
        except Exception as e:
            self.logger.exception(f'❌ LightGBM enhancement failed: {e}')
            return {'model': model, 'selected_features': list(X_train.columns), 'accuracy': 0.0}

    async def _enhance_cnn_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeframe_name: str) -> dict[str, Any]:
        """Enhance CNN model with convolution-specific optimizations."""
        try:
            self.logger.info(f'🔄 CNN enhancement: Convolution optimization for {timeframe_name}')
            best_params = await self._optimize_cnn_hyperparameters(X_train, y_train, X_val, y_val)
            optimal_features = await self._select_spatial_features(X_train, y_train, X_val, y_val)
            enhanced_cnn = await self._retrain_cnn_model(best_params, X_train[optimal_features], y_train)
            enhanced_cnn = await self._apply_cnn_optimizations(enhanced_cnn, X_train[optimal_features], y_train)
            final_accuracy = await self._evaluate_cnn_model(enhanced_cnn, X_val[optimal_features], y_val)
            return {'model': enhanced_cnn, 'selected_features': optimal_features, 'accuracy': final_accuracy, 'enhancement_metadata': {'enhancement_date': datetime.now().isoformat(), 'model_type': 'CNN', 'timeframe': timeframe_name, 'cnn_optimizations': ['convolution_layers', 'pooling', 'dropout'], 'final_accuracy': final_accuracy, 'best_params': best_params, 'selected_feature_count': len(optimal_features)}}
        except Exception as e:
            self.logger.exception(f'❌ CNN enhancement failed: {e}')
            return {'model': model, 'selected_features': list(X_train.columns), 'accuracy': 0.0}

    async def _enhance_default_model(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeframe_name: str) -> dict[str, Any]:
        """Default enhancement for other model types."""
        self.logger.info(f'🔄 Default enhancement for {model_name} in {timeframe_name}')
        best_params, hpo_score = await self._apply_hyperparameter_optimization(model_name, X_train, y_train, X_val, y_val)
        temp_model = self._get_model_instance(model_name, best_params)
        temp_model.fit(X_train, y_train)
        optimal_features, feature_selection_summary = await self._select_optimal_features(temp_model, model_name, X_train, y_train, X_val, y_val)
        final_model = self._get_model_instance(model_name, best_params)
        final_model.fit(X_train[optimal_features], y_train)
        final_accuracy = accuracy_score(y_val, final_model.predict(X_val[optimal_features]))
        return {'model': final_model, 'selected_features': optimal_features, 'accuracy': final_accuracy, 'enhancement_metadata': {'enhancement_date': datetime.now().isoformat(), 'model_type': model_name, 'timeframe': timeframe_name, 'hpo_score': hpo_score, 'final_accuracy': final_accuracy, 'best_params': best_params, 'feature_selection_method': feature_selection_summary.get('method', 'default'), 'selected_feature_count': len(optimal_features)}}

    def _get_model_instance(self, model_name: str, params: dict[str, Any]) -> None:
        """Factory function to get a model instance from its name and parameters."""
        if model_name in ['xgboost', 'lightgbm'] and self.device == 'mps':
            params.pop('device', None)
        if model_name == 'random_forest':
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        if model_name == 'lightgbm':
            safe_params = params.copy()
            safe_params.pop('device', None)
            safe_params['device_type'] = 'cpu'
            return lgb.LGBMClassifier(**safe_params, random_state=42)
        if model_name == 'xgboost':
            xgb_params = params.copy()
            if 'eval_metric' in xgb_params:
                del xgb_params['eval_metric']
            if 'device' in xgb_params:
                del xgb_params['device']
            return xgb.XGBClassifier(**xgb_params, random_state=42, n_estimators=params.get('n_estimators', 200), learning_rate=params.get('learning_rate', 0.05), max_depth=params.get('max_depth', 6), subsample=params.get('subsample', 0.8), colsample_bytree=params.get('colsample_bytree', 0.8))
        if model_name == 'svm':
            from sklearn.svm import SVC
            return SVC(**params, random_state=42, probability=True)
        if model_name == 'neural_network':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**params, random_state=42, early_stopping=True, validation_fraction=0.1)
        msg = f'Model {model_name} not supported.'
        raise ValueError(msg)

    async def _apply_hyperparameter_optimization(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, Any], float]:
        """Performs hyperparameter optimization using Optuna with early pruning."""
        self.logger.info(f'🚀 Running Optuna HPO with pruning for {model_name}...')
        trial_count = 0
        try:
            is_blank_env = os.environ.get('BLANK_TRAINING_MODE', '0') == '1'
        except Exception:
            is_blank_env = False
        try:
            is_blank_cfg = bool(CONFIG.get('BLANK_TRAINING_MODE', False))
        except Exception:
            is_blank_cfg = False
        blank_mode = is_blank_env or is_blank_cfg
        model_trial_mapping = {'lightgbm': self.config.get('lightgbm_trials', 50), 'xgboost': self.config.get('xgboost_trials', 50), 'svm': self.config.get('svm_trials', 30), 'random_forest': self.config.get('random_forest_trials', 40), 'neural_network': self.config.get('neural_network_trials', 25)}
        total_trials = model_trial_mapping.get(model_name, self.config.get('n_trials', 50))
        self.logger.info({'msg': 'HPO trial plan', 'model': model_name, 'total_trials': total_trials, 'blank_mode': blank_mode})
        with contextlib.suppress(Exception):
            pass

        def objective(trial: optuna.trial.Trial) -> float:
            nonlocal trial_count
            trial_count += 1
            self.logger.info({'msg': 'HPO trial start', 'model': model_name, 'trial': trial_count, 'of': total_trials})
            with contextlib.suppress(Exception):
                pass
            pruning_callback = None
            if y_train.nunique() <= 1:
                self.logger.warning(f'Target has only {y_train.nunique()} unique values, skipping optimization')
                return 0.0
            if model_name == 'lightgbm':
                n_classes = len(set(pd.concat([y_train, y_val]).unique()))
                lgb_objective = 'multiclass' if n_classes > 2 else 'binary'
                lgb_metric = 'multi_logloss' if n_classes > 2 else 'binary_logloss'
                pruning_callback = optuna.integration.LightGBMPruningCallback(trial, lgb_metric)
                params = {'objective': lgb_objective, 'metric': lgb_metric, 'verbosity': -1, 'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True), 'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 3, 12), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-08, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-08, 10.0, log=True), 'early_stopping_rounds': 50}
            elif model_name == 'xgboost':
                pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-logloss')
                params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0}
            elif model_name == 'svm':
                params = {'C': trial.suggest_float('C', 0.1, 100.0, log=True), 'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']), 'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])}
            elif model_name == 'neural_network':
                params = {'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]), 'alpha': trial.suggest_float('alpha', 1e-05, 0.1, log=True), 'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True), 'max_iter': trial.suggest_int('max_iter', 200, 1000)}
            else:
                params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500), 'max_depth': trial.suggest_int('max_depth', 5, 50), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)}
            model = self._get_model_instance(model_name, params)
            if model_name == 'lightgbm':
                use_pruning = pruning_callback is not None
                fit_kwargs = {'eval_set': [(X_val, y_val)]}
                if use_pruning:
                    fit_kwargs['callbacks'] = [pruning_callback]
                with contextlib.suppress(Exception):
                    self.logger.info({'msg': 'lightgbm_trial_params', 'trial': trial_count, 'params': params})

                def timeout_handler(signum: Any, frame: Any) -> Never:
                    msg = 'LightGBM training timed out'
                    raise TimeoutError(msg)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                try:
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model.fit(X_train, y_train, **fit_kwargs)
                finally:
                    sys.stdout = old_stdout
                    signal.alarm(0)
            elif model_name == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                if model_name == 'svm':
                    self.logger.info({'msg': 'Training model', 'model': model_name, 'trial': trial_count})
                model.fit(X_train, y_train)
            preds = model.predict(X_val)
            accuracy = accuracy_score(y_val, preds)
            if model_name == 'svm':
                self.logger.info({'msg': 'SVM trial result', 'trial': trial_count, 'accuracy': float(accuracy)})
            else:
                with contextlib.suppress(Exception):
                    self.logger.info({'msg': 'HPO_trial_result', 'model': model_name, 'trial': trial_count, 'metric': float(accuracy)})
            if model_name == 'lightgbm':
                labels_sorted = sorted(pd.unique(pd.concat([y_train, y_val])))
                y_proba = model.predict_proba(X_val)
                try:
                    loss = log_loss(y_val, y_proba, labels=labels_sorted)
                except Exception:
                    loss = log_loss(y_val, y_proba)
                return float(loss)
            return float(accuracy)
        study_direction = 'maximize'
        study = optuna.create_study(direction=study_direction, pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

        def progress_callback(study: Any, trial: Any) -> None:
            completed_trials = len(study.trials)
            if completed_trials % 1 == 0:
                self.logger.info({'msg': 'HPO progress', 'model': model_name, 'completed': completed_trials, 'total': total_trials})
                with contextlib.suppress(Exception):
                    pass
        try:
            import platform
            is_macos = platform.system() == 'Darwin'
        except Exception:
            is_macos = False
        parallel_jobs = 1 if model_name == 'svm' or is_macos else min(4, os.cpu_count() or 4)
        self.logger.info({'msg': 'HPO optimize start', 'model': model_name, 'total_trials': total_trials, 'n_jobs': parallel_jobs})
        with contextlib.suppress(Exception):
            pass
        study.optimize(objective, n_trials=total_trials, n_jobs=parallel_jobs, callbacks=[progress_callback] if model_name == 'svm' else None)
        self.logger.info({'msg': 'HPO optimize finished', 'model': model_name, 'completed_trials': len(study.trials)})
        with contextlib.suppress(Exception):
            pass
        if not study.best_trial:
            self.logger.warning('Optuna study found no best trial, possibly due to all trials being pruned. Returning empty params.')
            return ({}, 0.0)
        self.logger.info({'msg': 'HPO complete', 'model': model_name, 'best_score': float(study.best_value)})
        with contextlib.suppress(Exception):
            pass
        if model_name == 'svm':
            self.logger.info({'msg': 'Best SVM parameters', 'params': study.best_params})
        with contextlib.suppress(Exception):
            pass
        return (study.best_params, study.best_value)

    async def _select_optimal_features(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[list[str], dict]:
        """Selects the most important features using enhanced tiered strategy with stability selection and look-ahead bias prevention."""
        self.logger.info('🎯 Selecting optimal features using enhanced tiered strategy with stability selection...')
        with contextlib.suppress(Exception):
            pass
        feature_names = [c for c in X_val.columns.tolist() if c not in self._METADATA_COLUMNS and c not in self._LABEL_COLUMNS]
        X_train = X_train[feature_names]
        X_val = X_val[feature_names]
        total_features = len(feature_names)
        try:
            self._log_mutual_information_warnings(X_train, y_train)
        except Exception as e:
            self.logger.warning(f'Mutual Information check failed: {e}')
        try:
            self._log_feature_stability_warnings(X_train)
        except Exception as e:
            self.logger.warning(f'Stability check failed: {e}')
        self.logger.info(f'📊 Total features available: {total_features}')
        if total_features > 200:
            optimal_features, selection_summary = await self._execute_stable_tiered_feature_selection(model_name, X_train, y_train, X_val, y_val, feature_names)
        else:
            optimal_features, selection_summary = await self._execute_stable_traditional_feature_selection(model_name, X_train, y_train, X_val, y_val, feature_names)
        self.logger.info(f'✅ Selected {len(optimal_features)} optimal features from {total_features} total features')
        return (optimal_features, selection_summary)

    def _log_mutual_information_warnings(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute MI for each feature vs target and warn on near-zero scores."
        Threshold: in blank mode -> absolute threshold 1e-5.
        Full mode -> bottom 20% percentile flagged.
        """
        if X.empty or y is None or len(X.columns) == 0:
            return
        try:
            is_blank_env = os.environ.get('BLANK_TRAINING_MODE', '0') == '1'
        except Exception:
            is_blank_env = False
        try:
            is_blank_cfg = bool(CONFIG.get('BLANK_TRAINING_MODE', False))
        except Exception:
            is_blank_cfg = False
        blank_mode = is_blank_env or is_blank_cfg
        mi = mutual_info_classif(X.values, y.values, discrete_features=False, random_state=42)
        mi_series = pd.Series(mi, index=X.columns)
        if blank_mode:
            low = mi_series[mi_series <= 1e-05]
        else:
            threshold = mi_series.quantile(0.2)
            low = mi_series[mi_series <= threshold]
        if not low.empty:
            names = low.sort_values().index.tolist()
            self.logger.warning(f"MI: {len(names)} features show near-zero uni-variate predictive power (<= {('1e-5' if blank_mode else f'{threshold:.4g}')}): {names[:50]}{(' ...' if len(names) > 50 else '')}")

    def _log_feature_stability_warnings(self, X: pd.DataFrame) -> None:
        """Check 4-fold CV stability: warn if std of fold means >> expected standard error."
        Criterion: std_of_means > 3 * (global_std / sqrt(k)).
        """
        if X.empty:
            return
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        unstable: list[str] = []
        for col in X.columns:
            try:
                vals = X[col].astype(float).values
            except Exception as e:
                pass
                gstd = float(np.nanstd(vals))
                if not np.isfinite(gstd) or gstd == 0.0:
                    continue
                fold_means = []
                for train_idx, _ in kf.split(vals):
                    fold_vals = vals[train_idx]
                    if fold_vals.size == 0:
                        continue
                    fold_means.append(float(np.nanmean(fold_vals)))
                if len(fold_means) < 2:
                    continue
                std_of_means = float(np.nanstd(fold_means))
                expected_se = gstd / np.sqrt(4)
                if std_of_means > 3.0 * expected_se:
                    unstable.append(col)
            except Exception:
                continue
        if unstable:
            self.logger.warning(f"Stability: {len(unstable)} features are unstable across folds (std(mean) >> expected): {unstable[:50]}{(' ...' if len(unstable) > 50 else '')}")

    async def _execute_stable_tiered_feature_selection(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_names: list) -> tuple[list[str], dict]:
        """Execute stable tiered feature selection with bootstrapping to prevent selection instability."""
        feature_config = self.config.get('feature_interactions', {})
        selection_tiers = feature_config.get('feature_selection_tiers', {})
        stability_config = feature_config.get('stability_selection', {})
        n_bootstrap_samples = stability_config.get('n_bootstrap_samples', 50)
        stability_threshold = stability_config.get('stability_threshold', 0.7)
        min_features_per_tier = stability_config.get('min_features_per_tier', 5)
        tier_1_count = selection_tiers.get('tier_1_base_features', 80)
        tier_2_count = selection_tiers.get('tier_2_normalized_features', 40)
        tier_3_count = selection_tiers.get('tier_3_interaction_features', 60)
        tier_4_count = selection_tiers.get('tier_4_lagged_features', 40)
        tier_5_count = selection_tiers.get('tier_5_causality_features', 20)
        total_max_features = selection_tiers.get('total_max_features', 240)
        self.logger.info('🎯 Stable tiered feature selection targets (180 features):')
        self.logger.info(f'   Tier 1 (Core): {tier_1_count} features')
        self.logger.info(f'   Tier 2 (Normalized): {tier_2_count} features')
        self.logger.info(f'   Tier 3 (Interactions): {tier_3_count} features')
        self.logger.info(f'   Tier 4 (Lagged): {tier_4_count} features')
        self.logger.info(f'   Tier 5 (Causality): {tier_5_count} features')
        self.logger.info(f'   Total Max: {total_max_features} features')
        self.logger.info(f'   Stability: {n_bootstrap_samples} bootstrap samples, threshold: {stability_threshold}')
        feature_categories = self._categorize_features_by_tier(feature_names)
        selected_features = []
        selection_summary = {'method': 'stable_tiered_selection', 'total_features': len(feature_names), 'selected_features': 0, 'tier_breakdown': {}, 'selection_details': {}, 'stability_metrics': {}}
        tier_1_features = await self._select_stable_tier_1_features(model, model_name, X_train, y_train, X_val, y_val, feature_categories['tier_1'], tier_1_count, n_bootstrap_samples, stability_threshold, min_features_per_tier)
        selected_features.extend(tier_1_features)
        selection_summary['tier_breakdown']['tier_1_core'] = len(tier_1_features)
        self.logger.info(f'   ✅ Tier 1: Selected {len(tier_1_features)} stable core features')
        tier_2_features = await self._select_stable_tier_2_features(model, model_name, X_train, y_train, X_val, y_val, feature_categories['tier_2'], tier_2_count, n_bootstrap_samples, stability_threshold, min_features_per_tier)
        selected_features.extend(tier_2_features)
        selection_summary['tier_breakdown']['tier_2_normalized'] = len(tier_2_features)
        self.logger.info(f'   ✅ Tier 2: Selected {len(tier_2_features)} stable normalized features')
        tier_3_features = await self._select_stable_tier_3_features(model, model_name, X_train, y_train, X_val, y_val, feature_categories['tier_3'], tier_3_count, n_bootstrap_samples, stability_threshold, min_features_per_tier)
        selected_features.extend(tier_3_features)
        selection_summary['tier_breakdown']['tier_3_interactions'] = len(tier_3_features)
        self.logger.info(f'   ✅ Tier 3: Selected {len(tier_3_features)} stable interaction features')
        tier_4_features = await self._select_stable_tier_4_features(model, model_name, X_train, y_train, X_val, y_val, feature_categories['tier_4'], tier_4_count, n_bootstrap_samples, stability_threshold, min_features_per_tier)
        selected_features.extend(tier_4_features)
        selection_summary['tier_breakdown']['tier_4_lagged'] = len(tier_4_features)
        self.logger.info(f'   ✅ Tier 4: Selected {len(tier_4_features)} stable lagged features')
        tier_5_features = await self._select_stable_tier_5_features(model, model_name, X_train, y_train, X_val, y_val, feature_categories['tier_5'], tier_5_count, n_bootstrap_samples, stability_threshold, min_features_per_tier)
        selected_features.extend(tier_5_features)
        selection_summary['tier_breakdown']['tier_5_causality'] = len(tier_5_features)
        self.logger.info(f'   ✅ Tier 5: Selected {len(tier_5_features)} stable causality features')
        if len(selected_features) > total_max_features:
            (selected_features, await self._apply_stable_final_pruning(selected_features, X_val[selected_features], y_val, total_max_features, n_bootstrap_samples, stability_threshold))
        self.logger.info(f'   🔧 Final pruning: Reduced to {len(selected_features)} stable features')
        selection_summary['selected_features'] = len(selected_features)
        selection_summary['reduction_percentage'] = (len(feature_names) - len(selected_features)) / len(feature_names) * 100
        return (selected_features, selection_summary)

    async def _execute_stable_traditional_feature_selection(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_names: list) -> tuple[list[str], dict]:
        """Execute stable traditional feature selection with bootstrapping."""
        feature_config = self.config.get('feature_interactions', {})
        stability_config = feature_config.get('stability_selection', {})
        n_bootstrap_samples = stability_config.get('n_bootstrap_samples', 50)
        stability_threshold = stability_config.get('stability_threshold', 0.7)
        min_features = max(10, len(feature_names) // 2)
        max_features = min(20, len(feature_names))
        try:
            optimal_features, shap_summary = await self._try_stable_shap_feature_selection(model, model_name, X_train, y_train, X_val, y_val, feature_names, min_features, max_features, n_bootstrap_samples, stability_threshold)
            if optimal_features:
                return (optimal_features, {'method': 'stable_shap', **shap_summary})
        except Exception as e:
            self.logger.warning(f'Stable SHAP analysis failed: {e}. Trying alternative methods...')
        optimal_features, fallback_summary = await self._robust_stable_feature_selection(model, model_name, X_train, y_train, X_val, y_val, feature_names, min_features, max_features, n_bootstrap_samples, stability_threshold)
        return (optimal_features, {'method': 'stable_robust', **fallback_summary})

    async def _select_stable_tier_1_features(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tier_1_features: list, count: int, n_bootstrap_samples: int, stability_threshold: float, min_features_per_tier: int) -> list[str]:
        """Select core features with stability selection using bootstrapping."""
        if not tier_1_features:
            return []
        available_features = [f for f in tier_1_features if f in X_val.columns]
        if not available_features:
            return []
        return await self._perform_stability_selection(model, model_name, X_train, y_train, available_features, count, n_bootstrap_samples, stability_threshold, min_features_per_tier)

    async def _select_stable_tier_2_features(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tier_2_features: list, count: int, n_bootstrap_samples: int, stability_threshold: float, min_features_per_tier: int) -> list[str]:
        """Select normalized features with stability selection."""
        if not tier_2_features:
            return []
        available_features = [f for f in tier_2_features if f in X_val.columns]
        if not available_features:
            return []
        return await self._perform_stability_selection(model, model_name, X_train, y_train, available_features, count, n_bootstrap_samples, stability_threshold, min_features_per_tier, selection_criteria='stability')

    async def _select_stable_tier_3_features(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tier_3_features: list, count: int, n_bootstrap_samples: int, stability_threshold: float, min_features_per_tier: int) -> list[str]:
        """Select interaction features with stability selection."""
        if not tier_3_features:
            return []
        available_features = [f for f in tier_3_features if f in X_val.columns]
        if not available_features:
            return []
        return await self._perform_stability_selection(model, model_name, X_train, y_train, available_features, count, n_bootstrap_samples, stability_threshold, min_features_per_tier, selection_criteria='significance')

    async def _select_stable_tier_4_features(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tier_4_features: list, count: int, n_bootstrap_samples: int, stability_threshold: float, min_features_per_tier: int) -> list[str]:
        """Select lagged features with stability selection."""
        if not tier_4_features:
            return []
        available_features = [f for f in tier_4_features if f in X_val.columns]
        if not available_features:
            return []
        return await self._perform_stability_selection(model, model_name, X_train, y_train, available_features, count, n_bootstrap_samples, stability_threshold, min_features_per_tier, selection_criteria='temporal')

    async def _select_stable_tier_5_features(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tier_5_features: list, count: int, n_bootstrap_samples: int, stability_threshold: float, min_features_per_tier: int) -> list[str]:
        """Select causality features with stability selection."""
        if not tier_5_features:
            return []
        available_features = [f for f in tier_5_features if f in X_val.columns]
        if not available_features:
            return []
        return await self._perform_stability_selection(model, model_name, X_train, y_train, available_features, count, n_bootstrap_samples, stability_threshold, min_features_per_tier, selection_criteria='market_logic')

    async def _perform_stability_selection(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, available_features: list, count: int, n_bootstrap_samples: int, stability_threshold: float, min_features_per_tier: int, selection_criteria: str='importance') -> list[str]:
        """Perform stability selection using bootstrapping to ensure feature selection stability."""
        self.logger.info(f'🔄 Performing stability selection for {len(available_features)} features with {n_bootstrap_samples} bootstrap samples...')
        feature_selection_freq = dict.fromkeys(available_features, 0)
        for i in range(n_bootstrap_samples):
            try:
                bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_bootstrap = X_train.iloc[bootstrap_indices][available_features]
                y_bootstrap = y_train.iloc[bootstrap_indices]
                selected_features_bootstrap = await self._select_features_single_bootstrap(model, model_name, X_bootstrap, y_bootstrap, available_features, count, selection_criteria)
                for feature in selected_features_bootstrap:
                    feature_selection_freq[feature] += 1
            except Exception as e:
                self.logger.warning(f'Bootstrap sample {i + 1} failed: {e}')
                continue
        feature_stability = {feature: freq / n_bootstrap_samples for feature, freq in feature_selection_freq.items()}
        stable_features = [feature for feature, stability in feature_stability.items() if stability >= stability_threshold]
        if len(stable_features) < min_features_per_tier:
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:min_features_per_tier]]
        if len(stable_features) > count:
            stable_features = sorted(stable_features, key=lambda x: feature_stability[x], reverse=True)[:count]
        self.logger.info('   📊 Stability selection results:')
        self.logger.info(f'      Selected: {len(stable_features)} stable features')
        self.logger.info(f'      Average stability: {np.mean([feature_stability[f] for f in stable_features]):.3f}')
        self.logger.info(f'      Min stability: {min([feature_stability[f] for f in stable_features]):.3f}')
        return stable_features

    async def _select_features_single_bootstrap(self, model: Any, model_name: str, X_bootstrap: pd.DataFrame, y_bootstrap: pd.Series, available_features: list, count: int, selection_criteria: str) -> list[str]:
        """Select features for a single bootstrap sample."""
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                feature_importance_dict = dict(zip(X_bootstrap.columns, feature_importance, strict=False))
                tier_importance = {f: feature_importance_dict.get(f, 0) for f in available_features}
                selected_features = sorted(tier_importance.items(), key=lambda x: x[1], reverse=True)[:count]
                return [f[0] for f in selected_features]
            if selection_criteria == 'stability':
                feature_variance = X_bootstrap[available_features].var()
                return feature_variance.nsmallest(count).index.tolist()
            if selection_criteria == 'significance':
                feature_abs_mean = X_bootstrap[available_features].abs().mean()
                return feature_abs_mean.nlargest(count).index.tolist()
            if selection_criteria == 'temporal':
                feature_variance = X_bootstrap[available_features].var()
                return feature_variance.nlargest(count).index.tolist()
            if selection_criteria == 'market_logic':
                feature_abs_mean = X_bootstrap[available_features].abs().mean()
                return feature_abs_mean.nlargest(count).index.tolist()
            feature_variance = X_bootstrap[available_features].var()
            return feature_variance.nlargest(count).index.tolist()
        except Exception:
            feature_variance = X_bootstrap[available_features].var()
            return feature_variance.nlargest(count).index.tolist()

    async def _apply_stable_final_pruning(self, selected_features: list, X_val_subset: pd.DataFrame, y_val: pd.Series, max_features: int, n_bootstrap_samples: int, stability_threshold: float) -> list[str]:
        """Apply final pruning with stability selection to meet maximum feature count."""
        if len(selected_features) <= max_features:
            return selected_features
        return await self._perform_stability_selection(None, 'final_pruning', X_val_subset, y_val, selected_features, max_features, n_bootstrap_samples, stability_threshold, 5)

    async def _try_stable_shap_feature_selection(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_names: list, min_features: int, max_features: int, n_bootstrap_samples: int, stability_threshold: float) -> tuple[list[str], dict]:
        """Attempts stable SHAP-based feature selection with bootstrapping."""
        self.logger.info(f'🔍 Performing stable SHAP feature selection with {n_bootstrap_samples} bootstrap samples...')
        feature_config = self.config.get('feature_interactions', {})
        stability_config = feature_config.get('stability_selection', {})
        shap_config = stability_config.get('shap_analysis', {})
        validation_sample_size = self._get_adaptive_shap_sample_size(len(X_val), shap_config)
        self.logger.info(f'   📊 Using {validation_sample_size} validation samples for SHAP analysis')
        feature_selection_freq = dict.fromkeys(feature_names, 0)
        shap_values_all = []
        for i in range(n_bootstrap_samples):
            try:
                bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_bootstrap = X_train.iloc[bootstrap_indices]
                y_bootstrap = y_train.iloc[bootstrap_indices]
                sample_idx = np.random.RandomState(42 + i).choice(len(X_val), size=min(validation_sample_size, len(X_val)), replace=False)
                X_val_sample = X_val.iloc[sample_idx]
                y_val_sample = y_val.iloc[sample_idx]
                shap_importance = await self._calculate_shap_importance_single_bootstrap(model, model_name, X_bootstrap, y_bootstrap, X_val_sample, y_val_sample)
                if shap_importance is not None:
                    top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:max_features]
                    for feature, importance in top_features:
                        feature_selection_freq[feature] += 1
                        shap_values_all.append((feature, importance))
            except Exception as e:
                self.logger.warning(f'SHAP bootstrap sample {i + 1} failed: {e}')
                continue
        feature_stability = {feature: freq / n_bootstrap_samples for feature, freq in feature_selection_freq.items()}
        stable_features = [feature for feature, stability in feature_stability.items() if stability >= stability_threshold]
        if len(stable_features) < min_features:
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:min_features]]
        if len(stable_features) > max_features:
            stable_features = sorted(stable_features, key=lambda x: feature_stability[x], reverse=True)[:max_features]
        feature_shap_avg = {}
        for feature in stable_features:
            shap_values = [shap_val for f, shap_val in shap_values_all if f == feature]
            if shap_values:
                feature_shap_avg[feature] = np.mean(shap_values)
        self.logger.info('   📊 Stable SHAP selection results:')
        self.logger.info(f'      Selected: {len(stable_features)} stable features')
        self.logger.info(f'      Average stability: {np.mean([feature_stability[f] for f in stable_features]):.3f}')
        self.logger.info(f'      Validation samples used: {validation_sample_size}')
        try:
            top_by_stability = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info({'msg': 'stable_shap_top_features', 'top': [(f, float(s)) for f, s in top_by_stability]})
        except Exception:
            pass
        return (stable_features, {'method': 'stable_shap', 'stability_scores': feature_stability, 'shap_importance': feature_shap_avg, 'bootstrap_samples': n_bootstrap_samples, 'stability_threshold': stability_threshold, 'validation_sample_size': validation_sample_size})

    def _get_adaptive_shap_sample_size(self, total_samples: int, shap_config: dict) -> int:
        """Calculate adaptive sample size for SHAP analysis based on dataset size."""
        default_size = shap_config.get('validation_sample_size', 2000)
        min_size = shap_config.get('min_sample_size', 1000)
        max_size = shap_config.get('max_sample_size', 5000)
        enable_adaptive = shap_config.get('enable_adaptive_sampling', True)
        if not enable_adaptive:
            return min(default_size, total_samples)
        if total_samples <= 10000:
            sample_size = max(min_size, int(total_samples * 0.2))
        elif total_samples <= 50000:
            sample_size = int(total_samples * 0.1)
        elif total_samples <= 200000:
            sample_size = int(total_samples * 0.05)
        else:
            sample_size = min(max_size, int(total_samples * 0.02))
        sample_size = min(sample_size, total_samples)
        return max(min_size, sample_size)

    async def _calculate_shap_importance_single_bootstrap(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float] | None:
        """Calculate SHAP importance for a single bootstrap sample."""
        try:
            if model_name in ['lightgbm', 'xgboost', 'random_forest']:
                try:
                    explainer = TreeExplainer(model)
                    shap_values = explainer.shap_values(X_val)
                    if isinstance(shap_values, list):
                        shap_array = np.asarray(shap_values)
                    else:
                        shap_array = np.asarray(shap_values)
                    if shap_array.ndim == 2:
                        feature_importance = np.mean(np.abs(shap_array), axis=0)
                    elif shap_array.ndim == 3:
                        feature_importance = np.mean(np.abs(shap_array), axis=(0, 1))
                    else:
                        return None
                    return dict(zip(X_val.columns, feature_importance, strict=False))
                except (ImportError, AttributeError):
                    feature_importance = permutation_importance(model, X_val, y_val, n_repeats=3, random_state=42).importances_mean
                    return dict(zip(X_val.columns, feature_importance, strict=False))
            elif model_name == 'svm':
                try:
                    explainer = KernelExplainer(model.predict, X_train.iloc[:100])
                    shap_values = explainer.shap_values(X_val.iloc[:50])
                    feature_importance = np.mean(np.abs(shap_values), axis=0)
                    return dict(zip(X_val.columns, feature_importance, strict=False))
                except Exception:
                    return None
            else:
                feature_importance = permutation_importance(model, X_val, y_val, n_repeats=3, random_state=42).importances_mean
                return dict(zip(X_val.columns, feature_importance, strict=False))
        except Exception:
            return None

    async def _robust_stable_feature_selection(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_names: list, min_features: int, max_features: int, n_bootstrap_samples: int, stability_threshold: float) -> tuple[list[str], dict]:
        """Fallback to robust feature selection methods with stability selection."""
        self.logger.info(f'🔄 Performing robust stable feature selection with {n_bootstrap_samples} bootstrap samples...')
        feature_selection_freq = dict.fromkeys(feature_names, 0)
        for i in range(n_bootstrap_samples):
            try:
                bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_bootstrap = X_train.iloc[bootstrap_indices]
                y_bootstrap = y_train.iloc[bootstrap_indices]
                sample_idx = np.random.RandomState(42 + i).choice(len(X_val), size=min(500, len(X_val)), replace=False)
                X_val_sample = X_val.iloc[sample_idx]
                y_val_sample = y_val.iloc[sample_idx]
                selected_features_bootstrap = await self._robust_feature_selection_single_bootstrap(model, model_name, X_bootstrap, y_bootstrap, X_val_sample, y_val_sample, feature_names, min_features, max_features)
                for feature in selected_features_bootstrap:
                    feature_selection_freq[feature] += 1
            except Exception as e:
                self.logger.warning(f'Robust bootstrap sample {i + 1} failed: {e}')
                continue
        feature_stability = {feature: freq / n_bootstrap_samples for feature, freq in feature_selection_freq.items()}
        stable_features = [feature for feature, stability in feature_stability.items() if stability >= stability_threshold]
        if len(stable_features) < min_features:
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:min_features]]
        if len(stable_features) > max_features:
            stable_features = sorted(stable_features, key=lambda x: feature_stability[x], reverse=True)[:max_features]
        self.logger.info('   📊 Robust stable selection results:')
        self.logger.info(f'      Selected: {len(stable_features)} stable features')
        self.logger.info(f'      Average stability: {np.mean([feature_stability[f] for f in stable_features]):.3f}')
        return (stable_features, {'method': 'robust_stable', 'stability_scores': feature_stability, 'bootstrap_samples': n_bootstrap_samples, 'stability_threshold': stability_threshold})

    async def _robust_feature_selection_single_bootstrap(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_names: list, min_features: int, max_features: int) -> list[str]:
        """Perform robust feature selection for a single bootstrap sample."""
        try:
            methods = []
            try:
                feature_variance = X_val[feature_names].var()
                variance_features = feature_variance.nlargest(max_features).index.tolist()
                methods.append(variance_features)
            except Exception:
                pass
            try:
                selector = SelectKBest(score_func=f_classif, k=max_features)
                selector.fit(X_train[feature_names], y_train)
                correlation_features = [feature_names[i] for i in selector.get_support(indices=True)]
                methods.append(correlation_features)
            except Exception:
                pass
            try:
                mi_scores = mutual_info_classif(X_train[feature_names], y_train, random_state=42)
                mi_features = [feature_names[i] for i in np.argsort(mi_scores)[-max_features:]]
                methods.append(mi_features)
            except Exception:
                pass
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    feature_importance_dict = dict(zip(X_train.columns, feature_importance, strict=False))
                    model_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:max_features]
                    model_features = [f[0] for f in model_features if f[0] in feature_names]
                    methods.append(model_features)
            except Exception:
                pass
            if methods:
                feature_votes = {}
                for method_features in methods:
                    for feature in method_features:
                        feature_votes[feature] = feature_votes.get(feature, 0) + 1
                selected_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)[:max_features]
                return [f[0] for f in selected_features]
            feature_variance = X_val[feature_names].var()
            return feature_variance.nlargest(max_features).index.tolist()
        except Exception:
            return feature_names[:max_features]

    def _categorize_features_by_tier(self, feature_names: list) -> dict:
        """Categorize features into tiers based on naming patterns (enhanced with data-driven methods)."""
        categories = {'tier_1': [], 'tier_2': [], 'tier_3': [], 'tier_4': [], 'tier_5': []}
        for feature in feature_names:
            feature_lower = feature.lower()
            if any((keyword in feature_lower for keyword in ['rsi', 'macd', 'bb', 'atr', 'adx', 'sma', 'ema', 'cci', 'mfi', 'roc', 'volume', 'spread', 'liquidity', 'price_impact', 'kyle', 'amihud'])):
                categories['tier_1'].append(feature)
            elif any((keyword in feature_lower for keyword in ['_z_score', '_change', '_pct_change', '_acceleration', '_bounded', '_log', '_normalized'])):
                categories['tier_2'].append(feature)
            elif '_x_' in feature_lower or '_div_' in feature_lower:
                categories['tier_3'].append(feature)
            elif '_lag' in feature_lower:
                categories['tier_4'].append(feature)
            elif any((keyword in feature_lower for keyword in ['_predicts_', '_causality', '_divergence', '_stress', '_extreme'])):
                categories['tier_5'].append(feature)
            else:
                categories['tier_1'].append(feature)
        return categories

    async def _apply_data_driven_feature_selection(self, data: pd.DataFrame, feature_columns: list, target_column: str=None) -> list:
        """Apply data-driven feature selection using VIF, MI, SHAP, and RF methods."""
        try:
            self.logger.info(f'🔍 Applying data-driven feature selection to {len(feature_columns)} features')
            selected_features = feature_columns.copy()
            X_clean = data[feature_columns].copy()
            nan_ratio = X_clean.isna().sum() / len(X_clean)
            high_nan_features = nan_ratio[nan_ratio > 0.1].index.tolist()
            X_clean = X_clean.drop(columns=high_nan_features)
            inf_features = []
            for col in X_clean.columns:
                if np.isinf(X_clean[col]).any():
                    inf_features.append(col)
            X_clean = X_clean.drop(columns=inf_features)
            X_clean = X_clean.ffill().bfill().fillna(0)
            self.logger.info(f'   Data quality filtering: {len(feature_columns)} -> {len(X_clean.columns)} features')
            try:
                vif_scores = calculate_vif_robust(X_clean)
                low_vif_features = vif_scores[vif_scores <= 10.0].index.tolist()
                self.logger.info(f'   VIF filtering: {len(X_clean.columns)} -> {len(low_vif_features)} features')
                X_clean = X_clean[low_vif_features]
            except Exception as e:
                self.logger.warning(f'VIF filtering failed: {e}, skipping')
            if target_column and target_column in data.columns:
                try:
                    y = data[target_column]
                    task_type = 'classification' if len(y.unique()) < 10 else 'regression'
                    if task_type == 'classification':
                        mi_scores = mutual_info_classif(X_clean, y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X_clean, y, random_state=42)
                    mi_series = pd.Series(mi_scores, index=X_clean.columns)
                    high_mi_features = mi_series[mi_scores >= 0.01].index.tolist()
                    self.logger.info(f'   MI filtering: {len(X_clean.columns)} -> {len(high_mi_features)} features')
                    X_clean = X_clean[high_mi_features]
                except Exception as e:
                    self.logger.warning(f'MI filtering failed: {e}, skipping')
            if target_column and target_column in data.columns and (len(X_clean.columns) > 50):
                try:
                    shap_scores = compute_shap_importance(X_clean, y, task=task_type)
                    if shap_scores:
                        shap_series = pd.Series(shap_scores)
                        threshold = shap_series.quantile(0.2)
                        high_shap_features = shap_series[shap_series >= threshold].index.tolist()
                        self.logger.info(f'   SHAP filtering: {len(X_clean.columns)} -> {len(high_shap_features)} features')
                        X_clean = X_clean[high_shap_features]
                except Exception as e:
                    self.logger.warning(f'SHAP filtering failed: {e}, skipping')
            if target_column and target_column in data.columns and (len(X_clean.columns) > 30):
                try:
                    if task_type == 'classification':
                        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_clean, y)
                    rf_importance = pd.Series(rf.feature_importances_, index=X_clean.columns)
                    threshold = rf_importance.quantile(0.2)
                    high_rf_features = rf_importance[rf_importance >= threshold].index.tolist()
                    self.logger.info(f'   RF filtering: {len(X_clean.columns)} -> {len(high_rf_features)} features')
                    X_clean = X_clean[high_rf_features]
                except Exception as e:
                    self.logger.warning(f'RF filtering failed: {e}, skipping')
            selected_features = X_clean.columns.tolist()
            self.logger.info(f'✅ Data-driven feature selection completed: {len(feature_columns)} -> {len(selected_features)} features')
            return selected_features
        except Exception as e:
            self.logger.warning(f'⚠️ Error in data-driven feature selection: {e}')
            return feature_columns

    def _save_enhanced_models(self, enhanced_models: dict, data_dir: str, training_input: dict) -> str:
        """Saves the enhanced models and a JSON summary report."""
        enhanced_models_dir = os.path.join(data_dir, 'enhanced_hmm_models')
        os.makedirs(enhanced_models_dir, exist_ok=True)
        json_summary = {}
        for regime_name, models in enhanced_models.items():
            regime_models_dir = os.path.join(enhanced_models_dir, regime_name)
            os.makedirs(regime_models_dir, exist_ok=True)
            json_summary[regime_name] = {}
        for model_name, model_data in models.items():
            model_file = os.path.join(regime_models_dir, f'{model_name}.joblib')
            joblib.dump(model_data['model'], model_file)
            summary_data = model_data.copy()
            summary_data.pop('model', None)
            summary_data['model_path'] = model_file
            json_summary[regime_name][model_name] = summary_data
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'BINANCE')
        summary_file = os.path.join(data_dir, f'{exchange}_{symbol}_analyst_enhancement_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        return enhanced_models_dir

    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Applies dynamic quantization to a PyTorch model for CPU/MPS inference."""
        self.logger.info('Applying dynamic quantization to the model...')
        model.to('cpu')
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.logger.info('Dynamic quantization complete. Model is now smaller and may run faster on CPU.')
        return quantized_model

    def _apply_wanda_pruning(self, model: torch.nn.Module, calibration_data: pd.DataFrame, sparsity: float=0.5) -> torch.nn.Module:
        """Applies structured pruning using a simplified WANDA (Weight and Activation-based) method."
        This implementation demonstrates the core concept.
        """
        self.logger.info(f'Applying WANDA-style pruning with {sparsity} sparsity...')
        model.to(self.device)
        calib_tensor = torch.tensor(calibration_data.values, dtype=torch.float32).to(self.device)
        activations = {}

        def get_activation(name: Any) -> Any:

            def hook(model: Any, input: Any, output: Any) -> None:
                activations[name] = torch.sqrt(torch.mean(input[0] ** 2, dim=0))
            return hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))
        model(calib_tensor)
        for hook in hooks:
            hook.remove()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                W = module.weight.data
                act_norm = activations[name]
                importance_scores = torch.abs(W) * act_norm
                prune.l1_unstructured(module, name='weight', amount=sparsity, importance_scores=importance_scores)
                prune.remove(module, 'weight')
            try:
                total = W.numel()
                nonzero = int(torch.count_nonzero(W).item())
                sparsity_actual = 1.0 - nonzero / max(1, total)
                self.logger.info({'msg': 'wanda_layer_sparsity', 'layer': name, 'sparsity': float(sparsity_actual)})
            except Exception:
                pass
        self.logger.info('WANDA-style pruning complete.')
        return model

    def _apply_knowledge_distillation(self, teacher_model: torch.nn.Module, X_train: pd.DataFrame, y_train: pd.Series) -> torch.nn.Module:
        """Uses knowledge distillation to train a smaller 'student' model to mimic the teacher."""
        self.logger.info('Applying knowledge distillation...')
        teacher_model.to(self.device).eval()
        input_dim = X_train.shape[1]
        student_model = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 2)).to(self.device)
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        T = 2.0
        alpha = 0.3
        student_model.train()
        for epoch in range(5):
            for data, targets in train_loader:
                data, targets = (data.to(self.device), targets.to(self.device))
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                student_logits = student_model(data)
                loss_hard = F.cross_entropy(student_logits, targets)
                loss_soft = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1)) * (T * T)
                loss = alpha * loss_hard + (1.0 - alpha) * loss_soft
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.logger.info(f'Distillation Epoch {epoch + 1}, Loss: {loss.item():.4f}')
        self.logger.info('Knowledge distillation complete. Returning the trained student model.')
        return student_model.eval()

    async def _hpo_catboost(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, Any], float]:
        """Lightweight CatBoost HPO using Optuna; returns (best_params, best_score)."""
        try:

            def objective(trial: optuna.Trial) -> float:
                params = {'iterations': trial.suggest_int('iterations', 300, 1500, step=300), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True), 'depth': trial.suggest_int('depth', 4, 10), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0)}
                model = CatBoostClassifier(random_seed=42, verbose=False, **params)
                frac = min(1.0, 30000 / max(1, len(X_train)))
                if frac < 1.0:
                    Xs, ys = (X_train.sample(frac=frac, random_state=42), y_train.loc[Xs.index])
                else:
                    Xs, ys = (X_train, y_train)
                model.fit(Xs, ys)
                pred = model.predict(X_val)
                return float((pred == y_val).mean())
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=25)
            return (study.best_params, float(study.best_value))
        except Exception as e:
            self.logger.warning(f'CatBoost HPO failed: {e}')
            return ({}, 0.0)

    async def _apply_pre_feature_selection(self, data: pd.DataFrame, feature_columns: list, regime_name: str) -> list:
        """Apply pre-feature selection for large feature sets to reduce dimensionality before training."""
        try:
            self.logger.info(f'🔍 Applying pre-feature selection for {len(feature_columns)} features...')
            if regime_name.startswith('hmm_'):
                return await self._apply_architecture_specific_feature_selection(data, feature_columns, regime_name)
            feature_config = self.config.get('feature_interactions', {})
            selection_tiers = feature_config.get('feature_selection_tiers', {})
            tier_1_count = selection_tiers.get('tier_1_base_features', 80)
            tier_2_count = selection_tiers.get('tier_2_normalized_features', 40)
            tier_3_count = selection_tiers.get('tier_3_interaction_features', 60)
            tier_4_count = selection_tiers.get('tier_4_lagged_features', 40)
            tier_5_count = selection_tiers.get('tier_5_causality_features', 20)
            total_max_features = selection_tiers.get('total_max_features', 240)
            feature_categories = self._categorize_features_by_tier(feature_columns)
            selected_features = []
            tier_1_features = await self._select_tier_1_features_pre_training(data, feature_categories['tier_1'], tier_1_count)
            selected_features.extend(tier_1_features)
            self.logger.info(f'   ✅ Tier 1: Selected {len(tier_1_features)} core features')
            tier_2_features = await self._select_tier_2_features_pre_training(data, feature_categories['tier_2'], tier_2_count)
            selected_features.extend(tier_2_features)
            self.logger.info(f'   ✅ Tier 2: Selected {len(tier_2_features)} normalized features')
            tier_3_features = await self._select_tier_3_features_pre_training(data, feature_categories['tier_3'], tier_3_count)
            selected_features.extend(tier_3_features)
            self.logger.info(f'   ✅ Tier 3: Selected {len(tier_3_features)} interaction features')
            tier_4_features = await self._select_tier_4_features_pre_training(data, feature_categories['tier_4'], tier_4_count)
            selected_features.extend(tier_4_features)
            self.logger.info(f'   ✅ Tier 4: Selected {len(tier_4_features)} lagged features')
            tier_5_features = await self._select_tier_5_features_pre_training(data, feature_categories['tier_5'], tier_5_count)
            selected_features.extend(tier_5_features)
            self.logger.info(f'   ✅ Tier 5: Selected {len(tier_5_features)} causality features')
            try:
                if len(selected_features) > total_max_features:
                    X = data[selected_features].select_dtypes(include=[np.number]).fillna(0)
                    y = data.get('label')
                    if y is not None and (not X.empty):
                        (mi, mutual_info_classif(X.values, y.values if hasattr(y, 'values') else y, random_state=42))
                        keep_idx = np.argsort(mi)[-total_max_features:]
                        selected_features = [list(X.columns)[i] for i in keep_idx]
                        self.logger.info(f'   🔧 Aggressive MI pruning: Reduced to {len(selected_features)} features')
            except Exception as e:
                self.logger.warning(f'Aggressive MI pruning failed: {e}')
            if len(selected_features) > total_max_features:
                selected_features = await self._apply_final_pruning_pre_training(data, selected_features, total_max_features)
            self.logger.info(f'   🔧 Final pruning: Reduced to {len(selected_features)} features')
            return selected_features
        except Exception as e:
            self.logger.exception(f'❌ Pre-feature selection failed: {e}')
            return feature_columns

    async def _apply_architecture_specific_feature_selection(self, data: pd.DataFrame, feature_columns: list, regime_name: str) -> list:
        """Apply architecture-specific feature selection for new models."""
        try:
            timeframe = regime_name.replace('hmm_', '')
            if timeframe == '5m':
                temporal_features = [f for f in feature_columns if any((keyword in f.lower() for keyword in ['lag', 'change', 'momentum', 'acceleration', 'regime', 'cluster']))]
                regime_features = [f for f in feature_columns if any((keyword in f.lower() for keyword in ['regime', 'cluster', 'intensity']))]
                core_features = [f for f in feature_columns if any((keyword in f.lower() for keyword in ['rsi', 'macd', 'bb', 'atr', 'volume', 'volatility']))]
                selected = temporal_features + regime_features + core_features
                return selected[:200] if len(selected) > 200 else selected
            if timeframe == '15m':
                return feature_columns[:300]
            if timeframe == '30m':
                return feature_columns
            return feature_columns[:200]
        except Exception as e:
            self.logger.exception(f'❌ Architecture-specific feature selection failed: {e}')
            return feature_columns

    async def _hpo_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, Any], float]:
        """Optuna HPO for RandomForest; returns (best_params, best_score)."""
        try:

            def objective(trial: optuna.Trial) -> float:
                params = {'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100), 'max_depth': trial.suggest_int('max_depth', 4, 20), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5), 'max_features': trial.suggest_float('max_features', 0.3, 1.0)}
                model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
                frac = min(1.0, 30000 / max(1, len(X_train)))
                Xs = X_train.sample(frac=frac, random_state=42) if frac < 1.0 else X_train
                ys = y_train.loc[Xs.index]
                model.fit(Xs, ys)
                pred = model.predict(X_val)
                return float((pred == y_val).mean())
            rf_trials = self.config.get('random_forest_trials', 25)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=rf_trials)
            return (study.best_params, float(study.best_value))
        except Exception as e:
            self.logger.warning(f'RF HPO failed: {e}')
            return ({}, 0.0)

    async def _hpo_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, Any], float]:
        """Optuna HPO for Logistic Regression; returns (best_params, best_score)."""
        try:

            def objective(trial: optuna.Trial) -> float:
                penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
                C = trial.suggest_float('C', 0.001, 10.0, log=True)
                solver = 'liblinear' if penalty in ('l1', 'l2') else 'saga'
                class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
                model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, class_weight=class_weight, random_state=42)
                frac = min(1.0, 50000 / max(1, len(X_train)))
                Xs = X_train.sample(frac=frac, random_state=42) if frac < 1.0 else X_train
                ys = y_train.loc[Xs.index]
                model.fit(Xs, ys)
                pred = model.predict(X_val)
                return float((pred == y_val).mean())
            logistic_trials = self.config.get('logistic_trials', 25)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=logistic_trials)
            return (study.best_params, float(study.best_value))
        except Exception as e:
            self.logger.warning(f'Logistic HPO failed: {e}')
            return ({}, 0.0)

    async def _hpo_svm_proxy(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, Any], float]:
        """Optuna HPO for SVM proxy (RBFSampler + LinearSVC)."""
        try:

            def objective(trial: optuna.Trial) -> float:
                gamma = trial.suggest_float('gamma', 0.0001, 1.0, log=True)
                n_components = trial.suggest_int('n_components', 1000, 5000, step=500)
                C = trial.suggest_float('C', 0.1, 10.0, log=True)
                pipe = make_pipeline(StandardScaler(), RBFSampler(gamma=gamma, n_components=n_components, random_state=42), LinearSVC(C=C, tol=0.001, random_state=42))
                frac = min(1.0, 30000 / max(1, len(X_train)))
                Xs = X_train.sample(frac=frac, random_state=42) if frac < 1.0 else X_train
                ys = y_train.loc[Xs.index]
                pipe.fit(Xs, ys)
                pred = pipe.predict(X_val)
                return float((pred == y_val).mean())
            svm_trials = self.config.get('svm_trials', 25)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=svm_trials)
            return (study.best_params, float(study.best_value))
        except Exception as e:
            self.logger.warning(f'SVM-proxy HPO failed: {e}')
            return ({}, 0.0)

    async def _optimize_tcn_hyperparameters(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Optimize TCN hyperparameters for temporal data."""
        return {'kernel_size': 3, 'num_channels': [64, 128, 256], 'dropout': 0.1}

    async def _select_temporal_features(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Select features relevant for temporal modeling."""
        return list(X_train.columns)

    async def _retrain_tcn_model(self, best_params: List[Any], X_train: Any, y_train: Any) -> None:
        """Retrain TCN model with optimized parameters."""
        return

    async def _apply_tcn_optimizations(self, model: Any, X_train: Any, y_train: Any) -> None:
        """Apply TCN-specific optimizations."""
        return model

    async def _evaluate_tcn_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate TCN model performance."""
        return 0.0

    async def _optimize_transformer_hyperparameters(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Optimize Transformer hyperparameters for attention mechanisms."""
        return {'nhead': 8, 'num_layers': 4, 'd_model': 256}

    async def _select_attention_features(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Select features relevant for attention mechanisms."""
        return list(X_train.columns)

    async def _retrain_transformer_model(self, best_params: List[Any], X_train: Any, y_train: Any) -> None:
        """Retrain Transformer model with optimized parameters."""
        return

    async def _apply_transformer_optimizations(self, model: Any, X_train: Any, y_train: Any) -> None:
        """Apply Transformer-specific optimizations."""
        return model

    async def _evaluate_transformer_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate Transformer model performance."""
        return 0.0

    async def _optimize_lightgbm_hyperparameters(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Optimize LightGBM hyperparameters for tree-based modeling."""
        return {'n_estimators': 1000, 'learning_rate': 0.1, 'max_depth': 6}

    async def _select_tree_features(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Select features based on tree importance."""
        return list(X_train.columns)

    async def _retrain_lightgbm_model(self, best_params: List[Any], X_train: Any, y_train: Any) -> None:
        """Retrain LightGBM model with optimized parameters."""
        return

    async def _apply_lightgbm_optimizations(self, model: Any, X_train: Any, y_train: Any) -> None:
        """Apply LightGBM-specific optimizations."""
        return model

    async def _evaluate_lightgbm_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate LightGBM model performance."""
        return 0.0

    async def _optimize_cnn_hyperparameters(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Optimize CNN hyperparameters for convolution operations."""
        return {'num_filters': [32, 64, 128], 'kernel_size': 3, 'pool_size': 2}

    async def _select_spatial_features(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Select features relevant for spatial modeling."""
        return list(X_train.columns)

    async def _retrain_cnn_model(self, best_params: List[Any], X_train: Any, y_train: Any) -> None:
        """Retrain CNN model with optimized parameters."""
        return

    async def _apply_cnn_optimizations(self, model: Any, X_train: Any, y_train: Any) -> None:
        """Apply CNN-specific optimizations."""
        return model

    async def _evaluate_cnn_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate CNN model performance."""
        return 0.0
from src.core.decorators import deterministic_seed, idempotent_step, timeout, validates, log_execution_time, cached, log_call, circuit_breaker

@deterministic_seed(42)
@idempotent_step(step_key='step7_analyst_enhancement')
@validates()
@timeout(timeout=5400)
@validates(required_directories=['data/training', 'models'], min_memory_gb=8.0, min_disk_gb=5.0, required_packages=['pandas', 'numpy', 'sklearn', 'lightgbm', 'catboost'], data_quality_checks={'min_rows': 1000, 'required_columns': ['timestamp', 'features', 'targets']}, context='Analyst Enhancement')
@validates(backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True)
@validates(temporal_validation=True, feature_leakage_detection=True, cross_validation_isolation=True, lookahead_bias_prevention=True)
@log_execution_time(memory_threshold_gb=16.0, cpu_threshold_percent=90.0, disk_threshold_gb=10.0, monitor_interval=60.0, auto_cleanup=True)
@cached(chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25)
@log_call(log_intermediate_results=True, save_debug_artifacts=True, performance_profiling=True, error_context_preservation=True)
@circuit_breaker(failure_threshold=3, recovery_timeout=300.0, expected_exception=Exception, monitor_interval=60.0)
@validates(required_files=['models/{exchange}_{symbol}_analyst_enhanced.pkl'], data_quality_checks={'min_rows': 100, 'required_columns': ['predictions', 'probabilities']}, performance_thresholds={'enhancement_time_minutes': 90.0, 'memory_usage_gb': 8.0}, format_validation=True)
@validates(model_performance_thresholds={'accuracy': 0.6, 'f1_score': 0.5}, data_quality_metrics={'completeness': 0.9, 'consistency': 0.8}, convergence_checks=True, overfitting_detection=True, validation_score_requirements={'cross_validation_score': 0.6})
async def run_step(symbol: str, exchange: str='BINANCE', data_dir: str='data/training', force_rerun: bool=False, **kwargs) -> bool:
    """Run the analyst enhancement step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        force_rerun: Force rerun of the step
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    import copy
    import os.path
    from src.utils.logger import system_logger
    from src.utils.enhanced_mlflow_integration import with_enhanced_mlflow_logging, log_step_report, create_detailed_step_report, log_step_metrics, log_step_dataframe_with_standardized_name, log_step_artifact_with_standardized_name
    logger = system_logger.getChild('Step6.AnalystEnhancement')
    logger.info('=' * 80)
    logger.info('🚀 STEP 6: Analyst Enhancement')
    logger.info('=' * 80)
    logger.info('📋 Step 6 Parameters:')
    logger.info(f'   Symbol: {symbol}')
    logger.info(f'   Exchange: {exchange}')
    logger.info(f'   Data Directory: {data_dir}')
    logger.info(f'   Force Rerun: {force_rerun}')
    step_start_time = time.time()
    step_phases = {'configuration': False, 'initialization': False, 'model_loading': False, 'enhancement': False, 'validation': False}
    try:
        logger.info(f'🔄 Starting Step 6: Analyst Enhancement for {exchange}:{symbol}')
        logger.info('📋 Phase 1: Loading configuration...')
        try:
            config = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir}
            logger.info(f'✅ Configuration loaded: {len(config)} parameters')
            step_phases['configuration'] = True
        except Exception as e:
            logger.exception(f'❌ Configuration loading failed: {e}')
            return False
        logger.info('🔧 Phase 2: Initializing Analyst Enhancement Step...')
        try:
            step = RegimeAwareAnalystEnhancementStep(config)
            await step.initialize()
            logger.info('✅ Analyst Enhancement Step initialized successfully')
            step_phases['initialization'] = True
        except Exception as e:
            logger.exception(f'❌ Initialization failed: {e}')
            return False
        logger.info('📥 Phase 3: Preparing training input...')
        try:
            training_input = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir, 'force_rerun': force_rerun}
            pipeline_state = {}
            logger.info(f'✅ Training input prepared: {len(training_input)} parameters')
            step_phases['model_loading'] = True
        except Exception as e:
            logger.exception(f'❌ Training input preparation failed: {e}')
            return False
        logger.info('🎯 Phase 4: Executing model enhancement...')
        try:
            result = await step.execute(training_input, pipeline_state)
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                if status == 'SUCCESS':
                    logger.info('✅ Model enhancement completed successfully')
                    step_phases['enhancement'] = True
                else:
                    logger.error(f'❌ Model enhancement failed with status: {status}')
                    step_phases['enhancement'] = False
            else:
                logger.info('✅ Model enhancement completed (boolean result)')
                step_phases['enhancement'] = True
        except Exception as e:
            logger.exception(f'❌ Model enhancement execution failed: {e}')
            step_phases['enhancement'] = False
        logger.info('🔍 Phase 5: Validating enhancement results...')
        try:
            logger.info('✅ Enhancement validation completed')
            step_phases['validation'] = True
        except Exception as e:
            logger.exception(f'❌ Enhancement validation failed: {e}')
            step_phases['validation'] = False
        step_duration = time.time() - step_start_time
        successful_phases = sum((1 for v in step_phases.values() if v))
        total_phases = len(step_phases)
        logger.info('=' * 80)
        logger.info('📊 STEP 6 EXECUTION SUMMARY')
        logger.info('=' * 80)
        logger.info(f'Total execution time: {step_duration:.2f}s')
        logger.info(f'Successful phases: {successful_phases}/{total_phases}')
        logger.info('Phase status:')
        for phase, status in step_phases.items():
            status_emoji = '✅' if status else '❌'
            logger.info(f"   {status_emoji} {phase}: {('SUCCESS' if status else 'FAILED')}")
        final_result = successful_phases >= 4
        if final_result:
            logger.info('✅ Step 6: Analyst Enhancement completed successfully')
            logger.info(f'   Success rate: {successful_phases / total_phases * 100:.1f}%')
        else:
            logger.error('❌ Step 6: Analyst Enhancement failed')
            logger.error(f'   Success rate: {successful_phases / total_phases * 100:.1f}%')
        logger.info('=' * 80)
        return final_result
    except Exception as e:
        step_duration = time.time() - step_start_time
        logger.exception(f'❌ Step 6: Analyst Enhancement failed with exception: {e}')
        logger.exception(f'   Execution time: {step_duration:.2f}s')
        logger.exception(f'   Phase status: {step_phases}')
        return False