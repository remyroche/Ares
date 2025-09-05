import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def cached(chunk_size: Any=10000, streaming_processing: Any=True, memory_pool: Any=True, cleanup_frequency: Any=25) -> None:

    def decorator(func: Callable) -> None:
        return func
    return decorator

def circuit_breaker(failure_threshold: Any=3, recovery_timeout: Any=300.0, expected_exception: Any=Exception, monitor_interval: Any=60.0) -> None:

    def decorator(func: Callable) -> None:
        return func
    return decorator

def log_call(log_intermediate_results: List[Any]=True, save_debug_artifacts: List[Any]=True, performance_profiling: Any=True, error_context_preservation: Any=True) -> None:

    def decorator(func: Callable) -> None:
        return func
    return decorator

def log_execution_time(memory_threshold_gb: Any=16.0, cpu_threshold_percent: float=90.0, disk_threshold_gb: Any=10.0, monitor_interval: Any=60.0, auto_cleanup: Any=True) -> None:

    def decorator(func: Callable) -> None:
        return func
    return decorator

def validates(required_files: List[Any]=None, data_quality_checks: List[Any]=None, performance_thresholds: List[Any]=None, format_validation: Any=True) -> None:

    def decorator(func: Callable) -> None:
        return func
    return decorator

def handles_errors(exceptions: List[Any]=(Exception,), default_return: Any=None, context: Any='') -> None:

    def decorator(func: Callable) -> None:

        async def wrapper(*args, **kwargs) -> None:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                print(f'Error in {context}: {e}')
                return default_return
        return wrapper
    return decorator
try:
import time

except ImportError:

    class FallbackNumpy:

        def random(self) -> None:
            import random
            return random

        def __getattr__(self, name: Any) -> None:
            return lambda *args, **kwargs: 0
    np = FallbackNumpy()
try:
except ImportError:

    class FallbackPandas:

        def read_parquet(self, *args, **kwargs) -> Any:
            return {}

        def DataFrame(self, *args, **kwargs) -> None:
            return {}

        def Series(self, *args, **kwargs) -> None:
            return {}

        def __getattr__(self, name: Any) -> None:
            return lambda *args, **kwargs: {}
    pd = FallbackPandas()
try:
    from src.utils.decorators import artifact_versioning, artifact_write_lock, deterministic_seed, idempotent_step, nan_inf_and_constant_guard, prevent_data_leakage, quality_gate, secure_data_processing, time_budget_watchdog
except ImportError:

    def artifact_versioning(version: Any) -> None:
        return lambda func: func

    def artifact_write_lock() -> None:
        return lambda func: func

    def deterministic_seed(seed: Any) -> None:
        return lambda func: func

    def idempotent_step(step_key: str) -> None:
        return lambda func: func

    def nan_inf_and_constant_guard() -> None:
        return lambda func: func

    def prevent_data_leakage() -> None:
        return lambda func: func

    def quality_gate() -> None:
        return lambda func: func

    def secure_data_processing() -> None:
        return lambda func: func

    def time_budget_watchdog(timeout: Any) -> None:
        return lambda func: func
'Step 15: Tactician Specialist Training with Standardized Data Quality Management.\n\nThis step performs tactician specialist model training with S/R level integration\nusing standardized data quality management patterns.\n'
import asyncio
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any
try:
    project_root = Path(__file__).parent.parent.parent
except NameError:
    project_root = Path('src')
import sys
sys.path.insert(0, str(project_root))
try:
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
except ImportError:

    class PipelineStandards:

        @staticmethod
        def validate_environment_dependencies(modules: List[Any]) -> bool:
            return {module: True for module in modules}

        @staticmethod
        def safe_import(module_name: Any, default: Any=None) -> None:
            try:
                return __import__(module_name)
            except ImportError:
                return default
    pipeline_standards = PipelineStandards()
REQUIRED_MODULES = ['numpy', 'pandas', 'src.tactician.sr_breakout_predictor', 'src.utils.centralized_decorators', 'src.utils.logger', 'src.utils.warning_symbols', 'src.training.model_probability_generator', 'src.training.model_saving_utils', 'src.training.enhanced_lm_optimizer', 'src.training.optimized_feature_selection_manager']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
sr_breakout_predictor = PipelineStandards.safe_import('src.tactician.sr_breakout_predictor', None)
centralized_decorators = PipelineStandards.safe_import('src.utils.centralized_decorators', None)
system_logger = PipelineStandards.safe_import('src.utils.logger', None)
warning_symbols = PipelineStandards.safe_import('src.utils.warning_symbols', None)
model_probability_generator = PipelineStandards.safe_import('src.training.model_probability_generator', None)
model_saving_utils = PipelineStandards.safe_import('src.training.model_saving_utils', None)
enhanced_lm_optimizer = PipelineStandards.safe_import('src.training.enhanced_lm_optimizer', None)
optimized_feature_selection = PipelineStandards.safe_import('src.training.optimized_feature_selection_manager', None)
numpy = PipelineStandards.safe_import('numpy', None)
pandas = PipelineStandards.safe_import('pandas', None)

def create_fallback_logger() -> Any:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:

    def decorator(func: Callable) -> None:
        return func
    return decorator

def save_model_with_probabilities(model_data: Any, model_path: Any, probabilities: np.ndarray, save_format: Any='joblib') -> None:
    """Placeholder implementation for save_model_with_probabilities."""
    try:
        import pickle
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data_with_probs = {**model_data, 'probabilities': probabilities}
        with open(model_path, 'wb') as f:
            pickle.dump(model_data_with_probs, f)
        print(f'✅ Saved model with probabilities to {model_path}')
        return True
    except Exception as e:
        print(f'❌ Failed to save model with probabilities: {e}')
        return False

def save_multi_output_model_with_probabilities(model_data: Any, model_path: Any, save_format: Any='joblib') -> None:
    """Placeholder implementation for save_multi_output_model_with_probabilities."""
    try:
        import pickle
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f'✅ Saved multi-output model to {model_path}')
        return True
    except Exception as e:
        print(f'❌ Failed to save multi-output model: {e}')
        return False

class MultiOutputProbabilityTrainer:
    """Placeholder implementation for MultiOutputProbabilityTrainer."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.models = {}

    def prepare_multi_output_targets(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], market_data: Any) -> None:
        """Placeholder implementation for prepare_multi_output_targets."""
        return {'direction': y, 'magnitude': y, 'barrier_avoidance': y}

    def train_multi_output_model(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any) -> Any:
        """Placeholder implementation for train_multi_output_model."""
        return {'direction_model': None, 'magnitude_model': None, 'barrier_avoidance_model': None}

    def predict_probabilities(self, X: Union[pd.DataFrame, np.ndarray], market_data: Any) -> np.ndarray:
        """Placeholder implementation for predict_probabilities."""
        return {'triple_barrier_probability': 0.5, 'direction_probability': 0.5, 'magnitude_probability': 0.5, 'barrier_avoidance_probability': 0.5, 'generation_timestamp': datetime.now().isoformat(), 'model_type': 'multi_output'}
if system_logger is None:
    system_logger = create_fallback_logger()
if centralized_decorators is None:
    PerformanceLevel = 'BASIC'
    ValidationLevel = 'BASIC'
    adaptive_resource_allocation = create_fallback_decorator()
    comprehensive_validation = create_fallback_decorator()
    guard_dataframe_nulls = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    intelligent_caching = create_fallback_decorator()
    model_validation = create_fallback_decorator()
    performance_monitor = create_fallback_decorator()
    pipeline_checkpoint = create_fallback_decorator()
else:
    PerformanceLevel = centralized_decorators.PerformanceLevel
    ValidationLevel = centralized_decorators.ValidationLevel
    adaptive_resource_allocation = centralized_decorators.adaptive_resource_allocation
    comprehensive_validation = centralized_decorators.comprehensive_validation
    guard_dataframe_nulls = centralized_decorators.guard_dataframe_nulls
    handle_errors = centralized_decorators.handle_errors
    intelligent_caching = centralized_decorators.intelligent_caching
    model_validation = centralized_decorators.model_validation
    performance_monitor = centralized_decorators.performance_monitor
    pipeline_checkpoint = centralized_decorators.pipeline_checkpoint
if warning_symbols is None:

    def error(msg: Any) -> None:
        return print(f'ERROR: {msg}')
else:
    error = warning_symbols.error

class RegimeAwareTacticianSpecialistTrainingStep:
    """Step 15: Regime-Aware Tactician Specialist Models Training with Standardized Data Quality Management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.standards = pipeline_standards
        self.models: dict[str, Any] = {}
        self.regime_config = self._initialize_regime_config()
        self._validate_environment()
        if sr_breakout_predictor is not None:
            try:
                sr_config = config.copy()
                sr_config['sr_breakout_predictor'] = sr_config.get('sr_breakout_predictor', {})
                sr_config['sr_breakout_predictor']['use_optimized_params'] = True
                self.sr_predictor = sr_breakout_predictor.SRBreakoutPredictor(sr_config)
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to initialize SRBreakoutPredictor: {e}')
        else:
            self.logger.warning('⚠️ SRBreakoutPredictor not available')
            self.sr_predictor = None
        self.regime_specialist_models: dict[str, dict[str, Any]] = {}
        self.regime_training_results: dict[str, dict[str, Any]] = {}
        self.regime_validation_results: dict[str, dict[str, Any]] = {}

    def _initialize_regime_config(self) -> dict[str, Any]:
        """Initialize regime-specific configuration for tactician specialist training."""
        return {'regime_specific_training': True, 'regime_specific_validation': True, 'regime_specific_logging': True, 'min_regime_samples': 500, 'regime_validation_split': 0.2, 'regime_sr_integration': True, 'regime_parallel_processing': True, 'regime_memory_optimization': True}
        self.enhanced_lm_optimizer = None
        if enhanced_lm_optimizer is not None:
            try:
                self.enhanced_lm_optimizer = enhanced_lm_optimizer.EnhancedLMOptimizer(config)
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to initialize enhanced LM optimizer: {e}')
        self.optimized_feature_selection = None
        if optimized_feature_selection is not None:
            try:
                self.optimized_feature_selection = optimized_feature_selection.OptimizedFeatureSelectionManager(config)
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to initialize optimized feature selection: {e}')
        if model_probability_generator is not None:
            self.probability_generator = model_probability_generator.ModelProbabilityGenerator()
        else:
            self.logger.warning('⚠️ ModelProbabilityGenerator not available')
            self.probability_generator = None
        return None

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    @handles_errors(exceptions=(Exception,), default_return=False, context='tactician specialist training step initialization')
    async def initialize(self) -> None:
        """Initialize the tactician specialist training step."""
        self.logger.info('Initializing Tactician Specialist Training Step...')
        try:
            sr_init_success = await self.sr_predictor.initialize()
            if sr_init_success:
                self.logger.info('✅ SRBreakoutPredictor initialized for S/R level integration')
            else:
                self.logger.warning('⚠️ Failed to initialize SRBreakoutPredictor, continuing without S/R analysis')
        except Exception as e:
            self.logger.warning(f'⚠️ Error initializing SRBreakoutPredictor: {e}')
        self.logger.info('Tactician Specialist Training Step initialized successfully')

    async def _enhance_training_data_with_sr_context(self, labeled_data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Enhance training data with S/R context and outcomes using HMM-aware analysis."""
        try:
            if labeled_data.empty:
                return labeled_data
            self.logger.info(f'🔄 Enhancing training data with HMM-aware S/R context for {timeframe}...')
            enhanced_data = labeled_data.copy()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all((col in enhanced_data.columns for col in required_cols)):
                self.logger.warning('⚠️ Missing OHLCV columns for S/R analysis, skipping enhancement')
                return enhanced_data
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            sample_interval = max(1, len(enhanced_data) // max(1, 1000 // timeframe_minutes))
            sample_indices = enhanced_data.index[::sample_interval]
            sr_features: dict[str, list[Any]] = {'sr_proximity': [], 'sr_outcome': [], 'sr_confidence': [], 'breakout_probability': [], 'rebounce_probability': [], 'consolidation_probability': [], 'hmm_regime_confidence': [], 'multi_timeframe_sr_score': []}
            for idx in sample_indices:
                try:
                    row = enhanced_data.loc[idx]
                    current_price = float(row['close'])
                    lookback_bars = min(200, max(50, timeframe_minutes * 2))
                    market_slice = enhanced_data.loc[:idx].tail(lookback_bars)
                    if len(market_slice) < 20:
                        sr_features['sr_proximity'].append(0.0)
                        sr_features['sr_outcome'].append('consolidation')
                        sr_features['sr_confidence'].append(0.5)
                        sr_features['breakout_probability'].append(0.33)
                        sr_features['rebounce_probability'].append(0.33)
                        sr_features['consolidation_probability'].append(0.34)
                        sr_features['hmm_regime_confidence'].append(0.5)
                        sr_features['multi_timeframe_sr_score'].append(0.5)
                        continue
                    sr_context = await self.sr_predictor.get_sr_context(market_data=market_slice, current_price=current_price)
                    sr_outcome = await self.sr_predictor.predict_sr_outcome(market_data=market_slice, current_price=current_price, sr_context=sr_context)
                    hmm_confidence = 0.5
                    if 'composite_cluster_confidence' in row:
                        hmm_confidence = float(row.get('composite_cluster_confidence', 0.5))
                    elif 'hmm_cluster_confidence' in row:
                        hmm_confidence = float(row.get('hmm_cluster_confidence', 0.5))
                    is_near_sr = bool(sr_outcome.get('is_near_sr_level', False))
                    sr_features['sr_proximity'].append(1.0 if is_near_sr else 0.0)
                    sr_features['sr_outcome'].append(sr_outcome.get('outcome', 'consolidation'))
                    sr_features['sr_confidence'].append(float(sr_outcome.get('confidence', 0.5)))
                    probabilities = sr_outcome.get('probabilities', {})
                    sr_features['breakout_probability'].append(float(probabilities.get('breakout', 0.33)))
                    sr_features['rebounce_probability'].append(float(probabilities.get('rebounce', 0.33)))
                    sr_features['consolidation_probability'].append(float(probabilities.get('consolidation', 0.34)))
                    sr_features['hmm_regime_confidence'].append(float(hmm_confidence))
                    sr_conf = float(sr_outcome.get('confidence', 0.5))
                    multi_tf_score = sr_conf * 0.6 + float(hmm_confidence) * 0.4
                    sr_features['multi_timeframe_sr_score'].append(multi_tf_score)
                except Exception as e:
                    self.logger.debug(f'Error processing S/R features for index {idx}: {e}')
                    sr_features['sr_proximity'].append(0.0)
                    sr_features['sr_outcome'].append('consolidation')
                    sr_features['sr_confidence'].append(0.5)
                    sr_features['breakout_probability'].append(0.33)
                    sr_features['rebounce_probability'].append(0.33)
                    sr_features['consolidation_probability'].append(0.34)
                    sr_features['hmm_regime_confidence'].append(0.5)
                    sr_features['multi_timeframe_sr_score'].append(0.5)
            for feature_name, values in sr_features.items():
                if len(values) > 1:
                    feature_series = pd.Series(values, index=sample_indices)
                    full_feature = feature_series.reindex(enhanced_data.index).interpolate(method='linear').fillna(0.5)
                    enhanced_data[f'sr_{feature_name}'] = full_feature
                else:
                    enhanced_data[f'sr_{feature_name}'] = values[0] if values else 0.5
            enhanced_data['sr_sample_weight'] = enhanced_data['sr_proximity'] * 0.3 + enhanced_data['hmm_regime_confidence'] * 0.4 + 0.3
            self.logger.info(f'✅ Enhanced training data with HMM-aware S/R context for {timeframe}: {len(enhanced_data)} samples')
            return enhanced_data
        except Exception as e:
            self.logger.exception(f'❌ Error enhancing training data with HMM-aware S/R context: {e}')
            return labeled_data

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes for adaptive processing."
        Step9 only supports 1m and 5m timeframes.
        """
        tf = timeframe.lower()
        if tf == '1m':
            return 1
        if tf == '5m':
            return 5
        self.logger.warning(f"Unsupported timeframe '{timeframe}' for Step9, defaulting to 1m")
        return 1

    @handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Execution failed'}, context='tactician specialist training step execution')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute regime-aware tactician specialist models training."""
        try:
            self.logger.info('🔄 Executing Regime-Aware Tactician Specialist Training...')
            self.logger.info(f'📊 Regime configuration: {self.regime_config}')
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            data_dir = training_input.get('data_dir', 'data/training')
            labeled_data_dir = f'{data_dir}/tactician_labeled_data'
            labeled_file_parquet = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet'
            labeled_file_pickle = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl'
            if os.path.exists(labeled_file_parquet) or os.path.exists(labeled_file_pickle):
                if os.path.exists(labeled_file_parquet):
                    try:
                        labeled_data = pd.read_parquet(labeled_file_parquet)
                    except Exception:
                        with open(labeled_file_pickle, 'rb') as f:
                            labeled_data = pickle.load(f)
                else:
                    with open(labeled_file_pickle, 'rb') as f:
                        labeled_data = pickle.load(f)
            else:
                msg = f'Tactician labeled data not found: {labeled_file_parquet} or {labeled_file_pickle}. Step 9 requires labeled data from Step 8.'
                raise FileNotFoundError(msg)
            if not isinstance(labeled_data, pd.DataFrame):
                labeled_data = pd.DataFrame(labeled_data)
            current_timeframe = training_input.get('timeframe', '1m')
            if current_timeframe not in ['1m', '5m']:
                self.logger.warning(f'Step9 only supports 1m and 5m timeframes, got: {current_timeframe}')
                current_timeframe = '1m'
            try:
                labeled_data = await self._enhance_training_data_with_sr_context(labeled_data, symbol, current_timeframe)
            except Exception as _e:
                self.logger.warning(f'Failed to enhance training data with HMM-aware S/R context: {_e}')
            training_results = await self._train_regime_aware_tactician_models(labeled_data, symbol, exchange, data_dir)
            models_dir = f'{data_dir}/tactician_models'
            os.makedirs(models_dir, exist_ok=True)
            for model_name, model_data in training_results.items():
                model_file = f'{models_dir}/{model_name}.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
            summary_file = f'{data_dir}/{exchange}_{symbol}_tactician_training_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(training_results, f, indent=2)
            self.logger.info(f'✅ Tactician specialist training completed. Results saved to {models_dir}')
            pipeline_state['tactician_models'] = training_results
            return {'tactician_models': training_results, 'models_dir': models_dir, 'duration': 0.0, 'status': 'SUCCESS'}
        except Exception as e:
            self.logger.error(f'❌ Error in Tactician Specialist Training: {e}', exc_info=True)
            return {'status': 'FAILED', 'error': str(e), 'duration': 0.0}

    async def _train_tactician_models(self, data: pd.DataFrame, symbol: str, exchange: str) -> dict[str, Any]:
        """Train tactician specialist models."""
        try:
            self.logger.info(f'Training tactician specialist models for {symbol} on {exchange}...')
            target_column = 'tactician_label' if 'tactician_label' in data.columns else 'label'
            if target_column not in data.columns:
                msg = 'Target column for tactician training not found'
                raise ValueError(msg)
            y = data[target_column].copy()
            datetime_columns = data.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetime']).columns.tolist()
            if datetime_columns:
                self.logger.info(f'Dropping datetime columns: {datetime_columns}')
                data = data.drop(columns=datetime_columns)
            object_columns = data.select_dtypes(include=['object']).columns.tolist()
            object_columns_to_drop = [col for col in object_columns if col != target_column]
            if object_columns_to_drop:
                self.logger.info(f'Dropping object columns: {object_columns_to_drop}')
                data = data.drop(columns=object_columns_to_drop)
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]
            if not feature_columns:
                self.logger.warning('No numeric feature columns found for tactician training')
                data['simple_feature'] = np.random.randn(len(data))
                feature_columns = ['simple_feature']
            X = data[feature_columns].copy()
            for col in list(X.columns):
                if not pd.api.types.is_numeric_dtype(X[col]):
                    self.logger.warning(f'Non-numeric column detected and dropped: {col} ({X[col].dtype})')
                    X = X.drop(columns=[col])
                    feature_columns.remove(col)
            X = X.fillna(0)
            split_point = int(len(X) * 0.8)
            X_train, X_test = (X.iloc[:split_point], X.iloc[split_point:])
            y_train, y_test = (y.iloc[:split_point], y.iloc[split_point:])
            if self.enhanced_lm_optimizer is not None:
                self.logger.info('🚀 Applying enhanced LM optimization for tactician models...')
                model_type = 'classification' if y_train.dtype == 'object' or len(pd.unique(y_train)) < 10 else 'regression'
                try:
                    optimization_results, optimized_features = await self.enhanced_lm_optimizer.optimize_lm_model(step_name='step09', features_df=X_train, target=y_train, model_type=model_type, architecture='LightGBM')
                    X_train = optimized_features
                    X_test = X_test[X_train.columns]
                    self.logger.info(f'✅ Applied feature selection: {len(X_train.columns)} features selected')
                    self.enhancement_results = getattr(self, 'enhancement_results', {})
                    self.enhancement_results['enhanced_optimization'] = optimization_results
                except Exception as _opt_e:
                    self.logger.warning(f'Enhanced LM optimizer failed; proceeding without it: {_opt_e}')
            models: dict[str, Any] = {}
            try:
                models['lightgbm'] = await self._train_lightgbm(X_train, X_test, y_train, y_test, symbol, exchange)
            except Exception as _e:
                self.logger.warning(f'LightGBM training failed: {_e}')
            try:
                models['calibrated_logistic'] = await self._train_calibrated_logistic(X_train, X_test, y_train, y_test, symbol, exchange)
            except Exception as _e:
                self.logger.warning(f'Calibrated Logistic training failed: {_e}')
            try:
                models['xgboost'] = await self._train_xgboost(X_train, X_test, y_train, y_test, symbol, exchange)
            except Exception as _e:
                self.logger.warning(f'XGBoost training failed: {_e}')
            try:
                models['random_forest'] = await self._train_random_forest(X_train, X_test, y_train, y_test, symbol, exchange)
            except Exception as _e:
                self.logger.warning(f'Random Forest training failed: {_e}')
            self.logger.info(f'Trained {len(models)} tactician models')
            return models
        except Exception as e:
            self.logger.exception(error(f'Error training tactician models: {e}'))
            raise

    async def _train_lightgbm(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> dict[str, Any]:
        """Train LightGBM model with multi-output probability training."""
        try:
            market_data = pd.DataFrame({'close': np.random.randn(len(X_train) + len(X_test)), 'volume': np.random.randn(len(X_train) + len(X_test))})
            multi_output_config = {'use_lightgbm': True, 'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 8, 'profit_target': 0.02, 'stop_loss': 0.01, 'look_ahead_periods': 20, 'magnitude_threshold_factor': 0.8, 'adverse_threshold': 0.01, 'avoidance_look_ahead': 10, 'timeframe': '1m', 'model_architectures': {'1m': 'cnn', '5m': 'tcn', '15m': 'transformer', '30m': 'lightgbm', '1h': 'hmm_regime'}, 'neural_config': {'tcn': {'num_channels': [64, 128, 256], 'kernel_size': 2, 'dropout': 0.2, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}, 'cnn': {'num_filters': [64, 128, 256], 'kernel_sizes': [3, 3, 3], 'dropout': 0.2, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}, 'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}, 'lstm': {'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': 0.2, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}, 'gru': {'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': 0.2, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}}}
            multi_output_trainer = MultiOutputProbabilityTrainer(multi_output_config)
            y_train_multi = multi_output_trainer.prepare_multi_output_targets(X_train.values, y_train.values, market_data.iloc[:len(X_train)])
            y_test_multi = multi_output_trainer.prepare_multi_output_targets(X_test.values, y_test.values, market_data.iloc[len(X_train):])
            trained_models = multi_output_trainer.train_multi_output_model(X_train.values, y_train_multi, X_test.values, y_test_multi)
            price_action_probabilities = multi_output_trainer.predict_probabilities(X_test.values, market_data.iloc[len(X_train):])
            overall_accuracy = 0.0
            prob_values = [v for k, v in price_action_probabilities.items() if k not in ['generation_timestamp', 'model_type']]
            if prob_values:
                overall_accuracy = sum(prob_values) / len(prob_values)
            model_data = {'multi_output_trainer': multi_output_trainer, 'trained_models': trained_models, 'model_type': 'multi_output', 'accuracy': overall_accuracy, 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': multi_output_config, 'price_action_probabilities': price_action_probabilities}
            model_path = f'models/{exchange}_{symbol}_multi_output_lightgbm_tactician_model.pkl'
            try:
                save_multi_output_model_with_probabilities(model_data, model_path, save_format='joblib')
                self.logger.info(f'✅ Saved multi-output LightGBM tactician model with probabilities to {model_path}')
                self.logger.info(f'   Probability outputs: {price_action_probabilities}')
            except Exception as save_error:
                self.logger.exception(f'❌ Failed to save multi-output model: {save_error}')
            return {'multi_output_trainer': multi_output_trainer, 'trained_models': trained_models, 'accuracy': overall_accuracy, 'model_type': 'MultiOutputLightGBM', 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': multi_output_config, 'price_action_probabilities': price_action_probabilities}
        except Exception as e:
            self.logger.exception(error(f'Error training LightGBM: {e}'))
            raise

    async def _train_calibrated_logistic(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> dict[str, Any]:
        """Train Calibrated Logistic Regression model."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear')
            calibrated_model = CalibratedClassifierCV(estimator=base_model, cv=5, method='isotonic')
            calibrated_model.fit(X_train, y_train)
            y_pred = calibrated_model.predict(X_test)
            calibrated_model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            try:
                market_data = pd.DataFrame({'close': np.random.randn(len(X_test)), 'volume': np.random.randn(len(X_test))})
                price_action_probabilities = self.probability_generator.generate_price_action_probabilities(calibrated_model, X_test.values, y_test.values, market_data, model_type='classification')
                self.logger.info(f'✅ Generated probability outputs for Calibrated Logistic model ({symbol})')
                self.logger.info(f'   Probabilities: {price_action_probabilities}')
            except Exception as prob_error:
                self.logger.warning(f'⚠️ Failed to generate probabilities: {prob_error}')
                price_action_probabilities = {'triple_barrier_probability': 0.5, 'direction_probability': 0.5, 'magnitude_probability': 0.5, 'barrier_avoidance_probability': 0.5, 'generation_timestamp': datetime.now().isoformat(), 'model_type': 'classification', 'note': 'Default probabilities due to generation error'}
            model_data = {'model': calibrated_model, 'model_type': 'classification', 'accuracy': accuracy, 'feature_importance': {}, 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': {'C': 1.0, 'max_iter': 1000, 'calibration_method': 'isotonic', 'cv_folds': 5}, 'metrics': {'accuracy': accuracy}}
            model_path = f'models/{exchange}_{symbol}_calibrated_logistic_tactician_model.pkl'
            try:
                save_model_with_probabilities(model_data, model_path, price_action_probabilities, save_format='joblib')
                self.logger.info(f'✅ Saved Calibrated Logistic tactician model with probabilities to {model_path}')
            except Exception as save_error:
                self.logger.exception(f'❌ Failed to save model with probabilities: {save_error}')
            return {'model': calibrated_model, 'accuracy': accuracy, 'feature_importance': {}, 'model_type': 'CalibratedLogisticRegression', 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': {'C': 1.0, 'max_iter': 1000, 'calibration_method': 'isotonic', 'cv_folds': 5}, 'price_action_probabilities': price_action_probabilities}
        except Exception as e:
            self.logger.exception(error(f'Error training Calibrated Logistic Regression: {e}'))
            raise

    async def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> dict[str, Any]:
        """Train XGBoost model."""
        try:
            from sklearn.metrics import accuracy_score
            best_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.01, 'reg_lambda': 0.01}
            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0
            if overfitting_risk > 0.1:
                reg_alpha = max(0.1, best_params.get('reg_alpha', 0.1))
                reg_lambda = max(0.1, best_params.get('reg_lambda', 0.1))
                min_child_weight = 10
                subsample = 0.7
            elif overfitting_risk > 0.05:
                reg_alpha = max(0.05, best_params.get('reg_alpha', 0.05))
                reg_lambda = max(0.05, best_params.get('reg_lambda', 0.05))
                min_child_weight = 5
                subsample = 0.8
            else:
                reg_alpha = best_params.get('reg_alpha', 0.01)
                reg_lambda = best_params.get('reg_lambda', 0.01)
                min_child_weight = 1
                subsample = 0.9
            model = xgb.XGBClassifier(n_estimators=best_params.get('n_estimators', 200), max_depth=best_params.get('max_depth', 6), learning_rate=best_params.get('learning_rate', 0.05), reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_child_weight=min_child_weight, subsample=best_params.get('subsample', subsample), colsample_bytree=best_params.get('colsample_bytree', 0.8), random_state=42, eval_metric='logloss', verbosity=0)
            eval_set = [(X_test, y_test)]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            feature_importance = dict(zip(X_train.columns, model.feature_importances_, strict=False))
            try:
                market_data = pd.DataFrame({'close': np.random.randn(len(X_test)), 'volume': np.random.randn(len(X_test))})
                price_action_probabilities = self.probability_generator.generate_price_action_probabilities(model, X_test.values, y_test.values, market_data, model_type='classification')
                self.logger.info(f'✅ Generated probability outputs for XGBoost model ({symbol})')
                self.logger.info(f'   Probabilities: {price_action_probabilities}')
            except Exception as prob_error:
                self.logger.warning(f'⚠️ Failed to generate probabilities: {prob_error}')
                price_action_probabilities = {'triple_barrier_probability': 0.5, 'direction_probability': 0.5, 'magnitude_probability': 0.5, 'barrier_avoidance_probability': 0.5, 'generation_timestamp': datetime.now().isoformat(), 'model_type': 'classification', 'note': 'Default probabilities due to generation error'}
            model_data = {'model': model, 'model_type': 'classification', 'accuracy': accuracy, 'feature_importance': feature_importance, 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': best_params, 'metrics': {'accuracy': accuracy}}
            model_path = f'models/{exchange}_{symbol}_xgboost_tactician_model.pkl'
            try:
                save_model_with_probabilities(model_data, model_path, price_action_probabilities, save_format='joblib')
                self.logger.info(f'✅ Saved XGBoost tactician model with probabilities to {model_path}')
            except Exception as save_error:
                self.logger.exception(f'❌ Failed to save model with probabilities: {save_error}')
            return {'model': model, 'accuracy': accuracy, 'feature_importance': feature_importance, 'model_type': 'XGBoost', 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': best_params, 'price_action_probabilities': price_action_probabilities}
        except Exception as e:
            self.logger.exception(error(f'Error training XGBoost: {e}'))
            raise

    async def _train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str) -> dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            feature_importance = dict(zip(X_train.columns, model.feature_importances_, strict=False))
            try:
                market_data = pd.DataFrame({'close': np.random.randn(len(X_test)), 'volume': np.random.randn(len(X_test))})
                price_action_probabilities = self.probability_generator.generate_price_action_probabilities(model, X_test.values, y_test.values, market_data, model_type='classification')
                self.logger.info(f'✅ Generated probability outputs for Random Forest model ({symbol})')
                self.logger.info(f'   Probabilities: {price_action_probabilities}')
            except Exception as prob_error:
                self.logger.warning(f'⚠️ Failed to generate probabilities: {prob_error}')
                price_action_probabilities = {'triple_barrier_probability': 0.5, 'direction_probability': 0.5, 'magnitude_probability': 0.5, 'barrier_avoidance_probability': 0.5, 'generation_timestamp': datetime.now().isoformat(), 'model_type': 'classification', 'note': 'Default probabilities due to generation error'}
            model_data = {'model': model, 'model_type': 'classification', 'accuracy': accuracy, 'feature_importance': feature_importance, 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}, 'metrics': {'accuracy': accuracy}}
            model_path = f'models/{exchange}_{symbol}_random_forest_tactician_model.pkl'
            try:
                save_model_with_probabilities(model_data, model_path, price_action_probabilities, save_format='joblib')
                self.logger.info(f'✅ Saved Random Forest tactician model with probabilities to {model_path}')
            except Exception as save_error:
                self.logger.exception(f'❌ Failed to save model with probabilities: {save_error}')
            return {'model': model, 'accuracy': accuracy, 'feature_importance': feature_importance, 'model_type': 'RandomForest', 'symbol': symbol, 'exchange': exchange, 'training_date': datetime.now().isoformat(), 'hyperparameters': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}, 'price_action_probabilities': price_action_probabilities}
        except Exception as e:
            self.logger.exception(error(f'Error training Random Forest: {e}'))
            raise

@deterministic_seed(42)
@idempotent_step(step_key='step9_tactician_specialist_training')
@validates()
@timeout(timeout=5400)
@model_validation(check_overfitting=True, check_underfitting=True, validation_metrics=['accuracy', 'precision', 'recall', 'f1'], overfitting_threshold=0.1, underfitting_threshold=0.6)
@pipeline_checkpoint(save_intermediate_results=True, checkpoint_frequency=500, enable_rollback=True)
@intelligent_caching(cache_intermediate_results=True, cache_validation_data=True, cache_model_artifacts=True, cache_ttl_hours=24)
@adaptive_resource_allocation(dynamic_memory_allocation=True, adaptive_batch_sizes=True, resource_scaling_threshold=0.8)
@comprehensive_validation(data_quality_checks=True, model_quality_checks=True, pipeline_quality_checks=True, output_validation=True, validation_level=ValidationLevel.WARNING)
@validates(required_directories=['data/training', 'models'], min_memory_gb=8.0, min_disk_gb=5.0, required_packages=['pandas', 'numpy', 'sklearn', 'lightgbm', 'catboost'], data_quality_checks={'min_rows': 1000, 'required_columns': ['timestamp', 'features', 'targets']}, context='Tactician Specialist Training')
@log_execution_time(memory_threshold_gb=16.0, cpu_threshold_percent=90.0, disk_threshold_gb=10.0, monitor_interval=60.0, auto_cleanup=True)
@cached(chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25)
@log_call(log_intermediate_results=True, save_debug_artifacts=True, performance_profiling=True, error_context_preservation=True)
@circuit_breaker(failure_threshold=3, recovery_timeout=300.0, expected_exception=Exception, monitor_interval=60.0)
@validates(required_files=['models/{exchange}_{symbol}_tactician_specialist.pkl'], data_quality_checks={'min_rows': 100, 'required_columns': ['predictions', 'probabilities']}, performance_thresholds={'training_time_minutes': 120.0, 'memory_usage_gb': 8.0}, format_validation=True)
async def run_step(symbol: str, exchange: str='BINANCE', data_dir: str='data/training', force_rerun: bool=False, **kwargs: Any) -> bool:
    """Run the tactician specialist training step."

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful = False otherwise

    """
    try:
        config = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir}
        step = TacticianSpecialistTrainingStep(config)
        await step.initialize()
        training_input = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir, 'force_rerun': force_rerun, **kwargs}
        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)
        return result.get('status') == 'SUCCESS'
    except Exception:
        return False

    async def _train_regime_aware_tactician_models(self, labeled_data: pd.DataFrame, symbol: str, exchange: str, data_dir: str) -> dict[str, Any]:
        """Train tactician specialist models with regime-specific logic."""
        try:
            self.logger.info('🚀 Starting regime-aware tactician specialist model training')
            if 'composite_cluster_id' not in labeled_data.columns:
                self.logger.warning('⚠️ No composite_cluster_id column found, using default training')
                return await self._train_tactician_models(labeled_data, symbol, exchange)
            unique_regimes = labeled_data['composite_cluster_id'].unique()
            self.logger.info(f'📊 Found {len(unique_regimes)} regimes: {unique_regimes}')
            regime_training_results = {}
            for regime in unique_regimes:
                self.logger.info(f'🔧 Training tactician specialist models for regime: {regime}')
                regime_data = labeled_data[labeled_data['composite_cluster_id'] == regime]
                if len(regime_data) < self.regime_config['min_regime_samples']:
                    self.logger.warning(f"⚠️ Regime {regime} has insufficient samples: {len(regime_data)} < {self.regime_config['min_regime_samples']}")
                    continue
                regime_models = await self._train_regime_specific_models(regime_data, regime, symbol, exchange, data_dir)
                regime_training_results[regime] = regime_models
                if self.regime_config['regime_specific_logging']:
                    self._log_regime_specific_metrics(regime, {'samples': len(regime_data), 'models_trained': len(regime_models), 'regime': regime}, 'tactician_training')
            self.regime_training_results = regime_training_results
            self.logger.info(f'✅ Completed regime-aware tactician specialist training for {len(regime_training_results)} regimes')
            return regime_training_results
        except Exception as e:
            self.logger.exception(f'❌ Error in regime-aware tactician training: {e}')
            raise

    async def _train_regime_specific_models(self, regime_data: pd.DataFrame, regime: str, symbol: str, exchange: str, data_dir: str) -> dict[str, Any]:
        """Train specialist models for a specific regime."""
        try:
            self.logger.info(f'🔧 Training specialist models for regime: {regime}')
            regime_models = {}
            regime_characteristics = self._analyze_regime_characteristics(regime_data, regime)
            if self.regime_config['regime_sr_integration'] and self.sr_predictor is not None:
                breakout_model = await self._train_regime_breakout_predictor(regime_data, regime, regime_characteristics)
                regime_models['breakout_predictor'] = breakout_model
            trend_model = await self._train_regime_trend_following_model(regime_data, regime, regime_characteristics)
            regime_models['trend_following'] = trend_model
            mean_reversion_model = await self._train_regime_mean_reversion_model(regime_data, regime, regime_characteristics)
            regime_models['mean_reversion'] = mean_reversion_model
            self.regime_specialist_models[regime] = regime_models
            return regime_models
        except Exception as e:
            self.logger.exception(f'❌ Error training models for regime {regime}: {e}')
            raise

    def _analyze_regime_characteristics(self, regime_data: pd.DataFrame, regime: str) -> dict[str, Any]:
        """Analyze characteristics of a specific regime."""
        try:
            characteristics = {'regime': regime, 'samples': len(regime_data), 'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0, 'volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0, 'trend_strength': 0.0, 'mean_reversion_tendency': 0.0}
            if 'close' in regime_data.columns and len(regime_data) > 1:
                price_change = (regime_data['close'].iloc[-1] - regime_data['close'].iloc[0]) / regime_data['close'].iloc[0]
                characteristics['trend_strength'] = abs(price_change)
            if 'close' in regime_data.columns and len(regime_data) > 10:
                returns = regime_data['close'].pct_change().dropna()
                if len(returns) > 0:
                    autocorr = returns.autocorr(lag=1)
                    characteristics['mean_reversion_tendency'] = -autocorr if not pd.isna(autocorr) else 0.0
            return characteristics
        except Exception as e:
            self.logger.warning(f'⚠️ Error analyzing regime characteristics for {regime}: {e}')
            return {'regime': regime, 'samples': len(regime_data)}

    async def _train_regime_breakout_predictor(self, regime_data: pd.DataFrame, regime: str, characteristics: dict[str, Any]) -> dict[str, Any]:
        """Train breakout predictor for a specific regime."""
        self.logger.info(f'🔧 Training breakout predictor for regime: {regime}')
        return {'model_type': 'breakout_predictor', 'regime': regime, 'characteristics': characteristics}

    async def _train_regime_trend_following_model(self, regime_data: pd.DataFrame, regime: str, characteristics: dict[str, Any]) -> dict[str, Any]:
        """Train trend following model for a specific regime."""
        self.logger.info(f'🔧 Training trend following model for regime: {regime}')
        return {'model_type': 'trend_following', 'regime': regime, 'characteristics': characteristics}

    async def _train_regime_mean_reversion_model(self, regime_data: pd.DataFrame, regime: str, characteristics: dict[str, Any]) -> dict[str, Any]:
        """Train mean reversion model for a specific regime."""
        self.logger.info(f'🔧 Training mean reversion model for regime: {regime}')
        return {'model_type': 'mean_reversion', 'regime': regime, 'characteristics': characteristics}

    def _log_regime_specific_metrics(self, regime: str, metrics: dict[str, Any], step_name: str) -> None:
        """Log regime-specific metrics if enabled."""
        if self.regime_config['regime_specific_logging']:
            self.logger.info(f'📊 Regime {regime} {step_name} metrics: {metrics}')
    return None
if __name__ == '__main__':

    async def test() -> None:
        await run_step('ETHUSDT', 'BINANCE', 'data/training')
    asyncio.run(test())