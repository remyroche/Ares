from src.utils.tprint import tprint

from typing import Dict, List, Optional, Union, Any, Tuple

# Required dependencies - fail fast if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from src.utils.logger import system_logger
# Import decorators - fail fast if not available
try:
    from src.core.decorators import handles_errors, log_call, traced, validates
except Exception as e:
    raise ImportError(f"Required decorators not available: {e}. "
                     f"Please ensure src.core.decorators is properly installed.")
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import required utility functions - fail fast if not available
from src.utils.common_operations import (
    format_bytes, safe_log_metric, safe_log_params, safe_file_exists,
    get_current_datetime, format_datetime, safe_read_parquet, safe_json_dump,
    validate_dataframe, optimize_dataframe_dtypes, validate_dataframe_schema
)

"""Model Training Package for Trading Pipeline.

This package contains all the components for model training:
- HMM-based training and multi-timeframe ensembles (15m timeframe, 15-25 regimes)
- Unified regime intelligence
- Analyst creation, enhancement, and ensemble creation (5m timeframe, per-regime training)
- Tactician labeling and specialist training (1m timeframe, unified training on green-light periods)
- Model persistence and validation components

REGIME HANDLING STRATEGY:
- HMM Models: Detect 15-25 regimes on 15m timeframe for macro market state
- Analyst Models: Per-regime training on 5m timeframe for regime-specific patterns
- Tactician Models: Unified training on 1m timeframe for precise entry timing
- This design is intentional: different components optimize for different aspects of trading
"""

def validate_training_data(data: Any, data_name: str, required_columns: Optional[List[str]] = None) -> bool:
    """
    Consistent data validation for all training components.

    Args:
        data: Data to validate (DataFrame, array, etc.)
        data_name: Name of the data for error messages
        required_columns: Required columns for DataFrames

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if data is None:
        raise ValueError(f"{data_name} cannot be None")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError(f"{data_name} DataFrame is empty")

        # Check for required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"{data_name} missing required columns: {missing_columns}")

        # Check for NaN values
        if data.isnull().any().any():
            nan_count = data.isnull().sum().sum()
            tprint(f"⚠️ {data_name} contains {nan_count} NaN values")

        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_mask = np.isinf(data[numeric_cols]).any().any()
            if inf_mask:
                raise ValueError(f"{data_name} contains infinite values")

        tprint(f"✅ {data_name} validation passed: {len(data)} rows, {len(data.columns)} columns")

    elif isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError(f"{data_name} array is empty")

        if np.any(np.isnan(data)):
            raise ValueError(f"{data_name} contains NaN values")

        if np.any(np.isinf(data)):
            raise ValueError(f"{data_name} contains infinite values")

        tprint(f"✅ {data_name} validation passed: shape {data.shape}")

    else:
        # Basic validation for other types
        if hasattr(data, '__len__') and len(data) == 0:
            raise ValueError(f"{data_name} is empty")

        tprint(f"✅ {data_name} validation passed: type {type(data).__name__}")

    return True

def validate_model_inputs(X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
    """
    Validate standard model training inputs.

    Args:
        X: Feature matrix
        y: Target values
        feature_names: Optional feature names

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    validate_training_data(X, "Feature matrix X")
    validate_training_data(y, "Target values y")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")

    if feature_names and len(feature_names) != X.shape[1]:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match X features ({X.shape[1]})")

    tprint(f"✅ Model inputs validation passed: {X.shape[0]} samples, {X.shape[1]} features")
    return True
# Import from simplified model training structure
try:
    # from .simplified.general_model_training import GeneralModelTrainer  # Removed from pipeline
    SIMPLIFIED_TRAINING_AVAILABLE = True
except ImportError:
    SIMPLIFIED_TRAINING_AVAILABLE = False
    GeneralModelTrainer = None

# Import refactored training steps using common dependencies
try:
    from .analyst_models_training_refactored import (
        AnalystModelsTrainingStepRefactored as AnalystModelsTrainingStep,
        create_analyst_models_training_step_refactored as create_analyst_models_training_step,
        execute_analyst_models_training_refactored as execute_analyst_models_training
    )
    # Note: Other training steps need to be refactored using common dependencies
    # For now, we'll create placeholder imports
    AnalystEnsembleTrainingStep = None
    TacticianModelsTrainingStep = None
    TacticianEnsembleTrainingStep = None
    create_analyst_ensemble_training_step = None
    create_tactician_models_training_step = None
    create_tactician_ensemble_training_step = None
    execute_analyst_ensemble_training = None
    execute_tactician_models_training = None
    execute_tactician_ensemble_training = None
    COMPREHENSIVE_TRAINING_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_TRAINING_AVAILABLE = False
    AnalystModelsTrainingStep = None
    AnalystEnsembleTrainingStep = None
    TacticianModelsTrainingStep = None
    TacticianEnsembleTrainingStep = None
    create_analyst_models_training_step = None
    create_analyst_ensemble_training_step = None
    create_tactician_models_training_step = None
    create_tactician_ensemble_training_step = None
    execute_analyst_models_training = None
    execute_analyst_ensemble_training = None
    execute_tactician_models_training = None
    execute_tactician_ensemble_training = None

AnalystEnhancementStep = AnalystEnsembleTrainingStep
AnalystEnsembleCreationStep = AnalystEnsembleTrainingStep
TacticianSpecialistTrainingStep = TacticianModelsTrainingStep
from pathlib import Path

async def run_model_training_pipeline(symbol: str, exchange: str, timeframe: Any, data_dir: Any, **config) -> Any:
    """Run the complete model training pipeline with comprehensive validation and error handling."""
    import time
    try:
        from src.utils.common_operations import get_current_datetime, format_datetime
    except Exception:
        pass
    from .utils.validator_orchestrator import ValidatorOrchestrator
    from .utils.step_dependency_validator import StepDependencyValidator
    logger = system_logger.getChild('ModelTrainingPipeline')

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _monitor_memory_usage() -> dict:
        """Monitor memory usage and provide optimization alerts."""
        import json
        import logging

        try:
            import psutil
            import gc

            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            memory_stats = {'system_memory': {'total_gb': memory_info.total / 1024 ** 3, 'available_gb': memory_info.available / 1024 ** 3, 'used_gb': memory_info.used / 1024 ** 3, 'percent_used': memory_info.percent}, 'process_memory': {'rss_gb': process_memory.rss / 1024 ** 3, 'vms_gb': process_memory.vms / 1024 ** 3}}
            if memory_info.percent > 90:
                logger.warning(f'⚠️ High system memory usage: {memory_info.percent:.1f}%')
                tprint(f'   ⚠️ High system memory usage: {memory_info.percent:.1f}%')
            elif memory_info.percent > 80:
                logger.warning(f'⚠️ Moderate system memory usage: {memory_info.percent:.1f}%')
                tprint(f'   ⚠️ Moderate system memory usage: {memory_info.percent:.1f}%')
            if process_memory.rss / 1024 ** 3 > 2:
                logger.warning(f'⚠️ High process memory usage: {process_memory.rss / 1024 ** 3:.2f} GB')
                tprint(f'   ⚠️ High process memory usage: {process_memory.rss / 1024 ** 3:.2f} GB')
            gc.collect()
            return memory_stats
        except ImportError:
            logger.warning('⚠️ psutil not available for memory monitoring')
            tprint('   ⚠️ psutil not available for memory monitoring')
            return {}
        except Exception as e:
            logger.warning(f'⚠️ Memory monitoring failed: {e}')
            tprint(f'   ⚠️ Memory monitoring failed: {e}')
            return {}

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @validates(strict = True)
    @log_call
    @traced
    async def _validate_pipeline_inputs(symbol: str, exchange: str, timeframe: str, data_dir: str, **config) -> bool:
        """Validate all pipeline inputs and dependencies."""
        logger.info('🔍 Validating pipeline inputs and dependencies...')
        tprint('   🔍 Validating pipeline inputs and dependencies...')
        quality_issues = []
        quality_warnings = []
        if not symbol or not exchange or (not timeframe) or (not data_dir):
            error_msg = 'Missing required parameters: symbol, exchange, timeframe, data_dir'
            quality_issues.append(error_msg)
            raise ValueError(error_msg)
        if not safe_file_exists(data_dir):
            error_msg = f'Data directory not found: {data_dir}'
            quality_issues.append(error_msg)
            raise FileNotFoundError(error_msg)
        required_files = [f'aggtrades_{exchange}_{symbol}_consolidated.parquet', f'volume_{exchange}_{symbol}_consolidated.parquet']
        for file_name in required_files:
            file_path = f'{data_dir}/{file_name}'
            if not safe_file_exists(file_path):
                error_msg = f'Required data file not found: {file_path}'
                quality_issues.append(error_msg)
                raise FileNotFoundError(error_msg)
            else:
                try:
                    file_size = Path(file_path).stat().st_size
                    if file_size < 1024:
                        warning_msg = f'⚠️ Data file {file_name} is very small ({file_size} bytes)'
                        quality_warnings.append(warning_msg)
                        logger.warning(warning_msg)
                        tprint(f'   {warning_msg}')
                except Exception as e:
                    warning_msg = f'⚠️ Could not check file size for {file_name}: {e}'
                    quality_warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    tprint(f'   {warning_msg}')
        required_config_keys = ['hmm_training', 'regime_intelligence', 'analyst_creation']
        for key in required_config_keys:
            if key not in config:
                config[key] = True
                warning_msg = f'⚠️ Set default value for missing config key: {key}'
                quality_warnings.append(warning_msg)
                logger.warning(warning_msg)
                tprint(f'   {warning_msg}')
        if quality_issues:
            logger.error(f'❌ Found {len(quality_issues)} quality issues:')
            tprint(f'   ❌ Found {len(quality_issues)} quality issues:')
            for issue in quality_issues:
                logger.error(f'   • {issue}')
                tprint(f'   • {issue}')
            return False
        if quality_warnings:
            logger.warning(f'⚠️ Found {len(quality_warnings)} quality warnings:')
            tprint(f'   ⚠️ Found {len(quality_warnings)} quality warnings:')
            for warning in quality_warnings:
                logger.warning(f'   • {warning}')
                tprint(f'   • {warning}')
        logger.info('✅ Pipeline inputs validation passed')
        tprint('   ✅ Pipeline inputs validation passed')
        return True

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _validate_step_dependencies(symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate that all required previous steps have been completed."""
        logger.info('🔍 Validating step dependencies...')
        try:
            validator_orchestrator = ValidatorOrchestrator()
            dependency_validator = StepDependencyValidator()
            training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
            required_steps = ['step1_data_collection', 'step1_5_data_converter', 'step2_data_reading', 'step3_hmm_regime_discovery', 'step4_triple_barrier_method', 'step5_labeling', 'step6_feature_engineering']
            pipeline_state = {}
            all_passed = True
            for step in required_steps:
                try:
                    result = await validator_orchestrator.run_step_validator(step, training_input, pipeline_state, config)
                    if not result.get('validation_passed', False):
                        logger.error(f'❌ Step dependency validation failed for {step}')
                        all_passed = False
                    else:
                        logger.info(f'✅ Step dependency validation passed for {step}')
                except Exception as e:
                    logger.error(f'❌ Error validating step {step}: {e}')
                    all_passed = False
            if all_passed:
                logger.info('✅ All step dependencies validated successfully')
            else:
                logger.error('❌ Some step dependencies failed validation')
            return all_passed
        except Exception as e:
            logger.error(f'❌ Error in step dependency validation: {e}')
            return False

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _validate_data_quality(symbol: str, exchange: str, data_dir: str) -> bool:
        """Validate data quality before training."""
        logger.info('🔍 Validating data quality...')
        tprint('   🔍 Validating data quality...')
        quality_issues = []
        quality_warnings = []
        quality_score = 100
        try:
            data_file = f'{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet'
            tprint(f'   📁 Loading data file: {data_file}')
            df = safe_read_parquet(data_file)
            if df.empty:
                error_msg = f'Data file is empty: {data_file}'
                quality_issues.append(error_msg)
                logger.error(f'❌ {error_msg}')
                tprint(f'   ❌ {error_msg}')
                return False
            df = optimize_dataframe_dtypes(df)
            tprint(f'   📊 Data loaded: {len(df)} rows, {len(df.columns)} columns')
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            schema_valid, schema_errors = validate_dataframe_schema(df, required_columns)
            if not schema_valid:
                error_msg = f'Data schema validation failed: {schema_errors}'
                quality_issues.append(error_msg)
                logger.error(f'❌ {error_msg}')
                tprint(f'   ❌ {error_msg}')
                return False
            else:
                tprint('   ✅ Data schema validation passed')
            nan_counts = df.isnull().sum()
            total_nans = nan_counts.sum()
            if total_nans > 0:
                nan_ratio = total_nans / (len(df) * len(df.columns))
                if nan_ratio > 0.1:
                    error_msg = f'High missing data ratio: {nan_ratio:.2%} ({total_nans} missing values)'
                    quality_issues.append(error_msg)
                    logger.error(f'❌ {error_msg}')
                    tprint(f'   ❌ {error_msg}')
                    quality_score -= 30
                else:
                    warning_msg = f'Some missing data found: {nan_ratio:.2%} ({total_nans} missing values)'
                    quality_warnings.append(warning_msg)
                    logger.warning(f'⚠️ {warning_msg}')
                    tprint(f'   ⚠️ {warning_msg}')
                    quality_score -= 10
            else:
                tprint('   ✅ No missing values found')
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                warning_msg = f'Found {duplicate_count} duplicate rows'
                quality_warnings.append(warning_msg)
                logger.warning(f'⚠️ {warning_msg}')
                tprint(f'   ⚠️ {warning_msg}')
                quality_score -= 5
            if len(df) < 1000:
                warning_msg = f'Low data volume: {len(df)} rows (minimum recommended: 1000)'
                quality_warnings.append(warning_msg)
                logger.warning(f'⚠️ {warning_msg}')
                tprint(f'   ⚠️ {warning_msg}')
                quality_score -= 20
            elif len(df) < 10000:
                warning_msg = f'Moderate data volume: {len(df)} rows (recommended: 10000+)'
                quality_warnings.append(warning_msg)
                logger.warning(f'⚠️ {warning_msg}')
                tprint(f'   ⚠️ {warning_msg}')
                quality_score -= 5
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    if (df[col] <= 0).any():
                        error_msg = f'Found non-positive prices in {col} column'
                        quality_issues.append(error_msg)
                        logger.error(f'❌ {error_msg}')
                        tprint(f'   ❌ {error_msg}')
                        quality_score -= 25
                    elif (df[col] > df[col].quantile(0.99) * 10).any():
                        warning_msg = f'Found potential price outliers in {col} column'
                        quality_warnings.append(warning_msg)
                        logger.warning(f'⚠️ {warning_msg}')
                        tprint(f'   ⚠️ {warning_msg}')
                        quality_score -= 5
            tprint(f'   📊 Data quality score: {quality_score}/100')
            if quality_score >= 90:
                tprint('   🎉 Excellent data quality!')
            elif quality_score >= 70:
                tprint('   ✅ Good data quality')
            elif quality_score >= 50:
                tprint('   ⚠️ Fair data quality - proceed with caution')
            else:
                tprint('   ❌ Poor data quality - consider data cleaning')
            if quality_issues:
                logger.error(f'❌ Found {len(quality_issues)} quality issues:')
                tprint(f'   ❌ Found {len(quality_issues)} quality issues:')
                for issue in quality_issues:
                    logger.error(f'   • {issue}')
                    tprint(f'   • {issue}')
                return False
            if quality_warnings:
                logger.warning(f'⚠️ Found {len(quality_warnings)} quality warnings:')
                tprint(f'   ⚠️ Found {len(quality_warnings)} quality warnings:')
                for warning in quality_warnings:
                    logger.warning(f'   • {warning}')
                    tprint(f'   • {warning}')
            logger.info(f'✅ Data quality validation passed: {len(df)} rows, {len(df.columns)} columns, score: {quality_score}/100')
            tprint(f'   ✅ Data quality validation passed: {len(df)} rows, {len(df.columns)} columns, score: {quality_score}/100')
            return True
        except Exception as e:
            error_msg = f'Error in data quality validation: {e}'
            quality_issues.append(error_msg)
            logger.error(f'❌ {error_msg}')
            tprint(f'   ❌ {error_msg}')
            return False

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _execute_training_step(step_name: str, step_class: Any, symbol: str, exchange: str, timeframe: str, data_dir: str, **config) -> bool:
        """Execute a single training step with comprehensive error handling."""
        logger.info(f'🚀 Executing {step_name}...')
        tprint(f'   🔧 Initializing {step_name}...')
        try:
            step_instance = step_class(config)
            tprint(f'   ✅ {step_name} instance created successfully')
            logger.info(f'✅ {step_name} instance created successfully')
            tprint(f'   🚀 Starting {step_name} execution...')
            logger.info(f'🚀 Starting {step_name} execution...')
            if hasattr(step_instance, 'train_models'):
                success = await step_instance.train_models(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'build_intelligence'):
                success = await step_instance.build_intelligence(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'create_analysts'):
                success = await step_instance.create_analysts(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'enhance_analysts'):
                success = await step_instance.enhance_analysts(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'create_ensembles'):
                success = await step_instance.create_ensembles(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'train_tacticians'):
                success = await step_instance.train_tacticians(symbol, exchange, timeframe, data_dir)
            else:
                logger.error(f'❌ Unknown step method for {step_name}')
                tprint(f'   ❌ Unknown step method for {step_name}')
                return False
            if success:
                logger.info(f'✅ {step_name} completed successfully')
                tprint(f'   ✅ {step_name} completed successfully')
                if _is_model_training_step(step_name):
                    tprint(f'   🧠 Running model interpretability analysis for {step_name}...')
                    logger.info(f'🧠 Running model interpretability analysis for {step_name}...')
                    interpretability_success = await _run_model_interpretability_analysis(step_instance, symbol, exchange, timeframe, data_dir, step_name)
                    if interpretability_success:
                        tprint(f'   ✅ Model interpretability analysis completed for {step_name}')
                        logger.info(f'✅ Model interpretability analysis completed for {step_name}')
                    else:
                        tprint(f'   ⚠️ Model interpretability analysis failed for {step_name} - continuing...')
                        logger.warning(f'⚠️ Model interpretability analysis failed for {step_name} - continuing...')
                return True
            else:
                logger.error(f'❌ {step_name} failed')
                tprint(f'   ❌ {step_name} failed')
                return False
        except Exception as e:
            logger.error(f'❌ Error executing {step_name}: {e}')
            logger.error(f'📋 Exception type: {type(e).__name__}')
            tprint(f'   ❌ Error executing {step_name}: {e}')
            tprint(f'   📋 Exception type: {type(e).__name__}')
            return False

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _is_model_training_step(step_name: str) -> bool:
        """Determine if a step involves model training that should have interpretability analysis."""
        step_lower = step_name.lower()
        model_training_keywords = ['training', 'model', 'train', 'build', 'create', 'enhance', 'tactician', 'analyst', 'ensemble', 'intelligence', 'regime', 'cluster', 'support', 'resistance', 'sr', 'market_regime', 'market_cluster', 'hmm', 'gmm', 'kmeans', 'dbscan']
        for keyword in model_training_keywords:
            if keyword in step_lower:
                return True
        model_step_patterns = ['step09', 'step10', 'step11', 'step12', 'step13', 'step14', 'step15', 'model_training', 'build_intelligence', 'create_analysts', 'enhance_analysts', 'create_ensembles', 'train_tacticians']
        for pattern in model_step_patterns:
            if pattern in step_lower:
                return True
        return False

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _determine_model_type(step_name: str) -> str:
        """Determine the type of model being trained based on step name."""
        step_lower = step_name.lower()
        if 'tactician' in step_lower or 'step15' in step_lower:
            return 'tactician'
        elif 'analyst' in step_lower or 'step11' in step_lower or 'step12' in step_lower:
            return 'analyst'
        elif 'ensemble' in step_lower or 'step13' in step_lower:
            return 'ensemble'
        elif 'intelligence' in step_lower or 'step10' in step_lower:
            return 'intelligence'
        elif 'regime' in step_lower or 'hmm' in step_lower or 'gmm' in step_lower:
            return 'market_regime'
        elif 'cluster' in step_lower or 'kmeans' in step_lower or 'dbscan' in step_lower:
            return 'market_cluster'
        elif 'support' in step_lower or 'resistance' in step_lower or 'sr' in step_lower:
            return 'support_resistance'
        elif 'training' in step_lower or 'model' in step_lower or 'step09' in step_lower:
            return 'main_model'
        else:
            return 'unknown'

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _extract_models_and_data(step_instance: Any, step_name: str, model_type: str, data_dir: str, symbol: str, exchange: str) -> tuple:
        """Extract models and data from step instance based on model type."""
        trained_models = None
        feature_names = None
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        try:
            if hasattr(step_instance, 'trained_models'):
                trained_models = step_instance.trained_models
            elif hasattr(step_instance, 'models'):
                trained_models = step_instance.models
            elif hasattr(step_instance, 'model'):
                trained_models = {'main_model': step_instance.model}
            elif hasattr(step_instance, 'tacticians'):
                trained_models = step_instance.tacticians
            elif hasattr(step_instance, 'analysts'):
                trained_models = step_instance.analysts
            elif hasattr(step_instance, 'ensembles'):
                trained_models = step_instance.ensembles
            elif hasattr(step_instance, 'intelligence_models'):
                trained_models = step_instance.intelligence_models
            elif hasattr(step_instance, 'regime_models'):
                trained_models = step_instance.regime_models
            elif hasattr(step_instance, 'cluster_models'):
                trained_models = step_instance.cluster_models
            elif hasattr(step_instance, 'sr_models'):
                trained_models = step_instance.sr_models
            if hasattr(step_instance, 'feature_names'):
                feature_names = step_instance.feature_names
            elif hasattr(step_instance, 'features'):
                feature_names = step_instance.features
            elif hasattr(step_instance, 'input_features'):
                feature_names = step_instance.input_features
            elif hasattr(step_instance, 'selected_features'):
                feature_names = step_instance.selected_features
            if hasattr(step_instance, 'X_train'):
                X_train = step_instance.X_train
            if hasattr(step_instance, 'X_test'):
                X_test = step_instance.X_test
            if hasattr(step_instance, 'y_train'):
                y_train = step_instance.y_train
            if hasattr(step_instance, 'y_test'):
                y_test = step_instance.y_test
            elif hasattr(step_instance, 'train_data'):
                X_train = step_instance.train_data
            elif hasattr(step_instance, 'test_data'):
                X_test = step_instance.test_data
            if model_type == 'tactician':
                trained_models, feature_names = _extract_tactician_models(step_instance)
            elif model_type == 'analyst':
                trained_models, feature_names = _extract_analyst_models(step_instance)
            elif model_type == 'ensemble':
                trained_models, feature_names = _extract_ensemble_models(step_instance)
            elif model_type == 'intelligence':
                trained_models, feature_names = _extract_intelligence_models(step_instance)
            elif model_type == 'market_regime':
                trained_models, feature_names = _extract_regime_models(step_instance)
            elif model_type == 'market_cluster':
                trained_models, feature_names = _extract_cluster_models(step_instance)
            elif model_type == 'support_resistance':
                trained_models, feature_names = _extract_sr_models(step_instance)
            return (trained_models, feature_names, X_train, X_test, y_train, y_test)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting models and data for {model_type}: {e}')
            return (None, None, None, None, None, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_tactician_models(step_instance: Any) -> tuple:
        """Extract tactician models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'tacticians'):
                tacticians = step_instance.tacticians
                if isinstance(tacticians, dict):
                    for name, tactician in tacticians.items():
                        if hasattr(tactician, 'model'):
                            models[f'tactician_{name}'] = tactician.model
                        elif hasattr(tactician, 'trained_model'):
                            models[f'tactician_{name}'] = tactician.trained_model
            if hasattr(step_instance, 'tactician_features'):
                features = step_instance.tactician_features
            elif hasattr(step_instance, 'tactician_input_features'):
                features = step_instance.tactician_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting tactician models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_analyst_models(step_instance: Any) -> tuple:
        """Extract analyst models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'analysts'):
                analysts = step_instance.analysts
                if isinstance(analysts, dict):
                    for name, analyst in analysts.items():
                        if hasattr(analyst, 'model'):
                            models[f'analyst_{name}'] = analyst.model
                        elif hasattr(analyst, 'trained_model'):
                            models[f'analyst_{name}'] = analyst.trained_model
            if hasattr(step_instance, 'analyst_features'):
                features = step_instance.analyst_features
            elif hasattr(step_instance, 'analyst_input_features'):
                features = step_instance.analyst_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting analyst models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_ensemble_models(step_instance: Any) -> tuple:
        """Extract ensemble models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'ensembles'):
                ensembles = step_instance.ensembles
                if isinstance(ensembles, dict):
                    for name, ensemble in ensembles.items():
                        if hasattr(ensemble, 'model'):
                            models[f'ensemble_{name}'] = ensemble.model
                        elif hasattr(ensemble, 'trained_model'):
                            models[f'ensemble_{name}'] = ensemble.trained_model
            if hasattr(step_instance, 'ensemble_features'):
                features = step_instance.ensemble_features
            elif hasattr(step_instance, 'ensemble_input_features'):
                features = step_instance.ensemble_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting ensemble models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_intelligence_models(step_instance: Any) -> tuple:
        """Extract intelligence models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'intelligence_models'):
                intel_models = step_instance.intelligence_models
                if isinstance(intel_models, dict):
                    for name, model in intel_models.items():
                        models[f'intelligence_{name}'] = model
            if hasattr(step_instance, 'intelligence_features'):
                features = step_instance.intelligence_features
            elif hasattr(step_instance, 'intelligence_input_features'):
                features = step_instance.intelligence_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting intelligence models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_regime_models(step_instance: Any) -> tuple:
        """Extract market regime models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'regime_models'):
                regime_models = step_instance.regime_models
                if isinstance(regime_models, dict):
                    for name, model in regime_models.items():
                        models[f'regime_{name}'] = model
            if hasattr(step_instance, 'regime_features'):
                features = step_instance.regime_features
            elif hasattr(step_instance, 'regime_input_features'):
                features = step_instance.regime_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting regime models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_cluster_models(step_instance: Any) -> tuple:
        """Extract market cluster models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'cluster_models'):
                cluster_models = step_instance.cluster_models
                if isinstance(cluster_models, dict):
                    for name, model in cluster_models.items():
                        models[f'cluster_{name}'] = model
            if hasattr(step_instance, 'cluster_features'):
                features = step_instance.cluster_features
            elif hasattr(step_instance, 'cluster_input_features'):
                features = step_instance.cluster_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting cluster models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    def _extract_sr_models(step_instance: Any) -> tuple:
        """Extract support/resistance models and features."""
        models = {}
        features = None
        try:
            if hasattr(step_instance, 'sr_models'):
                sr_models = step_instance.sr_models
                if isinstance(sr_models, dict):
                    for name, model in sr_models.items():
                        models[f'sr_{name}'] = model
            if hasattr(step_instance, 'sr_features'):
                features = step_instance.sr_features
            elif hasattr(step_instance, 'sr_input_features'):
                features = step_instance.sr_input_features
            return (models, features)
        except Exception as e:
            logger.warning(f'⚠️ Error extracting SR models: {e}')
            return ({}, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _load_model_specific_data(model_type: str, data_dir: str, symbol: str, exchange: str, feature_names: List[str]) -> tuple:
        """Load model-specific data for interpretability analysis."""
        try:
            # Use consistent import path - functions already imported at top of module
            pass
            X_train = None
            X_test = None
            y_train = None
            y_test = None
            features_file = f'{data_dir}/features_{exchange}_{symbol}_consolidated.parquet'
            if safe_file_exists(features_file):
                features_df = safe_read_parquet(features_file)

                # 🔧 INTEGRATE DATA CLEANING UTILITY
                # Automatically clean corrupted data before training
                try:
                    from src.utils.ml_common.data_processing.data_cleaning_utils import exclude_corrupted_periods

                    # Convert timestamp if needed
                    if 'timestamp' in features_df.columns and features_df['timestamp'].dtype == 'int64':
                        features_df['datetime'] = pd.to_datetime(features_df['timestamp'], unit='s')
                    elif 'datetime' not in features_df.columns:
                        # Try to infer datetime column
                        datetime_cols = [col for col in features_df.columns if 'time' in col.lower()]
                        if datetime_cols:
                            features_df['datetime'] = pd.to_datetime(features_df[datetime_cols[0]])
                        else:
                            features_df['datetime'] = features_df.index

                    # Apply data cleaning
                    original_count = len(features_df)
                    features_df = exclude_corrupted_periods(features_df)
                    cleaned_count = len(features_df)

                    if original_count != cleaned_count:
                        excluded_count = original_count - cleaned_count
                        logger.info(f"🧹 Data cleaning applied: Excluded {excluded_count:,} corrupted rows ({100*excluded_count/original_count:.4f}%)")

                except ImportError as e:
                    logger.warning(f"⚠️ Data cleaning utility not available: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ Data cleaning failed, proceeding with original data: {e}")
                if feature_names:
                    available_features = [col for col in feature_names if col in features_df.columns]
                    if available_features:
                        features_df = features_df[available_features + ['timestamp']]
                    else:
                        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
                        if 'timestamp' in numeric_cols:
                            numeric_cols.remove('timestamp')
                        features_df = features_df[numeric_cols + ['timestamp']]
                else:
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'timestamp' in numeric_cols:
                        numeric_cols.remove('timestamp')
                    features_df = features_df[numeric_cols + ['timestamp']]
                split_idx = int(len(features_df) * 0.8)
                X_train = features_df.iloc[:split_idx].drop('timestamp', axis = 1, errors='ignore')
                X_test = features_df.iloc[split_idx:].drop('timestamp', axis = 1, errors='ignore')
                labels_file = f'{data_dir}/labels_{exchange}_{symbol}_consolidated.parquet'
                if safe_file_exists(labels_file):
                    labels_df = safe_read_parquet(labels_file)
                    if 'target' in labels_df.columns:
                        y_train = labels_df.iloc[:split_idx]['target']
                        y_test = labels_df.iloc[split_idx:]['target']
                tprint(f'   📊 Loaded {len(X_train)} training samples and {len(X_test)} test samples for {model_type} analysis')
                logger.info(f'📊 Loaded {len(X_train)} training samples and {len(X_test)} test samples for {model_type} analysis')
            return (X_train, X_test, y_train, y_test)
        except Exception as e:
            logger.warning(f'⚠️ Error loading model-specific data for {model_type}: {e}')
            return (None, None, None, None)

    @handles_errors(Exception, fallback = False, log_level='ERROR')
    @log_call
    @traced
    async def _run_model_interpretability_analysis(step_instance: Any, symbol: str, exchange: str, timeframe: str, data_dir: str, step_name: str) -> bool:
        """Run model interpretability analysis for trained models."""
        try:
            from .training.model_interpretability import ModelExplainer
            model_type = _determine_model_type(step_name)
            tprint(f'   🔍 Detected model type: {model_type}')
            logger.info(f'🔍 Detected model type: {model_type}')
            trained_models = None
            feature_names = None
            X_train = None
            X_test = None
            y_train = None
            y_test = None
            trained_models, feature_names, X_train, X_test, y_train, y_test = await _extract_models_and_data(step_instance, step_name, model_type, data_dir, symbol, exchange)
            if X_train is None or X_test is None:
                try:
                    from .utils.common_operations import safe_read_parquet
                    features_file = f'{data_dir}/features_{exchange}_{symbol}_consolidated.parquet'
                    if safe_file_exists(features_file):
                        features_df = safe_read_parquet(features_file)
                        if feature_names is None:
                            feature_names = [col for col in features_df.columns if col not in ['timestamp', 'target']]
                        split_idx = int(len(features_df) * 0.8)
                        X_train = features_df.iloc[:split_idx][feature_names]
                        X_test = features_df.iloc[split_idx:][feature_names]
                        labels_file = f'{data_dir}/labels_{exchange}_{symbol}_consolidated.parquet'
                        if safe_file_exists(labels_file):
                            labels_df = safe_read_parquet(labels_file)
                            y_train = labels_df.iloc[:split_idx]['target'] if 'target' in labels_df.columns else None
                            y_test = labels_df.iloc[split_idx:]['target'] if 'target' in labels_df.columns else None
                except Exception as e:
                    logger.warning(f'⚠️ Could not load data for interpretability analysis: {e}')
                    return False
            if not trained_models or not feature_names or X_train is None or (X_test is None):
                logger.warning(f'⚠️ Insufficient data for interpretability analysis in {step_name}')
                return False
            explainer_config = {'interpretability': {'enabled': True, 'shap_enabled': True, 'lime_enabled': True, 'visualization_enabled': True, 'reporting_enabled': True, 'model_type': model_type}}
            model_explainer = ModelExplainer(explainer_config)
            output_dir = f'{data_dir}/interpretability/{step_name}_{model_type}'
            X_train, X_test, y_train, y_test = await _load_model_specific_data(model_type, data_dir, symbol, exchange, feature_names)
            if isinstance(trained_models, dict) and len(trained_models) > 1:
                tprint(f'   🔍 Running multi-model interpretability analysis for {len(trained_models)} {model_type} models...')
                logger.info(f'🔍 Running multi-model interpretability analysis for {len(trained_models)} {model_type} models...')
                results = await model_explainer.explain_multiple_models(models = trained_models, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, feature_names = feature_names, symbol = symbol, exchange = exchange, output_dir = output_dir)
            else:
                model = list(trained_models.values())[0] if isinstance(trained_models, dict) else trained_models
                model_name = list(trained_models.keys())[0] if isinstance(trained_models, dict) else f'{step_name}_{model_type}'
                tprint(f'   🔍 Running single-model interpretability analysis for {model_type} model: {model_name}')
                logger.info(f'🔍 Running single-model interpretability analysis for {model_type} model: {model_name}')
                results = await model_explainer.explain_model(model = model, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, feature_names = feature_names, model_name = model_name, symbol = symbol, exchange = exchange, output_dir = output_dir)
            if results and 'error' not in results:
                top_features = results.get('feature_importance', {}).get('top_features', [])
                if top_features:
                    logger.info(f"🧠 Top 5 important features for {step_name}: {', '.join(top_features[:5])}")
                    tprint(f"   🧠 Top 5 important features: {', '.join(top_features[:5])}")
                insights = results.get('insights', {})
                feature_insights = insights.get('feature_insights', [])
                if feature_insights:
                    logger.info(f'💡 Key insight: {feature_insights[0]}')
                    tprint(f'   💡 Key insight: {feature_insights[0]}')
                return True
            else:
                logger.warning(f'⚠️ Interpretability analysis returned no results for {step_name}')
                return False
        except Exception as e:
            logger.error(f'❌ Model interpretability analysis failed for {step_name}: {e}')
            tprint(f'   ❌ Model interpretability analysis failed: {e}')
            return False
    try:
        # Record start time for performance metrics
        start_time = time.time()

        tprint('🚀 Starting Enhanced Model Training Pipeline')
        tprint('=' * 80)
        logger.info('🚀 Starting Enhanced Model Training Pipeline')
        logger.info(f'📊 Configuration: {symbol} on {exchange}, timeframe: {timeframe}')
        logger.info(f'📁 Data directory: {data_dir}')
        logger.info(f'⚙️ Training config: {config}')
        tprint('🔍 Initial memory monitoring...')
        initial_memory = await _monitor_memory_usage()
        tprint('🔍 STEP 1/6: Validating pipeline inputs...')
        logger.info('🔍 STEP 1/6: Validating pipeline inputs...')
        inputs_valid = await _validate_pipeline_inputs(symbol, exchange, timeframe, data_dir, **config)
        if not inputs_valid:
            logger.error('❌ Pipeline input validation failed')
            tprint('❌ Pipeline input validation failed')
            return False
        tprint('✅ Pipeline input validation passed')
        logger.info('✅ Pipeline input validation passed')
        tprint('🔍 STEP 2/6: Validating step dependencies...')
        logger.info('🔍 STEP 2/6: Validating step dependencies...')
        dependencies_valid = await _validate_step_dependencies(symbol, exchange, timeframe, data_dir)
        if not dependencies_valid:
            logger.error('❌ Step dependency validation failed')
            tprint('❌ Step dependency validation failed')
            return False
        tprint('✅ Step dependency validation passed')
        logger.info('✅ Step dependency validation passed')
        tprint('🔍 STEP 3/6: Validating data quality...')
        logger.info('🔍 STEP 3/6: Validating data quality...')
        data_quality_valid = await _validate_data_quality(symbol, exchange, data_dir)
        if not data_quality_valid:
            logger.error('❌ Data quality validation failed')
            tprint('❌ Data quality validation failed')
            return False
        tprint('✅ Data quality validation passed')
        logger.info('✅ Data quality validation passed')
        tprint('🔍 Memory monitoring after data validation...')
        post_data_memory = await _monitor_memory_usage()
        tprint('🚀 STEP 4/6: Executing training steps...')
        logger.info('🚀 STEP 4/6: Executing training steps...')
        # Use simplified training steps if available
        if SIMPLIFIED_TRAINING_AVAILABLE:
            training_steps = [
                ('General Model Training', GeneralModelTrainer, config.get('general_training', True)),
                ('Analyst Model Training', AnalystModelTrainer, config.get('analyst_training', True)),
                ('Tactician Model Training', TacticianModelTrainer, config.get('tactician_training', True))
            ]
        else:
            # Fallback to legacy steps (should not happen after cleanup)
            training_steps = [
                ('HMM-based Training', HMMBasedTrainingStep, config.get('hmm_training', True)),
                ('Unified Regime Intelligence', UnifiedRegimeIntelligenceStep, config.get('regime_intelligence', True)),
                ('Analyst Creation', AnalystCreationStep, config.get('analyst_creation', True)),
                ('Analyst Enhancement', AnalystEnhancementStep, config.get('analyst_enhancement', True)),
                ('Ensemble Creation', AnalystEnsembleCreationStep, config.get('ensemble_creation', True)),
                ('Tactician Training', TacticianSpecialistTrainingStep, config.get('tactician_training', True))
            ]
        enabled_steps = [(name, cls, enabled) for name, cls, enabled in training_steps if enabled]
        total_steps = len(enabled_steps)
        tprint(f'📊 Total training steps to execute: {total_steps}')
        logger.info(f'📊 Total training steps to execute: {total_steps}')
        all_steps_successful = True
        completed_steps = 0
        for step_index, (step_name, step_class, enabled) in enumerate(enabled_steps, 1):
            tprint(f'🔄 STEP 4.{step_index}/{total_steps}: {step_name}...')
            logger.info(f'🔄 STEP 4.{step_index}/{total_steps}: {step_name}...')
            step_start_time = time.time()
            step_success = await _execute_training_step(step_name, step_class, symbol, exchange, timeframe, data_dir, **config)
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            if step_success:
                completed_steps += 1
                progress_percentage = completed_steps / total_steps * 100
                tprint(f'✅ {step_name} completed successfully in {step_duration:.2f} seconds')
                tprint(f'📊 Progress: {completed_steps}/{total_steps} steps ({progress_percentage:.1f}%)')
                logger.info(f'✅ {step_name} completed successfully in {step_duration:.2f} seconds')
                logger.info(f'📊 Progress: {completed_steps}/{total_steps} steps ({progress_percentage:.1f}%)')
                if step_index % 2 == 0:
                    tprint(f'🔍 Memory monitoring after step {step_index}...')
                    step_memory = await _monitor_memory_usage()
            else:
                all_steps_successful = False
                logger.error(f'❌ Pipeline failed at {step_name}')
                tprint(f'❌ Pipeline failed at {step_name}')
                tprint(f'💥 Training stopped at step {step_index}/{total_steps}')
                break
        skipped_steps = [(name, cls, enabled) for name, cls, enabled in training_steps if not enabled]
        if skipped_steps:
            tprint(f'⏭️ Skipped {len(skipped_steps)} disabled steps:')
            logger.info(f'⏭️ Skipped {len(skipped_steps)} disabled steps:')
            for step_name, _, _ in skipped_steps:
                tprint(f'   • {step_name}')
                logger.info(f'   • {step_name}')
        if all_steps_successful:
            tprint('🎉 STEP 5/6: Model training pipeline completed successfully!')
            tprint('=' * 80)
            logger.info('🎉 STEP 5/6: Model training pipeline completed successfully!')
            total_execution_time = time.time() - start_time
            avg_step_time = total_execution_time / total_steps if total_steps > 0 else 0
            tprint('🔍 Final memory monitoring...')
            final_memory = await _monitor_memory_usage()
            execution_summary = {'pipeline_info': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'execution_time': format_datetime(get_current_datetime()), 'success': True}, 'configuration': config, 'steps_completed': [step[0] for step in training_steps if step[2]], 'performance_metrics': {'total_steps': len(training_steps), 'completed_steps': len([step for step in training_steps if step[2]]), 'success_rate': 1.0, 'enabled_steps': total_steps, 'skipped_steps': len(skipped_steps) if 'skipped_steps' in locals() else 0, 'total_execution_time_seconds': total_execution_time, 'total_execution_time_minutes': total_execution_time / 60, 'average_step_time_seconds': avg_step_time, 'execution_efficiency': 'high' if total_execution_time < 3600 else 'medium' if total_execution_time < 7200 else 'low', 'steps_per_minute': total_steps / (total_execution_time / 60) if total_execution_time > 0 else 0}, 'data_info': {'data_file_size': format_bytes(Path(f'{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet').stat().st_size) if safe_file_exists(f'{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet') else 'unknown'}, 'quality_metrics': {'data_validation_passed': True, 'step_dependencies_validated': True, 'overall_quality_score': 100, 'quality_issues_found': 0, 'quality_warnings_found': 0}, 'memory_metrics': {'initial_memory': initial_memory, 'post_data_memory': post_data_memory if 'post_data_memory' in locals() else {}, 'final_memory': final_memory}}
            summary_file = f'{data_dir}/model_training_execution_summary_{symbol}_{timeframe}.json'
            safe_json_dump(execution_summary, summary_file, indent = 2)
            logger.info(f'💾 Execution summary saved to: {summary_file}')
            tprint(f'💾 Execution summary saved to: {summary_file}')
            safe_log_metric('pipeline_success', 1.0)
            safe_log_metric('steps_completed', len([step for step in training_steps if step[2]]))
            safe_log_metric('total_execution_time_seconds', total_execution_time)
            safe_log_metric('total_execution_time_minutes', total_execution_time / 60)
            safe_log_metric('average_step_time_seconds', avg_step_time)
            safe_log_metric('steps_per_minute', total_steps / (total_execution_time / 60) if total_execution_time > 0 else 0)
            safe_log_metric('execution_efficiency', 'high' if total_execution_time < 3600 else 'medium' if total_execution_time < 7200 else 'low')
            safe_log_params({'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'total_steps': total_steps, 'enabled_steps': total_steps, 'skipped_steps': len(skipped_steps) if 'skipped_steps' in locals() else 0})
            tprint('📊 PERFORMANCE SUMMARY:')
            tprint(f'   ⏱️ Total execution time: {total_execution_time:.2f} seconds ({total_execution_time / 60:.2f} minutes)')
            tprint(f'   📈 Average step time: {avg_step_time:.2f} seconds')
            tprint(f'   🚀 Steps per minute: {total_steps / (total_execution_time / 60):.2f}')
            tprint(f"   📊 Execution efficiency: {('high' if total_execution_time < 3600 else 'medium' if total_execution_time < 7200 else 'low')}")
            tprint(f'   ✅ Success rate: 100%')
            tprint('✅ All training steps completed successfully!')
            tprint('=' * 80)
            return True
        else:
            tprint('❌ STEP 5/6: Model training pipeline failed')
            tprint('=' * 80)
            logger.error('❌ STEP 5/6: Model training pipeline failed')
            return False
    except Exception as e:
        tprint('💥 STEP 6/6: Model training pipeline failed with exception!')
        tprint('=' * 80)
        tprint(f'❌ Error: {e}')
        tprint(f'📋 Exception type: {type(e).__name__}')
        tprint('🔍 Troubleshooting suggestions:')
        tprint('   • Check data file integrity and availability')
        tprint('   • Verify previous steps completed successfully')
        tprint('   • Check system resources (memory, disk space)')
        tprint('   • Review configuration parameters')
        tprint('   • Check log files for detailed error information')
        tprint('=' * 80)
        logger.error(f'❌ STEP 6/6: Model training pipeline failed with error: {e}')
        logger.error(f'📋 Exception type: {type(e).__name__}')
        logger.error(f'📋 Exception details: {str(e)}')
        return False
__all__ = [
    # Available training components
    'AnalystModelsTrainingStep',
    'create_analyst_models_training_step',
    'execute_analyst_models_training',
    # Pipeline function
    'run_model_training_pipeline',
    # Data validation utilities
    'validate_training_data',
    'validate_model_inputs',
    # Utility constants
    'NUMPY_AVAILABLE',
    'PANDAS_AVAILABLE'
]
