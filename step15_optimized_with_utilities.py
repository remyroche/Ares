"""
Step 15: Optimized Tactician Specialist Training Implementation with Proper Utility Integration

This optimized version integrates with the specified utility modules and core components:
- src/utils/common_operations.py
- src/utils/math_validation.py  
- src/utils/parquet_utils.py
- src/core/decorators/
- src/core/errors/
"""

import asyncio
import json
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import logging

# Core imports
from src.core.decorators import (
    handles_errors, timeout, circuit_breaker, cached, log_call, 
    log_execution_time, traced, validates, error_boundary
)
from src.core.errors import (
    ValidationError, DataIntegrityError, NotFoundError, 
    ServiceUnavailableError, AppError
)

# Utility imports
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_read_parquet, safe_to_parquet,
    ensure_directory, safe_file_exists, safe_mean, safe_std,
    safe_fillna, safe_copy, safe_deepcopy, optimize_dataframe_dtypes,
    validate_dataframe_schema, validate_data_quality, safe_gather,
    create_async_task, safe_float, safe_int, format_bytes,
    get_current_datetime, standardize_price_action_probabilities
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

# System imports
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards

class Step15ValidationError(ValidationError):
    """Custom validation error for Step15 operations."""
    pass

class Step15DataError(DataIntegrityError):
    """Custom data integrity error for Step15 operations."""
    pass

class Step15OptimizedValidator:
    """Fast fail validation system for Step15 using proper utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("Step15Validator")
        self.max_samples = safe_int(config.get('max_samples', 1000000))
        self.min_samples = safe_int(config.get('min_samples', 100))
        self.max_memory_mb = safe_int(config.get('max_memory_mb', 8192))
        
    @validates()
    @log_call()
    def validate_inputs(self, training_input: Dict[str, Any]) -> Tuple[bool, str]:
        """Fast fail input validation using proper error handling."""
        try:
            # Check required parameters
            if not training_input.get('symbol'):
                raise Step15ValidationError("Symbol is required")
            
            if not training_input.get('exchange'):
                raise Step15ValidationError("Exchange is required")
            
            # Check data directory
            data_dir = training_input.get('data_dir')
            if not data_dir or not safe_file_exists(data_dir):
                raise Step15ValidationError(f"Data directory not found: {data_dir}")
            
            # Check for training data files
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            labeled_data_dir = f'{data_dir}/tactician_labeled_data'
            labeled_file_parquet = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet'
            labeled_file_pickle = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl'
            
            if not safe_file_exists(labeled_file_parquet) and not safe_file_exists(labeled_file_pickle):
                raise Step15ValidationError(f"No training data found for {symbol} on {exchange}")
            
            return True, "Input validation passed"
            
        except Step15ValidationError:
            raise
        except Exception as e:
            raise Step15ValidationError(f"Input validation error: {str(e)}")
    
    @validates()
    @log_call()
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate training data quality using math validation utilities."""
        try:
            if data.empty:
                raise Step15DataError("Training data is empty")
            
            # Check data size using safe integer conversion
            data_length = safe_int(len(data))
            if data_length < self.min_samples:
                raise Step15DataError(f"Insufficient samples: {data_length} < {self.min_samples}")
            
            if data_length > self.max_samples:
                raise Step15DataError(f"Dataset too large: {data_length} > {self.max_samples}")
            
            # Use math validation for percentage calculations
            total_cells = safe_int(len(data) * len(data.columns))
            missing_cells = safe_int(data.isnull().sum().sum())
            missing_ratio = safe_divide(missing_cells, total_cells, default=0.0)
            
            if missing_ratio > 0.5:
                raise Step15DataError(f"Too much missing data: {missing_ratio:.2%}")
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise Step15DataError(f"Missing required columns: {missing_cols}")
            
            # Check for target column
            target_cols = ['tactician_label', 'label']
            if not any(col in data.columns for col in target_cols):
                raise Step15DataError("No target column found (tactician_label or label)")
            
            return True, "Data quality validation passed"
            
        except (Step15DataError, MathValidationError):
            raise
        except Exception as e:
            raise Step15DataError(f"Data quality validation error: {str(e)}")
    
    @validates()
    @log_call()
    def validate_model_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate model parameters using math validation utilities."""
        try:
            # Check XGBoost parameters
            if 'xgboost' in params:
                xgb_params = params['xgboost']
                n_estimators = safe_int(xgb_params.get('n_estimators', 0))
                if n_estimators <= 0:
                    raise Step15ValidationError("Invalid n_estimators for XGBoost")
                
                max_depth = safe_int(xgb_params.get('max_depth', 0))
                if max_depth <= 0:
                    raise Step15ValidationError("Invalid max_depth for XGBoost")
                
                learning_rate = safe_float(xgb_params.get('learning_rate', 0))
                if not validate_range(learning_rate, 0.0, 1.0, "learning_rate"):
                    raise Step15ValidationError("Invalid learning_rate for XGBoost")
            
            # Check LightGBM parameters
            if 'lightgbm' in params:
                lgb_params = params['lightgbm']
                n_estimators = safe_int(lgb_params.get('n_estimators', 0))
                if n_estimators <= 0:
                    raise Step15ValidationError("Invalid n_estimators for LightGBM")
                
                max_depth = safe_int(lgb_params.get('max_depth', 0))
                if max_depth <= 0:
                    raise Step15ValidationError("Invalid max_depth for LightGBM")
            
            return True, "Model parameters validation passed"
            
        except (Step15ValidationError, MathValidationError):
            raise
        except Exception as e:
            raise Step15ValidationError(f"Model parameters validation error: {str(e)}")
    
    @validates()
    @log_call()
    def validate_regime_data(self, regime_data: pd.DataFrame, regime_id: str) -> Tuple[bool, str]:
        """Validate regime-specific data using proper validation."""
        try:
            min_regime_samples = safe_int(self.config.get('min_regime_samples', 500))
            regime_length = safe_int(len(regime_data))
            
            if regime_length < min_regime_samples:
                raise Step15DataError(f"Regime {regime_id} has insufficient samples: {regime_length} < {min_regime_samples}")
            
            if 'composite_cluster_id' not in regime_data.columns:
                raise Step15DataError(f"Missing composite_cluster_id column for regime {regime_id}")
            
            return True, f"Regime {regime_id} validation passed"
            
        except (Step15DataError, MathValidationError):
            raise
        except Exception as e:
            raise Step15DataError(f"Regime validation error: {str(e)}")

class Step15OptimizedProcessor:
    """Optimized data processing for Step15 using proper utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("Step15Processor")
        self.parquet_utils = get_parquet_utils()
        self._optimized_data_cache = {}
        
    @cached()
    @log_call()
    def optimize_dataframe(self, data: pd.DataFrame, cache_key: str = None) -> pd.DataFrame:
        """Optimize dataframe with caching using common operations utilities."""
        if cache_key and cache_key in self._optimized_data_cache:
            self.logger.info(f"Using cached optimized dataframe: {cache_key}")
            return self._optimized_data_cache[cache_key]
        
        try:
            # Use safe copy and optimization utilities
            optimized_data = safe_copy(data, deep=True)
            
            # Use common operations optimization
            optimized_data = optimize_dataframe_dtypes(optimized_data)
            
            # Cache if key provided
            if cache_key:
                self._optimized_data_cache[cache_key] = optimized_data
                self.logger.info(f"Cached optimized dataframe: {cache_key}")
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"Dataframe optimization failed: {e}")
            return safe_copy(data, deep=True)
    
    @log_call()
    def vectorized_sr_enhancement(self, data: pd.DataFrame, sr_predictor) -> pd.DataFrame:
        """Vectorized S/R context enhancement using math validation utilities."""
        try:
            if data.empty or sr_predictor is None:
                return safe_copy(data, deep=True)
            
            self.logger.info("Applying vectorized S/R enhancement...")
            enhanced_data = safe_copy(data, deep=True)
            
            # Vectorized S/R feature calculation
            data_length = safe_int(len(data))
            sample_interval = max(1, safe_int(safe_divide(data_length, 100, default=1)))
            sample_indices = data.index[::sample_interval]
            
            # Pre-allocate arrays for vectorized operations
            n_samples = len(sample_indices)
            sr_features = {
                'sr_proximity': np.zeros(n_samples),
                'sr_confidence': np.zeros(n_samples),
                'breakout_probability': np.zeros(n_samples),
                'rebounce_probability': np.zeros(n_samples),
                'consolidation_probability': np.zeros(n_samples)
            }
            
            # Vectorized processing with math validation
            for i, idx in enumerate(sample_indices):
                try:
                    row = data.loc[idx]
                    current_price = safe_float(row['close'])
                    
                    # Simplified S/R calculation
                    lookback = min(200, data_length)
                    market_slice = data.loc[:idx].tail(lookback)
                    
                    if len(market_slice) < 20:
                        sr_features['sr_proximity'][i] = 0.0
                        sr_features['sr_confidence'][i] = 0.5
                        sr_features['breakout_probability'][i] = 0.33
                        sr_features['rebounce_probability'][i] = 0.33
                        sr_features['consolidation_probability'][i] = 0.34
                        continue
                    
                    # Calculate S/R features using safe math operations
                    high_price = safe_float(market_slice['high'].max())
                    low_price = safe_float(market_slice['low'].min())
                    price_range = safe_float(high_price - low_price)
                    
                    if price_range > 0:
                        proximity = min(abs(current_price - high_price), abs(current_price - low_price))
                        proximity_ratio = safe_divide(proximity, price_range, default=0.0)
                        sr_features['sr_proximity'][i] = safe_float(1.0 - proximity_ratio)
                        sr_features['sr_confidence'][i] = safe_float(0.7 + proximity_ratio * 0.3)
                    else:
                        sr_features['sr_proximity'][i] = 0.0
                        sr_features['sr_confidence'][i] = 0.5
                    
                    # Simple probability distribution using safe math
                    base_prob = 0.3
                    proximity_boost = sr_features['sr_proximity'][i] * 0.2
                    sr_features['breakout_probability'][i] = safe_float(base_prob + proximity_boost)
                    sr_features['rebounce_probability'][i] = safe_float(base_prob + proximity_boost)
                    consolidation_prob = safe_float(1.0 - sr_features['breakout_probability'][i] - sr_features['rebounce_probability'][i])
                    sr_features['consolidation_probability'][i] = max(0.0, consolidation_prob)
                    
                except (MathValidationError, ValueError) as e:
                    self.logger.debug(f"Error processing S/R features for index {idx}: {e}")
                    sr_features['sr_proximity'][i] = 0.0
                    sr_features['sr_confidence'][i] = 0.5
                    sr_features['breakout_probability'][i] = 0.33
                    sr_features['rebounce_probability'][i] = 0.33
                    sr_features['consolidation_probability'][i] = 0.34
            
            # Interpolate features to full dataset
            for feature_name, values in sr_features.items():
                feature_series = pd.Series(values, index=sample_indices)
                full_feature = feature_series.reindex(data.index).interpolate(method='linear').fillna(0.5)
                enhanced_data[f'sr_{feature_name}'] = full_feature
            
            self.logger.info(f"✅ Vectorized S/R enhancement completed: {len(enhanced_data)} samples")
            return enhanced_data
            
        except Exception as e:
            self.logger.exception(f"Vectorized S/R enhancement failed: {e}")
            return safe_copy(data, deep=True)

class Step15OptimizedTrainer:
    """Optimized model training for Step15 using proper utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("Step15Trainer")
        self.model_cache = {}
        
    @traced(span_name="train_models_concurrent")
    @log_execution_time()
    async def train_models_concurrent(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                    y_train: pd.Series, y_test: pd.Series, 
                                    symbol: str, exchange: str) -> Dict[str, Any]:
        """Train models concurrently using safe async operations."""
        try:
            self.logger.info("Starting concurrent model training...")
            
            # Define training tasks
            training_tasks = [
                ('lightgbm', self._train_lightgbm_optimized),
                ('xgboost', self._train_xgboost_optimized),
                ('random_forest', self._train_random_forest_optimized),
                ('calibrated_logistic', self._train_calibrated_logistic_optimized)
            ]
            
            # Execute training concurrently using safe_gather
            coroutines = [
                self._execute_training_task_async(task_name, train_method, X_train, X_test, y_train, y_test, symbol, exchange)
                for task_name, train_method in training_tasks
            ]
            
            results = await safe_gather(*coroutines, return_exceptions=True)
            
            # Collect successful results
            models = {}
            for i, result in enumerate(results):
                task_name = training_tasks[i][0]
                if isinstance(result, Exception):
                    self.logger.warning(f"{task_name} training failed: {result}")
                elif result:
                    models[task_name] = result
                    self.logger.info(f"✅ {task_name} training completed")
            
            self.logger.info(f"Concurrent training completed: {len(models)} models trained")
            return models
            
        except Exception as e:
            self.logger.exception(f"Concurrent training failed: {e}")
            return {}
    
    @log_call()
    async def _execute_training_task_async(self, task_name: str, train_method, 
                                         X_train: pd.DataFrame, X_test: pd.DataFrame,
                                         y_train: pd.Series, y_test: pd.Series,
                                         symbol: str, exchange: str) -> Any:
        """Execute training task asynchronously with proper error handling."""
        try:
            return await train_method(X_train, X_test, y_train, y_test, symbol, exchange)
        except Exception as e:
            self.logger.warning(f"Training task {task_name} failed: {e}")
            return None
    
    @log_call()
    async def _train_lightgbm_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                      y_train: pd.Series, y_test: pd.Series,
                                      symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized LightGBM training with proper validation."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score
            
            # Optimized parameters with validation
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': safe_int(31),
                'learning_rate': validate_range(safe_float(0.05), 0.0, 1.0, "learning_rate"),
                'feature_fraction': validate_range(safe_float(0.9), 0.0, 1.0, "feature_fraction"),
                'bagging_fraction': validate_range(safe_float(0.8), 0.0, 1.0, "bagging_fraction"),
                'bagging_freq': safe_int(5),
                'verbose': -1,
                'random_state': 42
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            # Predictions with safe math
            y_pred = model.predict(X_test) > 0.5
            accuracy = safe_float(accuracy_score(y_test, y_pred))
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'LightGBM',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': get_current_datetime().isoformat()
            }
            
        except (MathValidationError, ValueError) as e:
            self.logger.exception(f"LightGBM training failed: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"LightGBM training failed: {e}")
            return None
    
    @log_call()
    async def _train_xgboost_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series, y_test: pd.Series,
                                     symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized XGBoost training with proper validation."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score
            
            # Optimized parameters with math validation
            params = {
                'n_estimators': safe_int(200),
                'max_depth': safe_int(6),
                'learning_rate': validate_range(safe_float(0.05), 0.0, 1.0, "learning_rate"),
                'subsample': validate_range(safe_float(0.8), 0.0, 1.0, "subsample"),
                'colsample_bytree': validate_range(safe_float(0.8), 0.0, 1.0, "colsample_bytree"),
                'reg_alpha': validate_positive(safe_float(0.01), "reg_alpha"),
                'reg_lambda': validate_positive(safe_float(0.01), "reg_lambda"),
                'random_state': 42,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            
            # Create model
            model = xgb.XGBClassifier(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Predictions with safe math
            y_pred = model.predict(X_test)
            accuracy = safe_float(accuracy_score(y_test, y_pred))
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'XGBoost',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': get_current_datetime().isoformat()
            }
            
        except (MathValidationError, ValueError) as e:
            self.logger.exception(f"XGBoost training failed: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"XGBoost training failed: {e}")
            return None
    
    @log_call()
    async def _train_random_forest_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                           y_train: pd.Series, y_test: pd.Series,
                                           symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized Random Forest training with proper validation."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            # Optimized parameters with validation
            params = {
                'n_estimators': safe_int(200),
                'max_depth': safe_int(10),
                'min_samples_split': safe_int(5),
                'min_samples_leaf': safe_int(2),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Create and train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions with safe math
            y_pred = model.predict(X_test)
            accuracy = safe_float(accuracy_score(y_test, y_pred))
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'RandomForest',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': get_current_datetime().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"Random Forest training failed: {e}")
            return None
    
    @log_call()
    async def _train_calibrated_logistic_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                 y_train: pd.Series, y_test: pd.Series,
                                                 symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized Calibrated Logistic Regression training with proper validation."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            
            # Base model with validated parameters
            base_model = LogisticRegression(
                C=validate_positive(safe_float(1.0), "C"),
                max_iter=safe_int(1000),
                random_state=42,
                solver='liblinear'
            )
            
            # Calibrated model
            model = CalibratedClassifierCV(
                estimator=base_model,
                cv=safe_int(5),
                method='isotonic'
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions with safe math
            y_pred = model.predict(X_test)
            accuracy = safe_float(accuracy_score(y_test, y_pred))
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'CalibratedLogistic',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': get_current_datetime().isoformat()
            }
            
        except (MathValidationError, ValueError) as e:
            self.logger.exception(f"Calibrated Logistic training failed: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"Calibrated Logistic training failed: {e}")
            return None

class Step15OptimizedImplementation:
    """Optimized Step15 implementation with proper utility integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("Step15Optimized")
        self.validator = Step15OptimizedValidator(config)
        self.processor = Step15OptimizedProcessor(config)
        self.trainer = Step15OptimizedTrainer(config)
        self.parquet_utils = get_parquet_utils()
        
    @handles_errors(exceptions=(Step15ValidationError, Step15DataError, MathValidationError), 
                   default_return={'status': 'FAILED', 'error': 'Execution failed'})
    @timeout(3600)  # 1 hour timeout
    @traced(span_name="execute_optimized_step15")
    @log_execution_time()
    async def execute_optimized(self, training_input: Dict[str, Any], 
                              pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized Step15 with proper utility integration."""
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting optimized Step15 execution with utility integration...")
            
            # Fast fail validation
            is_valid, validation_msg = self.validator.validate_inputs(training_input)
            if not is_valid:
                raise Step15ValidationError(f"Input validation failed: {validation_msg}")
            
            self.logger.info(f"✅ Input validation passed: {validation_msg}")
            
            # Load and validate data using parquet utils
            labeled_data = await self._load_training_data_optimized(training_input)
            if labeled_data is None:
                raise Step15DataError("Failed to load training data")
            
            # Validate data quality
            is_valid, validation_msg = self.validator.validate_data_quality(labeled_data)
            if not is_valid:
                raise Step15DataError(f"Data quality validation failed: {validation_msg}")
            
            self.logger.info(f"✅ Data quality validation passed: {validation_msg}")
            
            # Optimize data processing
            cache_key = f"{training_input['symbol']}_{training_input['exchange']}"
            labeled_data = self.processor.optimize_dataframe(labeled_data, cache_key)
            
            # Prepare training data
            X_train, X_test, y_train, y_test = await self._prepare_training_data_optimized(labeled_data)
            
            # Train models concurrently
            training_results = await self.trainer.train_models_concurrent(
                X_train, X_test, y_train, y_test,
                training_input['symbol'], training_input['exchange']
            )
            
            if not training_results:
                raise Step15DataError("No models were successfully trained")
            
            # Save results using proper utilities
            models_dir = await self._save_training_results_optimized(training_results, training_input)
            
            execution_time = safe_float(time.time() - start_time)
            
            self.logger.info(f"✅ Optimized Step15 execution completed in {execution_time:.2f}s")
            
            return {
                'status': 'SUCCESS',
                'tactician_models': training_results,
                'models_dir': models_dir,
                'duration': execution_time,
                'models_trained': safe_int(len(training_results))
            }
            
        except (Step15ValidationError, Step15DataError, MathValidationError) as e:
            execution_time = safe_float(time.time() - start_time)
            self.logger.error(f"❌ Optimized Step15 execution failed after {execution_time:.2f}s: {e}")
            return {'status': 'FAILED', 'error': str(e), 'duration': execution_time}
        except Exception as e:
            execution_time = safe_float(time.time() - start_time)
            self.logger.error(f"❌ Unexpected error in Step15 execution after {execution_time:.2f}s: {e}")
            return {'status': 'FAILED', 'error': str(e), 'duration': execution_time}
    
    @log_call()
    async def _load_training_data_optimized(self, training_input: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load training data using parquet utils with proper error handling."""
        try:
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            data_dir = training_input['data_dir']
            
            labeled_data_dir = f'{data_dir}/tactician_labeled_data'
            labeled_file_parquet = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet'
            labeled_file_pickle = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl'
            
            # Try parquet first using parquet utils
            if safe_file_exists(labeled_file_parquet):
                try:
                    data = self.parquet_utils.safe_read_parquet(labeled_file_parquet)
                    if data is not None:
                        self.logger.info(f"✅ Loaded training data from parquet: {len(data)} samples")
                        return data
                except Exception as e:
                    self.logger.warning(f"Failed to load parquet, trying pickle: {e}")
            
            # Fallback to pickle using safe operations
            if safe_file_exists(labeled_file_pickle):
                try:
                    with open(labeled_file_pickle, 'rb') as f:
                        data = pickle.load(f)
                    self.logger.info(f"✅ Loaded training data from pickle: {len(data)} samples")
                    return data
                except Exception as e:
                    self.logger.warning(f"Failed to load pickle: {e}")
            
            raise Step15DataError("No training data files found")
            
        except Step15DataError:
            raise
        except Exception as e:
            raise Step15DataError(f"Failed to load training data: {e}")
    
    @log_call()
    async def _prepare_training_data_optimized(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data using proper utilities and validation."""
        try:
            # Find target column
            target_column = 'tactician_label' if 'tactician_label' in data.columns else 'label'
            if target_column not in data.columns:
                raise Step15DataError("No target column found")
            
            y = data[target_column].copy()
            
            # Prepare features using safe operations
            datetime_columns = data.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetime']).columns.tolist()
            if datetime_columns:
                data = data.drop(columns=datetime_columns)
            
            object_columns = data.select_dtypes(include=['object']).columns.tolist()
            object_columns_to_drop = [col for col in object_columns if col != target_column]
            if object_columns_to_drop:
                data = data.drop(columns=object_columns_to_drop)
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]
            
            if not feature_columns:
                self.logger.warning("No numeric feature columns found, creating simple feature")
                data['simple_feature'] = np.random.randn(len(data))
                feature_columns = ['simple_feature']
            
            X = data[feature_columns].copy()
            X = safe_fillna(X, 0)
            
            # Train-test split using safe math
            data_length = safe_int(len(X))
            split_point = safe_int(safe_divide(data_length, 5, default=1) * 4)  # 80% split
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            self.logger.info(f"✅ Prepared training data: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
            
        except (Step15DataError, MathValidationError) as e:
            raise
        except Exception as e:
            raise Step15DataError(f"Failed to prepare training data: {e}")
    
    @log_call()
    async def _save_training_results_optimized(self, training_results: Dict[str, Any], 
                                             training_input: Dict[str, Any]) -> str:
        """Save training results using proper utilities and error handling."""
        try:
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            data_dir = training_input['data_dir']
            
            models_dir = f'{data_dir}/tactician_models'
            ensure_directory(models_dir)
            
            # Save individual models
            for model_name, model_data in training_results.items():
                model_file = f'{models_dir}/{model_name}.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
                self.logger.info(f"💾 Saved {model_name} model")
            
            # Save summary using safe JSON operations
            summary_file = f'{data_dir}/{exchange}_{symbol}_tactician_training_summary.json'
            safe_json_dump(training_results, summary_file, indent=2, default=str)
            
            self.logger.info(f"✅ Training results saved to {models_dir}")
            return models_dir
            
        except Exception as e:
            raise Step15DataError(f"Failed to save training results: {e}")

# Example usage with proper utility integration
@traced(span_name="run_optimized_step15_with_utilities")
@handles_errors(exceptions=(Step15ValidationError, Step15DataError, MathValidationError))
async def run_optimized_step15_with_utilities(symbol: str, exchange: str = 'BINANCE', 
                                            data_dir: str = None, **kwargs) -> bool:
    """Run optimized Step15 implementation with proper utility integration."""
    try:
        if data_dir is None:
            standards = PipelineStandards()
            data_dir = standards.build_path('training', exchange, symbol)
        
        config = {
            'symbol': symbol,
            'exchange': exchange,
            'data_dir': data_dir,
            'max_samples': 1000000,
            'min_samples': 100,
            'max_memory_mb': 8192,
            'min_regime_samples': 500
        }
        
        step = Step15OptimizedImplementation(config)
        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'data_dir': data_dir,
            **kwargs
        }
        
        pipeline_state = {}
        result = await step.execute_optimized(training_input, pipeline_state)
        
        return result.get('status') == 'SUCCESS'
        
    except (Step15ValidationError, Step15DataError, MathValidationError) as e:
        system_logger.error(f"Step15 validation/data error: {e}")
        return False
    except Exception as e:
        system_logger.error(f"Optimized Step15 execution failed: {e}")
        return False

if __name__ == "__main__":
    async def test():
        success = await run_optimized_step15_with_utilities('ETHUSDT', 'BINANCE')
        print(f"Optimized Step15 with utilities result: {success}")
    
    asyncio.run(test())