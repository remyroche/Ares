"""
Step 15: Optimized Tactician Specialist Training Implementation

This optimized version includes:
- Fast fail mechanisms for early validation
- Comprehensive validity checks
- Improved logic flow and error handling
- Performance optimizations
- Enhanced resource management
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

from src.core.decorators import handles_errors, timeout, circuit_breaker
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards

class Step15OptimizedValidator:
    """Fast fail validation system for Step15."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.max_samples = config.get('max_samples', 1000000)
        self.min_samples = config.get('min_samples', 100)
        self.max_memory_mb = config.get('max_memory_mb', 8192)
        
    def validate_inputs(self, training_input: Dict[str, Any]) -> Tuple[bool, str]:
        """Fast fail input validation."""
        try:
            # Check required parameters
            if not training_input.get('symbol'):
                return False, "Symbol is required"
            
            if not training_input.get('exchange'):
                return False, "Exchange is required"
            
            # Check data directory
            data_dir = training_input.get('data_dir')
            if not data_dir or not os.path.exists(data_dir):
                return False, f"Data directory not found: {data_dir}"
            
            # Check for training data files
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            labeled_data_dir = f'{data_dir}/tactician_labeled_data'
            labeled_file_parquet = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet'
            labeled_file_pickle = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl'
            
            if not os.path.exists(labeled_file_parquet) and not os.path.exists(labeled_file_pickle):
                return False, f"No training data found for {symbol} on {exchange}"
            
            return True, "Input validation passed"
            
        except Exception as e:
            return False, f"Input validation error: {str(e)}"
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate training data quality."""
        try:
            if data.empty:
                return False, "Training data is empty"
            
            # Check data size
            if len(data) < self.min_samples:
                return False, f"Insufficient samples: {len(data)} < {self.min_samples}"
            
            if len(data) > self.max_samples:
                return False, f"Dataset too large: {len(data)} > {self.max_samples}"
            
            # Check missing data
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.5:
                return False, f"Too much missing data: {missing_ratio:.2%}"
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check for target column
            target_cols = ['tactician_label', 'label']
            if not any(col in data.columns for col in target_cols):
                return False, "No target column found (tactician_label or label)"
            
            return True, "Data quality validation passed"
            
        except Exception as e:
            return False, f"Data quality validation error: {str(e)}"
    
    def validate_model_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate model parameters."""
        try:
            # Check XGBoost parameters
            if 'xgboost' in params:
                xgb_params = params['xgboost']
                if xgb_params.get('n_estimators', 0) <= 0:
                    return False, "Invalid n_estimators for XGBoost"
                if xgb_params.get('max_depth', 0) <= 0:
                    return False, "Invalid max_depth for XGBoost"
                if not 0 < xgb_params.get('learning_rate', 0) <= 1:
                    return False, "Invalid learning_rate for XGBoost"
            
            # Check LightGBM parameters
            if 'lightgbm' in params:
                lgb_params = params['lightgbm']
                if lgb_params.get('n_estimators', 0) <= 0:
                    return False, "Invalid n_estimators for LightGBM"
                if lgb_params.get('max_depth', 0) <= 0:
                    return False, "Invalid max_depth for LightGBM"
            
            return True, "Model parameters validation passed"
            
        except Exception as e:
            return False, f"Model parameters validation error: {str(e)}"
    
    def validate_regime_data(self, regime_data: pd.DataFrame, regime_id: str) -> Tuple[bool, str]:
        """Validate regime-specific data."""
        try:
            min_regime_samples = self.config.get('min_regime_samples', 500)
            
            if len(regime_data) < min_regime_samples:
                return False, f"Regime {regime_id} has insufficient samples: {len(regime_data)} < {min_regime_samples}"
            
            if 'composite_cluster_id' not in regime_data.columns:
                return False, f"Missing composite_cluster_id column for regime {regime_id}"
            
            return True, f"Regime {regime_id} validation passed"
            
        except Exception as e:
            return False, f"Regime validation error: {str(e)}"

class Step15OptimizedProcessor:
    """Optimized data processing for Step15."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self._optimized_data_cache = {}
        
    def optimize_dataframe(self, data: pd.DataFrame, cache_key: str = None) -> pd.DataFrame:
        """Optimize dataframe with caching."""
        if cache_key and cache_key in self._optimized_data_cache:
            self.logger.info(f"Using cached optimized dataframe: {cache_key}")
            return self._optimized_data_cache[cache_key]
        
        try:
            # Convert to efficient dtypes
            optimized_data = data.copy()
            
            # Optimize numeric columns
            for col in optimized_data.select_dtypes(include=[np.number]).columns:
                if optimized_data[col].dtype == 'float64':
                    optimized_data[col] = optimized_data[col].astype('float32')
                elif optimized_data[col].dtype == 'int64':
                    optimized_data[col] = optimized_data[col].astype('int32')
            
            # Cache if key provided
            if cache_key:
                self._optimized_data_cache[cache_key] = optimized_data
                self.logger.info(f"Cached optimized dataframe: {cache_key}")
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"Dataframe optimization failed: {e}")
            return data
    
    def vectorized_sr_enhancement(self, data: pd.DataFrame, sr_predictor) -> pd.DataFrame:
        """Vectorized S/R context enhancement."""
        try:
            if data.empty or sr_predictor is None:
                return data
            
            self.logger.info("Applying vectorized S/R enhancement...")
            enhanced_data = data.copy()
            
            # Vectorized S/R feature calculation
            sample_interval = max(1, len(data) // 100)  # Sample every 1% of data
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
            
            # Vectorized processing
            for i, idx in enumerate(sample_indices):
                try:
                    row = data.loc[idx]
                    current_price = float(row['close'])
                    
                    # Simplified S/R calculation
                    lookback = min(200, len(data))
                    market_slice = data.loc[:idx].tail(lookback)
                    
                    if len(market_slice) < 20:
                        sr_features['sr_proximity'][i] = 0.0
                        sr_features['sr_confidence'][i] = 0.5
                        sr_features['breakout_probability'][i] = 0.33
                        sr_features['rebounce_probability'][i] = 0.33
                        sr_features['consolidation_probability'][i] = 0.34
                        continue
                    
                    # Calculate S/R features
                    high_price = market_slice['high'].max()
                    low_price = market_slice['low'].min()
                    price_range = high_price - low_price
                    
                    if price_range > 0:
                        proximity = min(abs(current_price - high_price), abs(current_price - low_price)) / price_range
                        sr_features['sr_proximity'][i] = 1.0 - proximity
                        sr_features['sr_confidence'][i] = 0.7 + proximity * 0.3
                    else:
                        sr_features['sr_proximity'][i] = 0.0
                        sr_features['sr_confidence'][i] = 0.5
                    
                    # Simple probability distribution
                    sr_features['breakout_probability'][i] = 0.3 + sr_features['sr_proximity'][i] * 0.2
                    sr_features['rebounce_probability'][i] = 0.3 + sr_features['sr_proximity'][i] * 0.2
                    sr_features['consolidation_probability'][i] = 1.0 - sr_features['breakout_probability'][i] - sr_features['rebounce_probability'][i]
                    
                except Exception as e:
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
            return data

class Step15OptimizedTrainer:
    """Optimized model training for Step15."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.model_cache = {}
        
    async def train_models_concurrent(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                    y_train: pd.Series, y_test: pd.Series, 
                                    symbol: str, exchange: str) -> Dict[str, Any]:
        """Train models concurrently for better performance."""
        try:
            self.logger.info("Starting concurrent model training...")
            
            # Define training tasks
            training_tasks = [
                ('lightgbm', self._train_lightgbm_optimized),
                ('xgboost', self._train_xgboost_optimized),
                ('random_forest', self._train_random_forest_optimized),
                ('calibrated_logistic', self._train_calibrated_logistic_optimized)
            ]
            
            # Execute training concurrently
            results = await asyncio.gather(
                *[self._execute_training_task_async(task_name, train_method, X_train, X_test, y_train, y_test, symbol, exchange)
                  for task_name, train_method in training_tasks],
                return_exceptions=True
            )
            
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
    
    async def _execute_training_task_async(self, task_name: str, train_method, 
                                         X_train: pd.DataFrame, X_test: pd.DataFrame,
                                         y_train: pd.Series, y_test: pd.Series,
                                         symbol: str, exchange: str) -> Any:
        """Execute training task asynchronously."""
        try:
            return await train_method(X_train, X_test, y_train, y_test, symbol, exchange)
        except Exception as e:
            self.logger.warning(f"Training task {task_name} failed: {e}")
            return None
    
    async def _train_lightgbm_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                      y_train: pd.Series, y_test: pd.Series,
                                      symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized LightGBM training."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score
            
            # Optimized parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
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
            
            # Predictions
            y_pred = model.predict(X_test) > 0.5
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'LightGBM',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"LightGBM training failed: {e}")
            return None
    
    async def _train_xgboost_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series, y_test: pd.Series,
                                     symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized XGBoost training."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score
            
            # Optimized parameters
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
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
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'XGBoost',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"XGBoost training failed: {e}")
            return None
    
    async def _train_random_forest_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                           y_train: pd.Series, y_test: pd.Series,
                                           symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized Random Forest training."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            # Optimized parameters
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Create and train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'RandomForest',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"Random Forest training failed: {e}")
            return None
    
    async def _train_calibrated_logistic_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                 y_train: pd.Series, y_test: pd.Series,
                                                 symbol: str, exchange: str) -> Dict[str, Any]:
        """Optimized Calibrated Logistic Regression training."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            
            # Base model
            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
            
            # Calibrated model
            model = CalibratedClassifierCV(
                estimator=base_model,
                cv=5,
                method='isotonic'
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'CalibratedLogistic',
                'symbol': symbol,
                'exchange': exchange,
                'training_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"Calibrated Logistic training failed: {e}")
            return None

class Step15OptimizedImplementation:
    """Optimized Step15 implementation with all improvements."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.validator = Step15OptimizedValidator(config)
        self.processor = Step15OptimizedProcessor(config)
        self.trainer = Step15OptimizedTrainer(config)
        
    @handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Execution failed'})
    @timeout(3600)  # 1 hour timeout
    async def execute_optimized(self, training_input: Dict[str, Any], 
                              pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized Step15 with all improvements."""
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting optimized Step15 execution...")
            
            # Fast fail validation
            is_valid, validation_msg = self.validator.validate_inputs(training_input)
            if not is_valid:
                return {'status': 'FAILED', 'error': f"Input validation failed: {validation_msg}"}
            
            self.logger.info(f"✅ Input validation passed: {validation_msg}")
            
            # Load and validate data
            labeled_data = await self._load_training_data(training_input)
            if labeled_data is None:
                return {'status': 'FAILED', 'error': 'Failed to load training data'}
            
            # Validate data quality
            is_valid, validation_msg = self.validator.validate_data_quality(labeled_data)
            if not is_valid:
                return {'status': 'FAILED', 'error': f"Data quality validation failed: {validation_msg}"}
            
            self.logger.info(f"✅ Data quality validation passed: {validation_msg}")
            
            # Optimize data processing
            cache_key = f"{training_input['symbol']}_{training_input['exchange']}"
            labeled_data = self.processor.optimize_dataframe(labeled_data, cache_key)
            
            # Prepare training data
            X_train, X_test, y_train, y_test = await self._prepare_training_data(labeled_data)
            
            # Train models concurrently
            training_results = await self.trainer.train_models_concurrent(
                X_train, X_test, y_train, y_test,
                training_input['symbol'], training_input['exchange']
            )
            
            if not training_results:
                return {'status': 'FAILED', 'error': 'No models were successfully trained'}
            
            # Save results
            models_dir = await self._save_training_results(training_results, training_input)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ Optimized Step15 execution completed in {execution_time:.2f}s")
            
            return {
                'status': 'SUCCESS',
                'tactician_models': training_results,
                'models_dir': models_dir,
                'duration': execution_time,
                'models_trained': len(training_results)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Optimized Step15 execution failed after {execution_time:.2f}s: {e}")
            return {'status': 'FAILED', 'error': str(e), 'duration': execution_time}
    
    async def _load_training_data(self, training_input: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load training data with error handling."""
        try:
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            data_dir = training_input['data_dir']
            
            labeled_data_dir = f'{data_dir}/tactician_labeled_data'
            labeled_file_parquet = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet'
            labeled_file_pickle = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl'
            
            # Try parquet first
            if os.path.exists(labeled_file_parquet):
                try:
                    import pandas as pd
                    data = pd.read_parquet(labeled_file_parquet)
                    self.logger.info(f"✅ Loaded training data from parquet: {len(data)} samples")
                    return data
                except Exception as e:
                    self.logger.warning(f"Failed to load parquet, trying pickle: {e}")
            
            # Fallback to pickle
            if os.path.exists(labeled_file_pickle):
                with open(labeled_file_pickle, 'rb') as f:
                    data = pickle.load(f)
                self.logger.info(f"✅ Loaded training data from pickle: {len(data)} samples")
                return data
            
            self.logger.error("No training data files found")
            return None
            
        except Exception as e:
            self.logger.exception(f"Failed to load training data: {e}")
            return None
    
    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data with validation."""
        try:
            # Find target column
            target_column = 'tactician_label' if 'tactician_label' in data.columns else 'label'
            if target_column not in data.columns:
                raise ValueError("No target column found")
            
            y = data[target_column].copy()
            
            # Prepare features
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
            X = X.fillna(0)
            
            # Train-test split
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            self.logger.info(f"✅ Prepared training data: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.exception(f"Failed to prepare training data: {e}")
            raise
    
    async def _save_training_results(self, training_results: Dict[str, Any], 
                                   training_input: Dict[str, Any]) -> str:
        """Save training results with error handling."""
        try:
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            data_dir = training_input['data_dir']
            
            models_dir = f'{data_dir}/tactician_models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save individual models
            for model_name, model_data in training_results.items():
                model_file = f'{models_dir}/{model_name}.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
                self.logger.info(f"💾 Saved {model_name} model")
            
            # Save summary
            summary_file = f'{data_dir}/{exchange}_{symbol}_tactician_training_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Training results saved to {models_dir}")
            return models_dir
            
        except Exception as e:
            self.logger.exception(f"Failed to save training results: {e}")
            raise

# Example usage
async def run_optimized_step15(symbol: str, exchange: str = 'BINANCE', 
                              data_dir: str = None, **kwargs) -> bool:
    """Run optimized Step15 implementation."""
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
        
    except Exception as e:
        system_logger.error(f"Optimized Step15 execution failed: {e}")
        return False

if __name__ == "__main__":
    async def test():
        success = await run_optimized_step15('ETHUSDT', 'BINANCE')
        print(f"Optimized Step15 result: {success}")
    
    asyncio.run(test())