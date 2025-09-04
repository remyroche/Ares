#!/usr/bin/env python3
"""
Step-by-Step Validators for Backtesting Pipeline

This module provides specific validators for each stage of the backtesting pipeline,
ensuring data integrity and operational correctness at each step.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import time

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
)
from .validation_framework import (
    BacktestingValidator,
    ValidationResult,
    ValidationStatus,
    BacktestingValidationOrchestrator
)
from src.utils.compat import handle_errors


class DataLoadingValidator(BacktestingValidator):
    """Validator for data loading operations (Step 1)."""
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_data_files(self, symbol: str, exchange: str, data_dir: str = "data_cache") -> ValidationResult:
        """Validate that required data files exist and are accessible."""
        try:
            issues = []
            warnings = []
            
            # Expected data files
            expected_files = [
                f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"volume_{exchange}_{symbol}_consolidated.parquet",
                f"klines_{exchange}_{symbol}_1m.parquet"
            ]
            
            data_path = Path(data_dir)
            if not data_path.exists():
                issues.append(f"Data directory does not exist: {data_path}")
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Data directory validation failed for {symbol} on {exchange}",
                    errors=issues
                )
            
            # Check each expected file
            found_files = []
            for filename in expected_files:
                file_path = data_path / filename
                if file_path.exists():
                    found_files.append(filename)
                    # Check file size
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        issues.append(f"Empty file found: {filename}")
                    elif file_size < 1024:  # Less than 1KB
                        warnings.append(f"Very small file: {filename} ({file_size} bytes)")
                else:
                    issues.append(f"Missing required file: {filename}")
            
            # Check for additional data files
            all_files = list(data_path.glob(f"*{symbol}*"))
            if len(all_files) > len(expected_files):
                warnings.append(f"Found {len(all_files)} files for {symbol}, expected {len(expected_files)}")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Data files validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Data files validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Data files validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "data_directory": str(data_path),
                    "expected_files": expected_files,
                    "found_files": found_files,
                    "total_files": len(all_files)
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Data files validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_data_quality(self, data: pd.DataFrame, symbol: str, exchange: str) -> ValidationResult:
        """Validate the quality of loaded data."""
        try:
            issues = []
            warnings = []
            
            # Check data shape
            if data.empty:
                issues.append("Data is empty")
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Data quality validation failed for {symbol} on {exchange}",
                    errors=issues
                )
            
            # Check for required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            if "timestamp" in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                    issues.append("Timestamp column is not datetime type")
            
            # Check for null values
            null_counts = data.isnull().sum()
            high_null_cols = null_counts[null_counts > len(data) * 0.1]  # More than 10% null
            if not high_null_cols.empty:
                warnings.append(f"High null ratio in columns: {high_null_cols.to_dict()}")
            
            # Check for duplicates
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate rows")
            
            # Check timestamp continuity
            if "timestamp" in data.columns and len(data) > 1:
                time_diff = data["timestamp"].diff().dropna()
                if len(time_diff) > 0:
                    expected_interval = time_diff.mode().iloc[0]
                    large_gaps = time_diff > expected_interval * 3
                    if large_gaps.any():
                        warnings.append(f"Large time gaps detected in {large_gaps.sum()} locations")
            
            # Check price data consistency
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        issues.append(f"Non-positive values in {col} column")
                    if data[col].isnull().any():
                        issues.append(f"Null values in {col} column")
            
            # Check OHLC relationships
            if all(col in data.columns for col in price_columns):
                invalid_ohlc = (
                    (data["high"] < data["low"]) |
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"]) |
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"])
                )
                if invalid_ohlc.any():
                    issues.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Data quality validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Data quality validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Data quality validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "data_shape": data.shape,
                    "date_range": {
                        "start": data["timestamp"].min().isoformat() if "timestamp" in data.columns else None,
                        "end": data["timestamp"].max().isoformat() if "timestamp" in data.columns else None
                    },
                    "null_counts": null_counts.to_dict(),
                    "duplicate_count": duplicate_count
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Data quality validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class FeatureEngineeringValidator(BacktestingValidator):
    """Validator for feature engineering operations (Step 2)."""
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_feature_engineering_input(self, data: pd.DataFrame, symbol: str, exchange: str) -> ValidationResult:
        """Validate input data for feature engineering."""
        try:
            issues = []
            warnings = []
            
            # Check if data is suitable for feature engineering
            if data.empty:
                issues.append("Input data is empty")
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Feature engineering input validation failed for {symbol} on {exchange}",
                    errors=issues
                )
            
            # Check minimum data points
            min_data_points = 1000  # Minimum for meaningful feature engineering
            if len(data) < min_data_points:
                warnings.append(f"Limited data points for feature engineering: {len(data)} (minimum recommended: {min_data_points})")
            
            # Check for required price columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                issues.append(f"Missing required price columns: {missing_columns}")
            
            # Check for sufficient price variation
            if "close" in data.columns:
                price_std = data["close"].std()
                price_mean = data["close"].mean()
                if price_std / price_mean < 0.01:  # Less than 1% variation
                    warnings.append("Low price variation detected - features may not be meaningful")
            
            # Check for sufficient volume data
            if "volume" in data.columns:
                volume_std = data["volume"].std()
                volume_mean = data["volume"].mean()
                if volume_std / volume_mean < 0.1:  # Less than 10% variation
                    warnings.append("Low volume variation detected - volume-based features may not be meaningful")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Feature engineering input validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Feature engineering input validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Feature engineering input validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "input_data_shape": data.shape,
                    "data_points": len(data),
                    "price_variation": (data["close"].std() / data["close"].mean()) if "close" in data.columns else None,
                    "volume_variation": (data["volume"].std() / data["volume"].mean()) if "volume" in data.columns else None
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Feature engineering input validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_feature_engineering_output(self, features: pd.DataFrame, symbol: str, exchange: str) -> ValidationResult:
        """Validate the output of feature engineering."""
        try:
            issues = []
            warnings = []
            
            # Check if features were generated
            if features.empty:
                issues.append("No features were generated")
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Feature engineering output validation failed for {symbol} on {exchange}",
                    errors=issues
                )
            
            # Check feature count
            feature_count = len(features.columns)
            if feature_count == 0:
                issues.append("No feature columns generated")
            elif feature_count < 10:
                warnings.append(f"Low feature count: {feature_count} (recommended: 20+)")
            elif feature_count > 1000:
                warnings.append(f"Very high feature count: {feature_count} (may cause overfitting)")
            
            # Check for infinite values
            numeric_features = features.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                inf_count = np.isinf(numeric_features).sum().sum()
                if inf_count > 0:
                    issues.append(f"Found {inf_count} infinite values in features")
            
            # Check for excessive null values
            null_ratio = features.isnull().sum() / len(features)
            high_null_cols = null_ratio[null_ratio > 0.5]
            if not high_null_cols.empty:
                warnings.append(f"High null ratio in feature columns: {high_null_cols.to_dict()}")
            
            # Check for constant features
            constant_features = []
            for col in numeric_features.columns:
                if numeric_features[col].nunique() <= 1:
                    constant_features.append(col)
            if constant_features:
                warnings.append(f"Constant features detected: {constant_features}")
            
            # Check feature correlation
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                high_corr_pairs = []
                for col in upper_triangle.columns:
                    high_corr_cols = upper_triangle.index[upper_triangle[col] > 0.95].tolist()
                    if high_corr_cols:
                        high_corr_pairs.extend([(col, c) for c in high_corr_cols])
                
                if high_corr_pairs:
                    warnings.append(f"High correlation detected between features: {len(high_corr_pairs)} pairs")
            
            # Check feature scale
            if not numeric_features.empty:
                feature_std = numeric_features.std()
                very_large_std = feature_std[feature_std > 1000]
                if not very_large_std.empty:
                    warnings.append(f"Features with very large standard deviation: {very_large_std.to_dict()}")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Feature engineering output validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Feature engineering output validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Feature engineering output validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "feature_count": feature_count,
                    "feature_shape": features.shape,
                    "numeric_features": len(numeric_features.columns),
                    "constant_features": len(constant_features),
                    "high_correlation_pairs": len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Feature engineering output validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class ModelTrainingValidator(BacktestingValidator):
    """Validator for model training operations (Step 3)."""
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_training_data(self, X: pd.DataFrame, y: pd.Series, symbol: str, exchange: str) -> ValidationResult:
        """Validate training data for model training."""
        try:
            issues = []
            warnings = []
            
            # Check data shapes
            if X.empty:
                issues.append("Feature matrix is empty")
            if y.empty:
                issues.append("Target vector is empty")
            
            if X.empty or y.empty:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Training data validation failed for {symbol} on {exchange}",
                    errors=issues
                )
            
            # Check shape compatibility
            if len(X) != len(y):
                issues.append(f"Feature matrix and target vector length mismatch: {len(X)} vs {len(y)}")
            
            # Check for sufficient training samples
            min_samples = 1000
            if len(X) < min_samples:
                warnings.append(f"Limited training samples: {len(X)} (minimum recommended: {min_samples})")
            
            # Check target distribution
            if y.nunique() < 2:
                issues.append("Target has only one unique value - cannot train model")
            elif y.nunique() < 5:
                warnings.append(f"Target has limited unique values: {y.nunique()}")
            
            # Check for class imbalance
            if y.nunique() > 1:
                class_counts = y.value_counts()
                max_class_ratio = class_counts.max() / class_counts.sum()
                if max_class_ratio > 0.9:
                    warnings.append(f"Severe class imbalance detected: {max_class_ratio:.2%} in majority class")
            
            # Check feature matrix quality
            if X.isnull().any().any():
                null_count = X.isnull().sum().sum()
                warnings.append(f"Null values in feature matrix: {null_count}")
            
            # Check for constant features
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
            if constant_features:
                warnings.append(f"Constant features in training data: {constant_features}")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Training data validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Training data validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Training data validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "feature_shape": X.shape,
                    "target_shape": y.shape,
                    "target_unique_values": y.nunique(),
                    "class_distribution": y.value_counts().to_dict() if y.nunique() > 1 else None,
                    "constant_features": len(constant_features)
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Training data validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_trained_model(self, model: Any, symbol: str, exchange: str) -> ValidationResult:
        """Validate a trained model."""
        try:
            issues = []
            warnings = []
            
            # Check if model exists
            if model is None:
                issues.append("Model is None")
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Trained model validation failed for {symbol} on {exchange}",
                    errors=issues
                )
            
            # Check model type
            model_type = type(model).__name__
            
            # Check if model has required methods
            required_methods = ["predict", "fit"]
            missing_methods = []
            for method in required_methods:
                if not hasattr(model, method):
                    missing_methods.append(method)
            
            if missing_methods:
                issues.append(f"Model missing required methods: {missing_methods}")
            
            # Check model parameters
            if hasattr(model, "get_params"):
                try:
                    params = model.get_params()
                    if not params:
                        warnings.append("Model has no parameters")
                except Exception as e:
                    warnings.append(f"Could not retrieve model parameters: {str(e)}")
            
            # Check for model attributes that might indicate training issues
            if hasattr(model, "feature_importances_"):
                if model.feature_importances_ is None:
                    warnings.append("Model feature importances are None")
                elif np.all(model.feature_importances_ == 0):
                    warnings.append("All feature importances are zero")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Trained model validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Trained model validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Trained model validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "model_type": model_type,
                    "has_predict": hasattr(model, "predict"),
                    "has_fit": hasattr(model, "fit"),
                    "has_feature_importances": hasattr(model, "feature_importances_")
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Trained model validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class BacktestingExecutionValidator(BacktestingValidator):
    """Validator for backtesting execution operations (Step 4)."""
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_backtest_setup(self, config: Dict[str, Any], symbol: str, exchange: str) -> ValidationResult:
        """Validate backtesting configuration and setup."""
        try:
            issues = []
            warnings = []
            
            # Check required configuration parameters
            required_params = ["initial_capital", "commission", "slippage"]
            missing_params = set(required_params) - set(config.keys())
            if missing_params:
                issues.append(f"Missing required backtest parameters: {missing_params}")
            
            # Validate parameter values
            if "initial_capital" in config:
                capital = config["initial_capital"]
                if not isinstance(capital, (int, float)) or capital <= 0:
                    issues.append(f"Invalid initial capital: {capital}")
                elif capital < 1000:
                    warnings.append(f"Low initial capital: {capital}")
            
            if "commission" in config:
                commission = config["commission"]
                if not isinstance(commission, (int, float)) or commission < 0:
                    issues.append(f"Invalid commission rate: {commission}")
                elif commission > 0.01:  # More than 1%
                    warnings.append(f"High commission rate: {commission:.2%}")
            
            if "slippage" in config:
                slippage = config["slippage"]
                if not isinstance(slippage, (int, float)) or slippage < 0:
                    issues.append(f"Invalid slippage rate: {slippage}")
                elif slippage > 0.005:  # More than 0.5%
                    warnings.append(f"High slippage rate: {slippage:.2%}")
            
            # Check for optional parameters
            if "start_date" in config and "end_date" in config:
                try:
                    start_date = pd.to_datetime(config["start_date"])
                    end_date = pd.to_datetime(config["end_date"])
                    if start_date >= end_date:
                        issues.append("Start date must be before end date")
                    elif (end_date - start_date).days < 30:
                        warnings.append("Backtest period is less than 30 days")
                except Exception as e:
                    issues.append(f"Invalid date format: {str(e)}")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Backtest setup validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Backtest setup validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Backtest setup validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "config_keys": list(config.keys()),
                    "missing_required_params": list(missing_params)
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Backtest setup validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_backtest_results(self, results: Dict[str, Any], symbol: str, exchange: str) -> ValidationResult:
        """Validate backtesting results."""
        try:
            issues = []
            warnings = []
            
            # Check required result fields
            required_fields = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"]
            missing_fields = set(required_fields) - set(results.keys())
            if missing_fields:
                issues.append(f"Missing required result fields: {missing_fields}")
            
            # Validate numeric results
            for field in required_fields:
                if field in results:
                    value = results[field]
                    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                        issues.append(f"Invalid value for {field}: {value}")
            
            # Check for reasonable ranges
            if "total_return" in results:
                if results["total_return"] > 10.0:  # 1000% return
                    warnings.append(f"Unusually high total return: {results['total_return']:.2%}")
                elif results["total_return"] < -0.9:  # -90% return
                    warnings.append(f"Unusually low total return: {results['total_return']:.2%}")
            
            if "sharpe_ratio" in results:
                if results["sharpe_ratio"] > 5.0:
                    warnings.append(f"Unusually high Sharpe ratio: {results['sharpe_ratio']:.2f}")
                elif results["sharpe_ratio"] < -2.0:
                    warnings.append(f"Unusually low Sharpe ratio: {results['sharpe_ratio']:.2f}")
            
            if "max_drawdown" in results:
                if results["max_drawdown"] > 0.5:  # 50% drawdown
                    warnings.append(f"High maximum drawdown: {results['max_drawdown']:.2%}")
            
            if "win_rate" in results:
                if results["win_rate"] > 0.8:  # 80% win rate
                    warnings.append(f"Unusually high win rate: {results['win_rate']:.2%}")
                elif results["win_rate"] < 0.2:  # 20% win rate
                    warnings.append(f"Unusually low win rate: {results['win_rate']:.2%}")
            
            if "total_trades" in results:
                if results["total_trades"] == 0:
                    warnings.append("No trades executed during backtest")
                elif results["total_trades"] < 10:
                    warnings.append(f"Very few trades executed: {results['total_trades']}")
            
            # Check for consistency between related metrics
            if "total_return" in results and "max_drawdown" in results:
                if results["total_return"] > 0 and results["max_drawdown"] > results["total_return"]:
                    warnings.append("Maximum drawdown exceeds total return - check calculation")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Backtest results validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Backtest results validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Backtest results validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "result_fields": list(results.keys()),
                    "validation_timestamp": format_datetime(get_current_datetime())
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Backtest results validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class StepValidationOrchestrator:
    """Orchestrates step-by-step validation for the backtesting pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validators = {
            "data_loading": DataLoadingValidator(config),
            "feature_engineering": FeatureEngineeringValidator(config),
            "model_training": ModelTrainingValidator(config),
            "backtesting_execution": BacktestingExecutionValidator(config)
        }
    
    async def validate_step(
        self,
        step_name: str,
        data: Optional[Any] = None,
        symbol: str = "ETHUSDT",
        exchange: str = "BINANCE",
        **kwargs
    ) -> ValidationResult:
        """Validate a specific pipeline step."""
        try:
            self.logger.info(f"Validating step: {step_name}")
            
            if step_name not in self.validators:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Unknown step: {step_name}",
                    errors=[f"No validator available for step: {step_name}"]
                )
            
            validator = self.validators[step_name]
            
            # Route to appropriate validation method based on step and data type
            if step_name == "data_loading":
                if isinstance(data, pd.DataFrame):
                    result = validator.validate_data_quality(data, symbol, exchange)
                else:
                    result = validator.validate_data_files(symbol, exchange, kwargs.get("data_dir", "data_cache"))
            
            elif step_name == "feature_engineering":
                if "features" in kwargs:
                    result = validator.validate_feature_engineering_output(kwargs["features"], symbol, exchange)
                else:
                    result = validator.validate_feature_engineering_input(data, symbol, exchange)
            
            elif step_name == "model_training":
                if "model" in kwargs:
                    result = validator.validate_trained_model(kwargs["model"], symbol, exchange)
                elif "X" in kwargs and "y" in kwargs:
                    result = validator.validate_training_data(kwargs["X"], kwargs["y"], symbol, exchange)
                else:
                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        message=f"Insufficient data for model training validation",
                        errors=["Missing X, y, or model parameters"]
                    )
            
            elif step_name == "backtesting_execution":
                if "results" in kwargs:
                    result = validator.validate_backtest_results(kwargs["results"], symbol, exchange)
                elif "config" in kwargs:
                    result = validator.validate_backtest_setup(kwargs["config"], symbol, exchange)
                else:
                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        message=f"Insufficient data for backtesting validation",
                        errors=["Missing config or results parameters"]
                    )
            
            else:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    message=f"Unsupported step: {step_name}",
                    errors=[f"No validation logic for step: {step_name}"]
                )
            
            self.logger.info(f"Step validation completed: {result.status} - {result.message}")
            return result
            
        except Exception as e:
            self.logger.exception(f"Error in step validation: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Step validation error: {step_name} - {str(e)}",
                errors=[str(e)]
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        summary = {
            "step_summaries": {},
            "overall_summary": {},
            "timestamp": format_datetime(get_current_datetime())
        }
        
        all_results = []
        for step_name, validator in self.validators.items():
            step_summary = validator.get_summary()
            summary["step_summaries"][step_name] = step_summary
            all_results.extend(validator.validation_results)
        
        # Overall summary
        if all_results:
            total = len(all_results)
            passed = sum(1 for r in all_results if r.status == ValidationStatus.PASSED)
            failed = sum(1 for r in all_results if r.status == ValidationStatus.FAILED)
            warnings = sum(1 for r in all_results if r.status == ValidationStatus.WARNING)
            
            summary["overall_summary"] = {
                "total_validations": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": passed / total if total > 0 else 0.0
            }
        
        return summary