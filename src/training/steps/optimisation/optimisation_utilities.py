#!/usr/bin/env python3
"""
Optimisation Pipeline Utilities

Common utilities for the optimisation pipeline with:
- Data formatting and validation
- Analysis operations
- Data access control
- Pipeline state management
- Performance optimization
"""

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from src.utils.logger import system_logger
from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime
)
from src.utils.pipeline_protection_framework import (
    DataValidator,
    ValidationLevel,
    PipelineState,
    DataIntegrityCheck
)


class DataFormattingUtils:
    """Utilities for data formatting and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DataFormattingUtils")
        self.data_validator = DataValidator(config)
    
    def format_optimisation_data(self, 
                                data: pd.DataFrame,
                                target_column: str = "target",
                                feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Format data for optimisation operations."""
        try:
            self.logger.info("🔧 Formatting data for optimisation...")
            
            # Validate input data
            validation = self.data_validator.validate_dataframe(data)
            if not validation.passed:
                raise ValueError(f"Data validation failed: {validation}")
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # Check if target column exists
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Separate features and target
            X = data[feature_columns].copy()
            y = data[target_column].copy()
            
            # Handle missing values
            X = self._handle_missing_values(X)
            y = self._handle_missing_values(y)
            
            # Normalize features
            X_normalized = self._normalize_features(X)
            
            # Create formatted data structure
            formatted_data = {
                "features": X_normalized,
                "target": y,
                "feature_names": feature_columns,
                "target_name": target_column,
                "data_info": {
                    "n_samples": len(X),
                    "n_features": len(feature_columns),
                    "feature_types": {col: str(dtype) for col, dtype in X.dtypes.items()},
                    "target_type": str(y.dtype),
                    "checksum": self._calculate_checksum(data)
                }
            }
            
            self.logger.info(f"✅ Data formatted successfully: {len(X)} samples, {len(feature_columns)} features")
            return formatted_data
            
        except Exception as e:
            self.logger.exception(f"❌ Data formatting failed: {e}")
            raise
    
    def _handle_missing_values(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Handle missing values in data."""
        if isinstance(data, pd.DataFrame):
            # For DataFrames, use median for numeric columns and mode for categorical
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown', inplace=True)
        else:
            # For Series
            if data.dtype in ['int64', 'float64']:
                data.fillna(data.median(), inplace=True)
            else:
                data.fillna(data.mode()[0] if not data.mode().empty else 'unknown', inplace=True)
        
        return data
    
    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        try:
            # Only normalize numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_normalized = X.copy()
            
            for col in numeric_columns:
                if X[col].std() > 0:  # Avoid division by zero
                    X_normalized[col] = (X[col] - X[col].mean()) / X[col].std()
            
            return X_normalized
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed, using original data: {e}")
            return X
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data."""
        try:
            data_str = data.to_string()
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return ""


class AnalysisOperationsUtils:
    """Utilities for analysis operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("AnalysisOperationsUtils")
    
    def calculate_performance_metrics(self, 
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            self.logger.info("📊 Calculating performance metrics...")
            
            metrics = {}
            
            # Basic classification metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix, classification_report
            )
            
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC if probabilities are available
            if y_prob is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                except ValueError:
                    metrics["roc_auc"] = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Additional metrics
            metrics["n_samples"] = len(y_true)
            metrics["n_classes"] = len(np.unique(y_true))
            
            self.logger.info(f"✅ Performance metrics calculated: accuracy={metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Performance metrics calculation failed: {e}")
            raise
    
    def optimize_hyperparameters(self, 
                                model_class: Any,
                                X: pd.DataFrame,
                                y: pd.Series,
                                param_grid: Dict[str, List[Any]],
                                cv_folds: int = 5,
                                scoring: str = 'accuracy',
                                n_jobs: int = -1) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        try:
            self.logger.info("🔧 Optimizing hyperparameters...")
            
            from sklearn.model_selection import GridSearchCV
            
            # Create model instance
            model = model_class()
            
            # Perform grid search
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Extract results
            results = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "best_estimator": grid_search.best_estimator_,
                "cv_results": grid_search.cv_results_,
                "n_candidates": len(grid_search.cv_results_['params']),
                "optimization_time": time.time()
            }
            
            self.logger.info(f"✅ Hyperparameter optimization completed: best_score={results['best_score']:.3f}")
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ Hyperparameter optimization failed: {e}")
            raise
    
    def cross_validate_model(self, 
                           model: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv_folds: int = 5,
                           scoring: List[str] = None) -> Dict[str, Any]:
        """Perform cross-validation on a model."""
        try:
            self.logger.info("🔄 Performing cross-validation...")
            
            if scoring is None:
                scoring = ['accuracy', 'precision', 'recall', 'f1']
            
            from sklearn.model_selection import cross_validate
            
            cv_results = cross_validate(
                model,
                X, y,
                cv=cv_folds,
                scoring=scoring,
                return_train_score=True
            )
            
            # Calculate summary statistics
            summary = {}
            for metric in scoring:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                summary[metric] = {
                    "test_mean": np.mean(test_scores),
                    "test_std": np.std(test_scores),
                    "train_mean": np.mean(train_scores),
                    "train_std": np.std(train_scores),
                    "overfitting": np.mean(train_scores) - np.mean(test_scores)
                }
            
            results = {
                "cv_results": cv_results,
                "summary": summary,
                "cv_folds": cv_folds,
                "n_samples": len(X)
            }
            
            self.logger.info(f"✅ Cross-validation completed: {cv_folds} folds")
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ Cross-validation failed: {e}")
            raise


class DataAccessControl:
    """Utilities for data access control and security."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DataAccessControl")
        self.access_log: List[Dict[str, Any]] = []
    
    def validate_data_access(self, 
                           user_id: str,
                           data_path: str,
                           operation: str = "read") -> bool:
        """Validate data access permissions."""
        try:
            self.logger.info(f"🔐 Validating data access for user {user_id} on {data_path}")
            
            # Check if file exists
            if not safe_file_exists(data_path):
                self.logger.error(f"❌ Data file not found: {data_path}")
                return False
            
            # Check file permissions
            if not os.access(data_path, os.R_OK):
                self.logger.error(f"❌ No read permission for: {data_path}")
                return False
            
            if operation == "write" and not os.access(data_path, os.W_OK):
                self.logger.error(f"❌ No write permission for: {data_path}")
                return False
            
            # Log access
            access_entry = {
                "user_id": user_id,
                "data_path": data_path,
                "operation": operation,
                "timestamp": get_current_datetime().isoformat(),
                "allowed": True
            }
            self.access_log.append(access_entry)
            
            self.logger.info(f"✅ Data access validated for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Data access validation failed: {e}")
            return False
    
    def secure_data_loading(self, 
                          data_path: str,
                          user_id: str = "system",
                          validate_integrity: bool = True) -> Optional[pd.DataFrame]:
        """Securely load data with access control."""
        try:
            # Validate access
            if not self.validate_data_access(user_id, data_path, "read"):
                return None
            
            self.logger.info(f"📁 Loading data from: {data_path}")
            
            # Load data based on file extension
            if data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.pkl'):
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Validate data integrity
            if validate_integrity and isinstance(data, pd.DataFrame):
                validator = DataValidator(self.config)
                validation = validator.validate_dataframe(data)
                if not validation.passed:
                    self.logger.error(f"❌ Data integrity validation failed: {validation}")
                    return None
            
            self.logger.info(f"✅ Data loaded successfully: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.exception(f"❌ Secure data loading failed: {e}")
            return None
    
    def secure_data_saving(self, 
                          data: Any,
                          data_path: str,
                          user_id: str = "system",
                          backup_existing: bool = True) -> bool:
        """Securely save data with access control."""
        try:
            # Validate access
            if not self.validate_data_access(user_id, data_path, "write"):
                return False
            
            # Create backup if file exists
            if backup_existing and safe_file_exists(data_path):
                backup_path = f"{data_path}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(data_path, backup_path)
                self.logger.info(f"💾 Created backup: {backup_path}")
            
            # Ensure directory exists
            ensure_directory(Path(data_path).parent)
            
            self.logger.info(f"💾 Saving data to: {data_path}")
            
            # Save data based on file extension
            if data_path.endswith('.parquet'):
                data.to_parquet(data_path, index=False)
            elif data_path.endswith('.csv'):
                data.to_csv(data_path, index=False)
            elif data_path.endswith('.pkl'):
                with open(data_path, 'wb') as f:
                    pickle.dump(data, f)
            elif data_path.endswith('.json'):
                safe_json_dump(data, data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            self.logger.info(f"✅ Data saved successfully to: {data_path}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Secure data saving failed: {e}")
            return False


class PipelineStateManager:
    """Enhanced pipeline state management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PipelineStateManager")
        self.state_file = Path(config.get("state_file", "data_cache/optimisation_pipeline_state.json"))
        self._state: Optional[PipelineState] = None
    
    async def initialize_state(self) -> PipelineState:
        """Initialize pipeline state."""
        try:
            if safe_file_exists(self.state_file):
                state_data = safe_json_load(self.state_file)
                self._state = PipelineState(**state_data)
                self.logger.info(f"✅ Loaded pipeline state from {self.state_file}")
            else:
                self._state = PipelineState()
                self.logger.info("🆕 Created new pipeline state")
            
            return self._state
            
        except Exception as e:
            self.logger.exception(f"Error initializing pipeline state: {e}")
            self._state = PipelineState()
            return self._state
    
    async def save_state(self) -> None:
        """Save pipeline state."""
        try:
            if self._state is None:
                self.logger.warning("No state to save")
                return
            
            ensure_directory(self.state_file.parent)
            
            # Convert to dict for JSON serialization
            state_dict = {
                "current_step": self._state.current_step,
                "step_history": self._state.step_history,
                "data_checkpoints": self._state.data_checkpoints,
                "validation_results": self._state.validation_results,
                "error_log": [
                    {
                        **error,
                        "timestamp": error["timestamp"].isoformat() if isinstance(error.get("timestamp"), datetime) else str(error.get("timestamp", ""))
                    }
                    for error in self._state.error_log
                ],
                "performance_metrics": self._state.performance_metrics,
                "created_at": self._state.created_at.isoformat(),
                "updated_at": self._state.updated_at.isoformat()
            }
            
            safe_json_dump(state_dict, self.state_file, indent=2)
            self.logger.info(f"💾 Saved pipeline state to {self.state_file}")
            
        except Exception as e:
            self.logger.exception(f"Error saving pipeline state: {e}")
    
    def get_state(self) -> Optional[PipelineState]:
        """Get current state."""
        return self._state
    
    def update_step(self, step_name: str) -> None:
        """Update current step."""
        if self._state:
            self._state.update_step(step_name)
    
    def add_checkpoint(self, checkpoint_name: str, data: Any) -> None:
        """Add data checkpoint."""
        if self._state:
            self._state.add_checkpoint(checkpoint_name, data)
    
    def add_validation_result(self, step_name: str, result: Dict[str, Any]) -> None:
        """Add validation result."""
        if self._state:
            self._state.add_validation_result(step_name, result)
    
    def add_error(self, error: Dict[str, Any]) -> None:
        """Add error to log."""
        if self._state:
            self._state.add_error(error)


class PerformanceOptimizer:
    """Utilities for performance optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PerformanceOptimizer")
        self.max_workers = config.get("max_workers", mp.cpu_count())
    
    def parallel_processing(self, 
                          func: Callable,
                          data_chunks: List[Any],
                          use_processes: bool = False) -> List[Any]:
        """Execute function in parallel on data chunks."""
        try:
            self.logger.info(f"⚡ Starting parallel processing: {len(data_chunks)} chunks")
            
            if use_processes:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(func, data_chunks))
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(func, data_chunks))
            
            self.logger.info(f"✅ Parallel processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ Parallel processing failed: {e}")
            raise
    
    def optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        try:
            self.logger.info("🧠 Optimizing memory usage...")
            
            original_memory = data.memory_usage(deep=True).sum()
            
            # Optimize numeric columns
            for col in data.select_dtypes(include=[np.number]).columns:
                col_min = data[col].min()
                col_max = data[col].max()
                
                if data[col].dtype == 'int64':
                    if col_min >= 0:
                        if col_max < 255:
                            data[col] = data[col].astype(np.uint8)
                        elif col_max < 65535:
                            data[col] = data[col].astype(np.uint16)
                        elif col_max < 4294967295:
                            data[col] = data[col].astype(np.uint32)
                    else:
                        if col_min > -128 and col_max < 127:
                            data[col] = data[col].astype(np.int8)
                        elif col_min > -32768 and col_max < 32767:
                            data[col] = data[col].astype(np.int16)
                        elif col_min > -2147483648 and col_max < 2147483647:
                            data[col] = data[col].astype(np.int32)
                
                elif data[col].dtype == 'float64':
                    data[col] = data[col].astype(np.float32)
            
            # Optimize categorical columns
            for col in data.select_dtypes(include=['object']).columns:
                if data[col].nunique() / len(data) < 0.5:  # Less than 50% unique values
                    data[col] = data[col].astype('category')
            
            optimized_memory = data.memory_usage(deep=True).sum()
            memory_reduction = (original_memory - optimized_memory) / original_memory * 100
            
            self.logger.info(f"✅ Memory optimization completed: {memory_reduction:.1f}% reduction")
            return data
            
        except Exception as e:
            self.logger.exception(f"❌ Memory optimization failed: {e}")
            return data


# Global utility instances
_data_formatting_utils: Optional[DataFormattingUtils] = None
_analysis_operations_utils: Optional[AnalysisOperationsUtils] = None
_data_access_control: Optional[DataAccessControl] = None
_pipeline_state_manager: Optional[PipelineStateManager] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None


def initialize_optimisation_utilities(config: Dict[str, Any]) -> None:
    """Initialize optimisation utilities."""
    global _data_formatting_utils, _analysis_operations_utils, _data_access_control
    global _pipeline_state_manager, _performance_optimizer
    
    _data_formatting_utils = DataFormattingUtils(config)
    _analysis_operations_utils = AnalysisOperationsUtils(config)
    _data_access_control = DataAccessControl(config)
    _pipeline_state_manager = PipelineStateManager(config)
    _performance_optimizer = PerformanceOptimizer(config)
    
    system_logger.info("🔧 Optimisation utilities initialized")


def get_data_formatting_utils() -> DataFormattingUtils:
    """Get data formatting utils."""
    if _data_formatting_utils is None:
        raise RuntimeError("Data formatting utils not initialized")
    return _data_formatting_utils


def get_analysis_operations_utils() -> AnalysisOperationsUtils:
    """Get analysis operations utils."""
    if _analysis_operations_utils is None:
        raise RuntimeError("Analysis operations utils not initialized")
    return _analysis_operations_utils


def get_data_access_control() -> DataAccessControl:
    """Get data access control."""
    if _data_access_control is None:
        raise RuntimeError("Data access control not initialized")
    return _data_access_control


def get_pipeline_state_manager() -> PipelineStateManager:
    """Get pipeline state manager."""
    if _pipeline_state_manager is None:
        raise RuntimeError("Pipeline state manager not initialized")
    return _pipeline_state_manager


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer."""
    if _performance_optimizer is None:
        raise RuntimeError("Performance optimizer not initialized")
    return _performance_optimizer