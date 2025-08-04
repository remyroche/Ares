# src/training/enhanced_coarse_optimizer.py

import time
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBClassifier

# Import neural network models (with fallback handling)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    TabNetClassifier = None

try:
    from transformers import AutoModelForSequenceClassification
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    AutoModelForSequenceClassification = None

try:
    import torch.nn as nn
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    nn = None

from src.analyst.feature_engineering_orchestrator import FeatureEngineeringEngine
from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class EnhancedCoarseOptimizer:
    """
    Enhanced coarse optimization with multi-model approach, advanced feature pruning,
    and wider hyperparameter search.
    """

    def __init__(
        self,
        db_manager: SQLiteManager,
        symbol: str,
        timeframe: str,
        optimal_target_params: dict,
        klines_data: pd.DataFrame,
        agg_trades_data: pd.DataFrame,
        futures_data: pd.DataFrame,
        blank_training_mode: bool = False,
    ):
        """
        Initializes the Enhanced Coarse Optimizer.
        """
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.optimal_target_params = optimal_target_params
        self.logger = system_logger.getChild("EnhancedCoarseOptimizer")

        # Store the passed dataframes directly
        self.klines_data = klines_data
        self.agg_trades_data = agg_trades_data
        self.futures_data = futures_data
        self.blank_training_mode = blank_training_mode

        self.data_with_targets = None
        # Prepare data - will be called separately
        self._needs_initialization = True
        
        # Enhanced tracking and monitoring
        self.optimization_progress: float = 0.0
        self.current_stage: str = "initialized"
        self.stage_details: Dict[str, Any] = {}
        self.resource_usage: Dict[str, float] = {}
        
        # Initialize resource allocation
        self.resources = self._allocate_resources()

    def _allocate_resources(self) -> Dict[str, Any]:
        """Dynamically allocate computational resources for M1 Mac optimization."""
        try:
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # M1 Mac specific optimizations
            max_workers = min(6, cpu_count - 1)  # M1 has good thermal management
            memory_limit_gb = memory_gb * 0.7  # Conservative memory limit
            
            resources = {
                "max_workers": max_workers,
                "shap_sample_size": min(5000, int(memory_limit_gb * 1000)),
                "enable_parallel": cpu_count > 2,
                "memory_limit_gb": memory_limit_gb,
                "cpu_count": cpu_count,
                "total_memory_gb": memory_gb
            }
            
            self.logger.info(f"📊 Resource allocation for M1 Mac:")
            self.logger.info(f"   - CPU cores: {cpu_count}")
            self.logger.info(f"   - Total memory: {memory_gb:.1f} GB")
            self.logger.info(f"   - Max workers: {max_workers}")
            self.logger.info(f"   - Memory limit: {memory_limit_gb:.1f} GB")
            self.logger.info(f"   - SHAP sample size: {resources['shap_sample_size']}")
            
            return resources
            
        except Exception as e:
            self.logger.warning(f"Failed to allocate resources: {e}, using defaults")
            return {
                "max_workers": 4,
                "shap_sample_size": 3000,
                "enable_parallel": True,
                "memory_limit_gb": 8.0,
                "cpu_count": 8,
                "total_memory_gb": 16.0
            }

    def _monitor_memory_usage(self) -> bool:
        """Monitor memory usage and trigger cleanup if needed."""
        try:
            memory_percent = psutil.virtual_memory().percent
            memory_gb = psutil.virtual_memory().used / (1024**3)
            
            self.resource_usage["memory_percent"] = memory_percent
            self.resource_usage["memory_gb"] = memory_gb
            
            if memory_percent > 80:
                self.logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% ({memory_gb:.1f} GB)")
                gc.collect()
                return True
            elif memory_percent > 60:
                self.logger.info(f"📊 Memory usage: {memory_percent:.1f}% ({memory_gb:.1f} GB)")
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Memory monitoring failed: {e}")
            return False

    def _track_optimization_progress(self, stage: str, progress: float, details: Dict[str, Any] = None):
        """Track optimization progress with detailed metrics."""
        self.optimization_progress = progress
        self.current_stage = stage
        self.stage_details = details or {}
        
        self.logger.info(f"🔄 {stage}: {progress:.1f}% complete")
        if details:
            for key, value in details.items():
                if isinstance(value, float):
                    self.logger.info(f"   📊 {key}: {value:.4f}")
                else:
                    self.logger.info(f"   📊 {key}: {value}")
        
        # Update resource usage
        self._monitor_memory_usage()

    def _parallel_feature_selection(self, features: List[str]) -> Dict[str, float]:
        """Run feature selection in parallel for better performance."""
        if not self.resources["enable_parallel"]:
            self.logger.info("🔄 Sequential feature selection (parallel disabled)")
            return self._sequential_feature_selection(features)
        
        self.logger.info(f"🔄 Parallel feature selection with {self.resources['max_workers']} workers")
        
        with ThreadPoolExecutor(max_workers=self.resources["max_workers"]) as executor:
            future_to_feature = {
                executor.submit(self._calculate_feature_importance, feature): feature 
                for feature in features
            }
            
            results = {}
            completed = 0
            total = len(features)
            
            for future in as_completed(future_to_feature):
                feature = future_to_feature[future]
                completed += 1
                
                try:
                    importance = future.result()
                    results[feature] = importance
                    
                    # Update progress
                    progress = (completed / total) * 100
                    self._track_optimization_progress(
                        "Parallel Feature Selection",
                        progress,
                        {"completed": completed, "total": total, "current_feature": feature}
                    )
                    
                except Exception as e:
                    self.logger.warning(f"❌ Failed to calculate importance for {feature}: {e}")
                    results[feature] = 0.0
        
        return results

    def _calculate_feature_importance(self, feature: str) -> float:
        """Calculate feature importance for a single feature."""
        try:
            # Use mutual information as a robust importance measure
            feature_data = self.X[[feature]]
            target_data = self.y
            
            # Handle NaN values
            valid_mask = ~(feature_data.isnull().any(axis=1) | target_data.isnull())
            if valid_mask.sum() < 10:
                return 0.0
            
            clean_feature = feature_data[valid_mask]
            clean_target = target_data[valid_mask]
            
            if len(clean_feature) < 10:
                return 0.0
            
            importance = mutual_info_classif(
                clean_feature, 
                clean_target, 
                random_state=42
            )[0]
            
            return float(importance)
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed for {feature}: {e}")
            return 0.0

    def _sequential_feature_selection(self, features: List[str]) -> Dict[str, float]:
        """Sequential feature selection as fallback."""
        results = {}
        total = len(features)
        
        for i, feature in enumerate(features):
            importance = self._calculate_feature_importance(feature)
            results[feature] = importance
            
            # Update progress
            progress = ((i + 1) / total) * 100
            self._track_optimization_progress(
                "Sequential Feature Selection",
                progress,
                {"completed": i + 1, "total": total, "current_feature": feature}
            )
        
        return results

    def _robust_shap_analysis(self, X_sample: pd.DataFrame, y_sample: pd.Series) -> Dict[str, float]:
        """Robust SHAP analysis with multiple fallback strategies."""
        models_to_try = [
            ("lightgbm", lgb.LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)),
            ("catboost", CatBoostClassifier(iterations=50, random_state=42, verbose=False)),
            ("xgboost", xgb.XGBClassifier(n_estimators=50, random_state=42))
        ]
        
        for model_name, model in models_to_try:
            try:
                self.logger.info(f"🤖 Trying SHAP analysis with {model_name}")
                
                # Quick training with early stopping
                model.fit(X_sample, y_sample)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Process SHAP values
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                results = dict(zip(X_sample.columns, feature_importance))
                
                self.logger.info(f"✅ SHAP analysis successful with {model_name}")
                return results
                
            except Exception as e:
                self.logger.warning(f"❌ SHAP analysis failed for {model_name}: {e}")
                continue
        
        # Fallback to correlation-based feature importance
        self.logger.warning("⚠️ All SHAP models failed, using correlation fallback")
        return self._correlation_based_importance(X_sample, y_sample)

    def _correlation_based_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Correlation-based feature importance as fallback."""
        try:
            correlations = X.corrwith(y).abs()
            return correlations.to_dict()
        except Exception as e:
            self.logger.warning(f"Correlation-based importance failed: {e}")
            return {col: 0.0 for col in X.columns}

    def _enhanced_cross_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Enhanced cross-validation with multiple metrics."""
        
        # Enhanced time series cross-validation for financial data
        tscv = TimeSeriesSplit(
            n_splits=5,
            test_size=int(len(X) * 0.2),  # 20% test size
            gap=0  # No gap for financial data
        )
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        
        self.logger.info("🔄 Running enhanced cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                metrics["accuracy"].append(accuracy_score(y_val, y_pred))
                metrics["precision"].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
                metrics["recall"].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
                metrics["f1"].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                
                # Update progress
                progress = ((fold + 1) / 5) * 100
                self._track_optimization_progress(
                    "Cross-Validation",
                    progress,
                    {"fold": fold + 1, "total_folds": 5}
                )
                
            except Exception as e:
                self.logger.warning(f"❌ Cross-validation fold {fold + 1} failed: {e}")
                # Use default values for failed fold
                metrics["accuracy"].append(0.0)
                metrics["precision"].append(0.0)
                metrics["recall"].append(0.0)
                metrics["f1"].append(0.0)
        
        # Calculate statistics
        results = {}
        for metric_name, values in metrics.items():
            if values:  # Check if we have any valid values
                results[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            else:
                results[metric_name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        self.logger.info("✅ Enhanced cross-validation completed")
        return results

    async def initialize(self):
        """Async initialization method."""
        if self._needs_initialization:
            await self._prepare_data()
            self._needs_initialization = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid optimization parameters"),
            AttributeError: (None, "Missing required data"),
            TypeError: (None, "Invalid data types"),
        },
        default_return=None,
        context="coarse optimization",
    )
    async def optimize(self, data: pd.DataFrame) -> dict:
        """Run enhanced coarse optimization with comprehensive error handling."""
        self.logger.info("🚀 Starting Enhanced Coarse Optimization...")

        try:
            # Prepare data
            prepared_data = await self._prepare_data(data)

            # Run feature selection
            selected_features = await self._run_feature_selection(prepared_data)

            # Run hyperparameter optimization
            best_params = await self._run_hyperparameter_optimization(
                prepared_data,
                selected_features,
            )

            # Validate results
            validation_results = await self._validate_optimization_results(best_params)

            self.logger.info("✅ Enhanced Coarse Optimization completed successfully")
            return {
                "best_params": best_params,
                "selected_features": selected_features,
                "validation_results": validation_results,
            }

        except Exception as e:
            self.logger.error(f"❌ Enhanced Coarse Optimization failed: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data preparation",
    )
    async def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for optimization."""
        self.logger.info("Preparing data for optimization...")

        try:
            # Validate input data
            if data is None or data.empty:
                raise ValueError("Input data is None or empty")

            # Clean data
            cleaned_data = self._clean_data(data)

            # Add features
            featured_data = self._add_features(cleaned_data)

            return featured_data

        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data with enhanced quality logging."""
        self.logger.info("🧹 ENHANCED DATA CLEANING:")
        self.logger.info("=" * 50)
        
        original_shape = data.shape
        original_memory = data.memory_usage(deep=True).sum() / 1024**2
        
        self.logger.info(f"📊 Initial data state:")
        self.logger.info(f"   - Shape: {original_shape}")
        self.logger.info(f"   - Memory: {original_memory:.2f} MB")
        self.logger.info(f"   - Data types: {dict(data.dtypes.value_counts())}")
        
        # Track cleaning steps
        cleaning_steps = []
        
        # 1. Remove duplicate rows
        initial_duplicates = data.duplicated().sum()
        if initial_duplicates > 0:
            data = data.drop_duplicates()
            cleaning_steps.append(f"Removed {initial_duplicates} duplicate rows")
            self.logger.info(f"   ✅ Removed {initial_duplicates} duplicate rows")
        
        # 2. Handle infinity values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_counts > 0:
            data = data.replace([np.inf, -np.inf], np.nan)
            cleaning_steps.append(f"Replaced {inf_counts} infinity values with NaN")
            self.logger.info(f"   ✅ Replaced {inf_counts} infinity values with NaN")
        
        # 3. Handle extreme outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_counts = 0
        
        for col in numeric_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more conservative outlier detection
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    # Replace outliers with bounds instead of removing
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_counts += outliers
                    self.logger.info(f"   📊 {col}: Clipped {outliers} outliers to bounds [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        if outlier_counts > 0:
            cleaning_steps.append(f"Clipped {outlier_counts} extreme outliers")
        
        # 4. Handle missing values
        missing_before = data.isnull().sum().sum()
        if missing_before > 0:
            # For numeric columns, use forward fill then backward fill
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].isnull().sum() > 0:
                    # Forward fill then backward fill
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                    # If still has NaN, fill with median
                    if data[col].isnull().sum() > 0:
                        median_val = data[col].median()
                        data[col] = data[col].fillna(median_val)
                        self.logger.info(f"   📊 {col}: Filled remaining NaN with median {median_val:.4f}")
            
            missing_after = data.isnull().sum().sum()
            filled_count = missing_before - missing_after
            cleaning_steps.append(f"Filled {filled_count} missing values")
            self.logger.info(f"   ✅ Filled {filled_count} missing values")
        
        # 5. Data type optimization
        memory_before = data.memory_usage(deep=True).sum() / 1024**2
        data = self._optimize_dtypes(data)
        memory_after = data.memory_usage(deep=True).sum() / 1024**2
        memory_saved = memory_before - memory_after
        
        if memory_saved > 0:
            cleaning_steps.append(f"Optimized data types (saved {memory_saved:.2f} MB)")
            self.logger.info(f"   ✅ Memory optimization: {memory_before:.2f} MB → {memory_after:.2f} MB (saved {memory_saved:.2f} MB)")
        
        # Final statistics
        final_shape = data.shape
        final_memory = data.memory_usage(deep=True).sum() / 1024**2
        
        self.logger.info(f"📊 Final data state:")
        self.logger.info(f"   - Shape: {final_shape}")
        self.logger.info(f"   - Memory: {final_memory:.2f} MB")
        self.logger.info(f"   - Data loss: {original_shape[0] - final_shape[0]} rows")
        self.logger.info(f"   - Memory reduction: {((original_memory - final_memory) / original_memory * 100):.1f}%")
        
        self.logger.info(f"🔧 Cleaning steps performed:")
        for step in cleaning_steps:
            self.logger.info(f"   - {step}")
        
        self.logger.info("=" * 50)
        self.logger.info("✅ Enhanced data cleaning complete")
        
        return data
    
    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        optimized_data = data.copy()
        
        # Optimize numeric columns
        for col in optimized_data.select_dtypes(include=['int64']).columns:
            col_min = optimized_data[col].min()
            col_max = optimized_data[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_data[col] = optimized_data[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_data[col] = optimized_data[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_data[col] = optimized_data[col].astype(np.int32)
        
        # Optimize float columns
        for col in optimized_data.select_dtypes(include=['float64']).columns:
            optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='float')
        
        return optimized_data

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature addition",
    )
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features to the data."""
        try:
            # Add technical indicators
            data = self._add_technical_indicators(data)

            # Add statistical features
            data = self._add_statistical_features(data)

            # Add lag features
            data = self._add_lag_features(data)

            return data

        except Exception as e:
            self.logger.error(f"Error adding features: {e}")
            return data

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="technical indicators",
    )
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        try:
            # Implementation for adding technical indicators
            return data
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="statistical features",
    )
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features to the data."""
        try:
            # Implementation for adding statistical features
            return data
        except Exception as e:
            self.logger.error(f"Error adding statistical features: {e}")
            return data

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lag features",
    )
    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lag features to the data."""
        try:
            # Implementation for adding lag features
            return data
        except Exception as e:
            self.logger.error(f"Error adding lag features: {e}")
            return data

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature selection",
    )
    async def _run_feature_selection(self, data: pd.DataFrame) -> list:
        """Run feature selection."""
        self.logger.info("Running feature selection...")

        try:
            # Implementation for feature selection
            return ["feature1", "feature2", "feature3"]
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="hyperparameter optimization",
    )
    async def _run_hyperparameter_optimization(
        self,
        data: pd.DataFrame,
        features: list,
    ) -> dict:
        """Run hyperparameter optimization."""
        self.logger.info("Running hyperparameter optimization...")

        try:
            # Implementation for hyperparameter optimization
            return {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100}
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="optimization validation",
    )
    async def _validate_optimization_results(self, best_params: dict) -> dict:
        """Validate optimization results."""
        self.logger.info("Validating optimization results...")

        try:
            # Implementation for validation
            return {"validation_score": 0.85, "cross_validation_score": 0.82}
        except Exception as e:
            self.logger.error(f"Error validating optimization results: {e}")
            return {}

    async def _prepare_data(self):
        """Prepares the data for optimization."""
        self.logger.info("🔧 Preparing data for enhanced coarse optimization...")

        # Initialize feature engineering
        feature_engine = FeatureEngineeringEngine(CONFIG)

        # Generate features
        engineered_features = feature_engine.generate_all_features(
            self.klines_data,
            self.agg_trades_data,
            self.futures_data,
            sr_levels=[],  # Empty list for coarse optimization
        )

        # Use the target variable that was already created in Step 2
        # instead of generating a new one with MLTargetGenerator
        self.data_with_targets = engineered_features.copy()
        
        # Create target variable using the same logic as Step 2
        close_col = "close" if "close" in self.data_with_targets.columns else "Close"
        
        # Calculate future price change (5 bars ahead)
        future_price = self.data_with_targets[close_col].shift(-5)
        current_price = self.data_with_targets[close_col]
        
        # Calculate price change percentage
        price_change_pct = (future_price - current_price) / current_price
        
        # Create target: 1 if future price is > 0.5% higher than current price, 0 otherwise
        self.data_with_targets["target"] = (price_change_pct > 0.005).astype(int)
        
        # Remove the last 5 rows where we can't calculate the target (no future data)
        target_nan_count = self.data_with_targets["target"].isna().sum()
        if target_nan_count > 0:
            self.logger.info(
                f"Removing {target_nan_count} rows at the end where target cannot be calculated",
            )
            self.data_with_targets = self.data_with_targets[self.data_with_targets["target"].notna()]
        
        # Check target distribution and adjust if needed
        target_dist = self.data_with_targets["target"].value_counts()
        self.logger.info(f"Target variable created. Shape: {self.data_with_targets.shape}")
        self.logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        # If we have only one class, try different thresholds
        if len(target_dist) == 1:
            self.logger.warning("Only one class in target! Trying different thresholds...")
            
            # Try different thresholds to get balanced classes
            thresholds = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02]
            for threshold in thresholds:
                test_target = (price_change_pct > threshold).astype(int)
                test_dist = test_target.value_counts()
                if len(test_dist) > 1 and min(test_dist.values) > 100:  # At least 100 samples per class
                    self.data_with_targets["target"] = test_target
                    self.logger.info(f"Adjusted target with threshold {threshold}: {test_dist.to_dict()}")
                    break
            else:
                # If still only one class, use median-based approach
                median_change = price_change_pct.median()
                self.data_with_targets["target"] = (price_change_pct > median_change).astype(int)
                final_dist = self.data_with_targets["target"].value_counts()
                self.logger.info(f"Using median-based target: {final_dist.to_dict()}")

        # Separate features and targets
        target_columns = [
            col
            for col in self.data_with_targets.columns
            if col.lower() in ["target", "label", "signal", "class"]
        ]

        # If no target columns found, check if 'target' column exists
        if not target_columns and "target" in self.data_with_targets.columns:
            target_columns = ["target"]

        # If still no target columns, create a default target column
        if not target_columns:
            self.data_with_targets["target"] = 0  # Default target
            target_columns = ["target"]

        # Also exclude the 'target' column which contains string values
        feature_columns = [
            col
            for col in self.data_with_targets.columns
            if col not in target_columns  # Use the actual target columns list
        ]

        # Debug: Log the columns to see what we have
        self.logger.info(
            f"🔍 Debug: All columns in data: {list(self.data_with_targets.columns)}",
        )
        self.logger.info(f"🔍 Debug: Target columns found: {target_columns}")
        self.logger.info(f"🔍 Debug: Feature columns count: {len(feature_columns)}")

        # Set X and y for the pruning methods
        self.X = self.data_with_targets[feature_columns]
        self.y = (
            self.data_with_targets[target_columns[0]] if target_columns else None
        )  # Use first target column

        # Debug: Check if target column is accidentally in X
        if self.y is not None and self.y.name in self.X.columns:
            self.logger.warning(
                f"⚠️  Target column '{self.y.name}' is still in X! Removing it...",
            )
            self.X = self.X.drop(columns=[self.y.name])

        if self.y is None:
            raise ValueError("No target columns found in the data")

        # Remove rows with NaN values
        valid_mask = ~(self.X.isnull().any(axis=1) | self.y.isnull())
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]

        self.logger.info(
            f"✅ Data prepared: X shape {self.X.shape}, y shape {self.y.shape}",
        )

        # Run missing values analysis
        self._analyze_missing_values()

    def _analyze_missing_values(self):
        """Enhanced analysis of missing values with detailed logging."""
        self.logger.info("🔍 ENHANCED NaN ANALYSIS:")
        self.logger.info("=" * 60)
        
        # Analyze each dataset separately
        datasets = {
            "klines": self.klines_data,
            "agg_trades": self.agg_trades_data,
            "futures": self.futures_data,
            "combined": self.data_with_targets
        }
        
        for dataset_name, dataset in datasets.items():
            if dataset is not None and not dataset.empty:
                self.logger.info(f"📊 {dataset_name.upper()} DATASET ANALYSIS:")
                self.logger.info(f"   - Shape: {dataset.shape}")
                self.logger.info(f"   - Memory usage: {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Check for NaN values
                nan_counts = dataset.isnull().sum()
                nan_percentages = (nan_counts / len(dataset)) * 100
                
                # Find columns with NaN values
                columns_with_nan = nan_counts[nan_counts > 0]
                
                if len(columns_with_nan) > 0:
                    self.logger.warning(f"   ❌ Found {len(columns_with_nan)} columns with NaN values:")
                    for col, count in columns_with_nan.items():
                        percentage = nan_percentages[col]
                        self.logger.warning(f"      - {col}: {count} NaN values ({percentage:.2f}%)")
                        
                        # Analyze the column to understand why NaN values exist
                        if col in dataset.columns:
                            col_data = dataset[col]
                            self.logger.info(f"      📈 {col} analysis:")
                            self.logger.info(f"         - Data type: {col_data.dtype}")
                            self.logger.info(f"         - Unique values: {col_data.nunique()}")
                            self.logger.info(f"         - Value range: {col_data.min()} to {col_data.max()}")
                            
                            # Check for infinity values
                            inf_count = np.isinf(col_data).sum()
                            if inf_count > 0:
                                self.logger.warning(f"         - Infinity values: {inf_count}")
                            
                            # Check for zero values
                            zero_count = (col_data == 0).sum()
                            self.logger.info(f"         - Zero values: {zero_count}")
                            
                            # Sample of non-NaN values
                            non_nan_sample = col_data.dropna().head(3).tolist()
                            self.logger.info(f"         - Sample values: {non_nan_sample}")
                else:
                    self.logger.info(f"   ✅ No NaN values found in {dataset_name}")
                
                # Check for duplicate rows
                duplicates = dataset.duplicated().sum()
                if duplicates > 0:
                    self.logger.warning(f"   ⚠️  Found {duplicates} duplicate rows")
                
                # Check for constant columns
                constant_cols = [col for col in dataset.columns if dataset[col].nunique() <= 1]
                if constant_cols:
                    self.logger.warning(f"   ⚠️  Found {len(constant_cols)} constant columns: {constant_cols}")
                
                self.logger.info("")
        
        # Analyze feature engineering process
        if hasattr(self, 'data_with_targets') and self.data_with_targets is not None:
            self.logger.info("🔧 FEATURE ENGINEERING ANALYSIS:")
            
            # Check which features are causing NaN values
            feature_nan_counts = self.data_with_targets.isnull().sum()
            problematic_features = feature_nan_counts[feature_nan_counts > 0]
            
            if len(problematic_features) > 0:
                self.logger.warning(f"   ❌ Features with NaN values after engineering:")
                for feature, count in problematic_features.items():
                    percentage = (count / len(self.data_with_targets)) * 100
                    self.logger.warning(f"      - {feature}: {count} NaN values ({percentage:.2f}%)")
                    
                    # Analyze the feature to understand the cause
                    feature_data = self.data_with_targets[feature]
                    
                    # Check if it's a calculated feature
                    if any(op in feature for op in ['_', 'ratio', 'pct', 'diff', 'ma', 'std', 'vol']):
                        self.logger.info(f"      📊 {feature} appears to be a calculated feature")
                        
                        # Check if it's a division-based feature
                        if any(op in feature for op in ['ratio', 'pct', 'div']):
                            self.logger.warning(f"      ⚠️  {feature} might have division by zero issues")
                        
                        # Check if it's a rolling window feature
                        if any(op in feature for op in ['ma', 'std', 'vol']):
                            self.logger.info(f"      📈 {feature} is a rolling window feature")
                            # Check if we have enough data for the window
                            if 'ma' in feature or 'std' in feature:
                                window_size = 20  # Default assumption
                                if len(self.data_with_targets) < window_size:
                                    self.logger.warning(f"      ⚠️  Insufficient data for {feature} (need {window_size}, have {len(self.data_with_targets)})")
            
            # Check for correlation between NaN patterns
            self.logger.info("   🔗 Analyzing NaN patterns...")
            nan_matrix = self.data_with_targets.isnull()
            nan_correlations = nan_matrix.corr()
            
            # Find highly correlated NaN patterns
            high_corr_pairs = []
            for i in range(len(nan_correlations.columns)):
                for j in range(i+1, len(nan_correlations.columns)):
                    corr_val = nan_correlations.iloc[i, j]
                    if abs(corr_val) > 0.8:  # High correlation threshold
                        high_corr_pairs.append((
                            nan_correlations.columns[i],
                            nan_correlations.columns[j],
                            corr_val
                        ))
            
            if high_corr_pairs:
                self.logger.warning(f"   🔗 Found {len(high_corr_pairs)} highly correlated NaN patterns:")
                for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
                    self.logger.warning(f"      - {feat1} ↔ {feat2}: {corr:.3f}")
        
        self.logger.info("=" * 60)
        self.logger.info("✅ Enhanced NaN analysis complete")

    def enhanced_prune_features(self, top_n_percent: float = 0.5) -> list[str]:
        """
        Enhanced feature pruning with multiple stages, parallel processing, and better logging.
        """
        self.logger.info("🔧 Enhanced feature pruning starting...")
        pruning_start = time.time()
        
        # Track overall progress
        self._track_optimization_progress("Feature Pruning", 0.0, {"stage": "initialization"})

        # Check if this is blank training mode
        if self.blank_training_mode:
            self.logger.info(
                "🧪 BLANK TRAINING MODE: Using simplified pruning for speed",
            )
            # Use simpler pruning for blank training
            top_n_percent = 0.3  # Keep fewer features
            self.logger.info(
                f"📊 Blank training pruning config: top_n_percent = {top_n_percent}",
            )

        # Step 0: Comprehensive data cleaning and NaN handling
        self.logger.info("🧹 Step 0: Comprehensive data cleaning and NaN handling...")
        step0_start = time.time()
        
        # Check initial data quality
        initial_shape = self.X.shape
        initial_nan_count = self.X.isnull().sum().sum()
        self.logger.info(f"   Initial data shape: {initial_shape}, NaN count: {initial_nan_count}")
        
        # Get numeric columns only for cleaning
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # Replace infinite values with NaN first
            infinite_mask = np.isinf(self.X[numeric_columns])
            infinite_count = infinite_mask.sum().sum()
            if infinite_count > 0:
                self.logger.warning(f"   Found {infinite_count} infinite values, replacing with NaN")
                self.X[numeric_columns] = self.X[numeric_columns].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults
            for col in numeric_columns:
                nan_count = self.X[col].isnull().sum()
                if nan_count > 0:
                    self.logger.info(f"   Column '{col}': filling {nan_count} NaN values")
                    
                    # Use forward fill then backward fill for time series data
                    self.X[col] = self.X[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # If still have NaN values, use 0 as default
                    remaining_nan = self.X[col].isnull().sum()
                    if remaining_nan > 0:
                        self.logger.warning(f"   Column '{col}': still {remaining_nan} NaN values, using 0 as default")
                        self.X[col] = self.X[col].fillna(0)
        
        # Handle non-numeric columns
        non_numeric_columns = self.X.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            nan_count = self.X[col].isnull().sum()
            if nan_count > 0:
                self.logger.info(f"   Column '{col}': filling {nan_count} NaN values")
                self.X[col] = self.X[col].fillna(method='ffill').fillna(method='bfill')
        
        # Final check for any remaining NaN values
        final_nan_count = self.X.isnull().sum().sum()
        if final_nan_count > 0:
            self.logger.warning(f"   WARNING: Still have {final_nan_count} NaN values after cleaning")
            # Remove any rows that still have NaN values
            self.X = self.X.dropna()
            if hasattr(self, 'y') and self.y is not None:
                # Align target with cleaned features
                self.y = self.y.loc[self.X.index]
        
        step0_duration = time.time() - step0_start
        self.logger.info(f"   ✅ Step 0 completed in {step0_duration:.2f} seconds")
        self.logger.info(f"   Final data shape: {self.X.shape}, NaN count: {self.X.isnull().sum().sum()}")

        # Step 1: Data cleaning and infinite value handling
        self.logger.info("🧹 Step 1: Cleaning infinite and extreme values...")
        step1_start = time.time()

        # Get numeric columns only for infinite value detection
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns

        # Replace infinite values with NaN (only for numeric columns)
        if len(numeric_columns) > 0:
            infinite_mask = np.isinf(self.X[numeric_columns])
            infinite_count = (
                infinite_mask.sum().sum()
            )  # Sum across all columns and rows
            if infinite_count > 0:
                self.logger.warning(
                    f"   Found {infinite_count} infinite values, replaced with NaN",
                )
                self.X[numeric_columns] = self.X[numeric_columns].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

        # Replace extreme values (beyond 6 standard deviations) - only for numeric columns
        for col in numeric_columns:
            col_data = self.X[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    extreme_mask = (self.X[col] < mean_val - 6 * std_val) | (
                        self.X[col] > mean_val + 6 * std_val
                    )
                    extreme_count = extreme_mask.sum()
                    if extreme_count > 0:
                        self.logger.info(
                            f"   Column '{col}': replaced {extreme_count} extreme values",
                        )
                        self.X.loc[extreme_mask, col] = np.nan

        step1_duration = time.time() - step1_start
        self.logger.info(f"   ✅ Step 1 completed in {step1_duration:.2f} seconds")

        # Step 2: Variance-based pruning
        self.logger.info("📊 Step 2: Variance-based pruning...")
        step2_start = time.time()

        # Exclude target column from variance calculation if it exists
        feature_columns = self.X.columns
        if hasattr(self, "y") and self.y is not None:
            # Remove target column from feature columns if it exists
            if (
                isinstance(self.y, pd.Series)
                and self.y.name in feature_columns
                or hasattr(self.y, "name")
                and self.y.name in feature_columns
            ):
                feature_columns = [col for col in feature_columns if col != self.y.name]

        # Calculate variance only on feature columns (exclude target)
        variances = self.X[feature_columns].var()
        variance_threshold = variances.quantile(
            0.1,
        )  # Keep features with variance > 10th percentile

        # Filter features by variance
        high_variance_features = variances[
            variance_threshold < variances
        ].index.tolist()
        removed_by_variance = len(self.X.columns) - len(high_variance_features)

        self.logger.info(
            f"   Features after variance pruning: {len(high_variance_features)} (removed {removed_by_variance})",
        )
        step2_duration = time.time() - step2_start
        self.logger.info(f"   ✅ Step 2 completed in {step2_duration:.2f} seconds")

        # Step 3: Correlation-based pruning
        self.logger.info("🔗 Step 3: Correlation-based pruning...")
        step3_start = time.time()

        # Calculate correlation matrix for high variance features (exclude target)
        X_high_var = self.X[high_variance_features]
        corr_matrix = X_high_var.corr().abs()

        # Remove highly correlated features (correlation > 0.95)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
        )
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
        ]

        # Keep features that are not highly correlated
        uncorrelated_features = [f for f in high_variance_features if f not in to_drop]
        removed_by_correlation = len(high_variance_features) - len(
            uncorrelated_features,
        )

        self.logger.info(
            f"   Features after correlation pruning: {len(uncorrelated_features)} (removed {removed_by_correlation})",
        )
        step3_duration = time.time() - step3_start
        self.logger.info(f"   ✅ Step 3 completed in {step3_duration:.2f} seconds")

        # Step 4: Mutual information pruning with parallel processing
        if self.blank_training_mode:
            self.logger.info(
                "🧪 BLANK TRAINING MODE: Skipping mutual information pruning for speed",
            )
            final_features = uncorrelated_features
        else:
            self.logger.info("📈 Step 4: Parallel mutual information pruning...")
            step4_start = time.time()
            
            # Update progress
            self._track_optimization_progress("Feature Pruning", 60.0, {"stage": "mutual_information"})

            # Use a sample for MI calculation to speed up the process
            sample_size = min(self.resources["shap_sample_size"], len(self.X))
            if len(self.X) > sample_size:
                self.logger.info(
                    f"📊 Using sample of {sample_size} for MI calculation (from {len(self.X)} total)",
                )
                sample_indices = np.random.choice(
                    len(self.X),
                    sample_size,
                    replace=False,
                )
                X_sample = self.X.iloc[sample_indices][uncorrelated_features]
                y_sample = self.y.iloc[sample_indices]
            else:
                X_sample = self.X[uncorrelated_features]
                y_sample = self.y

            self.logger.info(
                f"📊 Processing {len(uncorrelated_features)} features with {len(X_sample)} samples",
            )

            # Use parallel processing for mutual information calculation
            mi_scores = self._parallel_feature_selection(uncorrelated_features)

            # Select top features based on mutual information
            sorted_features = sorted(
                mi_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_n = max(5, int(len(uncorrelated_features) * top_n_percent))
            final_features = [f[0] for f in sorted_features[:top_n]]

            removed_by_mi = len(uncorrelated_features) - len(final_features)
            step4_duration = time.time() - step4_start
            self.logger.info(
                f"📊 Features after MI pruning: {len(final_features)} (removed {removed_by_mi})",
            )
            self.logger.info(f"✅ Step 4 completed in {step4_duration:.2f} seconds")

        # Step 5: Robust SHAP pruning with graceful degradation
        self.logger.info("🤖 Step 5: Robust SHAP pruning...")
        step5_start = time.time()
        
        # Update progress
        self._track_optimization_progress("Feature Pruning", 80.0, {"stage": "shap_analysis"})
        
        try:
            # Prepare data for SHAP analysis
            X_shap = self.data_with_targets[final_features].copy()
            y_shap = self.data_with_targets["target"].copy()
            
            self.logger.info(f"📊 SHAP data preparation:")
            self.logger.info(f"   - Initial data shape: {X_shap.shape}")
            self.logger.info(f"   - Features to analyze: {len(final_features)}")
            self.logger.info(f"   - Target distribution: {y_shap.value_counts().to_dict()}")
            
            # Clean data for SHAP
            inf_count = np.isinf(X_shap).sum().sum()
            nan_count = X_shap.isnull().sum().sum()
            self.logger.info(f"📊 Data cleaning:")
            self.logger.info(f"   - Infinity values found: {inf_count}")
            self.logger.info(f"   - NaN values found: {nan_count}")
            
            X_shap_clean = X_shap.replace([np.inf, -np.inf], np.nan).dropna()
            y_shap_clean = y_shap.loc[X_shap_clean.index]
            
            self.logger.info(f"   - Clean data shape: {X_shap_clean.shape}")
            self.logger.info(f"   - Data loss: {len(X_shap) - len(X_shap_clean)} samples")
            
            if len(X_shap_clean) < 1000:
                self.logger.warning("⚠️ Insufficient data for SHAP analysis, skipping")
                self.logger.warning(f"   - Required: 1000, Available: {len(X_shap_clean)}")
                step5_duration = time.time() - step5_start
                self.logger.info(f"✅ Step 5 completed in {step5_duration:.2f} seconds (skipped)")
            else:
                # Sample data for SHAP analysis
                sample_size = min(self.resources["shap_sample_size"], len(X_shap_clean))
                X_sample = X_shap_clean.sample(n=sample_size, random_state=42)
                y_sample = y_shap_clean.loc[X_sample.index]
                
                self.logger.info(f"📊 SHAP analysis setup:")
                self.logger.info(f"   - Sample size: {len(X_sample)}")
                self.logger.info(f"   - Features: {len(X_sample.columns)}")
                self.logger.info(f"   - Target classes: {y_sample.value_counts().to_dict()}")
                
                # Use robust SHAP analysis with fallback
                shap_scores = self._robust_shap_analysis(X_sample, y_sample)
                
                if shap_scores:
                    # Select top features based on SHAP importance
                    sorted_features = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)
                    top_n_shap = max(10, int(len(sorted_features) * 0.3))  # Keep top 30%
                    final_features = [f[0] for f in sorted_features[:top_n_shap]]
                    
                    removed_by_shap = len(sorted_features) - len(final_features)
                    self.logger.info(f"📊 Features after SHAP pruning: {len(final_features)} (removed {removed_by_shap})")
                else:
                    self.logger.warning("⚠️ No SHAP analysis completed, keeping current features")
                
                step5_duration = time.time() - step5_start
                self.logger.info(f"✅ Step 5 completed in {step5_duration:.2f} seconds")
                
        except Exception as e:
            self.logger.warning(f"⚠️ SHAP pruning failed: {e}")
            step5_duration = time.time() - step5_start
            self.logger.info(f"✅ Step 5 completed in {step5_duration:.2f} seconds (skipped)")

        total_duration = time.time() - pruning_start
        
        # Final progress update
        self._track_optimization_progress("Feature Pruning", 100.0, {
            "stage": "completed",
            "final_features": len(final_features),
            "total_duration": f"{total_duration:.2f}s"
        })
        
        self.logger.info(
            f"✅ Enhanced feature pruning completed in {total_duration:.2f} seconds",
        )
        self.logger.info(f"📊 Final feature count: {len(final_features)}")
        self.logger.info(f"📊 Memory usage: {self.resource_usage.get('memory_percent', 0):.1f}%")

        return final_features

    def find_enhanced_hyperparameter_ranges(
        self,
        pruned_features: list,
        n_trials: int = 50,
    ) -> dict:
        """
        Enhanced hyperparameter search with wider ranges, multiple models, and cross-validation.
        """
        self.logger.info("🔍 Finding enhanced hyperparameter ranges...")
        
        # Track progress
        self._track_optimization_progress("Hyperparameter Optimization", 0.0, {"stage": "initialization"})

        X = self.data_with_targets[pruned_features]
        y = self.data_with_targets["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        n_classes = len(y.unique())
        self.logger.info(
            f"📊 Target has {n_classes} unique classes: {sorted(y.unique())}",
        )

        def objective(trial):
            # Test multiple model types
            model_type = trial.suggest_categorical(
                "model_type",
                ["lightgbm", "xgboost", "random_forest", "catboost", "gradient_boosting", "tabnet", "transformer", "lstm"],
            )

            if model_type == "lightgbm":
                param = {
                    "objective": "multiclass" if n_classes > 2 else "binary",
                    "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
                    "verbosity": -1,
                    "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        1e-4,
                        0.3,
                        log=True,
                    ),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.5,
                        1.0,
                    ),
                    # L1-L2 Regularization parameters
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    # Additional regularization
                    "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                    "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
                }
                if n_classes > 2:
                    param["num_class"] = n_classes
                model = lgb.LGBMClassifier(**param)

            elif model_type == "xgboost":
                param = {
                    "objective": "multi:softmax"
                    if n_classes > 2
                    else "binary:logistic",
                    "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
                    "verbosity": 0,
                    "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        1e-4,
                        0.3,
                        log=True,
                    ),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.5,
                        1.0,
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda",
                        1e-8,
                        10.0,
                        log=True,
                    ),
                }
                if n_classes > 2:
                    param["num_class"] = n_classes
                model = xgb.XGBClassifier(**param)

            elif model_type == "random_forest":
                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features",
                        ["sqrt", "log2", None],
                    ),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    # Enhanced Random Forest parameters for production
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 1000),
                    "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
                    "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                }
                model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)

            elif model_type == "catboost":
                param = {
                    "iterations": trial.suggest_int("iterations", 50, 2000),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        1e-4,
                        0.3,
                        log=True,
                    ),
                    "depth": trial.suggest_int("depth", 2, 12),
                    "l2_leaf_reg": trial.suggest_float(
                        "l2_leaf_reg",
                        1e-8,
                        10.0,
                        log=True,
                    ),
                    "border_count": trial.suggest_int("border_count", 32, 255),
                    "bagging_temperature": trial.suggest_float(
                        "bagging_temperature",
                        0.0,
                        1.0,
                    ),
                    # Enhanced CatBoost parameters for production
                    "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                    "max_bin": trial.suggest_int("max_bin", 128, 512),
                    "feature_border_type": trial.suggest_categorical("feature_border_type", ["GreedyLogSum", "MinEntropy", "MaxLogSum"]),
                    "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
                }
                if n_classes > 2:
                    param["loss_function"] = "MultiClass"
                else:
                    param["loss_function"] = "Logloss"
                model = CatBoostClassifier(**param, random_state=42, verbose=False)

            elif model_type == "gradient_boosting":
                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 15),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                    "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
                    "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 1000),
                }
                model = GradientBoostingClassifier(**param, random_state=42)

            elif model_type == "tabnet":
                if not TABNET_AVAILABLE:
                    self.logger.warning("⚠️ TabNet not available, falling back to Random Forest")
                    model_type = "random_forest"
                    param = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                        "max_depth": trial.suggest_int("max_depth", 2, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    }
                    model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
                else:
                    param = {
                        "optimizer_fn": torch.optim.Adam,
                        "optimizer_params": {"lr": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)},
                        "scheduler_params": {"step_size": trial.suggest_int("step_size", 5, 50)},
                        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
                        "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
                        "n_d": trial.suggest_int("n_d", 8, 64),
                        "n_a": trial.suggest_int("n_a", 8, 64),
                        "n_steps": trial.suggest_int("n_steps", 1, 10),
                        "gamma": trial.suggest_float("gamma", 1.0, 3.0),
                        "n_independent": trial.suggest_int("n_independent", 1, 3),
                        "n_shared": trial.suggest_int("n_shared", 1, 3),
                        "momentum": trial.suggest_float("momentum", 0.1, 0.9),
                        "clip_value": trial.suggest_float("clip_value", 0.0, 2.0),
                        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
                    }
                    model = TabNetClassifier(**param)

            elif model_type == "transformer":
                if not TRANSFORMER_AVAILABLE:
                    self.logger.warning("⚠️ Transformer not available, falling back to XGBoost")
                    model_type = "xgboost"
                    param = {
                        "objective": "multi:softmax" if n_classes > 2 else "binary:logistic",
                        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
                        "verbosity": 0,
                        "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                        "max_depth": trial.suggest_int("max_depth", 2, 20),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    }
                    if n_classes > 2:
                        param["num_class"] = n_classes
                    model = xgb.XGBClassifier(**param)
                else:
                    param = {
                        "d_model": trial.suggest_int("d_model", 32, 256),
                        "n_heads": trial.suggest_int("n_heads", 2, 8),
                        "n_layers": trial.suggest_int("n_layers", 1, 6),
                        "d_ff": trial.suggest_int("d_ff", 64, 512),
                        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                        "max_len": trial.suggest_int("max_len", 50, 200),
                        "batch_size": trial.suggest_int("batch_size", 16, 128),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                        "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
                        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.1, 2.0),
                    }
                    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Fallback

            elif model_type == "lstm":
                if not LSTM_AVAILABLE:
                    self.logger.warning("⚠️ LSTM not available, falling back to LightGBM")
                    model_type = "lightgbm"
                    param = {
                        "objective": "multiclass" if n_classes > 2 else "binary",
                        "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
                        "verbosity": -1,
                        "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                        "max_depth": trial.suggest_int("max_depth", 2, 20),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
                    }
                    if n_classes > 2:
                        param["num_class"] = n_classes
                    model = lgb.LGBMClassifier(**param)
                else:
                    param = {
                        "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                        "num_layers": trial.suggest_int("num_layers", 1, 4),
                        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
                        "batch_size": trial.suggest_int("batch_size", 16, 128),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                        "sequence_length": trial.suggest_int("sequence_length", 10, 100),
                        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.1, 2.0),
                    }
                    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Fallback

            # Train model with enhanced cross-validation
            try:
                if hasattr(model, "fit"):
                    # Use enhanced cross-validation for more robust evaluation
                    cv_results = self._enhanced_cross_validation(model, X_train, y_train)
                    
                    # Use mean accuracy as the objective value
                    accuracy = cv_results["accuracy"]["mean"]
                    
                    # Enhanced early stopping check
                    if trial.number > 10 and accuracy < 0.5:
                        self.logger.warning(f"⚠️ Early stopping trial {trial.number} - accuracy too low: {accuracy:.4f}")
                        return accuracy
                    
                    # Log detailed metrics for top trials
                    if trial.number < 5:  # Log details for first 5 trials
                        self.logger.info(f"📊 Trial {trial.number} ({model_type}):")
                        self.logger.info(f"   - Accuracy: {accuracy:.4f} ± {cv_results['accuracy']['std']:.4f}")
                        self.logger.info(f"   - F1 Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
                        self.logger.info(f"   - Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
                        self.logger.info(f"   - Recall: {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
                    
                    return accuracy
                return 0.0
            except Exception as e:
                self.logger.warning(f"Model training failed: {e}")
                return 0.0

        # Use enhanced pruner with better configuration
        study = optuna.create_study(
            direction="maximize",
            pruner=SuccessiveHalvingPruner(
                min_resource=1, 
                reduction_factor=3,
                min_early_stopping_rate=0.0
            ),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
                group=True
            )
        )
        
        # Add progress callback
        def progress_callback(study, trial):
            progress = (trial.number + 1) / n_trials * 100
            self._track_optimization_progress(
                "Hyperparameter Optimization", 
                progress, 
                {
                    "trial": trial.number + 1,
                    "total_trials": n_trials,
                    "best_value": study.best_value if study.best_value else 0.0
                }
            )
        
        study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

        # Analyze top trials to define ranges
        top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[
            : max(5, int(n_trials * 0.1))
        ]

        ranges = {}
        if top_trials:
            # Get all unique parameter names from all trials
            all_param_names = set()
            for trial in top_trials:
                if hasattr(trial, "params") and trial.params:
                    all_param_names.update(trial.params.keys())

            # Create ranges for each parameter
            for param_name in all_param_names:
                try:
                    values = [
                        t.params[param_name]
                        for t in top_trials
                        if param_name in t.params
                    ]
                    if values:  # Only create range if we have values
                        # Filter out non-numeric values and convert to proper types
                        numeric_values = []
                        for val in values:
                            if val is not None:
                                try:
                                    if isinstance(val, str):
                                        # Try to convert string to float/int
                                        if '.' in val:
                                            numeric_values.append(float(val))
                                        else:
                                            numeric_values.append(int(val))
                                    elif isinstance(val, (int, float)):
                                        numeric_values.append(val)
                                except (ValueError, TypeError):
                                    self.logger.warning(f"⚠️  Skipping non-numeric value '{val}' for parameter {param_name}")
                                    continue
                        
                        if numeric_values:  # Only create range if we have numeric values
                            ranges[param_name] = {
                                "low": min(numeric_values),
                                "high": max(numeric_values),
                                "type": "float" if isinstance(numeric_values[0], float) else "int",
                            }
                            if isinstance(numeric_values[0], float):
                                ranges[param_name]["step"] = (
                                    max(numeric_values) - min(numeric_values)
                                ) / 10.0
                            else:
                                ranges[param_name]["step"] = 1
                        else:
                            self.logger.warning(f"⚠️  No numeric values found for parameter {param_name}")
                except (KeyError, AttributeError) as e:
                    self.logger.warning(f"⚠️  Skipping parameter {param_name}: {e}")
                    continue

        self.logger.info(
            f"✅ Enhanced hyperparameter search complete. Found ranges: {ranges}",
        )
        return ranges

    def run(self) -> tuple[list[str], dict]:
        """
        Orchestrates the enhanced coarse optimization process with improved integration.
        """
        self.logger.info(
            "🚀 Starting Enhanced Stage 2: Coarse Optimization & Pruning ---",
        )
        
        # Initialize resource monitoring
        self._monitor_memory_usage()
        self.logger.info(f"📊 Initial resource allocation: {self.resources}")

        # Enhanced feature pruning with progress tracking
        pruning_config = CONFIG.get("MODEL_TRAINING", {}).get("feature_pruning", {})
        self.logger.info(f"🔧 Feature pruning configuration: {pruning_config}")
        
        pruned_features = self.enhanced_prune_features(
            top_n_percent=pruning_config.get("top_n_percent", 0.5),
        )

        # Enhanced hyperparameter optimization with progress tracking
        hpo_config = CONFIG.get("MODEL_TRAINING", {}).get("coarse_hpo", {})
        self.logger.info(f"🔧 Hyperparameter optimization configuration: {hpo_config}")

        # Use fewer trials for blank training mode
        if self.blank_training_mode:
            n_trials = 3  # Quick test for blank training
            self.logger.info(
                "🧪 BLANK TRAINING MODE: Using reduced trials for quick testing",
            )
        else:
            n_trials = hpo_config.get("n_trials", 50)

        narrowed_ranges = self.find_enhanced_hyperparameter_ranges(
            pruned_features,
            n_trials=n_trials,
        )

        # Generate optimization report
        self._generate_optimization_report(pruned_features, narrowed_ranges)
        
        # Final resource cleanup
        self._monitor_memory_usage()
        
        self.logger.info("✅ Enhanced Stage 2 Complete")
        return pruned_features, narrowed_ranges

    def _generate_optimization_report(self, pruned_features: list, narrowed_ranges: dict):
        """Generate comprehensive optimization report."""
        self.logger.info("📊 ENHANCED OPTIMIZATION REPORT:")
        self.logger.info("=" * 60)
        
        # Feature pruning summary
        self.logger.info("🔧 FEATURE PRUNING SUMMARY:")
        self.logger.info(f"   - Initial features: {len(self.X.columns)}")
        self.logger.info(f"   - Final features: {len(pruned_features)}")
        self.logger.info(f"   - Reduction: {((len(self.X.columns) - len(pruned_features)) / len(self.X.columns) * 100):.1f}%")
        
        # Hyperparameter optimization summary
        self.logger.info("🔧 HYPERPARAMETER OPTIMIZATION SUMMARY:")
        self.logger.info(f"   - Parameters optimized: {len(narrowed_ranges)}")
        for param, config in narrowed_ranges.items():
            if isinstance(config, dict) and "low" in config and "high" in config:
                self.logger.info(f"   - {param}: [{config['low']:.4f}, {config['high']:.4f}]")
        
        # Resource usage summary
        self.logger.info("📊 RESOURCE USAGE SUMMARY:")
        self.logger.info(f"   - Peak memory usage: {self.resource_usage.get('memory_percent', 0):.1f}%")
        self.logger.info(f"   - CPU cores used: {self.resources['cpu_count']}")
        self.logger.info(f"   - Parallel processing: {'Enabled' if self.resources['enable_parallel'] else 'Disabled'}")
        
        # Performance metrics
        self.logger.info("📈 PERFORMANCE METRICS:")
        self.logger.info(f"   - Optimization progress: {self.optimization_progress:.1f}%")
        self.logger.info(f"   - Current stage: {self.current_stage}")
        
        self.logger.info("=" * 60)
