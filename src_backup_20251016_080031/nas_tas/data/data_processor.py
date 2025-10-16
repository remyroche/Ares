"""
Unified Data Processor for NAS/TAS Systems

This module provides comprehensive data processing capabilities that consolidate
data handling logic previously scattered across NAS and TAS implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, tprint_logged, LogLevel
)


@dataclass
class DataProcessingConfig:
    """Configuration for data processing."""
    
    # Data cleaning
    handle_missing_values: bool = True
    missing_value_strategy: str = "median"  # mean, median, mode, drop, interpolate
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    
    # Feature scaling
    enable_scaling: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust, none
    scale_features: bool = True
    scale_target: bool = False
    
    # Feature engineering
    enable_feature_engineering: bool = True
    create_time_features: bool = True
    create_interaction_features: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = False
    reduction_method: str = "pca"  # pca, tsne, umap
    n_components: Optional[int] = None
    variance_threshold: float = 0.95
    
    # Feature selection
    enable_feature_selection: bool = False
    selection_method: str = "k_best"  # k_best, percentile, mutual_info
    n_features: int = 100
    feature_percentile: float = 50.0
    
    # Data validation
    validate_data: bool = True
    min_data_quality_score: float = 0.7
    
    # Memory optimization
    optimize_memory: bool = True
    use_category_dtype: bool = True
    downcast_numeric: bool = True
    
    # Custom processing functions
    custom_preprocessors: List[Callable] = field(default_factory=list)
    custom_postprocessors: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == 'custom_preprocessors' or field_name == 'custom_postprocessors':
                config_dict[field_name] = [func.__name__ for func in field_value]
            else:
                config_dict[field_name] = field_value
        return config_dict


@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""
    
    # Completeness metrics
    missing_value_percentage: float = 0.0
    complete_rows_percentage: float = 0.0
    complete_columns_percentage: float = 0.0
    
    # Consistency metrics
    duplicate_rows_count: int = 0
    duplicate_rows_percentage: float = 0.0
    
    # Validity metrics
    invalid_values_count: int = 0
    invalid_values_percentage: float = 0.0
    
    # Uniqueness metrics
    unique_values_ratio: float = 0.0
    high_cardinality_columns: List[str] = field(default_factory=list)
    
    # Distribution metrics
    skewed_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    
    # Overall quality score
    overall_quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'missing_value_percentage': self.missing_value_percentage,
            'complete_rows_percentage': self.complete_rows_percentage,
            'complete_columns_percentage': self.complete_columns_percentage,
            'duplicate_rows_count': self.duplicate_rows_count,
            'duplicate_rows_percentage': self.duplicate_rows_percentage,
            'invalid_values_count': self.invalid_values_count,
            'invalid_values_percentage': self.invalid_values_percentage,
            'unique_values_ratio': self.unique_values_ratio,
            'high_cardinality_columns': self.high_cardinality_columns,
            'skewed_columns': self.skewed_columns,
            'constant_columns': self.constant_columns,
            'overall_quality_score': self.overall_quality_score
        }


@dataclass
class DataValidationResult:
    """Result of data validation."""
    
    # Validation status
    validation_passed: bool = False
    validation_score: float = 0.0
    
    # Quality metrics
    quality_metrics: DataQualityMetrics = field(default_factory=DataQualityMetrics)
    
    # Issues found
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Data statistics
    data_shape: Tuple[int, int] = (0, 0)
    data_types: Dict[str, str] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    
    # Processing suggestions
    suggested_preprocessing: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'validation_passed': self.validation_passed,
            'validation_score': self.validation_score,
            'quality_metrics': self.quality_metrics.to_dict(),
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'data_shape': list(self.data_shape),
            'data_types': self.data_types,
            'memory_usage_mb': self.memory_usage_mb,
            'suggested_preprocessing': self.suggested_preprocessing
        }


class UnifiedDataProcessor:
    """
    Unified data processor for NAS/TAS systems.
    
    This class consolidates data processing logic that was previously
    scattered across NAS and TAS implementations, providing a unified
    interface for data preprocessing, cleaning, and validation.
    """
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize unified data processor.
        
        Args:
            config: Data processing configuration
        """
        tprint_info("Initializing Unified Data Processor")
        
        self.config = config or DataProcessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Log configuration
        tprint_structured({
            "data_processing_config": self.config.to_dict()
        }, LogLevel.INFO)
        
        # Processing components
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.dimension_reducer = None
        
        # Processing state
        tprint_success("Unified Data Processor initialized successfully")
        self.is_fitted = False
        self.feature_names = None
        self.target_name = None
        
        # Processing history
        self.processing_history = []
        
        tprint_info("Unified data processor initialized")
    
    def process_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], DataValidationResult]:
        """
        Process data with comprehensive preprocessing pipeline.
        
        Args:
            X: Feature data
            y: Target data (optional)
            fit: Whether to fit preprocessing components
            
        Returns:
            Tuple of (processed_X, processed_y, validation_result)
        """
        tprint_info("Starting data processing pipeline")
        start_time = datetime.now()
        
        # Log processing parameters
        X_shape = X.shape if hasattr(X, 'shape') else f"Unknown shape: {type(X)}"
        y_shape = y.shape if y is not None and hasattr(y, 'shape') else "No target"
        
        tprint_structured({
            "data_processing": {
                "input_X_shape": X_shape,
                "input_y_shape": y_shape,
                "fit_mode": fit,
                "data_type_X": type(X).__name__,
                "data_type_y": type(y).__name__ if y is not None else "None"
            }
        }, LogLevel.INFO)
        
        try:
            # Convert to DataFrame if needed
            tprint_debug("Converting data to appropriate formats")
            if isinstance(X, np.ndarray):
                tprint_debug(f"Converting numpy array to DataFrame: {X.shape}")
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
                tprint_success(f"Created DataFrame with {X.shape[1]} features")
            
            if y is not None and isinstance(y, np.ndarray):
                tprint_debug(f"Converting target numpy array to Series: {y.shape}")
                y = pd.Series(y, name="target")
                tprint_success("Created target Series")
            
            # Store original feature names
            if fit:
                tprint_debug("Storing feature and target names for fitting")
                self.feature_names = list(X.columns)
                self.target_name = y.name if y is not None else None
                tprint_success(f"Stored {len(self.feature_names)} feature names")
            
            # Validate data quality
            tprint_debug("Validating data quality")
            with tprint_timer("data_validation", LogLevel.DEBUG):
                validation_result = self.validate_data(X, y)
            
            tprint_structured({
                "validation_results": {
                    "validation_passed": validation_result.validation_passed,
                    "validation_score": validation_result.validation_score,
                    "min_required_score": self.config.min_data_quality_score
                }
            }, LogLevel.DEBUG)
            
            if not validation_result.validation_passed and self.config.validate_data:
                tprint_warning(f"Data validation failed (score: {validation_result.validation_score:.3f})")
                if validation_result.validation_score < self.config.min_data_quality_score:
                    error_msg = f"Data quality too low: {validation_result.validation_score:.3f} < {self.config.min_data_quality_score}"
                    tprint_error(error_msg)
                    raise ValueError(error_msg)
            else:
                tprint_success(f"Data validation passed (score: {validation_result.validation_score:.3f})")
            
            # Apply preprocessing pipeline
            tprint_debug("Starting preprocessing pipeline")
            processed_X = X.copy()
            processed_y = y.copy() if y is not None else None
            
            # Memory optimization
            if self.config.optimize_memory:
                tprint_debug("Optimizing memory usage")
                with tprint_timer("memory_optimization", LogLevel.DEBUG):
                    processed_X = self._optimize_memory(processed_X)
                    if processed_y is not None:
                        processed_y = self._optimize_memory(processed_y)
                tprint_success("Memory optimization completed")
            
            # Custom preprocessing
            if self.config.custom_preprocessors:
                tprint_debug(f"Applying {len(self.config.custom_preprocessors)} custom preprocessors")
                for i, preprocessor in enumerate(self.config.custom_preprocessors):
                    try:
                        tprint_debug(f"Applying custom preprocessor {i+1}/{len(self.config.custom_preprocessors)}: {preprocessor.__name__}")
                        processed_X = preprocessor(processed_X)
                        if processed_y is not None:
                            processed_y = preprocessor(processed_y)
                        tprint_success(f"Custom preprocessor {preprocessor.__name__} completed")
                    except Exception as e:
                        tprint_error(f"Error in custom preprocessor {preprocessor.__name__}: {e}")
            
            # Handle missing values
            if self.config.handle_missing_values:
                tprint_debug("Handling missing values")
                with tprint_timer("missing_values_handling", LogLevel.DEBUG):
                    processed_X, processed_y = self._handle_missing_values(
                        processed_X, processed_y, fit=fit
                    )
                tprint_success("Missing values handling completed")
            
            # Handle outliers
            if self.config.handle_outliers:
                tprint_debug("Handling outliers")
                with tprint_timer("outlier_handling", LogLevel.DEBUG):
                    processed_X = self._handle_outliers(processed_X, fit=fit)
                tprint_success("Outlier handling completed")
            
            # Feature engineering
            if self.config.enable_feature_engineering:
                tprint_debug("Engineering features")
                with tprint_timer("feature_engineering", LogLevel.DEBUG):
                    processed_X = self._engineer_features(processed_X, fit=fit)
                tprint_success("Feature engineering completed")
            
            # Feature scaling
            if self.config.enable_scaling:
                tprint_debug("Scaling features")
                with tprint_timer("feature_scaling", LogLevel.DEBUG):
                    processed_X, processed_y = self._scale_features(
                        processed_X, processed_y, fit=fit
                    )
                tprint_success("Feature scaling completed")
            
            # Feature selection
            if self.config.enable_feature_selection:
                tprint_debug("Selecting features")
                with tprint_timer("feature_selection", LogLevel.DEBUG):
                    processed_X = self._select_features(processed_X, processed_y, fit=fit)
                tprint_success("Feature selection completed")
            
            # Dimensionality reduction
            if self.config.enable_dimensionality_reduction:
                tprint_debug("Reducing dimensions")
                with tprint_timer("dimensionality_reduction", LogLevel.DEBUG):
                    processed_X = self._reduce_dimensions(processed_X, fit=fit)
                tprint_success("Dimensionality reduction completed")
            
            # Custom postprocessing
            if self.config.custom_postprocessors:
                tprint_debug(f"Applying {len(self.config.custom_postprocessors)} custom postprocessors")
                for i, postprocessor in enumerate(self.config.custom_postprocessors):
                    try:
                        tprint_debug(f"Applying custom postprocessor {i+1}/{len(self.config.custom_postprocessors)}: {postprocessor.__name__}")
                        processed_X = postprocessor(processed_X)
                        if processed_y is not None:
                            processed_y = postprocessor(processed_y)
                        tprint_success(f"Custom postprocessor {postprocessor.__name__} completed")
                    except Exception as e:
                        tprint_error(f"Error in custom postprocessor {postprocessor.__name__}: {e}")
            
            # Convert back to numpy arrays
            tprint_debug("Converting processed data back to numpy arrays")
            if isinstance(processed_X, pd.DataFrame):
                processed_X = processed_X.values
                tprint_success("Converted processed X to numpy array")
            
            if processed_y is not None and isinstance(processed_y, pd.Series):
                processed_y = processed_y.values
                tprint_success("Converted processed y to numpy array")
            
            # Update processing state
            if fit:
                self.is_fitted = True
                tprint_success("Data processor fitted successfully")
            
            # Record processing history
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_history.append({
                'timestamp': start_time,
                'duration': processing_time,
                'data_shape': X.shape,
                'processed_shape': processed_X.shape,
                'fit': fit
            })
            
            # Log final processing results
            tprint_structured({
                "processing_summary": {
                    "processing_time_seconds": processing_time,
                    "input_shape": X.shape,
                    "output_shape": processed_X.shape,
                    "fit_mode": fit,
                    "is_fitted": self.is_fitted,
                    "validation_passed": validation_result.validation_passed,
                    "validation_score": validation_result.validation_score
                }
            }, LogLevel.INFO)
            
            tprint_success(f"Data processing completed in {processing_time:.2f}s "
                          f"({X.shape} -> {processed_X.shape})")
            
            return processed_X, processed_y, validation_result
            
        except Exception as e:
            tprint_error(f"Error in data processing: {e}")
            tprint_structured({
                "processing_error": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            }, LogLevel.ERROR)
            self.logger.error(f"Error in data processing: {e}", exc_info=True)
            raise
    
    def validate_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DataValidationResult:
        """
        Validate data quality and provide recommendations.
        
        Args:
            X: Feature data
            y: Target data (optional)
            
        Returns:
            DataValidationResult with validation results
        """
        tprint_info("Validating data quality")
        
        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            
            if y is not None and isinstance(y, np.ndarray):
                y = pd.Series(y, name="target")
            
            result = DataValidationResult()
            result.data_shape = X.shape
            result.data_types = {col: str(dtype) for col, dtype in X.dtypes.items()}
            result.memory_usage_mb = X.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(X, y)
            result.quality_metrics = quality_metrics
            
            # Check for critical issues
            result.critical_issues = self._identify_critical_issues(X, quality_metrics)
            
            # Generate warnings
            result.warnings = self._generate_warnings(X, quality_metrics)
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(X, quality_metrics)
            
            # Generate preprocessing suggestions
            result.suggested_preprocessing = self._suggest_preprocessing(X, quality_metrics)
            
            # Calculate overall validation score
            result.validation_score = self._calculate_validation_score(quality_metrics, result.critical_issues)
            result.validation_passed = result.validation_score >= self.config.min_data_quality_score
            
            tprint_success(f"Data validation completed: {'PASSED' if result.validation_passed else 'FAILED'} "
                          f"(Score: {result.validation_score:.3f})")
            
            return result
            
        except Exception as e:
            tprint_error(f"Error in data validation: {e}")
            return DataValidationResult()
    
    def _optimize_memory(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Optimize memory usage of data."""
        if isinstance(data, pd.Series):
            if self.config.use_category_dtype and data.dtype == 'object':
                # Try to convert to category if it has few unique values
                if data.nunique() / len(data) < 0.5:
                    data = data.astype('category')
            
            if self.config.downcast_numeric and data.dtype in ['int64', 'float64']:
                if data.dtype == 'int64':
                    data = pd.to_numeric(data, downcast='integer')
                elif data.dtype == 'float64':
                    data = pd.to_numeric(data, downcast='float')
        
        elif isinstance(data, pd.DataFrame):
            for col in data.columns:
                if self.config.use_category_dtype and data[col].dtype == 'object':
                    if data[col].nunique() / len(data) < 0.5:
                        data[col] = data[col].astype('category')
                
                if self.config.downcast_numeric and data[col].dtype in ['int64', 'float64']:
                    if data[col].dtype == 'int64':
                        data[col] = pd.to_numeric(data[col], downcast='integer')
                    elif data[col].dtype == 'float64':
                        data[col] = pd.to_numeric(data[col], downcast='float')
        
        return data
    
    def _handle_missing_values(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Handle missing values in data."""
        if fit:
            if self.config.missing_value_strategy in ['mean', 'median', 'most_frequent']:
                self.imputer = SimpleImputer(strategy=self.config.missing_value_strategy)
            elif self.config.missing_value_strategy == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
            else:
                self.imputer = SimpleImputer(strategy='median')  # Default
        
        if self.imputer is not None:
            X_imputed = self.imputer.fit_transform(X) if fit else self.imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            
            if y is not None and y.isna().any():
                if fit:
                    y_imputer = SimpleImputer(strategy=self.config.missing_value_strategy)
                    y_imputed = y_imputer.fit_transform(y.values.reshape(-1, 1))
                    y = pd.Series(y_imputed.flatten(), index=y.index, name=y.name)
                else:
                    # For transform, we can't handle missing values in target
                    y = y.dropna()
        
        return X, y
    
    def _handle_outliers(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers in data."""
        if self.config.outlier_method == 'iqr':
            return self._handle_outliers_iqr(X)
        elif self.config.outlier_method == 'zscore':
            return self._handle_outliers_zscore(X)
        else:
            return X  # No outlier handling
    
    def _handle_outliers_iqr(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        X_clean = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
        
        return X_clean
    
    def _handle_outliers_zscore(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using Z-score method."""
        X_clean = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
            X_clean[col] = np.where(z_scores > self.config.outlier_threshold, 
                                   X[col].mean(), X_clean[col])
        
        return X_clean
    
    def _engineer_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Engineer new features."""
        X_engineered = X.copy()
        
        # Time features
        if self.config.create_time_features:
            X_engineered = self._create_time_features(X_engineered)
        
        # Interaction features
        if self.config.create_interaction_features:
            X_engineered = self._create_interaction_features(X_engineered)
        
        # Polynomial features
        if self.config.polynomial_features:
            X_engineered = self._create_polynomial_features(X_engineered)
        
        return X_engineered
    
    def _create_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        X_engineered = X.copy()
        
        # Look for common datetime column names
        datetime_columns = []
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['datetime', 'timestamp', 'date', 'time']):
                try:
                    # Try to convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(X[col]):
                        X_engineered[col] = pd.to_datetime(X[col], errors='coerce')
                    datetime_columns.append(col)
                except (ValueError, TypeError):
                    continue
        
        # Create time features for each datetime column
        for dt_col in datetime_columns:
            if pd.api.types.is_datetime64_any_dtype(X_engineered[dt_col]):
                # Basic time features
                X_engineered[f'{dt_col}_hour'] = X_engineered[dt_col].dt.hour
                X_engineered[f'{dt_col}_day_of_week'] = X_engineered[dt_col].dt.dayofweek
                X_engineered[f'{dt_col}_day_of_month'] = X_engineered[dt_col].dt.day
                X_engineered[f'{dt_col}_month'] = X_engineered[dt_col].dt.month
                X_engineered[f'{dt_col}_quarter'] = X_engineered[dt_col].dt.quarter
                X_engineered[f'{dt_col}_year'] = X_engineered[dt_col].dt.year
                
                # Cyclical encoding for periodic features
                X_engineered[f'{dt_col}_hour_sin'] = np.sin(2 * np.pi * X_engineered[f'{dt_col}_hour'] / 24)
                X_engineered[f'{dt_col}_hour_cos'] = np.cos(2 * np.pi * X_engineered[f'{dt_col}_hour'] / 24)
                X_engineered[f'{dt_col}_day_sin'] = np.sin(2 * np.pi * X_engineered[f'{dt_col}_day_of_week'] / 7)
                X_engineered[f'{dt_col}_day_cos'] = np.cos(2 * np.pi * X_engineered[f'{dt_col}_day_of_week'] / 7)
                X_engineered[f'{dt_col}_month_sin'] = np.sin(2 * np.pi * X_engineered[f'{dt_col}_month'] / 12)
                X_engineered[f'{dt_col}_month_cos'] = np.cos(2 * np.pi * X_engineered[f'{dt_col}_month'] / 12)
                
                # Business day features
                X_engineered[f'{dt_col}_is_weekend'] = (X_engineered[f'{dt_col}_day_of_week'] >= 5).astype(int)
                X_engineered[f'{dt_col}_is_month_start'] = X_engineered[dt_col].dt.is_month_start.astype(int)
                X_engineered[f'{dt_col}_is_month_end'] = X_engineered[dt_col].dt.is_month_end.astype(int)
                X_engineered[f'{dt_col}_is_quarter_start'] = X_engineered[dt_col].dt.is_quarter_start.astype(int)
                X_engineered[f'{dt_col}_is_quarter_end'] = X_engineered[dt_col].dt.is_quarter_end.astype(int)
                X_engineered[f'{dt_col}_is_year_start'] = X_engineered[dt_col].dt.is_year_start.astype(int)
                X_engineered[f'{dt_col}_is_year_end'] = X_engineered[dt_col].dt.is_year_end.astype(int)
        
        return X_engineered
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Create pairwise interactions for top features
        if len(numeric_cols) > 1:
            # Simple implementation - create interactions for first few columns
            for i, col1 in enumerate(numeric_cols[:3]):
                for col2 in numeric_cols[i+1:4]:
                    interaction_name = f"{col1}_x_{col2}"
                    X[interaction_name] = X[col1] * X[col2]
        
        return X
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns to avoid explosion
            for degree in range(2, self.config.polynomial_degree + 1):
                poly_name = f"{col}_poly_{degree}"
                X[poly_name] = X[col] ** degree
        
        return X
    
    def _scale_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Scale features and optionally target."""
        if fit:
            if self.config.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.config.scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                return X, y  # No scaling
        
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Scale target if requested
            if y is not None and self.config.scale_target:
                if fit:
                    y_scaler = StandardScaler()
                    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
                    y = pd.Series(y_scaled.flatten(), index=y.index, name=y.name)
        
        return X, y
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        fit: bool = True
    ) -> pd.DataFrame:
        """Select most important features."""
        if y is None:
            return X  # Can't do supervised feature selection without target
        
        if fit:
            if self.config.selection_method == 'k_best':
                self.feature_selector = SelectKBest(k=self.config.n_features)
            elif self.config.selection_method == 'percentile':
                self.feature_selector = SelectPercentile(percentile=self.config.feature_percentile)
            elif self.config.selection_method == 'mutual_info':
                self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=self.config.n_features)
        
        if self.feature_selector is not None:
            X_selected = self.feature_selector.fit_transform(X, y) if fit else self.feature_selector.transform(X)
            selected_features = X.columns[self.feature_selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X
    
    def _reduce_dimensions(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Reduce dimensionality of features."""
        if fit:
            if self.config.reduction_method == 'pca':
                self.dimension_reducer = PCA(n_components=self.config.n_components)
            elif self.config.reduction_method == 'tsne':
                self.dimension_reducer = TSNE(n_components=self.config.n_components or 2)
            else:
                return X
        
        if self.dimension_reducer is not None:
            X_reduced = self.dimension_reducer.fit_transform(X) if fit else self.dimension_reducer.transform(X)
            
            if self.config.reduction_method == 'pca':
                feature_names = [f"PC_{i+1}" for i in range(X_reduced.shape[1])]
            else:
                feature_names = [f"component_{i+1}" for i in range(X_reduced.shape[1])]
            
            X = pd.DataFrame(X_reduced, columns=feature_names, index=X.index)
        
        return X
    
    def _calculate_quality_metrics(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        metrics = DataQualityMetrics()
        
        # Completeness metrics
        total_cells = X.size
        missing_cells = X.isna().sum().sum()
        metrics.missing_value_percentage = (missing_cells / total_cells) * 100
        
        complete_rows = X.dropna().shape[0]
        metrics.complete_rows_percentage = (complete_rows / X.shape[0]) * 100
        
        complete_cols = X.dropna(axis=1).shape[1]
        metrics.complete_columns_percentage = (complete_cols / X.shape[1]) * 100
        
        # Consistency metrics
        metrics.duplicate_rows_count = X.duplicated().sum()
        metrics.duplicate_rows_percentage = (metrics.duplicate_rows_count / X.shape[0]) * 100
        
        # Uniqueness metrics
        unique_ratios = []
        high_cardinality = []
        
        for col in X.columns:
            unique_ratio = X[col].nunique() / len(X)
            unique_ratios.append(unique_ratio)
            
            if unique_ratio > 0.95:  # High cardinality threshold
                high_cardinality.append(col)
        
        metrics.unique_values_ratio = np.mean(unique_ratios)
        metrics.high_cardinality_columns = high_cardinality
        
        # Distribution metrics
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        skewed_cols = []
        constant_cols = []
        
        for col in numeric_cols:
            if X[col].std() == 0:
                constant_cols.append(col)
            elif abs(X[col].skew()) > 2:  # Highly skewed
                skewed_cols.append(col)
        
        metrics.skewed_columns = skewed_cols
        metrics.constant_columns = constant_cols
        
        # Calculate overall quality score
        metrics.overall_quality_score = self._calculate_overall_quality_score(metrics)
        
        return metrics
    
    def _calculate_overall_quality_score(self, metrics: DataQualityMetrics) -> float:
        """Calculate overall data quality score."""
        score = 1.0
        
        # Penalize missing values
        score -= (metrics.missing_value_percentage / 100) * 0.3
        
        # Penalize duplicates
        score -= (metrics.duplicate_rows_percentage / 100) * 0.2
        
        # Penalize high cardinality columns
        score -= len(metrics.high_cardinality_columns) * 0.05
        
        # Penalize constant columns
        score -= len(metrics.constant_columns) * 0.1
        
        # Penalize skewed columns
        score -= len(metrics.skewed_columns) * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _identify_critical_issues(self, X: pd.DataFrame, metrics: DataQualityMetrics) -> List[str]:
        """Identify critical data quality issues."""
        issues = []
        
        if metrics.missing_value_percentage > 50:
            issues.append(f"Too many missing values: {metrics.missing_value_percentage:.1f}%")
        
        if metrics.duplicate_rows_percentage > 20:
            issues.append(f"Too many duplicate rows: {metrics.duplicate_rows_percentage:.1f}%")
        
        if len(metrics.constant_columns) > X.shape[1] * 0.3:
            issues.append(f"Too many constant columns: {len(metrics.constant_columns)}")
        
        if X.shape[0] < 100:
            issues.append(f"Insufficient data: only {X.shape[0]} rows")
        
        return issues
    
    def _generate_warnings(self, X: pd.DataFrame, metrics: DataQualityMetrics) -> List[str]:
        """Generate data quality warnings."""
        warnings = []
        
        if metrics.missing_value_percentage > 10:
            warnings.append(f"High missing value percentage: {metrics.missing_value_percentage:.1f}%")
        
        if len(metrics.high_cardinality_columns) > 0:
            warnings.append(f"High cardinality columns detected: {metrics.high_cardinality_columns}")
        
        if len(metrics.skewed_columns) > 0:
            warnings.append(f"Highly skewed columns: {metrics.skewed_columns}")
        
        if X.shape[1] > 1000:
            warnings.append(f"High dimensionality: {X.shape[1]} features")
        
        return warnings
    
    def _generate_recommendations(self, X: pd.DataFrame, metrics: DataQualityMetrics) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        if metrics.missing_value_percentage > 5:
            recommendations.append("Consider imputation strategies for missing values")
        
        if metrics.duplicate_rows_percentage > 5:
            recommendations.append("Remove or handle duplicate rows")
        
        if len(metrics.constant_columns) > 0:
            recommendations.append("Remove constant columns")
        
        if len(metrics.high_cardinality_columns) > 0:
            recommendations.append("Consider encoding strategies for high cardinality columns")
        
        if len(metrics.skewed_columns) > 0:
            recommendations.append("Consider transformation for skewed columns")
        
        if X.shape[1] > 500:
            recommendations.append("Consider feature selection or dimensionality reduction")
        
        return recommendations
    
    def _suggest_preprocessing(self, X: pd.DataFrame, metrics: DataQualityMetrics) -> List[str]:
        """Suggest preprocessing steps."""
        suggestions = []
        
        if metrics.missing_value_percentage > 0:
            suggestions.append("handle_missing_values")
        
        if len(metrics.skewed_columns) > 0:
            suggestions.append("handle_outliers")
        
        if X.shape[1] > 100:
            suggestions.append("enable_feature_selection")
        
        if metrics.overall_quality_score < 0.8:
            suggestions.append("enable_scaling")
        
        return suggestions
    
    def _calculate_validation_score(
        self,
        metrics: DataQualityMetrics,
        critical_issues: List[str]
    ) -> float:
        """Calculate overall validation score."""
        score = metrics.overall_quality_score
        
        # Penalize critical issues heavily
        score -= len(critical_issues) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def save_preprocessing_pipeline(self, filepath: Union[str, Path]) -> bool:
        """Save fitted preprocessing pipeline."""
        try:
            if not self.is_fitted:
                tprint_warning("Pipeline not fitted - nothing to save")
                return False
            
            pipeline_data = {
                'config': self.config.to_dict(),
                'scaler': self.scaler,
                'imputer': self.imputer,
                'feature_selector': self.feature_selector,
                'dimension_reducer': self.dimension_reducer,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(pipeline_data, filepath)
            tprint_success(f"Preprocessing pipeline saved to {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to save preprocessing pipeline: {e}")
            return False
    
    def load_preprocessing_pipeline(self, filepath: Union[str, Path]) -> bool:
        """Load fitted preprocessing pipeline."""
        try:
            pipeline_data = joblib.load(filepath)
            
            self.config = DataProcessingConfig(**pipeline_data['config'])
            self.scaler = pipeline_data.get('scaler')
            self.imputer = pipeline_data.get('imputer')
            self.feature_selector = pipeline_data.get('feature_selector')
            self.dimension_reducer = pipeline_data.get('dimension_reducer')
            self.feature_names = pipeline_data.get('feature_names')
            self.target_name = pipeline_data.get('target_name')
            self.is_fitted = pipeline_data.get('is_fitted', False)
            
            tprint_success(f"Preprocessing pipeline loaded from {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to load preprocessing pipeline: {e}")
            return False
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing operations."""
        return {
            'is_fitted': self.is_fitted,
            'config': self.config.to_dict(),
            'processing_history': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'duration': entry['duration'],
                    'data_shape': entry['data_shape'],
                    'processed_shape': entry['processed_shape'],
                    'fit': entry['fit']
                }
                for entry in self.processing_history
            ],
            'components': {
                'scaler': self.scaler.__class__.__name__ if self.scaler else None,
                'imputer': self.imputer.__class__.__name__ if self.imputer else None,
                'feature_selector': self.feature_selector.__class__.__name__ if self.feature_selector else None,
                'dimension_reducer': self.dimension_reducer.__class__.__name__ if self.dimension_reducer else None
            }
        }