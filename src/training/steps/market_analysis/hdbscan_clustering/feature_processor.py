"""
Feature Processor

This module provides comprehensive feature preprocessing capabilities for
HDBSCAN-based regime discovery, including data cleaning, transformation,
and feature selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

# Import enhanced hardware optimization tools
from src.utils.hardware import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    optimize_dataframe_default, optimize_numpy_array_default
)

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, tprint_data_format, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class FeatureProcessorConfig:
    """Configuration for feature processing."""
    # Data cleaning
    handle_missing: str = 'drop'  # 'drop', 'fill', 'interpolate'
    handle_infinite: str = 'clip'  # 'drop', 'clip', 'replace'
    handle_outliers: str = 'winsorize'  # 'drop', 'winsorize', 'clip', 'none'
    
    # Outlier handling
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation', 'none'
    outlier_threshold: float = 3.0
    winsorize_limits: Tuple[float, float] = (0.01, 0.01)
    
    # Feature scaling
    scaling_method: str = 'standard'  # 'standard', 'robust', 'minmax', 'quantile', 'none'
    quantile_range: Tuple[float, float] = (0.25, 0.75)
    
    # Feature selection
    enable_feature_selection: bool = True
    selection_method: str = 'mutual_info'  # 'mutual_info', 'f_score', 'variance', 'correlation'
    n_features: int = 50
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    
    # Feature engineering
    enable_polynomial: bool = False
    polynomial_degree: int = 2
    enable_interactions: bool = True
    max_interactions: int = 10
    
    # Regime-specific processing
    enable_regime_aware_processing: bool = True
    regime_detection_method: str = 'variance'  # 'variance', 'entropy', 'volatility'
    regime_window: int = 20
    regime_threshold: float = 0.1
    enable_regime_normalization: bool = True
    enable_regime_scaling: bool = True
    
    # Dimensionality reduction
    enable_dr: bool = False
    dr_method: str = 'pca'  # 'pca', 'tsne', 'none'
    dr_components: int = 20
    dr_perplexity: float = 30.0
    
    # Data validation
    validate_data: bool = True
    min_variance: float = 1e-10
    max_correlation: float = 0.99
    max_missing_ratio: float = 0.5

@dataclass
class ProcessedFeatures:
    """Result of feature processing."""
    features_df: pd.DataFrame
    feature_names: List[str]
    processing_stats: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    dr_model: Optional[Any] = None

class FeatureProcessor:
    """
    Comprehensive feature processor for HDBSCAN regime discovery.
    
    Provides data cleaning, transformation, feature selection, and
    dimensionality reduction capabilities.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[FeatureProcessorConfig] = None):
        """
        Initialize feature processor.
        
        Args:
            config: Configuration for feature processing
        """
        tprint_info("Initializing FeatureProcessor")
        
        self.config = config or FeatureProcessorConfig()
        self.scaler = None
        self.feature_selector = None
        self.dr_model = None
        self.processing_stats = {}
        
        tprint_debug(f"Config: scaling_method={self.config.scaling_method}, enable_feature_selection={self.config.enable_feature_selection}")
        tprint_success("✅ FeatureProcessor initialized")
        
    @tprint_logged(LogLevel.INFO, include_args=True)
    @smart_cache(ttl=1800)  # Cache processed features for 30 minutes
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    @memory_efficient(memory_threshold_mb=150.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def process_features(self, 
                        features_df: pd.DataFrame,
                        target: Optional[np.ndarray] = None) -> ProcessedFeatures:
        """
        Process features through the complete pipeline.
        
        Args:
            features_df: Input features DataFrame
            target: Target variable for feature selection (optional)
            
        Returns:
            ProcessedFeatures with processed data and metadata
        """
        try:
            tprint_info("🔧 Starting feature processing pipeline...")
            tprint_debug(f"Input shape: {features_df.shape}, target shape: {target.shape if target is not None else 'None'}")
            
            # Data format validation for input features
            tprint_data_format(features_df, "input_features", LogLevel.DEBUG)
            if target is not None:
                tprint_data_format(target, "input_target", LogLevel.DEBUG)
            
            # Store original shape
            original_shape = features_df.shape
            self.processing_stats['original_shape'] = original_shape
            
            # Step 1: Data validation
            if self.config.validate_data:
                with tprint_timer("Data validation"):
                    features_df = self._validate_data(features_df)
                    tprint_debug(f"After validation: {features_df.shape}")
                    # Data format validation after validation step
                    tprint_data_format(features_df, "validated_features", LogLevel.DEBUG)
            else:
                tprint_debug("Data validation skipped")
            
            # Step 2: Data cleaning
            with tprint_timer("Data cleaning"):
                features_df = self._clean_data(features_df)
                tprint_debug(f"After cleaning: {features_df.shape}")
                # Data format validation after cleaning step
                tprint_data_format(features_df, "cleaned_features", LogLevel.DEBUG)
            
            # Step 3: Handle outliers
            if self.config.handle_outliers != 'none':
                with tprint_timer("Outlier handling"):
                    features_df = self._handle_outliers(features_df)
                    tprint_debug(f"After outlier handling: {features_df.shape}")
                    # Data format validation after outlier handling
                    tprint_data_format(features_df, "outlier_handled_features", LogLevel.DEBUG)
            else:
                tprint_debug("Outlier handling skipped")
            
            # Step 4: Feature scaling
            if self.config.scaling_method != 'none':
                with tprint_timer("Feature scaling"):
                    features_df = self._scale_features(features_df)
                    tprint_debug(f"After scaling: {features_df.shape}")
                    # Data format validation after scaling
                    tprint_data_format(features_df, "scaled_features", LogLevel.DEBUG)
            else:
                tprint_debug("Feature scaling skipped")
            
            # Step 5: Feature engineering
            if self.config.enable_polynomial or self.config.enable_interactions:
                with tprint_timer("Feature engineering"):
                    features_df = self._engineer_features(features_df)
                    tprint_debug(f"After feature engineering: {features_df.shape}")
                    # Data format validation after feature engineering
                    tprint_data_format(features_df, "engineered_features", LogLevel.DEBUG)
            else:
                tprint_debug("Feature engineering skipped")
            
            # Step 5.5: Regime-aware processing
            if self.config.enable_regime_aware_processing:
                with tprint_timer("Regime-aware processing"):
                    features_df = self._apply_regime_aware_processing(features_df)
                    tprint_debug(f"After regime-aware processing: {features_df.shape}")
            else:
                tprint_debug("Regime-aware processing skipped")
            
            # Step 6: Feature selection
            if self.config.enable_feature_selection:
                with tprint_timer("Feature selection"):
                    features_df, feature_importance = self._select_features(features_df, target)
                    tprint_debug(f"After feature selection: {features_df.shape}")
            else:
                feature_importance = None
                tprint_debug("Feature selection skipped")
            
            # Step 7: Dimensionality reduction
            if self.config.enable_dr:
                with tprint_timer("Dimensionality reduction"):
                    features_df, dr_model = self._reduce_dimensions(features_df)
                    tprint_debug(f"After dimensionality reduction: {features_df.shape}")
            else:
                dr_model = None
                tprint_debug("Dimensionality reduction skipped")
            
            # Step 8: Final validation
            with tprint_timer("Final validation"):
                features_df = self._final_validation(features_df)
                tprint_debug(f"After final validation: {features_df.shape}")
                # Data format validation for final output
                tprint_data_format(features_df, "final_processed_features", LogLevel.DEBUG)
            
            # Calculate processing statistics
            self.processing_stats.update({
                'final_shape': features_df.shape,
                'features_removed': original_shape[1] - features_df.shape[1],
                'samples_removed': original_shape[0] - features_df.shape[0],
                'missing_values': features_df.isnull().sum().sum(),
                'infinite_values': np.isinf(features_df.values).sum(),
                'zero_variance_features': self._count_zero_variance_features(features_df)
            })
            
            tprint_success(f"✅ Feature processing completed. Final shape: {features_df.shape}")
            tprint_debug(f"Processing stats: features_removed={self.processing_stats['features_removed']}, samples_removed={self.processing_stats['samples_removed']}")
            
            return ProcessedFeatures(
                features_df=features_df,
                feature_names=list(features_df.columns),
                processing_stats=self.processing_stats,
                feature_importance=feature_importance,
                dr_model=dr_model
            )
            
        except Exception as e:
            tprint_error(f"❌ Feature processing failed: {e}")
            # Return original data as fallback
            return ProcessedFeatures(
                features_df=features_df,
                feature_names=list(features_df.columns),
                processing_stats={'error': str(e)},
                feature_importance=None,
                dr_model=None
            )
    
    def _validate_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate input data."""
        try:
            logger.info("🔍 Validating input data...")
            
            # Check for empty DataFrame
            if features_df.empty:
                raise ValueError("Input DataFrame is empty")
            
            # Check for all NaN columns
            all_nan_cols = features_df.columns[features_df.isnull().all()].tolist()
            if all_nan_cols:
                logger.warning(f"⚠️ Removing columns with all NaN values: {all_nan_cols}")
                features_df = features_df.drop(columns=all_nan_cols)
            
            # Check for constant columns
            constant_cols = features_df.columns[features_df.nunique() <= 1].tolist()
            if constant_cols:
                logger.warning(f"⚠️ Removing constant columns: {constant_cols}")
                features_df = features_df.drop(columns=constant_cols)
            
            # Check for excessive missing values
            missing_ratio = features_df.isnull().sum() / len(features_df)
            high_missing_cols = missing_ratio[missing_ratio > self.config.max_missing_ratio].index.tolist()
            if high_missing_cols:
                logger.warning(f"⚠️ Removing columns with high missing ratio: {high_missing_cols}")
                features_df = features_df.drop(columns=high_missing_cols)
            
            # Check for excessive correlation
            if len(features_df.columns) > 1:
                corr_matrix = features_df.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
                high_corr_pairs = []
                for col in upper_tri.columns:
                    high_corr = upper_tri[col][upper_tri[col] > self.config.max_correlation].index.tolist()
                    if high_corr:
                        high_corr_pairs.extend([(col, hc) for hc in high_corr])
                
                if high_corr_pairs:
                    logger.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs")
                    # Remove one feature from each highly correlated pair
                    cols_to_remove = set()
                    for col1, col2 in high_corr_pairs:
                        if col1 not in cols_to_remove:
                            cols_to_remove.add(col2)
                    
                    if cols_to_remove:
                        logger.warning(f"⚠️ Removing highly correlated columns: {list(cols_to_remove)}")
                        features_df = features_df.drop(columns=list(cols_to_remove))
            
            logger.info(f"✅ Data validation completed. Shape: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return features_df
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _clean_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing and infinite values."""
        try:
            tprint_info("🧹 Cleaning data...")
            tprint_debug(f"Input shape: {features_df.shape}")
            
            # Handle missing values
            if self.config.handle_missing == 'drop':
                features_df = features_df.dropna()
                tprint_debug(f"After dropping missing values: {features_df.shape}")
            elif self.config.handle_missing == 'fill':
                # Fill with median for numeric columns
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
                tprint_debug(f"Filled missing values with median for {len(numeric_cols)} numeric columns")
            elif self.config.handle_missing == 'interpolate':
                # Interpolate missing values
                features_df = features_df.interpolate(method='linear')
                tprint_debug("Interpolated missing values using linear method")
            
            # Handle infinite values
            if self.config.handle_infinite == 'drop':
                features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
                tprint_debug(f"After dropping infinite values: {features_df.shape}")
            elif self.config.handle_infinite == 'clip':
                # Clip infinite values to large finite values
                features_df = features_df.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])
                tprint_debug("Clipped infinite values to max/min float values")
            elif self.config.handle_infinite == 'replace':
                # Replace infinite values with NaN and then fill
                features_df = features_df.replace([np.inf, -np.inf], np.nan)
                features_df = features_df.fillna(features_df.median())
                tprint_debug("Replaced infinite values with median")
            
            tprint_success(f"✅ Data cleaning completed. Shape: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Data cleaning failed: {e}")
            return features_df
    
    def _handle_outliers(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        try:
            logger.info("🎯 Handling outliers...")
            
            if self.config.outlier_method == 'none':
                return features_df
            
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            if self.config.outlier_method == 'winsorize':
                # Winsorize outliers
                for col in numeric_cols:
                    features_df[col] = self._winsorize_column(features_df[col])
            
            elif self.config.outlier_method == 'iqr':
                # Remove outliers using IQR method
                for col in numeric_cols:
                    Q1 = features_df[col].quantile(0.25)
                    Q3 = features_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    features_df[col] = features_df[col].clip(lower_bound, upper_bound)
            
            elif self.config.outlier_method == 'zscore':
                # Remove outliers using Z-score method
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(features_df[col]))
                    features_df[col] = features_df[col].where(z_scores < self.config.outlier_threshold)
            
            logger.info(f"✅ Outlier handling completed. Shape: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Outlier handling failed: {e}")
            return features_df
    
    def _winsorize_column(self, series: pd.Series) -> pd.Series:
        """Winsorize a single column."""
        try:
            lower_limit = series.quantile(self.config.winsorize_limits[0])
            upper_limit = series.quantile(1 - self.config.winsorize_limits[1])
            return series.clip(lower_limit, upper_limit)
        except Exception as e:
            logger.debug(f"Winsorization failed for column: {e}")
            return series
    
    def _scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the specified method."""
        try:
            logger.info(f"📏 Scaling features using {self.config.scaling_method} method...")
            
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            if self.config.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaling_method == 'robust':
                self.scaler = RobustScaler(quantile_range=self.config.quantile_range)
            elif self.config.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.config.scaling_method == 'quantile':
                self.scaler = QuantileTransformer(output_distribution='normal')
            else:
                logger.warning(f"⚠️ Unknown scaling method: {self.config.scaling_method}")
                return features_df
            
            # Fit and transform
            features_df[numeric_cols] = self.scaler.fit_transform(features_df[numeric_cols])
            
            logger.info("✅ Feature scaling completed")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature scaling failed: {e}")
            return features_df
    
    def _engineer_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features."""
        try:
            logger.info("🔧 Engineering features...")
            
            new_features = []
            new_names = []
            
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            # Polynomial features
            if self.config.enable_polynomial and len(numeric_cols) > 0:
                # Select top features by variance for polynomial expansion
                variances = features_df[numeric_cols].var()
                top_features = variances.nlargest(min(5, len(numeric_cols))).index
                
                for col in top_features:
                    for degree in range(2, self.config.polynomial_degree + 1):
                        poly_feature = features_df[col] ** degree
                        new_features.append(poly_feature)
                        new_names.append(f"{col}_deg_{degree}")
            
            # Interaction features
            if self.config.enable_interactions and len(numeric_cols) > 1:
                # Select top features for interactions
                variances = features_df[numeric_cols].var()
                top_features = variances.nlargest(min(5, len(numeric_cols))).index
                
                interaction_count = 0
                for i, col1 in enumerate(top_features):
                    for col2 in top_features[i+1:]:
                        if interaction_count >= self.config.max_interactions:
                            break
                        
                        interaction = features_df[col1] * features_df[col2]
                        new_features.append(interaction)
                        new_names.append(f"{col1}_x_{col2}")
                        interaction_count += 1
            
            # Add new features to DataFrame
            if new_features:
                new_features_df = pd.DataFrame(
                    np.column_stack(new_features),
                    columns=new_names,
                    index=features_df.index
                )
                features_df = pd.concat([features_df, new_features_df], axis=1)
                logger.info(f"✅ Added {len(new_features)} engineered features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return features_df
    
    def _select_features(self, features_df: pd.DataFrame, target: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
        """Select the most relevant features."""
        try:
            logger.info("🎯 Selecting features...")
            
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) <= self.config.n_features:
                logger.info("✅ No feature selection needed")
                return features_df, None
            
            # Remove low variance features
            variances = features_df[numeric_cols].var()
            high_var_cols = variances[variances > self.config.variance_threshold].index
            features_df = features_df[high_var_cols]
            
            if len(features_df.columns) <= self.config.n_features:
                logger.info("✅ Feature selection completed (variance filtering)")
                return features_df, None
            
            # Feature selection based on method
            if self.config.selection_method == 'mutual_info' and target is not None:
                # Mutual information
                mi_scores = mutual_info_regression(features_df, target)
                feature_scores = dict(zip(features_df.columns, mi_scores))
                
                # Select top features
                top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.n_features]
                selected_cols = [col for col, _ in top_features]
                
            elif self.config.selection_method == 'f_score' and target is not None:
                # F-score
                f_scores, _ = f_regression(features_df, target)
                feature_scores = dict(zip(features_df.columns, f_scores))
                
                # Select top features
                top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.n_features]
                selected_cols = [col for col, _ in top_features]
                
            elif self.config.selection_method == 'variance':
                # Variance-based selection
                variances = features_df.var()
                top_features = variances.nlargest(self.config.n_features)
                selected_cols = top_features.index.tolist()
                feature_scores = variances.to_dict()
                
            else:
                # Default: select first n_features
                selected_cols = features_df.columns[:self.config.n_features]
                feature_scores = None
            
            # Select features
            features_df = features_df[selected_cols]
            
            logger.info(f"✅ Feature selection completed. Selected {len(selected_cols)} features")
            
            return features_df, feature_scores
            
        except Exception as e:
            logger.error(f"❌ Feature selection failed: {e}")
            return features_df, None
    
    def _reduce_dimensions(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Reduce dimensionality using specified method."""
        try:
            logger.info(f"📉 Reducing dimensions using {self.config.dr_method}...")
            
            if self.config.dr_method == 'pca':
                self.dr_model = PCA(n_components=self.config.dr_components)
                reduced_features = self.dr_model.fit_transform(features_df)
                
                # Create new column names
                new_columns = [f"PC_{i+1}" for i in range(reduced_features.shape[1])]
                
            elif self.config.dr_method == 'tsne':
                self.dr_model = TSNE(
                    n_components=min(self.config.dr_components, 3),
                    perplexity=self.config.dr_perplexity,
                    random_state=42
                )
                reduced_features = self.dr_model.fit_transform(features_df)
                
                # Create new column names
                new_columns = [f"tSNE_{i+1}" for i in range(reduced_features.shape[1])]
                
            else:
                logger.warning(f"⚠️ Unknown DR method: {self.config.dr_method}")
                return features_df, None
            
            # Create new DataFrame
            features_df = pd.DataFrame(
                reduced_features,
                columns=new_columns,
                index=features_df.index
            )
            
            logger.info(f"✅ Dimensionality reduction completed. New shape: {features_df.shape}")
            
            return features_df, self.dr_model
            
        except Exception as e:
            logger.error(f"❌ Dimensionality reduction failed: {e}")
            return features_df, None
    
    def _final_validation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Final validation of processed features."""
        try:
            logger.info("🔍 Final validation...")
            
            # Check for NaN values
            if features_df.isnull().any().any():
                logger.warning("⚠️ Found NaN values in final features, filling with 0")
                features_df = features_df.fillna(0)
            
            # Check for infinite values
            if np.isinf(features_df.values).any():
                logger.warning("⚠️ Found infinite values in final features, clipping")
                features_df = features_df.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])
            
            # Check for zero variance features
            zero_var_cols = features_df.columns[features_df.var() < self.config.min_variance]
            if len(zero_var_cols) > 0:
                logger.warning(f"⚠️ Removing zero variance features: {zero_var_cols.tolist()}")
                features_df = features_df.drop(columns=zero_var_cols)
            
            logger.info("✅ Final validation completed")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Final validation failed: {e}")
            return features_df
    
    def _count_zero_variance_features(self, features_df: pd.DataFrame) -> int:
        """Count zero variance features."""
        try:
            variances = features_df.var()
            return (variances < self.config.min_variance).sum()
        except Exception as e:
            logger.debug(f"Zero variance count failed: {e}")
            return 0
    
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Transform new features using fitted processors."""
        try:
            logger.info("🔄 Transforming new features...")
            
            # Data format validation for input features
            tprint_data_format(features_df, "transform_input_features", LogLevel.DEBUG)
            
            # Apply scaling if available
            if self.scaler is not None:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                features_df[numeric_cols] = self.scaler.transform(features_df[numeric_cols])
                # Data format validation after scaling
                tprint_data_format(features_df, "scaled_transform_features", LogLevel.DEBUG)
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                features_df = features_df[self.feature_selector.get_support()]
                # Data format validation after feature selection
                tprint_data_format(features_df, "selected_transform_features", LogLevel.DEBUG)
            
            # Apply dimensionality reduction if available
            if self.dr_model is not None:
                if hasattr(self.dr_model, 'transform'):
                    reduced_features = self.dr_model.transform(features_df)
                    if self.config.dr_method == 'pca':
                        new_columns = [f"PC_{i+1}" for i in range(reduced_features.shape[1])]
                    else:
                        new_columns = [f"tSNE_{i+1}" for i in range(reduced_features.shape[1])]
                    
                    features_df = pd.DataFrame(
                        reduced_features,
                        columns=new_columns,
                        index=features_df.index
                    )
                    # Data format validation after dimensionality reduction
                    tprint_data_format(features_df, "reduced_transform_features", LogLevel.DEBUG)
            
            # Data format validation for final transformed output
            tprint_data_format(features_df, "final_transformed_features", LogLevel.DEBUG)
            
            logger.info("✅ Feature transformation completed")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature transformation failed: {e}")
            return features_df
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def get_scaler(self) -> Optional[Any]:
        """Get fitted scaler."""
        return self.scaler
    
    def get_dr_model(self) -> Optional[Any]:
        """Get fitted dimensionality reduction model."""
        return self.dr_model
    
    def _apply_regime_aware_processing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-aware processing to features."""
        try:
            logger.info("🎯 Applying regime-aware processing...")
            
            # Detect regimes in the data
            regimes = self._detect_regimes(features_df)
            
            # Apply regime-specific processing
            if regimes is not None:
                # Regime-specific normalization
                if self.config.enable_regime_normalization:
                    features_df = self._apply_regime_normalization(features_df, regimes)
                
                # Regime-specific scaling
                if self.config.enable_regime_scaling:
                    features_df = self._apply_regime_scaling(features_df, regimes)
                
                # Regime-aware feature selection
                features_df = self._apply_regime_aware_feature_selection(features_df, regimes)
                
                logger.info("✅ Regime-aware processing completed")
            else:
                logger.warning("⚠️ No regimes detected, skipping regime-aware processing")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Regime-aware processing failed: {e}")
            return features_df
    
    def _detect_regimes(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Detect regimes in the feature data."""
        try:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return None
            
            # Select primary feature for regime detection
            if 'returns' in numeric_cols:
                primary_feature = features_df['returns']
            elif 'volatility' in numeric_cols:
                primary_feature = features_df['volatility']
            else:
                # Use the first numeric column
                primary_feature = features_df[numeric_cols[0]]
            
            # Detect regimes based on method
            if self.config.regime_detection_method == 'variance':
                regimes = self._detect_regimes_by_variance(primary_feature)
            elif self.config.regime_detection_method == 'entropy':
                regimes = self._detect_regimes_by_entropy(primary_feature)
            elif self.config.regime_detection_method == 'volatility':
                regimes = self._detect_regimes_by_volatility(primary_feature)
            else:
                regimes = self._detect_regimes_by_variance(primary_feature)
            
            return regimes
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            return None
    
    def _detect_regimes_by_variance(self, feature: pd.Series) -> np.ndarray:
        """Detect regimes based on variance changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            rolling_var = feature.rolling(window=window).var()
            
            # Find variance change points
            var_changes = np.abs(rolling_var.diff()) > (threshold * rolling_var.rolling(window=window).mean())
            change_points = np.where(var_changes)[0]
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3  # 3 regimes max
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Variance-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _detect_regimes_by_entropy(self, feature: pd.Series) -> np.ndarray:
        """Detect regimes based on entropy changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            
            # Calculate rolling entropy
            rolling_entropy = []
            for i in range(window, len(feature)):
                window_data = feature[i-window:i]
                # Discretize data
                hist, _ = np.histogram(window_data, bins=10)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                rolling_entropy.append(entropy)
            
            rolling_entropy = np.array(rolling_entropy)
            
            # Find entropy change points
            entropy_changes = np.abs(np.diff(rolling_entropy)) > (threshold * np.std(rolling_entropy))
            change_points = np.where(entropy_changes)[0] + window
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Entropy-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _detect_regimes_by_volatility(self, feature: pd.Series) -> np.ndarray:
        """Detect regimes based on volatility changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            rolling_vol = feature.rolling(window=window).std()
            
            # Find volatility change points
            vol_changes = np.abs(rolling_vol.diff()) > (threshold * rolling_vol.rolling(window=window).mean())
            change_points = np.where(vol_changes)[0]
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Volatility-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _apply_regime_normalization(self, features_df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """Apply regime-specific normalization."""
        try:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            normalized_df = features_df.copy()
            
            unique_regimes = np.unique(regimes)
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 1:  # Need at least 2 samples
                    regime_data = features_df.loc[regime_mask, numeric_cols]
                    
                    # Normalize within regime
                    regime_mean = regime_data.mean()
                    regime_std = regime_data.std()
                    
                    # Avoid division by zero
                    regime_std = regime_std.replace(0, 1)
                    
                    normalized_df.loc[regime_mask, numeric_cols] = (
                        (regime_data - regime_mean) / regime_std
                    )
            
            logger.info(f"✅ Applied regime-specific normalization for {len(unique_regimes)} regimes")
            return normalized_df
            
        except Exception as e:
            logger.error(f"❌ Regime normalization failed: {e}")
            return features_df
    
    def _apply_regime_scaling(self, features_df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """Apply regime-specific scaling."""
        try:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            scaled_df = features_df.copy()
            
            unique_regimes = np.unique(regimes)
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 1:
                    regime_data = features_df.loc[regime_mask, numeric_cols]
                    
                    # Min-max scaling within regime
                    regime_min = regime_data.min()
                    regime_max = regime_data.max()
                    
                    # Avoid division by zero
                    regime_range = regime_max - regime_min
                    regime_range = regime_range.replace(0, 1)
                    
                    scaled_df.loc[regime_mask, numeric_cols] = (
                        (regime_data - regime_min) / regime_range
                    )
            
            logger.info(f"✅ Applied regime-specific scaling for {len(unique_regimes)} regimes")
            return scaled_df
            
        except Exception as e:
            logger.error(f"❌ Regime scaling failed: {e}")
            return features_df
    
    def _apply_regime_aware_feature_selection(self, features_df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """Apply regime-aware feature selection."""
        try:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            selected_cols = []
            
            unique_regimes = np.unique(regimes)
            
            # Select features that are informative across regimes
            for col in numeric_cols:
                feature_values = features_df[col].values
                
                # Calculate regime-specific variance
                regime_variances = []
                for regime in unique_regimes:
                    regime_mask = regimes == regime
                    if np.sum(regime_mask) > 1:
                        regime_data = feature_values[regime_mask]
                        regime_var = np.var(regime_data)
                        regime_variances.append(regime_var)
                
                if len(regime_variances) > 1:
                    # Feature is informative if it has different variance across regimes
                    variance_ratio = np.max(regime_variances) / (np.min(regime_variances) + 1e-10)
                    if variance_ratio > 2.0:  # Threshold for regime discrimination
                        selected_cols.append(col)
                else:
                    # If only one regime, keep feature if it has sufficient variance
                    if len(regime_variances) == 1 and regime_variances[0] > 1e-6:
                        selected_cols.append(col)
            
            # Keep non-numeric columns
            non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns
            selected_cols.extend(non_numeric_cols)
            
            if len(selected_cols) > 0:
                features_df = features_df[selected_cols]
                logger.info(f"✅ Regime-aware feature selection: {len(selected_cols)} features selected")
            else:
                logger.warning("⚠️ No features selected by regime-aware selection")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Regime-aware feature selection failed: {e}")
            return features_df