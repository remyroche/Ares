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
    
    def __init__(self, config: Optional[FeatureProcessorConfig] = None):
        """
        Initialize feature processor.
        
        Args:
            config: Configuration for feature processing
        """
        self.config = config or FeatureProcessorConfig()
        self.scaler = None
        self.feature_selector = None
        self.dr_model = None
        self.processing_stats = {}
        
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
            logger.info("🔧 Starting feature processing pipeline...")
            
            # Store original shape
            original_shape = features_df.shape
            self.processing_stats['original_shape'] = original_shape
            
            # Step 1: Data validation
            if self.config.validate_data:
                features_df = self._validate_data(features_df)
            
            # Step 2: Data cleaning
            features_df = self._clean_data(features_df)
            
            # Step 3: Handle outliers
            if self.config.handle_outliers != 'none':
                features_df = self._handle_outliers(features_df)
            
            # Step 4: Feature scaling
            if self.config.scaling_method != 'none':
                features_df = self._scale_features(features_df)
            
            # Step 5: Feature engineering
            if self.config.enable_polynomial or self.config.enable_interactions:
                features_df = self._engineer_features(features_df)
            
            # Step 6: Feature selection
            if self.config.enable_feature_selection:
                features_df, feature_importance = self._select_features(features_df, target)
            else:
                feature_importance = None
            
            # Step 7: Dimensionality reduction
            if self.config.enable_dr:
                features_df, dr_model = self._reduce_dimensions(features_df)
            else:
                dr_model = None
            
            # Step 8: Final validation
            features_df = self._final_validation(features_df)
            
            # Calculate processing statistics
            self.processing_stats.update({
                'final_shape': features_df.shape,
                'features_removed': original_shape[1] - features_df.shape[1],
                'samples_removed': original_shape[0] - features_df.shape[0],
                'missing_values': features_df.isnull().sum().sum(),
                'infinite_values': np.isinf(features_df.values).sum(),
                'zero_variance_features': self._count_zero_variance_features(features_df)
            })
            
            logger.info(f"✅ Feature processing completed. Final shape: {features_df.shape}")
            
            return ProcessedFeatures(
                features_df=features_df,
                feature_names=list(features_df.columns),
                processing_stats=self.processing_stats,
                feature_importance=feature_importance,
                dr_model=dr_model
            )
            
        except Exception as e:
            logger.error(f"❌ Feature processing failed: {e}")
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
    
    def _clean_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing and infinite values."""
        try:
            logger.info("🧹 Cleaning data...")
            
            # Handle missing values
            if self.config.handle_missing == 'drop':
                features_df = features_df.dropna()
            elif self.config.handle_missing == 'fill':
                # Fill with median for numeric columns
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
            elif self.config.handle_missing == 'interpolate':
                # Interpolate missing values
                features_df = features_df.interpolate(method='linear')
            
            # Handle infinite values
            if self.config.handle_infinite == 'drop':
                features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
            elif self.config.handle_infinite == 'clip':
                # Clip infinite values to large finite values
                features_df = features_df.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])
            elif self.config.handle_infinite == 'replace':
                # Replace infinite values with NaN and then fill
                features_df = features_df.replace([np.inf, -np.inf], np.nan)
                features_df = features_df.fillna(features_df.median())
            
            logger.info(f"✅ Data cleaning completed. Shape: {features_df.shape}")
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
            
            # Apply scaling if available
            if self.scaler is not None:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                features_df[numeric_cols] = self.scaler.transform(features_df[numeric_cols])
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                features_df = features_df[self.feature_selector.get_support()]
            
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