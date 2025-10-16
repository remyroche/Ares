"""
Advanced Preprocessing

Implementation for advanced data preprocessing in NAS.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif


class PreprocessingType(Enum):
    """Types of preprocessing operations."""
    NORMALIZATION = "normalization"
    STANDARDIZATION = "standardization"
    SCALING = "scaling"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FEATURE_SELECTION = "feature_selection"
    NOISE_REDUCTION = "noise_reduction"
    OUTLIER_DETECTION = "outlier_detection"


@dataclass
class PreprocessingConfig:
    """Configuration for advanced preprocessing."""
    preprocessing_types: List[PreprocessingType]
    normalization_method: str = "standard"
    dimensionality_reduction_components: int = 10
    feature_selection_k: int = 20
    outlier_threshold: float = 3.0
    noise_reduction_factor: float = 0.1


class AdvancedPreprocessor:
    """Advanced data preprocessing for NAS."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize advanced preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.preprocessing_pipeline = []
        self.fitted_transformers = {}
        self.preprocessing_history = []
        
    def preprocess_data(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Preprocess data using configured pipeline.
        
        Args:
            data: Input data
            target: Optional target data
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_info)
        """
        original_shape = data.shape
        preprocessed_data = data.copy()
        preprocessing_info = {
            'original_shape': original_shape,
            'preprocessing_steps': [],
            'transformation_params': {}
        }
        
        try:
            for preprocessing_type in self.config.preprocessing_types:
                if preprocessing_type == PreprocessingType.NORMALIZATION:
                    preprocessed_data, params = self._normalize_data(preprocessed_data)
                    preprocessing_info['preprocessing_steps'].append('normalization')
                    preprocessing_info['transformation_params']['normalization'] = params
                
                elif preprocessing_type == PreprocessingType.STANDARDIZATION:
                    preprocessed_data, params = self._standardize_data(preprocessed_data)
                    preprocessing_info['preprocessing_steps'].append('standardization')
                    preprocessing_info['transformation_params']['standardization'] = params
                
                elif preprocessing_type == PreprocessingType.SCALING:
                    preprocessed_data, params = self._scale_data(preprocessed_data)
                    preprocessing_info['preprocessing_steps'].append('scaling')
                    preprocessing_info['transformation_params']['scaling'] = params
                
                elif preprocessing_type == PreprocessingType.DIMENSIONALITY_REDUCTION:
                    preprocessed_data, params = self._reduce_dimensionality(preprocessed_data)
                    preprocessing_info['preprocessing_steps'].append('dimensionality_reduction')
                    preprocessing_info['transformation_params']['dimensionality_reduction'] = params
                
                elif preprocessing_type == PreprocessingType.FEATURE_SELECTION:
                    if target is not None:
                        preprocessed_data, params = self._select_features(preprocessed_data, target)
                        preprocessing_info['preprocessing_steps'].append('feature_selection')
                        preprocessing_info['transformation_params']['feature_selection'] = params
                
                elif preprocessing_type == PreprocessingType.NOISE_REDUCTION:
                    preprocessed_data, params = self._reduce_noise(preprocessed_data)
                    preprocessing_info['preprocessing_steps'].append('noise_reduction')
                    preprocessing_info['transformation_params']['noise_reduction'] = params
                
                elif preprocessing_type == PreprocessingType.OUTLIER_DETECTION:
                    preprocessed_data, params = self._detect_outliers(preprocessed_data)
                    preprocessing_info['preprocessing_steps'].append('outlier_detection')
                    preprocessing_info['transformation_params']['outlier_detection'] = params
            
            preprocessing_info['final_shape'] = preprocessed_data.shape
            preprocessing_info['data_quality_score'] = self._calculate_data_quality_score(preprocessed_data)
            
            # Record preprocessing
            preprocessing_record = {
                'original_data': data,
                'preprocessed_data': preprocessed_data,
                'preprocessing_info': preprocessing_info,
                'timestamp': np.datetime64('now')
            }
            self.preprocessing_history.append(preprocessing_record)
            
            return preprocessed_data, preprocessing_info
            
        except Exception as e:
            preprocessing_info['error'] = str(e)
            return data, preprocessing_info
    
    def _normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize data using specified method."""
        if self.config.normalization_method == "minmax":
            scaler = MinMaxScaler()
        elif self.config.normalization_method == "robust":
            scaler = RobustScaler()
        else:  # standard
            scaler = StandardScaler()
        
        normalized_data = scaler.fit_transform(data)
        
        params = {
            'method': self.config.normalization_method,
            'scaler_params': scaler.get_params()
        }
        
        return normalized_data, params
    
    def _standardize_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Standardize data to zero mean and unit variance."""
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)
        
        params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
        
        return standardized_data, params
    
    def _scale_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Scale data to specified range."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        params = {
            'data_min': scaler.data_min_.tolist(),
            'data_max': scaler.data_max_.tolist()
        }
        
        return scaled_data, params
    
    def _reduce_dimensionality(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Reduce dimensionality using PCA."""
        n_components = min(self.config.dimensionality_reduction_components, data.shape[1])
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        
        params = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist()
        }
        
        return reduced_data, params
    
    def _select_features(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Select best features using statistical tests."""
        k = min(self.config.feature_selection_k, data.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selected_data = selector.fit_transform(data, target)
        
        params = {
            'k': k,
            'selected_features': selector.get_support(indices=True).tolist(),
            'feature_scores': selector.scores_.tolist()
        }
        
        return selected_data, params
    
    def _reduce_noise(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Reduce noise in data."""
        # Simple noise reduction using moving average
        window_size = max(1, int(len(data) * self.config.noise_reduction_factor))
        
        if window_size > 1:
            # Apply moving average
            noise_reduced_data = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(data), i + window_size // 2 + 1)
                noise_reduced_data[i] = np.mean(data[start_idx:end_idx], axis=0)
        else:
            noise_reduced_data = data.copy()
        
        params = {
            'window_size': window_size,
            'noise_reduction_factor': self.config.noise_reduction_factor
        }
        
        return noise_reduced_data, params
    
    def _detect_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect and handle outliers."""
        # Use Z-score for outlier detection
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        outlier_mask = np.any(z_scores > self.config.outlier_threshold, axis=1)
        
        # Remove outliers
        cleaned_data = data[~outlier_mask]
        
        params = {
            'outlier_threshold': self.config.outlier_threshold,
            'outliers_removed': np.sum(outlier_mask),
            'outlier_indices': np.where(outlier_mask)[0].tolist()
        }
        
        return cleaned_data, params
    
    def _calculate_data_quality_score(self, data: np.ndarray) -> float:
        """Calculate data quality score."""
        # Calculate various quality metrics
        completeness = 1.0 - np.isnan(data).sum() / data.size
        consistency = 1.0 - np.std(data) / (np.mean(data) + 1e-8)
        diversity = len(np.unique(data, axis=0)) / len(data)
        
        # Combine metrics
        quality_score = (completeness + consistency + diversity) / 3.0
        
        return quality_score
    
    def create_preprocessing_pipeline(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> List[Callable]:
        """Create a preprocessing pipeline."""
        pipeline = []
        
        for preprocessing_type in self.config.preprocessing_types:
            if preprocessing_type == PreprocessingType.NORMALIZATION:
                pipeline.append(lambda x: self._normalize_data(x)[0])
            elif preprocessing_type == PreprocessingType.STANDARDIZATION:
                pipeline.append(lambda x: self._standardize_data(x)[0])
            elif preprocessing_type == PreprocessingType.SCALING:
                pipeline.append(lambda x: self._scale_data(x)[0])
            elif preprocessing_type == PreprocessingType.DIMENSIONALITY_REDUCTION:
                pipeline.append(lambda x: self._reduce_dimensionality(x)[0])
            elif preprocessing_type == PreprocessingType.FEATURE_SELECTION and target is not None:
                pipeline.append(lambda x: self._select_features(x, target)[0])
            elif preprocessing_type == PreprocessingType.NOISE_REDUCTION:
                pipeline.append(lambda x: self._reduce_noise(x)[0])
            elif preprocessing_type == PreprocessingType.OUTLIER_DETECTION:
                pipeline.append(lambda x: self._detect_outliers(x)[0])
        
        return pipeline
    
    def get_preprocessing_history(self) -> List[Dict]:
        """Get preprocessing history."""
        return self.preprocessing_history
    
    def get_data_quality_report(self, data: np.ndarray) -> Dict:
        """Get comprehensive data quality report."""
        return {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'missing_values': np.isnan(data).sum(),
            'infinite_values': np.isinf(data).sum(),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'quality_score': self._calculate_data_quality_score(data)
        }
