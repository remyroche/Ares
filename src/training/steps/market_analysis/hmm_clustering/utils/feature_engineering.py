"""
Feature engineering utilities for HMM clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering operations."""
    features: np.ndarray
    feature_names: List[str]
    scaler: Any
    feature_stats: Dict[str, Any]
    engineering_metadata: Dict[str, Any]
    execution_time: float
    matrix_ops_used: bool
    hardware_acceleration_used: bool


class FeatureEngineer:
    """Feature engineering utilities with hardware acceleration."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the feature engineer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for feature engineering")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for feature engineering: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for feature engineering")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for feature engineering: {e}")

    def extract_4d_features(self, data: Union[pd.DataFrame, np.ndarray], 
                          feature_config: Dict[str, Any] = None) -> FeatureEngineeringResult:
        """Extract 4D features (volume, volatility, momentum, trend).

        Args:
            data: Input data (DataFrame or numpy array)
            feature_config: Feature extraction configuration

        Returns:
            FeatureEngineeringResult with extracted features
        """
        import time
        start_time = time.time()
        
        try:
            # Monitor performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring("4d_feature_extraction")
            
            # Prepare data
            if isinstance(data, pd.DataFrame):
                data_array = data.values
                feature_names = data.columns.tolist()
            else:
                data_array = data
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            
            # Extract 4D features
            if feature_config is None:
                feature_config = self._get_default_4d_config()
            
            features_4d = self._extract_4d_features_from_data(data_array, feature_config)
            feature_names_4d = self._generate_4d_feature_names(feature_config)
            
            # Normalize features if configured
            scaler = None
            if feature_config.get('normalize_features', True):
                features_4d, scaler = self._normalize_features(features_4d, feature_config.get('scaling_method', 'standard'))
            
            # Calculate feature statistics
            feature_stats = self._calculate_feature_statistics(features_4d, feature_names_4d)
            
            # Stop performance monitoring
            perf_metrics = {}
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("4d_feature_extraction")
            
            execution_time = time.time() - start_time
            
            # Create result
            result = FeatureEngineeringResult(
                features=features_4d,
                feature_names=feature_names_4d,
                scaler=scaler,
                feature_stats=feature_stats,
                engineering_metadata={
                    'feature_config': feature_config,
                    'performance_metrics': perf_metrics,
                    'extraction_method': '4d_matrix_optimized' if self.matrix_ops else '4d_standard'
                },
                execution_time=execution_time,
                matrix_ops_used=self.matrix_ops is not None,
                hardware_acceleration_used=self.hardware_accelerator is not None
            )
            
            self.logger.info(f"✅ 4D feature extraction completed: {features_4d.shape} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ 4D feature extraction failed: {e}")
            return FeatureEngineeringResult(
                features=np.array([]),
                feature_names=[],
                scaler=None,
                feature_stats={},
                engineering_metadata={'error': str(e)},
                execution_time=execution_time,
                matrix_ops_used=False,
                hardware_acceleration_used=False
            )

    def _extract_4d_features_from_data(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Extract 4D features from data array.

        Args:
            data: Input data array
            config: Feature extraction configuration

        Returns:
            4D feature matrix
        """
        try:
            # Use matrix operations for feature extraction if available
            if self.matrix_ops is not None:
                return self._extract_4d_features_matrix_ops(data, config)
            else:
                return self._extract_4d_features_standard(data, config)
                
        except Exception as e:
            self.logger.warning(f"⚠️ 4D feature extraction failed: {e}")
            # Fallback to first 4 features
            return data[:, :min(4, data.shape[1])]

    def _extract_4d_features_matrix_ops(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Extract 4D features using matrix operations.

        Args:
            data: Input data array
            config: Feature extraction configuration

        Returns:
            4D feature matrix
        """
        try:
            # Use matrix operations for efficient feature extraction
            if hasattr(self.matrix_ops, 'extract_4d_features'):
                features_4d = self.matrix_ops.extract_4d_features(data, config)
            else:
                # Fallback to standard extraction with matrix operations for calculations
                features_4d = self._extract_4d_features_standard(data, config)
                
                # Apply matrix operations for feature transformations
                if config.get('apply_matrix_transformations', True):
                    features_4d = self._apply_matrix_transformations(features_4d)
            
            return features_4d
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix operations 4D feature extraction failed: {e}")
            return self._extract_4d_features_standard(data, config)

    def _extract_4d_features_standard(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Extract 4D features using standard methods.

        Args:
            data: Input data array
            config: Feature extraction configuration

        Returns:
            4D feature matrix
        """
        try:
            features_4d = []
            
            # Volume features
            if config.get('volume_features', True):
                volume_features = self._extract_volume_features(data)
                features_4d.append(volume_features)
            
            # Volatility features
            if config.get('volatility_features', True):
                volatility_features = self._extract_volatility_features(data)
                features_4d.append(volatility_features)
            
            # Momentum features
            if config.get('momentum_features', True):
                momentum_features = self._extract_momentum_features(data)
                features_4d.append(momentum_features)
            
            # Trend features
            if config.get('trend_features', True):
                trend_features = self._extract_trend_features(data)
                features_4d.append(trend_features)
            
            # Combine features
            if features_4d:
                features_4d = np.column_stack(features_4d)
            else:
                # Fallback to first 4 features
                features_4d = data[:, :min(4, data.shape[1])]
            
            return features_4d
            
        except Exception as e:
            self.logger.warning(f"⚠️ Standard 4D feature extraction failed: {e}")
            return data[:, :min(4, data.shape[1])]

    def _extract_volume_features(self, data: np.ndarray) -> np.ndarray:
        """Extract volume-related features.

        Args:
            data: Input data array

        Returns:
            Volume features
        """
        try:
            # Use first column as volume proxy
            volume = data[:, 0]
            
            # Calculate volume-based features
            volume_mean = np.mean(volume)
            volume_std = np.std(volume)
            
            # Normalize volume
            volume_normalized = (volume - volume_mean) / (volume_std + 1e-8)
            
            return volume_normalized
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume feature extraction failed: {e}")
            return data[:, 0]

    def _extract_volatility_features(self, data: np.ndarray) -> np.ndarray:
        """Extract volatility-related features.

        Args:
            data: Input data array

        Returns:
            Volatility features
        """
        try:
            # Use second column as volatility proxy
            if data.shape[1] > 1:
                price = data[:, 1]
            else:
                price = data[:, 0]
            
            # Calculate rolling volatility
            window_size = min(20, len(price) // 4)
            if window_size < 2:
                return np.zeros_like(price)
            
            volatility = []
            for i in range(len(price)):
                start_idx = max(0, i - window_size + 1)
                window_data = price[start_idx:i+1]
                if len(window_data) > 1:
                    vol = np.std(window_data)
                else:
                    vol = 0.0
                volatility.append(vol)
            
            volatility = np.array(volatility)
            
            # Normalize volatility
            vol_mean = np.mean(volatility)
            vol_std = np.std(volatility)
            volatility_normalized = (volatility - vol_mean) / (vol_std + 1e-8)
            
            return volatility_normalized
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility feature extraction failed: {e}")
            return np.zeros(data.shape[0])

    def _extract_momentum_features(self, data: np.ndarray) -> np.ndarray:
        """Extract momentum-related features.

        Args:
            data: Input data array

        Returns:
            Momentum features
        """
        try:
            # Use first column as price proxy
            price = data[:, 0]
            
            # Calculate momentum
            momentum = np.diff(price, prepend=price[0])
            
            # Calculate momentum features
            momentum_mean = np.mean(momentum)
            momentum_std = np.std(momentum)
            
            # Normalize momentum
            momentum_normalized = (momentum - momentum_mean) / (momentum_std + 1e-8)
            
            return momentum_normalized
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum feature extraction failed: {e}")
            return np.zeros(data.shape[0])

    def _extract_trend_features(self, data: np.ndarray) -> np.ndarray:
        """Extract trend-related features.

        Args:
            data: Input data array

        Returns:
            Trend features
        """
        try:
            # Use first column as price proxy
            price = data[:, 0]
            
            # Calculate trend using linear regression
            x = np.arange(len(price))
            
            # Simple trend calculation
            if len(price) > 1:
                trend = np.polyfit(x, price, 1)[0]
                trend_features = np.full_like(price, trend)
            else:
                trend_features = np.zeros_like(price)
            
            # Normalize trend
            trend_mean = np.mean(trend_features)
            trend_std = np.std(trend_features)
            trend_normalized = (trend_features - trend_mean) / (trend_std + 1e-8)
            
            return trend_normalized
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend feature extraction failed: {e}")
            return np.zeros(data.shape[0])

    def _apply_matrix_transformations(self, features: np.ndarray) -> np.ndarray:
        """Apply matrix transformations to features.

        Args:
            features: Input features

        Returns:
            Transformed features
        """
        try:
            if self.matrix_ops is not None:
                # Apply matrix operations for feature transformations
                if hasattr(self.matrix_ops, 'apply_feature_transformations'):
                    return self.matrix_ops.apply_feature_transformations(features)
                else:
                    # Apply standard transformations using matrix operations
                    return self._apply_standard_transformations(features)
            else:
                return self._apply_standard_transformations(features)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix transformations failed: {e}")
            return features

    def _apply_standard_transformations(self, features: np.ndarray) -> np.ndarray:
        """Apply standard feature transformations.

        Args:
            features: Input features

        Returns:
            Transformed features
        """
        try:
            # Apply log transformation if needed
            features_transformed = np.log1p(np.abs(features)) * np.sign(features)
            
            return features_transformed
            
        except Exception as e:
            self.logger.warning(f"⚠️ Standard transformations failed: {e}")
            return features

    def _normalize_features(self, features: np.ndarray, scaling_method: str = 'standard') -> Tuple[np.ndarray, Any]:
        """Normalize features using specified scaling method.

        Args:
            features: Input features
            scaling_method: Scaling method ('standard', 'robust', 'minmax')

        Returns:
            Tuple of (normalized_features, scaler)
        """
        try:
            # Use matrix operations for normalization if available
            if self.matrix_ops is not None and hasattr(self.matrix_ops, 'normalize_features'):
                normalized_features, scaler = self.matrix_ops.normalize_features(features, scaling_method)
                return normalized_features, scaler
            
            # Standard normalization
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            normalized_features = scaler.fit_transform(features)
            
            return normalized_features, scaler
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return features, None

    def _calculate_feature_statistics(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate feature statistics.

        Args:
            features: Feature matrix
            feature_names: Feature names

        Returns:
            Dictionary of feature statistics
        """
        try:
            stats = {
                'n_features': features.shape[1],
                'n_samples': features.shape[0],
                'feature_statistics': {}
            }
            
            for i, feature_name in enumerate(feature_names):
                feature_values = features[:, i]
                stats['feature_statistics'][feature_name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'median': float(np.median(feature_values)),
                    'q25': float(np.percentile(feature_values, 25)),
                    'q75': float(np.percentile(feature_values, 75))
                }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature statistics calculation failed: {e}")
            return {'error': str(e)}

    def _get_default_4d_config(self) -> Dict[str, Any]:
        """Get default 4D feature extraction configuration.

        Returns:
            Default configuration
        """
        return {
            'volume_features': True,
            'volatility_features': True,
            'momentum_features': True,
            'trend_features': True,
            'normalize_features': True,
            'scaling_method': 'standard',
            'apply_matrix_transformations': True
        }

    def _generate_4d_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Generate 4D feature names.

        Args:
            config: Feature configuration

        Returns:
            List of feature names
        """
        feature_names = []
        
        if config.get('volume_features', True):
            feature_names.append('volume_normalized')
        
        if config.get('volatility_features', True):
            feature_names.append('volatility_normalized')
        
        if config.get('momentum_features', True):
            feature_names.append('momentum_normalized')
        
        if config.get('trend_features', True):
            feature_names.append('trend_normalized')
        
        return feature_names


def create_feature_engineer(config: Dict[str, Any] = None) -> FeatureEngineer:
    """Create a feature engineer instance.

    Args:
        config: Configuration dictionary

    Returns:
        FeatureEngineer instance
    """
    return FeatureEngineer(config)