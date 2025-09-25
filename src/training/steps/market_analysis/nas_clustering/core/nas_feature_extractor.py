"""
NAS Feature Extractor - Neural Architecture Search Feature Extraction System

This module provides comprehensive feature extraction capabilities for NAS clustering,
including market data features, technical indicators, statistical features,
and neural architecture features.

Key Features:
- Market data feature extraction
- Technical indicator computation
- Statistical feature generation
- Neural architecture feature extraction
- Feature selection and ranking
- Feature validation and quality assessment
- Hardware-optimized feature computation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

# Import shared utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    calculate_data_quality_metrics, safe_merge_dataframes,
    safe_groupby_operation, safe_apply_function, create_summary_statistics
)
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite, validate_positive,
    safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile
)
from src.utils.serialization_utils import save_object, load_object
from src.utils.tprint import tprint

# Import ML utilities
from src.utils.ml_common.feature_selection import FeatureSelector
from src.utils.ml_common.common_operations import create_ml_pipeline

# Import hardware utilities
try:
    from src.utils.hardware.m1_gpu_utils import is_m1_available, is_mps_available, get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False
    def is_m1_available(): return False
    def is_mps_available(): return False
    def get_m1_gpu_manager(): return None
    def get_m1_memory_optimizer(): return None
    def get_m1_cpu_optimizer(): return None

# Import matrix operations
try:
    from src.utils.matrix_operations import MatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    MatrixOperations = None

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    # Basic configuration
    name: str = "NASFeatureExtractor"
    version: str = "1.0.0"
    
    # Feature types to extract
    market_features: bool = True
    technical_indicators: bool = True
    statistical_features: bool = True
    neural_features: bool = True
    custom_features: bool = False
    
    # Technical indicator parameters
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Statistical feature parameters
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    correlation_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Neural architecture features
    architecture_complexity: bool = True
    layer_features: bool = True
    activation_features: bool = True
    regularization_features: bool = True
    
    # Feature selection
    feature_selection: bool = True
    max_features: int = 100
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    mutual_info_threshold: float = 0.01
    
    # Hardware optimization
    use_hardware_optimization: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    memory_limit_gb: Optional[float] = None
    
    # Quality control
    validate_features: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    fill_missing: bool = True
    missing_strategy: str = "interpolate"  # interpolate, forward_fill, backward_fill, mean, median
    
    # Output configuration
    save_features: bool = True
    output_directory: str = "feature_extraction_results"
    verbose: bool = True

class NASFeatureExtractor:
    """
    Neural Architecture Search Feature Extractor.
    
    This class provides comprehensive feature extraction capabilities for NAS clustering,
    including market data features, technical indicators, statistical features,
    and neural architecture features.
    """
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        """
        Initialize NAS Feature Extractor.
        
        Args:
            config: Configuration object for the feature extractor
        """
        self.config = config or FeatureExtractionConfig()
        self.logger = logger.getChild('NASFeatureExtractor')
        
        # Initialize hardware managers
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if self.config.use_hardware_optimization and HARDWARE_UTILS_AVAILABLE:
            self._initialize_hardware_managers()
        
        # Initialize feature selector
        self.feature_selector = None
        if self.config.feature_selection:
            self.feature_selector = FeatureSelector()
        
        # Initialize matrix operations
        self.matrix_ops = None
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = MatrixOperations()
        
        # Storage for extracted features
        self.features = {}
        self.feature_names = []
        self.feature_importance = {}
        self.feature_quality = {}
        
        # Setup output directory
        self._setup_output_directory()
        
        # Initialize logging
        if self.config.verbose:
            tprint("🚀 NAS Feature Extractor initialized successfully")
            if self.config.use_hardware_optimization:
                tprint(f"🔧 Hardware optimization: {'Enabled' if HARDWARE_UTILS_AVAILABLE else 'Disabled'}")
    
    def _initialize_hardware_managers(self):
        """Initialize hardware-specific managers."""
        try:
            if is_m1_available():
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                if self.config.verbose:
                    tprint("🍎 M1 hardware optimization enabled")
            else:
                if self.config.verbose:
                    tprint("⚠️ M1 hardware not detected, using standard optimization")
        except Exception as e:
            self.logger.warning(f"Failed to initialize hardware managers: {e}")
    
    def _setup_output_directory(self):
        """Setup output directory for feature extraction results."""
        try:
            output_path = Path(self.config.output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (output_path / "features").mkdir(exist_ok=True)
            (output_path / "plots").mkdir(exist_ok=True)
            (output_path / "logs").mkdir(exist_ok=True)
            
            if self.config.verbose:
                tprint(f"📁 Output directory created: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Output directory setup failed: {e}")
    
    def extract_features(self, data: pd.DataFrame, 
                        target_column: Optional[str] = None,
                        architecture_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Extract comprehensive features from data.
        
        Args:
            data: Input DataFrame with market data
            target_column: Optional target column for supervised feature selection
            architecture_data: Optional neural architecture data
            
        Returns:
            DataFrame with extracted features
        """
        if self.config.verbose:
            tprint("🔍 Starting comprehensive feature extraction...")
        
        try:
            # Initialize feature storage
            all_features = []
            feature_names = []
            
            # Extract market features
            if self.config.market_features:
                market_features = self._extract_market_features(data)
                all_features.append(market_features)
                feature_names.extend(market_features.columns.tolist())
            
            # Extract technical indicators
            if self.config.technical_indicators:
                technical_features = self._extract_technical_indicators(data)
                all_features.append(technical_features)
                feature_names.extend(technical_features.columns.tolist())
            
            # Extract statistical features
            if self.config.statistical_features:
                statistical_features = self._extract_statistical_features(data)
                all_features.append(statistical_features)
                feature_names.extend(statistical_features.columns.tolist())
            
            # Extract neural architecture features
            if self.config.neural_features and architecture_data:
                neural_features = self._extract_neural_features(architecture_data)
                all_features.append(neural_features)
                feature_names.extend(neural_features.columns.tolist())
            
            # Extract custom features
            if self.config.custom_features:
                custom_features = self._extract_custom_features(data)
                all_features.append(custom_features)
                feature_names.extend(custom_features.columns.tolist())
            
            # Combine all features
            if all_features:
                combined_features = pd.concat(all_features, axis=1)
            else:
                combined_features = pd.DataFrame(index=data.index)
            
            # Feature validation and quality control
            if self.config.validate_features:
                combined_features = self._validate_features(combined_features)
            
            # Feature selection
            if self.config.feature_selection and self.feature_selector:
                combined_features = self._select_features(combined_features, target_column)
            
            # Store results
            self.features = combined_features
            self.feature_names = combined_features.columns.tolist()
            
            # Save features if requested
            if self.config.save_features:
                self._save_features(combined_features)
            
            if self.config.verbose:
                tprint(f"✅ Feature extraction completed: {len(combined_features.columns)} features extracted")
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _extract_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract market-related features."""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            if 'close' in data.columns:
                features['price_change'] = data['close'].pct_change()
                features['price_change_abs'] = features['price_change'].abs()
                features['price_log_return'] = np.log(data['close'] / data['close'].shift(1))
                
                # Price momentum
                for period in [1, 2, 5, 10, 20]:
                    features[f'price_momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            
            # Volume-based features
            if 'volume' in data.columns:
                features['volume_change'] = data['volume'].pct_change()
                features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                features['volume_std'] = data['volume'].rolling(20).std()
            
            # High-Low features
            if all(col in data.columns for col in ['high', 'low', 'close']):
                features['hl_ratio'] = data['high'] / data['low']
                features['hc_ratio'] = data['high'] / data['close']
                features['lc_ratio'] = data['low'] / data['close']
                features['hlc_range'] = (data['high'] - data['low']) / data['close']
            
            # Open-Close features
            if all(col in data.columns for col in ['open', 'close']):
                features['oc_ratio'] = data['open'] / data['close']
                features['oc_change'] = (data['close'] - data['open']) / data['open']
            
            # Remove infinite and NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Market feature extraction failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _extract_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicator features."""
        try:
            features = pd.DataFrame(index=data.index)
            
            if 'close' not in data.columns:
                return features
            
            close = data['close']
            
            # Simple Moving Averages
            for period in self.config.sma_periods:
                sma = close.rolling(period).mean()
                features[f'sma_{period}'] = sma
                features[f'sma_{period}_ratio'] = close / sma
                features[f'sma_{period}_change'] = sma.pct_change()
            
            # Exponential Moving Averages
            for period in self.config.ema_periods:
                ema = close.ewm(span=period).mean()
                features[f'ema_{period}'] = ema
                features[f'ema_{period}_ratio'] = close / ema
                features[f'ema_{period}_change'] = ema.pct_change()
            
            # RSI (Relative Strength Index)
            if len(close) >= self.config.rsi_period:
                rsi = self._calculate_rsi(close, self.config.rsi_period)
                features['rsi'] = rsi
                features['rsi_overbought'] = (rsi > 70).astype(int)
                features['rsi_oversold'] = (rsi < 30).astype(int)
            
            # MACD (Moving Average Convergence Divergence)
            if len(close) >= self.config.macd_slow:
                macd_line, signal_line, histogram = self._calculate_macd(
                    close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
                )
                features['macd'] = macd_line
                features['macd_signal'] = signal_line
                features['macd_histogram'] = histogram
                features['macd_crossover'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
            
            # Bollinger Bands
            if len(close) >= self.config.bollinger_period:
                upper, middle, lower = self._calculate_bollinger_bands(
                    close, self.config.bollinger_period, self.config.bollinger_std
                )
                features['bb_upper'] = upper
                features['bb_middle'] = middle
                features['bb_lower'] = lower
                features['bb_width'] = (upper - lower) / middle
                features['bb_position'] = (close - lower) / (upper - lower)
                features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean() * 0.5).astype(int)
            
            # Remove infinite and NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Technical indicator extraction failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features."""
        try:
            features = pd.DataFrame(index=data.index)
            
            if 'close' not in data.columns:
                return features
            
            close = data['close']
            
            # Rolling statistics
            for window in self.config.rolling_windows:
                if len(close) >= window:
                    rolling_mean = close.rolling(window).mean()
                    rolling_std = close.rolling(window).std()
                    rolling_skew = close.rolling(window).skew()
                    rolling_kurt = close.rolling(window).kurt()
                    
                    features[f'rolling_mean_{window}'] = rolling_mean
                    features[f'rolling_std_{window}'] = rolling_std
                    features[f'rolling_skew_{window}'] = rolling_skew
                    features[f'rolling_kurt_{window}'] = rolling_kurt
                    features[f'rolling_cv_{window}'] = rolling_std / rolling_mean
                    features[f'rolling_zscore_{window}'] = (close - rolling_mean) / rolling_std
            
            # Volatility features
            for window in self.config.volatility_windows:
                if len(close) >= window:
                    returns = close.pct_change()
                    volatility = returns.rolling(window).std()
                    features[f'volatility_{window}'] = volatility
                    features[f'volatility_ratio_{window}'] = volatility / volatility.rolling(window * 2).mean()
            
            # Correlation features
            if 'volume' in data.columns:
                for window in self.config.correlation_windows:
                    if len(close) >= window:
                        price_volume_corr = close.rolling(window).corr(data['volume'])
                        features[f'price_volume_corr_{window}'] = price_volume_corr
            
            # Remove infinite and NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Statistical feature extraction failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _extract_neural_features(self, architecture_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract neural architecture features."""
        try:
            features = {}
            
            # Architecture complexity features
            if self.config.architecture_complexity:
                if 'layers' in architecture_data:
                    layers = architecture_data['layers']
                    features['n_layers'] = len(layers)
                    features['total_neurons'] = sum(layers)
                    features['avg_layer_size'] = np.mean(layers)
                    features['max_layer_size'] = max(layers)
                    features['min_layer_size'] = min(layers)
                    features['layer_size_std'] = np.std(layers)
                    features['layer_size_cv'] = np.std(layers) / np.mean(layers) if np.mean(layers) > 0 else 0
            
            # Convert to DataFrame
            if features:
                feature_df = pd.DataFrame([features])
                return feature_df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.warning(f"Neural feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _extract_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract custom features."""
        try:
            features = pd.DataFrame(index=data.index)
            # Custom feature extraction logic can be added here
            return features
            
        except Exception as e:
            self.logger.warning(f"Custom feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception:
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_mult)
            lower = middle - (std * std_mult)
            return upper, middle, lower
        except Exception:
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean features."""
        try:
            # Remove features with too many missing values
            missing_threshold = 0.5
            features = features.loc[:, features.isnull().mean() < missing_threshold]
            
            # Remove features with zero variance
            features = features.loc[:, features.var() > 0]
            
            # Fill remaining missing values
            if self.config.fill_missing:
                if self.config.missing_strategy == "interpolate":
                    features = features.interpolate()
                elif self.config.missing_strategy == "forward_fill":
                    features = features.fillna(method='ffill')
                elif self.config.missing_strategy == "backward_fill":
                    features = features.fillna(method='bfill')
                elif self.config.missing_strategy == "mean":
                    features = features.fillna(features.mean())
                elif self.config.missing_strategy == "median":
                    features = features.fillna(features.median())
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature validation failed: {e}")
            return features
    
    def _select_features(self, features: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Select the most important features."""
        try:
            if self.feature_selector is None:
                return features
            
            # Limit number of features
            if len(features.columns) > self.config.max_features:
                # Use variance-based selection
                feature_vars = features.var()
                selected_features = feature_vars.nlargest(self.config.max_features).index
                features = features[selected_features]
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return features
    
    def _save_features(self, features: pd.DataFrame):
        """Save extracted features to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"features_{timestamp}.parquet"
            filepath = os.path.join(self.config.output_directory, "features", filename)
            
            features.to_parquet(filepath)
            
            if self.config.verbose:
                tprint(f"💾 Features saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Feature saving failed: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()
    
    def get_feature_quality(self) -> Dict[str, Any]:
        """Get feature quality metrics."""
        return self.feature_quality.copy()
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of extracted features."""
        if not self.features:
            return {"error": "No features extracted"}
        
        return {
            'n_features': len(self.features.columns),
            'feature_names': self.features.columns.tolist(),
            'data_shape': self.features.shape,
            'missing_values': self.features.isnull().sum().to_dict(),
            'feature_types': self.features.dtypes.to_dict(),
            'feature_stats': self.features.describe().to_dict()
        }
    
    def save_config(self, filepath: str) -> bool:
        """Save feature extraction configuration."""
        try:
            config_dict = self.config.__dict__
            success = save_object(config_dict, filepath)
            
            if success and self.config.verbose:
                tprint(f"💾 Configuration saved to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Configuration saving failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the feature extractor."""
        return f"NASFeatureExtractor(name='{self.config.name}', n_features={len(self.features.columns) if self.features else 0})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()

# Convenience functions
def create_feature_extractor(config: Optional[FeatureExtractionConfig] = None) -> NASFeatureExtractor:
    """Create a new NAS Feature Extractor instance."""
    return NASFeatureExtractor(config)

def quick_feature_extraction(data: pd.DataFrame, 
                           target_column: Optional[str] = None,
                           **kwargs) -> pd.DataFrame:
    """
    Quick feature extraction with default settings.
    
    Args:
        data: Input DataFrame
        target_column: Optional target column
        **kwargs: Additional configuration parameters
        
    Returns:
        DataFrame with extracted features
    """
    config = FeatureExtractionConfig(**kwargs)
    extractor = NASFeatureExtractor(config)
    return extractor.extract_features(data, target_column)

# Export main classes and functions
__all__ = [
    'NASFeatureExtractor',
    'FeatureExtractionConfig',
    'create_feature_extractor',
    'quick_feature_extraction'
]