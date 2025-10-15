"""
Feature Collection Utilities for Hybrid NAS-TAS Regime Detection.

Provides common feature collection utilities using the pre-existing feature_generator/ system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
import warnings
from datetime import datetime
# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

try:
    from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureResult
    from src.feature_generation.core.factory import FeatureFactory
    from src.feature_generation.categories.momentum import MomentumCalculator
    from src.feature_generation.categories.volatility import VolatilityCalculator
    from src.feature_generation.categories.volume import VolumeCalculator
    from src.feature_generation.categories.trend import TrendCalculator
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False

# Import shared FeatureConfig for compatibility
try:
    from src.training.steps.market_analysis.shared_utils.features import FeatureConfig
    SHARED_FEATURES_AVAILABLE = True
except ImportError:
    SHARED_FEATURES_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        UnifiedMatrixOperations,
        MatrixOperationRegistry,
        VectorBTRollingOptimizer,
        UnifiedVectorizationManager
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureCollectionConfig:
    """Configuration for feature collection operations."""
    use_standardized_features: bool = True
    feature_categories: List[str] = None
    lookback_periods: List[int] = None
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    validation_enabled: bool = True
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend']
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]


@dataclass
class FeatureCollectionResult:
    """Result from feature collection operations."""
    features: pd.DataFrame
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False


class StandardizedFeatureCalculator:
    """Standardized feature calculator using the existing feature_generator system."""
    
    def __init__(self, config: FeatureCollectionConfig):
        """Initialize the standardized feature calculator.
        
        Args:
            config: Feature collection configuration
        """
        tprint("🔧 Initializing StandardizedFeatureCalculator", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint(f"📊 Config: {config.feature_categories} categories, {config.lookback_periods} lookbacks", color="cyan")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                tprint("⚡ Initializing matrix operations for hardware acceleration", color="yellow")
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for feature calculation")
                tprint("✅ Matrix operations initialized successfully", color="green")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")
                tprint(f"⚠️ Matrix operations failed: {e}", color="red")
        
        # Initialize feature calculators
        tprint("🧮 Initializing feature calculators", color="blue")
        self.feature_calculators = self._initialize_feature_calculators()
        
        self.logger.info("✅ Standardized Feature Calculator initialized")
        tprint("✅ StandardizedFeatureCalculator initialization complete", color="green")
    
    def _initialize_feature_calculators(self) -> Dict[str, Any]:
        """Initialize feature calculators for different categories."""
        tprint("🔧 Setting up feature calculators for each category", color="blue")
        calculators = {}
        
        if FEATURE_GENERATION_AVAILABLE:
            try:
                tprint("📈 Creating momentum calculator", color="cyan")
                calculators['momentum'] = MomentumCalculator()
                tprint("📊 Creating volatility calculator", color="cyan")
                calculators['volatility'] = VolatilityCalculator()
                tprint("📦 Creating volume calculator", color="cyan")
                calculators['volume'] = VolumeCalculator()
                tprint("📈 Creating trend calculator", color="cyan")
                calculators['trend'] = TrendCalculator()
                self.logger.info("✅ Feature calculators initialized")
                tprint(f"✅ All {len(calculators)} feature calculators initialized", color="green")
            except Exception as e:
                self.logger.warning(f"⚠️ Feature calculator initialization failed: {e}")
                tprint(f"❌ Feature calculator initialization failed: {e}", color="red")
        
        return calculators
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all standardized features.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with calculated features
        """
        try:
            tprint(f"🔧 Starting feature calculation for {data.shape[0]} samples", color="blue")
            self.logger.info("🔧 Calculating standardized features")
            start_time = time.time()
            
            # Initialize features DataFrame
            features_df = pd.DataFrame(index=data.index)
            
            # Calculate features for each category
            tprint(f"📊 Processing {len(self.config.feature_categories)} feature categories", color="cyan")
            for category in self.config.feature_categories:
                tprint(f"🔄 Processing {category} features", color="yellow")
                if category in self.feature_calculators:
                    category_features = self._calculate_category_features(
                        data, category, self.feature_calculators[category]
                    )
                    if category_features is not None and not category_features.empty:
                        features_df = pd.concat([features_df, category_features], axis=1)
                        tprint(f"✅ {category}: {category_features.shape[1]} features added", color="green")
                    else:
                        tprint(f"⚠️ {category}: No features generated", color="yellow")
                else:
                    tprint(f"❌ {category}: Calculator not available", color="red")
            
            # Apply standardization if configured
            if self.config.use_standardized_features:
                tprint("🔧 Applying feature standardization", color="blue")
                features_df = self._standardize_features(features_df)
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Features calculated: {features_df.shape} in {processing_time:.2f}s")
            tprint(f"✅ Feature calculation complete: {features_df.shape[1]} features in {processing_time:.2f}s", color="green")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Feature calculation failed: {e}")
            tprint(f"❌ Feature calculation failed: {e}", color="red")
            return pd.DataFrame()
    
    def _calculate_category_features(self, data: pd.DataFrame, category: str, calculator: Any) -> Optional[pd.DataFrame]:
        """Calculate features for a specific category.
        
        Args:
            data: Market data
            category: Feature category
            calculator: Feature calculator
            
        Returns:
            DataFrame with category features
        """
        try:
            if category == 'momentum':
                return self._calculate_momentum_features(data, calculator)
            elif category == 'volatility':
                return self._calculate_volatility_features(data, calculator)
            elif category == 'volume':
                return self._calculate_volume_features(data, calculator)
            elif category == 'trend':
                return self._calculate_trend_features(data, calculator)
            else:
                self.logger.warning(f"⚠️ Unknown feature category: {category}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Feature calculation failed for {category}: {e}")
            return None
    
    def _calculate_momentum_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Calculate momentum features."""
        try:
            tprint(f"📈 Calculating momentum features for {len(self.config.lookback_periods)} lookback periods", color="cyan")
            features = pd.DataFrame(index=data.index)
            
            # Calculate momentum features for different lookback periods
            for lookback in self.config.lookback_periods:
                if hasattr(calculator, 'calculate_momentum'):
                    momentum = calculator.calculate_momentum(data, lookback)
                    if momentum is not None:
                        features[f'momentum_{lookback}'] = momentum
                else:
                    # Fallback calculation
                    if 'close' in data.columns:
                        momentum = data['close'].pct_change(lookback)
                        features[f'momentum_{lookback}'] = momentum
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum feature calculation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_volatility_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Calculate volatility features."""
        try:
            tprint(f"📊 Calculating volatility features for {len(self.config.lookback_periods)} lookback periods", color="cyan")
            features = pd.DataFrame(index=data.index)
            
            # Calculate volatility features for different lookback periods
            for lookback in self.config.lookback_periods:
                if hasattr(calculator, 'calculate_volatility'):
                    volatility = calculator.calculate_volatility(data, lookback)
                    if volatility is not None:
                        features[f'volatility_{lookback}'] = volatility
                else:
                    # Fallback calculation
                    if 'close' in data.columns:
                        returns = data['close'].pct_change()
                        volatility = returns.rolling(lookback).std()
                        features[f'volatility_{lookback}'] = volatility
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility feature calculation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_volume_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Calculate volume features."""
        try:
            tprint(f"📦 Calculating volume features for {len(self.config.lookback_periods)} lookback periods", color="cyan")
            features = pd.DataFrame(index=data.index)
            
            # Calculate volume features for different lookback periods
            for lookback in self.config.lookback_periods:
                if hasattr(calculator, 'calculate_volume_features'):
                    volume_features = calculator.calculate_volume_features(data, lookback)
                    if volume_features is not None:
                        for col in volume_features.columns:
                            features[f'volume_{col}_{lookback}'] = volume_features[col]
                else:
                    # Fallback calculation
                    if 'volume' in data.columns:
                        volume_ma = data['volume'].rolling(lookback).mean()
                        volume_ratio = data['volume'] / volume_ma
                        features[f'volume_ratio_{lookback}'] = volume_ratio
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume feature calculation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_trend_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Calculate trend features."""
        try:
            tprint(f"📈 Calculating trend features for {len(self.config.lookback_periods)} lookback periods", color="cyan")
            features = pd.DataFrame(index=data.index)
            
            # Calculate trend features for different lookback periods
            for lookback in self.config.lookback_periods:
                if hasattr(calculator, 'calculate_trend'):
                    trend = calculator.calculate_trend(data, lookback)
                    if trend is not None:
                        features[f'trend_{lookback}'] = trend
                else:
                    # Fallback calculation
                    if 'close' in data.columns:
                        # Simple trend calculation using linear regression slope
                        trend_scores = []
                        for i in range(len(data)):
                            if i >= lookback:
                                window_data = data['close'].iloc[i-lookback:i+1]
                                if len(window_data) > 1:
                                    x = np.arange(len(window_data))
                                    y = window_data.values
                                    slope = np.polyfit(x, y, 1)[0]
                                    trend_scores.append(slope)
                                else:
                                    trend_scores.append(0)
                            else:
                                trend_scores.append(0)
                        features[f'trend_{lookback}'] = trend_scores
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend feature calculation failed: {e}")
            return pd.DataFrame()
    
    def _standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features using z-score normalization.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            Standardized features DataFrame
        """
        try:
            standardized_features = features.copy()
            
            # Apply z-score standardization
            for column in standardized_features.columns:
                if standardized_features[column].dtype in ['float64', 'int64']:
                    mean_val = standardized_features[column].mean()
                    std_val = standardized_features[column].std()
                    if std_val > 0:
                        standardized_features[column] = (standardized_features[column] - mean_val) / std_val
                    else:
                        standardized_features[column] = 0
            
            return standardized_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature standardization failed: {e}")
            return features
    
    def get_primary_features(self) -> Dict[str, List[str]]:
        """Get primary features for each category.
        
        Returns:
            Dictionary mapping categories to feature names
        """
        primary_features = {
            'momentum': ['momentum_20', 'momentum_12'],
            'volatility': ['volatility_20', 'volatility_12'],
            'volume': ['volume_ratio_192m'],
            'trend': ['trend_score']
        }
        
        return primary_features


class FeatureCollectionManager:
    """Manager for feature collection operations with coordination between NAS and TAS."""
    
    def __init__(self, config: FeatureCollectionConfig):
        """Initialize the feature collection manager.
        
        Args:
            config: Feature collection configuration
        """
        tprint("🎯 Initializing FeatureCollectionManager", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.calculator = StandardizedFeatureCalculator(config)
        
        self.logger.info("✅ Feature Collection Manager initialized")
        tprint("✅ FeatureCollectionManager initialization complete", color="green")
    
    async def collect_features_for_nas(self, data: pd.DataFrame) -> FeatureCollectionResult:
        """Collect features specifically for NAS regime detection.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            FeatureCollectionResult with NAS-specific features
        """
        try:
            tprint(f"🧠 Starting NAS feature collection for {data.shape[0]} samples", color="blue")
            self.logger.info("🧠 Collecting features for NAS regime detection")
            start_time = time.time()
            
            # Calculate all features
            features = self.calculator.calculate_all_features(data)
            
            if features.empty:
                raise ValueError("No features calculated")
            
            # NAS-specific feature selection and processing
            tprint("🔍 Selecting NAS-specific features", color="yellow")
            nas_features = self._select_nas_features(features)
            
            processing_time = time.time() - start_time
            
            # Get feature categories
            tprint("📊 Categorizing features", color="cyan")
            feature_categories = self._categorize_features(nas_features)
            
            metadata = {
                'collection_type': 'nas_regime',
                'original_shape': data.shape,
                'features_shape': nas_features.shape,
                'feature_categories': feature_categories,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ NAS features collected: {nas_features.shape}")
            tprint(f"✅ NAS feature collection complete: {nas_features.shape[1]} features in {processing_time:.2f}s", color="green")
            
            return FeatureCollectionResult(
                features=nas_features,
                feature_names=list(nas_features.columns),
                feature_categories=feature_categories,
                metadata=metadata,
                processing_time=processing_time,
                success=True,
                hardware_optimization_applied=self.calculator.matrix_ops is not None,
                matrix_operations_used=self.calculator.matrix_ops is not None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ NAS feature collection failed: {e}")
            tprint(f"❌ NAS feature collection failed: {e}", color="red")
            return FeatureCollectionResult(
                features=pd.DataFrame(),
                feature_names=[],
                feature_categories={},
                metadata={'error': str(e)},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def collect_features_for_tas(self, data: pd.DataFrame) -> FeatureCollectionResult:
        """Collect features specifically for TAS regime detection.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            FeatureCollectionResult with TAS-specific features
        """
        try:
            tprint(f"🌳 Starting TAS feature collection for {data.shape[0]} samples", color="blue")
            self.logger.info("🌳 Collecting features for TAS regime detection")
            start_time = time.time()
            
            # Calculate all features
            features = self.calculator.calculate_all_features(data)
            
            if features.empty:
                raise ValueError("No features calculated")
            
            # TAS-specific feature selection and processing
            tprint("🔍 Selecting TAS-specific features", color="yellow")
            tas_features = self._select_tas_features(features)
            
            processing_time = time.time() - start_time
            
            # Get feature categories
            tprint("📊 Categorizing features", color="cyan")
            feature_categories = self._categorize_features(tas_features)
            
            metadata = {
                'collection_type': 'tas_regime',
                'original_shape': data.shape,
                'features_shape': tas_features.shape,
                'feature_categories': feature_categories,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ TAS features collected: {tas_features.shape}")
            tprint(f"✅ TAS feature collection complete: {tas_features.shape[1]} features in {processing_time:.2f}s", color="green")
            
            return FeatureCollectionResult(
                features=tas_features,
                feature_names=list(tas_features.columns),
                feature_categories=feature_categories,
                metadata=metadata,
                processing_time=processing_time,
                success=True,
                hardware_optimization_applied=self.calculator.matrix_ops is not None,
                matrix_operations_used=self.calculator.matrix_ops is not None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ TAS feature collection failed: {e}")
            tprint(f"❌ TAS feature collection failed: {e}", color="red")
            return FeatureCollectionResult(
                features=pd.DataFrame(),
                feature_names=[],
                feature_categories={},
                metadata={'error': str(e)},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _select_nas_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select features optimized for NAS regime detection.
        
        Args:
            features: All calculated features
            
        Returns:
            NAS-optimized features
        """
        try:
            tprint(f"🔍 Selecting NAS features from {features.shape[1]} total features", color="cyan")
            # NAS typically benefits from momentum and volatility features
            nas_feature_names = []
            
            for col in features.columns:
                if any(keyword in col.lower() for keyword in ['momentum', 'volatility', 'trend']):
                    nas_feature_names.append(col)
            
            if nas_feature_names:
                tprint(f"✅ Selected {len(nas_feature_names)} NAS-specific features", color="green")
                return features[nas_feature_names]
            else:
                # Fallback to first few features
                fallback_count = min(10, features.shape[1])
                tprint(f"⚠️ Using fallback: first {fallback_count} features", color="yellow")
                return features.iloc[:, :fallback_count]
                
        except Exception as e:
            self.logger.warning(f"⚠️ NAS feature selection failed: {e}")
            return features
    
    def _select_tas_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select features optimized for TAS regime detection.
        
        Args:
            features: All calculated features
            
        Returns:
            TAS-optimized features
        """
        try:
            tprint(f"🔍 Selecting TAS features from {features.shape[1]} total features", color="cyan")
            # TAS typically benefits from volume and trend features
            tas_feature_names = []
            
            for col in features.columns:
                if any(keyword in col.lower() for keyword in ['volume', 'trend', 'momentum']):
                    tas_feature_names.append(col)
            
            if tas_feature_names:
                tprint(f"✅ Selected {len(tas_feature_names)} TAS-specific features", color="green")
                return features[tas_feature_names]
            else:
                # Fallback to first few features
                fallback_count = min(10, features.shape[1])
                tprint(f"⚠️ Using fallback: first {fallback_count} features", color="yellow")
                return features.iloc[:, :fallback_count]
                
        except Exception as e:
            self.logger.warning(f"⚠️ TAS feature selection failed: {e}")
            return features
    
    def _categorize_features(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by type.
        
        Args:
            features: Features DataFrame
            
        Returns:
            Dictionary mapping categories to feature names
        """
        try:
            tprint(f"📊 Categorizing {features.shape[1]} features", color="cyan")
            categories = {
                'momentum': [],
                'volatility': [],
                'volume': [],
                'trend': [],
                'other': []
            }
            
            for col in features.columns:
                col_lower = col.lower()
                if 'momentum' in col_lower:
                    categories['momentum'].append(col)
                elif 'volatility' in col_lower:
                    categories['volatility'].append(col)
                elif 'volume' in col_lower:
                    categories['volume'].append(col)
                elif 'trend' in col_lower:
                    categories['trend'].append(col)
                else:
                    categories['other'].append(col)
            
            return categories
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature categorization failed: {e}")
            return {'other': list(features.columns)}
    
    def get_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about collected features.
        
        Args:
            features: Features DataFrame
            
        Returns:
            Feature statistics
        """
        try:
            tprint(f"📊 Calculating statistics for {features.shape[1]} features", color="cyan")
            stats = {
                'total_features': features.shape[1],
                'total_samples': features.shape[0],
                'feature_statistics': {},
                'missing_values': features.isnull().sum().to_dict(),
                'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().to_dict()
            }
            
            # Calculate statistics for each feature
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    stats['feature_statistics'][col] = {
                        'mean': float(features[col].mean()),
                        'std': float(features[col].std()),
                        'min': float(features[col].min()),
                        'max': float(features[col].max()),
                        'median': float(features[col].median())
                    }
            
            tprint(f"✅ Feature statistics calculated: {stats['total_features']} features, {stats['total_samples']} samples", color="green")
            return stats
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature statistics calculation failed: {e}")
            return {'error': str(e)}


def create_feature_collection_manager(config: FeatureCollectionConfig) -> FeatureCollectionManager:
    """Create a feature collection manager instance.
    
    Args:
        config: Feature collection configuration
        
    Returns:
        FeatureCollectionManager instance
    """
    tprint("🏭 Creating FeatureCollectionManager instance", color="blue")
    manager = FeatureCollectionManager(config)
    tprint("✅ FeatureCollectionManager created successfully", color="green")
    return manager

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
