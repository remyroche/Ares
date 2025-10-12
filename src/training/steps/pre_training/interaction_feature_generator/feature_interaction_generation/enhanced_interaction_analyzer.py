"""
Enhanced Interaction Feature Analyzer

This module provides advanced interaction analysis features that leverage the existing
feature bank from feature_generation/ to create sophisticated feature interactions,
divergences, volatility analysis, cross-timeframe clustering, and trend strength analysis.

Key Features:
- Feature divergence analysis using existing features
- Volatility analysis for feature interactions
- Cross-timeframe clustering features
- Trend strength analysis
- Quantile-based feature interactions
- VectorBT-optimized computations
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import warnings

from ...core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ...core.feature_bank import get_global_feature_bank, FeatureBank

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

# VectorBT Rolling Optimizer
try:
    from ...utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager,
        OperationType,
        OptimizationStrategy
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None

from ...utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class InteractionAnalysisConfig:
    """Configuration for enhanced interaction analysis."""
    # Feature bank integration
    use_feature_bank: bool = True
    feature_categories: List[str] = None
    
    # Divergence analysis
    enable_divergence_analysis: bool = True
    divergence_windows: List[int] = None
    divergence_threshold: float = 0.1
    
    # Volatility analysis
    enable_volatility_analysis: bool = True
    volatility_windows: List[int] = None
    volatility_methods: List[str] = None
    
    # Cross-timeframe clustering
    enable_cross_timeframe_clustering: bool = True
    clustering_timeframes: List[int] = None
    clustering_methods: List[str] = None
    
    # Trend strength analysis
    enable_trend_strength: bool = True
    trend_windows: List[int] = None
    trend_methods: List[str] = None
    
    # Quantile-based features
    enable_quantile_features: bool = True
    quantile_levels: List[float] = None
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    vectorbt_threshold: int = 1000
    enable_gpu: bool = True
    enable_parallel: bool = True
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'trend', 'volume', 'returns']
        if self.divergence_windows is None:
            self.divergence_windows = [5, 10, 20, 50]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50]
        if self.volatility_methods is None:
            self.volatility_methods = ['rolling_std', 'ewm_std', 'garman_klass']
        if self.clustering_timeframes is None:
            self.clustering_timeframes = [5, 15, 30, 60]
        if self.clustering_methods is None:
            self.clustering_methods = ['kmeans', 'hierarchical', 'dbscan']
        if self.trend_windows is None:
            self.trend_windows = [10, 20, 50]
        if self.trend_methods is None:
            self.trend_methods = ['linear_regression', 'polynomial', 'exponential']
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]


class EnhancedInteractionAnalyzer(VectorizedFeatureGenerator):
    """Enhanced interaction analyzer that leverages the feature bank for sophisticated analysis."""
    
    def __init__(self, config: Optional[InteractionAnalysisConfig] = None):
        """Initialize the enhanced interaction analyzer."""
        self.config = config or InteractionAnalysisConfig()
        
        # Initialize feature bank
        self.feature_bank = get_global_feature_bank()
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.config.enable_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel
            )
        
        # Initialize unified vectorization manager
        self.unified_manager = None
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        
        # Initialize base config
        base_config = FeatureConfig(
            name="enhanced_interaction_analyzer",
            category=FeatureCategory.INTERACTION,
            description="Enhanced interaction analysis using feature bank",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters=self.config.__dict__,
            matrix_optimized=True,
            gpu_accelerated=self.config.enable_gpu
        )
        
        super().__init__(base_config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        tprint_success("🚀 Enhanced Interaction Analyzer initialized")
        tprint_info(f"📊 Feature bank integration: {'✅' if self.config.use_feature_bank else '❌'}")
        tprint_info(f"📊 VectorBT optimization: {'✅' if self.config.enable_vectorbt else '❌'}")
    
    def analyze_feature_interactions(self, data: pd.DataFrame, 
                                   target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Analyze feature interactions using the feature bank.
        
        Args:
            data: Input OHLCV data
            target: Optional target series for analysis
            
        Returns:
            DataFrame with interaction analysis features
        """
        tprint_info("🔍 Starting enhanced feature interaction analysis...")
        start_time = time.time()
        
        # Generate base features from feature bank
        base_features = self._generate_base_features(data)
        
        # Initialize results
        interaction_features = {}
        
        # 1. Feature Divergence Analysis
        if self.config.enable_divergence_analysis:
            tprint_info("📊 Analyzing feature divergences...")
            divergence_features = self._analyze_feature_divergences(base_features, data)
            interaction_features.update(divergence_features)
        
        # 2. Volatility Analysis
        if self.config.enable_volatility_analysis:
            tprint_info("📈 Analyzing feature volatility...")
            volatility_features = self._analyze_feature_volatility(base_features, data)
            interaction_features.update(volatility_features)
        
        # 3. Cross-timeframe Clustering
        if self.config.enable_cross_timeframe_clustering:
            tprint_info("⏰ Analyzing cross-timeframe clustering...")
            clustering_features = self._analyze_cross_timeframe_clustering(base_features, data)
            interaction_features.update(clustering_features)
        
        # 4. Trend Strength Analysis
        if self.config.enable_trend_strength:
            tprint_info("📊 Analyzing trend strength...")
            trend_features = self._analyze_trend_strength(base_features, data)
            interaction_features.update(trend_features)
        
        # 5. Quantile-based Features
        if self.config.enable_quantile_features:
            tprint_info("📊 Analyzing quantile-based features...")
            quantile_features = self._analyze_quantile_features(base_features, data)
            interaction_features.update(quantile_features)
        
        # Create result DataFrame
        if interaction_features:
            result_df = pd.DataFrame(interaction_features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Enhanced interaction analysis completed in {execution_time:.3f}s")
        tprint_info(f"📊 Generated {len(result_df.columns)} interaction features")
        
        return result_df
    
    def _generate_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate base features using the feature bank."""
        tprint_debug("🔧 Generating base features from feature bank...")
        
        if not self.config.use_feature_bank:
            # Fallback to basic OHLCV features
            return self._generate_basic_features(data)
        
        try:
            # Generate features from specified categories
            base_features = self.feature_bank.generate_features(
                data=data,
                categories=self.config.feature_categories,
                use_optimized_pipeline=True
            )
            
            tprint_debug(f"✅ Generated {len(base_features.columns)} base features")
            return base_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature bank generation failed: {e}, using basic features")
            return self._generate_basic_features(data)
    
    def _generate_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features as fallback."""
        features = {}
        
        # Basic price features
        features['close'] = data['close']
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Basic volume features
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_returns'] = data['volume'].pct_change()
        
        # Basic volatility features
        features['volatility_20'] = data['close'].rolling(20).std()
        features['volatility_50'] = data['close'].rolling(50).std()
        
        # Basic momentum features
        features['momentum_5'] = data['close'].pct_change(5)
        features['momentum_20'] = data['close'].pct_change(20)
        
        return pd.DataFrame(features, index=data.index)
    
    def _analyze_feature_divergences(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze divergences between different features."""
        tprint_debug("🔍 Analyzing feature divergences...")
        divergence_features = {}
        
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) < 2:
            tprint_warning("⚠️ Need at least 2 numeric features for divergence analysis")
            return divergence_features
        
        # Analyze divergences for different windows
        for window in self.config.divergence_windows:
            for i, feature1 in enumerate(numeric_features):
                for feature2 in numeric_features[i+1:]:
                    try:
                        # Calculate rolling correlation
                        corr = self._optimized_rolling_operation(
                            features[feature1], 'corr', window, other=features[feature2]
                        )
                        
                        # Calculate divergence as deviation from expected correlation
                        expected_corr = corr.rolling(window*2).mean()
                        divergence = corr - expected_corr
                        
                        # Add divergence features
                        divergence_features[f'divergence_{feature1}_{feature2}_{window}'] = divergence
                        divergence_features[f'divergence_strength_{feature1}_{feature2}_{window}'] = abs(divergence)
                        divergence_features[f'divergence_direction_{feature1}_{feature2}_{window}'] = np.sign(divergence)
                        
                        # Price-feature divergences
                        if 'close' in data.columns:
                            price_corr = self._optimized_rolling_operation(
                                data['close'], 'corr', window, other=features[feature1]
                            )
                            price_divergence = price_corr - price_corr.rolling(window*2).mean()
                            divergence_features[f'price_divergence_{feature1}_{window}'] = price_divergence
                            
                    except Exception as e:
                        tprint_debug(f"⚠️ Divergence analysis failed for {feature1}-{feature2}: {e}")
        
        tprint_debug(f"✅ Generated {len(divergence_features)} divergence features")
        return divergence_features
    
    def _analyze_feature_volatility(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze volatility characteristics of features."""
        tprint_debug("📈 Analyzing feature volatility...")
        volatility_features = {}
        
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numeric_features:
            for window in self.config.volatility_windows:
                for method in self.config.volatility_methods:
                    try:
                        if method == 'rolling_std':
                            vol = self._optimized_rolling_operation(features[feature], 'std', window)
                        elif method == 'ewm_std':
                            vol = features[feature].ewm(span=window).std()
                        elif method == 'garman_klass' and 'high' in data.columns and 'low' in data.columns:
                            # Garman-Klass volatility for price features
                            if feature in ['close', 'returns']:
                                hl = np.log(data['high'] / data['low'])
                                co = np.log(data['close'] / data['open']) if 'open' in data.columns else 0
                                gk = 0.5 * hl**2 - (2*np.log(2)-1) * co**2
                                vol = self._optimized_rolling_operation(pd.Series(gk), 'mean', window)
                            else:
                                vol = self._optimized_rolling_operation(features[feature], 'std', window)
                        else:
                            vol = self._optimized_rolling_operation(features[feature], 'std', window)
                        
                        # Add volatility features
                        volatility_features[f'vol_{method}_{feature}_{window}'] = vol
                        volatility_features[f'vol_ratio_{feature}_{window}'] = vol / (vol.rolling(window*2).mean() + 1e-8)
                        volatility_features[f'vol_percentile_{feature}_{window}'] = vol.rolling(window*3).rank(pct=True)
                        
                        # Volatility clustering
                        vol_change = vol.pct_change()
                        volatility_features[f'vol_clustering_{feature}_{window}'] = vol_change.rolling(window).sum()
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Volatility analysis failed for {feature} ({method}): {e}")
        
        tprint_debug(f"✅ Generated {len(volatility_features)} volatility features")
        return volatility_features
    
    def _analyze_cross_timeframe_clustering(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze cross-timeframe clustering patterns."""
        tprint_debug("⏰ Analyzing cross-timeframe clustering...")
        clustering_features = {}
        
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numeric_features:
            for tf1 in self.config.clustering_timeframes:
                for tf2 in self.config.clustering_timeframes:
                    if tf1 >= tf2:
                        continue
                    
                    try:
                        # Calculate features at different timeframes
                        feature_tf1 = self._optimized_rolling_operation(features[feature], 'mean', tf1)
                        feature_tf2 = self._optimized_rolling_operation(features[feature], 'mean', tf2)
                        
                        # Cross-timeframe ratio
                        ratio = feature_tf1 / (feature_tf2 + 1e-8)
                        clustering_features[f'ctf_ratio_{feature}_{tf1}_{tf2}'] = ratio
                        
                        # Cross-timeframe momentum
                        momentum_tf1 = feature_tf1.pct_change(tf1)
                        momentum_tf2 = feature_tf2.pct_change(tf2)
                        momentum_divergence = momentum_tf1 - momentum_tf2
                        clustering_features[f'ctf_momentum_div_{feature}_{tf1}_{tf2}'] = momentum_divergence
                        
                        # Cross-timeframe volatility
                        vol_tf1 = self._optimized_rolling_operation(features[feature], 'std', tf1)
                        vol_tf2 = self._optimized_rolling_operation(features[feature], 'std', tf2)
                        vol_ratio = vol_tf1 / (vol_tf2 + 1e-8)
                        clustering_features[f'ctf_vol_ratio_{feature}_{tf1}_{tf2}'] = vol_ratio
                        
                        # Cross-timeframe correlation
                        corr = self._optimized_rolling_operation(
                            feature_tf1, 'corr', min(tf1, tf2), other=feature_tf2
                        )
                        clustering_features[f'ctf_corr_{feature}_{tf1}_{tf2}'] = corr
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Cross-timeframe clustering failed for {feature} ({tf1}-{tf2}): {e}")
        
        tprint_debug(f"✅ Generated {len(clustering_features)} cross-timeframe clustering features")
        return clustering_features
    
    def _analyze_trend_strength(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze trend strength of features."""
        tprint_debug("📊 Analyzing trend strength...")
        trend_features = {}
        
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numeric_features:
            for window in self.config.trend_windows:
                for method in self.config.trend_methods:
                    try:
                        if method == 'linear_regression':
                            # Linear regression slope
                            def calc_slope(series):
                                if len(series) < 2:
                                    return 0.0
                                try:
                                    x = np.arange(len(series))
                                    slope = np.polyfit(x, series, 1)[0]
                                    return slope
                                except:
                                    return 0.0
                            
                            trend_strength = self._optimized_rolling_operation(
                                features[feature], 'apply', window, func=calc_slope
                            )
                            
                        elif method == 'polynomial':
                            # Polynomial trend (quadratic)
                            def calc_poly_trend(series):
                                if len(series) < 3:
                                    return 0.0
                                try:
                                    x = np.arange(len(series))
                                    coeffs = np.polyfit(x, series, 2)
                                    return coeffs[0]  # Quadratic coefficient
                                except:
                                    return 0.0
                            
                            trend_strength = self._optimized_rolling_operation(
                                features[feature], 'apply', window, func=calc_poly_trend
                            )
                            
                        elif method == 'exponential':
                            # Exponential trend
                            def calc_exp_trend(series):
                                if len(series) < 2:
                                    return 0.0
                                try:
                                    x = np.arange(len(series))
                                    log_series = np.log(np.abs(series) + 1e-8)
                                    slope = np.polyfit(x, log_series, 1)[0]
                                    return slope
                                except:
                                    return 0.0
                            
                            trend_strength = self._optimized_rolling_operation(
                                features[feature], 'apply', window, func=calc_exp_trend
                            )
                        
                        else:
                            # Default to linear regression
                            def calc_slope(series):
                                if len(series) < 2:
                                    return 0.0
                                try:
                                    x = np.arange(len(series))
                                    slope = np.polyfit(x, series, 1)[0]
                                    return slope
                                except:
                                    return 0.0
                            
                            trend_strength = self._optimized_rolling_operation(
                                features[feature], 'apply', window, func=calc_slope
                            )
                        
                        # Add trend features
                        trend_features[f'trend_{method}_{feature}_{window}'] = trend_strength
                        trend_features[f'trend_strength_{feature}_{window}'] = abs(trend_strength)
                        trend_features[f'trend_direction_{feature}_{window}'] = np.sign(trend_strength)
                        
                        # Trend consistency
                        trend_consistency = (trend_strength > 0).rolling(window).mean()
                        trend_features[f'trend_consistency_{feature}_{window}'] = trend_consistency
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Trend strength analysis failed for {feature} ({method}): {e}")
        
        tprint_debug(f"✅ Generated {len(trend_features)} trend strength features")
        return trend_features
    
    def _analyze_quantile_features(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze quantile-based features."""
        tprint_debug("📊 Analyzing quantile-based features...")
        quantile_features = {}
        
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numeric_features:
            for window in self.config.volatility_windows:  # Reuse volatility windows
                for quantile_level in self.config.quantile_levels:
                    try:
                        # Calculate rolling quantiles
                        quantile_val = self._optimized_rolling_operation(
                            features[feature], 'quantile', window, q=quantile_level
                        )
                        
                        # Position within quantile range
                        feature_rank = features[feature].rolling(window).rank(pct=True)
                        quantile_position = (feature_rank <= quantile_level).astype(int)
                        
                        # Distance from quantile
                        quantile_distance = features[feature] - quantile_val
                        quantile_distance_pct = quantile_distance / (quantile_val + 1e-8)
                        
                        # Add quantile features
                        quantile_features[f'quantile_{quantile_level}_{feature}_{window}'] = quantile_val
                        quantile_features[f'quantile_position_{quantile_level}_{feature}_{window}'] = quantile_position
                        quantile_features[f'quantile_distance_{quantile_level}_{feature}_{window}'] = quantile_distance
                        quantile_features[f'quantile_distance_pct_{quantile_level}_{feature}_{window}'] = quantile_distance_pct
                        
                        # Quantile-based interactions
                        if quantile_level == 0.5:  # Median
                            median_deviation = features[feature] - quantile_val
                            quantile_features[f'median_deviation_{feature}_{window}'] = median_deviation
                            quantile_features[f'median_deviation_pct_{feature}_{window}'] = median_deviation / (quantile_val + 1e-8)
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Quantile analysis failed for {feature} (q={quantile_level}): {e}")
        
        tprint_debug(f"✅ Generated {len(quantile_features)} quantile features")
        return quantile_features
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT or fallback."""
        if self.rolling_optimizer and len(data) >= self.config.vectorbt_threshold:
            try:
                if operation == 'mean':
                    return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'corr':
                    other = kwargs.get('other')
                    return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
            except Exception as e:
                tprint_debug(f"⚠️ VectorBT rolling operation failed: {e}, using pandas fallback")
        
        # Fallback to pandas operations
        return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced interaction features."""
        # This method is required by the base class but we use analyze_feature_interactions instead
        result_df = self.analyze_feature_interactions(data)
        
        # Return the first column as a series (for compatibility)
        if not result_df.empty:
            return result_df.iloc[:, 0]
        else:
            return pd.Series(dtype=float, index=data.index)


def create_enhanced_interaction_analyzer(config: Optional[InteractionAnalysisConfig] = None) -> EnhancedInteractionAnalyzer:
    """Create an enhanced interaction analyzer."""
    return EnhancedInteractionAnalyzer(config)


def analyze_feature_interactions(data: pd.DataFrame, 
                               config: Optional[InteractionAnalysisConfig] = None,
                               target: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Analyze feature interactions using the enhanced analyzer.
    
    Args:
        data: Input OHLCV data
        config: Configuration for analysis
        target: Optional target series
        
    Returns:
        DataFrame with interaction analysis features
    """
    analyzer = create_enhanced_interaction_analyzer(config)
    return analyzer.analyze_feature_interactions(data, target)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(n_samples).cumsum() * 0.1,
        'high': 100 + np.random.randn(n_samples).cumsum() * 0.1 + np.random.uniform(0, 0.5, n_samples),
        'low': 100 + np.random.randn(n_samples).cumsum() * 0.1 - np.random.uniform(0, 0.5, n_samples),
        'close': 100 + np.random.randn(n_samples).cumsum() * 0.1,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Test enhanced interaction analysis
    config = InteractionAnalysisConfig(
        enable_vectorbt=True,
        divergence_windows=[5, 10, 20],
        volatility_windows=[10, 20],
        clustering_timeframes=[5, 15, 30],
        trend_windows=[10, 20],
        quantile_levels=[0.25, 0.5, 0.75]
    )
    
    try:
        features = analyze_feature_interactions(data, config)
        print(f"Generated {len(features.columns)} interaction features")
        print(f"Feature columns: {list(features.columns)[:10]}...")
        
    except Exception as e:
        print(f"Error: {e}")