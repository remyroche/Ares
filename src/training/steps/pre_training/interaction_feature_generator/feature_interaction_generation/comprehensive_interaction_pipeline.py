"""
Comprehensive Interaction Feature Pipeline

This module provides a comprehensive pipeline for generating enhanced interaction features
that leverages the existing feature bank and provides sophisticated analysis capabilities.

Key Features:
- Feature bank integration for base feature generation
- Enhanced interaction analysis (divergences, volatility, clustering, trend strength)
- VectorBT-optimized computations
- Quantile-based feature analysis
- Cross-timeframe analysis
- Advanced statistical interactions
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

# Import enhanced analyzers
from .enhanced_interaction_analyzer import (
    EnhancedInteractionAnalyzer, 
    InteractionAnalysisConfig,
    analyze_feature_interactions
)
from .enhanced_interaction_generators import create_enhanced_interaction_generators

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
class ComprehensiveInteractionConfig:
    """Configuration for comprehensive interaction feature pipeline."""
    # Feature bank integration
    use_feature_bank: bool = True
    feature_categories: List[str] = None
    feature_budget: int = 120
    
    # Analysis components
    enable_divergence_analysis: bool = True
    enable_volatility_analysis: bool = True
    enable_cross_timeframe_clustering: bool = True
    enable_trend_strength: bool = True
    enable_quantile_features: bool = True
    enable_enhanced_generators: bool = True
    
    # Analysis parameters
    divergence_windows: List[int] = None
    volatility_windows: List[int] = None
    clustering_timeframes: List[int] = None
    trend_windows: List[int] = None
    quantile_levels: List[float] = None
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    vectorbt_threshold: int = 1000
    enable_gpu: bool = True
    enable_parallel: bool = True
    
    # Performance
    max_interactions: int = 15
    enable_caching: bool = True
    chunk_size: int = 1000
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'trend', 'volume', 'returns']
        if self.divergence_windows is None:
            self.divergence_windows = [5, 10, 20, 50]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50]
        if self.clustering_timeframes is None:
            self.clustering_timeframes = [5, 15, 30, 60]
        if self.trend_windows is None:
            self.trend_windows = [10, 20, 50]
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]


class ComprehensiveInteractionPipeline(VectorizedFeatureGenerator):
    """Comprehensive pipeline for generating enhanced interaction features."""
    
    def __init__(self, config: Optional[ComprehensiveInteractionConfig] = None):
        """Initialize the comprehensive interaction pipeline."""
        self.config = config or ComprehensiveInteractionConfig()
        
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
        
        # Initialize enhanced interaction analyzer
        analyzer_config = InteractionAnalysisConfig(
            use_feature_bank=self.config.use_feature_bank,
            feature_categories=self.config.feature_categories,
            enable_divergence_analysis=self.config.enable_divergence_analysis,
            enable_volatility_analysis=self.config.enable_volatility_analysis,
            enable_cross_timeframe_clustering=self.config.enable_cross_timeframe_clustering,
            enable_trend_strength=self.config.enable_trend_strength,
            enable_quantile_features=self.config.enable_quantile_features,
            divergence_windows=self.config.divergence_windows,
            volatility_windows=self.config.volatility_windows,
            clustering_timeframes=self.config.clustering_timeframes,
            trend_windows=self.config.trend_windows,
            quantile_levels=self.config.quantile_levels,
            enable_vectorbt=self.config.enable_vectorbt,
            vectorbt_threshold=self.config.vectorbt_threshold,
            enable_gpu=self.config.enable_gpu,
            enable_parallel=self.config.enable_parallel
        )
        self.analyzer = EnhancedInteractionAnalyzer(analyzer_config)
        
        # Initialize enhanced generators
        self.enhanced_generators = []
        if self.config.enable_enhanced_generators:
            self.enhanced_generators = create_enhanced_interaction_generators()
        
        # Initialize base config
        base_config = FeatureConfig(
            name="comprehensive_interaction_pipeline",
            category=FeatureCategory.INTERACTION,
            description="Comprehensive interaction feature pipeline with feature bank integration",
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
        
        tprint_success("🚀 Comprehensive Interaction Pipeline initialized")
        tprint_info(f"📊 Feature bank integration: {'✅' if self.config.use_feature_bank else '❌'}")
        tprint_info(f"📊 VectorBT optimization: {'✅' if self.config.enable_vectorbt else '❌'}")
        tprint_info(f"📊 Enhanced generators: {len(self.enhanced_generators)}")
    
    def generate_comprehensive_features(self, data: pd.DataFrame, 
                                      target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate comprehensive interaction features.
        
        Args:
            data: Input OHLCV data
            target: Optional target series for analysis
            
        Returns:
            DataFrame with comprehensive interaction features
        """
        tprint_info("🔍 Starting comprehensive interaction feature generation...")
        start_time = time.time()
        
        all_features = {}
        
        # 1. Enhanced interaction analysis
        if any([
            self.config.enable_divergence_analysis,
            self.config.enable_volatility_analysis,
            self.config.enable_cross_timeframe_clustering,
            self.config.enable_trend_strength,
            self.config.enable_quantile_features
        ]):
            tprint_info("📊 Running enhanced interaction analysis...")
            try:
                analysis_features = self.analyzer.analyze_feature_interactions(data, target)
                all_features.update(analysis_features.to_dict('series'))
                tprint_success(f"✅ Generated {len(analysis_features.columns)} analysis features")
            except Exception as e:
                tprint_warning(f"⚠️ Enhanced analysis failed: {e}")
        
        # 2. Enhanced interaction generators
        if self.config.enable_enhanced_generators and self.enhanced_generators:
            tprint_info("🔧 Running enhanced interaction generators...")
            generator_features = self._run_enhanced_generators(data)
            all_features.update(generator_features)
            tprint_success(f"✅ Generated {len(generator_features)} generator features")
        
        # 3. Feature selection and optimization
        if len(all_features) > self.config.feature_budget:
            tprint_info(f"📊 Selecting top {self.config.feature_budget} features from {len(all_features)}...")
            all_features = self._select_top_features(all_features, target)
        
        # Create result DataFrame
        if all_features:
            result_df = pd.DataFrame(all_features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Comprehensive feature generation completed in {execution_time:.3f}s")
        tprint_info(f"📊 Generated {len(result_df.columns)} total features")
        
        return result_df
    
    def _run_enhanced_generators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Run enhanced interaction generators."""
        generator_features = {}
        
        for generator in self.enhanced_generators:
            try:
                result = generator.generate(data)
                if result.success:
                    generator_features[generator.config.name] = result.data
                else:
                    tprint_warning(f"⚠️ Generator {generator.config.name} failed: {result.error_message}")
            except Exception as e:
                tprint_warning(f"⚠️ Generator {generator.config.name} failed: {e}")
        
        return generator_features
    
    def _select_top_features(self, features: Dict[str, pd.Series], 
                           target: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """Select top features based on importance or correlation with target."""
        if not target is not None:
            # Select features based on correlation with target
            correlations = {}
            for name, series in features.items():
                try:
                    # Calculate correlation, handling NaN values
                    valid_mask = ~(series.isna() | target.isna())
                    if valid_mask.sum() > 10:  # Need minimum data points
                        corr = series[valid_mask].corr(target[valid_mask])
                        if not np.isnan(corr):
                            correlations[name] = abs(corr)
                except Exception:
                    continue
            
            # Sort by correlation and select top features
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_names = [name for name, _ in sorted_features[:self.config.feature_budget]]
            
        else:
            # Select features based on variance (fallback)
            variances = {}
            for name, series in features.items():
                try:
                    var = series.var()
                    if not np.isnan(var) and var > 0:
                        variances[name] = var
                except Exception:
                    continue
            
            sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
            selected_names = [name for name, _ in sorted_features[:self.config.feature_budget]]
        
        # Return selected features
        selected_features = {name: features[name] for name in selected_names if name in features}
        tprint_info(f"📊 Selected {len(selected_features)} features from {len(features)} candidates")
        
        return selected_features
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive interaction features."""
        # This method is required by the base class but we use generate_comprehensive_features instead
        result_df = self.generate_comprehensive_features(data)
        
        # Return the first column as a series (for compatibility)
        if not result_df.empty:
            return result_df.iloc[:, 0]
        else:
            return pd.Series(dtype=float, index=data.index)


def create_comprehensive_pipeline(config: Optional[ComprehensiveInteractionConfig] = None) -> ComprehensiveInteractionPipeline:
    """Create a comprehensive interaction pipeline."""
    return ComprehensiveInteractionPipeline(config)


def generate_comprehensive_interaction_features(data: pd.DataFrame, 
                                              config: Optional[ComprehensiveInteractionConfig] = None,
                                              target: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Generate comprehensive interaction features using the pipeline.
    
    Args:
        data: Input OHLCV data
        config: Configuration for the pipeline
        target: Optional target series for feature selection
        
    Returns:
        DataFrame with comprehensive interaction features
    """
    pipeline = create_comprehensive_pipeline(config)
    return pipeline.generate_comprehensive_features(data, target)


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
    
    # Create target series
    target = data['close'].pct_change().shift(-1)
    
    # Test comprehensive interaction pipeline
    config = ComprehensiveInteractionConfig(
        feature_budget=50,
        enable_divergence_analysis=True,
        enable_volatility_analysis=True,
        enable_cross_timeframe_clustering=True,
        enable_trend_strength=True,
        enable_quantile_features=True,
        enable_enhanced_generators=True,
        enable_vectorbt=True
    )
    
    try:
        features = generate_comprehensive_interaction_features(data, config, target)
        print(f"Generated {len(features.columns)} comprehensive interaction features")
        print(f"Feature columns: {list(features.columns)[:10]}...")
        
        # Show feature statistics
        print(f"\nFeature Statistics:")
        print(f"Mean values: {features.mean().head()}")
        print(f"Std values: {features.std().head()}")
        print(f"NaN count: {features.isnull().sum().sum()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()