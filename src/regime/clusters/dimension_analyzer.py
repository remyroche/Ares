"""
Market Dimension Analyzer.

This module analyzes implicit market dimensions captured by existing features
from the feature engineering pipeline. Instead of generating new features,
it uses the comprehensive feature set to discover underlying market dimensions
and their economic significance for regime identification.

Key Analysis Goals:
- Discover implicit dimensions in existing features
- Assess economic significance of each dimension
- Identify which market dynamics are most relevant for trading regimes
- Use existing feature engineering infrastructure for consistency

Integration with Feature Engineering:
- Uses features from src/feature_engineering/feature_generators.py
- Leverages cross-timeframe analysis features
- Incorporates microstructure proxies from existing pipeline
- Maintains compatibility with existing optimization system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

from src.utils.logger import system_logger

# Import feature engineering components
try:
    from src.feature_engineering.feature_generators import FeatureGenerators
    from src.feature_engineering.optimized_feature_orchestrator import OptimizedFeatureOrchestrator
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"Feature engineering components not available: {e}")
    FEATURE_ENGINEERING_AVAILABLE = False


class MarketDimension(Enum):
    """Enumeration of implicit market dimensions discovered from features."""
    LIQUIDITY = "liquidity"  # Volume-based liquidity proxies, spread indicators
    MICROSTRUCTURE = "microstructure"  # Order flow proxies, trade intensity
    VOLUME = "volume"  # Volume patterns, volume-price relationships
    MOMENTUM = "momentum"  # Price momentum, trend strength, persistence
    VOLATILITY = "volatility"  # Realized volatility, volatility clustering
    CORRELATION = "correlation"  # Auto-correlation, cross-timeframe correlations


@dataclass
class DimensionMetrics:
    """Container for dimension analysis metrics."""
    dimension: MarketDimension
    importance_score: float
    stability_score: float
    predictive_power: float
    regime_discriminability: float
    feature_names: List[str]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dimension': self.dimension.value,
            'importance_score': self.importance_score,
            'stability_score': self.stability_score,
            'predictive_power': self.predictive_power,
            'regime_discriminability': self.regime_discriminability,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }


@dataclass
class DimensionAnalysisConfig:
    """Configuration for dimension analysis."""
    # Analysis parameters
    lookback_periods: List[int] = None  # [5, 10, 20, 50, 100]
    min_regime_samples: int = 100
    significance_threshold: float = 0.05
    
    # Feature engineering parameters
    volume_windows: List[int] = None  # [5, 10, 20]
    volatility_windows: List[int] = None  # [10, 20, 50]
    momentum_windows: List[int] = None  # [5, 10, 20, 50]
    
    # Analysis methods
    use_pca: bool = True
    use_mutual_information: bool = True
    use_feature_importance: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100]
        if self.volume_windows is None:
            self.volume_windows = [5, 10, 20]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50]
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20, 50]


class MarketDimensionAnalyzer:
    """
    Comprehensive analyzer for market dimensions relevant to regime identification.
    
    This class provides tools to analyze various market dimensions and their
    importance for trading regime classification and model performance.
    """
    
    def __init__(self, config: Optional[DimensionAnalysisConfig] = None):
        """
        Initialize the market dimension analyzer.
        
        Args:
            config: Configuration for dimension analysis
        """
        self.config = config or DimensionAnalysisConfig()
        self.logger = system_logger.getChild('MarketDimensionAnalyzer')
        self.dimension_results: Dict[MarketDimension, DimensionMetrics] = {}
        
    def analyze_all_dimensions(self, 
                             market_data: pd.DataFrame,
                             regime_labels: Optional[np.ndarray] = None,
                             use_existing_features: bool = True) -> Dict[MarketDimension, DimensionMetrics]:
        """
        Analyze implicit market dimensions using existing feature engineering pipeline.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Optional regime labels for supervised analysis
            use_existing_features: Whether to use existing feature engineering pipeline
            
        Returns:
            Dictionary mapping dimensions to their analysis metrics
        """
        self.logger.info("🔍 Starting implicit market dimension analysis using feature engineering pipeline")
        
        # Generate comprehensive features using existing pipeline
        if use_existing_features and FEATURE_ENGINEERING_AVAILABLE:
            feature_data = self._generate_features_from_pipeline(market_data)
        else:
            # Fallback to basic feature generation
            feature_data = self._generate_basic_features(market_data)
        
        # Discover implicit dimensions from features
        results = self._discover_implicit_dimensions(feature_data, regime_labels)
        
        # Calculate overall rankings
        self._calculate_dimension_rankings(results)
        
        self.logger.info(f"✅ Completed analysis of {len(results)} implicit dimensions")
        return results
    
    def _generate_features_from_pipeline(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using the existing feature engineering pipeline."""
        self.logger.info("🔧 Generating features using existing feature engineering pipeline")
        
        try:
            # Initialize feature generators
            feature_generators = FeatureGenerators()
            
            # Generate comprehensive feature set
            features = pd.DataFrame(index=market_data.index)
            
            # Basic OHLCV features
            features = pd.concat([features, market_data], axis=1)
            
            # Generate technical indicators
            if hasattr(feature_generators, 'generate_technical_features'):
                tech_features = feature_generators.generate_technical_features(market_data)
                features = pd.concat([features, tech_features], axis=1)
            
            # Generate volume features
            if hasattr(feature_generators, 'generate_volume_features'):
                volume_features = feature_generators.generate_volume_features(market_data)
                features = pd.concat([features, volume_features], axis=1)
            
            # Generate volatility features
            if hasattr(feature_generators, 'generate_volatility_features'):
                vol_features = feature_generators.generate_volatility_features(market_data)
                features = pd.concat([features, vol_features], axis=1)
            
            # Generate momentum features
            if hasattr(feature_generators, 'generate_momentum_features'):
                momentum_features = feature_generators.generate_momentum_features(market_data)
                features = pd.concat([features, momentum_features], axis=1)
            
            # Generate microstructure proxies
            if hasattr(feature_generators, 'generate_microstructure_features'):
                micro_features = feature_generators.generate_microstructure_features(market_data)
                features = pd.concat([features, micro_features], axis=1)
            
            self.logger.info(f"✅ Generated {len(features.columns)} features using pipeline")
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to use feature engineering pipeline: {e}")
            return self._generate_basic_features(market_data)
    
    def _generate_basic_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features as fallback."""
        self.logger.info("🔧 Generating basic features (fallback)")
        
        features = market_data.copy()
        
        # Basic derived features
        if 'close' in features.columns:
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close']).diff()
            features['volatility_20'] = features['returns'].rolling(20).std()
            features['sma_20'] = features['close'].rolling(20).mean()
            features['sma_50'] = features['close'].rolling(50).mean()
        
        if 'volume' in features.columns:
            features['volume_sma_20'] = features['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma_20']
        
        return features.fillna(method='ffill').fillna(0)
    
    def _discover_implicit_dimensions(self, 
                                    feature_data: pd.DataFrame,
                                    regime_labels: Optional[np.ndarray] = None) -> Dict[MarketDimension, DimensionMetrics]:
        """Discover implicit market dimensions from feature data."""
        results = {}
        
        # Group features by implicit dimensions
        dimension_feature_groups = self._group_features_by_dimension(feature_data)
        
        # Analyze each dimension group
        for dimension, feature_group in dimension_feature_groups.items():
            self.logger.info(f"📊 Analyzing {dimension.value} dimension ({len(feature_group.columns)} features)")
            try:
                metrics = self._analyze_dimension_group(feature_group, dimension, regime_labels)
                results[dimension] = metrics
                self.dimension_results[dimension] = metrics
            except Exception as e:
                self.logger.error(f"❌ Error analyzing {dimension.value}: {e}")
                continue
        
        return results
    
    def _group_features_by_dimension(self, feature_data: pd.DataFrame) -> Dict[MarketDimension, pd.DataFrame]:
        """Group features by their implicit market dimension."""
        dimension_groups = {dimension: pd.DataFrame(index=feature_data.index) for dimension in MarketDimension}
        
        for column in feature_data.columns:
            column_lower = column.lower()
            
            # Liquidity dimension features
            if any(keyword in column_lower for keyword in ['volume', 'spread', 'liquidity', 'depth', 'impact']):
                dimension_groups[MarketDimension.LIQUIDITY][column] = feature_data[column]
            
            # Microstructure dimension features
            elif any(keyword in column_lower for keyword in ['tick', 'trade', 'order', 'flow', 'intensity', 'micro']):
                dimension_groups[MarketDimension.MICROSTRUCTURE][column] = feature_data[column]
            
            # Volume dimension features
            elif any(keyword in column_lower for keyword in ['vol', 'obv', 'vwap', 'turnover']):
                dimension_groups[MarketDimension.VOLUME][column] = feature_data[column]
            
            # Momentum dimension features
            elif any(keyword in column_lower for keyword in ['momentum', 'rsi', 'macd', 'roc', 'trend', 'ma', 'sma', 'ema']):
                dimension_groups[MarketDimension.MOMENTUM][column] = feature_data[column]
            
            # Volatility dimension features
            elif any(keyword in column_lower for keyword in ['volatility', 'vol', 'atr', 'std', 'var', 'garch']):
                dimension_groups[MarketDimension.VOLATILITY][column] = feature_data[column]
            
            # Correlation dimension features
            elif any(keyword in column_lower for keyword in ['corr', 'correlation', 'lag', 'autocorr']):
                dimension_groups[MarketDimension.CORRELATION][column] = feature_data[column]
            
            # Default to momentum if unclear
            else:
                dimension_groups[MarketDimension.MOMENTUM][column] = feature_data[column]
        
        # Remove empty dimension groups
        dimension_groups = {dim: df for dim, df in dimension_groups.items() if not df.empty}
        
        return dimension_groups
    
    def _analyze_dimension_group(self, 
                               feature_group: pd.DataFrame,
                               dimension: MarketDimension,
                               regime_labels: Optional[np.ndarray] = None) -> DimensionMetrics:
        """Analyze a group of features representing an implicit dimension."""
        
        # Calculate dimension metrics using the feature group
        importance_score = self._calculate_importance_score(feature_group, regime_labels)
        stability_score = self._calculate_stability_score(feature_group)
        predictive_power = self._calculate_predictive_power(feature_group, regime_labels)
        regime_discriminability = self._calculate_regime_discriminability(feature_group, regime_labels)
        
        # Additional dimension-specific metrics
        dimension_metrics = self._calculate_dimension_specific_metrics(feature_group, dimension)
        
        # Add economic significance analysis
        economic_significance = self._assess_economic_significance(feature_group, dimension, regime_labels)
        dimension_metrics.update(economic_significance)
        
        return DimensionMetrics(
            dimension=dimension,
            importance_score=importance_score,
            stability_score=stability_score,
            predictive_power=predictive_power,
            regime_discriminability=regime_discriminability,
            feature_names=list(feature_group.columns),
            metrics=dimension_metrics
        )
    
    def _assess_economic_significance(self, 
                                    feature_group: pd.DataFrame,
                                    dimension: MarketDimension,
                                    regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Assess the economic significance of a market dimension."""
        significance_metrics = {}
        
        try:
            # Economic significance based on return prediction
            if 'returns' in feature_group.columns or any('return' in col.lower() for col in feature_group.columns):
                return_cols = [col for col in feature_group.columns if 'return' in col.lower()]
                if return_cols:
                    returns = feature_group[return_cols[0]]
                    
                    # Information ratio (excess return / tracking error)
                    excess_return = returns.mean()
                    tracking_error = returns.std()
                    info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                    significance_metrics['information_ratio'] = float(info_ratio)
                    
                    # Sharpe ratio proxy
                    sharpe_proxy = excess_return / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
                    significance_metrics['sharpe_ratio_proxy'] = float(sharpe_proxy)
            
            # Regime-specific economic significance
            if regime_labels is not None:
                # Calculate regime-specific feature statistics
                unique_regimes = np.unique(regime_labels)
                regime_means = {}
                regime_stds = {}
                
                for regime in unique_regimes:
                    regime_mask = regime_labels == regime
                    regime_data = feature_group[regime_mask].mean(axis=1)  # Average across features
                    regime_means[regime] = float(regime_data.mean())
                    regime_stds[regime] = float(regime_data.std())
                
                # Economic separability (difference in means relative to pooled std)
                if len(regime_means) > 1:
                    regime_values = list(regime_means.values())
                    pooled_std = np.mean(list(regime_stds.values()))
                    separability = (max(regime_values) - min(regime_values)) / pooled_std if pooled_std > 0 else 0
                    significance_metrics['regime_separability'] = float(separability)
                
                significance_metrics['regime_means'] = regime_means
                significance_metrics['regime_stds'] = regime_stds
            
            # Feature coherence within dimension
            if len(feature_group.columns) > 1:
                correlation_matrix = feature_group.corr()
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                significance_metrics['dimension_coherence'] = float(avg_correlation)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate economic significance for {dimension.value}: {e}")
        
        return significance_metrics
    
    def _analyze_single_dimension(self, 
                                market_data: pd.DataFrame,
                                dimension: MarketDimension,
                                regime_labels: Optional[np.ndarray] = None) -> DimensionMetrics:
        """Analyze a single market dimension."""
        
        # Generate features for this dimension
        features = self._generate_dimension_features(market_data, dimension)
        
        if features.empty:
            raise ValueError(f"No features generated for {dimension.value}")
        
        # Calculate dimension metrics
        importance_score = self._calculate_importance_score(features, regime_labels)
        stability_score = self._calculate_stability_score(features)
        predictive_power = self._calculate_predictive_power(features, regime_labels)
        regime_discriminability = self._calculate_regime_discriminability(features, regime_labels)
        
        # Additional dimension-specific metrics
        dimension_metrics = self._calculate_dimension_specific_metrics(features, dimension)
        
        return DimensionMetrics(
            dimension=dimension,
            importance_score=importance_score,
            stability_score=stability_score,
            predictive_power=predictive_power,
            regime_discriminability=regime_discriminability,
            feature_names=list(features.columns),
            metrics=dimension_metrics
        )
    
    def _generate_dimension_features(self, 
                                   market_data: pd.DataFrame,
                                   dimension: MarketDimension) -> pd.DataFrame:
        """Generate features for a specific market dimension."""
        
        if dimension == MarketDimension.LIQUIDITY:
            return self._generate_liquidity_features(market_data)
        elif dimension == MarketDimension.MICROSTRUCTURE:
            return self._generate_microstructure_features(market_data)
        elif dimension == MarketDimension.VOLUME:
            return self._generate_volume_features(market_data)
        elif dimension == MarketDimension.MOMENTUM:
            return self._generate_momentum_features(market_data)
        elif dimension == MarketDimension.VOLATILITY:
            return self._generate_volatility_features(market_data)
        elif dimension == MarketDimension.CORRELATION:
            return self._generate_correlation_features(market_data)
        elif dimension == MarketDimension.SEASONALITY:
            return self._generate_seasonality_features(market_data)
        elif dimension == MarketDimension.NEWS_SENTIMENT:
            return self._generate_news_sentiment_features(market_data)
        else:
            return pd.DataFrame()
    
    def _generate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate liquidity-related features."""
        features = pd.DataFrame(index=data.index)
        
        # Basic liquidity proxies
        features['bid_ask_spread_proxy'] = (data['high'] - data['low']) / data['close']
        features['price_impact_proxy'] = data['volume'] / (data['high'] - data['low'])
        
        # Volume-based liquidity measures
        for window in self.config.volume_windows:
            features[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = data['volume'].rolling(window).std()
            features[f'volume_ratio_{window}'] = data['volume'] / features[f'volume_ma_{window}']
        
        # Amihud illiquidity measure proxy
        features['amihud_illiquidity'] = abs(data['close'].pct_change()) / data['volume']
        
        # Roll impact measure proxy
        features['roll_impact'] = abs(data['close'].diff()) / np.sqrt(data['volume'])
        
        return features.dropna()
    
    def _generate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features."""
        features = pd.DataFrame(index=data.index)
        
        # Tick size effects (using price granularity as proxy)
        features['price_granularity'] = data['close'] % 0.01  # Assuming 0.01 tick size
        
        # Order flow proxies
        features['buy_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        features['sell_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
        
        # Trade intensity proxies
        features['trade_intensity'] = data['volume'] / (data['high'] - data['low'])
        
        # Intrabar volatility
        features['intrabar_volatility'] = (data['high'] - data['low']) / data['open']
        
        # Price efficiency measures
        for window in [5, 10, 20]:
            features[f'price_efficiency_{window}'] = data['close'].rolling(window).std() / data['close'].rolling(window).mean()
        
        return features.dropna()
    
    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-related features."""
        features = pd.DataFrame(index=data.index)
        
        # Basic volume features
        for window in self.config.volume_windows:
            features[f'volume_sma_{window}'] = data['volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = data['volume'].rolling(window).std()
            features[f'volume_zscore_{window}'] = (data['volume'] - features[f'volume_sma_{window}']) / features[f'volume_std_{window}']
        
        # Volume-price relationships
        features['volume_price_correlation'] = data['volume'].rolling(20).corr(data['close'])
        features['volume_return_correlation'] = data['volume'].rolling(20).corr(data['close'].pct_change())
        
        # On-balance volume proxy
        features['obv'] = (data['volume'] * np.sign(data['close'].diff())).cumsum()
        
        # Volume rate of change
        for window in [5, 10, 20]:
            features[f'volume_roc_{window}'] = data['volume'].pct_change(window)
        
        # Volume momentum
        features['volume_momentum'] = data['volume'].rolling(10).mean() / data['volume'].rolling(20).mean()
        
        return features.dropna()
    
    def _generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-related features."""
        features = pd.DataFrame(index=data.index)
        
        # Price momentum
        for window in self.config.momentum_windows:
            features[f'price_momentum_{window}'] = data['close'].pct_change(window)
            features[f'price_acceleration_{window}'] = features[f'price_momentum_{window}'].diff()
        
        # Moving average convergence/divergence
        for short, long in [(5, 10), (10, 20), (20, 50)]:
            ma_short = data['close'].rolling(short).mean()
            ma_long = data['close'].rolling(long).mean()
            features[f'macd_{short}_{long}'] = ma_short - ma_long
            features[f'macd_signal_{short}_{long}'] = features[f'macd_{short}_{long}'].rolling(9).mean()
        
        # RSI-like momentum indicators
        for window in [14, 21, 50]:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window).mean()
            rs = gain / loss
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Trend strength
        for window in [10, 20, 50]:
            features[f'trend_strength_{window}'] = abs(data['close'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        return features.dropna()
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-related features."""
        features = pd.DataFrame(index=data.index)
        
        # Realized volatility
        returns = data['close'].pct_change()
        for window in self.config.volatility_windows:
            features[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        
        # Garman-Klass volatility
        features['gk_volatility'] = np.sqrt(
            0.5 * (np.log(data['high'] / data['low'])) ** 2 - 
            (2 * np.log(2) - 1) * (np.log(data['close'] / data['open'])) ** 2
        )
        
        # Volatility clustering
        for window in [10, 20, 50]:
            vol = returns.rolling(window).std()
            features[f'vol_clustering_{window}'] = vol.rolling(window).std()
        
        # Volatility momentum
        for short, long in [(10, 20), (20, 50)]:
            vol_short = returns.rolling(short).std()
            vol_long = returns.rolling(long).std()
            features[f'vol_momentum_{short}_{long}'] = vol_short / vol_long
        
        # Volatility mean reversion
        for window in [20, 50]:
            vol = returns.rolling(window).std()
            vol_ma = vol.rolling(window).mean()
            features[f'vol_mean_reversion_{window}'] = (vol - vol_ma) / vol_ma
        
        return features.dropna()
    
    def _generate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate correlation-related features (requires multiple assets - placeholder)."""
        features = pd.DataFrame(index=data.index)
        
        # Auto-correlation features
        returns = data['close'].pct_change()
        for lag in [1, 5, 10, 20]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(50).apply(lambda x: x.autocorr(lag))
        
        # Volume-price correlation
        for window in [10, 20, 50]:
            features[f'vol_price_corr_{window}'] = data['volume'].rolling(window).corr(data['close'])
        
        return features.dropna()
    
    def _generate_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate seasonality-related features."""
        features = pd.DataFrame(index=data.index)
        
        # Time-based features (assuming datetime index)
        if hasattr(data.index, 'hour'):
            features['hour_of_day'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month_of_year'] = data.index.month
        
        # Cyclical encoding
        if 'hour_of_day' in features.columns:
            features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
        
        if 'day_of_week' in features.columns:
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features.dropna()
    
    def _generate_news_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate news sentiment features (placeholder - requires external data)."""
        features = pd.DataFrame(index=data.index)
        
        # Placeholder features - in practice, these would come from news sentiment analysis
        features['sentiment_proxy'] = np.random.randn(len(data)) * 0.1  # Random placeholder
        features['news_volume_proxy'] = np.random.poisson(5, len(data))  # Random placeholder
        
        return features.dropna()
    
    def _calculate_importance_score(self, 
                                  features: pd.DataFrame,
                                  regime_labels: Optional[np.ndarray] = None) -> float:
        """Calculate feature importance score for regime discrimination."""
        if regime_labels is None:
            # Use unsupervised importance (variance-based)
            return float(features.var().mean())
        
        # Use supervised importance (mutual information or feature importance)
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features.fillna(0))
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(features_scaled, regime_labels[:len(features)])
            return float(np.mean(mi_scores))
            
        except Exception as e:
            self.logger.warning(f"Could not calculate supervised importance: {e}")
            return float(features.var().mean())
    
    def _calculate_stability_score(self, features: pd.DataFrame) -> float:
        """Calculate stability score of features over time."""
        # Split data into chunks and calculate correlation between chunks
        chunk_size = len(features) // 4
        if chunk_size < 10:
            return 0.5  # Default for small datasets
        
        chunk1 = features.iloc[:chunk_size]
        chunk2 = features.iloc[chunk_size:2*chunk_size]
        
        # Calculate correlation between feature statistics
        try:
            corr_means = np.corrcoef(chunk1.mean(), chunk2.mean())[0, 1]
            corr_stds = np.corrcoef(chunk1.std(), chunk2.std())[0, 1]
            
            stability = (corr_means + corr_stds) / 2
            return float(np.nan_to_num(stability, 0.5))
        except:
            return 0.5
    
    def _calculate_predictive_power(self, 
                                  features: pd.DataFrame,
                                  regime_labels: Optional[np.ndarray] = None) -> float:
        """Calculate predictive power of features."""
        if regime_labels is None:
            # Use autocorrelation as proxy for predictive power
            returns = features.pct_change().mean(axis=1)
            autocorr = returns.autocorr(1) if len(returns) > 1 else 0.0
            return float(abs(autocorr))
        
        # Use cross-validation score
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            scaler = StandardScaler()
            X = scaler.fit_transform(features.fillna(0))
            y = regime_labels[:len(features)]
            
            # Quick random forest evaluation
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
            
            return float(np.mean(scores))
            
        except Exception as e:
            self.logger.warning(f"Could not calculate predictive power: {e}")
            return 0.5
    
    def _calculate_regime_discriminability(self, 
                                         features: pd.DataFrame,
                                         regime_labels: Optional[np.ndarray] = None) -> float:
        """Calculate how well features discriminate between regimes."""
        if regime_labels is None:
            # Use clustering-based discriminability
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                X = scaler.fit_transform(features.fillna(0))
                
                # Try different numbers of clusters
                best_score = -1
                for n_clusters in [2, 3, 4, 5]:
                    if len(X) < n_clusters * 2:
                        continue
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
                    score = silhouette_score(X, labels)
                    best_score = max(best_score, score)
                
                return float(max(0, best_score))
                
            except Exception as e:
                self.logger.warning(f"Could not calculate unsupervised discriminability: {e}")
                return 0.5
        
        # Use supervised discriminability
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X = scaler.fit_transform(features.fillna(0))
            y = regime_labels[:len(features)]
            
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            
            # Use explained variance ratio as discriminability score
            explained_variance = np.sum(lda.explained_variance_ratio_)
            return float(min(1.0, explained_variance))
            
        except Exception as e:
            self.logger.warning(f"Could not calculate supervised discriminability: {e}")
            return 0.5
    
    def _calculate_dimension_specific_metrics(self, 
                                            features: pd.DataFrame,
                                            dimension: MarketDimension) -> Dict[str, float]:
        """Calculate dimension-specific metrics."""
        metrics = {}
        
        # Common metrics for all dimensions
        metrics['feature_count'] = len(features.columns)
        metrics['data_coverage'] = (1 - features.isnull().sum().sum() / features.size)
        metrics['feature_correlation'] = float(features.corr().abs().mean().mean())
        
        # Dimension-specific metrics
        if dimension == MarketDimension.VOLATILITY:
            # Volatility clustering measure
            if 'realized_vol_20' in features.columns:
                vol_series = features['realized_vol_20'].dropna()
                if len(vol_series) > 1:
                    metrics['volatility_clustering'] = float(vol_series.autocorr(1))
        
        elif dimension == MarketDimension.MOMENTUM:
            # Momentum persistence
            momentum_cols = [col for col in features.columns if 'momentum' in col]
            if momentum_cols:
                momentum_data = features[momentum_cols].mean(axis=1)
                if len(momentum_data) > 1:
                    metrics['momentum_persistence'] = float(abs(momentum_data.autocorr(1)))
        
        elif dimension == MarketDimension.VOLUME:
            # Volume predictability
            if 'volume_sma_10' in features.columns:
                vol_series = features['volume_sma_10'].dropna()
                if len(vol_series) > 1:
                    metrics['volume_predictability'] = float(abs(vol_series.autocorr(1)))
        
        return metrics
    
    def _calculate_dimension_rankings(self, results: Dict[MarketDimension, DimensionMetrics]):
        """Calculate overall rankings for dimensions."""
        # Calculate composite scores
        composite_scores = {}
        for dimension, metrics in results.items():
            composite_score = (
                metrics.importance_score * 0.3 +
                metrics.stability_score * 0.2 +
                metrics.predictive_power * 0.3 +
                metrics.regime_discriminability * 0.2
            )
            composite_scores[dimension] = composite_score
        
        # Add ranking to metrics
        sorted_dimensions = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (dimension, score) in enumerate(sorted_dimensions, 1):
            results[dimension].metrics['composite_score'] = score
            results[dimension].metrics['ranking'] = rank
    
    def get_top_dimensions(self, n: int = 3) -> List[Tuple[MarketDimension, DimensionMetrics]]:
        """Get top N dimensions by composite score."""
        if not self.dimension_results:
            return []
        
        sorted_results = sorted(
            self.dimension_results.items(),
            key=lambda x: x[1].metrics.get('composite_score', 0),
            reverse=True
        )
        
        return sorted_results[:n]
    
    def save_analysis_results(self, filepath: str):
        """Save analysis results to file."""
        results_dict = {
            dimension.value: metrics.to_dict() 
            for dimension, metrics in self.dimension_results.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"💾 Saved dimension analysis results to {filepath}")
    
    def load_analysis_results(self, filepath: str):
        """Load analysis results from file."""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.dimension_results = {}
        for dimension_name, metrics_dict in results_dict.items():
            dimension = MarketDimension(dimension_name)
            metrics = DimensionMetrics(
                dimension=dimension,
                importance_score=metrics_dict['importance_score'],
                stability_score=metrics_dict['stability_score'],
                predictive_power=metrics_dict['predictive_power'],
                regime_discriminability=metrics_dict['regime_discriminability'],
                feature_names=metrics_dict['feature_names'],
                metrics=metrics_dict['metrics']
            )
            self.dimension_results[dimension] = metrics
        
        self.logger.info(f"📂 Loaded dimension analysis results from {filepath}")
    
    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report."""
        if not self.dimension_results:
            return "No analysis results available. Run analyze_all_dimensions() first."
        
        report = []
        report.append("# Market Dimension Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Top dimensions
        top_dimensions = self.get_top_dimensions(5)
        report.append("## Top Performing Dimensions")
        report.append("")
        
        for rank, (dimension, metrics) in enumerate(top_dimensions, 1):
            report.append(f"{rank}. **{dimension.value.upper()}**")
            report.append(f"   - Composite Score: {metrics.metrics.get('composite_score', 0):.3f}")
            report.append(f"   - Importance: {metrics.importance_score:.3f}")
            report.append(f"   - Stability: {metrics.stability_score:.3f}")
            report.append(f"   - Predictive Power: {metrics.predictive_power:.3f}")
            report.append(f"   - Regime Discriminability: {metrics.regime_discriminability:.3f}")
            report.append(f"   - Feature Count: {len(metrics.feature_names)}")
            report.append("")
        
        # Detailed analysis
        report.append("## Detailed Dimension Analysis")
        report.append("")
        
        for dimension, metrics in self.dimension_results.items():
            report.append(f"### {dimension.value.upper()}")
            report.append(f"- **Features Generated**: {len(metrics.feature_names)}")
            report.append(f"- **Key Features**: {', '.join(metrics.feature_names[:5])}")
            if len(metrics.feature_names) > 5:
                report.append(f"  (and {len(metrics.feature_names) - 5} more)")
            report.append("")
            
            # Metrics
            report.append("**Performance Metrics:**")
            for key, value in metrics.metrics.items():
                if isinstance(value, float):
                    report.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    report.append(f"- {key.replace('_', ' ').title()}: {value}")
            report.append("")
        
        return "\n".join(report)