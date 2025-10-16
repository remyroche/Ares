"""
Feature Engineering for TAS

Comprehensive feature engineering system for tree architecture search including
technical indicators, regime features, and advanced feature transformations.
Now uses shared balanced feature extraction to prevent clustering imbalance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import shared balanced feature extractor
try:
    from src.training.steps.market_analysis.shared_utils.balanced_feature_extractor import (
        BalancedFeatureExtractor, BalancedFeatureConfig, create_tas_config
    )
    BALANCED_FEATURES_AVAILABLE = True
except ImportError:
    BALANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature types."""
    TECHNICAL_INDICATORS = "technical_indicators"
    PRICE_FEATURES = "price_features"
    VOLUME_FEATURES = "volume_features"
    VOLATILITY_FEATURES = "volatility_features"
    MOMENTUM_FEATURES = "momentum_features"
    TREND_FEATURES = "trend_features"
    REGIME_FEATURES = "regime_features"
    INTERACTION_FEATURES = "interaction_features"
    POLYNOMIAL_FEATURES = "polynomial_features"
    CROSS_TIMEFRAME_FEATURES = "cross_timeframe_features"

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Feature types to generate
    feature_types: List[FeatureType] = field(default_factory=lambda: [
        FeatureType.TECHNICAL_INDICATORS,
        FeatureType.PRICE_FEATURES,
        FeatureType.VOLUME_FEATURES,
        FeatureType.VOLATILITY_FEATURES,
        FeatureType.MOMENTUM_FEATURES,
        FeatureType.TREND_FEATURES
    ])
    
    # Technical indicators
    enable_technical_indicators: bool = True
    moving_averages: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21, 28])
    bollinger_periods: List[int] = field(default_factory=lambda: [20, 50])
    macd_periods: Tuple[int, int, int] = (12, 26, 9)
    
    # Price features
    enable_price_features: bool = True
    price_returns: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    price_ratios: bool = True
    price_position: bool = True
    
    # Volume features
    enable_volume_features: bool = True
    volume_returns: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    volume_ratios: bool = True
    volume_volatility: bool = True
    
    # Volatility features
    enable_volatility_features: bool = True
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    garch_features: bool = True
    volatility_of_volatility: bool = True
    
    # Momentum features
    enable_momentum_features: bool = True
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rate_of_change: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    stochastic_periods: Tuple[int, int] = (14, 3)
    
    # Trend features
    enable_trend_features: bool = True
    trend_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    trend_strength: bool = True
    trend_slope: bool = True
    
    # Regime features
    enable_regime_features: bool = True
    regime_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    regime_volatility: bool = True
    regime_trend: bool = True
    regime_volume: bool = True
    
    # Interaction features
    enable_interaction_features: bool = True
    interaction_pairs: List[Tuple[str, str]] = field(default_factory=list)
    interaction_methods: List[str] = field(default_factory=lambda: ['multiply', 'divide', 'add', 'subtract'])
    
    # Polynomial features
    enable_polynomial_features: bool = True
    polynomial_degree: int = 2
    polynomial_features: List[str] = field(default_factory=lambda: ['close', 'volume', 'volatility_20'])
    
    # Cross-timeframe features
    enable_cross_timeframe_features: bool = True
    cross_timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h', '4h', '1d'])
    cross_timeframe_features: List[str] = field(default_factory=lambda: ['close', 'volume', 'volatility'])
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_score", "chi2", "rfe"
    max_features: int = 100
    min_feature_importance: float = 0.01
    
    # Feature scaling
    enable_feature_scaling: bool = True
    scaling_method: str = "standard"  # "standard", "robust", "minmax", "quantile"
    
    # Output configuration
    save_features: bool = True
    output_directory: str = "engineered_features"
    cache_features: bool = True

@dataclass
class FeatureResult:
    """Result of feature engineering."""
    
    # Engineered features
    features: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, str]
    feature_importance: Dict[str, float]
    
    # Feature statistics
    feature_statistics: Dict[str, Any]
    feature_correlations: pd.DataFrame
    feature_redundancy: Dict[str, List[str]]
    
    # Feature engineering metadata
    engineering_steps: List[str]
    engineering_metadata: Dict[str, Any]
    feature_quality_score: float
    
    # Performance metrics
    engineering_time: float
    memory_usage: float
    features_generated: int
    features_selected: int
    
    # Metadata
    config: FeatureConfig
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class FeatureEngineer:
    """
    Comprehensive feature engineer for TAS.
    
    Provides technical indicators, regime features, interaction features,
    and advanced feature transformations for tree architecture search.
    """
    
    def __init__(self, config: FeatureConfig):
        """Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("✅ Feature Engineer initialized")
        self.logger.info(f"📊 Feature types: {[ft.value for ft in config.feature_types]}")
        self.logger.info(f"📊 Technical indicators: {config.enable_technical_indicators}")
        self.logger.info(f"📊 Regime features: {config.enable_regime_features}")
    
    def engineer_features(self, data: pd.DataFrame, regime_data: Optional[Dict[str, Any]] = None) -> FeatureResult:
        """
        Engineer features for TAS using shared balanced feature extraction.
        
        Args:
            data: Input data
            regime_data: Optional regime data
            
        Returns:
            Feature engineering result
        """
        self.logger.info("🚀 Starting balanced feature engineering")
        start_time = datetime.now()
        
        try:
            # Try to use shared balanced feature extractor first
            if BALANCED_FEATURES_AVAILABLE:
                self.logger.info("📊 Using shared balanced feature extractor")
                return self._engineer_features_balanced(data, regime_data, start_time)
            else:
                self.logger.warning("⚠️ Shared balanced features not available, using original method")
                return self._engineer_features_original(data, regime_data, start_time)
                
        except Exception as e:
            self.logger.error(f"❌ Balanced feature engineering failed: {e}")
            # Fallback to original method
            return self._engineer_features_original(data, regime_data, start_time)
    
    def _engineer_features_balanced(self, data: pd.DataFrame, regime_data: Optional[Dict[str, Any]], 
                                  start_time: datetime) -> FeatureResult:
        """Engineer features using shared balanced feature extractor."""
        try:
            # Create TAS-optimized configuration
            config = create_tas_config()
            extractor = BalancedFeatureExtractor(config)
            
            # Extract balanced features
            result = extractor.extract_balanced_features(data)
            
            if result.success:
                # Convert back to FeatureResult format
                features_df = pd.DataFrame(result.features, index=data.index, columns=result.feature_names)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return FeatureResult(
                    features=features_df,
                    feature_names=result.feature_names,
                    feature_types=result.feature_categories,
                    feature_importance={},
                    metadata={
                        'extraction_method': 'balanced_shared_extractor',
                        'balance_metrics': result.balance_metrics,
                        'processing_time': processing_time,
                        'extraction_metadata': result.extraction_metadata
                    },
                    processing_time=processing_time,
                    success=True
                )
            else:
                raise ValueError(f"Balanced feature extraction failed: {result.error_message}")
                
        except Exception as e:
            self.logger.error(f"Balanced feature engineering failed: {e}")
            raise
    
    def _engineer_features_original(self, data: pd.DataFrame, regime_data: Optional[Dict[str, Any]], 
                                  start_time: datetime) -> FeatureResult:
        """Original feature engineering method as fallback."""
        try:
            # Initialize feature engineering
            features = pd.DataFrame(index=data.index)
            feature_names = []
            feature_types = {}
            feature_importance = {}
            engineering_steps = []
            engineering_metadata = {}
            
            # Generate features based on configuration
            for feature_type in self.config.feature_types:
                if feature_type == FeatureType.TECHNICAL_INDICATORS and self.config.enable_technical_indicators:
                    tech_features = self._generate_technical_indicators(data)
                    features = pd.concat([features, tech_features], axis=1)
                    feature_names.extend(list(tech_features.columns))
                    feature_types.update({col: 'technical_indicator' for col in tech_features.columns})
                    engineering_steps.append('technical_indicators')
                
                elif feature_type == FeatureType.PRICE_FEATURES and self.config.enable_price_features:
                    price_features = self._generate_price_features(data)
                    features = pd.concat([features, price_features], axis=1)
                    feature_names.extend(list(price_features.columns))
                    feature_types.update({col: 'price_feature' for col in price_features.columns})
                    engineering_steps.append('price_features')
                
                elif feature_type == FeatureType.VOLUME_FEATURES and self.config.enable_volume_features:
                    volume_features = self._generate_volume_features(data)
                    features = pd.concat([features, volume_features], axis=1)
                    feature_names.extend(list(volume_features.columns))
                    feature_types.update({col: 'volume_feature' for col in volume_features.columns})
                    engineering_steps.append('volume_features')
                
                elif feature_type == FeatureType.VOLATILITY_FEATURES and self.config.enable_volatility_features:
                    volatility_features = self._generate_volatility_features(data)
                    features = pd.concat([features, volatility_features], axis=1)
                    feature_names.extend(list(volatility_features.columns))
                    feature_types.update({col: 'volatility_feature' for col in volatility_features.columns})
                    engineering_steps.append('volatility_features')
                
                elif feature_type == FeatureType.MOMENTUM_FEATURES and self.config.enable_momentum_features:
                    momentum_features = self._generate_momentum_features(data)
                    features = pd.concat([features, momentum_features], axis=1)
                    feature_names.extend(list(momentum_features.columns))
                    feature_types.update({col: 'momentum_feature' for col in momentum_features.columns})
                    engineering_steps.append('momentum_features')
                
                elif feature_type == FeatureType.TREND_FEATURES and self.config.enable_trend_features:
                    trend_features = self._generate_trend_features(data)
                    features = pd.concat([features, trend_features], axis=1)
                    feature_names.extend(list(trend_features.columns))
                    feature_types.update({col: 'trend_feature' for col in trend_features.columns})
                    engineering_steps.append('trend_features')
                
                elif feature_type == FeatureType.REGIME_FEATURES and self.config.enable_regime_features:
                    regime_features = self._generate_regime_features(data, regime_data)
                    features = pd.concat([features, regime_features], axis=1)
                    feature_names.extend(list(regime_features.columns))
                    feature_types.update({col: 'regime_feature' for col in regime_features.columns})
                    engineering_steps.append('regime_features')
                
                elif feature_type == FeatureType.INTERACTION_FEATURES and self.config.enable_interaction_features:
                    interaction_features = self._generate_interaction_features(features)
                    features = pd.concat([features, interaction_features], axis=1)
                    feature_names.extend(list(interaction_features.columns))
                    feature_types.update({col: 'interaction_feature' for col in interaction_features.columns})
                    engineering_steps.append('interaction_features')
                
                elif feature_type == FeatureType.POLYNOMIAL_FEATURES and self.config.enable_polynomial_features:
                    polynomial_features = self._generate_polynomial_features(features)
                    features = pd.concat([features, polynomial_features], axis=1)
                    feature_names.extend(list(polynomial_features.columns))
                    feature_types.update({col: 'polynomial_feature' for col in polynomial_features.columns})
                    engineering_steps.append('polynomial_features')
            
            # Apply feature scaling if configured
            if self.config.enable_feature_scaling:
                features = self._apply_feature_scaling(features)
                engineering_steps.append('feature_scaling')
            
            # Apply feature selection if configured
            if self.config.enable_feature_selection:
                features, feature_importance = self._apply_feature_selection(features)
                engineering_steps.append('feature_selection')
            
            # Calculate feature statistics
            feature_statistics = self._calculate_feature_statistics(features)
            feature_correlations = self._calculate_feature_correlations(features)
            feature_redundancy = self._calculate_feature_redundancy(features)
            
            # Calculate feature quality score
            feature_quality_score = self._calculate_feature_quality_score(features)
            
            # Calculate performance metrics
            engineering_time = (datetime.now() - start_time).total_seconds()
            memory_usage = features.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Create comprehensive result
            result = FeatureResult(
                # Engineered features
                features=features,
                feature_names=feature_names,
                feature_types=feature_types,
                feature_importance=feature_importance,
                
                # Feature statistics
                feature_statistics=feature_statistics,
                feature_correlations=feature_correlations,
                feature_redundancy=feature_redundancy,
                
                # Feature engineering metadata
                engineering_steps=engineering_steps,
                engineering_metadata=engineering_metadata,
                feature_quality_score=feature_quality_score,
                
                # Performance metrics
                engineering_time=engineering_time,
                memory_usage=memory_usage,
                features_generated=len(feature_names),
                features_selected=len(features.columns),
                
                # Metadata
                config=self.config
            )
            
            # Save features if configured
            if self.config.save_features:
                self._save_features(result)
            
            self.logger.info(f"✅ Feature engineering completed in {result.engineering_time:.2f}s")
            self.logger.info(f"📊 Features generated: {result.features_generated}")
            self.logger.info(f"📊 Features selected: {result.features_selected}")
            self.logger.info(f"📊 Feature quality score: {result.feature_quality_score:.3f}")
            self.logger.info(f"📊 Memory usage: {result.memory_usage:.2f} MB")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            raise
    
    def _generate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators."""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Moving averages
            for period in self.config.moving_averages:
                if 'close' in data.columns:
                    features[f'sma_{period}'] = rolling_mean(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).mean()
                    features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
            # RSI
            if 'close' in data.columns:
                for period in self.config.rsi_periods:
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            for period in self.config.bollinger_periods:
                if 'close' in data.columns:
                    sma = rolling_mean(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).mean()
                    std = rolling_std(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).std()
                    features[f'bb_upper_{period}'] = sma + (std * 2)
                    features[f'bb_lower_{period}'] = sma - (std * 2)
                    features[f'bb_middle_{period}'] = sma
                    features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
                    features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']
            
            # MACD
            if 'close' in data.columns:
                ema_12 = data['close'].ewm(span=self.config.macd_periods[0]).mean()
                ema_26 = data['close'].ewm(span=self.config.macd_periods[1]).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=self.config.macd_periods[2]).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
            
        except Exception as e:
            self.logger.warning(f"⚠️ Technical indicators generation failed: {e}")
        
        return features
    
    def _generate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Price returns
                for period in self.config.price_returns:
                    features[f'price_return_{period}'] = data['close'].pct_change(period)
                    features[f'log_return_{period}'] = np.log(data['close'] / data['close'].shift(period))
                
                # Price ratios
                if self.config.price_ratios:
                    if 'open' in data.columns:
                        features['open_close_ratio'] = data['open'] / data['close']
                    if 'high' in data.columns:
                        features['high_close_ratio'] = data['high'] / data['close']
                    if 'low' in data.columns:
                        features['low_close_ratio'] = data['low'] / data['close']
                
                # Price position within range
                if self.config.price_position and all(col in data.columns for col in ['high', 'low']):
                    features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
                    features['price_range'] = data['high'] - data['low']
                    features['price_range_ratio'] = features['price_range'] / data['close']
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price features generation failed: {e}")
        
        return features
    
    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'volume' in data.columns:
                # Volume returns
                for period in self.config.volume_returns:
                    features[f'volume_return_{period}'] = data['volume'].pct_change(period)
                    features[f'log_volume_{period}'] = np.log(data['volume'] + 1)
                
                # Volume ratios
                if self.config.volume_ratios:
                    for period in self.config.moving_averages:
                        volume_sma = rolling_mean(data["volume"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=period).mean()
                        features[f'volume_ratio_{period}'] = data['volume'] / volume_sma
                
                # Volume volatility
                if self.config.volume_volatility:
                    for period in self.config.volatility_periods:
                        features[f'volume_volatility_{period}'] = rolling_std(data["volume"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=period).std()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume features generation failed: {e}")
        
        return features
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Rolling volatility
                for period in self.config.volatility_periods:
                    features[f'volatility_{period}'] = rolling_std(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).std()
                    features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(window=period).mean()
                
                # GARCH-like features
                if self.config.garch_features:
                    returns = data['close'].pct_change()
                    features['squared_returns'] = returns ** 2
                    features['abs_returns'] = np.abs(returns)
                    
                    # Volatility of volatility
                    if self.config.volatility_of_volatility:
                        for period in self.config.volatility_periods:
                            features[f'vol_of_vol_{period}'] = features[f'volatility_{period}'].rolling(window=period).std()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility features generation failed: {e}")
        
        return features
    
    def _generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Price momentum
                for period in self.config.momentum_periods:
                    features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                
                # Rate of change
                for period in self.config.rate_of_change:
                    features[f'roc_{period}'] = data['close'].pct_change(period)
                
                # Stochastic oscillator
                if all(col in data.columns for col in ['high', 'low']):
                    k_period, d_period = self.config.stochastic_periods
                    lowest_low = data['low'].rolling(window=k_period).min()
                    highest_high = data['high'].rolling(window=k_period).max()
                    features['stoch_k'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
                    features['stoch_d'] = features['stoch_k'].rolling(window=d_period).mean()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum features generation failed: {e}")
        
        return features
    
    def _generate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Trend direction
                for period in self.config.trend_periods:
                    features[f'trend_{period}'] = np.where(data['close'] > data['close'].shift(period), 1, -1)
                
                # Trend strength
                if self.config.trend_strength:
                    for period in self.config.trend_periods:
                        features[f'trend_strength_{period}'] = np.abs(data['close'] - data['close'].shift(period))
                
                # Trend slope
                if self.config.trend_slope:
                    for period in self.config.trend_periods:
                        features[f'trend_slope_{period}'] = data['close'].rolling(window=period).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
                        )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend features generation failed: {e}")
        
        return features
    
    def _generate_regime_features(self, data: pd.DataFrame, regime_data: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Generate regime-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if regime_data is None:
                # Generate basic regime features
                if 'close' in data.columns:
                    for period in self.config.regime_periods:
                        # Volatility regime
                        if self.config.regime_volatility:
                            volatility = rolling_std(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).std()
                            features[f'regime_volatility_{period}'] = (volatility > self._vectorbt_rolling_operation(volatility, "mean", period)).astype(int)
                        
                        # Trend regime
                        if self.config.regime_trend:
                            features[f'regime_trend_{period}'] = (data['close'] > rolling_mean(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).mean()).astype(int)
                        
                        # Volume regime
                        if self.config.regime_volume and 'volume' in data.columns:
                            volume_ma = rolling_mean(data["volume"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=period).mean()
                            features[f'regime_volume_{period}'] = (data['volume'] > volume_ma).astype(int)
            else:
                # Use provided regime data
                regime_labels = regime_data.get('regime_labels', [])
                if len(regime_labels) == len(data):
                    features['regime_label'] = regime_labels
                    
                    # Generate regime-specific features
                    for regime_id in np.unique(regime_labels):
                        regime_mask = np.array(regime_labels) == regime_id
                        features[f'regime_{regime_id}'] = regime_mask.astype(int)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime features generation failed: {e}")
        
        return features
    
    def _generate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        interaction_features = pd.DataFrame(index=features.index)
        
        try:
            if not self.config.interaction_pairs:
                # Generate automatic interaction pairs
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        self.config.interaction_pairs.append((col1, col2))
            
            for col1, col2 in self.config.interaction_pairs:
                if col1 in features.columns and col2 in features.columns:
                    for method in self.config.interaction_methods:
                        if method == 'multiply':
                            interaction_features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
                        elif method == 'divide':
                            interaction_features[f'{col1}_div_{col2}'] = features[col1] / (features[col2] + 1e-8)
                        elif method == 'add':
                            interaction_features[f'{col1}_plus_{col2}'] = features[col1] + features[col2]
                        elif method == 'subtract':
                            interaction_features[f'{col1}_minus_{col2}'] = features[col1] - features[col2]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Interaction features generation failed: {e}")
        
        return interaction_features
    
    def _generate_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        polynomial_features = pd.DataFrame(index=features.index)
        
        try:
            for col in self.config.polynomial_features:
                if col in features.columns:
                    for degree in range(2, self.config.polynomial_degree + 1):
                        polynomial_features[f'{col}_power_{degree}'] = features[col] ** degree
            
        except Exception as e:
            self.logger.warning(f"⚠️ Polynomial features generation failed: {e}")
        
        return polynomial_features
    
    def _apply_feature_scaling(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling."""
        try:
            from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
            
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            
            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.config.scaling_method == "robust":
                scaler = RobustScaler()
            elif self.config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif self.config.scaling_method == "quantile":
                scaler = QuantileTransformer()
            else:
                scaler = StandardScaler()
            
            features_scaled = features.copy()
            features_scaled[numeric_cols] = scaler.fit_transform(features[numeric_cols])
            
            return features_scaled
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature scaling failed: {e}")
            return features
    
    def _apply_feature_selection(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply feature selection."""
        try:
            from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
            
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features_numeric = features[numeric_cols].fillna(0)
            
            if self.config.feature_selection_method == "mutual_info":
                scores = mutual_info_regression(features_numeric, features_numeric.iloc[:, 0])
            elif self.config.feature_selection_method == "f_score":
                scores, _ = f_regression(features_numeric, features_numeric.iloc[:, 0])
            else:
                scores = np.ones(len(numeric_cols))
            
            # Create feature importance dictionary
            feature_importance = dict(zip(numeric_cols, scores))
            
            # Select top features
            if len(numeric_cols) > self.config.max_features:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:self.config.max_features]
                selected_cols = [col for col, _ in top_features]
                features_selected = features[selected_cols]
            else:
                features_selected = features
                selected_cols = numeric_cols.tolist()
            
            # Filter by minimum importance
            if self.config.min_feature_importance > 0:
                important_cols = [col for col in selected_cols if feature_importance.get(col, 0) >= self.config.min_feature_importance]
                features_selected = features_selected[important_cols]
            
            return features_selected, feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection failed: {e}")
            return features, {}
    
    def _calculate_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics."""
        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            stats = {}
            
            for col in numeric_cols:
                stats[col] = {
                    'mean': float(features[col].mean()),
                    'std': float(features[col].std()),
                    'min': float(features[col].min()),
                    'max': float(features[col].max()),
                    'median': float(features[col].median()),
                    'skewness': float(features[col].skew()),
                    'kurtosis': float(features[col].kurtosis()),
                    'missing_ratio': float(features[col].isnull().sum() / len(features))
                }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature statistics calculation failed: {e}")
            return {}
    
    def _calculate_feature_correlations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature correlations."""
        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            return features[numeric_cols].corr()
        except Exception as e:
            self.logger.warning(f"⚠️ Feature correlations calculation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_feature_redundancy(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Calculate feature redundancy."""
        try:
            correlations = self._calculate_feature_correlations(features)
            redundancy = {}
            
            for col in correlations.columns:
                high_corr = correlations[col][abs(correlations[col]) > 0.8].index.tolist()
                high_corr = [c for c in high_corr if c != col]
                if high_corr:
                    redundancy[col] = high_corr
            
            return redundancy
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature redundancy calculation failed: {e}")
            return {}
    
    def _calculate_feature_quality_score(self, features: pd.DataFrame) -> float:
        """Calculate feature quality score."""
        try:
            # Calculate missing values ratio
            missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
            
            # Calculate feature variance
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                variance_ratio = (features[numeric_cols].var() == 0).sum() / len(numeric_cols)
            else:
                variance_ratio = 0
            
            # Calculate feature quality score
            quality_score = 1.0 - missing_ratio - variance_ratio
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature quality score calculation failed: {e}")
            return 0.0
    
    def _save_features(self, result: FeatureResult):
        """Save engineered features to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"engineered_features_{timestamp}.parquet"
            filepath = output_dir / filename
            
            result.features.to_parquet(filepath)
            
            # Save metadata
            metadata_file = output_dir / f"feature_metadata_{timestamp}.json"
            import json

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

except ImportError:
    
    cp = None
            metadata = {
                'feature_names': result.feature_names,
                'feature_types': result.feature_types,
                'feature_importance': result.feature_importance,
                'feature_quality_score': result.feature_quality_score,
                'engineering_time': result.engineering_time,
                'memory_usage': result.memory_usage,
                'features_generated': result.features_generated,
                'features_selected': result.features_selected
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"📁 Engineered features saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save features: {e}")
    
    def export_features(self, result: FeatureResult, filepath: str):
        """Export engineered features to file."""
        try:
            result.features.to_csv(filepath)
            self.logger.info(f"📁 Features exported to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to export features: {e}")