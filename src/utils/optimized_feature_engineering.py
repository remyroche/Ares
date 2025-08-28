#!/usr/bin/env python3
"""
Optimized Feature Engineering Module

This module provides a comprehensive, non-redundant feature engineering system
that ensures complete, functional feature generation without redundancy.
Integrates with centralized S/R logic and HMM regime management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
    handle_errors,
    validate_data_structure,
    monitor_feature_engineering,
    memory_efficient
)
from src.utils.centralized_sr_logic import CentralizedSRAnalyzer
from src.utils.enhanced_hmm_regime_manager import EnhancedHMMRegimeManager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class FeatureCategory(Enum):
    """Feature categories for organization."""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    TECHNICAL = "technical"
    SUPPORT_RESISTANCE = "support_resistance"
    REGIME = "regime"
    INTERACTION = "interaction"
    WAVELET = "wavelet"
    STATISTICAL = "statistical"


@dataclass
class FeatureInfo:
    """Feature information structure."""
    name: str
    category: FeatureCategory
    description: str
    dependencies: List[str]
    is_redundant: bool = False
    quality_score: float = 1.0


class OptimizedFeatureEngineering:
    """
    Optimized feature engineering system that eliminates redundancy and ensures
    complete, functional feature generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimizedFeatureEngineering")
        
        # Feature configuration
        self.enable_sr_features = config.get("enable_sr_features", True)
        self.enable_regime_features = config.get("enable_regime_features", True)
        self.enable_wavelet_features = config.get("enable_wavelet_features", True)
        self.enable_interaction_features = config.get("enable_interaction_features", True)
        
        # Quality thresholds
        self.min_feature_quality = config.get("min_feature_quality", 0.3)
        self.max_correlation_threshold = config.get("max_correlation_threshold", 0.95)
        
        # Initialize components
        self.sr_analyzer = CentralizedSRAnalyzer(config) if self.enable_sr_features else None
        self.regime_manager = EnhancedHMMRegimeManager(config) if self.enable_regime_features else None
        
        # Feature registry
        self.feature_registry: Dict[str, FeatureInfo] = {}
        self.generated_features: Dict[str, pd.Series] = {}
        
        # Cache
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._quality_cache: Dict[str, float] = {}
        
        # Initialize feature registry
        self._initialize_feature_registry()
    
    def _initialize_feature_registry(self) -> None:
        """Initialize the feature registry with all available features."""
        try:
            # Price-based features
            self._register_feature("log_returns", FeatureCategory.PRICE, "Log returns", [])
            self._register_feature("price_momentum", FeatureCategory.PRICE, "Price momentum", [])
            self._register_feature("price_acceleration", FeatureCategory.PRICE, "Price acceleration", ["price_momentum"])
            self._register_feature("price_velocity", FeatureCategory.PRICE, "Price velocity", ["log_returns"])
            
            # Volume features
            self._register_feature("volume_ratio", FeatureCategory.VOLUME, "Volume ratio", [])
            self._register_feature("volume_momentum", FeatureCategory.VOLUME, "Volume momentum", [])
            self._register_feature("volume_sma_ratio", FeatureCategory.VOLUME, "Volume SMA ratio", [])
            self._register_feature("volume_price_trend", FeatureCategory.VOLUME, "Volume-price trend", ["volume_ratio", "price_momentum"])
            
            # Volatility features
            self._register_feature("volatility_20", FeatureCategory.VOLATILITY, "20-period volatility", ["log_returns"])
            self._register_feature("volatility_50", FeatureCategory.VOLATILITY, "50-period volatility", ["log_returns"])
            self._register_feature("volatility_ratio", FeatureCategory.VOLATILITY, "Volatility ratio", ["volatility_20", "volatility_50"])
            self._register_feature("realized_volatility", FeatureCategory.VOLATILITY, "Realized volatility", ["log_returns"])
            
            # Momentum features
            self._register_feature("rsi", FeatureCategory.MOMENTUM, "Relative Strength Index", ["close"])
            self._register_feature("macd", FeatureCategory.MOMENTUM, "MACD", ["close"])
            self._register_feature("macd_signal", FeatureCategory.MOMENTUM, "MACD signal", ["macd"])
            self._register_feature("macd_histogram", FeatureCategory.MOMENTUM, "MACD histogram", ["macd", "macd_signal"])
            self._register_feature("stochastic_k", FeatureCategory.MOMENTUM, "Stochastic %K", ["high", "low", "close"])
            self._register_feature("stochastic_d", FeatureCategory.MOMENTUM, "Stochastic %D", ["stochastic_k"])
            
            # Technical features
            self._register_feature("bb_position", FeatureCategory.TECHNICAL, "Bollinger Band position", ["close"])
            self._register_feature("bb_width", FeatureCategory.TECHNICAL, "Bollinger Band width", ["close"])
            self._register_feature("bb_squeeze", FeatureCategory.TECHNICAL, "Bollinger Band squeeze", ["bb_width"])
            self._register_feature("atr", FeatureCategory.TECHNICAL, "Average True Range", ["high", "low", "close"])
            self._register_feature("adx", FeatureCategory.TECHNICAL, "Average Directional Index", ["high", "low", "close"])
            self._register_feature("cci", FeatureCategory.TECHNICAL, "Commodity Channel Index", ["high", "low", "close"])
            
            # Support/Resistance features
            if self.enable_sr_features:
                self._register_feature("sr_distance", FeatureCategory.SUPPORT_RESISTANCE, "Distance to S/R levels", [])
                self._register_feature("sr_strength", FeatureCategory.SUPPORT_RESISTANCE, "S/R strength", [])
                self._register_feature("sr_type", FeatureCategory.SUPPORT_RESISTANCE, "S/R type", [])
                self._register_feature("sr_breakout", FeatureCategory.SUPPORT_RESISTANCE, "S/R breakout", ["sr_distance"])
            
            # Regime features
            if self.enable_regime_features:
                self._register_feature("regime_id", FeatureCategory.REGIME, "Regime ID", [])
                self._register_feature("regime_confidence", FeatureCategory.REGIME, "Regime confidence", [])
                self._register_feature("regime_duration", FeatureCategory.REGIME, "Regime duration", [])
                self._register_feature("regime_transition_prob", FeatureCategory.REGIME, "Regime transition probability", [])
            
            # Interaction features
            if self.enable_interaction_features:
                self._register_feature("price_volume_interaction", FeatureCategory.INTERACTION, "Price-volume interaction", ["price_momentum", "volume_ratio"])
                self._register_feature("volatility_momentum_interaction", FeatureCategory.INTERACTION, "Volatility-momentum interaction", ["volatility_20", "rsi"])
                self._register_feature("trend_strength", FeatureCategory.INTERACTION, "Trend strength", ["price_momentum", "volatility_20"])
                self._register_feature("volume_trend_alignment", FeatureCategory.INTERACTION, "Volume-trend alignment", ["price_momentum", "volume_momentum"])
            
            # Wavelet features
            if self.enable_wavelet_features:
                self._register_feature("wavelet_approximation", FeatureCategory.WAVELET, "Wavelet approximation", ["close"])
                self._register_feature("wavelet_detail_1", FeatureCategory.WAVELET, "Wavelet detail level 1", ["close"])
                self._register_feature("wavelet_detail_2", FeatureCategory.WAVELET, "Wavelet detail level 2", ["close"])
                self._register_feature("wavelet_detail_3", FeatureCategory.WAVELET, "Wavelet detail level 3", ["close"])
            
            # Statistical features
            self._register_feature("price_zscore", FeatureCategory.STATISTICAL, "Price Z-score", ["close"])
            self._register_feature("volume_zscore", FeatureCategory.STATISTICAL, "Volume Z-score", ["volume"])
            self._register_feature("returns_skewness", FeatureCategory.STATISTICAL, "Returns skewness", ["log_returns"])
            self._register_feature("returns_kurtosis", FeatureCategory.STATISTICAL, "Returns kurtosis", ["log_returns"])
            
            self.logger.info(f"📋 Feature registry initialized with {len(self.feature_registry)} features")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing feature registry: {e}")
    
    def _register_feature(self, name: str, category: FeatureCategory, description: str, dependencies: List[str]) -> None:
        """Register a feature in the registry."""
        self.feature_registry[name] = FeatureInfo(
            name=name,
            category=category,
            description=description,
            dependencies=dependencies
        )
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="optimized_feature_engineering"
    )
    @validate_data_structure(required_columns=["open", "high", "low", "close", "volume"])
    @monitor_feature_engineering
    @memory_efficient
    async def generate_features(
        self, 
        df: pd.DataFrame,
        feature_categories: Optional[List[FeatureCategory]] = None,
        include_sr_analysis: bool = True,
        include_regime_analysis: bool = True
    ) -> pd.DataFrame:
        """
        Generate optimized features with redundancy elimination and quality control.
        
        Args:
            df: OHLCV DataFrame
            feature_categories: Specific feature categories to generate
            include_sr_analysis: Include support/resistance analysis
            include_regime_analysis: Include regime analysis
            
        Returns:
            DataFrame with generated features
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return pd.DataFrame()
        
        try:
            self.logger.info("🚀 Starting optimized feature generation...")
            
            # Generate cache key
            cache_key = self._generate_cache_key(df, feature_categories)
            
            # Check cache
            if cache_key in self._feature_cache:
                self.logger.info("📋 Using cached features")
                return self._feature_cache[cache_key]
            
            # Initialize features DataFrame
            features_df = pd.DataFrame(index=df.index)
            
            # Determine which features to generate
            features_to_generate = self._determine_features_to_generate(feature_categories)
            
            # Generate features by category
            for category in FeatureCategory:
                if category in features_to_generate:
                    self.logger.info(f"📊 Generating {category.value} features...")
                    category_features = await self._generate_category_features(df, category)
                    if not category_features.empty:
                        features_df = pd.concat([features_df, category_features], axis=1)
            
            # Generate support/resistance features
            if include_sr_analysis and self.enable_sr_features and self.sr_analyzer:
                self.logger.info("📊 Generating S/R features...")
                sr_features = await self._generate_sr_features(df)
                if not sr_features.empty:
                    features_df = pd.concat([features_df, sr_features], axis=1)
            
            # Generate regime features
            if include_regime_analysis and self.enable_regime_features and self.regime_manager:
                self.logger.info("📊 Generating regime features...")
                regime_features = await self._generate_regime_features(df)
                if not regime_features.empty:
                    features_df = pd.concat([features_df, regime_features], axis=1)
            
            # Quality control and redundancy elimination
            features_df = self._apply_quality_control(features_df)
            features_df = self._eliminate_redundancy(features_df)
            
            # Final validation
            features_df = self._validate_features(features_df)
            
            # Cache results
            self._feature_cache[cache_key] = features_df
            
            self.logger.info(f"✅ Feature generation completed. Shape: {features_df.shape}")
            self.logger.info(f"📊 Generated {len(features_df.columns)} features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in feature generation: {e}")
            return pd.DataFrame()
    
    def _generate_cache_key(self, df: pd.DataFrame, feature_categories: Optional[List[FeatureCategory]]) -> str:
        """Generate cache key for DataFrame and feature categories."""
        try:
            # Use DataFrame shape and last few values for cache key
            key_data = f"{df.shape}_{df['close'].iloc[-1]:.6f}_{len(df)}"
            if feature_categories:
                key_data += f"_{'_'.join([cat.value for cat in feature_categories])}"
            return str(hash(key_data))
        except Exception:
            return str(hash(str(df.shape)))
    
    def _determine_features_to_generate(self, feature_categories: Optional[List[FeatureCategory]]) -> List[FeatureCategory]:
        """Determine which feature categories to generate."""
        if feature_categories is None:
            return list(FeatureCategory)
        return feature_categories
    
    async def _generate_category_features(self, df: pd.DataFrame, category: FeatureCategory) -> pd.DataFrame:
        """Generate features for a specific category."""
        try:
            features = pd.DataFrame(index=df.index)
            
            if category == FeatureCategory.PRICE:
                features = self._generate_price_features(df)
            elif category == FeatureCategory.VOLUME:
                features = self._generate_volume_features(df)
            elif category == FeatureCategory.VOLATILITY:
                features = self._generate_volatility_features(df)
            elif category == FeatureCategory.MOMENTUM:
                features = self._generate_momentum_features(df)
            elif category == FeatureCategory.TECHNICAL:
                features = self._generate_technical_features(df)
            elif category == FeatureCategory.INTERACTION:
                features = self._generate_interaction_features(df)
            elif category == FeatureCategory.WAVELET:
                features = self._generate_wavelet_features(df)
            elif category == FeatureCategory.STATISTICAL:
                features = self._generate_statistical_features(df)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error generating {category.value} features: {e}")
            return pd.DataFrame()
    
    def _generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Log returns
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price momentum
            features['price_momentum'] = df['close'].pct_change(5)
            features['price_momentum_10'] = df['close'].pct_change(10)
            features['price_momentum_20'] = df['close'].pct_change(20)
            
            # Price acceleration
            features['price_acceleration'] = features['price_momentum'].diff()
            
            # Price velocity
            features['price_velocity'] = features['log_returns'].rolling(5).mean()
            
            # Price ranges
            features['price_range'] = (df['high'] - df['low']) / df['close']
            features['price_range_5'] = features['price_range'].rolling(5).mean()
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating price features: {e}")
            return pd.DataFrame()
    
    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Volume ratios
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
            features['volume_ratio_50'] = df['volume'] / df['volume'].rolling(50).mean()
            
            # Volume momentum
            features['volume_momentum'] = df['volume'].pct_change(5)
            features['volume_momentum_10'] = df['volume'].pct_change(10)
            
            # Volume SMA ratios
            features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Volume-price trend
            features['volume_price_trend'] = features['volume_ratio'] * features['price_momentum']
            
            # Volume volatility
            features['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
            
            return features.fillna(1)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating volume features: {e}")
            return pd.DataFrame()
    
    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Calculate log returns for volatility
            log_returns = np.log(df['close'] / df['close'].shift(1))
            
            # Rolling volatilities
            features['volatility_20'] = log_returns.rolling(20).std()
            features['volatility_50'] = log_returns.rolling(50).std()
            features['volatility_100'] = log_returns.rolling(100).std()
            
            # Volatility ratios
            features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']
            features['volatility_ratio_long'] = features['volatility_20'] / features['volatility_100']
            
            # Realized volatility
            features['realized_volatility'] = log_returns.rolling(20).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            # Volatility of volatility
            features['vol_of_vol'] = features['volatility_20'].rolling(20).std()
            
            # Parkinson volatility
            features['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(df['high'] / df['low']) ** 2).rolling(20).mean())
            )
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating volatility features: {e}")
            return pd.DataFrame()
    
    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Stochastic
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            features['stochastic_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            features['stochastic_d'] = features['stochastic_k'].rolling(3).mean()
            
            # Williams %R
            features['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
            
            # Rate of Change
            features['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            return features.fillna(50)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating momentum features: {e}")
            return pd.DataFrame()
    
    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            features['bb_position'] = (df['close'] - bb_middle) / bb_std
            features['bb_width'] = bb_std / bb_middle
            features['bb_squeeze'] = features['bb_width'].rolling(20).mean() / features['bb_width']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr'] = tr.rolling(14).mean()
            
            # ADX
            features['adx'] = self._calculate_adx(df)
            
            # CCI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            features['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Parabolic SAR
            features['psar'] = self._calculate_psar(df)
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating technical features: {e}")
            return pd.DataFrame()
    
    def _generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Get base features first
            price_features = self._generate_price_features(df)
            volume_features = self._generate_volume_features(df)
            volatility_features = self._generate_volatility_features(df)
            momentum_features = self._generate_momentum_features(df)
            
            # Price-volume interactions
            features['price_volume_interaction'] = price_features['price_momentum'] * volume_features['volume_ratio']
            features['price_volume_correlation'] = price_features['log_returns'].rolling(20).corr(volume_features['volume_ratio'])
            
            # Volatility-momentum interactions
            features['volatility_momentum_interaction'] = volatility_features['volatility_20'] * momentum_features['rsi']
            features['volatility_rsi_alignment'] = np.sign(volatility_features['volatility_20']) * np.sign(momentum_features['rsi'] - 50)
            
            # Trend strength
            features['trend_strength'] = abs(price_features['price_momentum']) / volatility_features['volatility_20']
            features['trend_strength_10'] = abs(price_features['price_momentum_10']) / volatility_features['volatility_20']
            
            # Volume-trend alignment
            features['volume_trend_alignment'] = np.sign(price_features['price_momentum']) * np.sign(volume_features['volume_momentum'])
            features['volume_trend_strength'] = features['volume_trend_alignment'] * abs(price_features['price_momentum'])
            
            # Momentum-volatility regime
            features['momentum_volatility_regime'] = np.where(
                (momentum_features['rsi'] > 70) & (volatility_features['volatility_20'] > volatility_features['volatility_20'].quantile(0.8)),
                'high_momentum_high_vol',
                np.where(
                    (momentum_features['rsi'] < 30) & (volatility_features['volatility_20'] > volatility_features['volatility_20'].quantile(0.8)),
                    'low_momentum_high_vol',
                    'normal'
                )
            )
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating interaction features: {e}")
            return pd.DataFrame()
    
    def _generate_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate wavelet-based features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Try to import pywt
            try:
                import pywt
            except ImportError:
                self.logger.warning("⚠️ PyWavelets not available, skipping wavelet features")
                return features
            
            # Use log returns for wavelet analysis
            log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            
            # Apply wavelet transform
            try:
                # Use Daubechies wavelet
                wavelet = 'db4'
                coeffs = pywt.wavedec(log_returns.values, wavelet, level=3)
                
                # Extract approximation and detail coefficients
                if len(coeffs) >= 4:
                    # Approximation coefficients (trend)
                    approx = coeffs[0]
                    # Detail coefficients (noise at different scales)
                    detail1 = coeffs[1]
                    detail2 = coeffs[2]
                    detail3 = coeffs[3]
                    
                    # Pad to match original length
                    features['wavelet_approximation'] = np.pad(approx, (0, len(log_returns) - len(approx)), 'edge')
                    features['wavelet_detail_1'] = np.pad(detail1, (0, len(log_returns) - len(detail1)), 'edge')
                    features['wavelet_detail_2'] = np.pad(detail2, (0, len(log_returns) - len(detail2)), 'edge')
                    features['wavelet_detail_3'] = np.pad(detail3, (0, len(log_returns) - len(detail3)), 'edge')
                    
                    # Wavelet energy features
                    features['wavelet_energy_1'] = features['wavelet_detail_1'] ** 2
                    features['wavelet_energy_2'] = features['wavelet_detail_2'] ** 2
                    features['wavelet_energy_3'] = features['wavelet_detail_3'] ** 2
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error in wavelet analysis: {e}")
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating wavelet features: {e}")
            return pd.DataFrame()
    
    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Z-scores
            features['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            features['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            
            # Returns statistics
            log_returns = np.log(df['close'] / df['close'].shift(1))
            features['returns_skewness'] = log_returns.rolling(20).skew()
            features['returns_kurtosis'] = log_returns.rolling(20).kurt()
            
            # Price statistics
            features['price_skewness'] = df['close'].rolling(20).skew()
            features['price_kurtosis'] = df['close'].rolling(20).kurt()
            
            # Volume statistics
            features['volume_skewness'] = df['volume'].rolling(20).skew()
            features['volume_kurtosis'] = df['volume'].rolling(20).kurt()
            
            # Percentile ranks
            features['price_percentile'] = df['close'].rolling(20).rank(pct=True)
            features['volume_percentile'] = df['volume'].rolling(20).rank(pct=True)
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating statistical features: {e}")
            return pd.DataFrame()
    
    async def _generate_sr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate support/resistance features using centralized S/R logic."""
        try:
            features = pd.DataFrame(index=df.index)
            
            if not self.sr_analyzer:
                return features
            
            # Analyze S/R levels
            sr_analysis = self.sr_analyzer.analyze_sr_levels(df, df['close'].iloc[-1])
            
            if sr_analysis.get("error"):
                self.logger.warning(f"⚠️ S/R analysis error: {sr_analysis['error']}")
                return features
            
            # Extract S/R features
            supports = sr_analysis.get("supports", [])
            resistances = sr_analysis.get("resistances", [])
            
            current_price = df['close'].iloc[-1]
            
            # Distance to nearest support/resistance
            if supports:
                nearest_support = max([s['price'] for s in supports if s['price'] < current_price], default=0)
                features['sr_distance_support'] = (current_price - nearest_support) / current_price if nearest_support > 0 else 0
                features['sr_strength_support'] = max([s['strength'] for s in supports if s['price'] < current_price], default=0)
            else:
                features['sr_distance_support'] = 0
                features['sr_strength_support'] = 0
            
            if resistances:
                nearest_resistance = min([r['price'] for r in resistances if r['price'] > current_price], default=float('inf'))
                features['sr_distance_resistance'] = (nearest_resistance - current_price) / current_price if nearest_resistance < float('inf') else 0
                features['sr_strength_resistance'] = max([r['strength'] for r in resistances if r['price'] > current_price], default=0)
            else:
                features['sr_distance_resistance'] = 0
                features['sr_strength_resistance'] = 0
            
            # Combined S/R features
            features['sr_distance'] = features['sr_distance_support'] + features['sr_distance_resistance']
            features['sr_strength'] = (features['sr_strength_support'] + features['sr_strength_resistance']) / 2
            
            # S/R breakout indicators
            features['sr_breakout'] = np.where(
                (features['sr_distance_support'] < 0.01) | (features['sr_distance_resistance'] < 0.01),
                1, 0
            )
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating S/R features: {e}")
            return pd.DataFrame()
    
    async def _generate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime features using enhanced HMM regime manager."""
        try:
            features = pd.DataFrame(index=df.index)
            
            if not self.regime_manager:
                return features
            
            # Train regime models if not already trained
            if not self.regime_manager.hmm_model:
                training_result = await self.regime_manager.train_regime_models(df)
                if not training_result.get("success"):
                    self.logger.warning(f"⚠️ Regime training failed: {training_result.get('error')}")
                    return features
            
            # Predict regime changes
            prediction_result = await self.regime_manager.predict_regime_changes(df)
            
            if not prediction_result.get("success"):
                self.logger.warning(f"⚠️ Regime prediction failed: {prediction_result.get('error')}")
                return features
            
            # Extract regime features
            current_regime = prediction_result.get("current_regime")
            if current_regime:
                features['regime_id'] = current_regime.regime_id
                features['regime_confidence'] = current_regime.confidence
                features['regime_duration'] = current_regime.duration
                features['regime_volatility'] = current_regime.volatility
                features['regime_momentum'] = current_regime.momentum
                features['regime_volume_profile'] = current_regime.volume_profile
            
            # Transition probabilities
            transition_probs = prediction_result.get("transition_probabilities", [])
            if transition_probs:
                features['regime_transition_prob'] = transition_probs[-1] if transition_probs else 0
            
            # HMM states
            hmm_states = prediction_result.get("hmm_states", [])
            if hmm_states:
                features['hmm_state'] = hmm_states[-1] if hmm_states else 0
            
            # Cluster labels
            cluster_labels = prediction_result.get("cluster_labels", [])
            if cluster_labels:
                features['cluster_id'] = cluster_labels[-1] if cluster_labels else 0
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating regime features: {e}")
            return pd.DataFrame()
    
    def _apply_quality_control(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control to features."""
        try:
            if features_df.empty:
                return features_df
            
            # Remove features with too many NaN values
            nan_threshold = 0.5
            nan_counts = features_df.isna().sum()
            valid_features = nan_counts[nan_counts / len(features_df) < nan_threshold].index
            features_df = features_df[valid_features]
            
            # Remove features with zero variance
            zero_var_features = features_df.columns[features_df.var() == 0]
            features_df = features_df.drop(columns=zero_var_features)
            
            # Remove features with infinite values
            inf_features = features_df.columns[np.isinf(features_df).any()]
            features_df = features_df.drop(columns=inf_features)
            
            # Fill remaining NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"🔍 Quality control: removed {len(zero_var_features) + len(inf_features)} low-quality features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in quality control: {e}")
            return features_df
    
    def _eliminate_redundancy(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Eliminate redundant features based on correlation."""
        try:
            if features_df.empty or len(features_df.columns) < 2:
                return features_df
            
            # Calculate correlation matrix
            corr_matrix = features_df.corr().abs()
            
            # Find highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > self.max_correlation_threshold)]
            
            # Remove redundant features
            features_df = features_df.drop(columns=high_corr_features)
            
            self.logger.info(f"🔍 Redundancy elimination: removed {len(high_corr_features)} redundant features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in redundancy elimination: {e}")
            return features_df
    
    def _validate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Final validation of features."""
        try:
            if features_df.empty:
                return features_df
            
            # Check for remaining issues
            issues = []
            
            # Check for NaN values
            if features_df.isna().any().any():
                issues.append("NaN values found")
            
            # Check for infinite values
            if np.isinf(features_df).any().any():
                issues.append("Infinite values found")
            
            # Check for constant columns
            constant_cols = features_df.columns[features_df.nunique() == 1]
            if len(constant_cols) > 0:
                issues.append(f"Constant columns found: {len(constant_cols)}")
                features_df = features_df.drop(columns=constant_cols)
            
            if issues:
                self.logger.warning(f"⚠️ Feature validation issues: {', '.join(issues)}")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in feature validation: {e}")
            return features_df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        try:
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth the values
            tr_smooth = tr.rolling(period).mean()
            plus_di = pd.Series(plus_dm).rolling(period).mean() / tr_smooth * 100
            minus_di = pd.Series(minus_dm).rolling(period).mean() / tr_smooth * 100
            
            # Calculate ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(period).mean()
            
            return adx.fillna(0)
            
        except Exception:
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_psar(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parabolic SAR."""
        try:
            # Simple SAR implementation
            psar = pd.Series(index=df.index, dtype=float)
            psar.iloc[0] = df['low'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    # Bullish
                    psar.iloc[i] = min(df['low'].iloc[i], psar.iloc[i-1])
                else:
                    # Bearish
                    psar.iloc[i] = max(df['high'].iloc[i], psar.iloc[i-1])
            
            return psar
            
        except Exception:
            return pd.Series([0] * len(df), index=df.index)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of generated features."""
        try:
            return {
                "total_features": len(self.feature_registry),
                "feature_categories": {cat.value: len([f for f in self.feature_registry.values() if f.category == cat]) for cat in FeatureCategory},
                "cache_stats": {
                    "feature_cache_size": len(self._feature_cache),
                    "quality_cache_size": len(self._quality_cache)
                },
                "components_available": {
                    "sr_analyzer": self.sr_analyzer is not None,
                    "regime_manager": self.regime_manager is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting feature summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._feature_cache.clear()
        self._quality_cache.clear()
        self.logger.info("Feature engineering cache cleared")