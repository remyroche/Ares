"""
Shared Feature Engineering Utilities

This module provides unified feature engineering capabilities for both TAS and NAS components,
eliminating code duplication and ensuring consistent feature extraction across the trading system.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('FeatureEngineering')

@dataclass
class FeatureSet:
    """Container for extracted features."""
    price_features: Dict[str, float]
    volatility_features: Dict[str, float]
    volume_features: Dict[str, float]
    technical_features: Dict[str, float]
    momentum_features: Dict[str, float]
    regime_features: Dict[str, float]
    metadata: Dict[str, Any]

class UnifiedFeatureEngine:
    """
    Unified feature engineering engine for both TAS and NAS components.
    
    Provides consistent feature extraction, normalization, and validation
    across all trading signal generation components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified feature engine.
        
        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.logger = logger.getChild('UnifiedFeatureEngine')
        
        # Feature extraction parameters
        self.min_data_points = self.config.get('min_data_points', 20)
        self.volatility_window = self.config.get('volatility_window', 20)
        self.momentum_window = self.config.get('momentum_window', 10)
        self.technical_window = self.config.get('technical_window', 50)
        
        # Feature normalization
        self.normalize_features = self.config.get('normalize_features', True)
        self.feature_clipping = self.config.get('feature_clipping', True)
        self.clip_threshold = self.config.get('clip_threshold', 3.0)  # 3-sigma clipping
        
        # Performance tracking
        self.feature_extraction_count = 0
        self.extraction_times = []
        
    @handles_errors
    @traced(span_name="extract_market_features")
    @log_execution_time()
    async def extract_market_features(
        self,
        market_data: pd.DataFrame,
        signal_type: str = "both",
        regime_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> FeatureSet:
        """
        Extract comprehensive market features for signal generation.
        
        Args:
            market_data: Market data DataFrame with OHLCV data
            signal_type: Type of signal ("nas", "tas", or "both")
            regime_data: Current regime information
            additional_context: Additional context for feature extraction
            
        Returns:
            FeatureSet: Comprehensive feature set for signal generation
        """
        try:
            if market_data.empty or len(market_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for feature extraction: {len(market_data)} rows")
                return self._create_empty_feature_set()
            
            tprint_info(f"🔄 Extracting {signal_type} features from {len(market_data)} data points")
            
            # Extract different feature categories
            price_features = self._extract_price_features(market_data)
            volatility_features = self._extract_volatility_features(market_data)
            volume_features = self._extract_volume_features(market_data)
            technical_features = self._extract_technical_features(market_data)
            momentum_features = self._extract_momentum_features(market_data)
            regime_features = self._extract_regime_features(market_data, regime_data)
            
            # Combine all features
            feature_set = FeatureSet(
                price_features=price_features,
                volatility_features=volatility_features,
                volume_features=volume_features,
                technical_features=technical_features,
                momentum_features=momentum_features,
                regime_features=regime_features,
                metadata={
                    'extraction_timestamp': datetime.now().isoformat(),
                    'signal_type': signal_type,
                    'data_points': len(market_data),
                    'feature_counts': {
                        'price': len(price_features),
                        'volatility': len(volatility_features),
                        'volume': len(volume_features),
                        'technical': len(technical_features),
                        'momentum': len(momentum_features),
                        'regime': len(regime_features)
                    }
                }
            )
            
            # Normalize features if requested
            if self.normalize_features:
                feature_set = self._normalize_feature_set(feature_set)
            
            self.feature_extraction_count += 1
            tprint_success(f"✅ Extracted {self._count_total_features(feature_set)} features")
            
            return feature_set
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            return self._create_empty_feature_set()
    
    def _extract_price_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract price-based features."""
        try:
            if len(market_data) < 20:
                return {}
            
            close_prices = market_data['close'].values
            features = {}
            
            # Price momentum features
            if len(close_prices) >= 2:
                features['returns_1d'] = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
            if len(close_prices) >= 6:
                features['returns_5d'] = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
            if len(close_prices) >= 21:
                features['returns_20d'] = (close_prices[-1] - close_prices[-21]) / close_prices[-21]
            
            # Moving average features
            if len(close_prices) >= 5:
                ma_5 = np.mean(close_prices[-5:])
                features['ma_5'] = ma_5
            if len(close_prices) >= 20:
                ma_20 = np.mean(close_prices[-20:])
                features['ma_20'] = ma_20
                features['ma_ratio_5_20'] = ma_5 / ma_20 if ma_20 > 0 else 1.0
            
            # Price position features
            if len(close_prices) >= 20:
                recent_high = np.max(close_prices[-20:])
                recent_low = np.min(close_prices[-20:])
                if recent_high > recent_low:
                    features['price_position'] = (close_prices[-1] - recent_low) / (recent_high - recent_low)
                else:
                    features['price_position'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price feature extraction failed: {e}")
            return {}
    
    def _extract_volatility_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-based features."""
        try:
            if len(market_data) < 20:
                return {}
            
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            features = {}
            
            # Volatility features
            if len(returns) >= 5:
                features['volatility_5d'] = np.std(returns[-5:])
            if len(returns) >= 20:
                features['volatility_20d'] = np.std(returns[-20:])
                if features['volatility_20d'] > 0:
                    features['volatility_ratio'] = features['volatility_5d'] / features['volatility_20d']
                else:
                    features['volatility_ratio'] = 1.0
            
            # High-low volatility
            if 'high' in market_data.columns and 'low' in market_data.columns:
                if len(market_data) >= 20:
                    hl_vol = np.mean((market_data['high'].iloc[-20:] - market_data['low'].iloc[-20:]) / market_data['close'].iloc[-20:])
                    features['hl_volatility'] = hl_vol
                else:
                    features['hl_volatility'] = features.get('volatility_20d', 0.02)
            else:
                features['hl_volatility'] = features.get('volatility_20d', 0.02)
            
            # Volatility regime classification
            if 'volatility_20d' in features:
                vol_20d = features['volatility_20d']
                if vol_20d > 0.05:
                    features['volatility_regime'] = 2.0  # High volatility
                elif vol_20d > 0.02:
                    features['volatility_regime'] = 1.0  # Medium volatility
                else:
                    features['volatility_regime'] = 0.0  # Low volatility
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility feature extraction failed: {e}")
            return {}
    
    def _extract_volume_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-based features."""
        try:
            if len(market_data) < 20 or 'volume' not in market_data.columns:
                return {}
            
            volumes = market_data['volume'].values
            features = {}
            
            # Volume features
            if len(volumes) >= 5:
                vol_ma_5 = np.mean(volumes[-5:])
                features['volume_ma_5'] = vol_ma_5
            if len(volumes) >= 20:
                vol_ma_20 = np.mean(volumes[-20:])
                features['volume_ma_20'] = vol_ma_20
                if vol_ma_20 > 0:
                    features['volume_ratio_5_20'] = vol_ma_5 / vol_ma_20
                else:
                    features['volume_ratio_5_20'] = 1.0
            
            # Volume volatility
            if len(volumes) >= 20:
                vol_std = np.std(volumes[-20:])
                if features.get('volume_ma_20', 0) > 0:
                    features['volume_cv'] = vol_std / features['volume_ma_20']
                else:
                    features['volume_cv'] = 0.0
            
            # Volume trend
            if 'volume_ratio_5_20' in features:
                features['volume_trend'] = features['volume_ratio_5_20'] - 1.0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume feature extraction failed: {e}")
            return {}
    
    def _extract_technical_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features."""
        try:
            if len(market_data) < 50:
                return {}
            
            close_prices = market_data['close'].values
            features = {}
            
            # RSI calculation
            if len(close_prices) >= 15:
                delta = np.diff(close_prices)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features['rsi'] = 100 - (100 / (1 + rs))
                else:
                    features['rsi'] = 100.0
                
                # Normalize RSI to [0, 1]
                features['rsi_normalized'] = features['rsi'] / 100.0
            
            # MACD approximation
            if len(close_prices) >= 26:
                ema_12 = np.mean(close_prices[-12:])
                ema_26 = np.mean(close_prices[-26:])
                macd = ema_12 - ema_26
                features['macd'] = macd / close_prices[-1]  # Normalize by price
            
            # Bollinger Bands position
            if len(close_prices) >= 20:
                ma_20 = np.mean(close_prices[-20:])
                std_20 = np.std(close_prices[-20:])
                upper_band = ma_20 + 2 * std_20
                lower_band = ma_20 - 2 * std_20
                
                if upper_band > lower_band:
                    features['bollinger_position'] = (close_prices[-1] - lower_band) / (upper_band - lower_band)
                else:
                    features['bollinger_position'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Technical feature extraction failed: {e}")
            return {}
    
    def _extract_momentum_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum-based features."""
        try:
            if len(market_data) < 10:
                return {}
            
            close_prices = market_data['close'].values
            features = {}
            
            # Price momentum
            if len(close_prices) >= 5:
                features['momentum_5d'] = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
            if len(close_prices) >= 10:
                features['momentum_10d'] = (close_prices[-1] - close_prices[-11]) / close_prices[-11]
            
            # Volume momentum
            if 'volume' in market_data.columns and len(market_data) >= 10:
                volumes = market_data['volume'].values
                vol_ma_10 = np.mean(volumes[-10:])
                if vol_ma_10 > 0:
                    features['volume_momentum'] = (volumes[-1] - vol_ma_10) / vol_ma_10
                else:
                    features['volume_momentum'] = 0.0
            
            # Combined momentum score
            price_momentum = features.get('momentum_5d', 0.0)
            volume_momentum = features.get('volume_momentum', 0.0)
            features['combined_momentum'] = (price_momentum + volume_momentum) / 2
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum feature extraction failed: {e}")
            return {}
    
    def _extract_regime_features(self, market_data: pd.DataFrame, regime_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract regime-related features."""
        try:
            features = {}
            
            # Add regime information if available
            if regime_data:
                features['regime_id'] = regime_data.get('regime_id', 0)
                features['regime_stability'] = regime_data.get('regime_stability', 0.5)
                features['regime_confidence'] = regime_data.get('confidence', 0.5)
            else:
                features['regime_id'] = 0
                features['regime_stability'] = 0.5
                features['regime_confidence'] = 0.5
            
            # Calculate trend strength
            if len(market_data) >= 20:
                features['trend_strength'] = self._calculate_trend_strength(market_data)
            else:
                features['trend_strength'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime feature extraction failed: {e}")
            return {'regime_id': 0, 'regime_stability': 0.5, 'regime_confidence': 0.5, 'trend_strength': 0.0}
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength indicator."""
        try:
            if len(market_data) < 20:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(market_data[-20:]))
            y = market_data['close'].iloc[-20:].values
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize by price
            normalized_slope = slope / market_data['close'].iloc[-1]
            return np.clip(normalized_slope * 100, -1, 1)  # Clamp to [-1, 1]
            
        except:
            return 0.0
    
    def _normalize_feature_set(self, feature_set: FeatureSet) -> FeatureSet:
        """Normalize features to prevent extreme values."""
        try:
            # Normalize each feature category
            feature_set.price_features = self._normalize_features(feature_set.price_features)
            feature_set.volatility_features = self._normalize_features(feature_set.volatility_features)
            feature_set.volume_features = self._normalize_features(feature_set.volume_features)
            feature_set.technical_features = self._normalize_features(feature_set.technical_features)
            feature_set.momentum_features = self._normalize_features(feature_set.momentum_features)
            feature_set.regime_features = self._normalize_features(feature_set.regime_features)
            
            return feature_set
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return feature_set
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize a feature dictionary."""
        try:
            if not features:
                return features
            
            # Convert to numpy array for normalization
            values = np.array(list(features.values()))
            
            # Apply clipping if enabled
            if self.feature_clipping:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val > 0:
                    values = np.clip(values, 
                                   mean_val - self.clip_threshold * std_val,
                                   mean_val + self.clip_threshold * std_val)
            
            # Normalize to [0, 1] range
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
            
            # Convert back to dictionary
            feature_names = list(features.keys())
            return dict(zip(feature_names, values))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return features
    
    def _create_empty_feature_set(self) -> FeatureSet:
        """Create empty feature set for fallback."""
        return FeatureSet(
            price_features={},
            volatility_features={},
            volume_features={},
            technical_features={},
            momentum_features={},
            regime_features={'regime_id': 0, 'regime_stability': 0.5, 'regime_confidence': 0.5},
            metadata={'extraction_timestamp': datetime.now().isoformat(), 'empty': True}
        )
    
    def _count_total_features(self, feature_set: FeatureSet) -> int:
        """Count total number of features in feature set."""
        return (len(feature_set.price_features) + 
                len(feature_set.volatility_features) + 
                len(feature_set.volume_features) + 
                len(feature_set.technical_features) + 
                len(feature_set.momentum_features) + 
                len(feature_set.regime_features))
    
    def get_feature_vector(self, feature_set: FeatureSet, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Convert feature set to numpy array for model input."""
        try:
            # Combine all features
            all_features = {}
            all_features.update(feature_set.price_features)
            all_features.update(feature_set.volatility_features)
            all_features.update(feature_set.volume_features)
            all_features.update(feature_set.technical_features)
            all_features.update(feature_set.momentum_features)
            all_features.update(feature_set.regime_features)
            
            if feature_order:
                # Use specified feature order
                return np.array([all_features.get(name, 0.0) for name in feature_order])
            else:
                # Use alphabetical order
                return np.array(list(all_features.values()))
                
        except Exception as e:
            self.logger.error(f"❌ Feature vector conversion failed: {e}")
            return np.array([])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get feature engineering performance metrics."""
        return {
            'total_extractions': self.feature_extraction_count,
            'avg_extraction_time': np.mean(self.extraction_times) if self.extraction_times else 0.0,
            'config': {
                'min_data_points': self.min_data_points,
                'normalize_features': self.normalize_features,
                'feature_clipping': self.feature_clipping
            }
        }

# Convenience functions
def create_feature_engine(config: Optional[Dict[str, Any]] = None) -> UnifiedFeatureEngine:
    """Create a configured feature engine."""
    return UnifiedFeatureEngine(config)

async def extract_features_for_signal(
    market_data: pd.DataFrame,
    signal_type: str = "both",
    regime_data: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> FeatureSet:
    """Extract features for signal generation with convenience function."""
    engine = create_feature_engine(config)
    return await engine.extract_market_features(market_data, signal_type, regime_data)