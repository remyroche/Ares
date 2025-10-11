"""
VectorBT Time Series Aware Directional Feature Selector

This module provides time series aware feature selection using VectorBT's
advanced time series analysis capabilities and regime detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Import VectorBT with fallback
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available. Using fallback implementations.")

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error, tprint_performance
from src.utils.dependency_manager import DependencyManager

logger = logging.getLogger(__name__)

@dataclass
class VectorBTDirectionalConfig:
    """Configuration for VectorBT directional feature selection."""
    # Regime detection parameters
    enable_regime_detection: bool = True
    regime_window: int = 50
    volatility_threshold: float = 1.5
    trend_threshold: float = 0.02
    
    # Time series analysis parameters
    enable_temporal_analysis: bool = True
    temporal_window: int = 20
    seasonality_detection: bool = True
    seasonality_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # Feature selection parameters
    max_features_per_regime: int = 30
    min_features_per_regime: int = 5
    regime_adaptation_threshold: float = 0.7
    
    # Cross-asset analysis
    enable_cross_asset: bool = True
    correlation_threshold: float = 0.8
    cross_asset_window: int = 30
    
    # Performance optimization
    enable_parallel: bool = True
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True

@dataclass
class RegimeInfo:
    """Information about detected market regime."""
    regime_type: str  # 'uptrend_high_vol', 'downtrend_low_vol', etc.
    confidence: float
    volatility_level: str  # 'high', 'medium', 'low'
    trend_direction: str  # 'up', 'down', 'sideways'
    regime_duration: int
    regime_strength: float

@dataclass
class TemporalFeatureInfo:
    """Information about temporal feature characteristics."""
    feature_name: str
    trend_strength: float
    seasonality_strength: float
    autocorrelation: float
    stationarity: bool
    temporal_importance: float

@dataclass
class VectorBTDirectionalResult:
    """Result of VectorBT directional feature selection."""
    selected_features: List[str]
    regime_info: RegimeInfo
    temporal_features: List[TemporalFeatureInfo]
    cross_asset_features: List[str]
    feature_scores: Dict[str, float]
    regime_adaptation: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class VectorBTDirectionalSelector:
    """Time series aware directional feature selector using VectorBT."""
    
    def __init__(self, config: Optional[VectorBTDirectionalConfig] = None):
        """Initialize VectorBT directional selector."""
        self.config = config or VectorBTDirectionalConfig()
        self.logger = logger.getChild('VectorBTDirectionalSelector')
        self.dependency_manager = DependencyManager()
        
        # Check VectorBT availability
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available. Using fallback implementations.")
            self.vectorbt_available = False
        else:
            self.vectorbt_available = True
            tprint_success("✅ VectorBT available for time series analysis")
        
        # Performance tracking
        self.performance_stats = {
            'selections_performed': 0,
            'regime_detections': 0,
            'temporal_analyses': 0,
            'total_time': 0.0
        }
        
        tprint_success("🚀 VectorBTDirectionalSelector initialized")
    
    def _detect_market_regime(self, prices: pd.DataFrame, returns: pd.Series) -> RegimeInfo:
        """Detect current market regime using VectorBT."""
        if not self.vectorbt_available or len(returns) < self.config.regime_window:
            return self._detect_market_regime_fallback(prices, returns)
        
        try:
            # Calculate volatility regime
            volatility = returns.rolling(window=self.config.regime_window).std() * np.sqrt(252)
            current_vol = volatility.iloc[-1] if not volatility.empty else 0.0
            avg_vol = volatility.mean() if not volatility.empty else 0.0
            
            # Determine volatility level
            if current_vol > avg_vol * self.config.volatility_threshold:
                vol_level = 'high'
            elif current_vol < avg_vol / self.config.volatility_threshold:
                vol_level = 'low'
            else:
                vol_level = 'medium'
            
            # Calculate trend regime
            if len(prices) > 0:
                price_series = prices.iloc[:, 0] if len(prices.columns) > 0 else prices.squeeze()
                
                # Use VectorBT for trend analysis
                sma_short = vbt.MA.run(price_series, window=10).ma
                sma_long = vbt.MA.run(price_series, window=30).ma
                
                if not sma_short.empty and not sma_long.empty:
                    trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                    
                    if trend_strength > self.config.trend_threshold:
                        trend_direction = 'up'
                    elif trend_strength < -self.config.trend_threshold:
                        trend_direction = 'down'
                    else:
                        trend_direction = 'sideways'
                else:
                    trend_direction = 'sideways'
                    trend_strength = 0.0
            else:
                trend_direction = 'sideways'
                trend_strength = 0.0
            
            # Calculate regime confidence
            confidence = min(abs(trend_strength) * 10, 1.0)  # Scale to 0-1
            
            # Determine regime type
            regime_type = f"{trend_direction}_{vol_level}_vol"
            
            # Calculate regime duration (simplified)
            regime_duration = self._calculate_regime_duration(returns, trend_direction)
            
            # Calculate regime strength
            regime_strength = confidence * (1.0 - current_vol / (avg_vol * 2))  # Penalize high volatility
            
            self.performance_stats['regime_detections'] += 1
            
            return RegimeInfo(
                regime_type=regime_type,
                confidence=confidence,
                volatility_level=vol_level,
                trend_direction=trend_direction,
                regime_duration=regime_duration,
                regime_strength=regime_strength
            )
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return self._detect_market_regime_fallback(prices, returns)
    
    def _detect_market_regime_fallback(self, prices: pd.DataFrame, returns: pd.Series) -> RegimeInfo:
        """Fallback regime detection without VectorBT."""
        if len(returns) < 10:
            return RegimeInfo(
                regime_type='unknown',
                confidence=0.0,
                volatility_level='medium',
                trend_direction='sideways',
                regime_duration=0,
                regime_strength=0.0
            )
        
        # Simple volatility calculation
        volatility = returns.rolling(window=min(20, len(returns))).std() * np.sqrt(252)
        current_vol = volatility.iloc[-1] if not volatility.empty else 0.0
        avg_vol = volatility.mean() if not volatility.empty else 0.0
        
        vol_level = 'high' if current_vol > avg_vol * 1.5 else 'low' if current_vol < avg_vol * 0.5 else 'medium'
        
        # Simple trend detection
        if len(prices) > 0:
            price_series = prices.iloc[:, 0] if len(prices.columns) > 0 else prices.squeeze()
            sma_short = price_series.rolling(window=10).mean()
            sma_long = price_series.rolling(window=30).mean()
            
            if not sma_short.empty and not sma_long.empty:
                trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                trend_direction = 'up' if trend_strength > 0.02 else 'down' if trend_strength < -0.02 else 'sideways'
            else:
                trend_direction = 'sideways'
        else:
            trend_direction = 'sideways'
        
        return RegimeInfo(
            regime_type=f"{trend_direction}_{vol_level}_vol",
            confidence=0.5,
            volatility_level=vol_level,
            trend_direction=trend_direction,
            regime_duration=0,
            regime_strength=0.5
        )
    
    def _calculate_regime_duration(self, returns: pd.Series, current_trend: str) -> int:
        """Calculate how long the current regime has been active."""
        if len(returns) < 10:
            return 0
        
        # Simple regime duration calculation
        # Look for trend changes in recent periods
        price_series = returns.cumsum()  # Approximate price from returns
        sma_short = price_series.rolling(window=5).mean()
        sma_long = price_series.rolling(window=15).mean()
        
        if sma_short.empty or sma_long.empty:
            return 0
        
        # Count consecutive periods with same trend
        duration = 0
        for i in range(len(sma_short) - 1, max(0, len(sma_short) - 20), -1):
            if i < len(sma_long):
                trend_strength = (sma_short.iloc[i] - sma_long.iloc[i]) / sma_long.iloc[i]
                current_trend_check = 'up' if trend_strength > 0.01 else 'down' if trend_strength < -0.01 else 'sideways'
                
                if current_trend_check == current_trend:
                    duration += 1
                else:
                    break
        
        return duration
    
    def _analyze_temporal_features(self, features: pd.DataFrame, feature_names: List[str]) -> List[TemporalFeatureInfo]:
        """Analyze temporal characteristics of features using VectorBT."""
        if not self.vectorbt_available or len(features) < 10:
            return self._analyze_temporal_features_fallback(features, feature_names)
        
        try:
            temporal_features = []
            
            for i, feature_name in enumerate(feature_names):
                if i >= features.shape[1]:
                    continue
                
                feature_series = features.iloc[:, i]
                
                # Calculate trend strength using VectorBT
                if len(feature_series) > 10:
                    # Linear trend
                    trend_slope = vbt.linear_regression(feature_series, window=min(20, len(feature_series))).slope
                    trend_strength = abs(trend_slope.iloc[-1]) if not trend_slope.empty else 0.0
                    
                    # Seasonality detection
                    seasonality_strength = 0.0
                    if self.config.seasonality_detection:
                        for period in self.config.seasonality_periods:
                            if len(feature_series) > period * 2:
                                # Simple seasonality check using autocorrelation
                                autocorr = feature_series.autocorr(lag=period)
                                seasonality_strength = max(seasonality_strength, abs(autocorr))
                    
                    # Autocorrelation
                    autocorr = feature_series.autocorr(lag=1) if len(feature_series) > 1 else 0.0
                    
                    # Stationarity test (simplified)
                    stationarity = self._test_stationarity(feature_series)
                    
                    # Calculate temporal importance
                    temporal_importance = (
                        trend_strength * 0.4 +
                        seasonality_strength * 0.3 +
                        abs(autocorr) * 0.3
                    )
                    
                    temporal_features.append(TemporalFeatureInfo(
                        feature_name=feature_name,
                        trend_strength=float(trend_strength),
                        seasonality_strength=float(seasonality_strength),
                        autocorrelation=float(autocorr),
                        stationarity=stationarity,
                        temporal_importance=float(temporal_importance)
                    ))
            
            self.performance_stats['temporal_analyses'] += len(temporal_features)
            return temporal_features
            
        except Exception as e:
            self.logger.warning(f"Temporal analysis failed: {e}")
            return self._analyze_temporal_features_fallback(features, feature_names)
    
    def _analyze_temporal_features_fallback(self, features: pd.DataFrame, feature_names: List[str]) -> List[TemporalFeatureInfo]:
        """Fallback temporal analysis without VectorBT."""
        temporal_features = []
        
        for i, feature_name in enumerate(feature_names):
            if i >= features.shape[1]:
                continue
            
            feature_series = features.iloc[:, i]
            
            if len(feature_series) < 3:
                continue
            
            # Simple trend calculation
            x = np.arange(len(feature_series))
            slope, _ = np.polyfit(x, feature_series.values, 1)
            trend_strength = abs(slope)
            
            # Simple autocorrelation
            autocorr = feature_series.autocorr(lag=1) if len(feature_series) > 1 else 0.0
            
            # Simple stationarity test
            stationarity = self._test_stationarity(feature_series)
            
            temporal_importance = trend_strength * 0.5 + abs(autocorr) * 0.5
            
            temporal_features.append(TemporalFeatureInfo(
                feature_name=feature_name,
                trend_strength=float(trend_strength),
                seasonality_strength=0.0,
                autocorrelation=float(autocorr),
                stationarity=stationarity,
                temporal_importance=float(temporal_importance)
            ))
        
        return temporal_features
    
    def _test_stationarity(self, series: pd.Series) -> bool:
        """Simple stationarity test."""
        if len(series) < 10:
            return True
        
        # Simple test: check if variance is relatively stable
        half_len = len(series) // 2
        first_half_var = series.iloc[:half_len].var()
        second_half_var = series.iloc[half_len:].var()
        
        if first_half_var == 0 or second_half_var == 0:
            return True
        
        variance_ratio = abs(first_half_var - second_half_var) / max(first_half_var, second_half_var)
        return variance_ratio < 0.5  # Threshold for stationarity
    
    def _analyze_cross_asset_features(self, features: pd.DataFrame, feature_names: List[str]) -> List[str]:
        """Analyze cross-asset feature relationships."""
        if not self.config.enable_cross_asset or len(features) < 2:
            return []
        
        try:
            cross_asset_features = []
            correlation_matrix = features.corr()
            
            for i, feature_name in enumerate(feature_names):
                if i >= features.shape[1]:
                    continue
                
                # Check correlations with other features
                feature_correlations = correlation_matrix.iloc[i].abs()
                high_corr_count = (feature_correlations > self.config.correlation_threshold).sum() - 1  # Exclude self-correlation
                
                # Features with moderate correlations are good for cross-asset analysis
                if 1 <= high_corr_count <= 3:  # Not too isolated, not too correlated
                    cross_asset_features.append(feature_name)
            
            return cross_asset_features
            
        except Exception as e:
            self.logger.warning(f"Cross-asset analysis failed: {e}")
            return []
    
    def _select_features_by_regime(self, 
                                 features: pd.DataFrame,
                                 feature_names: List[str],
                                 regime_info: RegimeInfo,
                                 temporal_features: List[TemporalFeatureInfo]) -> Tuple[List[str], Dict[str, float]]:
        """Select features based on detected regime."""
        selected_features = []
        feature_scores = {}
        
        # Filter features based on regime
        if regime_info.regime_type.startswith('up'):
            # Uptrend: prefer momentum and trend-following features
            regime_features = [f for f in temporal_features if f.trend_strength > 0.1]
        elif regime_info.regime_type.startswith('down'):
            # Downtrend: prefer mean-reversion and defensive features
            regime_features = [f for f in temporal_features if f.trend_strength < -0.1 or f.stationarity]
        else:
            # Sideways: prefer range-bound and volatility features
            regime_features = [f for f in temporal_features if f.stationarity or f.seasonality_strength > 0.1]
        
        # Sort by temporal importance
        regime_features.sort(key=lambda x: x.temporal_importance, reverse=True)
        
        # Select features within limits
        max_features = min(self.config.max_features_per_regime, len(regime_features))
        min_features = min(self.config.min_features_per_regime, len(regime_features))
        
        # Adjust selection based on regime confidence
        if regime_info.confidence > self.config.regime_adaptation_threshold:
            # High confidence: use more features
            n_features = min(max_features, int(len(regime_features) * 0.8))
        else:
            # Low confidence: use fewer features
            n_features = max(min_features, int(len(regime_features) * 0.5))
        
        selected_temporal = regime_features[:n_features]
        selected_features = [f.feature_name for f in selected_temporal]
        
        # Calculate feature scores
        for feature in selected_temporal:
            feature_scores[feature.feature_name] = feature.temporal_importance
        
        return selected_features, feature_scores
    
    def select_features(self, 
                       features: Union[np.ndarray, pd.DataFrame],
                       prices: Union[np.ndarray, pd.DataFrame],
                       returns: Union[np.ndarray, pd.Series],
                       feature_names: Optional[List[str]] = None) -> VectorBTDirectionalResult:
        """Select features using VectorBT time series analysis."""
        tprint("🔍 Starting VectorBT directional feature selection")
        
        start_time = datetime.now()
        
        try:
            # Prepare data
            if isinstance(features, np.ndarray):
                features_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
            else:
                features_df = features.copy()
            
            if isinstance(prices, np.ndarray):
                prices_df = pd.DataFrame(prices, columns=['price'])
            else:
                prices_df = prices.copy()
            
            if isinstance(returns, np.ndarray):
                returns_series = pd.Series(returns)
            else:
                returns_series = returns.copy()
            
            # Generate feature names if not provided
            if feature_names is None:
                feature_names = list(features_df.columns)
            
            # Detect market regime
            regime_info = self._detect_market_regime(prices_df, returns_series)
            tprint(f"📊 Detected regime: {regime_info.regime_type} (confidence: {regime_info.confidence:.2f})")
            
            # Analyze temporal features
            temporal_features = []
            if self.config.enable_temporal_analysis:
                temporal_features = self._analyze_temporal_features(features_df, feature_names)
                tprint(f"⏰ Analyzed {len(temporal_features)} temporal features")
            
            # Analyze cross-asset features
            cross_asset_features = []
            if self.config.enable_cross_asset:
                cross_asset_features = self._analyze_cross_asset_features(features_df, feature_names)
                tprint(f"🔗 Found {len(cross_asset_features)} cross-asset features")
            
            # Select features by regime
            selected_features, feature_scores = self._select_features_by_regime(
                features_df, feature_names, regime_info, temporal_features
            )
            
            # Create regime adaptation info
            regime_adaptation = {
                'regime_confidence': regime_info.confidence,
                'adaptation_applied': regime_info.confidence > self.config.regime_adaptation_threshold,
                'features_per_regime': {
                    'uptrend': len([f for f in temporal_features if f.trend_strength > 0.1]),
                    'downtrend': len([f for f in temporal_features if f.trend_strength < -0.1]),
                    'sideways': len([f for f in temporal_features if f.stationarity])
                }
            }
            
            # Create result
            result = VectorBTDirectionalResult(
                selected_features=selected_features,
                regime_info=regime_info,
                temporal_features=temporal_features,
                cross_asset_features=cross_asset_features,
                feature_scores=feature_scores,
                regime_adaptation=regime_adaptation,
                analysis_metadata={
                    'vectorbt_available': self.vectorbt_available,
                    'total_features': len(feature_names),
                    'selected_features': len(selected_features),
                    'analysis_time': (datetime.now() - start_time).total_seconds(),
                    'config': self.config.__dict__
                }
            )
            
            # Update performance stats
            self.performance_stats['selections_performed'] += 1
            self.performance_stats['total_time'] += (datetime.now() - start_time).total_seconds()
            
            tprint_success(f"✅ Directional selection completed: {len(selected_features)}/{len(feature_names)} features "
                         f"for {regime_info.regime_type} regime")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Directional feature selection failed: {e}")
            tprint_error(f"❌ Selection failed: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['selections_performed'] > 0:
            stats['avg_time_per_selection'] = stats['total_time'] / stats['selections_performed']
        else:
            stats['avg_time_per_selection'] = 0.0
        
        tprint_performance(f"📊 VectorBT Directional Stats: {stats['selections_performed']} selections, "
                         f"{stats['avg_time_per_selection']:.3f}s avg")
        
        return stats

def create_vectorbt_directional_selector(config: Optional[VectorBTDirectionalConfig] = None) -> VectorBTDirectionalSelector:
    """Create a VectorBT directional selector."""
    return VectorBTDirectionalSelector(config)