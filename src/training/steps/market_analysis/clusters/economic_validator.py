"""
Economic Regime Validator for Enhanced Regime Clustering.

This module provides comprehensive economic validation for regime clustering,
ensuring that identified regimes correspond to meaningful economic states.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)


class RegimeType(Enum):
    """Economic regime types."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeProfile:
    """Economic profile of a regime."""
    regime_id: int
    regime_type: RegimeType
    volatility_level: str  # "Low", "Medium", "High"
    trend_direction: str   # "Up", "Down", "Sideways", "Neutral"
    market_phase: str      # "Bull", "Bear", "Consolidation", "Crisis", "Recovery"
    risk_level: str        # "Low", "Medium", "High", "Extreme"
    economic_score: float  # Overall economic meaningfulness score
    characteristics: Dict[str, Any]  # Detailed characteristics


class EconomicRegimeValidator:
    """Validates and classifies economic regimes."""
    
    def __init__(self, lookback_periods: int = 20, volatility_threshold: float = 0.02):
        """
        Initialize economic regime validator.
        
        Args:
            lookback_periods: Number of periods for rolling calculations
            volatility_threshold: Threshold for high/low volatility classification
        """
        self.lookback_periods = lookback_periods
        self.volatility_threshold = volatility_threshold
        
    def validate_regime_economics(
        self, 
        market_data: pd.DataFrame, 
        labels: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Validate economic meaningfulness of regime clustering.
        
        Args:
            market_data: Market data with OHLCV columns
            labels: Cluster labels
            features: Optional feature matrix for additional analysis
            
        Returns:
            Dictionary containing economic validation results
        """
        try:
            tprint_info("Starting economic regime validation...")
            
            # Get unique regimes
            unique_regimes = np.unique(labels)
            regime_profiles = []
            
            # Analyze each regime
            for regime_id in unique_regimes:
                regime_mask = labels == regime_id
                regime_data = market_data[regime_mask]
                
                if len(regime_data) < 5:  # Skip regimes with too few samples
                    continue
                
                profile = self._analyze_regime_profile(regime_id, regime_data, market_data)
                regime_profiles.append(profile)
            
            # Calculate economic separation
            economic_separation = self._calculate_economic_separation(regime_profiles)
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(market_data, labels)
            
            # Calculate overall economic quality
            economic_quality = self._calculate_economic_quality(regime_profiles, economic_separation)
            
            # Generate economic insights
            economic_insights = self._generate_economic_insights(regime_profiles, economic_separation)
            
            results = {
                'regime_profiles': [profile.__dict__ for profile in regime_profiles],
                'economic_separation': economic_separation,
                'regime_stability': regime_stability,
                'economic_quality': economic_quality,
                'economic_insights': economic_insights,
                'validation_passed': economic_quality > 0.6
            }
            
            tprint_success(f"Economic validation completed. Quality: {economic_quality:.3f}")
            return results
            
        except Exception as e:
            tprint_error(f"Economic validation failed: {e}")
            return {'error': str(e), 'validation_passed': False}
    
    def _analyze_regime_profile(
        self, 
        regime_id: int, 
        regime_data: pd.DataFrame, 
        full_market_data: pd.DataFrame
    ) -> RegimeProfile:
        """Analyze economic profile of a single regime."""
        try:
            # Calculate basic metrics
            returns = regime_data['close'].pct_change().dropna() if 'close' in regime_data.columns else pd.Series()
            volatility = returns.std() if len(returns) > 0 else 0.0
            avg_return = returns.mean() if len(returns) > 0 else 0.0
            
            # Calculate trend metrics
            if 'close' in regime_data.columns and len(regime_data) > 1:
                price_change = (regime_data['close'].iloc[-1] - regime_data['close'].iloc[0]) / regime_data['close'].iloc[0]
                trend_strength = self._calculate_trend_strength(regime_data['close'])
            else:
                price_change = 0.0
                trend_strength = 0.0
            
            # Calculate volume metrics
            volume_ratio = self._calculate_volume_ratio(regime_data, full_market_data)
            
            # Calculate risk metrics
            max_drawdown = self._calculate_max_drawdown(regime_data['close']) if 'close' in regime_data.columns else 0.0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
            
            # Classify regime type
            regime_type = self._classify_regime_type(
                volatility, avg_return, price_change, trend_strength, volume_ratio, max_drawdown
            )
            
            # Determine characteristics
            volatility_level = self._classify_volatility_level(volatility, full_market_data)
            trend_direction = self._classify_trend_direction(price_change, trend_strength)
            market_phase = self._classify_market_phase(regime_type, volatility, avg_return)
            risk_level = self._classify_risk_level(volatility, max_drawdown, sharpe_ratio)
            
            # Calculate economic score
            economic_score = self._calculate_regime_economic_score(
                volatility, avg_return, trend_strength, volume_ratio, max_drawdown
            )
            
            # Create characteristics dictionary
            characteristics = {
                'volatility': volatility,
                'avg_return': avg_return,
                'price_change': price_change,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sample_count': len(regime_data),
                'duration_days': len(regime_data)
            }
            
            return RegimeProfile(
                regime_id=regime_id,
                regime_type=regime_type,
                volatility_level=volatility_level,
                trend_direction=trend_direction,
                market_phase=market_phase,
                risk_level=risk_level,
                economic_score=economic_score,
                characteristics=characteristics
            )
            
        except Exception as e:
            tprint_warning(f"Failed to analyze regime {regime_id}: {e}")
            return RegimeProfile(
                regime_id=regime_id,
                regime_type=RegimeType.UNKNOWN,
                volatility_level="Unknown",
                trend_direction="Unknown",
                market_phase="Unknown",
                risk_level="Unknown",
                economic_score=0.0,
                characteristics={}
            )
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression slope."""
        try:
            if len(prices) < 2:
                return 0.0
            
            x = np.arange(len(prices))
            y = prices.values
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize by price level
            normalized_slope = slope / prices.iloc[0] if prices.iloc[0] != 0 else 0.0
            return normalized_slope
            
        except Exception:
            return 0.0
    
    def _calculate_volume_ratio(self, regime_data: pd.DataFrame, full_market_data: pd.DataFrame) -> float:
        """Calculate volume ratio compared to market average."""
        try:
            if 'volume' not in regime_data.columns or 'volume' not in full_market_data.columns:
                return 1.0
            
            regime_avg_volume = regime_data['volume'].mean()
            market_avg_volume = full_market_data['volume'].mean()
            
            if market_avg_volume == 0:
                return 1.0
            
            return regime_avg_volume / market_avg_volume
            
        except Exception:
            return 1.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(prices) < 2:
                return 0.0
            
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
            
        except Exception:
            return 0.0
    
    def _classify_regime_type(
        self, 
        volatility: float, 
        avg_return: float, 
        price_change: float, 
        trend_strength: float,
        volume_ratio: float,
        max_drawdown: float
    ) -> RegimeType:
        """Classify regime type based on economic characteristics."""
        try:
            # Crisis detection
            if max_drawdown > 0.2 or (volatility > 0.05 and avg_return < -0.01):
                return RegimeType.CRISIS
            
            # High volatility regime
            if volatility > self.volatility_threshold * 2:
                return RegimeType.HIGH_VOLATILITY
            
            # Low volatility regime
            if volatility < self.volatility_threshold * 0.5:
                return RegimeType.LOW_VOLATILITY
            
            # Trending regimes
            if trend_strength > 0.001 and avg_return > 0:
                return RegimeType.TRENDING_UP
            elif trend_strength < -0.001 and avg_return < 0:
                return RegimeType.TRENDING_DOWN
            
            # Bull/Bear market detection
            if price_change > 0.1 and avg_return > 0.001:
                return RegimeType.BULL_MARKET
            elif price_change < -0.1 and avg_return < -0.001:
                return RegimeType.BEAR_MARKET
            
            # Recovery detection
            if avg_return > 0.005 and max_drawdown > 0.1:
                return RegimeType.RECOVERY
            
            # Sideways market
            if abs(trend_strength) < 0.0005 and abs(price_change) < 0.05:
                return RegimeType.SIDEWAYS
            
            return RegimeType.UNKNOWN
            
        except Exception:
            return RegimeType.UNKNOWN
    
    def _classify_volatility_level(self, volatility: float, market_data: pd.DataFrame) -> str:
        """Classify volatility level relative to market."""
        try:
            if 'close' not in market_data.columns:
                return "Medium"
            
            market_returns = market_data['close'].pct_change().dropna()
            market_volatility = market_returns.std()
            
            if volatility > market_volatility * 1.5:
                return "High"
            elif volatility < market_volatility * 0.7:
                return "Low"
            else:
                return "Medium"
                
        except Exception:
            return "Medium"
    
    def _classify_trend_direction(self, price_change: float, trend_strength: float) -> str:
        """Classify trend direction."""
        if trend_strength > 0.001 and price_change > 0.02:
            return "Up"
        elif trend_strength < -0.001 and price_change < -0.02:
            return "Down"
        elif abs(trend_strength) < 0.0005 and abs(price_change) < 0.02:
            return "Sideways"
        else:
            return "Neutral"
    
    def _classify_market_phase(self, regime_type: RegimeType, volatility: float, avg_return: float) -> str:
        """Classify market phase."""
        if regime_type == RegimeType.BULL_MARKET:
            return "Bull"
        elif regime_type == RegimeType.BEAR_MARKET:
            return "Bear"
        elif regime_type == RegimeType.CRISIS:
            return "Crisis"
        elif regime_type == RegimeType.RECOVERY:
            return "Recovery"
        elif volatility > self.volatility_threshold * 1.5:
            return "High Volatility"
        else:
            return "Consolidation"
    
    def _classify_risk_level(self, volatility: float, max_drawdown: float, sharpe_ratio: float) -> str:
        """Classify risk level."""
        if volatility > self.volatility_threshold * 2 or max_drawdown > 0.3:
            return "Extreme"
        elif volatility > self.volatility_threshold * 1.5 or max_drawdown > 0.15:
            return "High"
        elif volatility < self.volatility_threshold * 0.7 and max_drawdown < 0.05:
            return "Low"
        else:
            return "Medium"
    
    def _calculate_regime_economic_score(
        self, 
        volatility: float, 
        avg_return: float, 
        trend_strength: float,
        volume_ratio: float,
        max_drawdown: float
    ) -> float:
        """Calculate economic meaningfulness score for a regime."""
        try:
            # Base score from volatility distinctiveness
            vol_score = min(1.0, volatility / (self.volatility_threshold * 2))
            
            # Return distinctiveness
            return_score = min(1.0, abs(avg_return) / 0.02)
            
            # Trend distinctiveness
            trend_score = min(1.0, abs(trend_strength) / 0.002)
            
            # Volume distinctiveness
            volume_score = min(1.0, abs(volume_ratio - 1.0) / 0.5)
            
            # Risk distinctiveness
            risk_score = min(1.0, max_drawdown / 0.1)
            
            # Weighted average
            economic_score = (
                0.3 * vol_score +
                0.25 * return_score +
                0.2 * trend_score +
                0.15 * volume_score +
                0.1 * risk_score
            )
            
            return economic_score
            
        except Exception:
            return 0.0
    
    def _calculate_economic_separation(self, regime_profiles: List[RegimeProfile]) -> float:
        """Calculate economic separation between regimes."""
        try:
            if len(regime_profiles) < 2:
                return 0.0
            
            # Extract characteristics
            volatilities = [p.characteristics.get('volatility', 0) for p in regime_profiles]
            returns = [p.characteristics.get('avg_return', 0) for p in regime_profiles]
            trends = [p.characteristics.get('trend_strength', 0) for p in regime_profiles]
            
            # Calculate separation metrics
            vol_separation = np.std(volatilities) / (np.mean(volatilities) + 1e-8)
            return_separation = np.std(returns) / (np.std(returns) + 1e-8)
            trend_separation = np.std(trends) / (np.std(trends) + 1e-8)
            
            # Combined separation score
            separation = (vol_separation + return_separation + trend_separation) / 3
            
            return separation
            
        except Exception:
            return 0.0
    
    def _calculate_regime_stability(self, market_data: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate regime stability over time."""
        try:
            if len(labels) < 10:
                return 0.0
            
            # Calculate regime persistence
            regime_changes = np.sum(labels[1:] != labels[:-1])
            total_periods = len(labels) - 1
            
            if total_periods == 0:
                return 0.0
            
            # Stability score (higher = more stable)
            stability = 1.0 - (regime_changes / total_periods)
            
            return stability
            
        except Exception:
            return 0.0
    
    def _calculate_economic_quality(
        self, 
        regime_profiles: List[RegimeProfile], 
        economic_separation: float
    ) -> float:
        """Calculate overall economic quality score."""
        try:
            if not regime_profiles:
                return 0.0
            
            # Average economic score of regimes
            avg_economic_score = np.mean([p.economic_score for p in regime_profiles])
            
            # Regime diversity (different regime types)
            unique_types = len(set(p.regime_type for p in regime_profiles))
            type_diversity = min(1.0, unique_types / len(regime_profiles))
            
            # Combined quality score
            quality = 0.6 * avg_economic_score + 0.3 * economic_separation + 0.1 * type_diversity
            
            return quality
            
        except Exception:
            return 0.0
    
    def _generate_economic_insights(
        self, 
        regime_profiles: List[RegimeProfile], 
        economic_separation: float
    ) -> Dict[str, Any]:
        """Generate economic insights from regime analysis."""
        try:
            insights = {
                'total_regimes': len(regime_profiles),
                'regime_types': [p.regime_type.value for p in regime_profiles],
                'volatility_levels': [p.volatility_level for p in regime_profiles],
                'trend_directions': [p.trend_direction for p in regime_profiles],
                'market_phases': [p.market_phase for p in regime_profiles],
                'risk_levels': [p.risk_level for p in regime_profiles],
                'economic_separation': economic_separation,
                'regime_diversity': len(set(p.regime_type for p in regime_profiles)),
                'high_quality_regimes': len([p for p in regime_profiles if p.economic_score > 0.7]),
                'low_quality_regimes': len([p for p in regime_profiles if p.economic_score < 0.3])
            }
            
            return insights
            
        except Exception:
            return {}


def create_economic_validator(lookback_periods: int = 20, volatility_threshold: float = 0.02) -> EconomicRegimeValidator:
    """Create economic regime validator instance."""
    return EconomicRegimeValidator(lookback_periods=lookback_periods, volatility_threshold=volatility_threshold)