"""
Price Patterns Integration Interface

This module provides integration with the dedicated price patterns research modules
for consistent pattern definitions across the research framework.

Integration with:
- src/research/pattern_discovery_framework.py
- src/research/pure_price_action_patterns.py
- src/research/advanced_pattern_definitions.py

Key Features:
- Seamless integration with external price patterns research
- Fallback to internal pattern detection when modules unavailable
- Consistent pattern format and naming conventions
- Enhanced pattern analysis using dedicated research definitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.logger import system_logger

# Import price patterns research modules
try:
    from src.research.pattern_discovery_framework import (
        PatternDiscoveryOrchestrator,
        BasePatternDiscoverer,
        PatternType,
        PatternDefinition,
        PatternDiscoveryResult
    )
    from src.research.pure_price_action_patterns import (
        PurePriceActionPatternDiscovery,
        PurePatternType,
        PurePricePattern
    )
    PRICE_PATTERNS_MODULE_AVAILABLE = True
    system_logger.info("✅ Price patterns research modules available - using dedicated pattern definitions")
except ImportError as e:
    PRICE_PATTERNS_MODULE_AVAILABLE = False
    system_logger.warning(f"⚠️ Price patterns research modules not available: {e} - using fallback pattern detection")


@dataclass
class PatternIntegrationConfig:
    """Configuration for price patterns integration."""
    # External module settings
    use_external_patterns: bool = True
    use_pure_price_patterns: bool = True
    use_advanced_patterns: bool = True
    
    # Pattern filtering
    min_pattern_frequency: float = 0.02
    min_pattern_significance: float = 0.05
    min_predictability_score: float = 0.1
    
    # Integration settings
    pattern_alignment_method: str = "intersection"  # "intersection", "union", "external_only"
    
    # Fallback settings
    fallback_pattern_detection: bool = True
    internal_pattern_windows: List[int] = None
    
    def __post_init__(self):
        if self.internal_pattern_windows is None:
            self.internal_pattern_windows = [5, 10, 20, 50]


class PricePatternsIntegrator:
    """
    Integrator for price patterns research modules.
    
    This class provides seamless integration with the dedicated price patterns
    research modules while maintaining fallback capabilities.
    """
    
    def __init__(self, config: Optional[PatternIntegrationConfig] = None):
        self.config = config or PatternIntegrationConfig()
        self.logger = system_logger.getChild('PricePatternsIntegrator')
        self.external_available = PRICE_PATTERNS_MODULE_AVAILABLE
        
        # Initialize external pattern discovery if available
        if self.external_available and self.config.use_external_patterns:
            try:
                self.pattern_orchestrator = PatternDiscoveryOrchestrator()
                
                if self.config.use_pure_price_patterns:
                    self.pure_pattern_discovery = PurePriceActionPatternDiscovery()
                    
                self.logger.info("✅ External price pattern discovery initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize external pattern discovery: {e}")
                self.external_available = False
        else:
            self.pattern_orchestrator = None
            self.pure_pattern_discovery = None
    
    def detect_patterns(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect price patterns using external modules or fallback.
        
        Args:
            price_data: OHLCV price data
            
        Returns:
            Dictionary mapping pattern names to binary pattern indicators
        """
        patterns = {}
        
        # Try external price patterns research modules first
        if self.external_available and self.pattern_orchestrator:
            try:
                self.logger.info("🔍 Detecting patterns using external price patterns research")
                
                if 'close' not in price_data.columns:
                    self.logger.error("No close price data for external pattern detection")
                    return self._internal_pattern_detection(price_data)
                
                prices = price_data['close']
                
                # Use pattern discovery orchestrator
                discovery_results = self.pattern_orchestrator.discover_all_patterns(prices)
                
                # Convert to our expected format
                for pattern_name, pattern_result in discovery_results.items():
                    if pattern_result.is_valid_pattern:
                        patterns[pattern_name] = pattern_result.labels.astype(float)
                        self.logger.info(f"✅ Pattern '{pattern_name}': {pattern_result.frequency:.1%} frequency")
                    else:
                        self.logger.info(f"⚠️ Pattern '{pattern_name}' invalid: {pattern_result.frequency:.1%} frequency")
                
                # Use pure price action patterns if configured
                if self.config.use_pure_price_patterns and self.pure_pattern_discovery:
                    pure_results = self.pure_pattern_discovery.discover_all_patterns(prices)
                    
                    for pattern_name, pattern_result in pure_results.items():
                        if pattern_result.is_valid_pattern:
                            patterns[f"pure_{pattern_name}"] = pattern_result.labels.astype(float)
                            self.logger.info(f"✅ Pure pattern '{pattern_name}': {pattern_result.frequency:.1%} frequency")
                
                self.logger.info(f"✅ External pattern detection completed: {len(patterns)} valid patterns")
                
                if patterns:
                    return patterns
                else:
                    self.logger.warning("No valid patterns from external modules, using fallback")
                
            except Exception as e:
                self.logger.error(f"❌ External pattern detection failed: {e}")
                self.logger.info("🔄 Falling back to internal pattern detection")
        
        # Fallback to internal pattern detection
        if self.config.fallback_pattern_detection:
            patterns = self._internal_pattern_detection(price_data)
            self.logger.info(f"✅ Internal pattern detection completed: {len(patterns)} patterns")
        
        return patterns
    
    def _internal_pattern_detection(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Internal pattern detection as fallback."""
        
        patterns = {}
        
        if 'close' not in price_data.columns:
            self.logger.warning("No close price data available for internal pattern detection")
            return patterns
        
        prices = price_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Basic trend patterns
        sma_short = prices.rolling(10).mean()
        sma_long = prices.rolling(20).mean()
        
        patterns['trend_continuation'] = (
            ((prices > sma_short) & (sma_short > sma_long) & (returns > 0)) |
            ((prices < sma_short) & (sma_short < sma_long) & (returns < 0))
        ).astype(float)
        
        patterns['trend_reversal'] = (
            ((prices < sma_short) & (sma_short > sma_long)) |
            ((prices > sma_short) & (sma_short < sma_long))
        ).astype(float)
        
        # Momentum patterns
        momentum_5 = returns.rolling(5).mean()
        momentum_20 = returns.rolling(20).mean()
        
        patterns['momentum_persistence'] = (
            ((momentum_5 > 0) & (momentum_20 > 0) & (momentum_5 > momentum_20)) |
            ((momentum_5 < 0) & (momentum_20 < 0) & (momentum_5 < momentum_20))
        ).astype(float)
        
        # Volatility patterns
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(20).mean()
        
        patterns['volatility_expansion'] = (volatility > vol_ma * 1.5).astype(float)
        patterns['volatility_contraction'] = (volatility < vol_ma * 0.7).astype(float)
        
        # Mean reversion patterns
        price_zscore = (prices - sma_long) / prices.rolling(20).std()
        patterns['mean_reversion'] = (abs(price_zscore) > 1.5).astype(float)
        
        # Breakout patterns (if OHLC available)
        if all(col in price_data.columns for col in ['high', 'low']):
            rolling_high = price_data['high'].rolling(20).max()
            rolling_low = price_data['low'].rolling(20).min()
            
            patterns['breakout'] = (prices > rolling_high.shift(1)).astype(float)
            patterns['breakdown'] = (prices < rolling_low.shift(1)).astype(float)
        
        # Clean patterns
        for pattern_name in patterns:
            patterns[pattern_name] = patterns[pattern_name].fillna(0)
        
        self.logger.info(f"📊 Internal pattern detection completed: {len(patterns)} patterns")
        return patterns
    
    def get_pattern_definitions(self) -> Dict[str, Any]:
        """Get pattern definitions from external modules or internal fallback."""
        
        if self.external_available and self.pattern_orchestrator:
            try:
                # Get definitions from external modules
                definitions = {}
                
                # Get standard pattern definitions
                standard_discoverers = self.pattern_orchestrator._get_standard_discoverers()
                for discoverer in standard_discoverers:
                    pattern_def = discoverer.get_pattern_definition()
                    definitions[pattern_def.name] = {
                        'description': pattern_def.description,
                        'mathematical_formula': pattern_def.mathematical_formula,
                        'parameters': pattern_def.parameters,
                        'frequency_threshold': pattern_def.frequency_threshold,
                        'pattern_type': pattern_def.pattern_type.value
                    }
                
                # Get pure price action pattern definitions if available
                if self.config.use_pure_price_patterns and self.pure_pattern_discovery:
                    pure_discoverers = self.pure_pattern_discovery._get_all_discoverers()
                    for discoverer in pure_discoverers:
                        pattern_def = discoverer.get_pattern_definition()
                        definitions[f"pure_{pattern_def.name}"] = {
                            'description': pattern_def.description,
                            'mathematical_formula': pattern_def.mathematical_formula,
                            'parameters': pattern_def.parameters,
                            'frequency_threshold': pattern_def.frequency_threshold,
                            'pattern_type': pattern_def.pattern_type.value
                        }
                
                return definitions
                
            except Exception as e:
                self.logger.warning(f"Failed to get external pattern definitions: {e}")
        
        # Return internal pattern definitions
        return {
            'trend_continuation': {
                'description': 'Price continues in established trend direction',
                'mathematical_formula': '((price > sma_short) & (sma_short > sma_long) & (returns > 0)) | ((price < sma_short) & (sma_short < sma_long) & (returns < 0))',
                'parameters': {'short_window': 10, 'long_window': 20},
                'frequency_threshold': 0.05,
                'pattern_type': 'trend'
            },
            'momentum_persistence': {
                'description': 'Momentum continues in same direction with persistence',
                'mathematical_formula': '((momentum_5 > 0) & (momentum_20 > 0) & (momentum_5 > momentum_20)) | ((momentum_5 < 0) & (momentum_20 < 0) & (momentum_5 < momentum_20))',
                'parameters': {'short_window': 5, 'long_window': 20},
                'frequency_threshold': 0.05,
                'pattern_type': 'momentum'
            },
            'volatility_expansion': {
                'description': 'Volatility increases significantly above moving average',
                'mathematical_formula': 'volatility > volatility_ma * 1.5',
                'parameters': {'volatility_window': 20, 'expansion_threshold': 1.5},
                'frequency_threshold': 0.05,
                'pattern_type': 'volatility'
            },
            'mean_reversion': {
                'description': 'Price reverts toward moving average from extreme levels',
                'mathematical_formula': 'abs((price - sma) / rolling_std) > 1.5',
                'parameters': {'sma_window': 20, 'std_window': 20, 'zscore_threshold': 1.5},
                'frequency_threshold': 0.02,
                'pattern_type': 'mean_reversion'
            }
        }
    
    def validate_pattern_economic_significance(self,
                                             patterns: Dict[str, pd.Series],
                                             price_data: pd.DataFrame) -> Dict[str, bool]:
        """Validate economic significance of detected patterns."""
        
        if self.external_available and self.pattern_orchestrator:
            try:
                # Use external validation if available
                significance_results = {}
                
                for pattern_name, pattern_series in patterns.items():
                    # Use the pattern's built-in validation if it's from external module
                    # For now, use our internal validation
                    significance_results[pattern_name] = self._internal_significance_validation(
                        pattern_series, price_data
                    )
                
                return significance_results
                
            except Exception as e:
                self.logger.warning(f"External pattern validation failed: {e}")
        
        # Internal validation
        significance_results = {}
        
        for pattern_name, pattern_series in patterns.items():
            significance_results[pattern_name] = self._internal_significance_validation(
                pattern_series, price_data
            )
        
        return significance_results
    
    def _internal_significance_validation(self, 
                                        pattern_series: pd.Series,
                                        price_data: pd.DataFrame) -> bool:
        """Internal validation of pattern economic significance."""
        
        try:
            if 'close' not in price_data.columns:
                return False
            
            returns = price_data['close'].pct_change().fillna(0)
            
            # Align series lengths
            min_len = min(len(pattern_series), len(returns))
            pattern_aligned = pattern_series.iloc[:min_len]
            returns_aligned = returns.iloc[:min_len]
            
            # Calculate return differences when pattern occurs vs doesn't
            pattern_returns = returns_aligned[pattern_aligned > 0.5]
            no_pattern_returns = returns_aligned[pattern_aligned <= 0.5]
            
            if len(pattern_returns) < 10 or len(no_pattern_returns) < 10:
                return False
            
            # Statistical significance test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(pattern_returns, no_pattern_returns)
            
            # Economic significance: meaningful return difference
            return_diff = abs(pattern_returns.mean() - no_pattern_returns.mean())
            
            # Pattern is significant if:
            # 1. Statistically significant (p < 0.05)
            # 2. Economically meaningful (>0.1% daily return difference)
            # 3. Sufficient frequency (>2% occurrence rate)
            is_significant = (
                p_value < self.config.min_pattern_significance and
                return_diff > 0.001 and  # >0.1% daily
                pattern_aligned.mean() >= self.config.min_pattern_frequency
            )
            
            return is_significant
            
        except Exception as e:
            self.logger.warning(f"Pattern significance validation failed: {e}")
            return False


def integrate_with_price_patterns_research(price_data: pd.DataFrame,
                                         config: Optional[PatternIntegrationConfig] = None) -> Dict[str, pd.Series]:
    """
    Convenience function to integrate with price patterns research modules.
    
    Args:
        price_data: OHLCV price data
        config: Optional integration configuration
        
    Returns:
        Dictionary of detected and validated price patterns
    """
    integrator = PricePatternsIntegrator(config)
    patterns = integrator.detect_patterns(price_data)
    
    # Validate economic significance
    significance_results = integrator.validate_pattern_economic_significance(patterns, price_data)
    
    # Filter to only economically significant patterns
    significant_patterns = {
        name: pattern for name, pattern in patterns.items()
        if significance_results.get(name, False)
    }
    
    return significant_patterns


def get_available_pattern_types() -> List[str]:
    """Get list of available pattern types from external research."""
    
    if PRICE_PATTERNS_MODULE_AVAILABLE:
        try:
            # Get pattern types from external modules
            standard_types = [pt.value for pt in PatternType]
            pure_types = [pt.value for pt in PurePatternType]
            
            return standard_types + [f"pure_{pt}" for pt in pure_types]
            
        except Exception as e:
            system_logger.warning(f"Failed to get external pattern types: {e}")
    
    # Fallback pattern types
    return ['trend_continuation', 'momentum_persistence', 'volatility_expansion', 'mean_reversion', 'breakout']


# Create integrated price action analyzer function
def create_integrated_price_action_analyzer(config: Optional[PatternIntegrationConfig] = None):
    """
    Create price action analyzer that integrates with external price patterns research.
    
    Args:
        config: Optional integration configuration
        
    Returns:
        Enhanced analyzer with price patterns integration
    """
    try:
        from .enhanced_price_action_analysis import EnhancedPriceActionAnalyzer, FeaturePriceInteractionConfig
        
        # Create analyzer with pattern integration
        analyzer = EnhancedPriceActionAnalyzer(FeaturePriceInteractionConfig())
        
        # Replace pattern detection method with integrated version
        integrator = PricePatternsIntegrator(config)
        analyzer._detect_price_patterns = integrator.detect_patterns
        analyzer._pattern_integrator = integrator
        
        return analyzer
        
    except ImportError:
        system_logger.error("Enhanced price action analyzer not available")
        return None


# Example usage
if __name__ == "__main__":
    # Test integration with external price patterns research
    print("🔍 Testing Price Patterns Research Integration")
    
    # Create test price data
    np.random.seed(42)
    n_samples = 1000
    returns = np.random.randn(n_samples) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    price_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + abs(np.random.randn(n_samples)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(n_samples)) * 0.01),
        'open': prices * (1 + np.random.randn(n_samples) * 0.005),
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Test integration
    integrator = PricePatternsIntegrator()
    
    print(f"External price patterns research available: {integrator.external_available}")
    
    # Test pattern detection
    patterns = integrator.detect_patterns(price_data)
    print(f"Patterns detected: {len(patterns)}")
    
    for pattern_name, pattern_series in patterns.items():
        occurrence_rate = pattern_series.mean()
        print(f"  - {pattern_name}: {occurrence_rate:.1%} occurrence rate")
    
    # Test pattern definitions
    definitions = integrator.get_pattern_definitions()
    print(f"\nPattern definitions available: {len(definitions)}")
    
    # Test economic significance validation
    significance = integrator.validate_pattern_economic_significance(patterns, price_data)
    significant_patterns = sum(significance.values())
    print(f"Economically significant patterns: {significant_patterns}/{len(patterns)}")
    
    # Test available pattern types
    available_types = get_available_pattern_types()
    print(f"Available pattern types: {available_types}")
    
    print("\n✅ Integration framework ready for price patterns research modules")
    print("🔄 Automatically uses external research when available")
    print("📊 Fallback pattern detection working for immediate use")