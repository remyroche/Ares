"""
Optimized Lookback Integration

This module integrates optimized lookback periods from feature_lookback_optimization
with PID-based feature generation to ensure all features use the most effective
lookback periods.

Key Features:
- Integrates with feature_lookback_optimization results
- Applies optimized lookback periods to all feature generation
- Validates lookback period effectiveness
- Provides fallback mechanisms for missing optimization results
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('OptimizedLookbackIntegration')
except ImportError:
    logger = logging.getLogger('OptimizedLookbackIntegration')
    logger.setLevel(logging.INFO)


class IntegrationStatus(Enum):
    """Status of lookback integration."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FALLBACK = "fallback"
    FAILED = "failed"


@dataclass
class LookbackIntegrationResult:
    """Result of lookback integration."""
    optimized_lookback_periods: Dict[str, int] = field(default_factory=dict)
    integration_status: IntegrationStatus = IntegrationStatus.FAILED
    features_optimized: int = 0
    optimization_quality_score: float = 0.0
    fallback_periods_used: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    integration_time: float = 0.0
    optimization_source: str = "unknown"
    validation_passed: bool = False


class OptimizedLookbackIntegration:
    """
    Optimized Lookback Integration.
    
    Integrates optimized lookback periods from feature_lookback_optimization
    with PID-based feature generation.
    """
    
    def __init__(self):
        """Initialize the optimized lookback integration."""
        self.logger = logger.getChild('OptimizedLookbackIntegration')
        
        # Default lookback periods for common features
        self.default_lookback_periods = {
            # Technical indicators
            'rsi': 14,
            'sma': 20,
            'ema': 12,
            'macd': 26,
            'bollinger': 20,
            'stochastic': 14,
            'williams_r': 14,
            'cci': 20,
            'atr': 14,
            'adx': 14,
            
            # Price features
            'price_momentum': 10,
            'price_acceleration': 5,
            'price_volatility': 20,
            'price_trend': 15,
            
            # Volume features
            'volume_momentum': 10,
            'volume_profile': 20,
            'volume_ratio': 14,
            'volume_trend': 15,
            
            # Volatility features
            'volatility_regime': 20,
            'volatility_momentum': 10,
            'volatility_trend': 15,
            
            # Cross-timeframe features
            'cross_timeframe_momentum': 10,
            'cross_timeframe_volatility': 20,
            'cross_timeframe_trend': 15
        }
        
        self.logger.info("🔧 OptimizedLookbackIntegration initialized")
        self.logger.info(f"📊 Default lookback periods available for {len(self.default_lookback_periods)} feature types")
    
    def integrate_optimized_lookback_periods(
        self, 
        feature_lookback_optimization_result: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None
    ) -> LookbackIntegrationResult:
        """
        Integrate optimized lookback periods from feature_lookback_optimization.
        
        Args:
            feature_lookback_optimization_result: Results from feature_lookback_optimization
            feature_names: List of feature names to optimize
            
        Returns:
            LookbackIntegrationResult with integrated lookback periods
        """
        start_time = time.time()
        self.logger.info("🔧 Starting optimized lookback integration...")
        
        result = LookbackIntegrationResult()
        
        try:
            # Extract optimized lookback periods from optimization result
            if feature_lookback_optimization_result:
                optimized_periods = self._extract_optimized_periods(feature_lookback_optimization_result)
                if optimized_periods:
                    result.optimized_lookback_periods = optimized_periods
                    result.optimization_source = "feature_lookback_optimization"
                    result.integration_status = IntegrationStatus.SUCCESS
                    self.logger.info(f"✅ Extracted {len(optimized_periods)} optimized lookback periods")
                else:
                    self.logger.warning("⚠️ No optimized lookback periods found in optimization result")
                    result.integration_status = IntegrationStatus.FALLBACK
            else:
                self.logger.warning("⚠️ No feature_lookback_optimization result provided")
                result.integration_status = IntegrationStatus.FALLBACK
            
            # Apply fallback if needed
            if result.integration_status == IntegrationStatus.FALLBACK:
                self.logger.info("📊 Applying fallback lookback periods...")
                fallback_periods = self._apply_fallback_lookback_periods(feature_names)
                result.fallback_periods_used = fallback_periods
                result.optimized_lookback_periods.update(fallback_periods)
                result.optimization_source = "fallback_defaults"
                result.integration_status = IntegrationStatus.PARTIAL
            
            # Validate lookback periods
            validation_result = self._validate_lookback_periods(result.optimized_lookback_periods)
            result.validation_passed = validation_result['valid']
            result.optimization_quality_score = validation_result['quality_score']
            
            # Count optimized features
            result.features_optimized = len(result.optimized_lookback_periods)
            
            integration_time = time.time() - start_time
            result.integration_time = integration_time
            
            self.logger.info(f"✅ Optimized lookback integration completed in {integration_time:.3f}s")
            self.logger.info(f"📊 Integrated {result.features_optimized} lookback periods")
            self.logger.info(f"📊 Integration status: {result.integration_status.value}")
            self.logger.info(f"📊 Optimization quality score: {result.optimization_quality_score:.3f}")
            self.logger.info(f"📊 Validation passed: {result.validation_passed}")
            
            return result
            
        except Exception as e:
            integration_time = time.time() - start_time
            result.integration_time = integration_time
            result.integration_status = IntegrationStatus.FAILED
            
            self.logger.error(f"❌ Optimized lookback integration failed: {e}")
            
            # Apply emergency fallback
            emergency_fallback = self._apply_emergency_fallback()
            result.fallback_periods_used = emergency_fallback
            result.optimized_lookback_periods = emergency_fallback
            result.optimization_source = "emergency_fallback"
            result.integration_status = IntegrationStatus.FALLBACK
            
            return result
    
    def _extract_optimized_periods(
        self, 
        feature_lookback_optimization_result: Dict[str, Any]
    ) -> Dict[str, int]:
        """Extract optimized lookback periods from optimization result."""
        try:
            optimized_periods = {}
            
            # Try different possible structures in the optimization result
            optimization_results = feature_lookback_optimization_result.get('optimization_results', {})
            optimized_features = feature_lookback_optimization_result.get('optimized_features', {})
            
            # Extract from optimized_features
            if optimized_features:
                for feature_name, feature_data in optimized_features.items():
                    if isinstance(feature_data, dict):
                        lookback = feature_data.get('lookback', feature_data.get('best_lookback_period'))
                        if lookback and isinstance(lookback, (int, float)):
                            optimized_periods[feature_name] = int(lookback)
            
            # Extract from optimization_results
            if optimization_results:
                best_lookback_period = optimization_results.get('best_lookback_period')
                if best_lookback_period and isinstance(best_lookback_period, (int, float)):
                    # Apply to common feature types if no specific features found
                    if not optimized_periods:
                        for feature_type in ['rsi', 'sma', 'ema', 'macd']:
                            optimized_periods[feature_type] = int(best_lookback_period)
            
            # Extract from detailed results if available
            detailed_results = feature_lookback_optimization_result.get('detailed_results', {})
            if detailed_results:
                for feature_name, feature_data in detailed_results.items():
                    if isinstance(feature_data, dict):
                        lookback = feature_data.get('optimized_lookback', feature_data.get('best_period'))
                        if lookback and isinstance(lookback, (int, float)):
                            optimized_periods[feature_name] = int(lookback)
            
            self.logger.info(f"📊 Extracted {len(optimized_periods)} optimized lookback periods")
            return optimized_periods
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract optimized periods: {e}")
            return {}
    
    def _apply_fallback_lookback_periods(
        self, 
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Apply fallback lookback periods based on feature names."""
        try:
            fallback_periods = {}
            
            if feature_names:
                # Match feature names to default periods
                for feature_name in feature_names:
                    matched_period = self._match_feature_to_default_period(feature_name)
                    if matched_period:
                        fallback_periods[feature_name] = matched_period
                
                self.logger.info(f"📊 Matched {len(fallback_periods)} features to default lookback periods")
            else:
                # Use all default periods
                fallback_periods = self.default_lookback_periods.copy()
                self.logger.info(f"📊 Using all {len(fallback_periods)} default lookback periods")
            
            return fallback_periods
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply fallback lookback periods: {e}")
            return {}
    
    def _match_feature_to_default_period(self, feature_name: str) -> Optional[int]:
        """Match a feature name to a default lookback period."""
        try:
            feature_name_lower = feature_name.lower()
            
            # Direct matches
            for default_feature, period in self.default_lookback_periods.items():
                if default_feature in feature_name_lower:
                    return period
            
            # Pattern-based matching
            if any(pattern in feature_name_lower for pattern in ['rsi', 'relative_strength']):
                return self.default_lookback_periods['rsi']
            elif any(pattern in feature_name_lower for pattern in ['sma', 'simple_moving']):
                return self.default_lookback_periods['sma']
            elif any(pattern in feature_name_lower for pattern in ['ema', 'exponential_moving']):
                return self.default_lookback_periods['ema']
            elif any(pattern in feature_name_lower for pattern in ['macd']):
                return self.default_lookback_periods['macd']
            elif any(pattern in feature_name_lower for pattern in ['bollinger', 'bb']):
                return self.default_lookback_periods['bollinger']
            elif any(pattern in feature_name_lower for pattern in ['stochastic', 'stoch']):
                return self.default_lookback_periods['stochastic']
            elif any(pattern in feature_name_lower for pattern in ['williams', 'wr']):
                return self.default_lookback_periods['williams_r']
            elif any(pattern in feature_name_lower for pattern in ['cci', 'commodity_channel']):
                return self.default_lookback_periods['cci']
            elif any(pattern in feature_name_lower for pattern in ['atr', 'average_true_range']):
                return self.default_lookback_periods['atr']
            elif any(pattern in feature_name_lower for pattern in ['adx', 'average_directional']):
                return self.default_lookback_periods['adx']
            elif any(pattern in feature_name_lower for pattern in ['momentum']):
                return self.default_lookback_periods['price_momentum']
            elif any(pattern in feature_name_lower for pattern in ['volatility']):
                return self.default_lookback_periods['price_volatility']
            elif any(pattern in feature_name_lower for pattern in ['volume']):
                return self.default_lookback_periods['volume_momentum']
            elif any(pattern in feature_name_lower for pattern in ['trend']):
                return self.default_lookback_periods['price_trend']
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to match feature {feature_name} to default period: {e}")
            return None
    
    def _validate_lookback_periods(self, lookback_periods: Dict[str, int]) -> Dict[str, Any]:
        """Validate lookback periods for reasonableness."""
        try:
            validation_result = {
                'valid': True,
                'quality_score': 0.0,
                'issues': [],
                'recommendations': []
            }
            
            if not lookback_periods:
                validation_result['valid'] = False
                validation_result['issues'].append("No lookback periods provided")
                return validation_result
            
            # Check for reasonable ranges
            min_period = 1
            max_period = 200
            
            quality_scores = []
            for feature_name, period in lookback_periods.items():
                if not isinstance(period, (int, float)) or period < min_period or period > max_period:
                    validation_result['issues'].append(f"Invalid lookback period for {feature_name}: {period}")
                    validation_result['valid'] = False
                else:
                    # Calculate quality score based on period reasonableness
                    # Optimal range is typically 5-50 for most features
                    if 5 <= period <= 50:
                        quality_scores.append(1.0)
                    elif 1 <= period < 5 or 50 < period <= 100:
                        quality_scores.append(0.7)
                    else:
                        quality_scores.append(0.3)
            
            # Calculate overall quality score
            if quality_scores:
                validation_result['quality_score'] = float(np.mean(quality_scores))
            else:
                validation_result['quality_score'] = 0.0
            
            # Add recommendations
            if validation_result['quality_score'] < 0.7:
                validation_result['recommendations'].append("Consider re-optimizing lookback periods")
            
            if len(lookback_periods) < 5:
                validation_result['recommendations'].append("Consider optimizing more feature types")
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Lookback period validation failed: {e}")
            return {
                'valid': False,
                'quality_score': 0.0,
                'issues': [f"Validation error: {e}"],
                'recommendations': ["Review lookback period data"]
            }
    
    def _apply_emergency_fallback(self) -> Dict[str, int]:
        """Apply emergency fallback with minimal default periods."""
        try:
            emergency_periods = {
                'rsi': 14,
                'sma': 20,
                'ema': 12,
                'macd': 26,
                'momentum': 10,
                'volatility': 20,
                'volume': 10,
                'trend': 15
            }
            
            self.logger.warning("🚨 Applying emergency fallback lookback periods")
            return emergency_periods
            
        except Exception as e:
            self.logger.error(f"❌ Emergency fallback failed: {e}")
            return {'default': 20}  # Absolute fallback
    
    def get_integration_summary(self, result: LookbackIntegrationResult) -> Dict[str, Any]:
        """Get summary of integration results."""
        return {
            'integration_status': result.integration_status.value,
            'features_optimized': result.features_optimized,
            'optimization_quality_score': result.optimization_quality_score,
            'optimization_source': result.optimization_source,
            'validation_passed': result.validation_passed,
            'integration_time': result.integration_time,
            'fallback_periods_used': len(result.fallback_periods_used),
            'optimized_lookback_periods_sample': dict(list(result.optimized_lookback_periods.items())[:5])
        }