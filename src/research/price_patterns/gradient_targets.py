"""
Gradient-Based Pattern Target Generation

This module extends binary pattern labels with gradient-based intensity measurements.
Instead of just [0,1,0,1,...], we generate continuous values [0.0,0.8,0.2,0.9,...]
that measure the strength/intensity of each pattern occurrence.

Key Benefits:
1. Regression targets (not just classification)
2. Pattern strength measurement
3. Nuanced ML training (strong vs weak patterns)
4. Better signal quality assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.logger import system_logger


class IntensityMeasurementMethod(Enum):
    """Methods for measuring pattern intensity."""
    MAGNITUDE_BASED = "magnitude_based"
    PERSISTENCE_BASED = "persistence_based"
    QUALITY_BASED = "quality_based"
    COMPOSITE_SCORE = "composite_score"


@dataclass
class PatternIntensityMeasurement:
    """Measurement of pattern intensity."""
    pattern_name: str
    binary_labels: pd.Series  # Traditional [0,1,0,1,...]
    intensity_gradients: pd.Series  # New [0.0,0.8,0.2,0.9,...]
    intensity_method: IntensityMeasurementMethod
    intensity_statistics: Dict[str, float]
    correlation_with_outcomes: float  # How well intensity predicts future price moves


class GradientPatternTargetGenerator:
    """Generate gradient-based pattern targets for ML."""
    
    def __init__(self):
        self.logger = system_logger.getChild('GradientTargetGenerator')
    
    def generate_momentum_persistence_gradients(self, 
                                              prices: pd.Series,
                                              momentum_window: int = 5,
                                              persistence_window: int = 10) -> PatternIntensityMeasurement:
        """Generate gradient targets for momentum persistence patterns."""
        
        self.logger.info("🚀 Generating momentum persistence gradients")
        
        # Calculate momentum
        momentum = (prices - prices.shift(momentum_window)) / prices.shift(momentum_window)
        momentum = momentum.fillna(0)
        
        binary_labels = []
        intensity_gradients = []
        
        for i in range(len(momentum) - persistence_window):
            current_momentum = momentum.iloc[i]
            
            if abs(current_momentum) > 0.005:  # Minimum threshold
                future_momentum = momentum.iloc[i+1:i+persistence_window+1]
                
                # Calculate persistence components
                direction_persistence = (
                    np.sign(future_momentum) == np.sign(current_momentum)
                ).sum() / len(future_momentum)
                
                magnitude_ratios = abs(future_momentum) / abs(current_momentum)
                magnitude_persistence = (magnitude_ratios > 0.3).sum() / len(magnitude_ratios)
                
                # Binary label (traditional)
                binary_pattern = (direction_persistence >= 0.7 and magnitude_persistence >= 0.6)
                
                # Intensity gradient (new)
                intensity = (
                    abs(current_momentum) * 20 *  # Scale momentum magnitude
                    direction_persistence *        # Weight by direction consistency
                    magnitude_persistence         # Weight by magnitude persistence
                )
                intensity = min(intensity, 1.0)  # Cap at 1.0
                
                binary_labels.append(1 if binary_pattern else 0)
                intensity_gradients.append(intensity)
            else:
                binary_labels.append(0)
                intensity_gradients.append(0.0)
        
        binary_series = pd.Series(binary_labels, index=prices.index[:len(binary_labels)])
        intensity_series = pd.Series(intensity_gradients, index=prices.index[:len(intensity_gradients)])
        
        # Calculate intensity statistics
        intensity_stats = self._calculate_intensity_statistics(intensity_series)
        
        # Calculate correlation with future returns
        correlation = self._calculate_outcome_correlation(intensity_series, prices)
        
        return PatternIntensityMeasurement(
            pattern_name="momentum_persistence",
            binary_labels=binary_series,
            intensity_gradients=intensity_series,
            intensity_method=IntensityMeasurementMethod.COMPOSITE_SCORE,
            intensity_statistics=intensity_stats,
            correlation_with_outcomes=correlation
        )
    
    def generate_reversion_speed_gradients(self,
                                         prices: pd.Series,
                                         lookback_window: int = 20,
                                         reversion_window: int = 15) -> PatternIntensityMeasurement:
        """Generate gradient targets for price reversion patterns."""
        
        self.logger.info("🔄 Generating price reversion gradients")
        
        binary_labels = []
        intensity_gradients = []
        
        for i in range(lookback_window, len(prices) - reversion_window):
            reference_level = prices.iloc[i - lookback_window]
            current_price = prices.iloc[i]
            
            # Calculate deviation
            deviation = abs(current_price - reference_level) / reference_level
            
            if deviation > 0.02:  # Minimum deviation threshold
                # Look for reversion
                future_prices = prices.iloc[i+1:i+reversion_window+1]
                
                reversion_occurred = False
                reversion_speed = 0
                reversion_magnitude = 0
                
                for j, future_price in enumerate(future_prices):
                    future_deviation = abs(future_price - reference_level) / reference_level
                    if future_deviation < 0.5 * deviation:
                        reversion_occurred = True
                        reversion_speed = 1.0 / (j + 1)  # Faster = higher score
                        reversion_magnitude = (deviation - future_deviation) / deviation
                        break
                
                # Binary label
                binary_pattern = reversion_occurred
                
                # Intensity gradient
                if reversion_occurred:
                    intensity = (
                        deviation * 10 *          # Scale deviation magnitude
                        reversion_speed *         # Weight by speed
                        reversion_magnitude       # Weight by reversion completeness
                    )
                    intensity = min(intensity, 1.0)
                else:
                    # Even failed reversions have some intensity based on deviation
                    intensity = min(deviation * 5, 0.3)  # Cap failed reversions at 0.3
                
                binary_labels.append(1 if binary_pattern else 0)
                intensity_gradients.append(intensity)
            else:
                binary_labels.append(0)
                intensity_gradients.append(0.0)
        
        binary_series = pd.Series(binary_labels, index=prices.index[lookback_window:lookback_window+len(binary_labels)])
        intensity_series = pd.Series(intensity_gradients, index=prices.index[lookback_window:lookback_window+len(intensity_gradients)])
        
        intensity_stats = self._calculate_intensity_statistics(intensity_series)
        correlation = self._calculate_outcome_correlation(intensity_series, prices)
        
        return PatternIntensityMeasurement(
            pattern_name="price_reversion",
            binary_labels=binary_series,
            intensity_gradients=intensity_series,
            intensity_method=IntensityMeasurementMethod.COMPOSITE_SCORE,
            intensity_statistics=intensity_stats,
            correlation_with_outcomes=correlation
        )
    
    def generate_breakout_strength_gradients(self,
                                           prices: pd.Series,
                                           range_window: int = 30,
                                           continuation_window: int = 8) -> PatternIntensityMeasurement:
        """Generate gradient targets for range breakout patterns."""
        
        self.logger.info("📊 Generating range breakout gradients")
        
        binary_labels = []
        intensity_gradients = []
        
        for i in range(range_window, len(prices) - continuation_window):
            # Define range
            recent_prices = prices.iloc[i-range_window:i]
            range_high = recent_prices.max()
            range_low = recent_prices.min()
            range_size = (range_high - range_low) / range_low
            
            current_price = prices.iloc[i]
            
            # Check for established range
            if range_size < 0.08:
                # Check for breakout
                upper_breakout = current_price > range_high
                lower_breakout = current_price < range_low
                
                if upper_breakout or lower_breakout:
                    # Calculate breakout magnitude
                    if upper_breakout:
                        breakout_magnitude = (current_price - range_high) / range_high
                    else:
                        breakout_magnitude = (range_low - current_price) / range_low
                    
                    if breakout_magnitude > 0.005:  # Minimum breakout
                        # Check continuation
                        future_prices = prices.iloc[i+1:i+continuation_window+1]
                        
                        if upper_breakout:
                            continuation_count = (future_prices > range_high).sum()
                        else:
                            continuation_count = (future_prices < range_low).sum()
                        
                        continuation_strength = continuation_count / len(future_prices)
                        
                        # Binary label
                        binary_pattern = continuation_strength >= 0.6
                        
                        # Intensity gradient
                        range_quality = 1.0 - (range_size / 0.08)  # Tighter range = higher quality
                        intensity = (
                            breakout_magnitude * 20 *    # Scale breakout magnitude
                            continuation_strength *      # Weight by continuation
                            range_quality               # Weight by range quality
                        )
                        intensity = min(intensity, 1.0)
                        
                        binary_labels.append(1 if binary_pattern else 0)
                        intensity_gradients.append(intensity)
                    else:
                        binary_labels.append(0)
                        intensity_gradients.append(0.0)
                else:
                    binary_labels.append(0)
                    intensity_gradients.append(0.0)
            else:
                binary_labels.append(0)
                intensity_gradients.append(0.0)
        
        binary_series = pd.Series(binary_labels, index=prices.index[range_window:range_window+len(binary_labels)])
        intensity_series = pd.Series(intensity_gradients, index=prices.index[range_window:range_window+len(intensity_gradients)])
        
        intensity_stats = self._calculate_intensity_statistics(intensity_series)
        correlation = self._calculate_outcome_correlation(intensity_series, prices)
        
        return PatternIntensityMeasurement(
            pattern_name="range_breakout",
            binary_labels=binary_series,
            intensity_gradients=intensity_series,
            intensity_method=IntensityMeasurementMethod.COMPOSITE_SCORE,
            intensity_statistics=intensity_stats,
            correlation_with_outcomes=correlation
        )
    
    def generate_all_gradient_targets(self, prices: pd.Series) -> Dict[str, PatternIntensityMeasurement]:
        """Generate gradient targets for all patterns."""
        
        self.logger.info("🎯 Generating all gradient-based pattern targets")
        
        results = {}
        
        # Generate gradient targets for each pattern type
        try:
            results['momentum_persistence'] = self.generate_momentum_persistence_gradients(prices)
            self.logger.info("   ✅ Momentum persistence gradients generated")
        except Exception as e:
            self.logger.error(f"   ❌ Momentum persistence failed: {e}")
        
        try:
            results['price_reversion'] = self.generate_reversion_speed_gradients(prices)
            self.logger.info("   ✅ Price reversion gradients generated")
        except Exception as e:
            self.logger.error(f"   ❌ Price reversion failed: {e}")
        
        try:
            results['range_breakout'] = self.generate_breakout_strength_gradients(prices)
            self.logger.info("   ✅ Range breakout gradients generated")
        except Exception as e:
            self.logger.error(f"   ❌ Range breakout failed: {e}")
        
        self.logger.info(f"🎯 Gradient target generation completed: {len(results)} patterns")
        return results
    
    def _calculate_intensity_statistics(self, intensity_series: pd.Series) -> Dict[str, float]:
        """Calculate statistics for intensity gradients."""
        
        non_zero_intensities = intensity_series[intensity_series > 0]
        
        if len(non_zero_intensities) == 0:
            return {
                'mean_intensity': 0.0,
                'max_intensity': 0.0,
                'intensity_std': 0.0,
                'intensity_range': 0.0,
                'non_zero_count': 0
            }
        
        return {
            'mean_intensity': float(non_zero_intensities.mean()),
            'max_intensity': float(non_zero_intensities.max()),
            'intensity_std': float(non_zero_intensities.std()),
            'intensity_range': float(non_zero_intensities.max() - non_zero_intensities.min()),
            'non_zero_count': len(non_zero_intensities)
        }
    
    def _calculate_outcome_correlation(self, intensity_series: pd.Series, prices: pd.Series) -> float:
        """Calculate correlation between pattern intensity and future price movements."""
        
        if intensity_series.sum() == 0:
            return 0.0
        
        # Calculate future returns
        returns = prices.pct_change().fillna(0)
        
        # Align intensity with future returns (5-period forward)
        future_returns = returns.shift(-5)
        aligned_data = pd.concat([intensity_series, future_returns], axis=1).dropna()
        
        if len(aligned_data) < 20:
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
            return float(abs(correlation)) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def export_ml_ready_targets(self, 
                              gradient_results: Dict[str, PatternIntensityMeasurement]) -> Dict[str, pd.DataFrame]:
        """Export ML-ready targets in different formats."""
        
        exports = {
            'binary_only': pd.DataFrame(),
            'intensity_only': pd.DataFrame(),
            'combined': pd.DataFrame()
        }
        
        # Binary labels only
        binary_data = {}
        for pattern_name, measurement in gradient_results.items():
            binary_data[pattern_name] = measurement.binary_labels
        
        if binary_data:
            exports['binary_only'] = pd.DataFrame(binary_data)
        
        # Intensity gradients only
        intensity_data = {}
        for pattern_name, measurement in gradient_results.items():
            intensity_data[f"{pattern_name}_intensity"] = measurement.intensity_gradients
        
        if intensity_data:
            exports['intensity_only'] = pd.DataFrame(intensity_data)
        
        # Combined (both binary and intensity)
        combined_data = {**binary_data}
        for pattern_name, measurement in gradient_results.items():
            combined_data[f"{pattern_name}_intensity"] = measurement.intensity_gradients
        
        if combined_data:
            exports['combined'] = pd.DataFrame(combined_data)
        
        return exports
    
    def generate_gradient_report(self, 
                               gradient_results: Dict[str, PatternIntensityMeasurement]) -> str:
        """Generate report on gradient-based pattern targets."""
        
        report = []
        report.append("# Gradient-Based Pattern Target Report")
        report.append("=" * 60)
        report.append("")
        report.append("**Innovation**: Pattern intensity gradients for enhanced ML training")
        report.append("**Traditional**: Binary labels [0,1,0,1,...]")
        report.append("**Enhanced**: Intensity gradients [0.0,0.8,0.2,0.9,...]")
        report.append("")
        
        # Summary
        total_patterns = len(gradient_results)
        patterns_with_intensity = sum(
            1 for measurement in gradient_results.values()
            if measurement.intensity_statistics['non_zero_count'] > 0
        )
        
        report.append("## Gradient Target Summary")
        report.append("")
        report.append(f"- **Total Patterns**: {total_patterns}")
        report.append(f"- **Patterns with Intensity**: {patterns_with_intensity}")
        report.append("")
        
        # Pattern analysis
        for pattern_name, measurement in gradient_results.items():
            report.append(f"### {pattern_name.replace('_', ' ').title()}")
            report.append("")
            
            # Binary vs Gradient comparison
            binary_count = measurement.binary_labels.sum()
            intensity_count = measurement.intensity_statistics['non_zero_count']
            
            report.append(f"**Binary Labels**: {binary_count} pattern occurrences")
            report.append(f"**Intensity Gradients**: {intensity_count} non-zero intensities")
            
            if intensity_count > binary_count:
                additional = intensity_count - binary_count
                report.append(f"**Enhancement**: {additional} additional weak patterns captured")
            
            # Intensity statistics
            stats = measurement.intensity_statistics
            if stats['non_zero_count'] > 0:
                report.append(f"**Intensity Range**: {stats['intensity_range']:.3f}")
                report.append(f"**Average Intensity**: {stats['mean_intensity']:.3f}")
                report.append(f"**Max Intensity**: {stats['max_intensity']:.3f}")
            
            # Predictive correlation
            report.append(f"**Future Return Correlation**: {measurement.correlation_with_outcomes:.3f}")
            
            report.append("")
        
        # ML Training Benefits
        report.append("## ML Training Benefits")
        report.append("")
        report.append("### Binary Labels (Classification)")
        report.append("```python")
        report.append("# Traditional approach")
        report.append("X = market_features")
        report.append("y = binary_labels  # [0,1,0,1,...]")
        report.append("model = RandomForestClassifier()")
        report.append("model.fit(X, y)")
        report.append("```")
        report.append("")
        
        report.append("### Intensity Gradients (Regression)")
        report.append("```python")
        report.append("# Enhanced approach")
        report.append("X = market_features")
        report.append("y = intensity_gradients  # [0.0,0.8,0.2,0.9,...]")
        report.append("model = RandomForestRegressor()")
        report.append("model.fit(X, y)")
        report.append("```")
        report.append("")
        
        report.append("### Multi-Task Learning")
        report.append("```python")
        report.append("# Combined approach")
        report.append("X = market_features")
        report.append("y_binary = binary_labels")
        report.append("y_intensity = intensity_gradients")
        report.append("")
        report.append("# Train models for both tasks")
        report.append("classifier = train_pattern_classifier(X, y_binary)")
        report.append("regressor = train_pattern_regressor(X, y_intensity)")
        report.append("```")
        
        # Key advantages
        report.append("## Key Advantages of Gradient Targets")
        report.append("")
        report.append("✅ **Nuanced Training**: Distinguish strong vs weak patterns")
        report.append("✅ **Regression Targets**: Enable continuous prediction")
        report.append("✅ **Signal Quality**: Measure pattern strength/confidence")
        report.append("✅ **Better Predictions**: More information for ML models")
        report.append("✅ **Risk Management**: Scale positions by pattern intensity")
        
        return "\n".join(report)


# Example usage
def run_gradient_targets_example():
    """Example of gradient-based pattern target generation."""
    
    print("Gradient-Based Pattern Targets")
    print("=============================")
    print()
    print("🎯 ENHANCEMENT: Pattern intensity gradients")
    print("   Traditional: [0,1,0,1,0,1,...]")
    print("   Enhanced: [0.0,0.8,0.2,0.9,0.1,0.7,...]")
    print()
    print("Benefits:")
    print("1. Regression targets (not just classification)")
    print("2. Pattern strength measurement")
    print("3. Nuanced ML training")
    print("4. Better signal quality assessment")
    print()
    print("Usage:")
    print("```python")
    print("generator = GradientPatternTargetGenerator()")
    print("gradients = generator.generate_all_gradient_targets(prices)")
    print("ml_targets = generator.export_ml_ready_targets(gradients)")
    print("```")


if __name__ == "__main__":
    run_gradient_targets_example()