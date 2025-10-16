"""
Price Pattern ML Integration Framework

This module integrates mathematical price pattern definitions with the existing
src/research/clusters/ framework to create ML-ready targets and determine
which market dimensions are relevant for pattern prediction.

Integration Points:
- Extends src/research/clusters/dimension_economic_relevance.py
- Uses existing dimension discovery from src/research/clusters/
- Creates ML targets from mathematically defined patterns
- Filters dimensions by pattern-specific relevance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import system_logger

# Try to import existing framework components
try:
    from research.clusters.dimension_economic_relevance import (
        DimensionEconomicRelevanceAnalyzer,
        PriceActionInfluence
    )
    CLUSTERS_AVAILABLE = True
except ImportError:
    print("⚠️ src/research/clusters/ not available. Using standalone implementation.")
    CLUSTERS_AVAILABLE = False

class PricePattern(Enum):
    """Mathematically defined price patterns for ML applications."""

    # Momentum Patterns
    MOMENTUM_PERSISTENCE = "momentum_persistence"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MOMENTUM_DECAY = "momentum_decay"

    # Mean Reversion Patterns
    REVERSION_SPEED = "reversion_speed"
    OVERSOLD_BOUNCE = "oversold_bounce"
    OVERBOUGHT_DECLINE = "overbought_decline"

    # Volatility Patterns
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    VOLATILITY_CLUSTERING = "volatility_clustering"

    # Breakout Patterns
    CONFIRMED_BREAKOUT = "confirmed_breakout"
    FALSE_BREAKOUT = "false_breakout"
    RANGE_BOUND = "range_bound"

    # Trend Patterns
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"

@dataclass
class PatternDefinitionResult:
    """Result of pattern definition analysis."""
    pattern: PricePattern
    labels: pd.Series
    frequency: float  # How often pattern occurs
    predictability_score: float  # How predictable the pattern is
    economic_significance: float  # Economic value of pattern
    metadata: Dict[str, Any]

@dataclass
class PatternRelevanceResult:
    """Result of pattern-dimension relevance analysis."""
    pattern: PricePattern
    dimension_relevance: Dict[str, float]  # {dimension_name: relevance_score}
    best_dimensions: List[str]
    ml_accuracy: float  # ML prediction accuracy using best dimensions
    feature_importance: Dict[str, float]
    trading_implications: str

class MathematicalPatternDefinitions:
    """Mathematical definitions of price patterns for ML applications."""

    def __init__(self):
        self.logger = system_logger.getChild('PatternDefinitions')

    def momentum_persistence(self, prices: pd.Series,
                           momentum_window: int = 5,
                           persistence_window: int = 10,
                           momentum_threshold: float = 0.005) -> PatternDefinitionResult:
        """
        Pattern: Momentum persists for a specified number of periods.

        Mathematical Definition:
        If momentum(t) > threshold, then momentum(t+1:t+persistence_window)
        maintains same direction with decay rate < max_decay_rate
        """

        returns = prices.pct_change().fillna(0)
        momentum = returns.rolling(momentum_window).mean()

        labels = []

        for i in range(len(momentum) - persistence_window):
            current_momentum = momentum.iloc[i]

            if abs(current_momentum) > momentum_threshold:
                future_momentum = momentum.iloc[i+1:i+persistence_window+1]

                # Check direction persistence
                same_direction = (np.sign(future_momentum) == np.sign(current_momentum))
                persistence_rate = same_direction.sum() / len(future_momentum)

                # Check magnitude decay (gradual vs abrupt)
                magnitude_ratios = abs(future_momentum) / abs(current_momentum)
                gradual_decay = (magnitude_ratios > 0.3).sum() / len(magnitude_ratios)

                # Pattern exists if both conditions met
                pattern_exists = (persistence_rate >= 0.7) and (gradual_decay >= 0.6)
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)

        pattern_labels = pd.Series(labels, index=prices.index[:len(labels)])
        frequency = pattern_labels.sum() / len(pattern_labels)

        return PatternDefinitionResult(
            pattern=PricePattern.MOMENTUM_PERSISTENCE,
            labels=pattern_labels,
            frequency=frequency,
            predictability_score=self._calculate_predictability(pattern_labels),
            economic_significance=self._calculate_economic_significance(prices, pattern_labels),
            metadata={
                'momentum_window': momentum_window,
                'persistence_window': persistence_window,
                'momentum_threshold': momentum_threshold
            }
        )

    def reversion_speed(self, prices: pd.Series,
                       ma_window: int = 20,
                       deviation_threshold: float = 0.02,
                       reversion_window: int = 10) -> PatternDefinitionResult:
        """
        Pattern: Price reverts to mean within specified timeframe.

        Mathematical Definition:
        If |price(t) - MA(t)| > threshold, then price(t+k) is closer to MA(t)
        than price(t) for some k <= reversion_window
        """

        ma = prices.rolling(ma_window).mean()
        deviation = (prices - ma) / ma

        labels = []

        for i in range(ma_window, len(prices) - reversion_window):
            current_deviation = deviation.iloc[i]

            if abs(current_deviation) > deviation_threshold:
                current_price = prices.iloc[i]
                target_ma = ma.iloc[i]
                current_distance = abs(current_price - target_ma)

                # Look for reversion in future periods
                future_prices = prices.iloc[i+1:i+reversion_window+1]

                reversion_occurred = False
                for future_price in future_prices:
                    future_distance = abs(future_price - target_ma)
                    if future_distance < current_distance * 0.7:  # 30% closer
                        reversion_occurred = True
                        break

                labels.append(1 if reversion_occurred else 0)
            else:
                labels.append(0)

        pattern_labels = pd.Series(labels, index=prices.index[ma_window:ma_window+len(labels)])
        frequency = pattern_labels.sum() / len(pattern_labels)

        return PatternDefinitionResult(
            pattern=PricePattern.REVERSION_SPEED,
            labels=pattern_labels,
            frequency=frequency,
            predictability_score=self._calculate_predictability(pattern_labels),
            economic_significance=self._calculate_economic_significance(prices, pattern_labels),
            metadata={
                'ma_window': ma_window,
                'deviation_threshold': deviation_threshold,
                'reversion_window': reversion_window
            }
        )

    def volatility_expansion(self, prices: pd.Series,
                           vol_window: int = 20,
                           expansion_window: int = 10,
                           low_vol_percentile: float = 0.2,
                           high_vol_percentile: float = 0.8) -> PatternDefinitionResult:
        """
        Pattern: Low volatility followed by high volatility expansion.

        Mathematical Definition:
        If vol(t) < low_percentile, then vol(t+1:t+expansion_window) contains
        periods > high_percentile
        """

        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(vol_window).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)

        labels = []

        for i in range(100, len(volatility) - expansion_window):
            current_vol_percentile = vol_percentile.iloc[i]

            if current_vol_percentile < low_vol_percentile:
                future_vol_percentiles = vol_percentile.iloc[i+1:i+expansion_window+1]

                # Check for volatility expansion
                high_vol_periods = (future_vol_percentiles > high_vol_percentile).sum()
                expansion_rate = high_vol_periods / len(future_vol_percentiles)

                pattern_exists = expansion_rate >= 0.3  # 30% of periods high vol
                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)

        pattern_labels = pd.Series(labels, index=volatility.index[100:100+len(labels)])
        frequency = pattern_labels.sum() / len(pattern_labels)

        return PatternDefinitionResult(
            pattern=PricePattern.VOLATILITY_EXPANSION,
            labels=pattern_labels,
            frequency=frequency,
            predictability_score=self._calculate_predictability(pattern_labels),
            economic_significance=self._calculate_economic_significance(prices, pattern_labels),
            metadata={
                'vol_window': vol_window,
                'expansion_window': expansion_window,
                'low_vol_percentile': low_vol_percentile,
                'high_vol_percentile': high_vol_percentile
            }
        )

    def confirmed_breakout(self, prices: pd.Series,
                          bb_window: int = 20,
                          confirmation_window: int = 5,
                          min_continuation: float = 0.01) -> PatternDefinitionResult:
        """
        Pattern: Price breaks technical level and continues in breakout direction.

        Mathematical Definition:
        If price(t) > upper_band or price(t) < lower_band, then
        price(t+1:t+confirmation_window) continues beyond breakout level
        with minimum continuation magnitude
        """

        # Calculate Bollinger Bands
        ma = prices.rolling(bb_window).mean()
        std = prices.rolling(bb_window).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std

        labels = []

        for i in range(bb_window, len(prices) - confirmation_window):
            current_price = prices.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]

            # Check for breakout
            upper_breakout = current_price > current_upper
            lower_breakout = current_price < current_lower

            if upper_breakout or lower_breakout:
                future_prices = prices.iloc[i+1:i+confirmation_window+1]

                if upper_breakout:
                    # Confirm upward breakout
                    confirmation_count = (future_prices > current_upper).sum()
                    confirmation_rate = confirmation_count / len(future_prices)

                    max_future = future_prices.max()
                    continuation_magnitude = (max_future - current_price) / current_price

                    pattern_exists = (confirmation_rate >= 0.6) and (continuation_magnitude > min_continuation)

                elif lower_breakout:
                    # Confirm downward breakout
                    confirmation_count = (future_prices < current_lower).sum()
                    confirmation_rate = confirmation_count / len(future_prices)

                    min_future = future_prices.min()
                    continuation_magnitude = (current_price - min_future) / current_price

                    pattern_exists = (confirmation_rate >= 0.6) and (continuation_magnitude > min_continuation)

                labels.append(1 if pattern_exists else 0)
            else:
                labels.append(0)

        pattern_labels = pd.Series(labels, index=prices.index[bb_window:bb_window+len(labels)])
        frequency = pattern_labels.sum() / len(pattern_labels)

        return PatternDefinitionResult(
            pattern=PricePattern.CONFIRMED_BREAKOUT,
            labels=pattern_labels,
            frequency=frequency,
            predictability_score=self._calculate_predictability(pattern_labels),
            economic_significance=self._calculate_economic_significance(prices, pattern_labels),
            metadata={
                'bb_window': bb_window,
                'confirmation_window': confirmation_window,
                'min_continuation': min_continuation
            }
        )

    def _calculate_predictability(self, pattern_labels: pd.Series) -> float:
        """Calculate how predictable a pattern is (entropy-based measure)."""
        if len(pattern_labels) == 0:
            return 0.0

        # Calculate pattern frequency
        pattern_freq = pattern_labels.sum() / len(pattern_labels)

        if pattern_freq == 0 or pattern_freq == 1:
            return 0.0  # Completely predictable (never or always occurs)

        # Calculate entropy (lower entropy = more predictable)
        entropy = -pattern_freq * np.log2(pattern_freq) - (1 - pattern_freq) * np.log2(1 - pattern_freq)

        # Convert to predictability score (1 - normalized_entropy)
        max_entropy = 1.0  # Maximum entropy for binary variable
        predictability = 1.0 - (entropy / max_entropy)

        return predictability

    def _calculate_economic_significance(self, prices: pd.Series, pattern_labels: pd.Series) -> float:
        """Calculate economic significance of pattern."""
        if pattern_labels.sum() == 0:
            return 0.0

        returns = prices.pct_change().fillna(0)

        # Align returns with pattern labels
        aligned_returns = returns.loc[pattern_labels.index]

        # Calculate returns when pattern occurs vs when it doesn't
        pattern_returns = aligned_returns[pattern_labels == 1]
        no_pattern_returns = aligned_returns[pattern_labels == 0]

        if len(pattern_returns) == 0 or len(no_pattern_returns) == 0:
            return 0.0

        # Calculate difference in mean returns
        pattern_mean = pattern_returns.mean()
        no_pattern_mean = no_pattern_returns.mean()

        return abs(pattern_mean - no_pattern_mean) * 100  # Convert to percentage points

class PatternDimensionRelevanceAnalyzer:
    """Analyze which market dimensions are relevant for predicting specific patterns."""

    def __init__(self):
        self.logger = system_logger.getChild('PatternRelevanceAnalyzer')

    def analyze_pattern_dimension_relevance(self,
                                          market_data: pd.DataFrame,
                                          dimension_features: Dict[str, pd.DataFrame],
                                          pattern_result: PatternDefinitionResult) -> PatternRelevanceResult:
        """
        Analyze which dimensions are most relevant for predicting a specific pattern.

        Args:
            market_data: OHLCV market data
            dimension_features: Dictionary of {dimension_name: features_df}
            pattern_result: Result from pattern definition

        Returns:
            Analysis of dimension relevance for the pattern
        """

        self.logger.info(f"🎯 Analyzing dimension relevance for {pattern_result.pattern.value}")

        # Prepare features and target
        all_features = pd.concat(dimension_features.values(), axis=1)
        pattern_target = pattern_result.labels

        # Align features and target
        aligned_data = pd.concat([all_features, pattern_target], axis=1).dropna()

        if len(aligned_data) < 100:
            self.logger.warning(f"Insufficient data for {pattern_result.pattern.value} analysis")
            return self._create_empty_result(pattern_result.pattern)

        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]

        # Analyze relevance for each dimension
        dimension_relevance = {}
        dimension_accuracies = {}

        for dim_name, dim_features in dimension_features.items():
            # Get dimension features
            dim_feature_names = [col for col in X.columns if col in dim_features.columns]

            if not dim_feature_names:
                continue

            dim_X = X[dim_feature_names]

            # Calculate relevance metrics
            relevance_score = self._calculate_dimension_relevance(dim_X, y)
            dimension_relevance[dim_name] = relevance_score

            # Calculate ML prediction accuracy
            accuracy = self._calculate_ml_accuracy(dim_X, y)
            dimension_accuracies[dim_name] = accuracy

        # Find best dimensions
        best_dimensions = sorted(dimension_relevance.items(), key=lambda x: x[1], reverse=True)
        best_dimension_names = [dim[0] for dim in best_dimensions[:3]]

        # Calculate feature importance using best dimensions
        best_features = pd.concat([
            dimension_features[dim_name]
            for dim_name in best_dimension_names
        ], axis=1)

        aligned_best = pd.concat([best_features, pattern_target], axis=1).dropna()
        if len(aligned_best) > 50:
            feature_importance = self._calculate_feature_importance(
                aligned_best.iloc[:, :-1], aligned_best.iloc[:, -1]
            )
            ml_accuracy = max([dimension_accuracies.get(dim, 0.5) for dim in best_dimension_names])
        else:
            feature_importance = {}
            ml_accuracy = 0.5

        # Generate trading implications
        trading_implications = self._generate_trading_implications(
            pattern_result.pattern, best_dimension_names, ml_accuracy
        )

        return PatternRelevanceResult(
            pattern=pattern_result.pattern,
            dimension_relevance=dimension_relevance,
            best_dimensions=best_dimension_names,
            ml_accuracy=ml_accuracy,
            feature_importance=feature_importance,
            trading_implications=trading_implications
        )

    def _calculate_dimension_relevance(self, features: pd.DataFrame, target: pd.Series) -> float:
        """Calculate dimension relevance using multiple metrics."""

        if len(features) < 20:
            return 0.0

        relevance_scores = []

        # 1. Mutual information
        try:
            from sklearn.feature_selection import mutual_info_classif

            mi_scores = mutual_info_classif(features.fillna(0), target)
            relevance_scores.append(np.mean(mi_scores))
        except:
            pass

        # 2. Correlation-based relevance
        feature_target_corrs = []
        for col in features.columns:
            corr = abs(features[col].corr(target))
            if not np.isnan(corr):
                feature_target_corrs.append(corr)

        if feature_target_corrs:
            relevance_scores.append(np.mean(feature_target_corrs))

        # 3. Random Forest feature importance
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(features.fillna(0), target)
            relevance_scores.append(np.mean(rf.feature_importances_))
        except:
            pass

        return np.mean(relevance_scores) if relevance_scores else 0.0

    def _calculate_ml_accuracy(self, features: pd.DataFrame, target: pd.Series) -> float:
        """Calculate ML prediction accuracy using time series cross-validation."""

        if len(features) < 50 or target.sum() < 10:
            return 0.5  # Random baseline

        try:
            # Prepare data
            X = features.fillna(0).values
            y = target.values

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            accuracies = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_train_scaled, y_train)

                # Predict
                y_pred = rf.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            return np.mean(accuracies)

        except Exception as e:
            self.logger.warning(f"ML accuracy calculation failed: {e}")
            return 0.5

    def _calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using Random Forest."""

        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features.fillna(0), target)

            importance_dict = {}
            for feature, importance in zip(features.columns, rf.feature_importances_):
                importance_dict[feature] = float(importance)

            return importance_dict

        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}

    def _generate_trading_implications(self, pattern: PricePattern,
                                     best_dimensions: List[str],
                                     ml_accuracy: float) -> str:
        """Generate trading implications based on analysis results."""

        if ml_accuracy > 0.65:
            strength = "Strong"
        elif ml_accuracy > 0.58:
            strength = "Moderate"
        else:
            strength = "Weak"

        base_implication = f"{strength} predictive signal for {pattern.value}"

        if ml_accuracy > 0.6:
            dimension_desc = ", ".join(best_dimensions[:2])

            if pattern in [PricePattern.MOMENTUM_PERSISTENCE, PricePattern.MOMENTUM_ACCELERATION]:
                return f"{base_implication}. Use {dimension_desc} dimensions to enhance momentum strategies."
            elif pattern in [PricePattern.REVERSION_SPEED, PricePattern.OVERSOLD_BOUNCE]:
                return f"{base_implication}. Use {dimension_desc} dimensions to time mean reversion entries."
            elif pattern in [PricePattern.VOLATILITY_EXPANSION, PricePattern.VOLATILITY_CLUSTERING]:
                return f"{base_implication}. Use {dimension_desc} dimensions for volatility forecasting."
            elif pattern in [PricePattern.CONFIRMED_BREAKOUT, PricePattern.FALSE_BREAKOUT]:
                return f"{base_implication}. Use {dimension_desc} dimensions to confirm breakout signals."
            else:
                return f"{base_implication}. Use {dimension_desc} dimensions for pattern-based trading."
        else:
            return f"{base_implication}. Limited trading utility - consider as supporting indicator only."

    def _create_empty_result(self, pattern: PricePattern) -> PatternRelevanceResult:
        """Create empty result for failed analysis."""
        return PatternRelevanceResult(
            pattern=pattern,
            dimension_relevance={},
            best_dimensions=[],
            ml_accuracy=0.5,
            feature_importance={},
            trading_implications="Insufficient data for analysis"
        )

class PatternMLIntegrationOrchestrator:
    """Main orchestrator for pattern-ML integration."""

    def __init__(self):
        self.logger = system_logger.getChild('PatternMLIntegration')
        self.pattern_definitions = MathematicalPatternDefinitions()
        self.relevance_analyzer = PatternDimensionRelevanceAnalyzer()

    def run_comprehensive_pattern_analysis(self,
                                         market_data: pd.DataFrame,
                                         dimension_features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run comprehensive pattern analysis integrating with existing research framework.

        Args:
            market_data: OHLCV market data
            dimension_features: Dictionary from existing dimension discovery

        Returns:
            Complete analysis results
        """

        self.logger.info("🎯 Starting comprehensive pattern-ML integration analysis")

        results = {
            'pattern_definitions': {},
            'pattern_relevance': {},
            'ml_targets': {},
            'dimension_rankings': {},
            'trading_recommendations': []
        }

        # Define key patterns for analysis
        key_patterns = [
            ('momentum_persistence', self.pattern_definitions.momentum_persistence),
            ('reversion_speed', self.pattern_definitions.reversion_speed),
            ('volatility_expansion', self.pattern_definitions.volatility_expansion),
            ('confirmed_breakout', self.pattern_definitions.confirmed_breakout)
        ]

        # Analyze each pattern
        for pattern_name, pattern_func in key_patterns:
            self.logger.info(f"📊 Analyzing {pattern_name}")

            try:
                # Define pattern mathematically
                pattern_result = pattern_func(market_data['close'])
                results['pattern_definitions'][pattern_name] = pattern_result

                # Analyze dimension relevance for this pattern
                relevance_result = self.relevance_analyzer.analyze_pattern_dimension_relevance(
                    market_data, dimension_features, pattern_result
                )
                results['pattern_relevance'][pattern_name] = relevance_result

                # Create ML target
                results['ml_targets'][pattern_name] = pattern_result.labels

                # Log key findings
                self.logger.info(f"   Pattern frequency: {pattern_result.frequency:.3f}")
                self.logger.info(f"   Best dimension: {relevance_result.best_dimensions[0] if relevance_result.best_dimensions else 'None'}")
                self.logger.info(f"   ML accuracy: {relevance_result.ml_accuracy:.3f}")

            except Exception as e:
                self.logger.error(f"   Failed to analyze {pattern_name}: {e}")
                continue

        # Generate dimension rankings across all patterns
        results['dimension_rankings'] = self._calculate_overall_dimension_rankings(
            results['pattern_relevance']
        )

        # Generate trading recommendations
        results['trading_recommendations'] = self._generate_comprehensive_trading_recommendations(
            results
        )

        self.logger.info("✅ Comprehensive pattern analysis completed")
        return results

    def _calculate_overall_dimension_rankings(self,
                                            pattern_relevance: Dict[str, PatternRelevanceResult]) -> Dict[str, float]:
        """Calculate overall dimension rankings across all patterns."""

        dimension_scores = {}

        for pattern_name, relevance_result in pattern_relevance.items():
            for dim_name, relevance_score in relevance_result.dimension_relevance.items():
                if dim_name not in dimension_scores:
                    dimension_scores[dim_name] = []
                dimension_scores[dim_name].append(relevance_score)

        # Calculate average relevance across patterns
        dimension_rankings = {}
        for dim_name, scores in dimension_scores.items():
            dimension_rankings[dim_name] = np.mean(scores)

        return dict(sorted(dimension_rankings.items(), key=lambda x: x[1], reverse=True))

    def _generate_comprehensive_trading_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive trading recommendations."""

        recommendations = []

        # Overall pattern analysis
        pattern_relevance = results.get('pattern_relevance', {})
        dimension_rankings = results.get('dimension_rankings', {})

        high_accuracy_patterns = [
            name for name, result in pattern_relevance.items()
            if result.ml_accuracy > 0.6
        ]

        if high_accuracy_patterns:
            recommendations.append(f"✅ HIGH-CONFIDENCE PATTERNS: {', '.join(high_accuracy_patterns)}")
            recommendations.append("   → Develop ML models for these patterns with high prediction accuracy")

        # Top dimensions
        if dimension_rankings:
            top_dimensions = list(dimension_rankings.keys())[:3]
            recommendations.append(f"🏆 TOP DIMENSIONS: {', '.join(top_dimensions)}")
            recommendations.append("   → Focus feature engineering and model development on these dimensions")

        # Pattern-specific recommendations
        for pattern_name, relevance_result in pattern_relevance.items():
            if relevance_result.ml_accuracy > 0.6:
                recommendations.append(f"\n📊 {pattern_name.upper()}:")
                recommendations.append(f"   - {relevance_result.trading_implications}")

                if relevance_result.best_dimensions:
                    best_dims = ', '.join(relevance_result.best_dimensions[:2])
                    recommendations.append(f"   - Key dimensions: {best_dims}")

        # Integration with existing framework
        recommendations.append(f"\n🔗 INTEGRATION WITH EXISTING RESEARCH:")
        recommendations.append("   - Use these patterns as supervised learning targets")
        recommendations.append("   - Filter existing dimension features by pattern relevance")
        recommendations.append("   - Enhance regime clustering with pattern-based validation")

        return recommendations

    def generate_ml_dataset(self,
                          market_data: pd.DataFrame,
                          dimension_features: Dict[str, pd.DataFrame],
                          analysis_results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate ML-ready dataset with features and targets.

        Returns:
            features_df: Features filtered by pattern relevance
            targets_df: Pattern-based targets for supervised learning
        """

        # Combine all ML targets
        ml_targets = analysis_results.get('ml_targets', {})
        targets_df = pd.DataFrame(ml_targets)

        # Get dimension rankings
        dimension_rankings = analysis_results.get('dimension_rankings', {})
        top_dimensions = [dim for dim, score in dimension_rankings.items() if score > 0.3]

        # Filter features by relevance
        relevant_features = []
        for dim_name in top_dimensions:
            if dim_name in dimension_features:
                dim_features = dimension_features[dim_name]
                # Add dimension prefix to column names
                dim_features_renamed = dim_features.add_prefix(f"{dim_name}_")
                relevant_features.append(dim_features_renamed)

        if relevant_features:
            features_df = pd.concat(relevant_features, axis=1)
        else:
            # Fallback: use all features
            features_df = pd.concat(dimension_features.values(), axis=1)

        # Align features and targets
        common_index = features_df.index.intersection(targets_df.index)
        features_df = features_df.loc[common_index]
        targets_df = targets_df.loc[common_index]

        return features_df, targets_df

# Example usage function
def run_pattern_ml_integration_example():
    """Example of how to use the pattern-ML integration framework."""

    print("Pattern-ML Integration Framework")
    print("===============================")
    print()
    print("This framework:")
    print("1. Defines price patterns mathematically")
    print("2. Determines which dimensions predict each pattern")
    print("3. Creates ML-ready targets and filtered features")
    print("4. Integrates with existing src/research/clusters/ framework")
    print()
    print("Key benefits:")
    print("- Precise pattern definitions for reproducible research")
    print("- Dimension filtering based on pattern relevance")
    print("- ML targets for supervised learning")
    print("- Enhanced economic validation")
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = PatternMLIntegrationOrchestrator()")
    print("results = orchestrator.run_comprehensive_pattern_analysis(")
    print("    market_data, dimension_features")
    print(")")
    print("features_df, targets_df = orchestrator.generate_ml_dataset(")
    print("    market_data, dimension_features, results")
    print(")")
    print("```")

if __name__ == "__main__":
    run_pattern_ml_integration_example()
