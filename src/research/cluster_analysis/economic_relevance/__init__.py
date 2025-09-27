"""
Economic Relevance Analysis

Analyze the economic relevance of implicit market dimensions through their 
relationships with price patterns and market states.

Main Components:
- EconomicRelevanceAnalyzer: Main orchestrator for relevance analysis
- PatternDimensionAnalyzer: Pattern-dimension relationships
- MarketStateRelevanceAnalyzer: Market state pattern analysis
- CausalAnalyzer: Causal relationship identification
- TradingSignificanceAnalyzer: Economic value measurement

Usage:
    from src.research.cluster_analysis.economic_relevance import (
        EconomicRelevanceAnalyzer,
        PatternDimensionAnalyzer,
        CausalAnalyzer
    )
"""

# NOTE: this module is executed under Python 3.10+, but we enable postponed evaluation
# of annotations to keep compatibility with older tooling versions.
from __future__ import annotations

# Import actual implementations
import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

class EconomicRelevanceAnalyzer:
    """Main orchestrator for economic relevance analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.relevance_results = {}
    
    def analyze_pattern_dimension_relevance(self, patterns, dimensions, market_states):
        """Analyze comprehensive pattern-dimension-state relationships."""
        
        # 1. Pattern-Dimension Relevance Matrix
        pattern_dimension_matrix = self._calculate_pattern_dimension_matrix(patterns, dimensions)
        
        # 2. Market State Effects
        market_state_effects = self._analyze_market_state_effects(patterns, market_states)
        
        # 3. Economic Significance
        economic_significance = self._measure_economic_significance(patterns, dimensions, market_states)
        
        # 4. Trading Recommendations
        trading_recommendations = self._generate_trading_recommendations(
            pattern_dimension_matrix, market_state_effects, economic_significance
        )
        
        results = {
            'pattern_dimension_matrix': pattern_dimension_matrix,
            'market_state_effects': market_state_effects,
            'causal_relationships': {},  # Would implement causal analysis
            'economic_significance': economic_significance,
            'trading_recommendations': trading_recommendations
        }
        
        self.relevance_results = results
        return results
    
    def _calculate_pattern_dimension_matrix(self, patterns, dimensions):
        """Calculate which dimensions predict which patterns."""
        
        relevance_matrix = {}
        
        for pattern_name, pattern_data in patterns.items():
            pattern_relevance = {}
            
            for dimension_name, dimension_features in dimensions.items():
                # Calculate relevance score using ML prediction accuracy
                relevance_score = self._calculate_ml_relevance(
                    dimension_features, pattern_data['labels']
                )
                pattern_relevance[dimension_name] = relevance_score
            
            relevance_matrix[pattern_name] = pattern_relevance
        
        return pd.DataFrame(relevance_matrix).T  # Transpose for patterns as rows
    
    def _calculate_ml_relevance(self, features, target):
        """Calculate ML-based relevance score."""
        
        try:
            # Align data
            common_index = features.index.intersection(target.index)
            if len(common_index) < 50:
                return 0.0
            
            X = features.loc[common_index].fillna(0)
            y = target.loc[common_index]
            
            # Use Random Forest for prediction
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Cross-validation score
            scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
            relevance_score = scores.mean()
            
            # Normalize to 0-1 range (accuracy above 0.5 baseline)
            relevance_score = max(0, (relevance_score - 0.5) * 2)
            
            return relevance_score
            
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception(
                "Failed to calculate ML relevance", exc_info=(type(exc), exc, exc.__traceback__)
            )
            return 0.0
    
    def _analyze_market_state_effects(self, patterns, market_states):
        """Analyze how patterns behave in different market states."""
        
        state_effects = {}
        
        if 'labels' not in market_states:
            return state_effects
        
        state_labels = market_states['labels']
        unique_states = state_labels.unique()
        
        for pattern_name, pattern_data in patterns.items():
            pattern_labels = pattern_data['labels']
            
            # Align data
            common_index = pattern_labels.index.intersection(state_labels.index)
            if len(common_index) < 50:
                continue
            
            aligned_patterns = pattern_labels.loc[common_index]
            aligned_states = state_labels.loc[common_index]
            
            pattern_state_effects = {}
            
            for state in unique_states:
                state_mask = aligned_states == state
                state_patterns = aligned_patterns[state_mask]
                
                if len(state_patterns) > 10:
                    pattern_frequency = state_patterns.mean()
                    pattern_intensity = pattern_data.get('intensity', pd.Series(0, index=pattern_labels.index))
                    state_intensity = pattern_intensity.loc[common_index][state_mask].mean()
                    
                    pattern_state_effects[f'state_{state}'] = {
                        'frequency': pattern_frequency,
                        'intensity': state_intensity,
                        'sample_size': len(state_patterns)
                    }
            
            state_effects[pattern_name] = pattern_state_effects
        
        return state_effects
    
    def _measure_economic_significance(self, patterns, dimensions, market_states):
        """Measure economic significance for trading."""
        
        economic_metrics = {}
        
        for pattern_name, pattern_data in patterns.items():
            pattern_labels = pattern_data['labels']
            
            # Simple economic metrics
            pattern_frequency = pattern_labels.mean()
            
            # Economic significance based on frequency and predictability
            if 0.05 <= pattern_frequency <= 0.25:  # Reasonable frequency
                frequency_score = 1.0
            elif 0.02 <= pattern_frequency <= 0.40:
                frequency_score = 0.7
            else:
                frequency_score = 0.3
            
            # Combine with pattern intensity if available
            if 'intensity' in pattern_data:
                intensity_mean = pattern_data['intensity'].mean()
                intensity_score = min(1.0, intensity_mean * 2)  # Normalize
            else:
                intensity_score = 0.5
            
            economic_score = (frequency_score * 0.6) + (intensity_score * 0.4)
            
            economic_metrics[pattern_name] = {
                'frequency': pattern_frequency,
                'frequency_score': frequency_score,
                'intensity_score': intensity_score,
                'economic_score': economic_score,
                'is_significant': economic_score > 0.6
            }
        
        return economic_metrics
    
    def _generate_trading_recommendations(self, relevance_matrix, state_effects, economic_significance):
        """Generate trading recommendations based on analysis."""
        
        recommendations = []
        
        # High-relevance pattern-dimension combinations
        if not relevance_matrix.empty:
            for pattern in relevance_matrix.index:
                best_dimension = relevance_matrix.loc[pattern].idxmax()
                best_score = relevance_matrix.loc[pattern].max()
                
                if best_score > 0.6:
                    economic_score = economic_significance.get(pattern, {}).get('economic_score', 0)
                    
                    if economic_score > 0.6:
                        recommendations.append({
                            'type': 'high_confidence',
                            'pattern': pattern,
                            'dimension': best_dimension,
                            'relevance_score': best_score,
                            'economic_score': economic_score,
                            'recommendation': f"Use {best_dimension} features to predict {pattern} patterns"
                        })
        
        # Market state specific recommendations
        for pattern, state_data in state_effects.items():
            best_state = None
            best_frequency = 0
            
            for state, metrics in state_data.items():
                if metrics['frequency'] > best_frequency and metrics['sample_size'] > 20:
                    best_frequency = metrics['frequency']
                    best_state = state
            
            if best_state and best_frequency > 0.3:
                recommendations.append({
                    'type': 'state_specific',
                    'pattern': pattern,
                    'state': best_state,
                    'frequency': best_frequency,
                    'recommendation': f"{pattern} patterns are strongest in {best_state} ({best_frequency:.1%} frequency)"
                })
        
        return recommendations
    
    def test_causal_relationships(self, dimensions, patterns, max_lag: int = 3):
        """Test causal relationships between dimensions and patterns."""

        causal_analyzer = CausalAnalyzer(max_lag=max_lag)
        return causal_analyzer.evaluate(dimensions=dimensions, patterns=patterns)

    def measure_economic_significance(self, dimensions, patterns, market_states):
        """Measure economic significance for trading."""
        return self._measure_economic_significance(patterns, dimensions, market_states)

    # Public wrappers for reuse across helper analyzers
    def calculate_pattern_dimension_matrix(self, patterns, dimensions):
        return self._calculate_pattern_dimension_matrix(patterns, dimensions)

    def analyze_market_state_effects(self, patterns, market_states):
        return self._analyze_market_state_effects(patterns, market_states)

    def generate_trading_recommendations(self, relevance_matrix, state_effects, economic_significance):
        return self._generate_trading_recommendations(relevance_matrix, state_effects, economic_significance)

# Placeholder classes - to be implemented during migration
class PatternDimensionAnalyzer:
    """Pattern-dimension relationship analysis."""

    def __init__(self, base_analyzer: EconomicRelevanceAnalyzer | None = None):
        self._analyzer = base_analyzer or EconomicRelevanceAnalyzer()

    def score(self, patterns: Dict[str, Dict], dimensions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a DataFrame describing how well each dimension predicts a pattern."""

        matrix = self._analyzer.calculate_pattern_dimension_matrix(patterns, dimensions)
        return matrix.fillna(0.0)

    def summarize(self, patterns: Dict[str, Dict], dimensions: Dict[str, pd.DataFrame], top_n: int = 3) -> Dict[str, Iterable[Tuple[str, float]]]:
        """Return the top ``top_n`` dimensions per pattern."""

        matrix = self.score(patterns, dimensions)
        summary: Dict[str, Iterable[Tuple[str, float]]] = {}

        for pattern in matrix.index:
            ranked = matrix.loc[pattern].sort_values(ascending=False)
            summary[pattern] = list(zip(ranked.index[:top_n], ranked.values[:top_n]))

        return summary


class MarketStateRelevanceAnalyzer:
    """Market state pattern analysis."""

    def __init__(self, base_analyzer: EconomicRelevanceAnalyzer | None = None):
        self._analyzer = base_analyzer or EconomicRelevanceAnalyzer()

    def analyze(self, patterns: Dict[str, Dict], market_states: Dict[str, pd.Series]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return market state impacts for each pattern."""

        return self._analyzer.analyze_market_state_effects(patterns, market_states)


class CausalAnalyzer:
    """Causal relationship identification."""

    def __init__(self, max_lag: int = 3, min_correlation: float = 0.2):
        self.max_lag = max(1, max_lag)
        self.min_correlation = max(0.0, min_correlation)

    def evaluate(self, dimensions: Dict[str, pd.DataFrame | pd.Series], patterns: Dict[str, Dict]) -> Dict[str, Dict[str, Dict[str, float | bool | int]]]:
        """Evaluate lagged correlations as a lightweight causal proxy."""

        results: Dict[str, Dict[str, Dict[str, float | bool | int]]] = {}

        for pattern_name, pattern_payload in patterns.items():
            labels = pattern_payload.get("labels")
            if labels is None or labels.empty:
                continue

            pattern_results: Dict[str, Dict[str, float | bool | int]] = {}

            for dimension_name, feature_payload in dimensions.items():
                feature_series = self._prepare_feature_series(feature_payload)
                common_index = labels.index.intersection(feature_series.index)
                if len(common_index) < 30:
                    continue

                aligned_labels = labels.loc[common_index].astype(float)
                aligned_features = feature_series.loc[common_index]

                best_lag, best_strength = self._find_best_leading_lag(aligned_features, aligned_labels)
                pattern_results[dimension_name] = {
                    "leading_lag": best_lag,
                    "causal_strength": best_strength,
                    "is_causal": best_strength >= self.min_correlation,
                }

            if pattern_results:
                results[pattern_name] = pattern_results

        return results

    @staticmethod
    def _prepare_feature_series(feature_payload: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(feature_payload, pd.DataFrame):
            return feature_payload.mean(axis=1)

        return feature_payload

    def _find_best_leading_lag(self, feature_series: pd.Series, labels: pd.Series) -> Tuple[int, float]:
        best_lag = 0
        best_strength = 0.0

        for lag in range(1, self.max_lag + 1):
            shifted = feature_series.shift(lag).dropna()
            aligned_index = shifted.index.intersection(labels.index)
            if len(aligned_index) < 20:
                continue

            aligned_features = shifted.loc[aligned_index]
            aligned_labels = labels.loc[aligned_index]

            if aligned_features.std() == 0 or aligned_labels.std() == 0:
                continue

            correlation = float(np.corrcoef(aligned_features, aligned_labels)[0, 1])
            strength = abs(correlation)

            if strength > best_strength:
                best_strength = strength
                best_lag = lag

        return best_lag, best_strength


class TradingSignificanceAnalyzer:
    """Economic value measurement."""

    def __init__(self, base_analyzer: EconomicRelevanceAnalyzer | None = None):
        self._analyzer = base_analyzer or EconomicRelevanceAnalyzer()

    def evaluate(self, patterns: Dict[str, Dict], dimensions: Dict[str, pd.DataFrame], market_states: Dict[str, pd.Series]):
        """Return economic metrics per pattern."""

        return self._analyzer.measure_economic_significance(dimensions, patterns, market_states)

    def opportunity_report(
        self,
        patterns: Dict[str, Dict],
        dimensions: Dict[str, pd.DataFrame],
        market_states: Dict[str, pd.Series],
        min_score: float = 0.6,
    ) -> Dict[str, Dict[str, float]]:
        """Return high confidence trading opportunities."""

        relevance_matrix = self._analyzer.calculate_pattern_dimension_matrix(patterns, dimensions)
        state_effects = self._analyzer.analyze_market_state_effects(patterns, market_states)
        economic_significance = self._analyzer.measure_economic_significance(dimensions, patterns, market_states)

        recommendations = self._analyzer.generate_trading_recommendations(
            relevance_matrix,
            state_effects,
            economic_significance,
        )

        return {
            rec["pattern"]: {
                "dimension": rec.get("dimension"),
                "relevance_score": rec.get("relevance_score", 0.0),
                "economic_score": rec.get("economic_score", 0.0),
            }
            for rec in recommendations
            if rec.get("economic_score", 0.0) >= min_score
        }

# Main exports
__all__ = [
    "EconomicRelevanceAnalyzer",
    "PatternDimensionAnalyzer",
    "MarketStateRelevanceAnalyzer",
    "CausalAnalyzer",
    "TradingSignificanceAnalyzer"
]