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
    from research.cluster_analysis.economic_relevance import (
        EconomicRelevanceAnalyzer,
        PatternDimensionAnalyzer,
        CausalAnalyzer
    )
"""

# Import actual implementations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
            
        except Exception as e:
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
    
    def test_causal_relationships(self, dimensions, patterns):
        """Test causal relationships between dimensions and patterns."""
        # This would implement Granger causality testing
        # For now, return placeholder structure
        causal_results = {}
        
        for pattern_name in patterns.keys():
            pattern_causality = {}
            for dimension_name in dimensions.keys():
                # Placeholder causal test result
                pattern_causality[dimension_name] = {
                    'granger_p_value': 0.5,  # Would be actual test result
                    'causal_strength': 0.3,
                    'is_causal': False
                }
            causal_results[pattern_name] = pattern_causality
        
        return causal_results
    
    def measure_economic_significance(self, dimensions, patterns, market_states):
        """Measure economic significance for trading."""
        return self._measure_economic_significance(patterns, dimensions, market_states)

# Placeholder classes - to be implemented during migration
class PatternDimensionAnalyzer:
    """Pattern-dimension relationship analysis."""
    pass

class MarketStateRelevanceAnalyzer:
    """Market state pattern analysis."""
    pass

class CausalAnalyzer:
    """Causal relationship identification."""
    pass

class TradingSignificanceAnalyzer:
    """Economic value measurement."""
    pass

# Main exports
__all__ = [
    "EconomicRelevanceAnalyzer",
    "PatternDimensionAnalyzer",
    "MarketStateRelevanceAnalyzer",
    "CausalAnalyzer",
    "TradingSignificanceAnalyzer"
]