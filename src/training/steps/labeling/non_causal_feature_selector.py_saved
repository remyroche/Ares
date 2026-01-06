"""
Non-Causal Feature Selector for Layer 2.5 Chaser

Selects features that are NOT identified as causal by the PC algorithm.
This ensures the Chaser maintains orthogonality to the Causal Anchor.

Key Features:
1. PC Algorithm Integration
2. Causal Parent Identification
3. Non-Causal Feature Filtering
4. Technical Indicator Prioritization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Any
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class NonCausalFeatureSelector:
    """
    Selects non-causal features for the Chaser to maintain orthogonality.
    
    The Chaser should only learn from features that are NOT causal parents,
    ensuring it focuses on temporary inefficiencies rather than structural relationships.
    """
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        pc_algorithm_results: Optional[Dict] = None,
        technical_feature_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        min_feature_importance: float = 0.01,
        max_features: int = 100,
        verbose: bool = True
    ):
        """
        Initialize Non-Causal Feature Selector.
        
        Args:
            causal_graph: Causal graph from PC algorithm
            pc_algorithm_results: Raw PC algorithm results
            technical_feature_patterns: Patterns for technical indicators
            exclude_patterns: Patterns to exclude
            min_feature_importance: Minimum feature importance threshold
            max_features: Maximum number of features to select
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.causal_graph = causal_graph
        self.pc_algorithm_results = pc_algorithm_results
        self.min_feature_importance = min_feature_importance
        self.max_features = max_features
        
        # Default technical feature patterns
        self.technical_feature_patterns = technical_feature_patterns or [
            'rsi', 'macd', 'bollinger', 'bb_', 'sma', 'ema', 'wma',
            'stoch', 'williams', 'cci', 'adx', 'atr', 'obv',
            'momentum', 'rate_of_change', 'roc', 'tsi',
            'volume_ratio', 'volume_weighted', 'vwap',
            'price_position', 'price_range', 'price_momentum',
            'volatility_ratio', 'volatility_rank', 'volatility_regime',
            'trend_strength', 'trend_direction', 'trend_consistency',
            'microstructure', 'order_flow', 'bid_ask', 'spread',
            'depth', 'imbalance', 'pressure', 'flow'
        ]
        
        # Default exclusion patterns (causal parents)
        self.exclude_patterns = exclude_patterns or [
            'inventory', 'liquidity', 'global_liquidity',
            'market_impact', 'structural', 'causal_',
            'volume_price_impact', 'volatility_price_impact',
            'liquidity_friction', 'execution_cost'
        ]
        
        # Cache for results
        self.causal_parents_ = None
        self.non_causal_features_ = None
        self.feature_scores_ = None
        
    def identify_causal_parents(self) -> Set[str]:
        """
        Identify causal parents from the causal graph or PC results.
        
        Returns:
            Set of causal parent feature names
        """
        try:
            if self.verbose:
                tprint_info("🔍 Identifying causal parents...")
            
            causal_parents = set()
            
            # Method 1: From causal graph
            if self.causal_graph is not None:
                for target, parents in self.causal_graph.items():
                    causal_parents.update(parents)
                    if self.verbose:
                        tprint_info(f"   - {target}: {len(parents)} parents")
            
            # Method 2: From PC algorithm results
            elif self.pc_algorithm_results is not None:
                if 'graph' in self.pc_algorithm_results:
                    graph = self.pc_algorithm_results['graph']
                    for node, edges in graph.items():
                        if 'parents' in edges:
                            causal_parents.update(edges['parents'])
                        if 'children' in edges:
                            causal_parents.update(edges['children'])
                
                if 'causal_strength' in self.pc_algorithm_results:
                    strength_matrix = self.pc_algorithm_results['causal_strength']
                    # Find strong causal relationships
                    strong_edges = np.where(np.abs(strength_matrix) > 0.1)
                    for i, j in zip(strong_edges[0], strong_edges[1]):
                        if i < len(strength_matrix) and j < len(strength_matrix):
                            # This would need feature names mapping
                            pass
            
            # Method 3: Default causal parents (common in financial systems)
            else:
                default_causal_parents = {
                    'volume', 'volatility', 'liquidity', 'inventory',
                    'global_liquidity', 'market_impact', 'spread',
                    'depth', 'order_flow_imbalance'
                }
                causal_parents.update(default_causal_parents)
                
                if self.verbose:
                    tprint_info("   - Using default causal parents")
            
            self.causal_parents_ = causal_parents
            
            if self.verbose:
                tprint_success(f"✅ Identified {len(causal_parents)} causal parents:")
                for parent in sorted(causal_parents):
                    tprint_info(f"   - {parent}")
            
            return causal_parents
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal parent identification failed: {e}")
            raise
    
    def filter_causal_features(
        self,
        all_features: List[str]
    ) -> List[str]:
        """
        Filter out causal features from the feature list.
        
        Args:
            all_features: List of all available features
            
        Returns:
            List of non-causal features
        """
        try:
            if self.verbose:
                tprint_info("🔍 Filtering causal features...")
            
            # Identify causal parents if not done yet
            if self.causal_parents_ is None:
                self.identify_causal_parents()
            
            non_causal_features = []
            excluded_count = 0
            
            for feature in all_features:
                # Check if feature matches any causal parent pattern
                is_causal = False
                
                # Direct match
                if feature in self.causal_parents_:
                    is_causal = True
                
                # Pattern match
                else:
                    feature_lower = feature.lower()
                    for causal_parent in self.causal_parents_:
                        if causal_parent.lower() in feature_lower:
                            is_causal = True
                            break
                
                # Exclusion pattern match
                if not is_causal:
                    for pattern in self.exclude_patterns:
                        if pattern.lower() in feature.lower():
                            is_causal = True
                            break
                
                if is_causal:
                    excluded_count += 1
                    if self.verbose and excluded_count <= 10:  # Show first 10
                        tprint_info(f"   ❌ Excluded (causal): {feature}")
                else:
                    non_causal_features.append(feature)
            
            if self.verbose:
                tprint_success(f"✅ Feature filtering complete:")
                tprint_info(f"   - Total features: {len(all_features)}")
                tprint_info(f"   - Excluded (causal): {excluded_count}")
                tprint_info(f"   - Remaining (non-causal): {len(non_causal_features)}")
            
            self.non_causal_features_ = non_causal_features
            return non_causal_features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Feature filtering failed: {e}")
            raise
    
    def prioritize_technical_features(
        self,
        non_causal_features: List[str],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Prioritize technical indicators among non-causal features.
        
        Args:
            non_causal_features: List of non-causal features
            feature_importance: Optional feature importance scores
            
        Returns:
            Prioritized list of features
        """
        try:
            if self.verbose:
                tprint_info("🎯 Prioritizing technical features...")
            
            # Score features based on technical patterns and importance
            feature_scores = {}
            
            for feature in non_causal_features:
                score = 0.0
                feature_lower = feature.lower()
                
                # Technical pattern matching
                for pattern in self.technical_feature_patterns:
                    if pattern in feature_lower:
                        score += 1.0
                        break  # Only count once per feature
                
                # Feature importance bonus
                if feature_importance and feature in feature_importance:
                    importance = feature_importance[feature]
                    if importance > self.min_feature_importance:
                        score += importance
                
                feature_scores[feature] = score
            
            # Sort by score
            sorted_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Limit to max_features
            selected_features = [feat for feat, score in sorted_features[:self.max_features]]
            
            # Store scores for analysis
            self.feature_scores_ = dict(sorted_features)
            
            if self.verbose:
                tprint_success(f"✅ Technical feature prioritization complete:")
                tprint_info(f"   - Selected {len(selected_features)} features (max: {self.max_features})")
                
                # Show top features
                for i, (feature, score) in enumerate(sorted_features[:10]):
                    tprint_info(f"   {i+1:2d}. {feature}: {score:.3f}")
            
            return selected_features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Technical feature prioritization failed: {e}")
            raise
    
    def select_non_causal_features(
        self,
        all_features: List[str],
        feature_importance: Optional[Dict[str, float]] = None,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete non-causal feature selection pipeline.
        
        Args:
            all_features: List of all available features
            feature_importance: Optional feature importance scores
            target_col: Target column to exclude
            
        Returns:
            Dictionary with selection results
        """
        try:
            if self.verbose:
                tprint_info("🚀 Starting Non-Causal Feature Selection...")
            
            # Remove target column if present
            if target_col and target_col in all_features:
                all_features = [f for f in all_features if f != target_col]
            
            # Step 1: Filter out causal features
            non_causal_features = self.filter_causal_features(all_features)
            
            # Step 2: Prioritize technical features
            selected_features = self.prioritize_technical_features(
                non_causal_features, feature_importance
            )
            
            # Compile results
            results = {
                'selected_features': selected_features,
                'causal_parents': list(self.causal_parents_) if self.causal_parents_ else [],
                'excluded_count': len(all_features) - len(selected_features),
                'selected_count': len(selected_features),
                'feature_scores': self.feature_scores_ or {},
                'selection_ratio': len(selected_features) / len(all_features) if all_features else 0
            }
            
            if self.verbose:
                tprint_success("✅ Non-Causal Feature Selection Complete:")
                tprint_info(f"   - Original features: {len(all_features)}")
                tprint_info(f"   - Causal parents excluded: {len(results['causal_parents'])}")
                tprint_info(f"   - Selected features: {len(selected_features)}")
                tprint_info(f"   - Selection ratio: {results['selection_ratio']:.2%}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Non-causal feature selection failed: {e}")
            raise
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of the selection process.
        
        Returns:
            Dictionary with selection summary
        """
        return {
            'causal_parents_count': len(self.causal_parents_) if self.causal_parents_ else 0,
            'non_causal_features_count': len(self.non_causal_features_) if self.non_causal_features_ else 0,
            'selected_features_count': len(self.feature_scores_) if self.feature_scores_ else 0,
            'technical_patterns_count': len(self.technical_feature_patterns),
            'exclude_patterns_count': len(self.exclude_patterns)
        }

# Convenience functions
def quick_non_causal_selection(
    all_features: List[str],
    causal_graph: Optional[Dict[str, List[str]]] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    **kwargs
) -> List[str]:
    """
    Quick non-causal feature selection with defaults.
    
    Args:
        all_features: List of all features
        causal_graph: Causal graph
        feature_importance: Feature importance scores
        **kwargs: Additional parameters
        
    Returns:
        List of selected non-causal features
    """
    selector = NonCausalFeatureSelector(causal_graph=causal_graph, **kwargs)
    results = selector.select_non_causal_features(all_features, feature_importance)
    return results['selected_features']

def create_technical_feature_patterns() -> List[str]:
    """
    Create comprehensive list of technical indicator patterns.
    
    Returns:
        List of technical feature patterns
    """
    return [
        # Momentum indicators
        'rsi', 'stoch', 'williams', 'cci', 'momentum', 'roc', 'tsi',
        
        # Moving averages
        'sma', 'ema', 'wma', 'hma', 'kama', 'mama', 'vwap', 'twap',
        
        # MACD family
        'macd', 'macd_signal', 'macd_histogram', 'macd_oscillator',
        
        # Bollinger Bands
        'bollinger', 'bb_', 'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
        
        # Volatility
        'atr', 'volatility_ratio', 'volatility_rank', 'volatility_regime',
        'volatility_breakout', 'volatility_contraction',
        
        # Volume indicators
        'obv', 'volume_ratio', 'volume_weighted', 'volume_price',
        'volume_trend', 'volume_momentum', 'accumulation_distribution',
        
        # Price patterns
        'price_position', 'price_range', 'price_momentum', 'price_acceleration',
        'price_oscillator', 'price_rate_of_change',
        
        # Trend indicators
        'trend_strength', 'trend_direction', 'trend_consistency', 'trend_momentum',
        'adx', 'adx_plus', 'adx_minus', 'trend_factor',
        
        # Market microstructure
        'microstructure', 'order_flow', 'bid_ask', 'spread', 'depth',
        'imbalance', 'pressure', 'flow', 'liquidity_trend',
        
        # Composite indicators
        'composite_score', 'combined_signal', 'aggregate_indicator',
        'multi_timeframe', 'cross_asset', 'inter_market'
    ]
