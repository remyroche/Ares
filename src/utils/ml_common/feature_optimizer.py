"""
Feature Optimization for Specialists

This module provides feature optimization utilities including deduplication,
pruning, and shared feature management across specialists.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict
import hashlib

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getLogger('FeatureOptimizer')


@dataclass
class FeatureAnalysis:
    """Analysis results for feature optimization"""
    total_features: int
    unique_features: int
    duplicate_features: int
    shared_features: int
    specialist_specific_features: int
    feature_overlap_matrix: pd.DataFrame
    recommendations: List[str]


class FeatureOptimizer:
    """Optimize feature computation and usage across specialists"""
    
    def __init__(self):
        self.feature_registry = {}
        self.specialist_feature_map = {}
        self.feature_importance_cache = {}
        self.shared_features = {
            'price_features': ['open', 'high', 'low', 'close', 'volume'],
            'return_features': ['returns_1', 'returns_5', 'returns_20', 'returns_60'],
            'volatility_features': ['volatility_5', 'volatility_20', 'volatility_60', 'atr'],
            'technical_features': ['rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle'],
            'time_features': ['hour', 'day_of_week', 'month', 'quarter'],
        }
        self.computed_features = {}
    
    def analyze_specialist_features(self, specialist_categories: Dict[str, Dict]) -> FeatureAnalysis:
        """Analyze feature overlap and optimization opportunities"""
        
        tprint_info("🔍 Analyzing specialist feature overlap...")
        
        # Extract all feature patterns
        all_patterns = []
        specialist_patterns = {}
        
        for specialist, config in specialist_categories.items():
            patterns = config.get('patterns', [])
            specialist_patterns[specialist] = patterns
            all_patterns.extend(patterns)
        
        # Find unique and duplicate patterns
        unique_patterns = list(set(all_patterns))
        duplicate_patterns = [p for p in set(all_patterns) if all_patterns.count(p) > 1]
        
        # Calculate feature overlap matrix
        overlap_matrix = self._calculate_feature_overlap_matrix(specialist_patterns)
        
        # Identify shared vs specialist-specific features
        shared_features = self._identify_shared_features(specialist_patterns)
        specialist_specific = self._identify_specialist_specific_features(specialist_patterns)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            len(all_patterns), len(unique_patterns), len(duplicate_patterns),
            len(shared_features), len(specialist_specific)
        )
        
        analysis = FeatureAnalysis(
            total_features=len(all_patterns),
            unique_features=len(unique_patterns),
            duplicate_features=len(duplicate_patterns),
            shared_features=len(shared_features),
            specialist_specific_features=len(specialist_specific),
            feature_overlap_matrix=overlap_matrix,
            recommendations=recommendations
        )
        
        tprint_info(f"  Total patterns: {analysis.total_features}")
        tprint_info(f"  Unique patterns: {analysis.unique_features}")
        tprint_info(f"  Duplicate patterns: {analysis.duplicate_features}")
        tprint_info(f"  Shared features: {analysis.shared_features}")
        tprint_info(f"  Specialist-specific: {analysis.specialist_specific_features}")
        
        return analysis
    
    def deduplicate_specialist_features(self, specialist_categories: Dict[str, Dict]) -> Dict[str, Dict]:
        """Remove duplicate features across specialists"""
        
        tprint_info("🔄 Deduplicating specialist features...")
        
        # Track used patterns
        used_patterns = set()
        optimized_categories = {}
        
        # Process specialists in order of importance (anchor first)
        specialist_order = ['xgb_macro', 'risk', 'liquidity', 'path', 'momentum', 
                           'xgb_meso', 'volume', 'microstructure', 'candlestick',
                           'spectral', 'reversion', 'volatility', 'smc']
        
        # Sort specialists by importance
        sorted_specialists = []
        for specialist in specialist_order:
            if specialist in specialist_categories:
                sorted_specialists.append(specialist)
        
        # Add any remaining specialists
        for specialist in specialist_categories:
            if specialist not in sorted_specialists:
                sorted_specialists.append(specialist)
        
        # Deduplicate while preserving specialist functionality
        for specialist in sorted_specialists:
            config = specialist_categories[specialist].copy()
            original_patterns = config['patterns']
            
            # Remove already used patterns
            unique_patterns = [p for p in original_patterns if p not in used_patterns]
            
            # If too few patterns remain, add some duplicates back
            if len(unique_patterns) < 2 and len(original_patterns) > 2:
                # Add back most important patterns
                for pattern in original_patterns:
                    if pattern not in unique_patterns and len(unique_patterns) < 3:
                        unique_patterns.append(pattern)
            
            # Update config
            config['patterns'] = unique_patterns
            optimized_categories[specialist] = config
            
            # Mark patterns as used
            used_patterns.update(unique_patterns)
            
            tprint_info(f"  {specialist}: {len(original_patterns)} → {len(unique_patterns)} patterns")
        
        tprint_success(f"✅ Deduplicated {len(specialist_categories)} specialists")
        return optimized_categories
    
    def prune_low_importance_features(self, specialist: str, features: pd.DataFrame, 
                                    importance_threshold: float = 0.01) -> pd.DataFrame:
        """Remove low-importance features to speed up training"""
        
        if specialist in self.feature_importance_cache:
            importance = self.feature_importance_cache[specialist]
            
            # Filter by importance threshold
            important_features = importance[importance > importance_threshold].index
            
            # Ensure we keep at least some features
            if len(important_features) < 5:
                # Keep top 5 features
                important_features = importance.nlargest(5).index
            
            # Filter features
            available_features = [f for f in important_features if f in features.columns]
            
            if available_features:
                tprint_info(f"  Pruned {specialist}: {len(features.columns)} → {len(available_features)} features")
                return features[available_features]
        
        return features
    
    def lazy_feature_evaluation(self, df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
        """Evaluate features only when needed (lazy evaluation)"""
        
        features = []
        feature_counts = defaultdict(int)
        
        # Count pattern matches
        for pattern in patterns:
            matching_cols = [col for col in df.columns if col.startswith(pattern)]
            feature_counts[pattern] = len(matching_cols)
            if matching_cols:
                features.extend(matching_cols)
        
        # Remove duplicates while preserving order
        unique_features = []
        seen = set()
        for feature in features:
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        if unique_features:
            result = df[unique_features].copy()
            tprint_info(f"  Lazy evaluation: {len(df.columns)} total → {len(unique_features)} relevant features")
            return result
        else:
            tprint_warning("  No matching features found for patterns")
            return pd.DataFrame(index=df.index)
    
    def get_or_compute_shared_features(self, df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """Get shared features from cache or compute them once"""
        
        if feature_type in self.computed_features:
            cached_features = self.computed_features[feature_type]
            # Check if cached features match current data
            if len(cached_features) == len(df) and cached_features.index.equals(df.index):
                tprint_info(f"  Using cached {feature_type}")
                return cached_features
        
        # Compute features
        computed = self._compute_shared_feature_type(df, feature_type)
        self.computed_features[feature_type] = computed
        
        tprint_info(f"  Computed {feature_type}: {len(computed.columns)} features")
        return computed
    
    def optimize_feature_pipeline(self, specialist_categories: Dict[str, Dict], 
                                 df: pd.DataFrame) -> Tuple[Dict[str, Dict], FeatureAnalysis]:
        """Complete feature optimization pipeline"""
        
        tprint_info("🚀 Starting feature optimization pipeline...")
        
        # Step 1: Analyze current feature usage
        analysis = self.analyze_specialist_features(specialist_categories)
        
        # Step 2: Deduplicate features
        optimized_categories = self.deduplicate_specialist_features(specialist_categories)
        
        # Step 3: Pre-compute shared features
        self._precompute_shared_features(df)
        
        tprint_success("✅ Feature optimization pipeline completed")
        return optimized_categories, analysis
    
    def _calculate_feature_overlap_matrix(self, specialist_patterns: Dict[str, List[str]]) -> pd.DataFrame:
        """Calculate feature overlap matrix between specialists"""
        
        specialists = list(specialist_patterns.keys())
        overlap_matrix = pd.DataFrame(index=specialists, columns=specialists, dtype=float)
        
        for spec1 in specialists:
            for spec2 in specialists:
                if spec1 == spec2:
                    overlap_matrix.loc[spec1, spec2] = 1.0
                else:
                    patterns1 = set(specialist_patterns[spec1])
                    patterns2 = set(specialist_patterns[spec2])
                    
                    if patterns1 and patterns2:
                        overlap = len(patterns1.intersection(patterns2))
                        total = len(patterns1.union(patterns2))
                        overlap_matrix.loc[spec1, spec2] = overlap / total if total > 0 else 0.0
                    else:
                        overlap_matrix.loc[spec1, spec2] = 0.0
        
        return overlap_matrix
    
    def _identify_shared_features(self, specialist_patterns: Dict[str, List[str]]) -> List[str]:
        """Identify features shared across multiple specialists"""
        
        pattern_counts = defaultdict(int)
        for patterns in specialist_patterns.values():
            for pattern in patterns:
                pattern_counts[pattern] += 1
        
        # Features used by 2+ specialists
        shared_features = [pattern for pattern, count in pattern_counts.items() if count >= 2]
        return shared_features
    
    def _identify_specialist_specific_features(self, specialist_patterns: Dict[str, List[str]]) -> List[str]:
        """Identify features unique to individual specialists"""
        
        pattern_counts = defaultdict(int)
        for patterns in specialist_patterns.values():
            for pattern in patterns:
                pattern_counts[pattern] += 1
        
        # Features used by only 1 specialist
        specific_features = [pattern for pattern, count in pattern_counts.items() if count == 1]
        return specific_features
    
    def _generate_optimization_recommendations(self, total: int, unique: int, duplicates: int,
                                             shared: int, specific: int) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        if duplicates > 0:
            recommendations.append(f"Remove {duplicates} duplicate feature patterns")
        
        if shared > total * 0.3:
            recommendations.append("Consider creating shared feature modules for common patterns")
        
        if specific > total * 0.7:
            recommendations.append("High specialization - good for diversity but may impact efficiency")
        
        if unique < total * 0.5:
            recommendations.append("High overlap detected - consider feature deduplication")
        
        if total > 100:
            recommendations.append("Large feature space - consider feature selection or dimensionality reduction")
        
        return recommendations
    
    def _compute_shared_feature_type(self, df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """Compute shared features of a specific type"""
        
        if feature_type not in self.shared_features:
            return pd.DataFrame(index=df.index)
        
        features = []
        base_features = self.shared_features[feature_type]
        
        for feature in base_features:
            if feature in df.columns:
                features.append(feature)
        
        if features:
            return df[features].copy()
        else:
            return pd.DataFrame(index=df.index)
    
    def _precompute_shared_features(self, df: pd.DataFrame):
        """Pre-compute all shared feature types"""
        
        tprint_info("📊 Pre-computing shared features...")
        
        for feature_type in self.shared_features.keys():
            self.get_or_compute_shared_features(df, feature_type)
    
    def cache_feature_importance(self, specialist: str, importance: pd.Series):
        """Cache feature importance for future pruning"""
        self.feature_importance_cache[specialist] = importance
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        
        return {
            'shared_feature_types': list(self.shared_features.keys()),
            'computed_features': list(self.computed_features.keys()),
            'cached_importance_specialists': list(self.feature_importance_cache.keys()),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of cached features"""
        
        total_size = 0
        for features in self.computed_features.values():
            # Rough estimate: 8 bytes per value
            total_size += len(features) * len(features.columns) * 8
        
        return total_size / (1024 * 1024)  # Convert to MB


def create_feature_optimizer() -> FeatureOptimizer:
    """Factory function to create feature optimizer"""
    return FeatureOptimizer()
