"""
Interaction Generator for Feature Interaction Generation

Generates feature interactions including product, ratio, difference, and log interactions
for the three-phase interaction generation pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
import warnings
from dataclasses import dataclass
import logging

from src.utils.tprint import tprint
from itertools import combinations
import gc

# Import optimization utilities
from .advanced_memory_manager import AdvancedMemoryManager, MemoryConfig
from .enhanced_vectorbt_manager import EnhancedVectorBTManager, VectorBTConfig
from .m1_parallel_processor import M1ParallelProcessor, ParallelConfig
from .data_structure_optimizer import DataStructureOptimizer, OptimizationConfig
from .smart_interaction_discovery import SmartInteractionDiscovery, InteractionConfig

logger = logging.getLogger(__name__)

@dataclass
class InteractionConfig:
    """Configuration for interaction generation."""
    
    # Interaction types to generate
    enable_product: bool = True
    enable_ratio: bool = True
    enable_difference: bool = True
    enable_log: bool = True
    
    # Ratio settings
    ratio_epsilon: float = 1e-8
    ratio_winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)
    
    # Log settings
    log_epsilon: float = 1e-8
    log_base: float = np.e  # Natural log
    
    # Selection settings
    max_interactions_per_pair: int = 4  # Product, ratio, diff, log
    max_pairs: int = 50
    
    # Memory optimization
    chunk_size: int = 10000
    enable_memory_optimization: bool = True
    
    # Interaction centrality threshold
    min_interaction_centrality: float = 0.1

class FeatureInteractionGenerator:
    """
    Generates feature interactions for interaction analysis.
    
    Creates four interaction types:
    1. Product: f1 × f2
    2. Ratio: f1 / (f2 + ε)
    3. Difference: f1 - f2
    4. Log: log(f1 + ε) or log(f1 / f2 + ε)
    """
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        """Initialize the interaction generator with advanced optimizations."""
        self.config = config or InteractionConfig()
        self.logger = logger
        
        # Initialize optimization components
        self.memory_manager = AdvancedMemoryManager(MemoryConfig())
        self.vectorbt_manager = EnhancedVectorBTManager(VectorBTConfig())
        self.parallel_processor = M1ParallelProcessor(ParallelConfig())
        self.data_optimizer = DataStructureOptimizer(OptimizationConfig())
        self.smart_discovery = SmartInteractionDiscovery(InteractionConfig())
        
        # Cache for computed interactions
        self._interaction_cache = {}
        self._centrality_cache = {}
        
    def _winsorize_series(self, series: pd.Series, 
                         quantiles: Optional[Tuple[float, float]] = None) -> pd.Series:
        """Winsorize a series to handle outliers."""
        if quantiles is None:
            quantiles = self.config.ratio_winsorize_quantiles
            
        lower_quantile, upper_quantile = quantiles
        lower_bound = series.quantile(lower_quantile)
        upper_bound = series.quantile(upper_quantile)
        
        return series.clip(lower_bound, upper_bound)
    
    def _safe_log(self, series: pd.Series, base: Optional[float] = None) -> pd.Series:
        """Safely compute logarithm with epsilon guard."""
        if base is None:
            base = self.config.log_base
            
        # Add epsilon to avoid log(0)
        safe_series = series + self.config.log_epsilon
        
        if base == np.e:
            return np.log(safe_series)
        elif base == 2:
            return np.log2(safe_series)
        elif base == 10:
            return np.log10(safe_series)
        else:
            return np.log(safe_series) / np.log(base)
    
    def _generate_product_interaction(self, f1: pd.Series, f2: pd.Series, 
                                    name1: str, name2: str) -> pd.Series:
        """Generate product interaction: f1 × f2"""
        interaction = f1 * f2
        return interaction
    
    def _generate_ratio_interaction(self, f1: pd.Series, f2: pd.Series, 
                                  name1: str, name2: str) -> pd.Series:
        """Generate ratio interaction: f1 / (f2 + ε)"""
        # Add epsilon to denominator to avoid division by zero
        denominator = f2 + self.config.ratio_epsilon
        ratio = f1 / denominator
        
        # Winsorize to handle extreme values
        ratio = self._winsorize_series(ratio)
        
        return ratio
    
    def _generate_difference_interaction(self, f1: pd.Series, f2: pd.Series, 
                                       name1: str, name2: str) -> pd.Series:
        """Generate difference interaction: f1 - f2"""
        interaction = f1 - f2
        return interaction
    
    def _generate_log_interaction(self, f1: pd.Series, f2: pd.Series, 
                                name1: str, name2: str) -> pd.Series:
        """Generate log interaction: log(f1 + ε) or log(f1 / f2 + ε)"""
        # Log of ratio (more meaningful than log of product)
        ratio = f1 / (f2 + self.config.ratio_epsilon)
        log_interaction = self._safe_log(ratio)
        
        # Winsorize to handle extreme values
        log_interaction = self._winsorize_series(log_interaction)
        
        return log_interaction
    
    def generate_interactions_for_pair(self, 
                                     f1: pd.Series, f2: pd.Series,
                                     name1: str, name2: str) -> Dict[str, pd.Series]:
        """Generate all interaction types for a feature pair."""
        interactions = {}
        
        # Ensure features are aligned
        common_index = f1.index.intersection(f2.index)
        f1_aligned = f1.loc[common_index]
        f2_aligned = f2.loc[common_index]
        
        if len(common_index) == 0:
            tprint(f"⚠️ No common index for {name1} and {name2}, skipping")
            return interactions
            
        # Generate product interaction
        if self.config.enable_product:
            try:
                product = self._generate_product_interaction(f1_aligned, f2_aligned, name1, name2)
                interactions[f"{name1}_x_{name2}"] = product
            except Exception as e:
                tprint(f"⚠️ Failed to generate product for {name1} × {name2}: {e}")
        
        # Generate ratio interaction
        if self.config.enable_ratio:
            try:
                ratio = self._generate_ratio_interaction(f1_aligned, f2_aligned, name1, name2)
                interactions[f"{name1}_div_{name2}"] = ratio
            except Exception as e:
                tprint(f"⚠️ Failed to generate ratio for {name1} ÷ {name2}: {e}")
        
        # Generate difference interaction
        if self.config.enable_difference:
            try:
                diff = self._generate_difference_interaction(f1_aligned, f2_aligned, name1, name2)
                interactions[f"{name1}_minus_{name2}"] = diff
            except Exception as e:
                tprint(f"⚠️ Failed to generate difference for {name1} - {name2}: {e}")
        
        # Generate log interaction
        if self.config.enable_log:
            try:
                log_interaction = self._generate_log_interaction(f1_aligned, f2_aligned, name1, name2)
                interactions[f"{name1}_log_{name2}"] = log_interaction
            except Exception as e:
                tprint(f"⚠️ Failed to generate log for {name1} log {name2}: {e}")
        
        return interactions
    
    def generate_interactions_from_centrality(self, 
                                            features_df: pd.DataFrame,
                                            interaction_centrality: Dict[Tuple[str, str], float],
                                            max_pairs: Optional[int] = None) -> pd.DataFrame:
        """Generate interactions based on interaction centrality scores."""
        if max_pairs is None:
            max_pairs = self.config.max_pairs
            
        tprint(f"📊 Generating interactions from {len(interaction_centrality)} pairs, selecting top {max_pairs}")
        
        # Sort pairs by interaction centrality
        sorted_pairs = sorted(interaction_centrality.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Take top pairs
        top_pairs = sorted_pairs[:max_pairs]
        
        all_interactions = {}
        interaction_metadata = {}
        
        for (f1_name, f2_name), centrality_score in top_pairs:
            if f1_name not in features_df.columns or f2_name not in features_df.columns:
                tprint(f"⚠️ Feature {f1_name} or {f2_name} not found, skipping")
                continue
                
            f1 = features_df[f1_name]
            f2 = features_df[f2_name]
            
            # Generate interactions for this pair
            pair_interactions = self.generate_interactions_for_pair(f1, f2, f1_name, f2_name)
            
            # Store interactions
            for interaction_name, interaction_series in pair_interactions.items():
                all_interactions[interaction_name] = interaction_series
                
                # Store metadata
                interaction_metadata[interaction_name] = {
                    'feature1': f1_name,
                    'feature2': f2_name,
                    'interaction_type': self._get_interaction_type(interaction_name),
                    'centrality_score': centrality_score,
                    'parent_features': [f1_name, f2_name]
                }
        
        # Convert to DataFrame
        if all_interactions:
            interactions_df = pd.DataFrame(all_interactions)
            tprint(f"✅ Generated {len(interactions_df.columns)} interactions from {len(top_pairs)} pairs")
        else:
            interactions_df = pd.DataFrame(index=features_df.index)
            tprint("⚠️ No interactions generated")
        
        # Store metadata
        self._interaction_cache['metadata'] = interaction_metadata
        
        return interactions_df
    
    def _get_interaction_type(self, interaction_name: str) -> str:
        """Determine interaction type from name."""
        if '_x_' in interaction_name:
            return 'product'
        elif '_div_' in interaction_name:
            return 'ratio'
        elif '_minus_' in interaction_name:
            return 'difference'
        elif '_log_' in interaction_name:
            return 'log'
        else:
            return 'unknown'
    
    def generate_interactions_from_top_features(self, 
                                              features_df: pd.DataFrame,
                                              top_features: List[str],
                                              max_pairs: Optional[int] = None) -> pd.DataFrame:
        """Generate interactions from top features using all combinations."""
        if max_pairs is None:
            max_pairs = self.config.max_pairs
            
        tprint(f"📊 Generating interactions from {len(top_features)} top features")
        
        # Filter features that exist in DataFrame
        available_features = [f for f in top_features if f in features_df.columns]
        
        if len(available_features) < 2:
            tprint("⚠️ Need at least 2 features to generate interactions")
            return pd.DataFrame(index=features_df.index)
        
        # Generate all combinations
        feature_pairs = list(combinations(available_features, 2))
        
        # Limit number of pairs
        if len(feature_pairs) > max_pairs:
            # Randomly sample pairs to avoid bias
            import random
            random.seed(42)  # For reproducibility
            feature_pairs = random.sample(feature_pairs, max_pairs)
        
        tprint(f"📊 Processing {len(feature_pairs)} feature pairs")
        
        all_interactions = {}
        interaction_metadata = {}
        
        for f1_name, f2_name in feature_pairs:
            f1 = features_df[f1_name]
            f2 = features_df[f2_name]
            
            # Generate interactions for this pair
            pair_interactions = self.generate_interactions_for_pair(f1, f2, f1_name, f2_name)
            
            # Store interactions
            for interaction_name, interaction_series in pair_interactions.items():
                all_interactions[interaction_name] = interaction_series
                
                # Store metadata
                interaction_metadata[interaction_name] = {
                    'feature1': f1_name,
                    'feature2': f2_name,
                    'interaction_type': self._get_interaction_type(interaction_name),
                    'parent_features': [f1_name, f2_name]
                }
        
        # Convert to DataFrame
        if all_interactions:
            interactions_df = pd.DataFrame(all_interactions)
            tprint(f"✅ Generated {len(interactions_df.columns)} interactions from {len(feature_pairs)} pairs")
        else:
            interactions_df = pd.DataFrame(index=features_df.index)
            tprint("⚠️ No interactions generated")
        
        # Store metadata
        self._interaction_cache['metadata'] = interaction_metadata
        
        return interactions_df
    
    def generate_interactions_chunked(self, 
                                    features_df: pd.DataFrame,
                                    feature_pairs: List[Tuple[str, str]],
                                    chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Generate interactions in chunks to manage memory."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
            
        tprint(f"🔄 [INTERACTION] Generating interactions in chunks of {chunk_size}")
        
        all_interactions = {}
        interaction_metadata = {}
        
        # Process pairs in chunks
        for i in range(0, len(feature_pairs), chunk_size):
            chunk_pairs = feature_pairs[i:i + chunk_size]
            tprint(f"📊 [INTERACTION] Processing chunk {i//chunk_size + 1}: pairs {i+1}-{min(i+chunk_size, len(feature_pairs))}")
            
            for f1_name, f2_name in chunk_pairs:
                if f1_name not in features_df.columns or f2_name not in features_df.columns:
                    continue
                    
                f1 = features_df[f1_name]
                f2 = features_df[f2_name]
                
                # Generate interactions for this pair
                pair_interactions = self.generate_interactions_for_pair(f1, f2, f1_name, f2_name)
                
                # Store interactions
                for interaction_name, interaction_series in pair_interactions.items():
                    all_interactions[interaction_name] = interaction_series
                    
                    # Store metadata
                    interaction_metadata[interaction_name] = {
                        'feature1': f1_name,
                        'feature2': f2_name,
                        'interaction_type': self._get_interaction_type(interaction_name),
                        'parent_features': [f1_name, f2_name]
                    }
            
            # Force garbage collection after each chunk
            if self.config.enable_memory_optimization:
                gc.collect()
        
        # Convert to DataFrame
        if all_interactions:
            interactions_df = pd.DataFrame(all_interactions)
            tprint(f"✅ Generated {len(interactions_df.columns)} interactions total")
        else:
            interactions_df = pd.DataFrame(index=features_df.index)
            tprint("⚠️ No interactions generated")
        
        # Store metadata
        self._interaction_cache['metadata'] = interaction_metadata
        
        return interactions_df
    
    def filter_interactions_by_variance(self, 
                                      interactions_df: pd.DataFrame,
                                      min_variance: float = 1e-8) -> pd.DataFrame:
        """Filter out interactions with low variance."""
        tprint(f"📊 Filtering interactions by variance threshold: {min_variance}")
        
        # Calculate variance for each interaction
        variances = interactions_df.var()
        
        # Filter interactions
        high_variance_interactions = variances[variances >= min_variance]
        
        filtered_df = interactions_df[high_variance_interactions.index]
        
        tprint(f"✅ Filtered to {len(filtered_df.columns)} interactions (removed {len(interactions_df.columns) - len(filtered_df.columns)} low-variance)")
        
        return filtered_df
    
    def filter_interactions_by_correlation(self, 
                                         interactions_df: pd.DataFrame,
                                         max_correlation: float = 0.98) -> pd.DataFrame:
        """Filter out highly correlated interactions."""
        tprint(f"📊 Filtering interactions by correlation threshold: {max_correlation}")
        
        if interactions_df.empty:
            return interactions_df
            
        # Calculate correlation matrix
        corr_matrix = interactions_df.corr().abs()
        
        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > max_correlation:
                    # Keep the first one, remove the second
                    to_remove.add(corr_matrix.columns[j])
        
        # Remove highly correlated interactions
        filtered_df = interactions_df.drop(columns=list(to_remove))
        
        tprint(f"✅ Filtered to {len(filtered_df.columns)} interactions (removed {len(to_remove)} highly correlated)")
        
        return filtered_df
    
    def get_interaction_metadata(self) -> Dict[str, Any]:
        """Get metadata about generated interactions."""
        return self._interaction_cache.get('metadata', {})
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary statistics about generated interactions."""
        metadata = self.get_interaction_metadata()
        
        if not metadata:
            return {}
        
        # Count by interaction type
        type_counts = {}
        for interaction_name, meta in metadata.items():
            interaction_type = meta.get('interaction_type', 'unknown')
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        # Count by parent features
        parent_counts = {}
        for interaction_name, meta in metadata.items():
            parent_features = meta.get('parent_features', [])
            for parent in parent_features:
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
        
        return {
            'total_interactions': len(metadata),
            'interaction_types': type_counts,
            'most_used_features': sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'metadata': metadata
        }

# Import tprint for logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
