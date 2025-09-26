from src.utils.tprint import tprint

"""
Backtesting-Enhanced Clustering for SR Levels

This module integrates the backtesting engine with the clustering system to create
SR levels based on learned quality rules from historical performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..logger import system_logger
from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel, create_sr_level_from_dict
# Import extensive clustering system from enhanced SR detection
try:
    from ...tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector
    EXTENSIVE_CLUSTERING_AVAILABLE = True
except ImportError:
    EXTENSIVE_CLUSTERING_AVAILABLE = False
    EnhancedSRDetector = None

# Base clustering algorithm class
class BaseClusteringAlgorithm:
    """Base class for clustering algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = system_logger.getChild(f'ClusteringAlgorithm.{name}')
        self.logger.info(f"Initializing {name} clustering algorithm")
    
    def cluster(self, levels: List[Dict], price_range: Tuple[float, float], **kwargs) -> 'ClusteringResult':
        """
        Cluster levels using the base clustering algorithm.
        
        This is a default implementation that provides basic clustering
        functionality. Subclasses should override this method with their
        specific clustering algorithms.
        
        Args:
            levels: List of level dictionaries to cluster
            price_range: Tuple of (min_price, max_price) for the clustering context
            **kwargs: Additional clustering parameters
            
        Returns:
            ClusteringResult containing the clustering results
        """
        self.logger.info(f"Starting base clustering with {len(levels)} levels, price range: {price_range}")
        self.logger.debug(f"Clustering parameters: {kwargs}")
        
        if not levels:
            self.logger.warning("No levels provided for clustering")
            return ClusteringResult(
                clusters=[],
                cluster_centers=[],
                quality_score=0.0,
                quality_enhanced=False,
                quality_metrics={'total_levels': 0},
                algorithm_used=self.name,
                parameters=kwargs
            )
        
        try:
            # Default clustering implementation using simple proximity-based approach
            proximity_threshold = kwargs.get('proximity_threshold', 0.01)  # 1% of price range
            min_price, max_price = price_range
            price_range_size = max_price - min_price
            absolute_proximity_threshold = proximity_threshold * price_range_size
            
            self.logger.info(f"Using proximity threshold: {absolute_proximity_threshold:.2f} ({proximity_threshold:.1%} of price range)")
            
            # Sort levels by price
            sorted_levels = sorted(enumerate(levels), key=lambda x: x[1].get('price', 0.0))
            
            clusters = []
            cluster_centers = []
            current_cluster = [sorted_levels[0][0]]  # Start with first level index
            
            for i in range(1, len(sorted_levels)):
                level_idx, level = sorted_levels[i]
                current_cluster_center = np.mean([levels[idx].get('price', 0.0) for idx in current_cluster])
                
                # Check proximity
                price_diff = abs(level.get('price', 0.0) - current_cluster_center)
                
                # Add to cluster if within proximity threshold
                if price_diff <= absolute_proximity_threshold:
                    current_cluster.append(level_idx)
                    self.logger.debug(f"Added level {level_idx} to current cluster (price_diff: {price_diff:.2f})")
                else:
                    # Finalize current cluster
                    if current_cluster:
                        clusters.append(current_cluster)
                        cluster_centers.append(current_cluster_center)
                        self.logger.debug(f"Finalized cluster with {len(current_cluster)} levels")
                    current_cluster = [level_idx]
            
            # Add final cluster
            if current_cluster:
                clusters.append(current_cluster)
                cluster_centers.append(np.mean([levels[idx].get('price', 0.0) for idx in current_cluster]))
                self.logger.debug(f"Added final cluster with {len(current_cluster)} levels")
            
            # Calculate quality score
            quality_score = self._calculate_base_quality_score(clusters, levels)
            
            self.logger.info(f"Base clustering completed: {len(clusters)} clusters, quality: {quality_score:.3f}")
            
            return ClusteringResult(
                clusters=clusters,
                cluster_centers=cluster_centers,
                quality_score=quality_score,
                quality_enhanced=False,
                quality_metrics={
                    'proximity_threshold': proximity_threshold,
                    'absolute_proximity_threshold': absolute_proximity_threshold,
                    'total_levels': len(levels),
                    'avg_cluster_size': len(levels) / len(clusters) if clusters else 0
                },
                algorithm_used=self.name,
                parameters=kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Base clustering failed: {e}")
            # Return single-level clusters as fallback
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                quality_score=0.5,
                quality_enhanced=False,
                algorithm_used=f'{self.name}_fallback',
                parameters=kwargs
            )
    
    def _calculate_base_quality_score(self, clusters: List[List[int]], levels: List[Dict]) -> float:
        """Calculate basic quality score for clustering result."""
        try:
            if not clusters:
                return 0.0
            
            total_quality = 0.0
            total_levels = 0
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Multi-level cluster quality based on price variance
                    cluster_prices = [levels[idx].get('price', 0.0) for idx in cluster]
                    price_variance = np.var(cluster_prices) if len(cluster_prices) > 1 else 0.0
                    cluster_quality = 1.0 / (1.0 + price_variance)  # Lower variance = higher quality
                else:
                    # Single-level cluster
                    cluster_quality = 0.8  # Good quality for individual levels
                
                total_quality += cluster_quality * len(cluster)
                total_levels += len(cluster)
            
            return total_quality / total_levels if total_levels > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.5

class StrengthProximityClustering(BaseClusteringAlgorithm):
    """Strength and proximity-based clustering for SR levels."""
    
    def __init__(self):
        super().__init__("StrengthProximity")
    
    def cluster(self, levels: List[Dict], price_range: Tuple[float, float], 
                proximity_threshold: float = 0.01, strength_similarity_threshold: float = 0.2) -> 'ClusteringResult':
        """Cluster levels based on strength and proximity using adaptive thresholds."""
        
        tprint(f"🎯 Starting strength-proximity clustering with {len(levels)} levels...")
        tprint(f"📊 Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
        tprint(f"⚙️ Proximity threshold: {proximity_threshold:.3f} ({proximity_threshold*100:.1f}% of price range)")
        tprint(f"⚙️ Strength similarity threshold: {strength_similarity_threshold:.3f}")
        
        self.logger.info(f"Starting strength-proximity clustering with {len(levels)} levels")
        self.logger.info(f"Price range: {price_range[0]:.2f} - {price_range[1]:.2f}")
        self.logger.info(f"Proximity threshold: {proximity_threshold:.3f}, Strength threshold: {strength_similarity_threshold:.3f}")
        
        if not levels:
            tprint("⚠️ No levels provided for clustering, returning empty result")
            self.logger.warning("No levels provided for clustering, returning empty result")
            return ClusteringResult(
                clusters=[],
                cluster_centers=[],
                quality_score=0.0,
                quality_enhanced=False,
                quality_metrics={'total_levels': 0},
                algorithm_used='StrengthProximity',
                parameters={}
            )
        
        try:
            # Convert proximity threshold to absolute price difference
            min_price, max_price = price_range
            price_range_size = max_price - min_price
            absolute_proximity_threshold = proximity_threshold * price_range_size
            
            tprint(f"🔢 Calculated absolute proximity threshold: ${absolute_proximity_threshold:.2f}")
            self.logger.info(f"Clustering {len(levels)} levels with proximity threshold: {absolute_proximity_threshold:.2f} ({proximity_threshold:.1%} of price range)")
            self.logger.info(f"Strength similarity threshold: {strength_similarity_threshold:.2f}")
            
            # Log level distribution
            tprint("📊 Analyzing level distribution...")
            level_types = {}
            strength_ranges = {'low': 0, 'medium': 0, 'high': 0}
            for level in levels:
                level_type = level.get('level_type', 'unknown')
                level_types[level_type] = level_types.get(level_type, 0) + 1
                
                strength = level.get('strength', 0.5)
                if strength < 0.33:
                    strength_ranges['low'] += 1
                elif strength < 0.67:
                    strength_ranges['medium'] += 1
                else:
                    strength_ranges['high'] += 1
            
            tprint(f"📈 Level type distribution: {level_types}")
            tprint(f"💪 Strength distribution: {strength_ranges}")
            self.logger.info(f"Level type distribution: {level_types}")
            self.logger.info(f"Strength distribution: {strength_ranges}")
            
            # Initialize clusters
            tprint("🔄 Starting clustering process...")
            clusters = []
            unassigned_levels = list(range(len(levels)))
            cluster_count = 0
            
            while unassigned_levels:
                cluster_count += 1
                tprint(f"   Creating cluster {cluster_count}...")
                
                # Start new cluster with strongest unassigned level
                current_cluster = []
                seed_idx = self._find_strongest_level(levels, unassigned_levels)
                current_cluster.append(seed_idx)
                unassigned_levels.remove(seed_idx)
                
                tprint(f"   Seed level: ${levels[seed_idx].get('price', 0.0):.2f} (strength: {levels[seed_idx].get('strength', 0.5):.3f})")
                
                # Find all levels that should be in this cluster
                initial_unassigned_count = len(unassigned_levels)
                self._grow_cluster(levels, current_cluster, unassigned_levels, 
                                 absolute_proximity_threshold, strength_similarity_threshold)
                levels_added = initial_unassigned_count - len(unassigned_levels)
                
                tprint(f"   Cluster {cluster_count} created with {len(current_cluster)} levels ({levels_added} levels added)")
                
                clusters.append(current_cluster)
                
                # Remove assigned levels from unassigned
                for idx in current_cluster[1:]:  # Skip seed (already removed)
                    if idx in unassigned_levels:
                        unassigned_levels.remove(idx)
            
            tprint(f"✅ Clustering process completed: {len(clusters)} clusters created")
            
            # Calculate cluster centers and quality
            tprint("📊 Calculating cluster centers and quality scores...")
            cluster_centers = []
            total_quality = 0.0
            
            for i, cluster in enumerate(clusters):
                if cluster:
                    # Calculate weighted center (by strength)
                    cluster_prices = [levels[idx].get('price', 0.0) for idx in cluster]
                    cluster_strengths = [levels[idx].get('strength', 0.5) for idx in cluster]
                    
                    # Weighted average by strength
                    total_strength = sum(cluster_strengths)
                    if total_strength > 0:
                        weighted_center = sum(p * s for p, s in zip(cluster_prices, cluster_strengths)) / total_strength
                    else:
                        weighted_center = sum(cluster_prices) / len(cluster_prices)
                    
                    cluster_centers.append(weighted_center)
                    
                    # Calculate cluster quality (cohesion)
                    cluster_quality = self._calculate_cluster_quality(levels, cluster, weighted_center)
                    total_quality += cluster_quality
                    
                    tprint(f"   Cluster {i+1}: {len(cluster)} levels, center: ${weighted_center:.2f}, quality: {cluster_quality:.3f}")
            
            # Overall quality score
            quality_score = total_quality / len(clusters) if clusters else 0.0
            
            tprint(f"🎉 Strength-proximity clustering completed!")
            tprint(f"📊 Final Results:")
            tprint(f"   - Input levels: {len(levels)}")
            tprint(f"   - Clusters created: {len(clusters)}")
            tprint(f"   - Average cluster size: {len(levels) / len(clusters):.1f} levels")
            tprint(f"   - Overall quality score: {quality_score:.3f}")
            
            self.logger.info(f"Strength-proximity clustering: {len(levels)} levels -> {len(clusters)} clusters")
            self.logger.info(f"Average cluster size: {len(levels) / len(clusters):.1f} levels")
            self.logger.info(f"Quality score: {quality_score:.3f}")
            
            # Log cluster details
            for i, cluster in enumerate(clusters):
                cluster_prices = [levels[idx].get('price', 0.0) for idx in cluster]
                cluster_strengths = [levels[idx].get('strength', 0.5) for idx in cluster]
                self.logger.debug(f"Cluster {i+1}: {len(cluster)} levels, price range: {min(cluster_prices):.2f}-{max(cluster_prices):.2f}, avg strength: {np.mean(cluster_strengths):.3f}")
            
            return ClusteringResult(
                clusters=clusters,
                cluster_centers=cluster_centers,
                quality_score=quality_score,
                quality_enhanced=True,
                quality_metrics={
                    'proximity_threshold': proximity_threshold,
                    'strength_similarity_threshold': strength_similarity_threshold,
                    'absolute_proximity_threshold': absolute_proximity_threshold,
                    'total_levels': len(levels),
                    'avg_cluster_size': len(levels) / len(clusters) if clusters else 0
                },
                algorithm_used='StrengthProximity',
                parameters={
                    'proximity_threshold': proximity_threshold,
                    'strength_similarity_threshold': strength_similarity_threshold,
                    'absolute_proximity_threshold': absolute_proximity_threshold
                }
            )
            
        except Exception as e:
            tprint(f"❌ Strength-proximity clustering failed: {e}")
            self.logger.error(f"Strength-proximity clustering failed: {e}")
            raise
    
    def _find_strongest_level(self, levels: List[Dict], available_indices: List[int]) -> int:
        """Find the level with highest strength among available indices."""
        if not available_indices:
            self.logger.warning("No available indices for finding strongest level")
            return 0
        
        self.logger.debug(f"Finding strongest level among {len(available_indices)} available indices")
        
        strongest_idx = available_indices[0]
        strongest_strength = levels[strongest_idx].get('strength', 0.5)
        
        for idx in available_indices[1:]:
            strength = levels[idx].get('strength', 0.5)
            if strength > strongest_strength:
                strongest_strength = strength
                strongest_idx = idx
        
        self.logger.debug(f"Selected strongest level at index {strongest_idx} with strength {strongest_strength:.3f}")
        return strongest_idx
    
    def _grow_cluster(self, levels: List[Dict], current_cluster: List[int], 
                     unassigned_levels: List[int], proximity_threshold: float, 
                     strength_similarity_threshold: float) -> None:
        """Grow cluster by adding nearby levels with similar strength."""
        
        tprint(f"      Growing cluster with {len(current_cluster)} levels, checking {len(unassigned_levels)} unassigned levels...")
        self.logger.debug(f"Growing cluster with {len(current_cluster)} levels, checking {len(unassigned_levels)} unassigned levels")
        
        # Get cluster characteristics
        cluster_prices = [levels[i].get('price', 0.0) for i in current_cluster]
        cluster_strengths = [levels[i].get('strength', 0.5) for i in current_cluster]
        cluster_center_price = sum(cluster_prices) / len(cluster_prices)
        cluster_avg_strength = sum(cluster_strengths) / len(cluster_strengths)
        
        tprint(f"      Cluster center: ${cluster_center_price:.2f}, avg strength: {cluster_avg_strength:.3f}")
        self.logger.debug(f"Cluster center: {cluster_center_price:.2f}, avg strength: {cluster_avg_strength:.3f}")
        
        # Find levels to add to cluster
        levels_to_add = []
        proximity_rejected = 0
        strength_rejected = 0
        
        for idx in unassigned_levels:
            level_price = levels[idx].get('price', 0.0)
            level_strength = levels[idx].get('strength', 0.5)
            
            # Check proximity
            price_distance = abs(level_price - cluster_center_price)
            if price_distance > proximity_threshold:
                proximity_rejected += 1
                continue
            
            # Check strength similarity
            strength_difference = abs(level_strength - cluster_avg_strength)
            if strength_difference > strength_similarity_threshold:
                strength_rejected += 1
                continue
            
            # Level qualifies for this cluster
            levels_to_add.append(idx)
            tprint(f"         Adding level ${level_price:.2f} (strength: {level_strength:.3f}) to cluster")
            self.logger.debug(f"Adding level {idx} (price: {level_price:.2f}, strength: {level_strength:.3f}) to cluster")
        
        tprint(f"      Cluster growth: {len(levels_to_add)} levels added, {proximity_rejected} rejected by proximity, {strength_rejected} rejected by strength")
        self.logger.debug(f"Cluster growth: {len(levels_to_add)} levels added, {proximity_rejected} rejected by proximity, {strength_rejected} rejected by strength")
        
        # Add qualifying levels
        current_cluster.extend(levels_to_add)
    
    def _calculate_cluster_quality(self, levels: List[Dict], cluster: List[int], 
                                 cluster_center: float) -> float:
        """Calculate quality score for a cluster based on cohesion."""
        if not cluster:
            self.logger.warning("Empty cluster provided for quality calculation")
            return 0.0
        
        cluster_prices = [levels[i].get('price', 0.0) for i in cluster]
        cluster_strengths = [levels[i].get('strength', 0.5) for i in cluster]
        
        # Price cohesion (lower variance = higher quality)
        price_variance = np.var(cluster_prices) if len(cluster_prices) > 1 else 0.0
        price_cohesion = 1.0 / (1.0 + price_variance)
        
        # Strength cohesion (lower variance = higher quality)
        strength_variance = np.var(cluster_strengths) if len(cluster_strengths) > 1 else 0.0
        strength_cohesion = 1.0 / (1.0 + strength_variance)
        
        # Average strength (higher = better)
        avg_strength = sum(cluster_strengths) / len(cluster_strengths)
        
        # Combined quality score
        quality = 0.4 * price_cohesion + 0.3 * strength_cohesion + 0.3 * avg_strength
        
        self.logger.debug(f"Cluster quality calculation: price_cohesion={price_cohesion:.3f}, strength_cohesion={strength_cohesion:.3f}, avg_strength={avg_strength:.3f}, final_quality={quality:.3f}")
        
        return quality

@dataclass
class ClusteringResult:
    """Result of clustering SR levels."""
    clusters: List[List[int]]  # List of clusters, each containing level indices
    cluster_centers: List[float]  # Center price for each cluster
    quality_score: float = 0.0  # Overall quality score
    quality_enhanced: bool = False  # Whether quality enhancement was used
    quality_metrics: Dict[str, Any] = None  # Additional quality metrics
    algorithm_used: str = "unknown"  # Algorithm used for clustering
    parameters: Dict[str, Any] = None  # Parameters used for clustering
    
    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = {}

@dataclass
class BacktestingEnhancedConfig:
    """Configuration for backtesting-enhanced clustering."""
    # Backtesting parameters
    backtest_config: BacktestConfig = None
    
    # Clustering parameters
    proximity_threshold: float = 0.005  # 0.5% of price range (HARD RULE: no merging beyond 0.5%)
    strength_similarity_threshold: float = 0.15  # 15% strength difference (stricter)
    
    # Quality filtering
    min_quality_score: float = 0.01  # Minimum quality score to keep level (extremely lenient filtering)
    quality_weight_in_clustering: float = 0.4  # Weight of quality in clustering decisions
    
    # Learning parameters
    min_levels_for_learning: int = 50  # Minimum levels needed to learn rules
    learning_update_frequency: int = 100  # Update rules every N new levels
    
    # Performance thresholds
    min_success_rate: float = 0.6  # Minimum success rate for good levels
    min_bounce_strength: float = 0.002  # Minimum 0.2% bounce strength

class BacktestingEnhancedClustering:
    """Clustering system enhanced with backtesting-based quality assessment."""
    
    def __init__(self, config: Optional[BacktestingEnhancedConfig] = None):
        self.config = config or BacktestingEnhancedConfig()
        self.logger = system_logger.getChild('BacktestingEnhancedClustering')
        
        tprint("🚀 Initializing BacktestingEnhancedClustering system...")
        self.logger.info("🚀 Initializing BacktestingEnhancedClustering system...")
        
        tprint(f"📋 Configuration loaded:")
        tprint(f"   - Proximity threshold: {self.config.proximity_threshold:.3f} ({self.config.proximity_threshold*100:.1f}% of price range)")
        tprint(f"   - Strength similarity threshold: {self.config.strength_similarity_threshold:.3f}")
        tprint(f"   - Min quality score: {self.config.min_quality_score:.3f}")
        tprint(f"   - Quality weight in clustering: {self.config.quality_weight_in_clustering:.3f}")
        tprint(f"   - Min levels for learning: {self.config.min_levels_for_learning}")
        tprint(f"   - Learning update frequency: {self.config.learning_update_frequency}")
        tprint(f"   - Min success rate: {self.config.min_success_rate:.3f}")
        tprint(f"   - Min bounce strength: {self.config.min_bounce_strength:.3f}")
        
        self.logger.info("📋 Configuration loaded:")
        self.logger.info(f"   - Proximity threshold: {self.config.proximity_threshold:.3f} ({self.config.proximity_threshold*100:.1f}% of price range)")
        self.logger.info(f"   - Strength similarity threshold: {self.config.strength_similarity_threshold:.3f}")
        self.logger.info(f"   - Min quality score: {self.config.min_quality_score:.3f}")
        self.logger.info(f"   - Quality weight in clustering: {self.config.quality_weight_in_clustering:.3f}")
        self.logger.info(f"   - Min levels for learning: {self.config.min_levels_for_learning}")
        self.logger.info(f"   - Learning update frequency: {self.config.learning_update_frequency}")
        self.logger.info(f"   - Min success rate: {self.config.min_success_rate:.3f}")
        self.logger.info(f"   - Min bounce strength: {self.config.min_bounce_strength:.3f}")
        
        # Initialize components
        tprint("🔧 Initializing backtesting engine...")
        self.logger.info("🔧 Initializing backtesting engine...")
        self.backtesting_engine = SRBacktestingEngine(self.config.backtest_config)
        tprint("✅ Backtesting engine initialized successfully")
        self.logger.info("✅ Backtesting engine initialized successfully")
        
        # Initialize extensive clustering system if available
        tprint("🔍 Checking for enhanced SR detector...")
        self.logger.info("🔍 Checking for enhanced SR detector...")
        if EXTENSIVE_CLUSTERING_AVAILABLE:
            tprint("✅ Enhanced SR detector available, initializing...")
            self.logger.info("✅ Enhanced SR detector available, initializing...")
            self.enhanced_sr_detector = EnhancedSRDetector({})
            tprint("✅ Enhanced SR detector initialized successfully")
            self.logger.info("✅ Enhanced SR detector initialized successfully")
        else:
            error_msg = "❌ Enhanced SR detector not available - required for backtesting-enhanced clustering"
            tprint(error_msg)
            self.logger.error(error_msg)
            raise ImportError("Enhanced SR detector is required for backtesting-enhanced clustering")
        
        # Initialize strength-proximity clustering algorithm
        tprint("🎯 Initializing strength-proximity clustering algorithm...")
        self.logger.info("🎯 Initializing strength-proximity clustering algorithm...")
        self.strength_proximity_clustering = StrengthProximityClustering()
        tprint("✅ Strength-proximity clustering algorithm initialized")
        self.logger.info("✅ Strength-proximity clustering algorithm initialized")
        
        # Learning state
        tprint("🧠 Initializing learning state...")
        self.logger.info("🧠 Initializing learning state...")
        self.learned_rules = {}
        self.quality_predictions = {}
        self.levels_processed = 0
        tprint("✅ Learning state initialized")
        self.logger.info("✅ Learning state initialized")
        
        tprint("🎉 BacktestingEnhancedClustering initialization completed successfully!")
        self.logger.info("🎉 BacktestingEnhancedClustering initialization completed successfully!")
        
    def cluster_with_backtesting(self, levels: List[Dict], data: pd.DataFrame, 
                                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster levels using backtesting-enhanced approach."""
        try:
            tprint("=" * 80)
            tprint("🚀 STARTING BACKTESTING-ENHANCED CLUSTERING")
            tprint("=" * 80)
            tprint(f"📊 Input Summary:")
            tprint(f"   - Levels to cluster: {len(levels)}")
            tprint(f"   - Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
            tprint(f"   - Price range size: ${price_range[1] - price_range[0]:.2f}")
            tprint(f"   - Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            tprint(f"   - Data time range: {data.index.min()} to {data.index.max()}")
            
            self.logger.info("=" * 80)
            self.logger.info("🚀 STARTING BACKTESTING-ENHANCED CLUSTERING")
            self.logger.info("=" * 80)
            self.logger.info(f"📊 Input Summary:")
            self.logger.info(f"   - Levels to cluster: {len(levels)}")
            self.logger.info(f"   - Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
            self.logger.info(f"   - Price range size: ${price_range[1] - price_range[0]:.2f}")
            self.logger.info(f"   - Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            self.logger.info(f"   - Data time range: {data.index.min()} to {data.index.max()}")
            
            # Step 1: Backtest levels to assess quality
            tprint("\n" + "=" * 60)
            tprint("📊 STEP 1: BACKTESTING LEVELS TO ASSESS QUALITY")
            tprint("=" * 60)
            self.logger.info("📊 STEP 1: BACKTESTING LEVELS TO ASSESS QUALITY")
            backtest_results = self._backtest_levels(levels, data)
            
            # Step 2: Learn/update quality rules
            tprint("\n" + "=" * 60)
            tprint("🧠 STEP 2: LEARNING/UPDATING QUALITY RULES")
            tprint("=" * 60)
            self.logger.info("🧠 STEP 2: LEARNING/UPDATING QUALITY RULES")
            if len(backtest_results) >= self.config.min_levels_for_learning:
                tprint(f"✅ Sufficient levels for learning: {len(backtest_results)} >= {self.config.min_levels_for_learning}")
                self.logger.info(f"✅ Sufficient levels for learning: {len(backtest_results)} >= {self.config.min_levels_for_learning}")
                self._update_quality_rules(backtest_results, data)
            else:
                tprint(f"⚠️ Insufficient levels for learning: {len(backtest_results)} < {self.config.min_levels_for_learning}")
                self.logger.warning(f"⚠️ Insufficient levels for learning: {len(backtest_results)} < {self.config.min_levels_for_learning}")
            
            # Step 3: Filter levels based on quality
            tprint("\n" + "=" * 60)
            tprint("🔍 STEP 3: FILTERING LEVELS BASED ON QUALITY")
            tprint("=" * 60)
            self.logger.info("🔍 STEP 3: FILTERING LEVELS BASED ON QUALITY")
            quality_filtered_levels = self._filter_by_quality(levels, backtest_results)
            
            # Step 4: Enhance level data with quality scores
            tprint("\n" + "=" * 60)
            tprint("✨ STEP 4: ENHANCING LEVEL DATA WITH QUALITY SCORES")
            tprint("=" * 60)
            self.logger.info("✨ STEP 4: ENHANCING LEVEL DATA WITH QUALITY SCORES")
            enhanced_levels = self._enhance_levels_with_quality(quality_filtered_levels, backtest_results)
            
            # Step 5: Cluster using quality-enhanced approach
            tprint("\n" + "=" * 60)
            tprint("🎯 STEP 5: CLUSTERING USING QUALITY-ENHANCED APPROACH")
            tprint("=" * 60)
            self.logger.info("🎯 STEP 5: CLUSTERING USING QUALITY-ENHANCED APPROACH")
            clustering_result = self._cluster_quality_enhanced(enhanced_levels, price_range, data)
            
            # Step 6: Post-process clusters with quality validation
            tprint("\n" + "=" * 60)
            tprint("✅ STEP 6: POST-PROCESSING CLUSTERS WITH QUALITY VALIDATION")
            tprint("=" * 60)
            self.logger.info("✅ STEP 6: POST-PROCESSING CLUSTERS WITH QUALITY VALIDATION")
            final_result = self._validate_clusters_with_backtesting(clustering_result, data)
            
            self.levels_processed += len(levels)
            
            tprint("\n" + "=" * 80)
            tprint("🎉 BACKTESTING-ENHANCED CLUSTERING COMPLETED!")
            tprint("=" * 80)
            tprint(f"📈 Final Results:")
            tprint(f"   - Input levels: {len(levels)}")
            tprint(f"   - Final clusters: {len(final_result.clusters)}")
            tprint(f"   - Quality score: {final_result.quality_score:.3f}")
            tprint(f"   - Quality enhanced: {final_result.quality_enhanced}")
            tprint(f"   - Total levels processed: {self.levels_processed}")
            tprint("=" * 80)
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 BACKTESTING-ENHANCED CLUSTERING COMPLETED!")
            self.logger.info("=" * 80)
            self.logger.info(f"📈 Final Results:")
            self.logger.info(f"   - Input levels: {len(levels)}")
            self.logger.info(f"   - Final clusters: {len(final_result.clusters)}")
            self.logger.info(f"   - Quality score: {final_result.quality_score:.3f}")
            self.logger.info(f"   - Quality enhanced: {final_result.quality_enhanced}")
            self.logger.info(f"   - Total levels processed: {self.levels_processed}")
            self.logger.info("=" * 80)
            
            return final_result
            
        except Exception as e:
            tprint(f"\n❌ BACKTESTING-ENHANCED CLUSTERING FAILED: {e}")
            tprint("🔄 Falling back to standard clustering...")
            self.logger.error(f"❌ Backtesting-enhanced clustering failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to standard clustering
            self.logger.warning("🔄 Falling back to standard clustering")
            return self._fallback_clustering(levels, price_range)
    
    def _backtest_levels(self, levels: List[Dict], data: pd.DataFrame) -> List[Any]:
        """Backtest all levels to assess their quality."""
        tprint(f"📈 Starting backtesting of {len(levels)} levels for quality assessment...")
        self.logger.info(f"📈 Starting backtesting of {len(levels)} levels for quality assessment...")
        
        # Convert to SRLevel objects
        tprint("🔄 Converting level dictionaries to SRLevel objects...")
        self.logger.info("🔄 Converting level dictionaries to SRLevel objects...")
        sr_levels = []
        conversion_errors = 0
        for i, level_dict in enumerate(levels):
            try:
                sr_level = create_sr_level_from_dict(level_dict)
                sr_levels.append(sr_level)
                if i % 10 == 0:  # Log progress every 10 levels
                    tprint(f"   Converted {i+1}/{len(levels)} levels to SRLevel objects...")
                    self.logger.debug(f"Converted {i+1}/{len(levels)} levels to SRLevel objects")
            except Exception as e:
                tprint(f"   ⚠️ Failed to create SRLevel from level {i+1}: {e}")
                self.logger.warning(f"Failed to create SRLevel from {level_dict}: {e}")
                conversion_errors += 1
                continue
        
        tprint(f"✅ Conversion completed: {len(sr_levels)} levels converted ({conversion_errors} errors)")
        self.logger.info(f"✅ Conversion completed: {len(sr_levels)} levels converted ({conversion_errors} errors)")
        
        if not sr_levels:
            tprint("❌ No valid SRLevel objects created, cannot proceed with backtesting")
            self.logger.error("No valid SRLevel objects created, cannot proceed with backtesting")
            return []
        
        # Log level distribution
        if sr_levels:
            support_count = sum(1 for level in sr_levels if level.level_type == 'support')
            resistance_count = sum(1 for level in sr_levels if level.level_type == 'resistance')
            avg_strength = np.mean([level.strength for level in sr_levels])
            avg_touches = np.mean([level.touches for level in sr_levels])
            
            tprint(f"📊 Level distribution:")
            tprint(f"   - Support levels: {support_count}")
            tprint(f"   - Resistance levels: {resistance_count}")
            tprint(f"   - Average strength: {avg_strength:.3f}")
            tprint(f"   - Average touches: {avg_touches:.1f}")
            
            self.logger.info(f"📊 Level distribution:")
            self.logger.info(f"   - Support levels: {support_count}")
            self.logger.info(f"   - Resistance levels: {resistance_count}")
            self.logger.info(f"   - Average strength: {avg_strength:.3f}")
            self.logger.info(f"   - Average touches: {avg_touches:.1f}")
        
        # Backtest levels
        tprint("🚀 Running backtesting engine on SRLevel objects...")
        self.logger.info("🚀 Running backtesting engine on SRLevel objects...")
        backtest_results = self.backtesting_engine.backtest_multiple_levels(sr_levels, data)
        
        if backtest_results:
            quality_scores = [r.quality_score for r in backtest_results]
            success_rates = [r.success_rate for r in backtest_results]
            bounce_strengths = [r.avg_bounce_strength for r in backtest_results]
            
            tprint(f"✅ Backtesting completed successfully!")
            tprint(f"📊 Backtesting Results Summary:")
            tprint(f"   - Levels backtested: {len(backtest_results)}")
            tprint(f"   - Quality scores - mean: {np.mean(quality_scores):.3f}, std: {np.std(quality_scores):.3f}")
            tprint(f"   - Quality scores - min: {np.min(quality_scores):.3f}, max: {np.max(quality_scores):.3f}")
            tprint(f"   - Success rates - mean: {np.mean(success_rates):.3f}, std: {np.std(success_rates):.3f}")
            tprint(f"   - Bounce strengths - mean: {np.mean(bounce_strengths):.3f}, std: {np.std(bounce_strengths):.3f}")
            
            self.logger.info(f"✅ Backtesting completed successfully!")
            self.logger.info(f"📊 Backtesting Results Summary:")
            self.logger.info(f"   - Levels backtested: {len(backtest_results)}")
            self.logger.info(f"   - Quality scores - mean: {np.mean(quality_scores):.3f}, std: {np.std(quality_scores):.3f}")
            self.logger.info(f"   - Quality scores - min: {np.min(quality_scores):.3f}, max: {np.max(quality_scores):.3f}")
            self.logger.info(f"   - Success rates - mean: {np.mean(success_rates):.3f}, std: {np.std(success_rates):.3f}")
            self.logger.info(f"   - Bounce strengths - mean: {np.mean(bounce_strengths):.3f}, std: {np.std(bounce_strengths):.3f}")
        else:
            tprint("⚠️ Backtesting completed but no results returned")
            self.logger.warning("⚠️ Backtesting completed but no results returned")
        
        return backtest_results
    
    def _update_quality_rules(self, backtest_results: List[Any], market_data: Optional[pd.DataFrame] = None) -> None:
        """Update quality rules based on backtesting results."""
        try:
            tprint(f"🧠 Updating quality rules from {len(backtest_results)} backtest results...")
            self.logger.info("🧠 Updating quality rules from backtesting results")
            self.logger.info(f"Processing {len(backtest_results)} backtest results")
            
            # Optimize SR parameters
            tprint("🎯 Optimizing SR parameters from backtesting results...")
            self.logger.info("Optimizing SR parameters from backtesting results")
            parameter_optimization_result = self.backtesting_engine.optimize_sr_parameters(
                backtest_results, 
                market_data=market_data
            )
            
            if parameter_optimization_result:
                tprint("✅ Successfully optimized SR parameters!")
                self.logger.info(f"✅ Successfully optimized SR parameters")
                
                # Store optimized parameters
                if parameter_optimization_result.get('optimization_success', False):
                    tprint("📝 Storing optimized parameters...")
                    self.logger.info("Storing optimized parameters")
                    self.optimized_parameters = parameter_optimization_result.get('optimized_parameters', {})
                    self.quality_thresholds = parameter_optimization_result.get('quality_thresholds', {})
                    tprint("✅ Parameters stored successfully")
                else:
                    tprint("⚠️ Parameter optimization failed, using fallback parameters")
                    self.logger.warning("Parameter optimization failed, using fallback parameters")
                    self.optimized_parameters = parameter_optimization_result.get('optimized_parameters', {})
                    self.quality_thresholds = parameter_optimization_result.get('quality_thresholds', {})
                
                # Log parameter optimization results
                optimized_params = parameter_optimization_result.get('optimized_parameters', {})
                if optimized_params:
                    tprint("🎯 Parameter optimization completed successfully!")
                    tprint(f"📊 Optimized parameters summary:")
                    self.logger.info(f"🎯 Parameter optimization completed successfully")
                    self.logger.info(f"Optimized parameters: {optimized_params}")
                    
                    # Log key parameters
                    key_params = ['touch_tolerance', 'min_bounce_strength', 'volume_threshold_multiplier', 'min_touches_required']
                    tprint("   Key optimized parameters:")
                    self.logger.info("Key optimized parameters:")
                    for param in key_params:
                        if param in optimized_params:
                            value = optimized_params[param]
                            if isinstance(value, float):
                                tprint(f"   - {param}: {value:.4f}")
                                self.logger.info(f"  - {param}: {value:.4f}")
                            else:
                                tprint(f"   - {param}: {value}")
                                self.logger.info(f"  - {param}: {value}")
                
                # Log quality thresholds
                quality_thresholds = parameter_optimization_result.get('quality_thresholds', {})
                if quality_thresholds:
                    tprint(f"📊 Quality thresholds calculated:")
                    self.logger.info(f"📊 Quality thresholds calculated:")
                    for threshold_name, threshold_value in quality_thresholds.items():
                        tprint(f"   - {threshold_name}: {threshold_value:.3f}")
                        self.logger.info(f"  - {threshold_name}: {threshold_value:.3f}")
                
                tprint("✅ Parameter optimization completed successfully!")
                self.logger.info(f"✅ Parameter optimization completed successfully")
            else:
                tprint("⚠️ Failed to optimize SR parameters")
                self.logger.warning("Failed to optimize SR parameters")
            
        except Exception as e:
            tprint(f"❌ Failed to update quality rules: {e}")
            self.logger.error(f"❌ Failed to update quality rules: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _filter_by_quality(self, levels: List[Dict], backtest_results: List[Any]) -> List[Dict]:
        """Filter levels based on quality assessment - only filter out VERY low quality levels."""
        try:
            tprint(f"🔍 Filtering {len(levels)} levels based on quality assessment...")
            self.logger.info(f"🔍 Filtering {len(levels)} levels based on quality assessment")
            
            # Create quality mapping
            tprint("📊 Creating quality mapping from backtest results...")
            quality_map = {r.level.price: r.quality_score for r in backtest_results}
            tprint(f"✅ Quality mapping created for {len(quality_map)} levels")
            self.logger.debug(f"Created quality mapping for {len(quality_map)} levels")
            
            # Calculate quality statistics to determine very low threshold
            quality_scores = list(quality_map.values())
            if quality_scores:
                quality_mean = np.mean(quality_scores)
                quality_std = np.std(quality_scores)
                quality_min = np.min(quality_scores)
                quality_max = np.max(quality_scores)
                quality_median = np.median(quality_scores)
                
                tprint(f"📈 Quality statistics:")
                tprint(f"   - Mean: {quality_mean:.3f}")
                tprint(f"   - Median: {quality_median:.3f}")
                tprint(f"   - Std: {quality_std:.3f}")
                tprint(f"   - Min: {quality_min:.3f}")
                tprint(f"   - Max: {quality_max:.3f}")
                
                self.logger.info(f"Quality statistics: mean={quality_mean:.3f}, std={quality_std:.3f}, min={quality_min:.3f}, max={quality_max:.3f}")
                
                # DISABLED: Quality filtering temporarily disabled to let hard 0.5% rule work
                # very_low_threshold = max(0.01, quality_mean - 3 * quality_std)
                very_low_threshold = 0.0  # Allow all levels through for now
                tprint(f"🎯 Quality filtering DISABLED - threshold: {very_low_threshold:.3f} (all levels allowed)")
                self.logger.info(f"Quality filtering DISABLED - threshold: {very_low_threshold:.3f} (all levels allowed)")
            else:
                very_low_threshold = 0.01  # Very lenient threshold to keep more levels
                tprint("⚠️ No quality scores available, using lenient threshold: 0.01")
                self.logger.warning("No quality scores available, using lenient threshold: 0.01")
            
            # Filter levels - only remove very low quality
            tprint("🔍 Applying quality filter...")
            filtered_levels = []
            filtered_count = 0
            kept_count = 0
            
            for i, level in enumerate(levels):
                quality_score = quality_map.get(level['price'], 0.0)
                
                # Only filter out VERY low quality levels
                if quality_score >= very_low_threshold:
                    # Add quality score to level data
                    level['backtest_quality'] = quality_score
                    level['quality_metrics'] = self._extract_quality_metrics(quality_score, backtest_results)
                    filtered_levels.append(level)
                    kept_count += 1
                    if i % 10 == 0:  # Log progress every 10 levels
                        tprint(f"   Processed {i+1}/{len(levels)} levels...")
                else:
                    filtered_count += 1
                    if filtered_count <= 5:  # Log first 5 filtered levels
                        tprint(f"   🗑️ Filtered out level ${level['price']:.2f} (quality: {quality_score:.3f} < {very_low_threshold:.3f})")
                    self.logger.debug(f"Filtered out very low quality level ${level['price']:.2f} (quality: {quality_score:.3f}, threshold: {very_low_threshold:.3f})")
            
            tprint(f"✅ Quality filtering completed!")
            tprint(f"📊 Filtering Results:")
            tprint(f"   - Input levels: {len(levels)}")
            tprint(f"   - Kept levels: {kept_count}")
            tprint(f"   - Filtered out: {filtered_count}")
            tprint(f"   - Filter rate: {filtered_count/len(levels)*100:.1f}%")
            tprint(f"   - Threshold used: {very_low_threshold:.3f}")
            
            self.logger.info(f"✅ Quality filtering completed: {len(levels)} -> {len(filtered_levels)} levels")
            self.logger.info(f"Filtered out {filtered_count} very low quality levels (threshold: {very_low_threshold:.3f})")
            
            if filtered_levels:
                filtered_qualities = [level['backtest_quality'] for level in filtered_levels]
                tprint(f"📈 Filtered level quality stats:")
                tprint(f"   - Mean: {np.mean(filtered_qualities):.3f}")
                tprint(f"   - Min: {np.min(filtered_qualities):.3f}")
                tprint(f"   - Max: {np.max(filtered_qualities):.3f}")
                self.logger.info(f"Filtered level quality stats: mean={np.mean(filtered_qualities):.3f}, min={np.min(filtered_qualities):.3f}, max={np.max(filtered_qualities):.3f}")
            
            return filtered_levels
            
        except Exception as e:
            tprint(f"❌ Quality filtering failed: {e}")
            self.logger.error(f"❌ Quality filtering failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return levels
    
    def _enhance_levels_with_quality(self, levels: List[Dict], backtest_results: List[Any]) -> List[Dict]:
        """Enhance level data with quality information."""
        try:
            tprint(f"✨ Enhancing {len(levels)} levels with quality information...")
            self.logger.info(f"✨ Enhancing {len(levels)} levels with quality information")
            
            # Create quality mapping
            tprint("📊 Creating quality mapping from backtest results...")
            quality_map = {r.level.price: r for r in backtest_results}
            tprint(f"✅ Quality mapping created for {len(quality_map)} backtest results")
            self.logger.debug(f"Created quality mapping for {len(quality_map)} backtest results")
            
            enhanced_levels = []
            enhancement_errors = 0
            enhancement_successes = 0
            
            tprint("🔄 Processing levels for enhancement...")
            for i, level in enumerate(levels):
                try:
                    enhanced_level = level.copy()
                    
                    # Add backtesting results
                    backtest_result = quality_map.get(level['price'])
                    if backtest_result:
                        enhanced_level.update({
                            'backtest_quality': backtest_result.quality_score,
                            'success_rate': backtest_result.success_rate,
                            'avg_bounce_strength': backtest_result.avg_bounce_strength,
                            'total_touches': backtest_result.total_touches,
                            'volume_confirmation': backtest_result.total_volume_at_level,
                            'time_persistence': backtest_result.time_persistence
                        })
                        enhancement_successes += 1
                        
                        if i % 10 == 0:  # Log progress every 10 levels
                            tprint(f"   Enhanced level {i+1}/{len(levels)}: ${level['price']:.2f}, quality={backtest_result.quality_score:.3f}")
                            self.logger.debug(f"Enhanced level {i+1}/{len(levels)}: price={level['price']:.2f}, quality={backtest_result.quality_score:.3f}")
                    else:
                        if enhancement_errors < 5:  # Log first 5 missing results
                            tprint(f"   ⚠️ No backtest result found for level at ${level['price']:.2f}")
                        self.logger.warning(f"No backtest result found for level at price {level['price']:.2f}")
                        enhancement_errors += 1
                    
                    enhanced_levels.append(enhanced_level)
                    
                except Exception as e:
                    if enhancement_errors < 5:  # Log first 5 errors
                        tprint(f"   ❌ Failed to enhance level {i+1}: {e}")
                    self.logger.warning(f"Failed to enhance level {i+1}: {e}")
                    enhancement_errors += 1
                    enhanced_levels.append(level)  # Add original level as fallback
            
            tprint(f"✅ Level enhancement completed!")
            tprint(f"📊 Enhancement Results:")
            tprint(f"   - Levels processed: {len(levels)}")
            tprint(f"   - Successfully enhanced: {enhancement_successes}")
            tprint(f"   - Enhancement errors: {enhancement_errors}")
            tprint(f"   - Success rate: {enhancement_successes/len(levels)*100:.1f}%")
            
            self.logger.info(f"✅ Level enhancement completed: {len(enhanced_levels)} levels enhanced ({enhancement_errors} errors)")
            
            # Log enhancement statistics
            if enhanced_levels:
                enhanced_qualities = [level.get('backtest_quality', 0.0) for level in enhanced_levels]
                enhanced_success_rates = [level.get('success_rate', 0.0) for level in enhanced_levels]
                enhanced_bounce_strengths = [level.get('avg_bounce_strength', 0.0) for level in enhanced_levels]
                
                tprint(f"📈 Enhanced level statistics:")
                tprint(f"   - Quality scores - mean: {np.mean(enhanced_qualities):.3f}, std: {np.std(enhanced_qualities):.3f}")
                tprint(f"   - Success rates - mean: {np.mean(enhanced_success_rates):.3f}, std: {np.std(enhanced_success_rates):.3f}")
                tprint(f"   - Bounce strengths - mean: {np.mean(enhanced_bounce_strengths):.3f}, std: {np.std(enhanced_bounce_strengths):.3f}")
                
                self.logger.info(f"Enhanced level quality stats: mean={np.mean(enhanced_qualities):.3f}, std={np.std(enhanced_qualities):.3f}")
                self.logger.info(f"Enhanced level success rates: mean={np.mean(enhanced_success_rates):.3f}, std={np.std(enhanced_success_rates):.3f}")
                self.logger.info(f"Enhanced level bounce strengths: mean={np.mean(enhanced_bounce_strengths):.3f}, std={np.std(enhanced_bounce_strengths):.3f}")
            
            return enhanced_levels
            
        except Exception as e:
            tprint(f"❌ Level enhancement failed: {e}")
            self.logger.error(f"❌ Level enhancement failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return levels
    
    def _cluster_quality_enhanced(self, levels: List[Dict], price_range: Tuple[float, float], data: pd.DataFrame = None) -> ClusteringResult:
        """Cluster levels using quality-enhanced approach."""
        try:
            tprint(f"🎯 Starting quality-enhanced clustering for {len(levels)} levels...")
            self.logger.info(f"🎯 Starting quality-enhanced clustering for {len(levels)} levels")
            
            # Use configured clustering parameters (disable automatic quality adjustment for now)
            tprint("🔧 Using configured clustering parameters...")
            self.logger.info("Using configured clustering parameters")

            # HARD RULE: Never merge levels more than 0.5% price apart
            adjusted_proximity = min(self.config.proximity_threshold, 0.005)
            adjusted_strength_threshold = self.config.strength_similarity_threshold

            tprint(f"📊 Clustering parameters (HARD RULE: max 0.5% proximity):")
            tprint(f"   - Proximity: {adjusted_proximity:.3f} (capped at 0.5%)")
            tprint(f"   - Strength threshold: {adjusted_strength_threshold:.3f}")

            self.logger.info(f"Clustering parameters: proximity={adjusted_proximity:.3f} (HARD RULE: max 0.5%), strength={adjusted_strength_threshold:.3f}")

            # Use extensive clustering system
            tprint("🚀 Running extensive clustering with configured parameters...")
            self.logger.info("Running extensive clustering with configured parameters")
            result = self._cluster_levels_extensive(
                levels=levels,
                price_range=price_range,
                proximity_threshold=adjusted_proximity,
                strength_similarity_threshold=adjusted_strength_threshold,
                data=data
            )
            
            # Add quality information to result
            tprint("✨ Adding quality information to clustering result...")
            result.quality_enhanced = True
            result.quality_metrics = self._calculate_cluster_quality_metrics(result, levels)
            
            tprint(f"✅ Quality-enhanced clustering completed!")
            tprint(f"📊 Quality-Enhanced Clustering Results:")
            tprint(f"   - Clusters created: {len(result.clusters)}")
            tprint(f"   - Quality score: {result.quality_score:.3f}")
            tprint(f"   - Quality enhanced: {result.quality_enhanced}")
            tprint(f"   - Quality metrics: {len(result.quality_metrics)} metrics calculated")
            
            self.logger.info(f"✅ Quality-enhanced clustering completed: {len(result.clusters)} clusters")
            self.logger.info(f"Quality metrics: {result.quality_metrics}")
            
            return result
            
        except Exception as e:
            tprint(f"❌ Quality-enhanced clustering failed: {e}")
            tprint("🔄 Falling back to standard clustering...")
            self.logger.error(f"❌ Quality-enhanced clustering failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.warning("🔄 Falling back to standard clustering")
            return self._fallback_clustering(levels, price_range)
    
    def _validate_clusters_with_backtesting(self, clustering_result: ClusteringResult, data: pd.DataFrame) -> ClusteringResult:
        """Validate clusters using backtesting."""
        try:
            self.logger.info(f"✅ Validating {len(clustering_result.clusters)} clusters with backtesting")
            
            validated_clusters = []
            cluster_quality_scores = []
            filtered_clusters = 0
            
            for i, cluster in enumerate(clustering_result.clusters):
                self.logger.debug(f"Validating cluster {i+1}: {len(cluster)} levels")
                
                if len(cluster) > 1:
                    # Validate cluster quality
                    cluster_quality = self._validate_cluster_quality(cluster, clustering_result.cluster_centers[i], data)
                    cluster_quality_scores.append(cluster_quality)
                    
                    if cluster_quality >= self.config.min_quality_score:
                        validated_clusters.append(cluster)
                        self.logger.debug(f"✅ Cluster {i+1} validated (quality: {cluster_quality:.3f} >= {self.config.min_quality_score:.3f})")
                    else:
                        filtered_clusters += 1
                        self.logger.debug(f"❌ Filtered out cluster {i+1} (quality: {cluster_quality:.3f} < {self.config.min_quality_score:.3f})")
                else:
                    # Single levels are always kept
                    validated_clusters.append(cluster)
                    cluster_quality_scores.append(1.0)
                    self.logger.debug(f"✅ Single-level cluster {i+1} kept (no validation needed)")
            
            # Update result
            clustering_result.clusters = validated_clusters
            clustering_result.cluster_centers = [clustering_result.cluster_centers[i] for i in range(len(validated_clusters))]
            clustering_result.quality_score = np.mean(cluster_quality_scores) if cluster_quality_scores else 0.0
            
            self.logger.info(f"✅ Cluster validation completed: {len(clustering_result.clusters)} clusters validated")
            self.logger.info(f"Filtered out {filtered_clusters} low-quality clusters")
            self.logger.info(f"Overall cluster quality score: {clustering_result.quality_score:.3f}")
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"❌ Cluster validation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return clustering_result
    
    def _adjust_proximity_by_quality(self, levels: List[Dict]) -> float:
        """Adjust proximity threshold based on level quality."""
        try:
            if not levels:
                self.logger.warning("No levels provided for proximity adjustment, using default")
                return self.config.proximity_threshold
            
            # Calculate average quality
            qualities = [level.get('backtest_quality', 0.5) for level in levels]
            avg_quality = np.mean(qualities)
            quality_std = np.std(qualities)
            
            self.logger.debug(f"Level quality stats: mean={avg_quality:.3f}, std={quality_std:.3f}")
            
            # Higher quality levels can be clustered more tightly
            # Lower quality levels need more separation
            quality_factor = 0.5 + (avg_quality * 0.5)  # Range: 0.5 to 1.0
            
            adjusted_proximity = self.config.proximity_threshold * quality_factor
            
            self.logger.debug(f"Proximity adjustment: original={self.config.proximity_threshold:.3f}, quality_factor={quality_factor:.3f}, adjusted={adjusted_proximity:.3f}")
            
            return adjusted_proximity
            
        except Exception as e:
            self.logger.warning(f"Proximity adjustment failed: {e}")
            return self.config.proximity_threshold
    
    def _adjust_strength_threshold_by_quality(self, levels: List[Dict]) -> float:
        """Adjust strength similarity threshold based on level quality."""
        try:
            if not levels:
                self.logger.warning("No levels provided for strength threshold adjustment, using default")
                return self.config.strength_similarity_threshold
            
            # Calculate quality variance
            qualities = [level.get('backtest_quality', 0.5) for level in levels]
            quality_variance = np.var(qualities)
            
            self.logger.debug(f"Level quality variance: {quality_variance:.3f}")
            
            # Higher variance means more diverse quality levels
            # Need stricter strength matching
            variance_factor = 1.0 + (quality_variance * 2.0)  # Range: 1.0 to 2.0
            
            adjusted_threshold = self.config.strength_similarity_threshold / variance_factor
            
            self.logger.debug(f"Strength threshold adjustment: original={self.config.strength_similarity_threshold:.3f}, variance_factor={variance_factor:.3f}, adjusted={adjusted_threshold:.3f}")
            
            return adjusted_threshold
            
        except Exception as e:
            self.logger.warning(f"Strength threshold adjustment failed: {e}")
            return self.config.strength_similarity_threshold
    
    def _validate_cluster_quality(self, cluster: List[int], cluster_center: float, data: pd.DataFrame) -> float:
        """Validate the quality of a cluster using backtesting."""
        try:
            self.logger.debug(f"Validating cluster quality for {len(cluster)} levels at center {cluster_center:.2f}")
            
            # Create a synthetic SR level for the cluster center
            cluster_center_level = SRLevel(
                price=cluster_center,
                level_type="support",  # Default to support, could be determined by context
                strength=len(cluster),  # Strength based on number of levels in cluster
                touches=len(cluster),   # Number of touches equals cluster size
                first_touch=None,      # Will be determined during backtesting
                last_touch=None,       # Will be determined during backtesting
                quality_score=0.0,     # Will be calculated by backtesting
                metadata={
                    "cluster_size": len(cluster),
                    "cluster_members": cluster,
                    "is_cluster_center": True,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # Backtest the cluster center level
            backtest_result = self.backtesting_engine.backtest_sr_level(
                level=cluster_center_level,
                data=data
            )
            
            # Extract quality score from backtest result
            cluster_quality = backtest_result.quality_score
            
            # Additional validation: compare with individual cluster members
            if cluster:
                # Get quality scores of individual cluster members
                member_qualities = []
                for level_idx in cluster:
                    if level_idx < len(self.levels):
                        level = self.levels[level_idx]
                        member_quality = level.get('backtest_quality', 0.5)
                        member_qualities.append(member_quality)
                
                if member_qualities:
                    avg_member_quality = np.mean(member_qualities)
                    min_member_quality = np.min(member_qualities)
                    max_member_quality = np.max(member_qualities)
                    
                    # Cluster center should be at least as good as the average member
                    # and ideally better than the worst member
                    quality_penalty = 0.0
                    if cluster_quality < avg_member_quality:
                        quality_penalty = (avg_member_quality - cluster_quality) * 0.5
                    
                    # Apply penalty if cluster center is significantly worse than members
                    cluster_quality = max(0.0, cluster_quality - quality_penalty)
                    
                    self.logger.debug(f"Cluster member qualities - avg: {avg_member_quality:.3f}, "
                                    f"min: {min_member_quality:.3f}, max: {max_member_quality:.3f}")
            
            # Ensure quality is within valid range
            cluster_quality = max(0.0, min(1.0, cluster_quality))
            
            self.logger.debug(f"Cluster quality validation result: {cluster_quality:.3f}")
            return cluster_quality
            
        except Exception as e:
            self.logger.warning(f"Cluster quality validation failed: {e}")
            return 0.5
    
    def _calculate_cluster_quality_metrics(self, result: ClusteringResult, levels: List[Dict]) -> Dict[str, Any]:
        """Calculate quality metrics for the clustering result."""
        try:
            self.logger.debug(f"Calculating quality metrics for {len(result.clusters)} clusters")
            
            if not result.clusters:
                self.logger.warning("No clusters to calculate metrics for")
                return {}
            
            # Calculate metrics for each cluster
            cluster_metrics = []
            for i, cluster in enumerate(result.clusters):
                if len(cluster) > 1:
                    cluster_levels = [levels[i] for i in cluster]
                    cluster_quality = np.mean([level.get('backtest_quality', 0.5) for level in cluster_levels])
                    cluster_variance = np.var([level.get('backtest_quality', 0.5) for level in cluster_levels])
                    
                    cluster_metrics.append({
                        'size': len(cluster),
                        'avg_quality': cluster_quality,
                        'quality_variance': cluster_variance
                    })
                    
                    self.logger.debug(f"Cluster {i+1} metrics: size={len(cluster)}, avg_quality={cluster_quality:.3f}, variance={cluster_variance:.3f}")
            
            metrics = {
                'total_clusters': len(result.clusters),
                'avg_cluster_quality': np.mean([m['avg_quality'] for m in cluster_metrics]) if cluster_metrics else 0.0,
                'avg_cluster_size': np.mean([m['size'] for m in cluster_metrics]) if cluster_metrics else 0.0,
                'quality_consistency': 1.0 - np.mean([m['quality_variance'] for m in cluster_metrics]) if cluster_metrics else 0.0
            }
            
            self.logger.debug(f"Overall cluster quality metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Cluster quality metrics calculation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _extract_quality_metrics(self, quality_score: float, backtest_results: List[Any]) -> Dict[str, float]:
        """Extract quality metrics for a level."""
        try:
            # Find the backtest result for this quality score
            for result in backtest_results:
                if abs(result.quality_score - quality_score) < 0.001:
                    metrics = {
                        'success_rate': result.success_rate,
                        'bounce_strength': result.avg_bounce_strength,
                        'total_touches': result.total_touches,
                        'volume_confirmation': result.total_volume_at_level,
                        'time_persistence': result.time_persistence
                    }
                    self.logger.debug(f"Extracted quality metrics for score {quality_score:.3f}: {metrics}")
                    return metrics
            
            self.logger.warning(f"No quality metrics found for score {quality_score:.3f}")
            return {}
            
        except Exception as e:
            self.logger.warning(f"Failed to extract quality metrics: {e}")
            return {}
    
    def _merge_rules(self, existing_rules: Dict[str, Any], new_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new rules with existing rules using weighted average."""
        try:
            self.logger.info("Merging new rules with existing learned rules")
            
            merged_rules = existing_rules.copy()
            
            # Merge discriminative features
            if 'discriminative_features' in new_rules:
                existing_features = merged_rules.get('discriminative_features', {})
                new_features = new_rules['discriminative_features']
                
                self.logger.debug(f"Merging discriminative features: {len(existing_features)} existing, {len(new_features)} new")
                
                for feature, info in new_features.items():
                    if feature in existing_features:
                        # Weighted average (70% existing, 30% new)
                        existing_info = existing_features[feature]
                        merged_info = {
                            'high_mean': 0.7 * existing_info['high_mean'] + 0.3 * info['high_mean'],
                            'low_mean': 0.7 * existing_info['low_mean'] + 0.3 * info['low_mean'],
                            'discriminative_power': 0.7 * existing_info['discriminative_power'] + 0.3 * info['discriminative_power'],
                            'threshold': 0.7 * existing_info['threshold'] + 0.3 * info['threshold']
                        }
                        existing_features[feature] = merged_info
                        self.logger.debug(f"Merged feature {feature} using weighted average")
                    else:
                        existing_features[feature] = info
                        self.logger.debug(f"Added new feature {feature}")
                
                merged_rules['discriminative_features'] = existing_features
            
            # Update quality threshold
            if 'quality_threshold' in new_rules:
                old_threshold = merged_rules.get('quality_threshold', 0.5)
                new_threshold = new_rules['quality_threshold']
                merged_threshold = 0.7 * old_threshold + 0.3 * new_threshold
                merged_rules['quality_threshold'] = merged_threshold
                self.logger.debug(f"Updated quality threshold: {old_threshold:.3f} -> {merged_threshold:.3f}")
            
            self.logger.info("✅ Rule merging completed successfully")
            return merged_rules
            
        except Exception as e:
            self.logger.error(f"❌ Rule merging failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return existing_rules
    
    def _fallback_clustering(self, levels: List[Dict], price_range: Tuple[float, float]) -> ClusteringResult:
        """Fallback to standard clustering if backtesting-enhanced approach fails."""
        self.logger.warning("🔄 Falling back to standard clustering")
        self.logger.info(f"Fallback clustering for {len(levels)} levels with price range {price_range}")
        
        try:
            result = self._cluster_levels_extensive(
                levels=levels,
                price_range=price_range,
                proximity_threshold=self.config.proximity_threshold,
                strength_similarity_threshold=self.config.strength_similarity_threshold,
                data=None  # No data available in fallback
            )
            
            self.logger.info(f"✅ Fallback clustering completed: {len(result.clusters)} clusters")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Fallback clustering also failed: {e}")
            # Return minimal result
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                quality_score=0.5,
                quality_enhanced=False,
                algorithm_used='fallback_individual',
                parameters={}
            )
    
    def _cluster_levels_extensive(self, levels: List[Dict], price_range: Tuple[float, float], 
                                 proximity_threshold: float, strength_similarity_threshold: float, 
                                 data: pd.DataFrame = None) -> ClusteringResult:
        """Extensive clustering implementation using enhanced SR detection system."""
        try:
            self.logger.info(f"Running extensive clustering with {len(levels)} levels")
            
            if not levels:
                self.logger.warning("No levels provided for extensive clustering")
                return ClusteringResult(clusters=[], cluster_centers=[], quality_score=0.0, algorithm_used='extensive_clustering', parameters={})
            
            # Use the new StrengthProximityClustering algorithm
            self.logger.info("Using StrengthProximityClustering algorithm")
            result = self.strength_proximity_clustering.cluster(
                levels=levels,
                price_range=price_range,
                proximity_threshold=proximity_threshold,
                strength_similarity_threshold=strength_similarity_threshold
            )
            
            self.logger.info(f"✅ Extensive clustering completed: {len(result.clusters)} clusters")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Extensive clustering failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to simple clustering
            self.logger.warning("🔄 Falling back to simple clustering")
            return self._cluster_levels_simple_fallback(levels, price_range, proximity_threshold, strength_similarity_threshold)
    
    def _cluster_levels_simple_fallback(self, levels: List[Dict], price_range: Tuple[float, float], 
                                       proximity_threshold: float, strength_similarity_threshold: float) -> ClusteringResult:
        """Simple fallback clustering implementation."""
        try:
            self.logger.info(f"Running simple fallback clustering with {len(levels)} levels")
            
            if not levels:
                self.logger.warning("No levels provided for simple fallback clustering")
                return ClusteringResult(clusters=[], cluster_centers=[], quality_score=0.0, algorithm_used='extensive_clustering', parameters={})
            
            # Sort levels by price
            sorted_levels = sorted(enumerate(levels), key=lambda x: x[1]['price'])
            self.logger.debug(f"Sorted {len(sorted_levels)} levels by price")
            
            clusters = []
            cluster_centers = []
            current_cluster = [sorted_levels[0][0]]  # Start with first level index
            
            for i in range(1, len(sorted_levels)):
                level_idx, level = sorted_levels[i]
                current_cluster_center = np.mean([levels[idx]['price'] for idx in current_cluster])
                
                # Check proximity
                price_diff = abs(level['price'] - current_cluster_center) / current_cluster_center
                
                # Check strength similarity
                current_cluster_strength = np.mean([levels[idx].get('strength', 0.5) for idx in current_cluster])
                strength_diff = abs(level.get('strength', 0.5) - current_cluster_strength)
                
                # Add to cluster if both proximity and strength are similar
                if price_diff <= proximity_threshold and strength_diff <= strength_similarity_threshold:
                    current_cluster.append(level_idx)
                    self.logger.debug(f"Added level {level_idx} to current cluster (price_diff: {price_diff:.3f}, strength_diff: {strength_diff:.3f})")
                else:
                    # Finalize current cluster
                    if current_cluster:
                        clusters.append(current_cluster)
                        cluster_centers.append(current_cluster_center)
                        self.logger.debug(f"Finalized cluster with {len(current_cluster)} levels")
                    current_cluster = [level_idx]
            
            # Add final cluster
            if current_cluster:
                clusters.append(current_cluster)
                cluster_centers.append(np.mean([levels[idx]['price'] for idx in current_cluster]))
                self.logger.debug(f"Added final cluster with {len(current_cluster)} levels")
            
            # Calculate quality score
            quality_score = self._calculate_overall_quality_score(clusters, levels)
            
            self.logger.info(f"✅ Simple fallback clustering completed: {len(clusters)} clusters, quality: {quality_score:.3f}")
            
            return ClusteringResult(
                clusters=clusters,
                cluster_centers=cluster_centers,
                quality_score=quality_score,
                algorithm_used='simple_fallback',
                parameters={}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Simple fallback clustering failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return single-level clusters as final fallback
            self.logger.warning("🔄 Using single-level clusters as final fallback")
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                cluster_centers=[level['price'] for level in levels],
                quality_score=0.5,
                algorithm_used='final_fallback',
                parameters={}
            )
    
    def _calculate_overall_quality_score(self, clusters: List[List[int]], levels: List[Dict]) -> float:
        """Calculate overall quality score for clustering result."""
        try:
            self.logger.debug(f"Calculating overall quality score for {len(clusters)} clusters")
            
            if not clusters:
                self.logger.warning("No clusters provided for quality score calculation")
                return 0.0
            
            total_quality = 0.0
            total_levels = 0
            
            for i, cluster in enumerate(clusters):
                cluster_quality = np.mean([levels[idx].get('backtest_quality', 0.5) for idx in cluster])
                total_quality += cluster_quality * len(cluster)
                total_levels += len(cluster)
                self.logger.debug(f"Cluster {i+1}: {len(cluster)} levels, avg quality: {cluster_quality:.3f}")
            
            overall_quality = total_quality / total_levels if total_levels > 0 else 0.0
            self.logger.debug(f"Overall quality score: {overall_quality:.3f}")
            
            return overall_quality
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.5
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning process."""
        self.logger.info("Generating learning summary")
        
        summary = {
            'levels_processed': self.levels_processed,
            'rules_learned': bool(self.learned_rules),
            'quality_rules_summary': self.backtesting_engine.get_quality_rules_summary(),
            'learned_features': list(self.learned_rules.get('discriminative_features', {}).keys()) if self.learned_rules else []
        }
        
        self.logger.info(f"Learning summary: {summary}")
        return summary
    
    def predict_level_quality(self, level: Dict[str, Any], data: pd.DataFrame) -> float:
        """Predict the quality of a level using learned rules."""
        try:
            self.logger.debug(f"Predicting quality for level at price {level.get('price', 0.0):.2f}")
            
            sr_level = create_sr_level_from_dict(level)
            predicted_quality = self.backtesting_engine.predict_level_quality(sr_level, data)
            
            self.logger.debug(f"Predicted quality: {predicted_quality:.3f}")
            return predicted_quality
            
        except Exception as e:
            self.logger.warning(f"Quality prediction failed: {e}")
            fallback_quality = level.get('strength', 0.5)
            self.logger.debug(f"Using fallback quality: {fallback_quality:.3f}")
            return fallback_quality
    
    def _log_parameter_optimization_summary(self, parameter_optimization_result: Dict[str, Any]) -> None:
        """Log a summary of parameter optimization results."""
        try:
            self.logger.info("Logging parameter optimization summary")
            
            optimization_success = parameter_optimization_result.get('optimization_success', False)
            optimization_method = parameter_optimization_result.get('optimization_method', 'unknown')
            optimization_score = parameter_optimization_result.get('optimization_score', 0.0)
            
            self.logger.info(f"Parameter optimization summary:")
            self.logger.info(f"  - Success: {optimization_success}")
            self.logger.info(f"  - Method: {optimization_method}")
            self.logger.info(f"  - Score: {optimization_score:.4f}")
            
            optimized_params = parameter_optimization_result.get('optimized_parameters', {})
            if optimized_params:
                self.logger.info(f"  - Parameters optimized: {len(optimized_params)}")
                
                # Log key parameters
                key_params = ['touch_tolerance', 'min_bounce_strength', 'volume_threshold_multiplier', 'min_touches_required']
                for param in key_params:
                    if param in optimized_params:
                        value = optimized_params[param]
                        self.logger.info(f"    - {param}: {value}")
            
            quality_thresholds = parameter_optimization_result.get('quality_thresholds', {})
            if quality_thresholds:
                self.logger.info(f"  - Quality thresholds calculated: {len(quality_thresholds)}")
                for threshold_name, threshold_value in quality_thresholds.items():
                    self.logger.info(f"    - {threshold_name}: {threshold_value:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log parameter optimization summary: {e}")

def get_backtesting_enhanced_clustering(config: Optional[BacktestingEnhancedConfig] = None) -> BacktestingEnhancedClustering:
    """Get a backtesting-enhanced clustering instance."""
    logger = system_logger.getChild('BacktestingEnhancedClustering')
    logger.info("Creating new BacktestingEnhancedClustering instance")
    
    try:
        instance = BacktestingEnhancedClustering(config)
        logger.info("✅ Successfully created BacktestingEnhancedClustering instance")
        return instance
    except Exception as e:
        logger.error(f"❌ Failed to create BacktestingEnhancedClustering instance: {e}")
        raise