"""
Backtesting-Enhanced Clustering for SR Levels

This module integrates the backtesting engine with the clustering system to create
SR levels based on learned quality rules from historical performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
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
        """Cluster levels - to be implemented by subclasses."""
        self.logger.info(f"Starting clustering with {len(levels)} levels, price range: {price_range}")
        self.logger.debug(f"Clustering parameters: {kwargs}")
        raise NotImplementedError

class StrengthProximityClustering(BaseClusteringAlgorithm):
    """Strength and proximity-based clustering for SR levels."""
    
    def __init__(self):
        super().__init__("StrengthProximity")
    
    def cluster(self, levels: List[Dict], price_range: Tuple[float, float], 
                proximity_threshold: float = 0.01, strength_similarity_threshold: float = 0.2) -> 'ClusteringResult':
        """Cluster levels based on strength and proximity using adaptive thresholds."""
        
        print(f"🎯 Starting strength-proximity clustering with {len(levels)} levels...")
        print(f"📊 Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
        print(f"⚙️ Proximity threshold: {proximity_threshold:.3f} ({proximity_threshold*100:.1f}% of price range)")
        print(f"⚙️ Strength similarity threshold: {strength_similarity_threshold:.3f}")
        
        self.logger.info(f"Starting strength-proximity clustering with {len(levels)} levels")
        self.logger.info(f"Price range: {price_range[0]:.2f} - {price_range[1]:.2f}")
        self.logger.info(f"Proximity threshold: {proximity_threshold:.3f}, Strength threshold: {strength_similarity_threshold:.3f}")
        
        if not levels:
            print("⚠️ No levels provided for clustering, returning empty result")
            self.logger.warning("No levels provided for clustering, returning empty result")
            return ClusteringResult(
                clusters=[],
                cluster_centers=[],
                quality_score=0.0,
                quality_enhanced=False,
                quality_metrics={'algorithm_used': 'StrengthProximity', 'total_levels': 0}
            )
        
        try:
            # Convert proximity threshold to absolute price difference
            min_price, max_price = price_range
            price_range_size = max_price - min_price
            absolute_proximity_threshold = proximity_threshold * price_range_size
            
            print(f"🔢 Calculated absolute proximity threshold: ${absolute_proximity_threshold:.2f}")
            self.logger.info(f"Clustering {len(levels)} levels with proximity threshold: {absolute_proximity_threshold:.2f} ({proximity_threshold:.1%} of price range)")
            self.logger.info(f"Strength similarity threshold: {strength_similarity_threshold:.2f}")
            
            # Log level distribution
            print("📊 Analyzing level distribution...")
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
            
            print(f"📈 Level type distribution: {level_types}")
            print(f"💪 Strength distribution: {strength_ranges}")
            self.logger.info(f"Level type distribution: {level_types}")
            self.logger.info(f"Strength distribution: {strength_ranges}")
            
            # Initialize clusters
            print("🔄 Starting clustering process...")
            clusters = []
            unassigned_levels = list(range(len(levels)))
            cluster_count = 0
            
            while unassigned_levels:
                cluster_count += 1
                print(f"   Creating cluster {cluster_count}...")
                
                # Start new cluster with strongest unassigned level
                current_cluster = []
                seed_idx = self._find_strongest_level(levels, unassigned_levels)
                current_cluster.append(seed_idx)
                unassigned_levels.remove(seed_idx)
                
                print(f"   Seed level: ${levels[seed_idx].get('price', 0.0):.2f} (strength: {levels[seed_idx].get('strength', 0.5):.3f})")
                
                # Find all levels that should be in this cluster
                initial_unassigned_count = len(unassigned_levels)
                self._grow_cluster(levels, current_cluster, unassigned_levels, 
                                 absolute_proximity_threshold, strength_similarity_threshold)
                levels_added = initial_unassigned_count - len(unassigned_levels)
                
                print(f"   Cluster {cluster_count} created with {len(current_cluster)} levels ({levels_added} levels added)")
                
                clusters.append(current_cluster)
                
                # Remove assigned levels from unassigned
                for idx in current_cluster[1:]:  # Skip seed (already removed)
                    if idx in unassigned_levels:
                        unassigned_levels.remove(idx)
            
            print(f"✅ Clustering process completed: {len(clusters)} clusters created")
            
            # Calculate cluster centers and quality
            print("📊 Calculating cluster centers and quality scores...")
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
                    
                    print(f"   Cluster {i+1}: {len(cluster)} levels, center: ${weighted_center:.2f}, quality: {cluster_quality:.3f}")
            
            # Overall quality score
            quality_score = total_quality / len(clusters) if clusters else 0.0
            
            print(f"🎉 Strength-proximity clustering completed!")
            print(f"📊 Final Results:")
            print(f"   - Input levels: {len(levels)}")
            print(f"   - Clusters created: {len(clusters)}")
            print(f"   - Average cluster size: {len(levels) / len(clusters):.1f} levels")
            print(f"   - Overall quality score: {quality_score:.3f}")
            
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
                    'algorithm_used': 'StrengthProximity',
                    'proximity_threshold': proximity_threshold,
                    'strength_similarity_threshold': strength_similarity_threshold,
                    'absolute_proximity_threshold': absolute_proximity_threshold,
                    'total_levels': len(levels),
                    'avg_cluster_size': len(levels) / len(clusters) if clusters else 0
                }
            )
            
        except Exception as e:
            print(f"❌ Strength-proximity clustering failed: {e}")
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
        
        print(f"      Growing cluster with {len(current_cluster)} levels, checking {len(unassigned_levels)} unassigned levels...")
        self.logger.debug(f"Growing cluster with {len(current_cluster)} levels, checking {len(unassigned_levels)} unassigned levels")
        
        # Get cluster characteristics
        cluster_prices = [levels[i].get('price', 0.0) for i in current_cluster]
        cluster_strengths = [levels[i].get('strength', 0.5) for i in current_cluster]
        cluster_center_price = sum(cluster_prices) / len(cluster_prices)
        cluster_avg_strength = sum(cluster_strengths) / len(cluster_strengths)
        
        print(f"      Cluster center: ${cluster_center_price:.2f}, avg strength: {cluster_avg_strength:.3f}")
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
            print(f"         Adding level ${level_price:.2f} (strength: {level_strength:.3f}) to cluster")
            self.logger.debug(f"Adding level {idx} (price: {level_price:.2f}, strength: {level_strength:.3f}) to cluster")
        
        print(f"      Cluster growth: {len(levels_to_add)} levels added, {proximity_rejected} rejected by proximity, {strength_rejected} rejected by strength")
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
    
    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = {}

@dataclass
class BacktestingEnhancedConfig:
    """Configuration for backtesting-enhanced clustering."""
    # Backtesting parameters
    backtest_config: BacktestConfig = None
    
    # Clustering parameters
    proximity_threshold: float = 0.01  # 1% of price range
    strength_similarity_threshold: float = 0.2  # 20% strength difference
    
    # Quality filtering
    min_quality_score: float = 0.3  # Minimum quality score to keep level
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
        
        print("🚀 Initializing BacktestingEnhancedClustering system...")
        self.logger.info("🚀 Initializing BacktestingEnhancedClustering system...")
        
        print(f"📋 Configuration loaded:")
        print(f"   - Proximity threshold: {self.config.proximity_threshold:.3f} ({self.config.proximity_threshold*100:.1f}% of price range)")
        print(f"   - Strength similarity threshold: {self.config.strength_similarity_threshold:.3f}")
        print(f"   - Min quality score: {self.config.min_quality_score:.3f}")
        print(f"   - Quality weight in clustering: {self.config.quality_weight_in_clustering:.3f}")
        print(f"   - Min levels for learning: {self.config.min_levels_for_learning}")
        print(f"   - Learning update frequency: {self.config.learning_update_frequency}")
        print(f"   - Min success rate: {self.config.min_success_rate:.3f}")
        print(f"   - Min bounce strength: {self.config.min_bounce_strength:.3f}")
        
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
        print("🔧 Initializing backtesting engine...")
        self.logger.info("🔧 Initializing backtesting engine...")
        self.backtesting_engine = SRBacktestingEngine(self.config.backtest_config)
        print("✅ Backtesting engine initialized successfully")
        self.logger.info("✅ Backtesting engine initialized successfully")
        
        # Initialize extensive clustering system if available
        print("🔍 Checking for enhanced SR detector...")
        self.logger.info("🔍 Checking for enhanced SR detector...")
        if EXTENSIVE_CLUSTERING_AVAILABLE:
            print("✅ Enhanced SR detector available, initializing...")
            self.logger.info("✅ Enhanced SR detector available, initializing...")
            self.enhanced_sr_detector = EnhancedSRDetector({})
            print("✅ Enhanced SR detector initialized successfully")
            self.logger.info("✅ Enhanced SR detector initialized successfully")
        else:
            print("⚠️ Enhanced SR detector not available, using fallback methods")
            self.logger.warning("⚠️ Enhanced SR detector not available, using fallback methods")
            self.enhanced_sr_detector = None
        
        # Initialize strength-proximity clustering algorithm
        print("🎯 Initializing strength-proximity clustering algorithm...")
        self.logger.info("🎯 Initializing strength-proximity clustering algorithm...")
        self.strength_proximity_clustering = StrengthProximityClustering()
        print("✅ Strength-proximity clustering algorithm initialized")
        self.logger.info("✅ Strength-proximity clustering algorithm initialized")
        
        # Learning state
        print("🧠 Initializing learning state...")
        self.logger.info("🧠 Initializing learning state...")
        self.learned_rules = {}
        self.quality_predictions = {}
        self.levels_processed = 0
        print("✅ Learning state initialized")
        self.logger.info("✅ Learning state initialized")
        
        print("🎉 BacktestingEnhancedClustering initialization completed successfully!")
        self.logger.info("🎉 BacktestingEnhancedClustering initialization completed successfully!")
        
    def cluster_with_backtesting(self, levels: List[Dict], data: pd.DataFrame, 
                                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster levels using backtesting-enhanced approach."""
        try:
            print("=" * 80)
            print("🚀 STARTING BACKTESTING-ENHANCED CLUSTERING")
            print("=" * 80)
            print(f"📊 Input Summary:")
            print(f"   - Levels to cluster: {len(levels)}")
            print(f"   - Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
            print(f"   - Price range size: ${price_range[1] - price_range[0]:.2f}")
            print(f"   - Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            print(f"   - Data time range: {data.index.min()} to {data.index.max()}")
            
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
            print("\n" + "=" * 60)
            print("📊 STEP 1: BACKTESTING LEVELS TO ASSESS QUALITY")
            print("=" * 60)
            self.logger.info("📊 STEP 1: BACKTESTING LEVELS TO ASSESS QUALITY")
            backtest_results = self._backtest_levels(levels, data)
            
            # Step 2: Learn/update quality rules
            print("\n" + "=" * 60)
            print("🧠 STEP 2: LEARNING/UPDATING QUALITY RULES")
            print("=" * 60)
            self.logger.info("🧠 STEP 2: LEARNING/UPDATING QUALITY RULES")
            if len(backtest_results) >= self.config.min_levels_for_learning:
                print(f"✅ Sufficient levels for learning: {len(backtest_results)} >= {self.config.min_levels_for_learning}")
                self.logger.info(f"✅ Sufficient levels for learning: {len(backtest_results)} >= {self.config.min_levels_for_learning}")
                self._update_quality_rules(backtest_results, data)
            else:
                print(f"⚠️ Insufficient levels for learning: {len(backtest_results)} < {self.config.min_levels_for_learning}")
                self.logger.warning(f"⚠️ Insufficient levels for learning: {len(backtest_results)} < {self.config.min_levels_for_learning}")
            
            # Step 3: Filter levels based on quality
            print("\n" + "=" * 60)
            print("🔍 STEP 3: FILTERING LEVELS BASED ON QUALITY")
            print("=" * 60)
            self.logger.info("🔍 STEP 3: FILTERING LEVELS BASED ON QUALITY")
            quality_filtered_levels = self._filter_by_quality(levels, backtest_results)
            
            # Step 4: Enhance level data with quality scores
            print("\n" + "=" * 60)
            print("✨ STEP 4: ENHANCING LEVEL DATA WITH QUALITY SCORES")
            print("=" * 60)
            self.logger.info("✨ STEP 4: ENHANCING LEVEL DATA WITH QUALITY SCORES")
            enhanced_levels = self._enhance_levels_with_quality(quality_filtered_levels, backtest_results)
            
            # Step 5: Cluster using quality-enhanced approach
            print("\n" + "=" * 60)
            print("🎯 STEP 5: CLUSTERING USING QUALITY-ENHANCED APPROACH")
            print("=" * 60)
            self.logger.info("🎯 STEP 5: CLUSTERING USING QUALITY-ENHANCED APPROACH")
            clustering_result = self._cluster_quality_enhanced(enhanced_levels, price_range, data)
            
            # Step 6: Post-process clusters with quality validation
            print("\n" + "=" * 60)
            print("✅ STEP 6: POST-PROCESSING CLUSTERS WITH QUALITY VALIDATION")
            print("=" * 60)
            self.logger.info("✅ STEP 6: POST-PROCESSING CLUSTERS WITH QUALITY VALIDATION")
            final_result = self._validate_clusters_with_backtesting(clustering_result, data)
            
            self.levels_processed += len(levels)
            
            print("\n" + "=" * 80)
            print("🎉 BACKTESTING-ENHANCED CLUSTERING COMPLETED!")
            print("=" * 80)
            print(f"📈 Final Results:")
            print(f"   - Input levels: {len(levels)}")
            print(f"   - Final clusters: {len(final_result.clusters)}")
            print(f"   - Quality score: {final_result.quality_score:.3f}")
            print(f"   - Quality enhanced: {final_result.quality_enhanced}")
            print(f"   - Total levels processed: {self.levels_processed}")
            print("=" * 80)
            
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
            print(f"\n❌ BACKTESTING-ENHANCED CLUSTERING FAILED: {e}")
            print("🔄 Falling back to standard clustering...")
            self.logger.error(f"❌ Backtesting-enhanced clustering failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to standard clustering
            self.logger.warning("🔄 Falling back to standard clustering")
            return self._fallback_clustering(levels, price_range)
    
    def _backtest_levels(self, levels: List[Dict], data: pd.DataFrame) -> List[Any]:
        """Backtest all levels to assess their quality."""
        print(f"📈 Starting backtesting of {len(levels)} levels for quality assessment...")
        self.logger.info(f"📈 Starting backtesting of {len(levels)} levels for quality assessment...")
        
        # Convert to SRLevel objects
        print("🔄 Converting level dictionaries to SRLevel objects...")
        self.logger.info("🔄 Converting level dictionaries to SRLevel objects...")
        sr_levels = []
        conversion_errors = 0
        for i, level_dict in enumerate(levels):
            try:
                sr_level = create_sr_level_from_dict(level_dict)
                sr_levels.append(sr_level)
                if i % 10 == 0:  # Log progress every 10 levels
                    print(f"   Converted {i+1}/{len(levels)} levels to SRLevel objects...")
                    self.logger.debug(f"Converted {i+1}/{len(levels)} levels to SRLevel objects")
            except Exception as e:
                print(f"   ⚠️ Failed to create SRLevel from level {i+1}: {e}")
                self.logger.warning(f"Failed to create SRLevel from {level_dict}: {e}")
                conversion_errors += 1
                continue
        
        print(f"✅ Conversion completed: {len(sr_levels)} levels converted ({conversion_errors} errors)")
        self.logger.info(f"✅ Conversion completed: {len(sr_levels)} levels converted ({conversion_errors} errors)")
        
        if not sr_levels:
            print("❌ No valid SRLevel objects created, cannot proceed with backtesting")
            self.logger.error("No valid SRLevel objects created, cannot proceed with backtesting")
            return []
        
        # Log level distribution
        if sr_levels:
            support_count = sum(1 for level in sr_levels if level.level_type == 'support')
            resistance_count = sum(1 for level in sr_levels if level.level_type == 'resistance')
            avg_strength = np.mean([level.strength for level in sr_levels])
            avg_touches = np.mean([level.touches for level in sr_levels])
            
            print(f"📊 Level distribution:")
            print(f"   - Support levels: {support_count}")
            print(f"   - Resistance levels: {resistance_count}")
            print(f"   - Average strength: {avg_strength:.3f}")
            print(f"   - Average touches: {avg_touches:.1f}")
            
            self.logger.info(f"📊 Level distribution:")
            self.logger.info(f"   - Support levels: {support_count}")
            self.logger.info(f"   - Resistance levels: {resistance_count}")
            self.logger.info(f"   - Average strength: {avg_strength:.3f}")
            self.logger.info(f"   - Average touches: {avg_touches:.1f}")
        
        # Backtest levels
        print("🚀 Running backtesting engine on SRLevel objects...")
        self.logger.info("🚀 Running backtesting engine on SRLevel objects...")
        backtest_results = self.backtesting_engine.backtest_multiple_levels(sr_levels, data)
        
        if backtest_results:
            quality_scores = [r.quality_score for r in backtest_results]
            success_rates = [r.success_rate for r in backtest_results]
            bounce_strengths = [r.avg_bounce_strength for r in backtest_results]
            
            print(f"✅ Backtesting completed successfully!")
            print(f"📊 Backtesting Results Summary:")
            print(f"   - Levels backtested: {len(backtest_results)}")
            print(f"   - Quality scores - mean: {np.mean(quality_scores):.3f}, std: {np.std(quality_scores):.3f}")
            print(f"   - Quality scores - min: {np.min(quality_scores):.3f}, max: {np.max(quality_scores):.3f}")
            print(f"   - Success rates - mean: {np.mean(success_rates):.3f}, std: {np.std(success_rates):.3f}")
            print(f"   - Bounce strengths - mean: {np.mean(bounce_strengths):.3f}, std: {np.std(bounce_strengths):.3f}")
            
            self.logger.info(f"✅ Backtesting completed successfully!")
            self.logger.info(f"📊 Backtesting Results Summary:")
            self.logger.info(f"   - Levels backtested: {len(backtest_results)}")
            self.logger.info(f"   - Quality scores - mean: {np.mean(quality_scores):.3f}, std: {np.std(quality_scores):.3f}")
            self.logger.info(f"   - Quality scores - min: {np.min(quality_scores):.3f}, max: {np.max(quality_scores):.3f}")
            self.logger.info(f"   - Success rates - mean: {np.mean(success_rates):.3f}, std: {np.std(success_rates):.3f}")
            self.logger.info(f"   - Bounce strengths - mean: {np.mean(bounce_strengths):.3f}, std: {np.std(bounce_strengths):.3f}")
        else:
            print("⚠️ Backtesting completed but no results returned")
            self.logger.warning("⚠️ Backtesting completed but no results returned")
        
        return backtest_results
    
    def _update_quality_rules(self, backtest_results: List[Any], market_data: Optional[pd.DataFrame] = None) -> None:
        """Update quality rules based on backtesting results."""
        try:
            print(f"🧠 Updating quality rules from {len(backtest_results)} backtest results...")
            self.logger.info("🧠 Updating quality rules from backtesting results")
            self.logger.info(f"Processing {len(backtest_results)} backtest results")
            
            # Learn new rules with weight optimization
            print("🔬 Learning new quality rules with weight optimization...")
            self.logger.info("Learning new quality rules with weight optimization")
            new_rules = self.backtesting_engine.learn_quality_rules(
                backtest_results, 
                optimize_weights=True, 
                market_data=market_data
            )
            
            if new_rules:
                print("✅ Successfully learned new quality rules!")
                self.logger.info(f"✅ Successfully learned new quality rules")
                
                # Merge with existing rules (weighted average)
                if self.learned_rules:
                    print("🔄 Merging new rules with existing learned rules...")
                    self.logger.info("Merging new rules with existing learned rules")
                    self.learned_rules = self._merge_rules(self.learned_rules, new_rules)
                    print("✅ Rules merged successfully")
                else:
                    print("📝 No existing rules found, using new rules as base")
                    self.logger.info("No existing rules found, using new rules as base")
                    self.learned_rules = new_rules
                
                # Log optimization results
                if new_rules.get('weight_optimization_enabled', False):
                    optimized_weights = new_rules.get('optimized_weights', {})
                    if optimized_weights:
                        print("🎯 Weight optimization completed successfully!")
                        print(f"📊 Optimized weights summary:")
                        self.logger.info(f"🎯 Weight optimization completed successfully")
                        self.logger.info(f"Optimized weights: {optimized_weights}")
                        
                        # Log top weights
                        sorted_weights = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)
                        print("   Top 5 optimized weights:")
                        self.logger.info("Top 5 optimized weights:")
                        for i, (feature, weight) in enumerate(sorted_weights[:5], 1):
                            print(f"   {i}. {feature}: {weight:.3f}")
                            self.logger.info(f"  {feature}: {weight:.3f}")
                    else:
                        print("⚠️ Weight optimization attempted but no optimized weights available")
                        self.logger.warning("⚠️ Weight optimization attempted but no optimized weights available")
                
                # Log quality predictors
                quality_predictors = new_rules.get('quality_predictors', {})
                if quality_predictors:
                    print(f"📊 Quality predictors identified: {len(quality_predictors)} features")
                    top_predictors = sorted(quality_predictors.items(), key=lambda x: abs(x[1].get('correlation', 0)), reverse=True)[:5]
                    print("   Top 5 quality predictors:")
                    self.logger.info(f"📊 Quality predictors identified: {len(quality_predictors)} features")
                    self.logger.info("Top 5 quality predictors:")
                    for i, (feature, info) in enumerate(top_predictors, 1):
                        corr = info.get('correlation', 0)
                        direction = "📈" if corr > 0 else "📉"
                        print(f"   {i}. {direction} {feature}: correlation={corr:.3f}")
                        self.logger.info(f"  {feature}: correlation={corr:.3f}")
                
                # Log model performance if available
                strength_model = new_rules.get('strength_scoring_model', {})
                if strength_model:
                    model_type = strength_model.get('model_type', 'Unknown')
                    r_squared = strength_model.get('r_squared', 0.0)
                    cv_r_squared = strength_model.get('cv_r_squared_mean', 0.0)
                    print(f"🤖 ML Model Performance:")
                    print(f"   - Model type: {model_type}")
                    print(f"   - R² score: {r_squared:.3f}")
                    print(f"   - CV R² score: {cv_r_squared:.3f}")
                    self.logger.info(f"🤖 ML Model Performance: {model_type}, R²={r_squared:.3f}, CV R²={cv_r_squared:.3f}")
                
                print("✅ Quality rules update completed successfully!")
                self.logger.info(f"✅ Quality rules update completed successfully")
            else:
                print("⚠️ No new quality rules learned from backtesting results")
                self.logger.warning("⚠️ No new quality rules learned from backtesting results")
            
        except Exception as e:
            print(f"❌ Failed to update quality rules: {e}")
            self.logger.error(f"❌ Failed to update quality rules: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _filter_by_quality(self, levels: List[Dict], backtest_results: List[Any]) -> List[Dict]:
        """Filter levels based on quality assessment - only filter out VERY low quality levels."""
        try:
            print(f"🔍 Filtering {len(levels)} levels based on quality assessment...")
            self.logger.info(f"🔍 Filtering {len(levels)} levels based on quality assessment")
            
            # Create quality mapping
            print("📊 Creating quality mapping from backtest results...")
            quality_map = {r.level.price: r.quality_score for r in backtest_results}
            print(f"✅ Quality mapping created for {len(quality_map)} levels")
            self.logger.debug(f"Created quality mapping for {len(quality_map)} levels")
            
            # Calculate quality statistics to determine very low threshold
            quality_scores = list(quality_map.values())
            if quality_scores:
                quality_mean = np.mean(quality_scores)
                quality_std = np.std(quality_scores)
                quality_min = np.min(quality_scores)
                quality_max = np.max(quality_scores)
                quality_median = np.median(quality_scores)
                
                print(f"📈 Quality statistics:")
                print(f"   - Mean: {quality_mean:.3f}")
                print(f"   - Median: {quality_median:.3f}")
                print(f"   - Std: {quality_std:.3f}")
                print(f"   - Min: {quality_min:.3f}")
                print(f"   - Max: {quality_max:.3f}")
                
                self.logger.info(f"Quality statistics: mean={quality_mean:.3f}, std={quality_std:.3f}, min={quality_min:.3f}, max={quality_max:.3f}")
                
                # Only filter out levels that are significantly below average (more than 2 standard deviations)
                very_low_threshold = max(0.1, quality_mean - 2 * quality_std)
                print(f"🎯 Calculated very low quality threshold: {very_low_threshold:.3f} (mean - 2*std)")
                self.logger.info(f"Calculated very low quality threshold: {very_low_threshold:.3f} (mean - 2*std)")
            else:
                very_low_threshold = 0.1  # Very conservative threshold
                print("⚠️ No quality scores available, using conservative threshold: 0.1")
                self.logger.warning("No quality scores available, using conservative threshold: 0.1")
            
            # Filter levels - only remove very low quality
            print("🔍 Applying quality filter...")
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
                        print(f"   Processed {i+1}/{len(levels)} levels...")
                else:
                    filtered_count += 1
                    if filtered_count <= 5:  # Log first 5 filtered levels
                        print(f"   🗑️ Filtered out level ${level['price']:.2f} (quality: {quality_score:.3f} < {very_low_threshold:.3f})")
                    self.logger.debug(f"Filtered out very low quality level ${level['price']:.2f} (quality: {quality_score:.3f}, threshold: {very_low_threshold:.3f})")
            
            print(f"✅ Quality filtering completed!")
            print(f"📊 Filtering Results:")
            print(f"   - Input levels: {len(levels)}")
            print(f"   - Kept levels: {kept_count}")
            print(f"   - Filtered out: {filtered_count}")
            print(f"   - Filter rate: {filtered_count/len(levels)*100:.1f}%")
            print(f"   - Threshold used: {very_low_threshold:.3f}")
            
            self.logger.info(f"✅ Quality filtering completed: {len(levels)} -> {len(filtered_levels)} levels")
            self.logger.info(f"Filtered out {filtered_count} very low quality levels (threshold: {very_low_threshold:.3f})")
            
            if filtered_levels:
                filtered_qualities = [level['backtest_quality'] for level in filtered_levels]
                print(f"📈 Filtered level quality stats:")
                print(f"   - Mean: {np.mean(filtered_qualities):.3f}")
                print(f"   - Min: {np.min(filtered_qualities):.3f}")
                print(f"   - Max: {np.max(filtered_qualities):.3f}")
                self.logger.info(f"Filtered level quality stats: mean={np.mean(filtered_qualities):.3f}, min={np.min(filtered_qualities):.3f}, max={np.max(filtered_qualities):.3f}")
            
            return filtered_levels
            
        except Exception as e:
            print(f"❌ Quality filtering failed: {e}")
            self.logger.error(f"❌ Quality filtering failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return levels
    
    def _enhance_levels_with_quality(self, levels: List[Dict], backtest_results: List[Any]) -> List[Dict]:
        """Enhance level data with quality information."""
        try:
            print(f"✨ Enhancing {len(levels)} levels with quality information...")
            self.logger.info(f"✨ Enhancing {len(levels)} levels with quality information")
            
            # Create quality mapping
            print("📊 Creating quality mapping from backtest results...")
            quality_map = {r.level.price: r for r in backtest_results}
            print(f"✅ Quality mapping created for {len(quality_map)} backtest results")
            self.logger.debug(f"Created quality mapping for {len(quality_map)} backtest results")
            
            enhanced_levels = []
            enhancement_errors = 0
            enhancement_successes = 0
            
            print("🔄 Processing levels for enhancement...")
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
                            print(f"   Enhanced level {i+1}/{len(levels)}: ${level['price']:.2f}, quality={backtest_result.quality_score:.3f}")
                            self.logger.debug(f"Enhanced level {i+1}/{len(levels)}: price={level['price']:.2f}, quality={backtest_result.quality_score:.3f}")
                    else:
                        if enhancement_errors < 5:  # Log first 5 missing results
                            print(f"   ⚠️ No backtest result found for level at ${level['price']:.2f}")
                        self.logger.warning(f"No backtest result found for level at price {level['price']:.2f}")
                        enhancement_errors += 1
                    
                    enhanced_levels.append(enhanced_level)
                    
                except Exception as e:
                    if enhancement_errors < 5:  # Log first 5 errors
                        print(f"   ❌ Failed to enhance level {i+1}: {e}")
                    self.logger.warning(f"Failed to enhance level {i+1}: {e}")
                    enhancement_errors += 1
                    enhanced_levels.append(level)  # Add original level as fallback
            
            print(f"✅ Level enhancement completed!")
            print(f"📊 Enhancement Results:")
            print(f"   - Levels processed: {len(levels)}")
            print(f"   - Successfully enhanced: {enhancement_successes}")
            print(f"   - Enhancement errors: {enhancement_errors}")
            print(f"   - Success rate: {enhancement_successes/len(levels)*100:.1f}%")
            
            self.logger.info(f"✅ Level enhancement completed: {len(enhanced_levels)} levels enhanced ({enhancement_errors} errors)")
            
            # Log enhancement statistics
            if enhanced_levels:
                enhanced_qualities = [level.get('backtest_quality', 0.0) for level in enhanced_levels]
                enhanced_success_rates = [level.get('success_rate', 0.0) for level in enhanced_levels]
                enhanced_bounce_strengths = [level.get('avg_bounce_strength', 0.0) for level in enhanced_levels]
                
                print(f"📈 Enhanced level statistics:")
                print(f"   - Quality scores - mean: {np.mean(enhanced_qualities):.3f}, std: {np.std(enhanced_qualities):.3f}")
                print(f"   - Success rates - mean: {np.mean(enhanced_success_rates):.3f}, std: {np.std(enhanced_success_rates):.3f}")
                print(f"   - Bounce strengths - mean: {np.mean(enhanced_bounce_strengths):.3f}, std: {np.std(enhanced_bounce_strengths):.3f}")
                
                self.logger.info(f"Enhanced level quality stats: mean={np.mean(enhanced_qualities):.3f}, std={np.std(enhanced_qualities):.3f}")
                self.logger.info(f"Enhanced level success rates: mean={np.mean(enhanced_success_rates):.3f}, std={np.std(enhanced_success_rates):.3f}")
                self.logger.info(f"Enhanced level bounce strengths: mean={np.mean(enhanced_bounce_strengths):.3f}, std={np.std(enhanced_bounce_strengths):.3f}")
            
            return enhanced_levels
            
        except Exception as e:
            print(f"❌ Level enhancement failed: {e}")
            self.logger.error(f"❌ Level enhancement failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return levels
    
    def _cluster_quality_enhanced(self, levels: List[Dict], price_range: Tuple[float, float], data: pd.DataFrame = None) -> ClusteringResult:
        """Cluster levels using quality-enhanced approach."""
        try:
            print(f"🎯 Starting quality-enhanced clustering for {len(levels)} levels...")
            self.logger.info(f"🎯 Starting quality-enhanced clustering for {len(levels)} levels")
            
            # Adjust clustering parameters based on quality
            print("🔧 Adjusting clustering parameters based on level quality...")
            self.logger.info("Adjusting clustering parameters based on level quality")
            adjusted_proximity = self._adjust_proximity_by_quality(levels)
            adjusted_strength_threshold = self._adjust_strength_threshold_by_quality(levels)
            
            print(f"📊 Quality-adjusted parameters:")
            print(f"   - Proximity: {adjusted_proximity:.3f} (original: {self.config.proximity_threshold:.3f})")
            print(f"   - Strength threshold: {adjusted_strength_threshold:.3f} (original: {self.config.strength_similarity_threshold:.3f})")
            
            self.logger.info(f"Quality-adjusted parameters: proximity={adjusted_proximity:.3f} (original: {self.config.proximity_threshold:.3f}), strength={adjusted_strength_threshold:.3f} (original: {self.config.strength_similarity_threshold:.3f})")
            
            # Use extensive clustering system
            print("🚀 Running extensive clustering with quality-enhanced parameters...")
            self.logger.info("Running extensive clustering with quality-enhanced parameters")
            result = self._cluster_levels_extensive(
                levels=levels,
                price_range=price_range,
                proximity_threshold=adjusted_proximity,
                strength_similarity_threshold=adjusted_strength_threshold,
                data=data
            )
            
            # Add quality information to result
            print("✨ Adding quality information to clustering result...")
            result.quality_enhanced = True
            result.quality_metrics = self._calculate_cluster_quality_metrics(result, levels)
            
            print(f"✅ Quality-enhanced clustering completed!")
            print(f"📊 Quality-Enhanced Clustering Results:")
            print(f"   - Clusters created: {len(result.clusters)}")
            print(f"   - Quality score: {result.quality_score:.3f}")
            print(f"   - Quality enhanced: {result.quality_enhanced}")
            print(f"   - Quality metrics: {len(result.quality_metrics)} metrics calculated")
            
            self.logger.info(f"✅ Quality-enhanced clustering completed: {len(result.clusters)} clusters")
            self.logger.info(f"Quality metrics: {result.quality_metrics}")
            
            return result
            
        except Exception as e:
            print(f"❌ Quality-enhanced clustering failed: {e}")
            print("🔄 Falling back to standard clustering...")
            self.logger.error(f"❌ Quality-enhanced clustering failed: {e}")
            import traceback
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
            import traceback
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
            
            # This would backtest the cluster center as a new level
            # For now, return average quality of cluster members
            # TODO: Implement actual cluster center backtesting
            
            # Placeholder implementation - return average quality
            cluster_quality = 0.7  # Placeholder
            
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
            import traceback
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
            import traceback
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
                quality_enhanced=False
            )
    
    def _cluster_levels_extensive(self, levels: List[Dict], price_range: Tuple[float, float], 
                                 proximity_threshold: float, strength_similarity_threshold: float, 
                                 data: pd.DataFrame = None) -> ClusteringResult:
        """Extensive clustering implementation using enhanced SR detection system."""
        try:
            self.logger.info(f"Running extensive clustering with {len(levels)} levels")
            
            if not levels:
                self.logger.warning("No levels provided for extensive clustering")
                return ClusteringResult(clusters=[], cluster_centers=[], quality_score=0.0)
            
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
            import traceback
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
                return ClusteringResult(clusters=[], cluster_centers=[], quality_score=0.0)
            
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
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ Simple fallback clustering failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return single-level clusters as final fallback
            self.logger.warning("🔄 Using single-level clusters as final fallback")
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                cluster_centers=[level['price'] for level in levels],
                quality_score=0.5
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