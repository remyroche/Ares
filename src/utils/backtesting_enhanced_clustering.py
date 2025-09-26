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

from .logger import system_logger
from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel, create_sr_level_from_dict
from .clustering_alternatives import get_clustering_manager, ClusteringResult

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
        
        # Initialize components
        self.backtesting_engine = SRBacktestingEngine(self.config.backtest_config)
        self.clustering_manager = get_clustering_manager()
        
        # Learning state
        self.learned_rules = {}
        self.quality_predictions = {}
        self.levels_processed = 0
        
    def cluster_with_backtesting(self, levels: List[Dict], data: pd.DataFrame, 
                                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster levels using backtesting-enhanced approach."""
        try:
            self.logger.info(f"Starting backtesting-enhanced clustering for {len(levels)} levels")
            
            # Step 1: Backtest levels to assess quality
            backtest_results = self._backtest_levels(levels, data)
            
            # Step 2: Learn/update quality rules
            if len(backtest_results) >= self.config.min_levels_for_learning:
                self._update_quality_rules(backtest_results)
            
            # Step 3: Filter levels based on quality
            quality_filtered_levels = self._filter_by_quality(levels, backtest_results)
            
            # Step 4: Enhance level data with quality scores
            enhanced_levels = self._enhance_levels_with_quality(quality_filtered_levels, backtest_results)
            
            # Step 5: Cluster using quality-enhanced approach
            clustering_result = self._cluster_quality_enhanced(enhanced_levels, price_range)
            
            # Step 6: Post-process clusters with quality validation
            final_result = self._validate_clusters_with_backtesting(clustering_result, data)
            
            self.levels_processed += len(levels)
            self.logger.info(f"Backtesting-enhanced clustering completed: {len(final_result.clusters)} clusters")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Backtesting-enhanced clustering failed: {e}")
            # Fallback to standard clustering
            return self._fallback_clustering(levels, price_range)
    
    def _backtest_levels(self, levels: List[Dict], data: pd.DataFrame) -> List[Any]:
        """Backtest all levels to assess their quality."""
        self.logger.info(f"Backtesting {len(levels)} levels for quality assessment")
        
        # Convert to SRLevel objects
        sr_levels = []
        for level_dict in levels:
            try:
                sr_level = create_sr_level_from_dict(level_dict)
                sr_levels.append(sr_level)
            except Exception as e:
                self.logger.warning(f"Failed to create SRLevel from {level_dict}: {e}")
                continue
        
        # Backtest levels
        backtest_results = self.backtesting_engine.backtest_multiple_levels(sr_levels, data)
        
        self.logger.info(f"Backtesting completed. Average quality: {np.mean([r.quality_score for r in backtest_results]):.3f}")
        return backtest_results
    
    def _update_quality_rules(self, backtest_results: List[Any]) -> None:
        """Update quality rules based on backtesting results."""
        try:
            self.logger.info("Updating quality rules from backtesting results")
            
            # Learn new rules
            new_rules = self.backtesting_engine.learn_quality_rules(backtest_results)
            
            if new_rules:
                # Merge with existing rules (weighted average)
                if self.learned_rules:
                    self.learned_rules = self._merge_rules(self.learned_rules, new_rules)
                else:
                    self.learned_rules = new_rules
                
                self.logger.info(f"Quality rules updated. Key features: {list(new_rules.get('discriminative_features', {}).keys())}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update quality rules: {e}")
    
    def _filter_by_quality(self, levels: List[Dict], backtest_results: List[Any]) -> List[Dict]:
        """Filter levels based on quality assessment - only filter out VERY low quality levels."""
        try:
            # Create quality mapping
            quality_map = {r.level.price: r.quality_score for r in backtest_results}
            
            # Calculate quality statistics to determine very low threshold
            quality_scores = list(quality_map.values())
            if quality_scores:
                quality_mean = np.mean(quality_scores)
                quality_std = np.std(quality_scores)
                # Only filter out levels that are significantly below average (more than 2 standard deviations)
                very_low_threshold = max(0.1, quality_mean - 2 * quality_std)
            else:
                very_low_threshold = 0.1  # Very conservative threshold
            
            # Filter levels - only remove very low quality
            filtered_levels = []
            for level in levels:
                quality_score = quality_map.get(level['price'], 0.0)
                
                # Only filter out VERY low quality levels
                if quality_score >= very_low_threshold:
                    # Add quality score to level data
                    level['backtest_quality'] = quality_score
                    level['quality_metrics'] = self._extract_quality_metrics(quality_score, backtest_results)
                    filtered_levels.append(level)
                else:
                    self.logger.debug(f"Filtered out very low quality level ${level['price']:.2f} (quality: {quality_score:.3f}, threshold: {very_low_threshold:.3f})")
            
            self.logger.info(f"Quality filtering (very low only): {len(levels)} -> {len(filtered_levels)} levels (threshold: {very_low_threshold:.3f})")
            return filtered_levels
            
        except Exception as e:
            self.logger.warning(f"Quality filtering failed: {e}")
            return levels
    
    def _enhance_levels_with_quality(self, levels: List[Dict], backtest_results: List[Any]) -> List[Dict]:
        """Enhance level data with quality information."""
        try:
            # Create quality mapping
            quality_map = {r.level.price: r for r in backtest_results}
            
            enhanced_levels = []
            for level in levels:
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
                
                enhanced_levels.append(enhanced_level)
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.warning(f"Level enhancement failed: {e}")
            return levels
    
    def _cluster_quality_enhanced(self, levels: List[Dict], price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster levels using quality-enhanced approach."""
        try:
            # Adjust clustering parameters based on quality
            adjusted_proximity = self._adjust_proximity_by_quality(levels)
            adjusted_strength_threshold = self._adjust_strength_threshold_by_quality(levels)
            
            self.logger.info(f"Quality-enhanced clustering: proximity={adjusted_proximity:.3f}, strength={adjusted_strength_threshold:.3f}")
            
            # Use quality-enhanced clustering
            result = self.clustering_manager.cluster_with_fallback(
                levels=levels,
                price_range=price_range,
                proximity_threshold=adjusted_proximity,
                strength_similarity_threshold=adjusted_strength_threshold,
                preferred_algorithm='strength_proximity'
            )
            
            # Add quality information to result
            result.quality_enhanced = True
            result.quality_metrics = self._calculate_cluster_quality_metrics(result, levels)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Quality-enhanced clustering failed: {e}")
            return self._fallback_clustering(levels, price_range)
    
    def _validate_clusters_with_backtesting(self, clustering_result: ClusteringResult, data: pd.DataFrame) -> ClusteringResult:
        """Validate clusters using backtesting."""
        try:
            validated_clusters = []
            cluster_quality_scores = []
            
            for i, cluster in enumerate(clustering_result.clusters):
                if len(cluster) > 1:
                    # Validate cluster quality
                    cluster_quality = self._validate_cluster_quality(cluster, clustering_result.cluster_centers[i], data)
                    cluster_quality_scores.append(cluster_quality)
                    
                    if cluster_quality >= self.config.min_quality_score:
                        validated_clusters.append(cluster)
                    else:
                        self.logger.debug(f"Filtered out cluster {i} (quality: {cluster_quality:.3f})")
                else:
                    # Single levels are always kept
                    validated_clusters.append(cluster)
                    cluster_quality_scores.append(1.0)
            
            # Update result
            clustering_result.clusters = validated_clusters
            clustering_result.cluster_centers = [clustering_result.cluster_centers[i] for i in range(len(validated_clusters))]
            clustering_result.quality_score = np.mean(cluster_quality_scores) if cluster_quality_scores else 0.0
            
            self.logger.info(f"Cluster validation: {len(clustering_result.clusters)} clusters validated")
            
            return clustering_result
            
        except Exception as e:
            self.logger.warning(f"Cluster validation failed: {e}")
            return clustering_result
    
    def _adjust_proximity_by_quality(self, levels: List[Dict]) -> float:
        """Adjust proximity threshold based on level quality."""
        try:
            if not levels:
                return self.config.proximity_threshold
            
            # Calculate average quality
            qualities = [level.get('backtest_quality', 0.5) for level in levels]
            avg_quality = np.mean(qualities)
            
            # Higher quality levels can be clustered more tightly
            # Lower quality levels need more separation
            quality_factor = 0.5 + (avg_quality * 0.5)  # Range: 0.5 to 1.0
            
            adjusted_proximity = self.config.proximity_threshold * quality_factor
            
            return adjusted_proximity
            
        except Exception as e:
            self.logger.warning(f"Proximity adjustment failed: {e}")
            return self.config.proximity_threshold
    
    def _adjust_strength_threshold_by_quality(self, levels: List[Dict]) -> float:
        """Adjust strength similarity threshold based on level quality."""
        try:
            if not levels:
                return self.config.strength_similarity_threshold
            
            # Calculate quality variance
            qualities = [level.get('backtest_quality', 0.5) for level in levels]
            quality_variance = np.var(qualities)
            
            # Higher variance means more diverse quality levels
            # Need stricter strength matching
            variance_factor = 1.0 + (quality_variance * 2.0)  # Range: 1.0 to 2.0
            
            adjusted_threshold = self.config.strength_similarity_threshold / variance_factor
            
            return adjusted_threshold
            
        except Exception as e:
            self.logger.warning(f"Strength threshold adjustment failed: {e}")
            return self.config.strength_similarity_threshold
    
    def _validate_cluster_quality(self, cluster: List[int], cluster_center: float, data: pd.DataFrame) -> float:
        """Validate the quality of a cluster using backtesting."""
        try:
            if not cluster or len(cluster) == 0:
                return 0.0
            
            # Create a synthetic SR level at the cluster center
            synthetic_level = {
                'price': cluster_center,
                'strength': self._calculate_cluster_strength(cluster),
                'type': 'support' if cluster_center < data['close'].iloc[-1] else 'resistance',
                'touches': self._calculate_cluster_touches(cluster),
                'volume': self._calculate_cluster_volume(cluster, data)
            }
            
            # Convert to SRLevel object
            sr_level = create_sr_level_from_dict(synthetic_level)
            
            # Backtest the synthetic level
            backtest_result = self.backtesting_engine.backtest_single_level(sr_level, data)
            
            if backtest_result:
                # Calculate composite quality score
                quality_score = self._calculate_composite_quality_score(backtest_result, cluster)
                return quality_score
            else:
                # Fallback to average quality of cluster members
                return self._calculate_average_member_quality(cluster)
            
        except Exception as e:
            self.logger.warning(f"Cluster quality validation failed: {e}")
            return 0.5
    
    def _calculate_cluster_strength(self, cluster: List[int]) -> float:
        """Calculate the strength of a cluster based on its members."""
        try:
            if not cluster:
                return 0.0
            
            # Simple strength calculation - could be enhanced with more sophisticated logic
            base_strength = len(cluster) * 0.1  # Each member adds 0.1 strength
            
            # Cap at reasonable maximum
            return min(base_strength, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Cluster strength calculation failed: {e}")
            return 0.5
    
    def _calculate_cluster_touches(self, cluster: List[int]) -> int:
        """Calculate total touches for the cluster."""
        try:
            # For now, return the number of cluster members as a proxy for touches
            # In a more sophisticated implementation, this would count actual touches
            return len(cluster)
            
        except Exception as e:
            self.logger.warning(f"Cluster touches calculation failed: {e}")
            return 1
    
    def _calculate_cluster_volume(self, cluster: List[int], data: pd.DataFrame) -> float:
        """Calculate volume associated with the cluster."""
        try:
            if 'volume' not in data.columns:
                return 0.0
            
            # Calculate average volume around cluster center
            # This is a simplified approach - could be enhanced with actual volume analysis
            recent_volume = data['volume'].tail(20).mean()
            return recent_volume * len(cluster) * 0.1  # Scale by cluster size
            
        except Exception as e:
            self.logger.warning(f"Cluster volume calculation failed: {e}")
            return 0.0
    
    def _calculate_composite_quality_score(self, backtest_result: Any, cluster: List[int]) -> float:
        """Calculate composite quality score from backtest result and cluster characteristics."""
        try:
            # Base quality from backtesting
            base_quality = backtest_result.quality_score
            
            # Cluster size factor (more members = higher confidence)
            cluster_size_factor = min(1.0, len(cluster) / 5.0)  # Normalize to max 5 members
            
            # Consistency factor (how similar are cluster members)
            consistency_factor = self._calculate_cluster_consistency(cluster)
            
            # Composite score with weights
            composite_score = (
                0.6 * base_quality +           # 60% from backtesting
                0.2 * cluster_size_factor +   # 20% from cluster size
                0.2 * consistency_factor      # 20% from consistency
            )
            
            return min(1.0, max(0.0, composite_score))
            
        except Exception as e:
            self.logger.warning(f"Composite quality calculation failed: {e}")
            return backtest_result.quality_score if backtest_result else 0.5
    
    def _calculate_cluster_consistency(self, cluster: List[int]) -> float:
        """Calculate how consistent the cluster members are."""
        try:
            if len(cluster) <= 1:
                return 1.0  # Single member is perfectly consistent
            
            # This is a placeholder for consistency calculation
            # In a real implementation, this would analyze the similarity of cluster members
            # For now, return a reasonable default
            return 0.8
            
        except Exception as e:
            self.logger.warning(f"Cluster consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_average_member_quality(self, cluster: List[int]) -> float:
        """Calculate average quality of cluster members as fallback."""
        try:
            if not cluster:
                return 0.0
            
            # This would need access to the original level data
            # For now, return a reasonable default based on cluster size
            return min(0.8, 0.3 + (len(cluster) * 0.1))
            
        except Exception as e:
            self.logger.warning(f"Average member quality calculation failed: {e}")
            return 0.5
    
    def _calculate_cluster_quality_metrics(self, result: ClusteringResult, levels: List[Dict]) -> Dict[str, Any]:
        """Calculate quality metrics for the clustering result."""
        try:
            if not result.clusters:
                return {}
            
            # Calculate metrics for each cluster
            cluster_metrics = []
            for cluster in result.clusters:
                if len(cluster) > 1:
                    cluster_levels = [levels[i] for i in cluster]
                    cluster_quality = np.mean([level.get('backtest_quality', 0.5) for level in cluster_levels])
                    cluster_metrics.append({
                        'size': len(cluster),
                        'avg_quality': cluster_quality,
                        'quality_variance': np.var([level.get('backtest_quality', 0.5) for level in cluster_levels])
                    })
            
            return {
                'total_clusters': len(result.clusters),
                'avg_cluster_quality': np.mean([m['avg_quality'] for m in cluster_metrics]) if cluster_metrics else 0.0,
                'avg_cluster_size': np.mean([m['size'] for m in cluster_metrics]) if cluster_metrics else 0.0,
                'quality_consistency': 1.0 - np.mean([m['quality_variance'] for m in cluster_metrics]) if cluster_metrics else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"Cluster quality metrics calculation failed: {e}")
            return {}
    
    def _extract_quality_metrics(self, quality_score: float, backtest_results: List[Any]) -> Dict[str, float]:
        """Extract quality metrics for a level."""
        # Find the backtest result for this quality score
        for result in backtest_results:
            if abs(result.quality_score - quality_score) < 0.001:
                return {
                    'success_rate': result.success_rate,
                    'bounce_strength': result.avg_bounce_strength,
                    'total_touches': result.total_touches,
                    'volume_confirmation': result.total_volume_at_level,
                    'time_persistence': result.time_persistence
                }
        
        return {}
    
    def _merge_rules(self, existing_rules: Dict[str, Any], new_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new rules with existing rules using weighted average."""
        try:
            merged_rules = existing_rules.copy()
            
            # Merge discriminative features
            if 'discriminative_features' in new_rules:
                existing_features = merged_rules.get('discriminative_features', {})
                new_features = new_rules['discriminative_features']
                
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
                    else:
                        existing_features[feature] = info
                
                merged_rules['discriminative_features'] = existing_features
            
            # Update quality threshold
            if 'quality_threshold' in new_rules:
                merged_rules['quality_threshold'] = 0.7 * merged_rules.get('quality_threshold', 0.5) + 0.3 * new_rules['quality_threshold']
            
            return merged_rules
            
        except Exception as e:
            self.logger.warning(f"Rule merging failed: {e}")
            return existing_rules
    
    def _fallback_clustering(self, levels: List[Dict], price_range: Tuple[float, float]) -> ClusteringResult:
        """Fallback to standard clustering if backtesting-enhanced approach fails."""
        self.logger.warning("Falling back to standard clustering")
        
        return self.clustering_manager.cluster_with_fallback(
            levels=levels,
            price_range=price_range,
            proximity_threshold=self.config.proximity_threshold,
            strength_similarity_threshold=self.config.strength_similarity_threshold,
            preferred_algorithm='strength_proximity'
        )
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning process."""
        return {
            'levels_processed': self.levels_processed,
            'rules_learned': bool(self.learned_rules),
            'quality_rules_summary': self.backtesting_engine.get_quality_rules_summary(),
            'learned_features': list(self.learned_rules.get('discriminative_features', {}).keys()) if self.learned_rules else []
        }
    
    def predict_level_quality(self, level: Dict[str, Any], data: pd.DataFrame) -> float:
        """Predict the quality of a level using learned rules."""
        try:
            sr_level = create_sr_level_from_dict(level)
            return self.backtesting_engine.predict_level_quality(sr_level, data)
        except Exception as e:
            self.logger.warning(f"Quality prediction failed: {e}")
            return level.get('strength', 0.5)

def get_backtesting_enhanced_clustering(config: Optional[BacktestingEnhancedConfig] = None) -> BacktestingEnhancedClustering:
    """Get a backtesting-enhanced clustering instance."""
    return BacktestingEnhancedClustering(config)