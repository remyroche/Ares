#!/usr/bin/env python3
"""
Regime Consolidator

This module implements the complete coverage regime consolidation algorithm
that takes HMM discovery outputs and creates balanced clusters for ML training.

Key Features:
- Similarity-based regime merging (preserves market information)
- Complete distribution coverage (100% accounted for)
- Balanced cluster sizes (3-8% each)
- Top 20 clusters capture 90-95% of market states
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ConsolidationConfig:
    """Configuration for regime consolidation."""
    
    # Target clustering parameters
    target_clusters: int = 20
    min_cluster_size_pct: float = 0.03  # 3%
    max_cluster_size_pct: float = 0.08  # 8%
    coverage_target: float = 0.95  # 95% coverage by top clusters
    
    # Similarity thresholds
    merge_similarity_threshold: float = 0.90  # Merge regimes 90%+ similar
    assignment_similarity_threshold: float = 0.70  # Assign remaining regimes 70%+ similar
    
    # Quality thresholds
    min_meaningful_size: float = 0.01  # 1% minimum for meaningful clusters
    
    # Output settings
    save_detailed_results: bool = True
    output_dir: str = "training/steps/market_analysis/clustering/results"
    
    # Validation settings
    validate_coverage: bool = True
    validate_balance: bool = True

@dataclass
class ConsolidationResult:
    """Result container for regime consolidation."""
    
    # Core results
    final_clusters: List[Dict[str, Any]]
    coverage_analysis: Dict[str, Any]
    merge_history: List[Dict[str, Any]]
    
    # Metadata
    original_regime_count: int
    final_cluster_count: int
    target_clusters: int
    
    # Validation metrics
    coverage_percentage: float
    top_clusters_coverage: float
    balance_score: float
    
    # Processing info
    processing_time: float
    config: ConsolidationConfig
    
    # Data integrity
    total_samples_accounted: int
    original_total_samples: int

class RegimeConsolidator:
    """
    Regime Consolidator for creating balanced clusters from HMM discovery outputs.
    
    This class implements the complete coverage regime consolidation algorithm
    that ensures all market regimes are accounted for while creating balanced,
    coherent clusters suitable for ML model training.
    """
    
    def __init__(self, config: Optional[ConsolidationConfig] = None):
        """Initialize the regime consolidator."""
        self.config = config or ConsolidationConfig()
        self.logger = logger.getChild("RegimeConsolidator")
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"RegimeConsolidator initialized with config: {self.config}")
    
    def consolidate_regimes(self, regime_data: pd.DataFrame) -> ConsolidationResult:
        """
        Main consolidation method that takes HMM discovery output and creates balanced clusters.
        
        Args:
            regime_data: DataFrame with regime features and sample counts from HMM discovery
            
        Returns:
            ConsolidationResult with complete consolidation analysis
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting regime consolidation with {len(regime_data)} regimes")
            
            # Validate input data
            self._validate_input_data(regime_data)
            
            # Extract regime features and sample counts
            regime_features = regime_data[['momentum_mean', 'volatility_mean', 'volume_mean', 'trend_mean']].values
            regime_sample_counts = regime_data['sample_count'].values
            regime_names = regime_data.index.tolist()
            
            total_samples = regime_sample_counts.sum()
            self.logger.info(f"Total market samples: {total_samples:,}")
            
            # Step 1: Merge similar regimes
            merged_features, merged_counts, merged_names, merge_history = self._merge_similar_regimes(
                regime_features, regime_sample_counts, regime_names
            )
            
            self.logger.info(f"After similarity merging: {len(merged_features)} regimes "
                           f"(reduction: {(len(regime_features) - len(merged_features)) / len(regime_features):.1%})")
            
            # Step 2: Create primary clusters
            primary_clusters, remaining_indices = self._create_primary_clusters(
                merged_features, merged_counts, merged_names
            )
            
            self.logger.info(f"Created {len(primary_clusters)} primary clusters")
            self.logger.info(f"Remaining regimes: {len(remaining_indices)}")
            
            # Step 3: Assign all remaining regimes (100% coverage)
            final_clusters = self._assign_all_remaining_regimes(
                primary_clusters, remaining_indices, merged_features, merged_counts, merged_names
            )
            
            # Step 4: Analyze coverage and validate
            coverage_analysis = self._analyze_distribution_coverage(final_clusters, total_samples)
            
            # Step 5: Optimize for coverage target if needed
            if coverage_analysis['top_clusters_coverage'] < self.config.coverage_target:
                final_clusters = self._optimize_for_coverage_target(
                    final_clusters, total_samples
                )
                coverage_analysis = self._analyze_distribution_coverage(final_clusters, total_samples)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ConsolidationResult(
                final_clusters=final_clusters,
                coverage_analysis=coverage_analysis,
                merge_history=merge_history,
                original_regime_count=len(regime_features),
                final_cluster_count=len(final_clusters),
                target_clusters=self.config.target_clusters,
                coverage_percentage=coverage_analysis['coverage_percentage'],
                top_clusters_coverage=coverage_analysis['top_clusters_coverage'],
                balance_score=self._calculate_balance_score(coverage_analysis),
                processing_time=processing_time,
                config=self.config,
                total_samples_accounted=coverage_analysis['total_covered'],
                original_total_samples=total_samples
            )
            
            # Validate results
            if self.config.validate_coverage:
                self._validate_coverage(result)
            
            if self.config.validate_balance:
                self._validate_balance(result)
            
            # Save results if requested
            if self.config.save_detailed_results:
                self._save_results(result)
            
            self.logger.info(f"Consolidation completed in {processing_time:.2f}s")
            self.logger.info(f"Final: {len(final_clusters)} clusters, "
                           f"{coverage_analysis['top_clusters_coverage']:.2%} coverage by top {self.config.target_clusters}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Regime consolidation failed: {e}")
            raise
    
    def _validate_input_data(self, regime_data: pd.DataFrame) -> None:
        """Validate input regime data."""
        required_columns = ['momentum_mean', 'volatility_mean', 'volume_mean', 'trend_mean', 'sample_count']
        
        missing_columns = [col for col in required_columns if col not in regime_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(regime_data) == 0:
            raise ValueError("Regime data is empty")
        
        if regime_data['sample_count'].sum() == 0:
            raise ValueError("Total sample count is zero")
        
        # Check for negative sample counts
        if (regime_data['sample_count'] < 0).any():
            raise ValueError("Found negative sample counts")
        
        self.logger.info(f"Input validation passed: {len(regime_data)} regimes, "
                        f"{regime_data['sample_count'].sum():,} total samples")
    
    def _merge_similar_regimes(self, features: np.ndarray, sample_counts: np.ndarray, 
                             regime_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """Merge similar regimes while tracking all merges."""
        
        # Normalize features for similarity calculation
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(normalized_features)
        
        # Find similar pairs
        similar_pairs = self._find_similar_pairs(similarity_matrix, self.config.merge_similarity_threshold)
        
        self.logger.info(f"Found {len(similar_pairs)} pairs of similar regimes "
                        f"(similarity > {self.config.merge_similarity_threshold})")
        
        # Track merge history
        merge_history = []
        merge_map = {i: i for i in range(len(features))}
        
        # Perform merges
        for regime_i, regime_j, similarity in similar_pairs:
            if merge_map[regime_i] != regime_i or merge_map[regime_j] != regime_j:
                continue
            
            # Weighted merge by sample count
            count_i = sample_counts[regime_i]
            count_j = sample_counts[regime_j]
            total_count = count_i + count_j
            
            merged_feature = (features[regime_i] * count_i + features[regime_j] * count_j) / total_count
            
            features[regime_i] = merged_feature
            sample_counts[regime_i] = total_count
            merge_map[regime_j] = regime_i
            
            merge_history.append({
                'merged_from': regime_names[regime_j],
                'merged_into': regime_names[regime_i],
                'similarity': similarity,
                'samples_before': [int(count_i), int(count_j)],
                'samples_after': int(total_count),
                'timestamp': datetime.now().isoformat()
            })
        
        # Collect unmerged regimes
        unique_regimes = set(merge_map.values())
        merged_features = []
        merged_counts = []
        merged_names = []
        
        for regime_idx in unique_regimes:
            merged_features.append(features[regime_idx])
            merged_counts.append(sample_counts[regime_idx])
            merged_names.append(regime_names[regime_idx])
        
        return np.array(merged_features), np.array(merged_counts), merged_names, merge_history
    
    def _find_similar_pairs(self, similarity_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
        """Find pairs of regimes with similarity above threshold."""
        
        similar_pairs = []
        n_regimes = similarity_matrix.shape[0]
        
        for i in range(n_regimes):
            for j in range(i + 1, n_regimes):
                if similarity_matrix[i, j] >= threshold:
                    similar_pairs.append((i, j, similarity_matrix[i, j]))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def _create_primary_clusters(self, features: np.ndarray, sample_counts: np.ndarray, 
                               regime_names: List[str]) -> Tuple[List[Dict], List[int]]:
        """Create primary clusters from the largest merged regimes."""
        
        total_samples = sample_counts.sum()
        min_cluster_size = int(total_samples * self.config.min_cluster_size_pct)
        max_cluster_size = int(total_samples * self.config.max_cluster_size_pct)
        
        # Sort regimes by sample count (descending)
        sorted_indices = np.argsort(sample_counts)[::-1]
        
        primary_clusters = []
        remaining_indices = []
        
        # Start with largest regimes as cluster seeds
        cluster_sample_count = 0
        current_cluster_regimes = []
        
        for idx in sorted_indices:
            regime_samples = sample_counts[idx]
            
            # Check if adding this regime would exceed cluster size limit
            if (cluster_sample_count + regime_samples > max_cluster_size and 
                current_cluster_regimes):
                # Finalize current cluster and start new one
                if len(current_cluster_regimes) > 0:
                    primary_clusters.append({
                        'regime_indices': current_cluster_regimes.copy(),
                        'sample_count': cluster_sample_count,
                        'regime_count': len(current_cluster_regimes)
                    })
                
                current_cluster_regimes = []
                cluster_sample_count = 0
            
            # Add regime to current cluster
            current_cluster_regimes.append(idx)
            cluster_sample_count += regime_samples
            
            # If we have enough clusters or run out of large regimes
            if (len(primary_clusters) >= self.config.target_clusters - 1 or 
                cluster_sample_count >= min_cluster_size):
                if len(current_cluster_regimes) > 0:
                    primary_clusters.append({
                        'regime_indices': current_cluster_regimes.copy(),
                        'sample_count': cluster_sample_count,
                        'regime_count': len(current_cluster_regimes)
                    })
                
                # Remaining regimes go to remaining list
                remaining_indices = sorted_indices[len(current_cluster_regimes):].tolist()
                break
        
        return primary_clusters, remaining_indices
    
    def _assign_all_remaining_regimes(self, primary_clusters: List[Dict], remaining_indices: List[int],
                                    features: np.ndarray, sample_counts: np.ndarray, 
                                    regime_names: List[str]) -> List[Dict]:
        """Assign ALL remaining regimes to clusters (ensures 100% coverage)."""
        
        final_clusters = []
        
        # Process each primary cluster
        for cluster in primary_clusters:
            cluster_regimes = cluster['regime_indices']
            
            # Add regimes from remaining list that are similar to this cluster
            cluster_centroid = features[cluster_regimes].mean(axis=0)
            
            # Find similar remaining regimes
            similar_remaining = self._find_similar_remaining_regimes(
                cluster_centroid, remaining_indices, features, self.config.assignment_similarity_threshold
            )
            
            # Add similar regimes to this cluster
            cluster_regimes.extend(similar_remaining)
            
            # Remove assigned regimes from remaining list
            for regime_idx in similar_remaining:
                remaining_indices.remove(regime_idx)
            
            # Update cluster statistics
            cluster_sample_count = sum(sample_counts[i] for i in cluster_regimes)
            
            final_clusters.append({
                'cluster_id': len(final_clusters),
                'regime_indices': cluster_regimes,
                'regime_names': [regime_names[i] for i in cluster_regimes],
                'sample_count': cluster_sample_count,
                'regime_count': len(cluster_regimes),
                'centroid': features[cluster_regimes].mean(axis=0).tolist(),
                'feature_ranges': self._calculate_feature_ranges(features[cluster_regimes])
            })
        
        # Handle any remaining unassigned regimes
        if remaining_indices:
            self.logger.info(f"Creating additional clusters for {len(remaining_indices)} remaining regimes")
            
            for regime_idx in remaining_indices:
                final_clusters.append({
                    'cluster_id': len(final_clusters),
                    'regime_indices': [regime_idx],
                    'regime_names': [regime_names[regime_idx]],
                    'sample_count': sample_counts[regime_idx],
                    'regime_count': 1,
                    'centroid': features[regime_idx].tolist(),
                    'feature_ranges': self._calculate_feature_ranges(features[[regime_idx]])
                })
        
        return final_clusters
    
    def _find_similar_remaining_regimes(self, cluster_centroid: np.ndarray, 
                                      remaining_indices: List[int], features: np.ndarray, 
                                      similarity_threshold: float) -> List[int]:
        """Find remaining regimes similar to a cluster centroid."""
        
        similar_regimes = []
        
        for regime_idx in remaining_indices:
            regime_feature = features[regime_idx]
            similarity = cosine_similarity([cluster_centroid], [regime_feature])[0][0]
            
            if similarity >= similarity_threshold:
                similar_regimes.append(regime_idx)
        
        return similar_regimes
    
    def _analyze_distribution_coverage(self, final_clusters: List[Dict], total_samples: int) -> Dict[str, Any]:
        """Analyze how well the clusters cover the distribution."""
        
        # Sort clusters by sample count (descending)
        sorted_clusters = sorted(final_clusters, key=lambda x: x['sample_count'], reverse=True)
        
        # Calculate coverage metrics
        cluster_coverages = []
        cumulative_coverage = 0
        
        for i, cluster in enumerate(sorted_clusters):
            cluster_coverage = cluster['sample_count'] / total_samples
            cluster_coverages.append({
                'cluster_id': cluster['cluster_id'],
                'coverage': cluster_coverage,
                'cumulative_coverage': cumulative_coverage + cluster_coverage,
                'regime_count': cluster['regime_count'],
                'sample_count': cluster['sample_count']
            })
            cumulative_coverage += cluster_coverage
        
        # Calculate top N coverage
        top_20_coverage = sum(cluster_coverages[i]['coverage'] for i in range(min(20, len(cluster_coverages))))
        
        return {
            'total_clusters': len(final_clusters),
            'total_covered': sum(cluster['sample_count'] for cluster in final_clusters),
            'coverage_percentage': sum(cluster['sample_count'] for cluster in final_clusters) / total_samples,
            'top_20_coverage': top_20_coverage,
            'top_clusters_coverage': top_20_coverage,
            'cluster_coverages': cluster_coverages
        }
    
    def _optimize_for_coverage_target(self, final_clusters: List[Dict], total_samples: int) -> List[Dict]:
        """Optimize clusters to meet coverage target."""
        
        # Sort by sample count
        sorted_clusters = sorted(final_clusters, key=lambda x: x['sample_count'], reverse=True)
        
        # If top clusters don't meet coverage target, merge smaller clusters into them
        top_clusters = sorted_clusters[:self.config.target_clusters]
        remaining_clusters = sorted_clusters[self.config.target_clusters:]
        
        top_coverage = sum(cluster['sample_count'] for cluster in top_clusters) / total_samples
        
        if top_coverage < self.config.coverage_target:
            self.logger.info(f"Top {self.config.target_clusters} clusters only cover {top_coverage:.2%}, "
                           f"merging remaining clusters to reach {self.config.coverage_target:.2%}")
            
            # Merge remaining clusters into top clusters
            for remaining_cluster in remaining_clusters:
                # Find best top cluster to merge with (most similar)
                best_merge_idx = self._find_best_merge_target(remaining_cluster, top_clusters)
                
                if best_merge_idx is not None:
                    # Merge remaining cluster into top cluster
                    self._merge_clusters(top_clusters[best_merge_idx], remaining_cluster)
            
            final_clusters = top_clusters
        
        return final_clusters
    
    def _find_best_merge_target(self, remaining_cluster: Dict, top_clusters: List[Dict]) -> Optional[int]:
        """Find the best top cluster to merge a remaining cluster into."""
        
        remaining_centroid = np.array(remaining_cluster['centroid'])
        best_similarity = -1
        best_idx = None
        
        for i, top_cluster in enumerate(top_clusters):
            top_centroid = np.array(top_cluster['centroid'])
            similarity = cosine_similarity([remaining_centroid], [top_centroid])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
        
        return best_idx
    
    def _merge_clusters(self, target_cluster: Dict, source_cluster: Dict) -> None:
        """Merge source cluster into target cluster."""
        
        # Combine regime indices and names
        target_cluster['regime_indices'].extend(source_cluster['regime_indices'])
        target_cluster['regime_names'].extend(source_cluster['regime_names'])
        
        # Update counts
        target_cluster['sample_count'] += source_cluster['sample_count']
        target_cluster['regime_count'] += source_cluster['regime_count']
        
        # Update centroid (weighted average)
        total_samples = target_cluster['sample_count']
        target_weight = (target_cluster['sample_count'] - source_cluster['sample_count']) / total_samples
        source_weight = source_cluster['sample_count'] / total_samples
        
        target_cluster['centroid'] = (
            np.array(target_cluster['centroid']) * target_weight + 
            np.array(source_cluster['centroid']) * source_weight
        ).tolist()
    
    def _calculate_feature_ranges(self, features: np.ndarray) -> Dict[str, List[float]]:
        """Calculate feature ranges for a set of regimes."""
        
        return {
            'momentum': [float(features[:, 0].min()), float(features[:, 0].max())],
            'volatility': [float(features[:, 1].min()), float(features[:, 1].max())],
            'volume': [float(features[:, 2].min()), float(features[:, 2].max())],
            'trend': [float(features[:, 3].min()), float(features[:, 3].max())]
        }
    
    def _calculate_balance_score(self, coverage_analysis: Dict) -> float:
        """Calculate how well cluster sizes are balanced."""
        
        cluster_coverages = coverage_analysis['cluster_coverages']
        if not cluster_coverages:
            return 0.0
        
        # Calculate how many clusters are in the target size range
        target_min = self.config.min_cluster_size_pct
        target_max = self.config.max_cluster_size_pct
        
        balanced_clusters = sum(1 for cluster in cluster_coverages 
                              if target_min <= cluster['coverage'] <= target_max)
        
        return balanced_clusters / len(cluster_coverages)
    
    def _validate_coverage(self, result: ConsolidationResult) -> None:
        """Validate that coverage requirements are met."""
        
        if result.coverage_percentage < 0.999:  # Allow for small rounding errors
            raise ValueError(f"Coverage validation failed: {result.coverage_percentage:.4f} < 0.999")
        
        if result.top_clusters_coverage < self.config.coverage_target:
            raise ValueError(f"Top clusters coverage validation failed: "
                           f"{result.top_clusters_coverage:.4f} < {self.config.coverage_target}")
        
        self.logger.info("Coverage validation passed")
    
    def _validate_balance(self, result: ConsolidationResult) -> None:
        """Validate that cluster balance is reasonable."""
        
        if result.balance_score < 0.5:  # At least 50% of clusters should be balanced
            self.logger.warning(f"Low balance score: {result.balance_score:.2f}")
        
        self.logger.info(f"Balance validation: score = {result.balance_score:.2f}")
    
    def _save_results(self, result: ConsolidationResult) -> None:
        """Save detailed results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = self.output_dir / f"regime_consolidation_results_{timestamp}.json"
        
        # Convert result to serializable format
        results_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'target_clusters': result.config.target_clusters,
                    'min_cluster_size_pct': result.config.min_cluster_size_pct,
                    'max_cluster_size_pct': result.config.max_cluster_size_pct,
                    'coverage_target': result.config.coverage_target
                },
                'original_regime_count': result.original_regime_count,
                'final_cluster_count': result.final_cluster_count,
                'processing_time': result.processing_time
            },
            'coverage_analysis': result.coverage_analysis,
            'final_clusters': result.final_clusters,
            'merge_history': result.merge_history
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save summary CSV
        summary_file = self.output_dir / f"cluster_summary_{timestamp}.csv"
        
        summary_data = []
        for cluster_info in result.coverage_analysis['cluster_coverages']:
            cluster_id = cluster_info['cluster_id']
            cluster_data = next(c for c in result.final_clusters if c['cluster_id'] == cluster_id)
            
            summary_data.append({
                'cluster_id': cluster_id,
                'sample_count': cluster_info['sample_count'],
                'sample_percentage': cluster_info['coverage'],
                'regime_count': cluster_info['regime_count'],
                'momentum_mean': cluster_data['centroid'][0],
                'volatility_mean': cluster_data['centroid'][1],
                'volume_mean': cluster_data['centroid'][2],
                'trend_mean': cluster_data['centroid'][3]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Cluster summary saved to {summary_file}")


def create_consolidation_config(
    target_clusters: int = 20,
    min_cluster_size_pct: float = 0.03,
    max_cluster_size_pct: float = 0.08,
    coverage_target: float = 0.95,
    **kwargs
) -> ConsolidationConfig:
    """Create a consolidation configuration with custom parameters."""
    
    return ConsolidationConfig(
        target_clusters=target_clusters,
        min_cluster_size_pct=min_cluster_size_pct,
        max_cluster_size_pct=max_cluster_size_pct,
        coverage_target=coverage_target,
        **kwargs
    )