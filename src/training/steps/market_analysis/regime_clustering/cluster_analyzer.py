#!/usr/bin/env python3
"""
Cluster Analyzer for Regime Clustering Results.

This module provides analysis and reporting tools for regime clustering results,
including visualization, interpretation, and export capabilities.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from src.utils.logger import system_logger
from src.utils.tprint import tprint


class ClusterAnalyzer:
    """
    Analyzes and visualizes regime clustering results.
    
    Provides functionality for:
    - Cluster interpretation and naming
    - Statistical analysis of clusters
    - Export for ML training
    - Visualization of results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the cluster analyzer."""
        self.config = config
        self.logger = system_logger.getChild('ClusterAnalyzer')
        
        tprint("🔧 ClusterAnalyzer initialized")
    
    def analyze_cluster_characteristics(self, 
                                      cluster_stats: Dict[str, Any],
                                      regime_coordinates: np.ndarray,
                                      cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze characteristics of each cluster for interpretation.
        
        Args:
            cluster_stats: Statistics for each cluster
            regime_coordinates: 3D coordinates for each regime
            cluster_labels: Cluster assignment for each regime
            
        Returns:
            Dictionary with cluster characteristics
        """
        tprint("🔍 Analyzing cluster characteristics")
        
        characteristics = {}
        
        for cluster_id, stats in cluster_stats.items():
            # Get cluster regimes
            regime_ids = stats['regime_ids']
            cluster_coords = regime_coordinates[regime_ids]
            
            # Calculate characteristics
            centroid = np.array(stats['centroid'])
            
            # Interpret cluster based on centroid position
            interpretation = self._interpret_cluster(centroid)
            
            # Calculate regime diversity within cluster
            diversity = self._calculate_cluster_diversity(cluster_coords)
            
            # Calculate market condition representation
            market_conditions = self._analyze_market_conditions(centroid)
            
            characteristics[cluster_id] = {
                'cluster_id': cluster_id,
                'interpretation': interpretation,
                'market_conditions': market_conditions,
                'diversity_score': diversity,
                'centroid': centroid.tolist(),
                'regime_count': stats['regime_count'],
                'sample_count': stats['sample_count'],
                'percentage': stats['percentage']
            }
        
        tprint(f"✅ Analyzed characteristics for {len(characteristics)} clusters")
        return characteristics
    
    def _interpret_cluster(self, centroid: np.ndarray) -> Dict[str, str]:
        """
        Interpret cluster characteristics based on centroid position.
        
        Args:
            centroid: 3D centroid coordinates [momentum, volatility, volume]
            
        Returns:
            Dictionary with cluster interpretation
        """
        momentum, volatility, volume = centroid
        
        # Momentum interpretation
        if momentum < 2:
            momentum_desc = "Low Momentum"
        elif momentum < 5:
            momentum_desc = "Medium Momentum"
        else:
            momentum_desc = "High Momentum"
        
        # Volatility interpretation
        if volatility < 2:
            volatility_desc = "Low Volatility"
        elif volatility < 5:
            volatility_desc = "Medium Volatility"
        else:
            volatility_desc = "High Volatility"
        
        # Volume interpretation
        if volume < 2:
            volume_desc = "Low Volume"
        elif volume < 5:
            volume_desc = "Medium Volume"
        else:
            volume_desc = "High Volume"
        
        # Combined interpretation
        if momentum < 3 and volatility < 3 and volume < 3:
            market_type = "Quiet Market"
        elif momentum > 6 and volatility > 6 and volume > 6:
            market_type = "Active Market"
        elif volatility > 6:
            market_type = "Volatile Market"
        elif momentum > 6:
            market_type = "Trending Market"
        elif volume > 6:
            market_type = "High Activity Market"
        else:
            market_type = "Balanced Market"
        
        return {
            'momentum': momentum_desc,
            'volatility': volatility_desc,
            'volume': volume_desc,
            'market_type': market_type,
            'description': f"{market_type} with {momentum_desc.lower()}, {volatility_desc.lower()}, and {volume_desc.lower()}"
        }
    
    def _calculate_cluster_diversity(self, cluster_coords: np.ndarray) -> float:
        """
        Calculate diversity score for regimes within a cluster.
        
        Args:
            cluster_coords: Coordinates of regimes in the cluster
            
        Returns:
            Diversity score (0-1, higher means more diverse)
        """
        if len(cluster_coords) < 2:
            return 0.0
        
        # Calculate standard deviation for each dimension
        std_scores = np.std(cluster_coords, axis=0)
        
        # Normalize by maximum possible std (assuming 0-8 range)
        max_std = np.std([0, 8])  # Maximum std for 0-8 range
        normalized_std = std_scores / max_std
        
        # Average across dimensions
        diversity_score = np.mean(normalized_std)
        
        return min(1.0, diversity_score)
    
    def _analyze_market_conditions(self, centroid: np.ndarray) -> Dict[str, Any]:
        """
        Analyze what market conditions this cluster represents.
        
        Args:
            centroid: 3D centroid coordinates
            
        Returns:
            Dictionary with market condition analysis
        """
        momentum, volatility, volume = centroid
        
        conditions = {
            'momentum_level': momentum,
            'volatility_level': volatility,
            'volume_level': volume,
            'risk_level': 'Low' if volatility < 3 else 'Medium' if volatility < 6 else 'High',
            'activity_level': 'Low' if volume < 3 else 'Medium' if volume < 6 else 'High',
            'trend_strength': 'Weak' if momentum < 3 else 'Medium' if momentum < 6 else 'Strong'
        }
        
        # Trading implications
        if volatility < 3 and momentum < 3:
            conditions['trading_implication'] = 'Suitable for conservative strategies'
        elif volatility > 6 and momentum > 6:
            conditions['trading_implication'] = 'Suitable for aggressive strategies'
        elif volatility > 6:
            conditions['trading_implication'] = 'Requires risk management focus'
        elif momentum > 6:
            conditions['trading_implication'] = 'Suitable for trend-following strategies'
        else:
            conditions['trading_implication'] = 'Suitable for balanced strategies'
        
        return conditions
    
    def create_cluster_summary(self, 
                             cluster_stats: Dict[str, Any],
                             characteristics: Dict[str, Any],
                             validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive summary of clustering results.
        
        Args:
            cluster_stats: Statistics for each cluster
            characteristics: Cluster characteristics
            validation_results: Validation metrics
            
        Returns:
            Dictionary with comprehensive summary
        """
        tprint("📊 Creating cluster summary")
        
        # Overall statistics
        total_clusters = len(cluster_stats)
        total_samples = sum(stats['sample_count'] for stats in cluster_stats.values())
        
        # Cluster size distribution
        cluster_sizes = [stats['sample_count'] for stats in cluster_stats.values()]
        size_stats = {
            'min_size': min(cluster_sizes),
            'max_size': max(cluster_sizes),
            'mean_size': np.mean(cluster_sizes),
            'median_size': np.median(cluster_sizes),
            'size_std': np.std(cluster_sizes)
        }
        
        # Market type distribution
        market_types = {}
        for char in characteristics.values():
            market_type = char['interpretation']['market_type']
            market_types[market_type] = market_types.get(market_type, 0) + 1
        
        # Quality assessment
        overall_quality = validation_results.get('overall_quality', {})
        quality_level = overall_quality.get('quality_level', 'Unknown')
        
        summary = {
            'overview': {
                'total_clusters': total_clusters,
                'total_samples': total_samples,
                'quality_level': quality_level,
                'overall_quality_score': overall_quality.get('overall_score', 0.0)
            },
            'size_distribution': size_stats,
            'market_type_distribution': market_types,
            'cluster_details': {
                'statistics': cluster_stats,
                'characteristics': characteristics
            },
            'validation': validation_results,
            'recommendations': overall_quality.get('recommendations', [])
        }
        
        tprint("✅ Cluster summary created")
        return summary
    
    def export_for_ml_training(self, 
                             cluster_stats: Dict[str, Any],
                             characteristics: Dict[str, Any],
                             output_dir: str) -> Dict[str, str]:
        """
        Export clustering results in formats suitable for ML training.
        
        Args:
            cluster_stats: Statistics for each cluster
            characteristics: Cluster characteristics
            output_dir: Directory to save exports
            
        Returns:
            Dictionary with paths to exported files
        """
        tprint("💾 Exporting results for ML training")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 1. Export cluster mapping (regime_id -> cluster_id)
        cluster_mapping = {}
        for cluster_id, stats in cluster_stats.items():
            for regime_id in stats['regime_ids']:
                cluster_mapping[regime_id] = cluster_id
        
        mapping_file = output_path / "cluster_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(cluster_mapping, f, indent=2)
        exported_files['cluster_mapping'] = str(mapping_file)
        
        # 2. Export cluster characteristics for ML model naming
        characteristics_file = output_path / "cluster_characteristics.json"
        with open(characteristics_file, 'w') as f:
            json.dump(characteristics, f, indent=2)
        exported_files['cluster_characteristics'] = str(characteristics_file)
        
        # 3. Export cluster summary
        summary = self.create_cluster_summary(cluster_stats, characteristics, {})
        summary_file = output_path / "cluster_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        exported_files['cluster_summary'] = str(summary_file)
        
        # 4. Export CSV for easy analysis
        cluster_df_data = []
        for cluster_id, stats in cluster_stats.items():
            char = characteristics.get(cluster_id, {})
            row = {
                'cluster_id': cluster_id,
                'regime_count': stats['regime_count'],
                'sample_count': stats['sample_count'],
                'percentage': stats['percentage'],
                'market_type': char.get('interpretation', {}).get('market_type', 'Unknown'),
                'momentum_level': stats['centroid'][0],
                'volatility_level': stats['centroid'][1],
                'volume_level': stats['centroid'][2],
                'diversity_score': char.get('diversity_score', 0.0)
            }
            cluster_df_data.append(row)
        
        cluster_df = pd.DataFrame(cluster_df_data)
        csv_file = output_path / "cluster_analysis.csv"
        cluster_df.to_csv(csv_file, index=False)
        exported_files['cluster_analysis_csv'] = str(csv_file)
        
        tprint(f"✅ Exported {len(exported_files)} files to {output_dir}")
        return exported_files
    
    def generate_cluster_names(self, characteristics: Dict[str, Any]) -> Dict[int, str]:
        """
        Generate meaningful names for clusters based on their characteristics.
        
        Args:
            characteristics: Cluster characteristics
            
        Returns:
            Dictionary mapping cluster_id to cluster name
        """
        cluster_names = {}
        
        for cluster_id, char in characteristics.items():
            interpretation = char['interpretation']
            market_type = interpretation['market_type']
            
            # Create descriptive name
            momentum = char['centroid'][0]
            volatility = char['centroid'][1]
            volume = char['centroid'][2]
            
            # Generate name based on characteristics
            if market_type == "Quiet Market":
                name = f"Quiet_M{momentum:.0f}_V{volatility:.0f}_Vol{volume:.0f}"
            elif market_type == "Active Market":
                name = f"Active_M{momentum:.0f}_V{volatility:.0f}_Vol{volume:.0f}"
            elif market_type == "Volatile Market":
                name = f"Volatile_M{momentum:.0f}_V{volatility:.0f}_Vol{volume:.0f}"
            elif market_type == "Trending Market":
                name = f"Trending_M{momentum:.0f}_V{volatility:.0f}_Vol{volume:.0f}"
            elif market_type == "High Activity Market":
                name = f"HighActivity_M{momentum:.0f}_V{volatility:.0f}_Vol{volume:.0f}"
            else:
                name = f"Balanced_M{momentum:.0f}_V{volatility:.0f}_Vol{volume:.0f}"
            
            cluster_names[int(cluster_id)] = name
        
        return cluster_names