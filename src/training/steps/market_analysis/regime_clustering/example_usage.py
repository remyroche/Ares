#!/usr/bin/env python3
"""
Example Usage of Regime Clustering Pipeline.

This script demonstrates how to use the regime clustering pipeline to cluster
HMM regimes into larger, coherent groups suitable for ML model training.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.training.steps.market_analysis.regime_clustering.main_clustering_pipeline import RegimeClusteringPipeline
from src.training.steps.market_analysis.regime_clustering.config import get_config_template, create_custom_config


def example_with_template_config():
    """Example using a predefined configuration template."""
    print("🔧 Example 1: Using predefined configuration template")
    
    # Use the 'balanced' template configuration
    config = get_config_template('balanced')
    
    # Create pipeline
    pipeline = RegimeClusteringPipeline(config.to_dict())
    
    # Example paths (replace with actual paths)
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    output_dir = "/workspace/outputs/regime_clustering_example"
    
    # Run clustering pipeline
    try:
        results = pipeline.run_clustering_pipeline(hmm_outcome_path, output_dir)
        
        print(f"✅ Clustering completed successfully!")
        print(f"📊 Total clusters: {results['summary']['overview']['total_clusters']}")
        print(f"⭐ Quality level: {results['summary']['overview']['quality_level']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return None


def example_with_custom_config():
    """Example using a custom configuration."""
    print("\n🔧 Example 2: Using custom configuration")
    
    # Create custom configuration
    config = create_custom_config(
        target_clusters=18,  # Slightly fewer clusters
        min_cluster_size_pct=0.035,  # 3.5% minimum
        max_cluster_size_pct=0.075,  # 7.5% maximum
        max_noise_pct=0.04,  # 4% maximum noise
        linkage_method='complete'  # Different linkage method
    )
    
    # Create pipeline
    pipeline = RegimeClusteringPipeline(config.to_dict())
    
    # Example paths (replace with actual paths)
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    output_dir = "/workspace/outputs/regime_clustering_custom"
    
    # Run clustering pipeline
    try:
        results = pipeline.run_clustering_pipeline(hmm_outcome_path, output_dir)
        
        print(f"✅ Custom clustering completed successfully!")
        print(f"📊 Total clusters: {results['summary']['overview']['total_clusters']}")
        print(f"⭐ Quality level: {results['summary']['overview']['quality_level']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Custom clustering failed: {e}")
        return None


def example_with_validation_focus():
    """Example focusing on high-quality clustering validation."""
    print("\n🔧 Example 3: High-quality validation focus")
    
    # Create configuration focused on quality
    config = create_custom_config(
        target_clusters=16,  # Fewer, larger clusters
        min_cluster_size_pct=0.04,  # 4% minimum
        max_cluster_size_pct=0.08,  # 8% maximum
        max_noise_pct=0.03,  # 3% maximum noise
        min_silhouette_score=0.35,  # Higher quality threshold
        min_constraint_satisfaction=0.85,  # Stricter constraints
        linkage_method='ward'  # Ward linkage for compact clusters
    )
    
    # Create pipeline
    pipeline = RegimeClusteringPipeline(config.to_dict())
    
    # Example paths (replace with actual paths)
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    output_dir = "/workspace/outputs/regime_clustering_high_quality"
    
    # Run clustering pipeline
    try:
        results = pipeline.run_clustering_pipeline(hmm_outcome_path, output_dir)
        
        print(f"✅ High-quality clustering completed successfully!")
        print(f"📊 Total clusters: {results['summary']['overview']['total_clusters']}")
        print(f"⭐ Quality level: {results['summary']['overview']['quality_level']}")
        print(f"🎯 Quality score: {results['summary']['overview']['overall_quality_score']:.3f}")
        
        # Print validation details
        validation = results['validation_results']
        print(f"📈 Silhouette score: {validation['validity']['silhouette_score']:.3f}")
        print(f"📏 Size compliance: {validation['size_distribution']['constraint_satisfaction']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ High-quality clustering failed: {e}")
        return None


def analyze_results(results):
    """Analyze and display clustering results."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("📊 CLUSTERING RESULTS ANALYSIS")
    print("="*60)
    
    # Overview
    overview = results['summary']['overview']
    print(f"Total Clusters: {overview['total_clusters']}")
    print(f"Total Samples: {overview['total_samples']:,}")
    print(f"Quality Level: {overview['quality_level']}")
    print(f"Quality Score: {overview['overall_quality_score']:.3f}")
    
    # Market type distribution
    market_types = results['summary']['market_type_distribution']
    print(f"\nMarket Type Distribution:")
    for market_type, count in sorted(market_types.items()):
        print(f"  {market_type}: {count} clusters")
    
    # Cluster details
    cluster_stats = results['clustering_results']['clustering_results']['cluster_stats']
    print(f"\nCluster Size Distribution:")
    for cluster_id, stats in sorted(cluster_stats.items(), key=lambda x: x[1]['sample_count'], reverse=True)[:5]:
        print(f"  Cluster {cluster_id}: {stats['sample_count']:,} samples ({stats['percentage']:.1f}%)")
    
    # Recommendations
    recommendations = results['recommendations']
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("="*60)


def main():
    """Run all examples."""
    print("🚀 Regime Clustering Pipeline Examples")
    print("="*60)
    
    # Run examples
    results1 = example_with_template_config()
    results2 = example_with_custom_config()
    results3 = example_with_validation_focus()
    
    # Analyze results
    if results1:
        analyze_results(results1)
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()