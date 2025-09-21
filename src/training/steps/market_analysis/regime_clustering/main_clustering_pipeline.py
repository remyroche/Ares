#!/usr/bin/env python3
"""
Main Clustering Pipeline for HMM Regime Consolidation.

This is the main entry point for clustering HMM regimes into larger, coherent clusters
suitable for ML model training. Orchestrates the entire clustering pipeline including
validation, analysis, and export.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.tprint import tprint

from .regime_clusterer import RegimeClusterer
from .cluster_validator import ClusterValidator
from .cluster_analyzer import ClusterAnalyzer


class RegimeClusteringPipeline:
    """
    Main pipeline for clustering HMM regimes.
    
    Orchestrates the complete clustering process:
    1. Load HMM regime discovery results
    2. Perform clustering with size constraints
    3. Validate cluster quality
    4. Analyze cluster characteristics
    5. Export results for ML training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the clustering pipeline."""
        self.config = config
        self.logger = system_logger.getChild('RegimeClusteringPipeline')
        
        # Initialize components
        self.clusterer = RegimeClusterer(config)
        self.validator = ClusterValidator(config)
        self.analyzer = ClusterAnalyzer(config)
        
        tprint("🔧 RegimeClusteringPipeline initialized")
        self.logger.info("Pipeline initialized with config: %s", config)
    
    def run_clustering_pipeline(self, 
                              hmm_outcome_path: str,
                              output_dir: str,
                              save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Run the complete clustering pipeline.
        
        Args:
            hmm_outcome_path: Path to HMM regime discovery outcome file
            output_dir: Directory to save results
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with complete pipeline results
        """
        tprint("🚀 Starting regime clustering pipeline")
        
        pipeline_start_time = datetime.now()
        
        # Step 1: Cluster regimes
        tprint("Step 1: Clustering regimes")
        clustering_results = self.clusterer.cluster_regimes(hmm_outcome_path)
        
        if save_intermediate:
            self._save_intermediate_results(clustering_results, output_dir, "clustering_results")
        
        # Step 2: Validate clustering quality
        tprint("Step 2: Validating cluster quality")
        validation_results = self.validator.validate_clustering_results(
            self.clusterer.cluster_labels,
            self.clusterer.regime_coordinates,
            self.clusterer.cluster_stats,
            self.clusterer.regime_data['regime_assignments']
        )
        
        if save_intermediate:
            self._save_intermediate_results(validation_results, output_dir, "validation_results")
        
        # Step 3: Analyze cluster characteristics
        tprint("Step 3: Analyzing cluster characteristics")
        characteristics = self.analyzer.analyze_cluster_characteristics(
            self.clusterer.cluster_stats,
            self.clusterer.regime_coordinates,
            self.clusterer.cluster_labels
        )
        
        # Step 4: Create comprehensive summary
        tprint("Step 4: Creating comprehensive summary")
        summary = self.analyzer.create_cluster_summary(
            self.clusterer.cluster_stats,
            characteristics,
            validation_results
        )
        
        # Step 5: Export results for ML training
        tprint("Step 5: Exporting results for ML training")
        exported_files = self.analyzer.export_for_ml_training(
            self.clusterer.cluster_stats,
            characteristics,
            output_dir
        )
        
        # Step 6: Generate cluster names
        cluster_names = self.analyzer.generate_cluster_names(characteristics)
        
        # Compile final results
        pipeline_results = {
            'pipeline_metadata': {
                'timestamp': pipeline_start_time.isoformat(),
                'execution_time': (datetime.now() - pipeline_start_time).total_seconds(),
                'config': self.config,
                'hmm_input_file': hmm_outcome_path,
                'output_directory': output_dir
            },
            'clustering_results': clustering_results,
            'validation_results': validation_results,
            'cluster_characteristics': characteristics,
            'cluster_names': cluster_names,
            'summary': summary,
            'exported_files': exported_files,
            'recommendations': self._generate_pipeline_recommendations(
                clustering_results, validation_results, characteristics
            )
        }
        
        # Save final results
        final_output_file = Path(output_dir) / "regime_clustering_results.json"
        with open(final_output_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        pipeline_end_time = datetime.now()
        execution_time = (pipeline_end_time - pipeline_start_time).total_seconds()
        
        tprint(f"🎉 Pipeline completed successfully in {execution_time:.2f} seconds!")
        tprint(f"📁 Results saved to: {output_dir}")
        
        # Print summary
        self._print_pipeline_summary(pipeline_results)
        
        return pipeline_results
    
    def _save_intermediate_results(self, results: Dict[str, Any], output_dir: str, filename: str) -> None:
        """Save intermediate results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved intermediate results to {file_path}")
    
    def _generate_pipeline_recommendations(self, 
                                         clustering_results: Dict[str, Any],
                                         validation_results: Dict[str, Any],
                                         characteristics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        # Get validation recommendations
        overall_quality = validation_results.get('overall_quality', {})
        recommendations.extend(overall_quality.get('recommendations', []))
        
        # Check cluster count
        total_clusters = len(clustering_results['clustering_results']['cluster_stats'])
        target_clusters = self.config.get('target_clusters', 20)
        
        if abs(total_clusters - target_clusters) > 5:
            recommendations.append(
                f"Cluster count ({total_clusters}) differs significantly from target ({target_clusters}). "
                "Consider adjusting clustering parameters."
            )
        
        # Check for imbalanced clusters
        cluster_percentages = [
            stats['percentage'] for stats in clustering_results['clustering_results']['cluster_stats'].values()
        ]
        min_pct, max_pct = min(cluster_percentages), max(cluster_percentages)
        
        if max_pct - min_pct > 10:  # More than 10% difference
            recommendations.append(
                "Significant cluster size imbalance detected. Consider merging small clusters "
                "or splitting large ones for better ML training distribution."
            )
        
        # Check market type diversity
        market_types = set()
        for char in characteristics.values():
            market_types.add(char['interpretation']['market_type'])
        
        if len(market_types) < 5:
            recommendations.append(
                f"Limited market type diversity ({len(market_types)} types). "
                "Consider adjusting clustering to capture more market conditions."
            )
        
        return recommendations
    
    def _print_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of pipeline results."""
        clustering = results['clustering_results']['clustering_results']
        validation = results['validation_results']
        summary = results['summary']
        
        tprint("\n" + "="*60)
        tprint("🎯 REGIME CLUSTERING PIPELINE SUMMARY")
        tprint("="*60)
        
        # Overview
        overview = summary['overview']
        tprint(f"📊 Total Clusters: {overview['total_clusters']}")
        tprint(f"📈 Total Samples: {overview['total_samples']:,}")
        tprint(f"⭐ Quality Level: {overview['quality_level']}")
        tprint(f"🎯 Quality Score: {overview['overall_quality_score']:.3f}")
        
        # Size distribution
        size_stats = summary['size_distribution']
        tprint(f"\n📏 Cluster Size Distribution:")
        tprint(f"   Min: {size_stats['min_size']:,} samples")
        tprint(f"   Max: {size_stats['max_size']:,} samples")
        tprint(f"   Mean: {size_stats['mean_size']:.0f} samples")
        tprint(f"   Std: {size_stats['size_std']:.0f} samples")
        
        # Market types
        market_types = summary['market_type_distribution']
        tprint(f"\n🏪 Market Type Distribution:")
        for market_type, count in sorted(market_types.items()):
            tprint(f"   {market_type}: {count} clusters")
        
        # Validation metrics
        validity = validation['validity']
        tprint(f"\n✅ Validation Metrics:")
        tprint(f"   Silhouette Score: {validity['silhouette_score']:.3f}")
        tprint(f"   Calinski-Harabasz: {validity['calinski_harabasz_score']:.1f}")
        tprint(f"   Davies-Bouldin: {validity['davies_bouldin_score']:.3f}")
        
        # Recommendations
        recommendations = results['recommendations']
        if recommendations:
            tprint(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                tprint(f"   {i}. {rec}")
        
        tprint("="*60)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for regime clustering."""
    return {
        'target_clusters': 20,
        'min_cluster_size_pct': 0.03,  # 3%
        'max_cluster_size_pct': 0.08,  # 8%
        'max_noise_pct': 0.05,  # 5%
        'linkage_method': 'ward',
        'min_samples_per_regime': 5,
        'min_silhouette_score': 0.3,
        'max_size_variance': 0.01,
        'min_constraint_satisfaction': 0.8
    }


def main():
    """Main entry point for the clustering pipeline."""
    parser = argparse.ArgumentParser(description='Cluster HMM regimes into coherent groups')
    parser.add_argument('--hmm-outcome', required=True, 
                       help='Path to HMM regime discovery outcome JSON file')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save clustering results')
    parser.add_argument('--config', 
                       help='Path to configuration JSON file (optional)')
    parser.add_argument('--target-clusters', type=int, default=20,
                       help='Target number of clusters (default: 20)')
    parser.add_argument('--min-cluster-size', type=float, default=0.03,
                       help='Minimum cluster size as percentage (default: 0.03)')
    parser.add_argument('--max-cluster-size', type=float, default=0.08,
                       help='Maximum cluster size as percentage (default: 0.08)')
    parser.add_argument('--max-noise', type=float, default=0.05,
                       help='Maximum noise percentage (default: 0.05)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config['target_clusters'] = args.target_clusters
    config['min_cluster_size_pct'] = args.min_cluster_size
    config['max_cluster_size_pct'] = args.max_cluster_size
    config['max_noise_pct'] = args.max_noise
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    try:
        pipeline = RegimeClusteringPipeline(config)
        results = pipeline.run_clustering_pipeline(
            args.hmm_outcome,
            args.output_dir
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())