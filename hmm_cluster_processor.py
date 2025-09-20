"""
HMM Cluster Processor

Integration script to process HMM clustering results with validation and merging.
This script demonstrates how to integrate the validation and merging tools with 
your existing market analysis pipeline.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from cluster_validation import ClusterValidator
from cluster_merger import ClusterMerger, validate_and_merge_clusters

class HMMClusterProcessor:
    """
    Processor for HMM clustering results with validation and merging capabilities
    """
    
    def __init__(self, 
                 min_cluster_size: int = 50,
                 max_cluster_size: int = 1000,
                 similarity_threshold: float = 0.8,
                 validation_threshold: float = 0.7):
        """
        Initialize the processor
        
        Args:
            min_cluster_size: Minimum samples per cluster
            max_cluster_size: Maximum samples per cluster  
            similarity_threshold: Similarity threshold for merging
            validation_threshold: Minimum validation score to pass
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.similarity_threshold = similarity_threshold
        self.validation_threshold = validation_threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_hmm_outcome_file(self, outcome_file_path: str) -> Dict:
        """
        Process an HMM clustering outcome file
        
        Args:
            outcome_file_path: Path to the HMM clustering outcome JSON file
            
        Returns:
            Dict with processed results and recommendations
        """
        
        self.logger.info(f"Processing HMM outcome file: {outcome_file_path}")
        
        # Load the outcome file
        try:
            with open(outcome_file_path, 'r') as f:
                outcome_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading outcome file: {e}")
            return {'error': f"Failed to load outcome file: {e}"}
        
        # Extract cluster information
        cluster_info = self._extract_cluster_info(outcome_data)
        
        if 'error' in cluster_info:
            return cluster_info
        
        # Generate synthetic features for validation (since we don't have the raw data)
        # In practice, you would load your actual feature data
        synthetic_data = self._generate_synthetic_features(cluster_info)
        
        # Create cluster labels array
        cluster_labels = self._create_cluster_labels(cluster_info)
        
        # Process with validation and merging
        results = self._validate_and_process_clusters(
            synthetic_data, cluster_labels, cluster_info, outcome_data
        )
        
        # Generate recommendations
        results['recommendations'] = self._generate_processing_recommendations(results)
        
        # Save processed results
        output_path = self._save_processed_results(outcome_file_path, results)
        results['output_file'] = output_path
        
        return results
    
    def _extract_cluster_info(self, outcome_data: Dict) -> Dict:
        """Extract cluster information from outcome data"""
        
        try:
            hmm_models = outcome_data['artifacts']['hmm_clustering_result']['hmm_models']
            
            cluster_info = {
                'total_clusters': len(hmm_models),
                'cluster_data': {},
                'regime_counts': {},
                'sample_counts': {}
            }
            
            for model in hmm_models:
                cluster_id = model['cluster_id']
                regime_count = model['regime_count']
                
                cluster_info['cluster_data'][cluster_id] = model
                cluster_info['regime_counts'][cluster_id] = regime_count
                
                # Try to extract sample count if available
                # This would need to be adapted based on your actual data structure
                cluster_info['sample_counts'][cluster_id] = regime_count * 10  # Placeholder
            
            return cluster_info
            
        except Exception as e:
            self.logger.error(f"Error extracting cluster info: {e}")
            return {'error': f"Failed to extract cluster info: {e}"}
    
    def _generate_synthetic_features(self, cluster_info: Dict) -> np.ndarray:
        """
        Generate synthetic features for demonstration
        In practice, you would load your actual feature data here
        """
        
        total_samples = sum(cluster_info['sample_counts'].values())
        n_features = 15  # Typical number of features for market regimes
        
        # Generate synthetic data with some cluster structure
        np.random.seed(42)
        data = []
        
        sample_idx = 0
        for cluster_id, sample_count in cluster_info['sample_counts'].items():
            # Generate cluster-specific data
            cluster_center = np.random.randn(n_features) * 2
            cluster_data = np.random.randn(sample_count, n_features) + cluster_center
            data.append(cluster_data)
            sample_idx += sample_count
        
        synthetic_data = np.vstack(data)
        
        self.logger.warning("Using synthetic features for demonstration. "
                          "Replace with actual feature loading in production.")
        
        return synthetic_data
    
    def _create_cluster_labels(self, cluster_info: Dict) -> np.ndarray:
        """Create cluster labels array from cluster info"""
        
        labels = []
        
        for cluster_id, sample_count in cluster_info['sample_counts'].items():
            labels.extend([cluster_id] * sample_count)
        
        return np.array(labels)
    
    def _validate_and_process_clusters(self, 
                                     data: np.ndarray,
                                     cluster_labels: np.ndarray,
                                     cluster_info: Dict,
                                     outcome_data: Dict) -> Dict:
        """Validate and process clusters"""
        
        # Extract regime features if available
        regime_features = self._extract_regime_features(outcome_data)
        
        # Run validation and merging
        final_labels, validation_results, merge_report = validate_and_merge_clusters(
            data=data,
            cluster_labels=cluster_labels,
            regime_features=regime_features,
            min_cluster_size=self.min_cluster_size,
            similarity_threshold=self.similarity_threshold
        )
        
        # Analyze final results
        final_cluster_info = self._analyze_final_clusters(final_labels, cluster_info)
        
        results = {
            'original_clustering': {
                'total_clusters': len(np.unique(cluster_labels)),
                'cluster_info': cluster_info
            },
            'validation_results': validation_results,
            'merge_report': merge_report,
            'final_clustering': {
                'total_clusters': len(np.unique(final_labels)),
                'cluster_labels': final_labels.tolist(),
                'cluster_analysis': final_cluster_info
            },
            'processing_summary': {
                'validation_passed': validation_results['validation_passed'],
                'merging_performed': 'executed_merges' in merge_report,
                'cluster_reduction': len(np.unique(cluster_labels)) - len(np.unique(final_labels)),
                'improvement_achieved': self._assess_improvement(validation_results, merge_report)
            }
        }
        
        return results
    
    def _extract_regime_features(self, outcome_data: Dict) -> Optional[Dict]:
        """Extract regime features from outcome data"""
        
        # This would be implemented based on your specific outcome data structure
        # Look for regime characteristics, market conditions, etc.
        
        try:
            # Placeholder - adapt based on your data structure
            regime_features = {
                'symbol': outcome_data.get('metadata', {}).get('symbol', 'unknown'),
                'timeframe': outcome_data.get('metadata', {}).get('timeframe', 'unknown'),
                'regime_reduction': outcome_data.get('metadata', {}).get('regime_to_cluster_reduction', 'unknown')
            }
            
            return regime_features
            
        except Exception as e:
            self.logger.warning(f"Could not extract regime features: {e}")
            return None
    
    def _analyze_final_clusters(self, final_labels: np.ndarray, original_cluster_info: Dict) -> Dict:
        """Analyze final cluster distribution"""
        
        unique_labels, counts = np.unique(final_labels, return_counts=True)
        
        analysis = {
            'cluster_sizes': dict(zip(unique_labels.astype(int), counts.astype(int))),
            'size_statistics': {
                'min': int(np.min(counts)),
                'max': int(np.max(counts)),
                'mean': float(np.mean(counts)),
                'median': float(np.median(counts)),
                'std': float(np.std(counts))
            },
            'quality_metrics': {
                'clusters_below_min_size': int(np.sum(counts < self.min_cluster_size)),
                'clusters_above_max_size': int(np.sum(counts > self.max_cluster_size)),
                'size_balance_score': float(1.0 / (1.0 + np.std(counts) / np.mean(counts)))
            }
        }
        
        return analysis
    
    def _assess_improvement(self, validation_results: Dict, merge_report: Dict) -> bool:
        """Assess whether processing improved the clustering"""
        
        if 'final_validation' not in merge_report:
            return False
        
        original_score = validation_results.get('overall_score', 0.0)
        final_score = merge_report['final_validation'].get('overall_score', 0.0)
        
        return final_score > original_score
    
    def _generate_processing_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on processing results"""
        
        recommendations = []
        
        # Validation-based recommendations
        if not results['processing_summary']['validation_passed']:
            recommendations.append("Clustering validation failed. Consider adjusting clustering parameters.")
        
        # Merging-based recommendations
        if results['processing_summary']['merging_performed']:
            reduction = results['processing_summary']['cluster_reduction']
            recommendations.append(f"Successfully merged {reduction} micro-clusters. "
                                 f"Final clustering has {results['final_clustering']['total_clusters']} clusters.")
        else:
            recommendations.append("No cluster merging was needed. Original clustering structure preserved.")
        
        # Quality-based recommendations
        final_analysis = results['final_clustering']['cluster_analysis']
        quality = final_analysis['quality_metrics']
        
        if quality['clusters_below_min_size'] > 0:
            recommendations.append(f"{quality['clusters_below_min_size']} clusters still below minimum size. "
                                 f"Consider more aggressive merging or different clustering parameters.")
        
        if quality['size_balance_score'] < 0.5:
            recommendations.append("Cluster sizes are still unbalanced. Consider hierarchical clustering approach.")
        
        # Improvement assessment
        if results['processing_summary']['improvement_achieved']:
            recommendations.append("Processing successfully improved clustering quality.")
        else:
            recommendations.append("Processing did not significantly improve clustering. "
                                 "Consider alternative clustering approaches.")
        
        return recommendations
    
    def _save_processed_results(self, original_file_path: str, results: Dict) -> str:
        """Save processed results to file"""
        
        original_path = Path(original_file_path)
        output_path = original_path.parent / f"{original_path.stem}_processed.json"
        
        # Remove numpy arrays for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Processed results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving processed results: {e}")
            return ""
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def print_processing_summary(self, results: Dict) -> None:
        """Print a summary of processing results"""
        
        print("=" * 80)
        print("HMM CLUSTER PROCESSING SUMMARY")
        print("=" * 80)
        
        # Original clustering
        orig = results['original_clustering']
        print(f"\nORIGINAL CLUSTERING:")
        print(f"  Total clusters: {orig['total_clusters']}")
        
        # Processing results
        summary = results['processing_summary']
        print(f"\nPROCESSING RESULTS:")
        print(f"  Validation passed: {summary['validation_passed']}")
        print(f"  Merging performed: {summary['merging_performed']}")
        print(f"  Cluster reduction: {summary['cluster_reduction']}")
        print(f"  Improvement achieved: {summary['improvement_achieved']}")
        
        # Final clustering
        final = results['final_clustering']
        print(f"\nFINAL CLUSTERING:")
        print(f"  Total clusters: {final['total_clusters']}")
        
        analysis = final['cluster_analysis']
        stats = analysis['size_statistics']
        print(f"  Size range: {stats['min']} - {stats['max']}")
        print(f"  Mean size: {stats['mean']:.1f}")
        
        quality = analysis['quality_metrics']
        print(f"  Clusters below min size: {quality['clusters_below_min_size']}")
        print(f"  Size balance score: {quality['size_balance_score']:.3f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("=" * 80)


def process_hmm_outcome_file(file_path: str, 
                           min_cluster_size: int = 50,
                           similarity_threshold: float = 0.8) -> Dict:
    """
    Convenience function to process an HMM outcome file
    
    Args:
        file_path: Path to HMM clustering outcome JSON file
        min_cluster_size: Minimum samples per cluster
        similarity_threshold: Similarity threshold for merging
        
    Returns:
        Dict with processing results
    """
    
    processor = HMMClusterProcessor(
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold
    )
    
    results = processor.process_hmm_outcome_file(file_path)
    processor.print_processing_summary(results)
    
    return results


# Example usage with your specific file
def example_usage():
    """Example of processing the specific HMM outcome file"""
    
    file_path = "/workspace/outcomes/market_analysis_hmm_clustering_outcome_20250920_165525.json"
    
    # Process the file
    results = process_hmm_outcome_file(
        file_path=file_path,
        min_cluster_size=50,  # Require at least 50 samples per cluster
        similarity_threshold=0.75  # Merge clusters with >75% similarity
    )
    
    return results

if __name__ == "__main__":
    example_usage()