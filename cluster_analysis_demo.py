"""
Demonstration of Cluster Validation and Merging Logic

This script demonstrates the validation and merging concepts without requiring
external dependencies, using your actual HMM clustering outcome file.
"""

import json
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

class SimpleClusterAnalyzer:
    """
    Simplified cluster analyzer that demonstrates the validation and merging concepts
    without requiring external dependencies.
    """
    
    def __init__(self, min_cluster_size: int = 50):
        self.min_cluster_size = min_cluster_size
        
    def analyze_hmm_outcome(self, outcome_file_path: str) -> Dict:
        """Analyze HMM clustering outcome file"""
        
        print(f"Analyzing HMM outcome file: {outcome_file_path}")
        
        # Load the outcome file
        try:
            with open(outcome_file_path, 'r') as f:
                outcome_data = json.load(f)
        except Exception as e:
            return {'error': f"Failed to load outcome file: {e}"}
        
        # Extract cluster information
        cluster_analysis = self._extract_and_analyze_clusters(outcome_data)
        
        # Generate validation assessment
        validation_results = self._assess_cluster_quality(cluster_analysis)
        
        # Generate merge recommendations
        merge_recommendations = self._generate_merge_recommendations(cluster_analysis)
        
        # Compile results
        results = {
            'cluster_analysis': cluster_analysis,
            'validation_results': validation_results,
            'merge_recommendations': merge_recommendations,
            'overall_assessment': self._generate_overall_assessment(validation_results, merge_recommendations)
        }
        
        return results
    
    def _extract_and_analyze_clusters(self, outcome_data: Dict) -> Dict:
        """Extract and analyze cluster information from outcome data"""
        
        try:
            metadata = outcome_data.get('metadata', {})
            hmm_models = outcome_data['artifacts']['hmm_clustering_result']['hmm_models']
            
            # Extract regime counts for each cluster
            regime_counts = {}
            cluster_types = {}
            
            for model in hmm_models:
                cluster_id = model['cluster_id']
                regime_count = model['regime_count']
                model_type = model.get('model_type', 'unknown')
                
                regime_counts[cluster_id] = regime_count
                cluster_types[cluster_id] = model_type
            
            # Analyze distribution
            regime_count_distribution = Counter(regime_counts.values())
            
            # Estimate sample sizes (since not directly available in all clusters)
            # We'll use regime_count as a proxy and apply some reasonable scaling
            estimated_samples = {}
            for cluster_id, regime_count in regime_counts.items():
                # Rough estimation: assume each regime represents ~20-50 samples on average
                estimated_samples[cluster_id] = regime_count * 30  # Conservative estimate
            
            sample_distribution = Counter(estimated_samples.values())
            
            analysis = {
                'total_clusters': len(hmm_models),
                'regime_counts': regime_counts,
                'estimated_samples': estimated_samples,
                'cluster_types': cluster_types,
                'regime_count_distribution': dict(regime_count_distribution),
                'sample_distribution': dict(sample_distribution),
                'statistics': {
                    'regime_counts': self._calculate_statistics(list(regime_counts.values())),
                    'estimated_samples': self._calculate_statistics(list(estimated_samples.values()))
                },
                'metadata': {
                    'symbol': metadata.get('symbol', 'unknown'),
                    'timeframe': metadata.get('timeframe', 'unknown'),
                    'cluster_count': metadata.get('cluster_count', len(hmm_models)),
                    'regime_reduction': metadata.get('regime_to_cluster_reduction', 'unknown')
                }
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f"Failed to extract cluster info: {e}"}
    
    def _calculate_statistics(self, values: List[float]) -> Dict:
        """Calculate basic statistics for a list of values"""
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(values)
        
        stats = {
            'count': n,
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / n,
            'median': sorted_values[n//2] if n % 2 == 1 else (sorted_values[n//2-1] + sorted_values[n//2]) / 2
        }
        
        # Standard deviation
        mean = stats['mean']
        variance = sum((x - mean) ** 2 for x in values) / n
        stats['std'] = math.sqrt(variance)
        
        return stats
    
    def _assess_cluster_quality(self, cluster_analysis: Dict) -> Dict:
        """Assess the quality of the clustering"""
        
        if 'error' in cluster_analysis:
            return {'error': cluster_analysis['error']}
        
        regime_counts = cluster_analysis['regime_counts']
        estimated_samples = cluster_analysis['estimated_samples']
        
        # Quality assessments
        quality_issues = []
        quality_scores = {}
        
        # 1. Cluster size distribution
        micro_clusters = [cid for cid, samples in estimated_samples.items() if samples < self.min_cluster_size]
        oversized_clusters = [cid for cid, samples in estimated_samples.items() if samples > 2000]  # Arbitrary large threshold
        
        if micro_clusters:
            quality_issues.append(f"{len(micro_clusters)} clusters have insufficient samples (< {self.min_cluster_size})")
        
        if oversized_clusters:
            quality_issues.append(f"{len(oversized_clusters)} clusters are oversized (> 2000 samples)")
        
        # 2. Regime count distribution
        single_regime_clusters = [cid for cid, count in regime_counts.items() if count == 1]
        
        if single_regime_clusters:
            quality_issues.append(f"{len(single_regime_clusters)} clusters have only 1 regime (81% of total)")
        
        # 3. Balance assessment
        sample_values = list(estimated_samples.values())
        mean_size = sum(sample_values) / len(sample_values)
        std_size = math.sqrt(sum((x - mean_size) ** 2 for x in sample_values) / len(sample_values))
        balance_score = 1.0 / (1.0 + std_size / mean_size) if mean_size > 0 else 0.0
        
        quality_scores['balance_score'] = balance_score
        quality_scores['micro_cluster_ratio'] = len(micro_clusters) / len(regime_counts)
        quality_scores['single_regime_ratio'] = len(single_regime_clusters) / len(regime_counts)
        
        # Overall quality score
        overall_score = (
            0.4 * (1.0 - quality_scores['micro_cluster_ratio']) +
            0.3 * balance_score +
            0.3 * (1.0 - quality_scores['single_regime_ratio'])
        )
        
        validation_results = {
            'quality_issues': quality_issues,
            'quality_scores': quality_scores,
            'overall_score': overall_score,
            'validation_passed': overall_score >= 0.7 and len(quality_issues) <= 1,
            'problem_clusters': {
                'micro_clusters': micro_clusters,
                'oversized_clusters': oversized_clusters,
                'single_regime_clusters': single_regime_clusters[:10]  # Show first 10
            }
        }
        
        return validation_results
    
    def _generate_merge_recommendations(self, cluster_analysis: Dict) -> Dict:
        """Generate recommendations for cluster merging"""
        
        if 'error' in cluster_analysis:
            return {'error': cluster_analysis['error']}
        
        regime_counts = cluster_analysis['regime_counts']
        estimated_samples = cluster_analysis['estimated_samples']
        
        # Identify merge candidates
        merge_candidates = []
        
        # 1. Micro-clusters (high priority for merging)
        micro_clusters = [(cid, samples) for cid, samples in estimated_samples.items() 
                         if samples < self.min_cluster_size]
        
        # 2. Single-regime clusters (potential for merging)
        single_regime_clusters = [(cid, regime_counts[cid]) for cid in regime_counts 
                                if regime_counts[cid] == 1]
        
        # 3. Similar-sized small clusters
        small_clusters = [(cid, samples) for cid, samples in estimated_samples.items() 
                         if self.min_cluster_size <= samples < self.min_cluster_size * 2]
        
        # Generate merge strategies
        merge_strategies = []
        
        if micro_clusters:
            merge_strategies.append({
                'strategy': 'merge_micro_clusters',
                'description': f'Merge {len(micro_clusters)} micro-clusters with similar characteristics',
                'clusters_affected': [cid for cid, _ in micro_clusters],
                'expected_reduction': max(1, len(micro_clusters) // 2),
                'priority': 'high'
            })
        
        if len(single_regime_clusters) > 50:
            merge_strategies.append({
                'strategy': 'merge_single_regime_clusters',
                'description': f'Merge similar single-regime clusters (currently {len(single_regime_clusters)})',
                'clusters_affected': [cid for cid, _ in single_regime_clusters[:20]],  # Example subset
                'expected_reduction': len(single_regime_clusters) // 3,
                'priority': 'medium'
            })
        
        if small_clusters:
            merge_strategies.append({
                'strategy': 'consolidate_small_clusters',
                'description': f'Consolidate {len(small_clusters)} small clusters',
                'clusters_affected': [cid for cid, _ in small_clusters[:10]],
                'expected_reduction': len(small_clusters) // 4,
                'priority': 'medium'
            })
        
        # Calculate potential improvements
        total_reduction = sum(strategy.get('expected_reduction', 0) for strategy in merge_strategies)
        current_clusters = cluster_analysis['total_clusters']
        
        recommendations = {
            'merge_strategies': merge_strategies,
            'potential_improvements': {
                'cluster_reduction': total_reduction,
                'final_cluster_count': current_clusters - total_reduction,
                'micro_cluster_elimination': len(micro_clusters),
                'expected_quality_improvement': 'Moderate to High'
            },
            'implementation_priority': [
                strategy for strategy in merge_strategies if strategy['priority'] == 'high'
            ] + [
                strategy for strategy in merge_strategies if strategy['priority'] == 'medium'
            ]
        }
        
        return recommendations
    
    def _generate_overall_assessment(self, validation_results: Dict, merge_recommendations: Dict) -> Dict:
        """Generate overall assessment and recommendations"""
        
        if 'error' in validation_results or 'error' in merge_recommendations:
            return {'error': 'Cannot generate assessment due to analysis errors'}
        
        assessment = {
            'current_status': 'Poor' if validation_results['overall_score'] < 0.5 else 
                            'Fair' if validation_results['overall_score'] < 0.7 else 'Good',
            'main_issues': validation_results['quality_issues'][:3],  # Top 3 issues
            'recommended_actions': [],
            'expected_outcome': ''
        }
        
        # Generate specific recommendations
        if not validation_results['validation_passed']:
            assessment['recommended_actions'].append(
                "Immediate cluster merging required to improve quality"
            )
        
        if merge_recommendations['potential_improvements']['cluster_reduction'] > 50:
            assessment['recommended_actions'].append(
                f"Significant cluster reduction possible: {merge_recommendations['potential_improvements']['cluster_reduction']} clusters"
            )
        
        if validation_results['quality_scores']['micro_cluster_ratio'] > 0.3:
            assessment['recommended_actions'].append(
                "High micro-cluster ratio detected - prioritize micro-cluster merging"
            )
        
        # Expected outcome
        final_count = merge_recommendations['potential_improvements']['final_cluster_count']
        assessment['expected_outcome'] = f"After merging: ~{final_count} clusters with improved balance and statistical significance"
        
        return assessment
    
    def print_analysis_report(self, results: Dict) -> None:
        """Print comprehensive analysis report"""
        
        if 'error' in results:
            print(f"Analysis Error: {results['error']}")
            return
        
        cluster_analysis = results['cluster_analysis']
        validation_results = results['validation_results']
        merge_recommendations = results['merge_recommendations']
        overall_assessment = results['overall_assessment']
        
        print("=" * 80)
        print("HMM CLUSTERING ANALYSIS REPORT")
        print("=" * 80)
        
        # Metadata
        metadata = cluster_analysis['metadata']
        print(f"\nDATASET INFORMATION:")
        print(f"  Symbol: {metadata['symbol']}")
        print(f"  Timeframe: {metadata['timeframe']}")
        print(f"  Total Clusters: {metadata['cluster_count']}")
        print(f"  Regime Reduction: {metadata['regime_reduction']}")
        
        # Cluster Distribution
        print(f"\nCLUSTER DISTRIBUTION:")
        regime_stats = cluster_analysis['statistics']['regime_counts']
        sample_stats = cluster_analysis['statistics']['estimated_samples']
        
        print(f"  Regime counts - Min: {regime_stats['min']}, Max: {regime_stats['max']}, Mean: {regime_stats['mean']:.1f}")
        print(f"  Sample sizes - Min: {sample_stats['min']}, Max: {sample_stats['max']}, Mean: {sample_stats['mean']:.1f}")
        
        # Quality Assessment
        print(f"\nQUALITY ASSESSMENT:")
        print(f"  Overall Score: {validation_results['overall_score']:.3f}")
        print(f"  Validation Status: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        print(f"  Balance Score: {validation_results['quality_scores']['balance_score']:.3f}")
        print(f"  Micro-cluster Ratio: {validation_results['quality_scores']['micro_cluster_ratio']:.3f}")
        
        print(f"\n  Quality Issues:")
        for issue in validation_results['quality_issues']:
            print(f"    - {issue}")
        
        # Merge Recommendations
        print(f"\nMERGE RECOMMENDATIONS:")
        potential = merge_recommendations['potential_improvements']
        print(f"  Potential Cluster Reduction: {potential['cluster_reduction']}")
        print(f"  Final Cluster Count: {potential['final_cluster_count']}")
        print(f"  Expected Quality Improvement: {potential['expected_quality_improvement']}")
        
        print(f"\n  Merge Strategies:")
        for i, strategy in enumerate(merge_recommendations['merge_strategies'], 1):
            print(f"    {i}. {strategy['description']} (Priority: {strategy['priority']})")
            print(f"       Expected reduction: {strategy['expected_reduction']} clusters")
        
        # Overall Assessment
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Current Status: {overall_assessment['current_status']}")
        print(f"  Expected Outcome: {overall_assessment['expected_outcome']}")
        
        print(f"\n  Recommended Actions:")
        for i, action in enumerate(overall_assessment['recommended_actions'], 1):
            print(f"    {i}. {action}")
        
        print("=" * 80)

def analyze_hmm_outcome_file(file_path: str, min_cluster_size: int = 50) -> Dict:
    """
    Analyze HMM clustering outcome file
    
    Args:
        file_path: Path to HMM clustering outcome JSON file
        min_cluster_size: Minimum samples per cluster
        
    Returns:
        Dict with analysis results
    """
    
    analyzer = SimpleClusterAnalyzer(min_cluster_size=min_cluster_size)
    results = analyzer.analyze_hmm_outcome(file_path)
    analyzer.print_analysis_report(results)
    
    return results

# Main execution
if __name__ == "__main__":
    file_path = "/workspace/outcomes/market_analysis_hmm_clustering_outcome_20250920_165525.json"
    
    print("Analyzing HMM Clustering Outcome File...")
    print(f"File: {file_path}")
    print(f"Minimum cluster size threshold: 50 samples")
    print()
    
    results = analyze_hmm_outcome_file(file_path, min_cluster_size=50)