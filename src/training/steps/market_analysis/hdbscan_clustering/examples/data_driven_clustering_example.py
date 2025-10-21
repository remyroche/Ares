"""
Data-Driven Clustering Parameters Example

This example demonstrates how to use the data-driven clustering parameter
optimization system to replace hardcoded parameters with adaptive, data-driven alternatives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import time
from pathlib import Path

# Import tprint utilities for extensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

# Import data-driven optimization components
from ..config.data_driven_config import DataDrivenClusteringConfig
from ..optimization.data_driven_clustering_optimizer import DataDrivenClusteringOptimizer
from ..optimization.data_driven_feature_weights import DataDrivenFeatureWeightOptimizer
from ..optimization.data_driven_merging_thresholds import DataDrivenMergingThresholdOptimizer
from ..optimization.data_driven_temporal_windows import DataDrivenTemporalWindowOptimizer
from ..optimization.data_driven_validation_thresholds import DataDrivenValidationThresholdOptimizer

# Import clustering components
from ..hdbscan_clusterer import HDBSCANClusterer
from ..similarity_merger_data_driven import DataDrivenSimilarityMerger

# Import feature preparation
from ...clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep

logger = logging.getLogger(__name__)

class DataDrivenClusteringExample:
    """
    Example demonstrating data-driven clustering parameter optimization.
    
    This class shows how to replace hardcoded parameters with data-driven
    alternatives for better clustering performance.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[DataDrivenClusteringConfig] = None):
        """Initialize the example with configuration."""
        tprint_info("🔧 Initializing DataDrivenClusteringExample")
        start_time = time.perf_counter()
        
        self.config = config or DataDrivenClusteringConfig()
        self.optimization_results = {}
        
        init_time = time.perf_counter() - start_time
        tprint_success(f"✅ DataDrivenClusteringExample initialized in {init_time:.3f}s")
        
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def run_complete_example(self, 
                           market_data: pd.DataFrame,
                           features: np.ndarray,
                           feature_names: List[str]) -> Dict[str, Any]:
        """
        Run complete data-driven clustering optimization example.
        
        Args:
            market_data: Market data for analysis
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        try:
            tprint_info("🚀 Starting complete data-driven clustering example...")
            tprint_debug(f"Market data shape: {market_data.shape}, Features shape: {features.shape}")
            
            results = {
                'start_time': time.time(),
                'steps_completed': [],
                'optimization_results': {},
                'recommendations': {},
                'performance_metrics': {}
            }
            
            # Step 1: Data-driven feature weight optimization
            tprint_info("📊 Step 1: Optimizing feature group weights...")
            with tprint_timer("Feature weight optimization"):
                feature_weight_results = self._optimize_feature_weights(
                    features, feature_names, market_data
                )
            results['steps_completed'].append('feature_weights')
            results['optimization_results']['feature_weights'] = feature_weight_results
            tprint_success("✅ Feature weight optimization completed")
            
            # Step 2: Data-driven temporal window optimization
            tprint_info("⏰ Step 2: Optimizing temporal window sizes...")
            with tprint_timer("Temporal window optimization"):
                temporal_window_results = self._optimize_temporal_windows(market_data)
            results['steps_completed'].append('temporal_windows')
            results['optimization_results']['temporal_windows'] = temporal_window_results
            tprint_success("✅ Temporal window optimization completed")
            
            # Step 3: Data-driven merging threshold optimization
            tprint_info("🔗 Step 3: Optimizing regime merging thresholds...")
            with tprint_timer("Merging threshold optimization"):
                merging_threshold_results = self._optimize_merging_thresholds(features)
            results['steps_completed'].append('merging_thresholds')
            results['optimization_results']['merging_thresholds'] = merging_threshold_results
            tprint_success("✅ Merging threshold optimization completed")
            
            # Step 4: Data-driven validation threshold optimization
            tprint_info("📊 Step 4: Optimizing cluster validation thresholds...")
            with tprint_timer("Validation threshold optimization"):
                validation_threshold_results = self._optimize_validation_thresholds(features)
            results['steps_completed'].append('validation_thresholds')
            results['optimization_results']['validation_thresholds'] = validation_threshold_results
            tprint_success("✅ Validation threshold optimization completed")
            
            # Step 5: Generate recommendations
            tprint_info("💡 Step 5: Generating recommendations...")
            with tprint_timer("Recommendation generation"):
                recommendations = self._generate_recommendations(results['optimization_results'])
            results['recommendations'] = recommendations
            tprint_success("✅ Recommendations generated")
            
            # Step 6: Calculate performance metrics
            tprint_info("📈 Step 6: Calculating performance metrics...")
            with tprint_timer("Performance metrics calculation"):
                performance_metrics = self._calculate_performance_metrics(results)
            results['performance_metrics'] = performance_metrics
            tprint_success("✅ Performance metrics calculated")
            
            results['end_time'] = time.time()
            results['total_time'] = results['end_time'] - results['start_time']
            
            tprint_success(f"✅ Complete example finished in {results['total_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Complete example failed: {e}")
            raise
    
    def _optimize_feature_weights(self, 
                                 features: np.ndarray,
                                 feature_names: List[str],
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize feature group weights using data-driven methods."""
        try:
            # Create optimizer
            optimizer = DataDrivenFeatureWeightOptimizer(self.config.feature_weights)
            
            # Create simple clustering function for optimization
            def clustering_func(features):
                from sklearn.cluster import KMeans
                n_clusters = min(5, features.shape[0] // 10)
                if n_clusters < 2:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                return kmeans.fit_predict(features)
            
            # Optimize weights
            result = optimizer.optimize_weights(
                features=features,
                feature_names=feature_names,
                market_data=market_data,
                clustering_func=clustering_func
            )
            
            return {
                'optimal_weights': result.optimal_weights,
                'optimization_score': result.optimization_score,
                'validation_scores': result.validation_scores,
                'feature_importance_scores': result.feature_importance_scores,
                'n_trials': len(result.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Feature weight optimization failed: {e}")
            return {'error': str(e)}
    
    def _optimize_temporal_windows(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize temporal window sizes using data-driven methods."""
        try:
            # Create optimizer
            optimizer = DataDrivenTemporalWindowOptimizer(self.config.temporal_windows)
            
            # Create windowed clustering function
            def windowed_clustering_func(data, windows):
                # This is a simplified example - in practice, you'd implement
                # proper windowing logic
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(data) // 10)
                if n_clusters < 2:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                return kmeans.fit_predict(data.values)
            
            # Optimize windows
            result = optimizer.optimize_windows(
                market_data=market_data,
                clustering_func=windowed_clustering_func
            )
            
            return {
                'optimal_windows': result.optimal_windows,
                'optimization_score': result.optimization_score,
                'validation_scores': result.validation_scores,
                'volatility_adaptation': result.volatility_adaptation,
                'n_trials': len(result.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Temporal window optimization failed: {e}")
            return {'error': str(e)}
    
    def _optimize_merging_thresholds(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimize regime merging thresholds using data-driven methods."""
        try:
            # Create optimizer
            optimizer = DataDrivenMergingThresholdOptimizer(self.config.merging_thresholds)
            
            # Create initial clustering
            from sklearn.cluster import KMeans
            n_clusters = min(8, features.shape[0] // 10)
            if n_clusters < 2:
                n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            initial_labels = kmeans.fit_predict(features)
            
            # Create merging function
            def merging_func(labels, features, thresholds):
                # This is a simplified example - in practice, you'd implement
                # proper merging logic based on thresholds
                return labels
            
            # Optimize thresholds
            result = optimizer.optimize_thresholds(
                cluster_labels=initial_labels,
                features=features,
                merging_func=merging_func
            )
            
            return {
                'optimal_thresholds': result.optimal_thresholds,
                'optimization_score': result.optimization_score,
                'validation_scores': result.validation_scores,
                'merging_statistics': result.merging_statistics,
                'n_trials': len(result.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Merging threshold optimization failed: {e}")
            return {'error': str(e)}
    
    def _optimize_validation_thresholds(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimize cluster validation thresholds using data-driven methods."""
        try:
            # Create optimizer
            optimizer = DataDrivenValidationThresholdOptimizer(self.config.validation_thresholds)
            
            # Create clustering function
            def clustering_func(features):
                from sklearn.cluster import KMeans
                n_clusters = min(5, features.shape[0] // 10)
                if n_clusters < 2:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                return kmeans.fit_predict(features)
            
            # Optimize thresholds
            result = optimizer.optimize_thresholds(
                features=features,
                clustering_func=clustering_func
            )
            
            return {
                'optimal_thresholds': result.optimal_thresholds,
                'optimization_score': result.optimization_score,
                'validation_scores': result.validation_scores,
                'statistical_validation': result.statistical_validation,
                'bootstrap_validation': result.bootstrap_validation,
                'n_trials': len(result.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Validation threshold optimization failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on optimization results."""
        try:
            recommendations = {
                'feature_weights': {},
                'temporal_windows': {},
                'merging_thresholds': {},
                'validation_thresholds': {},
                'overall': {}
            }
            
            # Feature weight recommendations
            if 'feature_weights' in optimization_results:
                fw_results = optimization_results['feature_weights']
                if 'optimal_weights' in fw_results:
                    weights = fw_results['optimal_weights']
                    recommendations['feature_weights'] = {
                        'recommended_weights': weights,
                        'insights': self._analyze_feature_weights(weights),
                        'implementation_notes': [
                            "Replace hardcoded weights (w_returns=0.50, w_vol=0.30, w_volume=0.20) with optimized values",
                            "Apply weights using sqrt() for variance scaling",
                            "Consider re-optimizing weights periodically as market conditions change"
                        ]
                    }
            
            # Temporal window recommendations
            if 'temporal_windows' in optimization_results:
                tw_results = optimization_results['temporal_windows']
                if 'optimal_windows' in tw_results:
                    windows = tw_results['optimal_windows']
                    recommendations['temporal_windows'] = {
                        'recommended_windows': windows,
                        'insights': self._analyze_temporal_windows(windows, tw_results.get('volatility_adaptation', {})),
                        'implementation_notes': [
                            "Replace hardcoded window_size=300 with optimized value",
                            "Replace hardcoded smoothing_window=5 with optimized value",
                            "Consider volatility-adaptive windows for different market regimes"
                        ]
                    }
            
            # Merging threshold recommendations
            if 'merging_thresholds' in optimization_results:
                mt_results = optimization_results['merging_thresholds']
                if 'optimal_thresholds' in mt_results:
                    thresholds = mt_results['optimal_thresholds']
                    recommendations['merging_thresholds'] = {
                        'recommended_thresholds': thresholds,
                        'insights': self._analyze_merging_thresholds(thresholds),
                        'implementation_notes': [
                            "Replace hardcoded similarity_threshold=0.8 with optimized value",
                            "Replace hardcoded distance_threshold=0.2 with optimized value",
                            "Replace hardcoded p_value_threshold=0.05 with optimized value"
                        ]
                    }
            
            # Validation threshold recommendations
            if 'validation_thresholds' in optimization_results:
                vt_results = optimization_results['validation_thresholds']
                if 'optimal_thresholds' in vt_results:
                    thresholds = vt_results['optimal_thresholds']
                    recommendations['validation_thresholds'] = {
                        'recommended_thresholds': thresholds,
                        'insights': self._analyze_validation_thresholds(thresholds),
                        'implementation_notes': [
                            "Replace hardcoded min_silhouette=0.2 with optimized value",
                            "Replace hardcoded max_dbi=2.5 with optimized value",
                            "Use statistical significance testing for threshold validation"
                        ]
                    }
            
            # Overall recommendations
            recommendations['overall'] = {
                'summary': "Data-driven optimization completed successfully",
                'next_steps': [
                    "Integrate optimized parameters into production clustering pipeline",
                    "Set up periodic re-optimization schedule",
                    "Monitor clustering performance with new parameters",
                    "Consider implementing adaptive parameter updates"
                ],
                'benefits': [
                    "Improved clustering quality through data-driven parameter selection",
                    "Better adaptation to different market conditions",
                    "Reduced reliance on hardcoded heuristics",
                    "Enhanced regime discovery accuracy"
                ]
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_weights(self, weights: Dict[str, float]) -> List[str]:
        """Analyze feature weight optimization results."""
        insights = []
        
        # Find dominant feature group
        max_weight_group = max(weights.items(), key=lambda x: x[1])
        insights.append(f"Dominant feature group: {max_weight_group[0]} (weight: {max_weight_group[1]:.3f})")
        
        # Check for balanced weights
        weight_values = list(weights.values())
        weight_std = np.std(weight_values)
        if weight_std < 0.1:
            insights.append("Feature groups are well-balanced")
        else:
            insights.append("Feature groups show significant imbalance - consider rebalancing")
        
        # Compare to hardcoded weights
        hardcoded_weights = {'returns': 0.50, 'volatility': 0.30, 'volume': 0.20}
        for group, optimized_weight in weights.items():
            if group in hardcoded_weights:
                hardcoded_weight = hardcoded_weights[group]
                change = (optimized_weight - hardcoded_weight) / hardcoded_weight * 100
                insights.append(f"{group} weight changed by {change:+.1f}% from hardcoded value")
        
        return insights
    
    def _analyze_temporal_windows(self, windows: Dict[str, int], volatility_adaptation: Dict[str, Any]) -> List[str]:
        """Analyze temporal window optimization results."""
        insights = []
        
        window_size = windows.get('window_size', 300)
        smoothing_window = windows.get('smoothing_window', 5)
        
        # Compare to hardcoded values
        hardcoded_window_size = 300
        hardcoded_smoothing_window = 5
        
        window_change = (window_size - hardcoded_window_size) / hardcoded_window_size * 100
        smoothing_change = (smoothing_window - hardcoded_smoothing_window) / hardcoded_smoothing_window * 100
        
        insights.append(f"Window size changed by {window_change:+.1f}% from hardcoded value")
        insights.append(f"Smoothing window changed by {smoothing_change:+.1f}% from hardcoded value")
        
        # Volatility adaptation analysis
        if volatility_adaptation:
            volatility_regime = volatility_adaptation.get('volatility_regime', 'unknown')
            adaptation_score = volatility_adaptation.get('adaptation_score', 0.0)
            
            insights.append(f"Volatility regime detected: {volatility_regime}")
            insights.append(f"Volatility adaptation score: {adaptation_score:.3f}")
            
            if adaptation_score > 0.7:
                insights.append("Windows are well-adapted to current volatility regime")
            else:
                insights.append("Windows may need further adaptation to volatility regime")
        
        return insights
    
    def _analyze_merging_thresholds(self, thresholds: Dict[str, float]) -> List[str]:
        """Analyze merging threshold optimization results."""
        insights = []
        
        # Compare to hardcoded values
        hardcoded_thresholds = {
            'similarity_threshold': 0.8,
            'distance_threshold': 0.2,
            'p_value_threshold': 0.05
        }
        
        for threshold_name, optimized_value in thresholds.items():
            if threshold_name in hardcoded_thresholds:
                hardcoded_value = hardcoded_thresholds[threshold_name]
                change = (optimized_value - hardcoded_value) / hardcoded_value * 100
                insights.append(f"{threshold_name} changed by {change:+.1f}% from hardcoded value")
        
        # Analyze threshold relationships
        sim_thresh = thresholds.get('similarity_threshold', 0.8)
        dist_thresh = thresholds.get('distance_threshold', 0.2)
        
        if sim_thresh > 0.9:
            insights.append("High similarity threshold suggests distinct regimes")
        elif sim_thresh < 0.6:
            insights.append("Low similarity threshold suggests similar regimes")
        
        if dist_thresh > 0.3:
            insights.append("High distance threshold suggests well-separated clusters")
        elif dist_thresh < 0.1:
            insights.append("Low distance threshold suggests close clusters")
        
        return insights
    
    def _analyze_validation_thresholds(self, thresholds: Dict[str, float]) -> List[str]:
        """Analyze validation threshold optimization results."""
        insights = []
        
        # Compare to hardcoded values
        hardcoded_thresholds = {
            'min_silhouette': 0.2,
            'max_dbi': 2.5,
            'min_stability': 0.7
        }
        
        for threshold_name, optimized_value in thresholds.items():
            if threshold_name in hardcoded_thresholds:
                hardcoded_value = hardcoded_thresholds[threshold_name]
                change = (optimized_value - hardcoded_value) / hardcoded_value * 100
                insights.append(f"{threshold_name} changed by {change:+.1f}% from hardcoded value")
        
        # Analyze threshold quality
        min_silhouette = thresholds.get('min_silhouette', 0.2)
        max_dbi = thresholds.get('max_dbi', 2.5)
        
        if min_silhouette > 0.4:
            insights.append("High silhouette threshold suggests good cluster separation")
        elif min_silhouette < 0.1:
            insights.append("Low silhouette threshold suggests poor cluster separation")
        
        if max_dbi < 2.0:
            insights.append("Low DBI threshold suggests compact clusters")
        elif max_dbi > 3.0:
            insights.append("High DBI threshold suggests loose clusters")
        
        return insights
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the optimization process."""
        try:
            metrics = {
                'total_optimization_time': results.get('total_time', 0.0),
                'steps_completed': len(results.get('steps_completed', [])),
                'success_rate': 0.0,
                'average_optimization_score': 0.0,
                'parameter_improvements': {}
            }
            
            # Calculate success rate
            optimization_results = results.get('optimization_results', {})
            successful_optimizations = 0
            total_optimizations = len(optimization_results)
            
            scores = []
            for step, step_results in optimization_results.items():
                if 'error' not in step_results:
                    successful_optimizations += 1
                    if 'optimization_score' in step_results:
                        scores.append(step_results['optimization_score'])
            
            if total_optimizations > 0:
                metrics['success_rate'] = successful_optimizations / total_optimizations
            
            if scores:
                metrics['average_optimization_score'] = np.mean(scores)
            
            # Calculate parameter improvements
            for step, step_results in optimization_results.items():
                if 'error' not in step_results:
                    metrics['parameter_improvements'][step] = {
                        'optimization_score': step_results.get('optimization_score', 0.0),
                        'n_trials': step_results.get('n_trials', 0),
                        'status': 'success'
                    }
                else:
                    metrics['parameter_improvements'][step] = {
                        'status': 'failed',
                        'error': step_results.get('error', 'Unknown error')
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate synthetic market data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
    
    # Generate price data
    price = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n_samples))
    
    # Generate volume data
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # Generate volatility data
    volatility = np.random.exponential(0.02, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'close': price,
        'volume': volume,
        'volatility': volatility
    })
    
    data.set_index('timestamp', inplace=True)
    
    return data


def create_sample_features(market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Create sample features for testing."""
    np.random.seed(42)
    
    n_samples = len(market_data)
    
    # Generate synthetic features
    features = np.random.randn(n_samples, 20)
    feature_names = [
        'return_1h', 'return_4h', 'return_1d',
        'volatility_1h', 'volatility_4h', 'volatility_1d',
        'volume_1h', 'volume_4h', 'volume_1d',
        'rsi_14', 'rsi_21', 'rsi_50',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr_14', 'atr_21'
    ]
    
    return features, feature_names


def main():
    """Main function to run the data-driven clustering example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    logger.info("Creating sample market data...")
    market_data = create_sample_market_data(1000)
    features, feature_names = create_sample_features(market_data)
    
    # Create configuration
    config = DataDrivenClusteringConfig()
    
    # Create example instance
    example = DataDrivenClusteringExample(config)
    
    # Run complete example
    logger.info("Running complete data-driven clustering example...")
    results = example.run_complete_example(market_data, features, feature_names)
    
    # Print results
    logger.info("=" * 80)
    logger.info("DATA-DRIVEN CLUSTERING OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"Total optimization time: {results['total_time']:.2f} seconds")
    logger.info(f"Steps completed: {results['steps_completed']}")
    
    # Print optimization results
    for step, step_results in results['optimization_results'].items():
        logger.info(f"\n{step.upper()} OPTIMIZATION:")
        if 'error' in step_results:
            logger.error(f"  Error: {step_results['error']}")
        else:
            logger.info(f"  Optimization score: {step_results.get('optimization_score', 'N/A'):.4f}")
            logger.info(f"  Number of trials: {step_results.get('n_trials', 'N/A')}")
    
    # Print recommendations
    logger.info("\nRECOMMENDATIONS:")
    for category, recs in results['recommendations'].items():
        if isinstance(recs, dict) and 'insights' in recs:
            logger.info(f"\n{category.upper()}:")
            for insight in recs['insights']:
                logger.info(f"  • {insight}")
    
    # Print performance metrics
    logger.info("\nPERFORMANCE METRICS:")
    metrics = results['performance_metrics']
    logger.info(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
    logger.info(f"  Average optimization score: {metrics.get('average_optimization_score', 0):.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Example completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()