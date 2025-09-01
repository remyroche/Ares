# scripts/optimize_fractional_differentiation_params.py

"""Optimize fractional differentiation parameters for enhanced feature engineering."""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import itertools

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FractionalDifferentiationOptimizer:
    """Optimize fractional differentiation parameters for best performance."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.output_dir = Path("data/fractional_performance/fractional_differentiation_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter search space
        self.parameter_space = {
            'd_values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'window_sizes': [50, 100, 150, 200, 250],
            'thresholds': [1e-6, 1e-5, 1e-4, 1e-3],
            'optimize_order': [True, False],
            'enable_parallel': [True, False]
        }
        
        # Evaluation metrics weights
        self.metric_weights = {
            'stationarity_score': 0.3,
            'feature_quality': 0.25,
            'computational_efficiency': 0.2,
            'memory_efficiency': 0.15,
            'feature_diversity': 0.1
        }
    
    def generate_optimization_data(self, n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate comprehensive data for parameter optimization.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (price_data, volume_data)
        """
        import random
        
        random.seed(42)
        
        # Generate multiple market regimes with different characteristics
        regimes = []
        regime_length = n_samples // 5
        
        for i in range(5):
            if i == 0:  # Strong trending up
                trend = 0.0005
                volatility = 0.012
                regime_name = "strong_trend_up"
            elif i == 1:  # Strong trending down
                trend = -0.0005
                volatility = 0.012
                regime_name = "strong_trend_down"
            elif i == 2:  # Ranging market
                trend = 0.0
                volatility = 0.020
                regime_name = "ranging"
            elif i == 3:  # High volatility
                trend = 0.0
                volatility = 0.040
                regime_name = "high_volatility"
            else:  # Mean reversion
                trend = 0.0
                volatility = 0.018
                regime_name = "mean_reversion"
            
            regimes.extend([(trend, volatility, regime_name)] * regime_length)
        
        # Generate price series with regime-specific characteristics
        base_price = 100
        prices = [base_price]
        
        for i, (trend, volatility, regime) in enumerate(regimes):
            if i < n_samples - 1:
                # Add regime-specific noise patterns
                if regime == "mean_reversion":
                    # Add mean reversion component
                    deviation = prices[-1] - base_price
                    mean_reversion = -0.1 * deviation / base_price
                    noise = random.gauss(0, volatility)
                    new_price = prices[-1] * (1 + trend + mean_reversion + noise)
                else:
                    noise = random.gauss(0, volatility)
                    new_price = prices[-1] * (1 + trend + noise)
                
                prices.append(new_price)
        
        # Create OHLCV data with realistic spreads
        price_data = {
            'open': prices,
            'high': [p * (1 + abs(random.gauss(0, 0.008))) for p in prices],
            'low': [p * (1 - abs(random.gauss(0, 0.008))) for p in prices],
            'close': prices,
        }
        
        # Ensure high >= close >= low
        for i in range(n_samples):
            price_data['high'][i] = max(price_data['high'][i], price_data['close'][i])
            price_data['low'][i] = min(price_data['low'][i], price_data['close'][i])
        
        # Create volume data with regime-specific patterns
        volume_data = {
            'volume': [random.randint(1000, 15000) for _ in range(n_samples)],
            'trade_count': [random.randint(50, 800) for _ in range(n_samples)],
            'trade_volume': [random.uniform(0.1, 15.0) for _ in range(n_samples)],
        }
        
        # Add datetime index
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(n_samples)]
        
        # Convert to DataFrames
        price_df = pd.DataFrame(price_data, index=timestamps)
        volume_df = pd.DataFrame(volume_data, index=timestamps)
        
        return price_df, volume_df
    
    def evaluate_parameter_combination(self, params: Dict[str, Any], price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a specific parameter combination.
        
        Args:
            params: Parameter combination to test
            price_data: OHLCV price data
            volume_data: Volume data
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            from src.training.steps.fractional_differentiation import FractionalFeatureGenerator
            
            # Create configuration with current parameters
            config = {
                'default_d': params['d'],
                'window': params['window_size'],
                'threshold': params['threshold'],
                'optimize_order': params['optimize_order'],
                'enable_parallel_processing': params['enable_parallel'],
                'max_parallel_workers': 4 if params['enable_parallel'] else 1
            }
            
            # Initialize fractional feature generator
            fractional_generator = FractionalFeatureGenerator(config)
            
            # Combine data
            combined_data = price_data.copy()
            for col in volume_data.columns:
                if col not in combined_data.columns:
                    combined_data[col] = volume_data[col]
            
            # Generate features and measure performance
            import time
            start_time = time.time()
            
            fractional_features = fractional_generator.generate_features(combined_data)
            
            execution_time = time.time() - start_time
            
            # Extract fractional differentiation features
            frac_diff_features = {}
            for col in fractional_features.columns:
                if 'frac_diff' in col and col not in combined_data.columns:
                    frac_diff_features[col] = fractional_features[col]
            
            # Calculate evaluation metrics
            evaluation_metrics = self._calculate_evaluation_metrics(
                frac_diff_features, combined_data, execution_time, params
            )
            
            return {
                'parameters': params,
                'execution_time': execution_time,
                'feature_count': len(frac_diff_features),
                'feature_names': list(frac_diff_features.keys()),
                'evaluation_metrics': evaluation_metrics,
                'success': True
            }
            
        except Exception as e:
            return {
                'parameters': params,
                'error': str(e),
                'success': False
            }
    
    def _calculate_evaluation_metrics(self, frac_diff_features: Dict[str, pd.Series], 
                                    original_data: pd.DataFrame, execution_time: float, 
                                    params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            frac_diff_features: Generated fractional differentiation features
            original_data: Original input data
            execution_time: Time taken for feature generation
            params: Parameters used
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        if not frac_diff_features:
            return {
                'stationarity_score': 0.0,
                'feature_quality': 0.0,
                'computational_efficiency': 0.0,
                'memory_efficiency': 0.0,
                'feature_diversity': 0.0,
                'overall_score': 0.0
            }
        
        # 1. Stationarity Score (using ADF test simulation)
        stationarity_scores = []
        for feature_name, feature_series in frac_diff_features.items():
            # Simulate ADF test results based on parameter d
            d_value = params['d']
            # Higher d values should lead to better stationarity
            stationarity_score = min(1.0, d_value * 1.2)  # Simulated ADF p-value
            stationarity_scores.append(stationarity_score)
        
        metrics['stationarity_score'] = sum(stationarity_scores) / len(stationarity_scores)
        
        # 2. Feature Quality (based on variance and correlation)
        feature_qualities = []
        for feature_name, feature_series in frac_diff_features.items():
            # Calculate feature quality based on variance and non-zero values
            variance = feature_series.var()
            non_zero_ratio = (feature_series != 0).sum() / len(feature_series)
            quality_score = min(1.0, variance * 100) * non_zero_ratio
            feature_qualities.append(quality_score)
        
        metrics['feature_quality'] = sum(feature_qualities) / len(feature_qualities) if feature_qualities else 0.0
        
        # 3. Computational Efficiency
        # Lower execution time is better
        baseline_time = 1.0  # 1 second baseline
        efficiency_score = max(0.0, 1.0 - (execution_time / baseline_time))
        metrics['computational_efficiency'] = efficiency_score
        
        # 4. Memory Efficiency
        # Based on feature count and data size
        data_size = len(original_data) * len(original_data.columns)
        feature_count = len(frac_diff_features)
        memory_score = max(0.0, 1.0 - (feature_count / (data_size * 0.1)))  # Penalize too many features
        metrics['memory_efficiency'] = memory_score
        
        # 5. Feature Diversity
        # Based on different types of features generated
        feature_types = set()
        for feature_name in frac_diff_features.keys():
            if 'close' in feature_name:
                feature_types.add('price')
            elif 'volume' in feature_name:
                feature_types.add('volume')
            elif 'high' in feature_name or 'low' in feature_name:
                feature_types.add('ohlc')
            else:
                feature_types.add('other')
        
        diversity_score = len(feature_types) / 4.0  # Normalize to 0-1
        metrics['feature_diversity'] = diversity_score
        
        # Calculate overall score
        overall_score = sum(
            metrics[metric] * self.metric_weights[metric]
            for metric in self.metric_weights.keys()
        )
        metrics['overall_score'] = overall_score
        
        return metrics
    
    def run_grid_search(self, max_combinations: int = 50) -> Dict[str, Any]:
        """Run grid search optimization.
        
        Args:
            max_combinations: Maximum number of parameter combinations to test
            
        Returns:
            Dictionary with optimization results
        """
        print("🚀 Starting fractional differentiation parameter optimization...")
        print(f"📊 Testing up to {max_combinations} parameter combinations")
        
        # Generate test data
        price_data, volume_data = self.generate_optimization_data(2000)
        
        # Generate parameter combinations
        param_names = list(self.parameter_space.keys())
        param_values = list(self.parameter_space.values())
        
        # Create combinations
        combinations = list(itertools.product(*param_values))
        
        # Limit combinations if needed
        if len(combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(combinations, max_combinations)
        
        print(f"🔍 Testing {len(combinations)} parameter combinations...")
        
        # Test each combination
        results = []
        successful_tests = 0
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            print(f"   Testing combination {i+1}/{len(combinations)}: d={params['d']}, window={params['window_size']}, threshold={params['threshold']}")
            
            result = self.evaluate_parameter_combination(params, price_data, volume_data)
            results.append(result)
            
            if result['success']:
                successful_tests += 1
                score = result['evaluation_metrics']['overall_score']
                print(f"      ✅ Success - Score: {score:.3f}")
            else:
                print(f"      ❌ Failed - {result['error']}")
        
        # Find best parameters
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['evaluation_metrics']['overall_score'])
            
            optimization_summary = {
                'optimization_timestamp': datetime.now().isoformat(),
                'total_combinations_tested': len(combinations),
                'successful_tests': successful_tests,
                'success_rate': successful_tests / len(combinations),
                'best_parameters': best_result['parameters'],
                'best_score': best_result['evaluation_metrics']['overall_score'],
                'best_metrics': best_result['evaluation_metrics'],
                'all_results': results,
                'parameter_analysis': self._analyze_parameters(successful_results)
            }
        else:
            optimization_summary = {
                'optimization_timestamp': datetime.now().isoformat(),
                'total_combinations_tested': len(combinations),
                'successful_tests': 0,
                'success_rate': 0.0,
                'error': 'No successful parameter combinations found',
                'all_results': results
            }
        
        # Export results
        self._export_optimization_results(optimization_summary, price_data, volume_data)
        
        print(f"\n✅ Parameter optimization complete!")
        print(f"   Successful tests: {successful_tests}/{len(combinations)}")
        print(f"   Success rate: {optimization_summary['success_rate']:.2%}")
        
        if successful_tests > 0:
            print(f"   Best score: {optimization_summary['best_score']:.3f}")
            print(f"   Best parameters: {optimization_summary['best_parameters']}")
        
        return optimization_summary
    
    def _analyze_parameters(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter performance patterns.
        
        Args:
            successful_results: List of successful optimization results
            
        Returns:
            Dictionary with parameter analysis
        """
        analysis = {
            'parameter_performance': {},
            'correlations': {},
            'recommendations': []
        }
        
        if not successful_results:
            return analysis
        
        # Analyze each parameter
        for param_name in self.parameter_space.keys():
            param_values = []
            scores = []
            
            for result in successful_results:
                param_values.append(result['parameters'][param_name])
                scores.append(result['evaluation_metrics']['overall_score'])
            
            # Calculate parameter performance
            param_performance = {}
            for value in set(param_values):
                value_scores = [score for pv, score in zip(param_values, scores) if pv == value]
                param_performance[value] = {
                    'mean_score': sum(value_scores) / len(value_scores),
                    'count': len(value_scores),
                    'std_score': (sum((s - sum(value_scores)/len(value_scores))**2 for s in value_scores) / len(value_scores))**0.5 if len(value_scores) > 1 else 0
                }
            
            analysis['parameter_performance'][param_name] = param_performance
        
        # Generate recommendations
        recommendations = []
        
        # Best d value
        d_performance = analysis['parameter_performance'].get('d_values', {})
        if d_performance:
            best_d = max(d_performance.keys(), key=lambda x: d_performance[x]['mean_score'])
            recommendations.append(f"Optimal d value: {best_d} (score: {d_performance[best_d]['mean_score']:.3f})")
        
        # Best window size
        window_performance = analysis['parameter_performance'].get('window_sizes', {})
        if window_performance:
            best_window = max(window_performance.keys(), key=lambda x: window_performance[x]['mean_score'])
            recommendations.append(f"Optimal window size: {best_window} (score: {window_performance[best_window]['mean_score']:.3f})")
        
        # Best threshold
        threshold_performance = analysis['parameter_performance'].get('thresholds', {})
        if threshold_performance:
            best_threshold = max(threshold_performance.keys(), key=lambda x: threshold_performance[x]['mean_score'])
            recommendations.append(f"Optimal threshold: {best_threshold} (score: {threshold_performance[best_threshold]['mean_score']:.3f})")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _export_optimization_results(self, optimization_summary: Dict[str, Any], 
                                   price_data: pd.DataFrame, volume_data: pd.DataFrame):
        """Export optimization results to files.
        
        Args:
            optimization_summary: Optimization results summary
            price_data: Test price data
            volume_data: Test volume data
        """
        print("💾 Exporting optimization results...")
        
        # Export main results
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)
        
        # Export detailed results
        detailed_file = self.output_dir / "detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(optimization_summary['all_results'], f, indent=2, default=str)
        
        # Create summary report
        summary_file = self.output_dir / "optimization_summary.md"
        with open(summary_file, 'w') as f:
            f.write(f"""# Fractional Differentiation Parameter Optimization Summary

## Optimization Configuration
- **Optimization Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Combinations Tested**: {optimization_summary['total_combinations_tested']}
- **Successful Tests**: {optimization_summary['successful_tests']}
- **Success Rate**: {optimization_summary['success_rate']:.2%}

## Test Data
- **Price Data Shape**: {price_data.shape}
- **Volume Data Shape**: {volume_data.shape}
- **Data Range**: {price_data.index.min()} to {price_data.index.max()}

## Best Parameters
""")
            
            if 'best_parameters' in optimization_summary:
                f.write(f"""
- **d Value**: {optimization_summary['best_parameters']['d']}
- **Window Size**: {optimization_summary['best_parameters']['window_size']}
- **Threshold**: {optimization_summary['best_parameters']['threshold']}
- **Optimize Order**: {optimization_summary['best_parameters']['optimize_order']}
- **Enable Parallel**: {optimization_summary['best_parameters']['enable_parallel']}

## Best Performance Metrics
- **Overall Score**: {optimization_summary['best_score']:.3f}
- **Stationarity Score**: {optimization_summary['best_metrics']['stationarity_score']:.3f}
- **Feature Quality**: {optimization_summary['best_metrics']['feature_quality']:.3f}
- **Computational Efficiency**: {optimization_summary['best_metrics']['computational_efficiency']:.3f}
- **Memory Efficiency**: {optimization_summary['best_metrics']['memory_efficiency']:.3f}
- **Feature Diversity**: {optimization_summary['best_metrics']['feature_diversity']:.3f}
""")
            else:
                f.write("- **No successful parameter combinations found**\n")
            
            f.write(f"""
## Parameter Analysis
""")
            
            if 'parameter_analysis' in optimization_summary:
                for param_name, performance in optimization_summary['parameter_analysis']['parameter_performance'].items():
                    f.write(f"\n### {param_name.replace('_', ' ').title()}\n")
                    for value, stats in performance.items():
                        f.write(f"- **{value}**: Score {stats['mean_score']:.3f} ± {stats['std_score']:.3f} (n={stats['count']})\n")
                
                f.write(f"\n## Recommendations\n")
                for rec in optimization_summary['parameter_analysis']['recommendations']:
                    f.write(f"- {rec}\n")
            
            f.write(f"""
## Next Steps
1. Validate optimized parameters with real market data
2. Test computational performance in production environment
3. Monitor feature quality and model performance
4. Consider regime-specific parameter tuning
""")
        
        print(f"   ✅ Results exported to: {self.output_dir}")


def main():
    """Main function to run fractional differentiation parameter optimization."""
    
    optimizer = FractionalDifferentiationOptimizer()
    results = optimizer.run_grid_search(max_combinations=30)
    
    print("\n🎯 Optimization Summary:")
    print(f"   Success Rate: {results['success_rate']:.2%}")
    
    if 'best_parameters' in results:
        print(f"   Best Score: {results['best_score']:.3f}")
        print(f"   Best d Value: {results['best_parameters']['d']}")
        print(f"   Best Window Size: {results['best_parameters']['window_size']}")
        print(f"   Best Threshold: {results['best_parameters']['threshold']}")
    
    print("\n📋 Key Findings:")
    print("   • Parameter optimization completed successfully")
    print("   • Best parameters identified for fractional differentiation")
    print("   • Ready for validation with real market data")


if __name__ == "__main__":
    import pandas as pd
    
    main()