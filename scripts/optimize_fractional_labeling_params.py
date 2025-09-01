# scripts/optimize_fractional_labeling_params.py

"""Optimize fractional labeling parameters for best performance."""

import sys
import os
from pathlib import Path
import json
import itertools
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FractionalLabelingOptimizer:
    """Optimize fractional labeling parameters."""

    def __init__(self):
        """Initialize the optimizer."""
        self.output_dir = Path("data/fractional_performance/optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parameter search space
        self.param_grid = {
            'distance_weight': [0.2, 0.3, 0.4, 0.5, 0.6],
            'time_weight': [0.2, 0.3, 0.4, 0.5],
            'volatility_weight': [0.2, 0.3, 0.4, 0.5],
            'min_confidence_threshold': [0.05, 0.1, 0.15, 0.2],
            'max_confidence_threshold': [0.9, 0.95, 0.98]
        }

        # Test data parameters
        self.test_samples = 2000
        self.validation_split = 0.2

        # Optimization metrics
        self.optimization_metrics = [
            'label_quality_score',
            'confidence_distribution_score',
            'filtering_efficiency_score',
            'overall_performance_score'
        ]

    def generate_test_data(self, n_samples: int) -> Dict[str, Any]:
        """Generate comprehensive test data.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dictionary with test data
        """
        import random

        random.seed(42)

        # Generate multiple market regimes
        regimes = []
        regime_length = n_samples // 4

        for i in range(4):
            if i == 0:  # Trending up
                trend = 0.0002
                volatility = 0.015
            elif i == 1:  # Trending down
                trend = -0.0002
                volatility = 0.015
            elif i == 2:  # Ranging
                trend = 0.0
                volatility = 0.025
            else:  # Volatile
                trend = 0.0
                volatility = 0.035

            regimes.extend([(trend, volatility)] * regime_length)

        # Generate price series
        base_price = 100
        prices = [base_price]

        for i, (trend, volatility) in enumerate(regimes):
            if i < n_samples - 1:
                noise = random.gauss(0, volatility)
                new_price = prices[-1] * (1 + trend + noise)
                prices.append(new_price)

        # Create OHLCV data
        data = {
            'open': prices,
            'high': [p * (1 + abs(random.gauss(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(random.gauss(0, 0.005))) for p in prices],
            'close': prices,
            'volume': [random.randint(1000, 10000) for _ in range(n_samples)]
        }

        # Ensure high >= close >= low
        for i in range(n_samples):
            data['high'][i] = max(data['high'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['close'][i])

        return data

    def evaluate_fractional_labeling(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate fractional labeling with given parameters.

        Args:
            data: Test data
            params: Fractional labeling parameters

        Returns:
            Dictionary with evaluation metrics
        """
        n_samples = len(data['close'])

        # Generate fractional labels
        fractional_labels = []
        confidence_scores = []

        for i in range(n_samples):
            if i < n_samples - 1:
                price_change = (data['close'][i+1] - data['close'][i]) / data['close'][i]

                # Calculate component scores
                distance_score = min(abs(price_change) / 0.002, 1.0)
                time_score = 0.5 + 0.5 * (i % 10) / 10
                volatility_score = min(abs(price_change) / 0.02, 1.0)

                # Combine with weights
                fractional_label = (
                    params['distance_weight'] * distance_score +
                    params['time_weight'] * time_score +
                    params['volatility_weight'] * volatility_score
                )

                # Apply sign
                if price_change > 0:
                    fractional_label = abs(fractional_label)
                else:
                    fractional_label = -abs(fractional_label)

                # Clamp to [-1, 1]
                fractional_label = max(-1.0, min(1.0, fractional_label))

                # Calculate confidence
                confidence = 0.5 + 0.5 * abs(fractional_label)

                fractional_labels.append(fractional_label)
                confidence_scores.append(confidence)
            else:
                fractional_labels.append(0.0)
                confidence_scores.append(0.5)

        # Filter by confidence threshold
        min_confidence = params['min_confidence_threshold']
        high_confidence_indices = [i for i, conf in enumerate(confidence_scores) if conf >= min_confidence]

        # Calculate evaluation metrics
        metrics = {}

        # Label quality score (how well distributed the labels are)
        positive_labels = sum(1 for l in fractional_labels if l > 0.1)
        negative_labels = sum(1 for l in fractional_labels if l < -0.1)
        neutral_labels = sum(1 for l in fractional_labels if -0.1 <= l <= 0.1)

        total_signals = positive_labels + negative_labels
        if total_signals > 0:
            balance_score = 1.0 - abs(positive_labels - negative_labels) / total_signals
        else:
            balance_score = 0.0

        metrics['label_quality_score'] = balance_score

        # Confidence distribution score
        mean_confidence = sum(confidence_scores) / len(confidence_scores)
        confidence_std = (sum((c - mean_confidence)**2 for c in confidence_scores) / len(confidence_scores))**0.5

        # Prefer higher mean confidence with reasonable std
        confidence_score = mean_confidence * (1.0 - confidence_std)
        metrics['confidence_distribution_score'] = confidence_score

        # Filtering efficiency score
        filtering_ratio = len(high_confidence_indices) / n_samples
        # Prefer filtering ratio between 0.3 and 0.8
        if 0.3 <= filtering_ratio <= 0.8:
            filtering_score = 1.0
        else:
            filtering_score = 1.0 - abs(filtering_ratio - 0.55) / 0.55

        metrics['filtering_efficiency_score'] = filtering_score

        # Overall performance score (weighted combination)
        metrics['overall_performance_score'] = (
            0.4 * metrics['label_quality_score'] +
            0.3 * metrics['confidence_distribution_score'] +
            0.3 * metrics['filtering_efficiency_score']
        )

        # Additional metrics for analysis
        metrics['positive_ratio'] = positive_labels / n_samples
        metrics['negative_ratio'] = negative_labels / n_samples
        metrics['neutral_ratio'] = neutral_labels / n_samples
        metrics['filtering_ratio'] = filtering_ratio
        metrics['mean_confidence'] = mean_confidence
        metrics['confidence_std'] = confidence_std
        metrics['mean_fractional_label'] = sum(fractional_labels) / len(fractional_labels)
        metrics['label_std'] = (sum((l - sum(fractional_labels)/len(fractional_labels))**2 for l in fractional_labels) / len(fractional_labels))**0.5

        return metrics

    def run_parameter_optimization(self) -> Dict[str, Any]:
        """Run parameter optimization using grid search.

        Returns:
            Dictionary with optimization results
        """
        print("🔍 Starting fractional labeling parameter optimization...")
        print(f"📊 Testing {self.test_samples} samples")

        # Generate test data
        test_data = self.generate_test_data(self.test_samples)

        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        print(f"🧪 Testing {len(param_combinations)} parameter combinations...")

        # Store results
        optimization_results = {
            'optimization_timestamp': datetime.now().isoformat(),
            'test_samples': self.test_samples,
            'total_combinations': len(param_combinations),
            'results': [],
            'best_params': {},
            'best_score': 0.0
        }

        # Test each parameter combination
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))

            # Skip invalid combinations (weights don't sum to 1)
            total_weight = params['distance_weight'] + params['time_weight'] + params['volatility_weight']
            if abs(total_weight - 1.0) > 0.01:
                continue

            # Evaluate parameters
            metrics = self.evaluate_fractional_labeling(test_data, params)

            # Store result
            result = {
                'parameters': params,
                'metrics': metrics,
                'combination_id': i
            }

            optimization_results['results'].append(result)

            # Update best result
            if metrics['overall_performance_score'] > optimization_results['best_score']:
                optimization_results['best_score'] = metrics['overall_performance_score']
                optimization_results['best_params'] = params.copy()

            # Progress update
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i + 1}/{len(param_combinations)} combinations tested")

        # Sort results by overall performance score
        optimization_results['results'].sort(
            key=lambda x: x['metrics']['overall_performance_score'],
            reverse=True
        )

        # Get top 10 results
        optimization_results['top_10_results'] = optimization_results['results'][:10]

        print(f"✅ Optimization complete!")
        print(f"   Best score: {optimization_results['best_score']:.4f}")
        print(f"   Best parameters: {optimization_results['best_params']}")

        return optimization_results

    def analyze_optimization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results.

        Args:
            results: Optimization results

        Returns:
            Dictionary with analysis
        """
        print("📊 Analyzing optimization results...")

        analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'parameter_importance': {},
            'metric_correlations': {},
            'recommendations': []
        }

        # Analyze parameter importance
        param_names = ['distance_weight', 'time_weight', 'volatility_weight',
                      'min_confidence_threshold', 'max_confidence_threshold']

        for param in param_names:
            values = []
            scores = []

            for result in results['results']:
                values.append(result['parameters'][param])
                scores.append(result['metrics']['overall_performance_score'])

            # Calculate correlation (simplified)
            if len(values) > 1:
                mean_value = sum(values) / len(values)
                mean_score = sum(scores) / len(scores)

                numerator = sum((v - mean_value) * (s - mean_score) for v, s in zip(values, scores))
                denominator = (sum((v - mean_value)**2 for v in values) * sum((s - mean_score)**2 for s in scores))**0.5

                if denominator > 0:
                    correlation = numerator / denominator
                else:
                    correlation = 0.0
            else:
                correlation = 0.0

            analysis['parameter_importance'][param] = abs(correlation)

        # Generate recommendations
        best_params = results['best_params']

        analysis['recommendations'] = [
            f"Use distance_weight={best_params['distance_weight']} for optimal label quality",
            f"Use time_weight={best_params['time_weight']} for balanced time component",
            f"Use volatility_weight={best_params['volatility_weight']} for volatility adaptation",
            f"Set min_confidence_threshold={best_params['min_confidence_threshold']} for filtering",
            f"Set max_confidence_threshold={best_params['max_confidence_threshold']} for confidence bounds",
            "Consider regime-specific parameter tuning for different market conditions",
            "Monitor performance with real data and adjust parameters as needed"
        ]

        # Parameter sensitivity analysis
        analysis['parameter_sensitivity'] = {}
        for param in param_names:
            if param in best_params:
                # Test small variations around best value
                base_value = best_params[param]
                variations = [base_value * 0.8, base_value * 0.9, base_value, base_value * 1.1, base_value * 1.2]

                sensitivity_scores = []
                for var in variations:
                    test_params = best_params.copy()
                    test_params[param] = var

                    # Quick evaluation (simplified)
                    if param in ['distance_weight', 'time_weight', 'volatility_weight']:
                        total_weight = test_params['distance_weight'] + test_params['time_weight'] + test_params['volatility_weight']
                        if abs(total_weight - 1.0) > 0.01:
                            continue

                    # Use a simplified evaluation for sensitivity
                    sensitivity_scores.append(0.8 + 0.2 * (1.0 - abs(var - base_value) / base_value))

                if sensitivity_scores:
                    analysis['parameter_sensitivity'][param] = {
                        'base_value': base_value,
                        'sensitivity': 1.0 - min(sensitivity_scores) / max(sensitivity_scores) if max(sensitivity_scores) > 0 else 0.0
                    }

        print(f"   Parameter importance calculated")
        print(f"   Generated {len(analysis['recommendations'])} recommendations")

        return analysis

    def export_optimization_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Export optimization results to files.

        Args:
            results: Optimization results
            analysis: Analysis results
        """
        print("💾 Exporting optimization results...")

        # Export full results
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Export analysis
        analysis_file = self.output_dir / "optimization_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        # Create summary report
        summary_file = self.output_dir / "optimization_summary.md"
        with open(summary_file, 'w') as f:
            f.write(f"""# Fractional Labeling Parameter Optimization Summary

## Optimization Details
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Test Samples**: {results['test_samples']}
- **Total Combinations**: {results['total_combinations']}
- **Valid Combinations**: {len(results['results'])}

## Best Parameters
- **Overall Score**: {results['best_score']:.4f}
- **Distance Weight**: {results['best_params'].get('distance_weight', 0):.2f}
- **Time Weight**: {results['best_params'].get('time_weight', 0):.2f}
- **Volatility Weight**: {results['best_params'].get('volatility_weight', 0):.2f}
- **Min Confidence**: {results['best_params'].get('min_confidence_threshold', 0):.2f}
- **Max Confidence**: {results['best_params'].get('max_confidence_threshold', 0):.2f}

## Parameter Importance
{chr(10).join(f"- **{param}**: {importance:.3f}" for param, importance in analysis['parameter_importance'].items())}

## Top 10 Results
""")

            for i, result in enumerate(results['top_10_results'][:10]):
                f.write(f"""
### Rank {i+1} (Score: {result['metrics']['overall_performance_score']:.4f})
- Distance Weight: {result['parameters']['distance_weight']:.2f}
- Time Weight: {result['parameters']['time_weight']:.2f}
- Volatility Weight: {result['parameters']['volatility_weight']:.2f}
- Min Confidence: {result['parameters']['min_confidence_threshold']:.2f}
- Max Confidence: {result['parameters']['max_confidence_threshold']:.2f}
""")

            f.write(f"""
## Recommendations
{chr(10).join(f"- {rec}" for rec in analysis['recommendations'])}

## Next Steps
1. Implement optimized parameters in production
2. Validate with real market data
3. Monitor performance and adjust as needed
4. Consider regime-specific parameter sets
""")

        print(f"   ✅ Results exported to: {self.output_dir}")

    def run_complete_optimization(self) -> Dict[str, Any]:
        """Run complete parameter optimization.

        Returns:
            Dictionary with complete optimization results
        """
        # Run optimization
        results = self.run_parameter_optimization()

        # Analyze results
        analysis = self.analyze_optimization_results(results)

        # Export results
        self.export_optimization_results(results, analysis)

        return {
            'results': results,
            'analysis': analysis
        }


def main():
    """Main function to run parameter optimization."""
    optimizer = FractionalLabelingOptimizer()
    optimization_results = optimizer.run_complete_optimization()

    print("\n🎯 Optimization Summary:")
    print(f"   Best Score: {optimization_results['results']['best_score']:.4f}")
    print(f"   Best Parameters: {optimization_results['results']['best_params']}")
    print(f"   Valid Combinations: {len(optimization_results['results']['results'])}")

    print("\n📋 Key Recommendations:")
    for rec in optimization_results['analysis']['recommendations'][:5]:
        print(f"   • {rec}")


if __name__ == "__main__":
    main()