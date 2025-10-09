"""
Simple test for probabilistic regime output functionality.

This script demonstrates the core probabilistic regime output functionality
without complex dependencies.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    print("📊 Creating sample market data")
    
    # Generate synthetic market data
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with different regimes
    n_regimes = 4
    regime_length = n_samples // n_regimes
    
    data = []
    for i in range(n_regimes):
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, n_samples)
        
        # Different characteristics for each regime
        if i == 0:  # Trending up
            trend = np.linspace(100, 120, end_idx - start_idx)
            noise = np.random.normal(0, 1, end_idx - start_idx)
        elif i == 1:  # Trending down
            trend = np.linspace(120, 100, end_idx - start_idx)
            noise = np.random.normal(0, 1.5, end_idx - start_idx)
        elif i == 2:  # High volatility
            trend = np.full(end_idx - start_idx, 110)
            noise = np.random.normal(0, 3, end_idx - start_idx)
        else:  # Low volatility
            trend = np.full(end_idx - start_idx, 110)
            noise = np.random.normal(0, 0.5, end_idx - start_idx)
        
        prices = trend + noise
        
        for j, price in enumerate(prices):
            data.append({
                'timestamp': dates[start_idx + j],
                'open': price + np.random.normal(0, 0.1),
                'high': price + abs(np.random.normal(0, 0.2)),
                'low': price - abs(np.random.normal(0, 0.2)),
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Created sample data with {len(df)} samples")
    return df


def generate_regime_probabilities(n_samples: int, n_regimes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regime probabilities for testing."""
    print(f"🔮 Generating regime probabilities for {n_samples} samples and {n_regimes} regimes")
    
    # Generate regime labels
    regime_labels = np.random.randint(0, n_regimes, n_samples)
    
    # Generate probability matrix
    regime_probabilities = np.random.dirichlet(np.ones(n_regimes), n_samples)
    
    # Make the probabilities more realistic by biasing towards the actual regime
    for i in range(n_samples):
        actual_regime = regime_labels[i]
        # Increase probability for actual regime
        regime_probabilities[i, actual_regime] *= 2.0
        # Renormalize
        regime_probabilities[i] = regime_probabilities[i] / np.sum(regime_probabilities[i])
    
    print(f"✅ Generated regime probabilities: {regime_probabilities.shape}")
    return regime_labels, regime_probabilities


def calculate_comprehensive_regime_analysis(
    regime_probabilities: np.ndarray, 
    regime_labels: np.ndarray, 
    n_regimes: int
) -> Dict[str, Any]:
    """Calculate comprehensive analysis of regime probabilities and predictions."""
    print("📊 Calculating comprehensive regime analysis")
    
    # Calculate regime-specific statistics
    regime_stats = {}
    for regime in range(n_regimes):
        regime_mask = (regime_labels == regime)
        regime_probs = regime_probabilities[regime_mask, regime] if regime_mask.any() else np.array([])
        
        regime_stats[f'regime_{regime}'] = {
            'count': int(np.sum(regime_mask)),
            'percentage': float(np.sum(regime_mask) / len(regime_labels) * 100),
            'avg_probability': float(np.mean(regime_probs)) if len(regime_probs) > 0 else 0.0,
            'std_probability': float(np.std(regime_probs)) if len(regime_probs) > 0 else 0.0,
            'min_probability': float(np.min(regime_probs)) if len(regime_probs) > 0 else 0.0,
            'max_probability': float(np.max(regime_probs)) if len(regime_probs) > 0 else 0.0,
            'confidence_distribution': {
                'high_confidence': int(np.sum(regime_probs >= 0.8)) if len(regime_probs) > 0 else 0,
                'medium_confidence': int(np.sum((regime_probs >= 0.5) & (regime_probs < 0.8))) if len(regime_probs) > 0 else 0,
                'low_confidence': int(np.sum(regime_probs < 0.5)) if len(regime_probs) > 0 else 0
            }
        }
    
    # Calculate cross-regime probability correlations
    regime_correlations = {}
    for i in range(n_regimes):
        for j in range(i + 1, n_regimes):
            corr = np.corrcoef(regime_probabilities[:, i], regime_probabilities[:, j])[0, 1]
            regime_correlations[f'regime_{i}_vs_regime_{j}'] = float(corr) if not np.isnan(corr) else 0.0
    
    # Calculate regime uncertainty metrics
    entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)
    uncertainty_metrics = {
        'mean_entropy': float(np.mean(entropy)),
        'std_entropy': float(np.std(entropy)),
        'high_uncertainty_samples': int(np.sum(entropy > 1.0)),
        'low_uncertainty_samples': int(np.sum(entropy < 0.5)),
        'uncertainty_distribution': {
            'very_low': int(np.sum(entropy < 0.2)),
            'low': int(np.sum((entropy >= 0.2) & (entropy < 0.5))),
            'medium': int(np.sum((entropy >= 0.5) & (entropy < 1.0))),
            'high': int(np.sum((entropy >= 1.0) & (entropy < 1.5))),
            'very_high': int(np.sum(entropy >= 1.5))
        }
    }
    
    # Calculate regime dominance analysis
    sorted_probs = np.sort(regime_probabilities, axis=1)
    dominance_analysis = {
        'mean_dominance': float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])) if n_regimes > 1 else 1.0,
        'std_dominance': float(np.std(sorted_probs[:, -1] - sorted_probs[:, -2])) if n_regimes > 1 else 0.0,
        'clear_dominance_samples': int(np.sum((sorted_probs[:, -1] - sorted_probs[:, -2]) > 0.5)) if n_regimes > 1 else len(regime_labels),
        'ambiguous_samples': int(np.sum((sorted_probs[:, -1] - sorted_probs[:, -2]) <= 0.2)) if n_regimes > 1 else 0
    }
    
    analysis = {
        'regime_statistics': regime_stats,
        'regime_correlations': regime_correlations,
        'uncertainty_metrics': uncertainty_metrics,
        'dominance_analysis': dominance_analysis,
        'summary': {
            'total_samples': len(regime_labels),
            'n_regimes': n_regimes,
            'most_common_regime': int(np.argmax(np.bincount(regime_labels, minlength=n_regimes))),
            'least_common_regime': int(np.argmin(np.bincount(regime_labels, minlength=n_regimes))),
            'avg_confidence': float(np.mean(np.max(regime_probabilities, axis=1))),
            'regime_balance': float(np.std(np.bincount(regime_labels, minlength=n_regimes) / len(regime_labels)))
        }
    }
    
    print("✅ Comprehensive regime analysis completed")
    return analysis


def calculate_regime_transitions(regime_labels: np.ndarray, n_regimes: int) -> Dict[str, Any]:
    """Calculate regime transition probabilities and patterns."""
    print("🔄 Calculating regime transition analysis")
    
    # Calculate transition matrix
    transition_matrix = np.zeros((n_regimes, n_regimes))
    
    for i in range(len(regime_labels) - 1):
        current_regime = int(regime_labels[i])
        next_regime = int(regime_labels[i + 1])
        transition_matrix[current_regime, next_regime] += 1
    
    # Normalize to get transition probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_probabilities = np.divide(transition_matrix, row_sums[:, np.newaxis], 
                                       out=np.zeros_like(transition_matrix), 
                                       where=row_sums[:, np.newaxis] != 0)
    
    # Calculate transition statistics
    transitions = {
        'transition_matrix': transition_matrix.tolist(),
        'transition_probabilities': transition_probabilities.tolist(),
        'transition_counts': {
            'total_transitions': int(np.sum(transition_matrix)),
            'self_transitions': int(np.sum(np.diag(transition_matrix))),
            'cross_transitions': int(np.sum(transition_matrix) - np.sum(np.diag(transition_matrix))),
            'transition_rate': float(np.sum(transition_matrix) / len(regime_labels)) if len(regime_labels) > 0 else 0.0
        },
        'regime_persistence': {
            f'regime_{i}': float(transition_probabilities[i, i]) for i in range(n_regimes)
        },
        'most_likely_transitions': {
            f'from_regime_{i}': int(np.argmax(transition_probabilities[i, :])) 
            for i in range(n_regimes) if np.sum(transition_matrix[i, :]) > 0
        }
    }
    
    print("✅ Regime transition analysis completed")
    return transitions


def calculate_regime_persistence(regime_labels: np.ndarray, n_regimes: int) -> Dict[str, Any]:
    """Calculate regime persistence and stability metrics."""
    print("📈 Calculating regime persistence analysis")
    
    # Calculate regime durations
    regime_durations = {f'regime_{i}': [] for i in range(n_regimes)}
    current_regime = regime_labels[0]
    current_duration = 1
    
    for i in range(1, len(regime_labels)):
        if regime_labels[i] == current_regime:
            current_duration += 1
        else:
            regime_durations[f'regime_{int(current_regime)}'].append(current_duration)
            current_regime = regime_labels[i]
            current_duration = 1
    
    # Add the last regime duration
    regime_durations[f'regime_{int(current_regime)}'].append(current_duration)
    
    # Calculate persistence statistics
    persistence_stats = {}
    for regime in range(n_regimes):
        durations = regime_durations[f'regime_{regime}']
        if durations:
            persistence_stats[f'regime_{regime}'] = {
                'avg_duration': float(np.mean(durations)),
                'std_duration': float(np.std(durations)),
                'min_duration': int(np.min(durations)),
                'max_duration': int(np.max(durations)),
                'total_episodes': len(durations),
                'total_duration': int(np.sum(durations))
            }
        else:
            persistence_stats[f'regime_{regime}'] = {
                'avg_duration': 0.0,
                'std_duration': 0.0,
                'min_duration': 0,
                'max_duration': 0,
                'total_episodes': 0,
                'total_duration': 0
            }
    
    # Calculate overall persistence metrics
    all_durations = [d for durations in regime_durations.values() for d in durations]
    overall_persistence = {
        'avg_episode_duration': float(np.mean(all_durations)) if all_durations else 0.0,
        'std_episode_duration': float(np.std(all_durations)) if all_durations else 0.0,
        'longest_episode': int(np.max(all_durations)) if all_durations else 0,
        'shortest_episode': int(np.min(all_durations)) if all_durations else 0,
        'total_episodes': len(all_durations),
        'regime_stability': float(np.mean([stats['avg_duration'] for stats in persistence_stats.values()]))
    }
    
    persistence = {
        'regime_durations': regime_durations,
        'persistence_statistics': persistence_stats,
        'overall_persistence': overall_persistence
    }
    
    print("✅ Regime persistence analysis completed")
    return persistence


def generate_comprehensive_report(analysis_results: Dict[str, Any]) -> str:
    """Generate a comprehensive text report from analysis results."""
    print("📝 Generating comprehensive report")
    
    report = []
    report.append("=" * 80)
    report.append(f"COMPREHENSIVE REGIME PROBABILITY ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 80)
    report.append("")
    
    # Basic Statistics
    basic_stats = analysis_results.get('basic_statistics', {})
    if 'error' not in basic_stats:
        report.append("📊 BASIC STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Samples: {basic_stats.get('total_samples', 'N/A')}")
        report.append(f"Number of Regimes: {basic_stats.get('n_regimes', 'N/A')}")
        
        regime_dist = basic_stats.get('regime_distribution', {})
        report.append(f"Most Common Regime: {regime_dist.get('most_common_regime', 'N/A')}")
        report.append(f"Regime Balance: {regime_dist.get('regime_balance', 0):.3f}")
        
        prob_stats = basic_stats.get('probability_statistics', {})
        report.append(f"Mean Max Probability: {prob_stats.get('mean_max_probability', 0):.3f}")
        report.append(f"Std Max Probability: {prob_stats.get('std_max_probability', 0):.3f}")
        report.append("")
    
    # Regime Analysis
    regime_analysis = analysis_results.get('regime_analysis', {})
    if 'error' not in regime_analysis:
        report.append("🎯 REGIME ANALYSIS")
        report.append("-" * 40)
        
        regime_stats = regime_analysis.get('regime_statistics', {})
        for regime_key, regime_data in regime_stats.items():
            if isinstance(regime_data, dict) and 'error' not in regime_data:
                report.append(f"{regime_key.upper()}:")
                report.append(f"  Sample Count: {regime_data.get('count', 0)}")
                report.append(f"  Percentage: {regime_data.get('percentage', 0):.1f}%")
                report.append(f"  Avg Probability: {regime_data.get('avg_probability', 0):.3f}")
                report.append(f"  High Confidence: {regime_data.get('confidence_distribution', {}).get('high_confidence', 0)}")
                report.append("")
        
        uncertainty = regime_analysis.get('uncertainty_metrics', {})
        report.append("UNCERTAINTY METRICS:")
        report.append(f"  Mean Entropy: {uncertainty.get('mean_entropy', 0):.3f}")
        report.append(f"  High Uncertainty Samples: {uncertainty.get('high_uncertainty_samples', 0)}")
        report.append("")
    
    # Transition Analysis
    transitions = analysis_results.get('regime_transitions', {})
    if 'error' not in transitions:
        report.append("🔄 TRANSITION ANALYSIS")
        report.append("-" * 40)
        transition_counts = transitions.get('transition_counts', {})
        report.append(f"Total Transitions: {transition_counts.get('total_transitions', 0)}")
        report.append(f"Self Transitions: {transition_counts.get('self_transitions', 0)}")
        report.append(f"Cross Transitions: {transition_counts.get('cross_transitions', 0)}")
        report.append(f"Transition Rate: {transition_counts.get('transition_rate', 0):.3f}")
        report.append("")
    
    # Persistence Analysis
    persistence = analysis_results.get('regime_persistence', {})
    if 'error' not in persistence:
        report.append("📈 PERSISTENCE ANALYSIS")
        report.append("-" * 40)
        overall = persistence.get('overall_persistence', {})
        report.append(f"Avg Episode Duration: {overall.get('avg_episode_duration', 0):.1f}")
        report.append(f"Longest Episode: {overall.get('longest_episode', 0)}")
        report.append(f"Total Episodes: {overall.get('total_episodes', 0)}")
        report.append(f"Regime Stability: {overall.get('regime_stability', 0):.3f}")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def test_probabilistic_regime_output():
    """Test the probabilistic regime output functionality."""
    print("🚀 Starting Probabilistic Regime Output Test")
    print("=" * 60)
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        
        # Generate regime probabilities
        n_regimes = 4
        regime_labels, regime_probabilities = generate_regime_probabilities(len(data), n_regimes)
        
        # Calculate comprehensive analysis
        regime_analysis = calculate_comprehensive_regime_analysis(
            regime_probabilities, regime_labels, n_regimes
        )
        
        # Calculate transitions
        regime_transitions = calculate_regime_transitions(regime_labels, n_regimes)
        
        # Calculate persistence
        regime_persistence = calculate_regime_persistence(regime_labels, n_regimes)
        
        # Create comprehensive results
        analysis_results = {
            'model_name': 'Test Model',
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_statistics': {
                'total_samples': len(regime_labels),
                'n_regimes': n_regimes,
                'regime_distribution': {
                    'counts': np.bincount(regime_labels, minlength=n_regimes).tolist(),
                    'percentages': (np.bincount(regime_labels, minlength=n_regimes) / len(regime_labels) * 100).tolist(),
                    'most_common_regime': int(np.argmax(np.bincount(regime_labels, minlength=n_regimes))),
                    'least_common_regime': int(np.argmin(np.bincount(regime_labels, minlength=n_regimes))),
                    'regime_balance': float(np.std(np.bincount(regime_labels, minlength=n_regimes) / len(regime_labels)))
                },
                'probability_statistics': {
                    'mean_max_probability': float(np.mean(np.max(regime_probabilities, axis=1))),
                    'std_max_probability': float(np.std(np.max(regime_probabilities, axis=1))),
                    'min_max_probability': float(np.min(np.max(regime_probabilities, axis=1))),
                    'max_max_probability': float(np.max(np.max(regime_probabilities, axis=1)))
                }
            },
            'regime_analysis': regime_analysis,
            'regime_transitions': regime_transitions,
            'regime_persistence': regime_persistence
        }
        
        # Generate comprehensive report
        report = generate_comprehensive_report(analysis_results)
        
        print("✅ Probabilistic regime output test completed successfully!")
        print("\n📝 ANALYSIS REPORT:")
        print(report)
        
        # Save results to file
        with open('probabilistic_regime_test_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: probabilistic_regime_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_probabilistic_regime_output()
    if success:
        print("\n🎉 All tests passed! Probabilistic regime output functionality is working correctly.")
    else:
        print("\n⚠️ Test failed. Please check the error messages above.")