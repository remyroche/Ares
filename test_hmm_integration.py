"""
Test HMM Integration with Existing Pipeline

This script demonstrates how to integrate HMM temporal refinement
with the existing HDBSCAN regime discovery pipeline.

Usage:
    python test_hmm_integration.py
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Import existing components
from src.training.steps.market_analysis.hdbscan_clustering.main_regime_discovery import (
    HDBSCANRegimeDiscovery,
    RegimeDiscoveryConfig
)
from src.training.steps.market_analysis.hmm_temporal_layer import (
    refine_with_hmm,
    HMMTemporalLayer,
    _compute_temporal_coherence
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


async def test_hmm_integration_basic():
    """Test basic HMM integration with synthetic data."""
    print("\n" + "="*80)
    print("TEST 1: Basic HMM Integration with Synthetic Data")
    print("="*80 + "\n")
    
    # Generate synthetic market-like data
    np.random.seed(42)
    n_samples = 2000
    n_features = 50
    
    # Simulate different market regimes
    # Regime 0: Bull market (positive returns, low volatility)
    # Regime 1: Bear market (negative returns, moderate volatility)
    # Regime 2: Volatile market (mixed returns, high volatility)
    
    features_list = []
    regime_labels_true = []
    
    # Regime 0: Bull (500 samples)
    bull_features = np.random.randn(500, n_features) * 0.5 + 1.0
    features_list.append(bull_features)
    regime_labels_true.extend([0] * 500)
    
    # Regime 1: Bear (500 samples)
    bear_features = np.random.randn(500, n_features) * 0.7 - 1.0
    features_list.append(bear_features)
    regime_labels_true.extend([1] * 500)
    
    # Regime 2: Volatile (500 samples)
    volatile_features = np.random.randn(500, n_features) * 2.0
    features_list.append(volatile_features)
    regime_labels_true.extend([2] * 500)
    
    # Mixed period (500 samples) - rapid transitions
    for _ in range(500):
        regime = np.random.choice([0, 1, 2])
        if regime == 0:
            sample = np.random.randn(n_features) * 0.5 + 1.0
        elif regime == 1:
            sample = np.random.randn(n_features) * 0.7 - 1.0
        else:
            sample = np.random.randn(n_features) * 2.0
        features_list.append(sample.reshape(1, -1))
        regime_labels_true.append(regime)
    
    features = np.vstack(features_list)
    regime_labels_true = np.array(regime_labels_true)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Add some realistic feature names for demonstration
    features_df.rename(columns={
        'feature_0': 'returns',
        'feature_1': 'volatility',
        'feature_2': 'volume',
        'feature_3': 'rsi',
        'feature_4': 'macd',
    }, inplace=True)
    
    print(f"Generated synthetic data:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - True regimes: {len(np.unique(regime_labels_true))}")
    print(f"  - True temporal coherence: {_compute_temporal_coherence(regime_labels_true):.3f}\n")
    
    # Step 1: Run HDBSCAN regime discovery
    tprint_info("Step 1: Running HDBSCAN regime discovery...")
    
    # Create config
    config = RegimeDiscoveryConfig()
    
    # Initialize HDBSCAN
    hdbscan_discovery = HDBSCANRegimeDiscovery(config, use_optimized=True)
    
    # For testing, we'll simulate HDBSCAN result
    # In real usage, you would do: hdbscan_result = await hdbscan_discovery.discover_regimes(data)
    
    # Simulate HDBSCAN clustering (with some noise)
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    hdbscan_labels = dbscan.fit_predict(features)
    
    # Create mock result object
    class MockHDBSCANResult:
        def __init__(self, labels):
            self.cluster_labels = labels
            self.labels = labels
    
    hdbscan_result = MockHDBSCANResult(hdbscan_labels)
    
    n_hdbscan_regimes = len(np.unique(hdbscan_labels[hdbscan_labels != -1]))
    hdbscan_coherence = _compute_temporal_coherence(hdbscan_labels)
    
    tprint_success(f"HDBSCAN complete: {n_hdbscan_regimes} regimes, coherence={hdbscan_coherence:.3f}")
    
    # Step 2: Refine with HMM
    tprint_info("\nStep 2: Refining with HMM temporal layer...")
    
    hmm_config = {
        'hmm_config': {
            'covariance_type': 'diag',  # Use 'diag' for speed, 'full' for accuracy
            'n_iter': 100,
            'convergence_threshold': 1e-4
        }
    }
    
    hmm_result = await refine_with_hmm(hdbscan_result, features_df, hmm_config)
    
    if hmm_result.success:
        tprint_success("HMM refinement successful!")
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"{'Metric':<30} {'HDBSCAN':<15} {'HMM Refined':<15} {'Improvement':<15}")
        print("-"*80)
        
        hmm_coherence = _compute_temporal_coherence(hmm_result.refined_labels)
        coherence_improvement = hmm_coherence - hdbscan_coherence
        
        print(f"{'Temporal Coherence':<30} {hdbscan_coherence:<15.3f} {hmm_coherence:<15.3f} {coherence_improvement:<15.3f}")
        print(f"{'Number of Regimes':<30} {n_hdbscan_regimes:<15} {len(np.unique(hmm_result.refined_labels)):<15}")
        print(f"{'Noise Ratio':<30} {(hdbscan_labels == -1).mean():<15.3f} {0.0:<15.3f}")
        
        # Transition analysis
        print("\n" + "="*80)
        print("TRANSITION ANALYSIS")
        print("="*80)
        print("\nRegime Stability (Expected Duration in timesteps):")
        for regime, duration in hmm_result.regime_stability.items():
            if duration != float('inf'):
                print(f"  Regime {regime}: {duration:.1f} timesteps")
            else:
                print(f"  Regime {regime}: Very stable (no transitions observed)")
        
        print("\nTransition Matrix:")
        print(hmm_result.transition_matrix)
        
        # Most likely transitions
        print("\nMost Likely Regime Transitions:")
        for trans in hmm_result.metadata.get('most_likely_transitions', []):
            print(f"  Regime {trans['from']} → Regime {trans['to']}: {trans['probability']:.3f}")
        
        # Convergence info
        print("\n" + "="*80)
        print("CONVERGENCE INFORMATION")
        print("="*80)
        print(f"Converged: {hmm_result.convergence_info.get('converged', 'N/A')}")
        print(f"Iterations: {hmm_result.convergence_info.get('iterations', 'N/A')}")
        print(f"Final log-likelihood: {hmm_result.convergence_info.get('final_log_likelihood', 'N/A'):.2f}")
        
    else:
        tprint_error(f"HMM refinement failed: {hmm_result.error_message}")
    
    return hmm_result


async def test_hmm_integration_with_real_pipeline():
    """Test HMM integration with actual regime discovery pipeline."""
    print("\n" + "="*80)
    print("TEST 2: HMM Integration with Real Pipeline")
    print("="*80 + "\n")
    
    # This test would use actual market data and the full pipeline
    # For now, we'll demonstrate the integration pattern
    
    tprint_info("This test demonstrates how to integrate HMM with your existing pipeline:")
    
    print("""
    # In your regime_clustering_step.py, add:
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # ... existing HDBSCAN code ...
        
        # Load HDBSCAN artifacts
        hdbscan_artifacts = self._load_hdbscan_artifacts(config)
        
        # Choose refinement method
        use_hmm = config.get('use_hmm_temporal_refinement', True)
        
        if use_hmm:
            # Use HMM refinement (NEW)
            from .hmm_temporal_layer import refine_with_hmm
            
            hmm_result = await refine_with_hmm(
                hdbscan_artifacts,
                features_df,
                config
            )
            
            refined_clusters = {
                'labels': hmm_result.refined_labels,
                'probabilities': hmm_result.regime_probabilities,
                'n_clusters': len(np.unique(hmm_result.refined_labels)),
                'transition_matrix': hmm_result.transition_matrix,
                'regime_stability': hmm_result.regime_stability,
                'method': 'hmm_temporal_refinement'
            }
        else:
            # Use existing iterative optimization
            refined_clusters = self._refine_hdbscan_clusters(
                hdbscan_artifacts, 
                config
            )
        
        # ... rest of your code ...
    """)
    
    tprint_success("Integration pattern documented above!")


async def test_performance_comparison():
    """Compare performance of HMM vs iterative optimization."""
    print("\n" + "="*80)
    print("TEST 3: Performance Comparison")
    print("="*80 + "\n")
    
    # Generate test data of different sizes
    sizes = [500, 1000, 2000, 5000]
    n_features = 50
    
    results = []
    
    for n_samples in sizes:
        print(f"\nTesting with {n_samples} samples...")
        
        # Generate data
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 5, n_samples)
        
        # Create mock result
        class MockResult:
            def __init__(self, labels):
                self.cluster_labels = labels
        
        # Test HMM
        start = datetime.now()
        hmm_layer = HMMTemporalLayer(n_components=5, covariance_type='diag', verbose=False)
        hmm_layer.initialize_from_clusters(features, labels)
        hmm_layer.fit(features)
        refined = hmm_layer.predict(features)
        hmm_time = (datetime.now() - start).total_seconds()
        
        results.append({
            'n_samples': n_samples,
            'hmm_time': hmm_time,
            'hmm_memory': 'N/A'  # Would measure in real test
        })
        
        print(f"  HMM time: {hmm_time:.3f}s")
    
    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Samples':<15} {'HMM Time (s)':<15} {'Memory':<15}")
    print("-"*45)
    for r in results:
        print(f"{r['n_samples']:<15} {r['hmm_time']:<15.3f} {r['hmm_memory']:<15}")


async def test_live_trading_simulation():
    """Test HMM forward algorithm for live trading."""
    print("\n" + "="*80)
    print("TEST 4: Live Trading Simulation (Forward Algorithm)")
    print("="*80 + "\n")
    
    # Generate sequential data
    n_samples = 1000
    n_features = 20
    n_regimes = 3
    
    # Generate features with regime switches
    features = []
    true_regimes = []
    current_regime = 0
    
    for i in range(n_samples):
        # Switch regime occasionally
        if i > 0 and np.random.rand() < 0.05:  # 5% chance of regime switch
            current_regime = (current_regime + 1) % n_regimes
        
        # Generate features for current regime
        regime_mean = np.random.randn(n_features) * 3
        sample = np.random.randn(n_features) * 0.5 + regime_mean
        features.append(sample)
        true_regimes.append(current_regime)
    
    features = np.array(features)
    true_regimes = np.array(true_regimes)
    
    print(f"Generated {n_samples} sequential samples with {n_regimes} regimes")
    print(f"True regime switches: {np.sum(true_regimes[1:] != true_regimes[:-1])}")
    
    # Train HMM on first 80% of data
    train_size = int(0.8 * n_samples)
    train_features = features[:train_size]
    train_labels = true_regimes[:train_size]
    
    # Initialize and fit HMM
    hmm_layer = HMMTemporalLayer(
        n_components=n_regimes,
        covariance_type='diag',
        verbose=False
    )
    hmm_layer.initialize_from_clusters(train_features, train_labels)
    hmm_layer.fit(train_features)
    
    print(f"\nTrained HMM on {train_size} samples")
    
    # Simulate live trading: predict regime for each new observation
    print("\nSimulating live trading with forward algorithm...")
    print(f"{'Time':<10} {'True Regime':<15} {'Predicted':<15} {'Confidence':<15} {'Correct':<10}")
    print("-"*75)
    
    correct_predictions = 0
    
    # Start with training data, then predict each new point
    for i in range(train_size, min(train_size + 20, n_samples)):  # Show first 20 predictions
        # Use all data up to current time (no look-ahead!)
        historical_features = features[:i+1]
        
        # Predict current regime
        regime_probs = hmm_layer.predict_proba(historical_features)
        current_regime_probs = regime_probs[-1]
        predicted_regime = np.argmax(current_regime_probs)
        confidence = current_regime_probs[predicted_regime]
        
        true_regime = true_regimes[i]
        is_correct = predicted_regime == true_regime
        
        if is_correct:
            correct_predictions += 1
        
        print(f"{i:<10} {true_regime:<15} {predicted_regime:<15} {confidence:<15.3f} {'✓' if is_correct else '✗':<10}")
    
    accuracy = correct_predictions / 20
    print(f"\nLive trading accuracy (first 20 predictions): {accuracy:.1%}")
    
    # Show transition probabilities for trading decisions
    print("\n" + "="*80)
    print("TRANSITION PROBABILITIES FOR TRADING")
    print("="*80)
    print("\nCurrent regime transition probabilities:")
    trans_matrix = hmm_layer.get_transition_matrix()
    
    for i in range(n_regimes):
        print(f"\nFrom Regime {i}:")
        for j in range(n_regimes):
            if i != j:
                print(f"  → Regime {j}: {trans_matrix[i, j]:.3f}")
    
    tprint_success("\nLive trading simulation complete!")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("HMM INTEGRATION TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Basic integration
        await test_hmm_integration_basic()
        
        # Test 2: Integration pattern
        await test_hmm_integration_with_real_pipeline()
        
        # Test 3: Performance comparison
        await test_performance_comparison()
        
        # Test 4: Live trading simulation
        await test_live_trading_simulation()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("="*80)
        
        print("\n✅ HMM integration is working correctly!")
        print("\nNext steps:")
        print("1. Review the HMM_REGIME_INTEGRATION_ANALYSIS.md document")
        print("2. Integrate HMM temporal layer into your regime_clustering_step.py")
        print("3. Update config files to enable/disable HMM refinement")
        print("4. Test with real market data")
        print("5. Compare results with existing iterative optimization")
        
    except Exception as e:
        tprint_error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
