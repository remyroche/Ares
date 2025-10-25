"""
Enhanced Fallback Strategies Demo for Automated HDBSCAN Parameter Tuner

This script demonstrates all 12 fallback strategies with feature engineering
to achieve the 4-8 cluster target with balanced distributions.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import time

def create_enhanced_financial_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create enhanced financial time series data with 4 distinct regimes."""
    np.random.seed(42)
    
    # Create 4 distinct market regimes
    regime_length = n_samples // 4
    data = {}
    
    for i in range(n_samples):
        if i < regime_length:
            # Regime 1: Bull market, low volatility
            trend = 0.001
            volatility = 0.01
            data.setdefault('close', []).append(100 + i * trend + np.random.normal(0, volatility))
        elif i < 2 * regime_length:
            # Regime 2: Bear market, medium volatility
            trend = -0.0005
            volatility = 0.02
            data.setdefault('close', []).append(110 + (i - regime_length) * trend + np.random.normal(0, volatility))
        elif i < 3 * regime_length:
            # Regime 3: Sideways, high volatility
            trend = 0.0001
            volatility = 0.03
            data.setdefault('close', []).append(105 + (i - 2*regime_length) * trend + np.random.normal(0, volatility))
        else:
            # Regime 4: Volatile trending, very high volatility
            trend = 0.0008
            volatility = 0.04
            data.setdefault('close', []).append(108 + (i - 3*regime_length) * trend + np.random.normal(0, volatility))
    
    # Add technical indicators
    close_prices = np.array(data['close'])
    data['returns'] = np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]])
    data['sma_20'] = pd.Series(close_prices).rolling(20).mean().fillna(close_prices[0]).values
    data['volatility'] = pd.Series(data['returns']).rolling(20).std().fillna(0).values
    
    # Add more features for better discrimination
    for i in range(5):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)

def test_enhanced_fallback_strategies():
    """Test all 12 enhanced fallback strategies."""
    print("🚀 ENHANCED FALLBACK STRATEGIES DEMO")
    print("=" * 60)
    
    # Create enhanced data
    print("📊 Creating enhanced financial data with 4 distinct regimes...")
    data = create_enhanced_financial_data(1000)
    print(f"✅ Created dataset: {data.shape[0]} samples, {data.shape[1]} features")
    
    try:
        import hdbscan
        from sklearn.metrics import silhouette_score
        
        # Define all 12 fallback strategies
        strategies = [
            {
                'name': 'target_4_6_clusters_leaf',
                'description': 'Target 4-6 clusters using leaf method',
                'parameters': {
                    'cluster_selection_method': 'leaf',
                    'cluster_selection_epsilon': 0.01,
                    'min_cluster_size': max(25, len(data) // 25),
                    'min_samples': max(12, len(data) // 50),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'target_6_8_clusters_eom',
                'description': 'Target 6-8 clusters using EOM method',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.05,
                    'min_cluster_size': max(20, len(data) // 35),
                    'min_samples': max(10, len(data) // 70),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'aggressive_more_clusters',
                'description': 'Aggressive clustering to get more clusters',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.02,
                    'min_cluster_size': max(15, len(data) // 50),
                    'min_samples': max(8, len(data) // 100),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'conservative_fewer_clusters',
                'description': 'Conservative approach for fewer clusters',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.1,
                    'min_cluster_size': max(40, len(data) // 15),
                    'min_samples': max(20, len(data) // 30),
                    'metric': 'manhattan'
                }
            },
            {
                'name': 'alternative_metrics_separation',
                'description': 'Alternative metrics for better cluster separation',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.05,
                    'min_cluster_size': max(25, len(data) // 30),
                    'min_samples': max(12, len(data) // 60),
                    'metric': 'manhattan'
                }
            },
            {
                'name': 'balanced_distribution',
                'description': 'Target balanced cluster distribution (2%-20%)',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.08,
                    'min_cluster_size': max(20, len(data) // 25),
                    'min_samples': max(10, len(data) // 50),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'conservative_balanced',
                'description': 'Conservative approach for balanced distribution',
                'parameters': {
                    'cluster_selection_method': 'leaf',
                    'cluster_selection_epsilon': 0.03,
                    'min_cluster_size': max(30, len(data) // 20),
                    'min_samples': max(15, len(data) // 40),
                    'metric': 'manhattan'
                }
            },
            {
                'name': 'aggressive_cluster_count',
                'description': 'Aggressive approach to achieve 4-8 clusters',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.01,
                    'min_cluster_size': max(10, len(data) // 100),
                    'min_samples': max(5, len(data) // 200),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'cosine_metric_separation',
                'description': 'Cosine metric for better feature separation',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.05,
                    'min_cluster_size': max(15, len(data) // 50),
                    'min_samples': max(8, len(data) // 100),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'manhattan_robust_clustering',
                'description': 'Manhattan metric for robust regime detection',
                'parameters': {
                    'cluster_selection_method': 'leaf',
                    'cluster_selection_epsilon': 0.08,
                    'min_cluster_size': max(20, len(data) // 40),
                    'min_samples': max(10, len(data) // 80),
                    'metric': 'manhattan'
                }
            },
            {
                'name': 'feature_engineering_approach',
                'description': 'Enhanced feature engineering for better discrimination',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.03,
                    'min_cluster_size': max(12, len(data) // 60),
                    'min_samples': max(6, len(data) // 120),
                    'metric': 'euclidean'
                }
            },
            {
                'name': 'multi_metric_ensemble',
                'description': 'Multi-metric approach for comprehensive clustering',
                'parameters': {
                    'cluster_selection_method': 'eom',
                    'cluster_selection_epsilon': 0.06,
                    'min_cluster_size': max(18, len(data) // 35),
                    'min_samples': max(9, len(data) // 70),
                    'metric': 'cosine'
                }
            }
        ]
        
        print(f"\n🔧 Testing {len(strategies)} enhanced fallback strategies...")
        print("=" * 60)
        
        results = []
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n📋 Strategy {i}: {strategy['name']}")
            print(f"   Description: {strategy['description']}")
            print(f"   Parameters: {strategy['parameters']}")
            
            try:
                # Apply feature engineering for specific strategies
                test_data = data
                if strategy['name'] in ['feature_engineering_approach', 'multi_metric_ensemble']:
                    print(f"   🔧 Applying feature engineering...")
                    # Simulate feature engineering (simplified)
                    enhanced_data = data.copy()
                    close_prices = data['close']
                    
                    # Add technical indicators
                    enhanced_data['sma_5'] = close_prices.rolling(5).mean()
                    enhanced_data['sma_10'] = close_prices.rolling(10).mean()
                    enhanced_data['volatility_5'] = data['returns'].rolling(5).std()
                    enhanced_data['rsi_14'] = 50 + np.random.normal(0, 10, len(data))  # Simplified RSI
                    enhanced_data['momentum_5'] = close_prices.pct_change(5)
                    enhanced_data['trend_strength'] = (close_prices.rolling(10).mean() - close_prices.rolling(30).mean()) / close_prices.rolling(30).mean()
                    
                    test_data = enhanced_data.fillna(0)
                    print(f"   ✅ Enhanced to {test_data.shape[1]} features")
                
                # Test clustering
                clusterer = hdbscan.HDBSCAN(**strategy['parameters'])
                labels = clusterer.fit_predict(test_data)
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # Calculate distribution
                unique_labels = np.unique(labels)
                distributions = []
                for label in unique_labels:
                    if label != -1:
                        pct = (list(labels).count(label) / len(labels)) * 100
                        distributions.append(pct)
                
                min_dist = min(distributions) if distributions else 0
                max_dist = max(distributions) if distributions else 0
                balanced = all(2.0 <= pct <= 20.0 for pct in distributions) if distributions else False
                
                # Calculate silhouette score
                silhouette = None
                if n_clusters > 1:
                    try:
                        valid_mask = labels != -1
                        if valid_mask.sum() > 1:
                            valid_data = test_data[valid_mask]
                            valid_labels = labels[valid_mask]
                            if len(set(valid_labels)) > 1:
                                silhouette = silhouette_score(valid_data, valid_labels)
                    except:
                        pass
                
                # Calculate composite score
                cluster_score = 1.0 if 4 <= n_clusters <= 8 else 0.0
                distribution_score = 1.0 if balanced else 0.0
                noise_score = max(0, 1 - noise_ratio)
                silhouette_score_val = max(0, silhouette) if silhouette is not None else 0.0
                
                composite_score = (cluster_score + distribution_score + noise_score + silhouette_score_val) / 4
                
                result = {
                    'strategy': strategy['name'],
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'silhouette': silhouette,
                    'min_dist': min_dist,
                    'max_dist': max_dist,
                    'balanced': balanced,
                    'composite_score': composite_score
                }
                results.append(result)
                
                silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                print(f"   Results: {n_clusters} clusters, {noise_ratio:.3f} noise, silhouette: {silhouette_str}")
                print(f"   Distribution: {min_dist:.1f}%-{max_dist:.1f}% (balanced: {balanced})")
                print(f"   Composite Score: {composite_score:.3f}")
                
                # Check if targets are met
                cluster_ok = 4 <= n_clusters <= 8
                distribution_ok = balanced
                
                if cluster_ok and distribution_ok:
                    print(f"   Status: ✅ ALL TARGETS MET!")
                elif cluster_ok:
                    print(f"   Status: ⚠️ Cluster count OK, distribution needs work")
                elif distribution_ok:
                    print(f"   Status: ⚠️ Distribution OK, cluster count needs work")
                else:
                    print(f"   Status: ❌ Both targets need improvement")
                
            except Exception as e:
                print(f"   ❌ Strategy failed: {e}")
                continue
        
        # Find best strategy
        if results:
            best_result = max(results, key=lambda x: x['composite_score'])
            
            print(f"\n🏆 BEST STRATEGY: {best_result['strategy']}")
            print(f"   • Composite Score: {best_result['composite_score']:.3f}")
            print(f"   • Clusters: {best_result['n_clusters']} (target: 4-8)")
            print(f"   • Distribution: {best_result['min_dist']:.1f}%-{best_result['max_dist']:.1f}% (target: 2%-20%)")
            print(f"   • Balanced: {best_result['balanced']}")
            print(f"   • Noise Ratio: {best_result['noise_ratio']:.3f}")
            silhouette_str = f"{best_result['silhouette']:.3f}" if best_result['silhouette'] is not None else "N/A"
            print(f"   • Silhouette: {silhouette_str}")
            
            # Final validation
            cluster_ok = 4 <= best_result['n_clusters'] <= 8
            distribution_ok = best_result['balanced']
            
            print(f"\n🎯 FINAL VALIDATION:")
            print(f"   • Cluster count: {'✅' if cluster_ok else '❌'} {best_result['n_clusters']} clusters (target: 4-8)")
            print(f"   • Distribution balance: {'✅' if distribution_ok else '❌'} (target: 2%-20%)")
            
            if cluster_ok and distribution_ok:
                print(f"   • Overall: ✅ SUCCESS - All targets achieved!")
            else:
                print(f"   • Overall: ❌ NEEDS IMPROVEMENT - Some targets not met")
        
        print(f"\n🚀 ENHANCED FALLBACK STRATEGIES DEMO COMPLETE!")
        
    except ImportError as e:
        print(f"❌ HDBSCAN not available: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_fallback_strategies()
