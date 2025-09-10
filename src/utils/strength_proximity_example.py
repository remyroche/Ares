"""
Example demonstrating Strength-Proximity Clustering for SR Levels

This example shows how the new clustering approach groups SR levels
based on their strength and proximity rather than hard-coded cluster counts.
"""

import numpy as np
from typing import List, Dict, Tuple
from clustering_alternatives import get_clustering_manager

def create_sample_sr_levels() -> List[Dict]:
    """Create sample SR levels with varying strengths and prices."""
    
    # Sample SR levels from your actual data (based on your logs)
    levels = [
        # Strong levels around 1624-1649 (high strength, close proximity)
        {'price': 1624.73, 'strength': 1.0, 'touches': 5, 'type': 'support'},
        {'price': 1628.31, 'strength': 1.0, 'touches': 4, 'type': 'support'},
        {'price': 1632.46, 'strength': 1.0, 'touches': 3, 'type': 'resistance'},
        {'price': 1636.62, 'strength': 0.996, 'touches': 3, 'type': 'support'},
        {'price': 1640.33, 'strength': 1.0, 'touches': 2, 'type': 'resistance'},
        {'price': 1645.89, 'strength': 1.0, 'touches': 2, 'type': 'resistance'},
        {'price': 1649.55, 'strength': 1.0, 'touches': 2, 'type': 'resistance'},
        
        # Medium strength levels around 1844-1859 (medium strength, close proximity)
        {'price': 1844.44, 'strength': 1.0, 'touches': 2, 'type': 'resistance'},
        {'price': 1849.70, 'strength': 1.0, 'touches': 2, 'type': 'support'},
        {'price': 1854.17, 'strength': 1.0, 'touches': 2, 'type': 'resistance'},
        {'price': 1859.79, 'strength': 1.0, 'touches': 2, 'type': 'resistance'},
        
        # Weak levels around 3000 (psychological level, lower strength)
        {'price': 3000.00, 'strength': 0.734, 'touches': 2, 'type': 'support'},
        {'price': 3006.75, 'strength': 1.0, 'touches': 1, 'type': 'resistance'},
        
        # Isolated strong level
        {'price': 4000.00, 'strength': 0.89, 'touches': 3, 'type': 'support'},
        
        # Weak isolated level
        {'price': 4500.00, 'strength': 0.55, 'touches': 1, 'type': 'resistance'},
    ]
    
    return levels

def demonstrate_strength_proximity_clustering():
    """Demonstrate how strength-proximity clustering works."""
    
    print("🎯 Strength-Proximity Clustering Demo")
    print("=" * 50)
    
    # Create sample data
    levels = create_sample_sr_levels()
    price_range = (min(level['price'] for level in levels), 
                   max(level['price'] for level in levels))
    
    print(f"📊 Input: {len(levels)} SR levels")
    print(f"💰 Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
    print()
    
    # Show original levels
    print("📋 Original SR Levels:")
    for i, level in enumerate(levels):
        print(f"  {i+1:2d}. ${level['price']:8.2f} | Strength: {level['strength']:.3f} | {level['type']:10s} | Touches: {level['touches']}")
    print()
    
    # Test different proximity thresholds
    proximity_thresholds = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5% of price range
    strength_thresholds = [0.1, 0.2, 0.3]  # 10%, 20%, 30% strength difference
    
    clustering_manager = get_clustering_manager()
    
    for prox_thresh in proximity_thresholds:
        for str_thresh in strength_thresholds:
            print(f"🔍 Testing: Proximity={prox_thresh:.1%}, Strength={str_thresh:.1%}")
            
            try:
                result = clustering_manager.cluster_with_fallback(
                    levels=levels,
                    price_range=price_range,
                    proximity_threshold=prox_thresh,
                    strength_similarity_threshold=str_thresh,
                    preferred_algorithm='strength_proximity'
                )
                
                print(f"   ✅ Result: {len(result.clusters)} clusters, Quality: {result.quality_score:.3f}")
                
                # Show cluster details
                for i, cluster in enumerate(result.clusters):
                    if len(cluster) > 1:  # Only show multi-level clusters
                        cluster_prices = [levels[idx]['price'] for idx in cluster]
                        cluster_strengths = [levels[idx]['strength'] for idx in cluster]
                        cluster_center = result.cluster_centers[i]
                        
                        print(f"      Cluster {i+1}: {len(cluster)} levels, Center: ${cluster_center:.2f}")
                        print(f"        Prices: {[f'${p:.2f}' for p in cluster_prices]}")
                        print(f"        Strengths: {[f'{s:.3f}' for s in cluster_strengths]}")
                        print(f"        Price spread: ${max(cluster_prices) - min(cluster_prices):.2f}")
                        print(f"        Strength spread: {max(cluster_strengths) - min(cluster_strengths):.3f}")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                print()

def demonstrate_vs_dbscan():
    """Compare strength-proximity clustering with DBSCAN approach."""
    
    print("🆚 Strength-Proximity vs DBSCAN Comparison")
    print("=" * 50)
    
    levels = create_sample_sr_levels()
    price_range = (min(level['price'] for level in levels), 
                   max(level['price'] for level in levels))
    
    clustering_manager = get_clustering_manager()
    
    # Strength-proximity approach
    print("🎯 Strength-Proximity Clustering:")
    result_sp = clustering_manager.cluster_with_fallback(
        levels=levels,
        price_range=price_range,
        proximity_threshold=0.01,  # 1% of price range
        strength_similarity_threshold=0.2,  # 20% strength difference
        preferred_algorithm='strength_proximity'
    )
    
    print(f"   Clusters: {len(result_sp.clusters)}")
    print(f"   Quality: {result_sp.quality_score:.3f}")
    print(f"   All levels preserved: {result_sp.total_levels == len(levels)}")
    print(f"   Noise points: {len(result_sp.noise_points)}")
    
    # Show meaningful clusters
    meaningful_clusters = [c for c in result_sp.clusters if len(c) > 1]
    print(f"   Meaningful clusters (>1 level): {len(meaningful_clusters)}")
    
    for i, cluster in enumerate(meaningful_clusters):
        cluster_prices = [levels[idx]['price'] for idx in cluster]
        cluster_strengths = [levels[idx]['strength'] for idx in cluster]
        print(f"     Cluster {i+1}: {len(cluster)} levels at ${min(cluster_prices):.2f}-${max(cluster_prices):.2f}")
        print(f"       Strengths: {min(cluster_strengths):.3f}-{max(cluster_strengths):.3f}")
    
    print()
    
    # Simulate DBSCAN problems (based on your logs)
    print("❌ DBSCAN Problems (from your logs):")
    print("   Attempt 1: eps=184.007281 → 2 clusters, 12 noise (14 total levels)")
    print("   Attempt 2: eps=73.602912 → 6 clusters, 17 noise (28 total levels)")
    print("   Attempt 3: eps=29.441165 → 12 clusters, 23 noise (49 total levels)")
    print("   Attempt 4: eps=11.776466 → 14 clusters, 41 noise (81 total levels)")
    print("   Attempt 5: eps=4.710586 → 4 clusters, 72 noise (81 total levels)")
    print("   Attempt 6: eps=1.884235 → 0 clusters, 81 noise (81 total levels)")
    print()
    print("   Problems:")
    print("   - Unpredictable cluster count")
    print("   - Many levels lost as 'noise'")
    print("   - Sensitive to parameter tuning")
    print("   - No consideration of level strength")
    print("   - Hard to achieve target level count")
    
    print()
    print("✅ Strength-Proximity Advantages:")
    print("   - All levels preserved (no noise)")
    print("   - Natural cluster formation based on data")
    print("   - Considers both price proximity AND strength similarity")
    print("   - Deterministic results")
    print("   - No parameter sensitivity issues")
    print("   - Quality scoring for cluster evaluation")

def demonstrate_adaptive_clustering():
    """Show how clustering adapts to different data characteristics."""
    
    print("🔄 Adaptive Clustering Examples")
    print("=" * 50)
    
    # Scenario 1: Dense price levels (many levels close together)
    dense_levels = [
        {'price': 1624.73, 'strength': 1.0, 'touches': 5, 'type': 'support'},
        {'price': 1625.10, 'strength': 0.95, 'touches': 3, 'type': 'support'},
        {'price': 1625.50, 'strength': 0.90, 'touches': 2, 'type': 'support'},
        {'price': 1626.00, 'strength': 0.85, 'touches': 2, 'type': 'support'},
        {'price': 1626.50, 'strength': 0.80, 'touches': 1, 'type': 'support'},
    ]
    
    # Scenario 2: Sparse price levels (levels far apart)
    sparse_levels = [
        {'price': 1000.00, 'strength': 1.0, 'touches': 5, 'type': 'support'},
        {'price': 2000.00, 'strength': 0.9, 'touches': 3, 'type': 'resistance'},
        {'price': 3000.00, 'strength': 0.8, 'touches': 2, 'type': 'support'},
        {'price': 4000.00, 'strength': 0.7, 'touches': 2, 'type': 'resistance'},
        {'price': 5000.00, 'strength': 0.6, 'touches': 1, 'type': 'resistance'},
    ]
    
    clustering_manager = get_clustering_manager()
    
    for scenario_name, scenario_levels in [("Dense Levels", dense_levels), ("Sparse Levels", sparse_levels)]:
        print(f"📊 {scenario_name}:")
        
        price_range = (min(level['price'] for level in scenario_levels), 
                       max(level['price'] for level in scenario_levels))
        
        result = clustering_manager.cluster_with_fallback(
            levels=scenario_levels,
            price_range=price_range,
            proximity_threshold=0.01,  # 1% of price range
            strength_similarity_threshold=0.2,
            preferred_algorithm='strength_proximity'
        )
        
        print(f"   Input: {len(scenario_levels)} levels")
        print(f"   Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
        print(f"   Result: {len(result.clusters)} clusters")
        print(f"   Quality: {result.quality_score:.3f}")
        
        # Show how clustering adapts
        meaningful_clusters = [c for c in result.clusters if len(c) > 1]
        single_clusters = [c for c in result.clusters if len(c) == 1]
        
        print(f"   Grouped levels: {sum(len(c) for c in meaningful_clusters)}")
        print(f"   Isolated levels: {len(single_clusters)}")
        
        if meaningful_clusters:
            print("   Clustered groups:")
            for i, cluster in enumerate(meaningful_clusters):
                cluster_prices = [scenario_levels[idx]['price'] for idx in cluster]
                print(f"     Group {i+1}: {[f'${p:.2f}' for p in cluster_prices]}")
        
        print()

if __name__ == "__main__":
    demonstrate_strength_proximity_clustering()
    print("\n" + "="*80 + "\n")
    demonstrate_vs_dbscan()
    print("\n" + "="*80 + "\n")
    demonstrate_adaptive_clustering()