from src.utils.tprint import tprint

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
    
    tprint("🎯 Strength-Proximity Clustering Demo")
    tprint("=" * 50)
    
    # Create sample data
    levels = create_sample_sr_levels()
    price_range = (min(level['price'] for level in levels), 
                   max(level['price'] for level in levels))
    
    tprint(f"📊 Input: {len(levels)} SR levels")
    tprint(f"💰 Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
    tprint()
    
    # Show original levels
    tprint("📋 Original SR Levels:")
    for i, level in enumerate(levels):
        tprint(f"  {i+1:2d}. ${level['price']:8.2f} | Strength: {level['strength']:.3f} | {level['type']:10s} | Touches: {level['touches']}")
    tprint()
    
    # Test different proximity thresholds
    proximity_thresholds = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5% of price range
    strength_thresholds = [0.1, 0.2, 0.3]  # 10%, 20%, 30% strength difference
    
    clustering_manager = get_clustering_manager()
    
    for prox_thresh in proximity_thresholds:
        for str_thresh in strength_thresholds:
            tprint(f"🔍 Testing: Proximity={prox_thresh:.1%}, Strength={str_thresh:.1%}")
            
            try:
                result = clustering_manager.cluster_with_fallback(
                    levels=levels,
                    price_range=price_range,
                    proximity_threshold=prox_thresh,
                    strength_similarity_threshold=str_thresh,
                    preferred_algorithm='strength_proximity'
                )
                
                tprint(f"   ✅ Result: {len(result.clusters)} clusters, Quality: {result.quality_score:.3f}")
                
                # Show cluster details
                for i, cluster in enumerate(result.clusters):
                    if len(cluster) > 1:  # Only show multi-level clusters
                        cluster_prices = [levels[idx]['price'] for idx in cluster]
                        cluster_strengths = [levels[idx]['strength'] for idx in cluster]
                        cluster_center = result.cluster_centers[i]
                        
                        tprint(f"      Cluster {i+1}: {len(cluster)} levels, Center: ${cluster_center:.2f}")
                        tprint(f"        Prices: {[f'${p:.2f}' for p in cluster_prices]}")
                        tprint(f"        Strengths: {[f'{s:.3f}' for s in cluster_strengths]}")
                        tprint(f"        Price spread: ${max(cluster_prices) - min(cluster_prices):.2f}")
                        tprint(f"        Strength spread: {max(cluster_strengths) - min(cluster_strengths):.3f}")
                
                tprint()
                
            except Exception as e:
                tprint(f"   ❌ Failed: {e}")
                tprint()

def demonstrate_vs_dbscan():
    """Compare strength-proximity clustering with DBSCAN approach."""
    
    tprint("🆚 Strength-Proximity vs DBSCAN Comparison")
    tprint("=" * 50)
    
    levels = create_sample_sr_levels()
    price_range = (min(level['price'] for level in levels), 
                   max(level['price'] for level in levels))
    
    clustering_manager = get_clustering_manager()
    
    # Strength-proximity approach
    tprint("🎯 Strength-Proximity Clustering:")
    result_sp = clustering_manager.cluster_with_fallback(
        levels=levels,
        price_range=price_range,
        proximity_threshold=0.01,  # 1% of price range
        strength_similarity_threshold=0.2,  # 20% strength difference
        preferred_algorithm='strength_proximity'
    )
    
    tprint(f"   Clusters: {len(result_sp.clusters)}")
    tprint(f"   Quality: {result_sp.quality_score:.3f}")
    tprint(f"   All levels preserved: {result_sp.total_levels == len(levels)}")
    tprint(f"   Noise points: {len(result_sp.noise_points)}")
    
    # Show meaningful clusters
    meaningful_clusters = [c for c in result_sp.clusters if len(c) > 1]
    tprint(f"   Meaningful clusters (>1 level): {len(meaningful_clusters)}")
    
    for i, cluster in enumerate(meaningful_clusters):
        cluster_prices = [levels[idx]['price'] for idx in cluster]
        cluster_strengths = [levels[idx]['strength'] for idx in cluster]
        tprint(f"     Cluster {i+1}: {len(cluster)} levels at ${min(cluster_prices):.2f}-${max(cluster_prices):.2f}")
        tprint(f"       Strengths: {min(cluster_strengths):.3f}-{max(cluster_strengths):.3f}")
    
    tprint()
    
    # Simulate DBSCAN problems (based on your logs)
    tprint("❌ DBSCAN Problems (from your logs):")
    tprint("   Attempt 1: eps=184.007281 → 2 clusters, 12 noise (14 total levels)")
    tprint("   Attempt 2: eps=73.602912 → 6 clusters, 17 noise (28 total levels)")
    tprint("   Attempt 3: eps=29.441165 → 12 clusters, 23 noise (49 total levels)")
    tprint("   Attempt 4: eps=11.776466 → 14 clusters, 41 noise (81 total levels)")
    tprint("   Attempt 5: eps=4.710586 → 4 clusters, 72 noise (81 total levels)")
    tprint("   Attempt 6: eps=1.884235 → 0 clusters, 81 noise (81 total levels)")
    tprint()
    tprint("   Problems:")
    tprint("   - Unpredictable cluster count")
    tprint("   - Many levels lost as 'noise'")
    tprint("   - Sensitive to parameter tuning")
    tprint("   - No consideration of level strength")
    tprint("   - Hard to achieve target level count")
    
    tprint()
    tprint("✅ Strength-Proximity Advantages:")
    tprint("   - All levels preserved (no noise)")
    tprint("   - Natural cluster formation based on data")
    tprint("   - Considers both price proximity AND strength similarity")
    tprint("   - Deterministic results")
    tprint("   - No parameter sensitivity issues")
    tprint("   - Quality scoring for cluster evaluation")

def demonstrate_adaptive_clustering():
    """Show how clustering adapts to different data characteristics."""
    
    tprint("🔄 Adaptive Clustering Examples")
    tprint("=" * 50)
    
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
        tprint(f"📊 {scenario_name}:")
        
        price_range = (min(level['price'] for level in scenario_levels), 
                       max(level['price'] for level in scenario_levels))
        
        result = clustering_manager.cluster_with_fallback(
            levels=scenario_levels,
            price_range=price_range,
            proximity_threshold=0.01,  # 1% of price range
            strength_similarity_threshold=0.2,
            preferred_algorithm='strength_proximity'
        )
        
        tprint(f"   Input: {len(scenario_levels)} levels")
        tprint(f"   Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
        tprint(f"   Result: {len(result.clusters)} clusters")
        tprint(f"   Quality: {result.quality_score:.3f}")
        
        # Show how clustering adapts
        meaningful_clusters = [c for c in result.clusters if len(c) > 1]
        single_clusters = [c for c in result.clusters if len(c) == 1]
        
        tprint(f"   Grouped levels: {sum(len(c) for c in meaningful_clusters)}")
        tprint(f"   Isolated levels: {len(single_clusters)}")
        
        if meaningful_clusters:
            tprint("   Clustered groups:")
            for i, cluster in enumerate(meaningful_clusters):
                cluster_prices = [scenario_levels[idx]['price'] for idx in cluster]
                tprint(f"     Group {i+1}: {[f'${p:.2f}' for p in cluster_prices]}")
        
        tprint()

if __name__ == "__main__":
    demonstrate_strength_proximity_clustering()
    tprint("\n" + "="*80 + "\n")
    demonstrate_vs_dbscan()
    tprint("\n" + "="*80 + "\n")
    demonstrate_adaptive_clustering()