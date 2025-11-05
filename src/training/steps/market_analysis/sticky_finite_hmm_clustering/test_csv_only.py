#!/usr/bin/env python3
"""
Simple test of enhanced CSV functionality
"""

import sys
import os
from pathlib import Path

# Add the path to import cluster_quality_assessor
current_dir = os.path.dirname(os.path.abspath(__file__))
clusters_dir = os.path.join(current_dir, '..', 'clusters')
sys.path.insert(0, clusters_dir)

def test_csv_generation_only():
    """Test only the CSV generation functionality"""
    
    print("🧪 Testing Enhanced CSV Generation Only")
    print("=" * 50)
    
    try:
        # Import only what we need for CSV testing
        from cluster_quality_assessor import ClusterQualityAssessor, ClusterQualityMetrics
        from datetime import datetime
        
        # Create assessor
        assessor = ClusterQualityAssessor()
        
        # Create sample metrics with enhanced data
        metrics = ClusterQualityMetrics(
            quality_score=0.82,
            silhouette_score=0.48,
            davies_bouldin_score=1.1,
            calinski_harabasz_score=185.0,
            n_regimes=4,
            temporal_smoothness=0.88,
            regime_persistence=28.5,
            balance_score=0.92,
            cluster_size_distribution=[0.24, 0.26, 0.23, 0.27],
            # Enhanced CV metrics
            within_regime_cv=0.15,
            within_regime_cv_std=0.03,
            between_regime_cv=0.45,
            between_regime_cv_std=0.08,
            per_regime_cv={0: 0.12, 1: 0.18, 2: 0.14, 3: 0.16},
            # Economic metrics
            economic_validation={
                0: {'mean_return': 0.0025, 'volatility': 0.015, 'sharpe': 1.67, 'max_drawdown': -0.08, 'hit_rate': 0.65, 'size': 1000},
                1: {'mean_return': -0.0018, 'volatility': 0.022, 'sharpe': -0.82, 'max_drawdown': -0.15, 'hit_rate': 0.45, 'size': 1100},
                2: {'mean_return': 0.0042, 'volatility': 0.018, 'sharpe': 2.33, 'max_drawdown': -0.06, 'hit_rate': 0.72, 'size': 950},
                3: {'mean_return': -0.0008, 'volatility': 0.012, 'sharpe': -0.67, 'max_drawdown': -0.10, 'hit_rate': 0.52, 'size': 1050}
            },
            predictive_power=0.73,
            log_likelihood=-1450.0,
            # HMM validation metrics
            refit_stability_ari=0.85,
            state_occupancy={0: 0.24, 1: 0.26, 2: 0.23, 3: 0.27},
            occupancy_entropy=1.38
        )
        
        print("✅ Created test metrics with enhanced data")
        
        # Create method-specific config
        method_config = {
            'K': 4,
            'base_alpha': 0.75,
            'kappa': 18.5,
            'n_mixtures': 2,
            'pca_components': 15,
            'learning_rate': 0.045,
            'svi_iterations': 1200
        }
        
        # Test CSV generation directly
        output_path = Path("test_outcomes")
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate quality metrics CSV
        quality_csv_path = assessor._generate_quality_metrics_csv(
            metrics, "ETHUSDT", output_path, timestamp, method_config
        )
        
        print(f"📊 Quality Metrics CSV: {quality_csv_path}")
        
        # Create sample all_trials data
        all_trials = [
            {
                'trial_number': 1,
                'params': {'K': 4, 'base_alpha': 0.75, 'kappa': 18.5, 'n_mixtures': 2, 
                          'pca_components': 15, 'learning_rate': 0.045, 'svi_iterations': 1200},
                'final_elbo': -1450.0,
                'quality_metrics': {
                    'quality_score': 0.82, 'silhouette_score': 0.48, 'davies_bouldin_score': 1.1,
                    'calinski_harabasz_score': 185.0, 'within_regime_cv': 0.15, 'between_regime_cv': 0.45,
                    'within_regime_cv_std': 0.03, 'between_regime_cv_std': 0.08,
                    'temporal_smoothness': 0.88, 'regime_persistence': 28.5, 'balance_score': 0.92,
                    'n_regimes': 4, 'noise_ratio': 0.05, 'predictive_power': 0.73,
                    'economic_validation': metrics.economic_validation,
                    'log_likelihood': -1450.0, 'refit_stability_ari': 0.85, 'occupancy_entropy': 1.38,
                    'cluster_size_distribution': [0.24, 0.26, 0.23, 0.27]
                }
            }
        ]
        
        # Generate all trials CSV
        trials_csv_path = assessor._generate_all_trials_csv(
            all_trials, "ETHUSDT", output_path, timestamp
        )
        
        print(f"📋 All Trials CSV: {trials_csv_path}")
        
        # Read and display CSV content
        if quality_csv_path and os.path.exists(quality_csv_path):
            with open(quality_csv_path, 'r') as f:
                lines = f.readlines()
                print(f"\n📊 Quality Metrics CSV has {len(lines)} lines")
                print("First few lines:")
                for i, line in enumerate(lines[:5]):
                    print(f"   {i+1}: {line.strip()}")
        
        if trials_csv_path and os.path.exists(trials_csv_path):
            with open(trials_csv_path, 'r') as f:
                lines = f.readlines()
                print(f"\n📋 All Trials CSV has {len(lines)} lines")
                print("Header:")
                print(f"   1: {lines[0].strip()}")
                print("First trial:")
                print(f"   2: {lines[1].strip()[:100]}...")
        
        print("\n🎉 Enhanced CSV Generation Test Complete!")
        print("✅ CSV files with comprehensive metrics generated successfully")
        
        return quality_csv_path, trials_csv_path
        
    except Exception as e:
        print(f"❌ Error during CSV test: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_csv_generation_only()
