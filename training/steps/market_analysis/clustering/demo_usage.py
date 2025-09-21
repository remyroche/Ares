#!/usr/bin/env python3
"""
Demonstration of how to use the regime clustering pipeline.

This script shows the complete workflow from HMM discovery results to ML-ready outputs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_pipeline_usage():
    """Demonstrate the complete pipeline usage."""
    
    print("=" * 80)
    print("REGIME CLUSTERING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. OVERVIEW")
    print("-" * 40)
    print("This pipeline takes HMM discovery results and creates balanced clusters for ML training.")
    print("Key features:")
    print("  • Consolidates many small regimes into 20-ish balanced clusters")
    print("  • Ensures 90-95% coverage of market states")
    print("  • Creates ML-ready training datasets")
    print("  • Provides cluster profiles and market regime interpretations")
    
    print("\n2. COMMAND LINE USAGE")
    print("-" * 40)
    print("Basic usage:")
    print("  python run_clustering_pipeline.py \\")
    print("    --hmm-results /path/to/hmm_results.json \\")
    print("    --symbol BTCUSDT \\")
    print("    --timeframe 1h")
    
    print("\nFrom outcomes directory:")
    print("  python run_clustering_pipeline.py \\")
    print("    --outcomes-dir /path/to/outcomes \\")
    print("    --symbol ETHUSDT \\")
    print("    --timeframe 15m")
    
    print("\nCustom parameters:")
    print("  python run_clustering_pipeline.py \\")
    print("    --hmm-results /path/to/hmm_results.json \\")
    print("    --symbol BTCUSDT \\")
    print("    --timeframe 1h \\")
    print("    --target-clusters 25 \\")
    print("    --min-cluster-size 0.02 \\")
    print("    --max-cluster-size 0.10 \\")
    print("    --coverage-target 0.98")
    
    print("\n3. PYTHON MODULE USAGE")
    print("-" * 40)
    print("""
from training.steps.market_analysis.clustering import create_clustering_pipeline

# Create pipeline with custom configuration
pipeline = create_clustering_pipeline(
    target_clusters=20,
    min_cluster_size_pct=0.03,
    max_cluster_size_pct=0.08,
    coverage_target=0.95
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    hmm_results_file="path/to/hmm_results.json",
    symbol="BTCUSDT",
    timeframe="1h"
)

# Access results
consolidation_result = results['consolidation_result']
training_dataset = results['training_dataset']
cluster_profiles = results['cluster_profiles']
""")
    
    print("\n4. CONFIGURATION OPTIONS")
    print("-" * 40)
    print("ConsolidationConfig parameters:")
    print("  • target_clusters: Number of target clusters (default: 20)")
    print("  • min_cluster_size_pct: Minimum cluster size (default: 0.03 = 3%)")
    print("  • max_cluster_size_pct: Maximum cluster size (default: 0.08 = 8%)")
    print("  • coverage_target: Target coverage by top clusters (default: 0.95 = 95%)")
    print("  • merge_similarity_threshold: Threshold for merging regimes (default: 0.90)")
    print("  • assignment_similarity_threshold: Threshold for assigning regimes (default: 0.70)")
    
    print("\n5. OUTPUT FILES")
    print("-" * 40)
    print("ML Training Files:")
    print("  • training_dataset_{symbol}_{timeframe}_{timestamp}.csv")
    print("  • cluster_labels_{symbol}_{timeframe}_{timestamp}.npy")
    print("  • cluster_metadata_{symbol}_{timeframe}_{timestamp}.json")
    
    print("\nAnalysis Files:")
    print("  • cluster_profiles_{symbol}_{timeframe}_{timestamp}.json")
    print("  • ml_outputs_summary_{symbol}_{timeframe}_{timestamp}.json")
    print("  • pipeline_results_{symbol}_{timeframe}_{timestamp}.json")
    
    print("\n6. EXPECTED RESULTS")
    print("-" * 40)
    print("Typical results for BTCUSDT 1h data:")
    print("  • Original HMM regimes: 4,000+ small regimes")
    print("  • Final clusters: 20 balanced clusters")
    print("  • Coverage: 100% of all regimes accounted for")
    print("  • Top 20 coverage: 90-95% of market states")
    print("  • Training samples: 50,000+ samples")
    print("  • Trainable clusters: 15-20 clusters")
    
    print("\n7. MARKET REGIME INTERPRETATION")
    print("-" * 40)
    print("Clusters are automatically interpreted as market regimes:")
    print("  • High_Volatility_Bull: High volatility + positive momentum")
    print("  • Low_Volatility_Sideways: Low volatility + neutral momentum")
    print("  • Moderate_Volatility_Bear: Moderate volatility + negative momentum")
    print("  • etc.")
    
    print("\n8. ML TRAINING INTEGRATION")
    print("-" * 40)
    print("""
# Load training data
import pandas as pd
import numpy as np

features = pd.read_csv('training_dataset_BTCUSDT_1h_20241201_120000.csv')
labels = np.load('cluster_labels_BTCUSDT_1h_20241201_120000.npy')

# Train ML model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")
""")
    
    print("\n9. QUALITY METRICS")
    print("-" * 40)
    print("The pipeline provides several quality metrics:")
    print("  • Coverage Percentage: % of total samples covered by all clusters")
    print("  • Top Clusters Coverage: % covered by top N clusters")
    print("  • Balance Score: How well cluster sizes are balanced")
    print("  • Trainability: Number of clusters suitable for ML training")
    
    print("\n10. TROUBLESHOOTING")
    print("-" * 40)
    print("Common issues and solutions:")
    print("  • Low coverage: Increase coverage_target or adjust cluster size constraints")
    print("  • Unbalanced clusters: Adjust min/max cluster size percentages")
    print("  • Too many small clusters: Increase merge_similarity_threshold")
    print("  • Validation warnings: Check HMM discovery results quality")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    print("\nTo get started:")
    print("1. Ensure you have HMM discovery results")
    print("2. Install required dependencies (numpy, pandas, scikit-learn, scipy)")
    print("3. Run: python run_clustering_pipeline.py --help")
    print("4. Or use the Python module as shown above")

def create_sample_config():
    """Create a sample configuration file."""
    
    config = {
        "clustering_config": {
            "target_clusters": 20,
            "min_cluster_size_pct": 0.03,
            "max_cluster_size_pct": 0.08,
            "coverage_target": 0.95,
            "merge_similarity_threshold": 0.90,
            "assignment_similarity_threshold": 0.70
        },
        "output_config": {
            "output_dir": "training/steps/market_analysis/clustering",
            "save_detailed_results": True,
            "validate_coverage": True,
            "validate_balance": True
        },
        "ml_config": {
            "min_samples_required": 1000,
            "feature_names": ["momentum", "volatility", "volume", "trend"],
            "target_models": ["RandomForestClassifier", "GradientBoostingClassifier"]
        }
    }
    
    config_file = Path("clustering_config_sample.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSample configuration saved to: {config_file}")
    print("You can modify this file and use it as a reference for your clustering setup.")

if __name__ == "__main__":
    demo_pipeline_usage()
    create_sample_config()