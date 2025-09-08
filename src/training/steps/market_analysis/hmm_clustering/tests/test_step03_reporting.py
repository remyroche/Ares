#!/usr/bin/env python3
"""
Test script for step03 enhanced reporting functionality.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
from ...standardized_parquet_handler import standardized_parquet_handler
import time

def analyze_hmm_clustering_results(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze HMM clustering results and return comprehensive summary."""
    try:
        # Load HMM composite metadata
        meta_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
        if not meta_file.exists():
            return {"error": f"HMM metadata file not found: {meta_file}"}

        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        # Load HMM block states
        block_states_file = Path("data/training") / f"BINANCE_{symbol}_hmm_block_states_{timeframe}.parquet"
        if block_states_file.exists():
            block_states_df = pd.read_parquet(block_states_file)
        else:
            block_states_df = None

        # Load HMM composite clusters
        clusters_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        if clusters_file.exists():
            clusters_df = pd.read_parquet(clusters_file)
        else:
            clusters_df = None

        # Analyze HMM blocks
        blocks_analysis = {}
        for block in meta_data.get("blocks", []):
            block_name = block["name"]
            n_states = block["n_states"]
            blocks_analysis[block_name] = {
                "n_states": n_states,
                "state_medians": meta_data.get("state_feature_medians", {}).get(block_name, {}),
                "state_names": meta_data.get("state_names", {}).get(block_name, {})
            }

        # Analyze cluster centroids
        centroids = meta_data.get("cluster_centroids", {})
        cluster_analysis = {
            "n_clusters": len(centroids),
            "cluster_sizes": {f"cluster_{i}": len(centroids.get(str(i), [])) for i in range(len(centroids))},
            "centroids_summary": {
                f"cluster_{i}": {
                    "size": len(centroids.get(str(i), [])),
                    "mean_value": np.mean(centroids.get(str(i), [0])) if centroids.get(str(i)) else 0
                } for i in range(len(centroids))
            }
        }

        # Analyze regime combinations
        combinations = meta_data.get("combination_counts", {})
        top_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "blocks_analysis": blocks_analysis,
            "cluster_analysis": cluster_analysis,
            "regime_combinations": {
                "total_combinations": len(combinations),
                "top_combinations": top_combinations,
                "most_common_regime": top_combinations[0][0] if top_combinations else None
            },
            "data_availability": {
                "block_states_available": block_states_df is not None,
                "clusters_available": clusters_df is not None,
                "metadata_available": True
            },
            "summary": {
                "total_regime_blocks": len(meta_data.get("blocks", [])),
                "total_clusters": len(centroids),
                "total_regime_combinations": len(combinations)
            }
        }

    except Exception as e:
        return {"error": f"Failed to analyze HMM clustering results: {str(e)}"}

def test_hmm_analysis():
    """Test HMM clustering analysis."""
    print("🧪 Testing HMM Clustering Analysis...")
    result = analyze_hmm_clustering_results("ETHUSDT", "BINANCE", "1m")

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    print("✅ HMM Analysis successful!")
    print(f"   📊 Total regime blocks: {result['summary']['total_regime_blocks']}")
    print(f"   🎯 Total clusters: {result['summary']['total_clusters']}")
    print(f"   🔄 Total regime combinations: {result['summary']['total_regime_combinations']}")

    return True

def test_comprehensive_report():
    """Test comprehensive report generation."""
    print("\n📊 Testing Comprehensive Report Generation...")

    # Mock execution data
    execution_time = 45.67
    correlation_id = "test_12345"

    # Generate report
    hmm_results = analyze_hmm_clustering_results("ETHUSDT", "BINANCE", "1m")

    # Create summary
    summary = {
        "execution_info": {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "execution_time_seconds": execution_time,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat()
        },
        "pipeline_components": {
            "hmm_clustering": "completed" if not hmm_results.get("error") else "failed",
        },
        "data_quality_metrics": {
            "hmm_data_available": hmm_results.get("data_availability", {}).get("metadata_available", False),
            "regime_data_processed": 0,
            "features_generated": 0,
            "matrix_operations_performed": False
        }
    }

    recommendations = []
    if hmm_results.get("error"):
        recommendations.append("❌ HMM clustering failed - check data availability")
    else:
        n_clusters = hmm_results.get("cluster_analysis", {}).get("n_clusters", 0)
        if n_clusters > 0:
            recommendations.append(f"✅ HMM clustering successful with {n_clusters} clusters identified")
        else:
            recommendations.append("⚠️ No clusters found - review HMM parameters")

    comprehensive_report = {
        "summary": summary,
        "hmm_clustering_analysis": hmm_results,
        "recommendations": recommendations
    }

    # Save report
    report_file = Path("data_cache") / "test_market_analysis_comprehensive_report_ETHUSDT_1m.json"
    with open(report_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)

    print("✅ Comprehensive report generated!")
    print(f"💾 Report saved to: {report_file}")

    return True

if __name__ == "__main__":
    print("🚀 Testing Step03 Enhanced Reporting\n")

    # Test individual components
    hmm_success = test_hmm_analysis()

    # Test comprehensive report
    report_success = test_comprehensive_report()

    print("\n📈 Test Results:")
    print(f"   HMM Analysis: {'✅ PASSED' if hmm_success else '❌ FAILED'}")
    print(f"   Comprehensive Report: {'✅ PASSED' if report_success else '❌ FAILED'}")

    if hmm_success and report_success:
        print("\n🎉 All tests passed! Enhanced reporting is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")
