#!/usr/bin/env python3
"""
HMM Cluster Relevance Testing Script

This script provides comprehensive testing of HMM clusters generated in Step 3
to validate their relevance before proceeding to ML model training.

Usage:
    python test_hmm_cluster_relevance.py --data_path path/to/cluster_data.parquet
    python test_hmm_cluster_relevance.py --config_path path/to/config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HMMClusterValidator:
    """Comprehensive HMM cluster validation and testing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results = {}

    def test_cluster_predictive_power(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Test the predictive power of clusters for future price movements."""
        results = {}

        if "composite_cluster_id" not in cluster_data.columns:
            print("⚠️ No composite_cluster_id found in data")
            return {"error": "Missing composite_cluster_id column"}

        # 1. Regime Transition Predictability
        regimes = cluster_data["composite_cluster_id"].values
        transition_counts = {}

        for i in range(len(regimes) - 1):
            current = regimes[i]
            next_regime = regimes[i + 1]

            if current not in transition_counts:
                transition_counts[current] = {}
            if next_regime not in transition_counts[current]:
                transition_counts[current][next_regime] = 0
            transition_counts[current][next_regime] += 1

        # Calculate predictability scores
        predictability_scores = {}
        for regime, transitions in transition_counts.items():
            total_transitions = sum(transitions.values())
            if total_transitions > 0:
                probabilities = [count / total_transitions for count in transitions.values()]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                max_entropy = np.log2(len(transitions))
                predictability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                predictability_scores[regime] = predictability

        results["transition_predictability"] = predictability_scores
        results["avg_predictability"] = np.mean(list(predictability_scores.values())) if predictability_scores else 0
        results["transition_matrix"] = transition_counts

        return results

    def test_cluster_stability(self, cluster_data: pd.DataFrame, window_size: int = 1000) -> Dict[str, Any]:
        """Test cluster stability over rolling windows."""
        results = {}

        if "composite_cluster_id" not in cluster_data.columns:
            return {"error": "Missing composite_cluster_id column"}

        # Calculate cluster consistency over rolling windows
        stability_scores = []
        window_data_list = []

        for i in range(0, len(cluster_data) - window_size, window_size // 2):
            window_data = cluster_data.iloc[i:i+window_size]
            window_clusters = window_data["composite_cluster_id"].values

            # Calculate cluster distribution consistency
            unique_clusters = np.unique(window_clusters)
            cluster_counts = np.bincount(window_clusters, minlength=max(unique_clusters) + 1)
            cluster_proportions = cluster_counts / len(window_clusters)

            # Calculate entropy (lower = more stable)
            valid_proportions = cluster_proportions[cluster_proportions > 0]
            if len(valid_proportions) > 1:
                entropy = -sum(p * np.log2(p) for p in valid_proportions)
                max_entropy = np.log2(len(valid_proportions))
                stability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                stability_scores.append(stability)
                window_data_list.append({
                    "start_idx": i,
                    "end_idx": i + window_size,
                    "stability": stability,
                    "cluster_distribution": dict(zip(range(len(cluster_counts)), cluster_counts))
                })

        results["stability_scores"] = stability_scores
        results["avg_stability"] = np.mean(stability_scores) if stability_scores else 0
        results["stability_std"] = np.std(stability_scores) if stability_scores else 0
        results["window_analysis"] = window_data_list

        return results

    def test_market_condition_differentiation(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Test if clusters effectively differentiate market conditions."""
        results = {}

        if "composite_cluster_id" not in cluster_data.columns:
            return {"error": "Missing composite_cluster_id column"}

        # Calculate average characteristics for each cluster
        cluster_characteristics = {}

        for cluster_id in cluster_data["composite_cluster_id"].unique():
            cluster_mask = cluster_data["composite_cluster_id"] == cluster_id
            cluster_subset = cluster_data[cluster_mask]

            characteristics = {
                "avg_volatility": cluster_subset["volatility_20"].mean() if "volatility_20" in cluster_subset.columns else 0,
                "avg_momentum": cluster_subset["price_momentum_10"].mean() if "price_momentum_10" in cluster_subset.columns else 0,
                "avg_volume": cluster_subset["volume_ratio_10"].mean() if "volume_ratio_10" in cluster_subset.columns else 1,
                "avg_returns": cluster_subset["returns"].mean() if "returns" in cluster_subset.columns else 0,
                "size": len(cluster_subset),
                "percentage": len(cluster_subset) / len(cluster_data) * 100
            }
            cluster_characteristics[cluster_id] = characteristics

        # Calculate differentiation scores
        differentiation_scores = {}
        for cluster_id, char in cluster_characteristics.items():
            # Calculate how different this cluster is from others
            differences = []
            for other_id, other_char in cluster_characteristics.items():
                if other_id != cluster_id:
                    diff = abs(char["avg_volatility"] - other_char["avg_volatility"]) + \
                           abs(char["avg_momentum"] - other_char["avg_momentum"]) + \
                           abs(char["avg_volume"] - other_char["avg_volume"])
                    differences.append(diff)

            differentiation_scores[cluster_id] = np.mean(differences) if differences else 0

        results["cluster_characteristics"] = cluster_characteristics
        results["differentiation_scores"] = differentiation_scores
        results["avg_differentiation"] = np.mean(list(differentiation_scores.values())) if differentiation_scores else 0

        return results

    def test_return_predictability(self, cluster_data: pd.DataFrame, forward_periods: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """Test if clusters can predict future returns."""
        results = {}

        if "composite_cluster_id" not in cluster_data.columns or "close" not in cluster_data.columns:
            return {"error": "Missing required columns: composite_cluster_id or close"}

        for period in forward_periods:
            # Calculate forward returns
            cluster_data[f"forward_return_{period}"] = cluster_data["close"].pct_change(period).shift(-period)

            # Calculate average returns by cluster
            cluster_returns = {}
            for cluster_id in cluster_data["composite_cluster_id"].unique():
                cluster_mask = cluster_data["composite_cluster_id"] == cluster_id
                cluster_subset = cluster_data[cluster_mask]

                # Remove NaN values
                valid_returns = cluster_subset[f"forward_return_{period}"].dropna()
                if len(valid_returns) > 0:
                    cluster_returns[cluster_id] = {
                        "mean_return": valid_returns.mean(),
                        "std_return": valid_returns.std(),
                        "sharpe_ratio": valid_returns.mean() / valid_returns.std() if valid_returns.std() > 0 else 0,
                        "sample_size": len(valid_returns),
                        "positive_return_pct": (valid_returns > 0).mean()
                    }

            # Calculate return predictability score
            if cluster_returns:
                return_spreads = []
                for cluster_id, returns in cluster_returns.items():
                    for other_id, other_returns in cluster_returns.items():
                        if cluster_id != other_id:
                            spread = abs(returns["mean_return"] - other_returns["mean_return"])
                            return_spreads.append(spread)

                predictability_score = np.mean(return_spreads) if return_spreads else 0
            else:
                predictability_score = 0

            results[f"period_{period}"] = {
                "cluster_returns": cluster_returns,
                "predictability_score": predictability_score
            }

        return results

    def calculate_cluster_quality_metrics(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate traditional cluster quality metrics."""
        results = {}

        if "composite_cluster_id" not in cluster_data.columns:
            return {"error": "Missing composite_cluster_id column"}

        # Prepare features for clustering metrics
        feature_columns = [col for col in cluster_data.columns
                          if col not in ["composite_cluster_id", "timestamp", "close", "high", "low", "open", "volume"]]

        if len(feature_columns) == 0:
            return {"error": "No feature columns found for quality metrics"}

        # Use first few features for dimensionality reduction
        features_for_metrics = cluster_data[feature_columns[:min(10, len(feature_columns))]].fillna(0)
        cluster_labels = cluster_data["composite_cluster_id"].values

        try:
            # Silhouette score
            results["silhouette_score"] = silhouette_score(features_for_metrics, cluster_labels)
        except Exception:
            results["silhouette_score"] = 0.0

        try:
            # Calinski-Harabasz score
            results["calinski_harabasz_score"] = calinski_harabasz_score(features_for_metrics, cluster_labels)
        except Exception:
            results["calinski_harabasz_score"] = 0.0

        try:
            # Davies-Bouldin score
            results["davies_bouldin_score"] = davies_bouldin_score(features_for_metrics, cluster_labels)
        except Exception:
            results["davies_bouldin_score"] = float('inf')

        # Cluster size distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        results["cluster_sizes"] = dict(zip(unique_labels, counts))
        results["min_cluster_size"] = np.min(counts)
        results["max_cluster_size"] = np.max(counts)
        results["mean_cluster_size"] = np.mean(counts)
        results["std_cluster_size"] = np.std(counts)
        results["cluster_balance"] = results["std_cluster_size"] / results["mean_cluster_size"] if results["mean_cluster_size"] > 0 else 0

        return results

    def comprehensive_validation(self, cluster_data: pd.DataFrame,
                               quality_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Comprehensive validation of HMM clusters."""
        if quality_thresholds is None:
            quality_thresholds = {
                "min_silhouette": 0.3,
                "min_predictability": 0.4,
                "min_stability": 0.5,
                "min_differentiation": 0.1,
                "min_return_predictability": 0.001
            }

        results = {
            "quality_metrics": {},
            "predictive_power": {},
            "stability": {},
            "market_differentiation": {},
            "return_predictability": {},
            "overall_score": 0,
            "recommendations": []
        }

        print("🔍 Running comprehensive cluster validation...")

        # 1. Quality Metrics
        print("  📊 Calculating quality metrics...")
        results["quality_metrics"] = self.calculate_cluster_quality_metrics(cluster_data)

        # 2. Predictive Power
        print("  🔮 Testing predictive power...")
        results["predictive_power"] = self.test_cluster_predictive_power(cluster_data)

        # 3. Stability
        print("  ⏱️ Testing stability...")
        results["stability"] = self.test_cluster_stability(cluster_data)

        # 4. Market Differentiation
        print("  📈 Testing market differentiation...")
        results["market_differentiation"] = self.test_market_condition_differentiation(cluster_data)

        # 5. Return Predictability
        print("  💰 Testing return predictability...")
        results["return_predictability"] = self.test_return_predictability(cluster_data)

        # 6. Calculate Overall Score
        scores = []

        # Silhouette score
        silhouette = results["quality_metrics"].get("silhouette_score", 0)
        if silhouette > quality_thresholds["min_silhouette"]:
            scores.append(1.0)
        else:
            scores.append(silhouette / quality_thresholds["min_silhouette"])

        # Predictive power score
        predictability = results["predictive_power"].get("avg_predictability", 0)
        if predictability > quality_thresholds["min_predictability"]:
            scores.append(1.0)
        else:
            scores.append(predictability / quality_thresholds["min_predictability"])

        # Stability score
        stability = results["stability"].get("avg_stability", 0)
        if stability > quality_thresholds["min_stability"]:
            scores.append(1.0)
        else:
            scores.append(stability / quality_thresholds["min_stability"])

        # Differentiation score
        differentiation = results["market_differentiation"].get("avg_differentiation", 0)
        if differentiation > quality_thresholds["min_differentiation"]:
            scores.append(1.0)
        else:
            scores.append(differentiation / quality_thresholds["min_differentiation"])

        results["overall_score"] = np.mean(scores) if scores else 0

        # 7. Generate Recommendations
        if results["overall_score"] < 0.6:
            results["recommendations"].append("Consider reducing number of clusters or adjusting HMM parameters")
        if predictability < quality_thresholds["min_predictability"]:
            results["recommendations"].append("Clusters show low predictive power - consider feature engineering improvements")
        if stability < quality_thresholds["min_stability"]:
            results["recommendations"].append("Clusters are unstable over time - consider longer lookback periods")
        if silhouette < quality_thresholds["min_silhouette"]:
            results["recommendations"].append("Low silhouette score - clusters may not be well-separated")
        if differentiation < quality_thresholds["min_differentiation"]:
            results["recommendations"].append("Low differentiation - clusters may not capture distinct market conditions")

        return results

    def generate_report(self, validation_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("# HMM Cluster Validation Report")
        report.append("")

        # Overall Score
        report.append(f"## Overall Assessment")
        report.append(f"**Overall Score**: {validation_results['overall_score']:.3f}")

        score_level = "🟢 Good" if validation_results['overall_score'] > 0.7 else \
                     "🟡 Moderate" if validation_results['overall_score'] > 0.5 else "🔴 Poor"
        report.append(f"**Quality Level**: {score_level}")
        report.append("")

        # Quality Metrics
        if "quality_metrics" in validation_results and "error" not in validation_results["quality_metrics"]:
            qm = validation_results["quality_metrics"]
            report.append("## Quality Metrics")
            report.append(f"- **Silhouette Score**: {qm.get('silhouette_score', 0):.4f}")
            report.append(f"- **Calinski-Harabasz Score**: {qm.get('calinski_harabasz_score', 0):.2f}")
            report.append(f"- **Davies-Bouldin Score**: {qm.get('davies_bouldin_score', 0):.4f}")
            report.append(f"- **Cluster Balance**: {qm.get('cluster_balance', 0):.4f}")
            report.append("")

        # Predictive Power
        if "predictive_power" in validation_results and "error" not in validation_results["predictive_power"]:
            pp = validation_results["predictive_power"]
            report.append("## Predictive Power")
            report.append(f"- **Average Predictability**: {pp.get('avg_predictability', 0):.4f}")
            report.append("")

        # Stability
        if "stability" in validation_results and "error" not in validation_results["stability"]:
            st = validation_results["stability"]
            report.append("## Stability")
            report.append(f"- **Average Stability**: {st.get('avg_stability', 0):.4f}")
            report.append(f"- **Stability Std**: {st.get('stability_std', 0):.4f}")
            report.append("")

        # Market Differentiation
        if "market_differentiation" in validation_results and "error" not in validation_results["market_differentiation"]:
            md = validation_results["market_differentiation"]
            report.append("## Market Differentiation")
            report.append(f"- **Average Differentiation**: {md.get('avg_differentiation', 0):.4f}")
            report.append("")

        # Return Predictability
        if "return_predictability" in validation_results:
            rp = validation_results["return_predictability"]
            report.append("## Return Predictability")
            for period_key, period_data in rp.items():
                if "error" not in period_data:
                    report.append(f"- **{period_key}**: {period_data.get('predictability_score', 0):.6f}")
            report.append("")

        # Recommendations
        if validation_results.get("recommendations"):
            report.append("## Recommendations")
            for i, rec in enumerate(validation_results["recommendations"], 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Cluster Details
        if "market_differentiation" in validation_results and "error" not in validation_results["market_differentiation"]:
            md = validation_results["market_differentiation"]
            if "cluster_characteristics" in md:
                report.append("## Cluster Details")
                for cluster_id, char in md["cluster_characteristics"].items():
                    report.append(f"### Cluster {cluster_id}")
                    report.append(f"- **Size**: {char.get('size', 0)} ({char.get('percentage', 0):.1f}%)")
                    report.append(f"- **Avg Volatility**: {char.get('avg_volatility', 0):.6f}")
                    report.append(f"- **Avg Momentum**: {char.get('avg_momentum', 0):.6f}")
                    report.append(f"- **Avg Volume**: {char.get('avg_volume', 0):.6f}")
                    report.append("")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_path}")

        return report_text

    def create_visualizations(self, cluster_data: pd.DataFrame, validation_results: Dict[str, Any],
                            output_dir: Optional[str] = None) -> None:
        """Create visualizations for cluster analysis."""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HMM Cluster Analysis', fontsize=16, fontweight='bold')

        # 1. Cluster Size Distribution
        if "quality_metrics" in validation_results and "cluster_sizes" in validation_results["quality_metrics"]:
            cluster_sizes = validation_results["quality_metrics"]["cluster_sizes"]
            axes[0, 0].bar(cluster_sizes.keys(), cluster_sizes.values())
            axes[0, 0].set_title('Cluster Size Distribution')
            axes[0, 0].set_xlabel('Cluster ID')
            axes[0, 0].set_ylabel('Number of Samples')

        # 2. Stability Over Time
        if "stability" in validation_results and "stability_scores" in validation_results["stability"]:
            stability_scores = validation_results["stability"]["stability_scores"]
            axes[0, 1].plot(stability_scores)
            axes[0, 1].set_title('Cluster Stability Over Time')
            axes[0, 1].set_xlabel('Window Index')
            axes[0, 1].set_ylabel('Stability Score')
            axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
            axes[0, 1].legend()

        # 3. Market Characteristics by Cluster
        if "market_differentiation" in validation_results and "cluster_characteristics" in validation_results["market_differentiation"]:
            cluster_chars = validation_results["market_differentiation"]["cluster_characteristics"]
            cluster_ids = list(cluster_chars.keys())
            volatilities = [cluster_chars[cid]["avg_volatility"] for cid in cluster_ids]
            momentums = [cluster_chars[cid]["avg_momentum"] for cid in cluster_ids]

            scatter = axes[1, 0].scatter(volatilities, momentums, c=cluster_ids, cmap='viridis', s=100, alpha=0.7)
            axes[1, 0].set_title('Market Characteristics by Cluster')
            axes[1, 0].set_xlabel('Average Volatility')
            axes[1, 0].set_ylabel('Average Momentum')
            plt.colorbar(scatter, ax=axes[1, 0], label='Cluster ID')

        # 4. Overall Score Breakdown
        metrics = []
        scores = []

        if "quality_metrics" in validation_results:
            silhouette = validation_results["quality_metrics"].get("silhouette_score", 0)
            metrics.append("Silhouette")
            scores.append(min(silhouette / 0.3, 1.0) if silhouette > 0 else 0)

        if "predictive_power" in validation_results:
            predictability = validation_results["predictive_power"].get("avg_predictability", 0)
            metrics.append("Predictability")
            scores.append(min(predictability / 0.4, 1.0) if predictability > 0 else 0)

        if "stability" in validation_results:
            stability = validation_results["stability"].get("avg_stability", 0)
            metrics.append("Stability")
            scores.append(min(stability / 0.5, 1.0) if stability > 0 else 0)

        if "market_differentiation" in validation_results:
            differentiation = validation_results["market_differentiation"].get("avg_differentiation", 0)
            metrics.append("Differentiation")
            scores.append(min(differentiation / 0.1, 1.0) if differentiation > 0 else 0)

        if metrics:
            bars = axes[1, 1].bar(metrics, scores)
            axes[1, 1].set_title('Quality Score Breakdown')
            axes[1, 1].set_ylabel('Normalized Score')
            axes[1, 1].set_ylim(0, 1.1)

            # Color bars based on score
            for bar, score in zip(bars, scores):
                if score > 0.7:
                    bar.set_color('green')
                elif score > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_path / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to: {output_path / 'cluster_analysis.png'}")
        else:
            plt.show()

        plt.close()


def main():
    """Main function to run cluster validation."""
    parser = argparse.ArgumentParser(description="Test HMM cluster relevance")
    parser.add_argument("--data_path", type=str, help="Path to cluster data parquet file")
    parser.add_argument("--config_path", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output_dir", type=str, help="Output directory for reports and visualizations")
    parser.add_argument("--thresholds", type=str, help="JSON string with quality thresholds")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)

    # Load quality thresholds
    quality_thresholds = None
    if args.thresholds:
        quality_thresholds = json.loads(args.thresholds)

    # Load cluster data
    if args.data_path:
        print(f"📂 Loading cluster data from: {args.data_path}")
        cluster_data = pd.read_parquet(args.data_path)
    else:
        print("❌ Please provide --data_path argument")
        return

    print(f"📊 Loaded {len(cluster_data)} samples with {len(cluster_data.columns)} columns")
    print(f"🔍 Found clusters: {cluster_data['composite_cluster_id'].unique() if 'composite_cluster_id' in cluster_data.columns else 'None'}")

    # Initialize validator
    validator = HMMClusterValidator(config)

    # Run comprehensive validation
    validation_results = validator.comprehensive_validation(cluster_data, quality_thresholds)

    # Generate report
    report = validator.generate_report(validation_results,
                                     output_path=f"{args.output_dir}/cluster_validation_report.md" if args.output_dir else None)

    # Create visualizations
    if args.output_dir:
        validator.create_visualizations(cluster_data, validation_results, args.output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Overall Score: {validation_results['overall_score']:.3f}")

    if validation_results['overall_score'] > 0.7:
        print("✅ Clusters are of good quality - safe to proceed with ML training")
    elif validation_results['overall_score'] > 0.5:
        print("⚠️ Clusters are of moderate quality - consider improvements before ML training")
    else:
        print("❌ Clusters are of poor quality - significant improvements needed before ML training")

    if validation_results.get("recommendations"):
        print("\nRecommendations:")
        for i, rec in enumerate(validation_results["recommendations"], 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()