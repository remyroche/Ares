"""Reporting utilities for NAS/TAS regime analysis."""
from __future__ import annotations

from typing import Any, Dict

from src.utils.tprint import tprint


def print_detailed_metrics(distribution: Dict[str, Any], metrics: Dict[str, Any], regime_type: str) -> None:
    """Print detailed statistics for a single regime type."""
    tprint(f"\n🔍 {regime_type} REGIME DETAILED ANALYSIS", "INFO")
    tprint("-" * 60, "INFO")

    tprint("📊 Distribution Statistics:", "INFO")
    for regime, count in distribution["regime_counts"].items():
        percentage = distribution["regime_percentages"][regime]
        tprint(f"   {regime}: {count} samples ({percentage}%)", "INFO")

    tprint("📈 Balance Metrics:", "INFO")
    tprint(f"   Min: {distribution['regime_balance']['min_percentage']:.1f}%", "INFO")
    tprint(f"   Max: {distribution['regime_balance']['max_percentage']:.1f}%", "INFO")
    tprint(f"   Std: {distribution['regime_balance']['std_percentage']:.1f}%", "INFO")
    tprint(f"   Balance Score: {distribution['regime_balance']['balance_score']:.3f}", "INFO")

    tprint("🎯 Clustering Quality Metrics:", "INFO")
    tprint(
        "   Silhouette Score: "
        f"{metrics['silhouette_score']:.4f} ({metrics['interpretation']['silhouette']})",
        "INFO",
    )
    tprint(
        "   Davies-Bouldin Score: "
        f"{metrics['davies_bouldin_score']:.4f} ({metrics['interpretation']['davies_bouldin']})",
        "INFO",
    )
    tprint(
        "   Calinski-Harabasz Score: "
        f"{metrics.get('calinski_harabasz_score', 0.0):.4f}",
        "INFO",
    )
    tprint(
        "   CV Score: "
        f"{metrics['cv_score']:.4f} ({metrics['interpretation']['cv_score']})",
        "INFO",
    )
    tprint("-" * 60, "INFO")


def print_analysis_summary(analysis: Dict[str, Any]) -> None:
    """Print a formatted summary of the analysis."""
    tprint("\n" + "=" * 80, "INFO")
    tprint("📊 REGIME ANALYSIS SUMMARY", "INFO")
    tprint("=" * 80, "INFO")

    nas_dist = analysis["nas_analysis"]["distribution"]
    nas_metrics = analysis["nas_analysis"]["clustering_metrics"]
    tprint(
        f"\n🔬 NAS REGIMES ({nas_dist['num_regimes']} regimes, {nas_dist['total_samples']} samples)",
        "INFO",
    )
    tprint(
        "   Distribution: "
        f"{nas_dist['regime_balance']['min_percentage']:.1f}% - "
        f"{nas_dist['regime_balance']['max_percentage']:.1f}% (std: "
        f"{nas_dist['regime_balance']['std_percentage']:.1f}%)",
        "INFO",
    )
    tprint(f"   Balance Score: {nas_dist['regime_balance']['balance_score']:.3f}", "INFO")
    tprint(
        "   Silhouette: "
        f"{nas_metrics['silhouette_score']:.3f} ({nas_metrics['interpretation']['silhouette']})",
        "INFO",
    )
    tprint(
        "   Davies-Bouldin: "
        f"{nas_metrics['davies_bouldin_score']:.3f} ({nas_metrics['interpretation']['davies_bouldin']})",
        "INFO",
    )
    tprint(
        "   CV Score: "
        f"{nas_metrics['cv_score']:.3f} ({nas_metrics['interpretation']['cv_score']})",
        "INFO",
    )

    tas_dist = analysis["tas_analysis"]["distribution"]
    tas_metrics = analysis["tas_analysis"]["clustering_metrics"]
    tprint(
        f"\n🎯 TAS REGIMES ({tas_dist['num_regimes']} regimes, {tas_dist['total_samples']} samples)",
        "INFO",
    )
    tprint(
        "   Distribution: "
        f"{tas_dist['regime_balance']['min_percentage']:.1f}% - "
        f"{tas_dist['regime_balance']['max_percentage']:.1f}% (std: "
        f"{tas_dist['regime_balance']['std_percentage']:.1f}%)",
        "INFO",
    )
    tprint(f"   Balance Score: {tas_dist['regime_balance']['balance_score']:.3f}", "INFO")
    tprint(
        "   Silhouette: "
        f"{tas_metrics['silhouette_score']:.3f} ({tas_metrics['interpretation']['silhouette']})",
        "INFO",
    )
    tprint(
        "   Davies-Bouldin: "
        f"{tas_metrics['davies_bouldin_score']:.3f} ({tas_metrics['interpretation']['davies_bouldin']})",
        "INFO",
    )
    tprint(
        "   CV Score: "
        f"{tas_metrics['cv_score']:.3f} ({tas_metrics['interpretation']['cv_score']})",
        "INFO",
    )

    tprint("\n" + "=" * 80, "INFO")
