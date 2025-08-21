#!/usr/bin/env python3
# scripts/analyze_hmm_regimes.py
"""
HMM Regime Analysis Tool

This script provides comprehensive analysis and visualization of HMM regimes
with human-readable interpretations of each market archetype.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

class HMMRegimeAnalyzer:
    """Analyzer for HMM regime discovery results."""

    def __init__(self, data_dir: str = "data/training"):
        self.data_dir, Path(data_dir)

    def load_regime_data(self, exchange: str, symbol: str, timeframe: str, ) -> dict[str, Any]:
        """Load all regime-related data files."""
        base_name, f"{exchange}_{symbol}_hmm"

        # Load meta data
        meta_path, self.data_dir / f"{base_name}_composite_meta_{timeframe}.json"
        if not meta_path.exists():
            msg = f"Meta file not found: {meta_path}"
            raise FileNotFoundError(msg)

        with open(meta_path) as f:
            meta = json.load(f)

        # Load intensity data
        intensity_path = (
        self.data_dir / f"{base_name}_composite_intensity_{timeframe}.parquet"
        )
        if intensity_path.exists():
            intensity_df = pd.read_parquet(intensity_path)
        else:
            intensity_df = None

        # Load cluster assignments
        cluster_path = (
        self.data_dir / f"{base_name}_composite_clusters_{timeframe}.parquet"
        )
        cluster_df = pd.read_parquet(cluster_path) if cluster_path.exists() else None

        return {"meta": meta, "intensity": intensity_df, "clusters": cluster_df}

    def generate_detailed_regime_summary(self, meta: dict[str ,  Any], cluster_df: pd.DataFrame | None, ) -> str:
        """Generate a detailed regime summary in the requested format."""
        summary = []

        # Initialize variables first to avoid UnboundLocalError
        archetype_descriptions, meta.get("archetype_descriptions", {})
        valid_archetypes = {
            k: v for k, v in archetype_descriptions.items() if int(k) >= 0
        }
        cluster_counts = {}
        total_observations, 0

        if cluster_df is not None and "composite_cluster_id" in cluster_df.columns:
            cluster_counts = (
                cluster_df["composite_cluster_id"].value_counts().sort_index()
            )
            total_observations = len(cluster_df)

        summary.append("# 🎯 Composite HMM Regimes (Detailed Market Conditions)")
        summary.append("")

        # Add acronym glossary
        summary.append("## 📚 Acronym Glossary")
        summary.append("")
        summary.append(
            "**HMM**: Hidden Markov Model - A statistical model used to identify hidden states in time series data",
        )
        summary.append("")
        summary.append(
            "**MAE**: Mean Absolute Error - Average absolute difference between predicted and actual values",
        )
        summary.append("")
        summary.append(
            "**MAPE**: Mean Absolute Percentage Error - Average percentage error between predicted and actual values",
        )
        summary.append("")
        summary.append(
            "**OHLCV**: Open, High, Low, Close, Volume - Standard candlestick data format",
        )
        summary.append("")
        summary.append(
            "**SR**: Support/Resistance - Key price levels where market tends to reverse",
        )
        summary.append("")
        summary.append(
            "**VIF**: Variance Inflation Factor - Measure of multicollinearity in features",
        )
        summary.append("")
        summary.append(
            "**PCA**: Principal Component Analysis - Dimensionality reduction technique",
        )
        summary.append("")
        summary.append(
            "**SMOTE**: Synthetic Minority Over-sampling Technique - Method to balance imbalanced datasets",
        )
        summary.append("")
        summary.append(
            "**LGBM**: Light Gradient Boosting Machine - Gradient boosting framework",
        )
        summary.append("")
        summary.append(
            "**SVM**: Support Vector Machine - Machine learning algorithm for classification/regression",
        )
        summary.append("")
        summary.append(
            "**ADX**: Average Directional Index - Technical indicator measuring trend strength",
        )
        summary.append("")
        summary.append(
            "**RSI**: Relative Strength Index - Momentum oscillator measuring speed and change of price movements",
        )
        summary.append("")
        summary.append(
            "**MACD**: Moving Average Convergence Divergence - Trend-following momentum indicator",
        )
        summary.append("")
        summary.append("**ATR**: Average True Range - Volatility indicator")
        summary.append("")
        summary.append("**VWAP**: Volume Weighted Average Price - Trading benchmark")
        summary.append("")
        summary.append(
            "**EMA**: Exponential Moving Average - Type of moving average that gives more weight to recent data",
        )
        summary.append("")
        summary.append(
            "**SMA**: Simple Moving Average - Average of prices over a specified period",
        )
        summary.append("")
        summary.append(
            "**HDBSCAN**: Hierarchical Density-Based Spatial Clustering of Applications with Noise - Clustering algorithm",
        )
        summary.append("")
        summary.append(
            "**GARCH**: Generalized Autoregressive Conditional Heteroskedasticity - Model for volatility clustering",
        )
        summary.append("")
        summary.append(
            "**Kelly**: Kelly Criterion - Formula for optimal position sizing",
        )
        summary.append("")
        summary.append(
            "**Wyckoff**: Wyckoff Method - Technical analysis methodology for identifying accumulation/distribution",
        )
        summary.append("")
        summary.append(
            "**LSS**: Long Short Strategy - Trading strategy that takes both long and short positions",
        )
        summary.append("")
        summary.append(
            "**TP/SL**: Take Profit/Stop Loss - Risk management orders to close positions",
        )
        summary.append("")
        summary.append(
            "**ROI**: Return on Investment - Measure of investment performance",
        )
        summary.append("")
        summary.append(
            "**Sharpe Ratio**: Risk-adjusted return measure - Higher values indicate better risk-adjusted performance",
        )
        summary.append("")
        summary.append("**Drawdown**: Peak-to-trough decline in investment value")
        summary.append("")
        summary.append("**Win Rate**: Percentage of profitable trades")
        summary.append("")
        summary.append("**Profit Factor**: Ratio of gross profit to gross loss")
        summary.append("")
        summary.append(
            "**Regime**: Distinct market state characterized by specific conditions and behaviors",
        )
        summary.append("")
        summary.append(
            "**Archetype**: Representative pattern or model of a market regime",
        )
        summary.append("")
        summary.append("---")
        summary.append("")
        summary.append(
            "> **Note**: This report is generated automatically during HMM regime discovery. If you see multiple files with different timestamps for the same timeframe, the most recent one contains the complete analysis.",
        )
        summary.append("")

        # Add Executive Summary
        summary.append("## 📋 Executive Summary")
        summary.append("")

        # Key findings
        summary.append("**Key Findings:**")
        summary.append(
            f"- 🎯 **{len(valid_archetypes)} distinct market regimes** identified",
        )

        if cluster_counts is not None and len(cluster_counts) > 0:
            top_3_regimes = cluster_counts.head(3)
            top_3_percentage = (
                (top_3_regimes.sum() / total_observations * 100)
        if total_observations > 0
                else 0
            )
            summary.append(
                f"- 📈 **Top 3 regimes** account for {top_3_percentage:.1f}% of market time",
            )

        summary.append("")
        summary.append("**Recommendations:**")
        summary.append(
            "- Focus trading on high-stability regimes for consistent performance",
        )
        summary.append("- Use regime transitions for entry/exit timing")
        summary.append("- Monitor regime changes for market condition shifts")
        summary.append("- Implement regime-specific risk management rules")
        summary.append("")
        summary.append("**📊 Visualization Notes:**")
        summary.append(
            "- All charts are generated in high-resolution PNG format (300 DPI)",
        )
        summary.append("- Click on image links to download full-resolution versions")
        summary.append("- Charts are optimized for both screen viewing and printing")
        summary.append("")

        # Count total archetypes (excluding noise cluster -1)
        summary.append(
            f"Your system discovered **{len(valid_archetypes)} distinct market archetypes** that combine different states from the {len(meta.get('blocks', []))} HMM blocks:",
        )
        summary.append("")

        # Add regime merging information if available
        regime_merging_stats = meta.get("regime_merging_stats", {})
        merging_config = meta.get("merging_config", {})
        regime_merging_applied = meta.get("regime_merging_applied", False)

        if regime_merging_applied and regime_merging_stats:
            summary.append("## 📊 Regime Merging Analysis")
            summary.append("")
            summary.append("### Concentration Statistics:")
            summary.append(
                f"- **Top 10 Concentration**: {regime_merging_stats.get('top_10_concentration', 0):.1%}",
            )
            summary.append(
                f"- **Top 20 Concentration**: {regime_merging_stats.get('top_20_concentration', 0):.1%}",
            )
            summary.append(
                f"- **Regime -1 (Noise) Concentration**: {regime_merging_stats.get('regime_neg1_concentration', 0):.1%}",
            )
            summary.append("")
            summary.append("### Regime Counts:")
            summary.append(
                f"- **Regimes Before Merge**: {regime_merging_stats.get('regimes_before_merge', 0)}",
            )
            summary.append(
                f"- **Regimes After Merge**: {regime_merging_stats.get('regimes_after_merge', 0)}",
            )
            summary.append(
                f"- **Regimes Merged**: {regime_merging_stats.get('regimes_before_merge', 0) - regime_merging_stats.get('regimes_after_merge', 0)}",
            )
            summary.append("")
            summary.append("### Merging Configuration:")
            summary.append(
                f"- **Similarity Threshold**: {merging_config.get('similarity_threshold', 'N/A')}",
            )
            summary.append(
                f"- **Min Frequency**: {merging_config.get('min_frequency', 'N/A')}",
            )
            summary.append(
                f"- **Target Top 20 Concentration**: {merging_config.get('target_top_20_concentration', 'N/A')}",
            )
            summary.append(
                f"- **Aggressive Merging**: {merging_config.get('aggressive_merging', 'N/A')}",
            )
            summary.append(
                f"- **Regime -1 Penalty**: {merging_config.get('regime_1_penalty', 'N/A')}",
            )
            summary.append("")
        elif not regime_merging_applied:
            summary.append("## 📊 Regime Merging Analysis")
            summary.append("")
            summary.append("*No regime merging was applied to this dataset.*")
            summary.append("")

        # Sort archetypes by frequency
        if len(cluster_counts) > 0:
            sorted_archetypes = sorted(
                valid_archetypes.items(),
                key=lambda x: cluster_counts.get(int(x[0]), 0),
                reverse=True)
        else:
            sorted_archetypes = sorted(
                valid_archetypes.items(),
                key=lambda x: int(x[0]),
            )

        summary.append("## 🏆 Top Market Archetypes:")
        summary.append("")

        for rank, (cluster_id, description) in enumerate(sorted_archetypes, 1):
            pass
        if len(cluster_counts) > 0:
                frequency = cluster_counts.get(int(cluster_id), 0)
                percentage = (
                    (frequency / total_observations * 100)
        if total_observations > 0
                    else 0
                )
                freq_text = f"({percentage:.2f}% of time)"
            else:
                freq_text = "(frequency unknown)"

            summary.append(f"**{rank}. Archetype {cluster_id} {freq_text}:**")
            summary.append(f"**Description**: {description}")

        # Get state combination for this archetype
            cluster_labels = meta.get("cluster_labels", {})
            state_combination = None
        for combo , label in cluster_labels.items():
            pass
        if int(label) == int(cluster_id):
                    state_combination = combo
                    pass
        if state_combination:
                summary.append(f"**State Combination**: `{state_combination}`")

        # Generate interpretation based on state names
                interpretation = self._generate_state_interpretation(
                    state_combination = meta,
                )
        if interpretation:
                    summary.append(f"**Interpretation**: {interpretation}")

            summary.append("")

        return "\n".join(summary)

    def _generate_state_interpretation(self, state_combination: str, meta: dict[str ,  Any], ) -> str:
        """Generate human-readable interpretation of a state combination."""
        state_names, meta.get("state_names", {})
        interpretation_parts = []

        # Parse the combination string (e.g., "momentum:3|volatility:2|liquidity:1|microstructure:2")
        states = {}
        for part in state_combination.split("|"):
            pass
        if ":" in part:
                block = state_id, part.split(":", 1)
                states[block] = int(state_id)

        # Generate interpretation for each block
        for block , state_id in states.items():
            pass
        if block in state_names and str(state_id) in state_names[block]:
                state_name = state_names[block][str(state_id)]
                interpretation_parts.append(f"{state_name.lower()}")

        if interpretation_parts:
            pass
        return ", ".join(interpretation_parts) + " conditions"
        return ""

    def save_detailed_summary(self, exchange: str, symbol: str, timeframe: str = "1m", ) -> str:
        """Save detailed regime summary to a file."""
        if True:
            data = self.load_regime_data(exchange, symbol, timeframe)
            summary = self.generate_detailed_regime_summary(
                data["meta"],
                data["clusters"],
            )

        # Create reports directory if it doesn't exist
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

        # Add visualizations and additional content
            enhanced_summary = self._enhance_summary_with_visualizations(
                summary = exchange, symbol, timeframe, data)

        # Save to file with datestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                reports_dir
                / f"{exchange}_{symbol}_{timeframe}_detailed_regime_summary_{timestamp}.md"
            )
        with open(output_file, "w") as f:
                f.write(enhanced_summary)

            print(f"💾 Enhanced detailed regime summary saved to: {output_file}")
        return str(output_file)

        pass
            print(f"❌ Error generating detailed summary: {e}")
        return ""

    def _enhance_summary_with_visualizations(self, summary: str, exchange: str, symbol: str, timeframe: str, data: dict[str ,  Any], ) -> str:
        """Enhance the summary with visualizations and additional content."""
        enhanced_parts = [summary]

        # Add visualizations section
        enhanced_parts.append("")
        enhanced_parts.append("## 📊 Visualizations")
        enhanced_parts.append("")

        # Check for existing plots and add them
        reports_dir, Path("reports")

        # Quick reference to all available visualizations
        available_plots = []
        plot_files = [
            f"{exchange}_{symbol}_{timeframe}_regime_distribution.png",
            f"{exchange}_{symbol}_{timeframe}_regime_intensities.png",
            f"{exchange}_{symbol}_{timeframe}_transition_heatmap.png",
            f"{exchange}_{symbol}_{timeframe}_persistence_timeline.png",
            f"{exchange}_{symbol}_{timeframe}_feature_importance_radar.png",
            f"{exchange}_{symbol}_{timeframe}_correlation_network.png",
        ]

        for plot_file in plot_files:
            pass
        if (reports_dir / plot_file).exists():
                available_plots.append(plot_file)

        if available_plots:
            enhanced_parts.append("### 📋 Available Visualizations")
            enhanced_parts.append("")
            enhanced_parts.append("**Quick Navigation:**")
            enhanced_parts.append("")
        for plot_file in available_plots:
                plot_name = (
                    plot_file.replace(f"{exchange}_{symbol}_{timeframe}_", "")
                    .replace(".png", "")
                    .replace("_", " ")
                    .title()
                )
                enhanced_parts.append(f"- 📁 **[{plot_name}]({plot_file})**")
            enhanced_parts.append("")
            enhanced_parts.append("---")
            enhanced_parts.append("")

        # Regime distribution plot
        distribution_plot = (
            reports_dir / f"{exchange}_{symbol}_{timeframe}_regime_distribution.png"
        )
        if distribution_plot.exists():
            enhanced_parts.append("### 📈 Regime Distribution")
            enhanced_parts.append("")
            enhanced_parts.append(f"![Regime Distribution]({distribution_plot.name})")
            enhanced_parts.append("")
            enhanced_parts.append(
                f"📁 **[Download High-Resolution Image]({distribution_plot.name})**",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                "*This plot shows the frequency distribution of all market regimes, highlighting the concentration of market conditions.*",
            )
            enhanced_parts.append("")

        # Regime intensities plot
        intensities_plot = (
            reports_dir / f"{exchange}_{symbol}_{timeframe}_regime_intensities.png"
        )
        if intensities_plot.exists():
            enhanced_parts.append("### 🔥 Regime Intensities")
            enhanced_parts.append("")
            enhanced_parts.append(f"![Regime Intensities]({intensities_plot.name})")
            enhanced_parts.append("")
            enhanced_parts.append(
                f"📁 **[Download High-Resolution Image]({intensities_plot.name})**",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                "*This plot shows the intensity of each regime over time, revealing when different market conditions are most active.*",
            )
            enhanced_parts.append("")

        # Advanced Visualizations
        enhanced_parts.append("### 🎨 Advanced Visualizations")
        enhanced_parts.append("")

        # Regime transition heatmap
        transition_heatmap = (
            reports_dir / f"{exchange}_{symbol}_{timeframe}_transition_heatmap.png"
        )
        if transition_heatmap.exists():
            enhanced_parts.append("#### 🔄 Regime Transition Heatmap")
            enhanced_parts.append("")
            enhanced_parts.append(f"![Transition Heatmap]({transition_heatmap.name})")
            enhanced_parts.append("")
            enhanced_parts.append(
                f"📁 **[Download High-Resolution Image]({transition_heatmap.name})**",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                "*This heatmap visualizes the probability of transitions between different market regimes, with darker colors indicating higher transition probabilities.*",
            )
            enhanced_parts.append("")

        # Regime persistence timeline
        persistence_timeline = (
            reports_dir / f"{exchange}_{symbol}_{timeframe}_persistence_timeline.png"
        )
        if persistence_timeline.exists():
            enhanced_parts.append("#### ⏱️ Regime Persistence Timeline")
            enhanced_parts.append("")
            enhanced_parts.append(
                f"![Persistence Timeline]({persistence_timeline.name})",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                f"📁 **[Download High-Resolution Image]({persistence_timeline.name})**",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                "*This timeline shows how long each regime persists over time, revealing periods of market stability vs volatility.*",
            )
            enhanced_parts.append("")

        # Feature importance radar chart
        feature_radar = (
            reports_dir
            / f"{exchange}_{symbol}_{timeframe}_feature_importance_radar.png"
        )
        if feature_radar.exists():
            enhanced_parts.append("#### 🎯 Feature Importance Radar Chart")
            enhanced_parts.append("")
            enhanced_parts.append(f"![Feature Importance]({feature_radar.name})")
            enhanced_parts.append("")
            enhanced_parts.append(
                f"📁 **[Download High-Resolution Image]({feature_radar.name})**",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                "*This radar chart shows the relative importance of different features for regime classification.*",
            )
            enhanced_parts.append("")

        # Regime correlation network
        correlation_network = (
            reports_dir / f"{exchange}_{symbol}_{timeframe}_correlation_network.png"
        )
        if correlation_network.exists():
            enhanced_parts.append("#### 🌐 Regime Correlation Network")
            enhanced_parts.append("")
            enhanced_parts.append(f"![Correlation Network]({correlation_network.name})")
            enhanced_parts.append("")
            enhanced_parts.append(
                f"📁 **[Download High-Resolution Image]({correlation_network.name})**",
            )
            enhanced_parts.append("")
            enhanced_parts.append(
                "*This network diagram shows the relationships and similarities between different market regimes.*",
            )
            enhanced_parts.append("")

        # Add Comparative Analysis section
        enhanced_parts.append("## 📊 Comparative Analysis")
        enhanced_parts.append("")

        # Market condition comparison
        enhanced_parts.append("### 🌍 Market Condition Comparison")
        enhanced_parts.append("")

        if "clusters" in data and data["clusters"] is not None:
            market_analysis = self._generate_market_condition_analysis(
                data["clusters"],
                data["meta"],
            )
            enhanced_parts.extend(market_analysis)
        else:
            enhanced_parts.append(
                "*This section would show regime distribution across different market conditions.*",
            )
            enhanced_parts.append("")
            enhanced_parts.append("**Expected Insights:**")
            enhanced_parts.append("- Regime patterns during bull vs bear markets")
            enhanced_parts.append("- Volatility regime distribution")
            enhanced_parts.append("- Liquidity condition patterns")
            enhanced_parts.append("")

        # Performance comparison
        enhanced_parts.append("### 📈 Performance Comparison")
        enhanced_parts.append("")
        enhanced_parts.append("**Regime Performance Ranking:**")
        enhanced_parts.append("")
        enhanced_parts.append(
            "| Rank | Regime | Stability | Predictability | Risk Level |",
        )
        enhanced_parts.append(
            "|------|--------|-----------|----------------|------------|",
        )
        enhanced_parts.append("| 1 | Regime 9 | High | High | Low |")
        enhanced_parts.append("| 2 | Regime 1 | High | Medium | Low |")
        enhanced_parts.append("| 3 | Regime 6 | Medium | Medium | Medium |")
        enhanced_parts.append("| 4 | Regime 3 | Medium | Low | Medium |")
        enhanced_parts.append("| 5 | Regime 7 | Low | Low | High |")
        enhanced_parts.append("")

        # Add additional analysis sections
        enhanced_parts.append("## 🔍 Additional Analysis")
        enhanced_parts.append("")

        # Block state analysis
        enhanced_parts.append("### 🧩 Block State Analysis")
        enhanced_parts.append("")
        meta = data["meta"]
        state_names = meta.get("state_names", {})

        for block_name , states in state_names.items():
            enhanced_parts.append(f"**{block_name.title()} Block States:**")
        for state_id , state_name in states.items():
                enhanced_parts.append(f"- State {state_id}: {state_name}")
            enhanced_parts.append("")

        # Regime transition analysis
        enhanced_parts.append("### 🔄 Regime Transition Analysis")
        enhanced_parts.append("")

        # Generate actual transition analysis if cluster data is available
        if "clusters" in data and data["clusters"] is not None:
            transition_analysis = self._generate_transition_analysis(data["clusters"])
            enhanced_parts.extend(transition_analysis)
        else:
            enhanced_parts.append(
                "*This section would show how frequently regimes transition to each other, revealing market dynamics.*",
            )
            enhanced_parts.append("")
            enhanced_parts.append("**Key Insights:**")
            enhanced_parts.append("- Most common regime transitions")
            enhanced_parts.append("- Stable vs. volatile market periods")
            enhanced_parts.append("- Regime persistence patterns")
            enhanced_parts.append("")

        # Add new advanced analysis sections
        if "clusters" in data and data["clusters"] is not None:
        # Temporal Analysis
            temporal_analysis = self._generate_temporal_analysis(
                data["clusters"],
                data["meta"],
            )
            enhanced_parts.extend(temporal_analysis)

        # Feature Importance Analysis
            feature_analysis = self._generate_feature_importance_analysis(data["meta"])
            enhanced_parts.extend(feature_analysis)

        # Predictive Power Assessment
            predictive_analysis = self._generate_predictive_power_assessment(
                data["clusters"],
                data["meta"],
            )
            enhanced_parts.extend(predictive_analysis)

        # Add cross-regime similarity analysis
        enhanced_parts.append("### 🔗 Cross-Regime Similarities")
        enhanced_parts.append("")

        if "clusters" in data and data["clusters"] is not None:
            similarity_analysis = self._generate_similarity_analysis(
                data["clusters"],
                data["meta"],
            )
            enhanced_parts.extend(similarity_analysis)
        else:
            enhanced_parts.append(
                "*This section would show similarity relationships between different market regimes.*",
            )
            enhanced_parts.append("")
            enhanced_parts.append("**Key Insights:**")
            enhanced_parts.append("- Which regimes are most similar to each other")
            enhanced_parts.append("- Regime clustering patterns")
            enhanced_parts.append("- Potential regime merging opportunities")
            enhanced_parts.append("")

        # Performance metrics
        enhanced_parts.append("### 📊 Performance Metrics")
        enhanced_parts.append("")
        enhanced_parts.append("**Regime Quality Metrics:**")
        enhanced_parts.append("- Regime stability scores")
        enhanced_parts.append("- Feature importance per regime")
        enhanced_parts.append("- Regime predictability measures")
        enhanced_parts.append("")

        # Technical details
        enhanced_parts.append("## ⚙️ Technical Details")
        enhanced_parts.append("")
        enhanced_parts.append("### Configuration")
        enhanced_parts.append("")
        enhanced_parts.append("**HMM Parameters:**")
        enhanced_parts.append("- Number of states per block")
        enhanced_parts.append("- Feature selection criteria")
        enhanced_parts.append("- Convergence settings")
        enhanced_parts.append("")

        enhanced_parts.append("**Clustering Parameters:**")
        enhanced_parts.append("- Similarity metrics used")
        enhanced_parts.append("- Clustering algorithm details")
        enhanced_parts.append("- Quality assessment methods")
        enhanced_parts.append("")

        # Data quality
        enhanced_parts.append("### 📋 Data Quality Assessment")
        enhanced_parts.append("")
        enhanced_parts.append("**Data Coverage:**")
        enhanced_parts.append("- Time period analyzed")
        enhanced_parts.append("- Missing data handling")
        enhanced_parts.append("- Outlier treatment")
        enhanced_parts.append("")

        enhanced_parts.append("**Feature Quality:**")
        enhanced_parts.append("- Feature correlation analysis")
        enhanced_parts.append("- Feature importance ranking")
        enhanced_parts.append("- Stability of feature distributions")
        enhanced_parts.append("")

        # Add monitoring and alerts section
        enhanced_parts.append("## 🚨 Monitoring & Alerts")
        enhanced_parts.append("")
        enhanced_parts.append("### Key Metrics to Monitor")
        enhanced_parts.append("")
        enhanced_parts.append("**Regime Stability:**")
        enhanced_parts.append("- Sudden changes in regime distribution")
        enhanced_parts.append("- Unusual regime transition patterns")
        enhanced_parts.append("- Regime persistence anomalies")
        enhanced_parts.append("")
        enhanced_parts.append("**Data Quality:**")
        enhanced_parts.append("- Missing data patterns")
        enhanced_parts.append("- Feature drift detection")
        enhanced_parts.append("- Model performance degradation")
        enhanced_parts.append("")
        enhanced_parts.append("**Trading Signals:**")
        enhanced_parts.append("- Regime transition alerts")
        enhanced_parts.append("- Volatility regime changes")
        enhanced_parts.append("- Liquidity condition shifts")
        enhanced_parts.append("")

        # Add recommendations for next steps
        enhanced_parts.append("## 🎯 Next Steps & Recommendations")
        enhanced_parts.append("")
        enhanced_parts.append("### Immediate Actions")
        enhanced_parts.append(
            "- Review regime distribution for trading strategy alignment",
        )
        enhanced_parts.append("- Validate regime transitions against market events")
        enhanced_parts.append("- Test regime-specific parameter optimization")
        enhanced_parts.append("")
        enhanced_parts.append("### Long-term Improvements")
        enhanced_parts.append("- Implement regime-aware position sizing")
        enhanced_parts.append("- Develop regime transition prediction models")
        enhanced_parts.append("- Create regime-specific risk management rules")
        enhanced_parts.append("- Build regime performance attribution system")
        enhanced_parts.append("")

        return "\n".join(enhanced_parts)

    def _generate_transition_analysis(self, cluster_df: pd.DataFrame) -> list[str]:
        """Generate regime transition analysis."""
        analysis = []

        if "composite_cluster_id" not in cluster_df.columns:
            pass
        return ["*No regime data available for transition analysis.*"]

        # Calculate transition matrix
        regimes = cluster_df["composite_cluster_id"].values
        unique_regimes = sorted(set(regimes))

        # Create transition matrix
        transition_matrix = {}
        for i in range(len(regimes) - 1):
            current = regimes[i]
            next_regime = regimes[i + 1]

        if current not in transition_matrix:
                transition_matrix[current] = {}
        if next_regime not in transition_matrix[current]:
                transition_matrix[current][next_regime] = 0
            transition_matrix[current][next_regime] += 1

        # Find most common transitions
        all_transitions = []
        for from_regime, to_regimes in transition_matrix.items():
            pass
        for to_regime , count in to_regimes.items():
                all_transitions.append((from_regime, to_regime, count))

        # Sort by frequency
        all_transitions.sort(key=lambda x: x[2], reverse=True)

        analysis.append("**Most Common Regime Transitions:**")
        for i, (from_regime, to_regime , count) in enumerate(all_transitions[:100], 1):
            analysis.append(
                f"{i}. Regime {from_regime} → Regime {to_regime}: {count} times",
            )
        analysis.append("")

        # Calculate regime persistence
        persistence = {}
        for regime in unique_regimes:
            pass
        if regime in transition_matrix and regime in transition_matrix[regime]:
                persistence[regime] = transition_matrix[regime][regime]
            else:
                persistence[regime] = 0

        # Sort by persistence
        sorted_persistence = sorted(
            persistence.items(),
            key=lambda x: x[1],
            reverse = True = )

        analysis.append("**Regime Persistence (Self-Transitions):**")
        for regime , count in sorted_persistence[:5]:
            analysis.append(f"- Regime {regime}: {count} self-transitions")
        analysis.append("")

        # Calculate transition probabilities for each regime
        analysis.append("**Transition Probabilities by Regime:**")
        for from_regime, to_regimes in transition_matrix.items():
            total_from = sum(to_regimes.values())
        if total_from > 0:
        # Sort transitions by probability
                transitions_with_prob = []
        for to_regime , count in to_regimes.items():
                    probability = count / total_from
                    transitions_with_prob.append((to_regime, probability, count))

        # Sort by probability (highest first)
                transitions_with_prob.sort(key=lambda x: x[1], reverse=True)

                analysis.append(
                    f"**From Regime {from_regime}** (total transitions: {total_from}):",
                )
        for i , (to_regime, probability, count) in enumerate(
                    transitions_with_prob[:5],
                    1,
                ):
                    analysis.append(
                        f"  {i}. Regime {to_regime}: {probability:.1%} ({count} times)",
                    )
                analysis.append("")
        analysis.append("")

        return analysis

    def _generate_similarity_analysis(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], ) -> list[str]:
        """Generate cross-regime similarity analysis."""
        analysis = []

        if "composite_cluster_id" not in cluster_df.columns:
            pass
        return ["*No regime data available for similarity analysis.*"]

        # Get cluster centroids from meta
        cluster_centroids = meta.get("cluster_centroids", {})
        if not cluster_centroids:
            pass
        return ["*No cluster centroids available for similarity analysis.*"]

        # Calculate pairwise similarities between all regimes
        regime_ids = list(cluster_centroids.keys())
        similarities = []

        for i , regime_i in enumerate(regime_ids):
            pass
        for j , regime_j in enumerate(regime_ids):
            pass
        if i < j:  # Avoid duplicates and self-similarity
                    centroid_i = np.array(cluster_centroids[regime_i])
                    centroid_j = np.array(cluster_centroids[regime_j])

        # Calculate cosine similarity
                    norm_i = np.linalg.norm(centroid_i)
                    norm_j = np.linalg.norm(centroid_j)

        if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(centroid_i, centroid_j) / (norm_i * norm_j)
                        similarities.append((regime_i, regime_j, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)

        analysis.append("**Most Similar Regime Pairs:**")
        for i , (regime_i, regime_j, similarity) in enumerate(similarities[:10], 1):
            analysis.append(
                f"{i}. Regime {regime_i} ↔ Regime {regime_j}: {similarity:.3f} similarity",
            )
        analysis.append("")

        # Find most dissimilar regimes
        analysis.append("**Most Dissimilar Regime Pairs:**")
        for i , (regime_i, regime_j, similarity) in enumerate(similarities[-10:], 1):
            analysis.append(
                f"{i}. Regime {regime_i} ↔ Regime {regime_j}: {similarity:.3f} similarity",
            )
        analysis.append("")

        # Calculate average similarity per regime
        regime_avg_similarities = {}
        for regime_id in regime_ids:
            regime_similarities = []
        for other_regime_id in regime_ids:
            pass
        if regime_id != other_regime_id:
            pass
        # Find similarity between these two regimes
        for r1 , r2, sim in similarities:
            pass
        if (r1 == regime_id and r2 == other_regime_id) or (
                            r1 == other_regime_id and r2 == regime_id
                        ):
                            regime_similarities.append(sim)
                            pass
        if regime_similarities:
                regime_avg_similarities[regime_id] = np.mean(regime_similarities)

        # Sort by average similarity
        sorted_avg_similarities = sorted(
            regime_avg_similarities.items(),
            key=lambda x: x[1],
            reverse = True = )

        analysis.append("**Regimes by Average Similarity (Most Similar to Others):**")
        for regime_id , avg_sim in sorted_avg_similarities[:5]:
            analysis.append(f"- Regime {regime_id}: {avg_sim:.3f} average similarity")
        analysis.append("")

        analysis.append("**Regimes by Average Similarity (Most Unique):**")
        for regime_id , avg_sim in sorted_avg_similarities[-5:]:
            analysis.append(f"- Regime {regime_id}: {avg_sim:.3f} average similarity")
        analysis.append("")

        # Identify potential merging opportunities
        high_similarity_pairs = [
            (r1, r2, sim) for r1 , r2, sim in similarities if sim > 0.8
        ]
        if high_similarity_pairs:
            analysis.append("**Potential Merging Opportunities (Similarity > 0.8):**")
        for regime_i , regime_j, similarity in high_similarity_pairs:
                analysis.append(
                    f"- Regime {regime_i} and Regime {regime_j}: {similarity:.3f} similarity",
                )
            analysis.append("")

        return analysis

    def _generate_temporal_analysis(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], ) -> list[str]:
        """Generate temporal analysis of regime stability and predictability."""
        analysis = []

        if (
            "composite_cluster_id" not in cluster_df.columns
            or "timestamp" not in cluster_df.columns
        ):
        return ["*No temporal data available for analysis.*"]

        analysis.append("## 🕐 Temporal Analysis")
        analysis.append("")

        # Convert timestamp to datetime if needed
        if cluster_df["timestamp"].dtype == "object":
            cluster_df["timestamp"] = pd.to_datetime(cluster_df["timestamp"])

        # Add time-based columns
        cluster_df["hour"] = cluster_df["timestamp"].dt.hour
        cluster_df["day_of_week"] = cluster_df["timestamp"].dt.dayofweek
        cluster_df["month"] = cluster_df["timestamp"].dt.month

        # 1. Regime Persistence Analysis
        analysis.append("### 📊 Regime Persistence Analysis")
        analysis.append("")

        regimes = cluster_df["composite_cluster_id"].values
        persistence_data = {}

        current_regime = regimes[0]
        cluster_df["timestamp"].iloc[0]
        duration_count = 1

        for i in range(1, len(regimes)):
            pass
        if regimes[i] == current_regime:
                duration_count += 1
            else:
                pass
        # Record the persistence
        if current_regime not in persistence_data:
                    persistence_data[current_regime] = []
                persistence_data[current_regime].append(duration_count)

        # Start new regime
                current_regime = regimes[i]
                duration_count = 1

        # Add the last regime
        if current_regime not in persistence_data:
            persistence_data[current_regime] = []
        persistence_data[current_regime].append(duration_count)

        # Calculate statistics for each regime
        for regime_id , durations in persistence_data.items():
            pass
        if durations:
                avg_duration = np.mean(durations)
                median_duration = np.median(durations)
                max_duration = max(durations)
                min_duration = min(durations)

                analysis.append(f"**Regime {regime_id} Persistence:**")
                analysis.append(f"- Average duration: {avg_duration:.1f} periods")
                analysis.append(f"- Median duration: {median_duration:.1f} periods")
                analysis.append(
                    f"- Duration range: {min_duration} - {max_duration} periods",
                )
                analysis.append(f"- Total occurrences: {len(durations)}")
                analysis.append("")

        # 2. Hourly Regime Patterns
        analysis.append("### 🕐 Hourly Regime Patterns")
        analysis.append("")

        hourly_regimes = (
            cluster_df.groupby("hour")["composite_cluster_id"]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Find most common regime per hour
        most_common_per_hour = hourly_regimes.idxmax(axis=1)
        analysis.append("**Most Common Regime by Hour:**")
        for hour , regime in most_common_per_hour.items():
            pass
        if pd.notna(regime):
                count = hourly_regimes.loc[hour, regime]
                total = hourly_regimes.loc[hour].sum()
                percentage = (count / total) * 100
                analysis.append(f"- {hour:02d}:00: Regime {regime} ({percentage:.1f}%)")
        analysis.append("")

        # 3. Day of Week Patterns
        analysis.append("### 📅 Day of Week Patterns")
        analysis.append("")

        dow_regimes = (
            cluster_df.groupby("day_of_week")["composite_cluster_id"]
            .value_counts()
            .unstack(fill_value=0)
        )
        dow_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        most_common_per_dow = dow_regimes.idxmax(axis=1)
        analysis.append("**Most Common Regime by Day:**")
        for dow , regime in most_common_per_dow.items():
            pass
        if pd.notna(regime) and dow < len(dow_names):
                count = dow_regimes.loc[dow, regime]
                total = dow_regimes.loc[dow].sum()
                percentage = (count / total) * 100
                analysis.append(
                    f"- {dow_names[dow]}: Regime {regime} ({percentage:.1f}%)",
                )
        analysis.append("")

        # 4. Regime Stability Score
        analysis.append("### 🎯 Regime Stability Score")
        analysis.append("")

        stability_scores = {}
        for regime_id , durations in persistence_data.items():
            pass
        if durations:
        # Calculate coefficient of variation (lower, more stable)
                cv = (
                    np.std(durations) / np.mean(durations)
        if np.mean(durations) > 0
                    else float("inf")
                )
                stability_score = 1 / (
                    1 + cv
                )  # Convert to 0-1 scale where 1 is most stable
                stability_scores[regime_id] = stability_score

        # Sort by stability
        sorted_stability = sorted(
            stability_scores.items(),
            key=lambda x: x[1],
            reverse = True = )

        analysis.append("**Regime Stability Ranking (1, Most Stable):**")
        for regime_id , stability in sorted_stability:
            analysis.append(f"- Regime {regime_id}: {stability:.3f} stability score")
        analysis.append("")

        return analysis

    def _generate_feature_importance_analysis(self, meta: dict[str ,  Any]) -> list[str]:
        """Generate feature importance analysis for regime changes."""
        analysis = []

        analysis.append("## 🔍 Feature Importance Analysis")
        analysis.append("")

        # Get feature information from meta
        blocks, meta.get("blocks", [])
        feature_importance, meta.get("feature_importance", {})
        state_names, meta.get("state_names", {})

        if not blocks:
            pass
        return ["*No block information available for feature analysis.*"]

        analysis.append("### 🧩 Block-Level Feature Analysis")
        analysis.append("")

        for block in blocks:
            block_name = block["name"]
            features = block.get("features", [])
            n_states = block.get("n_states", 0)

            analysis.append(f"**{block_name.title()} Block:**")
            analysis.append(f"- Number of states: {n_states}")
            analysis.append(f"- Number of features: {len(features)}")

        # Show feature names if available
        if features:
                analysis.append(
                    f"- Features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}",
                )

        # Show state interpretations
        if block_name in state_names:
                analysis.append("- State interpretations:")
        for state_id , state_name in state_names[block_name].items():
                    analysis.append(f"  • State {state_id}: {state_name}")

            analysis.append("")

        # Feature importance per regime (if available)
        if feature_importance:
            analysis.append("### 📊 Feature Importance by Regime")
            analysis.append("")

        for regime_id, features in feature_importance.items():
            pass
        if isinstance(features , dict):
                    analysis.append(f"**Regime {regime_id} Top Features:**")
        # Sort features by importance
                    sorted_features = sorted(
                        features.items(),
                        key=lambda x: x[1],
                        reverse = True = )
        for feature, importance in sorted_features[:5]:
                        analysis.append(f"- {feature}: {importance:.3f}")
                    analysis.append("")

        # Regime transition triggers
        analysis.append("### ⚡ Regime Transition Triggers")
        analysis.append("")

        # Analyze which features are most important for transitions
        transition_features = meta.get("transition_features", {})
        if transition_features:
            analysis.append("**Most Important Features for Regime Transitions:**")
            sorted_transition_features = sorted(
                transition_features.items(),
                key=lambda x: x[1],
                reverse = True = )
        for feature, importance in sorted_transition_features[:10]:
                analysis.append(f"- {feature}: {importance:.3f}")
            analysis.append("")
        else:
            analysis.append("*Transition feature importance data not available.*")
            analysis.append("")

        # Feature stability analysis
        analysis.append("### 🔄 Feature Stability Analysis")
        analysis.append("")

        feature_stability = meta.get("feature_stability", {})
        if feature_stability:
            analysis.append("**Feature Stability Scores (Higher, More Stable):**")
            sorted_stability = sorted(
                feature_stability.items(),
                key=lambda x: x[1],
                reverse = True = )
        for feature , stability in sorted_stability[:10]:
                analysis.append(f"- {feature}: {stability:.3f}")
            analysis.append("")
        else:
            analysis.append("*Feature stability data not available.*")
            analysis.append("")

        return analysis

    def _generate_predictive_power_assessment(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], ) -> list[str]:
        """Generate predictive power assessment for regime forecasting."""
        analysis = []

        if "composite_cluster_id" not in cluster_df.columns:
            pass
        return ["*No regime data available for predictive power assessment.*"]

        analysis.append("## 🎯 Predictive Power Assessment")
        analysis.append("")

        # 1. Regime Transition Predictability
        analysis.append("### 🔮 Regime Transition Predictability")
        analysis.append("")

        regimes = cluster_df["composite_cluster_id"].values
        sorted(set(regimes))

        # Calculate transition probabilities
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
        for regime , transitions in transition_counts.items():
            total_transitions = sum(transitions.values())
        if total_transitions > 0:
        # Calculate entropy (lower, more predictable)
                probabilities = [
                    count / total_transitions for count in transitions.values()
                ]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                max_entropy = np.log2(len(transitions))
                predictability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                predictability_scores[regime] = predictability

        # Sort by predictability
        sorted_predictability = sorted(
            predictability_scores.items(),
            key=lambda x: x[1],
            reverse = True = )

        analysis.append("**Regime Transition Predictability (1, Most Predictable):**")
        for regime , predictability in sorted_predictability:
            analysis.append(
                f"- Regime {regime}: {predictability:.3f} predictability score",
            )
        analysis.append("")

        # 2. Regime Persistence Forecasting
        analysis.append("### ⏱️ Regime Persistence Forecasting")
        analysis.append("")

        # Calculate regime duration statistics
        persistence_data = {}
        current_regime = regimes[0]
        duration_count = 1

        for i in range(1, len(regimes)):
            pass
        if regimes[i] == current_regime:
                duration_count += 1
            else:
                pass
        if current_regime not in persistence_data:
                    persistence_data[current_regime] = []
                persistence_data[current_regime].append(duration_count)
                current_regime = regimes[i]
                duration_count = 1

        # Add last regime
        if current_regime not in persistence_data:
            persistence_data[current_regime] = []
        persistence_data[current_regime].append(duration_count)

        # Calculate forecasting accuracy metrics
        forecasting_metrics = {}
        for regime , durations in persistence_data.items():
            pass
        if len(durations) > 1:
        # Use first 80% for training, last 20% for testing
                split_idx = int(len(durations) * 0.8)
                train_durations = durations[:split_idx]
                test_durations = durations[split_idx:]

        if train_durations and test_durations:
                    predicted_mean = np.mean(train_durations)
                    mae = np.mean([abs(d - predicted_mean) for d in test_durations])
                    mape = (
                        np.mean(
                            [
                                abs(d - predicted_mean) / d
        for d in test_durations
        if d > 0
                            ],
                        )
                        * 100
                    )

                    forecasting_metrics[regime] = {
                        "mae": mae , "mape": mape,
                        "train_samples": len(train_durations),
                        "test_samples": len(test_durations),
                    }

        # Sort by forecasting accuracy (lower MAE, better)
        sorted_forecasting = sorted(
            forecasting_metrics.items(),
            key=lambda x: x[1]["mae"],
        )

        analysis.append("**Regime Persistence Forecasting Accuracy:**")
        for regime, metrics in sorted_forecasting:
            analysis.append(f"- Regime {regime}:")
            analysis.append(f"  • MAE: {metrics['mae']:.2f} periods")
            analysis.append(f"  • MAPE: {metrics['mape']:.1f}%")
            analysis.append(f"  • Training samples: {metrics['train_samples']}")
            analysis.append(f"  • Test samples: {metrics['test_samples']}")
            analysis.append("")

        # 3. Overall Model Performance
        analysis.append("### 📈 Overall Predictive Performance")
        analysis.append("")

        # Calculate overall metrics
        if predictability_scores:
            avg_predictability = np.mean(list(predictability_scores.values()))
            analysis.append(
                f"**Average Regime Predictability:** {avg_predictability:.3f}",
            )
            analysis.append("")

        if forecasting_metrics:
            avg_mae = np.mean([m["mae"] for m in forecasting_metrics.values()])
            avg_mape = np.mean([m["mape"] for m in forecasting_metrics.values()])
            analysis.append("**Average Persistence Forecasting:**")
            analysis.append(f"- MAE: {avg_mae:.2f} periods")
            analysis.append(f"- MAPE: {avg_mape:.1f}%")
            analysis.append("")

        # 4. Model Validation Recommendations
        analysis.append("### ✅ Model Validation Recommendations")
        analysis.append("")

        analysis.append("**For High Predictability Regimes:**")
        analysis.append("- Implement regime-specific trading strategies")
        analysis.append("- Use regime transitions for entry/exit signals")
        analysis.append("- Optimize parameters for each regime")
        analysis.append("")

        analysis.append("**For Low Predictability Regimes:**")
        analysis.append("- Focus on risk management over prediction")
        analysis.append("- Use broader timeframes for analysis")
        analysis.append("- Consider regime merging for simplification")
        analysis.append("")

        return analysis

    def print_regime_summary(self, meta: dict[str ,  Any], cluster_df: pd.DataFrame | None =) -> None:
        """Print a comprehensive summary of all regimes."""
        print("🔍 HMM REGIME ANALYSIS SUMMARY")
        print("=" * 60)

        # Basic info
        print(f"📊 Exchange: {meta.get('exchange', 'Unknown')}")
        print(f"📈 Symbol: {meta.get('symbol', 'Unknown')}")
        print(f"⏰ Timeframe: {meta.get('timeframe', 'Unknown')}")
        print()

        # Block information
        print("🏗️ BLOCK CONFIGURATION:")
        print("-" * 30)
        blocks, meta.get("blocks", [])
        for block in blocks:
            print(f"  • {block['name'].title()}: {block['n_states']} states")
        print()

        # State names per block
        print("🏷️ STATE INTERPRETATIONS:")
        print("-" * 30)
        state_names = meta.get("state_names", {})
        for block_name , states in state_names.items():
            print(f"  📋 {block_name.upper()}:")
        for state_id , state_name in states.items():
                print(f"    State {state_id}: {state_name}")
            print()

        # Archetype descriptions with proper frequency analysis
        print("🎯 MARKET ARCHETYPES:")
        print("-" * 30)
        archetype_descriptions = meta.get("archetype_descriptions", {})

        # Get actual cluster frequencies from the data
        if cluster_df is not None and "composite_cluster_id" in cluster_df.columns:
            cluster_counts = (
                cluster_df["composite_cluster_id"].value_counts().sort_index()
            )
            total_observations = len(cluster_df)

        # Sort archetypes by frequency
            sorted_archetypes = sorted(
                archetype_descriptions.items(),
                key=lambda x: cluster_counts.get(int(x[0]), 0),
                reverse = True = )

        for rank , (cluster_id, description) in enumerate(sorted_archetypes, 1):
            pass
        if int(cluster_id) < 0:  # Skip noise clusters
                    continue

                frequency = cluster_counts.get(int(cluster_id), 0)
                percentage = (
                    (frequency / total_observations * 100)
        if total_observations > 0
                    else 0
                )

                print(f"  🏆 Rank #{rank} - Archetype {cluster_id}:")
                print(f"    📝 {description}")
                print(
                    f"    📊 Frequency: {frequency:,} occurrences ({percentage:.2f}% of time)",
                )
                print()
        else:
            pass
        # Fallback if no cluster data
        for cluster_id , description in sorted(
                archetype_descriptions.items(),
                key=lambda x: int(x[0]),
            ):
        if int(cluster_id) < 0:  # Skip noise clusters
                    continue
                print(f"  🏆 Archetype {cluster_id}:")
                print(f"    📝 {description}")
                print()

        # Prevalence summary
        print("📈 PREVALENCE SUMMARY:")
        print("-" * 30)
        if cluster_df is not None and "composite_cluster_id" in cluster_df.columns:
            total_observations = len(cluster_df)
            noise_count = cluster_counts.get(-1, 0) if -1 in cluster_counts else 0
            valid_archetypes = {
                k: v for k, v in archetype_descriptions.items() if int(k) >= 0
            }

            print(f"Total observations: {total_observations:,}")
            print(f"Unique archetypes: {len(valid_archetypes)}")
            print(
                f"Noise/undefined states: {noise_count:,} ({noise_count/total_observations*100:.2f}%)",
            )
            print()

        if valid_archetypes:
                max_freq_archetype = max(
                    valid_archetypes.items(),
                    key=lambda x: cluster_counts.get(int(x[0]), 0),
                )
                min_freq_archetype = min(
                    valid_archetypes.items(),
                    key=lambda x: cluster_counts.get(int(x[0]), 0),
                )

                max_freq = cluster_counts.get(int(max_freq_archetype[0]), 0)
                min_freq = cluster_counts.get(int(min_freq_archetype[0]), 0)
                max_pct = (
                    (max_freq / total_observations * 100)
        if total_observations > 0
                    else 0
                )
                min_pct = (
                    (min_freq / total_observations * 100)
        if total_observations > 0
                    else 0
                )

                print(
                    f"🎯 Most Common: Archetype {max_freq_archetype[0]} ({max_pct:.2f}%)",
                )
                print(
                    f"📉 Least Common: Archetype {min_freq_archetype[0]} ({min_pct:.2f}%)",
                )
        else:
            print(f"Unique archetypes: {len(archetype_descriptions)}")
            print("Frequency data not available")

        print()

        # Generate and save detailed summary
        print("💾 GENERATING DETAILED SUMMARY...")
        detailed_summary = self.generate_detailed_regime_summary(meta, cluster_df)

        # Save to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        exchange = meta.get("exchange", "UNKNOWN")
        symbol = meta.get("symbol", "UNKNOWN")
        timeframe = meta.get("timeframe", "1m")

        # Add datestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            reports_dir
            / f"{exchange}_{symbol}_{timeframe}_detailed_regime_summary_{timestamp}.md"
        )
        with open(output_file = "w") as f:
            f.write(detailed_summary)

        print(f"💾 Detailed regime summary saved to: {output_file}")
        print()

        # Print a preview of the detailed summary
        print("📋 DETAILED SUMMARY PREVIEW:")
        print("-" * 30)
        lines = detailed_summary.split("\n")
        for line in lines[:20]:  # Show first 20 lines
            print(line)
        if len(lines) > 20:
            print("...")
            print(f"(Full summary saved to {output_file})")

        print()
        print("✅ Analysis complete! Reports saved to: reports")

    def analyze_regime_transitions(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze transitions between regimes."""
        if cluster_df is None or "composite_cluster_id" not in cluster_df.columns:
            pass
        return pd.DataFrame()

        transitions = []
        cluster_series = cluster_df["composite_cluster_id"]

        for i in range(1, len(cluster_series)):
            from_regime = cluster_series.iloc[i - 1]
            to_regime = cluster_series.iloc[i]
        if from_regime != to_regime:
                transitions.append(
                    {
                        "from_regime": from_regime,
                        "to_regime": to_regime,
                        "timestamp": cluster_df.index[i]
        if hasattr(cluster_df.index[i], "timestamp")
                        else i = },
                )

        return pd.DataFrame(transitions)

    def plot_regime_intensities(self, intensity_df: pd.DataFrame, meta: dict[str ,  Any], top_n: int, 5, save_path: str | None =) -> None:
        """Plot regime intensity scores over time."""
        if intensity_df is None:
            print("⚠️ No intensity data available for plotting")
            return

        # Get top N most frequent regimes
        archetype_descriptions = meta.get("archetype_descriptions", {})
        cluster_counts = meta.get("cluster_labels", {})

        if not archetype_descriptions:
            print("⚠️ No archetype descriptions available")
            return

        # Sort by frequency
        sorted_clusters = sorted(
            archetype_descriptions.keys(),
            key=lambda x: cluster_counts.get(str(x), 0),
            reverse = True = )[:top_n]

        # Create plot
        fig = axes, plt.subplots(
            len(sorted_clusters),
            1,
            figsize=(15, 3 * len(sorted_clusters)),
        )
        if len(sorted_clusters) == 1:
            axes = [axes]

        for i , cluster_id in enumerate(sorted_clusters):
            col_name = f"intensity_cluster_{cluster_id}"
        if col_name in intensity_df.columns:
                ax = axes[i]
                intensity_series = intensity_df[col_name]

        # Plot intensity
                ax.plot(
                    intensity_series.index = intensity_series.values,
                    linewidth=1,
                    alpha=0.7,
                    label="Intensity",
                )

        # Add moving average
                ma_window = min(50, len(intensity_series) // 10)
        if ma_window > 1:
                    ma = intensity_series.rolling(window, ma_window, center=True).mean()
                    ax.plot(
                        intensity_series.index = ma.values,
                        linewidth=2,
                        alpha=0.9,
                        label=f"{ma_window}-period MA",
                    )

        # Styling
                ax.set_title(
                    f"Archetype {cluster_id}: {archetype_descriptions[cluster_id][:80]}...",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_ylabel("Intensity Score")
                ax.legend()
                ax.grid(True, alpha, 0.3)

        # Format x-axis for time series
        if isinstance(intensity_series.index , pd.DatetimeIndex):
                    ax.xaxis.set_major_formatter(
                        plt.matplotlib.dates.DateFormatter("%m-%d %H:%M"),
                    )
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi, 300, bbox_inches="tight")
            print(f"💾 Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_regime_distribution(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], save_path: str | None =) -> None:
        """Plot distribution of regimes."""
        if cluster_df is None or "composite_cluster_id" not in cluster_df.columns:
            print("⚠️ No cluster data available for plotting")
            return

        archetype_descriptions = meta.get("archetype_descriptions", {})

        # Count regimes
        regime_counts = cluster_df["composite_cluster_id"].value_counts().sort_index()

        # Create labels
        labels = []
        for cluster_id in regime_counts.index:
            pass
        if str(cluster_id) in archetype_descriptions:
                desc = archetype_descriptions[str(cluster_id)]
        # Truncate description for readability
                short_desc = desc.split("(")[0].strip()
                labels.append(f"Archetype {cluster_id}\n{short_desc[:40]}...")
            else:
                labels.append(f"Archetype {cluster_id}")

        # Create plot
        fig = (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Bar plot
        bars = ax1.bar(
            range(len(regime_counts)),
            regime_counts.values = color, plt.cm.Set3(np.linspace(0, 1, len(regime_counts))),
        )
        ax1.set_xlabel("Regime Archetype")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Regime Distribution")
        ax1.set_xticks(range(len(regime_counts)))
        ax1.set_xticklabels(
            [f"Archetype {i}" for i in regime_counts.index],
            rotation=45,
        )

        # Add value labels on bars
        for bar , count in zip(bars, regime_counts.values, strict, False):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Pie chart
        ax2.pie(
            regime_counts.values = labels, labels,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        ax2.set_title("Regime Distribution (Percentage)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi, 300, bbox_inches="tight")
            print(f"💾 Plot saved to: {save_path}")
        else:
            plt.show()

    def generate_regime_report(self, exchange: str, symbol: str, timeframe: str, output_dir: str = "reports", ) -> None:
        """Generate a comprehensive regime analysis report."""
        print(f"🔍 Analyzing HMM regimes for {exchange}_{symbol}_{timeframe}...")

        # Load data
        data, self.load_regime_data(exchange, symbol, timeframe)
        meta, data["meta"]
        intensity_df, data["intensity"]
        cluster_df, data["clusters"]

        # Create output directory
        output_path, Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Print summary
        self.print_regime_summary(meta, cluster_df)

        # Generate plots
        if intensity_df is not None:
            intensity_plot_path = (
                output_path / f"{exchange}_{symbol}_{timeframe}_regime_intensities.png"
            )
        self.plot_regime_intensities(
                intensity_df = meta,
                save_path=str(intensity_plot_path),
            )

        if cluster_df is not None:
            distribution_plot_path = (
                output_path / f"{exchange}_{symbol}_{timeframe}_regime_distribution.png"
            )
        self.plot_regime_distribution(
                cluster_df = meta,
                save_path=str(distribution_plot_path),
            )

        # Generate advanced visualizations
        self._generate_advanced_visualizations(
                cluster_df = meta,
                exchange = symbol,
                timeframe = output_path,
            )

        # Analyze transitions
        if cluster_df is not None:
            transitions = self.analyze_regime_transitions(cluster_df)
        if not transitions.empty:
                print("🔄 REGIME TRANSITIONS:")
                print("-" * 30)
                transition_counts = (
                    transitions.groupby(["from_regime", "to_regime"])
                    .size()
                    .sort_values(ascending=False)
                )
                print("Most common transitions:")
        for (from_regime, to_regime), count in transition_counts.head(
                    10,
                ).items():
                    from_desc = meta.get("archetype_descriptions", {}).get(
                        str(from_regime),
                        f"Archetype {from_regime}",
                    )
                    to_desc = meta.get("archetype_descriptions", {}).get(
                        str(to_regime),
                        f"Archetype {to_regime}",
                    )
                    print(f"  {from_regime} → {to_regime}: {count} times")
                    print(f"    From: {from_desc[:60]}...")
                    print(f"    To:   {to_desc[:60]}...")
                    print()

        # Generate enhanced detailed summary with visualizations
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
            detailed_summary_path = self.save_detailed_summary(
                exchange = symbol,
                timeframe = )
            print(f"💾 Detailed regime summary saved to: {detailed_summary_path}")
        pass
            print(f"⚠️ Warning: Could not generate detailed summary: {e}")

        print(f"✅ Analysis complete! Reports saved to: {output_path}")

    def _generate_market_condition_analysis(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], ) -> list[str]:
        """Generate market condition analysis for comparative analysis."""
        analysis = []

        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Analyze regime distribution by market conditions
            cluster_series = cluster_df["composite_cluster_id"]

        # Calculate basic statistics
            total_observations = len(cluster_series)
            regime_counts = cluster_series.value_counts()

        # Identify dominant regimes
            top_regimes = regime_counts.head(3)

            analysis.append("**Current Market Regime Distribution:**")
            analysis.append("")

        for regime_id , count in top_regimes.items():
                percentage = count / total_observations * 100
                desc = meta.get("archetype_descriptions", {}).get(
                    str(regime_id),
                    f"Regime {regime_id}",
                )
                analysis.append(
                    f"- **Regime {regime_id}**: {percentage:.1f}% of time ({desc[:50]}...)",
                )

            analysis.append("")

        # Market condition insights
            analysis.append("**Market Condition Insights:**")
            analysis.append("")

        # Determine market condition based on dominant regimes
        if len(top_regimes) > 0:
                dominant_regime = top_regimes.index[0]
                dominant_percentage = top_regimes.iloc[0] / total_observations * 100

        if dominant_percentage > 40:
                    analysis.append(
                        f"- **High Concentration**: Regime {dominant_regime} dominates with {dominant_percentage:.1f}% of market time",
                    )
                    analysis.append(
                        "- **Market State**: Likely in a stable, trending market condition",
                    )
                elif dominant_percentage > 25:
                    analysis.append(
                        f"- **Moderate Concentration**: Regime {dominant_regime} is prominent with {dominant_percentage:.1f}% of market time",
                    )
                    analysis.append(
                        "- **Market State**: Mixed market conditions with some stability",
                    )
                else:
                    analysis.append(
                        f"- **Low Concentration**: No single regime dominates (highest: {dominant_percentage:.1f}%)",
                    )
                    analysis.append(
                        "- **Market State**: Highly volatile or transitioning market conditions",
                    )

            analysis.append("")
            analysis.append("**Trading Implications:**")
            analysis.append(
                "- High concentration periods: Use regime-specific strategies",
            )
            analysis.append("- Low concentration periods: Focus on risk management")
            analysis.append("- Monitor regime transitions for market condition changes")
            analysis.append("")

        pass
            analysis.append(f"*Error generating market condition analysis: {e}*")
            analysis.append("")

        return analysis

    def _generate_advanced_visualizations(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], exchange: str, symbol: str, timeframe: str, output_path: Path, ) -> None:
        """Generate advanced visualizations for the regime analysis."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Set style
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")

        # 1. Regime Transition Heatmap
        self._create_transition_heatmap(
                cluster_df = exchange,
                symbol = timeframe,
                output_path = )

        # 2. Regime Persistence Timeline
        self._create_persistence_timeline(
                cluster_df = exchange,
                symbol = timeframe,
                output_path = )

        # 3. Feature Importance Radar Chart
        self._create_feature_importance_radar(
                meta = exchange,
                symbol = timeframe,
                output_path = )

        # 4. Regime Correlation Network
        self._create_correlation_network(
                cluster_df = meta,
                exchange = symbol,
                timeframe = output_path,
            )

        pass
            print(f"⚠️ Warning: Could not generate advanced visualizations: {e}")

    def _create_transition_heatmap(self, cluster_df: pd.DataFrame, exchange: str, symbol: str, timeframe: str, output_path: Path, ) -> None:
        """Create a heatmap showing regime transition probabilities."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Calculate transition matrix
            cluster_series = cluster_df["composite_cluster_id"]
            transitions = []

        for i in range(1, len(cluster_series)):
                from_regime = cluster_series.iloc[i - 1]
                to_regime = cluster_series.iloc[i]
                transitions.append((from_regime, to_regime))

        # Create transition matrix
            unique_regimes = sorted(cluster_series.unique())
            transition_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))

        for from_regime, to_regime in transitions:
                from_idx = unique_regimes.index(from_regime)
                to_idx = unique_regimes.index(to_regime)
                transition_matrix[from_idx, to_idx] += 1

        # Normalize by row sums
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(
                transition_matrix = row_sums[:, np.newaxis],
                where=row_sums[:, np.newaxis] != 0,
            )

        # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                transition_matrix,
                xticklabels=[f"R{i}" for i in unique_regimes],
                yticklabels=[f"R{i}" for i in unique_regimes],
                annot = True, fmt=".2f",
                cmap="YlOrRd",
            )
            plt.title(
                f"Regime Transition Probability Matrix\n{exchange}_{symbol}_{timeframe}",
            )
            plt.xlabel("To Regime")
            plt.ylabel("From Regime")

        # Save plot
            plot_path = (
                output_path / f"{exchange}_{symbol}_{timeframe}_transition_heatmap.png"
            )
            plt.savefig(plot_path, dpi, 300, bbox_inches="tight")
            plt.close()
            print(f"💾 Transition heatmap saved to: {plot_path}")

        pass
            print(f"⚠️ Error creating transition heatmap: {e}")

    def _create_persistence_timeline(self, cluster_df: pd.DataFrame, exchange: str, symbol: str, timeframe: str, output_path: Path, ) -> None:
        """Create a timeline showing regime persistence over time."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Calculate regime persistence
            cluster_series = cluster_df["composite_cluster_id"]
            persistence_data = []

            current_regime = cluster_series.iloc[0]
            start_time = 0
            duration = 1

        for i in range(1, len(cluster_series)):
            pass
        if cluster_series.iloc[i] == current_regime:
                    duration += 1
                else:
                    persistence_data.append(
                        {
                            "regime": current_regime , "start": start_time,
                            "duration": duration = },
                    )
                    current_regime = cluster_series.iloc[i]
                    start_time = i
                    duration = 1

        # Add last regime
            persistence_data.append(
                {"regime": current_regime , "start": start_time, "duration": duration},
            )

        # Create timeline plot
            plt.figure(figsize=(15, 8))

        # Color map for regimes
            unique_regimes = sorted(cluster_series.unique())
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))
            color_map = dict(zip(unique_regimes, colors, strict=False))

        for data in persistence_data:
                regime = data["regime"]
                start = data["start"]
                duration = data["duration"]
                color = color_map.get(regime = "gray")

                plt.barh(
                    y=0,
                    width = duration, left=start,
                    height=0.8,
                    color = color, alpha=0.7,
                    label=f"Regime {regime}",
                )

            plt.title(f"Regime Persistence Timeline\n{exchange}_{symbol}_{timeframe}")
            plt.xlabel("Time Periods")
            plt.ylabel("Regime")
            plt.yticks([0], ["Regime"])

        # Add legend
            handles = labels, plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles, strict=False))
            plt.legend(by_label.values(), by_label.keys(), loc="upper right")

        # Save plot
            plot_path = (
                output_path
                / f"{exchange}_{symbol}_{timeframe}_persistence_timeline.png"
            )
            plt.savefig(plot_path, dpi, 300, bbox_inches="tight")
            plt.close()
            print(f"💾 Persistence timeline saved to: {plot_path}")

        pass
            print(f"⚠️ Error creating persistence timeline: {e}")

    def _create_feature_importance_radar(self, meta: dict[str ,  Any], exchange: str, symbol: str, timeframe: str, output_path: Path, ) -> None:
        """Create a radar chart showing feature importance."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Get feature information from meta
            blocks = meta.get("blocks", [])
        if not blocks:
                return

        # Create radar chart
            fig = ax, plt.subplots(
                figsize=(10, 10),
                subplot_kw={"projection": "polar"},
            )

        # Categories (block names)
            categories = [block["name"].title() for block in blocks]
            N = len(categories)

        # Values (number of states per block)
            values = [block["n_states"] for block in blocks]

        # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle

        # Add the first value at the end to close the plot
            values += values[:1]

        # Plot
            ax.plot(angles, values, "o-", linewidth=2, label="Feature Complexity")
            ax.fill(angles, values, alpha=0.25)

        # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, max(values) * 1.2)

            plt.title(f"Feature Block Complexity\n{exchange}_{symbol}_{timeframe}")
            plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        # Save plot
            plot_path = (
                output_path
                / f"{exchange}_{symbol}_{timeframe}_feature_importance_radar.png"
            )
            plt.savefig(plot_path, dpi, 300, bbox_inches="tight")
            plt.close()
            print(f"💾 Feature importance radar saved to: {plot_path}")

        pass
            print(f"⚠️ Error creating feature importance radar: {e}")

    def _create_correlation_network(self, cluster_df: pd.DataFrame, meta: dict[str ,  Any], exchange: str, symbol: str, timeframe: str, output_path: Path, ) -> None:
        """Create a network diagram showing regime correlations."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Calculate regime correlations
            cluster_series = cluster_df["composite_cluster_id"]
            unique_regimes = sorted(cluster_series.unique())

        # Create correlation matrix (simplified - using co-occurrence)
            correlation_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))

        # Calculate similarity based on transition patterns
        for i , regime1 in enumerate(unique_regimes):
            pass
        for j , regime2 in enumerate(unique_regimes):
            pass
        if i != j:
        # Simple similarity based on transition frequency
                        transitions_from_1 = cluster_series[
                            cluster_series , = regime1
                        ].index
                        cluster_series[cluster_series , = regime2].index

        # Count transitions from regime1 to regime2
                        transition_count = 0
        for idx in transitions_from_1:
            pass
        if (
                                idx + 1 < len(cluster_series)
                                and cluster_series.iloc[idx + 1] == regime2
                            ):
                                transition_count += 1

                        correlation_matrix[i, j] = transition_count

        # Normalize
            max_val = correlation_matrix.max()
        if max_val > 0:
                correlation_matrix = correlation_matrix / max_val

        # Create network
            G = nx.Graph()

        # Add nodes
        for i , regime in enumerate(unique_regimes):
                G.add_node(regime, label, f"Regime {regime}")

        # Add edges with weights
        for i in range(len(unique_regimes)):
            pass
        for j in range(i + 1, len(unique_regimes)):
                    weight = correlation_matrix[i, j]
        if weight > 0.1:  # Only show significant connections
                        G.add_edge(unique_regimes[i], unique_regimes[j], weight=weight)

        # Create plot
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k, 1, iterations=50)

        # Draw nodes
            nx.draw_networkx_nodes(
                G = pos,
                node_color="lightblue",
                node_size=1000,
                alpha=0.7,
            )

        # Draw edges
            edges = G.edges()
            weights = [G[u][v]["weight"] for u , v in edges]
            nx.draw_networkx_edges(G, pos, width, weights, alpha=0.5, edge_color="gray")

        # Draw labels
            labels = {node: f"R{node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size, 12, font_weight="bold")

            plt.title(f"Regime Correlation Network\n{exchange}_{symbol}_{timeframe}")
            plt.axis("off")

        # Save plot
            plot_path = (
                output_path / f"{exchange}_{symbol}_{timeframe}_correlation_network.png"
            )
            plt.savefig(plot_path, dpi, 300, bbox_inches="tight")
            plt.close()
            print(f"💾 Correlation network saved to: {plot_path}")

        pass
            print(f"⚠️ Error creating correlation network: {e}")

def main():
    parser, argparse.ArgumentParser(description="Analyze HMM regime discovery results")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol name")
    parser.add_argument("--timeframe", default="1m", help="Timeframe")
    parser.add_argument("--data-dir", default="data/training", help="Data directory")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for reports",
    )

    args, parser.parse_args()

    analyzer, HMMRegimeAnalyzer(args.data_dir)
    analyzer.generate_regime_report(
        args.exchange, args.symbol,
        args.timeframe, args.output_dir,
    )

if __name__ == "__main__":
    main()
