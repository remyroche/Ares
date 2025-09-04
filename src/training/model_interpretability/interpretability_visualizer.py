#!/usr/bin/env python3
"""Interpretability Visualizer for Model Analysis.

This module provides visualization capabilities for model interpretability results
including SHAP and LIME analysis outputs.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import time

from src.core.decorators import handles_errors, validates, log_call, traced
from src.utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_json_dump, safe_json_load, safe_file_exists,
    timed_operation, format_bytes, safe_log_metric, safe_log_params
)
from src.utils.logger import system_logger

class InterpretabilityVisualizer:
    """Visualizer for model interpretability results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the interpretability visualizer."""
        self.config = config
        self.logger = system_logger.getChild("InterpretabilityVisualizer")
        self.matplotlib_available = False
        self.seaborn_available = False
        
        # Check visualization libraries availability
        self._check_visualization_libraries()
    
    def _check_visualization_libraries(self):
        """Check if visualization libraries are available."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            self.plt = plt
            self.matplotlib = matplotlib
            self.matplotlib_available = True
            self.logger.info("✅ Matplotlib available for visualization")
            print("✅ Matplotlib available for visualization")
        except ImportError:
            self.logger.warning("⚠️ Matplotlib not available - install with: pip install matplotlib")
            print("⚠️ Matplotlib not available - install with: pip install matplotlib")
            self.matplotlib_available = False
        
        try:
            import seaborn as sns
            self.sns = sns
            self.seaborn_available = True
            self.logger.info("✅ Seaborn available for enhanced visualization")
            print("✅ Seaborn available for enhanced visualization")
        except ImportError:
            self.logger.warning("⚠️ Seaborn not available - install with: pip install seaborn")
            print("⚠️ Seaborn not available - install with: pip install seaborn")
            self.seaborn_available = False
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def create_visualizations(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """Create comprehensive visualizations for interpretability results."""
        if not self.matplotlib_available:
            self.logger.warning("⚠️ Matplotlib not available, skipping visualizations")
            return {"error": "Matplotlib not available"}
        
        self.logger.info("🎨 Creating interpretability visualizations...")
        print("🎨 Creating interpretability visualizations...")
        
        visualizations = {
            "plots_created": [],
            "summary_plots": [],
            "feature_importance_plots": [],
            "comparison_plots": [],
            "insight_plots": []
        }
        
        try:
            # Ensure output directory exists
            ensure_directory(output_dir)
            
            # 1. Feature Importance Comparison Plot
            print("📊 Creating feature importance comparison plot...")
            self.logger.info("📊 Creating feature importance comparison plot...")
            
            importance_plot = await self._create_feature_importance_comparison_plot(results, output_dir)
            if importance_plot:
                visualizations["feature_importance_plots"].append(importance_plot)
                visualizations["plots_created"].append(importance_plot)
            
            # 2. Top Features Summary Plot
            print("📈 Creating top features summary plot...")
            self.logger.info("📈 Creating top features summary plot...")
            
            summary_plot = await self._create_top_features_summary_plot(results, output_dir)
            if summary_plot:
                visualizations["summary_plots"].append(summary_plot)
                visualizations["plots_created"].append(summary_plot)
            
            # 3. Feature Importance Distribution Plot
            print("📊 Creating feature importance distribution plot...")
            self.logger.info("📊 Creating feature importance distribution plot...")
            
            distribution_plot = await self._create_feature_importance_distribution_plot(results, output_dir)
            if distribution_plot:
                visualizations["feature_importance_plots"].append(distribution_plot)
                visualizations["plots_created"].append(distribution_plot)
            
            # 4. Model Comparison Plot (if multiple models)
            if "individual_results" in results:
                print("📊 Creating model comparison plot...")
                self.logger.info("📊 Creating model comparison plot...")
                
                comparison_plot = await self._create_model_comparison_plot(results, output_dir)
                if comparison_plot:
                    visualizations["comparison_plots"].append(comparison_plot)
                    visualizations["plots_created"].append(comparison_plot)
            
            # 5. Insights Summary Plot
            print("💡 Creating insights summary plot...")
            self.logger.info("💡 Creating insights summary plot...")
            
            insights_plot = await self._create_insights_summary_plot(results, output_dir)
            if insights_plot:
                visualizations["insight_plots"].append(insights_plot)
                visualizations["plots_created"].append(insights_plot)
            
            print(f"✅ Created {len(visualizations['plots_created'])} visualization plots")
            self.logger.info(f"✅ Created {len(visualizations['plots_created'])} visualization plots")
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create visualizations: {e}")
            print(f"❌ Failed to create visualizations: {e}")
            return visualizations
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_feature_importance_comparison_plot(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Create feature importance comparison plot."""
        try:
            # Extract feature importance data
            feature_importance = results.get("feature_importance", {})
            if not feature_importance:
                return None
            
            # Get top 10 features
            top_features = list(feature_importance.keys())[:10]
            if not top_features:
                return None
            
            # Create comparison data
            comparison_data = []
            
            # SHAP-based importance
            shap_results = results.get("shap_results", {})
            shap_importance = shap_results.get("feature_importance", {})
            
            # LIME-based importance
            lime_results = results.get("lime_results", {})
            lime_importance = lime_results.get("feature_importance", {})
            
            for feature in top_features:
                row = {"Feature": feature}
                
                # Combined importance
                if feature in feature_importance:
                    row["Combined"] = feature_importance[feature]
                
                # SHAP importance
                if feature in shap_importance:
                    row["SHAP"] = shap_importance[feature]
                
                # LIME importance
                if feature in lime_importance:
                    row["LIME"] = lime_importance[feature].get("importance_score", 0)
                
                comparison_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(comparison_data)
            df = df.set_index("Feature")
            
            # Create plot
            fig, ax = self.plt.subplots(figsize=(12, 8))
            
            # Plot bars
            x = np.arange(len(df))
            width = 0.25
            
            if "Combined" in df.columns:
                ax.bar(x - width, df["Combined"], width, label="Combined", alpha=0.8)
            if "SHAP" in df.columns:
                ax.bar(x, df["SHAP"], width, label="SHAP", alpha=0.8)
            if "LIME" in df.columns:
                ax.bar(x + width, df["LIME"], width, label="LIME", alpha=0.8)
            
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance Score")
            ax.set_title("Feature Importance Comparison (Top 10 Features)")
            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save plot
            plot_path = f"{output_dir}/feature_importance_comparison.png"
            self.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.plt.close()
            
            print(f"✅ Feature importance comparison plot saved: {plot_path}")
            self.logger.info(f"✅ Feature importance comparison plot saved: {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create feature importance comparison plot: {e}")
            print(f"❌ Failed to create feature importance comparison plot: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_top_features_summary_plot(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Create top features summary plot."""
        try:
            # Extract top features
            feature_importance = results.get("feature_importance", {})
            top_features = feature_importance.get("top_features", [])
            
            if not top_features:
                return None
            
            # Get top 15 features
            top_15 = top_features[:15]
            importance_scores = [feature_importance.get("combined_ranking", {}).get(f, 0) for f in top_15]
            
            # Create plot
            fig, ax = self.plt.subplots(figsize=(10, 8))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_15))
            bars = ax.barh(y_pos, importance_scores, alpha=0.7)
            
            # Color bars based on importance
            colors = self.plt.cm.viridis(np.linspace(0, 1, len(top_15)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_15)
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 15 Most Important Features")
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', ha='left', va='center', fontsize=8)
            
            # Invert y-axis to show highest importance at top
            ax.invert_yaxis()
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save plot
            plot_path = f"{output_dir}/top_features_summary.png"
            self.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.plt.close()
            
            print(f"✅ Top features summary plot saved: {plot_path}")
            self.logger.info(f"✅ Top features summary plot saved: {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create top features summary plot: {e}")
            print(f"❌ Failed to create top features summary plot: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_feature_importance_distribution_plot(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Create feature importance distribution plot."""
        try:
            # Extract feature importance data
            feature_importance = results.get("feature_importance", {})
            combined_ranking = feature_importance.get("combined_ranking", {})
            
            if not combined_ranking:
                return None
            
            # Get all importance scores
            importance_scores = list(combined_ranking.values())
            
            # Create plot
            fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(importance_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel("Importance Score")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Distribution of Feature Importance Scores")
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(importance_scores, vert=True)
            ax2.set_ylabel("Importance Score")
            ax2.set_title("Feature Importance Scores - Box Plot")
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_score = np.mean(importance_scores)
            median_score = np.median(importance_scores)
            std_score = np.std(importance_scores)
            
            ax1.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
            ax1.axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score:.3f}')
            ax1.legend()
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save plot
            plot_path = f"{output_dir}/feature_importance_distribution.png"
            self.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.plt.close()
            
            print(f"✅ Feature importance distribution plot saved: {plot_path}")
            self.logger.info(f"✅ Feature importance distribution plot saved: {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create feature importance distribution plot: {e}")
            print(f"❌ Failed to create feature importance distribution plot: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_model_comparison_plot(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Create model comparison plot for multiple models."""
        try:
            individual_results = results.get("individual_results", {})
            if len(individual_results) < 2:
                return None
            
            # Extract top features for each model
            model_top_features = {}
            for model_name, model_results in individual_results.items():
                feature_importance = model_results.get("feature_importance", {})
                top_features = feature_importance.get("top_features", [])
                model_top_features[model_name] = top_features[:10]  # Top 10 features
            
            # Find common features across models
            all_features = set()
            for features in model_top_features.values():
                all_features.update(features)
            
            # Create comparison matrix
            comparison_data = []
            for feature in all_features:
                row = {"Feature": feature}
                for model_name, features in model_top_features.items():
                    row[model_name] = 1 if feature in features else 0
                comparison_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(comparison_data)
            df = df.set_index("Feature")
            
            # Create heatmap
            fig, ax = self.plt.subplots(figsize=(10, max(8, len(df) * 0.3)))
            
            # Create heatmap
            im = ax.imshow(df.values, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(df.columns)))
            ax.set_yticks(range(len(df.index)))
            ax.set_xticklabels(df.columns)
            ax.set_yticklabels(df.index)
            
            # Add colorbar
            cbar = self.plt.colorbar(im, ax=ax)
            cbar.set_label('Feature Present in Top 10')
            
            # Add text annotations
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    text = ax.text(j, i, df.iloc[i, j], ha="center", va="center", color="black")
            
            ax.set_title("Model Comparison - Top 10 Features")
            ax.set_xlabel("Models")
            ax.set_ylabel("Features")
            
            # Rotate x-axis labels
            self.plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save plot
            plot_path = f"{output_dir}/model_comparison_heatmap.png"
            self.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.plt.close()
            
            print(f"✅ Model comparison plot saved: {plot_path}")
            self.logger.info(f"✅ Model comparison plot saved: {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create model comparison plot: {e}")
            print(f"❌ Failed to create model comparison plot: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_insights_summary_plot(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Create insights summary plot."""
        try:
            # Extract insights
            insights = results.get("insights", {})
            feature_insights = insights.get("feature_insights", [])
            model_insights = insights.get("model_insights", [])
            recommendations = insights.get("recommendations", [])
            
            # Create summary text
            summary_text = []
            summary_text.append("MODEL INTERPRETABILITY INSIGHTS")
            summary_text.append("=" * 40)
            summary_text.append("")
            
            if feature_insights:
                summary_text.append("FEATURE INSIGHTS:")
                for insight in feature_insights[:5]:  # Top 5 insights
                    summary_text.append(f"• {insight}")
                summary_text.append("")
            
            if model_insights:
                summary_text.append("MODEL INSIGHTS:")
                for insight in model_insights[:3]:  # Top 3 insights
                    summary_text.append(f"• {insight}")
                summary_text.append("")
            
            if recommendations:
                summary_text.append("RECOMMENDATIONS:")
                for rec in recommendations[:3]:  # Top 3 recommendations
                    summary_text.append(f"• {rec}")
            
            # Create plot with text
            fig, ax = self.plt.subplots(figsize=(12, 10))
            ax.axis('off')
            
            # Add text
            full_text = "\n".join(summary_text)
            ax.text(0.05, 0.95, full_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            ax.set_title("Model Interpretability Insights Summary", fontsize=16, fontweight='bold')
            
            # Save plot
            plot_path = f"{output_dir}/insights_summary.png"
            self.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.plt.close()
            
            print(f"✅ Insights summary plot saved: {plot_path}")
            self.logger.info(f"✅ Insights summary plot saved: {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create insights summary plot: {e}")
            print(f"❌ Failed to create insights summary plot: {e}")
            return None