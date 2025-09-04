#!/usr/bin/env python3
"""Interpretability Reporter for Model Analysis.

This module provides comprehensive reporting capabilities for model interpretability results
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

class InterpretabilityReporter:
    """Reporter for model interpretability results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the interpretability reporter."""
        self.config = config
        self.logger = system_logger.getChild("InterpretabilityReporter")
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> str:
        """Generate comprehensive interpretability report."""
        self.logger.info("📄 Generating comprehensive interpretability report...")
        print("📄 Generating comprehensive interpretability report...")
        
        try:
            # Ensure output directory exists
            ensure_directory(output_dir)
            
            # Generate different report formats
            reports_created = []
            
            # 1. JSON Report
            json_report_path = await self._generate_json_report(results, output_dir)
            if json_report_path:
                reports_created.append(json_report_path)
            
            # 2. Markdown Report
            markdown_report_path = await self._generate_markdown_report(results, output_dir)
            if markdown_report_path:
                reports_created.append(markdown_report_path)
            
            # 3. HTML Report
            html_report_path = await self._generate_html_report(results, output_dir)
            if html_report_path:
                reports_created.append(html_report_path)
            
            # 4. Summary Report
            summary_report_path = await self._generate_summary_report(results, output_dir)
            if summary_report_path:
                reports_created.append(summary_report_path)
            
            print(f"✅ Generated {len(reports_created)} interpretability reports")
            self.logger.info(f"✅ Generated {len(reports_created)} interpretability reports")
            
            return reports_created[0] if reports_created else None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate interpretability report: {e}")
            print(f"❌ Failed to generate interpretability report: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_json_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Generate JSON report."""
        try:
            # Create comprehensive JSON report
            json_report = {
                "report_metadata": {
                    "generated_at": format_datetime(get_current_datetime()),
                    "report_type": "model_interpretability",
                    "version": "1.0"
                },
                "model_info": results.get("model_info", {}),
                "shap_analysis": results.get("shap_results", {}),
                "lime_analysis": results.get("lime_results", {}),
                "feature_importance": results.get("feature_importance", {}),
                "insights": results.get("insights", {}),
                "visualizations": results.get("visualizations", {}),
                "performance_metrics": results.get("performance_metrics", {})
            }
            
            # Save JSON report
            json_path = f"{output_dir}/interpretability_report.json"
            safe_json_dump(json_report, json_path, indent=2)
            
            print(f"✅ JSON report saved: {json_path}")
            self.logger.info(f"✅ JSON report saved: {json_path}")
            
            return json_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate JSON report: {e}")
            print(f"❌ Failed to generate JSON report: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Generate Markdown report."""
        try:
            # Extract data
            model_info = results.get("model_info", {})
            feature_importance = results.get("feature_importance", {})
            insights = results.get("insights", {})
            shap_results = results.get("shap_results", {})
            lime_results = results.get("lime_results", {})
            visualizations = results.get("visualizations", {})
            
            # Generate Markdown content
            markdown_content = []
            
            # Header
            markdown_content.append("# Model Interpretability Report")
            markdown_content.append("")
            markdown_content.append(f"**Generated:** {format_datetime(get_current_datetime())}")
            markdown_content.append(f"**Model:** {model_info.get('model_name', 'Unknown')}")
            markdown_content.append(f"**Symbol:** {model_info.get('symbol', 'Unknown')}")
            markdown_content.append(f"**Exchange:** {model_info.get('exchange', 'Unknown')}")
            markdown_content.append("")
            
            # Executive Summary
            markdown_content.append("## Executive Summary")
            markdown_content.append("")
            summary_insights = insights.get("summary", [])
            for insight in summary_insights:
                markdown_content.append(f"- {insight}")
            markdown_content.append("")
            
            # Feature Importance
            markdown_content.append("## Feature Importance Analysis")
            markdown_content.append("")
            
            top_features = feature_importance.get("top_features", [])
            if top_features:
                markdown_content.append("### Top 10 Most Important Features")
                markdown_content.append("")
                for i, feature in enumerate(top_features[:10], 1):
                    importance_score = feature_importance.get("combined_ranking", {}).get(feature, 0)
                    markdown_content.append(f"{i}. **{feature}** - Importance Score: {importance_score:.4f}")
                markdown_content.append("")
            
            # SHAP Analysis
            if shap_results and "error" not in shap_results:
                markdown_content.append("## SHAP Analysis")
                markdown_content.append("")
                markdown_content.append("SHAP (SHapley Additive exPlanations) analysis provides global and local explanations for model predictions.")
                markdown_content.append("")
                
                shap_importance = shap_results.get("feature_importance", {})
                if shap_importance:
                    markdown_content.append("### SHAP Feature Importance (Top 5)")
                    markdown_content.append("")
                    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
                    for i, (feature, score) in enumerate(sorted_shap[:5], 1):
                        markdown_content.append(f"{i}. **{feature}** - SHAP Score: {score:.4f}")
                    markdown_content.append("")
                
                plots_created = shap_results.get("plots_created", [])
                if plots_created:
                    markdown_content.append("### SHAP Visualizations")
                    markdown_content.append("")
                    for plot in plots_created:
                        plot_name = Path(plot).name
                        markdown_content.append(f"- `{plot_name}`")
                    markdown_content.append("")
            
            # LIME Analysis
            if lime_results and "error" not in lime_results:
                markdown_content.append("## LIME Analysis")
                markdown_content.append("")
                markdown_content.append("LIME (Local Interpretable Model-agnostic Explanations) provides local explanations for individual predictions.")
                markdown_content.append("")
                
                lime_importance = lime_results.get("feature_importance", {})
                if lime_importance:
                    markdown_content.append("### LIME Feature Importance (Top 5)")
                    markdown_content.append("")
                    sorted_lime = sorted(lime_importance.items(), key=lambda x: x[1].get("importance_score", 0), reverse=True)
                    for i, (feature, data) in enumerate(sorted_lime[:5], 1):
                        score = data.get("importance_score", 0)
                        markdown_content.append(f"{i}. **{feature}** - LIME Score: {score:.4f}")
                    markdown_content.append("")
                
                plots_created = lime_results.get("plots_created", [])
                if plots_created:
                    markdown_content.append("### LIME Visualizations")
                    markdown_content.append("")
                    for plot in plots_created:
                        plot_name = Path(plot).name
                        markdown_content.append(f"- `{plot_name}`")
                    markdown_content.append("")
            
            # Insights and Recommendations
            markdown_content.append("## Insights and Recommendations")
            markdown_content.append("")
            
            feature_insights = insights.get("feature_insights", [])
            if feature_insights:
                markdown_content.append("### Feature Insights")
                markdown_content.append("")
                for insight in feature_insights:
                    markdown_content.append(f"- {insight}")
                markdown_content.append("")
            
            model_insights = insights.get("model_insights", [])
            if model_insights:
                markdown_content.append("### Model Insights")
                markdown_content.append("")
                for insight in model_insights:
                    markdown_content.append(f"- {insight}")
                markdown_content.append("")
            
            recommendations = insights.get("recommendations", [])
            if recommendations:
                markdown_content.append("### Recommendations")
                markdown_content.append("")
                for rec in recommendations:
                    markdown_content.append(f"- {rec}")
                markdown_content.append("")
            
            # Risk Assessment
            risk_assessment = insights.get("risk_assessment", [])
            if risk_assessment:
                markdown_content.append("### Risk Assessment")
                markdown_content.append("")
                for risk in risk_assessment:
                    markdown_content.append(f"- {risk}")
                markdown_content.append("")
            
            # Visualizations
            if visualizations and "plots_created" in visualizations:
                markdown_content.append("## Generated Visualizations")
                markdown_content.append("")
                plots_created = visualizations["plots_created"]
                for plot in plots_created:
                    plot_name = Path(plot).name
                    markdown_content.append(f"- `{plot_name}`")
                markdown_content.append("")
            
            # Footer
            markdown_content.append("---")
            markdown_content.append("")
            markdown_content.append("*This report was generated automatically by the Model Interpretability System.*")
            
            # Save Markdown report
            markdown_path = f"{output_dir}/interpretability_report.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            
            print(f"✅ Markdown report saved: {markdown_path}")
            self.logger.info(f"✅ Markdown report saved: {markdown_path}")
            
            return markdown_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate Markdown report: {e}")
            print(f"❌ Failed to generate Markdown report: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_html_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Generate HTML report."""
        try:
            # Extract data
            model_info = results.get("model_info", {})
            feature_importance = results.get("feature_importance", {})
            insights = results.get("insights", {})
            shap_results = results.get("shap_results", {})
            lime_results = results.get("lime_results", {})
            visualizations = results.get("visualizations", {})
            
            # Generate HTML content
            html_content = []
            
            # HTML Header
            html_content.append("<!DOCTYPE html>")
            html_content.append("<html lang='en'>")
            html_content.append("<head>")
            html_content.append("    <meta charset='UTF-8'>")
            html_content.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
            html_content.append("    <title>Model Interpretability Report</title>")
            html_content.append("    <style>")
            html_content.append("        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }")
            html_content.append("        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }")
            html_content.append("        h2 { color: #34495e; margin-top: 30px; }")
            html_content.append("        h3 { color: #7f8c8d; }")
            html_content.append("        .summary { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }")
            html_content.append("        .feature-list { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }")
            html_content.append("        .insight { background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-left: 4px solid #27ae60; }")
            html_content.append("        .recommendation { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }")
            html_content.append("        .risk { background-color: #f8d7da; padding: 10px; margin: 10px 0; border-left: 4px solid #dc3545; }")
            html_content.append("        table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
            html_content.append("        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
            html_content.append("        th { background-color: #f2f2f2; }")
            html_content.append("        .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; }")
            html_content.append("    </style>")
            html_content.append("</head>")
            html_content.append("<body>")
            
            # Title
            html_content.append(f"<h1>Model Interpretability Report</h1>")
            html_content.append(f"<p><strong>Generated:</strong> {format_datetime(get_current_datetime())}</p>")
            html_content.append(f"<p><strong>Model:</strong> {model_info.get('model_name', 'Unknown')}</p>")
            html_content.append(f"<p><strong>Symbol:</strong> {model_info.get('symbol', 'Unknown')}</p>")
            html_content.append(f"<p><strong>Exchange:</strong> {model_info.get('exchange', 'Unknown')}</p>")
            
            # Executive Summary
            html_content.append("<h2>Executive Summary</h2>")
            html_content.append("<div class='summary'>")
            summary_insights = insights.get("summary", [])
            for insight in summary_insights:
                html_content.append(f"<p>• {insight}</p>")
            html_content.append("</div>")
            
            # Feature Importance
            html_content.append("<h2>Feature Importance Analysis</h2>")
            top_features = feature_importance.get("top_features", [])
            if top_features:
                html_content.append("<h3>Top 10 Most Important Features</h3>")
                html_content.append("<div class='feature-list'>")
                html_content.append("<table>")
                html_content.append("<tr><th>Rank</th><th>Feature</th><th>Importance Score</th></tr>")
                for i, feature in enumerate(top_features[:10], 1):
                    importance_score = feature_importance.get("combined_ranking", {}).get(feature, 0)
                    html_content.append(f"<tr><td>{i}</td><td><strong>{feature}</strong></td><td>{importance_score:.4f}</td></tr>")
                html_content.append("</table>")
                html_content.append("</div>")
            
            # SHAP Analysis
            if shap_results and "error" not in shap_results:
                html_content.append("<h2>SHAP Analysis</h2>")
                html_content.append("<p>SHAP (SHapley Additive exPlanations) analysis provides global and local explanations for model predictions.</p>")
                
                shap_importance = shap_results.get("feature_importance", {})
                if shap_importance:
                    html_content.append("<h3>SHAP Feature Importance (Top 5)</h3>")
                    html_content.append("<div class='feature-list'>")
                    html_content.append("<table>")
                    html_content.append("<tr><th>Rank</th><th>Feature</th><th>SHAP Score</th></tr>")
                    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
                    for i, (feature, score) in enumerate(sorted_shap[:5], 1):
                        html_content.append(f"<tr><td>{i}</td><td><strong>{feature}</strong></td><td>{score:.4f}</td></tr>")
                    html_content.append("</table>")
                    html_content.append("</div>")
            
            # LIME Analysis
            if lime_results and "error" not in lime_results:
                html_content.append("<h2>LIME Analysis</h2>")
                html_content.append("<p>LIME (Local Interpretable Model-agnostic Explanations) provides local explanations for individual predictions.</p>")
                
                lime_importance = lime_results.get("feature_importance", {})
                if lime_importance:
                    html_content.append("<h3>LIME Feature Importance (Top 5)</h3>")
                    html_content.append("<div class='feature-list'>")
                    html_content.append("<table>")
                    html_content.append("<tr><th>Rank</th><th>Feature</th><th>LIME Score</th></tr>")
                    sorted_lime = sorted(lime_importance.items(), key=lambda x: x[1].get("importance_score", 0), reverse=True)
                    for i, (feature, data) in enumerate(sorted_lime[:5], 1):
                        score = data.get("importance_score", 0)
                        html_content.append(f"<tr><td>{i}</td><td><strong>{feature}</strong></td><td>{score:.4f}</td></tr>")
                    html_content.append("</table>")
                    html_content.append("</div>")
            
            # Insights and Recommendations
            html_content.append("<h2>Insights and Recommendations</h2>")
            
            feature_insights = insights.get("feature_insights", [])
            if feature_insights:
                html_content.append("<h3>Feature Insights</h3>")
                for insight in feature_insights:
                    html_content.append(f"<div class='insight'>{insight}</div>")
            
            model_insights = insights.get("model_insights", [])
            if model_insights:
                html_content.append("<h3>Model Insights</h3>")
                for insight in model_insights:
                    html_content.append(f"<div class='insight'>{insight}</div>")
            
            recommendations = insights.get("recommendations", [])
            if recommendations:
                html_content.append("<h3>Recommendations</h3>")
                for rec in recommendations:
                    html_content.append(f"<div class='recommendation'>{rec}</div>")
            
            risk_assessment = insights.get("risk_assessment", [])
            if risk_assessment:
                html_content.append("<h3>Risk Assessment</h3>")
                for risk in risk_assessment:
                    html_content.append(f"<div class='risk'>{risk}</div>")
            
            # Footer
            html_content.append("<div class='footer'>")
            html_content.append("<p><em>This report was generated automatically by the Model Interpretability System.</em></p>")
            html_content.append("</div>")
            
            html_content.append("</body>")
            html_content.append("</html>")
            
            # Save HTML report
            html_path = f"{output_dir}/interpretability_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            
            print(f"✅ HTML report saved: {html_path}")
            self.logger.info(f"✅ HTML report saved: {html_path}")
            
            return html_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate HTML report: {e}")
            print(f"❌ Failed to generate HTML report: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_summary_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> Optional[str]:
        """Generate summary report."""
        try:
            # Extract key information
            model_info = results.get("model_info", {})
            feature_importance = results.get("feature_importance", {})
            insights = results.get("insights", {})
            
            # Create summary
            summary = {
                "report_summary": {
                    "generated_at": format_datetime(get_current_datetime()),
                    "model_name": model_info.get("model_name", "Unknown"),
                    "symbol": model_info.get("symbol", "Unknown"),
                    "exchange": model_info.get("exchange", "Unknown"),
                    "total_features": model_info.get("feature_count", 0)
                },
                "key_findings": {
                    "top_5_features": feature_importance.get("top_features", [])[:5],
                    "feature_importance_scores": {
                        feature: score for feature, score in 
                        list(feature_importance.get("combined_ranking", {}).items())[:5]
                    }
                },
                "insights_summary": {
                    "feature_insights": insights.get("feature_insights", [])[:3],
                    "model_insights": insights.get("model_insights", [])[:3],
                    "recommendations": insights.get("recommendations", [])[:3]
                },
                "analysis_status": {
                    "shap_analysis_completed": "shap_results" in results and "error" not in results.get("shap_results", {}),
                    "lime_analysis_completed": "lime_results" in results and "error" not in results.get("lime_results", {}),
                    "visualizations_created": len(results.get("visualizations", {}).get("plots_created", []))
                }
            }
            
            # Save summary report
            summary_path = f"{output_dir}/interpretability_summary.json"
            safe_json_dump(summary, summary_path, indent=2)
            
            print(f"✅ Summary report saved: {summary_path}")
            self.logger.info(f"✅ Summary report saved: {summary_path}")
            
            return summary_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary report: {e}")
            print(f"❌ Failed to generate summary report: {e}")
            return None