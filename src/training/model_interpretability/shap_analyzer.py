#!/usr/bin/env python3
"""SHAP Analyzer for Model Interpretability.

This module provides SHAP (SHapley Additive exPlanations) analysis for understanding
model predictions and feature importance.
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

class SHAPAnalyzer:
    """SHAP analyzer for model interpretability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SHAP analyzer."""
        self.config = config
        self.logger = system_logger.getChild("SHAPAnalyzer")
        self.shap_available = False
        
        # Check SHAP availability
        self._check_shap_availability()
    
    def _check_shap_availability(self):
        """Check if SHAP is available and initialize if possible."""
        try:
            import shap
            self.shap = shap
            self.shap_available = True
            self.logger.info("✅ SHAP library available and initialized")
            print("✅ SHAP library available and initialized")
        except ImportError:
            self.logger.warning("⚠️ SHAP library not available - install with: pip install shap")
            print("⚠️ SHAP library not available - install with: pip install shap")
            self.shap_available = False
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def analyze_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        model_name: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Perform SHAP analysis on a trained model."""
        if not self.shap_available:
            self.logger.warning("⚠️ SHAP not available, skipping SHAP analysis")
            return {"error": "SHAP library not available"}
        
        self.logger.info(f"🧠 Starting SHAP analysis for {model_name}")
        print(f"🧠 Starting SHAP analysis for {model_name}")
        
        results = {
            "model_name": model_name,
            "analysis_timestamp": format_datetime(get_current_datetime()),
            "feature_names": feature_names,
            "shap_values": None,
            "feature_importance": {},
            "summary_stats": {},
            "local_explanations": {},
            "global_explanations": {},
            "plots_created": [],
            "performance_metrics": {}
        }
        
        try:
            # Ensure output directory exists
            ensure_directory(output_dir)
            
            # Step 1: Create SHAP explainer
            print("🔧 Creating SHAP explainer...")
            self.logger.info("🔧 Creating SHAP explainer...")
            
            explainer = await self._create_shap_explainer(model, X_train)
            if explainer is None:
                return {"error": "Failed to create SHAP explainer"}
            
            # Step 2: Calculate SHAP values
            print("📊 Calculating SHAP values...")
            self.logger.info("📊 Calculating SHAP values...")
            
            shap_values = await self._calculate_shap_values(explainer, X_test)
            if shap_values is None:
                return {"error": "Failed to calculate SHAP values"}
            
            results["shap_values"] = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
            
            # Step 3: Analyze feature importance
            print("📈 Analyzing feature importance...")
            self.logger.info("📈 Analyzing feature importance...")
            
            feature_importance = await self._analyze_feature_importance(shap_values, feature_names)
            results["feature_importance"] = feature_importance
            
            # Step 4: Generate summary statistics
            print("📊 Generating summary statistics...")
            self.logger.info("📊 Generating summary statistics...")
            
            summary_stats = await self._generate_summary_stats(shap_values, feature_names)
            results["summary_stats"] = summary_stats
            
            # Step 5: Create visualizations
            print("🎨 Creating SHAP visualizations...")
            self.logger.info("🎨 Creating SHAP visualizations...")
            
            plots_created = await self._create_shap_plots(
                explainer, shap_values, X_test, feature_names, model_name, output_dir
            )
            results["plots_created"] = plots_created
            
            # Step 6: Generate local explanations
            print("🔍 Generating local explanations...")
            self.logger.info("🔍 Generating local explanations...")
            
            local_explanations = await self._generate_local_explanations(
                shap_values, X_test, feature_names, num_samples=10
            )
            results["local_explanations"] = local_explanations
            
            # Step 7: Generate global explanations
            print("🌍 Generating global explanations...")
            self.logger.info("🌍 Generating global explanations...")
            
            global_explanations = await self._generate_global_explanations(
                shap_values, feature_names
            )
            results["global_explanations"] = global_explanations
            
            # Log metrics
            safe_log_metric("shap_analysis_success", 1.0)
            safe_log_metric("shap_features_analyzed", len(feature_names))
            safe_log_metric("shap_plots_created", len(plots_created))
            
            print("✅ SHAP analysis completed successfully!")
            self.logger.info("✅ SHAP analysis completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ SHAP analysis failed: {e}")
            print(f"❌ SHAP analysis failed: {e}")
            return {"error": str(e)}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_shap_explainer(self, model: Any, X_train: pd.DataFrame) -> Optional[Any]:
        """Create appropriate SHAP explainer for the model."""
        try:
            # Determine model type and create appropriate explainer
            model_type = type(model).__name__.lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'gradient' in model_type:
                # Tree-based models
                explainer = self.shap.TreeExplainer(model)
                print("✅ Created TreeExplainer for tree-based model")
                self.logger.info("✅ Created TreeExplainer for tree-based model")
            elif 'linear' in model_type or 'logistic' in model_type:
                # Linear models
                explainer = self.shap.LinearExplainer(model, X_train)
                print("✅ Created LinearExplainer for linear model")
                self.logger.info("✅ Created LinearExplainer for linear model")
            else:
                # Generic explainer using background data
                background = X_train.sample(min(100, len(X_train)), random_state=42)
                explainer = self.shap.Explainer(model, background)
                print("✅ Created generic Explainer with background data")
                self.logger.info("✅ Created generic Explainer with background data")
            
            return explainer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create SHAP explainer: {e}")
            print(f"❌ Failed to create SHAP explainer: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _calculate_shap_values(self, explainer: Any, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate SHAP values for test data."""
        try:
            # Limit test data size for performance
            max_samples = min(1000, len(X_test))
            X_test_sample = X_test.sample(max_samples, random_state=42)
            
            print(f"📊 Calculating SHAP values for {len(X_test_sample)} samples...")
            self.logger.info(f"📊 Calculating SHAP values for {len(X_test_sample)} samples...")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case - take the first class
                shap_values = shap_values[0]
            
            print(f"✅ SHAP values calculated: {shap_values.shape}")
            self.logger.info(f"✅ SHAP values calculated: {shap_values.shape}")
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate SHAP values: {e}")
            print(f"❌ Failed to calculate SHAP values: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _analyze_feature_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance from SHAP values."""
        try:
            # Calculate mean absolute SHAP values as feature importance
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(feature_names):
                if i < len(mean_shap_values):
                    feature_importance[feature] = float(mean_shap_values[i])
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            print(f"✅ Feature importance calculated for {len(feature_importance)} features")
            self.logger.info(f"✅ Feature importance calculated for {len(feature_importance)} features")
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze feature importance: {e}")
            print(f"❌ Failed to analyze feature importance: {e}")
            return {}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_summary_stats(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Generate summary statistics from SHAP values."""
        try:
            stats = {
                "total_features": len(feature_names),
                "total_samples": shap_values.shape[0],
                "shap_value_stats": {
                    "mean": float(np.mean(shap_values)),
                    "std": float(np.std(shap_values)),
                    "min": float(np.min(shap_values)),
                    "max": float(np.max(shap_values)),
                    "median": float(np.median(shap_values))
                },
                "feature_stats": {}
            }
            
            # Calculate statistics for each feature
            for i, feature in enumerate(feature_names):
                if i < shap_values.shape[1]:
                    feature_shap = shap_values[:, i]
                    stats["feature_stats"][feature] = {
                        "mean": float(np.mean(feature_shap)),
                        "std": float(np.std(feature_shap)),
                        "min": float(np.min(feature_shap)),
                        "max": float(np.max(feature_shap)),
                        "mean_abs": float(np.mean(np.abs(feature_shap)))
                    }
            
            print("✅ Summary statistics generated successfully")
            self.logger.info("✅ Summary statistics generated successfully")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary statistics: {e}")
            print(f"❌ Failed to generate summary statistics: {e}")
            return {}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_shap_plots(
        self,
        explainer: Any,
        shap_values: np.ndarray,
        X_test: pd.DataFrame,
        feature_names: List[str],
        model_name: str,
        output_dir: str
    ) -> List[str]:
        """Create SHAP visualization plots."""
        plots_created = []
        
        try:
            # Limit data for plotting
            max_samples = min(100, len(X_test))
            X_test_sample = X_test.sample(max_samples, random_state=42)
            shap_values_sample = shap_values[:max_samples]
            
            # 1. Summary plot
            print("📊 Creating SHAP summary plot...")
            self.logger.info("📊 Creating SHAP summary plot...")
            
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                self.shap.summary_plot(shap_values_sample, X_test_sample, feature_names=feature_names, show=False)
                summary_plot_path = f"{output_dir}/shap_summary_plot_{model_name}.png"
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created.append(summary_plot_path)
                print(f"✅ Summary plot saved: {summary_plot_path}")
                self.logger.info(f"✅ Summary plot saved: {summary_plot_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create summary plot: {e}")
            
            # 2. Bar plot
            print("📊 Creating SHAP bar plot...")
            self.logger.info("📊 Creating SHAP bar plot...")
            
            try:
                plt.figure(figsize=(10, 6))
                self.shap.summary_plot(shap_values_sample, X_test_sample, feature_names=feature_names, plot_type="bar", show=False)
                bar_plot_path = f"{output_dir}/shap_bar_plot_{model_name}.png"
                plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created.append(bar_plot_path)
                print(f"✅ Bar plot saved: {bar_plot_path}")
                self.logger.info(f"✅ Bar plot saved: {bar_plot_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create bar plot: {e}")
            
            # 3. Waterfall plot for first sample
            print("📊 Creating SHAP waterfall plot...")
            self.logger.info("📊 Creating SHAP waterfall plot...")
            
            try:
                plt.figure(figsize=(10, 6))
                self.shap.waterfall_plot(
                    self.shap.Explanation(
                        values=shap_values_sample[0],
                        base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                        data=X_test_sample.iloc[0].values,
                        feature_names=feature_names
                    ),
                    show=False
                )
                waterfall_plot_path = f"{output_dir}/shap_waterfall_plot_{model_name}.png"
                plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created.append(waterfall_plot_path)
                print(f"✅ Waterfall plot saved: {waterfall_plot_path}")
                self.logger.info(f"✅ Waterfall plot saved: {waterfall_plot_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create waterfall plot: {e}")
            
            print(f"✅ Created {len(plots_created)} SHAP plots")
            self.logger.info(f"✅ Created {len(plots_created)} SHAP plots")
            
            return plots_created
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create SHAP plots: {e}")
            print(f"❌ Failed to create SHAP plots: {e}")
            return plots_created
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_local_explanations(
        self,
        shap_values: np.ndarray,
        X_test: pd.DataFrame,
        feature_names: List[str],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """Generate local explanations for individual predictions."""
        try:
            local_explanations = {}
            
            # Limit number of samples
            num_samples = min(num_samples, len(X_test), len(shap_values))
            
            for i in range(num_samples):
                sample_id = f"sample_{i}"
                explanation = {
                    "sample_index": i,
                    "feature_values": X_test.iloc[i].to_dict(),
                    "shap_values": shap_values[i].tolist(),
                    "feature_contributions": {}
                }
                
                # Calculate feature contributions
                for j, feature in enumerate(feature_names):
                    if j < len(shap_values[i]):
                        explanation["feature_contributions"][feature] = {
                            "value": float(X_test.iloc[i, j]),
                            "shap_value": float(shap_values[i, j]),
                            "contribution": float(shap_values[i, j])
                        }
                
                # Sort by absolute contribution
                sorted_contributions = dict(sorted(
                    explanation["feature_contributions"].items(),
                    key=lambda x: abs(x[1]["contribution"]),
                    reverse=True
                ))
                explanation["top_contributors"] = list(sorted_contributions.keys())[:5]
                
                local_explanations[sample_id] = explanation
            
            print(f"✅ Generated local explanations for {num_samples} samples")
            self.logger.info(f"✅ Generated local explanations for {num_samples} samples")
            
            return local_explanations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate local explanations: {e}")
            print(f"❌ Failed to generate local explanations: {e}")
            return {}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_global_explanations(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate global explanations for the model."""
        try:
            global_explanations = {
                "feature_importance_ranking": [],
                "positive_contributors": [],
                "negative_contributors": [],
                "feature_interactions": {},
                "model_behavior": {}
            }
            
            # Calculate mean SHAP values for each feature
            mean_shap_values = np.mean(shap_values, axis=0)
            abs_mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Feature importance ranking
            feature_importance = []
            for i, feature in enumerate(feature_names):
                if i < len(abs_mean_shap_values):
                    feature_importance.append({
                        "feature": feature,
                        "importance": float(abs_mean_shap_values[i]),
                        "mean_contribution": float(mean_shap_values[i])
                    })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            global_explanations["feature_importance_ranking"] = feature_importance
            
            # Positive and negative contributors
            positive_contributors = [f for f in feature_importance if f["mean_contribution"] > 0]
            negative_contributors = [f for f in feature_importance if f["mean_contribution"] < 0]
            
            global_explanations["positive_contributors"] = positive_contributors[:5]
            global_explanations["negative_contributors"] = negative_contributors[:5]
            
            # Model behavior insights
            global_explanations["model_behavior"] = {
                "most_important_feature": feature_importance[0]["feature"] if feature_importance else None,
                "total_features": len(feature_names),
                "positive_features": len(positive_contributors),
                "negative_features": len(negative_contributors),
                "feature_diversity": len(set([f["feature"] for f in feature_importance[:10]]))
            }
            
            print("✅ Generated global explanations successfully")
            self.logger.info("✅ Generated global explanations successfully")
            
            return global_explanations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate global explanations: {e}")
            print(f"❌ Failed to generate global explanations: {e}")
            return {}