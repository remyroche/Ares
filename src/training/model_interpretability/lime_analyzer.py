#!/usr/bin/env python3
"""LIME Analyzer for Model Interpretability.

This module provides LIME (Local Interpretable Model-agnostic Explanations) analysis
for understanding individual model predictions.
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

class LIMEAnalyzer:
    """LIME analyzer for model interpretability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LIME analyzer."""
        self.config = config
        self.logger = system_logger.getChild("LIMEAnalyzer")
        self.lime_available = False
        
        # Check LIME availability
        self._check_lime_availability()
    
    def _check_lime_availability(self):
        """Check if LIME is available and initialize if possible."""
        try:
            import lime
            import lime.lime_tabular
            self.lime = lime
            self.lime_tabular = lime.lime_tabular
            self.lime_available = True
            self.logger.info("✅ LIME library available and initialized")
            print("✅ LIME library available and initialized")
        except ImportError:
            self.logger.warning("⚠️ LIME library not available - install with: pip install lime")
            print("⚠️ LIME library not available - install with: pip install lime")
            self.lime_available = False
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def analyze_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        feature_names: List[str],
        model_name: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Perform LIME analysis on a trained model."""
        if not self.lime_available:
            self.logger.warning("⚠️ LIME not available, skipping LIME analysis")
            return {"error": "LIME library not available"}
        
        self.logger.info(f"🔍 Starting LIME analysis for {model_name}")
        print(f"🔍 Starting LIME analysis for {model_name}")
        
        results = {
            "model_name": model_name,
            "analysis_timestamp": format_datetime(get_current_datetime()),
            "feature_names": feature_names,
            "local_explanations": {},
            "feature_importance": {},
            "explanation_consistency": {},
            "plots_created": [],
            "performance_metrics": {}
        }
        
        try:
            # Ensure output directory exists
            ensure_directory(output_dir)
            
            # Step 1: Create LIME explainer
            print("🔧 Creating LIME explainer...")
            self.logger.info("🔧 Creating LIME explainer...")
            
            explainer = await self._create_lime_explainer(X_test, feature_names)
            if explainer is None:
                return {"error": "Failed to create LIME explainer"}
            
            # Step 2: Generate local explanations
            print("🔍 Generating local explanations...")
            self.logger.info("🔍 Generating local explanations...")
            
            local_explanations = await self._generate_local_explanations(
                explainer, model, X_test, feature_names, num_samples=20
            )
            results["local_explanations"] = local_explanations
            
            # Step 3: Analyze feature importance
            print("📈 Analyzing feature importance...")
            self.logger.info("📈 Analyzing feature importance...")
            
            feature_importance = await self._analyze_feature_importance(local_explanations, feature_names)
            results["feature_importance"] = feature_importance
            
            # Step 4: Analyze explanation consistency
            print("📊 Analyzing explanation consistency...")
            self.logger.info("📊 Analyzing explanation consistency...")
            
            explanation_consistency = await self._analyze_explanation_consistency(local_explanations, feature_names)
            results["explanation_consistency"] = explanation_consistency
            
            # Step 5: Create visualizations
            print("🎨 Creating LIME visualizations...")
            self.logger.info("🎨 Creating LIME visualizations...")
            
            plots_created = await self._create_lime_plots(
                explainer, model, X_test, feature_names, model_name, output_dir
            )
            results["plots_created"] = plots_created
            
            # Log metrics
            safe_log_metric("lime_analysis_success", 1.0)
            safe_log_metric("lime_features_analyzed", len(feature_names))
            safe_log_metric("lime_explanations_generated", len(local_explanations))
            safe_log_metric("lime_plots_created", len(plots_created))
            
            print("✅ LIME analysis completed successfully!")
            self.logger.info("✅ LIME analysis completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ LIME analysis failed: {e}")
            print(f"❌ LIME analysis failed: {e}")
            return {"error": str(e)}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_lime_explainer(self, X_test: pd.DataFrame, feature_names: List[str]) -> Optional[Any]:
        """Create LIME tabular explainer."""
        try:
            # Use a subset of data for training the explainer
            training_data = X_test.sample(min(1000, len(X_test)), random_state=42).values
            
            # Create LIME explainer
            explainer = self.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=['prediction'],
                mode='regression' if len(np.unique(training_data.flatten())) > 10 else 'classification',
                discretize_continuous=True,
                random_state=42
            )
            
            print("✅ Created LIME tabular explainer")
            self.logger.info("✅ Created LIME tabular explainer")
            
            return explainer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create LIME explainer: {e}")
            print(f"❌ Failed to create LIME explainer: {e}")
            return None
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_local_explanations(
        self,
        explainer: Any,
        model: Any,
        X_test: pd.DataFrame,
        feature_names: List[str],
        num_samples: int = 20
    ) -> Dict[str, Any]:
        """Generate local explanations for individual predictions."""
        try:
            local_explanations = {}
            
            # Limit number of samples for performance
            num_samples = min(num_samples, len(X_test))
            sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
            
            for i, sample_idx in enumerate(sample_indices):
                sample_id = f"sample_{i}"
                sample_data = X_test.iloc[sample_idx].values
                
                try:
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        sample_data,
                        model.predict,
                        num_features=min(10, len(feature_names)),
                        top_labels=1
                    )
                    
                    # Extract explanation data
                    explanation_data = {
                        "sample_index": int(sample_idx),
                        "feature_values": X_test.iloc[sample_idx].to_dict(),
                        "prediction": float(model.predict([sample_data])[0]),
                        "explanation": {},
                        "top_features": [],
                        "feature_contributions": {}
                    }
                    
                    # Get explanation as list
                    explanation_list = explanation.as_list()
                    
                    for feature, contribution in explanation_list:
                        explanation_data["explanation"][feature] = contribution
                        explanation_data["feature_contributions"][feature] = {
                            "contribution": contribution,
                            "abs_contribution": abs(contribution)
                        }
                    
                    # Sort by absolute contribution
                    sorted_contributions = dict(sorted(
                        explanation_data["feature_contributions"].items(),
                        key=lambda x: x[1]["abs_contribution"],
                        reverse=True
                    ))
                    explanation_data["top_features"] = list(sorted_contributions.keys())[:5]
                    
                    local_explanations[sample_id] = explanation_data
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to explain sample {sample_idx}: {e}")
                    continue
            
            print(f"✅ Generated local explanations for {len(local_explanations)} samples")
            self.logger.info(f"✅ Generated local explanations for {len(local_explanations)} samples")
            
            return local_explanations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate local explanations: {e}")
            print(f"❌ Failed to generate local explanations: {e}")
            return {}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _analyze_feature_importance(
        self,
        local_explanations: Dict[str, Any],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature importance from LIME explanations."""
        try:
            feature_importance = {}
            feature_contributions = {feature: [] for feature in feature_names}
            
            # Collect contributions for each feature across all explanations
            for sample_id, explanation in local_explanations.items():
                contributions = explanation.get("feature_contributions", {})
                for feature, contrib_data in contributions.items():
                    if feature in feature_contributions:
                        feature_contributions[feature].append(contrib_data["contribution"])
            
            # Calculate importance metrics for each feature
            for feature in feature_names:
                contributions = feature_contributions[feature]
                if contributions:
                    feature_importance[feature] = {
                        "mean_contribution": float(np.mean(contributions)),
                        "std_contribution": float(np.std(contributions)),
                        "mean_abs_contribution": float(np.mean(np.abs(contributions))),
                        "frequency": len(contributions) / len(local_explanations),
                        "importance_score": float(np.mean(np.abs(contributions)))
                    }
                else:
                    feature_importance[feature] = {
                        "mean_contribution": 0.0,
                        "std_contribution": 0.0,
                        "mean_abs_contribution": 0.0,
                        "frequency": 0.0,
                        "importance_score": 0.0
                    }
            
            # Sort by importance score
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1]["importance_score"],
                reverse=True
            ))
            
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
    async def _analyze_explanation_consistency(
        self,
        local_explanations: Dict[str, Any],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze consistency of explanations across samples."""
        try:
            consistency_analysis = {
                "feature_consistency": {},
                "explanation_stability": {},
                "top_features_consistency": {},
                "overall_consistency_score": 0.0
            }
            
            # Analyze feature consistency
            feature_appearances = {feature: 0 for feature in feature_names}
            feature_contributions = {feature: [] for feature in feature_names}
            
            for sample_id, explanation in local_explanations.items():
                top_features = explanation.get("top_features", [])
                contributions = explanation.get("feature_contributions", {})
                
                # Count feature appearances in top features
                for feature in top_features:
                    if feature in feature_appearances:
                        feature_appearances[feature] += 1
                
                # Collect contributions
                for feature, contrib_data in contributions.items():
                    if feature in feature_contributions:
                        feature_contributions[feature].append(contrib_data["contribution"])
            
            # Calculate consistency metrics
            total_explanations = len(local_explanations)
            for feature in feature_names:
                appearance_rate = feature_appearances[feature] / total_explanations
                contributions = feature_contributions[feature]
                
                if contributions:
                    contribution_std = np.std(contributions)
                    contribution_mean = np.mean(np.abs(contributions))
                    consistency_score = appearance_rate * (1 / (1 + contribution_std)) if contribution_std > 0 else appearance_rate
                else:
                    consistency_score = 0.0
                
                consistency_analysis["feature_consistency"][feature] = {
                    "appearance_rate": appearance_rate,
                    "contribution_std": float(np.std(contributions)) if contributions else 0.0,
                    "consistency_score": consistency_score
                }
            
            # Calculate overall consistency score
            consistency_scores = [data["consistency_score"] for data in consistency_analysis["feature_consistency"].values()]
            consistency_analysis["overall_consistency_score"] = float(np.mean(consistency_scores))
            
            # Top features consistency
            sorted_consistency = dict(sorted(
                consistency_analysis["feature_consistency"].items(),
                key=lambda x: x[1]["consistency_score"],
                reverse=True
            ))
            consistency_analysis["top_features_consistency"] = dict(list(sorted_consistency.items())[:10])
            
            print("✅ Explanation consistency analysis completed")
            self.logger.info("✅ Explanation consistency analysis completed")
            
            return consistency_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze explanation consistency: {e}")
            print(f"❌ Failed to analyze explanation consistency: {e}")
            return {}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _create_lime_plots(
        self,
        explainer: Any,
        model: Any,
        X_test: pd.DataFrame,
        feature_names: List[str],
        model_name: str,
        output_dir: str
    ) -> List[str]:
        """Create LIME visualization plots."""
        plots_created = []
        
        try:
            # Create plots for a few representative samples
            num_plots = min(5, len(X_test))
            sample_indices = np.random.choice(len(X_test), num_plots, replace=False)
            
            for i, sample_idx in enumerate(sample_indices):
                try:
                    sample_data = X_test.iloc[sample_idx].values
                    
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        sample_data,
                        model.predict,
                        num_features=min(10, len(feature_names)),
                        top_labels=1
                    )
                    
                    # Save explanation as HTML
                    html_path = f"{output_dir}/lime_explanation_{model_name}_sample_{i}.html"
                    explanation.save_to_file(html_path)
                    plots_created.append(html_path)
                    
                    print(f"✅ LIME explanation plot saved: {html_path}")
                    self.logger.info(f"✅ LIME explanation plot saved: {html_path}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to create LIME plot for sample {sample_idx}: {e}")
                    continue
            
            print(f"✅ Created {len(plots_created)} LIME plots")
            self.logger.info(f"✅ Created {len(plots_created)} LIME plots")
            
            return plots_created
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create LIME plots: {e}")
            print(f"❌ Failed to create LIME plots: {e}")
            return plots_created