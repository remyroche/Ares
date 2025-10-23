#!/usr/bin/env python3
from src.utils.tprint import tprint

import numpy as np
import pandas as pd
from src.utils.logger import system_logger
from ..core.decorators import handles_errors

"""Model Explainer for Trading Pipeline.

This module provides comprehensive model interpretability using SHAP and LIME
to understand what features are most important for model predictions.
"""

from src.utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_json_dump, safe_json_load, safe_file_exists,
    timed_operation, format_bytes, safe_log_metric, safe_log_params
)

class ModelExplainer:
    """Comprehensive model explainer using SHAP and LIME."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the model explainer."""
        self.config = config
        self.logger = system_logger.getChild("ModelExplainer")
        self.shap_analyzer = None
        self.lime_analyzer = None
        self.visualizer = None
        self.reporter = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize SHAP, LIME, and visualization components."""
        try:
            from .shap_analyzer import SHAPAnalyzer
            from .lime_analyzer import LIMEAnalyzer
            from .interpretability_visualizer import InterpretabilityVisualizer
            from .interpretability_reporter import InterpretabilityReporter
            import json
            import logging
            import time
            import typing

            self.shap_analyzer = SHAPAnalyzer(self.config)
            self.lime_analyzer = LIMEAnalyzer(self.config)
            self.visualizer = InterpretabilityVisualizer(self.config)
            self.reporter = InterpretabilityReporter(self.config)

            self.logger.info("✅ Model interpretability components initialized successfully")

        except ImportError as e:
            self.logger.warning(f"⚠️ Some interpretability components not available: {e}")
            self.logger.warning("Proceeding with available components only")

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @validates(strict = True)
    @log_call
    @traced
    async def explain_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        model_name: str,
        symbol: str,
        exchange: str,
        output_dir: str = "models/interpretability"
    ) -> Dict[str, Any]:
        """Generate comprehensive model explanations using SHAP and LIME."""
        self.logger.info(f"🔍 Starting model interpretability analysis for {model_name}")
        tprint(f"🔍 Starting model interpretability analysis for {model_name}")

        # Ensure output directory exists
        ensure_directory(output_dir)

        # Initialize results dictionary
        results = {
            "model_info": {
                "model_name": model_name,
                "symbol": symbol,
                "exchange": exchange,
                "analysis_timestamp": format_datetime(get_current_datetime()),
                "feature_count": len(feature_names),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            },
            "shap_results": {},
            "lime_results": {},
            "feature_importance": {},
            "insights": {},
            "visualizations": {},
            "performance_metrics": {}
        }

        try:
            # Step 1: SHAP Analysis
            tprint("🧠 STEP 1/4: Running SHAP analysis...")
            self.logger.info("🧠 STEP 1/4: Running SHAP analysis...")

            if self.shap_analyzer:
                shap_results = await self.shap_analyzer.analyze_model(
                    model = model,
                    X_train = X_train,
                    X_test = X_test,
                    feature_names = feature_names,
                    model_name = model_name,
                    output_dir = output_dir
                )
                results["shap_results"] = shap_results
                tprint("✅ SHAP analysis completed successfully")
                self.logger.info("✅ SHAP analysis completed successfully")
            else:
                tprint("⚠️ SHAP analyzer not available, skipping SHAP analysis")
                self.logger.warning("⚠️ SHAP analyzer not available, skipping SHAP analysis")

            # Step 2: LIME Analysis
            tprint("🔍 STEP 2/4: Running LIME analysis...")
            self.logger.info("🔍 STEP 2/4: Running LIME analysis...")

            if self.lime_analyzer:
                lime_results = await self.lime_analyzer.analyze_model(
                    model = model,
                    X_test = X_test,
                    feature_names = feature_names,
                    model_name = model_name,
                    output_dir = output_dir
                )
                results["lime_results"] = lime_results
                tprint("✅ LIME analysis completed successfully")
                self.logger.info("✅ LIME analysis completed successfully")
            else:
                tprint("⚠️ LIME analyzer not available, skipping LIME analysis")
                self.logger.warning("⚠️ LIME analyzer not available, skipping LIME analysis")

            # Step 3: Feature Importance Analysis
            tprint("📊 STEP 3/4: Analyzing feature importance...")
            self.logger.info("📊 STEP 3/4: Analyzing feature importance...")

            feature_importance = await self._analyze_feature_importance(
                model = model,
                X_train = X_train,
                feature_names = feature_names,
                shap_results = results.get("shap_results", {}),
                lime_results = results.get("lime_results", {})
            )
            results["feature_importance"] = feature_importance
            tprint("✅ Feature importance analysis completed successfully")
            self.logger.info("✅ Feature importance analysis completed successfully")

            # Step 4: Generate Insights and Visualizations
            tprint("📈 STEP 4/4: Generating insights and visualizations...")
            self.logger.info("📈 STEP 4/4: Generating insights and visualizations...")

            insights = await self._generate_insights(results)
            results["insights"] = insights

            if self.visualizer:
                visualizations = await self.visualizer.create_visualizations(
                    results = results,
                    output_dir = output_dir
                )
                results["visualizations"] = visualizations
                tprint("✅ Visualizations created successfully")
                self.logger.info("✅ Visualizations created successfully")

            # Generate comprehensive report
            if self.reporter:
                # Extract symbol and exchange from config or results
                symbol = self.config.get("symbol", results.get("symbol", "UNKNOWN"))
                exchange = self.config.get("exchange", results.get("exchange", "UNKNOWN"))
                model_type = self.config.get("model_type", results.get("model_type", "general"))

                report_path = await self.reporter.generate_report(
                    results = results,
                    output_dir = output_dir,
                    model_type = model_type,
                    symbol = symbol,
                    exchange = exchange
                )
                results["report_path"] = report_path
                tprint(f"📄 Comprehensive report generated: {report_path}")
                self.logger.info(f"📄 Comprehensive report generated: {report_path}")

            # Log metrics
            safe_log_metric("interpretability_analysis_success", 1.0)
            safe_log_metric("features_analyzed", len(feature_names))
            safe_log_metric("shap_analysis_completed", 1.0 if results.get("shap_results") else 0.0)
            safe_log_metric("lime_analysis_completed", 1.0 if results.get("lime_results") else 0.0)

            tprint("🎉 Model interpretability analysis completed successfully!")
            self.logger.info("🎉 Model interpretability analysis completed successfully!")

            return results

        except Exception as e:
            self.logger.error(f"❌ Model interpretability analysis failed: {e}")
            tprint(f"❌ Model interpretability analysis failed: {e}")
            raise

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _analyze_feature_importance(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: List[str],
        shap_results: Dict[str, Any],
        lime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods."""
        self.logger.info("📊 Analyzing feature importance using multiple methods...")

        feature_importance = {
            "model_based": {},
            "shap_based": {},
            "lime_based": {},
            "combined_ranking": {},
            "top_features": [],
            "insights": []
        }

        try:
            # 1. Model-based feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    feature_importance["model_based"][feature] = float(model_importance[i])
                tprint("✅ Model-based feature importance extracted")
                self.logger.info("✅ Model-based feature importance extracted")

            # 2. SHAP-based feature importance
            if shap_results and "feature_importance" in shap_results:
                feature_importance["shap_based"] = shap_results["feature_importance"]
                tprint("✅ SHAP-based feature importance extracted")
                self.logger.info("✅ SHAP-based feature importance extracted")

            # 3. LIME-based feature importance
            if lime_results and "feature_importance" in lime_results:
                feature_importance["lime_based"] = lime_results["feature_importance"]
                tprint("✅ LIME-based feature importance extracted")
                self.logger.info("✅ LIME-based feature importance extracted")

            # 4. Combined ranking
            combined_scores = {}
            for feature in feature_names:
                scores = []

                # Model-based score
                if feature in feature_importance["model_based"]:
                    scores.append(feature_importance["model_based"][feature])

                # SHAP-based score
                if feature in feature_importance["shap_based"]:
                    scores.append(feature_importance["shap_based"][feature])

                # LIME-based score
                if feature in feature_importance["lime_based"]:
                    scores.append(feature_importance["lime_based"][feature])

                if scores:
                    combined_scores[feature] = np.mean(scores)

            # Sort by combined score
            sorted_features = sorted(combined_scores.items(), key = lambda x: x[1], reverse = True)
            feature_importance["combined_ranking"] = dict(sorted_features)
            feature_importance["top_features"] = [feature for feature, _ in sorted_features[:10]]

            # Generate insights
            insights = []
            if feature_importance["top_features"]:
                insights.append(f"Top 3 most important features: {', '.join(feature_importance['top_features'][:3])}")

            if len(feature_importance["top_features"]) > 0:
                top_feature = feature_importance["top_features"][0]
                top_score = feature_importance["combined_ranking"][top_feature]
                insights.append(f"Most important feature '{top_feature}' has a combined importance score of {top_score:.4f}")

            feature_importance["insights"] = insights

            tprint(f"✅ Feature importance analysis completed - {len(feature_importance['top_features'])} top features identified")
            self.logger.info(f"✅ Feature importance analysis completed - {len(feature_importance['top_features'])} top features identified")

            return feature_importance

        except Exception as e:
            self.logger.error(f"❌ Feature importance analysis failed: {e}")
            tprint(f"❌ Feature importance analysis failed: {e}")
            return feature_importance

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from interpretability results."""
        self.logger.info("💡 Generating insights from interpretability results...")

        insights = {
            "summary": [],
            "feature_insights": [],
            "model_insights": [],
            "recommendations": [],
            "risk_assessment": [],
            "model_type_insights": []
        }

        try:
            # Get model type from config
            model_type = self.config.get("interpretability", {}).get("model_type", "unknown")

            # Summary insights
            model_info = results.get("model_info", {})
            insights["summary"].append(f"Analyzed {model_info.get('feature_count', 0)} features for {model_info.get('model_name', 'unknown')} {model_type} model")
            insights["summary"].append(f"Training samples: {model_info.get('training_samples', 0)}, Test samples: {model_info.get('test_samples', 0)}")

            # Feature insights
            feature_importance = results.get("feature_importance", {})
            top_features = feature_importance.get("top_features", [])

            if top_features:
                insights["feature_insights"].append(f"Top 5 most important features: {', '.join(top_features[:5])}")

                # Model-type specific feature analysis
                model_type_insights = await self._generate_model_type_specific_insights(
                    model_type, top_features, feature_importance
                )
                insights["model_type_insights"] = model_type_insights

                # Analyze feature types
                technical_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['rsi', 'macd', 'bollinger', 'sma', 'ema'])]
                if technical_features:
                    insights["feature_insights"].append(f"Technical indicators are prominent: {', '.join(technical_features[:3])}")

                price_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['price', 'close', 'open', 'high', 'low'])]
                if price_features:
                    insights["feature_insights"].append(f"Price-based features are important: {', '.join(price_features[:3])}")

                volume_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['volume', 'vol'])]
                if volume_features:
                    insights["feature_insights"].append(f"Volume-based features are significant: {', '.join(volume_features[:3])}")

            # Model insights
            shap_results = results.get("shap_results", {})
            if shap_results:
                insights["model_insights"].append("SHAP analysis completed successfully - model behavior is interpretable")

            lime_results = results.get("lime_results", {})
            if lime_results:
                insights["model_insights"].append("LIME analysis completed successfully - local explanations available")

            # Model-type specific recommendations
            model_type_recommendations = await self._generate_model_type_recommendations(model_type, top_features)
            insights["recommendations"].extend(model_type_recommendations)

            # General recommendations
            if len(top_features) > 0:
                insights["recommendations"].append("Focus on the top 5 features for model optimization")
                insights["recommendations"].append("Consider feature engineering for the most important features")

            if len(top_features) < 5:
                insights["recommendations"].append("Consider adding more features to improve model performance")

            # Risk assessment
            model_type_risks = await self._generate_model_type_risk_assessment(model_type, top_features)
            insights["risk_assessment"].extend(model_type_risks)

            if len(top_features) > 0:
                insights["risk_assessment"].append("Model has clear feature dependencies - monitor top features for stability")

            tprint(f"✅ Generated {len(insights['summary']) + len(insights['feature_insights']) + len(insights['model_insights'])} insights")
            self.logger.info(f"✅ Generated {len(insights['summary']) + len(insights['feature_insights']) + len(insights['model_insights'])} insights")

            return insights

        except Exception as e:
            self.logger.error(f"❌ Insight generation failed: {e}")
            tprint(f"❌ Insight generation failed: {e}")
            return insights

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_model_type_specific_insights(
        self,
        model_type: str,
        top_features: List[str],
        feature_importance: Dict[str, Any]
    ) -> List[str]:
        """Generate model-type specific insights."""
        insights = []

        try:
            if model_type == "tactician":
                insights.append("Tactician models focus on execution strategy and risk management")
                execution_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['execution', 'risk', 'position', 'size'])]
                if execution_features:
                    insights.append(f"Execution-focused features are prominent: {', '.join(execution_features[:3])}")

            elif model_type == "analyst":
                insights.append("Analyst models focus on market analysis and prediction")
                analysis_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['trend', 'momentum', 'volatility', 'pattern'])]
                if analysis_features:
                    insights.append(f"Analysis-focused features are prominent: {', '.join(analysis_features[:3])}")

            elif model_type == "ensemble":
                insights.append("Ensemble models combine multiple model predictions")
                diversity_score = len(set(top_features[:10])) / min(10, len(top_features))
                insights.append(f"Feature diversity score: {diversity_score:.2f} (higher is better for ensemble)")

            elif model_type == "intelligence":
                insights.append("Intelligence models focus on market intelligence and insights")
                intelligence_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['sentiment', 'news', 'social', 'macro'])]
                if intelligence_features:
                    insights.append(f"Intelligence-focused features are prominent: {', '.join(intelligence_features[:3])}")

            elif model_type == "market_regime":
                insights.append("Market regime models identify different market states")
                regime_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['regime', 'state', 'volatility', 'trend'])]
                if regime_features:
                    insights.append(f"Regime-identifying features are prominent: {', '.join(regime_features[:3])}")

            elif model_type == "market_cluster":
                insights.append("Market cluster models group similar market conditions")
                cluster_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['cluster', 'similarity', 'distance', 'pattern'])]
                if cluster_features:
                    insights.append(f"Clustering features are prominent: {', '.join(cluster_features[:3])}")

            elif model_type == "support_resistance":
                insights.append("Support/Resistance models identify key price levels")
                sr_features = [f for f in top_features if any(keyword in f.lower() for keyword in ['support', 'resistance', 'level', 'breakout'])]
                if sr_features:
                    insights.append(f"Support/Resistance features are prominent: {', '.join(sr_features[:3])}")

            return insights

        except Exception as e:
            self.logger.error(f"❌ Model type specific insights generation failed: {e}")
            return []

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_model_type_recommendations(
        self,
        model_type: str,
        top_features: List[str]
    ) -> List[str]:
        """Generate model-type specific recommendations."""
        recommendations = []

        try:
            if model_type == "tactician":
                recommendations.append("Monitor execution features for real-time risk management")
                recommendations.append("Consider position sizing based on top feature values")

            elif model_type == "analyst":
                recommendations.append("Focus on trend and momentum features for market analysis")
                recommendations.append("Consider feature engineering for pattern recognition")

            elif model_type == "ensemble":
                recommendations.append("Ensure diversity in base models for better ensemble performance")
                recommendations.append("Monitor feature importance consistency across ensemble members")

            elif model_type == "intelligence":
                recommendations.append("Integrate external data sources for enhanced intelligence")
                recommendations.append("Monitor sentiment and news features for market insights")

            elif model_type == "market_regime":
                recommendations.append("Use regime identification for adaptive trading strategies")
                recommendations.append("Monitor regime transitions for strategy adjustments")

            elif model_type == "market_cluster":
                recommendations.append("Use clustering for market condition classification")
                recommendations.append("Monitor cluster stability for reliable classifications")

            elif model_type == "support_resistance":
                recommendations.append("Use S/R levels for entry/exit point identification")
                recommendations.append("Monitor level breaks for trend confirmation")

            return recommendations

        except Exception as e:
            self.logger.error(f"❌ Model type recommendations generation failed: {e}")
            return []

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_model_type_risk_assessment(
        self,
        model_type: str,
        top_features: List[str]
    ) -> List[str]:
        """Generate model-type specific risk assessment."""
        risks = []

        try:
            if model_type == "tactician":
                risks.append("Monitor execution risk - tactician models directly affect trading decisions")
                risks.append("Ensure position sizing features are stable and reliable")

            elif model_type == "analyst":
                risks.append("Monitor analysis accuracy - incorrect analysis can lead to poor decisions")
                risks.append("Ensure trend and momentum features are not overfitted")

            elif model_type == "ensemble":
                risks.append("Monitor ensemble diversity - low diversity can lead to correlated failures")
                risks.append("Ensure base model independence for robust ensemble performance")

            elif model_type == "intelligence":
                risks.append("Monitor external data quality - poor intelligence data affects all decisions")
                risks.append("Ensure sentiment features are not biased or manipulated")

            elif model_type == "market_regime":
                risks.append("Monitor regime identification accuracy - wrong regime leads to wrong strategy")
                risks.append("Ensure regime transitions are detected promptly")

            elif model_type == "market_cluster":
                risks.append("Monitor cluster stability - unstable clusters lead to inconsistent classifications")
                risks.append("Ensure clustering features are representative of market conditions")

            elif model_type == "support_resistance":
                risks.append("Monitor S/R level accuracy - wrong levels lead to poor entry/exit points")
                risks.append("Ensure level breaks are detected reliably")

            return risks

        except Exception as e:
            self.logger.error(f"❌ Model type risk assessment generation failed: {e}")
            return []

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def explain_multiple_models(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        symbol: str,
        exchange: str,
        output_dir: str = "models/interpretability"
    ) -> Dict[str, Any]:
        """Generate explanations for multiple models."""
        self.logger.info(f"🔍 Starting multi-model interpretability analysis for {len(models)} models")
        tprint(f"🔍 Starting multi-model interpretability analysis for {len(models)} models")

        results = {
            "models_analyzed": len(models),
            "symbol": symbol,
            "exchange": exchange,
            "analysis_timestamp": format_datetime(get_current_datetime()),
            "individual_results": {},
            "comparative_analysis": {},
            "ensemble_insights": {}
        }

        try:
            # Analyze each model individually
            for model_name, model in models.items():
                tprint(f"🔍 Analyzing model: {model_name}")
                self.logger.info(f"🔍 Analyzing model: {model_name}")

                model_results = await self.explain_model(
                    model = model,
                    X_train = X_train,
                    X_test = X_test,
                    y_train = y_train,
                    y_test = y_test,
                    feature_names = feature_names,
                    model_name = model_name,
                    symbol = symbol,
                    exchange = exchange,
                    output_dir = f"{output_dir}/{model_name}"
                )

                results["individual_results"][model_name] = model_results

            # Perform comparative analysis
            tprint("📊 Performing comparative analysis...")
            self.logger.info("📊 Performing comparative analysis...")

            comparative_analysis = await self._perform_comparative_analysis(results["individual_results"])
            results["comparative_analysis"] = comparative_analysis

            # Generate ensemble insights
            tprint("🎯 Generating ensemble insights...")
            self.logger.info("🎯 Generating ensemble insights...")

            ensemble_insights = await self._generate_ensemble_insights(results["individual_results"])
            results["ensemble_insights"] = ensemble_insights

            # Save comprehensive results
            results_file = f"{output_dir}/multi_model_interpretability_results.json"
            safe_json_dump(results, results_file, indent = 2)

            tprint(f"🎉 Multi-model interpretability analysis completed successfully!")
            tprint(f"📄 Results saved to: {results_file}")
            self.logger.info(f"🎉 Multi-model interpretability analysis completed successfully!")
            self.logger.info(f"📄 Results saved to: {results_file}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Multi-model interpretability analysis failed: {e}")
            tprint(f"❌ Multi-model interpretability analysis failed: {e}")
            raise

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _perform_comparative_analysis(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across multiple models."""
        self.logger.info("📊 Performing comparative analysis across models...")

        comparative_analysis = {
            "feature_consistency": {},
            "model_agreement": {},
            "top_features_comparison": {},
            "insights": []
        }

        try:
            model_names = list(individual_results.keys())

            if len(model_names) < 2:
                comparative_analysis["insights"].append("Only one model analyzed - no comparative analysis possible")
                return comparative_analysis

            # Compare top features across models
            all_top_features = {}
            for model_name, results in individual_results.items():
                feature_importance = results.get("feature_importance", {})
                top_features = feature_importance.get("top_features", [])
                all_top_features[model_name] = top_features

            # Find common top features
            if all_top_features:
                all_features = set()
                for features in all_top_features.values():
                    all_features.update(features)

                feature_consistency = {}
                for feature in all_features:
                    count = sum(1 for features in all_top_features.values() if feature in features)
                    feature_consistency[feature] = count / len(model_names)

                comparative_analysis["feature_consistency"] = feature_consistency

                # Find most consistent features
                consistent_features = {k: v for k, v in feature_consistency.items() if v >= 0.5}
                if consistent_features:
                    most_consistent = sorted(consistent_features.items(), key = lambda x: x[1], reverse = True)[:5]
                    comparative_analysis["insights"].append(f"Most consistent features across models: {', '.join([f[0] for f in most_consistent])}")

            tprint("✅ Comparative analysis completed successfully")
            self.logger.info("✅ Comparative analysis completed successfully")

            return comparative_analysis

        except Exception as e:
            self.logger.error(f"❌ Comparative analysis failed: {e}")
            tprint(f"❌ Comparative analysis failed: {e}")
            return comparative_analysis

    @handles_errors(Exception, fallback = False, log_level="ERROR")
    @log_call
    @traced
    async def _generate_ensemble_insights(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights for ensemble of models."""
        self.logger.info("🎯 Generating ensemble insights...")

        ensemble_insights = {
            "ensemble_summary": [],
            "feature_consensus": {},
            "model_diversity": {},
            "recommendations": []
        }

        try:
            model_names = list(individual_results.keys())
            ensemble_insights["ensemble_summary"].append(f"Analyzed ensemble of {len(model_names)} models: {', '.join(model_names)}")

            # Analyze feature consensus
            all_features = set()
            for results in individual_results.values():
                feature_importance = results.get("feature_importance", {})
                top_features = feature_importance.get("top_features", [])
                all_features.update(top_features)

            if all_features:
                ensemble_insights["feature_consensus"]["total_unique_features"] = len(all_features)
                ensemble_insights["recommendations"].append(f"Consider ensemble methods using {len(all_features)} unique important features")

            tprint("✅ Ensemble insights generated successfully")
            self.logger.info("✅ Ensemble insights generated successfully")

            return ensemble_insights

        except Exception as e:
            self.logger.error(f"❌ Ensemble insights generation failed: {e}")
            tprint(f"❌ Ensemble insights generation failed: {e}")
            return ensemble_insights
