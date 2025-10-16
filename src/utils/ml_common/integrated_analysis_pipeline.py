#!/usr/bin/env python3
"""
Integrated Analysis Pipeline

This module demonstrates how to properly integrate the enhanced ML Common features
into the existing trading system pipeline.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

# Import enhanced features
from ...feature_selection.analysis.feature_importance_analyzer import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig, ImportanceMethod,
    analyze_feature_importance, get_important_features
)
from .data_drift_detector import (
    DataDriftDetector, DriftDetectionConfig, DriftMethod, DriftSeverity,
    detect_data_drift, get_drifted_features
)

# HMM tooling has been deprecated and removed

# Import system utilities
from ..logger import get_logger

@dataclass
class IntegratedAnalysisConfig:
    """Configuration for integrated analysis pipeline."""
    # Feature importance settings
    feature_importance_methods: List[ImportanceMethod] = field(default_factory=lambda: [
        ImportanceMethod.RANDOM_FOREST,
        ImportanceMethod.LASSO,
        ImportanceMethod.MUTUAL_INFO
    ])
    top_k_features: int = 20

    # Drift detection settings
    drift_threshold: float = 0.05
    warning_threshold: float = 0.1
    critical_threshold: float = 0.2

    # HMM settings have been removed

    # Output settings
    save_results: bool = True
    output_directory: str = "integrated_analysis_results"

class IntegratedAnalysisPipeline:
    """Integrated analysis pipeline combining all enhanced features."""

    def __init__(self, config: Optional[IntegratedAnalysisConfig] = None):
        self.config = config or IntegratedAnalysisConfig()
        self.logger = get_logger("IntegratedAnalysisPipeline")

        # Initialize components
        self._initialize_components()

        self.logger.info("🚀 Integrated Analysis Pipeline initialized")

    def _initialize_components(self):
        """Initialize all analysis components."""
        # Feature importance analyzer
        feature_config = FeatureImportanceConfig(
            methods=self.config.feature_importance_methods,
            top_k_features=self.config.top_k_features,
            save_results=self.config.save_results,
            output_directory=f"{self.config.output_directory}/feature_importance"
        )
        self.feature_analyzer = FeatureImportanceAnalyzer(feature_config)

        # Data drift detector
        drift_config = DriftDetectionConfig(
            drift_threshold=self.config.drift_threshold,
            warning_threshold=self.config.warning_threshold,
            critical_threshold=self.config.critical_threshold,
            save_results=self.config.save_results,
            output_directory=f"{self.config.output_directory}/drift_detection"
        )
        self.drift_detector = DataDriftDetector(drift_config)

        # Regime detection using alternative methods (HMM deprecated)
        self._regime_detection_available = True
        self.logger.info("ℹ️ Using alternative regime detection methods (HMM deprecated).")

    def analyze_comprehensive(self,
                            current_data: pd.DataFrame,
                            reference_data: Optional[pd.DataFrame] = None,
                            target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all features."""

        start_time = time.time()
        self.logger.info("🔍 Starting comprehensive analysis pipeline")

        results = {}

        # 1. Feature Importance Analysis
        self.logger.info("📊 Step 1: Feature Importance Analysis")
        feature_results = self._analyze_feature_importance(current_data, target_column)
        results['feature_importance'] = feature_results

        # 2. Data Drift Detection (if reference data available)
        if reference_data is not None:
            self.logger.info("🔍 Step 2: Data Drift Detection")
            drift_results = self._detect_data_drift(reference_data, current_data)
            results['drift_detection'] = drift_results
        else:
            self.logger.info("⏭️ Step 2: Skipping drift detection (no reference data)")
            results['drift_detection'] = {'status': 'skipped', 'reason': 'no_reference_data'}

        # 3. Regime Detection (using alternative methods)
        self.logger.info("🔍 Step 3: Regime Detection (using alternative methods)")
        regime_results = self._detect_regimes(current_data)
        results['regime_detection'] = regime_results

        # 4. Integrated Analysis
        self.logger.info("🔍 Step 4: Integrated Analysis")
        integrated_results = self._integrate_analysis(results)
        results['integrated_analysis'] = integrated_results

        # 5. Generate Recommendations
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations

        total_time = time.time() - start_time
        self.logger.info(f"✅ Comprehensive analysis completed in {total_time:.3f}s")

        # Save results
        if self.config.save_results:
            self._save_results(results)

        return results

    def _analyze_feature_importance(self, data: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze feature importance."""
        try:
            if target_column and target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Use the enhanced analyzer with advanced tools integration
                result = self.feature_analyzer.analyze_with_advanced_tools(X, y)

                return {
                    'status': 'success',
                    'top_features': result.get_top_features("ensemble", self.config.top_k_features),
                    'stability_scores': result.stability_scores,
                    'method_scores': result.method_scores,
                    'overall_quality': result.meta_info.get('analysis_time', 0)
                }
            else:
                # Analyze without target (unsupervised)
                result = self.feature_analyzer.analyze_features(data)

                return {
                    'status': 'success_unsupervised',
                    'top_features': result.get_top_features("ensemble", self.config.top_k_features),
                    'stability_scores': result.stability_scores,
                    'method_scores': result.method_scores
                }

        except Exception as e:
            self.logger.error(f"❌ Feature importance analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data."""
        try:
            # Remove non-numeric columns for drift detection
            numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
            ref_numeric = reference_data[numeric_columns]
            cur_numeric = current_data[numeric_columns]

            # Detect drift
            report = self.drift_detector.detect_drift(ref_numeric, cur_numeric)

            return {
                'status': 'success',
                'drift_rate': report.drifted_features / report.total_features if report.total_features > 0 else 0,
                'critical_features': [r.feature_name for r in report.drift_results if r.severity == DriftSeverity.CRITICAL],
                'high_drift_features': [r.feature_name for r in report.drift_results if r.severity == DriftSeverity.HIGH],
                'recommendations': report.recommendations,
                'severity_summary': {sev.value: count for sev, count in report.severity_summary.items()}
            }

        except Exception as e:
            self.logger.error(f"❌ Data drift detection failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _detect_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regimes using alternative methods."""
        # Using volatility-based regime detection (HMM deprecated)
        try:
            # Simple regime detection based on price movements and volatility
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(window=20).std()

            # Define regimes based on volatility thresholds
            high_vol_threshold = volatility.quantile(0.75)
            low_vol_threshold = volatility.quantile(0.25)

            regimes = []
            current_regime = None
            regime_start = 0

            for i in range(len(data)):
                if volatility.iloc[i] > high_vol_threshold:
                    regime = 'high_volatility'
                elif volatility.iloc[i] < low_vol_threshold:
                    regime = 'low_volatility'
                else:
                    regime = 'normal_volatility'

                if regime != current_regime:
                    if current_regime is not None:
                        regimes.append({
                            'regime': current_regime,
                            'start_idx': regime_start,
                            'end_idx': i - 1,
                            'duration': i - regime_start
                        })
                    current_regime = regime
                    regime_start = i

            # Add the last regime
            if current_regime is not None:
                regimes.append({
                    'regime': current_regime,
                    'start_idx': regime_start,
                    'end_idx': len(data) - 1,
                    'duration': len(data) - regime_start
                })

            return {
                'status': 'success',
                'regimes': regimes,
                'n_regimes': len(set(r['regime'] for r in regimes)),
                'method': 'volatility_based'
            }

        except Exception as exc:
            self.logger.error(f"❌ Basic regime detection failed: {exc}")
            return {
                'status': 'error',
                'error': str(exc),
                'regimes': None,
                'method': 'failed'
            }

    def _integrate_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate all analysis results."""
        try:
            integration_results = {
                'feature_regime_correlation': {},
                'drift_regime_impact': {},
                'overall_stability': 0.0,
                'key_insights': []
            }

            # Analyze correlation between feature importance and regimes
            if (results.get('feature_importance', {}).get('status') == 'success' and
                results.get('regime_detection', {}).get('status') == 'success'):

                # Get top features and regime labels
                top_features = results['feature_importance'].get('top_features', [])
                regime_labels = results['regime_detection'].get('regime_labels', [])

                if top_features and regime_labels:
                    # Analyze feature importance across regimes
                    integration_results['feature_regime_correlation'] = {
                        'top_features': top_features[:10],
                        'regime_analysis': 'Features show varying importance across regimes'
                    }

            # Analyze drift impact on regimes
            if (results.get('drift_detection', {}).get('status') == 'success' and
                results.get('regime_detection', {}).get('status') == 'success'):

                drift_rate = results['drift_detection'].get('drift_rate', 0)
                regime_stability = results['regime_detection'].get('regime_stability', 0)

                integration_results['drift_regime_impact'] = {
                    'drift_rate': drift_rate,
                    'regime_stability': regime_stability,
                    'impact_assessment': 'High' if drift_rate > 0.2 else 'Low'
                }

            # Calculate overall stability
            stability_components = []
            if results.get('feature_importance', {}).get('status') == 'success':
                stability_components.append(0.8)  # Feature stability
            if results.get('drift_detection', {}).get('status') == 'success':
                drift_rate = results['drift_detection'].get('drift_rate', 0)
                stability_components.append(1 - drift_rate)  # Drift stability
            if results.get('regime_detection', {}).get('status') == 'success':
                regime_stability = results['regime_detection'].get('regime_stability', 0)
                stability_components.append(regime_stability)  # Regime stability

            if stability_components:
                integration_results['overall_stability'] = np.mean(stability_components)

            # Generate key insights
            insights = []
            if results.get('drift_detection', {}).get('critical_features'):
                insights.append(f"Critical drift detected in {len(results['drift_detection']['critical_features'])} features")
            if results.get('regime_detection', {}).get('regime_stability', 0) > 0.8:
                insights.append("Market regimes show high stability")
            if results.get('feature_importance', {}).get('top_features'):
                insights.append(f"Top {len(results['feature_importance']['top_features'][:5])} features identified")

            integration_results['key_insights'] = insights

            return integration_results

        except Exception as e:
            self.logger.error(f"❌ Integration analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        try:
            # Drift-based recommendations
            if results.get('drift_detection', {}).get('status') == 'success':
                drift_rate = results['drift_detection'].get('drift_rate', 0)
                if drift_rate > 0.2:
                    recommendations.append("CRITICAL: High data drift detected - consider retraining models")
                elif drift_rate > 0.1:
                    recommendations.append("WARNING: Moderate data drift detected - monitor closely")

                critical_features = results['drift_detection'].get('critical_features', [])
                if critical_features:
                    recommendations.append(f"Focus on {len(critical_features)} features showing critical drift")

            # Regime-based recommendations
            if results.get('regime_detection', {}).get('status') == 'success':
                regime_stability = results['regime_detection'].get('regime_stability', 0)
                if regime_stability < 0.7:
                    recommendations.append("Regime instability detected - consider regime-aware modeling")

                n_regimes = results['regime_detection'].get('n_regimes', 0)
                if n_regimes > 6:
                    recommendations.append("Many regimes detected - consider regime consolidation")

            # Feature importance recommendations
            if results.get('feature_importance', {}).get('status') == 'success':
                top_features = results['feature_importance'].get('top_features', [])
                if len(top_features) > 50:
                    recommendations.append("High feature dimensionality - consider feature selection")
                elif len(top_features) < 10:
                    recommendations.append("Low feature dimensionality - consider feature engineering")

            # Overall stability recommendations
            if results.get('integrated_analysis', {}).get('overall_stability', 0) < 0.6:
                recommendations.append("Overall system instability - comprehensive review recommended")

        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations

    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save comprehensive results
            import json
            results_file = output_dir / f"comprehensive_analysis_{int(time.time())}.json"

            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)

            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            self.logger.info(f"💾 Results saved to {results_file}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

# Convenience functions for pipeline integration
def run_comprehensive_analysis(current_data: pd.DataFrame,
                             reference_data: Optional[pd.DataFrame] = None,
                             target_column: Optional[str] = None,
                             config: Optional[IntegratedAnalysisConfig] = None) -> Dict[str, Any]:
    """Run comprehensive analysis pipeline."""
    pipeline = IntegratedAnalysisPipeline(config)
    return pipeline.analyze_comprehensive(current_data, reference_data, target_column)

def detect_regime_changes(current_data: pd.DataFrame,
                         reference_data: pd.DataFrame,
                         config: Optional[IntegratedAnalysisConfig] = None) -> Dict[str, Any]:
    """Detect regime changes using drift detection and regime analysis."""
    pipeline = IntegratedAnalysisPipeline(config)

    # Focus on regime change detection
    results = pipeline.analyze_comprehensive(current_data, reference_data)

    # Extract regime change insights
    regime_changes = {
        'drift_impact': results.get('drift_detection', {}),
        'regime_stability': results.get('regime_detection', {}),
        'stability_assessment': results.get('integrated_analysis', {}),
        'recommendations': results.get('recommendations', [])
    }

    return regime_changes
