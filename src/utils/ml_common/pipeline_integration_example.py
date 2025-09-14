#!/usr/bin/env python3
"""
Pipeline Integration Example

This module shows how to integrate the enhanced ML Common features
into the existing trading system pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Import enhanced features
from .integrated_analysis_pipeline import (
    IntegratedAnalysisPipeline, IntegratedAnalysisConfig,
    run_comprehensive_analysis, detect_regime_changes
)
from ..feature_selection.feature_importance_analyzer import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig, ImportanceMethod
)
from .data_drift_detector import (
    DataDriftDetector, DriftDetectionConfig, DriftMethod
)

# Import existing pipeline components
from ..logger import get_logger

class EnhancedPipelineIntegration:
    """Integration of enhanced features into existing pipeline."""
    
    def __init__(self):
        self.logger = get_logger("EnhancedPipelineIntegration")
        
    def integrate_with_step08_feature_selection(self, 
                                             symbol: str, 
                                             exchange: str, 
                                             timeframe: str, 
                                             data_dir: str,
                                             matrix_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with existing Step08 feature selection."""
        
        self.logger.info(f"🔗 Integrating enhanced analysis with Step08 for {symbol}")
        
        try:
            # Load data
            data_file = Path(data_dir) / f"{symbol.lower()}_{timeframe}_features.parquet"
            if not data_file.exists():
                self.logger.warning(f"⚠️ Data file not found: {data_file}")
                return {'status': 'error', 'error': 'Data file not found'}
            
            data = pd.read_parquet(data_file)
            
            # Configure enhanced analysis
            config = IntegratedAnalysisConfig(
                feature_importance_methods=[
                    ImportanceMethod.RANDOM_FOREST,
                    ImportanceMethod.LASSO,
                    ImportanceMethod.MUTUAL_INFO
                ],
                top_k_features=50,
                drift_threshold=0.05,
                hmm_n_components=4,
                save_results=True,
                output_directory=f"enhanced_analysis/{symbol}_{timeframe}"
            )
            
            # Run comprehensive analysis
            results = run_comprehensive_analysis(
                current_data=data,
                reference_data=None,  # Could load historical reference data
                target_column=None,   # Could specify target if available
                config=config
            )
            
            # Extract feature selection insights
            feature_insights = self._extract_feature_selection_insights(results)
            
            # Integrate with existing matrix operations
            enhanced_matrix_data = self._enhance_matrix_data(matrix_data, feature_insights)
            
            self.logger.info("✅ Enhanced analysis integrated successfully")
            
            return {
                'status': 'success',
                'enhanced_analysis': results,
                'feature_insights': feature_insights,
                'enhanced_matrix_data': enhanced_matrix_data
            }
            
        except Exception as e:
            self.logger.error(f"❌ Integration failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def integrate_with_hmm_regime_discovery(self, 
                                          symbol: str, 
                                          exchange: str, 
                                          timeframe: str, 
                                          data_dir: str,
                                          regime_labels: np.ndarray) -> Dict[str, Any]:
        """Integrate with existing HMM regime discovery."""
        
        self.logger.info(f"🔗 Integrating enhanced analysis with HMM regime discovery for {symbol}")
        
        try:
            # Load data
            data_file = Path(data_dir) / f"{symbol.lower()}_{timeframe}_features.parquet"
            if not data_file.exists():
                return {'status': 'error', 'error': 'Data file not found'}
            
            data = pd.read_parquet(data_file)
            
            # Configure for regime-aware analysis
            config = IntegratedAnalysisConfig(
                feature_importance_methods=[
                    ImportanceMethod.RANDOM_FOREST,
                    ImportanceMethod.MUTUAL_INFO,
                    ImportanceMethod.PERMUTATION
                ],
                top_k_features=30,
                drift_threshold=0.03,  # Stricter for regime analysis
                hmm_n_components=len(np.unique(regime_labels)),
                save_results=True,
                output_directory=f"regime_analysis/{symbol}_{timeframe}"
            )
            
            # Run regime-aware analysis
            results = run_comprehensive_analysis(
                current_data=data,
                reference_data=None,
                target_column=None,
                config=config
            )
            
            # Analyze regime-specific insights
            regime_insights = self._analyze_regime_specific_insights(results, regime_labels)
            
            self.logger.info("✅ Regime-aware analysis integrated successfully")
            
            return {
                'status': 'success',
                'regime_analysis': results,
                'regime_insights': regime_insights,
                'regime_stability': self._assess_regime_stability(regime_labels)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Regime integration failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def monitor_data_quality_over_time(self, 
                                     symbol: str, 
                                     exchange: str, 
                                     timeframe: str, 
                                     data_dir: str,
                                     historical_reference: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Monitor data quality and drift over time."""
        
        self.logger.info(f"🔍 Monitoring data quality for {symbol}")
        
        try:
            # Load current data
            current_file = Path(data_dir) / f"{symbol.lower()}_{timeframe}_features.parquet"
            if not current_file.exists():
                return {'status': 'error', 'error': 'Current data file not found'}
            
            current_data = pd.read_parquet(current_file)
            
            # Load reference data if not provided
            if historical_reference is None:
                # Try to load historical reference data
                reference_file = Path(data_dir) / f"{symbol.lower()}_{timeframe}_reference.parquet"
                if reference_file.exists():
                    historical_reference = pd.read_parquet(reference_file)
                else:
                    self.logger.warning("⚠️ No reference data available for drift detection")
                    return {'status': 'warning', 'message': 'No reference data available'}
            
            # Configure drift detection
            drift_config = DriftDetectionConfig(
                methods=[
                    DriftMethod.KS_TEST,
                    DriftMethod.PSI,
                    DriftMethod.WASSERSTEIN
                ],
                drift_threshold=0.05,
                warning_threshold=0.1,
                critical_threshold=0.2,
                enable_alerts=True,
                save_results=True,
                output_directory=f"drift_monitoring/{symbol}_{timeframe}"
            )
            
            # Detect drift
            detector = DataDriftDetector(drift_config)
            drift_report = detector.detect_drift(historical_reference, current_data)
            
            # Generate monitoring insights
            monitoring_insights = {
                'drift_rate': drift_report.drifted_features / drift_report.total_features,
                'critical_features': [r.feature_name for r in drift_report.drift_results if r.severity.value == 'critical'],
                'recommendations': drift_report.recommendations,
                'alert_level': self._determine_alert_level(drift_report)
            }
            
            self.logger.info(f"✅ Data quality monitoring completed - drift rate: {monitoring_insights['drift_rate']:.2%}")
            
            return {
                'status': 'success',
                'drift_report': drift_report,
                'monitoring_insights': monitoring_insights,
                'alert_required': monitoring_insights['alert_level'] in ['high', 'critical']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Data quality monitoring failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_feature_selection_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights for feature selection."""
        feature_importance = results.get('feature_importance', {})
        
        if feature_importance.get('status') != 'success':
            return {'status': 'error', 'error': 'Feature importance analysis failed'}
        
        return {
            'top_features': feature_importance.get('top_features', [])[:20],
            'feature_stability': feature_importance.get('stability_scores', {}),
            'method_agreement': self._calculate_method_agreement(feature_importance.get('method_scores', {})),
            'selection_recommendations': self._generate_selection_recommendations(feature_importance)
        }
    
    def _analyze_regime_specific_insights(self, results: Dict[str, Any], regime_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime-specific insights."""
        regime_detection = results.get('regime_detection', {})
        feature_importance = results.get('feature_importance', {})
        
        insights = {
            'regime_count': len(np.unique(regime_labels)),
            'regime_distribution': {str(i): int(np.sum(regime_labels == i)) for i in np.unique(regime_labels)},
            'regime_stability': regime_detection.get('regime_stability', 0),
            'feature_regime_correlation': {}
        }
        
        # Analyze feature importance across regimes
        if feature_importance.get('status') == 'success':
            top_features = feature_importance.get('top_features', [])
            insights['feature_regime_correlation'] = {
                'top_features_per_regime': self._analyze_features_per_regime(top_features, regime_labels),
                'regime_specific_importance': 'Features show varying importance across regimes'
            }
        
        return insights
    
    def _assess_regime_stability(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Assess regime stability."""
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        
        # Calculate stability metrics
        regime_balance = 1 - (np.std(counts) / np.mean(counts)) if len(counts) > 1 else 1.0
        min_regime_size = np.min(counts)
        max_regime_size = np.max(counts)
        
        return {
            'regime_balance': regime_balance,
            'min_regime_size': int(min_regime_size),
            'max_regime_size': int(max_regime_size),
            'stability_score': regime_balance * (min_regime_size / max_regime_size),
            'stability_assessment': 'stable' if regime_balance > 0.7 else 'unstable'
        }
    
    def _calculate_method_agreement(self, method_scores: Dict[str, Dict[str, float]]) -> float:
        """Calculate agreement between different feature importance methods."""
        if len(method_scores) < 2:
            return 1.0
        
        # Get top features from each method
        method_top_features = {}
        for method, scores in method_scores.items():
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            method_top_features[method] = [f for f, _ in sorted_features[:10]]
        
        # Calculate overlap between methods
        all_features = set()
        for features in method_top_features.values():
            all_features.update(features)
        
        if not all_features:
            return 0.0
        
        # Calculate average overlap
        overlaps = []
        methods = list(method_top_features.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                features1 = set(method_top_features[methods[i]])
                features2 = set(method_top_features[methods[j]])
                overlap = len(features1.intersection(features2)) / len(features1.union(features2))
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _generate_selection_recommendations(self, feature_importance: Dict[str, Any]) -> List[str]:
        """Generate feature selection recommendations."""
        recommendations = []
        
        top_features = feature_importance.get('top_features', [])
        stability_scores = feature_importance.get('stability_scores', {})
        
        if len(top_features) > 100:
            recommendations.append("High feature count - consider aggressive feature selection")
        elif len(top_features) < 20:
            recommendations.append("Low feature count - consider feature engineering")
        
        # Check stability of top features
        stable_features = [f for f, score in stability_scores.items() if score > 0.7]
        if len(stable_features) < len(top_features) * 0.5:
            recommendations.append("Many features show low stability - prioritize stable features")
        
        return recommendations
    
    def _analyze_features_per_regime(self, top_features: List[str], regime_labels: np.ndarray) -> Dict[str, List[str]]:
        """Analyze feature importance per regime."""
        # This is a simplified implementation
        # In practice, you would analyze feature importance within each regime
        regime_features = {}
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_size = np.sum(regime_mask)
            
            if regime_size > 50:  # Minimum samples per regime
                # Take a subset of top features for this regime
                n_features = min(10, len(top_features))
                regime_features[str(regime)] = top_features[:n_features]
        
        return regime_features
    
    def _determine_alert_level(self, drift_report) -> str:
        """Determine alert level based on drift report."""
        critical_count = len([r for r in drift_report.drift_results if r.severity.value == 'critical'])
        high_count = len([r for r in drift_report.drift_results if r.severity.value == 'high'])
        
        if critical_count > 0:
            return 'critical'
        elif high_count > 3:
            return 'high'
        elif high_count > 0:
            return 'medium'
        else:
            return 'low'
    
    def _enhance_matrix_data(self, matrix_data: Dict[str, Any], feature_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing matrix data with feature insights."""
        enhanced_data = matrix_data.copy()
        
        # Add feature importance information
        enhanced_data['feature_importance'] = {
            'top_features': feature_insights.get('top_features', []),
            'method_agreement': feature_insights.get('method_agreement', 0),
            'stability_scores': feature_insights.get('feature_stability', {}),
            'recommendations': feature_insights.get('selection_recommendations', [])
        }
        
        return enhanced_data

# Usage examples for pipeline integration
def integrate_with_existing_pipeline():
    """Example of how to integrate with existing pipeline."""
    
    integration = EnhancedPipelineIntegration()
    
    # Example 1: Integrate with Step08 feature selection
    step08_results = integration.integrate_with_step08_feature_selection(
        symbol="ETHUSDT",
        exchange="binance", 
        timeframe="1m",
        data_dir="historical_data",
        matrix_data={}  # Existing matrix operations data
    )
    
    # Example 2: Integrate with HMM regime discovery
    regime_labels = np.random.randint(0, 4, 1000)  # Example regime labels
    hmm_results = integration.integrate_with_hmm_regime_discovery(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1m", 
        data_dir="historical_data",
        regime_labels=regime_labels
    )
    
    # Example 3: Monitor data quality over time
    monitoring_results = integration.monitor_data_quality_over_time(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1m",
        data_dir="historical_data"
    )
    
    return {
        'step08_integration': step08_results,
        'hmm_integration': hmm_results,
        'quality_monitoring': monitoring_results
    }

if __name__ == "__main__":
    # Run integration example
    results = integrate_with_existing_pipeline()
    print("Pipeline integration completed successfully")