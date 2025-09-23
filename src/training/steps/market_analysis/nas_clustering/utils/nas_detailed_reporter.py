"""
NAS Detailed Reporter for comprehensive analysis reporting.

This module provides detailed reporting capabilities for NAS clustering results,
including regime analysis, economic significance, trading viability, and ML training features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NASDetailedReporter:
    """Comprehensive reporter for NAS clustering results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS detailed reporter.
        
        Args:
            config: Reporter configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Reporting settings
        self.enable_detailed_analysis = config.get('enable_detailed_analysis', True)
        self.enable_economic_reporting = config.get('enable_economic_reporting', True)
        self.enable_trading_reporting = config.get('enable_trading_reporting', True)
        self.enable_ml_training_reporting = config.get('enable_ml_training_reporting', True)
        self.enable_micro_regime_reporting = config.get('enable_micro_regime_reporting', True)
        
        # Output settings
        self.output_format = config.get('output_format', 'json')
        self.include_visualizations = config.get('include_visualizations', False)
        self.include_recommendations = config.get('include_recommendations', True)
        
        self.logger.info("✅ NAS Detailed Reporter initialized")
    
    def generate_comprehensive_report(self, clustering_result: Any, 
                                    feature_result: Any,
                                    micro_regime_result: Any,
                                    regime_optimization: Any,
                                    market_data: np.ndarray,
                                    timestamps: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive detailed report.
        
        Args:
            clustering_result: NAS clustering result
            feature_result: Feature extraction result
            micro_regime_result: Micro-regime detection result
            regime_optimization: Regime optimization result
            market_data: Market data array
            timestamps: Timestamps array
            
        Returns:
            Dictionary with comprehensive report
        """
        try:
            self.logger.info("📊 Generating comprehensive detailed report")
            
            # Executive summary
            executive_summary = self._generate_executive_summary(
                clustering_result, feature_result, micro_regime_result, regime_optimization
            )
            
            # Regime analysis
            regime_analysis = self._generate_regime_analysis(
                clustering_result, market_data, timestamps
            )
            
            # Economic analysis
            economic_analysis = self._generate_economic_analysis(
                clustering_result, market_data, timestamps
            )
            
            # Trading analysis
            trading_analysis = self._generate_trading_analysis(
                clustering_result, market_data, timestamps
            )
            
            # Micro-regime analysis
            micro_regime_analysis = self._generate_micro_regime_analysis(
                micro_regime_result, market_data, timestamps
            )
            
            # ML training analysis
            ml_training_analysis = self._generate_ml_training_analysis(
                clustering_result, feature_result, micro_regime_result
            )
            
            # Performance metrics
            performance_metrics = self._generate_performance_metrics(
                clustering_result, feature_result, micro_regime_result, regime_optimization
            )
            
            # Recommendations
            recommendations = self._generate_recommendations(
                clustering_result, feature_result, micro_regime_result, regime_optimization
            )
            
            # Create comprehensive report
            comprehensive_report = {
                'executive_summary': executive_summary,
                'regime_analysis': regime_analysis,
                'economic_analysis': economic_analysis,
                'trading_analysis': trading_analysis,
                'micro_regime_analysis': micro_regime_analysis,
                'ml_training_analysis': ml_training_analysis,
                'performance_metrics': performance_metrics,
                'recommendations': recommendations,
                'metadata': {
                    'report_timestamp': datetime.now().isoformat(),
                    'report_version': '1.0.0',
                    'nas_clustering_version': '1.0.0',
                    'report_type': 'comprehensive_nas_analysis'
                }
            }
            
            self.logger.info("✅ Comprehensive detailed report generated")
            return comprehensive_report
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive report generation failed: {e}")
            return {
                'error': str(e),
                'report_timestamp': datetime.now().isoformat(),
                'report_type': 'comprehensive_nas_analysis'
            }
    
    def _generate_executive_summary(self, clustering_result: Any, feature_result: Any,
                                  micro_regime_result: Any, regime_optimization: Any) -> Dict[str, Any]:
        """Generate executive summary."""
        try:
            return {
                'analysis_overview': {
                    'method': 'nas_clustering',
                    'timeframe': clustering_result.metadata.get('timeframe', '15m'),
                    'n_regimes': clustering_result.metadata.get('n_regimes', 0),
                    'n_samples': len(clustering_result.labels),
                    'n_features': len(feature_result.feature_names) if feature_result else 0,
                    'execution_time': clustering_result.execution_time,
                    'success': clustering_result.success
                },
                
                'key_findings': {
                    'regime_quality': clustering_result.quality_metrics.get('nas_score', 0.0),
                    'economic_significance': np.mean(clustering_result.economic_significance_scores),
                    'trading_viability': np.mean(clustering_result.trading_viability_scores),
                    'micro_regime_accuracy': micro_regime_result.detection_accuracy if micro_regime_result else 0.0,
                    'regime_optimization': regime_optimization.optimal_n_regimes if regime_optimization else 0
                },
                
                'performance_summary': {
                    'overall_quality': 'High' if clustering_result.quality_metrics.get('nas_score', 0.0) > 0.7 else 'Medium' if clustering_result.quality_metrics.get('nas_score', 0.0) > 0.5 else 'Low',
                    'economic_relevance': 'High' if np.mean(clustering_result.economic_significance_scores) > 0.7 else 'Medium' if np.mean(clustering_result.economic_significance_scores) > 0.5 else 'Low',
                    'trading_viability': 'High' if np.mean(clustering_result.trading_viability_scores) > 0.7 else 'Medium' if np.mean(clustering_result.trading_viability_scores) > 0.5 else 'Low',
                    'micro_regime_detection': 'Good' if micro_regime_result.detection_accuracy > 0.7 else 'Fair' if micro_regime_result.detection_accuracy > 0.5 else 'Poor' if micro_regime_result else 'N/A'
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Executive summary generation failed: {e}")
            return {}
    
    def _generate_regime_analysis(self, clustering_result: Any, 
                                market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Generate regime analysis."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            # Regime distribution
            regime_distribution = {}
            regime_percentages = {}
            regime_durations = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                count = np.sum(regime_mask)
                percentage = (count / len(labels)) * 100
                
                regime_distribution[f'regime_{label}'] = int(count)
                regime_percentages[f'regime_{label}'] = float(percentage)
                
                # Calculate regime duration
                regime_indices = np.where(regime_mask)[0]
                if len(regime_indices) > 0:
                    duration = len(regime_indices)
                    regime_durations[f'regime_{label}'] = int(duration)
            
            # Regime characteristics
            regime_characteristics = {}
            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue
                
                regime_data = market_data[regime_mask]
                
                if regime_data.shape[1] >= 4:
                    close_prices = regime_data[:, 3]
                    high_prices = regime_data[:, 1]
                    low_prices = regime_data[:, 2]
                    
                    # Price characteristics
                    price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
                    price_volatility = np.std(close_prices) / np.mean(close_prices)
                    price_range = (np.max(high_prices) - np.min(low_prices)) / np.mean(close_prices)
                    
                    regime_characteristics[f'regime_{label}'] = {
                        'price_change': float(price_change),
                        'price_volatility': float(price_volatility),
                        'price_range': float(price_range),
                        'trend_direction': 'bullish' if price_change > 0.01 else 'bearish' if price_change < -0.01 else 'sideways',
                        'volatility_level': 'high' if price_volatility > 0.05 else 'medium' if price_volatility > 0.02 else 'low'
                    }
            
            # Regime transitions
            regime_transitions = self._calculate_regime_transitions(labels)
            
            return {
                'regime_distribution': regime_distribution,
                'regime_percentages': regime_percentages,
                'regime_durations': regime_durations,
                'regime_characteristics': regime_characteristics,
                'regime_transitions': regime_transitions,
                'n_regimes': len(unique_labels),
                'regime_stability': self._calculate_regime_stability(labels)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime analysis generation failed: {e}")
            return {}
    
    def _generate_economic_analysis(self, clustering_result: Any,
                                  market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Generate economic analysis."""
        try:
            labels = clustering_result.labels
            economic_scores = clustering_result.economic_significance_scores
            unique_labels = np.unique(labels)
            
            # Overall economic significance
            overall_economic_significance = np.mean(economic_scores)
            economic_significance_std = np.std(economic_scores)
            
            # Regime-wise economic analysis
            regime_economic_analysis = {}
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_economic_scores = economic_scores[regime_mask]
                    regime_economic_analysis[f'regime_{label}'] = {
                        'mean_economic_significance': float(np.mean(regime_economic_scores)),
                        'std_economic_significance': float(np.std(regime_economic_scores)),
                        'min_economic_significance': float(np.min(regime_economic_scores)),
                        'max_economic_significance': float(np.max(regime_economic_scores)),
                        'economic_level': 'high' if np.mean(regime_economic_scores) > 0.7 else 'medium' if np.mean(regime_economic_scores) > 0.5 else 'low'
                    }
            
            # Economic regime ranking
            economic_ranking = sorted(
                regime_economic_analysis.items(),
                key=lambda x: x[1]['mean_economic_significance'],
                reverse=True
            )
            
            # Economic significance distribution
            economic_distribution = {
                'high_significance': int(np.sum(economic_scores > 0.7)),
                'medium_significance': int(np.sum((economic_scores > 0.5) & (economic_scores <= 0.7))),
                'low_significance': int(np.sum(economic_scores <= 0.5))
            }
            
            return {
                'overall_economic_significance': float(overall_economic_significance),
                'economic_significance_std': float(economic_significance_std),
                'regime_economic_analysis': regime_economic_analysis,
                'economic_ranking': economic_ranking,
                'economic_distribution': economic_distribution,
                'top_economic_regimes': [item[0] for item in economic_ranking[:3]]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic analysis generation failed: {e}")
            return {}
    
    def _generate_trading_analysis(self, clustering_result: Any,
                                 market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Generate trading analysis."""
        try:
            labels = clustering_result.labels
            trading_scores = clustering_result.trading_viability_scores
            unique_labels = np.unique(labels)
            
            # Overall trading viability
            overall_trading_viability = np.mean(trading_scores)
            trading_viability_std = np.std(trading_scores)
            
            # Regime-wise trading analysis
            regime_trading_analysis = {}
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_trading_scores = trading_scores[regime_mask]
                    regime_trading_analysis[f'regime_{label}'] = {
                        'mean_trading_viability': float(np.mean(regime_trading_scores)),
                        'std_trading_viability': float(np.std(regime_trading_scores)),
                        'min_trading_viability': float(np.min(regime_trading_scores)),
                        'max_trading_viability': float(np.max(regime_trading_scores)),
                        'trading_level': 'high' if np.mean(regime_trading_scores) > 0.7 else 'medium' if np.mean(regime_trading_scores) > 0.5 else 'low'
                    }
            
            # Trading regime ranking
            trading_ranking = sorted(
                regime_trading_analysis.items(),
                key=lambda x: x[1]['mean_trading_viability'],
                reverse=True
            )
            
            # Trading viability distribution
            trading_distribution = {
                'high_viability': int(np.sum(trading_scores > 0.7)),
                'medium_viability': int(np.sum((trading_scores > 0.5) & (trading_scores <= 0.7))),
                'low_viability': int(np.sum(trading_scores <= 0.5))
            }
            
            return {
                'overall_trading_viability': float(overall_trading_viability),
                'trading_viability_std': float(trading_viability_std),
                'regime_trading_analysis': regime_trading_analysis,
                'trading_ranking': trading_ranking,
                'trading_distribution': trading_distribution,
                'top_trading_regimes': [item[0] for item in trading_ranking[:3]]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading analysis generation failed: {e}")
            return {}
    
    def _generate_micro_regime_analysis(self, micro_regime_result: Any,
                                       market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Generate micro-regime analysis."""
        try:
            if micro_regime_result is None:
                return {}
            
            micro_regimes = micro_regime_result.micro_regimes
            micro_regime_types = micro_regime_result.micro_regime_types
            micro_regime_scores = micro_regime_result.micro_regime_scores
            detection_accuracy = micro_regime_result.detection_accuracy
            
            # Micro-regime distribution
            unique_micro_regimes = np.unique(micro_regimes)
            micro_regime_distribution = {}
            micro_regime_percentages = {}
            
            for regime_id in unique_micro_regimes:
                regime_mask = micro_regimes == regime_id
                count = np.sum(regime_mask)
                percentage = (count / len(micro_regimes)) * 100
                
                micro_regime_distribution[f'micro_regime_{regime_id}'] = int(count)
                micro_regime_percentages[f'micro_regime_{regime_id}'] = float(percentage)
            
            # Micro-regime type analysis
            micro_regime_type_analysis = {}
            for i, regime_type in enumerate(micro_regime_types):
                regime_mask = micro_regimes == i
                if np.any(regime_mask):
                    regime_scores = micro_regime_scores[regime_mask]
                    micro_regime_type_analysis[regime_type.value] = {
                        'count': int(np.sum(regime_mask)),
                        'mean_score': float(np.mean(regime_scores)),
                        'std_score': float(np.std(regime_scores)),
                        'quality_level': 'high' if np.mean(regime_scores) > 0.7 else 'medium' if np.mean(regime_scores) > 0.5 else 'low'
                    }
            
            return {
                'detection_accuracy': float(detection_accuracy),
                'n_micro_regimes': len(unique_micro_regimes),
                'micro_regime_distribution': micro_regime_distribution,
                'micro_regime_percentages': micro_regime_percentages,
                'micro_regime_type_analysis': micro_regime_type_analysis,
                'micro_regime_types': [t.value for t in micro_regime_types]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime analysis generation failed: {e}")
            return {}
    
    def _generate_ml_training_analysis(self, clustering_result: Any, feature_result: Any,
                                     micro_regime_result: Any) -> Dict[str, Any]:
        """Generate ML training analysis."""
        try:
            # Feature analysis
            feature_analysis = {
                'n_features': len(feature_result.feature_names) if feature_result else 0,
                'feature_names': feature_result.feature_names if feature_result else [],
                'feature_metadata': feature_result.feature_metadata if feature_result else {}
            }
            
            # Regime features for ML training
            regime_features = {
                'regime_labels': clustering_result.labels.tolist(),
                'regime_centers': clustering_result.cluster_centers.tolist(),
                'regime_statistics': clustering_result.statistics,
                'regime_quality_metrics': clustering_result.quality_metrics
            }
            
            # Economic features for ML training
            economic_features = {
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'mean_economic_significance': float(np.mean(clustering_result.economic_significance_scores)),
                'std_economic_significance': float(np.std(clustering_result.economic_significance_scores))
            }
            
            # Trading features for ML training
            trading_features = {
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                'mean_trading_viability': float(np.mean(clustering_result.trading_viability_scores)),
                'std_trading_viability': float(np.std(clustering_result.trading_viability_scores))
            }
            
            # Micro-regime features for ML training
            micro_regime_features = {}
            if micro_regime_result is not None:
                micro_regime_features = {
                    'micro_regime_labels': micro_regime_result.micro_regimes.tolist(),
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types],
                    'micro_regime_scores': micro_regime_result.micro_regime_scores.tolist(),
                    'detection_accuracy': micro_regime_result.detection_accuracy
                }
            
            # ML model compatibility
            ml_model_compatibility = {
                'supported_models': ['DeepScale', 'LGBM', 'XGBoost', 'RandomForest', 'SVM', 'NeuralNetwork'],
                'feature_types': ['regime_features', 'transition_features', 'economic_features', 'trading_features', 'micro_regime_features'],
                'data_ready': True,
                'feature_count': len(feature_result.feature_names) if feature_result else 0
            }
            
            return {
                'feature_analysis': feature_analysis,
                'regime_features': regime_features,
                'economic_features': economic_features,
                'trading_features': trading_features,
                'micro_regime_features': micro_regime_features,
                'ml_model_compatibility': ml_model_compatibility
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ ML training analysis generation failed: {e}")
            return {}
    
    def _generate_performance_metrics(self, clustering_result: Any, feature_result: Any,
                                     micro_regime_result: Any, regime_optimization: Any) -> Dict[str, Any]:
        """Generate performance metrics."""
        try:
            return {
                'clustering_performance': {
                    'silhouette_score': clustering_result.quality_metrics.get('silhouette_score', 0.0),
                    'nas_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                    'calinski_harabasz_score': clustering_result.quality_metrics.get('calinski_harabasz_score', 0.0),
                    'execution_time': clustering_result.execution_time
                },
                
                'regime_optimization_performance': {
                    'optimal_n_regimes': regime_optimization.optimal_n_regimes if regime_optimization else 0,
                    'optimization_scores': regime_optimization.optimization_scores if regime_optimization else {},
                    'optimization_method': regime_optimization.optimization_method if regime_optimization else 'unknown'
                },
                
                'micro_regime_performance': {
                    'detection_accuracy': micro_regime_result.detection_accuracy if micro_regime_result else 0.0,
                    'n_micro_regimes': len(micro_regime_result.micro_regime_types) if micro_regime_result else 0
                },
                
                'feature_extraction_performance': {
                    'n_features': len(feature_result.feature_names) if feature_result else 0,
                    'execution_time': feature_result.execution_time if feature_result else 0.0
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics generation failed: {e}")
            return {}
    
    def _generate_recommendations(self, clustering_result: Any, feature_result: Any,
                                micro_regime_result: Any, regime_optimization: Any) -> List[str]:
        """Generate recommendations."""
        try:
            recommendations = []
            
            # Quality recommendations
            nas_score = clustering_result.quality_metrics.get('nas_score', 0.0)
            if nas_score < 0.5:
                recommendations.append("Consider adjusting NAS architecture parameters to improve regime quality")
            
            # Economic significance recommendations
            economic_significance = np.mean(clustering_result.economic_significance_scores)
            if economic_significance < 0.6:
                recommendations.append("Consider enhancing economic significance features for better regime detection")
            
            # Trading viability recommendations
            trading_viability = np.mean(clustering_result.trading_viability_scores)
            if trading_viability < 0.6:
                recommendations.append("Consider improving trading viability features for better trading decisions")
            
            # Micro-regime recommendations
            if micro_regime_result and micro_regime_result.detection_accuracy < 0.6:
                recommendations.append("Consider adjusting micro-regime detection sensitivity for better micro-regime detection")
            
            # Regime optimization recommendations
            if regime_optimization and regime_optimization.optimal_n_regimes < 8:
                recommendations.append("Consider increasing regime count for better market state differentiation")
            elif regime_optimization and regime_optimization.optimal_n_regimes > 15:
                recommendations.append("Consider reducing regime count to avoid over-segmentation")
            
            # ML training recommendations
            if feature_result and len(feature_result.feature_names) < 20:
                recommendations.append("Consider enhancing feature extraction for better ML model performance")
            
            recommendations.append("ML training features are ready for DeepScale, LGBM, XGBoost, and other ML models")
            recommendations.append("Consider using regime features as input features for downstream ML models")
            recommendations.append("Micro-regime features can provide additional context for ML model training")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Recommendations generation failed: {e}")
            return []
    
    def _calculate_regime_transitions(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime transitions."""
        try:
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            transition_counts = {}
            
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                
                if current_regime in unique_labels and next_regime in unique_labels:
                    current_idx = np.where(unique_labels == current_regime)[0][0]
                    next_idx = np.where(unique_labels == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
                    
                    transition_key = f'regime_{current_regime}_to_regime_{next_regime}'
                    transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
            
            # Calculate transition probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            return {
                'transition_matrix': transition_matrix.tolist(),
                'transition_probabilities': transition_probs.tolist(),
                'transition_counts': transition_counts,
                'n_transitions': int(np.sum(transition_matrix))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transitions calculation failed: {e}")
            return {}
    
    def _calculate_regime_stability(self, labels: np.ndarray) -> float:
        """Calculate regime stability."""
        try:
            if len(labels) < 2:
                return 0.0
            
            # Calculate regime changes
            regime_changes = np.sum(np.diff(labels) != 0)
            total_periods = len(labels) - 1
            
            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return float(stability)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability calculation failed: {e}")
            return 0.0
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> bool:
        """Save detailed report to file.
        
        Args:
            report: Comprehensive report dictionary
            output_path: Output file path
            
        Returns:
            Success status
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"✅ Detailed report saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save detailed report: {e}")
            return False