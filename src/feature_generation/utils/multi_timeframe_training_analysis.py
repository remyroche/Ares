"""
Multi-Timeframe Training Analysis Framework

This module provides tools to analyze whether multi-timeframe training
is beneficial when using cross-timeframe features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

from src.utils.logger import system_logger

logger = system_logger.getChild('MultiTimeframeTrainingAnalysis')

@dataclass
class TrainingApproach:
    """Configuration for different training approaches."""
    name: str
    timeframes: List[str]
    target_timeframe: str
    use_cross_timeframe_features: bool
    use_regime_awareness: bool
    description: str

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    approach: str
    timeframe: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    inference_time: float
    model_complexity: int
    feature_importance: Dict[str, float]
    # Multi-output specific metrics
    multi_output_accuracy: Optional[float] = None
    per_output_metrics: Optional[Dict[str, Dict[str, float]]] = None
    overall_mse: Optional[float] = None
    overall_mae: Optional[float] = None
    overall_r2: Optional[float] = None
    confidence_calibration: Optional[Dict[str, float]] = None

@dataclass
class MultiOutputAnalysis:
    """Multi-output analysis results."""
    output_names: List[str]
    per_output_performance: Dict[str, Dict[str, float]]
    overall_performance: Dict[str, float]
    output_correlations: Dict[str, Dict[str, float]]
    stacking_ensemble_performance: Dict[str, float]
    confidence_calibration: Dict[str, float]
    feature_importance_per_output: Dict[str, Dict[str, float]]

class MultiTimeframeTrainingAnalyzer:
    """
    Analyzer to determine the best training approach when using cross-timeframe features.
    """
    
    def __init__(self):
        self.logger = logger.getChild('MultiTimeframeTrainingAnalyzer')
        self.results = {}
        self.multi_output_results = {}
    
    def define_training_approaches(self) -> List[TrainingApproach]:
        """Define different training approaches to compare."""
        approaches = [
            TrainingApproach(
                name="single_timeframe_base",
                timeframes=['1m'],
                target_timeframe='1m',
                use_cross_timeframe_features=True,
                use_regime_awareness=False,
                description="Single timeframe (1m) with cross-timeframe features"
            ),
            TrainingApproach(
                name="single_timeframe_medium",
                timeframes=['5m'],
                target_timeframe='5m',
                use_cross_timeframe_features=True,
                use_regime_awareness=False,
                description="Single timeframe (5m) with cross-timeframe features"
            ),
            TrainingApproach(
                name="multi_timeframe_parallel",
                timeframes=['1m', '5m', '15m'],
                target_timeframe='1m',  # Primary target
                use_cross_timeframe_features=True,
                use_regime_awareness=False,
                description="Multi-timeframe parallel training with cross-timeframe features"
            ),
            TrainingApproach(
                name="multi_timeframe_hierarchical",
                timeframes=['1m', '5m', '15m'],
                target_timeframe='1m',
                use_cross_timeframe_features=True,
                use_regime_awareness=False,
                description="Hierarchical multi-timeframe training"
            ),
            TrainingApproach(
                name="regime_aware_single",
                timeframes=['1m'],
                target_timeframe='1m',
                use_cross_timeframe_features=True,
                use_regime_awareness=True,
                description="Regime-aware single timeframe training"
            ),
            TrainingApproach(
                name="regime_aware_multi",
                timeframes=['1m', '5m', '15m'],
                target_timeframe='1m',
                use_cross_timeframe_features=True,
                use_regime_awareness=True,
                description="Regime-aware multi-timeframe training"
            )
        ]
        
        return approaches
    
    def analyze_multi_output_performance(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        output_names: List[str],
        approach: str,
        timeframe: str
    ) -> MultiOutputAnalysis:
        """
        Analyze multi-output model performance.
        
        Args:
            y_true: True labels (n_samples, n_outputs)
            y_pred: Predicted values (n_samples, n_outputs)
            output_names: List of output names
            approach: Training approach name
            timeframe: Timeframe name
            
        Returns:
            MultiOutputAnalysis object
        """
        self.logger.info(f"🔄 Analyzing multi-output performance for {approach} ({timeframe})")
        
        try:
            n_outputs = y_true.shape[1]
            per_output_performance = {}
            overall_metrics = {}
            
            # Calculate metrics for each output
            for i, output_name in enumerate(output_names):
                y_true_output = y_true[:, i]
                y_pred_output = y_pred[:, i]
                
                # Basic regression metrics
                mse = np.mean((y_true_output - y_pred_output) ** 2)
                mae = np.mean(np.abs(y_true_output - y_pred_output))
                r2 = 1 - (np.sum((y_true_output - y_pred_output) ** 2) / 
                         np.sum((y_true_output - np.mean(y_true_output)) ** 2))
                
                # Classification metrics (if applicable)
                if len(np.unique(y_true_output)) <= 10:  # Likely classification
                    accuracy = np.mean(y_true_output == y_pred_output)
                    precision = self._calculate_precision(y_true_output, y_pred_output)
                    recall = self._calculate_recall(y_true_output, y_pred_output)
                    f1 = self._calculate_f1_score(y_true_output, y_pred_output)
                else:
                    accuracy = r2  # Use R² as accuracy for regression
                    precision = None
                    recall = None
                    f1 = None
                
                per_output_performance[output_name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'accuracy': float(accuracy),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Add to overall metrics
                overall_metrics[f'{output_name}_mse'] = float(mse)
                overall_metrics[f'{output_name}_mae'] = float(mae)
                overall_metrics[f'{output_name}_r2'] = float(r2)
                overall_metrics[f'{output_name}_accuracy'] = float(accuracy)
            
            # Calculate overall performance
            overall_metrics['overall_mse'] = float(np.mean([m['mse'] for m in per_output_performance.values()]))
            overall_metrics['overall_mae'] = float(np.mean([m['mae'] for m in per_output_performance.values()]))
            overall_metrics['overall_r2'] = float(np.mean([m['r2'] for m in per_output_performance.values()]))
            overall_metrics['overall_accuracy'] = float(np.mean([m['accuracy'] for m in per_output_performance.values()]))
            
            # Calculate output correlations
            output_correlations = self._calculate_output_correlations(y_pred, output_names)
            
            # Calculate stacking ensemble performance
            stacking_performance = self._calculate_stacking_ensemble_performance(y_true, y_pred, output_names)
            
            # Calculate confidence calibration (placeholder)
            confidence_calibration = self._calculate_confidence_calibration(y_true, y_pred, output_names)
            
            # Calculate feature importance per output (placeholder)
            feature_importance_per_output = self._calculate_feature_importance_per_output(y_true, y_pred, output_names)
            
            analysis = MultiOutputAnalysis(
                output_names=output_names,
                per_output_performance=per_output_performance,
                overall_performance=overall_metrics,
                output_correlations=output_correlations,
                stacking_ensemble_performance=stacking_performance,
                confidence_calibration=confidence_calibration,
                feature_importance_per_output=feature_importance_per_output
            )
            
            self.multi_output_results[f"{approach}_{timeframe}"] = analysis
            self.logger.info(f"✅ Multi-output analysis completed for {approach} ({timeframe})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Multi-output analysis failed: {e}")
            raise
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision for classification."""
        try:
            from sklearn.metrics import precision_score
            return float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        except:
            return 0.0
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall for classification."""
        try:
            from sklearn.metrics import recall_score
            return float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        except:
            return 0.0
    
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score for classification."""
        try:
            from sklearn.metrics import f1_score
            return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        except:
            return 0.0
    
    def _calculate_output_correlations(
        self, 
        y_pred: np.ndarray, 
        output_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between outputs."""
        correlations = {}
        
        for i, output1 in enumerate(output_names):
            correlations[output1] = {}
            for j, output2 in enumerate(output_names):
                if i != j:
                    corr = np.corrcoef(y_pred[:, i], y_pred[:, j])[0, 1]
                    correlations[output1][output2] = float(corr) if not np.isnan(corr) else 0.0
        
        return correlations
    
    def _calculate_stacking_ensemble_performance(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        output_names: List[str]
    ) -> Dict[str, float]:
        """Calculate stacking ensemble specific performance metrics."""
        try:
            # Calculate ensemble diversity
            diversity_metrics = {}
            
            # Calculate prediction variance across outputs
            pred_variance = np.var(y_pred, axis=1)
            diversity_metrics['prediction_variance'] = float(np.mean(pred_variance))
            
            # Calculate output stability
            output_stability = []
            for i in range(len(output_names)):
                output_std = np.std(y_pred[:, i])
                output_stability.append(output_std)
            diversity_metrics['output_stability'] = float(np.mean(output_stability))
            
            # Calculate ensemble agreement
            ensemble_agreement = 1 - np.mean(pred_variance) / np.var(y_pred)
            diversity_metrics['ensemble_agreement'] = float(ensemble_agreement)
            
            return diversity_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate stacking ensemble performance: {e}")
            return {}
    
    def _calculate_confidence_calibration(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        output_names: List[str]
    ) -> Dict[str, float]:
        """Calculate confidence calibration metrics."""
        try:
            calibration_metrics = {}
            
            # Calculate Brier score for each output
            for i, output_name in enumerate(output_names):
                y_true_output = y_true[:, i]
                y_pred_output = y_pred[:, i]
                
                # Simple Brier score calculation
                brier_score = np.mean((y_true_output - y_pred_output) ** 2)
                calibration_metrics[f'{output_name}_brier_score'] = float(brier_score)
            
            # Calculate overall calibration
            overall_brier = np.mean([(y_true - y_pred) ** 2])
            calibration_metrics['overall_brier_score'] = float(overall_brier)
            
            return calibration_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate confidence calibration: {e}")
            return {}
    
    def _calculate_feature_importance_per_output(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        output_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate feature importance per output (placeholder)."""
        # This would require access to the actual model and features
        # For now, return empty dictionaries
        feature_importance = {}
        for output_name in output_names:
            feature_importance[output_name] = {}
        
        return feature_importance
    
    def compare_multi_output_approaches(self) -> Dict[str, Any]:
        """Compare different multi-output training approaches."""
        self.logger.info("🔄 Comparing multi-output training approaches")
        
        if not self.multi_output_results:
            self.logger.warning("⚠️ No multi-output results available for comparison")
            return {}
        
        comparison = {
            'approaches': list(self.multi_output_results.keys()),
            'overall_performance': {},
            'per_output_performance': {},
            'stacking_ensemble_performance': {},
            'recommendations': []
        }
        
        # Compare overall performance
        for approach, analysis in self.multi_output_results.items():
            comparison['overall_performance'][approach] = analysis.overall_performance
            comparison['per_output_performance'][approach] = analysis.per_output_performance
            comparison['stacking_ensemble_performance'][approach] = analysis.stacking_ensemble_performance
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_multi_output_recommendations()
        
        return comparison
    
    def _generate_multi_output_recommendations(self) -> List[str]:
        """Generate recommendations for multi-output training."""
        recommendations = []
        
        if not self.multi_output_results:
            return recommendations
        
        # Analyze overall performance
        best_overall = max(self.multi_output_results.items(), 
                          key=lambda x: x[1].overall_performance.get('overall_r2', 0))
        recommendations.append(f"Best overall performance: {best_overall[0]} (R²: {best_overall[1].overall_performance.get('overall_r2', 0):.3f})")
        
        # Analyze per-output performance
        for approach, analysis in self.multi_output_results.items():
            for output_name, metrics in analysis.per_output_performance.items():
                if metrics['r2'] > 0.7:
                    recommendations.append(f"Strong performance for {output_name} in {approach}: R² = {metrics['r2']:.3f}")
        
        # Analyze stacking ensemble performance
        best_ensemble = max(self.multi_output_results.items(),
                           key=lambda x: x[1].stacking_ensemble_performance.get('ensemble_agreement', 0))
        recommendations.append(f"Best ensemble agreement: {best_ensemble[0]} (agreement: {best_ensemble[1].stacking_ensemble_performance.get('ensemble_agreement', 0):.3f})")
        
        return recommendations
    
    def analyze_feature_utilization(
        self,
        cross_timeframe_features: pd.DataFrame,
        target_returns: pd.Series
    ) -> Dict[str, Any]:
        """Analyze how well cross-timeframe features capture multi-timeframe information."""
        
        analysis = {
            'feature_categories': {},
            'timeframe_correlations': {},
            'feature_importance_by_timeframe': {},
            'redundancy_analysis': {},
            'information_content': {}
        }
        
        try:
            # Categorize features by timeframe
            feature_categories = self._categorize_features_by_timeframe(cross_timeframe_features)
            analysis['feature_categories'] = feature_categories
            
            # Analyze correlations between different timeframe features
            timeframe_correlations = self._analyze_timeframe_correlations(
                cross_timeframe_features, feature_categories
            )
            analysis['timeframe_correlations'] = timeframe_correlations
            
            # Calculate feature importance for different timeframes
            feature_importance = self._calculate_feature_importance_by_timeframe(
                cross_timeframe_features, target_returns, feature_categories
            )
            analysis['feature_importance_by_timeframe'] = feature_importance
            
            # Analyze redundancy between timeframes
            redundancy_analysis = self._analyze_redundancy(cross_timeframe_features, feature_categories)
            analysis['redundancy_analysis'] = redundancy_analysis
            
            # Calculate information content
            information_content = self._calculate_information_content(
                cross_timeframe_features, target_returns
            )
            analysis['information_content'] = information_content
            
            self.logger.info("✅ Feature utilization analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Feature utilization analysis failed: {e}")
        
        return analysis
    
    def _categorize_features_by_timeframe(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by their timeframe origin."""
        categories = {
            'base_features': [],
            'cross_timeframe_features': [],
            'interaction_features': [],
            'specialized_features': []
        }
        
        for col in features.columns:
            if any(tf in col for tf in ['1m', '5m', '15m', '30m']):
                if '_' in col and any(tf in col for tf in ['1m', '5m', '15m', '30m']):
                    # Cross-timeframe feature (contains multiple timeframes)
                    categories['cross_timeframe_features'].append(col)
                else:
                    # Single timeframe feature
                    categories['base_features'].append(col)
            elif any(term in col for term in ['corr_', 'mom_', 'vol_', 'volume_']):
                categories['interaction_features'].append(col)
            elif any(term in col for term in ['microstructure', 'order_flow', 'momentum_divergence']):
                categories['specialized_features'].append(col)
            else:
                categories['base_features'].append(col)
        
        return categories
    
    def _analyze_timeframe_correlations(
        self,
        features: pd.DataFrame,
        feature_categories: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Analyze correlations between different timeframe features."""
        correlations = {}
        
        try:
            # Get timeframe-specific features
            timeframe_features = {}
            for tf in ['1m', '5m', '15m', '30m']:
                tf_features = [col for col in features.columns if tf in col and '_' not in col.split(tf)[0]]
                if tf_features:
                    timeframe_features[tf] = tf_features
            
            # Calculate pairwise correlations between timeframes
            for tf1, features1 in timeframe_features.items():
                for tf2, features2 in timeframe_features.items():
                    if tf1 != tf2 and features1 and features2:
                        # Calculate average correlation between timeframe features
                        corr_values = []
                        for f1 in features1:
                            for f2 in features2:
                                if f1 in features.columns and f2 in features.columns:
                                    corr = features[f1].corr(features[f2])
                                    if not np.isnan(corr):
                                        corr_values.append(abs(corr))
                        
                        if corr_values:
                            correlations[f'{tf1}_{tf2}'] = np.mean(corr_values)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Timeframe correlation analysis failed: {e}")
        
        return correlations
    
    def _calculate_feature_importance_by_timeframe(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_categories: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate feature importance grouped by timeframe."""
        importance_by_timeframe = {}
        
        try:
            for category, feature_list in feature_categories.items():
                if not feature_list:
                    continue
                
                category_importance = {}
                for feature in feature_list:
                    if feature in features.columns:
                        try:
                            # Calculate correlation with target
                            corr = features[feature].corr(target)
                            if not np.isnan(corr):
                                category_importance[feature] = abs(corr)
                        except:
                            category_importance[feature] = 0.0
                
                if category_importance:
                    # Sort by importance
                    sorted_importance = dict(sorted(
                        category_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ))
                    importance_by_timeframe[category] = sorted_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance calculation failed: {e}")
        
        return importance_by_timeframe
    
    def _analyze_redundancy(
        self,
        features: pd.DataFrame,
        feature_categories: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Analyze redundancy between different timeframe features."""
        redundancy_analysis = {
            'high_correlation_pairs': [],
            'redundant_features': [],
            'unique_information_ratio': 0.0
        }
        
        try:
            # Find high correlation pairs
            corr_matrix = features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_pairs = []
            for col in upper_triangle.columns:
                high_corr = upper_triangle[col][upper_triangle[col] > 0.95]
                for feature, corr in high_corr.items():
                    high_corr_pairs.append((feature, col, corr))
            
            redundancy_analysis['high_correlation_pairs'] = high_corr_pairs
            
            # Identify redundant features
            redundant_features = set()
            for feature1, feature2, corr in high_corr_pairs:
                # Keep the feature with more information (higher variance)
                var1 = features[feature1].var()
                var2 = features[feature2].var()
                
                if var1 > var2:
                    redundant_features.add(feature2)
                else:
                    redundant_features.add(feature1)
            
            redundancy_analysis['redundant_features'] = list(redundant_features)
            
            # Calculate unique information ratio
            total_features = len(features.columns)
            unique_features = total_features - len(redundant_features)
            redundancy_analysis['unique_information_ratio'] = unique_features / total_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Redundancy analysis failed: {e}")
        
        return redundancy_analysis
    
    def _calculate_information_content(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Calculate information content of features."""
        information_content = {
            'mutual_information': {},
            'variance_explained': {},
            'feature_diversity': 0.0,
            'information_density': 0.0
        }
        
        try:
            # Calculate mutual information with target
            for col in features.columns:
                try:
                    # Simple correlation-based mutual information proxy
                    corr = features[col].corr(target)
                    if not np.isnan(corr):
                        information_content['mutual_information'][col] = abs(corr)
                except:
                    information_content['mutual_information'][col] = 0.0
            
            # Calculate variance explained
            total_variance = features.var().sum()
            for col in features.columns:
                feature_variance = features[col].var()
                information_content['variance_explained'][col] = feature_variance / total_variance
            
            # Calculate feature diversity (inverse of average correlation)
            corr_matrix = features.corr().abs()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            information_content['feature_diversity'] = 1 - avg_correlation
            
            # Calculate information density
            high_info_features = sum(1 for mi in information_content['mutual_information'].values() if mi > 0.1)
            information_content['information_density'] = high_info_features / len(features.columns)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Information content calculation failed: {e}")
        
        return information_content
    
    def recommend_training_approach(
        self,
        feature_analysis: Dict[str, Any],
        data_size: int,
        available_compute: str = "medium"
    ) -> Dict[str, Any]:
        """Recommend the best training approach based on analysis."""
        
        recommendations = {
            'recommended_approach': 'single_timeframe_base',
            'confidence': 0.0,
            'reasoning': [],
            'alternatives': [],
            'implementation_notes': []
        }
        
        try:
            # Analyze feature utilization
            cross_timeframe_ratio = len(feature_analysis.get('feature_categories', {}).get('cross_timeframe_features', []))
            total_features = sum(len(features) for features in feature_analysis.get('feature_categories', {}).values())
            cross_timeframe_ratio = cross_timeframe_ratio / total_features if total_features > 0 else 0
            
            # Analyze redundancy
            unique_info_ratio = feature_analysis.get('redundancy_analysis', {}).get('unique_information_ratio', 0.0)
            
            # Analyze information content
            feature_diversity = feature_analysis.get('information_content', {}).get('feature_diversity', 0.0)
            information_density = feature_analysis.get('information_content', {}).get('information_density', 0.0)
            
            # Make recommendation based on analysis
            if cross_timeframe_ratio > 0.3 and unique_info_ratio > 0.7 and feature_diversity > 0.5:
                # Cross-timeframe features are comprehensive and diverse
                recommendations['recommended_approach'] = 'single_timeframe_base'
                recommendations['confidence'] = 0.9
                recommendations['reasoning'].append("Cross-timeframe features provide comprehensive multi-timeframe information")
                recommendations['reasoning'].append("High feature diversity and unique information content")
                recommendations['reasoning'].append("Single timeframe training will be more efficient and stable")
                
            elif cross_timeframe_ratio < 0.2 or unique_info_ratio < 0.5:
                # Cross-timeframe features are limited
                recommendations['recommended_approach'] = 'multi_timeframe_parallel'
                recommendations['confidence'] = 0.7
                recommendations['reasoning'].append("Limited cross-timeframe features may miss important patterns")
                recommendations['reasoning'].append("Multi-timeframe training can capture additional information")
                
            elif data_size > 100000 and available_compute == "high":
                # Large dataset with high compute - consider regime-aware
                recommendations['recommended_approach'] = 'regime_aware_single'
                recommendations['confidence'] = 0.8
                recommendations['reasoning'].append("Large dataset allows for regime-specific modeling")
                recommendations['reasoning'].append("Cross-timeframe features + regime awareness provides best of both worlds")
                
            else:
                # Default to single timeframe
                recommendations['recommended_approach'] = 'single_timeframe_base'
                recommendations['confidence'] = 0.6
                recommendations['reasoning'].append("Balanced approach with good performance and simplicity")
            
            # Add implementation notes
            if recommendations['recommended_approach'] == 'single_timeframe_base':
                recommendations['implementation_notes'].append("Use 1m as base timeframe for high-frequency trading")
                recommendations['implementation_notes'].append("Include all cross-timeframe features in single model")
                recommendations['implementation_notes'].append("Focus on feature selection to avoid overfitting")
                
            elif recommendations['recommended_approach'] == 'multi_timeframe_parallel':
                recommendations['implementation_notes'].append("Train separate models for each timeframe")
                recommendations['implementation_notes'].append("Use ensemble methods to combine predictions")
                recommendations['implementation_notes'].append("Consider different feature sets for each timeframe")
                
            elif recommendations['recommended_approach'] == 'regime_aware_single':
                recommendations['implementation_notes'].append("Implement regime detection first")
                recommendations['implementation_notes'].append("Train separate models for each regime")
                recommendations['implementation_notes'].append("Use cross-timeframe features in each regime model")
            
            # Add alternatives
            if recommendations['recommended_approach'] != 'single_timeframe_base':
                recommendations['alternatives'].append({
                    'approach': 'single_timeframe_base',
                    'reason': 'Simpler and often more effective with good cross-timeframe features'
                })
            
            if recommendations['recommended_approach'] != 'regime_aware_single':
                recommendations['alternatives'].append({
                    'approach': 'regime_aware_single',
                    'reason': 'Better for markets with distinct regimes'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            recommendations['reasoning'].append(f"Analysis failed: {e}")
        
        return recommendations
    
    def generate_implementation_guide(
        self,
        recommended_approach: str,
        feature_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate implementation guide for the recommended approach."""
        
        guide = {
            'approach': recommended_approach,
            'implementation_steps': [],
            'code_examples': {},
            'configuration': {},
            'monitoring_metrics': [],
            'expected_benefits': [],
            'potential_risks': []
        }
        
        try:
            if recommended_approach == 'single_timeframe_base':
                guide['implementation_steps'] = [
                    "1. Use 1m as the base timeframe for training",
                    "2. Include all cross-timeframe features in the feature set",
                    "3. Apply advanced feature selection to reduce dimensionality",
                    "4. Train a single model with all features",
                    "5. Validate on out-of-sample data",
                    "6. Monitor performance and retrain as needed"
                ]
                
                guide['code_examples'] = {
                    'feature_preparation': '''
# Prepare features for single timeframe training
from src.feature_generation.utils.optimized_cross_timeframe_analysis_integration import (
    analyze_cross_timeframes_optimized
)

# Get cross-timeframe features
result = await analyze_cross_timeframes_optimized(
    data_dir="historical_data",
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframes=['1m', '5m', '15m', '30m']
)

# Use all cross-timeframe features
features = result.cross_timeframe_features
selected_features = result.selected_features['final']
''',
                    'model_training': '''
# Train single model with cross-timeframe features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare data
X = features[selected_features]
y = target_returns  # 1m returns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
'''
                }
                
                guide['configuration'] = {
                    'base_timeframe': '1m',
                    'feature_selection_method': 'mutual_info',
                    'model_type': 'RandomForestRegressor',
                    'validation_method': 'time_series_split',
                    'retraining_frequency': 'daily'
                }
                
                guide['monitoring_metrics'] = [
                    'Model accuracy and precision',
                    'Feature importance stability',
                    'Prediction latency',
                    'Memory usage',
                    'Cross-timeframe feature utilization'
                ]
                
                guide['expected_benefits'] = [
                    'Simpler architecture and deployment',
                    'Better interpretability',
                    'Reduced overfitting risk',
                    'Faster training and inference',
                    'Comprehensive multi-timeframe information capture'
                ]
                
                guide['potential_risks'] = [
                    'May miss regime-specific patterns',
                    'Feature engineering complexity',
                    'Single point of failure',
                    'Less specialized for specific timeframes'
                ]
            
            elif recommended_approach == 'multi_timeframe_parallel':
                guide['implementation_steps'] = [
                    "1. Train separate models for each timeframe",
                    "2. Use timeframe-specific feature sets",
                    "3. Implement ensemble prediction combination",
                    "4. Validate each model independently",
                    "5. Monitor performance across timeframes",
                    "6. Implement dynamic model selection"
                ]
                
                guide['code_examples'] = {
                    'multi_model_training': '''
# Train models for different timeframes
models = {}
timeframes = ['1m', '5m', '15m']

for tf in timeframes:
    # Get timeframe-specific features
    tf_features = get_timeframe_features(features, tf)
    tf_target = get_timeframe_target(targets, tf)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(tf_features, tf_target)
    models[tf] = model

# Ensemble prediction
def ensemble_predict(features):
    predictions = {}
    for tf, model in models.items():
        tf_features = get_timeframe_features(features, tf)
        predictions[tf] = model.predict(tf_features)
    
    # Combine predictions (e.g., weighted average)
    return combine_predictions(predictions)
'''
                }
                
                guide['configuration'] = {
                    'timeframes': ['1m', '5m', '15m'],
                    'model_type': 'RandomForestRegressor',
                    'ensemble_method': 'weighted_average',
                    'validation_method': 'time_series_split',
                    'retraining_frequency': 'daily'
                }
                
                guide['monitoring_metrics'] = [
                    'Individual model performance',
                    'Ensemble prediction accuracy',
                    'Model correlation',
                    'Prediction consistency across timeframes',
                    'Computational overhead'
                ]
                
                guide['expected_benefits'] = [
                    'Specialized models for each timeframe',
                    'Better uncertainty estimation',
                    'Robustness through diversity',
                    'Flexible prediction horizons'
                ]
                
                guide['potential_risks'] = [
                    'Increased complexity',
                    'Higher computational cost',
                    'Model coordination challenges',
                    'Potential overfitting with multiple models'
                ]
            
            elif recommended_approach == 'regime_aware_single':
                guide['implementation_steps'] = [
                    "1. Implement regime detection algorithm",
                    "2. Train separate models for each regime",
                    "3. Use cross-timeframe features in each regime model",
                    "4. Implement regime transition handling",
                    "5. Validate regime-specific performance",
                    "6. Monitor regime detection accuracy"
                ]
                
                guide['code_examples'] = {
                    'regime_detection': '''
# Detect market regimes
# HMMRegimeDetector no longer available from deleted hmm_clustering module
# from src.training.steps.market_analysis.hmm_clustering import HMMRegimeDetector

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# regime_detector = HMMRegimeDetector()
# regimes = regime_detector.detect_regimes(features)

# Train regime-specific models
regime_models = {}
for regime in np.unique(regimes):
    regime_mask = regimes == regime
    regime_features = features[regime_mask]
    regime_target = target_returns[regime_mask]
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(regime_features, regime_target)
    regime_models[regime] = model

# Regime-aware prediction
def regime_aware_predict(features):
    current_regime = regime_detector.predict_regime(features.iloc[-1:])
    model = regime_models[current_regime]
    return model.predict(features)
'''
                }
                
                guide['configuration'] = {
                    'regime_detection_method': 'hmm_clustering',
                    'regime_count': 3,
                    'model_type': 'RandomForestRegressor',
                    'regime_transition_threshold': 0.7,
                    'retraining_frequency': 'daily'
                }
                
                guide['monitoring_metrics'] = [
                    'Regime detection accuracy',
                    'Regime-specific model performance',
                    'Regime transition frequency',
                    'Model stability across regimes',
                    'Cross-timeframe feature utilization by regime'
                ]
                
                guide['expected_benefits'] = [
                    'Adaptive to market conditions',
                    'Specialized models for different regimes',
                    'Better performance in volatile markets',
                    'Comprehensive multi-timeframe information'
                ]
                
                guide['potential_risks'] = [
                    'Regime detection errors',
                    'Increased complexity',
                    'Model switching overhead',
                    'Requires sufficient data for each regime'
                ]
            
        except Exception as e:
            self.logger.error(f"❌ Implementation guide generation failed: {e}")
            guide['implementation_steps'].append(f"Error generating guide: {e}")
        
        return guide

# Convenience function
def analyze_training_approach(
    cross_timeframe_features: pd.DataFrame,
    target_returns: pd.Series,
    data_size: int,
    available_compute: str = "medium"
) -> Dict[str, Any]:
    """
    Analyze and recommend the best training approach for cross-timeframe features.
    
    Args:
        cross_timeframe_features: DataFrame with cross-timeframe features
        target_returns: Series with target returns
        data_size: Size of the dataset
        available_compute: Available compute resources ("low", "medium", "high")
    
    Returns:
        Dictionary with analysis results and recommendations
    """
    analyzer = MultiTimeframeTrainingAnalyzer()
    
    # Analyze feature utilization
    feature_analysis = analyzer.analyze_feature_utilization(
        cross_timeframe_features, target_returns
    )
    
    # Get recommendation
    recommendation = analyzer.recommend_training_approach(
        feature_analysis, data_size, available_compute
    )
    
    # Generate implementation guide
    implementation_guide = analyzer.generate_implementation_guide(
        recommendation['recommended_approach'], feature_analysis
    )
    
    return {
        'feature_analysis': feature_analysis,
        'recommendation': recommendation,
        'implementation_guide': implementation_guide
    }
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
