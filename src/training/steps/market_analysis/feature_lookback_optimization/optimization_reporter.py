"""
Optimization Reporter Module.

This module provides comprehensive reporting capabilities for feature lookback optimization,
including detailed metrics, visualizations, and actionable insights.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import asdict

from src.utils.logger import system_logger

logger = system_logger.getChild('OptimizationReporter')

class OptimizationReporter:
    """
    Comprehensive reporter for feature lookback optimization results.
    
    Provides detailed reporting, visualizations, and actionable insights
    for optimization results and performance metrics.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the optimization reporter."""
        self.logger = logger.getChild('OptimizationReporter')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different report types
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.insights_dir = self.output_dir / "insights"
        
        for dir_path in [self.metrics_dir, self.visualizations_dir, self.insights_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_comprehensive_report(
        self,
        optimization_result: Dict[str, Any],
        metrics: Any,
        validation_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        
        Args:
            optimization_result: Complete optimization results
            metrics: Optimization metrics object
            validation_results: Data and pipeline validation results
            performance_metrics: Performance monitoring metrics
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Comprehensive report dictionary
        """
        self.logger.info("📊 Generating comprehensive optimization report...")
        
        try:
            # Generate timestamp for report
            report_timestamp = datetime.now().isoformat()
            report_id = f"{symbol}_{exchange}_{timeframe}_{report_timestamp[:10]}"
            
            # Create comprehensive report structure
            report = {
                'report_metadata': {
                    'report_id': report_id,
                    'generation_timestamp': report_timestamp,
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'reporter_version': '2.0.0'
                },
                'executive_summary': self._generate_executive_summary(optimization_result, metrics),
                'optimization_results': self._format_optimization_results(optimization_result),
                'performance_analysis': self._analyze_performance(metrics, performance_metrics),
                'data_quality_assessment': self._assess_data_quality(validation_results),
                'feature_analysis': self._analyze_features(optimization_result),
                'recommendations': self._generate_recommendations(optimization_result, metrics, validation_results),
                'risk_assessment': self._assess_risks(optimization_result, metrics),
                'next_steps': self._suggest_next_steps(optimization_result, metrics),
                'technical_details': self._generate_technical_details(optimization_result, metrics, performance_metrics)
            }
            
            # Save report to file
            self._save_report(report, report_id)
            
            # Generate visualizations
            self._generate_visualizations(optimization_result, metrics, report_id)
            
            # Generate insights
            self._generate_insights(report, report_id)
            
            self.logger.info(f"✅ Comprehensive report generated: {report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return {'error': str(e), 'report_id': 'failed'}
    
    def _generate_executive_summary(self, optimization_result: Dict[str, Any], metrics: Any) -> Dict[str, Any]:
        """Generate executive summary of optimization results."""
        try:
            optimization_results = optimization_result.get('optimization_results', {})
            optimized_features = optimization_result.get('optimized_features', {})
            
            summary = {
                'status': 'SUCCESS' if optimization_results.get('best_score', 0) > 0 else 'FAILED',
                'key_metrics': {
                    'best_lookback_period': optimization_results.get('best_lookback_period', 0),
                    'best_score': optimization_results.get('best_score', 0.0),
                    'total_features_optimized': len(optimized_features),
                    'optimization_method': optimization_results.get('optimization_method', 'unknown')
                },
                'performance_indicators': {
                    'validation_score': getattr(metrics, 'validation_score', 0.0),
                    'stability_score': getattr(metrics, 'stability_score', 0.0),
                    'regime_coverage': getattr(metrics, 'regime_coverage', 0.0),
                    'error_rate': getattr(metrics, 'error_rate', 0.0)
                },
                'optimization_time': getattr(metrics, 'optimization_time', 0.0),
                'convergence_iterations': getattr(metrics, 'convergence_iterations', 0)
            }
            
            # Add performance rating
            overall_score = (
                summary['key_metrics']['best_score'] * 0.4 +
                summary['performance_indicators']['validation_score'] * 0.3 +
                summary['performance_indicators']['stability_score'] * 0.3
            )
            
            if overall_score >= 0.8:
                summary['performance_rating'] = 'EXCELLENT'
            elif overall_score >= 0.6:
                summary['performance_rating'] = 'GOOD'
            elif overall_score >= 0.4:
                summary['performance_rating'] = 'FAIR'
            else:
                summary['performance_rating'] = 'POOR'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            return {'error': str(e)}
    
    def _format_optimization_results(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format optimization results for reporting."""
        try:
            optimization_results = optimization_result.get('optimization_results', {})
            optimized_features = optimization_result.get('optimized_features', {})
            optimization_metrics = optimization_result.get('optimization_metrics', {})
            
            # Format feature results
            formatted_features = {}
            for feature_name, feature_data in optimized_features.items():
                formatted_features[feature_name] = {
                    'optimal_lookback': feature_data.get('lookback', 0),
                    'performance_score': feature_data.get('score', 0.0),
                    'optimization_method': feature_data.get('method', 'unknown'),
                    'confidence_level': self._calculate_confidence_level(feature_data)
                }
            
            # Sort features by performance score
            sorted_features = dict(sorted(
                formatted_features.items(),
                key=lambda x: x[1]['performance_score'],
                reverse=True
            ))
            
            return {
                'overall_results': {
                    'best_lookback_period': optimization_results.get('best_lookback_period', 0),
                    'best_score': optimization_results.get('best_score', 0.0),
                    'optimization_method': optimization_results.get('optimization_method', 'unknown'),
                    'convergence_achieved': optimization_metrics.get('convergence_iterations', 0) > 0
                },
                'feature_results': sorted_features,
                'top_performing_features': list(sorted_features.keys())[:5],
                'optimization_metadata': optimization_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Optimization results formatting failed: {e}")
            return {'error': str(e)}
    
    def _analyze_performance(self, metrics: Any, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and resource usage."""
        try:
            analysis = {
                'resource_usage': {
                    'memory_usage_mb': getattr(metrics, 'memory_usage_mb', 0.0),
                    'cpu_usage_percent': getattr(metrics, 'cpu_usage_percent', 0.0),
                    'optimization_time': getattr(metrics, 'optimization_time', 0.0)
                },
                'efficiency_metrics': {
                    'convergence_iterations': getattr(metrics, 'convergence_iterations', 0),
                    'features_per_second': len(performance_metrics.get('execution_times', {})) / max(1, getattr(metrics, 'optimization_time', 1.0)),
                    'memory_efficiency': getattr(metrics, 'memory_usage_mb', 0.0) / max(1, getattr(metrics, 'total_features_optimized', 1))
                },
                'performance_trends': self._analyze_performance_trends(performance_metrics),
                'bottlenecks': self._identify_bottlenecks(performance_metrics)
            }
            
            # Add performance assessment
            memory_usage = analysis['resource_usage']['memory_usage_mb']
            cpu_usage = analysis['resource_usage']['cpu_usage_percent']
            optimization_time = analysis['resource_usage']['optimization_time']
            
            if memory_usage > 1000 or cpu_usage > 80 or optimization_time > 300:
                analysis['performance_assessment'] = 'NEEDS_OPTIMIZATION'
            elif memory_usage > 500 or cpu_usage > 60 or optimization_time > 120:
                analysis['performance_assessment'] = 'MODERATE'
            else:
                analysis['performance_assessment'] = 'EFFICIENT'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}
    
    def _assess_data_quality(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality based on validation results."""
        try:
            data_validation = validation_results.get('data_validation', {})
            pipeline_validation = validation_results.get('pipeline_validation', {})
            
            assessment = {
                'overall_quality_score': data_validation.get('data_quality_score', 0.0),
                'data_completeness': data_validation.get('data_completeness', 0.0),
                'validation_status': {
                    'data_valid': data_validation.get('is_valid', False),
                    'pipeline_valid': pipeline_validation.get('is_valid', False)
                },
                'quality_issues': {
                    'data_warnings': data_validation.get('warnings', []),
                    'pipeline_warnings': pipeline_validation.get('warnings', [])
                },
                'quality_rating': self._calculate_quality_rating(data_validation)
            }
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {e}")
            return {'error': str(e)}
    
    def _analyze_features(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimized features for patterns and insights."""
        try:
            optimized_features = optimization_result.get('optimized_features', {})
            
            if not optimized_features:
                return {'error': 'No features to analyze'}
            
            # Extract lookback periods and scores
            lookback_periods = [feature.get('lookback', 0) for feature in optimized_features.values()]
            scores = [feature.get('score', 0.0) for feature in optimized_features.values()]
            
            analysis = {
                'feature_statistics': {
                    'total_features': len(optimized_features),
                    'average_lookback': np.mean(lookback_periods) if lookback_periods else 0,
                    'median_lookback': np.median(lookback_periods) if lookback_periods else 0,
                    'lookback_std': np.std(lookback_periods) if lookback_periods else 0,
                    'average_score': np.mean(scores) if scores else 0,
                    'score_std': np.std(scores) if scores else 0
                },
                'feature_categories': self._categorize_features(optimized_features),
                'performance_distribution': self._analyze_performance_distribution(scores),
                'lookback_patterns': self._analyze_lookback_patterns(lookback_periods)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Feature analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, optimization_result: Dict[str, Any], metrics: Any, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on optimization results."""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if getattr(metrics, 'validation_score', 0) < 0.7:
                recommendations.append({
                    'category': 'DATA_QUALITY',
                    'priority': 'HIGH',
                    'title': 'Improve Data Quality',
                    'description': 'Low validation score indicates data quality issues',
                    'action': 'Review data preprocessing and cleaning procedures',
                    'expected_impact': 'HIGH'
                })
            
            if getattr(metrics, 'stability_score', 0) < 0.6:
                recommendations.append({
                    'category': 'OPTIMIZATION',
                    'priority': 'MEDIUM',
                    'title': 'Improve Feature Stability',
                    'description': 'High variability in lookback periods suggests instability',
                    'action': 'Consider regularization or ensemble methods',
                    'expected_impact': 'MEDIUM'
                })
            
            if getattr(metrics, 'regime_coverage', 0) < 0.8:
                recommendations.append({
                    'category': 'REGIME_AWARENESS',
                    'priority': 'MEDIUM',
                    'title': 'Enhance Regime Coverage',
                    'description': 'Low regime coverage may limit optimization effectiveness',
                    'action': 'Implement regime-aware optimization strategies',
                    'expected_impact': 'MEDIUM'
                })
            
            if getattr(metrics, 'memory_usage_mb', 0) > 1000:
                recommendations.append({
                    'category': 'PERFORMANCE',
                    'priority': 'LOW',
                    'title': 'Optimize Memory Usage',
                    'description': 'High memory usage may impact scalability',
                    'action': 'Implement data chunking or memory optimization',
                    'expected_impact': 'LOW'
                })
            
            # Add positive recommendations for good performance
            if getattr(metrics, 'validation_score', 0) > 0.8 and getattr(metrics, 'stability_score', 0) > 0.7:
                recommendations.append({
                    'category': 'OPTIMIZATION',
                    'priority': 'LOW',
                    'title': 'Consider Advanced Techniques',
                    'description': 'Good performance metrics suggest readiness for advanced optimization',
                    'action': 'Explore ensemble methods or advanced feature selection',
                    'expected_impact': 'MEDIUM'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendations generation failed: {e}")
            return [{'error': str(e)}]
    
    def _assess_risks(self, optimization_result: Dict[str, Any], metrics: Any) -> Dict[str, Any]:
        """Assess risks associated with optimization results."""
        try:
            risks = {
                'overfitting_risk': self._assess_overfitting_risk(optimization_result, metrics),
                'data_quality_risk': self._assess_data_quality_risk(metrics),
                'performance_risk': self._assess_performance_risk(metrics),
                'stability_risk': self._assess_stability_risk(metrics),
                'overall_risk_level': 'LOW'
            }
            
            # Calculate overall risk level
            risk_scores = [
                risks['overfitting_risk']['score'],
                risks['data_quality_risk']['score'],
                risks['performance_risk']['score'],
                risks['stability_risk']['score']
            ]
            
            avg_risk_score = np.mean(risk_scores)
            
            if avg_risk_score >= 0.8:
                risks['overall_risk_level'] = 'HIGH'
            elif avg_risk_score >= 0.5:
                risks['overall_risk_level'] = 'MEDIUM'
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {'error': str(e)}
    
    def _suggest_next_steps(self, optimization_result: Dict[str, Any], metrics: Any) -> List[Dict[str, Any]]:
        """Suggest next steps based on optimization results."""
        next_steps = []
        
        try:
            # Always suggest validation
            next_steps.append({
                'step': 'VALIDATION',
                'description': 'Validate optimization results on out-of-sample data',
                'priority': 'HIGH',
                'estimated_effort': 'MEDIUM'
            })
            
            # Suggest based on performance
            if getattr(metrics, 'validation_score', 0) > 0.8:
                next_steps.append({
                    'step': 'PRODUCTION_DEPLOYMENT',
                    'description': 'Deploy optimized features to production environment',
                    'priority': 'HIGH',
                    'estimated_effort': 'HIGH'
                })
            
            # Suggest based on regime coverage
            if getattr(metrics, 'regime_coverage', 0) < 0.8:
                next_steps.append({
                    'step': 'REGIME_ENHANCEMENT',
                    'description': 'Enhance regime detection and regime-aware optimization',
                    'priority': 'MEDIUM',
                    'estimated_effort': 'HIGH'
                })
            
            # Suggest monitoring
            next_steps.append({
                'step': 'MONITORING_SETUP',
                'description': 'Set up monitoring for optimized features in production',
                'priority': 'MEDIUM',
                'estimated_effort': 'MEDIUM'
            })
            
            return next_steps
            
        except Exception as e:
            self.logger.error(f"Next steps suggestion failed: {e}")
            return [{'error': str(e)}]
    
    def _generate_technical_details(self, optimization_result: Dict[str, Any], metrics: Any, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical details for advanced users."""
        try:
            return {
                'optimization_parameters': optimization_result.get('optimization_metrics', {}),
                'performance_metrics': performance_metrics,
                'convergence_details': {
                    'iterations': getattr(metrics, 'convergence_iterations', 0),
                    'final_score': getattr(metrics, 'best_score', 0.0),
                    'optimization_time': getattr(metrics, 'optimization_time', 0.0)
                },
                'resource_utilization': {
                    'memory_usage_mb': getattr(metrics, 'memory_usage_mb', 0.0),
                    'cpu_usage_percent': getattr(metrics, 'cpu_usage_percent', 0.0),
                    'error_rate': getattr(metrics, 'error_rate', 0.0)
                },
                'algorithm_details': optimization_result.get('optimization_results', {}),
                'feature_details': optimization_result.get('optimized_features', {})
            }
            
        except Exception as e:
            self.logger.error(f"Technical details generation failed: {e}")
            return {'error': str(e)}
    
    def _save_report(self, report: Dict[str, Any], report_id: str) -> None:
        """Save report to file."""
        try:
            report_file = self.metrics_dir / f"{report_id}_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Report saving failed: {e}")
    
    def _generate_visualizations(self, optimization_result: Dict[str, Any], metrics: Any, report_id: str) -> None:
        """Generate visualizations for the report."""
        try:
            # This would generate charts and plots
            # For now, we'll create a placeholder
            viz_data = {
                'report_id': report_id,
                'visualizations': [
                    'feature_performance_chart',
                    'lookback_distribution_histogram',
                    'optimization_convergence_plot',
                    'performance_metrics_dashboard'
                ],
                'generation_timestamp': datetime.now().isoformat()
            }
            
            viz_file = self.visualizations_dir / f"{report_id}_visualizations.json"
            with open(viz_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
            
            self.logger.info(f"Visualizations data saved to: {viz_file}")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
    
    def _generate_insights(self, report: Dict[str, Any], report_id: str) -> None:
        """Generate actionable insights from the report."""
        try:
            insights = {
                'report_id': report_id,
                'key_insights': [
                    f"Optimization achieved {report['executive_summary']['key_metrics']['best_score']:.3f} score",
                    f"Performance rating: {report['executive_summary']['performance_rating']}",
                    f"Total features optimized: {report['executive_summary']['key_metrics']['total_features_optimized']}"
                ],
                'actionable_items': [
                    item['action'] for item in report['recommendations']
                ],
                'risk_summary': report['risk_assessment']['overall_risk_level'],
                'generation_timestamp': datetime.now().isoformat()
            }
            
            insights_file = self.insights_dir / f"{report_id}_insights.json"
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2)
            
            self.logger.info(f"Insights saved to: {insights_file}")
            
        except Exception as e:
            self.logger.error(f"Insights generation failed: {e}")
    
    # Helper methods
    def _calculate_confidence_level(self, feature_data: Dict[str, Any]) -> str:
        """Calculate confidence level for a feature."""
        score = feature_data.get('score', 0.0)
        if score >= 0.8:
            return 'HIGH'
        elif score >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_quality_rating(self, data_validation: Dict[str, Any]) -> str:
        """Calculate overall data quality rating."""
        quality_score = data_validation.get('data_quality_score', 0.0)
        if quality_score >= 0.9:
            return 'EXCELLENT'
        elif quality_score >= 0.7:
            return 'GOOD'
        elif quality_score >= 0.5:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _analyze_performance_trends(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Placeholder for performance trend analysis
        return {
            'memory_trend': 'STABLE',
            'cpu_trend': 'STABLE',
            'execution_time_trend': 'STABLE'
        }
    
    def _identify_bottlenecks(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for high memory usage
        if performance_metrics.get('memory_usage', [0])[-1] > 1000:
            bottlenecks.append('HIGH_MEMORY_USAGE')
        
        # Check for high CPU usage
        if performance_metrics.get('cpu_usage', [0])[-1] > 80:
            bottlenecks.append('HIGH_CPU_USAGE')
        
        return bottlenecks
    
    def _categorize_features(self, optimized_features: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'technical_indicators': [],
            'price_features': [],
            'volume_features': [],
            'other': []
        }
        
        for feature_name in optimized_features.keys():
            if any(indicator in feature_name.lower() for indicator in ['rsi', 'sma', 'ema', 'macd']):
                categories['technical_indicators'].append(feature_name)
            elif any(price in feature_name.lower() for price in ['open', 'high', 'low', 'close']):
                categories['price_features'].append(feature_name)
            elif 'volume' in feature_name.lower():
                categories['volume_features'].append(feature_name)
            else:
                categories['other'].append(feature_name)
        
        return categories
    
    def _analyze_performance_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze performance score distribution."""
        if not scores:
            return {'error': 'No scores to analyze'}
        
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'quartiles': {
                'q1': np.percentile(scores, 25),
                'q3': np.percentile(scores, 75)
            }
        }
    
    def _analyze_lookback_patterns(self, lookback_periods: List[int]) -> Dict[str, Any]:
        """Analyze patterns in lookback periods."""
        if not lookback_periods:
            return {'error': 'No lookback periods to analyze'}
        
        return {
            'range': [min(lookback_periods), max(lookback_periods)],
            'most_common': max(set(lookback_periods), key=lookback_periods.count),
            'distribution': {
                'short_term': sum(1 for p in lookback_periods if p <= 10),
                'medium_term': sum(1 for p in lookback_periods if 10 < p <= 30),
                'long_term': sum(1 for p in lookback_periods if p > 30)
            }
        }
    
    def _assess_overfitting_risk(self, optimization_result: Dict[str, Any], metrics: Any) -> Dict[str, Any]:
        """Assess overfitting risk."""
        # Simple overfitting assessment based on validation score
        validation_score = getattr(metrics, 'validation_score', 0.0)
        
        if validation_score < 0.5:
            risk_level = 'HIGH'
            score = 0.8
        elif validation_score < 0.7:
            risk_level = 'MEDIUM'
            score = 0.5
        else:
            risk_level = 'LOW'
            score = 0.2
        
        return {
            'risk_level': risk_level,
            'score': score,
            'description': f'Overfitting risk based on validation score of {validation_score:.3f}'
        }
    
    def _assess_data_quality_risk(self, metrics: Any) -> Dict[str, Any]:
        """Assess data quality risk."""
        # This would be based on data validation results
        return {
            'risk_level': 'LOW',
            'score': 0.2,
            'description': 'Data quality risk assessment'
        }
    
    def _assess_performance_risk(self, metrics: Any) -> Dict[str, Any]:
        """Assess performance risk."""
        memory_usage = getattr(metrics, 'memory_usage_mb', 0.0)
        
        if memory_usage > 2000:
            risk_level = 'HIGH'
            score = 0.8
        elif memory_usage > 1000:
            risk_level = 'MEDIUM'
            score = 0.5
        else:
            risk_level = 'LOW'
            score = 0.2
        
        return {
            'risk_level': risk_level,
            'score': score,
            'description': f'Performance risk based on memory usage of {memory_usage:.1f}MB'
        }
    
    def _assess_stability_risk(self, metrics: Any) -> Dict[str, Any]:
        """Assess stability risk."""
        stability_score = getattr(metrics, 'stability_score', 0.0)
        
        if stability_score < 0.4:
            risk_level = 'HIGH'
            score = 0.8
        elif stability_score < 0.6:
            risk_level = 'MEDIUM'
            score = 0.5
        else:
            risk_level = 'LOW'
            score = 0.2
        
        return {
            'risk_level': risk_level,
            'score': score,
            'description': f'Stability risk based on stability score of {stability_score:.3f}'
        }