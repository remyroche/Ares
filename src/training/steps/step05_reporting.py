from ..standardized_parquet_handler import standardized_parquet_handler
"""
Step05 Focused Reporting Module

This module provides focused reporting capabilities for Step05 labeling,
integrating validation, financial calculations, and error handling modules.
"""

import json
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from src.utils.logger import system_logger
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from .step05_validation import Step05Validator, ValidationResult, LookaheadBiasResult
from .step05_financial import Step05FinancialCalculator, TradingPerformance, RiskMetrics
from .step05_error_handling import Step05ErrorHandler, ErrorSeverity, ErrorCategory

logger = system_logger.getChild('Step05Reporting')
financial_logger = get_financial_metrics_logger()

class Step05Reporter:
    """Focused reporter for Step05 labeling operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.validator = Step05Validator(config)
        self.financial_calculator = Step05FinancialCalculator(config)
        self.error_handler = Step05ErrorHandler(config)
        
    def generate_comprehensive_report(self, labeled_data: pd.DataFrame,
                                    labeling_results: Dict[str, Any],
                                    performance_data: Dict[str, Any],
                                    validation_results: Dict[str, Any],
                                    meta_labeling_analysis: Dict[str, Any],
                                    symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate comprehensive report using modular components.
        
        Args:
            labeled_data: DataFrame with labeled data
            labeling_results: Results from labeling process
            performance_data: Performance metrics
            validation_results: Validation results
            meta_labeling_analysis: Meta-labeling analysis
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Comprehensive report dictionary
        """
        try:
            self.logger.info("📊 Generating comprehensive Step05 report...")
            
            with financial_metrics_context("Step05_Reporting", symbol, exchange, timeframe):
                financial_logger.log_step_start("Step05_Reporting", symbol, exchange, timeframe)
                
                # Generate report sections using modular components
                report = {
                    'metadata': self._generate_metadata(symbol, exchange, timeframe),
                    'validation_results': self._generate_validation_section(labeled_data, validation_results),
                    'financial_analysis': self._generate_financial_section(labeled_data),
                    'performance_metrics': self._generate_performance_section(performance_data),
                    'label_quality': self._generate_label_quality_section(labeled_data),
                    'meta_labeling_analysis': self._generate_meta_labeling_section(meta_labeling_analysis),
                    'error_summary': self.error_handler.get_error_summary(),
                    'recommendations': self._generate_recommendations(labeled_data, labeling_results),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Log financial metrics
                self._log_financial_metrics(report, symbol, exchange, timeframe)
                
                financial_logger.log_step_end("Step05_Reporting", symbol, exchange, timeframe, success=True)
                
                self.logger.info("✅ Comprehensive report generated successfully")
                return report
                
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            financial_logger.log_step_end("Step05_Reporting", symbol, exchange, timeframe, success=False, error_message=str(e))
            
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'metadata': self._generate_metadata(symbol, exchange, timeframe)
            }
    
    def _generate_metadata(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'report_version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'modules_used': [
                'step05_validation',
                'step05_financial',
                'step05_error_handling',
                'step05_reporting'
            ]
        }
    
    def _generate_validation_section(self, labeled_data: pd.DataFrame, 
                                   validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation section using validation module."""
        try:
            # Perform lookahead bias validation
            barrier_params = {
                'profit_take_multiplier': 0.002,
                'stop_loss_multiplier': 0.001,
                'time_barrier_minutes': 30,
                'max_lookahead': 100
            }
            
            lookahead_bias_result = self.validator.validate_lookahead_bias(labeled_data, barrier_params)
            
            # Perform data integrity validation
            data_integrity_result = self.validator.validate_data_integrity(labeled_data)
            
            # Perform label quality validation
            label_quality_result = self.validator.validate_label_quality(labeled_data)
            
            return {
                'lookahead_bias': {
                    'bias_detected': lookahead_bias_result.bias_detected,
                    'bias_score': lookahead_bias_result.bias_score,
                    'temporal_violations': lookahead_bias_result.temporal_violations,
                    'future_data_leakage': lookahead_bias_result.future_data_leakage,
                    'recommendations': lookahead_bias_result.recommendations
                },
                'data_integrity': {
                    'passed': data_integrity_result.passed,
                    'score': data_integrity_result.score,
                    'warnings': data_integrity_result.warnings,
                    'errors': data_integrity_result.errors,
                    'recommendations': data_integrity_result.recommendations
                },
                'label_quality': {
                    'passed': label_quality_result.passed,
                    'score': label_quality_result.score,
                    'warnings': label_quality_result.warnings,
                    'errors': label_quality_result.errors,
                    'recommendations': label_quality_result.recommendations
                },
                'overall_validation': {
                    'passed': all([
                        not lookahead_bias_result.bias_detected,
                        data_integrity_result.passed,
                        label_quality_result.passed
                    ]),
                    'critical_issues': len(data_integrity_result.errors) + len(label_quality_result.errors)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Validation section generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_financial_section(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate financial analysis section using financial module."""
        try:
            # Calculate transaction costs
            transaction_costs = self.financial_calculator.calculate_transaction_costs(labeled_data)
            
            # Calculate trading performance
            trading_performance = self.financial_calculator.calculate_trading_performance(
                labeled_data, transaction_costs
            )
            
            # Calculate risk metrics
            risk_metrics = self.financial_calculator.calculate_risk_metrics(labeled_data)
            
            # Calculate position sizing
            position_sizes = self.financial_calculator.calculate_position_sizing(labeled_data)
            
            return {
                'trading_performance': asdict(trading_performance),
                'risk_metrics': asdict(risk_metrics),
                'transaction_costs': {
                    'total_costs': transaction_costs.sum(),
                    'avg_cost_per_trade': transaction_costs.mean(),
                    'cost_distribution': {
                        'min': transaction_costs.min(),
                        'max': transaction_costs.max(),
                        'median': transaction_costs.median(),
                        'std': transaction_costs.std()
                    }
                },
                'position_sizing': {
                    'avg_position_size': position_sizes.mean(),
                    'position_size_distribution': {
                        'min': position_sizes.min(),
                        'max': position_sizes.max(),
                        'median': position_sizes.median(),
                        'std': position_sizes.std()
                    }
                },
                'financial_summary': {
                    'net_return': trading_performance.net_return,
                    'sharpe_ratio': trading_performance.sharpe_ratio,
                    'max_drawdown': trading_performance.max_drawdown,
                    'win_rate': trading_performance.win_rate,
                    'cost_impact': trading_performance.cost_impact
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Financial section generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_performance_section(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics section."""
        try:
            return {
                'execution_metrics': {
                    'execution_time': performance_data.get('execution_time', 0),
                    'memory_usage': performance_data.get('memory_usage', 0),
                    'cpu_usage': performance_data.get('cpu_usage', 0),
                    'processing_efficiency': performance_data.get('processing_efficiency', 0)
                },
                'labeling_metrics': {
                    'label_creation_rate': performance_data.get('label_creation_rate', 0),
                    'meta_labeling_time': performance_data.get('meta_labeling_time', 0),
                    'validation_time': performance_data.get('validation_time', 0)
                },
                'optimization_metrics': {
                    'optimization_effectiveness': performance_data.get('optimization_effectiveness', 0),
                    'total_function_calls': performance_data.get('total_function_calls', 0),
                    'successful_operations': performance_data.get('successful_operations', 0),
                    'error_rate': performance_data.get('error_rate', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance section generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_label_quality_section(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate label quality analysis section."""
        try:
            if 'label' not in labeled_data.columns:
                return {'error': 'No label column found'}
            
            labels = labeled_data['label'].dropna()
            
            if len(labels) == 0:
                return {'error': 'No valid labels found'}
            
            # Calculate label distribution
            label_counts = labels.value_counts()
            total_labels = len(labels)
            
            # Calculate quality metrics
            label_distribution = {
                'buy': int(label_counts.get(1, 0)),
                'sell': int(label_counts.get(-1, 0)),
                'hold': int(label_counts.get(0, 0))
            }
            
            # Calculate balance
            if len(label_counts) > 1:
                max_count = label_counts.max()
                min_count = label_counts.min()
                balance_ratio = min_count / max_count if max_count > 0 else 0
            else:
                balance_ratio = 0
            
            # Calculate confidence if available
            confidence_score = 0.5  # Default
            if 'label_confidence' in labeled_data.columns:
                confidence_scores = labeled_data['label_confidence'].dropna()
                if len(confidence_scores) > 0:
                    confidence_score = confidence_scores.mean()
            
            return {
                'label_distribution': label_distribution,
                'total_labels': total_labels,
                'balance_ratio': balance_ratio,
                'confidence_score': confidence_score,
                'quality_assessment': {
                    'balanced': balance_ratio > 0.3,
                    'sufficient_samples': total_labels > 100,
                    'high_confidence': confidence_score > 0.7
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Label quality section generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_meta_labeling_section(self, meta_labeling_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate meta-labeling analysis section."""
        try:
            return {
                'meta_labels_created': meta_labeling_analysis.get('meta_labels_created', 0),
                'success_rate': meta_labeling_analysis.get('success_rate', 0),
                'avg_confidence': meta_labeling_analysis.get('avg_confidence', 0),
                'quality_score': meta_labeling_analysis.get('quality_score', 0),
                'agreement_rate': meta_labeling_analysis.get('agreement_rate', 0),
                'computation_time': meta_labeling_analysis.get('computation_time', 0),
                'optimization_gain': meta_labeling_analysis.get('optimization_gain', 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Meta-labeling section generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, labeled_data: pd.DataFrame, 
                                labeling_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            # Validation-based recommendations
            validation_section = self._generate_validation_section(labeled_data, {})
            if 'lookahead_bias' in validation_section:
                bias_result = validation_section['lookahead_bias']
                if bias_result.get('bias_detected', False):
                    recommendations.append("Address lookahead bias issues in labeling logic")
            
            # Financial-based recommendations
            financial_section = self._generate_financial_section(labeled_data)
            if 'financial_summary' in financial_section:
                financial_summary = financial_section['financial_summary']
                if financial_summary.get('cost_impact', 0) > 0.1:
                    recommendations.append("Reduce transaction costs by optimizing trade frequency")
                if financial_summary.get('sharpe_ratio', 0) < 1.0:
                    recommendations.append("Improve risk-adjusted returns through better signal quality")
            
            # Label quality recommendations
            label_quality = self._generate_label_quality_section(labeled_data)
            if 'quality_assessment' in label_quality:
                quality = label_quality['quality_assessment']
                if not quality.get('balanced', True):
                    recommendations.append("Address label imbalance through balanced sampling")
                if not quality.get('sufficient_samples', True):
                    recommendations.append("Increase sample size for more reliable labeling")
            
            # Error-based recommendations
            error_summary = self.error_handler.get_error_summary()
            if error_summary.get('critical_errors', 0) > 0:
                recommendations.append("Address critical errors before production deployment")
            if error_summary.get('resolution_rate', 1.0) < 0.8:
                recommendations.append("Improve error recovery mechanisms")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendations generation failed: {e}")
            return ["Review system configuration and error logs"]
    
    def _log_financial_metrics(self, report: Dict[str, Any], symbol: str, 
                             exchange: str, timeframe: str):
        """Log key financial metrics from the report."""
        try:
            if 'financial_analysis' in report:
                financial_analysis = report['financial_analysis']
                
                if 'financial_summary' in financial_analysis:
                    summary = financial_analysis['financial_summary']
                    
                    financial_logger.log_financial_metric(
                        metric_name="net_return",
                        metric_value=summary.get('net_return', 0.0),
                        metric_type="return",
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )
                    
                    financial_logger.log_financial_metric(
                        metric_name="sharpe_ratio",
                        metric_value=summary.get('sharpe_ratio', 0.0),
                        metric_type="risk_adjusted_return",
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )
                    
                    financial_logger.log_financial_metric(
                        metric_name="max_drawdown",
                        metric_value=summary.get('max_drawdown', 0.0),
                        metric_type="risk",
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )
            
            self.logger.info("💰 Financial metrics logged successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not log financial metrics: {e}")
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "reports/step05") -> Dict[str, str]:
        """Save report to various formats."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"step05_report_{timestamp}"
            
            saved_files = {}
            
            # Save JSON report
            json_path = output_path / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            saved_files['json'] = str(json_path)
            
            # Save Markdown report
            md_path = output_path / f"{base_filename}.md"
            markdown_content = self._generate_markdown_report(report)
            with open(md_path, 'w') as f:
                f.write(markdown_content)
            saved_files['markdown'] = str(md_path)
            
            self.logger.info(f"✅ Report saved to {output_path}")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ Report saving failed: {e}")
            return {'error': str(e)}
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate Markdown version of the report."""
        try:
            md_lines = [
                "# Step05 Labeling Report",
                f"**Generated:** {report.get('timestamp', 'Unknown')}",
                "",
                "## Executive Summary",
                ""
            ]
            
            # Add metadata
            if 'metadata' in report:
                metadata = report['metadata']
                md_lines.extend([
                    f"- **Symbol:** {metadata.get('symbol', 'N/A')}",
                    f"- **Exchange:** {metadata.get('exchange', 'N/A')}",
                    f"- **Timeframe:** {metadata.get('timeframe', 'N/A')}",
                    f"- **Report Version:** {metadata.get('report_version', 'N/A')}",
                    ""
                ])
            
            # Add validation results
            if 'validation_results' in report:
                validation = report['validation_results']
                md_lines.extend([
                    "## Validation Results",
                    ""
                ])
                
                if 'overall_validation' in validation:
                    overall = validation['overall_validation']
                    status = "✅ PASSED" if overall.get('passed', False) else "❌ FAILED"
                    md_lines.append(f"**Overall Status:** {status}")
                    md_lines.append(f"**Critical Issues:** {overall.get('critical_issues', 0)}")
                    md_lines.append("")
            
            # Add financial summary
            if 'financial_analysis' in report:
                financial = report['financial_analysis']
                if 'financial_summary' in financial:
                    summary = financial['financial_summary']
                    md_lines.extend([
                        "## Financial Summary",
                        "",
                        f"- **Net Return:** {summary.get('net_return', 0):.2%}",
                        f"- **Sharpe Ratio:** {summary.get('sharpe_ratio', 0):.2f}",
                        f"- **Max Drawdown:** {summary.get('max_drawdown', 0):.2%}",
                        f"- **Win Rate:** {summary.get('win_rate', 0):.2%}",
                        f"- **Cost Impact:** {summary.get('cost_impact', 0):.2%}",
                        ""
                    ])
            
            # Add recommendations
            if 'recommendations' in report:
                recommendations = report['recommendations']
                if recommendations:
                    md_lines.extend([
                        "## Recommendations",
                        ""
                    ])
                    for i, rec in enumerate(recommendations, 1):
                        md_lines.append(f"{i}. {rec}")
                    md_lines.append("")
            
            return "\n".join(md_lines)
            
        except Exception as e:
            self.logger.error(f"❌ Markdown generation failed: {e}")
            return f"# Step05 Report\n\nError generating report: {e}"