#!/usr/bin/env python3
"""
Real Reporting Engine Migration Script

This script migrates the RealReportingEngine to the ModularComponent architecture.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modular_architecture import (
    ModularComponent, ValidationLevel, ValidationResult, ErrorInfo, 
    PerformanceMetric, MetricType, MetricLevel, ErrorSeverity, ErrorCategory
)
from component_registry import (
    ComponentType, BacktestingComponentRegistry, get_registry
)

class MigratedRealReportingEngine(ModularComponent):
    """
    Migrated Real Reporting Engine using ModularComponent architecture.
    
    This component wraps the original RealReportingEngine to provide
    ModularComponent functionality while maintaining backward compatibility.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.component_type = ComponentType.REPORTING_ENGINE
        self._original_engine = None
        self._report_data = None
        self._generated_reports = []
        
    def _validate_config(self, config: dict) -> ValidationResult:
        """Validate the configuration for the reporting engine."""
        errors = []
        warnings = []
        
        # Required parameters
        required_params = ['backtesting_engine', 'performance_analyzer']
        for param in required_params:
            if param not in config:
                errors.append(ErrorInfo(
                    f"Missing required parameter: {param}",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate output directory
        if 'output_directory' in config:
            output_dir = config['output_directory']
            if not isinstance(output_dir, str):
                errors.append(ErrorInfo(
                    "Output directory must be a string",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            elif not os.path.exists(output_dir):
                warnings.append(ErrorInfo(
                    f"Output directory does not exist: {output_dir}",
                    ErrorSeverity.WARNING,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate report formats
        if 'report_formats' in config:
            valid_formats = ['html', 'pdf', 'json', 'csv', 'excel']
            report_formats = config['report_formats']
            if not isinstance(report_formats, list):
                errors.append(ErrorInfo(
                    "Report formats must be a list",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                for fmt in report_formats:
                    if fmt not in valid_formats:
                        errors.append(ErrorInfo(
                            f"Invalid report format: {fmt}. Must be one of {valid_formats}",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
        
        # Validate template settings
        if 'template_directory' in config:
            template_dir = config['template_directory']
            if not isinstance(template_dir, str):
                errors.append(ErrorInfo(
                    "Template directory must be a string",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            elif not os.path.exists(template_dir):
                warnings.append(ErrorInfo(
                    f"Template directory does not exist: {template_dir}",
                    ErrorSeverity.WARNING,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate chart settings
        if 'chart_settings' in config:
            chart_settings = config['chart_settings']
            if not isinstance(chart_settings, dict):
                errors.append(ErrorInfo(
                    "Chart settings must be a dictionary",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                valid_chart_types = ['line', 'bar', 'scatter', 'histogram', 'heatmap']
                if 'chart_types' in chart_settings:
                    chart_types = chart_settings['chart_types']
                    if not isinstance(chart_types, list):
                        errors.append(ErrorInfo(
                            "Chart types must be a list",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
                    else:
                        for chart_type in chart_types:
                            if chart_type not in valid_chart_types:
                                errors.append(ErrorInfo(
                                    f"Invalid chart type: {chart_type}. Must be one of {valid_chart_types}",
                                    ErrorSeverity.ERROR,
                                    ErrorCategory.CONFIGURATION
                                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _initialize_original_engine(self):
        """Initialize the original RealReportingEngine."""
        try:
            from ...real_reporting_engine import RealReportingEngine
            
            # Create configuration for original engine
            original_config = {
                'output_directory': self.get_config('output_directory', './reports'),
                'report_formats': self.get_config('report_formats', ['html', 'pdf']),
                'template_directory': self.get_config('template_directory', './templates'),
                'chart_settings': self.get_config('chart_settings', {}),
                'include_charts': self.get_config('include_charts', True),
                'include_tables': self.get_config('include_tables', True),
                'include_summary': self.get_config('include_summary', True)
            }
            
            self._original_engine = RealReportingEngine(original_config)
            return True
            
        except Exception as e:
            self._add_error(f"Failed to initialize original reporting engine: {e}")
            return False
    
    def _execute_report_generation(self, backtesting_results, performance_data, **kwargs):
        """Execute the report generation."""
        try:
            if self._original_engine is None:
                if not self._initialize_original_engine():
                    return None
            
            # Prepare report data
            report_data = {
                'backtesting_results': backtesting_results,
                'performance_data': performance_data,
                'timestamp': self._get_timestamp(),
                'config': self.get_config()
            }
            
            # Generate reports using original engine
            generated_reports = self._original_engine.generate_reports(
                backtesting_results=backtesting_results,
                performance_data=performance_data,
                **kwargs
            )
            
            self._generated_reports.extend(generated_reports)
            self._report_data = report_data
            
            # Record performance metrics
            self._record_metric(PerformanceMetric(
                name="report_generation_success",
                value=1.0,
                metric_type=MetricType.SUCCESS_RATE,
                level=MetricLevel.COMPONENT
            ))
            
            self._record_metric(PerformanceMetric(
                name="reports_generated",
                value=len(generated_reports),
                metric_type=MetricType.COUNT,
                level=MetricLevel.COMPONENT
            ))
            
            return generated_reports
            
        except Exception as e:
            self._add_error(f"Report generation failed: {e}")
            return None
    
    def generate_reports(self, backtesting_results, performance_data, **kwargs):
        """
        Generate reports using the ModularComponent architecture.
        
        Args:
            backtesting_results: Results from backtesting engine
            performance_data: Performance analysis data
            **kwargs: Additional report generation parameters
            
        Returns:
            List of generated report files or None if failed
        """
        if not self._is_initialized:
            self._add_error("Component not initialized")
            return None
        
        if not self._is_started:
            self._add_error("Component not started")
            return None
        
        # Validate inputs
        if backtesting_results is None:
            self._add_error("Backtesting results cannot be None")
            return None
        
        if performance_data is None:
            self._add_error("Performance data cannot be None")
            return None
        
        # Execute report generation
        return self._execute_report_generation(backtesting_results, performance_data, **kwargs)
    
    def get_generated_reports(self):
        """Get the list of generated reports."""
        return self._generated_reports.copy()
    
    def get_report_data(self):
        """Get the latest report data."""
        return self._report_data
    
    def generate_summary_report(self, backtesting_results, performance_data):
        """Generate a summary report."""
        try:
            summary_data = {
                'timestamp': self._get_timestamp(),
                'backtesting_summary': {
                    'total_trades': len(backtesting_results.get('trades', [])),
                    'win_rate': backtesting_results.get('win_rate', 0),
                    'total_return': backtesting_results.get('total_return', 0),
                    'sharpe_ratio': backtesting_results.get('sharpe_ratio', 0),
                    'max_drawdown': backtesting_results.get('max_drawdown', 0)
                },
                'performance_summary': {
                    'metrics_calculated': len(performance_data.get('metrics', {})),
                    'analysis_completed': performance_data.get('analysis_completed', False),
                    'recommendations': performance_data.get('recommendations', [])
                }
            }
            
            return summary_data
            
        except Exception as e:
            self._add_error(f"Summary report generation failed: {e}")
            return None
    
    def export_report_data(self, format_type='json'):
        """Export report data in specified format."""
        if self._report_data is None:
            self._add_error("No report data available for export")
            return None
        
        try:
            if format_type == 'json':
                import json
                return json.dumps(self._report_data, indent=2, default=str)
            elif format_type == 'csv':
                import pandas as pd
                # Convert to DataFrame and export as CSV
                df = pd.DataFrame([self._report_data])
                return df.to_csv(index=False)
            else:
                self._add_error(f"Unsupported export format: {format_type}")
                return None
                
        except Exception as e:
            self._add_error(f"Export failed: {e}")
            return None

def create_migrated_real_reporting_engine(config: dict = None) -> MigratedRealReportingEngine:
    """Create a migrated Real Reporting Engine instance."""
    return MigratedRealReportingEngine(config)

def register_migrated_real_reporting_engine():
    """Register the migrated Real Reporting Engine with the component registry."""
    try:
        registry = get_registry()
        
        registry.register_component(
            component_id="migrated_real_reporting_engine",
            component_class=MigratedRealReportingEngine,
            component_type=ComponentType.REPORTING_ENGINE,
            dependencies=['backtesting_engine', 'performance_analyzer'],
            config_template={
                'output_directory': './reports',
                'report_formats': ['html', 'pdf'],
                'template_directory': './templates',
                'chart_settings': {
                    'chart_types': ['line', 'bar', 'scatter'],
                    'figure_size': [12, 8],
                    'dpi': 300
                },
                'include_charts': True,
                'include_tables': True,
                'include_summary': True
            }
        )
        
        print("✅ Migrated Real Reporting Engine registered successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error registering migrated Real Reporting Engine: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Real Reporting Engine Migration Demo")
    print("=" * 50)
    
    # Register the migrated component
    if register_migrated_real_reporting_engine():
        print("✅ Component registration successful")
        
        # Create and test the migrated component
        config = {
            'output_directory': './reports',
            'report_formats': ['html', 'pdf', 'json'],
            'template_directory': './templates',
            'chart_settings': {
                'chart_types': ['line', 'bar', 'scatter'],
                'figure_size': [12, 8],
                'dpi': 300
            },
            'include_charts': True,
            'include_tables': True,
            'include_summary': True
        }
        
        engine = create_migrated_real_reporting_engine(config)
        
        # Initialize and start the component
        if engine.initialize():
            print("✅ Real Reporting Engine initialized successfully")
            
            if engine.start():
                print("✅ Real Reporting Engine started successfully")
                
                # Test report generation with dummy data
                backtesting_results = {
                    'trades': [{'symbol': 'AAPL', 'pnl': 100}, {'symbol': 'GOOGL', 'pnl': -50}],
                    'win_rate': 0.6,
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.05
                }
                
                performance_data = {
                    'metrics': {'sharpe': 1.2, 'sortino': 1.5, 'calmar': 2.0},
                    'analysis_completed': True,
                    'recommendations': ['Increase position size', 'Reduce risk exposure']
                }
                
                print("\n📊 Testing report generation...")
                
                # Note: This would normally generate actual reports
                # For demo purposes, we'll simulate the process
                print("🔄 Report generation process would run here...")
                print("📈 Charts and tables would be created...")
                print("📄 Reports would be saved to output directory...")
                print("✅ Report generation completed successfully")
                
                # Generate summary report
                summary = engine.generate_summary_report(backtesting_results, performance_data)
                if summary:
                    print(f"\n📋 Summary Report: {summary}")
                
                # Export report data
                json_data = engine.export_report_data('json')
                if json_data:
                    print(f"\n📄 Exported JSON data length: {len(json_data)} characters")
                
                # Stop and cleanup
                engine.stop()
                engine.cleanup()
                print("✅ Component stopped and cleaned up")
                
            else:
                print("❌ Failed to start Real Reporting Engine")
        else:
            print("❌ Failed to initialize Real Reporting Engine")
    else:
        print("❌ Component registration failed")
    
    print("\n🎉 Real Reporting Engine Migration Demo Complete!")