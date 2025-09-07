#!/usr/bin/env python3
"""
S/R Parameter Interdependence Analysis Tool

This script analyzes parameter relationships and provides recommendations
for managing overfitting and parameter interdependence risks.
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SRParameterAnalyzer:
    """Analyze S/R parameter relationships and provide optimization recommendations."""

    def __init__(self, config_path: str = "src/config/sr_optimization_config.yaml"):
        """Initialize the parameter analyzer."""
        self.config_path = Path(config_path)
        self.sensitivity_config_path = Path("src/config/sr_parameter_sensitivity_config.yaml")
        self.load_configurations()

    def load_configurations(self):
        """Load configuration files."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            with open(self.sensitivity_config_path, 'r') as f:
                self.sensitivity_config = yaml.safe_load(f)

            print("✅ Configurations loaded successfully")

        except FileNotFoundError as e:
            print(f"❌ Configuration file not found: {e}")
            self.config = {}
            self.sensitivity_config = {}

    def analyze_parameter_interdependence(self) -> Dict[str, any]:
        """Analyze parameter relationships and dependencies."""
        print("\n🔍 Analyzing Parameter Interdependence...")

        # Extract parameter ranges
        param_ranges = self.config.get('parameter_ranges', {})

        # Define known parameter relationships
        relationships = {
            'touch_proximity_dbscan': {
                'params': ['touch_proximity_threshold', 'dbscan_eps'],
                'expected_correlation': 0.8,
                'relationship_type': 'direct'
            },
            'volume_sensitivity_timeframe': {
                'params': ['volume_spike_threshold', 'timeframe'],
                'expected_correlation': -0.6,
                'relationship_type': 'inverse'
            },
            'fractal_pivot_complexity': {
                'params': ['fractal_period', 'pivot_period'],
                'expected_correlation': 0.5,
                'relationship_type': 'direct'
            }
        }

        analysis_results = {}

        for relationship_name, relationship_info in relationships.items():
            print(f"  📊 Analyzing {relationship_name}...")

            # Check if parameters exist in config
            params_available = all(
                param in str(param_ranges) or param in str(self.config)
                for param in relationship_info['params']
            )

            if params_available:
                analysis_results[relationship_name] = {
                    'status': 'available_for_analysis',
                    'expected_correlation': relationship_info['expected_correlation'],
                    'relationship_type': relationship_info['relationship_type'],
                    'recommendations': self._generate_relationship_recommendations(relationship_info)
                }
            else:
                analysis_results[relationship_name] = {
                    'status': 'parameters_not_found',
                    'expected_correlation': relationship_info['expected_correlation'],
                    'relationship_type': relationship_info['relationship_type']
                }

        return analysis_results

    def _generate_relationship_recommendations(self, relationship_info: Dict) -> List[str]:
        """Generate specific recommendations for parameter relationships."""
        recommendations = []

        if relationship_info['relationship_type'] == 'direct':
            recommendations.append(
                f"Parameters {relationship_info['params']} have direct relationship. "
                f"When optimizing, adjust them together to maintain correlation ~{relationship_info['expected_correlation']}"
            )

        elif relationship_info['relationship_type'] == 'inverse':
            recommendations.append(
                f"Parameters {relationship_info['params']} have inverse relationship. "
                f"Increase one when decreasing the other to maintain balance."
            )

        # Add general recommendations
        recommendations.append(
            "Consider joint optimization of these parameters rather than independent optimization"
        )

        recommendations.append(
            "Monitor parameter correlation during backtesting to detect relationship breakdown"
        )

        return recommendations

    def analyze_overfitting_risks(self) -> Dict[str, any]:
        """Analyze potential overfitting risks in current configuration."""
        print("\n🎯 Analyzing Overfitting Risks...")

        risks = {}

        # Count total parameters
        total_params = self._count_total_parameters()
        risks['parameter_count'] = {
            'total_parameters': total_params,
            'risk_level': 'high' if total_params > 20 else 'medium' if total_params > 10 else 'low',
            'recommendation': self._get_parameter_count_recommendation(total_params)
        }

        # Analyze parameter ranges
        param_ranges_analysis = self._analyze_parameter_ranges()
        risks['parameter_ranges'] = param_ranges_analysis

        # Check for regularization
        regularization_check = self._check_regularization_settings()
        risks['regularization'] = regularization_check

        # Cross-validation assessment
        cv_assessment = self._assess_cross_validation_setup()
        risks['cross_validation'] = cv_assessment

        return risks

    def _count_total_parameters(self) -> int:
        """Count total number of parameters in the configuration."""
        param_ranges = self.config.get('parameter_ranges', {})

        total_params = 0
        for section_name, section_params in param_ranges.items():
            if isinstance(section_params, dict):
                for param_name, param_value in section_params.items():
                    if isinstance(param_value, list) and len(param_value) == 2:
                        total_params += 1
                    elif isinstance(param_value, dict):
                        # Count nested parameters
                        for nested_param in param_value.values():
                            if isinstance(nested_param, list) and len(nested_param) == 2:
                                total_params += 1

        return total_params

    def _get_parameter_count_recommendation(self, count: int) -> str:
        """Get recommendation based on parameter count."""
        if count > 20:
            return ("High risk of overfitting. Consider reducing parameters or using stronger regularization. "
                   "Implement feature selection to identify most important parameters.")
        elif count > 10:
            return ("Medium risk. Consider parameter grouping and hierarchical optimization. "
                   "Use cross-validation with multiple folds.")
        else:
            return "Low risk. Current parameter count is manageable."

    def _analyze_parameter_ranges(self) -> Dict[str, any]:
        """Analyze parameter ranges for optimization potential."""
        param_ranges = self.config.get('parameter_ranges', {})

        range_analysis = {
            'total_ranges_analyzed': 0,
            'wide_ranges': [],
            'narrow_ranges': [],
            'recommendations': []
        }

        for section_name, section_params in param_ranges.items():
            if isinstance(section_params, dict):
                for param_name, param_value in section_params.items():
                    if isinstance(param_value, list) and len(param_value) == 2:
                        range_analysis['total_ranges_analyzed'] += 1

                        min_val, max_val = param_value
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            range_width = abs(max_val - min_val)
                            avg_value = (min_val + max_val) / 2

                            # Calculate relative range width
                            if avg_value != 0:
                                relative_width = range_width / abs(avg_value)
                            else:
                                relative_width = range_width

                            if relative_width > 1.0:  # Wide range
                                range_analysis['wide_ranges'].append(f"{param_name}: {relative_width:.2f}x")
                            elif relative_width < 0.1:  # Narrow range
                                range_analysis['narrow_ranges'].append(f"{param_name}: {relative_width:.2f}x")

        # Generate recommendations
        if range_analysis['wide_ranges']:
            range_analysis['recommendations'].append(
                f"Consider narrowing ranges for: {', '.join(range_analysis['wide_ranges'][:3])}"
            )

        if range_analysis['narrow_ranges']:
            range_analysis['recommendations'].append(
                f"Consider expanding ranges for better exploration: {', '.join(range_analysis['narrow_ranges'][:3])}"
            )

        return range_analysis

    def _check_regularization_settings(self) -> Dict[str, any]:
        """Check if regularization is properly configured."""
        regularization = self.config.get('parameter_ranges', {}).get('regularization', {})

        if not regularization:
            return {
                'status': 'missing',
                'recommendation': 'Add regularization parameters to prevent overfitting'
            }

        required_regularization = ['l1_penalty', 'l2_penalty', 'parameter_correlation_penalty']
        present_regularization = [k for k in required_regularization if k in regularization]

        return {
            'status': 'partial' if len(present_regularization) < len(required_regularization) else 'complete',
            'present_parameters': present_regularization,
            'missing_parameters': [k for k in required_regularization if k not in regularization],
            'recommendation': 'Regularization configuration is good' if len(present_regularization) == len(required_regularization) else f'Add missing regularization: {", ".join([k for k in required_regularization if k not in regularization])}'
        }

    def _assess_cross_validation_setup(self) -> Dict[str, any]:
        """Assess cross-validation configuration."""
        validation = self.config.get('validation', {})

        assessment = {
            'walk_forward_enabled': validation.get('enable_walk_forward', False),
            'cv_folds': self.config.get('cv_folds', 5),
            'out_of_sample_ratio': validation.get('out_of_sample_ratio', 0.2),
            'recommendations': []
        }

        if not assessment['walk_forward_enabled']:
            assessment['recommendations'].append("Enable walk-forward validation for better overfitting detection")

        if assessment['cv_folds'] < 5:
            assessment['recommendations'].append("Consider increasing CV folds to at least 5 for more robust validation")

        if assessment['out_of_sample_ratio'] < 0.3:
            assessment['recommendations'].append("Consider increasing out-of-sample ratio to at least 30%")

        if not assessment['recommendations']:
            assessment['recommendations'].append("Cross-validation setup looks good")

        return assessment

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        print("\n📊 Generating Comprehensive Analysis Report...")

        interdependence = self.analyze_parameter_interdependence()
        overfitting_risks = self.analyze_overfitting_risks()

        report = []
        report.append("=" * 80)
        report.append("S/R PARAMETER ANALYSIS REPORT")
        report.append("=" * 80)

        # Parameter Interdependence Section
        report.append("\n🔗 PARAMETER INTERDEPENDENCE ANALYSIS")
        report.append("-" * 50)

        for relationship_name, analysis in interdependence.items():
            report.append(f"\n📈 {relationship_name.upper()}")
            report.append(f"Status: {analysis['status']}")
            report.append(f"Expected Correlation: {analysis['expected_correlation']}")
            report.append(f"Relationship Type: {analysis['relationship_type']}")

            if 'recommendations' in analysis:
                report.append("Recommendations:")
                for rec in analysis['recommendations']:
                    report.append(f"  • {rec}")

        # Overfitting Risks Section
        report.append("\n🎯 OVERFITTING RISK ANALYSIS")
        report.append("-" * 50)

        for risk_category, risk_analysis in overfitting_risks.items():
            report.append(f"\n🔍 {risk_category.upper().replace('_', ' ')}")
            report.append(f"Risk Level: {risk_analysis.get('risk_level', 'unknown')}")

            if 'total_parameters' in risk_analysis:
                report.append(f"Total Parameters: {risk_analysis['total_parameters']}")

            if 'recommendation' in risk_analysis:
                report.append(f"Recommendation: {risk_analysis['recommendation']}")

            if 'recommendations' in risk_analysis and risk_analysis['recommendations']:
                report.append("Specific Recommendations:")
                for rec in risk_analysis['recommendations']:
                    report.append(f"  • {rec}")

        # Summary and Next Steps
        report.append("\n🎯 SUMMARY & NEXT STEPS")
        report.append("-" * 50)
        report.append("1. Implement walk-forward validation for robust testing")
        report.append("2. Add parameter regularization to prevent overfitting")
        report.append("3. Monitor parameter correlations during optimization")
        report.append("4. Consider hierarchical parameter optimization")
        report.append("5. Set up automated parameter drift detection")
        report.append("6. Implement early stopping based on validation performance")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

def main():
    """Main function to run the parameter analysis."""
    print("🚀 S/R Parameter Interdependence Analysis Tool")
    print("=" * 60)

    analyzer = SRParameterAnalyzer()

    if not analyzer.config:
        print("❌ Failed to load configuration. Please check file paths.")
        return

    # Generate and display report
    report = analyzer.generate_report()
    print(report)

    # Save report to file
    report_file = Path("sr_parameter_analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    main()
