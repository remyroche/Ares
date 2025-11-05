"""
Comprehensive Quality Assessment for Statsmodels Clustering

This module provides comprehensive quality assessment including stability metrics,
calibration tests, residual analysis, and economic validation.

Key Features:
- Stability metrics (ARI/NMI across bootstrap samples)
- Calibration tests for regime probabilities
- Residual analysis (serial correlation, heteroscedasticity)
- Sensitivity analysis and change point detection
- Integration with cluster_quality_assessor.py
"""

from .stability_metrics import StabilityMetricsCalculator
from .calibration_tests import CalibrationTester
from .residual_tests import ResidualAnalyzer
from .sensitivity_analysis import SensitivityAnalyzer
from .change_point_detection import ChangePointDetector
from .economic_validation import EconomicValidator
from .quality_integration import QualityAssessmentIntegrator

__all__ = [
    'StabilityMetricsCalculator',
    'CalibrationTester',
    'ResidualAnalyzer', 
    'SensitivityAnalyzer',
    'ChangePointDetector',
    'EconomicValidator',
    'QualityAssessmentIntegrator'
]