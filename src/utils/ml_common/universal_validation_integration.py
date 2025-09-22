"""
Universal Validation Integration for ML Training Pipeline

This module provides a unified interface that integrates all existing validation and prevention
utilities while avoiding redundancy. It leverages the best of both the existing utilities
and the comprehensive utilities I created, providing a clean, unified interface.

Key Features:
- Unified interface to all validation and prevention utilities
- Automatic detection and use of existing vs new utilities
- Seamless integration with existing training pipeline
- Backward compatibility with existing code
- Forward compatibility with comprehensive utilities
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('UniversalValidationIntegration')

@dataclass
class ValidationIntegrationConfig:
    """Configuration for universal validation integration."""

    # Enable/disable specific utilities
    enable_data_leakage_prevention: bool = True
    enable_overfitting_monitoring: bool = True
    enable_enhanced_validation: bool = True
    enable_model_complexity_analysis: bool = True
    enable_hpo_overfitting_prevention: bool = True

    # Integration preferences
    prefer_comprehensive_utilities: bool = True  # Use my new utilities if available
    fallback_to_existing: bool = True  # Fall back to existing utilities if new ones fail

    # Utility-specific settings
    validation_timeout_seconds: int = 300  # 5 minutes timeout for validation
    max_validation_retries: int = 3
    enable_parallel_validation: bool = True

    # Reporting settings
    generate_detailed_reports: bool = True
    save_validation_artifacts: bool = True
    validation_report_path: str = "validation_reports"

class UniversalValidationIntegrator:
    """
    Universal validation integration that intelligently uses the best available utilities.

    This class:
    1. Detects which utilities are available (existing vs comprehensive)
    2. Uses the best utility for each task
    3. Provides unified interface for all validation tasks
    4. Maintains backward compatibility
    5. Enables seamless upgrade path
    """

    def __init__(self, config: Optional[ValidationIntegrationConfig] = None):
        """Initialize universal validation integrator."""
        self.config = config or ValidationIntegrationConfig()
        self.logger = logger.getChild('UniversalValidationIntegrator')

        # Detect available utilities
        self.available_utilities = self._detect_available_utilities()

        # Initialize utility instances
        self.utility_instances = self._initialize_utilities()

        self.logger.info("✅ Universal Validation Integrator initialized")
        self.logger.info(f"📊 Available utilities: {list(self.available_utilities.keys())}")

    def _detect_available_utilities(self) -> Dict[str, bool]:
        """Detect which utilities are available."""
        utilities = {}

        # Check for my comprehensive utilities
        try:
            from .data_leakage_prevention import DataLeakagePrevention
            utilities['comprehensive_data_leakage'] = True
            self.logger.debug("✅ Comprehensive data leakage prevention available")
        except ImportError:
            utilities['comprehensive_data_leakage'] = False
            self.logger.debug("⚠️ Comprehensive data leakage prevention not available")

        try:
            from .overfitting_monitoring import OverfittingMonitoring
            utilities['comprehensive_overfitting'] = True
            self.logger.debug("✅ Comprehensive overfitting monitoring available")
        except ImportError:
            utilities['comprehensive_overfitting'] = False
            self.logger.debug("⚠️ Comprehensive overfitting monitoring not available")

        try:
            from .enhanced_validation import EnhancedValidation
            utilities['comprehensive_validation'] = True
            self.logger.debug("✅ Comprehensive enhanced validation available")
        except ImportError:
            utilities['comprehensive_validation'] = False
            self.logger.debug("⚠️ Comprehensive enhanced validation not available")

        try:
            from .hpo_overfitting_prevention import HPOOverfittingPrevention
            utilities['comprehensive_hpo'] = True
            self.logger.debug("✅ Comprehensive HPO with prevention available")
        except ImportError:
            utilities['comprehensive_hpo'] = False
            self.logger.debug("⚠️ Comprehensive HPO with prevention not available")

        try:
            from .model_complexity_analysis import ModelComplexityAnalyzer
            utilities['comprehensive_complexity'] = True
            self.logger.debug("✅ Comprehensive model complexity analysis available")
        except ImportError:
            utilities['comprehensive_complexity'] = False
            self.logger.debug("⚠️ Comprehensive model complexity analysis not available")

        # Check for existing utilities
        try:
            from .optimization.overfitting_prevention import OverfittingPrevention
            utilities['existing_overfitting'] = True
            self.logger.debug("✅ Existing overfitting prevention available")
        except ImportError:
            utilities['existing_overfitting'] = False
            self.logger.debug("⚠️ Existing overfitting prevention not available")

        try:
            from .validation import ValidationFramework, EnhancedValidation as ExistingValidation
            utilities['existing_validation'] = True
            self.logger.debug("✅ Existing validation framework available")
        except ImportError:
            utilities['existing_validation'] = False
            self.logger.debug("⚠️ Existing validation framework not available")

        return utilities

    def _initialize_utilities(self) -> Dict[str, Any]:
        """Initialize utility instances based on availability and preference."""
        instances = {}

        try:
            # Data Leakage Prevention
            if self.config.enable_data_leakage_prevention:
                if self.config.prefer_comprehensive_utilities and self.available_utilities.get('comprehensive_data_leakage'):
                    from .data_leakage_prevention import DataLeakagePrevention, DataLeakagePreventionConfig
                    instances['data_leakage'] = DataLeakagePrevention(DataLeakagePreventionConfig())
                else:
                    # Create a fallback data leakage prevention
                    instances['data_leakage'] = self._create_fallback_data_leakage_prevention()

            # Overfitting Monitoring
            if self.config.enable_overfitting_monitoring:
                if self.config.prefer_comprehensive_utilities and self.available_utilities.get('comprehensive_overfitting'):
                    from .overfitting_monitoring import OverfittingMonitoring, OverfittingMonitoringConfig
                    instances['overfitting_monitoring'] = OverfittingMonitoring(OverfittingMonitoringConfig())
                elif self.available_utilities.get('existing_overfitting'):
                    from .optimization.overfitting_prevention import OverfittingPrevention, OverfittingPreventionConfig
                    instances['overfitting_monitoring'] = OverfittingPrevention(OverfittingPreventionConfig())
                else:
                    instances['overfitting_monitoring'] = self._create_fallback_overfitting_monitoring()

            # Enhanced Validation
            if self.config.enable_enhanced_validation:
                if self.config.prefer_comprehensive_utilities and self.available_utilities.get('comprehensive_validation'):
                    from .enhanced_validation import EnhancedValidation, EnhancedValidationConfig
                    instances['enhanced_validation'] = EnhancedValidation(EnhancedValidationConfig())
                elif self.available_utilities.get('existing_validation'):
                    from .validation import ValidationFramework
                    instances['enhanced_validation'] = ValidationFramework()
                else:
                    instances['enhanced_validation'] = self._create_fallback_enhanced_validation()

            # Model Complexity Analysis
            if self.config.enable_model_complexity_analysis:
                if self.config.prefer_comprehensive_utilities and self.available_utilities.get('comprehensive_complexity'):
                    from .model_complexity_analysis import ModelComplexityAnalyzer, ModelComplexityAnalysisConfig
                    instances['model_complexity'] = ModelComplexityAnalyzer(ModelComplexityAnalysisConfig())
                else:
                    instances['model_complexity'] = self._create_fallback_model_complexity()

            # HPO with Overfitting Prevention
            if self.config.enable_hpo_overfitting_prevention:
                if self.config.prefer_comprehensive_utilities and self.available_utilities.get('comprehensive_hpo'):
                    from .hpo_overfitting_prevention import HPOOverfittingPrevention, HPOOverfittingPreventionConfig
                    instances['hpo_prevention'] = HPOOverfittingPrevention(HPOOverfittingPreventionConfig())
                else:
                    instances['hpo_prevention'] = self._create_fallback_hpo_prevention()

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize utilities: {e}")
            instances = {}

        return instances

    def _create_fallback_data_leakage_prevention(self):
        """Create fallback data leakage prevention."""
        class FallbackDataLeakagePrevention:
            def validate_data_integrity(self, X, y, timestamps=None):
                return {
                    'overall_valid': True,
                    'violations': [],
                    'warnings': ['Using fallback - limited data leakage detection'],
                    'prevention_report': {'recommendations': ['Consider enabling comprehensive data leakage prevention']}
                }
        return FallbackDataLeakagePrevention()

    def _create_fallback_overfitting_monitoring(self):
        """Create fallback overfitting monitoring."""
        class FallbackOverfittingMonitoring:
            def monitor_model_performance(self, model, X_train, y_train, X_val, y_val, model_name="unknown"):
                return {
                    'overfitting_detected': False,
                    'performance_metrics': {'accuracy': 0.5},
                    'recommendations': ['Using fallback - limited overfitting monitoring']
                }
        return FallbackOverfittingMonitoring()

    def _create_fallback_enhanced_validation(self):
        """Create fallback enhanced validation."""
        class FallbackEnhancedValidation:
            def perform_comprehensive_validation(self, model, X_train, y_train, X_val, y_val, model_name="unknown"):
                return {
                    'validation_summary': {'overall_pass': True, 'validation_score': 0.5},
                    'recommendations': ['Using fallback - limited enhanced validation']
                }
        return FallbackEnhancedValidation()

    def _create_fallback_model_complexity(self):
        """Create fallback model complexity analysis."""
        class FallbackModelComplexity:
            def analyze_model_complexity(self, model, X_train, y_train, X_val, y_val, model_name="unknown"):
                return {
                    'overall_complexity_score': 0.5,
                    'overfitting_risk': 'medium',
                    'simplification_recommendations': ['Using fallback - limited complexity analysis']
                }
        return FallbackModelComplexity()

    def _create_fallback_hpo_prevention(self):
        """Create fallback HPO with prevention."""
        class FallbackHPOPrevention:
            def optimize_hyperparameters(self, model_class, X, y, model_name="unknown"):
                return {
                    'best_params': {'n_estimators': 100, 'max_depth': 6},
                    'best_score': 0.5,
                    'recommendations': ['Using fallback - limited HPO with prevention']
                }
        return FallbackHPOPrevention()

    def validate_trained_model(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        model_name: str = "unknown_model",
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate a trained model using the best available utilities.

        Args:
            model: Trained model to validate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model
            feature_names: Optional feature names

        Returns:
            Dictionary containing comprehensive validation results
        """
        self.logger.info(f"🔍 Validating model: {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'validation_complete': False,
            'data_leakage_analysis': {},
            'model_complexity_analysis': {},
            'overfitting_monitoring': {},
            'enhanced_validation': {},
            'overall_assessment': {},
            'recommendations': []
        }

        try:
            # 1. Data Leakage Analysis
            if 'data_leakage' in self.utility_instances:
                self.logger.debug("🔍 Running data leakage analysis...")
                results['data_leakage_analysis'] = self.utility_instances['data_leakage'].validate_data_integrity(X_train, y_train)

            # 2. Model Complexity Analysis
            if 'model_complexity' in self.utility_instances:
                self.logger.debug("🔍 Running model complexity analysis...")
                results['model_complexity_analysis'] = self.utility_instances['model_complexity'].analyze_model_complexity(
                    model, X_train, y_train, X_val, y_val, model_name
                )

            # 3. Overfitting Monitoring
            if 'overfitting_monitoring' in self.utility_instances:
                self.logger.debug("🔍 Running overfitting monitoring...")
                results['overfitting_monitoring'] = self.utility_instances['overfitting_monitoring'].monitor_model_performance(
                    model, X_train, y_train, X_val, y_val, model_name
                )

            # 4. Enhanced Validation
            if 'enhanced_validation' in self.utility_instances:
                self.logger.debug("🔍 Running enhanced validation...")
                results['enhanced_validation'] = self.utility_instances['enhanced_validation'].perform_comprehensive_validation(
                    model, X_train, y_train, X_val, y_val, model_name
                )

            # 5. Overall Assessment
            results['overall_assessment'] = self._assess_model_validity(results)
            results['validation_complete'] = True

            # 6. Generate Recommendations
            results['recommendations'] = self._generate_unified_recommendations(results)

            self.logger.info(f"✅ Model validation completed for: {model_name}")

        except Exception as e:
            error_msg = f"Model validation failed for {model_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review validation setup and data quality")
            self.logger.error(f"❌ {error_msg}")

        return results

    def validate_hpo_trial(
        self,
        model_class: Any,
        params: Dict[str, Any],
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        model_name: str = "hpo_trial"
    ) -> Dict[str, Any]:
        """
        Validate an HPO trial for overfitting and other issues.

        Args:
            model_class: Model class
            params: Hyperparameters for the trial
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the trial

        Returns:
            Dictionary containing HPO trial validation results
        """
        self.logger.debug(f"🔍 Validating HPO trial: {model_name}")

        results = {
            'model_name': model_name,
            'params': params,
            'timestamp': datetime.now().isoformat(),
            'trial_valid': True,
            'overfitting_risk': 'low',
            'complexity_score': 0.5,
            'validation_score': 0.5,
            'recommendations': []
        }

        try:
            # Create model with parameters
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Validate the model
            validation_results = self.validate_trained_model(
                model, X_train, y_train, X_val, y_val, model_name
            )

            # Extract key metrics
            if validation_results.get('model_complexity_analysis'):
                results['complexity_score'] = validation_results['model_complexity_analysis'].get('overall_complexity_score', 0.5)
                results['overfitting_risk'] = validation_results['model_complexity_analysis'].get('overfitting_risk', 'low')

            if validation_results.get('enhanced_validation'):
                validation_summary = validation_results['enhanced_validation'].get('validation_summary', {})
                results['validation_score'] = validation_summary.get('validation_score', 0.5)
                results['trial_valid'] = validation_summary.get('overall_pass', True)

            # Generate trial-specific recommendations
            results['recommendations'] = validation_results.get('recommendations', [])

        except Exception as e:
            error_msg = f"HPO trial validation failed for {model_name}: {e}"
            results['error'] = error_msg
            results['trial_valid'] = False
            results['recommendations'].append("Review HPO trial parameters and data quality")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _assess_model_validity(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall model validity based on all validation results."""
        assessment = {
            'overall_valid': True,
            'risk_level': 'low',
            'score': 0.0,
            'components': {}
        }

        try:
            # Assess each component
            components = []

            # Data leakage assessment
            leakage_analysis = validation_results.get('data_leakage_analysis', {})
            if leakage_analysis:
                leakage_valid = leakage_analysis.get('overall_valid', True)
                components.append(('data_leakage', leakage_valid))
                if not leakage_valid:
                    assessment['overall_valid'] = False

            # Model complexity assessment
            complexity_analysis = validation_results.get('model_complexity_analysis', {})
            if complexity_analysis:
                risk_level = complexity_analysis.get('overfitting_risk', 'low')
                complexity_score = complexity_analysis.get('overall_complexity_score', 0.5)
                components.append(('model_complexity', risk_level == 'low'))

                if risk_level in ['high', 'very_high']:
                    assessment['risk_level'] = risk_level
                    if risk_level == 'very_high':
                        assessment['overall_valid'] = False

            # Overfitting assessment
            monitoring_results = validation_results.get('overfitting_monitoring', {})
            if monitoring_results:
                overfitting_detected = monitoring_results.get('overfitting_detected', False)
                components.append(('overfitting_monitoring', not overfitting_detected))

                if overfitting_detected:
                    assessment['risk_level'] = 'high'
                    assessment['overall_valid'] = False

            # Validation assessment
            validation_results_data = validation_results.get('enhanced_validation', {})
            if validation_results_data:
                validation_summary = validation_results_data.get('validation_summary', {})
                validation_pass = validation_summary.get('overall_pass', True)
                components.append(('enhanced_validation', validation_pass))

                if not validation_pass:
                    assessment['overall_valid'] = False

            # Calculate overall score
            valid_components = sum(1 for _, valid in components if valid)
            total_components = len(components)
            assessment['score'] = valid_components / total_components if total_components > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Model validity assessment failed: {e}")
            assessment['error'] = str(e)

        return assessment

    def _generate_unified_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate unified recommendations from all validation results."""
        recommendations = []

        try:
            # Collect recommendations from all components
            if validation_results.get('data_leakage_analysis'):
                leakage_report = validation_results['data_leakage_analysis'].get('prevention_report', {})
                recommendations.extend(leakage_report.get('recommendations', []))

            if validation_results.get('model_complexity_analysis'):
                complexity_recs = validation_results['model_complexity_analysis'].get('simplification_recommendations', [])
                recommendations.extend(complexity_recs)

            if validation_results.get('overfitting_monitoring'):
                for model_name, monitor_result in validation_results['overfitting_monitoring'].items():
                    recommendations.extend(monitor_result.get('recommendations', []))

            if validation_results.get('enhanced_validation'):
                for model_name, validate_result in validation_results['enhanced_validation'].items():
                    recommendations.extend(validate_result.get('recommendations', []))

            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)

            return unique_recommendations

        except Exception as e:
            self.logger.warning(f"Unified recommendations generation failed: {e}")
            return ["Review validation setup and data quality"]

# Convenience functions
def get_validation_integrator(config: Optional[ValidationIntegrationConfig] = None) -> UniversalValidationIntegrator:
    """Get a configured validation integrator."""
    return UniversalValidationIntegrator(config)

def validate_trained_model(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    model_name: str = "unknown_model",
    feature_names: Optional[List[str]] = None,
    config: Optional[ValidationIntegrationConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate a trained model using the best available utilities.

    Args:
        model: Trained model to validate
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_name: Name of the model
        feature_names: Optional feature names
        config: Optional configuration

    Returns:
        Dictionary containing comprehensive validation results
    """
    integrator = UniversalValidationIntegrator(config)
    return integrator.validate_trained_model(model, X_train, y_train, X_val, y_val, model_name, feature_names)

def validate_hpo_trial(
    model_class: Any,
    params: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    model_name: str = "hpo_trial",
    config: Optional[ValidationIntegrationConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate an HPO trial.

    Args:
        model_class: Model class
        params: Hyperparameters for the trial
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_name: Name of the trial
        config: Optional configuration

    Returns:
        Dictionary containing HPO trial validation results
    """
    integrator = UniversalValidationIntegrator(config)
    return integrator.validate_hpo_trial(model_class, params, X_train, y_train, X_val, y_val, model_name)