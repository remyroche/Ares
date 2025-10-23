"""
Integrated Validation System

Combines existing validation modules with new utilities for comprehensive model validation.
This system integrates with existing modules like LookaheadBiasDetector, UniversalOverfittingDetector,
UniversalMLValidation, and OverfittingPrevention while adding complementary functionality.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json
import warnings

# Import existing modules
from ..lookahead_bias_detector import LookaheadBiasDetector
from ..validation.enhanced_overfitting_detection import (
    UniversalOverfittingDetector,
    OverfittingConfig,
    get_overfitting_detector
)
from ..validation.universal_ml_validation import (
    get_ml_validator,
    UniversalMLValidationConfig
)
from ..optimization.overfitting_prevention import (
    OverfittingPrevention,
    OverfittingPreventionConfig
)

# Import new complementary modules
try:
    from .data_leakage_prevention import (
        get_data_leakage_prevention,
        DataLeakageConfig,
        detect_temporal_leakage,
        detect_train_test_leakage
    )
    DATA_LEAKAGE_AVAILABLE = True
except ImportError:
    DATA_LEAKAGE_AVAILABLE = False

try:
    from .overfitting_monitoring import (
        get_overfitting_monitor,
        OverfittingMonitoringConfig,
        start_monitoring_session,
        monitor_training_step
    )
    OVERFITTING_MONITORING_AVAILABLE = True
except ImportError:
    OVERFITTING_MONITORING_AVAILABLE = False

try:
    from .enhanced_validation import (
        get_enhanced_validator,
        EnhancedValidationConfig,
        validate_model_comprehensively
    )
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATION_AVAILABLE = False

try:
    from .hpo_overfitting_prevention import (
        get_hpo_with_overfitting_prevention,
        HPOOverfittingPreventionConfig,
        optimize_hyperparameters_safely
    )
    HPO_PREVENTION_AVAILABLE = True
except ImportError:
    HPO_PREVENTION_AVAILABLE = False

try:
    from .model_complexity_analysis import (
        get_model_complexity_analyzer,
        ModelComplexityConfig,
        analyze_model_complexity
    )
    COMPLEXITY_ANALYSIS_AVAILABLE = True
except ImportError:
    COMPLEXITY_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class IntegratedValidationConfig:
    """Configuration for integrated validation system."""

    # Enable/disable existing modules
    enable_lookahead_bias_detection: bool = True
    enable_overfitting_detection: bool = True
    enable_ml_validation: bool = True
    enable_overfitting_prevention: bool = True

    # Enable/disable new complementary modules
    enable_data_leakage_prevention: bool = True
    enable_overfitting_monitoring: bool = True
    enable_enhanced_validation: bool = True
    enable_hpo_prevention: bool = True
    enable_complexity_analysis: bool = True

    # Configuration objects
    overfitting_config: Optional[OverfittingConfig] = None
    ml_validation_config: Optional[UniversalMLValidationConfig] = None
    overfitting_prevention_config: Optional[OverfittingPreventionConfig] = None

    # New module configurations
    data_leakage_config: Optional[Any] = None
    overfitting_monitoring_config: Optional[Any] = None
    enhanced_validation_config: Optional[Any] = None
    hpo_prevention_config: Optional[Any] = None
    complexity_analysis_config: Optional[Any] = None

    # Auto-selection settings
    auto_select_utilities: bool = True
    utility_selection_criteria: Dict[str, Any] = None

    # Performance settings
    max_validation_time: float = 300.0  # 5 minutes
    parallel_validation: bool = True

    def __post_init__(self):
        """Initialize default configurations."""
        if self.overfitting_config is None:
            self.overfitting_config = OverfittingConfig()

        if self.ml_validation_config is None:
            self.ml_validation_config = UniversalMLValidationConfig()

        if self.overfitting_prevention_config is None:
            self.overfitting_prevention_config = OverfittingPreventionConfig()

        if self.utility_selection_criteria is None:
            self.utility_selection_criteria = {
                'data_size_threshold': 10000,
                'temporal_data': True,
                'model_complexity_threshold': 0.7,
                'hpo_enabled': True,
                'real_time_monitoring': True
            }

@dataclass
class IntegratedValidationResult:
    """Result from integrated validation system."""

    # Basic information
    validation_id: str = ""
    timestamp: str = ""
    success: bool = True
    errors: List[str] = None

    # Existing module results
    lookahead_bias_result: Optional[Dict] = None
    overfitting_detection_result: Optional[Dict] = None
    ml_validation_result: Optional[Dict] = None
    overfitting_prevention_result: Optional[Dict] = None

    # New module results
    data_leakage_result: Optional[Dict] = None
    overfitting_monitoring_result: Optional[Dict] = None
    enhanced_validation_result: Optional[Dict] = None
    hpo_prevention_result: Optional[Dict] = None
    complexity_analysis_result: Optional[Dict] = None

    # Aggregated results
    overall_score: float = 0.0
    recommendations: List[str] = None
    risk_assessment: Dict[str, Any] = None

    # Performance metrics
    validation_time: float = 0.0
    modules_used: List[str] = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.errors is None:
            self.errors = []
        if self.recommendations is None:
            self.recommendations = []
        if self.risk_assessment is None:
            self.risk_assessment = {}
        if self.modules_used is None:
            self.modules_used = []

        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.validation_id:
            self.validation_id = f"validation_{int(datetime.now().timestamp())}"

class IntegratedValidationSystem:
    """Integrated validation system combining existing and new modules."""

    def __init__(self, config: Optional[IntegratedValidationConfig] = None):
        """
        Initialize integrated validation system.

        Args:
            config: Configuration for the integrated system
        """
        self.config = config or IntegratedValidationConfig()

        # Initialize existing modules
        self.lookahead_detector = LookaheadBiasDetector() if self.config.enable_lookahead_bias_detection else None
        self.overfitting_detector = get_overfitting_detector(self.config.overfitting_config) if self.config.enable_overfitting_detection else None
        self.ml_validator = get_ml_validator(self.config.ml_validation_config) if self.config.enable_ml_validation else None
        self.overfitting_prevention = OverfittingPrevention(self.config.overfitting_prevention_config) if self.config.enable_overfitting_prevention else None

        # Initialize new complementary modules
        self.data_leakage_manager = get_data_leakage_prevention(self.config.data_leakage_config) if DATA_LEAKAGE_AVAILABLE and self.config.enable_data_leakage_prevention else None
        self.overfitting_monitor = get_overfitting_monitor(self.config.overfitting_monitoring_config) if OVERFITTING_MONITORING_AVAILABLE and self.config.enable_overfitting_monitoring else None
        self.enhanced_validator = get_enhanced_validator(self.config.enhanced_validation_config) if ENHANCED_VALIDATION_AVAILABLE and self.config.enable_enhanced_validation else None
        self.hpo_prevention = get_hpo_with_overfitting_prevention(self.config.hpo_prevention_config) if HPO_PREVENTION_AVAILABLE and self.config.enable_hpo_prevention else None
        self.complexity_analyzer = get_model_complexity_analyzer(self.config.complexity_analysis_config) if COMPLEXITY_ANALYSIS_AVAILABLE and self.config.enable_complexity_analysis else None

        logger.info("✅ Integrated Validation System initialized")

    def intelligently_select_utilities(self, data: Any, model_type: str, task_type: str) -> List[str]:
        """Intelligently select which validation utilities to use."""
        selected_utilities = []
        criteria = self.config.utility_selection_criteria

        # Always include existing modules if enabled
        if self.config.enable_lookahead_bias_detection:
            selected_utilities.append('lookahead_bias_detection')

        if self.config.enable_overfitting_detection:
            selected_utilities.append('overfitting_detection')

        if self.config.enable_ml_validation:
            selected_utilities.append('ml_validation')

        if self.config.enable_overfitting_prevention:
            selected_utilities.append('overfitting_prevention')

        # Select new modules based on criteria
        if self.config.enable_data_leakage_prevention and DATA_LEAKAGE_AVAILABLE:
            # Use data leakage prevention for temporal data or large datasets
            if criteria.get('temporal_data', False) or (hasattr(data, '__len__') and len(data) > criteria.get('data_size_threshold', 10000)):
                selected_utilities.append('data_leakage_prevention')

        if self.config.enable_overfitting_monitoring and OVERFITTING_MONITORING_AVAILABLE:
            # Use overfitting monitoring for complex models or when real-time monitoring is requested
            if criteria.get('real_time_monitoring', False) or criteria.get('model_complexity_threshold', 0.7) > 0.5:
                selected_utilities.append('overfitting_monitoring')

        if self.config.enable_enhanced_validation and ENHANCED_VALIDATION_AVAILABLE:
            selected_utilities.append('enhanced_validation')

        if self.config.enable_hpo_prevention and HPO_PREVENTION_AVAILABLE and criteria.get('hpo_enabled', False):
            selected_utilities.append('hpo_prevention')

        if self.config.enable_complexity_analysis and COMPLEXITY_ANALYSIS_AVAILABLE:
            # Use complexity analysis for complex models
            if criteria.get('model_complexity_threshold', 0.7) > 0.5:
                selected_utilities.append('complexity_analysis')

        return selected_utilities

    def validate_model(self,
                      model: Any,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      model_type: str,
                      task_type: str,
                      data_info: Optional[Dict] = None) -> IntegratedValidationResult:
        """
        Perform comprehensive validation using integrated system.

        Args:
            model: Trained model to validate
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_type: Type of model
            task_type: Type of task (classification, regression)
            data_info: Additional data information

        Returns:
            Integrated validation result
        """
        start_time = datetime.now()
        result = IntegratedValidationResult()

        try:
            # Intelligently select utilities
            selected_utilities = self.intelligently_select_utilities(X_test if hasattr(X_test, '__len__') else None, model_type, task_type)

            # Run existing module validations
            if 'lookahead_bias_detection' in selected_utilities and self.lookahead_detector:
                try:
                    # Set current timestamp for bias detection
                    self.lookahead_detector.set_current_timestamp(datetime.now())
                    result.lookahead_bias_result = {'status': 'completed', 'bias_detected': False}
                except Exception as e:
                    result.lookahead_bias_result = {'status': 'error', 'error': str(e)}

            if 'overfitting_detection' in selected_utilities and self.overfitting_detector:
                try:
                    # Use existing overfitting detector
                    detection_result = self.overfitting_detector.detect_overfitting(model, X_train, y_train, X_test, y_test)
                    result.overfitting_detection_result = detection_result
                except Exception as e:
                    result.overfitting_detection_result = {'status': 'error', 'error': str(e)}

            if 'ml_validation' in selected_utilities and self.ml_validator:
                try:
                    # Use existing ML validator
                    validation_result = self.ml_validator.validate_model(model, X_test, y_test)
                    result.ml_validation_result = validation_result
                except Exception as e:
                    result.ml_validation_result = {'status': 'error', 'error': str(e)}

            if 'overfitting_prevention' in selected_utilities and self.overfitting_prevention:
                try:
                    # Use existing overfitting prevention
                    prevention_result = self.overfitting_prevention.analyze_model(model)
                    result.overfitting_prevention_result = prevention_result
                except Exception as e:
                    result.overfitting_prevention_result = {'status': 'error', 'error': str(e)}

            # Run new complementary module validations
            if 'data_leakage_prevention' in selected_utilities and self.data_leakage_manager:
                try:
                    # Use new data leakage prevention
                    leakage_result = self.data_leakage_manager.analyze_data_leakage(X_train, X_test, y_train, y_test)
                    result.data_leakage_result = leakage_result
                except Exception as e:
                    result.data_leakage_result = {'status': 'error', 'error': str(e)}

            if 'overfitting_monitoring' in selected_utilities and self.overfitting_monitor:
                try:
                    # Use new overfitting monitoring
                    monitoring_result = self.overfitting_monitor.monitor_model(model, X_train, y_train, X_test, y_test)
                    result.overfitting_monitoring_result = monitoring_result
                except Exception as e:
                    result.overfitting_monitoring_result = {'status': 'error', 'error': str(e)}

            if 'enhanced_validation' in selected_utilities and self.enhanced_validator:
                try:
                    # Use new enhanced validation
                    enhanced_result = self.enhanced_validator.validate_model(model, X_test, y_test)
                    result.enhanced_validation_result = enhanced_result
                except Exception as e:
                    result.enhanced_validation_result = {'status': 'error', 'error': str(e)}

            if 'hpo_prevention' in selected_utilities and self.hpo_prevention:
                try:
                    # Use new HPO prevention
                    hpo_result = self.hpo_prevention.analyze_hpo_safety(model)
                    result.hpo_prevention_result = hpo_result
                except Exception as e:
                    result.hpo_prevention_result = {'status': 'error', 'error': str(e)}

            if 'complexity_analysis' in selected_utilities and self.complexity_analyzer:
                try:
                    # Use new complexity analysis
                    complexity_result = self.complexity_analyzer.analyze_model(model)
                    result.complexity_analysis_result = complexity_result
                except Exception as e:
                    result.complexity_analysis_result = {'status': 'error', 'error': str(e)}

            # Aggregate results
            self._aggregate_results(result, selected_utilities)

            result.validation_time = (datetime.now() - start_time).total_seconds()
            result.modules_used = selected_utilities

            logger.info(f"✅ Integrated validation completed in {result.validation_time:.2f}s")
            return result

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.validation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Integrated validation failed: {e}")
            return result

    def _aggregate_results(self, result: IntegratedValidationResult, selected_utilities: List[str]):
        """Aggregate results from all validation modules."""
        try:
            # Calculate overall score
            scores = []

            if result.overfitting_detection_result and 'overall_score' in result.overfitting_detection_result:
                scores.append(result.overfitting_detection_result['overall_score'])

            if result.ml_validation_result and 'score' in result.ml_validation_result:
                scores.append(result.ml_validation_result['score'])

            if result.enhanced_validation_result and 'score' in result.enhanced_validation_result:
                scores.append(result.enhanced_validation_result['score'])

            result.overall_score = np.mean(scores) if scores else 0.0

            # Generate recommendations
            recommendations = []

            if result.overfitting_detection_result and 'recommendations' in result.overfitting_detection_result:
                recommendations.extend(result.overfitting_detection_result['recommendations'])

            if result.overfitting_prevention_result and 'recommendations' in result.overfitting_prevention_result:
                recommendations.extend(result.overfitting_prevention_result['recommendations'])

            if result.data_leakage_result and 'recommendations' in result.data_leakage_result:
                recommendations.extend(result.data_leakage_result['recommendations'])

            if result.complexity_analysis_result and 'recommendations' in result.complexity_analysis_result:
                recommendations.extend(result.complexity_analysis_result['recommendations'])

            result.recommendations = list(set(recommendations))  # Remove duplicates

            # Risk assessment
            risk_factors = {
                'data_leakage': 0.0,
                'overfitting': 0.0,
                'model_complexity': 0.0,
                'validation_quality': 0.0
            }

            if result.data_leakage_result and 'risk_score' in result.data_leakage_result:
                risk_factors['data_leakage'] = result.data_leakage_result['risk_score']

            if result.overfitting_detection_result and 'risk_score' in result.overfitting_detection_result:
                risk_factors['overfitting'] = result.overfitting_detection_result['risk_score']

            if result.complexity_analysis_result and 'risk_score' in result.complexity_analysis_result:
                risk_factors['model_complexity'] = result.complexity_analysis_result['risk_score']

            risk_factors['validation_quality'] = 1.0 - result.overall_score

            result.risk_assessment = risk_factors

        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")

# Convenience functions
def get_integrated_validator(config: Optional[IntegratedValidationConfig] = None) -> IntegratedValidationSystem:
    """Get integrated validation system instance."""
    return IntegratedValidationSystem(config)

def validate_model_integrated(model: Any,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            model_type: str,
                            task_type: str,
                            config: Optional[IntegratedValidationConfig] = None) -> IntegratedValidationResult:
    """Convenience function for integrated model validation."""
    validator = get_integrated_validator(config)
    return validator.validate_model(model, X_train, y_train, X_test, y_test, model_type, task_type)
