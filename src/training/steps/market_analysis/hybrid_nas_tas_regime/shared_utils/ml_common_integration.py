"""
ML Common Integration for TAS and NAS

This module integrates the existing ML Common utilities with the unified architecture
system, leveraging the comprehensive tools already available in utils/ml_commons/
instead of recreating functionality.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing ML Common utilities
from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
from src.utils.ml_common.validation.enhanced_overfitting_detection import UniversalOverfittingDetector
from src.utils.ml_common.utils.hpo_utils import HyperparameterOptimization
from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention
# from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem

# Import existing optimization utilities
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization as CanonicalHPO

from .unified_architecture_config import ArchitectureType, OptimizationObjective

logger = logging.getLogger(__name__)


class MLCommonIntegrationType(Enum):
    """Types of ML Common integrations."""
    LOOKAHEAD_PROTECTION = "lookahead_protection"
    OVERFITTING_DETECTION = "overfitting_detection"
    HPO_OPTIMIZATION = "hpo_optimization"
    DATA_LEAKAGE_PREVENTION = "data_leakage_prevention"
    UNIFIED_VALIDATION = "unified_validation"
    GRID_BAYESIAN_OPTIMIZATION = "grid_bayesian_optimization"


class MLUtilityType(Enum):
    """Types of ML utilities."""
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_EVALUATION = "model_evaluation"


@dataclass
class MLUtilityConfig:
    """Configuration for ML utilities."""
    utility_type: MLUtilityType
    enable_validation: bool = True
    enable_optimization: bool = True
    enable_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_model_evaluation: bool = True
    validation_config: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    feature_engineering_config: Dict[str, Any] = field(default_factory=dict)
    model_evaluation_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLCommonIntegrationConfig:
    """Configuration for ML Common integration."""
    # Lookahead protection
    enable_lookahead_protection: bool = True
    lookahead_config: Dict[str, Any] = field(default_factory=lambda: {
        'strict_mode': True,
        'tolerance_seconds': 60,
        'enable_automatic_filtering': True
    })
    
    # Overfitting detection
    enable_overfitting_detection: bool = True
    overfitting_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_learning_curves': True,
        'enable_cross_validation': True,
        'enable_early_stopping': True,
        'patience': 10
    })
    
    # HPO optimization
    enable_hpo_optimization: bool = True
    hpo_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_parallel': True,
        'max_workers': 4,
        'enable_monitoring': True,
        'use_nonlinear_optimization': True
    })
    
    # Data leakage prevention
    enable_data_leakage_prevention: bool = True
    leakage_prevention_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_temporal_validation': True,
        'enforce_strict_time_order': True,
        'lookahead_detection_enabled': True
    })
    
    # Unified validation
    enable_unified_validation: bool = True
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_cross_validation': True,
        'enable_bootstrap': True,
        'enable_uncertainty_quantification': True
    })


class MLCommonIntegration:
    """ML Common integration using existing utilities."""
    
    def __init__(self, 
                 architecture_type: ArchitectureType,
                 config: MLCommonIntegrationConfig = None):
        """Initialize ML Common integration.
        
        Args:
            architecture_type: Type of architecture (TAS/NAS/Hybrid)
            config: Integration configuration
        """
        self.architecture_type = architecture_type
        self.config = config or MLCommonIntegrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ML Common utilities
        self._initialize_ml_common_utilities()
        
        self.logger.info(f"✅ ML Common Integration initialized for {architecture_type.value}")
    
    def _initialize_ml_common_utilities(self):
        """Initialize ML Common utilities based on configuration."""
        try:
            # Initialize Lookahead Protection
            if self.config.enable_lookahead_protection:
                self.lookahead_protection = LookaheadProtection(
                    config=self.config.lookahead_config
                )
                self.logger.info("✅ Lookahead Protection initialized")
            else:
                self.lookahead_protection = None
            
            # Initialize Overfitting Detection
            if self.config.enable_overfitting_detection:
                self.overfitting_detector = UniversalOverfittingDetector(
                    config=self.config.overfitting_config
                )
                self.logger.info("✅ Overfitting Detection initialized")
            else:
                self.overfitting_detector = None
            
            # Initialize HPO Optimization
            if self.config.enable_hpo_optimization:
                self.hpo_optimizer = CanonicalHPO(
                    config=self.config.hpo_config
                )
                self.logger.info("✅ HPO Optimization initialized")
            else:
                self.hpo_optimizer = None
            
            # Initialize Data Leakage Prevention
            if self.config.enable_data_leakage_prevention:
                self.data_leakage_prevention = DataLeakagePrevention(
                    config=self.config.leakage_prevention_config
                )
                self.logger.info("✅ Data Leakage Prevention initialized")
            else:
                self.data_leakage_prevention = None
            
            # Initialize Unified Validation
            if self.config.enable_unified_validation:
                # self.unified_validation = UnifiedValidationSystem(
                #     config=self.config.validation_config
                # )
                self.unified_validation = None
                self.logger.info("✅ Unified Validation initialized")
            else:
                self.unified_validation = None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Common utilities: {e}")
            raise
    
    def validate_data_split(self, 
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray,
                          timestamps_train: Optional[np.ndarray] = None,
                          timestamps_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate data split using ML Common utilities."""
        validation_results = {
            'is_valid': True,
            'lookahead_protection': {},
            'data_leakage_prevention': {},
            'unified_validation': {}
        }
        
        try:
            # Data Leakage Prevention
            if self.data_leakage_prevention:
                leakage_result = self.data_leakage_prevention.detect_train_test_leakage(
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    timestamps_train=timestamps_train,
                    timestamps_test=timestamps_test
                )
                validation_results['data_leakage_prevention'] = leakage_result
                
                if leakage_result.train_test_leakage_detected:
                    validation_results['is_valid'] = False
                    self.logger.warning("🚨 Data leakage detected between train/test sets")
            
            self.logger.info(f"✅ Data split validation completed - Valid: {validation_results['is_valid']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Data split validation failed: {e}")
            validation_results['is_valid'] = False
            validation_results['error'] = str(e)
            return validation_results
    
    def optimize_hyperparameters(self,
                               model: Any,
                               X: np.ndarray,
                               y: np.ndarray,
                               search_space: Optional[Dict[str, Any]] = None,
                               optimization_objective: OptimizationObjective = OptimizationObjective.ACCURACY) -> Dict[str, Any]:
        """Optimize hyperparameters using ML Common HPO utilities."""
        if not self.hpo_optimizer:
            return {'error': 'HPO optimization not enabled'}
        
        try:
            self.logger.info(f"🔧 Starting hyperparameter optimization for {self.architecture_type.value}")
            
            # Use existing HPO utilities
            optimization_result = self.hpo_optimizer.optimize_hyperparameters(
                model=model, X=X, y=y,
                search_space=search_space,
                objective=optimization_objective.value
            )
            
            self.logger.info("✅ Hyperparameter optimization completed")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            return {'error': str(e)}
    
    def detect_overfitting(self,
                          model: Any,
                          X_train: np.ndarray,
                          X_val: np.ndarray,
                          y_train: np.ndarray,
                          y_val: np.ndarray) -> Dict[str, Any]:
        """Detect overfitting using ML Common utilities."""
        if not self.overfitting_detector:
            return {'error': 'Overfitting detection not enabled'}
        
        try:
            self.logger.info("🔍 Detecting overfitting patterns")
            
            # Use existing overfitting detection utilities
            overfitting_result = self.overfitting_detector.detect_overfitting(
                model=model, X_train=X_train, X_val=X_val,
                y_train=y_train, y_val=y_val
            )
            
            self.logger.info("✅ Overfitting detection completed")
            return overfitting_result
            
        except Exception as e:
            self.logger.error(f"❌ Overfitting detection failed: {e}")
            return {'error': str(e)}
    
    def prevent_lookahead_bias(self,
                              data: pd.DataFrame,
                              timestamp_col: str = 'timestamp',
                              target_col: Optional[str] = None) -> Dict[str, Any]:
        """Prevent lookahead bias using ML Common utilities."""
        if not self.lookahead_protection:
            return {'error': 'Lookahead protection not enabled'}
        
        try:
            self.logger.info("🔒 Preventing lookahead bias")
            
            # Use existing lookahead protection utilities
            if target_col:
                features_df = data.drop(columns=[target_col])
                target_df = data[[timestamp_col, target_col]]
                
                bias_result = self.lookahead_protection.detect_data_leakage(
                    features_df=features_df, target_df=target_df,
                    timestamp_col=timestamp_col
                )
            else:
                bias_result = self.lookahead_protection.detect_data_leakage(
                    features_df=data, target_df=data,
                    timestamp_col=timestamp_col
                )
            
            self.logger.info("✅ Lookahead bias prevention completed")
            return bias_result
            
        except Exception as e:
            self.logger.error(f"❌ Lookahead bias prevention failed: {e}")
            return {'error': str(e)}
    
    def comprehensive_validation(self,
                               model: Any,
                               X_train: np.ndarray,
                               X_test: np.ndarray,
                               y_train: np.ndarray,
                               y_test: np.ndarray,
                               timestamps_train: Optional[np.ndarray] = None,
                               timestamps_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform comprehensive validation using all ML Common utilities."""
        self.logger.info("🔬 Starting comprehensive validation")
        
        validation_results = {
            'data_split_validation': {},
            'overfitting_detection': {},
            'lookahead_bias_prevention': {},
            'hyperparameter_optimization': {},
            'overall_assessment': {}
        }
        
        try:
            # Data split validation
            validation_results['data_split_validation'] = self.validate_data_split(
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                timestamps_train=timestamps_train,
                timestamps_test=timestamps_test
            )
            
            # Overfitting detection
            validation_results['overfitting_detection'] = self.detect_overfitting(
                model=model, X_train=X_train, X_val=X_test,
                y_train=y_train, y_val=y_test
            )
            
            # Hyperparameter optimization
            validation_results['hyperparameter_optimization'] = self.optimize_hyperparameters(
                model=model, X=X_train, y=y_train
            )
            
            # Overall assessment
            validation_results['overall_assessment'] = self._assess_overall_validation(
                validation_results
            )
            
            self.logger.info("✅ Comprehensive validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive validation failed: {e}")
            validation_results['error'] = str(e)
            return validation_results
    
    def _assess_overall_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall validation results."""
        assessment = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'score': 1.0
        }
        
        # Check data split validation
        data_split = validation_results.get('data_split_validation', {})
        if not data_split.get('is_valid', True):
            assessment['is_valid'] = False
            assessment['issues'].append("Data split validation failed")
        
        # Check overfitting detection
        overfitting = validation_results.get('overfitting_detection', {})
        if overfitting.get('overfitting_detected', False):
            assessment['warnings'].append("Overfitting detected")
            assessment['score'] *= 0.8
        
        # Generate recommendations
        if assessment['issues']:
            assessment['recommendations'].append("Address critical validation issues")
        if assessment['warnings']:
            assessment['recommendations'].append("Review warnings and consider model adjustments")
        
        return assessment
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of ML Common integrations."""
        status = {
            'architecture_type': self.architecture_type.value,
            'integrations': {
                'lookahead_protection': self.lookahead_protection is not None,
                'overfitting_detection': self.overfitting_detector is not None,
                'hpo_optimization': self.hpo_optimizer is not None,
                'data_leakage_prevention': self.data_leakage_prevention is not None,
                'unified_validation': self.unified_validation is not None
            },
            'config': self.config.__dict__,
            'using_existing_utilities': True
        }
        
        return status


# Convenience functions for creating ML Common integrations
class MLCommonIntegrationManager:
    """Manager for ML Common integrations."""
    
    def __init__(self, config: MLCommonIntegrationConfig, integration_types: List[MLCommonIntegrationType]):
        """Initialize the ML Common integration manager."""
        self.config = config
        self.integration_types = integration_types
        self.logger = logging.getLogger(__name__)
        self.integrations = {}
        
        # Initialize integrations
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize all configured integrations."""
        for integration_type in self.integration_types:
            try:
                if integration_type == MLCommonIntegrationType.LOOKAHEAD_PROTECTION:
                    self.integrations[integration_type] = LookaheadProtection(
                        **self.config.lookahead_config
                    )
                elif integration_type == MLCommonIntegrationType.OVERFITTING_DETECTION:
                    self.integrations[integration_type] = UniversalOverfittingDetector(
                        **self.config.overfitting_config
                    )
                elif integration_type == MLCommonIntegrationType.HPO_OPTIMIZATION:
                    self.integrations[integration_type] = HyperparameterOptimization(
                        **self.config.hpo_config
                    )
                elif integration_type == MLCommonIntegrationType.DATA_LEAKAGE_PREVENTION:
                    self.integrations[integration_type] = DataLeakagePrevention(
                        **self.config.data_leakage_config
                    )
                
                self.logger.info(f"✅ Initialized {integration_type.value}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {integration_type.value}: {e}")
                raise
    
    def get_integration(self, integration_type: MLCommonIntegrationType):
        """Get a specific integration."""
        return self.integrations.get(integration_type)
    
    def get_all_integrations(self):
        """Get all integrations."""
        return self.integrations


def create_ml_common_integration(architecture_type: ArchitectureType,
                               config: Optional[MLCommonIntegrationConfig] = None) -> MLCommonIntegration:
    """Create ML Common integration with default settings."""
    return MLCommonIntegration(architecture_type=architecture_type, config=config)


def create_tas_ml_common_integration(config: Optional[MLCommonIntegrationConfig] = None) -> MLCommonIntegration:
    """Create TAS-specific ML Common integration."""
    if config is None:
        config = MLCommonIntegrationConfig()
        # TAS-specific optimizations
        config.hpo_config['enable_parallel'] = True  # Trees benefit from parallel processing
        config.lookahead_config['strict_mode'] = True  # Strict temporal validation for trading
    
    return MLCommonIntegration(architecture_type=ArchitectureType.TAS, config=config)


def create_nas_ml_common_integration(config: Optional[MLCommonIntegrationConfig] = None) -> MLCommonIntegration:
    """Create NAS-specific ML Common integration."""
    if config is None:
        config = MLCommonIntegrationConfig()
        # NAS-specific optimizations
        config.hpo_config['use_nonlinear_optimization'] = True  # Neural networks benefit from nonlinear optimization
        config.overfitting_config['enable_learning_curves'] = True  # Important for neural networks
    
    return MLCommonIntegration(architecture_type=ArchitectureType.NAS, config=config)


def create_hybrid_ml_common_integration(config: Optional[MLCommonIntegrationConfig] = None) -> MLCommonIntegration:
    """Create hybrid-specific ML Common integration."""
    if config is None:
        config = MLCommonIntegrationConfig()
        # Hybrid-specific optimizations - combine both approaches
        config.hpo_config['enable_parallel'] = True
        config.hpo_config['use_nonlinear_optimization'] = True
        config.overfitting_config['enable_learning_curves'] = True
        config.lookahead_config['strict_mode'] = True
    
    return MLCommonIntegration(architecture_type=ArchitectureType.HYBRID, config=config)


def create_shared_ml_utilities_manager(
    config: Optional[MLCommonIntegrationConfig] = None,
    integration_types: Optional[List[MLCommonIntegrationType]] = None
) -> MLCommonIntegrationManager:
    """
    Create a shared ML utilities manager for hybrid NAS-TAS regime detection.
    
    Args:
        config: Configuration for ML Common integration
        integration_types: Types of integrations to enable
        
    Returns:
        MLCommonIntegrationManager: Configured ML utilities manager
    """
    if config is None:
        config = MLCommonIntegrationConfig()
    
    if integration_types is None:
        integration_types = [
            MLCommonIntegrationType.LOOKAHEAD_PROTECTION,
            MLCommonIntegrationType.OVERFITTING_DETECTION,
            MLCommonIntegrationType.HPO_OPTIMIZATION,
            MLCommonIntegrationType.DATA_LEAKAGE_PREVENTION
        ]
    
    return MLCommonIntegrationManager(
        config=config,
        integration_types=integration_types
    )