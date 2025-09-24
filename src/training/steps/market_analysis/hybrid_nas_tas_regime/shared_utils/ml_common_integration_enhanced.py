"""
Enhanced ML Common Integration for TAS and NAS

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

# Import existing ML Common utilities
from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
from src.utils.ml_common.validation.enhanced_overfitting_detection import EnhancedOverfittingDetector
from src.utils.ml_common.utils.hpo_utils import HyperparameterOptimization
from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention
from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem

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


class EnhancedMLCommonIntegration:
    """Enhanced ML Common integration using existing utilities."""
    
    def __init__(self, 
                 architecture_type: ArchitectureType,
                 config: MLCommonIntegrationConfig = None):
        """Initialize enhanced ML Common integration.
        
        Args:
            architecture_type: Type of architecture (TAS/NAS/Hybrid)
            config: Integration configuration
        """
        self.architecture_type = architecture_type
        self.config = config or MLCommonIntegrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ML Common utilities
        self._initialize_ml_common_utilities()
        
        self.logger.info(f"✅ Enhanced ML Common Integration initialized for {architecture_type.value}")
    
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
                self.overfitting_detector = EnhancedOverfittingDetector(
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
                self.unified_validation = UnifiedValidationSystem(
                    config=self.config.validation_config
                )
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
        """Validate data split using ML Common utilities.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            timestamps_train: Optional training timestamps
            timestamps_test: Optional test timestamps
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'lookahead_protection': {},
            'data_leakage_prevention': {},
            'overfitting_detection': {},
            'unified_validation': {}
        }
        
        try:
            # Lookahead Protection Validation
            if self.lookahead_protection and timestamps_train is not None and timestamps_test is not None:
                # Create DataFrames for lookahead protection
                train_df = pd.DataFrame(X_train)
                train_df['timestamp'] = timestamps_train
                train_df['target'] = y_train
                
                test_df = pd.DataFrame(X_test)
                test_df['timestamp'] = timestamps_test
                test_df['target'] = y_test
                
                # Validate temporal integrity
                lookahead_result = self.lookahead_protection.detect_data_leakage(
                    features_df=train_df,
                    target_df=test_df,
                    timestamp_col='timestamp'
                )
                validation_results['lookahead_protection'] = lookahead_result
                
                if lookahead_result.get('leakage_detected', False):
                    validation_results['is_valid'] = False
                    self.logger.warning("🚨 Lookahead bias detected in data split")
            
            # Data Leakage Prevention
            if self.data_leakage_prevention:
                leakage_result = self.data_leakage_prevention.detect_train_test_leakage(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    timestamps_train=timestamps_train,
                    timestamps_test=timestamps_test
                )
                validation_results['data_leakage_prevention'] = leakage_result
                
                if leakage_result.train_test_leakage_detected:
                    validation_results['is_valid'] = False
                    self.logger.warning("🚨 Data leakage detected between train/test sets")
            
            # Unified Validation
            if self.unified_validation:
                unified_result = self.unified_validation.validate_data_split(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test
                )
                validation_results['unified_validation'] = unified_result
                
                if not unified_result.get('is_valid', True):
                    validation_results['is_valid'] = False
                    self.logger.warning("🚨 Unified validation failed")
            
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
        """Optimize hyperparameters using ML Common HPO utilities.
        
        Args:
            model: Model to optimize
            X: Features
            y: Targets
            search_space: Optional search space definition
            optimization_objective: Optimization objective
            
        Returns:
            Optimization results
        """
        if not self.hpo_optimizer:
            return {'error': 'HPO optimization not enabled'}
        
        try:
            self.logger.info(f"🔧 Starting hyperparameter optimization for {self.architecture_type.value}")
            
            # Use existing HPO utilities
            optimization_result = self.hpo_optimizer.optimize_hyperparameters(
                model=model,
                X=X,
                y=y,
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
        """Detect overfitting using ML Common utilities.
        
        Args:
            model: Model to analyze
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Overfitting detection results
        """
        if not self.overfitting_detector:
            return {'error': 'Overfitting detection not enabled'}
        
        try:
            self.logger.info("🔍 Detecting overfitting patterns")
            
            # Use existing overfitting detection utilities
            overfitting_result = self.overfitting_detector.detect_overfitting(
                model=model,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val
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
        """Prevent lookahead bias using ML Common utilities.
        
        Args:
            data: Data to analyze
            timestamp_col: Timestamp column name
            target_col: Optional target column name
            
        Returns:
            Lookahead bias prevention results
        """
        if not self.lookahead_protection:
            return {'error': 'Lookahead protection not enabled'}
        
        try:
            self.logger.info("🔒 Preventing lookahead bias")
            
            # Use existing lookahead protection utilities
            if target_col:
                features_df = data.drop(columns=[target_col])
                target_df = data[[timestamp_col, target_col]]
                
                bias_result = self.lookahead_protection.detect_data_leakage(
                    features_df=features_df,
                    target_df=target_df,
                    timestamp_col=timestamp_col
                )
            else:
                bias_result = self.lookahead_protection.detect_data_leakage(
                    features_df=data,
                    target_df=data,
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
        """Perform comprehensive validation using all ML Common utilities.
        
        Args:
            model: Model to validate
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            timestamps_train: Optional training timestamps
            timestamps_test: Optional test timestamps
            
        Returns:
            Comprehensive validation results
        """
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
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                timestamps_train=timestamps_train,
                timestamps_test=timestamps_test
            )
            
            # Overfitting detection
            validation_results['overfitting_detection'] = self.detect_overfitting(
                model=model,
                X_train=X_train,
                X_val=X_test,
                y_train=y_train,
                y_val=y_test
            )
            
            # Hyperparameter optimization
            validation_results['hyperparameter_optimization'] = self.optimize_hyperparameters(
                model=model,
                X=X_train,
                y=y_train
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
        
        # Check lookahead bias
        lookahead = validation_results.get('lookahead_bias_prevention', {})
        if lookahead.get('leakage_detected', False):
            assessment['is_valid'] = False
            assessment['issues'].append("Lookahead bias detected")
        
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
            'config': self.config.__dict__
        }
        
        return status


# Convenience functions for creating enhanced ML Common integrations
def create_enhanced_ml_common_integration(architecture_type: ArchitectureType,
                                        config: Optional[MLCommonIntegrationConfig] = None) -> EnhancedMLCommonIntegration:
    """Create enhanced ML Common integration with default settings."""
    return EnhancedMLCommonIntegration(architecture_type=architecture_type, config=config)


def create_tas_ml_common_integration(config: Optional[MLCommonIntegrationConfig] = None) -> EnhancedMLCommonIntegration:
    """Create TAS-specific ML Common integration."""
    if config is None:
        config = MLCommonIntegrationConfig()
        # TAS-specific optimizations
        config.hpo_config['enable_parallel'] = True  # Trees benefit from parallel processing
        config.lookahead_config['strict_mode'] = True  # Strict temporal validation for trading
    
    return EnhancedMLCommonIntegration(architecture_type=ArchitectureType.TAS, config=config)


def create_nas_ml_common_integration(config: Optional[MLCommonIntegrationConfig] = None) -> EnhancedMLCommonIntegration:
    """Create NAS-specific ML Common integration."""
    if config is None:
        config = MLCommonIntegrationConfig()
        # NAS-specific optimizations
        config.hpo_config['use_nonlinear_optimization'] = True  # Neural networks benefit from nonlinear optimization
        config.overfitting_config['enable_learning_curves'] = True  # Important for neural networks
    
    return EnhancedMLCommonIntegration(architecture_type=ArchitectureType.NAS, config=config)


def create_hybrid_ml_common_integration(config: Optional[MLCommonIntegrationConfig] = None) -> EnhancedMLCommonIntegration:
    """Create hybrid-specific ML Common integration."""
    if config is None:
        config = MLCommonIntegrationConfig()
        # Hybrid-specific optimizations - combine both approaches
        config.hpo_config['enable_parallel'] = True
        config.hpo_config['use_nonlinear_optimization'] = True
        config.overfitting_config['enable_learning_curves'] = True
        config.lookahead_config['strict_mode'] = True
    
    return EnhancedMLCommonIntegration(architecture_type=ArchitectureType.HYBRID, config=config)