"""
Negative Learning Pipeline Integration

This module provides drop-in integration for the negative learning plugin
into existing Analyst/Tactician training pipelines.

Key Features:
- Drop-in integration with minimal code changes
- Time-series safe feature generation
- Automatic constraint application
- Performance monitoring
- Backward compatibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
import warnings

from src.utils.logger import system_logger
from src.feature_generation.categories.negative_learning_integration import (
    NegativeLearningPipelineManager,
    create_negative_learning_pipeline
)
from src.feature_generation.categories.negative_learning_selection import (
    NegativeLearningFeatureSelector,
    create_feature_selector
)
from src.feature_generation.categories.negative_learning_constraints import (
    ModelConstraintManager,
    create_constraint_manager,
    ModelType
)
from src.feature_generation.categories.negative_learning_validation import (

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
    NegativeLearningValidator,
    create_negative_learning_validator
)


class AnalystNegativeLearningIntegration:
    """
    Drop-in integration for Analyst pipeline with negative learning.
    Minimal code changes required for existing Analyst training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('AnalystNegativeLearningIntegration')
        
        # Initialize negative learning components
        self.pipeline_manager = create_negative_learning_pipeline(
            self.config.get('negative_learning', {})
        )
        self.feature_selector = create_feature_selector(
            self.config.get('feature_selection', {})
        )
        self.constraint_manager = create_constraint_manager(
            self.config.get('constraints', {})
        )
        self.validator = create_negative_learning_validator(
            self.config.get('validation', {})
        )
        
        # State
        self.is_initialized = False
        self.last_retrain_timestamp = None
        self.negative_features = []
        self.model_constraints = None
    
    def initialize_negative_learning(
        self,
        analyst_features: pd.DataFrame,
        analyst_target: pd.Series,
        retrain_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Initialize negative learning for Analyst pipeline.
        Call this once per retrain cycle.
        
        Args:
            analyst_features: 1h Analyst features
            analyst_target: 1h Analyst target
            retrain_timestamp: Retrain timestamp
            
        Returns:
            Initialization results
        """
        self.logger.info("🎯 Initializing Analyst negative learning...")
        
        try:
            # Retrain negative learning pipeline
            retrain_results = self.pipeline_manager.retrain_negative_learning(
                analyst_features=analyst_features,
                analyst_target=analyst_target,
                tactician_features=pd.DataFrame(),  # Not needed for Analyst
                tactician_target=pd.Series(dtype=float),
                retrain_timestamp=retrain_timestamp
            )
            
            # Get enhanced features to identify negative learning features
            enhanced_features = self.pipeline_manager.get_analyst_features(analyst_features)
            self.negative_features = [
                col for col in enhanced_features.columns 
                if col not in analyst_features.columns
            ]
            
            # Generate model constraints
            self.model_constraints = self.constraint_manager.generate_constraints(
                features_df=enhanced_features,
                feature_names=enhanced_features.columns.tolist(),
                negative_learning_features=self.negative_features,
                failure_contexts=self.pipeline_manager.analyst_integration.negative_learning_plugin.get_failure_contexts(),
                model_type=ModelType.LIGHTGBM,
                target=analyst_target
            )
            
            self.is_initialized = True
            self.last_retrain_timestamp = retrain_timestamp or datetime.now()
            
            self.logger.info(f"✅ Analyst negative learning initialized with {len(self.negative_features)} negative features")
            
            return {
                'status': 'success',
                'negative_features': self.negative_features,
                'retrain_results': retrain_results,
                'model_constraints': self.model_constraints
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Analyst negative learning: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def enhance_analyst_features(
        self,
        features_df: pd.DataFrame,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Enhance Analyst features with negative learning.
        Call this for each inference.
        
        Args:
            features_df: 1h Analyst features
            inference_timestamp: Current inference timestamp
            
        Returns:
            Enhanced features with negative learning
        """
        if not self.is_initialized:
            self.logger.warning("Negative learning not initialized, returning original features")
            return features_df
        
        try:
            enhanced_features = self.pipeline_manager.get_analyst_features(
                features_df, inference_timestamp
            )
            
            self.logger.debug(f"Enhanced Analyst features: {features_df.shape[1]} -> {enhanced_features.shape[1]}")
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Failed to enhance Analyst features: {e}")
            return features_df
    
    def get_analyst_model_config(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        model_type: str = 'lightgbm'
    ) -> Dict[str, Any]:
        """
        Get Analyst model configuration with negative learning constraints.
        
        Args:
            base_config: Base model configuration
            model_type: Model type ('lightgbm', 'xgboost', 'catboost')
            
        Returns:
            Enhanced model configuration
        """
        if not self.is_initialized or not self.model_constraints:
            return base_config or {}
        
        config = base_config or {}
        
        # Add monotone constraints
        if self.model_constraints.monotone_constraints:
            if model_type == 'lightgbm':
                config['monotone_constraints'] = self.model_constraints.monotone_constraints
            elif model_type == 'catboost':
                config['monotone_constraints'] = self.model_constraints.monotone_constraints
        
        # Add feature constraints
        if self.model_constraints.feature_caps:
            config['feature_caps'] = self.model_constraints.feature_caps
        
        return config
    
    def get_analyst_sample_weights(
        self,
        features_df: pd.DataFrame,
        base_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get Analyst sample weights with uncertainty weighting.
        
        Args:
            features_df: Feature matrix
            base_weights: Base sample weights
            
        Returns:
            Enhanced sample weights
        """
        if not self.is_initialized or not self.model_constraints:
            return base_weights or pd.Series(1.0, index=features_df.index)
        
        return self.model_constraints.sample_weights or (base_weights or pd.Series(1.0, index=features_df.index))
    
    def validate_analyst_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Validate Analyst negative learning performance.
        
        Args:
            features_df: Feature matrix
            target: Target variable
            
        Returns:
            Validation results
        """
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        try:
            enhanced_features = self.enhance_analyst_features(features_df)
            
            validation_results = self.validator.validate_negative_learning(
                features_df=enhanced_features,
                target=target,
                negative_features=self.negative_features,
                failure_contexts=self.pipeline_manager.analyst_integration.negative_learning_plugin.get_failure_contexts()
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate Analyst performance: {e}")
            return {'status': 'failed', 'error': str(e)}


class TacticianNegativeLearningIntegration:
    """
    Drop-in integration for Tactician pipeline with negative learning.
    Minimal code changes required for existing Tactician training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('TacticianNegativeLearningIntegration')
        
        # Initialize negative learning components
        self.pipeline_manager = create_negative_learning_pipeline(
            self.config.get('negative_learning', {})
        )
        self.feature_selector = create_feature_selector(
            self.config.get('feature_selection', {})
        )
        self.constraint_manager = create_constraint_manager(
            self.config.get('constraints', {})
        )
        self.validator = create_negative_learning_validator(
            self.config.get('validation', {})
        )
        
        # State
        self.is_initialized = False
        self.last_retrain_timestamp = None
        self.negative_features = []
        self.model_constraints = None
    
    def initialize_negative_learning(
        self,
        tactician_features: pd.DataFrame,
        tactician_target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        retrain_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Initialize negative learning for Tactician pipeline.
        Call this once per retrain cycle.
        
        Args:
            tactician_features: 15m Tactician features
            tactician_target: 15m Tactician target
            analyst_outputs: Analyst ensemble outputs
            retrain_timestamp: Retrain timestamp
            
        Returns:
            Initialization results
        """
        self.logger.info("🎯 Initializing Tactician negative learning...")
        
        try:
            # Retrain negative learning pipeline
            retrain_results = self.pipeline_manager.retrain_negative_learning(
                analyst_features=pd.DataFrame(),  # Not needed for Tactician
                analyst_target=pd.Series(dtype=float),
                tactician_features=tactician_features,
                tactician_target=tactician_target,
                analyst_outputs=analyst_outputs,
                retrain_timestamp=retrain_timestamp
            )
            
            # Get enhanced features to identify negative learning features
            enhanced_features = self.pipeline_manager.get_tactician_features(
                tactician_features, analyst_outputs
            )
            self.negative_features = [
                col for col in enhanced_features.columns 
                if col not in tactician_features.columns
            ]
            
            # Generate model constraints
            self.model_constraints = self.constraint_manager.generate_constraints(
                features_df=enhanced_features,
                feature_names=enhanced_features.columns.tolist(),
                negative_learning_features=self.negative_features,
                failure_contexts=self.pipeline_manager.tactician_integration.negative_learning_plugin.get_failure_contexts(),
                model_type=ModelType.LIGHTGBM,
                target=tactician_target
            )
            
            self.is_initialized = True
            self.last_retrain_timestamp = retrain_timestamp or datetime.now()
            
            self.logger.info(f"✅ Tactician negative learning initialized with {len(self.negative_features)} negative features")
            
            return {
                'status': 'success',
                'negative_features': self.negative_features,
                'retrain_results': retrain_results,
                'model_constraints': self.model_constraints
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Tactician negative learning: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def enhance_tactician_features(
        self,
        features_df: pd.DataFrame,
        analyst_outputs: Optional[pd.DataFrame] = None,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Enhance Tactician features with negative learning.
        Call this for each inference.
        
        Args:
            features_df: 15m Tactician features
            analyst_outputs: Current Analyst outputs
            inference_timestamp: Current inference timestamp
            
        Returns:
            Enhanced features with negative learning
        """
        if not self.is_initialized:
            self.logger.warning("Negative learning not initialized, returning original features")
            return features_df
        
        try:
            enhanced_features = self.pipeline_manager.get_tactician_features(
                features_df, analyst_outputs, inference_timestamp
            )
            
            self.logger.debug(f"Enhanced Tactician features: {features_df.shape[1]} -> {enhanced_features.shape[1]}")
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Failed to enhance Tactician features: {e}")
            return features_df
    
    def get_tactician_model_config(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        model_type: str = 'lightgbm'
    ) -> Dict[str, Any]:
        """
        Get Tactician model configuration with negative learning constraints.
        
        Args:
            base_config: Base model configuration
            model_type: Model type ('lightgbm', 'xgboost', 'catboost')
            
        Returns:
            Enhanced model configuration
        """
        if not self.is_initialized or not self.model_constraints:
            return base_config or {}
        
        config = base_config or {}
        
        # Add monotone constraints
        if self.model_constraints.monotone_constraints:
            if model_type == 'lightgbm':
                config['monotone_constraints'] = self.model_constraints.monotone_constraints
            elif model_type == 'catboost':
                config['monotone_constraints'] = self.model_constraints.monotone_constraints
        
        # Add feature constraints
        if self.model_constraints.feature_caps:
            config['feature_caps'] = self.model_constraints.feature_caps
        
        return config
    
    def get_tactician_sample_weights(
        self,
        features_df: pd.DataFrame,
        base_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get Tactician sample weights with uncertainty weighting.
        
        Args:
            features_df: Feature matrix
            base_weights: Base sample weights
            
        Returns:
            Enhanced sample weights
        """
        if not self.is_initialized or not self.model_constraints:
            return base_weights or pd.Series(1.0, index=features_df.index)
        
        return self.model_constraints.sample_weights or (base_weights or pd.Series(1.0, index=features_df.index))
    
    def validate_tactician_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Validate Tactician negative learning performance.
        
        Args:
            features_df: Feature matrix
            target: Target variable
            analyst_outputs: Analyst outputs
            
        Returns:
            Validation results
        """
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        try:
            enhanced_features = self.enhance_tactician_features(features_df, analyst_outputs)
            
            validation_results = self.validator.validate_negative_learning(
                features_df=enhanced_features,
                target=target,
                negative_features=self.negative_features,
                failure_contexts=self.pipeline_manager.tactician_integration.negative_learning_plugin.get_failure_contexts()
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate Tactician performance: {e}")
            return {'status': 'failed', 'error': str(e)}


class NegativeLearningPipelineIntegrator:
    """
    Main integrator that provides drop-in integration for both Analyst and Tactician pipelines.
    Handles coordination and provides unified interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('NegativeLearningPipelineIntegrator')
        
        # Initialize integrations
        self.analyst_integration = AnalystNegativeLearningIntegration(
            self.config.get('analyst', {})
        )
        self.tactician_integration = TacticianNegativeLearningIntegration(
            self.config.get('tactician', {})
        )
        
        # State
        self.is_initialized = False
        self.initialization_results = {}
    
    def initialize_negative_learning(
        self,
        analyst_features: pd.DataFrame,
        analyst_target: pd.Series,
        tactician_features: pd.DataFrame,
        tactician_target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        retrain_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Initialize negative learning for both Analyst and Tactician pipelines.
        
        Args:
            analyst_features: 1h Analyst features
            analyst_target: 1h Analyst target
            tactician_features: 15m Tactician features
            tactician_target: 15m Tactician target
            analyst_outputs: Analyst ensemble outputs
            retrain_timestamp: Retrain timestamp
            
        Returns:
            Initialization results for both pipelines
        """
        self.logger.info("🎯 Initializing negative learning for both pipelines...")
        
        # Initialize Analyst
        analyst_results = self.analyst_integration.initialize_negative_learning(
            analyst_features, analyst_target, retrain_timestamp
        )
        
        # Initialize Tactician
        tactician_results = self.tactician_integration.initialize_negative_learning(
            tactician_features, tactician_target, analyst_outputs, retrain_timestamp
        )
        
        self.initialization_results = {
            'analyst': analyst_results,
            'tactician': tactician_results,
            'timestamp': retrain_timestamp or datetime.now()
        }
        
        self.is_initialized = (
            analyst_results.get('status') == 'success' and
            tactician_results.get('status') == 'success'
        )
        
        if self.is_initialized:
            self.logger.info("✅ Negative learning initialized for both pipelines")
        else:
            self.logger.warning("⚠️ Negative learning initialization had issues")
        
        return self.initialization_results
    
    def get_enhanced_features(
        self,
        analyst_features: pd.DataFrame,
        tactician_features: pd.DataFrame,
        analyst_outputs: Optional[pd.DataFrame] = None,
        inference_timestamp: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get enhanced features for both Analyst and Tactician.
        
        Args:
            analyst_features: 1h Analyst features
            tactician_features: 15m Tactician features
            analyst_outputs: Current Analyst outputs
            inference_timestamp: Current inference timestamp
            
        Returns:
            Tuple of (enhanced_analyst_features, enhanced_tactician_features)
        """
        if not self.is_initialized:
            self.logger.warning("Negative learning not initialized, returning original features")
            return analyst_features, tactician_features
        
        # Enhance Analyst features
        enhanced_analyst = self.analyst_integration.enhance_analyst_features(
            analyst_features, inference_timestamp
        )
        
        # Enhance Tactician features
        enhanced_tactician = self.tactician_integration.enhance_tactician_features(
            tactician_features, analyst_outputs, inference_timestamp
        )
        
        return enhanced_analyst, enhanced_tactician
    
    def get_model_configs(
        self,
        analyst_base_config: Optional[Dict[str, Any]] = None,
        tactician_base_config: Optional[Dict[str, Any]] = None,
        analyst_model_type: str = 'lightgbm',
        tactician_model_type: str = 'lightgbm'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations for both Analyst and Tactician.
        
        Args:
            analyst_base_config: Base Analyst configuration
            tactician_base_config: Base Tactician configuration
            analyst_model_type: Analyst model type
            tactician_model_type: Tactician model type
            
        Returns:
            Dictionary with configurations for both pipelines
        """
        return {
            'analyst': self.analyst_integration.get_analyst_model_config(
                analyst_base_config, analyst_model_type
            ),
            'tactician': self.tactician_integration.get_tactician_model_config(
                tactician_base_config, tactician_model_type
            )
        }
    
    def get_sample_weights(
        self,
        analyst_features: pd.DataFrame,
        tactician_features: pd.DataFrame,
        analyst_base_weights: Optional[pd.Series] = None,
        tactician_base_weights: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Get sample weights for both Analyst and Tactician.
        
        Args:
            analyst_features: 1h Analyst features
            tactician_features: 15m Tactician features
            analyst_base_weights: Base Analyst weights
            tactician_base_weights: Base Tactician weights
            
        Returns:
            Tuple of (analyst_weights, tactician_weights)
        """
        analyst_weights = self.analyst_integration.get_analyst_sample_weights(
            analyst_features, analyst_base_weights
        )
        
        tactician_weights = self.tactician_integration.get_tactician_sample_weights(
            tactician_features, tactician_base_weights
        )
        
        return analyst_weights, tactician_weights
    
    def validate_performance(
        self,
        analyst_features: pd.DataFrame,
        analyst_target: pd.Series,
        tactician_features: pd.DataFrame,
        tactician_target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Validate performance for both pipelines.
        
        Args:
            analyst_features: 1h Analyst features
            analyst_target: 1h Analyst target
            tactician_features: 15m Tactician features
            tactician_target: 15m Tactician target
            analyst_outputs: Analyst outputs
            
        Returns:
            Validation results for both pipelines
        """
        analyst_validation = self.analyst_integration.validate_analyst_performance(
            analyst_features, analyst_target
        )
        
        tactician_validation = self.tactician_integration.validate_tactician_performance(
            tactician_features, tactician_target, analyst_outputs
        )
        
        return {
            'analyst': analyst_validation,
            'tactician': tactician_validation
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'is_initialized': self.is_initialized,
            'initialization_results': self.initialization_results,
            'analyst_negative_features': len(self.analyst_integration.negative_features),
            'tactician_negative_features': len(self.tactician_integration.negative_features),
            'last_retrain': self.analyst_integration.last_retrain_timestamp
        }


# Convenience functions for easy integration
def create_negative_learning_integrator(config: Optional[Dict[str, Any]] = None) -> NegativeLearningPipelineIntegrator:
    """Create a new negative learning pipeline integrator"""
    return NegativeLearningPipelineIntegrator(config)


def get_integration_config() -> Dict[str, Any]:
    """Get default integration configuration"""
    return {
        'analyst': {
            'negative_learning': {
                'max_negative_features': 8,
                'enable_gated_twins': True,
                'enable_exception_interactions': True,
                'enable_context_indicators': True
            },
            'feature_selection': {
                'stability_threshold': 0.6,
                'min_ic_improvement': 0.10
            },
            'constraints': {
                'enable_monotone_constraints': True,
                'enable_sample_weights': True,
                'weight_uncertainty_factor': 0.3
            }
        },
        'tactician': {
            'negative_learning': {
                'max_negative_features': 6,
                'enable_gated_twins': True,
                'enable_exception_interactions': True,
                'enable_context_indicators': False
            },
            'feature_selection': {
                'stability_threshold': 0.6,
                'min_ic_improvement': 0.10
            },
            'constraints': {
                'enable_monotone_constraints': True,
                'enable_sample_weights': True,
                'weight_uncertainty_factor': 0.3
            }
        }
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
