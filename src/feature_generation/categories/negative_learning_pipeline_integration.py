"""
Negative Learning Pipeline Integration

This module provides drop-in integration for the negative learning plugin
into existing Analyst/Tactician training pipelines.

Key Features:
    pass
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


            except Exception as e:
                pass
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
                                      pass
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
                                     pass
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

            except Exception as e:
                pass