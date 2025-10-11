"""
Negative Learning Integration Module

This module integrates the negative learning plugin into the existing Analyst/Tactician
pipelines with proper time-series safety and latency budget management.

Key Features:
- Analyst (1h) integration for HTF parent features
- Tactician (15m) integration for fast features
- Time-series safe feature generation
- Latency budget compliance
- Model constraint integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from src.utils.logger import system_logger
from src.feature_generation.categories.negative_learning import (

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
    NegativeLearningPlugin, 
    NegativeLearningConfig,
    FailureContextType
)


class AnalystNegativeLearningIntegration:
    """
    Integrates negative learning into the Analyst (1h) pipeline.
    Focuses on HTF parent features (trend/vol/anchor) for strategic decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('AnalystNegativeLearningIntegration')
        
        # Initialize negative learning plugin with Analyst-specific config
        nl_config = NegativeLearningConfig(
            max_negative_features=self.config.get('max_negative_features', 8),
            enable_gated_twins=True,
            enable_exception_interactions=True,
            enable_context_indicators=True,
            enable_monotone_constraints=True,
            enable_sample_weights=True
        )
        
        self.negative_learning_plugin = NegativeLearningPlugin(nl_config)
        self.is_fitted = False
        
        # Analyst-specific feature categories
        self.htf_parent_features = [
            'trend_strength', 'volatility_regime', 'volume_profile',
            'support_resistance_strength', 'momentum_htf', 'regime_probability'
        ]
    
    def fit_negative_learning(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        retrain_timestamp: Optional[datetime] = None
    ) -> 'AnalystNegativeLearningIntegration':
        """
        Fit negative learning on Analyst training data.
        Runs once per retrain cycle.
        
        Args:
            features_df: 1h timeframe feature matrix
            target: 1h target variable (returns)
            retrain_timestamp: Timestamp of retrain (for caching)
            
        Returns:
            Self for method chaining
        """
        self.logger.info("🎯 Fitting Analyst negative learning...")
        
        # Filter to HTF parent features
        available_htf_features = [f for f in self.htf_parent_features if f in features_df.columns]
        
        if not available_htf_features:
            self.logger.warning("No HTF parent features found for negative learning")
            return self
        
        # Fit the negative learning plugin
        self.negative_learning_plugin.fit(
            features_df, 
            target, 
            available_htf_features
        )
        
        self.is_fitted = True
        self.logger.info(f"✅ Analyst negative learning fitted with {len(available_htf_features)} HTF features")
        return self
    
    def generate_analyst_negative_features(
        self, 
        features_df: pd.DataFrame,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate negative learning features for Analyst inference.
        Time-series safe - no peeking past last HTF close.
        
        Args:
            features_df: 1h timeframe feature matrix
            inference_timestamp: Current inference timestamp
            
        Returns:
            Feature matrix with negative learning features
        """
        if not self.is_fitted:
            self.logger.warning("Negative learning not fitted, returning original features")
            return features_df
        
        self.logger.debug("🔄 Generating Analyst negative learning features...")
        
        # Ensure time-series safety
        if inference_timestamp:
            # Only use data up to the last complete HTF bar
            last_htf_close = inference_timestamp.replace(minute=0, second=0, microsecond=0)
            features_df = features_df[features_df.index <= last_htf_close]
        
        # Generate negative learning features
        enhanced_features = self.negative_learning_plugin.transform(features_df)
        
        self.logger.debug(f"✅ Generated {enhanced_features.shape[1] - features_df.shape[1]} negative learning features")
        return enhanced_features
    
    def get_analyst_monotone_constraints(self, feature_names: List[str]) -> List[int]:
        """Get monotone constraints for Analyst models"""
        return self.negative_learning_plugin.get_monotone_constraints(feature_names)
    
    def get_analyst_sample_weights(
        self, 
        features_df: pd.DataFrame, 
        base_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """Get sample weights for Analyst training"""
        return self.negative_learning_plugin.get_sample_weights(features_df, base_weights)
    
    def get_analyst_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for Analyst negative learning features"""
        return self.negative_learning_plugin.get_feature_importance_scores()


class TacticianNegativeLearningIntegration:
    """
    Integrates negative learning into the Tactician (15m) pipeline.
    Focuses on fast features (momentum, RVshort, VWAP_dist) for tactical decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('TacticianNegativeLearningIntegration')
        
        # Initialize negative learning plugin with Tactician-specific config
        nl_config = NegativeLearningConfig(
            max_negative_features=self.config.get('max_negative_features', 6),
            enable_gated_twins=True,
            enable_exception_interactions=True,
            enable_context_indicators=False,  # Lighter for Tactician
            enable_monotone_constraints=True,
            enable_sample_weights=True
        )
        
        self.negative_learning_plugin = NegativeLearningPlugin(nl_config)
        self.is_fitted = False
        
        # Tactician-specific feature categories
        self.fast_features = [
            'momentum_5m', 'momentum_15m', 'rsi_5m', 'rsi_15m',
            'vwap_distance', 'volume_weighted_price', 'volatility_short',
            'price_acceleration', 'order_flow_imbalance'
        ]
        
        # Analyst outputs to include
        self.analyst_outputs = [
            'p_trade', 'u_trade', 'confidence', 'regime_probability'
        ]
    
    def fit_negative_learning(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        retrain_timestamp: Optional[datetime] = None
    ) -> 'TacticianNegativeLearningIntegration':
        """
        Fit negative learning on Tactician training data.
        Includes Analyst outputs for context.
        
        Args:
            features_df: 15m timeframe feature matrix
            target: 15m target variable (returns)
            analyst_outputs: Analyst ensemble outputs
            retrain_timestamp: Timestamp of retrain
            
        Returns:
            Self for method chaining
        """
        self.logger.info("🎯 Fitting Tactician negative learning...")
        
        # Combine features with Analyst outputs
        combined_features = features_df.copy()
        if analyst_outputs is not None:
            # Ensure time alignment
            analyst_aligned = analyst_outputs.reindex(features_df.index, method='ffill')
            combined_features = pd.concat([combined_features, analyst_aligned], axis=1)
        
        # Filter to fast features and Analyst outputs
        candidate_features = []
        for feature_list in [self.fast_features, self.analyst_outputs]:
            candidate_features.extend([f for f in feature_list if f in combined_features.columns])
        
        if not candidate_features:
            self.logger.warning("No fast features found for negative learning")
            return self
        
        # Fit the negative learning plugin
        self.negative_learning_plugin.fit(
            combined_features, 
            target, 
            candidate_features
        )
        
        self.is_fitted = True
        self.logger.info(f"✅ Tactician negative learning fitted with {len(candidate_features)} features")
        return self
    
    def generate_tactician_negative_features(
        self, 
        features_df: pd.DataFrame,
        analyst_outputs: Optional[pd.DataFrame] = None,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate negative learning features for Tactician inference.
        Time-series safe - no peeking past last 15m close.
        
        Args:
            features_df: 15m timeframe feature matrix
            analyst_outputs: Current Analyst outputs
            inference_timestamp: Current inference timestamp
            
        Returns:
            Feature matrix with negative learning features
        """
        if not self.is_fitted:
            self.logger.warning("Negative learning not fitted, returning original features")
            return features_df
        
        self.logger.debug("🔄 Generating Tactician negative learning features...")
        
        # Ensure time-series safety
        if inference_timestamp:
            # Only use data up to the last complete 15m bar
            last_15m_close = inference_timestamp.replace(
                minute=(inference_timestamp.minute // 15) * 15, 
                second=0, 
                microsecond=0
            )
            features_df = features_df[features_df.index <= last_15m_close]
        
        # Combine with Analyst outputs
        combined_features = features_df.copy()
        if analyst_outputs is not None:
            analyst_aligned = analyst_outputs.reindex(features_df.index, method='ffill')
            combined_features = pd.concat([combined_features, analyst_aligned], axis=1)
        
        # Generate negative learning features
        enhanced_features = self.negative_learning_plugin.transform(combined_features)
        
        self.logger.debug(f"✅ Generated {enhanced_features.shape[1] - combined_features.shape[1]} negative learning features")
        return enhanced_features
    
    def get_tactician_monotone_constraints(self, feature_names: List[str]) -> List[int]:
        """Get monotone constraints for Tactician models"""
        return self.negative_learning_plugin.get_monotone_constraints(feature_names)
    
    def get_tactician_sample_weights(
        self, 
        features_df: pd.DataFrame, 
        base_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """Get sample weights for Tactician training"""
        return self.negative_learning_plugin.get_sample_weights(features_df, base_weights)
    
    def get_tactician_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for Tactician negative learning features"""
        return self.negative_learning_plugin.get_feature_importance_scores()


class NegativeLearningPipelineManager:
    """
    Manages the complete negative learning pipeline for both Analyst and Tactician.
    Handles coordination, caching, and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('NegativeLearningPipelineManager')
        
        # Initialize integrations
        self.analyst_integration = AnalystNegativeLearningIntegration(
            self.config.get('analyst', {})
        )
        self.tactician_integration = TacticianNegativeLearningIntegration(
            self.config.get('tactician', {})
        )
        
        # State tracking
        self.last_retrain_timestamp = None
        self.validation_results = {}
    
    def retrain_negative_learning(
        self,
        analyst_features: pd.DataFrame,
        analyst_target: pd.Series,
        tactician_features: pd.DataFrame,
        tactician_target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        retrain_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Retrain negative learning for both Analyst and Tactician.
        Called once per retrain cycle.
        
        Args:
            analyst_features: 1h Analyst features
            analyst_target: 1h Analyst target
            tactician_features: 15m Tactician features
            tactician_target: 15m Tactician target
            analyst_outputs: Analyst ensemble outputs for Tactician context
            retrain_timestamp: Retrain timestamp
            
        Returns:
            Retrain results and validation metrics
        """
        self.logger.info("🔄 Retraining negative learning pipeline...")
        
        retrain_results = {
            'analyst': {},
            'tactician': {},
            'validation': {}
        }
        
        # Retrain Analyst negative learning
        try:
            self.analyst_integration.fit_negative_learning(
                analyst_features, 
                analyst_target, 
                retrain_timestamp
            )
            retrain_results['analyst']['status'] = 'success'
            retrain_results['analyst']['failure_contexts'] = len(
                self.analyst_integration.negative_learning_plugin.get_failure_contexts()
            )
        except Exception as e:
            self.logger.error(f"Analyst negative learning retrain failed: {e}")
            retrain_results['analyst']['status'] = 'failed'
            retrain_results['analyst']['error'] = str(e)
        
        # Retrain Tactician negative learning
        try:
            self.tactician_integration.fit_negative_learning(
                tactician_features, 
                tactician_target, 
                analyst_outputs, 
                retrain_timestamp
            )
            retrain_results['tactician']['status'] = 'success'
            retrain_results['tactician']['failure_contexts'] = len(
                self.tactician_integration.negative_learning_plugin.get_failure_contexts()
            )
        except Exception as e:
            self.logger.error(f"Tactician negative learning retrain failed: {e}")
            retrain_results['tactician']['status'] = 'failed'
            retrain_results['tactician']['error'] = str(e)
        
        # Validate both pipelines
        try:
            self.validation_results = self._validate_pipelines(
                analyst_features, analyst_target,
                tactician_features, tactician_target,
                analyst_outputs
            )
            retrain_results['validation'] = self.validation_results
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            retrain_results['validation']['error'] = str(e)
        
        self.last_retrain_timestamp = retrain_timestamp or datetime.now()
        self.logger.info("✅ Negative learning pipeline retrain complete")
        
        return retrain_results
    
    def _validate_pipelines(
        self,
        analyst_features: pd.DataFrame,
        analyst_target: pd.Series,
        tactician_features: pd.DataFrame,
        tactician_target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Validate both Analyst and Tactician negative learning pipelines"""
        validation_results = {}
        
        # Validate Analyst pipeline
        if self.analyst_integration.is_fitted:
            analyst_enhanced = self.analyst_integration.generate_analyst_negative_features(
                analyst_features
            )
            analyst_validation = self.analyst_integration.negative_learning_plugin.validate(
                analyst_enhanced, analyst_target
            )
            validation_results['analyst'] = analyst_validation
        
        # Validate Tactician pipeline
        if self.tactician_integration.is_fitted:
            tactician_enhanced = self.tactician_integration.generate_tactician_negative_features(
                tactician_features, analyst_outputs
            )
            tactician_validation = self.tactician_integration.negative_learning_plugin.validate(
                tactician_enhanced, tactician_target
            )
            validation_results['tactician'] = tactician_validation
        
        return validation_results
    
    def get_analyst_features(
        self, 
        features_df: pd.DataFrame,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get enhanced Analyst features with negative learning"""
        return self.analyst_integration.generate_analyst_negative_features(
            features_df, inference_timestamp
        )
    
    def get_tactician_features(
        self, 
        features_df: pd.DataFrame,
        analyst_outputs: Optional[pd.DataFrame] = None,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get enhanced Tactician features with negative learning"""
        return self.tactician_integration.generate_tactician_negative_features(
            features_df, analyst_outputs, inference_timestamp
        )
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations with negative learning constraints"""
        return {
            'analyst': {
                'monotone_constraints': self.analyst_integration.get_analyst_monotone_constraints,
                'sample_weights': self.analyst_integration.get_analyst_sample_weights,
                'feature_importance': self.analyst_integration.get_analyst_feature_importance
            },
            'tactician': {
                'monotone_constraints': self.tactician_integration.get_tactician_monotone_constraints,
                'sample_weights': self.tactician_integration.get_tactician_sample_weights,
                'feature_importance': self.tactician_integration.get_tactician_feature_importance
            }
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics"""
        return {
            'last_retrain': self.last_retrain_timestamp,
            'analyst_fitted': self.analyst_integration.is_fitted,
            'tactician_fitted': self.tactician_integration.is_fitted,
            'validation_results': self.validation_results,
            'analyst_failure_contexts': len(
                self.analyst_integration.negative_learning_plugin.get_failure_contexts()
            ),
            'tactician_failure_contexts': len(
                self.tactician_integration.negative_learning_plugin.get_failure_contexts()
            )
        }


# Convenience functions for easy integration
def create_negative_learning_pipeline(config: Optional[Dict[str, Any]] = None) -> NegativeLearningPipelineManager:
    """Create a new negative learning pipeline manager"""
    return NegativeLearningPipelineManager(config)


def get_negative_learning_config() -> Dict[str, Any]:
    """Get default negative learning configuration"""
    return {
        'analyst': {
            'max_negative_features': 8,
            'enable_gated_twins': True,
            'enable_exception_interactions': True,
            'enable_context_indicators': True
        },
        'tactician': {
            'max_negative_features': 6,
            'enable_gated_twins': True,
            'enable_exception_interactions': True,
            'enable_context_indicators': False
        },
        'validation': {
            'stability_selection_bootstrap': 80,
            'stability_selection_threshold': 0.6,
            'min_ic_improvement': 0.10
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
