"""
Shared Feature Engineering Module

This module provides shared feature engineering utilities that ensure
consistency between training and inference (signal generation).

Features engineered here match exactly what is done during model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum


class TrainingRole(Enum):
    """Training role types."""
    ANALYST = "analyst"
    TACTICIAN = "tactician"


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering operation."""
    data: pd.DataFrame
    engineered_features: List[str]
    warnings: List[str]
    errors: List[str]


class FeatureEngineer:
    """Base class for feature engineering."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.engineered_feature_names: List[str] = []
    
    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features. Override in subclasses.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with engineered features added
        """
        return data
    
    def get_engineered_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        return self.engineered_feature_names.copy()
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
        """Safely divide two series, handling division by zero."""
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default)


class AnalystFeatureEngineer(FeatureEngineer):
    """
    Feature engineer for Analyst role.
    
    Base features come from feature_generation_final_feature_selection_step (300+ features selected down to 40/50/60).
    This module ONLY adds regime confidence features for each regime (4 regimes total).
    
    Engineered features:
    - regime_confidence_0, regime_confidence_1, regime_confidence_2, regime_confidence_3
    
    Source: regime_ensemble_training ML model outputs (regime_prob_0, regime_prob_1, regime_prob_2, regime_prob_3)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.engineered_feature_names = [
            'regime_confidence_0',
            'regime_confidence_1',
            'regime_confidence_2',
            'regime_confidence_3'
        ]
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        regime_probabilities: Optional[Dict[int, Union[pd.Series, np.ndarray, float]]] = None,
        allow_uniform_defaults: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Engineer features specific to Analyst role.
        
        ONLY adds regime confidence features - all other features come from feature_generation_final_feature_selection_step.
        
        Args:
            data: Input DataFrame with base features from feature_generation_final_feature_selection_step
            regime_probabilities: Dict mapping regime index to probability values
                                 e.g., {0: 0.7, 1: 0.2, 2: 0.05, 3: 0.05}
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with regime confidence features added
        """
        try:
            result_data = data.copy()
            warnings = []
            
            # Add regime confidence for each of the 4 regimes
            if regime_probabilities is not None:
                # Use provided regime probabilities dictionary
                for regime_idx in range(4):
                    if regime_idx in regime_probabilities:
                        prob = regime_probabilities[regime_idx]
                        
                        # Convert to series if needed
                        if isinstance(prob, (float, int)):
                            prob_series = pd.Series([prob] * len(result_data), index=result_data.index)
                        elif isinstance(prob, pd.Series):
                            prob_series = prob
                        else:
                            prob_array = np.asarray(prob)
                            prob_series = pd.Series(prob_array, index=result_data.index[:len(prob_array)])
                        
                        # Regime confidence is the probability itself (already 0-1)
                        result_data[f'regime_confidence_{regime_idx}'] = prob_series
                    else:
                        warnings.append(f"Regime {regime_idx} probability not provided, using default 0.25")
                        result_data[f'regime_confidence_{regime_idx}'] = 0.25
            
            # Try to extract from existing columns (regime_prob_0, regime_prob_1, etc.)
            elif any(f'regime_prob_{i}' in result_data.columns for i in range(4)):
                for regime_idx in range(4):
                    prob_col = f'regime_prob_{regime_idx}'
                    if prob_col in result_data.columns:
                        result_data[f'regime_confidence_{regime_idx}'] = result_data[prob_col]
                    else:
                        warnings.append(f"{prob_col} not found in data, using default 0.25")
                        result_data[f'regime_confidence_{regime_idx}'] = 0.25
            
            elif allow_uniform_defaults:
                warnings.append("No regime probabilities provided or found in data, using uniform distribution")
                for regime_idx in range(4):
                    result_data[f'regime_confidence_{regime_idx}'] = 0.25
            else:
                pass
            
            # Fill NaN values
            result_data = result_data.fillna(method='bfill').fillna(0.25)
            
            self.logger.info(f"Engineered {len(self.engineered_feature_names)} features for Analyst. Total columns: {len(result_data.columns)}")
            
            if warnings:
                self.logger.warning(f"Feature engineering warnings: {warnings}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Analyst feature engineering failed: {e}", exc_info=True)
            return data
    
    def engineer_features_from_regime_dict(
        self,
        data: pd.DataFrame,
        regime_probabilities: Optional[Dict[Any, float]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Engineer features from regime probabilities dictionary.
        
        Args:
            data: Input DataFrame
            regime_probabilities: Dictionary of regime probabilities (e.g., {RegimeType.TRENDING: 0.7})
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with engineered features
        """
        # Extract primary regime probability (highest value)
        regime_prob = None
        if regime_probabilities:
            regime_prob = max(regime_probabilities.values())
        
        return self.engineer_features(data, regime_probability=regime_prob, **kwargs)


class TacticianFeatureEngineer(FeatureEngineer):
    """
    Feature engineer for Tactician role.
    
    Base features come from feature_generation_final_feature_selection_step (300+ features selected down to 40/50/60).
    This module ONLY adds:
    - Regime confidence features for each regime (4 regimes total)
    - Analyst signal strength (single aggregated value from analyst models)
    
    Engineered features:
    - regime_confidence_0, regime_confidence_1, regime_confidence_2, regime_confidence_3
    - analyst_signal_strength
    
    Sources:
    - Regime features: regime_ensemble_training ML model outputs (regime_prob_0-3)
    - Analyst signal: analyst_ensemble_models predictions
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.engineered_feature_names = [
            'regime_confidence_0',
            'regime_confidence_1',
            'regime_confidence_2',
            'regime_confidence_3',
            'analyst_signal_strength',
        ]
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        regime_probabilities: Optional[Dict[int, Union[pd.Series, np.ndarray, float]]] = None,
        analyst_signal_strength: Optional[Union[pd.Series, np.ndarray, float]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Engineer features specific to Tactician role.
        
        ONLY adds regime confidence + analyst signal strength - all other features come from feature_generation_final_feature_selection_step.
        
        Args:
            data: Input DataFrame with base features from feature_generation_final_feature_selection_step
            regime_probabilities: Dict mapping regime index to probability values
                                 e.g., {0: 0.7, 1: 0.2, 2: 0.05, 3: 0.05}
            analyst_signal_strength: Aggregated signal strength from analyst ensemble models
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with engineered features added
        """
        try:
            result_data = data.copy()
            warnings = []
            
            # 1. Add regime confidence for each of the 4 regimes (same as Analyst)
            if regime_probabilities is not None:
                for regime_idx in range(4):
                    if regime_idx in regime_probabilities:
                        prob = regime_probabilities[regime_idx]
                        
                        # Convert to series if needed
                        if isinstance(prob, (float, int)):
                            prob_series = pd.Series([prob] * len(result_data), index=result_data.index)
                        elif isinstance(prob, pd.Series):
                            prob_series = prob
                        else:
                            prob_array = np.asarray(prob)
                            prob_series = pd.Series(prob_array, index=result_data.index[:len(prob_array)])
                        
                        result_data[f'regime_confidence_{regime_idx}'] = prob_series
                    else:
                        warnings.append(f"Regime {regime_idx} probability not provided, using default 0.25")
                        result_data[f'regime_confidence_{regime_idx}'] = 0.25
            
            # Try to extract from existing columns
            elif any(f'regime_prob_{i}' in result_data.columns for i in range(4)):
                for regime_idx in range(4):
                    prob_col = f'regime_prob_{regime_idx}'
                    if prob_col in result_data.columns:
                        result_data[f'regime_confidence_{regime_idx}'] = result_data[prob_col]
                    else:
                        warnings.append(f"{prob_col} not found in data, using default 0.25")
                        result_data[f'regime_confidence_{regime_idx}'] = 0.25
            else:
                warnings.append("No regime probabilities provided or found in data, using uniform distribution")
                for regime_idx in range(4):
                    result_data[f'regime_confidence_{regime_idx}'] = 0.25
            
            # 2. Add analyst signal strength
            if analyst_signal_strength is not None:
                # Convert to series if needed
                if isinstance(analyst_signal_strength, (float, int)):
                    result_data['analyst_signal_strength'] = analyst_signal_strength
                elif isinstance(analyst_signal_strength, pd.Series):
                    result_data['analyst_signal_strength'] = analyst_signal_strength
                else:
                    signal_array = np.asarray(analyst_signal_strength)
                    result_data['analyst_signal_strength'] = pd.Series(signal_array, index=result_data.index[:len(signal_array)])
            
            # Try to extract from existing analyst columns
            elif 'analyst_ensemble_predictions' in result_data.columns:
                result_data['analyst_signal_strength'] = result_data['analyst_ensemble_predictions']
            elif any('analyst' in col.lower() for col in result_data.columns):
                analyst_cols = [col for col in result_data.columns if 'analyst' in col.lower()]
                result_data['analyst_signal_strength'] = result_data[analyst_cols].mean(axis=1)
            else:
                warnings.append("No analyst signal strength provided or found in data, using default 0.0")
                result_data['analyst_signal_strength'] = 0.0
            
            # Fill NaN values
            result_data = result_data.fillna(method='bfill').fillna(0)
            
            self.logger.info(f"Engineered {len(self.engineered_feature_names)} features for Tactician. Total columns: {len(result_data.columns)}")
            
            if warnings:
                self.logger.warning(f"Feature engineering warnings: {warnings}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Tactician feature engineering failed: {e}", exc_info=True)
            return data


# Convenience functions
def engineer_analyst_features(
    data: pd.DataFrame,
    regime_probabilities: Optional[Dict[int, Union[pd.Series, np.ndarray, float]]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function to engineer Analyst features.
    
    Args:
        data: Input DataFrame with base features from feature_generation_final_feature_selection_step
        regime_probabilities: Dict mapping regime index (0-3) to probability values
        logger: Optional logger
        
    Returns:
        DataFrame with regime confidence features added
    """
    engineer = AnalystFeatureEngineer(logger=logger)
    return engineer.engineer_features(data, regime_probabilities=regime_probabilities)


def engineer_tactician_features(
    data: pd.DataFrame,
    regime_probabilities: Optional[Dict[int, Union[pd.Series, np.ndarray, float]]] = None,
    analyst_signal_strength: Optional[Union[pd.Series, np.ndarray, float]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function to engineer Tactician features.
    
    Args:
        data: Input DataFrame with base features from feature_generation_final_feature_selection_step
        regime_probabilities: Dict mapping regime index (0-3) to probability values
        analyst_signal_strength: Aggregated signal strength from analyst ensemble models
        logger: Optional logger
        
    Returns:
        DataFrame with regime confidence and analyst signal features added
    """
    engineer = TacticianFeatureEngineer(logger=logger)
    return engineer.engineer_features(
        data,
        regime_probabilities=regime_probabilities,
        analyst_signal_strength=analyst_signal_strength
    )
