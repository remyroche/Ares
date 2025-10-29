"""
Shared Feature Engineering Module

This module provides shared feature engineering utilities that ensure
consistency between training and inference (signal generation).

Features engineered here match exactly what is done during model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


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
    
    Engineers the same features as used in training:
    - Regime-based features (regime_strength, regime_confidence)
    - Market condition features (volume_price_trend, volume_momentum)
    - Volatility features (volatility_5d, volatility_20d, volatility_ratio)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.engineered_feature_names = [
            'regime_strength',
            'regime_confidence',
            'volume_price_trend',
            'volume_momentum',
            'volatility_5d',
            'volatility_20d',
            'volatility_ratio',
        ]
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        regime_probability: Optional[Union[pd.Series, np.ndarray, float]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Engineer features specific to Analyst role.
        
        This matches the logic in model_trainer.py:_engineer_analyst_features()
        
        Args:
            data: Input DataFrame with market data
            regime_probability: Optional regime probability value(s) to add
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with engineered features added
        """
        try:
            result_data = data.copy()
            warnings = []
            errors = []
            
            # 1. Add regime-based features
            if regime_probability is not None:
                # Handle different input types
                if isinstance(regime_probability, (float, int)):
                    # Single value - create series
                    regime_prob = pd.Series([regime_probability] * len(result_data), index=result_data.index)
                elif isinstance(regime_probability, pd.Series):
                    regime_prob = regime_probability
                elif isinstance(regime_probability, np.ndarray):
                    if len(regime_probability) == 1:
                        regime_prob = pd.Series([regime_probability[0]] * len(result_data), index=result_data.index)
                    else:
                        regime_prob = pd.Series(regime_probability, index=result_data.index[:len(regime_probability)])
                else:
                    regime_prob = None
                
                if regime_prob is not None:
                    # Ensure same length as data
                    if len(regime_prob) != len(result_data):
                        if len(regime_prob) == 1:
                            regime_prob = pd.Series([regime_prob.iloc[0]] * len(result_data), index=result_data.index)
                        else:
                            warnings.append(f"Regime probability length ({len(regime_prob)}) doesn't match data length ({len(result_data)}), using last value")
                            regime_prob = pd.Series([regime_prob.iloc[-1]] * len(result_data), index=result_data.index)
                    
                    result_data['regime_strength'] = regime_prob.abs()
                    result_data['regime_confidence'] = np.where(
                        regime_prob > 0.5,
                        regime_prob,
                        1 - regime_prob
                    )
            
            elif 'regime_probability' in result_data.columns:
                # Use existing column
                result_data['regime_strength'] = result_data['regime_probability'].abs()
                result_data['regime_confidence'] = np.where(
                    result_data['regime_probability'] > 0.5,
                    result_data['regime_probability'],
                    1 - result_data['regime_probability']
                )
            else:
                warnings.append("No regime_probability provided and not found in data")
            
            # 2. Add market condition features
            if 'volume' in result_data.columns and 'close' in result_data.columns:
                result_data['volume_price_trend'] = result_data['volume'] * result_data['close'].pct_change()
                # Handle division by zero for volume_momentum
                volume_5d = result_data['volume'].rolling(5).mean()
                volume_20d = result_data['volume'].rolling(20).mean()
                result_data['volume_momentum'] = self._safe_divide(volume_5d, volume_20d)
            else:
                missing_cols = []
                if 'volume' not in result_data.columns:
                    missing_cols.append('volume')
                if 'close' not in result_data.columns:
                    missing_cols.append('close')
                warnings.append(f"Missing columns for market condition features: {missing_cols}")
            
            # 3. Add volatility features
            if 'close' in result_data.columns:
                result_data['volatility_5d'] = result_data['close'].rolling(5).std()
                result_data['volatility_20d'] = result_data['close'].rolling(20).std()
                # Handle division by zero for volatility_ratio
                result_data['volatility_ratio'] = self._safe_divide(
                    result_data['volatility_5d'],
                    result_data['volatility_20d']
                )
            else:
                warnings.append("Missing 'close' column for volatility features")
            
            # Fill NaN values created by rolling operations
            result_data = result_data.fillna(method='bfill').fillna(0)
            
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
    
    Engineers the same features as used in training:
    - Timing features (hour, day_of_week, is_weekend)
    - Analyst signal features (analyst_signal_strength, analyst_signal_consistency)
    - Risk features (price_momentum, risk_adjusted_return)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.engineered_feature_names = [
            'hour',
            'day_of_week',
            'is_weekend',
            'analyst_signal_strength',
            'analyst_signal_consistency',
            'price_momentum',
            'risk_adjusted_return',
        ]
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        timestamp: Optional[Union[pd.Timestamp, datetime, str]] = None,
        analyst_confidence: Optional[float] = None,
        analyst_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Engineer features specific to Tactician role.
        
        This matches the logic in model_trainer.py:_engineer_tactician_features()
        
        Args:
            data: Input DataFrame with market data
            timestamp: Optional timestamp for timing features
            analyst_confidence: Optional analyst confidence value
            analyst_outputs: Optional analyst output dictionary
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with engineered features added
        """
        try:
            result_data = data.copy()
            warnings = []
            
            # 1. Add timing features
            if timestamp is not None:
                # Handle different timestamp types
                if isinstance(timestamp, str):
                    timestamp_dt = pd.to_datetime(timestamp)
                elif isinstance(timestamp, (pd.Timestamp, datetime)):
                    timestamp_dt = timestamp
                else:
                    timestamp_dt = None
                
                if timestamp_dt is not None:
                    result_data['hour'] = timestamp_dt.hour
                    result_data['day_of_week'] = timestamp_dt.dayofweek
                    result_data['is_weekend'] = 1 if timestamp_dt.dayofweek in [5, 6] else 0
                else:
                    warnings.append(f"Could not parse timestamp: {timestamp}")
            
            elif 'timestamp' in result_data.columns:
                # Use existing timestamp column
                timestamp_series = pd.to_datetime(result_data['timestamp'])
                result_data['hour'] = timestamp_series.dt.hour
                result_data['day_of_week'] = timestamp_series.dt.dayofweek
                result_data['is_weekend'] = result_data['day_of_week'].isin([5, 6]).astype(int)
            else:
                warnings.append("No timestamp provided and not found in data")
                # Set defaults for timing features
                result_data['hour'] = 12  # Default noon
                result_data['day_of_week'] = 0  # Default Monday
                result_data['is_weekend'] = 0
            
            # 2. Add analyst signal features
            analyst_columns = [col for col in result_data.columns if 'analyst' in col.lower()]
            
            if analyst_outputs:
                # Create analyst signal from outputs dictionary
                analyst_values = []
                for key, value in analyst_outputs.items():
                    if isinstance(value, (int, float)):
                        analyst_values.append(value)
                
                if analyst_values:
                    analyst_array = np.array(analyst_values)
                    if len(analyst_array) > 1:
                        result_data['analyst_signal_strength'] = np.mean(analyst_array)
                        result_data['analyst_signal_consistency'] = np.std(analyst_array)
                    else:
                        result_data['analyst_signal_strength'] = analyst_array[0]
                        result_data['analyst_signal_consistency'] = 0.0
                else:
                    result_data['analyst_signal_strength'] = analyst_confidence if analyst_confidence is not None else 0.5
                    result_data['analyst_signal_consistency'] = 0.0
            
            elif analyst_confidence is not None:
                # Use analyst confidence as signal strength
                result_data['analyst_signal_strength'] = analyst_confidence
                result_data['analyst_signal_consistency'] = 0.0
            
            elif analyst_columns:
                # Use existing analyst columns
                result_data['analyst_signal_strength'] = result_data[analyst_columns].mean(axis=1)
                result_data['analyst_signal_consistency'] = result_data[analyst_columns].std(axis=1)
            else:
                warnings.append("No analyst signals found, using defaults")
                result_data['analyst_signal_strength'] = 0.5
                result_data['analyst_signal_consistency'] = 0.0
            
            # 3. Add risk features
            if 'close' in result_data.columns:
                result_data['price_momentum'] = result_data['close'].pct_change(5)
                # Handle division by zero for risk_adjusted_return
                price_std = result_data['close'].rolling(20).std()
                result_data['risk_adjusted_return'] = self._safe_divide(
                    result_data['price_momentum'],
                    price_std
                )
            else:
                warnings.append("Missing 'close' column for risk features")
                result_data['price_momentum'] = 0.0
                result_data['risk_adjusted_return'] = 0.0
            
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
    regime_probability: Optional[Union[pd.Series, np.ndarray, float]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function to engineer Analyst features.
    
    Args:
        data: Input DataFrame
        regime_probability: Optional regime probability value(s)
        logger: Optional logger
        
    Returns:
        DataFrame with engineered features
    """
    engineer = AnalystFeatureEngineer(logger=logger)
    return engineer.engineer_features(data, regime_probability=regime_probability)


def engineer_tactician_features(
    data: pd.DataFrame,
    timestamp: Optional[Union[pd.Timestamp, datetime, str]] = None,
    analyst_confidence: Optional[float] = None,
    analyst_outputs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function to engineer Tactician features.
    
    Args:
        data: Input DataFrame
        timestamp: Optional timestamp
        analyst_confidence: Optional analyst confidence
        analyst_outputs: Optional analyst outputs dictionary
        logger: Optional logger
        
    Returns:
        DataFrame with engineered features
    """
    engineer = TacticianFeatureEngineer(logger=logger)
    return engineer.engineer_features(
        data,
        timestamp=timestamp,
        analyst_confidence=analyst_confidence,
        analyst_outputs=analyst_outputs
    )
