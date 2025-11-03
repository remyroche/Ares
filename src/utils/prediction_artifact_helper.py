"""
Prediction Artifact Helper

Helper utilities for Analyst and Tactician to save predictions with uncertainty metrics
using the PreTrainingArtifactManager system.

Key Features:
- Standardized artifact structure for predictions
- Automatic uncertainty calculation from ensemble predictions
- OHLCV + predictions + uncertainty in joint format
- Integration with PreTrainingArtifactManager
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    artifact_context
)
from src.utils.ml_common.uncertainty_calculator import get_global_uncertainty_calculator

logger = system_logger.getChild('PredictionArtifactHelper')


class PredictionArtifactHelper:
    """Helper for saving Analyst/Tactician predictions with uncertainty metrics."""
    
    def __init__(self):
        """Initialize the helper."""
        self.artifact_manager = get_pretraining_artifact_manager()
        self.uncertainty_calculator = get_global_uncertainty_calculator()
        self.logger = logger
    
    @handles_errors(fallback=None, context="save analyst predictions")
    def save_analyst_predictions(
        self,
        predictions: Dict[str, Any],
        ohlcv_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        ensemble_predictions: Optional[List[np.ndarray]] = None,
        model_predictions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save Analyst predictions with uncertainty metrics.
        
        Args:
            predictions: Main prediction output
            ohlcv_data: OHLCV price data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            ensemble_predictions: List of predictions from ensemble members
            model_predictions: Dict of predictions from different models
            metadata: Additional metadata
        
        Returns:
            Path to saved artifact file
        """
        try:
            # Calculate uncertainty metrics
            uncertainty_metrics = self.uncertainty_calculator.calculate_comprehensive_metrics(
                ensemble_predictions=ensemble_predictions,
                model_predictions=model_predictions,
                confidence_history=None  # Not applicable for single prediction
            )
            
            # Create predictions DataFrame
            pred_index = ohlcv_data.index if hasattr(ohlcv_data, 'index') else pd.RangeIndex(len(ohlcv_data))
            predictions_df = pd.DataFrame({
                'analyst_prediction': [predictions.get('prediction', 0.0)] * len(pred_index),
                'analyst_confidence': [predictions.get('confidence', 0.0)] * len(pred_index),
                'ensemble_variance': [uncertainty_metrics.get('ensemble_variance', 0.0)] * len(pred_index),
                'model_disagreement': [uncertainty_metrics.get('model_disagreement', 0.0)] * len(pred_index),
                'combined_uncertainty': [uncertainty_metrics.get('combined_uncertainty', 0.0)] * len(pred_index)
            }, index=pred_index)
            
            # Use artifact context manager
            with artifact_context(
                symbol=symbol,
                exchange=exchange,
                information='analyst_predictions',
                timeframe=timeframe,
                model='Analyst'
            ):
                # Create joint parquet file
                filepath = self.artifact_manager.create_joint_parquet_file(
                    step_name='live_predictions',
                    ohlcv_data=ohlcv_data,
                    labels_data=predictions_df,
                    features_data=None,
                    key='analyst_pred'
                )
                
                self.logger.info(f"✅ Saved Analyst predictions with uncertainty to {filepath}")
                return filepath
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save Analyst predictions: {e}")
            return None
    
    @handles_errors(fallback=None, context="save tactician predictions")
    def save_tactician_predictions(
        self,
        predictions: Dict[str, Any],
        ohlcv_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        micro_movements: Optional[Dict[str, Any]] = None,
        directional_analysis: Optional[Dict[str, Any]] = None,
        ensemble_predictions: Optional[List[np.ndarray]] = None,
        model_predictions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save Tactician predictions with uncertainty and micro-movement data.
        
        Args:
            predictions: Main prediction output
            ohlcv_data: OHLCV price data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            micro_movements: Micro-movement predictions
            directional_analysis: Directional analysis results
            ensemble_predictions: List of predictions from ensemble members
            model_predictions: Dict of predictions from different models
            metadata: Additional metadata
        
        Returns:
            Path to saved artifact file
        """
        try:
            # Calculate uncertainty metrics
            uncertainty_metrics = self.uncertainty_calculator.calculate_comprehensive_metrics(
                ensemble_predictions=ensemble_predictions,
                model_predictions=model_predictions,
                confidence_history=None
            )
            
            # Create predictions DataFrame
            pred_index = ohlcv_data.index if hasattr(ohlcv_data, 'index') else pd.RangeIndex(len(ohlcv_data))
            
            # Extract micro-movement data if available
            micro_long = micro_movements.get('micro_immediate_long', {}).get('probability', 0.5) if micro_movements else 0.5
            micro_short = micro_movements.get('micro_immediate_short', {}).get('probability', 0.5) if micro_movements else 0.5
            
            # Extract directional analysis if available
            dir_confidence = directional_analysis.get('directional_confidence', 0.0) if directional_analysis else 0.0
            dir_bias = directional_analysis.get('directional_bias', 'NEUTRAL') if directional_analysis else 'NEUTRAL'
            
            predictions_df = pd.DataFrame({
                'tactician_confidence': [predictions.get('confidence', 0.0)] * len(pred_index),
                'tactician_prediction': [predictions.get('prediction', 0.0)] * len(pred_index),
                'micro_long_prob': [micro_long] * len(pred_index),
                'micro_short_prob': [micro_short] * len(pred_index),
                'directional_confidence': [dir_confidence] * len(pred_index),
                'directional_bias': [dir_bias] * len(pred_index),
                'ensemble_variance': [uncertainty_metrics.get('ensemble_variance', 0.0)] * len(pred_index),
                'model_disagreement': [uncertainty_metrics.get('model_disagreement', 0.0)] * len(pred_index),
                'combined_uncertainty': [uncertainty_metrics.get('combined_uncertainty', 0.0)] * len(pred_index)
            }, index=pred_index)
            
            # Use artifact context manager
            with artifact_context(
                symbol=symbol,
                exchange=exchange,
                information='tactician_predictions',
                timeframe=timeframe,
                model='Tactician'
            ):
                # Create joint parquet file
                filepath = self.artifact_manager.create_joint_parquet_file(
                    step_name='live_predictions',
                    ohlcv_data=ohlcv_data,
                    labels_data=predictions_df,
                    features_data=None,
                    key='tactician_pred'
                )
                
                self.logger.info(f"✅ Saved Tactician predictions with uncertainty to {filepath}")
                return filepath
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save Tactician predictions: {e}")
            return None


# Global instance
_global_helper: Optional[PredictionArtifactHelper] = None


def get_prediction_artifact_helper() -> PredictionArtifactHelper:
    """Get global prediction artifact helper instance."""
    global _global_helper
    if _global_helper is None:
        _global_helper = PredictionArtifactHelper()
    return _global_helper


__all__ = [
    'PredictionArtifactHelper',
    'get_prediction_artifact_helper'
]

