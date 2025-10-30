"""
Prediction Cache Service for Real-Time Trading

This module provides a thread-safe rolling buffer cache for storing recent
Analyst and Tactician predictions, enabling confidence degradation tracking
and uncertainty analysis over time.

Key Features:
- Rolling buffer for last N candles of predictions
- Thread-safe concurrent access for real-time trading
- Automatic artifact loading on startup
- Confidence degradation calculation
- Uncertainty metrics aggregation
- Position-specific prediction history tracking
"""

import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Deque

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.ml_common.uncertainty_calculator import get_global_uncertainty_calculator, UncertaintyCalculator

logger = system_logger.getChild('PredictionCache')


class PredictionEntry:
    """Single prediction entry with timestamp and metadata."""
    
    def __init__(
        self,
        timestamp: datetime,
        predictions: Dict[str, Any],
        ohlcv: Optional[pd.Series] = None,
        uncertainty: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a prediction entry.
        
        Args:
            timestamp: When the prediction was made
            predictions: Prediction values (can be dict of model predictions)
            ohlcv: OHLCV data for this candle
            uncertainty: Uncertainty metrics for this prediction
            confidence: Overall confidence score
            metadata: Additional metadata
        """
        self.timestamp = timestamp
        self.predictions = predictions
        self.ohlcv = ohlcv
        self.uncertainty = uncertainty or {}
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'predictions': self.predictions,
            'ohlcv': self.ohlcv.to_dict() if self.ohlcv is not None else None,
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class PredictionCache:
    """
    Thread-safe cache for recent ML predictions.
    
    Maintains rolling buffers of recent predictions from Analyst and Tactician,
    enabling real-time uncertainty tracking and confidence degradation analysis.
    """
    
    def __init__(
        self,
        max_candles: int = 50,
        default_window: int = 8,
        enable_artifact_loading: bool = True,
        artifact_dir: Optional[str] = None
    ):
        """
        Initialize the prediction cache.
        
        Args:
            max_candles: Maximum number of candles to store in cache
            default_window: Default window size for degradation calculations
            enable_artifact_loading: Whether to load predictions from artifacts on startup
            artifact_dir: Directory to load/save artifacts
        """
        self.max_candles = max_candles
        self.default_window = default_window
        self.enable_artifact_loading = enable_artifact_loading
        self.artifact_dir = artifact_dir or "artifacts/predictions"
        
        # Thread-safe deques for predictions
        self._lock = threading.RLock()  # Reentrant lock for nested locking
        self._analyst_predictions: Deque[PredictionEntry] = deque(maxlen=max_candles)
        self._tactician_predictions: Deque[PredictionEntry] = deque(maxlen=max_candles)
        
        # Position-specific prediction history
        self._position_predictions: Dict[str, List[PredictionEntry]] = {}
        
        # Uncertainty calculator
        self.uncertainty_calculator = get_global_uncertainty_calculator()
        
        self.logger = logger.getChild('PredictionCache')
        self.logger.info(f"✅ PredictionCache initialized: max_candles={max_candles}, window={default_window}")
    
    @handles_errors(fallback=None, context="add analyst prediction")
    def add_analyst_prediction(
        self,
        predictions: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        ohlcv: Optional[pd.Series] = None,
        uncertainty: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an Analyst prediction to the cache.
        
        Args:
            predictions: Analyst prediction values
            timestamp: Prediction timestamp (defaults to now)
            ohlcv: OHLCV data for this candle
            uncertainty: Uncertainty metrics
            confidence: Confidence score
            metadata: Additional metadata
        """
        timestamp = timestamp or datetime.now()
        
        entry = PredictionEntry(
            timestamp=timestamp,
            predictions=predictions,
            ohlcv=ohlcv,
            uncertainty=uncertainty,
            confidence=confidence,
            metadata=metadata
        )
        
        with self._lock:
            self._analyst_predictions.append(entry)
            self.logger.debug(f"Added Analyst prediction at {timestamp}, cache size: {len(self._analyst_predictions)}")
    
    @handles_errors(fallback=None, context="add tactician prediction")
    def add_tactician_prediction(
        self,
        predictions: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        ohlcv: Optional[pd.Series] = None,
        uncertainty: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a Tactician prediction to the cache.
        
        Args:
            predictions: Tactician prediction values
            timestamp: Prediction timestamp (defaults to now)
            ohlcv: OHLCV data for this candle
            uncertainty: Uncertainty metrics
            confidence: Confidence score
            metadata: Additional metadata
        """
        timestamp = timestamp or datetime.now()
        
        entry = PredictionEntry(
            timestamp=timestamp,
            predictions=predictions,
            ohlcv=ohlcv,
            uncertainty=uncertainty,
            confidence=confidence,
            metadata=metadata
        )
        
        with self._lock:
            self._tactician_predictions.append(entry)
            self.logger.debug(f"Added Tactician prediction at {timestamp}, cache size: {len(self._tactician_predictions)}")
    
    @handles_errors(fallback=[], context="get recent analyst predictions")
    def get_recent_analyst_predictions(self, n_candles: Optional[int] = None) -> List[PredictionEntry]:
        """
        Get recent Analyst predictions.
        
        Args:
            n_candles: Number of recent candles to retrieve (defaults to default_window)
        
        Returns:
            List of recent prediction entries (oldest to newest)
        """
        n_candles = n_candles or self.default_window
        
        with self._lock:
            # Get last n_candles entries
            all_predictions = list(self._analyst_predictions)
            recent = all_predictions[-n_candles:] if len(all_predictions) >= n_candles else all_predictions
            return recent
    
    @handles_errors(fallback=[], context="get recent tactician predictions")
    def get_recent_tactician_predictions(self, n_candles: Optional[int] = None) -> List[PredictionEntry]:
        """
        Get recent Tactician predictions.
        
        Args:
            n_candles: Number of recent candles to retrieve (defaults to default_window)
        
        Returns:
            List of recent prediction entries (oldest to newest)
        """
        n_candles = n_candles or self.default_window
        
        with self._lock:
            # Get last n_candles entries
            all_predictions = list(self._tactician_predictions)
            recent = all_predictions[-n_candles:] if len(all_predictions) >= n_candles else all_predictions
            return recent
    
    @handles_errors(fallback=0.0, context="calculate confidence degradation")
    def calculate_confidence_degradation(
        self,
        source: str = 'tactician',
        window: Optional[int] = None,
        position_id: Optional[str] = None
    ) -> float:
        """
        Calculate confidence degradation over recent predictions.
        
        Args:
            source: Which predictions to use ('analyst' or 'tactician')
            window: Window size for calculation (defaults to default_window)
            position_id: If provided, uses position-specific history
        
        Returns:
            float: Degradation metric (negative = degradation, positive = improvement)
        """
        window = window or self.default_window
        
        # Get predictions based on source
        if position_id and position_id in self._position_predictions:
            with self._lock:
                predictions = self._position_predictions[position_id]
        elif source.lower() == 'analyst':
            predictions = self.get_recent_analyst_predictions(window)
        else:  # tactician
            predictions = self.get_recent_tactician_predictions(window)
        
        if len(predictions) < 2:
            self.logger.warning(f"Not enough predictions for degradation calculation: {len(predictions)}")
            return 0.0
        
        # Extract confidence scores
        confidence_values = []
        for entry in predictions:
            if entry.confidence is not None:
                confidence_values.append(entry.confidence)
            elif 'confidence' in entry.predictions:
                confidence_values.append(entry.predictions['confidence'])
            elif 'confidence_score' in entry.predictions:
                confidence_values.append(entry.predictions['confidence_score'])
        
        if len(confidence_values) < 2:
            self.logger.warning("Not enough confidence values for degradation calculation")
            return 0.0
        
        # Calculate degradation using uncertainty calculator
        degradation = self.uncertainty_calculator.calculate_confidence_degradation(
            confidence_series=confidence_values,
            window=window
        )
        
        return degradation
    
    @handles_errors(fallback={}, context="get uncertainty metrics")
    def get_uncertainty_metrics(
        self,
        source: str = 'tactician',
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive uncertainty metrics from recent predictions.
        
        Args:
            source: Which predictions to use ('analyst' or 'tactician')
            window: Window size for calculation
        
        Returns:
            Dict containing uncertainty metrics
        """
        window = window or self.default_window
        
        # Get recent predictions
        if source.lower() == 'analyst':
            predictions = self.get_recent_analyst_predictions(window)
        else:
            predictions = self.get_recent_tactician_predictions(window)
        
        if not predictions:
            return {}
        
        # Extract data for uncertainty calculation
        ensemble_predictions = []
        model_predictions = {}
        confidence_values = []
        
        for entry in predictions:
            # Extract ensemble predictions if available
            if 'ensemble_predictions' in entry.predictions:
                ensemble_predictions.append(entry.predictions['ensemble_predictions'])
            
            # Extract model-specific predictions
            for key, value in entry.predictions.items():
                if key.startswith('model_') or key in ['lightgbm', 'catboost', 'xgboost', 'neural_net']:
                    if key not in model_predictions:
                        model_predictions[key] = []
                    model_predictions[key].append(value)
            
            # Extract confidence
            if entry.confidence is not None:
                confidence_values.append(entry.confidence)
        
        # Calculate comprehensive metrics
        metrics = self.uncertainty_calculator.calculate_comprehensive_metrics(
            ensemble_predictions=ensemble_predictions if ensemble_predictions else None,
            model_predictions={k: np.array(v) for k, v in model_predictions.items()} if model_predictions else None,
            confidence_history=confidence_values if confidence_values else None
        )
        
        return metrics
    
    @handles_errors(fallback=None, context="register position")
    def register_position(
        self,
        position_id: str,
        entry_timestamp: datetime,
        snapshot_window: int = 8
    ) -> None:
        """
        Register a new position and snapshot current predictions.
        
        This saves the current predictions for later confidence degradation analysis.
        
        Args:
            position_id: Unique position identifier
            entry_timestamp: When the position was entered
            snapshot_window: How many candles to snapshot
        """
        with self._lock:
            # Snapshot recent predictions for this position
            snapshot = list(self._tactician_predictions)[-snapshot_window:]
            self._position_predictions[position_id] = snapshot
            
            self.logger.info(f"Registered position {position_id} with {len(snapshot)} prediction snapshots")
    
    @handles_errors(fallback=None, context="update position predictions")
    def update_position_predictions(
        self,
        position_id: str,
        new_prediction: PredictionEntry
    ) -> None:
        """
        Add a new prediction to a position's history.
        
        Args:
            position_id: Position identifier
            new_prediction: New prediction entry to add
        """
        with self._lock:
            if position_id in self._position_predictions:
                self._position_predictions[position_id].append(new_prediction)
                # Limit to reasonable size
                if len(self._position_predictions[position_id]) > self.max_candles:
                    self._position_predictions[position_id] = self._position_predictions[position_id][-self.max_candles:]
    
    @handles_errors(fallback=None, context="remove position")
    def remove_position(self, position_id: str) -> None:
        """
        Remove a position's prediction history.
        
        Args:
            position_id: Position identifier to remove
        """
        with self._lock:
            if position_id in self._position_predictions:
                del self._position_predictions[position_id]
                self.logger.debug(f"Removed position {position_id} from prediction cache")
    
    @handles_errors(fallback={}, context="get position metrics")
    def get_position_metrics(self, position_id: str) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a specific position.
        
        Args:
            position_id: Position identifier
        
        Returns:
            Dict containing position-specific metrics
        """
        with self._lock:
            if position_id not in self._position_predictions:
                return {}
            
            predictions = self._position_predictions[position_id]
            
            if not predictions:
                return {}
            
            # Extract confidence values
            confidence_values = [p.confidence for p in predictions if p.confidence is not None]
            
            # Calculate degradation
            degradation = 0.0
            if len(confidence_values) >= 2:
                degradation = self.uncertainty_calculator.calculate_confidence_degradation(confidence_values)
            
            # Get latest uncertainty
            latest_uncertainty = predictions[-1].uncertainty if predictions else {}
            
            return {
                'position_id': position_id,
                'num_predictions': len(predictions),
                'confidence_degradation': degradation,
                'entry_confidence': confidence_values[0] if confidence_values else None,
                'current_confidence': confidence_values[-1] if confidence_values else None,
                'latest_uncertainty': latest_uncertainty,
                'timestamp': predictions[-1].timestamp if predictions else None
            }
    
    def clear_cache(self) -> None:
        """Clear all cached predictions."""
        with self._lock:
            self._analyst_predictions.clear()
            self._tactician_predictions.clear()
            self._position_predictions.clear()
            self.logger.info("Cleared all prediction caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'analyst_predictions': len(self._analyst_predictions),
                'tactician_predictions': len(self._tactician_predictions),
                'active_positions': len(self._position_predictions),
                'max_candles': self.max_candles,
                'default_window': self.default_window
            }


# Global instance for convenience
_global_prediction_cache: Optional[PredictionCache] = None
_cache_lock = threading.Lock()


def get_global_prediction_cache(max_candles: int = 50, default_window: int = 8) -> PredictionCache:
    """
    Get or create the global prediction cache instance.
    
    Args:
        max_candles: Maximum number of candles to cache
        default_window: Default window for calculations
    
    Returns:
        PredictionCache: Global cache instance
    """
    global _global_prediction_cache
    
    with _cache_lock:
        if _global_prediction_cache is None:
            _global_prediction_cache = PredictionCache(
                max_candles=max_candles,
                default_window=default_window
            )
    
    return _global_prediction_cache


__all__ = [
    'PredictionEntry',
    'PredictionCache',
    'get_global_prediction_cache'
]


