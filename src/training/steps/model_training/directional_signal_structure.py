"""
Directional Signal Structure for Enhanced Analyst Signals

This module defines the enhanced signal structure that includes directional information
(short/long) for the Analyst's signals, enabling the Tactician to make more informed
timing decisions based on the expected direction of the trade.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalDirection(Enum):
    """Enumeration for signal directions."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class DirectionalSignal:
    """
    Enhanced signal structure that includes directional information.
    
    This replaces the simple binary green light signals with a more comprehensive
    structure that includes the expected direction of the trade.
    """
    # Core signal information
    is_active: bool  # Whether the signal is active (green light)
    direction: SignalDirection  # Expected direction: long, short, or neutral
    confidence: float  # Signal confidence (0.0 to 1.0)
    
    # Additional metadata
    strength: float  # Signal strength (0.0 to 1.0)
    expected_return: float  # Expected return for this direction
    risk_score: float  # Risk assessment (0.0 to 1.0)
    
    # Timing information
    duration_minutes: int  # Expected signal duration in minutes
    urgency: float  # Signal urgency (0.0 to 1.0)
    
    def __post_init__(self):
        """Validate signal data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.strength}")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"Risk score must be between 0.0 and 1.0, got {self.risk_score}")
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError(f"Urgency must be between 0.0 and 1.0, got {self.urgency}")
        if self.duration_minutes < 0:
            raise ValueError(f"Duration must be non-negative, got {self.duration_minutes}")


class DirectionalSignalArray:
    """
    Array-like container for directional signals with efficient operations.
    
    This class provides a more efficient way to handle arrays of directional signals
    compared to lists of DirectionalSignal objects.
    """
    
    def __init__(self, signals: Optional[np.ndarray] = None):
        """
        Initialize directional signal array.
        
        Args:
            signals: Array of directional signals with shape (n_samples, 8) where:
                    - Column 0: is_active (bool)
                    - Column 1: direction (0=neutral, 1=long, 2=short)
                    - Column 2: confidence (float)
                    - Column 3: strength (float)
                    - Column 4: expected_return (float)
                    - Column 5: risk_score (float)
                    - Column 6: duration_minutes (int)
                    - Column 7: urgency (float)
        """
        if signals is None:
            self.signals = np.array([])
        else:
            self.signals = signals
            self._validate_signals()
    
    def _validate_signals(self):
        """Validate the signal array structure."""
        if len(self.signals.shape) != 2 or self.signals.shape[1] != 8:
            raise ValueError(f"Signals must have shape (n_samples, 8), got {self.signals.shape}")
        
        # Validate data types and ranges
        if not np.all(np.isin(self.signals[:, 1], [0, 1, 2])):  # direction
            raise ValueError("Direction values must be 0 (neutral), 1 (long), or 2 (short)")
        
        for col, name in enumerate(['is_active', 'direction', 'confidence', 'strength', 
                                 'expected_return', 'risk_score', 'duration_minutes', 'urgency']):
            if col == 0:  # is_active
                if not np.all(np.isin(self.signals[:, col], [0, 1])):
                    raise ValueError(f"{name} must be 0 or 1")
            elif col in [2, 3, 4, 5, 7]:  # float columns
                if not np.all((self.signals[:, col] >= 0) & (self.signals[:, col] <= 1)):
                    raise ValueError(f"{name} must be between 0.0 and 1.0")
            elif col == 6:  # duration_minutes
                if not np.all(self.signals[:, col] >= 0):
                    raise ValueError(f"{name} must be non-negative")
    
    @classmethod
    def from_binary_signals(cls, binary_signals: np.ndarray, 
                          directions: Optional[np.ndarray] = None,
                          confidences: Optional[np.ndarray] = None) -> 'DirectionalSignalArray':
        """
        Create directional signals from binary signals (backward compatibility).
        
        Args:
            binary_signals: Binary array (0/1) indicating active signals
            directions: Optional array of directions (0=neutral, 1=long, 2=short)
            confidences: Optional array of confidence scores
            
        Returns:
            DirectionalSignalArray instance
        """
        n_samples = len(binary_signals)
        
        # Default values
        if directions is None:
            directions = np.ones(n_samples, dtype=int)  # Default to long
        if confidences is None:
            confidences = np.ones(n_samples, dtype=float)  # Default to 1.0
        
        # Create signal array
        signals = np.zeros((n_samples, 8), dtype=float)
        signals[:, 0] = binary_signals.astype(float)  # is_active
        signals[:, 1] = directions.astype(float)  # direction
        signals[:, 2] = confidences  # confidence
        signals[:, 3] = confidences  # strength (same as confidence by default)
        signals[:, 4] = 0.0  # expected_return (default)
        signals[:, 5] = 0.5  # risk_score (default)
        signals[:, 6] = 30  # duration_minutes (default)
        signals[:, 7] = 0.5  # urgency (default)
        
        return cls(signals)
    
    def get_active_signals(self) -> np.ndarray:
        """Get boolean array of active signals."""
        return self.signals[:, 0].astype(bool)
    
    def get_directions(self) -> np.ndarray:
        """Get array of signal directions."""
        return self.signals[:, 1].astype(int)
    
    def get_confidences(self) -> np.ndarray:
        """Get array of signal confidences."""
        return self.signals[:, 2]
    
    def get_strengths(self) -> np.ndarray:
        """Get array of signal strengths."""
        return self.signals[:, 3]
    
    def get_expected_returns(self) -> np.ndarray:
        """Get array of expected returns."""
        return self.signals[:, 4]
    
    def get_risk_scores(self) -> np.ndarray:
        """Get array of risk scores."""
        return self.signals[:, 5]
    
    def get_durations(self) -> np.ndarray:
        """Get array of signal durations."""
        return self.signals[:, 6].astype(int)
    
    def get_urgencies(self) -> np.ndarray:
        """Get array of signal urgencies."""
        return self.signals[:, 7]
    
    def filter_by_direction(self, direction: SignalDirection) -> 'DirectionalSignalArray':
        """Filter signals by direction."""
        direction_value = direction.value
        if direction_value == "long":
            direction_int = 1
        elif direction_value == "short":
            direction_int = 2
        else:  # neutral
            direction_int = 0
        
        mask = self.signals[:, 1] == direction_int
        return DirectionalSignalArray(self.signals[mask])
    
    def filter_by_confidence(self, min_confidence: float) -> 'DirectionalSignalArray':
        """Filter signals by minimum confidence."""
        mask = self.signals[:, 2] >= min_confidence
        return DirectionalSignalArray(self.signals[mask])
    
    def filter_active_only(self) -> 'DirectionalSignalArray':
        """Filter to only active signals."""
        mask = self.signals[:, 0] == 1
        return DirectionalSignalArray(self.signals[mask])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the signals."""
        if len(self.signals) == 0:
            return {
                'total_signals': 0,
                'active_signals': 0,
                'long_signals': 0,
                'short_signals': 0,
                'neutral_signals': 0,
                'avg_confidence': 0.0,
                'avg_strength': 0.0,
                'avg_expected_return': 0.0,
                'avg_risk_score': 0.0,
                'avg_duration': 0.0,
                'avg_urgency': 0.0
            }
        
        active_mask = self.signals[:, 0] == 1
        long_mask = self.signals[:, 1] == 1
        short_mask = self.signals[:, 1] == 2
        neutral_mask = self.signals[:, 1] == 0
        
        return {
            'total_signals': len(self.signals),
            'active_signals': int(np.sum(active_mask)),
            'long_signals': int(np.sum(long_mask)),
            'short_signals': int(np.sum(short_mask)),
            'neutral_signals': int(np.sum(neutral_mask)),
            'avg_confidence': float(np.mean(self.signals[:, 2])),
            'avg_strength': float(np.mean(self.signals[:, 3])),
            'avg_expected_return': float(np.mean(self.signals[:, 4])),
            'avg_risk_score': float(np.mean(self.signals[:, 5])),
            'avg_duration': float(np.mean(self.signals[:, 6])),
            'avg_urgency': float(np.mean(self.signals[:, 7]))
        }
    
    def to_binary_signals(self) -> np.ndarray:
        """Convert to binary signals for backward compatibility."""
        return self.signals[:, 0].astype(int)
    
    def __len__(self):
        """Return number of signals."""
        return len(self.signals)
    
    def __getitem__(self, index):
        """Get signal at index."""
        if isinstance(index, slice):
            return DirectionalSignalArray(self.signals[index])
        else:
            signal_data = self.signals[index]
            return DirectionalSignal(
                is_active=bool(signal_data[0]),
                direction=SignalDirection(["neutral", "long", "short"][int(signal_data[1])]),
                confidence=float(signal_data[2]),
                strength=float(signal_data[3]),
                expected_return=float(signal_data[4]),
                risk_score=float(signal_data[5]),
                duration_minutes=int(signal_data[6]),
                urgency=float(signal_data[7])
            )


def create_directional_signals_from_analyst_outputs(
    analyst_outputs: Dict[str, np.ndarray],
    market_data: Optional[np.ndarray] = None
) -> DirectionalSignalArray:
    """
    Create directional signals from analyst model outputs.
    
    Args:
        analyst_outputs: Dictionary containing analyst model outputs
        market_data: Optional market data for enhanced signal generation
        
    Returns:
        DirectionalSignalArray with directional signals
    """
    # Extract basic signals
    if 'signals' in analyst_outputs:
        binary_signals = analyst_outputs['signals']
    elif 'predictions' in analyst_outputs:
        # Convert predictions to binary signals
        predictions = analyst_outputs['predictions']
        binary_signals = (predictions > 0.5).astype(int)
    else:
        raise ValueError("analyst_outputs must contain 'signals' or 'predictions'")
    
    # Extract confidences
    if 'confidences' in analyst_outputs:
        confidences = analyst_outputs['confidences']
    else:
        confidences = np.ones(len(binary_signals), dtype=float)
    
    # Determine directions based on analyst outputs
    if 'directional_predictions' in analyst_outputs:
        # Use explicit directional predictions
        directional_preds = analyst_outputs['directional_predictions']
        directions = np.where(directional_preds > 0.5, 1, 2)  # 1=long, 2=short
    elif 'predictions' in analyst_outputs:
        # Infer direction from predictions
        predictions = analyst_outputs['predictions']
        directions = np.where(predictions > 0.5, 1, 2)  # 1=long, 2=short
    else:
        # Default to long direction
        directions = np.ones(len(binary_signals), dtype=int)
    
    # Create enhanced signal array
    signals = np.zeros((len(binary_signals), 8), dtype=float)
    signals[:, 0] = binary_signals.astype(float)  # is_active
    signals[:, 1] = directions.astype(float)  # direction
    signals[:, 2] = confidences  # confidence
    signals[:, 3] = confidences  # strength
    signals[:, 4] = 0.0  # expected_return (to be calculated)
    signals[:, 5] = 0.5  # risk_score (to be calculated)
    signals[:, 6] = 30  # duration_minutes (default)
    signals[:, 7] = 0.5  # urgency (default)
    
    return DirectionalSignalArray(signals)


def enhance_signals_with_market_data(
    signals: DirectionalSignalArray,
    market_data: np.ndarray,
    price_column: int = 0
) -> DirectionalSignalArray:
    """
    Enhance signals with market data to calculate expected returns and risk scores.
    
    Args:
        signals: Base directional signals
        market_data: Market data array with price information
        price_column: Column index for price data
        
    Returns:
        Enhanced DirectionalSignalArray
    """
    enhanced_signals = signals.signals.copy()
    
    # Calculate expected returns based on price momentum
    if len(market_data) > 1:
        price_changes = np.diff(market_data[:, price_column])
        
        for i in range(len(enhanced_signals)):
            if i < len(price_changes):
                # Calculate expected return based on direction and price momentum
                direction = int(enhanced_signals[i, 1])
                if direction == 1:  # long
                    enhanced_signals[i, 4] = max(0, price_changes[i])  # expected_return
                elif direction == 2:  # short
                    enhanced_signals[i, 4] = max(0, -price_changes[i])  # expected_return
                
                # Calculate risk score based on volatility
                if i >= 10:  # Need some history
                    recent_changes = price_changes[max(0, i-10):i+1]
                    volatility = np.std(recent_changes)
                    enhanced_signals[i, 5] = min(1.0, volatility * 10)  # risk_score
    
    return DirectionalSignalArray(enhanced_signals)