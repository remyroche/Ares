"""
Triple Barrier Labeling Utilities

This module provides common utilities for triple-barrier labeling that can be shared
across different labeling implementations to reduce code duplication.

Key Features:
- Vectorized triple-barrier label generation
- Confidence score calculation
- Horizon-based labeling
- Performance optimization
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    
    # Horizon settings
    min_horizon: int = 1
    max_horizon: int = 100
    default_horizon: int = 20
    
    # Confidence settings
    use_calibrated_confidence: bool = True
    confidence_calibration_window: int = 100
    
    # Performance settings
    use_vectorized_operations: bool = True
    batch_size: int = 1000
    
    # Quality settings
    min_activation_rate: float = 0.01
    max_activation_rate: float = 0.5


class TripleBarrierLabeler:
    """
    Common triple-barrier labeling implementation.
    
    This class provides shared functionality for triple-barrier labeling
    that can be used across different labeling schemes.
    """
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """Initialize the triple-barrier labeler."""
        self.config = config or TripleBarrierConfig()
    
    def generate_labels(self, bars: pd.DataFrame, upper_targets: pd.Series, 
                       lower_targets: pd.Series, eligibility_mask: pd.Series, 
                       horizon: int, volatility_series: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Generate triple-barrier labels with confidence scores.
        
        Args:
            bars: OHLCV DataFrame
            upper_targets: Upper barrier targets
            lower_targets: Lower barrier targets
            eligibility_mask: Boolean mask for eligible bars
            horizon: Maximum horizon for label generation
            volatility_series: Optional volatility series for confidence calculation
            
        Returns:
            Dictionary containing 'labels' and 'confidence' Series
        """
        try:
            if self.config.use_vectorized_operations:
                labels = self._generate_labels_vectorized(
                    bars, upper_targets, lower_targets, eligibility_mask, horizon
                )
            else:
                labels = self._generate_labels_loop(
                    bars, upper_targets, lower_targets, eligibility_mask, horizon
                )
            
            # Calculate confidence scores
            if self.config.use_calibrated_confidence and volatility_series is not None:
                confidence = self._calculate_calibrated_confidence(
                    labels, bars, volatility_series, upper_targets, lower_targets
                )
            else:
                confidence = self._calculate_simple_confidence(
                    labels, bars, upper_targets, lower_targets
                )
            
            return {
                'labels': labels,
                'confidence': confidence
            }
            
        except Exception as e:
            tprint_error(f"❌ Error in triple-barrier labeling: {e}")
            return {
                'labels': pd.Series(0, index=bars.index, dtype=int),
                'confidence': pd.Series(0.5, index=bars.index, dtype=float)
            }
    
    def _generate_labels_vectorized(self, bars: pd.DataFrame, upper_targets: pd.Series, 
                                  lower_targets: pd.Series, eligibility_mask: pd.Series, 
                                  horizon: int) -> pd.Series:
        """Vectorized triple-barrier label generation."""
        try:
            n_bars = len(bars)
            labels = pd.Series(0, index=bars.index, dtype=int)
            
            # Get eligible indices
            eligible_indices = eligibility_mask[eligibility_mask].index
            if len(eligible_indices) == 0:
                return labels
            
            # Convert to numpy arrays for vectorized operations
            eligible_positions = np.array([bars.index.get_loc(idx) for idx in eligible_indices])
            
            # Filter out positions too close to the end
            eligible_positions = eligible_positions[eligible_positions < n_bars - horizon]
            
            if len(eligible_positions) == 0:
                return labels
            
            # Vectorized target calculation
            upper_targets_array = upper_targets.iloc[eligible_positions].values
            lower_targets_array = lower_targets.iloc[eligible_positions].values
            
            # Create future price matrices for vectorized comparison
            future_prices_matrix = np.zeros((len(eligible_positions), horizon))
            for i, pos in enumerate(eligible_positions):
                future_prices_matrix[i, :] = bars['close'].iloc[pos+1:pos+1+horizon].values
            
            # Vectorized hit detection
            upper_hits_matrix = future_prices_matrix >= upper_targets_array.reshape(-1, 1)
            lower_hits_matrix = future_prices_matrix <= lower_targets_array.reshape(-1, 1)
            
            # Find first hits for each position
            upper_first_hits = np.argmax(upper_hits_matrix, axis=1)
            lower_first_hits = np.argmax(lower_hits_matrix, axis=1)
            
            # Handle cases where no hit occurs
            upper_no_hit = ~upper_hits_matrix.any(axis=1)
            lower_no_hit = ~lower_hits_matrix.any(axis=1)
            
            # Set first hits to a large number where no hit occurs
            upper_first_hits[upper_no_hit] = horizon + 1
            lower_first_hits[lower_no_hit] = horizon + 1
            
            # Determine labels based on which target is hit first
            upper_first = upper_first_hits < lower_first_hits
            lower_first = lower_first_hits < upper_first_hits
            
            # Set labels
            labels.iloc[eligible_positions[upper_first]] = 1
            labels.iloc[eligible_positions[lower_first]] = -1
            
            return labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in vectorized label generation: {e}")
            return self._generate_labels_loop(bars, upper_targets, lower_targets, eligibility_mask, horizon)
    
    def _generate_labels_loop(self, bars: pd.DataFrame, upper_targets: pd.Series, 
                            lower_targets: pd.Series, eligibility_mask: pd.Series, 
                            horizon: int) -> pd.Series:
        """Fallback loop-based triple-barrier label generation."""
        try:
            labels = pd.Series(0, index=bars.index, dtype=int)
            
            for i in range(len(bars) - horizon):
                if not eligibility_mask.iloc[i]:
                    continue
                
                upper_target = upper_targets.iloc[i]
                lower_target = lower_targets.iloc[i]
                
                # Check if price hits targets within horizon
                future_prices = bars['close'].iloc[i+1:i+horizon+1]
                if len(future_prices) == 0:
                    continue
                
                # Find first hit
                upper_hits = future_prices >= upper_target
                lower_hits = future_prices <= lower_target
                
                if upper_hits.any() and lower_hits.any():
                    # Both hit - check which comes first
                    upper_first_hit = upper_hits.idxmax() if upper_hits.any() else None
                    lower_first_hit = lower_hits.idxmax() if lower_hits.any() else None
                    
                    if upper_first_hit is not None and lower_first_hit is not None:
                        if upper_first_hit <= lower_first_hit:
                            labels.iloc[i] = 1  # Upper hit first
                        else:
                            labels.iloc[i] = -1  # Lower hit first
                    elif upper_first_hit is not None:
                        labels.iloc[i] = 1
                    elif lower_first_hit is not None:
                        labels.iloc[i] = -1
                elif upper_hits.any():
                    labels.iloc[i] = 1
                elif lower_hits.any():
                    labels.iloc[i] = -1
            
            return labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in loop-based label generation: {e}")
            return pd.Series(0, index=bars.index, dtype=int)
    
    def _calculate_calibrated_confidence(self, labels: pd.Series, bars: pd.DataFrame, 
                                       volatility_series: pd.Series, upper_targets: pd.Series, 
                                       lower_targets: pd.Series) -> pd.Series:
        """Calculate calibrated confidence scores using logistic regression."""
        try:
            if len(labels) < 50:
                return self._calculate_simple_confidence(labels, bars, upper_targets, lower_targets)
            
            # Prepare features for confidence calibration
            features = self._prepare_confidence_features(labels, bars, volatility_series, upper_targets, lower_targets)
            
            if features.empty or len(features) < 20:
                return self._calculate_simple_confidence(labels, bars, upper_targets, lower_targets)
            
            # Create target variable: 1 if label is correct, 0 otherwise
            target = self._create_confidence_target(labels, bars, upper_targets, lower_targets)
            
            if target.sum() < 5:
                return self._calculate_simple_confidence(labels, bars, upper_targets, lower_targets)
            
            # Train logistic regression model for confidence calibration
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import TimeSeriesSplit
            
            # Use time series split to avoid look-ahead bias
            tscv = TimeSeriesSplit(n_splits=min(3, len(features) // 20))
            
            calibrated_confidence = pd.Series(index=labels.index, dtype=float)
            
            for train_idx, test_idx in tscv.split(features):
                if len(train_idx) < 10 or len(test_idx) < 5:
                    continue
                
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train logistic regression
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(X_train_scaled, y_train)
                
                # Predict probabilities
                probabilities = lr.predict_proba(X_test_scaled)[:, 1]
                
                # Store calibrated confidence
                calibrated_confidence.iloc[test_idx] = probabilities
            
            # Fill any remaining NaN values with simple confidence
            nan_mask = calibrated_confidence.isna()
            if nan_mask.any():
                simple_confidence = self._calculate_simple_confidence(labels, bars, upper_targets, lower_targets)
                calibrated_confidence[nan_mask] = simple_confidence[nan_mask]
            
            return calibrated_confidence
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating calibrated confidence: {e}")
            return self._calculate_simple_confidence(labels, bars, upper_targets, lower_targets)
    
    def _calculate_simple_confidence(self, labels: pd.Series, bars: pd.DataFrame, 
                                   upper_targets: pd.Series, lower_targets: pd.Series) -> pd.Series:
        """Calculate simple confidence based on distance to opposite barrier."""
        try:
            confidence = pd.Series(0.5, index=labels.index, dtype=float)
            
            for i, (idx, label) in enumerate(labels.items()):
                if label == 0:
                    continue
                
                if idx not in bars.index:
                    continue
                
                current_price = bars.loc[idx, 'close']
                upper_target = upper_targets.loc[idx] if idx in upper_targets.index else current_price * 1.01
                lower_target = lower_targets.loc[idx] if idx in lower_targets.index else current_price * 0.99
                
                if label == 1:  # Upper target hit
                    distance_to_opposite = abs(current_price - lower_target)
                    total_range = upper_target - lower_target
                    if total_range > 0:
                        confidence.iloc[i] = min(1.0, distance_to_opposite / total_range)
                elif label == -1:  # Lower target hit
                    distance_to_opposite = abs(upper_target - current_price)
                    total_range = upper_target - lower_target
                    if total_range > 0:
                        confidence.iloc[i] = min(1.0, distance_to_opposite / total_range)
            
            return confidence
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating simple confidence: {e}")
            return pd.Series(0.5, index=labels.index, dtype=float)
    
    def _prepare_confidence_features(self, labels: pd.Series, bars: pd.DataFrame, 
                                   volatility_series: pd.Series, upper_targets: pd.Series, 
                                   lower_targets: pd.Series) -> pd.DataFrame:
        """Prepare features for confidence calibration."""
        try:
            features = []
            
            for i, (idx, label) in enumerate(labels.items()):
                if label == 0:
                    continue
                
                feature_row = []
                
                # Price-based features
                if idx in bars.index:
                    current_price = bars.loc[idx, 'close']
                    feature_row.extend([
                        current_price,
                        bars.loc[idx, 'high'] - bars.loc[idx, 'low'],  # Range
                        (bars.loc[idx, 'high'] - current_price) / current_price,  # Upper range
                        (current_price - bars.loc[idx, 'low']) / current_price,  # Lower range
                    ])
                
                # Volatility features
                if idx in volatility_series.index:
                    current_vol = volatility_series.loc[idx]
                    feature_row.extend([
                        current_vol,
                        current_vol * current_price,  # Price volatility
                    ])
                
                # Target features
                if idx in upper_targets.index and idx in lower_targets.index:
                    upper_target = upper_targets.loc[idx]
                    lower_target = lower_targets.loc[idx]
                    feature_row.extend([
                        (upper_target - current_price) / current_price,  # Distance to upper target
                        (current_price - lower_target) / current_price,  # Distance to lower target
                        (upper_target - lower_target) / current_price,  # Total range
                    ])
                
                # Time-based features
                if hasattr(idx, 'hour'):
                    feature_row.extend([
                        idx.hour,
                        idx.dayofweek,
                    ])
                
                features.append(feature_row)
            
            if features:
                feature_df = pd.DataFrame(features, index=labels[labels != 0].index)
                return feature_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_warning(f"⚠️ Error preparing confidence features: {e}")
            return pd.DataFrame()
    
    def _create_confidence_target(self, labels: pd.Series, bars: pd.DataFrame, 
                                upper_targets: pd.Series, lower_targets: pd.Series) -> pd.Series:
        """Create target variable for confidence calibration."""
        try:
            target = pd.Series(0, index=labels.index, dtype=int)
            
            for i, (idx, label) in enumerate(labels.items()):
                if label == 0:
                    continue
                
                if idx not in bars.index:
                    continue
                
                current_price = bars.loc[idx, 'close']
                
                # Look ahead to see if the predicted direction was correct
                future_window = min(20, len(bars) - i - 1)
                if future_window > 0:
                    future_prices = bars.iloc[i+1:i+1+future_window]['close']
                    
                    if label == 1:  # Predicted upward movement
                        # Check if price actually moved up significantly
                        max_future_price = future_prices.max()
                        if max_future_price > current_price * 1.005:  # 0.5% threshold
                            target.iloc[i] = 1
                    elif label == -1:  # Predicted downward movement
                        # Check if price actually moved down significantly
                        min_future_price = future_prices.min()
                        if min_future_price < current_price * 0.995:  # 0.5% threshold
                            target.iloc[i] = 1
            
            return target
            
        except Exception as e:
            tprint_warning(f"⚠️ Error creating confidence target: {e}")
            return pd.Series(0, index=labels.index, dtype=int)


# Convenience functions
def create_triple_barrier_labeler(config: Optional[TripleBarrierConfig] = None) -> TripleBarrierLabeler:
    """Create a triple-barrier labeler with specified configuration."""
    return TripleBarrierLabeler(config)


def generate_triple_barrier_labels(bars: pd.DataFrame, upper_targets: pd.Series, 
                                 lower_targets: pd.Series, eligibility_mask: pd.Series, 
                                 horizon: int, volatility_series: Optional[pd.Series] = None,
                                 config: Optional[TripleBarrierConfig] = None) -> Dict[str, pd.Series]:
    """Generate triple-barrier labels with default configuration."""
    labeler = TripleBarrierLabeler(config)
    return labeler.generate_labels(bars, upper_targets, lower_targets, eligibility_mask, horizon, volatility_series)