"""
Tactician Training Filter - Enhanced Training Data Selection

This module implements the enhanced Tactician training data filtering logic that trains
the Tactician on samples where the Analyst gives high confidence (>0.5) plus the next
45 minutes after confidence drops below 0.5. This simulates real trading conditions
where the Tactician may open a position after the Analyst gives a green light and
then "ride" it as long as short-term expectations remain good.

Key Features:
- Confidence-based training sample selection
- Post-confidence-drop window extension (45 minutes)
- Realistic trading simulation for Tactician training
- Configurable thresholds and time windows
- Efficient vectorized operations for large datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance
)

logger = system_logger.getChild('TacticianTrainingFilter')


@dataclass
class TacticianFilterConfig:
    """Configuration for Tactician training data filtering."""
    
    # Core filtering parameters
    confidence_threshold: float = 0.5
    post_drop_window_minutes: int = 45
    
    # Advanced filtering options
    min_confidence_duration_minutes: int = 1  # Minimum duration of high confidence
    max_gap_minutes: int = 60  # Maximum gap between high confidence periods
    enable_smoothing: bool = True  # Enable confidence smoothing
    smoothing_window: int = 5  # Window for confidence smoothing
    
    # Performance settings
    enable_vectorization: bool = True
    chunk_size: int = 10000  # Process data in chunks for memory efficiency
    
    # Validation settings
    min_samples_per_period: int = 10
    max_missing_ratio: float = 0.1


class TacticianTrainingFilter:
    """
    Enhanced Tactician training data filter that implements confidence-based
    training sample selection with post-confidence-drop window extension.
    
    Training Logic:
    1. Include all samples where Analyst confidence > threshold (default 0.5)
    2. Include the next N minutes (default 45) after confidence drops below threshold
    3. This simulates real trading where Tactician "rides" positions after initial signals weaken
    """
    
    def __init__(self, config: Optional[TacticianFilterConfig] = None):
        """Initialize the Tactician training filter."""
        self.config = config or TacticianFilterConfig()
        self.logger = logger.getChild('TacticianTrainingFilter')
        
        # Validation
        self._validate_config()
        
        # State tracking
        self.filter_stats = {}
        self.last_filter_result = None
        
        self.logger.info("🎯 Tactician Training Filter initialized")
        self.logger.info(f"   → Confidence threshold: {self.config.confidence_threshold}")
        self.logger.info(f"   → Post-drop window: {self.config.post_drop_window_minutes} minutes")
    
    def _validate_config(self):
        """Validate filter configuration."""
        if not (0.0 <= self.config.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.config.post_drop_window_minutes <= 0:
            raise ValueError("Post-drop window must be positive")
        
        if self.config.min_confidence_duration_minutes < 0:
            raise ValueError("Min confidence duration cannot be negative")
    
    def create_training_mask(self, 
                           analyst_confidence: pd.Series,
                           data_index: Optional[pd.DatetimeIndex] = None,
                           timeframe_minutes: int = 1) -> pd.Series:
        """
        Create training mask for Tactician based on Analyst confidence.
        
        Args:
            analyst_confidence: Series of Analyst confidence scores
            data_index: Optional datetime index (if confidence index is not datetime)
            timeframe_minutes: Timeframe in minutes (for calculating durations)
            
        Returns:
            Boolean series indicating which samples to include in training
        """
        try:
            tprint_info("🎯 Creating Tactician training mask...")
            
            # Prepare data
            confidence_data = self._prepare_confidence_data(analyst_confidence, data_index)
            
            # Apply smoothing if enabled
            if self.config.enable_smoothing:
                confidence_data = self._smooth_confidence(confidence_data)
            
            # Create base mask (high confidence periods)
            high_confidence_mask = confidence_data > self.config.confidence_threshold
            
            # Find confidence drop points
            drop_points = self._find_confidence_drop_points(confidence_data)
            
            # Extend training window after each drop
            post_drop_mask = self._create_post_drop_mask(
                drop_points, confidence_data.index, timeframe_minutes
            )
            
            # Combine masks
            training_mask = high_confidence_mask | post_drop_mask
            
            # Apply additional filtering
            training_mask = self._apply_additional_filtering(
                training_mask, confidence_data
            )
            
            # Calculate and log statistics
            self._calculate_filter_stats(
                high_confidence_mask, post_drop_mask, training_mask, confidence_data
            )
            
            self.last_filter_result = {
                'mask': training_mask,
                'high_confidence_mask': high_confidence_mask,
                'post_drop_mask': post_drop_mask,
                'drop_points': drop_points,
                'stats': self.filter_stats
            }
            
            tprint_success(f"✅ Training mask created: {training_mask.sum()}/{len(training_mask)} samples selected")
            return training_mask
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create training mask: {e}")
            raise
    
    def _prepare_confidence_data(self, 
                                confidence: pd.Series,
                                data_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """Prepare confidence data for filtering."""
        if data_index is not None:
            # Align confidence with data index
            confidence = confidence.reindex(data_index, method='ffill')
        
        # Validate confidence data
        if confidence.isna().all():
            raise ValueError("All confidence values are NaN")
        
        # Forward fill missing values (within reasonable limits)
        max_forward_fill = self.config.max_gap_minutes
        confidence = confidence.fillna(method='ffill', limit=max_forward_fill)
        
        # Fill remaining NaN with 0 (no confidence)
        confidence = confidence.fillna(0.0)
        
        # Clip to valid range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def _smooth_confidence(self, confidence: pd.Series) -> pd.Series:
        """Apply smoothing to confidence data to reduce noise."""
        window = self.config.smoothing_window
        if window > 1:
            # Use rolling mean for smoothing
            smoothed = confidence.rolling(window=window, center=True, min_periods=1).mean()
            return smoothed
        return confidence
    
    def _find_confidence_drop_points(self, confidence: pd.Series) -> List[pd.Timestamp]:
        """Find points where confidence drops below threshold."""
        threshold = self.config.confidence_threshold
        
        # Find transitions from above threshold to below threshold
        above_threshold = confidence > threshold
        below_threshold = confidence <= threshold
        
        # Find drop points (where we transition from above to below)
        drop_points = []
        
        for i in range(1, len(above_threshold)):
            if above_threshold.iloc[i-1] and below_threshold.iloc[i]:
                drop_points.append(confidence.index[i])
        
        self.logger.info(f"📍 Found {len(drop_points)} confidence drop points")
        return drop_points
    
    def _create_post_drop_mask(self, 
                              drop_points: List[pd.Timestamp],
                              index: pd.DatetimeIndex,
                              timeframe_minutes: int) -> pd.Series:
        """Create mask for post-confidence-drop periods."""
        post_drop_mask = pd.Series(False, index=index)
        
        window_minutes = self.config.post_drop_window_minutes
        
        for drop_point in drop_points:
            # Calculate end time for post-drop window
            end_time = drop_point + pd.Timedelta(minutes=window_minutes)
            
            # Create mask for this post-drop period
            period_mask = (index >= drop_point) & (index <= end_time)
            post_drop_mask |= period_mask
        
        return post_drop_mask
    
    def _apply_additional_filtering(self, 
                                  training_mask: pd.Series,
                                  confidence: pd.Series) -> pd.Series:
        """Apply additional filtering rules."""
        filtered_mask = training_mask.copy()
        
        # Remove very short training periods
        min_duration_samples = max(1, self.config.min_samples_per_period)
        
        # Find continuous periods in the mask
        mask_diff = filtered_mask.astype(int).diff()
        period_starts = mask_diff == 1
        period_ends = mask_diff == -1
        
        # For each period, check if it meets minimum duration
        current_period_start = None
        for i, (start, end) in enumerate(zip(period_starts, period_ends)):
            if start:
                current_period_start = i
            elif end and current_period_start is not None:
                period_length = i - current_period_start
                if period_length < min_duration_samples:
                    # Remove this short period
                    filtered_mask.iloc[current_period_start:i] = False
                current_period_start = None
        
        # Handle case where period extends to end of data
        if current_period_start is not None:
            period_length = len(filtered_mask) - current_period_start
            if period_length < min_duration_samples:
                filtered_mask.iloc[current_period_start:] = False
        
        return filtered_mask
    
    def _calculate_filter_stats(self, 
                               high_confidence_mask: pd.Series,
                               post_drop_mask: pd.Series,
                               training_mask: pd.Series,
                               confidence: pd.Series):
        """Calculate and store filtering statistics."""
        total_samples = len(confidence)
        high_conf_samples = high_confidence_mask.sum()
        post_drop_samples = post_drop_mask.sum()
        training_samples = training_mask.sum()
        
        # Calculate overlap (samples that are both high confidence and post-drop)
        overlap_samples = (high_confidence_mask & post_drop_mask).sum()
        
        # Calculate coverage statistics
        high_conf_coverage = high_conf_samples / total_samples
        post_drop_coverage = post_drop_samples / total_samples
        training_coverage = training_samples / total_samples
        
        # Confidence statistics
        avg_confidence = confidence.mean()
        avg_training_confidence = confidence[training_mask].mean()
        
        self.filter_stats = {
            'total_samples': total_samples,
            'high_confidence_samples': high_conf_samples,
            'post_drop_samples': post_drop_samples,
            'training_samples': training_samples,
            'overlap_samples': overlap_samples,
            'high_confidence_coverage': high_conf_coverage,
            'post_drop_coverage': post_drop_coverage,
            'training_coverage': training_coverage,
            'avg_confidence': avg_confidence,
            'avg_training_confidence': avg_training_confidence
        }
        
        # Log statistics
        tprint_info("📊 Filter Statistics:")
        tprint_info(f"   → Total samples: {total_samples:,}")
        tprint_info(f"   → Training samples: {training_samples:,} ({training_coverage:.1%})")
        tprint_info(f"   → High confidence: {high_conf_samples:,} ({high_conf_coverage:.1%})")
        tprint_info(f"   → Post-drop extension: {post_drop_samples:,} ({post_drop_coverage:.1%})")
        tprint_info(f"   → Overlap: {overlap_samples:,}")
        tprint_info(f"   → Avg confidence: {avg_confidence:.3f}")
        tprint_info(f"   → Avg training confidence: {avg_training_confidence:.3f}")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        return self.filter_stats.copy()
    
    def get_last_filter_result(self) -> Optional[Dict[str, Any]]:
        """Get the last filtering result."""
        return self.last_filter_result
    
    def visualize_filtering_result(self, 
                                  confidence: pd.Series,
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize the filtering result (requires matplotlib).
        
        Args:
            confidence: Confidence series used for filtering
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if self.last_filter_result is None:
                self.logger.warning("No filtering result available for visualization")
                return
            
            mask = self.last_filter_result['mask']
            high_conf_mask = self.last_filter_result['high_confidence_mask']
            post_drop_mask = self.last_filter_result['post_drop_mask']
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot confidence
            ax.plot(confidence.index, confidence.values, 'b-', alpha=0.7, label='Confidence')
            ax.axhline(y=self.config.confidence_threshold, color='r', linestyle='--', 
                      alpha=0.7, label=f'Threshold ({self.config.confidence_threshold})')
            
            # Highlight high confidence periods
            high_conf_data = confidence[high_conf_mask]
            ax.fill_between(high_conf_data.index, 0, high_conf_data.values, 
                          alpha=0.3, color='green', label='High Confidence')
            
            # Highlight post-drop periods
            post_drop_data = confidence[post_drop_mask]
            ax.fill_between(post_drop_data.index, 0, post_drop_data.values, 
                          alpha=0.3, color='orange', label='Post-Drop Extension')
            
            # Mark drop points
            drop_points = self.last_filter_result['drop_points']
            for drop_point in drop_points:
                ax.axvline(x=drop_point, color='red', linestyle=':', alpha=0.7)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Confidence')
            ax.set_title('Tactician Training Filter Results')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis for datetime
            if isinstance(confidence.index, pd.DatetimeIndex):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"📊 Filter visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")


def create_tactician_training_filter(
    confidence_threshold: float = 0.5,
    post_drop_window_minutes: int = 45,
    enable_smoothing: bool = True,
    smoothing_window: int = 5
) -> TacticianTrainingFilter:
    """
    Create a Tactician training filter with specified parameters.
    
    Args:
        confidence_threshold: Minimum confidence for high-confidence periods
        post_drop_window_minutes: Minutes to extend training after confidence drops
        enable_smoothing: Whether to smooth confidence data
        smoothing_window: Window size for confidence smoothing
        
    Returns:
        Configured TacticianTrainingFilter instance
    """
    config = TacticianFilterConfig(
        confidence_threshold=confidence_threshold,
        post_drop_window_minutes=post_drop_window_minutes,
        enable_smoothing=enable_smoothing,
        smoothing_window=smoothing_window
    )
    
    return TacticianTrainingFilter(config)


def apply_tactician_filtering(
    data: pd.DataFrame,
    analyst_confidence: pd.Series,
    confidence_threshold: float = 0.5,
    post_drop_window_minutes: int = 45
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply Tactician filtering to training data.
    
    Args:
        data: Training data DataFrame
        analyst_confidence: Analyst confidence scores
        confidence_threshold: Confidence threshold for filtering
        post_drop_window_minutes: Post-drop window in minutes
        
    Returns:
        Tuple of (filtered_data, filter_stats)
    """
    filter_component = create_tactician_training_filter(
        confidence_threshold=confidence_threshold,
        post_drop_window_minutes=post_drop_window_minutes
    )
    
    training_mask = filter_component.create_training_mask(analyst_confidence, data.index)
    filtered_data = data[training_mask].copy()
    filter_stats = filter_component.get_filter_stats()
    
    return filtered_data, filter_stats


if __name__ == '__main__':
    # Test the Tactician training filter
    print("🎯 Testing Tactician Training Filter")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Create realistic confidence pattern
    confidence = pd.Series(0.0, index=dates)
    
    # Add some high confidence periods
    confidence.iloc[100:150] = 0.8  # High confidence period 1
    confidence.iloc[300:350] = 0.9  # High confidence period 2
    confidence.iloc[500:550] = 0.7  # High confidence period 3
    
    # Add some medium confidence periods
    confidence.iloc[200:250] = 0.3  # Medium confidence (below threshold)
    confidence.iloc[400:450] = 0.4  # Medium confidence (below threshold)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(confidence))
    confidence = np.clip(confidence + noise, 0.0, 1.0)
    
    # Test filtering
    filter_component = create_tactician_training_filter(
        confidence_threshold=0.5,
        post_drop_window_minutes=45
    )
    
    training_mask = filter_component.create_training_mask(confidence)
    
    print(f"✅ Filtering completed:")
    print(f"   Selected {training_mask.sum()}/{len(training_mask)} samples for training")
    print(f"   Coverage: {training_mask.sum()/len(training_mask):.1%}")
    
    stats = filter_component.get_filter_stats()
    print(f"   High confidence samples: {stats['high_confidence_samples']}")
    print(f"   Post-drop extension samples: {stats['post_drop_samples']}")
    print(f"   Average confidence: {stats['avg_confidence']:.3f}")
    
    print('✅ Tactician Training Filter test completed!')