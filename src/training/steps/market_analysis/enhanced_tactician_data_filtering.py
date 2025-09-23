"""
Enhanced Tactician Data Filtering Component

This component implements the enhanced data filtering logic for Tactician training:
1. Train on all samples where Analyst gives confidence score > 0.5
2. Include the next 45 minutes after Analyst confidence drops below 0.5
3. This simulates real trading conditions where Tactician may open a position after
   Analyst green light and "ride" it as long as short-term expectations are good

Enhanced Features:
- Time-based filtering with configurable window (45 minutes)
- Confidence threshold filtering (0.5)
- Handles temporal sequences properly
- Memory-efficient processing
- Comprehensive error handling and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import timedelta
import traceback

# Enhanced logging imports
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_structured,
        LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

# Common utilities imports
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics, get_memory_usage
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False

# Math validation imports
try:
    from src.utils.math_validation import (
        safe_divide, validate_finite, validate_positive, validate_range
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

# Initialize logger
logger = logging.getLogger(__name__)


class EnhancedTacticianDataFilter:
    """
    Enhanced data filtering component for Tactician training with time-based logic.

    Key Features:
    - Filters data based on Analyst confidence > 0.5
    - Includes next 45 minutes after confidence drops below 0.5
    - Handles temporal sequences and maintains time continuity
    - Memory-efficient processing with optional chunking
    - Comprehensive validation and error handling
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        ride_duration_minutes: int = 45,
        enable_memory_optimization: bool = True,
        chunk_size: Optional[int] = None,
        validate_inputs: bool = True
    ):
        """
        Initialize the enhanced data filter.

        Args:
            confidence_threshold: Minimum Analyst confidence score for training (default: 0.5)
            ride_duration_minutes: How many minutes to include after confidence drops (default: 45)
            enable_memory_optimization: Whether to use memory-efficient processing
            chunk_size: Size of data chunks for memory optimization (None = auto)
            validate_inputs: Whether to validate input data
        """
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Initializing Enhanced Tactician Data Filter")
            tprint_debug(f"   Confidence threshold: {confidence_threshold}")
            tprint_debug(f"   Ride duration: {ride_duration_minutes} minutes")
            tprint_debug(f"   Memory optimization: {enable_memory_optimization}")

        self.confidence_threshold = confidence_threshold
        self.ride_duration_minutes = ride_duration_minutes
        self.enable_memory_optimization = enable_memory_optimization
        self.chunk_size = chunk_size
        self.validate_inputs = validate_inputs

        # Initialize state
        self._validate_configuration()
        self._initialize_components()

        if TPRINT_AVAILABLE:
            tprint_success("✅ Enhanced Tactician Data Filter initialized")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        try:
            if not (0.0 < self.confidence_threshold <= 1.0):
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")

            if not (0 < self.ride_duration_minutes <= 1440):  # Max 24 hours
                raise ValueError(f"Ride duration must be between 1 and 1440 minutes, got {self.ride_duration_minutes}")

            if self.chunk_size is not None and self.chunk_size <= 0:
                raise ValueError(f"Chunk size must be positive or None, got {self.chunk_size}")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Configuration validation failed: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize internal components and state."""
        self.filtering_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'green_light_samples': 0,
            'ride_samples': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }

        self.last_filtering_result = None

        if TPRINT_AVAILABLE:
            tprint_debug("✅ Internal components initialized")

    def filter_training_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        confidence_scores: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Filter training data using enhanced logic.

        Args:
            data: Input features (DataFrame or numpy array)
            confidence_scores: Analyst confidence scores for each sample
            timestamps: Timestamps for each sample (required for time-based filtering)
            return_stats: Whether to return detailed statistics

        Returns:
            Dictionary containing:
            - 'filtered_data': Filtered data array
            - 'filtered_indices': Indices of selected samples
            - 'filtering_stats': Detailed statistics (if requested)
            - 'filtering_mask': Boolean mask of selected samples
        """
        start_time = pd.Timestamp.now()
        if TPRINT_AVAILABLE:
            tprint_info("🔍 Starting enhanced data filtering for Tactician training")

        try:
            # Input validation
            if self.validate_inputs:
                self._validate_inputs(data, confidence_scores, timestamps)

            # Convert to numpy arrays for efficient processing
            data_array = self._convert_to_array(data)
            confidence_array = np.asarray(confidence_scores, dtype=np.float64)

            # Validate array consistency
            if data_array.shape[0] != len(confidence_array):
                raise ValueError(f"Data ({data_array.shape[0]}) and confidence ({len(confidence_array)}) length mismatch")

            # Create filtering mask
            filtering_mask = self._create_filtering_mask(confidence_array, timestamps)

            # Apply filtering
            filtered_data = data_array[filtering_mask]
            filtered_indices = np.where(filtering_mask)[0]

            # Update statistics
            self._update_filtering_stats(
                original_samples=len(data_array),
                filtered_samples=len(filtered_data),
                green_light_samples=np.sum(confidence_array >= self.confidence_threshold),
                ride_samples=len(filtered_data) - np.sum(confidence_array >= self.confidence_threshold)
            )

            # Log results
            if TPRINT_AVAILABLE:
                self._log_filtering_results()

            result = {
                'filtered_data': filtered_data,
                'filtered_indices': filtered_indices,
                'filtering_mask': filtering_mask
            }

            if return_stats:
                result['filtering_stats'] = self.filtering_stats.copy()

            self.last_filtering_result = result

            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Data filtering completed: {len(filtered_data)}/{len(data_array)} samples selected")

            return result

        except Exception as e:
            error_msg = f"Data filtering failed: {str(e)}"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
                tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_inputs(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        confidence_scores: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> None:
        """Validate input parameters."""
        try:
            if data is None:
                raise ValueError("Data cannot be None")

            if confidence_scores is None:
                raise ValueError("Confidence scores cannot be None")

            confidence_array = np.asarray(confidence_scores)
            if len(confidence_array) == 0:
                raise ValueError("Confidence scores cannot be empty")

            if not (0.0 <= confidence_array.min() and confidence_array.max() <= 1.0):
                raise ValueError(f"Confidence scores must be in [0, 1] range, got [{confidence_array.min():.3f}, {confidence_array.max():.3f}]")

            if timestamps is not None:
                timestamp_array = np.asarray(timestamps)
                if len(timestamp_array) != len(confidence_array):
                    raise ValueError(f"Timestamps ({len(timestamp_array)}) and confidence ({len(confidence_array)}) length mismatch")

                # Try to convert to datetime for validation
                try:
                    pd.to_datetime(timestamp_array)
                except Exception:
                    raise ValueError("Timestamps must be convertible to datetime")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Input validation failed: {e}")
            raise

    def _convert_to_array(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _create_filtering_mask(
        self,
        confidence_array: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create boolean mask for filtering based on confidence and time logic.

        Args:
            confidence_array: Array of confidence scores
            timestamps: Array of timestamps (required for time-based filtering)

        Returns:
            Boolean mask indicating which samples to include
        """
        try:
            # Step 1: Basic confidence filtering (> 0.5)
            confidence_mask = confidence_array >= self.confidence_threshold

            # Step 2: Time-based filtering (next 45 min after confidence drops)
            if timestamps is not None:
                time_mask = self._create_time_based_mask(confidence_array, timestamps)
                # Combine both masks
                final_mask = confidence_mask | time_mask
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ No timestamps provided, using confidence-only filtering")
                final_mask = confidence_mask

            return final_mask

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to create filtering mask: {e}")
            raise

    def _create_time_based_mask(
        self,
        confidence_array: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Create mask for samples in the 45-minute window after confidence drops below 0.5.

        Args:
            confidence_array: Array of confidence scores
            timestamps: Array of timestamps

        Returns:
            Boolean mask for time-based filtering
        """
        try:
            # Convert timestamps to pandas datetime for easier manipulation
            timestamp_series = pd.to_datetime(timestamps)
            time_mask = np.zeros(len(confidence_array), dtype=bool)

            # Find points where confidence drops below threshold
            confidence_below = confidence_array < self.confidence_threshold
            drop_points = np.where(confidence_below)[0]

            # For each drop point, include the next 45 minutes
            ride_duration = pd.Timedelta(minutes=self.ride_duration_minutes)

            for drop_idx in drop_points:
                drop_time = timestamp_series.iloc[drop_idx]
                end_time = drop_time + ride_duration

                # Find all samples within the ride duration window
                mask_in_window = (timestamp_series >= drop_time) & (timestamp_series <= end_time)
                time_mask = time_mask | mask_in_window

            return time_mask

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Time-based masking failed: {e}, falling back to confidence-only")
            return np.zeros(len(confidence_array), dtype=bool)

    def _update_filtering_stats(
        self,
        original_samples: int,
        filtered_samples: int,
        green_light_samples: int,
        ride_samples: int
    ) -> None:
        """Update filtering statistics."""
        self.filtering_stats.update({
            'total_samples': original_samples,
            'filtered_samples': filtered_samples,
            'green_light_samples': green_light_samples,
            'ride_samples': ride_samples,
            'filtering_ratio': filtered_samples / max(original_samples, 1),
            'green_light_ratio': green_light_samples / max(original_samples, 1),
            'ride_ratio': ride_samples / max(filtered_samples, 1)
        })

        # Update memory usage if available
        if COMMON_UTILITIES_AVAILABLE:
            try:
                self.filtering_stats['memory_usage_mb'] = get_memory_usage() / (1024 * 1024)
            except Exception:
                pass

    def _log_filtering_results(self) -> None:
        """Log detailed filtering results."""
        if not TPRINT_AVAILABLE:
            return

        stats = self.filtering_stats
        tprint_info("📊 Enhanced Data Filtering Results:")
        tprint_info(f"   Total samples: {stats['total_samples']","}")
        tprint_info(f"   Filtered samples: {stats['filtered_samples']","}")
        tprint_info(f"   Filtering ratio: {stats['filtering_ratio']:.2%}")
        tprint_info(f"   Green light samples: {stats['green_light_samples']} ({stats['green_light_ratio']:.2%})")
        tprint_info(f"   Ride samples: {stats['ride_samples']} ({stats['ride_ratio']:.2%})")

    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics."""
        return self.filtering_stats.copy()

    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        self.filtering_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'green_light_samples': 0,
            'ride_samples': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
        self.last_filtering_result = None

        if TPRINT_AVAILABLE:
            tprint_info("📊 Filtering statistics reset")

    def cleanup_resources(self) -> None:
        """Clean up resources and reset state."""
        self.reset_statistics()

        if TPRINT_AVAILABLE:
            tprint_info("🧹 Enhanced Data Filter resources cleaned up")


# Convenience functions for easy integration
def create_enhanced_data_filter(
    confidence_threshold: float = 0.5,
    ride_duration_minutes: int = 45,
    **kwargs
) -> EnhancedTacticianDataFilter:
    """Create an enhanced data filter instance."""
    return EnhancedTacticianDataFilter(
        confidence_threshold=confidence_threshold,
        ride_duration_minutes=ride_duration_minutes,
        **kwargs
    )


def filter_tactician_training_data(
    data: Union[pd.DataFrame, np.ndarray],
    confidence_scores: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5,
    ride_duration_minutes: int = 45,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to filter Tactician training data with enhanced logic.

    Args:
        data: Input features
        confidence_scores: Analyst confidence scores
        timestamps: Timestamps for each sample
        confidence_threshold: Minimum confidence for training
        ride_duration_minutes: Duration to include after confidence drops
        **kwargs: Additional arguments for filter configuration

    Returns:
        Dictionary with filtered data and statistics
    """
    filter_instance = create_enhanced_data_filter(
        confidence_threshold=confidence_threshold,
        ride_duration_minutes=ride_duration_minutes,
        **kwargs
    )

    return filter_instance.filter_training_data(data, confidence_scores, timestamps)


if __name__ == "__main__":
    # Example usage
    print("Enhanced Tactician Data Filtering Component")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    sample_data = np.random.randn(n_samples, 10)
    confidence_scores = np.random.uniform(0.3, 0.8, n_samples)
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')

    print(f"Sample data shape: {sample_data.shape}")
    print(f"Confidence range: [{confidence_scores.min():.3f}, {confidence_scores.max():.3f}]")

    # Create filter
    data_filter = create_enhanced_data_filter(
        confidence_threshold=0.5,
        ride_duration_minutes=45
    )

    # Filter data
    result = data_filter.filter_training_data(
        data=sample_data,
        confidence_scores=confidence_scores,
        timestamps=timestamps.values
    )

    print(f"\nFiltering Results:")
    print(f"Original samples: {n_samples}")
    print(f"Filtered samples: {result['filtered_data'].shape[0]}")
    print(f"Filtering ratio: {result['filtered_data'].shape[0]/n_samples:.2%}")

    print("\nFilter Configuration:")
    print(f"Confidence threshold: {data_filter.confidence_threshold}")
    print(f"Ride duration: {data_filter.ride_duration_minutes} minutes")

    print("\n✅ Enhanced Tactician Data Filtering ready for integration!")