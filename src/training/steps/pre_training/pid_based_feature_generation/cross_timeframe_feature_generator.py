"""
Cross Timeframe Feature Generator using Partial Information Decomposition

This module generates data-driven cross-timeframe features using PID analysis to identify
the most relevant cross-timeframe relationships up to 50 features.

Key Features:
- Uses optimized lookback periods from feature_lookback_optimization
- Leverages matrix_operations/ for all calculations
- Generates up to 50 cross-timeframe features based on PID analysis
- Comprehensive validation and error handling
- Hardware-optimized computations
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Import tprint for consistent logging
from src.utils.tprint import tprint

tprint("🔧 Loading cross timeframe feature generator...")

# Core dependencies with fallback support
try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    sliding_window_view = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import PID utilities
try:
    from src.training.utils.feature_selection.partial_information_decompositor import (
        PartialInformationDecompositor, PIDConfig, PIDResult
    )
    PID_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PID utilities not available: {e}")
    PID_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('CrossTimeframeFeatureGenerator')
except ImportError:
    logger = logging.getLogger('CrossTimeframeFeatureGenerator')
    logger.setLevel(logging.INFO)

# Import configuration constants
try:
    from .constants import (
        COMPUTATION, CROSS_TIMEFRAME, VALIDATION, 
        get_rolling_window_size, validate_lookback_period
    )
    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False
    # Fallback constants
    class _Constants:
        DEFAULT_ROLLING_WINDOW = 20
        MIN_ROLLING_WINDOW = 3
        MAX_LAG_PERIODS = 5
        STANDARD_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    COMPUTATION = _Constants()
    CROSS_TIMEFRAME = _Constants()
    
    def get_rolling_window_size(data_length, preferred=None):
        return min(20, max(3, data_length // 4))
    
    def validate_lookback_period(period, feature_type=None):
        return 1 <= period <= 200


class CrossTimeframeType(Enum):
    """Types of cross-timeframe features."""
    RATIO = "ratio"
    DIFFERENCE = "difference"
    CORRELATION = "correlation"
    LAG_CORRELATION = "lag_correlation"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND_ALIGNMENT = "trend_alignment"
    REGIME_CONSISTENCY = "regime_consistency"


@dataclass
class CrossTimeframeConfig:
    """Configuration for cross-timeframe feature generation."""
    # PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Feature Limits
    max_cross_timeframe_features: int = 50
    max_timeframe_pairs: int = 25
    max_lag_periods: int = 5
    
    # Timeframe Settings
    timeframes: List[str] = field(default_factory=lambda: CROSS_TIMEFRAME.STANDARD_TIMEFRAMES)
    
    # Cross Timeframe Types
    cross_timeframe_types: List[CrossTimeframeType] = field(default_factory=lambda: [
        CrossTimeframeType.RATIO,
        CrossTimeframeType.DIFFERENCE,
        CrossTimeframeType.CORRELATION,
        CrossTimeframeType.LAG_CORRELATION,
        CrossTimeframeType.MOMENTUM
    ])
    
    # Computational Settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Validation
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.95
    significance_threshold: float = 0.05
    
    # Hardware Optimization
    chunk_size_mb: int = 256
    max_memory_percent: float = 0.7


@dataclass
class CrossTimeframeResult:
    """Result of cross-timeframe feature generation."""
    cross_timeframe_features: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    cross_timeframe_scores: Dict[str, float] = field(default_factory=dict)
    pid_analysis: Optional[PIDResult] = None
    
    # Metadata
    total_features_generated: int = 0
    execution_time: float = 0.0
    optimization_used: bool = False
    matrix_ops_used: bool = False
    
    # Quality Metrics
    average_correlation: float = 0.0
    feature_stability_score: float = 0.0
    timeframe_coverage: float = 0.0
    lag_effectiveness: float = 0.0


class CrossTimeframeFeatureGenerator:
    """
    Cross Timeframe Feature Generator using Partial Information Decomposition.
    
    Generates data-driven cross-timeframe features using PID analysis to identify
    the most relevant cross-timeframe relationships up to 50 features.
    """
    
    def __init__(self, config: Optional[CrossTimeframeConfig] = None):
        """Initialize the cross-timeframe feature generator."""
        self.config = config or CrossTimeframeConfig()
        self.logger = logger.getChild('CrossTimeframeFeatureGenerator')

        # Initialize components
        self._initialize_components()

        # Rolling window cache for reusing statistics across feature computations
        self._rolling_cache: Dict[Tuple[str, Tuple[Any, ...]], np.ndarray] = {}
        self._rolling_cache_hits: int = 0
        self._rolling_cache_misses: int = 0

        self.logger.info("🔧 CrossTimeframeFeatureGenerator initialized")
        self.logger.info(f"📊 Max cross-timeframe features: {self.config.max_cross_timeframe_features}")
        self.logger.info(f"📊 Timeframes: {self.config.timeframes}")
        self.logger.info(f"📊 Cross-timeframe types: {[t.value for t in self.config.cross_timeframe_types]}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize PID decompositor
        if PID_AVAILABLE:
            pid_config = PIDConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                cross_timeframe_threshold=self.config.synergy_threshold,
                max_timeframe_lag=self.config.max_lag_periods,
                max_interaction_features=self.config.max_cross_timeframe_features
            )
            self.pid_decompositor = PartialInformationDecompositor(pid_config)
            self.logger.info("✅ PID Decompositor initialized")
        else:
            self.pid_decompositor = None
            self.logger.warning("⚠️ PID Decompositor not available")
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            self.logger.info("✅ Matrix Operations initialized")
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Matrix Operations not available")
    
    async def generate_cross_timeframe_features(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        optimized_lookback_periods: Optional[Dict[str, int]] = None,
        target: Optional[np.ndarray] = None
    ) -> CrossTimeframeResult:
        """
        Generate cross-timeframe features using PID analysis.
        
        Args:
            data: Input feature matrix
            feature_names: List of feature names
            optimized_lookback_periods: Optimized lookback periods from feature_lookback_optimization
            target: Target variable for PID analysis (optional) - now uses multi-horizon profit probabilities
            
        Returns:
            CrossTimeframeResult with generated cross-timeframe features
        """
        start_time = time.time()
        self.logger.info("🔧 Starting cross-timeframe feature generation...")
        
        result = CrossTimeframeResult()
        
        try:
            # Convert data to numpy array if needed
            if isinstance(data, pd.DataFrame):
                X = data.values
                if feature_names is None:
                    feature_names = list(data.columns)
            else:
                X = data
            
            self.logger.info(f"📊 Input data shape: {X.shape}")
            self.logger.info(f"📊 Feature count: {len(feature_names)}")
            
            # Apply optimized lookback periods if available
            if optimized_lookback_periods:
                X, feature_names = self._apply_optimized_lookback_periods(
                    X, feature_names, optimized_lookback_periods
                )
                result.optimization_used = True
                self.logger.info("✅ Applied optimized lookback periods")
            
            # Identify timeframe features
            timeframe_features = self._identify_timeframe_features(feature_names)
            self.logger.info(f"📊 Found {len(timeframe_features)} timeframe features")
            
            if len(timeframe_features) < 2:
                self.logger.warning("⚠️ Insufficient timeframe features for cross-timeframe analysis")
                return result
            
            # Generate synthetic timeframe data if needed
            if any('_tf' in tf for tf in timeframe_features):
                self.logger.info("🔧 Generating synthetic multi-timeframe data...")
                X_extended, extended_feature_names = self._create_synthetic_timeframe_data(
                    X, feature_names, timeframe_features
                )
                # Update data and feature names to include synthetic features
                X = X_extended
                feature_names = extended_feature_names
                self.logger.info(f"📊 Extended data shape: {X.shape} with synthetic timeframe features")
            
            # Perform PID analysis
            if self.pid_decompositor and target is not None:
                self.logger.info("🔍 Performing PID analysis...")
                pid_result = self.pid_decompositor.decompose_information(X, target, feature_names)
                result.pid_analysis = pid_result
                
                # Extract significant cross-timeframe relationships from PID
                significant_pairs = self._extract_significant_cross_timeframe_relationships(
                    pid_result, timeframe_features
                )
                self.logger.info(f"📊 Found {len(significant_pairs)} significant cross-timeframe relationships")
            else:
                # Fallback: use correlation-based approach
                self.logger.info("📊 Using correlation-based cross-timeframe detection")
                significant_pairs = self._correlation_based_cross_timeframe_detection(
                    X, feature_names, timeframe_features
                )
                self.logger.info(f"📊 Found {len(significant_pairs)} correlation-based cross-timeframe relationships")
            
            # Generate cross-timeframe features
            self.logger.info("🔧 Generating cross-timeframe features...")
            cross_timeframe_features, cross_timeframe_names = self._generate_cross_timeframe_matrix(
                X, feature_names, significant_pairs
            )
            
            # Calculate cross-timeframe scores
            cross_timeframe_scores = self._calculate_cross_timeframe_scores(
                cross_timeframe_features, cross_timeframe_names, target
            )
            
            # Store results
            result.cross_timeframe_features = {
                name: feature for name, feature in zip(cross_timeframe_names, cross_timeframe_features.T)
            }
            result.feature_names = cross_timeframe_names
            result.cross_timeframe_scores = cross_timeframe_scores
            result.total_features_generated = len(cross_timeframe_names)
            result.matrix_ops_used = self.matrix_ops is not None
            
            # Calculate quality metrics
            result.average_correlation = self._calculate_average_correlation(cross_timeframe_features)
            result.feature_stability_score = self._calculate_stability_score(cross_timeframe_features)
            result.timeframe_coverage = self._calculate_timeframe_coverage(cross_timeframe_names)
            result.lag_effectiveness = self._calculate_lag_effectiveness(cross_timeframe_names)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(f"✅ Cross-timeframe feature generation completed in {execution_time:.3f}s")
            self.logger.info(f"📊 Generated {result.total_features_generated} cross-timeframe features")
            self.logger.info(f"📊 Average correlation: {result.average_correlation:.3f}")
            self.logger.info(f"📊 Stability score: {result.feature_stability_score:.3f}")
            self.logger.info(f"📊 Timeframe coverage: {result.timeframe_coverage:.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.error(f"❌ Cross-timeframe feature generation failed: {e}")
            self.logger.error(f"❌ Error details: {traceback.format_exc()}")
            
            return result
    
    def _apply_optimized_lookback_periods(
        self,
        X: np.ndarray,
        feature_names: List[str],
        optimized_lookback_periods: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply optimized lookback periods to features."""
        try:
            if not optimized_lookback_periods:
                return X, feature_names

            self.logger.info(
                f"📊 Applying optimized lookback periods: {optimized_lookback_periods}"
            )

            if X.ndim != 2:
                self.logger.warning(
                    "⚠️ Expected 2D feature matrix when applying lookback periods"
                )
                return X, feature_names

            feature_index_map = {name: idx for idx, name in enumerate(feature_names)}

            valid_periods: Dict[str, Tuple[int, int]] = {}
            for feature_name, period in optimized_lookback_periods.items():
                if feature_name not in feature_index_map:
                    self.logger.debug(
                        "ℹ️ Optimized period provided for unknown feature '%s'", feature_name
                    )
                    continue

                try:
                    period_value = int(period)
                except (TypeError, ValueError):
                    self.logger.debug(
                        "ℹ️ Invalid lookback period '%s' for feature '%s'", period, feature_name
                    )
                    continue

                if period_value < 1:
                    self.logger.debug(
                        "ℹ️ Non-positive lookback period %s for feature '%s'", period_value, feature_name
                    )
                    continue

                if 'validate_lookback_period' in globals():
                    try:
                        if not validate_lookback_period(period_value):
                            self.logger.debug(
                                "ℹ️ Lookback period %s for feature '%s' failed validation",
                                period_value,
                                feature_name,
                            )
                            continue
                    except Exception as validation_error:  # pragma: no cover - defensive
                        self.logger.debug(
                            "ℹ️ Lookback validation failed for feature '%s': %s",
                            feature_name,
                            validation_error,
                        )
                        continue

                valid_periods[feature_name] = (feature_index_map[feature_name], period_value)

            if not valid_periods:
                return X, feature_names

            # Compute rolling transformations grouped by common window sizes to
            # leverage vectorized operations.
            transformed_columns: Dict[str, np.ndarray] = {}
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(X, columns=feature_names)
                for period_value in sorted({period for _, period in valid_periods.values()}):
                    period_columns = [
                        name
                        for name, (_, p_value) in valid_periods.items()
                        if p_value == period_value
                    ]
                    if not period_columns:
                        continue

                    rolled_df = df[period_columns].rolling(
                        window=period_value,
                        min_periods=1,
                    ).mean()
                    for column_name in period_columns:
                        transformed_columns[column_name] = rolled_df[column_name].to_numpy(copy=True)
            else:
                for feature_name, (idx, period_value) in valid_periods.items():
                    column_data = X[:, idx].astype(float, copy=False)

                    if period_value == 1:
                        transformed_columns[feature_name] = column_data.copy()
                        continue

                    # Vectorized cumulative sum approach to compute rolling mean
                    cumsum = np.cumsum(column_data, dtype=float)
                    rolling_sum = cumsum.copy()
                    rolling_sum[period_value:] = (
                        cumsum[period_value:] - cumsum[:-period_value]
                    )
                    counts = np.minimum(
                        np.arange(1, column_data.shape[0] + 1),
                        period_value,
                    )
                    transformed = rolling_sum / counts
                    transformed_columns[feature_name] = transformed

            updated_feature_arrays: List[np.ndarray] = []
            updated_feature_names: List[str] = []

            for feature_name in feature_names:
                if feature_name in valid_periods and feature_name in transformed_columns:
                    period_value = valid_periods[feature_name][1]
                    updated_feature_arrays.append(transformed_columns[feature_name])
                    updated_feature_names.append(f"{feature_name}_lb{period_value}")
                else:
                    updated_feature_arrays.append(X[:, feature_index_map[feature_name]])
                    updated_feature_names.append(feature_name)

            transformed_matrix = np.column_stack(updated_feature_arrays)
            return transformed_matrix, updated_feature_names

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply optimized lookback periods: {e}")
            return X, feature_names
    
    def _identify_timeframe_features(self, feature_names: List[str]) -> List[str]:
        """Identify features that contain timeframe information or create synthetic ones."""
        timeframe_features = []
        
        # First, check for explicit timeframe features
        for feature_name in feature_names:
            for timeframe in self.config.timeframes:
                if timeframe in feature_name.lower():
                    timeframe_features.append(feature_name)
                    break
        
        # If no explicit timeframe features found, create synthetic ones
        if len(timeframe_features) == 0:
            self.logger.info("📊 No explicit timeframe features found, creating synthetic multi-timeframe features")
            
            # Select base features that are good candidates for multi-timeframe analysis
            base_features = []
            for feature_name in feature_names:
                # Include price-related, volume-related, and return features
                if any(keyword in feature_name.lower() for keyword in 
                       ['close', 'open', 'high', 'low', 'volume', 'return', 'price']):
                    base_features.append(feature_name)
            
            # Create synthetic timeframe feature names (we'll generate the actual features later)
            synthetic_timeframes = ['short', 'medium', 'long']  # Representing different aggregation periods
            for base_feature in base_features[:5]:  # Limit to top 5 base features
                for tf in synthetic_timeframes:
                    synthetic_name = f"{base_feature}_{tf}_tf"
                    timeframe_features.append(synthetic_name)
            
            self.logger.info(f"📊 Created {len(timeframe_features)} synthetic timeframe features")
        
        return timeframe_features
    
    def _create_synthetic_timeframe_data(self, X: np.ndarray, feature_names: List[str],
                                       timeframe_features: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create synthetic multi-timeframe features from single timeframe data."""
        try:
            # Create a mapping of base features to their indices
            base_feature_indices = {}
            for tf_name in timeframe_features:
                if '_tf' in tf_name:
                    # Extract base feature name by removing the synthetic suffix
                    base_name = tf_name
                    for suffix in ('_short_tf', '_medium_tf', '_long_tf'):
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break

                    base_name = base_name.rstrip('_')

                    if base_name in feature_names:
                        base_feature_indices[tf_name] = feature_names.index(base_name)
            
            # Create synthetic timeframe features using different aggregation periods
            synthetic_features = []
            synthetic_names = []
            
            for tf_name, base_idx in base_feature_indices.items():
                base_data = X[:, base_idx]
                
                base_feature_name = feature_names[base_idx] if base_idx < len(feature_names) else tf_name

                if 'short_tf' in tf_name:
                    # Short timeframe: 5-period rolling mean (represents ~5min aggregation)
                    synthetic_data = self._rolling_aggregation(
                        base_data,
                        window=5,
                        agg_type='mean',
                        cache_params=('synthetic_mean', base_feature_name, 5)
                    )
                elif 'medium_tf' in tf_name:
                    # Medium timeframe: 15-period rolling mean (represents ~15min aggregation)
                    synthetic_data = self._rolling_aggregation(
                        base_data,
                        window=15,
                        agg_type='mean',
                        cache_params=('synthetic_mean', base_feature_name, 15)
                    )
                elif 'long_tf' in tf_name:
                    # Long timeframe: 60-period rolling mean (represents ~1hour aggregation)
                    synthetic_data = self._rolling_aggregation(
                        base_data,
                        window=60,
                        agg_type='mean',
                        cache_params=('synthetic_mean', base_feature_name, 60)
                    )
                else:
                    continue
                
                synthetic_features.append(synthetic_data)
                synthetic_names.append(tf_name)
            
            if synthetic_features:
                # Combine original features with synthetic timeframe features
                synthetic_matrix = np.column_stack(synthetic_features)
                X_extended = np.column_stack([X, synthetic_matrix])
                extended_names = feature_names + synthetic_names
                
                self.logger.info(f"📊 Created {len(synthetic_features)} synthetic timeframe features")
                return X_extended, extended_names
            else:
                self.logger.warning("⚠️ Failed to create synthetic timeframe features")
                return X, feature_names
                
        except Exception as e:
            self.logger.error(f"❌ Error creating synthetic timeframe data: {e}")
            return X, feature_names
    
    def _rolling_aggregation(
        self,
        data: np.ndarray,
        window: int,
        agg_type: str = 'mean',
        cache_params: Optional[Tuple[Any, ...]] = None
    ) -> np.ndarray:
        """Apply vectorized rolling aggregation to create synthetic timeframe data."""
        if not NUMPY_AVAILABLE:
            return data

        use_cache = cache_params is not None
        cached_result = None
        if use_cache:
            cached_result = self._cache_lookup('rolling_aggregation', cache_params)
            if cached_result is not None:
                return cached_result

        try:
            if agg_type != 'mean':
                return data

            window = max(1, min(window, len(data)))

            if self.matrix_ops and getattr(self.matrix_ops, 'vectorized_core', None) is not None and PANDAS_AVAILABLE:
                df = pd.DataFrame({'value': data})
                df = self._prepare_dataframe_for_vectorization(df)
                try:
                    vectorized_df = self.matrix_ops.vectorized_core.vectorized_rolling_features(
                        df[['value']],
                        windows=[window],
                        features=['value']
                    )
                    column_name = f'value_rolling_mean_{window}'
                    if column_name in vectorized_df:
                        result = vectorized_df[column_name].to_numpy()
                    else:
                        result = vectorized_df['value'].rolling(window=window, min_periods=1).mean().to_numpy()
                except Exception:
                    result = pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()
            elif PANDAS_AVAILABLE:
                result = pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()
            else:
                cumsum = np.cumsum(data, dtype=float)
                result = cumsum / np.arange(1, len(data) + 1)
                if window < len(data):
                    numerator = cumsum[window:] - cumsum[:-window]
                    result[window - 1:] = numerator / window

            if use_cache:
                self._cache_store('rolling_aggregation', cache_params, result)
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ Rolling aggregation failed: {e}")
            return data
    
    def _extract_significant_cross_timeframe_relationships(
        self, 
        pid_result: PIDResult, 
        timeframe_features: List[str]
    ) -> List[Tuple[str, str]]:
        """Extract significant cross-timeframe relationships from PID analysis using dynamic thresholds."""
        # Use the PID decompositor's significant pairs directly if available
        if hasattr(pid_result, 'significant_pairs') and pid_result.significant_pairs:
            # Filter for timeframe features only
            timeframe_pairs = []
            for feat1, feat2 in pid_result.significant_pairs:
                if feat1 in timeframe_features and feat2 in timeframe_features:
                    # Check if features are from different timeframes  
                    if self._are_different_timeframes(feat1, feat2):
                        timeframe_pairs.append((feat1, feat2))
                        if len(timeframe_pairs) >= self.config.max_timeframe_pairs:
                            break
            
            self.logger.info(f"📊 Using PID significant pairs: {len(timeframe_pairs)} cross-timeframe relationships")
            return timeframe_pairs
        
        # Fallback: use dynamic threshold approach similar to main PID decompositor
        timeframe_synergy = {
            (feat1, feat2): score for (feat1, feat2), score in pid_result.synergy.items()
            if feat1 in timeframe_features and feat2 in timeframe_features
        }
        
        if not timeframe_synergy:
            self.logger.warning("⚠️ No timeframe synergy pairs found")
            return []
        
        # Use dynamic threshold: select top 25% of timeframe pairs
        total_pairs = len(timeframe_synergy)
        target_count = max(int(total_pairs * 0.25), 2)  # At least 2 pairs
        target_count = min(target_count, self.config.max_timeframe_pairs)
        
        # Sort by synergy score and take top relationships
        synergy_items = sorted(timeframe_synergy.items(), key=lambda x: x[1], reverse=True)
        
        significant_pairs = []
        for (feat1, feat2), synergy_score in synergy_items[:target_count]:
            # Check if features are from different timeframes
            if self._are_different_timeframes(feat1, feat2):
                significant_pairs.append((feat1, feat2))
        
        self.logger.info(f"📊 Dynamic threshold: selected {len(significant_pairs)} cross-timeframe relationships from {total_pairs} pairs")
        return significant_pairs
    
    def _are_different_timeframes(self, feat1: str, feat2: str) -> bool:
        """Check if two features are from different timeframes."""
        tf1 = self._extract_timeframe(feat1)
        tf2 = self._extract_timeframe(feat2)
        return tf1 is not None and tf2 is not None and tf1 != tf2
    
    def _extract_timeframe(self, feature_name: str) -> Optional[str]:
        """Extract timeframe from feature name."""
        # Check for original timeframes first
        for timeframe in self.config.timeframes:
            if timeframe in feature_name.lower():
                return timeframe
        
        # Check for synthetic timeframe indicators
        if 'short_tf' in feature_name.lower():
            return 'short_tf'
        elif 'medium_tf' in feature_name.lower():
            return 'medium_tf'
        elif 'long_tf' in feature_name.lower():
            return 'long_tf'
        
        return None
    
    def _correlation_based_cross_timeframe_detection(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        timeframe_features: List[str]
    ) -> List[Tuple[str, str]]:
        """Fallback correlation-based cross-timeframe detection."""
        try:
            if self.matrix_ops:
                # Use matrix operations for correlation calculation
                correlation_matrix = self.matrix_ops.safe_correlation_matrix(X)
            else:
                # Fallback to numpy correlation
                correlation_matrix = np.corrcoef(X.T)
            
            significant_pairs = []
            timeframe_features_in_data = [
                f for f in timeframe_features if f in feature_names
            ]

            if len(timeframe_features_in_data) < 2:
                self.logger.warning("⚠️ Insufficient timeframe features available after filtering")
                return []

            for i, feat1 in enumerate(timeframe_features_in_data):
                for j, feat2 in enumerate(timeframe_features_in_data[i+1:], i+1):
                    # Check if features are from different timeframes
                    if self._are_different_timeframes(feat1, feat2):
                        idx1 = feature_names.index(feat1)
                        idx2 = feature_names.index(feat2)
                        corr = abs(correlation_matrix[idx1, idx2])
                        
                        if (self.config.min_correlation_threshold <= corr <= 
                            self.config.max_correlation_threshold):
                            significant_pairs.append((feat1, feat2))
                            
                            if len(significant_pairs) >= self.config.max_timeframe_pairs:
                                break
                
                if len(significant_pairs) >= self.config.max_timeframe_pairs:
                    break
            
            return significant_pairs
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation-based cross-timeframe detection failed: {e}")
            return []
    
    def _generate_cross_timeframe_matrix(
        self,
        X: np.ndarray,
        feature_names: List[str],
        significant_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate cross-timeframe feature matrix."""
        cross_timeframe_features = []
        cross_timeframe_names = []
        
        for feat1, feat2 in significant_pairs:
            try:
                # Find feature indices
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)
                
                x1, x2 = X[:, idx1], X[:, idx2]
                
                # Generate different types of cross-timeframe features
                for cross_timeframe_type in self.config.cross_timeframe_types:
                    cross_tf_feat, cross_tf_name = self._create_cross_timeframe_feature(
                        x1, x2, feat1, feat2, cross_timeframe_type
                    )
                    
                    if cross_tf_feat is not None:
                        cross_timeframe_features.append(cross_tf_feat)
                        cross_timeframe_names.append(cross_tf_name)
                        
                        if len(cross_timeframe_names) >= self.config.max_cross_timeframe_features:
                            break
                
                if len(cross_timeframe_names) >= self.config.max_cross_timeframe_features:
                    break
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"⚠️ Failed to create cross-timeframe feature for ({feat1}, {feat2}): {e}")
                continue
        
        if cross_timeframe_features:
            features_matrix = np.column_stack(cross_timeframe_features)
            # Drop near-constant features (low variance)
            variances = np.var(features_matrix, axis=0)
            keep_mask = variances > 1e-12
            if not np.all(keep_mask):
                features_matrix = features_matrix[:, keep_mask]
                cross_timeframe_names = [name for name, keep in zip(cross_timeframe_names, keep_mask) if keep]
            return features_matrix, cross_timeframe_names
        else:
            return np.array([]).reshape(X.shape[0], 0), []

    def _cache_lookup(self, operation: str, params: Tuple[Any, ...]) -> Optional[np.ndarray]:
        """Retrieve cached rolling window computations when available."""
        if not params:
            return None

        key = (operation, params)
        cached = self._rolling_cache.get(key)
        if cached is not None:
            self._rolling_cache_hits += 1
            return cached

        self._rolling_cache_misses += 1
        return None

    def _cache_store(self, operation: str, params: Tuple[Any, ...], value: np.ndarray) -> None:
        """Store rolling window computations for reuse."""
        if not params:
            return

        key = (operation, params)
        self._rolling_cache[key] = value

    def _prepare_dataframe_for_vectorization(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Optimize DataFrame before passing it to matrix operations."""
        if not PANDAS_AVAILABLE or df is None:
            return df

        if self.matrix_ops and getattr(self.matrix_ops, 'vectorized_core', None) is not None:
            vectorized_core = self.matrix_ops.vectorized_core
            try:
                if hasattr(vectorized_core, 'optimize_dataframe_for_processing'):
                    return vectorized_core.optimize_dataframe_for_processing(df.copy())
                if hasattr(vectorized_core, 'optimize_dataframe'):
                    return vectorized_core.optimize_dataframe(df.copy())
            except Exception:
                pass

            try:
                from src.utils.matrix_operations.convenience import optimize_dataframe  # type: ignore

                return optimize_dataframe(df.copy())
            except Exception:
                return df

        return df

    def _create_cross_timeframe_feature(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        feat1: str,
        feat2: str, 
        cross_timeframe_type: CrossTimeframeType
    ) -> Tuple[Optional[np.ndarray], str]:
        """Create a specific type of cross-timeframe feature."""
        try:
            tf1 = self._extract_timeframe(feat1)
            tf2 = self._extract_timeframe(feat2)
            
            if cross_timeframe_type == CrossTimeframeType.RATIO:
                if self.matrix_ops:
                    feature = self.matrix_ops.batch_process(
                        np.column_stack([x1, x2]), 
                        'safe_divide', 
                        numerator=x1, 
                        denominator=x2, 
                        default_value=0.0
                    )
                else:
                    feature = np.divide(x1, x2, out=np.zeros_like(x1), where=(x2 != 0))
                name = f"{feat1}_to_{feat2}_ratio"
                
            elif cross_timeframe_type == CrossTimeframeType.DIFFERENCE:
                feature = x1 - x2
                name = f"{feat1}_minus_{feat2}"
                
            elif cross_timeframe_type == CrossTimeframeType.CORRELATION:
                # Memory-efficient rolling correlation between timeframes
                window = get_rolling_window_size(len(x1), COMPUTATION.DEFAULT_ROLLING_WINDOW)
                if window >= COMPUTATION.MIN_ROLLING_WINDOW:
                    cache_params = tuple(sorted((feat1, feat2))) + (window, 'corr')
                    feature = self._compute_rolling_correlation_efficient(
                        x1,
                        x2,
                        window,
                        cache_params=cache_params
                    )
                else:
                    feature = np.zeros_like(x1, dtype=float)
                name = f"{feat1}_rolling_corr_{feat2}"

            elif cross_timeframe_type == CrossTimeframeType.LAG_CORRELATION:
                # True lag-based rolling correlation
                lag = min(CROSS_TIMEFRAME.MAX_LAG_PERIODS, len(x1) // 4)
                if lag > 0:
                    x1_lag = np.roll(x1, lag)
                    x1_lag[:lag] = x1_lag[lag]
                    window = get_rolling_window_size(len(x1), COMPUTATION.DEFAULT_ROLLING_WINDOW)
                    if window >= COMPUTATION.MIN_ROLLING_WINDOW:
                        x1_mean = np.convolve(x1_lag, np.ones(window)/window, mode='valid')
                        x2_mean = np.convolve(x2, np.ones(window)/window, mode='valid')
                        x1_centered = x1_lag[window-1:] - x1_mean
                        x2_centered = x2[window-1:] - x2_mean
                        num = np.convolve(x1_centered * x2_centered, np.ones(1), mode='valid')
                        den = np.sqrt(
                            np.convolve(x1_centered**2, np.ones(1), mode='valid') *
                            np.convolve(x2_centered**2, np.ones(1), mode='valid')
                        )
                        corr_valid = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
                        feature = np.zeros_like(x1, dtype=float)
                        feature[:window-1] = 0.0
                        feature[window-1:] = corr_valid
                        name = f"{feat1}_lag{lag}_rolling_corr_{feat2}"
                    else:
                        return None, ""
                else:
                    return None, ""
                
            elif cross_timeframe_type == CrossTimeframeType.MOMENTUM:
                # Momentum difference between timeframes
                momentum1 = np.diff(x1, prepend=x1[0])
                momentum2 = np.diff(x2, prepend=x2[0])
                feature = momentum1 - momentum2
                name = f"{feat1}_momentum_minus_{feat2}_momentum"
                
            elif cross_timeframe_type == CrossTimeframeType.VOLATILITY:
                # Memory-efficient rolling volatility ratio between timeframes
                window = min(20, len(x1))
                if window >= 3:
                    vol1 = self._compute_rolling_statistic_efficient(
                        x1,
                        window,
                        np.std,
                        cache_params=(feat1, window, 'std')
                    )
                    vol2 = self._compute_rolling_statistic_efficient(
                        x2,
                        window,
                        np.std,
                        cache_params=(feat2, window, 'std')
                    )
                    feature = np.divide(vol1, vol2, out=np.zeros_like(vol1), where=(vol2 != 0))
                    name = f"{feat1}_rolling_vol_ratio_{feat2}"
                else:
                    return None, ""

            elif cross_timeframe_type == CrossTimeframeType.TREND_ALIGNMENT:
                # Rolling trend alignment between timeframes using slope product
                window = min(20, len(x1))
                if window >= 3:
                    feature = self._compute_trend_alignment(
                        x1,
                        x2,
                        window,
                        cache_params=tuple(sorted((feat1, feat2))) + (window, 'trend_align')
                    )
                    name = f"{feat1}_rolling_trend_align_{feat2}"
                else:
                    return None, ""
                
            elif cross_timeframe_type == CrossTimeframeType.REGIME_CONSISTENCY:
                # Regime consistency between timeframes
                # Simplified: use moving average comparison
                ma1 = np.convolve(x1, np.ones(5)/5, mode='same')
                ma2 = np.convolve(x2, np.ones(5)/5, mode='same')
                feature = (ma1 > ma1.mean()) == (ma2 > ma2.mean())
                feature = feature.astype(float)
                name = f"{feat1}_regime_consistency_{feat2}"
                
            else:
                return None, ""
            
            # Validate feature
            if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                self.logger.warning(f"⚠️ Invalid values in {name}, skipping")
                return None, ""
            
            return feature, name
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create {cross_timeframe_type.value} cross-timeframe feature: {e}")
            return None, ""

    def _apply_known_rolling_stat(self, rolling_obj, stat_func):
        """Apply common rolling statistics using pandas-native implementations when possible."""
        func_name = getattr(stat_func, '__name__', '') if hasattr(stat_func, '__name__') else ''

        if func_name in {'std', 'nanstd'}:
            return rolling_obj.std(ddof=0)
        if func_name in {'var', 'nanvar'}:
            return rolling_obj.var(ddof=0)
        if func_name in {'mean', 'nanmean'}:
            return rolling_obj.mean()
        if func_name in {'min', 'amin'}:
            return rolling_obj.min()
        if func_name in {'max', 'amax'}:
            return rolling_obj.max()
        if func_name in {'median', 'nanmedian'}:
            return rolling_obj.median()

        return rolling_obj.apply(lambda arr: stat_func(arr), raw=True)

    def _calculate_cross_timeframe_scores(
        self,
        cross_timeframe_features: np.ndarray,
        cross_timeframe_names: List[str],
        target: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate importance scores for cross-timeframe features."""
        scores = {}
        
        if target is None:
            # Use variance as importance score
            for i, name in enumerate(cross_timeframe_names):
                scores[name] = float(np.var(cross_timeframe_features[:, i]))
        else:
            # Use correlation with target as importance score
            for i, name in enumerate(cross_timeframe_names):
                try:
                    if self.matrix_ops:
                        corr = self.matrix_ops.safe_correlation_matrix(
                            np.column_stack([cross_timeframe_features[:, i], target])
                        )[0, 1]
                    else:
                        corr = np.corrcoef(cross_timeframe_features[:, i], target)[0, 1]
                    scores[name] = abs(float(corr))
                except Exception:
                    scores[name] = 0.0
        
        return scores
    
    def _calculate_average_correlation(self, cross_timeframe_features: np.ndarray) -> float:
        """Calculate average correlation between cross-timeframe features."""
        try:
            if self.matrix_ops:
                corr_matrix = self.matrix_ops.safe_correlation_matrix(cross_timeframe_features)
            else:
                corr_matrix = np.corrcoef(cross_timeframe_features.T)
            
            # Get upper triangle (excluding diagonal)
            n = corr_matrix.shape[0]
            upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
            
            return float(np.mean(np.abs(upper_triangle)))
            
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, cross_timeframe_features: np.ndarray) -> float:
        """Calculate stability score based on feature consistency."""
        try:
            # Calculate coefficient of variation for each feature
            cv_scores = []
            for i in range(cross_timeframe_features.shape[1]):
                feature = cross_timeframe_features[:, i]
                mean_val = np.mean(feature)
                std_val = np.std(feature)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    cv_scores.append(cv)
            
            if cv_scores:
                # Lower CV = higher stability
                avg_cv = np.mean(cv_scores)
                stability_score = max(0.0, 1.0 - avg_cv)
                return float(stability_score)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_timeframe_coverage(self, cross_timeframe_names: List[str]) -> float:
        """Calculate coverage of different timeframes."""
        try:
            covered_timeframes = set()
            for name in cross_timeframe_names:
                for timeframe in self.config.timeframes:
                    if timeframe in name.lower():
                        covered_timeframes.add(timeframe)
            
            return len(covered_timeframes) / len(self.config.timeframes)
            
        except Exception:
            return 0.0
    
    def _calculate_lag_effectiveness(self, cross_timeframe_names: List[str]) -> float:
        """Calculate effectiveness of lag-based features."""
        try:
            lag_features = [name for name in cross_timeframe_names if 'lag_' in name]
            return len(lag_features) / len(cross_timeframe_names) if cross_timeframe_names else 0.0
            
        except Exception:
            return 0.0
    
    def _compute_rolling_correlation_efficient(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        window: int,
        cache_params: Optional[Tuple[Any, ...]] = None
    ) -> np.ndarray:
        """Compute rolling correlation using vectorized operations."""
        if not NUMPY_AVAILABLE:
            return np.zeros_like(x1, dtype=float)

        use_cache = cache_params is not None
        cached_result = None
        if use_cache:
            cached_result = self._cache_lookup('rolling_corr', cache_params)
            if cached_result is not None:
                return cached_result

        try:
            window = max(1, min(window, len(x1)))
            if window <= 1:
                return np.zeros_like(x1, dtype=float)

            if self.matrix_ops and getattr(self.matrix_ops, 'vectorized_core', None) is not None and PANDAS_AVAILABLE:
                df = pd.DataFrame({'x1': x1, 'x2': x2})
                df = self._prepare_dataframe_for_vectorization(df)
                try:
                    corr_series = df['x1'].rolling(window=window, min_periods=window).corr(df['x2'])
                except Exception:
                    series1 = pd.Series(x1)
                    series2 = pd.Series(x2)
                    corr_series = series1.rolling(window=window, min_periods=window).corr(series2)
                result = corr_series.fillna(0.0).to_numpy()
            elif PANDAS_AVAILABLE:
                series1 = pd.Series(x1)
                series2 = pd.Series(x2)
                corr_series = series1.rolling(window=window, min_periods=window).corr(series2)
                result = corr_series.fillna(0.0).to_numpy()
            else:
                if sliding_window_view is None:
                    return np.zeros_like(x1, dtype=float)

                view1 = sliding_window_view(x1, window)
                view2 = sliding_window_view(x2, window)
                mean1 = view1.mean(axis=-1)
                mean2 = view2.mean(axis=-1)
                centered1 = view1 - mean1[:, None]
                centered2 = view2 - mean2[:, None]
                denom = (window - 1) if window > 1 else 1
                cov = np.sum(centered1 * centered2, axis=-1) / denom
                std1 = np.sqrt(np.sum(centered1 ** 2, axis=-1) / denom)
                std2 = np.sqrt(np.sum(centered2 ** 2, axis=-1) / denom)
                corr_valid = np.divide(
                    cov,
                    std1 * std2,
                    out=np.zeros_like(cov),
                    where=(std1 > 0) & (std2 > 0)
                )
                result = np.zeros_like(x1, dtype=float)
                result[window - 1:] = corr_valid

            if use_cache:
                self._cache_store('rolling_corr', cache_params, result)

            return result

        except Exception as e:
            self.logger.warning(f"Efficient rolling correlation failed: {e}, using fallback")
            return np.zeros_like(x1, dtype=float)
    
    def _compute_rolling_statistic_efficient(
        self,
        x: np.ndarray,
        window: int,
        stat_func,
        cache_params: Optional[Tuple[Any, ...]] = None
    ) -> np.ndarray:
        """Compute rolling statistics with vectorized operations."""
        if not NUMPY_AVAILABLE:
            return np.zeros_like(x, dtype=float)

        use_cache = cache_params is not None
        cached_result = None
        if use_cache:
            cached_result = self._cache_lookup('rolling_stat', cache_params)
            if cached_result is not None:
                return cached_result

        try:
            window = max(1, min(window, len(x)))
            if window <= 1:
                if len(x) == 0:
                    return np.array([], dtype=float)

                val = float(stat_func(np.asarray([x[0]], dtype=float)))
                result_array = np.full_like(x, val, dtype=float)
                if use_cache:
                    self._cache_store('rolling_stat', cache_params, result_array)
                return result_array

            if self.matrix_ops and getattr(self.matrix_ops, 'vectorized_core', None) is not None and PANDAS_AVAILABLE:
                df = pd.DataFrame({'value': x})
                df = self._prepare_dataframe_for_vectorization(df)
                try:
                    rolling_obj = df['value'].rolling(window=window, min_periods=window)
                    result_series = self._apply_known_rolling_stat(rolling_obj, stat_func)
                except Exception:
                    series = pd.Series(x)
                    rolling_obj = series.rolling(window=window, min_periods=window)
                    result_series = self._apply_known_rolling_stat(rolling_obj, stat_func)
            elif PANDAS_AVAILABLE:
                series = pd.Series(x)
                rolling_obj = series.rolling(window=window, min_periods=window)
                result_series = self._apply_known_rolling_stat(rolling_obj, stat_func)
            else:
                if sliding_window_view is None:
                    return np.zeros_like(x, dtype=float)

                windows = sliding_window_view(x, window)
                stats = np.apply_along_axis(stat_func, 1, windows)
                result_array = np.zeros_like(x, dtype=float)
                result_array[window - 1:] = stats
                if use_cache:
                    self._cache_store('rolling_stat', cache_params, result_array)
                return result_array

            result_array = result_series.fillna(0.0).to_numpy()

            if use_cache:
                self._cache_store('rolling_stat', cache_params, result_array)

            return result_array

        except Exception as e:
            self.logger.warning(f"Efficient rolling statistic failed: {e}")
            return np.zeros_like(x, dtype=float)

    def _compute_trend_alignment(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        window: int,
        cache_params: Optional[Tuple[Any, ...]] = None
    ) -> np.ndarray:
        """Compute rolling trend alignment via vectorized slope products."""
        if not NUMPY_AVAILABLE or sliding_window_view is None:
            return np.zeros_like(x1, dtype=float)

        use_cache = cache_params is not None
        cached_result = None
        if use_cache:
            cached_result = self._cache_lookup('trend_alignment', cache_params)
            if cached_result is not None:
                return cached_result

        window = max(2, min(window, len(x1)))
        if window <= 1:
            return np.zeros_like(x1, dtype=float)

        try:
            idx = np.arange(window, dtype=float)
            idx_centered = idx - idx.mean()
            denom = np.sum(idx_centered ** 2)
            if denom == 0:
                return np.zeros_like(x1, dtype=float)

            view1 = sliding_window_view(x1, window)
            view2 = sliding_window_view(x2, window)

            centered1 = view1 - view1.mean(axis=-1, keepdims=True)
            centered2 = view2 - view2.mean(axis=-1, keepdims=True)

            slope1 = centered1 @ idx_centered / denom
            slope2 = centered2 @ idx_centered / denom
            slope_prod = slope1 * slope2

            result = np.zeros_like(x1, dtype=float)
            result[window - 1:] = slope_prod

            if use_cache:
                self._cache_store('trend_alignment', cache_params, result)

            return result

        except Exception as exc:
            self.logger.warning(f"⚠️ Trend alignment computation failed: {exc}")
            return np.zeros_like(x1, dtype=float)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            'pid_available': PID_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'pandas_available': PANDAS_AVAILABLE,
            'rolling_cache_hits': self._rolling_cache_hits,
            'rolling_cache_misses': self._rolling_cache_misses
        }
        
        if self.matrix_ops:
            metrics['matrix_ops_stats'] = self.matrix_ops.get_performance_stats()
            metrics['hardware_info'] = self.matrix_ops.get_hardware_info()
        
        return metrics
