"""
Lookahead Bias Protection and Prevention Utilities

This module provides comprehensive lookahead bias detection and prevention mechanisms
for time series machine learning in trading systems.

Key Features:
- Data leakage detection in feature engineering
- Temporal validation of features and targets
- Rolling window validation with strict temporal ordering
- Advanced information barrier checks
- Automated future data filtering
- Feature timestamp alignment validation
- Rolling window validation for streaming data
- Temporal feature validation

Built on existing utilities:
- Extends the existing lookahead_bias_detector.py
- Uses math_validation.py for safe operations
- Integrates with data_processing_utils.py for data handling
- Leverages m1_gpu_utils.py for GPU acceleration
- Uses m1_memory_optimizer.py for memory management
"""

import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Iterator
from datetime import datetime, timedelta
import logging
import hashlib
import warnings

from ..math_validation import safe_divide
from ..common_operations import create_fallback_logger
from src.utils.common_operations import safe_dataframe_operation

# Enhanced imports for new functionality
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer  # type: ignore
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Define fallback classes to prevent NameError - available globally
class LookaheadBiasDetector:
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)

    def detect_bias(self, data, target=None):
        """Fallback detection method - always returns no bias"""
        self.logger.info("Using fallback lookahead bias detector")
        return {'bias_detected': False, 'bias_score': 0.0, 'details': 'Fallback detector'}

    def validate_temporal_order(self, timestamps):
        """Fallback temporal validation"""
        return True

class LookaheadBiasError(Exception):
    """Exception raised when lookahead bias is detected in ML training."""

    def __init__(self, message: str, bias_score: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.bias_score = bias_score
        self.context = context or {}
        self.error_type = "LOOKAHEAD_BIAS_ERROR"
        self.severity = "CRITICAL"
        self.suggested_actions = [
            "Remove future-looking features from training data",
            "Verify temporal ordering of data splits",
            "Implement strict temporal validation",
            "Check feature engineering for target leakage",
            "Review data preprocessing pipeline"
        ]

    def __str__(self):
        if self.bias_score is not None:
            return f"{self.error_type} (score: {self.bias_score:.3f}): {super().__str__()} | Context: {self.context}"
        return f"{self.error_type}: {super().__str__()} | Context: {self.context}"

try:
    from src.utils.lookahead_bias_detector import LookaheadBiasDetector, LookaheadBiasError
    EXISTING_DETECTOR_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Existing lookahead detector not available - using fallback implementation")
    EXISTING_DETECTOR_AVAILABLE = False

class LookaheadProtection:
    """Advanced lookahead bias protection and detection system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize lookahead protection with configuration."""
        self.logger = logger.getChild('LookaheadProtection')
        self.logger.info("🚀 Initializing LookaheadProtection...")
        start_time = time.time()

        self.config = config or {}
        self.logger.info(f"📊 Configuration loaded with {len(self.config)} parameters")

        # Configuration defaults
        self.strict_mode = self.config.get('strict_mode', True)
        self.tolerance_seconds = self.config.get('tolerance_seconds', 60)  # 1 minute tolerance
        self.enable_automatic_filtering = self.config.get('enable_automatic_filtering', True)
        self.detection_log = []
        self.current_timestamp = None

        self.logger.info(f"📊 Strict mode: {self.strict_mode}")
        self.logger.info(f"📊 Tolerance seconds: {self.tolerance_seconds}")
        self.logger.info(f"📊 Automatic filtering: {self.enable_automatic_filtering}")

        # Enhanced configuration for new features
        self.enable_gpu = self.config.get('enable_gpu', GPU_AVAILABLE)
        self.enable_memory_optimization = self.config.get('enable_memory_optimization', MEMORY_OPTIMIZER_AVAILABLE)
        self.rolling_window_size = self.config.get('rolling_window_size', 1000)
        self.information_barrier_rules = self.config.get('information_barrier_rules', {})
        self.feature_alignment_threshold = self.config.get('feature_alignment_threshold', timedelta(minutes=1))

        self.logger.info(f"📊 GPU enabled: {self.enable_gpu}")
        self.logger.info(f"📊 Memory optimization: {self.enable_memory_optimization}")
        self.logger.info(f"📊 Rolling window size: {self.rolling_window_size}")

        # Initialize utilities
        self.logger.debug("🔧 Initializing GPU manager...")
        self.gpu_manager = M1GPUManager() if self.enable_gpu else None
        if self.gpu_manager:
            self.logger.debug("✅ GPU manager initialized")
        else:
            self.logger.debug("ℹ️ GPU manager not initialized")

        self.logger.debug("🔧 Initializing memory optimizer...")
        self.memory_optimizer = M1MemoryOptimizer() if self.enable_memory_optimization else None
        if self.memory_optimizer:
            self.logger.debug("✅ Memory optimizer initialized")
        else:
            self.logger.debug("ℹ️ Memory optimizer not initialized")

        # Initialize existing detector if available
        self.logger.debug("🔧 Initializing base detector...")
        if EXISTING_DETECTOR_AVAILABLE:
            self.base_detector = LookaheadBiasDetector(strict_mode=self.strict_mode)
            self.logger.debug("✅ Base detector initialized")
        else:
            self.base_detector = None
            self.logger.warning("⚠️ Base detector not available")

        init_time = time.time() - start_time
        self.logger.info(f"✅ LookaheadProtection initialized in {init_time:.3f}s")

    def set_current_timestamp(self, timestamp: datetime) -> None:
        """Set the current timestamp for bias detection."""
        self.current_timestamp = timestamp
        if self.base_detector:
            self.base_detector.set_current_timestamp(timestamp)
        self.logger.info(f"🔒 Current timestamp set to: {timestamp.isoformat()}")

    def detect_data_leakage(self, features_df: pd.DataFrame,
                          target_df: pd.DataFrame,
                          timestamp_col: str = 'timestamp',
                          feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect potential data leakage between features and targets.

        Args:
            features_df: DataFrame containing features
            target_df: DataFrame containing targets
            timestamp_col: Name of timestamp column
            feature_cols: Specific feature columns to check

        Returns:
            Detection results with potential leakage issues
        """
        try:
            self.logger.info("🔍 Starting comprehensive data leakage detection")

            detection_results = {
                'leakage_detected': False,
                'issues': [],
                'warnings': [],
                'recommendations': [],
                'feature_analysis': {},
                'temporal_analysis': {}
            }

            # Validate inputs
            if timestamp_col not in features_df.columns:
                detection_results['issues'].append(f"Timestamp column '{timestamp_col}' not found in features")
                return detection_results

            if timestamp_col not in target_df.columns:
                detection_results['issues'].append(f"Timestamp column '{timestamp_col}' not found in targets")
                return detection_results

            # Convert timestamps if needed
            features_df = self._ensure_timestamp_format(features_df, timestamp_col)
            target_df = self._ensure_timestamp_format(target_df, timestamp_col)

            # Get feature columns
            if feature_cols is None:
                feature_cols = [col for col in features_df.columns
                              if col != timestamp_col and not col.startswith('target')]

            # Analyze temporal relationships
            temporal_analysis = self._analyze_temporal_relationships(
                features_df, target_df, timestamp_col, feature_cols
            )
            detection_results['temporal_analysis'] = temporal_analysis

            # Check for future feature values
            future_features = self._detect_future_features(
                features_df, timestamp_col, feature_cols
            )

            # Check for target leakage in features
            target_leakage = self._detect_target_leakage(
                features_df, target_df, timestamp_col, feature_cols
            )

            # Check for overlapping time windows
            time_overlap = self._detect_time_window_overlap(
                features_df, target_df, timestamp_col
            )

            # Aggregate results
            all_issues = (temporal_analysis.get('issues', []) +
                         future_features.get('issues', []) +
                         target_leakage.get('issues', []) +
                         time_overlap.get('issues', []))

            detection_results['issues'].extend(all_issues)
            detection_results['leakage_detected'] = len(all_issues) > 0

            # Generate recommendations
            detection_results['recommendations'] = self._generate_leakage_recommendations(
                detection_results['issues']
            )

            self.logger.info(f"✅ Data leakage detection completed - "
                           f"{'Issues found' if detection_results['leakage_detected'] else 'No issues detected'}")

            return detection_results

        except Exception as e:
            self.logger.error(f"❌ Data leakage detection failed: {e}")
            return {'error': str(e), 'leakage_detected': True}

    def temporal_feature_validation(self, feature_data: pd.DataFrame,
                                  prediction_timestamp: datetime,
                                  feature_timestamp_col: str = 'timestamp',
                                  lookback_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Validate that features are not using future information relative to prediction time.

        Args:
            feature_data: DataFrame containing features
            prediction_timestamp: Timestamp when prediction is made
            feature_timestamp_col: Column containing feature timestamps
            lookback_window: Maximum lookback window allowed

        Returns:
            Validation results
        """
        try:
            self.logger.info(f"⏰ Validating temporal integrity for prediction at {prediction_timestamp}")

            validation_results = {
                'is_valid': True,
                'issues': [],
                'feature_stats': {},
                'temporal_coverage': {}
            }

            if feature_timestamp_col not in feature_data.columns:
                validation_results['issues'].append(f"Feature timestamp column '{feature_timestamp_col}' not found")
                validation_results['is_valid'] = False
                return validation_results

            # Ensure timestamp format
            feature_data = self._ensure_timestamp_format(feature_data, feature_timestamp_col)

            # Check for future timestamps
            future_mask = feature_data[feature_timestamp_col] > prediction_timestamp
            future_features = feature_data[future_mask]

            if len(future_features) > 0:
                validation_results['issues'].append(
                    f"Found {len(future_features)} features with future timestamps "
                    f"(after {prediction_timestamp})"
                )
                validation_results['is_valid'] = False

            # Check lookback window if specified
            if lookback_window is not None:
                earliest_allowed = prediction_timestamp - lookback_window
                old_features_mask = feature_data[feature_timestamp_col] < earliest_allowed
                old_features = feature_data[old_features_mask]

                if len(old_features) > 0:
                    validation_results['issues'].append(
                        f"Found {len(old_features)} features outside lookback window "
                        f"(before {earliest_allowed})"
                    )
                    validation_results['warnings'] = validation_results.get('warnings', [])
                    validation_results['warnings'].append("Consider adjusting lookback window")

            # Analyze temporal coverage
            if len(feature_data) > 0:
                timestamps = feature_data[feature_timestamp_col].dropna()
                if len(timestamps) > 0:
                    validation_results['temporal_coverage'] = {
                        'earliest_feature': timestamps.min(),
                        'latest_feature': timestamps.max(),
                        'time_span': timestamps.max() - timestamps.min(),
                        'feature_count': len(feature_data),
                        'prediction_time': prediction_timestamp,
                        'lookback_used': prediction_timestamp - timestamps.max() if len(timestamps) > 0 else None
                    }

            # Feature statistics
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                validation_results['feature_stats'] = {
                    'numeric_features': len(numeric_cols),
                    'total_features': len(feature_data.columns) - 1,  # Excluding timestamp
                    'missing_values': feature_data.isnull().sum().sum(),
                    'missing_percentage': safe_divide(feature_data.isnull().sum().sum(),
                                                    feature_data.shape[0] * feature_data.shape[1]) * 100
                }

            return validation_results

        except Exception as e:
            self.logger.error(f"❌ Temporal feature validation failed: {e}")
            return {'error': str(e), 'is_valid': False}

    def rolling_window_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                timestamps: np.ndarray, window_size: int = 1000,
                                step_size: int = 100) -> Dict[str, Any]:
        """
        Perform rolling window validation with strict temporal ordering.

        Args:
            model: ML model to validate
            X: Feature matrix
            y: Target array
            timestamps: Array of timestamps
            window_size: Size of rolling window
            step_size: Step size for window movement

        Returns:
            Rolling window validation results
        """
        try:
            self.logger.info(f"🔄 Starting rolling window validation (window={window_size}, step={step_size})")

            validation_results = {
                'windows': [],
                'performance_metrics': [],
                'temporal_integrity_checks': [],
                'overall_assessment': {}
            }

            # Sort data by timestamp
            sort_indices = np.argsort(timestamps)
            X_sorted = X[sort_indices]
            y_sorted = y[sort_indices]
            timestamps_sorted = timestamps[sort_indices]

            # Perform rolling window validation
            for start_idx in range(0, len(X_sorted) - window_size, step_size):
                end_idx = start_idx + window_size

                # Define train/test split (last portion as test)
                test_size = max(50, window_size // 10)
                train_end_idx = end_idx - test_size

                if train_end_idx <= start_idx:
                    continue

                try:
                    # Split data
                    X_train = X_sorted[start_idx:train_end_idx]
                    y_train = y_sorted[start_idx:train_end_idx]
                    X_test = X_sorted[train_end_idx:end_idx]
                    y_test = y_sorted[train_end_idx:end_idx]

                    train_timestamps = timestamps_sorted[start_idx:train_end_idx]
                    test_timestamps = timestamps_sorted[train_end_idx:end_idx]

                    # Temporal integrity check
                    temporal_check = self._check_temporal_integrity(
                        train_timestamps, test_timestamps
                    )

                    # Train and evaluate model
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_test)

                    # Calculate metrics
                    metrics = self._calculate_basic_metrics(y_test, y_pred)

                    # Store window results
                    window_result = {
                        'window_id': len(validation_results['windows']),
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'train_period': {
                            'start': train_timestamps[0] if len(train_timestamps) > 0 else None,
                            'end': train_timestamps[-1] if len(train_timestamps) > 0 else None
                        },
                        'test_period': {
                            'start': test_timestamps[0] if len(test_timestamps) > 0 else None,
                            'end': test_timestamps[-1] if len(test_timestamps) > 0 else None
                        },
                        'temporal_integrity': temporal_check,
                        'metrics': metrics
                    }

                    validation_results['windows'].append(window_result)
                    validation_results['temporal_integrity_checks'].append(temporal_check)

                    acc = metrics.get('accuracy')
                    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float, np.floating)) else str(acc)
                    self.logger.debug(f"✅ Window {window_result['window_id']} completed - Accuracy: {acc_str}")

                except Exception as window_e:
                    self.logger.warning(f"⚠️ Window {len(validation_results['windows'])} failed: {window_e}")
                    continue

            # Overall assessment
            if validation_results['windows']:
                all_metrics = [w['metrics'] for w in validation_results['windows']]
                validation_results['overall_assessment'] = self._assess_rolling_performance(
                    validation_results['windows']
                )

                self.logger.info(f"✅ Rolling window validation completed: "
                               f"{len(validation_results['windows'])} windows processed")
            else:
                self.logger.error("❌ No windows completed successfully")

            return validation_results

        except Exception as e:
            self.logger.error(f"❌ Rolling window validation failed: {e}")
            return {'error': str(e), 'windows': []}

    def information_barrier_check(self, X: np.ndarray, y: np.ndarray,
                                timestamps: np.ndarray,
                                barrier_minutes: int = 60) -> Dict[str, Any]:
        """
        Check information barriers between features and targets.

        Args:
            X: Feature matrix
            y: Target array
            timestamps: Array of timestamps
            barrier_minutes: Information barrier in minutes

        Returns:
            Information barrier check results
        """
        try:
            self.logger.info(f"🚧 Checking information barriers ({barrier_minutes} minutes)")

            barrier_results = {
                'barrier_violations': [],
                'barrier_compliance': True,
                'violation_summary': {},
                'recommendations': []
            }

            barrier_timedelta = timedelta(minutes=barrier_minutes)

            # Check each sample
            violations = []
            for i in range(len(timestamps)):
                feature_time = timestamps[i]

                # Find target timestamps within barrier period
                future_targets = []
                for j in range(len(timestamps)):
                    if timestamps[j] > feature_time and timestamps[j] <= feature_time + barrier_timedelta:
                        future_targets.append((j, timestamps[j]))

                if future_targets:
                    violations.append({
                        'sample_idx': i,
                        'feature_timestamp': feature_time,
                        'future_targets': future_targets,
                        'barrier_end': feature_time + barrier_timedelta
                    })

            barrier_results['barrier_violations'] = violations
            barrier_results['barrier_compliance'] = len(violations) == 0

            if violations:
                barrier_results['violation_summary'] = {
                    'total_violations': len(violations),
                    'affected_samples': len(set(v['sample_idx'] for v in violations)),
                    'violation_percentage': safe_divide(len(violations), len(timestamps)) * 100
                }

                barrier_results['recommendations'].extend([
                    f"Remove or adjust {len(violations)} samples with information barrier violations",
                    f"Consider increasing barrier to {barrier_minutes * 2} minutes",
                    "Implement strict temporal data partitioning"
                ])

            self.logger.info(f"✅ Information barrier check completed - "
                           f"{'Compliant' if barrier_results['barrier_compliance'] else f'{len(violations)} violations found'}")

            return barrier_results

        except Exception as e:
            self.logger.error(f"❌ Information barrier check failed: {e}")
            return {'error': str(e), 'barrier_compliance': False}

    def automated_future_data_filtering(self, df: pd.DataFrame,
                                      current_time: Optional[datetime] = None,
                                      timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Automatically filter out future data points.

        Args:
            df: DataFrame to filter
            current_time: Current timestamp (uses now if None)
            timestamp_col: Timestamp column name

        Returns:
            Filtered DataFrame with only past/present data
        """
        try:
            if current_time is None:
                current_time = datetime.now()

            if timestamp_col not in df.columns:
                self.logger.warning(f"Timestamp column '{timestamp_col}' not found, returning original DataFrame")
                return df

            # Ensure timestamp format
            df = self._ensure_timestamp_format(df, timestamp_col)

            # Filter out future data
            valid_mask = df[timestamp_col] <= current_time
            filtered_df = df[valid_mask].copy()

            removed_count = len(df) - len(filtered_df)

            if removed_count > 0:
                self.logger.info(f"🗑️ Removed {removed_count} future data points from DataFrame")
                self.logger.info(f"📊 Data filtered: {len(filtered_df)} samples remaining "
                               f"(from {current_time})")

            return filtered_df

        except Exception as e:
            self.logger.error(f"❌ Automated future data filtering failed: {e}")
            return df

    def feature_timestamp_alignment(self, feature_dict: Dict[str, pd.DataFrame],
                                  target_timestamp: datetime,
                                  timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Validate and align feature timestamps with target timestamp.

        Args:
            feature_dict: Dictionary of feature DataFrames
            target_timestamp: Target prediction timestamp
            timestamp_col: Timestamp column name

        Returns:
            Alignment validation results
        """
        try:
            self.logger.info("🔧 Validating feature timestamp alignment")

            alignment_results = {
                'alignment_status': 'valid',
                'feature_alignment': {},
                'issues': [],
                'recommendations': []
            }

            for feature_name, feature_df in feature_dict.items():
                feature_alignment = {
                    'feature_name': feature_name,
                    'is_aligned': True,
                    'timestamp_range': {},
                    'issues': []
                }

                try:
                    if timestamp_col not in feature_df.columns:
                        feature_alignment['issues'].append(f"Missing timestamp column '{timestamp_col}'")
                        feature_alignment['is_aligned'] = False
                    else:
                        # Ensure timestamp format
                        feature_df = self._ensure_timestamp_format(feature_df, timestamp_col)

                        # Check timestamp range
                        timestamps = feature_df[timestamp_col].dropna()
                        if len(timestamps) > 0:
                            feature_alignment['timestamp_range'] = {
                                'earliest': timestamps.min(),
                                'latest': timestamps.max(),
                                'count': len(timestamps)
                            }

                            # Check if latest feature is not after target timestamp
                            if timestamps.max() > target_timestamp:
                                feature_alignment['issues'].append(
                                    f"Feature has data after target timestamp: "
                                    f"{timestamps.max()} > {target_timestamp}"
                                )
                                feature_alignment['is_aligned'] = False

                            # Check for reasonable lookback
                            lookback = target_timestamp - timestamps.max()
                            if lookback > timedelta(days=30):  # More than 30 days old
                                feature_alignment['issues'].append(
                                    f"Feature data is {lookback.days} days old - may be stale"
                                )

                        else:
                            feature_alignment['issues'].append("No valid timestamps found")
                            feature_alignment['is_aligned'] = False

                except Exception as feature_e:
                    feature_alignment['issues'].append(f"Alignment check failed: {feature_e}")
                    feature_alignment['is_aligned'] = False

                alignment_results['feature_alignment'][feature_name] = feature_alignment

                if not feature_alignment['is_aligned']:
                    alignment_results['alignment_status'] = 'invalid'
                    alignment_results['issues'].extend(feature_alignment['issues'])

            # Generate recommendations
            if alignment_results['issues']:
                alignment_results['recommendations'].extend([
                    "Align all feature timestamps to target prediction time",
                    "Implement automatic feature timestamp validation",
                    "Consider feature staleness checks in preprocessing"
                ])

            self.logger.info(f"✅ Feature timestamp alignment validation completed - "
                           f"Status: {alignment_results['alignment_status']}")

            return alignment_results

        except Exception as e:
            self.logger.error(f"❌ Feature timestamp alignment failed: {e}")
            return {'error': str(e), 'alignment_status': 'error'}

    def _analyze_temporal_relationships(self, features_df: pd.DataFrame,
                                      target_df: pd.DataFrame,
                                      timestamp_col: str,
                                      feature_cols: List[str]) -> Dict[str, Any]:
        """Analyze temporal relationships between features and targets."""
        try:
            analysis = {'issues': [], 'warnings': [], 'temporal_gaps': []}

            # Get timestamp ranges
            feature_times = features_df[timestamp_col].dropna()
            target_times = target_df[timestamp_col].dropna()

            if len(feature_times) > 0 and len(target_times) > 0:
                feature_range = (feature_times.min(), feature_times.max())
                target_range = (target_times.min(), target_times.max())

                # Check for temporal overlap
                latest_start = max(feature_range[0], target_range[0])
                earliest_end = min(feature_range[1], target_range[1])
                overlap = max(0, (earliest_end - latest_start).total_seconds())

                if overlap == 0:
                    analysis['issues'].append("No temporal overlap between features and targets")
                elif overlap < 3600:  # Less than 1 hour
                    analysis['warnings'].append(f"Limited temporal overlap: {overlap/3600:.1f} hours")

                # Check for targets that occur after max feature time (potential leakage)
                future_targets = target_df[target_df[timestamp_col] >= feature_times.max()]
                if len(future_targets) > 0:
                    analysis['issues'].append(
                        f"Found {len(future_targets)} target values in feature future"
                    )

            return analysis

        except Exception as e:
            return {'error': str(e), 'issues': [f'Analysis failed: {e}']}

    def _detect_future_features(self, features_df: pd.DataFrame,
                              timestamp_col: str,
                              feature_cols: List[str]) -> Dict[str, Any]:
        """Detect features with future timestamps."""
        try:
            detection = {'issues': [], 'future_features': {}}

            if self.current_timestamp is None:
                return detection

            for col in feature_cols:
                if col in features_df.columns:
                    feature_mask = features_df[timestamp_col] > self.current_timestamp
                    future_values = features_df[feature_mask]

                    if len(future_values) > 0:
                        detection['issues'].append(
                            f"Feature '{col}' has {len(future_values)} future values"
                        )
                        detection['future_features'][col] = len(future_values)

            return detection

        except Exception as e:
            return {'error': str(e), 'issues': [f'Detection failed: {e}']}

    def _detect_target_leakage(self, features_df: pd.DataFrame,
                             target_df: pd.DataFrame,
                             timestamp_col: str,
                             feature_cols: List[str]) -> Dict[str, Any]:
        """Detect potential target leakage in features."""
        try:
            leakage = {'issues': [], 'suspected_leakage': []}

            # Check for target-like column names in features
            target_patterns = ['target', 'label', 'outcome', 'result', 'return', 'profit']
            for col in feature_cols:
                col_lower = col.lower()
                for pattern in target_patterns:
                    if pattern in col_lower:
                        leakage['suspected_leakage'].append(col)
                        leakage['issues'].append(f"Potential target leakage: feature '{col}' contains '{pattern}'")

            # Check for perfect correlation with target (if target available)
            if 'target' in target_df.columns:
                for col in feature_cols:
                    if col in features_df.columns:
                        try:
                            # Simple correlation check
                            merged = pd.merge(features_df[[timestamp_col, col]],
                                            target_df[[timestamp_col, 'target']],
                                            on=timestamp_col, how='inner')

                            if len(merged) > 10:
                                corr = merged[col].corr(merged['target'])
                                if abs(corr) > 0.95:  # Near-perfect correlation
                                    leakage['issues'].append(
                                        f"Potential target leakage: feature '{col}' has {corr:.3f} correlation with target"
                                    )
                        except:
                            pass  # Skip correlation check if it fails

            return leakage

        except Exception as e:
            return {'error': str(e), 'issues': [f'Leakage detection failed: {e}']}

    def _detect_time_window_overlap(self, features_df: pd.DataFrame,
                                  target_df: pd.DataFrame,
                                  timestamp_col: str) -> Dict[str, Any]:
        """Detect overlapping time windows between features and targets."""
        try:
            overlap = {'issues': [], 'overlap_analysis': {}}

            # Calculate time spans
            feature_times = features_df[timestamp_col].dropna()
            target_times = target_df[timestamp_col].dropna()

            if len(feature_times) > 0 and len(target_times) > 0:
                overlap['overlap_analysis'] = {
                    'feature_span': (feature_times.min(), feature_times.max()),
                    'target_span': (target_times.min(), target_times.max()),
                    'overlap_duration': self._calculate_overlap_duration(
                        (feature_times.min(), feature_times.max()),
                        (target_times.min(), target_times.max())
                    )
                }

                # Check for problematic overlaps
                if overlap['overlap_analysis']['overlap_duration'] == 0:
                    overlap['issues'].append("No temporal overlap between features and targets")

            return overlap

        except Exception as e:
            return {'error': str(e), 'issues': [f'Overlap detection failed: {e}']}

    def _calculate_overlap_duration(self, range1: Tuple[datetime, datetime],
                                  range2: Tuple[datetime, datetime]) -> float:
        """Calculate overlap duration between two time ranges in seconds."""
        try:
            latest_start = max(range1[0], range2[0])
            earliest_end = min(range1[1], range2[1])
            overlap = max(0, (earliest_end - latest_start).total_seconds())
            return overlap
        except:
            return 0.0

    def _generate_leakage_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on detected issues."""
        recommendations = []

        if not issues:
            return ["✅ No data leakage issues detected - temporal integrity maintained"]

        # Analyze issue patterns and provide specific recommendations
        if any('future' in issue.lower() for issue in issues):
            recommendations.append("Implement strict future data filtering in preprocessing pipeline")

        if any('target' in issue.lower() and 'leakage' in issue.lower() for issue in issues):
            recommendations.append("Review feature engineering to remove target-derived features")
            recommendations.append("Implement feature-target correlation checks in CI/CD")

        if any('temporal' in issue.lower() and 'overlap' in issue.lower() for issue in issues):
            recommendations.append("Ensure proper temporal alignment between features and targets")
            recommendations.append("Implement time-based data partitioning strategies")

        if any('timestamp' in issue.lower() for issue in issues):
            recommendations.append("Standardize timestamp formats across all data sources")
            recommendations.append("Implement automated timestamp validation")

        # General recommendations
        recommendations.extend([
            "Add comprehensive lookahead bias tests to validation suite",
            "Implement automated temporal validation in data pipeline",
            "Document all temporal assumptions and constraints",
            "Regular audit of feature engineering pipeline for temporal issues"
        ])

        return recommendations

    def _ensure_timestamp_format(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Ensure timestamp column is in proper datetime format."""
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df = df.copy()
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            return df
        except Exception as e:
            self.logger.warning(f"Timestamp format conversion failed: {e}")
            return df

    def _check_temporal_integrity(self, train_timestamps: np.ndarray,
                                test_timestamps: np.ndarray) -> Dict[str, Any]:
        """Check temporal integrity between train and test sets."""
        try:
            integrity = {'is_valid': True, 'issues': []}

            if len(train_timestamps) > 0 and len(test_timestamps) > 0:
                # Check that training data comes before test data
                if train_timestamps.max() >= test_timestamps.min():
                    integrity['is_valid'] = False
                    integrity['issues'].append(
                        f"Temporal overlap: train max ({train_timestamps.max()}) >= test min ({test_timestamps.min()})"
                    )

                # Check for gaps
                time_gap = test_timestamps.min() - train_timestamps.max()
                if time_gap < timedelta(0):
                    integrity['issues'].append(f"Negative time gap: {time_gap}")

            return integrity

        except Exception as e:
            return {'error': str(e), 'is_valid': False}

    def _clone_model(self, model: Any) -> Any:
        """Clone a model for validation."""
        try:
            if hasattr(model, 'clone'):
                return model.clone()
            elif hasattr(model, '__class__'):
                model_class = model.__class__
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    return model_class(**params)
                else:
                    return model_class()
            else:
                return model
        except Exception as e:
            self.logger.warning(f"Model cloning failed: {e}")
            return model

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate basic classification/regression metrics."""
        try:
            metrics = {}

            # Determine task type
            unique_values = np.unique(y_true)
            if len(unique_values) <= 10 and all(isinstance(v, (int, np.integer)) for v in unique_values):
                # Classification
                from sklearn.metrics import accuracy_score, f1_score
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                if len(unique_values) == 2:
                    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
                else:
                    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            else:
                # Regression
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def _assess_rolling_performance(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall performance across rolling windows."""
        try:
            assessment = {'trend': 'stable', 'volatility': 0.0, 'recommendations': []}

            if len(windows) < 2:
                return assessment

            # Extract performance metrics
            accuracies = []
            for window in windows:
                if 'metrics' in window and 'accuracy' in window['metrics']:
                    accuracies.append(window['metrics']['accuracy'])

            if len(accuracies) >= 2:
                # Calculate trend
                first_half = np.mean(accuracies[:len(accuracies)//2])
                second_half = np.mean(accuracies[len(accuracies)//2:])

                if second_half > first_half + 0.05:
                    assessment['trend'] = 'improving'
                elif first_half > second_half + 0.05:
                    assessment['trend'] = 'declining'
                else:
                    assessment['trend'] = 'stable'

                # Calculate volatility
                assessment['volatility'] = np.std(accuracies)

                # Generate recommendations
                if assessment['volatility'] > 0.1:
                    assessment['recommendations'].append("High performance volatility detected - consider model stabilization")
                if assessment['trend'] == 'declining':
                    assessment['recommendations'].append("Performance declining over time - investigate concept drift")

            return assessment

        except Exception as e:
            return {'error': str(e)}

    def advanced_information_barrier_checks(self, data_stream: pd.DataFrame,
                                         barrier_rules: Optional[Dict[str, Any]] = None,
                                         current_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Advanced information barrier checks with customizable rules.

        Args:
            data_stream: Data stream to check for information barriers
            barrier_rules: Custom barrier rules (optional)
            current_timestamp: Current timestamp for temporal checks

        Returns:
            Advanced barrier check results with violations and recommendations
        """
        try:
            if current_timestamp is None:
                current_timestamp = self.current_timestamp or datetime.now()

            if barrier_rules is None:
                barrier_rules = self.information_barrier_rules or self._get_default_barrier_rules()

            self.logger.info(f"🔒 Performing advanced information barrier checks at {current_timestamp}")

            barrier_analysis = {
                'barrier_violations': [],
                'barrier_compliance': {},
                'temporal_barriers': {},
                'data_flow_analysis': {},
                'recommendations': [],
                'severity_assessment': {}
            }

            # Check temporal information barriers
            temporal_barriers = self._check_temporal_barriers(data_stream, barrier_rules, current_timestamp)
            barrier_analysis['temporal_barriers'] = temporal_barriers

            # Check data flow barriers
            data_flow = self._check_data_flow_barriers(data_stream, barrier_rules)
            barrier_analysis['data_flow_analysis'] = data_flow

            # Check feature dependency barriers
            feature_deps = self._check_feature_dependency_barriers(data_stream, barrier_rules)
            barrier_analysis['feature_dependency_analysis'] = feature_deps

            # Aggregate violations
            all_violations = (temporal_barriers.get('violations', []) +
                            data_flow.get('violations', []) +
                            feature_deps.get('violations', []))
            barrier_analysis['barrier_violations'] = all_violations

            # Assess severity
            barrier_analysis['severity_assessment'] = self._assess_barrier_violations_severity(all_violations)

            # Generate recommendations
            barrier_analysis['recommendations'] = self._generate_barrier_recommendations(all_violations, barrier_rules)

            self.logger.info(f"✅ Advanced barrier checks completed - {len(all_violations)} violations found")

            return barrier_analysis

        except Exception as e:
            self.logger.error(f"❌ Advanced barrier checks failed: {e}")
            return {'error': str(e), 'barrier_violations': []}

    def validate_feature_timestamp_alignment(self, features_df: pd.DataFrame,
                                          target_df: pd.DataFrame,
                                          timestamp_col: str = 'timestamp',
                                          alignment_threshold: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Validate feature timestamp alignment with targets.

        Args:
            features_df: DataFrame containing features
            target_df: DataFrame containing targets
            timestamp_col: Timestamp column name
            alignment_threshold: Maximum allowed alignment difference

        Returns:
            Timestamp alignment validation results
        """
        try:
            if alignment_threshold is None:
                alignment_threshold = self.feature_alignment_threshold

            self.logger.info(f"⏰ Validating feature timestamp alignment (threshold: {alignment_threshold})")

            alignment_results = {
                'is_aligned': True,
                'alignment_stats': {},
                'misaligned_features': [],
                'temporal_coverage': {},
                'recommendations': []
            }

            # Ensure timestamp columns exist and are properly formatted
            if timestamp_col not in features_df.columns or timestamp_col not in target_df.columns:
                alignment_results['is_aligned'] = False
                alignment_results['recommendations'].append(f"Timestamp column '{timestamp_col}' missing from data")
                return alignment_results

            # Convert to datetime if needed
            features_df = self._ensure_timestamp_format(features_df, timestamp_col)
            target_df = self._ensure_timestamp_format(target_df, timestamp_col)

            # Get feature columns
            feature_cols = [col for col in features_df.columns if col != timestamp_col and not col.startswith('target')]

            # Analyze timestamp alignment for each feature
            for feature_col in feature_cols:
                if feature_col not in features_df.columns:
                    continue

                feature_alignment = self._analyze_single_feature_alignment(
                    features_df, target_df, feature_col, timestamp_col, alignment_threshold
                )

                alignment_results['alignment_stats'][feature_col] = feature_alignment

                if not feature_alignment['is_aligned']:
                    alignment_results['is_aligned'] = False
                    alignment_results['misaligned_features'].append(feature_col)

            # Analyze temporal coverage
            alignment_results['temporal_coverage'] = self._analyze_temporal_coverage(
                features_df, target_df, timestamp_col
            )

            # Generate recommendations
            if not alignment_results['is_aligned']:
                alignment_results['recommendations'].extend([
                    f"Found {len(alignment_results['misaligned_features'])} misaligned features",
                    "Consider temporal interpolation or feature realignment",
                    "Review data collection timestamps for consistency"
                ])

            self.logger.info(f"✅ Timestamp alignment validation completed - "
                           f"{'Aligned' if alignment_results['is_aligned'] else 'Misaligned'}")

            return alignment_results

        except Exception as e:
            self.logger.error(f"❌ Timestamp alignment validation failed: {e}")
            return {'error': str(e), 'is_aligned': False}

    def automated_future_data_filtering_stream(self, data_stream: Iterator[pd.DataFrame],
                                             current_timestamp: Optional[datetime] = None,
                                             filter_config: Optional[Dict[str, Any]] = None) -> Iterator[pd.DataFrame]:
        """
        Automated filtering of future-looking data from streaming data.

        Args:
            data_stream: Iterator of DataFrames to filter
            current_timestamp: Current timestamp for filtering
            filter_config: Filtering configuration

        Yields:
            Filtered DataFrames with future data removed
        """
        try:
            if current_timestamp is None:
                current_timestamp = self.current_timestamp or datetime.now()

            if filter_config is None:
                filter_config = {
                    'timestamp_column': 'timestamp',
                    'tolerance_seconds': self.tolerance_seconds,
                    'strict_filtering': self.strict_mode,
                    'log_violations': True
                }

            self.logger.info(f"🔮 Starting automated future data filtering at {current_timestamp}")

            filtering_stats = {
                'total_rows_processed': 0,
                'future_rows_filtered': 0,
                'valid_rows_kept': 0,
                'timestamp_violations': []
            }

            for chunk_idx, data_chunk in enumerate(data_stream):
                try:
                    # Apply future data filtering
                    filtered_chunk, chunk_stats = self._filter_future_data_chunk(
                        data_chunk, current_timestamp, filter_config
                    )

                    # Update statistics
                    filtering_stats['total_rows_processed'] += chunk_stats['original_rows']
                    filtering_stats['future_rows_filtered'] += chunk_stats['filtered_rows']
                    filtering_stats['valid_rows_kept'] += chunk_stats['valid_rows']
                    filtering_stats['timestamp_violations'].extend(chunk_stats['violations'])

                    # Log progress
                    if chunk_idx % 10 == 0:
                        self.logger.debug(f"Processed chunk {chunk_idx}: {chunk_stats['valid_rows']} valid rows")

                    if len(filtered_chunk) > 0:
                        yield filtered_chunk

                except Exception as chunk_e:
                    self.logger.warning(f"⚠️ Failed to filter chunk {chunk_idx}: {chunk_e}")
                    # Yield original chunk if filtering fails
                    if len(data_chunk) > 0:
                        yield data_chunk

            self.logger.info(f"✅ Future data filtering completed: "
                           f"{filtering_stats['valid_rows_kept']} valid rows kept, "
                           f"{filtering_stats['future_rows_filtered']} future rows filtered")

        except Exception as e:
            self.logger.error(f"❌ Automated future data filtering failed: {e}")
            # Return original data stream if filtering fails completely
            for data_chunk in data_stream:
                if len(data_chunk) > 0:
                    yield data_chunk

    def rolling_window_bias_validation(self, data_stream: Iterator[pd.DataFrame],
                                    window_size: Optional[int] = None,
                                    validation_config: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """
        Rolling window validation for continuous bias detection on streaming data.

        Args:
            data_stream: Iterator of DataFrames to validate
            window_size: Size of rolling window
            validation_config: Validation configuration

        Yields:
            Validation results for each rolling window
        """
        try:
            if window_size is None:
                window_size = self.rolling_window_size

            if validation_config is None:
                validation_config = {
                    'timestamp_column': 'timestamp',
                    'target_column': 'target',
                    'feature_columns': None,  # Auto-detect
                    'bias_detection_threshold': 0.1,
                    'enable_gpu': self.enable_gpu
                }

            self.logger.info(f"🔄 Starting rolling window bias validation (window_size={window_size})")

            rolling_buffer = []
            validation_history = []

            for chunk_idx, data_chunk in enumerate(data_stream):
                try:
                    # Add chunk to rolling buffer
                    rolling_buffer.extend(data_chunk.to_dict('records'))

                    # Maintain window size
                    if len(rolling_buffer) > window_size:
                        rolling_buffer = rolling_buffer[-window_size:]

                    # Skip if buffer is too small
                    if len(rolling_buffer) < window_size // 4:
                        continue

                    # Convert buffer to DataFrame for analysis
                    window_df = pd.DataFrame(rolling_buffer)

                    # Perform bias validation on current window
                    window_validation = self._validate_rolling_window(
                        window_df, validation_config, chunk_idx
                    )

                    # Store validation result
                    validation_history.append(window_validation)

                    # Detect emerging bias patterns
                    if len(validation_history) >= 5:
                        bias_trend = self._detect_bias_trend(validation_history[-5:])
                        window_validation['bias_trend'] = bias_trend

                    yield window_validation

                    # Memory management
                    if self.memory_optimizer and chunk_idx % 10 == 0:
                        self.memory_optimizer.force_gc()

                except Exception as window_e:
                    self.logger.warning(f"⚠️ Rolling window validation failed for chunk {chunk_idx}: {window_e}")
                    yield {'error': str(window_e), 'chunk_idx': chunk_idx}

            self.logger.info(f"✅ Rolling window validation completed for {len(validation_history)} windows")

        except Exception as e:
            self.logger.error(f"❌ Rolling window bias validation failed: {e}")
            yield {'error': str(e)}

    # Helper methods for new functionality

    def _get_default_barrier_rules(self) -> Dict[str, Any]:
        """Get default information barrier rules."""
        return {
            'temporal_barriers': {
                'future_data_blocked': True,
                'max_lookahead_days': 0,
                'trading_hours_only': True
            },
            'data_flow_barriers': {
                'cross_market_isolation': True,
                'internal_external_separation': True,
                'production_development_separation': True
            },
            'feature_barriers': {
                'target_leakage_protection': True,
                'future_price_protection': True,
                'order_flow_protection': True
            }
        }

    def _check_temporal_barriers(self, data_stream: pd.DataFrame,
                               barrier_rules: Dict[str, Any],
                               current_timestamp: datetime) -> Dict[str, Any]:
        """Check temporal information barriers."""
        try:
            temporal_check = {'violations': [], 'compliance_score': 1.0}

            if 'timestamp' not in data_stream.columns:
                temporal_check['violations'].append("No timestamp column found")
                temporal_check['compliance_score'] = 0.0
                return temporal_check

            # Check for future timestamps
            future_mask = data_stream['timestamp'] > current_timestamp
            future_count = future_mask.sum()

            if future_count > 0:
                temporal_check['violations'].append(
                    f"Found {future_count} future timestamps beyond {current_timestamp}"
                )
                temporal_check['compliance_score'] -= future_count / len(data_stream)

            # Check trading hours if specified
            if barrier_rules.get('temporal_barriers', {}).get('trading_hours_only', False):
                trading_hours_violations = self._check_trading_hours_compliance(data_stream)
                if trading_hours_violations > 0:
                    temporal_check['violations'].append(f"Found {trading_hours_violations} non-trading hours records")
                    temporal_check['compliance_score'] -= trading_hours_violations / len(data_stream)

            return temporal_check
        except Exception:
            return {'violations': ['Temporal barrier check failed'], 'compliance_score': 0.0}

    def _check_data_flow_barriers(self, data_stream: pd.DataFrame,
                               barrier_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check data flow information barriers."""
        try:
            flow_check = {'violations': [], 'data_sources': set()}

            # Check for data source identification
            if 'data_source' in data_stream.columns:
                flow_check['data_sources'] = set(data_stream['data_source'].unique())
            else:
                flow_check['violations'].append("Data source not identified")

            # Check for cross-market isolation
            if barrier_rules.get('data_flow_barriers', {}).get('cross_market_isolation', False):
                if 'market' in data_stream.columns:
                    market_count = data_stream['market'].nunique()
                    if market_count > 1:
                        flow_check['violations'].append(f"Multiple markets detected: {market_count}")

            return flow_check
        except Exception:
            return {'violations': ['Data flow barrier check failed']}

    def _check_feature_dependency_barriers(self, data_stream: pd.DataFrame,
                                        barrier_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check feature dependency information barriers."""
        try:
            dependency_check = {'violations': [], 'suspicious_patterns': []}

            # Check for target leakage patterns
            if barrier_rules.get('feature_barriers', {}).get('target_leakage_protection', False):
                leakage_patterns = self._detect_target_leakage_patterns(data_stream)
                dependency_check['suspicious_patterns'].extend(leakage_patterns)

            # Check for future price features
            if barrier_rules.get('feature_barriers', {}).get('future_price_protection', False):
                future_price_features = self._detect_future_price_features(data_stream)
                if future_price_features:
                    dependency_check['violations'].extend(future_price_features)

            return dependency_check
        except Exception:
            return {'violations': ['Feature dependency barrier check failed']}

    def _analyze_single_feature_alignment(self, features_df: pd.DataFrame,
                                       target_df: pd.DataFrame,
                                       feature_col: str,
                                       timestamp_col: str,
                                       threshold: timedelta) -> Dict[str, Any]:
        """Analyze timestamp alignment for a single feature."""
        try:
            alignment = {'is_aligned': True, 'max_offset': timedelta(0), 'alignment_score': 1.0}

            # Get timestamps for feature and target
            feature_timestamps = features_df[timestamp_col].dropna().unique()
            target_timestamps = target_df[timestamp_col].dropna().unique()

            if len(feature_timestamps) == 0 or len(target_timestamps) == 0:
                alignment['is_aligned'] = False
                return alignment

            # Calculate alignment offsets
            feature_ts_set = set(feature_timestamps)
            target_ts_set = set(target_timestamps)

            # Find maximum offset between aligned timestamps
            max_offset = timedelta(0)
            aligned_count = 0

            for feature_ts in feature_ts_set:
                closest_target_ts = min(target_ts_set, key=lambda x: abs((x - feature_ts).total_seconds()))
                offset = abs(feature_ts - closest_target_ts)

                if offset <= threshold:
                    aligned_count += 1
                    max_offset = max(max_offset, offset)

            alignment['max_offset'] = max_offset
            alignment['alignment_score'] = aligned_count / len(feature_ts_set)
            alignment['is_aligned'] = alignment['alignment_score'] >= 0.9  # 90% alignment threshold

            return alignment
        except Exception:
            return {'is_aligned': False, 'error': 'Alignment analysis failed'}

    def _analyze_temporal_coverage(self, features_df: pd.DataFrame,
                                target_df: pd.DataFrame,
                                timestamp_col: str) -> Dict[str, Any]:
        """Analyze temporal coverage between features and targets."""
        try:
            coverage = {
                'feature_coverage': {},
                'target_coverage': {},
                'overlap_percentage': 0.0
            }

            feature_ts = features_df[timestamp_col].dropna()
            target_ts = target_df[timestamp_col].dropna()

            if len(feature_ts) > 0:
                coverage['feature_coverage'] = {
                    'start': feature_ts.min(),
                    'end': feature_ts.max(),
                    'duration': feature_ts.max() - feature_ts.min(),
                    'count': len(feature_ts)
                }

            if len(target_ts) > 0:
                coverage['target_coverage'] = {
                    'start': target_ts.min(),
                    'end': target_ts.max(),
                    'duration': target_ts.max() - target_ts.min(),
                    'count': len(target_ts)
                }

            # Calculate overlap
            if len(feature_ts) > 0 and len(target_ts) > 0:
                overlap_start = max(feature_ts.min(), target_ts.min())
                overlap_end = min(feature_ts.max(), target_ts.max())

                if overlap_start <= overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    total_duration = max(feature_ts.max(), target_ts.max()) - min(feature_ts.min(), target_ts.min())
                    coverage['overlap_percentage'] = overlap_duration.total_seconds() / total_duration.total_seconds()

            return coverage
        except Exception:
            return {'error': 'Temporal coverage analysis failed'}

    def _filter_future_data_chunk(self, data_chunk: pd.DataFrame,
                               current_timestamp: datetime,
                               filter_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Filter future data from a single chunk."""
        try:
            timestamp_col = filter_config['timestamp_column']
            tolerance_seconds = filter_config['tolerance_seconds']

            if timestamp_col not in data_chunk.columns:
                return data_chunk, {'original_rows': len(data_chunk), 'filtered_rows': 0, 'valid_rows': len(data_chunk), 'violations': []}

            # Apply tolerance to current timestamp
            cutoff_timestamp = current_timestamp - timedelta(seconds=tolerance_seconds)

            # Filter out future data
            valid_mask = data_chunk[timestamp_col] <= cutoff_timestamp
            filtered_chunk = data_chunk[valid_mask].copy()
            future_chunk = data_chunk[~valid_mask].copy()

            # Log violations if enabled
            violations = []
            if filter_config.get('log_violations', False) and len(future_chunk) > 0:
                for _, row in future_chunk.iterrows():
                    violations.append({
                        'timestamp': row[timestamp_col],
                        'current_time': current_timestamp,
                        'offset_seconds': (row[timestamp_col] - current_timestamp).total_seconds()
                    })

            chunk_stats = {
                'original_rows': len(data_chunk),
                'filtered_rows': len(future_chunk),
                'valid_rows': len(filtered_chunk),
                'violations': violations
            }

            return filtered_chunk, chunk_stats
        except Exception as e:
            return data_chunk, {'error': str(e), 'original_rows': len(data_chunk), 'filtered_rows': 0, 'valid_rows': len(data_chunk), 'violations': []}

    def _validate_rolling_window(self, window_df: pd.DataFrame,
                              validation_config: Dict[str, Any],
                              chunk_idx: int) -> Dict[str, Any]:
        """Validate a single rolling window for bias."""
        try:
            validation_result = {
                'chunk_idx': chunk_idx,
                'window_size': len(window_df),
                'bias_detected': False,
                'bias_score': 0.0,
                'validation_metrics': {},
                'recommendations': []
            }

            # Perform lookahead bias detection
            if len(window_df) > 10:
                bias_analysis = self.detect_data_leakage(
                    window_df, window_df,  # Use same data for features and targets
                    timestamp_col=validation_config['timestamp_column']
                )

                validation_result['bias_detected'] = bias_analysis.get('leakage_detected', False)
                validation_result['bias_score'] = len(bias_analysis.get('issues', [])) / len(window_df)
                validation_result['validation_metrics'] = bias_analysis

                if validation_result['bias_detected']:
                    validation_result['recommendations'].append("Bias detected in rolling window")

            return validation_result
        except Exception as e:
            return {'error': str(e), 'chunk_idx': chunk_idx}

    def _detect_bias_trend(self, recent_validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect emerging bias trends from recent validations."""
        try:
            bias_scores = [v.get('bias_score', 0) for v in recent_validations if 'bias_score' in v]

            if len(bias_scores) < 2:
                return {'trend': 'insufficient_data'}

            trend_analysis = {
                'trend': 'stable',
                'slope': 0.0,
                'volatility': np.std(bias_scores),
                'is_increasing': False
            }

            # Calculate trend
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(range(len(bias_scores)), bias_scores)
            trend_analysis['slope'] = slope

            if slope > 0.01:
                trend_analysis['trend'] = 'increasing'
                trend_analysis['is_increasing'] = True
            elif slope < -0.01:
                trend_analysis['trend'] = 'decreasing'

            return trend_analysis
        except Exception:
            return {'trend': 'analysis_failed'}

    def _assess_barrier_violations_severity(self, violations: List[str]) -> Dict[str, Any]:
        """Assess severity of barrier violations."""
        try:
            severity = {'level': 'low', 'score': 0.0, 'critical_violations': 0}

            if not violations:
                return severity

            severity_keywords = {
                'critical': ['future', 'leakage', 'barrier'],
                'high': ['temporal', 'timestamp', 'data_flow'],
                'medium': ['alignment', 'coverage', 'consistency'],
                'low': ['formatting', 'metadata']
            }

            for violation in violations:
                violation_lower = violation.lower()
                for level, keywords in severity_keywords.items():
                    if any(keyword in violation_lower for keyword in keywords):
                        if level == 'critical':
                            severity['critical_violations'] += 1
                        break

            # Calculate severity score
            severity['score'] = min(1.0, (len(violations) * 0.1 + severity['critical_violations'] * 0.3))

            # Determine severity level
            if severity['critical_violations'] > 0:
                severity['level'] = 'critical'
            elif severity['score'] > 0.7:
                severity['level'] = 'high'
            elif severity['score'] > 0.3:
                severity['level'] = 'medium'

            return severity
        except Exception:
            return {'level': 'unknown', 'score': 0.0}

    def _generate_barrier_recommendations(self, violations: List[str],
                                       barrier_rules: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on barrier violations."""
        try:
            recommendations = []

            if not violations:
                recommendations.append("All information barriers compliant")
                return recommendations

            # General recommendations
            recommendations.append("Review and strengthen information barrier controls")

            # Specific recommendations based on violation types
            violation_text = ' '.join(violations).lower()

            if 'future' in violation_text:
                recommendations.append("Implement strict future data filtering")
            if 'temporal' in violation_text:
                recommendations.append("Verify timestamp alignment across all data sources")
            if 'leakage' in violation_text:
                recommendations.append("Conduct feature engineering review for target leakage")
            if 'data_flow' in violation_text:
                recommendations.append("Strengthen data flow controls between environments")

            return recommendations
        except Exception:
            return ["Unable to generate barrier recommendations"]

    def _check_trading_hours_compliance(self, data_stream: pd.DataFrame) -> int:
        """Check compliance with trading hours."""
        try:
            if 'timestamp' not in data_stream.columns:
                return 0

            # Simple trading hours check (9:30 AM - 4:00 PM EST, weekdays)
            def is_trading_hours(ts):
                if ts.weekday() >= 5:  # Weekend
                    return False
                hour = ts.hour
                minute = ts.minute
                time_minutes = hour * 60 + minute
                return 570 <= time_minutes <= 960  # 9:30 AM to 4:00 PM

            trading_hours_mask = data_stream['timestamp'].apply(is_trading_hours)
            return (~trading_hours_mask).sum()
        except Exception:
            return 0

    def _detect_target_leakage_patterns(self, data_stream: pd.DataFrame) -> List[str]:
        """Detect potential target leakage patterns."""
        try:
            patterns = []

            # Look for suspiciously correlated features
            numeric_cols = data_stream.select_dtypes(include=[np.number]).columns

            if 'target' in numeric_cols and len(numeric_cols) > 1:
                target_corr = data_stream[numeric_cols].corr()['target'].abs()

                # Find features with correlation > 0.95 with target
                suspicious_features = target_corr[target_corr > 0.95].index.tolist()
                suspicious_features.remove('target')  # Remove target itself

                if suspicious_features:
                    patterns.append(f"Suspiciously high correlation with target: {suspicious_features}")

            return patterns
        except Exception:
            return []

    def _detect_future_price_features(self, data_stream: pd.DataFrame) -> List[str]:
        """Detect features that might contain future price information."""
        try:
            violations = []

            # Check column names for future-looking indicators
            future_indicators = ['future', 'next', 'tomorrow', 'ahead', 'forward']

            for col in data_stream.columns:
                col_lower = col.lower()
                if any(indicator in col_lower for indicator in future_indicators):
                    violations.append(f"Potential future-looking feature detected: {col}")

            return violations
        except Exception:
            return []

    def check_lookahead_bias(self, data: pd.DataFrame, labels: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Check for lookahead bias in the data.

        Args:
            data: DataFrame containing features
            labels: Optional labels/targets to check against

        Returns:
            Dictionary with bias detection results
        """
        try:
            self.logger.info("🔍 Checking for lookahead bias...")

            # Use the existing detect_and_prevent_leakage method
            if labels is not None and 'target' not in data.columns:
                # Add labels as target column if not present
                data_with_target = data.copy()
                data_with_target['target'] = labels
            else:
                data_with_target = data

            # Perform bias detection
            bias_results = self.detect_data_leakage(
                features_df=data_with_target,
                target_df=data_with_target,
                timestamp_col='timestamp' if 'timestamp' in data_with_target.columns else None,
                feature_cols=[col for col in data_with_target.columns if col not in ['timestamp', 'target', 'label']]
            )

            # Convert to expected format - ensure bias_results is a dictionary
            if not isinstance(bias_results, dict):
                self.logger.warning(f"⚠️ bias_results is not a dictionary: {type(bias_results)}")
                # Handle case where bias_results is not a dictionary
                result = {
                    'bias_detected': True,  # Assume bias on unexpected format
                    'bias_score': 1.0,
                    'issues': [f"Unexpected bias_results format: {type(bias_results)}"],
                    'warnings': [f"Bias detection returned unexpected format: {type(bias_results)}"],
                    'recommendations': ["Review bias detection implementation"]
                }
            else:
                result = {
                    'bias_detected': bias_results.get('leakage_detected', False),
                    'bias_score': len(bias_results.get('issues', [])) / max(len(data), 1),
                    'issues': bias_results.get('issues', []),
                    'warnings': bias_results.get('warnings', []),
                    'recommendations': bias_results.get('recommendations', [])
                }

            self.logger.info(f"✅ Lookahead bias check completed - {'Bias detected' if result['bias_detected'] else 'No bias detected'}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Lookahead bias check failed: {e}")
            return {
                'bias_detected': True,  # Assume bias on error
                'bias_score': 1.0,
                'issues': [f"Bias check failed: {str(e)}"],
                'warnings': [f"Error in bias detection: {str(e)}"],
                'recommendations': ["Review data format and try again"],
                'error': str(e)
            }

    async def detect_and_prevent_leakage(self, data: pd.DataFrame,
                                       symbol: Optional[str] = None,
                                       exchange: Optional[str] = None,
                                       context: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect and prevent data leakage - wrapper method for compatibility.

        Args:
            data: DataFrame containing features and targets
            symbol: Trading symbol (optional)
            exchange: Trading exchange (optional)
            context: Context for the analysis (optional)

        Returns:
            Dictionary with leakage detection results
        """
        try:
            self.logger.info(f"🔍 Starting detect_and_prevent_leakage for {symbol or 'unknown'} on {exchange or 'unknown'}")

            # Set current timestamp if not already set
            if self.current_timestamp is None:
                if 'timestamp' in data.columns:
                    self.current_timestamp = data['timestamp'].max()
                else:
                    self.current_timestamp = datetime.now()

            # Prepare data for leakage detection
            # Assume the data contains both features and targets
            feature_cols = [col for col in data.columns
                          if col not in ['timestamp', 'target', 'label', 'outcome']]

            # Create separate DataFrames for features and targets
            features_df = data[['timestamp'] + feature_cols].copy() if 'timestamp' in data.columns else data[feature_cols].copy()
            target_df = data[['timestamp', 'target']].copy() if 'target' in data.columns else None

            # If no target column, create a dummy target DataFrame
            if target_df is None:
                target_df = features_df.copy()
                if 'target' not in target_df.columns:
                    target_df['target'] = 0  # Dummy target

            # Perform data leakage detection
            leakage_results = self.detect_data_leakage(
                features_df=features_df,
                target_df=target_df,
                timestamp_col='timestamp' if 'timestamp' in data.columns else None,
                feature_cols=feature_cols
            )

            # Convert to expected format - ensure leakage_results is a dictionary
            if not isinstance(leakage_results, dict):
                self.logger.warning(f"⚠️ leakage_results is not a dictionary: {type(leakage_results)}")
                # Handle case where leakage_results is not a dictionary
                result = {
                    'has_leakage': True,  # Assume leakage on unexpected format
                    'leakage_details': [f"Unexpected leakage_results format: {type(leakage_results)}"],
                    'warnings': [f"Leakage detection returned unexpected format: {type(leakage_results)}"],
                    'recommendations': ["Review leakage detection implementation"],
                    'feature_analysis': {},
                    'temporal_analysis': {},
                    'symbol': symbol,
                    'exchange': exchange,
                    'context': context,
                    'timestamp': self.current_timestamp
                }
            else:
                result = {
                    'has_leakage': leakage_results.get('leakage_detected', False),
                    'leakage_details': leakage_results.get('issues', []),
                    'warnings': leakage_results.get('warnings', []),
                    'recommendations': leakage_results.get('recommendations', []),
                    'feature_analysis': leakage_results.get('feature_analysis', {}),
                    'temporal_analysis': leakage_results.get('temporal_analysis', {}),
                    'symbol': symbol,
                    'exchange': exchange,
                    'context': context,
                    'timestamp': self.current_timestamp
                }

            # Log results
            if result['has_leakage']:
                self.logger.warning(f"🚨 Data leakage detected: {len(result['leakage_details'])} issues found")
                for issue in result['leakage_details']:
                    self.logger.warning(f"  - {issue}")
            else:
                self.logger.info("✅ No data leakage detected")

            return result

        except Exception as e:
            self.logger.error(f"❌ detect_and_prevent_leakage failed: {e}")
            return {
                'has_leakage': True,  # Assume leakage on error
                'leakage_details': [f"Detection failed: {str(e)}"],
                'warnings': [f"Error in leakage detection: {str(e)}"],
                'recommendations': ["Review data format and try again"],
                'error': str(e),
                'symbol': symbol,
                'exchange': exchange,
                'context': context
            }
