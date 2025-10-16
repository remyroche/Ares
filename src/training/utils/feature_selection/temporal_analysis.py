from src.utils.tprint import tprint

"""
Temporal Analysis Component

This module provides temporal analysis capabilities for feature selection,
including time-based feature importance analysis and regime-specific selection.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
from collections import defaultdict

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.TemporalAnalysis")
    tprint("✅ Custom logger available for FeatureSelection.TemporalAnalysis")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.TemporalAnalysis")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited temporal analysis functionality")

# Import optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.matrix_operations import get_unified_matrix_operations
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logger.warning("⚠️ Optimization utilities not available - using standard operations")

# Import common operations utilities
try:
    from src.utils.ml_common.utils import get_memory_usage
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError:
    COMMON_OPERATIONS_AVAILABLE = False

# Set matrix operations availability based on optimization imports
MATRIX_OPERATIONS_AVAILABLE = OPTIMIZATION_AVAILABLE

class TemporalAnalyzer:
    """Temporal analysis for feature selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize temporal analyzer."""
        self.config = config or {}
        self.logger = logger.getChild('TemporalAnalyzer')

        # Temporal analysis parameters
        self.window_sizes = self.config.get('window_sizes', [100, 200, 500])
        self.overlap_ratio = self.config.get('overlap_ratio', 0.5)
        self.min_window_size = self.config.get('min_window_size', 50)
        self.regime_detection_threshold = self.config.get('regime_detection_threshold', 0.1)

        # Initialize optimization tools
        self._initialize_optimization_tools()

        _LOGGER.info("⏰ TemporalAnalyzer initialized")
        _LOGGER.info(f"⚙️ Window sizes: {self.window_sizes}")
        _LOGGER.info(f"⚙️ Overlap ratio: {self.overlap_ratio}")

    def _initialize_optimization_tools(self):
        """Initialize hardware optimization utilities."""
        try:
            if OPTIMIZATION_AVAILABLE and COMMON_OPERATIONS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                if self.gpu_manager:
                    _LOGGER.info("✅ M1 GPU manager initialized for temporal analysis")
                if self.memory_optimizer:
                    _LOGGER.info("✅ M1 memory optimizer initialized for temporal analysis")
                if self.cpu_optimizer:
                    _LOGGER.info("✅ M1 CPU optimizer initialized for temporal analysis")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

        try:
            if OPTIMIZATION_AVAILABLE and MATRIX_OPERATIONS_AVAILABLE:
                self.matrix_ops = get_unified_matrix_operations()
                _LOGGER.info("✅ Unified matrix operations initialized for temporal analysis")
            else:
                self.matrix_ops = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ Matrix operations initialization failed: {e}")
            self.matrix_ops = None

    def analyze_temporal_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: List[str],
                                          temporal_indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze temporal evolution of feature importance."""
        start_time = time.time()
        _LOGGER.info(f"⏰ Starting temporal feature importance analysis...")
        _LOGGER.info(f"📊 Parameters - Data shape: {X.shape}, Window sizes: {self.window_sizes}")

        try:
            n_samples, n_features = X.shape

            # Use provided temporal indices or create default
            if temporal_indices is None:
                temporal_indices = np.arange(n_samples)

            # Analyze each window size
            window_results = {}
            feature_temporal_importance = defaultdict(list)

            for window_size in self.window_sizes:
                if window_size < self.min_window_size or window_size > n_samples:
                    _LOGGER.warning(f"⚠️ Window size {window_size} invalid, skipping")
                    continue

                _LOGGER.debug(f"🔄 Analyzing window size: {window_size}")

                # Calculate step size based on overlap
                step_size = int(window_size * (1 - self.overlap_ratio))
                if step_size < 1:
                    step_size = 1

                # Analyze sliding windows
                window_importances = []
                window_positions = []

                for start_idx in range(0, n_samples - window_size + 1, step_size):
                    end_idx = start_idx + window_size

                    X_window = X[start_idx:end_idx]
                    y_window = y[start_idx:end_idx]

                    # Calculate feature importance for this window
                    importance_scores = self._calculate_window_feature_importance(
                        X_window, y_window, feature_names
                    )

                    if importance_scores:
                        window_importances.append(importance_scores)
                        window_positions.append((start_idx, end_idx))

                        # Record temporal importance
                        for feature, importance in importance_scores.items():
                            feature_temporal_importance[feature].append({
                                'window_start': start_idx,
                                'window_end': end_idx,
                                'window_size': window_size,
                                'importance': importance,
                                'temporal_position': (start_idx + end_idx) / 2
                            })

                window_results[window_size] = {
                    'window_importances': window_importances,
                    'window_positions': window_positions,
                    'n_windows': len(window_importances)
                }

            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(feature_temporal_importance)

            # Analyze cross-timeframe behavior
            cross_timeframe_analysis = self._analyze_cross_timeframe_behavior(feature_temporal_importance)

            # Identify optimal features by timeframe
            optimal_features_by_timeframe = self._identify_optimal_features_by_timeframe(feature_temporal_importance)

            # Calculate temporal decay
            temporal_decay = self._calculate_temporal_decay(feature_temporal_importance)

            # Calculate temporal scores for each feature
            temporal_scores = self._calculate_temporal_scores(feature_temporal_importance, feature_names)

            execution_time = time.time() - start_time

            result = {
                'window_results': window_results,
                'feature_temporal_importance': dict(feature_temporal_importance),
                'temporal_patterns': temporal_patterns,
                'cross_timeframe_analysis': cross_timeframe_analysis,
                'optimal_features_by_timeframe': optimal_features_by_timeframe,
                'temporal_decay': temporal_decay,
                'temporal_scores': temporal_scores,  # NEW: Individual feature scores
                'method': 'temporal_importance_analysis',
                'parameters': {
                    'window_sizes': self.window_sizes,
                    'overlap_ratio': self.overlap_ratio,
                    'min_window_size': self.min_window_size
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Temporal feature importance analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Analyzed {len(window_results)} window sizes")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Temporal feature importance analysis failed: {e}")
            return {
                'window_results': {},
                'feature_temporal_importance': {},
                'temporal_patterns': {},
                'cross_timeframe_analysis': {},
                'optimal_features_by_timeframe': {},
                'temporal_decay': {},
                'method': 'temporal_importance_analysis',
                'error': str(e),
                'success': False
            }

    def analyze_regime_specific_importance(self, X: np.ndarray, y: np.ndarray,
                                         feature_names: List[str],
                                         regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze feature importance across different regimes."""
        start_time = time.time()
        _LOGGER.info(f"⏰ Starting regime-specific importance analysis...")
        _LOGGER.info(f"📊 Parameters - Data shape: {X.shape}")

        try:
            n_samples, n_features = X.shape

            # Auto-detect regimes if not provided
            if regime_labels is None:
                regime_labels = self._detect_regimes(y)

            unique_regimes = np.unique(regime_labels)
            _LOGGER.info(f"📊 Detected {len(unique_regimes)} regimes: {unique_regimes}")

            # Analyze each regime
            regime_results = {}
            regime_feature_importance = {}

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]

                if len(X_regime) < self.min_window_size:
                    _LOGGER.warning(f"⚠️ Regime {regime} has insufficient data: {len(X_regime)} samples")
                    continue

                _LOGGER.debug(f"🔄 Analyzing regime: {regime} ({len(X_regime)} samples)")

                # Calculate feature importance for this regime
                regime_importance = self._calculate_regime_feature_importance(
                    X_regime, y_regime, feature_names, regime
                )

                regime_results[regime] = {
                    'n_samples': len(X_regime),
                    'feature_importance': regime_importance,
                    'regime_stats': {
                        'mean_target': np.mean(y_regime),
                        'std_target': np.std(y_regime),
                        'min_target': np.min(y_regime),
                        'max_target': np.max(y_regime)
                    }
                }

                regime_feature_importance[regime] = regime_importance

            # Analyze regime differences
            regime_differences = self._analyze_regime_differences(regime_feature_importance)

            # Identify regime-specific features
            regime_specific_features = self._identify_regime_specific_features(regime_feature_importance)

            execution_time = time.time() - start_time

            result = {
                'regime_results': regime_results,
                'regime_feature_importance': regime_feature_importance,
                'regime_differences': regime_differences,
                'regime_specific_features': regime_specific_features,
                'unique_regimes': unique_regimes.tolist(),
                'method': 'regime_specific_analysis',
                'parameters': {
                    'min_window_size': self.min_window_size,
                    'regime_detection_threshold': self.regime_detection_threshold
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Regime-specific importance analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Analyzed {len(unique_regimes)} regimes")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Regime-specific importance analysis failed: {e}")
            return {
                'regime_results': {},
                'regime_feature_importance': {},
                'regime_differences': {},
                'regime_specific_features': {},
                'method': 'regime_specific_analysis',
                'error': str(e),
                'success': False
            }

    def _calculate_window_feature_importance(self, X_window: np.ndarray, y_window: np.ndarray,
                                           feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for a specific window."""
        try:
            if not SKLEARN_AVAILABLE or len(X_window) < 10:
                # Fallback to simple correlation-based importance
                importances = {}
                for i, feature in enumerate(feature_names):
                    corr = np.corrcoef(X_window[:, i], y_window)[0, 1]
                    importances[feature] = abs(corr) if not np.isnan(corr) else 0.0
                return importances

            # Use memory optimization if available
            if self.memory_optimizer:
                memory_status = self.memory_optimizer.check_memory_status()
                if memory_status.get('pressure', False):
                    _LOGGER.debug("🧠 Memory pressure detected, using optimized parameters")
                    n_estimators = 25  # Reduce for memory pressure
                else:
                    n_estimators = 50
            else:
                n_estimators = 50

            # Use tree-based model for importance
            is_classification = len(np.unique(y_window)) <= 10

            if is_classification:
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

            model.fit(X_window, y_window)
            importances = model.feature_importances_

            # Use optimized matrix operations if available
            if self.matrix_ops:
                # Use optimized operations for importance processing
                importances = self.matrix_ops.normalize_vector(importances)

            # Ensure proper mapping between importances and feature names
            return {feature_names[i]: importances[i] for i in range(min(len(feature_names), len(importances)))}

        except Exception as e:
            _LOGGER.debug(f"⚠️ Window importance calculation failed: {e}")
            return {}

    def _analyze_temporal_patterns(self, feature_temporal_importance: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze temporal patterns in feature importance."""
        try:
            patterns = {}

            for feature, importance_history in feature_temporal_importance.items():
                if not importance_history:
                    continue

                # Extract importance values and temporal positions
                importances = [item['importance'] for item in importance_history]
                temporal_positions = [item['temporal_position'] for item in importance_history]

                # Calculate temporal trend
                if len(importances) > 1:
                    # Simple linear trend
                    temporal_trend = np.polyfit(temporal_positions, importances, 1)[0]

                    # Calculate volatility
                    importance_volatility = np.std(importances)

                    # Calculate stability (inverse of volatility)
                    importance_stability = 1.0 / (1.0 + importance_volatility)

                    patterns[feature] = {
                        'temporal_trend': temporal_trend,
                        'importance_volatility': importance_volatility,
                        'importance_stability': importance_stability,
                        'mean_importance': np.mean(importances),
                        'max_importance': np.max(importances),
                        'min_importance': np.min(importances),
                        'n_observations': len(importances)
                    }

            return patterns

        except Exception as e:
            _LOGGER.warning(f"⚠️ Temporal pattern analysis failed: {e}")
            return {}

    def _analyze_cross_timeframe_behavior(self, feature_temporal_importance: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze cross-timeframe behavior of features."""
        try:
            cross_timeframe_analysis = {}

            # Group by window size
            window_sizes = set()
            for importance_history in feature_temporal_importance.values():
                for item in importance_history:
                    window_sizes.add(item['window_size'])

            for window_size in window_sizes:
                window_analysis = {}

                for feature, importance_history in feature_temporal_importance.items():
                    # Filter for this window size
                    window_importances = [item['importance'] for item in importance_history
                                        if item['window_size'] == window_size]

                    if window_importances:
                        window_analysis[feature] = {
                            'mean_importance': np.mean(window_importances),
                            'std_importance': np.std(window_importances),
                            'n_observations': len(window_importances)
                        }

                cross_timeframe_analysis[window_size] = window_analysis

            return cross_timeframe_analysis

        except Exception as e:
            _LOGGER.warning(f"⚠️ Cross-timeframe analysis failed: {e}")
            return {}

    def _identify_optimal_features_by_timeframe(self, feature_temporal_importance: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Identify optimal features for different timeframes."""
        try:
            optimal_features = {}

            # Group by window size
            window_sizes = set()
            for importance_history in feature_temporal_importance.values():
                for item in importance_history:
                    window_sizes.add(item['window_size'])

            for window_size in window_sizes:
                feature_scores = {}

                for feature, importance_history in feature_temporal_importance.items():
                    # Filter for this window size
                    window_importances = [item['importance'] for item in importance_history
                                        if item['window_size'] == window_size]

                    if window_importances:
                        # Score based on mean importance and stability
                        mean_importance = np.mean(window_importances)
                        stability = 1.0 / (1.0 + np.std(window_importances))
                        feature_scores[feature] = mean_importance * stability

                # Sort features by score
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                optimal_features[window_size] = sorted_features

            return optimal_features

        except Exception as e:
            _LOGGER.warning(f"⚠️ Optimal feature identification failed: {e}")
            return {}

    def _calculate_temporal_decay(self, feature_temporal_importance: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate temporal decay of feature importance."""
        try:
            temporal_decay = {}

            for feature, importance_history in feature_temporal_importance.items():
                if len(importance_history) < 2:
                    continue

                # Sort by temporal position
                sorted_history = sorted(importance_history, key=lambda x: x['temporal_position'])
                importances = [item['importance'] for item in sorted_history]
                temporal_positions = [item['temporal_position'] for item in sorted_history]

                # Calculate decay rate
                if len(importances) > 1:
                    # Simple exponential decay fit
                    try:
                        # Normalize temporal positions
                        norm_positions = np.array(temporal_positions) - temporal_positions[0]
                        norm_positions = norm_positions / (norm_positions[-1] + 1e-10)

                        # Fit exponential decay
                        log_importances = np.log(np.maximum(importances, 1e-10))
                        decay_rate = np.polyfit(norm_positions, log_importances, 1)[0]

                        temporal_decay[feature] = {
                            'decay_rate': decay_rate,
                            'initial_importance': importances[0],
                            'final_importance': importances[-1],
                            'decay_ratio': importances[-1] / (importances[0] + 1e-10)
                        }
                    except Exception:
                        temporal_decay[feature] = {
                            'decay_rate': 0.0,
                            'initial_importance': importances[0],
                            'final_importance': importances[-1],
                            'decay_ratio': 1.0
                        }

            return temporal_decay

        except Exception as e:
            _LOGGER.warning(f"⚠️ Temporal decay calculation failed: {e}")
            return {}

    def _detect_regimes(self, y: np.ndarray) -> np.ndarray:
        """Auto-detect regimes in the target variable."""
        try:
            # Simple regime detection based on target value changes
            regime_labels = np.zeros(len(y), dtype=int)
            current_regime = 0

            # Use rolling window to detect regime changes
            window_size = min(50, len(y) // 10)
            if window_size < 10:
                return regime_labels

            for i in range(window_size, len(y)):
                # Calculate rolling statistics
                window_data = y[i-window_size:i]
                current_value = y[i]

                # Detect significant change
                window_mean = np.mean(window_data)
                window_std = np.std(window_data)

                if window_std > 0:
                    z_score = abs(current_value - window_mean) / window_std
                    if z_score > 2.0:  # Significant change
                        current_regime += 1

                regime_labels[i] = current_regime

            return regime_labels

        except Exception as e:
            _LOGGER.warning(f"⚠️ Regime detection failed: {e}")
            return np.zeros(len(y), dtype=int)

    def _calculate_regime_feature_importance(self, X_regime: np.ndarray, y_regime: np.ndarray,
                                           feature_names: List[str], regime: Any) -> Dict[str, float]:
        """Calculate feature importance for a specific regime."""
        try:
            if not SKLEARN_AVAILABLE or len(X_regime) < 10:
                # Fallback to correlation-based importance
                importances = {}
                for i, feature in enumerate(feature_names):
                    corr = np.corrcoef(X_regime[:, i], y_regime)[0, 1]
                    importances[feature] = abs(corr) if not np.isnan(corr) else 0.0
                return importances

            # Use tree-based model
            is_classification = len(np.unique(y_regime)) <= 10

            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            model.fit(X_regime, y_regime)
            importances = model.feature_importances_

            # Ensure proper mapping between importances and feature names
            return {feature_names[i]: importances[i] for i in range(min(len(feature_names), len(importances)))}

        except Exception as e:
            _LOGGER.debug(f"⚠️ Regime importance calculation failed: {e}")
            return {}

    def _analyze_regime_differences(self, regime_feature_importance: Dict[Any, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze differences in feature importance across regimes."""
        try:
            regime_differences = {}

            # Get all features
            all_features = set()
            for regime_importance in regime_feature_importance.values():
                all_features.update(regime_importance.keys())

            # Calculate differences for each feature
            for feature in all_features:
                feature_importances = []
                regimes = []

                for regime, importance_dict in regime_feature_importance.items():
                    if feature in importance_dict:
                        feature_importances.append(importance_dict[feature])
                        regimes.append(regime)

                if len(feature_importances) > 1:
                    regime_differences[feature] = {
                        'mean_importance': np.mean(feature_importances),
                        'std_importance': np.std(feature_importances),
                        'max_importance': np.max(feature_importances),
                        'min_importance': np.min(feature_importances),
                        'importance_range': np.max(feature_importances) - np.min(feature_importances),
                        'regimes': regimes,
                        'regime_importances': dict(zip(regimes, feature_importances))
                    }

            return regime_differences

        except Exception as e:
            _LOGGER.warning(f"⚠️ Regime difference analysis failed: {e}")
            return {}

    def _identify_regime_specific_features(self, regime_feature_importance: Dict[Any, Dict[str, float]]) -> Dict[str, Any]:
        """Identify features that are specific to certain regimes."""
        try:
            regime_specific_features = {}

            # Get all features and regimes
            all_features = set()
            all_regimes = list(regime_feature_importance.keys())

            for regime_importance in regime_feature_importance.values():
                all_features.update(regime_importance.keys())

            # For each feature, identify which regimes it's most important in
            for feature in all_features:
                feature_regime_importance = {}

                for regime, importance_dict in regime_feature_importance.items():
                    if feature in importance_dict:
                        feature_regime_importance[regime] = importance_dict[feature]

                if feature_regime_importance:
                    # Find regime with highest importance
                    best_regime = max(feature_regime_importance.items(), key=lambda x: x[1])

                    # Calculate regime specificity (how much more important in best regime)
                    mean_importance = np.mean(list(feature_regime_importance.values()))
                    specificity_ratio = best_regime[1] / (mean_importance + 1e-10)

                    regime_specific_features[feature] = {
                        'best_regime': best_regime[0],
                        'best_regime_importance': best_regime[1],
                        'mean_importance': mean_importance,
                        'specificity_ratio': specificity_ratio,
                        'regime_importances': feature_regime_importance
                    }

            return regime_specific_features

        except Exception as e:
            _LOGGER.warning(f"⚠️ Regime-specific feature identification failed: {e}")
            return {}

    def _calculate_temporal_scores(self, feature_temporal_importance: Dict[str, List[Dict]],
                                 feature_names: List[str]) -> Dict[str, float]:
        """Calculate temporal scores for each feature based on temporal stability and importance."""
        try:
            temporal_scores = {}

            for feature in feature_names:
                if feature not in feature_temporal_importance:
                    temporal_scores[feature] = 0.0
                    continue

                feature_data = feature_temporal_importance[feature]
                if not feature_data:
                    temporal_scores[feature] = 0.0
                    continue

                # Extract importance values across all windows
                importances = [item['importance'] for item in feature_data]

                if not importances:
                    temporal_scores[feature] = 0.0
                    continue

                # Calculate temporal stability metrics
                mean_importance = np.mean(importances)
                std_importance = np.std(importances)
                min_importance = np.min(importances)
                max_importance = np.max(importances)

                # Calculate temporal consistency (inverse of coefficient of variation)
                if mean_importance > 0:
                    coefficient_of_variation = std_importance / mean_importance
                    temporal_consistency = 1.0 / (1.0 + coefficient_of_variation)
                else:
                    temporal_consistency = 0.0

                # Calculate temporal range (how much importance varies)
                if max_importance > min_importance:
                    temporal_range = (max_importance - min_importance) / max_importance
                else:
                    temporal_range = 0.0

                # Calculate temporal trend (linear trend over time)
                temporal_positions = [item['temporal_position'] for item in feature_data]
                if len(temporal_positions) > 1:
                    # Calculate correlation between time and importance
                    temporal_trend = np.corrcoef(temporal_positions, importances)[0, 1]
                    if np.isnan(temporal_trend):
                        temporal_trend = 0.0
                else:
                    temporal_trend = 0.0

                # Calculate temporal persistence (how often feature is important)
                importance_threshold = np.percentile(importances, 50)  # Median as threshold
                persistence_ratio = np.mean([imp >= importance_threshold for imp in importances])

                # Combine metrics into temporal score
                # Weight: mean importance (40%), consistency (25%), persistence (20%), trend (15%)
                temporal_score = (
                    mean_importance * 0.4 +
                    temporal_consistency * 0.25 +
                    persistence_ratio * 0.2 +
                    abs(temporal_trend) * 0.15  # Use absolute trend (both positive and negative trends are valuable)
                )

                # Normalize to 0-1 range
                temporal_score = max(0.0, min(1.0, temporal_score))
                temporal_scores[feature] = temporal_score

            # Normalize all scores to 0-1 range
            if temporal_scores:
                max_score = max(temporal_scores.values())
                min_score = min(temporal_scores.values())
                if max_score > min_score:
                    for feature in temporal_scores:
                        temporal_scores[feature] = (temporal_scores[feature] - min_score) / (max_score - min_score)

            _LOGGER.info(f"📊 Calculated temporal scores for {len(temporal_scores)} features")
            return temporal_scores

        except Exception as e:
            _LOGGER.error(f"❌ Temporal score calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}
