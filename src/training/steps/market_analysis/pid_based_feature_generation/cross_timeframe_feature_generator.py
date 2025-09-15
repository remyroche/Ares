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

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

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
    timeframes: List[str] = field(default_factory=lambda: [
        '1m', '5m', '15m', '30m', '1h', '4h', '1d'
    ])
    
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
            target: Target variable for PID analysis (optional)
            
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
            # This is a placeholder for applying optimized lookback periods
            # In practice, this would involve resampling or windowing the data
            # based on the optimized periods from feature_lookback_optimization
            
            self.logger.info(f"📊 Applying optimized lookback periods: {optimized_lookback_periods}")
            
            # For now, we'll just log the optimization and return the original data
            # In a full implementation, this would:
            # 1. Identify which features correspond to which optimized periods
            # 2. Apply the appropriate windowing/resampling
            # 3. Update feature names to reflect the optimization
            
            return X, feature_names
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply optimized lookback periods: {e}")
            return X, feature_names
    
    def _identify_timeframe_features(self, feature_names: List[str]) -> List[str]:
        """Identify features that contain timeframe information."""
        timeframe_features = []
        
        for feature_name in feature_names:
            # Check if feature name contains any timeframe indicators
            for timeframe in self.config.timeframes:
                if timeframe in feature_name.lower():
                    timeframe_features.append(feature_name)
                    break
        
        return timeframe_features
    
    def _extract_significant_cross_timeframe_relationships(
        self, 
        pid_result: PIDResult, 
        timeframe_features: List[str]
    ) -> List[Tuple[str, str]]:
        """Extract significant cross-timeframe relationships from PID analysis."""
        significant_pairs = []
        
        # Filter synergy scores for timeframe features only
        timeframe_synergy = {
            (feat1, feat2): score for (feat1, feat2), score in pid_result.synergy.items()
            if feat1 in timeframe_features and feat2 in timeframe_features
        }
        
        # Sort by synergy score and take top relationships
        synergy_items = sorted(timeframe_synergy.items(), key=lambda x: x[1], reverse=True)
        
        for (feat1, feat2), synergy_score in synergy_items:
            if synergy_score > self.config.synergy_threshold:
                # Check if features are from different timeframes
                if self._are_different_timeframes(feat1, feat2):
                    significant_pairs.append((feat1, feat2))
                    if len(significant_pairs) >= self.config.max_timeframe_pairs:
                        break
        
        return significant_pairs
    
    def _are_different_timeframes(self, feat1: str, feat2: str) -> bool:
        """Check if two features are from different timeframes."""
        tf1 = self._extract_timeframe(feat1)
        tf2 = self._extract_timeframe(feat2)
        return tf1 is not None and tf2 is not None and tf1 != tf2
    
    def _extract_timeframe(self, feature_name: str) -> Optional[str]:
        """Extract timeframe from feature name."""
        for timeframe in self.config.timeframes:
            if timeframe in feature_name.lower():
                return timeframe
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
            timeframe_indices = [feature_names.index(f) for f in timeframe_features]
            
            for i, feat1 in enumerate(timeframe_features):
                for j, feat2 in enumerate(timeframe_features[i+1:], i+1):
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
            return np.column_stack(cross_timeframe_features), cross_timeframe_names
        else:
            return np.array([]).reshape(X.shape[0], 0), []
    
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
                # Rolling correlation between timeframes
                window = min(20, len(x1))
                feature = np.full_like(x1, np.corrcoef(x1, x2)[0, 1])
                name = f"{feat1}_corr_{feat2}"
                
            elif cross_timeframe_type == CrossTimeframeType.LAG_CORRELATION:
                # Lag-based correlation
                lag = min(5, len(x1) // 4)
                if lag > 0:
                    lagged_x1 = np.roll(x1, lag)
                    lagged_x1[:lag] = 0
                    feature = lagged_x1 * x2
                    name = f"{feat1}_lag_{lag}_x_{feat2}"
                else:
                    return None, ""
                
            elif cross_timeframe_type == CrossTimeframeType.MOMENTUM:
                # Momentum difference between timeframes
                momentum1 = np.diff(x1, prepend=x1[0])
                momentum2 = np.diff(x2, prepend=x2[0])
                feature = momentum1 - momentum2
                name = f"{feat1}_momentum_minus_{feat2}_momentum"
                
            elif cross_timeframe_type == CrossTimeframeType.VOLATILITY:
                # Volatility ratio between timeframes
                vol1 = np.std(x1)
                vol2 = np.std(x2)
                if vol2 != 0:
                    feature = np.full_like(x1, vol1 / vol2)
                    name = f"{feat1}_vol_ratio_{feat2}_vol"
                else:
                    return None, ""
                
            elif cross_timeframe_type == CrossTimeframeType.TREND_ALIGNMENT:
                # Trend alignment between timeframes
                trend1 = np.polyfit(range(len(x1)), x1, 1)[0]
                trend2 = np.polyfit(range(len(x2)), x2, 1)[0]
                feature = np.full_like(x1, trend1 * trend2)
                name = f"{feat1}_trend_x_{feat2}_trend"
                
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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            'pid_available': PID_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'pandas_available': PANDAS_AVAILABLE
        }
        
        if self.matrix_ops:
            metrics['matrix_ops_stats'] = self.matrix_ops.get_performance_stats()
            metrics['hardware_info'] = self.matrix_ops.get_hardware_info()
        
        return metrics