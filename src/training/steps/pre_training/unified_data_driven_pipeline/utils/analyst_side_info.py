"""
Analyst Side Information Handler

This module extracts and processes Analyst artifacts to create side information A
for conditional mutual information computation. Supports multiple extraction methods
with priority ordering and automatic degradation handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
    tprint_info("✅ Hardware optimizations available for Analyst side info")
except ImportError:
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning("⚠️ Hardware optimizations not available, using standard computations")

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
    tprint_info("✅ VectorBT optimizations available for Analyst side info")
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning("⚠️ VectorBT optimizations not available, using standard computations")

# Import ML utilities
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.utils.lookahead_protection import LookaheadValidator
    ML_UTILITIES_AVAILABLE = True
    tprint_info("✅ ML utilities available for Analyst side info")
except ImportError:
    ML_UTILITIES_AVAILABLE = False
    tprint_warning("⚠️ ML utilities not available, using standard implementations")

# Import common utilities
try:
    from src.utils.common_operations import safe_divide, safe_log
    from src.utils.common_utilities import validate_inputs, handle_missing_data
    from src.utils.math_validation import validate_numerical, check_finite
    COMMON_UTILITIES_AVAILABLE = True
    tprint_info("✅ Common utilities available for Analyst side info")
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False
    tprint_warning("⚠️ Common utilities not available, using standard implementations")

logger = logging.getLogger(__name__)

@dataclass
class AnalystSideInfoResult:
    """Result from Analyst side information extraction."""
    A: np.ndarray  # Side information array (n_samples, n_dims)
    source: str  # Source type: 'oof_confidence', 'multi_channel', 'binary_opportunity'
    n_dims: int  # Number of dimensions in A
    I_Y_A: float  # Strength of Analyst signal I(Y;A)
    degraded_to_unconditional: bool  # True if A was too weak
    extraction_metadata: Dict[str, Any]
    is_valid: bool = True

@dataclass
class AnalystSideInfoConfig:
    """Configuration for Analyst side information extraction."""
    # Priority order for extraction
    prefer_oof_confidence: bool = True
    prefer_multi_channel: bool = True
    fallback_to_binary: bool = True
    
    # Dimensionality reduction
    max_A_dims: int = 2  # Reduce A to ≤2 dims for CMI efficiency
    use_pca_reduction: bool = True
    use_rank_mean_fallback: bool = True
    
    # Binary label processing
    min_samples_per_bin: int = 100  # For isotonic calibration
    enable_isotonic_calibration: bool = True
    
    # Degenerate A handling
    weak_A_threshold: float = 0.005  # AUC-equivalent; degrade to unconditional MI
    enable_auto_degradation: bool = True
    
    # Missing data handling
    max_missing_ratio: float = 0.3  # Max ratio of missing values allowed
    enable_missing_data_alignment: bool = True

class AnalystSideInfoHandler:
    """
    Handler for extracting and processing Analyst side information.
    
    Priority order:
    1. OOF/confidence series (preferred)
    2. Multi-channel [conf, quality, dist] → PCA reduction
    3. Binary opportunity label → isotonic calibration
    4. Degenerate A → auto-degrade to unconditional MI
    """
    
    def __init__(self, config: Optional[AnalystSideInfoConfig] = None):
        """Initialize Analyst side information handler."""
        self.config = config or AnalystSideInfoConfig()
        self.logger = logger
        
        # Statistics tracking
        self._extraction_stats = {
            'total_extractions': 0,
            'oof_confidence_used': 0,
            'multi_channel_used': 0,
            'binary_opportunity_used': 0,
            'degraded_to_unconditional': 0,
            'pca_reductions': 0,
            'isotonic_calibrations': 0,
            'missing_data_aligned': 0
        }
        
        tprint_info("🎯 Analyst Side Information Handler initialized")
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize VectorBT optimizations
        self._init_vectorbt_optimizations()
        
        # Initialize ML utilities
        self._init_ml_utilities()
    
    def _init_hardware_optimizations(self):
        """Initialize hardware optimizations for M1 chip."""
        if HARDWARE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.gpu_optimizer = M1GPUOptimizer()
                self.memory_optimizer = M1MemoryOptimizer()
                self.cpu_optimizer = M1CPUOptimizer()
                tprint_success("✅ Hardware optimizations initialized for Analyst side info")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.gpu_optimizer = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_optimizer = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_vectorbt_optimizations(self):
        """Initialize VectorBT optimizations for efficient rolling computations."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ VectorBT optimizations initialized for Analyst side info")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT optimization initialization failed: {e}")
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
    
    def _init_ml_utilities(self):
        """Initialize ML utilities for cross-validation and data leakage detection."""
        if ML_UTILITIES_AVAILABLE:
            try:
                self.purged_kfold = PurgedKFold
                self.data_leakage_detector = DataLeakageDetector()
                self.lookahead_validator = LookaheadValidator()
                tprint_success("✅ ML utilities initialized for Analyst side info")
            except Exception as e:
                tprint_warning(f"⚠️ ML utility initialization failed: {e}")
                self.purged_kfold = None
                self.data_leakage_detector = None
                self.lookahead_validator = None
        else:
            self.purged_kfold = None
            self.data_leakage_detector = None
            self.lookahead_validator = None
    
    def extract_side_info(self, pipeline_state: Dict[str, Any], 
                         target_series: Optional[pd.Series] = None,
                         data_index: Optional[pd.Index] = None) -> AnalystSideInfoResult:
        """
        Extract Analyst side information from pipeline state.
        
        Args:
            pipeline_state: Pipeline state containing Analyst artifacts
            target_series: Target series for computing I(Y;A)
            data_index: Index to align side information with data
            
        Returns:
            AnalystSideInfoResult with processed side information
        """
        try:
            self._extraction_stats['total_extractions'] += 1
            
            # Try extraction methods in priority order
            if self.config.prefer_oof_confidence:
                result = self._extract_oof_confidence(pipeline_state, target_series, data_index)
                if result.is_valid:
                    self._extraction_stats['oof_confidence_used'] += 1
                    return result
            
            if self.config.prefer_multi_channel:
                result = self._extract_multi_channel(pipeline_state, target_series, data_index)
                if result.is_valid:
                    self._extraction_stats['multi_channel_used'] += 1
                    return result
            
            if self.config.fallback_to_binary:
                result = self._extract_binary_opportunity(pipeline_state, target_series, data_index)
                if result.is_valid:
                    self._extraction_stats['binary_opportunity_used'] += 1
                    return result
            
            # If all methods failed, return invalid result
            tprint_warning("⚠️ All Analyst side information extraction methods failed")
            return AnalystSideInfoResult(
                A=np.array([]),
                source='none',
                n_dims=0,
                I_Y_A=0.0,
                degraded_to_unconditional=True,
                extraction_metadata={'error': 'No valid Analyst artifacts found'},
                is_valid=False
            )
            
        except Exception as e:
            tprint_error(f"❌ Analyst side information extraction failed: {e}")
            return AnalystSideInfoResult(
                A=np.array([]),
                source='error',
                n_dims=0,
                I_Y_A=0.0,
                degraded_to_unconditional=True,
                extraction_metadata={'error': str(e)},
                is_valid=False
            )
    
    def _extract_oof_confidence(self, pipeline_state: Dict[str, Any],
                               target_series: Optional[pd.Series],
                               data_index: Optional[pd.Index]) -> AnalystSideInfoResult:
        """Extract OOF confidence series from Analyst artifacts."""
        try:
            # Look for analyst_profit_labeler_artifacts
            analyst_artifacts = pipeline_state.get('analyst_profit_labeler_artifacts')
            if analyst_artifacts is None:
                return self._create_invalid_result("No analyst artifacts found")
            
            # Extract confidence scores
            confidence_series = None
            
            if isinstance(analyst_artifacts, dict):
                mhlr = analyst_artifacts.get('multi_horizon_labeling_result')
                if mhlr and isinstance(mhlr, dict):
                    conf_scores = mhlr.get('confidence_scores')
                    if isinstance(conf_scores, pd.DataFrame):
                        # Prefer 'opportunity' column, fallback to max across numeric columns
                        if 'opportunity' in conf_scores.columns:
                            confidence_series = conf_scores['opportunity']
                        else:
                            num_cols = conf_scores.select_dtypes(include=[np.number])
                            if len(num_cols.columns) > 0:
                                confidence_series = num_cols.max(axis=1)
                    elif isinstance(conf_scores, pd.Series):
                        confidence_series = conf_scores
            else:
                # Try to get confidence from object attributes
                mhlr = getattr(analyst_artifacts, 'multi_horizon_labeling_result', None)
                if mhlr:
                    conf_scores = getattr(mhlr, 'confidence_scores', None)
                    if isinstance(conf_scores, pd.DataFrame):
                        if 'opportunity' in conf_scores.columns:
                            confidence_series = conf_scores['opportunity']
                        else:
                            num_cols = conf_scores.select_dtypes(include=[np.number])
                            if len(num_cols.columns) > 0:
                                confidence_series = num_cols.max(axis=1)
                    elif isinstance(conf_scores, pd.Series):
                        confidence_series = conf_scores
            
            if confidence_series is None:
                return self._create_invalid_result("No confidence series found in Analyst artifacts")
            
            # Align with data index if provided
            if data_index is not None:
                confidence_series = confidence_series.reindex(data_index).fillna(0.0)
            
            # Convert to numpy array and ensure 2D
            A = confidence_series.values.reshape(-1, 1)
            
            # Check for degenerate A
            I_Y_A = self._compute_I_Y_A(A, target_series)
            degraded = self._check_degenerate_A(I_Y_A)
            
            if degraded:
                self._extraction_stats['degraded_to_unconditional'] += 1
                tprint_warning("⚠️ Analyst confidence signal too weak, degrading to unconditional MI")
            
            return AnalystSideInfoResult(
                A=A,
                source='oof_confidence',
                n_dims=1,
                I_Y_A=I_Y_A,
                degraded_to_unconditional=degraded,
                extraction_metadata={
                    'method': 'oof_confidence',
                    'original_length': len(confidence_series),
                    'aligned_length': len(A),
                    'missing_ratio': confidence_series.isnull().sum() / len(confidence_series)
                },
                is_valid=True
            )
            
        except Exception as e:
            return self._create_invalid_result(f"OOF confidence extraction failed: {e}")
    
    def _extract_multi_channel(self, pipeline_state: Dict[str, Any],
                              target_series: Optional[pd.Series],
                              data_index: Optional[pd.Index]) -> AnalystSideInfoResult:
        """Extract multi-channel Analyst information [conf, quality, dist]."""
        try:
            analyst_artifacts = pipeline_state.get('analyst_profit_labeler_artifacts')
            if analyst_artifacts is None:
                return self._create_invalid_result("No analyst artifacts found")
            
            # Collect multiple channels
            channels = []
            channel_names = []
            
            # Extract confidence
            conf_series = self._extract_confidence_channel(analyst_artifacts, data_index)
            if conf_series is not None:
                channels.append(conf_series)
                channel_names.append('confidence')
            
            # Extract quality (if available)
            quality_series = self._extract_quality_channel(analyst_artifacts, data_index)
            if quality_series is not None:
                channels.append(quality_series)
                channel_names.append('quality')
            
            # Extract distance to anchor (if available)
            dist_series = self._extract_distance_channel(analyst_artifacts, data_index)
            if dist_series is not None:
                channels.append(dist_series)
                channel_names.append('distance')
            
            if len(channels) < 2:
                return self._create_invalid_result("Insufficient multi-channel data")
            
            # Stack channels
            A_multi = np.column_stack(channels)
            
            # Reduce dimensionality if needed
            if A_multi.shape[1] > self.config.max_A_dims:
                A_reduced = self._reduce_dimensionality(A_multi, target_series)
                self._extraction_stats['pca_reductions'] += 1
            else:
                A_reduced = A_multi
            
            # Check for degenerate A
            I_Y_A = self._compute_I_Y_A(A_reduced, target_series)
            degraded = self._check_degenerate_A(I_Y_A)
            
            if degraded:
                self._extraction_stats['degraded_to_unconditional'] += 1
                tprint_warning("⚠️ Multi-channel Analyst signal too weak, degrading to unconditional MI")
            
            return AnalystSideInfoResult(
                A=A_reduced,
                source='multi_channel',
                n_dims=A_reduced.shape[1],
                I_Y_A=I_Y_A,
                degraded_to_unconditional=degraded,
                extraction_metadata={
                    'method': 'multi_channel',
                    'channels': channel_names,
                    'original_dims': A_multi.shape[1],
                    'reduced_dims': A_reduced.shape[1],
                    'n_samples': len(A_reduced)
                },
                is_valid=True
            )
            
        except Exception as e:
            return self._create_invalid_result(f"Multi-channel extraction failed: {e}")
    
    def _extract_binary_opportunity(self, pipeline_state: Dict[str, Any],
                                   target_series: Optional[pd.Series],
                                   data_index: Optional[pd.Index]) -> AnalystSideInfoResult:
        """Extract binary opportunity labels from Analyst artifacts."""
        try:
            analyst_artifacts = pipeline_state.get('analyst_profit_labeler_artifacts')
            if analyst_artifacts is None:
                return self._create_invalid_result("No analyst artifacts found")
            
            # Extract binary opportunity labels
            opportunity_series = None
            
            if isinstance(analyst_artifacts, dict):
                mhlr = analyst_artifacts.get('multi_horizon_labeling_result')
                if mhlr and isinstance(mhlr, dict):
                    labels = mhlr.get('labels')
                    if isinstance(labels, pd.DataFrame):
                        if 'opportunity' in labels.columns:
                            opportunity_series = labels['opportunity']
                        else:
                            # Use first binary column
                            binary_cols = labels.select_dtypes(include=[np.number])
                            if len(binary_cols.columns) > 0:
                                opportunity_series = binary_cols.iloc[:, 0]
                    elif isinstance(labels, pd.Series):
                        opportunity_series = labels
            else:
                mhlr = getattr(analyst_artifacts, 'multi_horizon_labeling_result', None)
                if mhlr:
                    labels = getattr(mhlr, 'labels', None)
                    if isinstance(labels, pd.DataFrame):
                        if 'opportunity' in labels.columns:
                            opportunity_series = labels['opportunity']
                        else:
                            binary_cols = labels.select_dtypes(include=[np.number])
                            if len(binary_cols.columns) > 0:
                                opportunity_series = binary_cols.iloc[:, 0]
                    elif isinstance(labels, pd.Series):
                        opportunity_series = labels
            
            if opportunity_series is None:
                return self._create_invalid_result("No opportunity labels found")
            
            # Align with data index if provided
            if data_index is not None:
                opportunity_series = opportunity_series.reindex(data_index).fillna(0)
            
            # Apply isotonic calibration if conditions are met
            if (self.config.enable_isotonic_calibration and 
                len(opportunity_series) >= self.config.min_samples_per_bin):
                try:
                    # Check if we have enough samples per bin
                    unique_values = opportunity_series.unique()
                    min_samples_per_bin = min(opportunity_series.value_counts())
                    
                    if min_samples_per_bin >= self.config.min_samples_per_bin:
                        # Apply isotonic calibration
                        iso_reg = IsotonicRegression(out_of_bounds='clip')
                        A_calibrated = iso_reg.fit_transform(
                            opportunity_series.values, 
                            target_series.values if target_series is not None else opportunity_series.values
                        )
                        self._extraction_stats['isotonic_calibrations'] += 1
                        tprint_info("✅ Applied isotonic calibration to binary opportunity labels")
                    else:
                        A_calibrated = opportunity_series.values.astype(float)
                        tprint_info("ℹ️ Skipped isotonic calibration: insufficient samples per bin")
                except Exception as e:
                    tprint_warning(f"⚠️ Isotonic calibration failed: {e}, using raw values")
                    A_calibrated = opportunity_series.values.astype(float)
            else:
                A_calibrated = opportunity_series.values.astype(float)
            
            # Ensure 2D array
            A = A_calibrated.reshape(-1, 1)
            
            # Check for degenerate A
            I_Y_A = self._compute_I_Y_A(A, target_series)
            degraded = self._check_degenerate_A(I_Y_A)
            
            if degraded:
                self._extraction_stats['degraded_to_unconditional'] += 1
                tprint_warning("⚠️ Binary opportunity signal too weak, degrading to unconditional MI")
            
            return AnalystSideInfoResult(
                A=A,
                source='binary_opportunity',
                n_dims=1,
                I_Y_A=I_Y_A,
                degraded_to_unconditional=degraded,
                extraction_metadata={
                    'method': 'binary_opportunity',
                    'isotonic_calibrated': self.config.enable_isotonic_calibration,
                    'original_length': len(opportunity_series),
                    'aligned_length': len(A),
                    'unique_values': len(np.unique(A_calibrated))
                },
                is_valid=True
            )
            
        except Exception as e:
            return self._create_invalid_result(f"Binary opportunity extraction failed: {e}")
    
    def _extract_confidence_channel(self, analyst_artifacts: Any, data_index: Optional[pd.Index]) -> Optional[pd.Series]:
        """Extract confidence channel from Analyst artifacts."""
        try:
            if isinstance(analyst_artifacts, dict):
                mhlr = analyst_artifacts.get('multi_horizon_labeling_result')
                if mhlr and isinstance(mhlr, dict):
                    conf_scores = mhlr.get('confidence_scores')
                    if isinstance(conf_scores, pd.DataFrame):
                        if 'opportunity' in conf_scores.columns:
                            return conf_scores['opportunity']
                        else:
                            num_cols = conf_scores.select_dtypes(include=[np.number])
                            if len(num_cols.columns) > 0:
                                return num_cols.max(axis=1)
            return None
        except Exception:
            return None
    
    def _extract_quality_channel(self, analyst_artifacts: Any, data_index: Optional[pd.Index]) -> Optional[pd.Series]:
        """Extract quality channel from Analyst artifacts."""
        try:
            # Look for quality scores in metadata or additional fields
            if isinstance(analyst_artifacts, dict):
                mhlr = analyst_artifacts.get('multi_horizon_labeling_result')
                if mhlr and isinstance(mhlr, dict):
                    # Check for quality scores
                    quality_scores = mhlr.get('quality_scores')
                    if isinstance(quality_scores, pd.DataFrame):
                        if 'opportunity' in quality_scores.columns:
                            return quality_scores['opportunity']
                        else:
                            num_cols = quality_scores.select_dtypes(include=[np.number])
                            if len(num_cols.columns) > 0:
                                return num_cols.max(axis=1)
            return None
        except Exception:
            return None
    
    def _extract_distance_channel(self, analyst_artifacts: Any, data_index: Optional[pd.Index]) -> Optional[pd.Series]:
        """Extract distance to anchor channel from Analyst artifacts."""
        try:
            # Look for distance information in metadata
            if isinstance(analyst_artifacts, dict):
                mhlr = analyst_artifacts.get('multi_horizon_labeling_result')
                if mhlr and isinstance(mhlr, dict):
                    # Check for distance information
                    metadata = mhlr.get('metadata', {})
                    if 'distance_to_anchor' in metadata:
                        return pd.Series(metadata['distance_to_anchor'])
            return None
        except Exception:
            return None
    
    def _reduce_dimensionality(self, A: np.ndarray, target_series: Optional[pd.Series]) -> np.ndarray:
        """Reduce A dimensionality using PCA or rank-mean."""
        try:
            if A.shape[1] <= self.config.max_A_dims:
                return A
            
            if self.config.use_pca_reduction:
                # Use PCA to reduce to max_A_dims
                pca = PCA(n_components=self.config.max_A_dims)
                A_reduced = pca.fit_transform(A)
                tprint_info(f"✅ Reduced A from {A.shape[1]} to {A_reduced.shape[1]} dims using PCA")
                return A_reduced
            elif self.config.use_rank_mean_fallback:
                # Use rank-normalized mean as fallback
                A_ranked = np.zeros((A.shape[0], self.config.max_A_dims))
                for i in range(self.config.max_A_dims):
                    if i < A.shape[1]:
                        A_ranked[:, i] = A[:, i]
                    else:
                        # Use mean of remaining dimensions
                        A_ranked[:, i] = np.mean(A[:, i:], axis=1)
                tprint_info(f"✅ Reduced A from {A.shape[1]} to {A_ranked.shape[1]} dims using rank-mean")
                return A_ranked
            else:
                # Simple truncation
                return A[:, :self.config.max_A_dims]
                
        except Exception as e:
            tprint_warning(f"⚠️ Dimensionality reduction failed: {e}, using truncation")
            return A[:, :self.config.max_A_dims]
    
    def _compute_I_Y_A(self, A: np.ndarray, target_series: Optional[pd.Series]) -> float:
        """Compute I(Y;A) to assess Analyst signal strength."""
        try:
            if target_series is None or len(A) == 0:
                return 0.0
            
            # Align A and target_series
            if len(A) != len(target_series):
                return 0.0
            
            # Simple correlation-based approximation of I(Y;A)
            if A.shape[1] == 1:
                # Single dimension
                correlation = np.corrcoef(A.flatten(), target_series.values)[0, 1]
                if np.isnan(correlation):
                    return 0.0
                # Convert correlation to approximate MI
                return 0.5 * np.log(1 / (1 - correlation**2 + 1e-10))
            else:
                # Multiple dimensions - use maximum correlation
                correlations = []
                for i in range(A.shape[1]):
                    corr = np.corrcoef(A[:, i], target_series.values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                if not correlations:
                    return 0.0
                
                max_corr = max(correlations)
                return 0.5 * np.log(1 / (1 - max_corr**2 + 1e-10))
                
        except Exception as e:
            tprint_warning(f"⚠️ I(Y;A) computation failed: {e}")
            return 0.0
    
    def _check_degenerate_A(self, I_Y_A: float) -> bool:
        """Check if A is too weak (degenerate)."""
        if not self.config.enable_auto_degradation:
            return False
        
        return I_Y_A < self.config.weak_A_threshold
    
    def _create_invalid_result(self, error_message: str) -> AnalystSideInfoResult:
        """Create an invalid result with error message."""
        return AnalystSideInfoResult(
            A=np.array([]),
            source='error',
            n_dims=0,
            I_Y_A=0.0,
            degraded_to_unconditional=True,
            extraction_metadata={'error': error_message},
            is_valid=False
        )
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self._extraction_stats.copy()

def create_analyst_side_info_handler(config: Optional[AnalystSideInfoConfig] = None) -> AnalystSideInfoHandler:
    """Create an Analyst side information handler with default configuration."""
    return AnalystSideInfoHandler(config)
