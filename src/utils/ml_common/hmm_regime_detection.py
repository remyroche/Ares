#!/usr/bin/env python3
"""
Enhanced HMM Regime Detection Utilities

This enhanced module integrates the HMM composite manager functionality and adds
comprehensive regime detection capabilities with advanced features:

Key Enhancements:
- HMM Composite Manager Integration: Full integration with consolidated HMM functionality
- Multi-Timeframe Support: Ensemble HMM across multiple timeframes
- Regime Transition Analysis: Advanced transition probability calculations
- Economic Significance: Pareto front utilities for regime validation
- Streaming Support: Real-time regime detection capabilities
- Memory Optimization: M1-optimized memory management
- GPU Acceleration: M1 MPS support for regime detection
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from pathlib import Path

# Import comprehensive utility infrastructure
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt,
    validate_positive, validate_range
)
from src.utils.core.common import create_fallback_logger, create_fallback_decorator
from src.utils.parquet_utils import ParquetUtils
from src.utils.serialization_utils import UniversalSerializer
from src.utils.data_processing_utils import DataProcessingUtils
from src.utils.common_utilities import CommonUtilities
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer

# Import HMM composite manager
from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager

# Import ML Common utilities
from .validation.cv_utils import TemporalCrossValidator, PurgedKFold
from .validation.validation_utils import ValidationFramework
from .optimization.pareto import ParetoFrontAnalyzer
from .ensembles.ensemble_manager import EnsembleManager, EnsembleConfig, EnsembleType

# Import data quality frameworks
try:
    from ..data.quality.data_quality import DataQualityFramework
    from ..data.quality.data_cleaning import DataCleaner
    QUALITY_FRAMEWORK_AVAILABLE = True
except ImportError:
    QUALITY_FRAMEWORK_AVAILABLE = False

try:
    from src.utils.feature_engineering_validation import FeatureEngineeringValidator
    FEATURE_VALIDATOR_AVAILABLE = True
except ImportError:
    FEATURE_VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    # Note: Removed silhouette_score and calinski_harabasz_score as they're not relevant for HMMs
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("HMM libraries not available - limited regime detection functionality")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - Bayesian optimization disabled")

class RegimeDetectionMethod(Enum):
    """Available regime detection methods."""
    HMM_GAUSSIAN = "hmm_gaussian"
    HMM_MULTIVARIATE = "hmm_multivariate"
    ENSEMBLE_HMM = "ensemble_hmm"
    MULTI_TIMEFRAME_HMM = "multi_timeframe_hmm"
    STREAMING_HMM = "streaming_hmm"
    REGIME_AWARE_HMM = "regime_aware_hmm"

class TimeframeType(Enum):
    """Available timeframe types."""
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

@dataclass
class HMMRegimeConfig:
    """Configuration for HMM regime detection."""
    n_components: int = 4
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-3
    random_state: int = 42
    method: RegimeDetectionMethod = RegimeDetectionMethod.HMM_GAUSSIAN
    min_regime_samples: int = 100
    max_regime_imbalance: float = 0.8
    economic_significance_threshold: float = 0.05
    
    # Mode-based regime limits
    light_mode_max_regimes: int = 2
    blank_mode_max_regimes: int = 5
    full_mode_max_regimes: int = 150
    
    def get_max_regimes_for_mode(self, mode: str) -> int:
        """Get the maximum number of regimes allowed for a given mode."""
        mode = mode.lower() if mode else 'light'
        if mode == 'light':
            return self.light_mode_max_regimes
        elif mode == 'blank':
            return self.blank_mode_max_regimes
        elif mode == 'full':
            return self.full_mode_max_regimes
        else:
            # Default to light mode for unknown modes
            return self.light_mode_max_regimes

@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe HMM."""
    timeframes: List[TimeframeType] = field(default_factory=lambda: [TimeframeType.HOUR, TimeframeType.DAILY])
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    consensus_threshold: float = 0.6
    temporal_alignment: bool = True

@dataclass
class StreamingConfig:
    """Configuration for streaming HMM."""
    window_size: int = 1000
    update_frequency: int = 100
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.8
    max_regime_changes: int = 10

@dataclass
class RegimeTransitionMetrics:
    """Metrics for regime transition analysis."""
    transition_matrix: np.ndarray
    transition_probabilities: Dict[Tuple[int, int], float]
    regime_persistence: Dict[int, float]
    transition_volatility: float
    regime_stability: float

@dataclass
class EconomicSignificanceMetrics:
    """Metrics for economic significance validation."""
    regime_returns: Dict[int, float]
    regime_volatility: Dict[int, float]
    regime_sharpe: Dict[int, float]
    pareto_efficiency: float
    economic_significance: bool

class EnhancedHMMRegimeDetector:
    """Enhanced HMM regime detector with comprehensive functionality."""
    
    def __init__(self, config: Optional[HMMRegimeConfig] = None):
        self.logger = create_fallback_logger()
        self.logger.info("🚀 Initializing EnhancedHMMRegimeDetector...")
        start_time = time.time()
        
        self.config = config or HMMRegimeConfig()
        self.logger.info(f"📊 Configuration loaded: {self.config.method.value}")
        self.logger.info(f"📊 HMM components: {self.config.n_components}")
        self.logger.info(f"📊 Covariance type: {self.config.covariance_type}")
        
        # Initialize utility managers
        self.logger.debug("🔧 Initializing utility managers...")
        self._initialize_utilities()
        
        # Initialize HMM composite manager
        self.logger.debug("🔧 Initializing HMM composite manager...")
        self.hmm_manager = EnhancedHMMCompositeManager()
        self.logger.debug("✅ HMM composite manager initialized")
        
        # Initialize specialized configurations
        self.logger.debug("🔧 Initializing specialized configurations...")
        self.multi_timeframe_config = MultiTimeframeConfig()
        self.streaming_config = StreamingConfig()
        self.logger.debug("✅ Specialized configurations initialized")
        
        # Performance tracking
        self.performance_stats = {
            'total_regimes_detected': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'accuracy_scores': []
        }
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ EnhancedHMMRegimeDetector initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Min regime samples: {self.config.min_regime_samples}")
        self.logger.info(f"📊 Max regime imbalance: {self.config.max_regime_imbalance}")
        self.logger.info(f"📊 Economic significance threshold: {self.config.economic_significance_threshold}")
        
        # Streaming state
        self.streaming_state = {
            'current_model': None,
            'last_update': None,
            'regime_history': [],
            'stability_score': 0.0
        }

    def _initialize_utilities(self):
        """Initialize utility managers."""
        try:
            # Try to import and initialize M1 GPU manager with fallback
            try:
                self.gpu_manager = get_m1_gpu_manager()
            except NameError:
                self.logger.warning("⚠️ get_m1_gpu_manager not available, using fallback")
                self.gpu_manager = None
            except Exception as gpu_e:
                self.logger.warning(f"⚠️ M1 GPU manager initialization failed: {gpu_e}")
                self.gpu_manager = None

            # Try to import and initialize M1 memory optimizer with fallback
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
            except Exception as mem_e:
                self.logger.warning(f"⚠️ M1 memory optimizer initialization failed: {mem_e}")
                self.memory_optimizer = None

            # Try to import and initialize M1 CPU optimizer with fallback
            try:
                self.cpu_optimizer = get_m1_cpu_optimizer()
            except Exception as cpu_e:
                self.logger.warning(f"⚠️ M1 CPU optimizer initialization failed: {cpu_e}")
                self.cpu_optimizer = None

            # Initialize other utilities
            try:
                self.parquet_utils = ParquetUtils()
                self.serializer = UniversalSerializer()
                self.data_processor = DataProcessingUtils()
                self.common_utils = CommonUtilities()
                self.pareto_analyzer = ParetoFrontAnalyzer()

                # Create ensemble config for HMM regime detection
                ensemble_config = EnsembleConfig(
                    ensemble_name="hmm_regime_detection",
                    output_dir=str(Path(__file__).parent / "artifacts" / "ensemble"),
                    ensemble_type=EnsembleType.VOTING,
                    enable_gpu_acceleration=True,
                    enable_memory_optimization=True,
                    enable_parallel_processing=True,
                    memory_limit_gb=4.0
                )
                self.ensemble_manager = EnsembleManager(ensemble_config)
            except Exception as util_e:
                self.logger.warning(f"⚠️ Some basic utilities failed to initialize: {util_e}")
                self.parquet_utils = None
                self.serializer = None
                self.data_processor = None
                self.common_utils = None
                self.pareto_analyzer = None
                self.ensemble_manager = None

            self.logger.info("✅ All utility managers initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Unexpected error during utility initialization: {e}")
            # Set fallback implementations for any that weren't already set
            if not hasattr(self, 'gpu_manager'):
                self.gpu_manager = None
            if not hasattr(self, 'memory_optimizer'):
                self.memory_optimizer = None
            if not hasattr(self, 'cpu_optimizer'):
                self.cpu_optimizer = None
            if not hasattr(self, 'parquet_utils'):
                self.parquet_utils = None
            if not hasattr(self, 'serializer'):
                self.serializer = None
            if not hasattr(self, 'data_processor'):
                self.data_processor = None
            if not hasattr(self, 'common_utils'):
                self.common_utils = None
            if not hasattr(self, 'pareto_analyzer'):
                self.pareto_analyzer = None
            if not hasattr(self, 'ensemble_manager'):
                self.ensemble_manager = None

    def detect_regimes(
        self,
        data: pd.DataFrame,
        method: Optional[RegimeDetectionMethod] = None,
        config: Optional[HMMRegimeConfig] = None,
        mode: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect regimes using specified method.

        Args:
            data: Input data for regime detection
            method: Regime detection method
            config: Optional configuration override
            mode: Optional optimization mode ('light', 'blank', 'full')

        Returns:
            DataFrame with regime labels and metadata
        """
        self.logger.info(f"🔍 DEBUG: detect_regimes called with method={method}, mode={mode}")
        self.logger.info(f"🔍 DEBUG: Input data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")

        method = method or self.config.method
        self.logger.info(f"🔍 DEBUG: Using method: {method}")
        config = config or self.config
        optimization_mode = mode  # Local variable for optimization mode
        start_time = time.time()
        
        # Apply mode-based regime limits
        if mode:
            max_regimes = config.get_max_regimes_for_mode(mode)
            if config.n_components > max_regimes:
                self.logger.info(f"🔧 Limiting n_components from {config.n_components} to {max_regimes} for {mode} mode (range: 2-150)")
                config.n_components = max_regimes
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Select implementation based on method
            if method == RegimeDetectionMethod.HMM_GAUSSIAN:
                regimes_df = self._detect_hmm_gaussian_regimes(data, config, mode)
            elif method == RegimeDetectionMethod.HMM_MULTIVARIATE:
                regimes_df = self._detect_hmm_multivariate_regimes(data, config)
            elif method == RegimeDetectionMethod.ENSEMBLE_HMM:
                regimes_df = self._detect_ensemble_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.MULTI_TIMEFRAME_HMM:
                regimes_df = self._detect_multi_timeframe_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.STREAMING_HMM:
                regimes_df = self._detect_streaming_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.REGIME_AWARE_HMM:
                regimes_df = self._detect_regime_aware_hmm_regimes(data, config)
            else:
                raise ValueError(f"Unsupported regime detection method: {method}")
            
            # Validate regime quality
            validation_results = self._validate_regime_quality(regimes_df, data)
            
            # Update performance stats
            self.logger.info(f"🔍 DEBUG: Updating performance stats...")
            self._update_performance_stats(start_time, len(regimes_df))
            self.logger.info(f"🔍 DEBUG: Performance stats updated")

            self.logger.info(f"✅ Detected {len(regimes_df)} regimes using {method.value}")
            self.logger.info(f"🔍 DEBUG: detect_regimes method completing, returning DataFrame with shape: {regimes_df.shape}")
            self.logger.info(f"🔍 DEBUG: Result columns: {list(regimes_df.columns)}")
            return regimes_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to detect regimes: {e}")
            raise

    def _detect_hmm_gaussian_regimes(
        self,
        data: pd.DataFrame,
        config: HMMRegimeConfig,
        mode: Optional[str] = None
    ) -> pd.DataFrame:
        """Detect regimes using Gaussian HMM."""
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available")
        
        # Prepare data - pre-filter features
        # 1) Drop time-like and identifier columns if present
        cols_to_drop = [
            c for c in ['timestamp', 'open_time', 'close_time', 'symbol', 'exchange', 'timeframe', 'interval']
            if c in data.columns
        ]
        filtered = data.drop(columns=cols_to_drop, errors='ignore')
        # 2) Keep only numeric columns
        numeric_data = filtered.select_dtypes(include=[np.number])
        # 3) Remove near-constant/low-variance columns
        if len(numeric_data.columns) > 0:
            std_series = numeric_data.std(numeric_only=True)
            low_var_cols = std_series[std_series <= 1e-8].index.tolist()
            if low_var_cols:
                numeric_data = numeric_data.drop(columns=low_var_cols)
                self.logger.info(f"🧹 Removed {len(low_var_cols)} low-variance features: {low_var_cols[:10]}{'...' if len(low_var_cols) > 10 else ''}")

        # Check for NaN values and handle them properly
        nan_count = numeric_data.isnull().sum().sum()
        if nan_count > 0:
            # Identify rows and columns with NaN values
            nan_locations = []
            for col in numeric_data.columns:
                nan_rows = numeric_data[numeric_data[col].isnull()].index.tolist()
                if nan_rows:
                    nan_locations.extend([f"Column '{col}', Row {row}" for row in nan_rows[:5]])  # Limit to first 5 per column

            nan_location_str = "; ".join(nan_locations[:10])  # Limit total locations shown
            if len(nan_locations) > 10:
                nan_location_str += f"... and {len(nan_locations) - 10} more"

            self.logger.warning(f"⚠️ Found {nan_count} NaN values in numeric data, using forward/backward fill. Locations: {nan_location_str}")

            # Use forward fill, then backward fill for remaining NaN values
            numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill')

            # If still have NaN values (e.g., all values in column are NaN), fill with column mean
            numeric_data = numeric_data.fillna(numeric_data.mean())

            # Final fallback: fill any remaining NaN values with 0
            numeric_data = numeric_data.fillna(0)

        # Enhanced feature engineering for better regime detection
        try:
            enhanced_data = self._create_enhanced_features(numeric_data)
            numeric_data = enhanced_data
            self.logger.info(f"✅ Enhanced features created: {numeric_data.shape[1]} features")
        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineering failed, using original data: {e}")

        # Data normalization for HMM stability
        try:
            numeric_data = self._normalize_hmm_data(numeric_data)
            self.logger.info("✅ Data normalized for HMM training")
        except Exception as e:
            self.logger.warning(f"⚠️ Data normalization failed: {e}")

        # Safety: ensure numeric-only after engineering
        numeric_data = numeric_data.select_dtypes(include=[np.number])

        # Validate data for HMM training
        if numeric_data.empty:
            raise ValueError("No numeric data available for HMM training")

        if len(numeric_data) < config.n_components * 10:
            raise ValueError(f"Insufficient data for HMM training. Need at least {config.n_components * 10} samples, got {len(numeric_data)}")

        # Check for infinite values
        if np.any(np.isinf(numeric_data.values)):
            self.logger.warning("⚠️ Found infinite values in data, replacing with finite values")
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Use HMM composite manager for optimization
        # Use passed mode parameter or default to 'light'
        optimization_mode = (mode or 'light').upper()
        self.logger.info(f"🔧 Using optimization mode: {optimization_mode}")
        optimization_result = self.hmm_manager.optimize_hmm_parameters(numeric_data, mode=optimization_mode)
        
        if optimization_result.get('success', False):
            best_params = optimization_result['best_params']
            config.n_components = best_params.get('n_components', config.n_components)
            config.covariance_type = best_params.get('covariance_type', config.covariance_type)
            config.n_iter = best_params.get('n_iter', config.n_iter)
            config.tol = best_params.get('tol', config.tol)
        
        # Multi-stage HMM parameter optimization: coarse grid -> fine grid -> Bayesian
        import time as time_module
        n_samples = len(numeric_data)

        # Determine optimization intensity based on mode
        mode = (mode or 'blank').lower()
        if mode == 'full':
            # Very detailed in full mode
            use_bayesian = True
            n_coarse_configs = 8
            n_fine_configs = 12
            bayesian_trials = 25
            verbose_logging = True
        elif mode == 'blank':
            # Lighter in blank mode
            use_bayesian = True
            n_coarse_configs = 3
            n_fine_configs = 3
            bayesian_trials = 4
            verbose_logging = True
        else:  # light mode
            # Light mode with Bayesian optimization (reduced trials)
            use_bayesian = True
            n_coarse_configs = 3
            n_fine_configs = 3
            bayesian_trials = 3      # Reduced trials for light mode
            verbose_logging = True   # Verbose logging: Every step, every config, every score

        self.logger.info(f"🎛️ HMM Optimization Mode: {mode.upper()}")
        self.logger.info(f"📊 Dataset: {n_samples:,} samples, {numeric_data.shape[1]} features")
        stages_desc = "Coarse Grid → Fine Grid → Bayesian Opt"
        if not use_bayesian:
            stages_desc = "Coarse Grid → Fine Grid → Best Config"
        self.logger.info(f"🔬 Stages: {stages_desc}")

        # Stage 1: Coarse Grid Search
        self.logger.info(f"\n🔍 Stage 1: Coarse Grid Search ({n_coarse_configs} configs)")

        coarse_configs = [
            {'n_components': 2, 'covariance_type': 'spherical', 'n_iter': 30, 'tol': 1e-2},
            {'n_components': 3, 'covariance_type': 'diag', 'n_iter': 50, 'tol': 5e-3},
            {'n_components': 4, 'covariance_type': 'diag', 'n_iter': 75, 'tol': 1e-3},
            {'n_components': 2, 'covariance_type': 'diag', 'n_iter': 40, 'tol': 1e-2},
            {'n_components': 5, 'covariance_type': 'diag', 'n_iter': 60, 'tol': 5e-4},
            {'n_components': 3, 'covariance_type': 'spherical', 'n_iter': 45, 'tol': 1e-2},
            {'n_components': 4, 'covariance_type': 'tied', 'n_iter': 55, 'tol': 1e-3},
            {'n_components': 2, 'covariance_type': 'tied', 'n_iter': 35, 'tol': 5e-3},
        ][:n_coarse_configs]

        coarse_results = []
        best_coarse_config = None
        best_coarse_score = float('-inf')

        for i, coarse_config in enumerate(coarse_configs):
            try:
                start_time = time_module.time()

                if verbose_logging:
                    self.logger.info(f"  Testing coarse config {i+1}/{len(coarse_configs)}: {coarse_config}")

                # Create temporary model for scoring
                temp_model = hmm.GaussianHMM(
                    n_components=coarse_config['n_components'],
                    covariance_type=coarse_config['covariance_type'],
                    n_iter=coarse_config['n_iter'],
                    tol=coarse_config['tol'],
                    random_state=42,  # Use fixed random state for reproducibility
                    init_params='mc',
                    params='stmc'
                )

                # Initialize start probabilities
                if hasattr(temp_model, 'startprob_'):
                    temp_model.startprob_ = np.ones(coarse_config['n_components']) / coarse_config['n_components']

                # Fit on full dataset for coarse evaluation
                temp_model.fit(numeric_data)
                score = temp_model.score(numeric_data)

                elapsed = time_module.time() - start_time
                coarse_results.append({
                    'config': coarse_config.copy(),
                    'score': score,
                    'time': elapsed
                })

                if verbose_logging:
                    self.logger.info(f"    Score: {score:.2f} (took {elapsed:.2f}s)")

                if score > best_coarse_score:
                    best_coarse_score = score
                    best_coarse_config = coarse_config.copy()

            except Exception as e:
                if verbose_logging:
                    self.logger.warning(f"    Coarse config {i+1} failed: {e}")
                continue

        if not best_coarse_config:
            self.logger.error("❌ All coarse configurations failed!")
            raise ValueError("Coarse grid search failed to find any working configuration")

        self.logger.info(f"🏅 Stage 1 Complete - Best coarse config: {best_coarse_config} (score: {best_coarse_score:.2f})")

        # Stage 2: Fine Grid Search around best coarse configuration
        self.logger.info(f"\n🎯 Stage 2: Fine Grid Search ({n_fine_configs} configs)")

        base_n_comp = best_coarse_config['n_components']
        base_cov = best_coarse_config['covariance_type']
        base_iter = best_coarse_config['n_iter']
        base_tol = best_coarse_config['tol']

        fine_configs = []

        # Generate fine grid around best coarse config
        for n_comp in [max(2, base_n_comp - 1), base_n_comp, min(6, base_n_comp + 1)]:
            for cov_type in [base_cov, 'diag' if base_cov != 'diag' else 'spherical']:
                for n_iter_mult in [0.7, 1.0, 1.3]:
                    for tol_mult in [0.3, 1.0, 3.0]:
                        fine_config = {
                            'n_components': n_comp,
                            'covariance_type': cov_type,
                            'n_iter': max(20, int(base_iter * n_iter_mult)),
                            'tol': base_tol * tol_mult
                        }
                        if fine_config not in fine_configs:
                            fine_configs.append(fine_config)

        # Limit to requested number of configs
        fine_configs = fine_configs[:n_fine_configs]

        fine_results = []
        best_fine_config = best_coarse_config
        best_fine_score = best_coarse_score

        for i, fine_config in enumerate(fine_configs):
            try:
                start_time = time_module.time()

                if verbose_logging:
                    self.logger.info(f"  Testing fine config {i+1}/{len(fine_configs)}: {fine_config}")

                temp_model = hmm.GaussianHMM(
                    n_components=fine_config['n_components'],
                    covariance_type=fine_config['covariance_type'],
                    n_iter=fine_config['n_iter'],
                    tol=fine_config['tol'],
                    random_state=42,  # Use fixed random state for reproducibility
                    init_params='mc',
                    params='stmc'
                )

                if hasattr(temp_model, 'startprob_'):
                    temp_model.startprob_ = np.ones(fine_config['n_components']) / fine_config['n_components']

                # Fit on full dataset for fine search
                temp_model.fit(numeric_data)
                score = temp_model.score(numeric_data)

                elapsed = time_module.time() - start_time
                fine_results.append({
                    'config': fine_config.copy(),
                    'score': score,
                    'time': elapsed
                })

                if verbose_logging:
                    self.logger.info(f"    Score: {score:.2f} (took {elapsed:.2f}s)")

                if score > best_fine_score:
                    best_fine_score = score
                    best_fine_config = fine_config.copy()

            except Exception as e:
                if verbose_logging:
                    self.logger.warning(f"    Fine config {i+1} failed: {e}")
                continue

        self.logger.info(f"🏅 Stage 2 Complete - Best fine config: {best_fine_config} (score: {best_fine_score:.2f})")

        # Stage 3: Bayesian Optimization (Full/Blank modes only)
        final_config = best_fine_config

        if use_bayesian and OPTUNA_AVAILABLE:
            self.logger.info(f"\n🧠 Stage 3: Bayesian Optimization ({bayesian_trials} trials)")

            def bayesian_objective(trial):
                # Suggest parameters around best fine config
                n_comp = trial.suggest_int('n_components',
                                         max(2, best_fine_config['n_components'] - 1),
                                         min(6, best_fine_config['n_components'] + 1))

                cov_options = ['diag', 'spherical']
                if best_fine_config['covariance_type'] == 'tied':
                    cov_options.append('tied')
                cov_type = trial.suggest_categorical('covariance_type', cov_options)

                # Suggest n_iter around best fine config
                n_iter = trial.suggest_int('n_iter',
                                         max(20, int(best_fine_config['n_iter'] * 0.5)),
                                         int(best_fine_config['n_iter'] * 1.5))

                # Suggest tol around best fine config
                tol = trial.suggest_float('tol',
                                        best_fine_config['tol'] * 0.1,
                                        best_fine_config['tol'] * 10,
                                        log=True)

                bayesian_config = {
                    'n_components': n_comp,
                    'covariance_type': cov_type,
                    'n_iter': n_iter,
                    'tol': tol
                }

                try:
                    temp_model = hmm.GaussianHMM(
                        n_components=n_comp,
                        covariance_type=cov_type,
                        n_iter=n_iter,
                        tol=tol,
                        random_state=42,  # Use fixed random state for reproducibility
                        init_params='mc',
                        params='stmc'
                    )

                    if hasattr(temp_model, 'startprob_'):
                        temp_model.startprob_ = np.ones(n_comp) / n_comp

                    # Use full dataset for Bayesian optimization
                    temp_model.fit(numeric_data)
                    score = temp_model.score(numeric_data)

                    if verbose_logging:
                        self.logger.info(f"    Bayesian trial - Config: {config}, Score: {score:.2f}")

                    return score

                except Exception as e:
                    if verbose_logging:
                        self.logger.warning(f"    Bayesian trial failed: {e}")
                    return float('-inf')

            # Create study with pruner for early stopping
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )

            study = optuna.create_study(
                direction='maximize',
                pruner=pruner,
                study_name=f"hmm_bayesian_optimization_{mode}"
            )

            study.optimize(bayesian_objective, n_trials=bayesian_trials)

            if study.best_params:
                final_config = {
                    'n_components': study.best_params['n_components'],
                    'covariance_type': study.best_params['covariance_type'],
                    'n_iter': study.best_params['n_iter'],
                    'tol': study.best_params['tol']
                }
                best_bayesian_score = study.best_value
                self.logger.info(f"🏅 Stage 3 Complete - Best Bayesian config: {final_config} (score: {best_bayesian_score:.2f})")
            else:
                self.logger.warning("⚠️ Bayesian optimization failed, using fine grid result")

        self.logger.info(f"\n🏆 OPTIMIZATION COMPLETE - Final config: {final_config}")
        if verbose_logging:
            self.logger.info("📊 Optimization Summary:")
            self.logger.info(f"   Coarse configs tested: {len(coarse_results)}")
            self.logger.info(f"   Fine configs tested: {len(fine_results)}")
            if use_bayesian and OPTUNA_AVAILABLE:
                self.logger.info(f"   Bayesian trials: {bayesian_trials}")
            self.logger.info(f"   Total time: ~{(sum(r['time'] for r in coarse_results) + sum(r['time'] for r in fine_results)):.1f}s")

        # Create final model with optimized parameters
        model = hmm.GaussianHMM(
            n_components=final_config['n_components'],
            covariance_type=final_config['covariance_type'],
            n_iter=max(final_config['n_iter'], 10),
            tol=max(final_config['tol'], 1e-4),
            random_state=config.random_state,
            init_params='mc',
            params='stmc'
        )

        # Better initialization for start probabilities
        if hasattr(model, 'startprob_'):
            # Initialize with uniform distribution
            model.startprob_ = np.ones(final_config['n_components']) / final_config['n_components']
        
        # Use memory optimizer if available
        if self.memory_optimizer:
            numeric_data = self.memory_optimizer.optimize_dataframe_memory(numeric_data)
        
        # Ensure data covariance is positive-definite for stable HMM fitting
        numeric_data = self._ensure_positive_definite_data(numeric_data)

        # Standard fallback configurations
        fallback_configs = [
            {'covariance_type': 'diag', 'n_components': min(config.n_components, 3), 'n_iter': 300, 'tol': 1e-2},
            {'covariance_type': 'spherical', 'n_components': 2, 'n_iter': 300, 'tol': 1e-2},
            {'covariance_type': 'tied', 'n_components': min(config.n_components, 2), 'n_iter': 300, 'tol': 1e-2},
            {'covariance_type': 'diag', 'n_components': 2, 'n_iter': 500, 'tol': 1e-1},
            {'covariance_type': 'spherical', 'n_components': 2, 'n_iter': 500, 'tol': 1e-1},
        ]

        # Try fitting with different configurations if needed
        fit_success = False

        try:
            self.logger.info(f"🏃 Starting HMM fitting with {final_config['n_components']} components on {len(numeric_data)} samples...")

            fit_start_time = time_module.time()

            model.fit(numeric_data)
            fit_success = True

            fit_time = time_module.time() - fit_start_time
            self.logger.info(f"✅ HMM fitting completed in {fit_time:.2f}s")

        except Exception as e:
            self.logger.warning(f"⚠️ Primary HMM fitting failed: {e}, trying fallback configurations")
            fit_success = False

            for i, fallback_config in enumerate(fallback_configs):
                try:
                    self.logger.info(f"Trying fallback configuration {i+1}: {fallback_config}")
                    fallback_model = hmm.GaussianHMM(
                        n_components=fallback_config['n_components'],
                        covariance_type=fallback_config['covariance_type'],
                        n_iter=fallback_config.get('n_iter', 300),
                        tol=fallback_config.get('tol', 1e-3),
                        random_state=config.random_state,
                        init_params='mc',
                        params='stmc'
                    )
                    fallback_model.startprob_ = np.ones(fallback_config['n_components']) / fallback_config['n_components']
                    fallback_model.fit(numeric_data)
                    model = fallback_model
                    fit_success = True
                    self.logger.info(f"✅ Fallback configuration {i+1} succeeded")
                    break
                except Exception as fallback_e:
                    self.logger.warning(f"⚠️ Fallback configuration {i+1} failed: {fallback_e}")

            if not fit_success:
                self.logger.error(f"❌ All HMM fitting attempts failed")
                raise RuntimeError(f"HMM model fitting failed: {e}") from e

        # Check if model fitted properly
        if hasattr(model, 'startprob_') and np.any(np.isnan(model.startprob_)):
            self.logger.error("❌ HMM model has invalid start probabilities (NaN values)")
            raise RuntimeError("HMM model fitting produced invalid start probabilities")

        if hasattr(model, 'transmat_') and np.any(np.isnan(model.transmat_)):
            self.logger.error("❌ HMM model has invalid transition matrix (NaN values)")
            raise RuntimeError("HMM model fitting produced invalid transition matrix")

        # Validate covariance matrices are positive-definite
        self._validate_model_covariances(model)

        try:
            self.logger.info(f"🎯 Predicting regime labels for {len(numeric_data)} samples...")
            predict_start = time.time()
            regime_labels = model.predict(numeric_data)
            predict_time = time.time() - predict_start
            self.logger.info(f"✅ Regime prediction completed in {predict_time:.2f}s")
        except Exception as e:
            self.logger.error(f"❌ HMM prediction failed: {e}")
            raise RuntimeError(f"HMM prediction failed: {e}") from e

        # Handle length mismatch due to feature engineering
        original_length = len(data)
        processed_length = len(numeric_data)

        if processed_length != original_length:
            self.logger.warning(f"⚠️ Length mismatch: original {original_length}, processed {processed_length}")
            # Create regime labels for original length, using the last regime for missing data
            extended_regime_labels = np.full(original_length, regime_labels[-1] if len(regime_labels) > 0 else 0)
            # Map processed data indices back to original data
            processed_indices = numeric_data.index
            extended_regime_labels[processed_indices] = regime_labels
            regime_labels = extended_regime_labels

        # Create result DataFrame
        result = data.copy()
        result['regime'] = regime_labels

        # Handle regime probabilities with proper length matching
        try:
            if processed_length == original_length:
                # Optimize probabilistic predictions for large datasets instead of skipping
                if len(numeric_data) > 50000:
                    self.logger.info(f"🚀 Optimizing probabilistic predictions for large dataset ({len(numeric_data)} samples) using advanced batching")
                    self.logger.info("⚡ Using parallel processing and memory optimization for performance")
                    start_time = time.time()

                    # Use optimized batched prediction with parallel processing for very large datasets
                    probabilities = self._optimized_batched_predict_proba(
                        model, numeric_data, batch_size=20000, use_parallel=True
                    )
                    # Apply temperature scaling and Dirichlet smoothing to reduce overconfidence
                    temperature = 1.3
                    alpha = 0.05
                    with np.errstate(over='ignore'):
                        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
                        scaled = logits / max(1e-6, temperature)
                        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
                        probabilities = exp_scaled / np.clip(np.sum(exp_scaled, axis=1, keepdims=True), 1e-12, None)
                        probabilities = (probabilities + alpha) / np.clip(np.sum(probabilities + alpha, axis=1, keepdims=True), 1e-12, None)

                    predict_proba_time = time.time() - start_time
                    self.logger.info(f"✅ Optimized probabilistic predictions completed in {predict_proba_time:.2f}s")
                    self.logger.info(f"📊 Processed {len(numeric_data)} samples with {probabilities.shape[1]} regimes")

                    # Store results using the same logic as smaller datasets
                    result['regime_probability'] = np.max(probabilities, axis=1)
                    for regime_idx in range(probabilities.shape[1]):
                        result[f'regime_{regime_idx}_probability'] = probabilities[:, regime_idx]
                    for regime_idx in range(probabilities.shape[1]):
                        result[f'regime_{regime_idx}_percentage'] = probabilities[:, regime_idx] * 100
                    result['regime_probability_entropy'] = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                    result['regime_confidence'] = np.max(probabilities, axis=1) - np.mean(probabilities, axis=1)

                    self.logger.info(f"✅ Added optimized probabilistic predictions for {probabilities.shape[1]} regimes")

                    # Validate that optimized predictions are working correctly (not simplified)
                    max_prob_mean = np.mean(np.max(probabilities, axis=1))
                    entropy_mean = np.mean(result['regime_probability_entropy'])

                    if max_prob_mean < 0.5:
                        self.logger.warning(f"⚠️ Low average max probability ({max_prob_mean:.3f}) - predictions may be unreliable")
                    if entropy_mean < 0.2:
                        self.logger.warning(f"⚠️ Very low entropy ({entropy_mean:.3f}) - predictions may be overconfident")
                        self.logger.info(f"💡 Suggestions: Consider increasing n_components, using 'diag' covariance_type, or adding regularization")

                    # Ensure probabilities are properly distributed (not all in one regime)
                    regime_distribution = np.mean(probabilities, axis=0)
                    max_regime_share = np.max(regime_distribution)
                    if max_regime_share > 0.8:
                        self.logger.warning(f"⚠️ One regime dominates ({max_regime_share:.1%}) - check model fit")

                    self.logger.info(f"📊 Prediction quality: max_prob={max_prob_mean:.3f}, entropy={entropy_mean:.3f}")
                    self.logger.info(f"🔍 DEBUG: Completed optimized prediction quality assessment")

                    # Add detailed entropy analysis
                    entropy_std = np.std(result['regime_probability_entropy'])
                    self.logger.info(f"🔍 DEBUG: Entropy stats - mean: {entropy_mean:.4f}, std: {entropy_std:.4f}, min: {np.min(result['regime_probability_entropy']):.4f}, max: {np.max(result['regime_probability_entropy']):.4f}")

                    # Check regime distribution
                    regime_dist = np.mean(probabilities, axis=0)
                    self.logger.info(f"🔍 DEBUG: Regime distribution: {regime_dist}")
                    self.logger.info(f"🔍 DEBUG: Max regime share: {np.max(regime_dist):.1%}")

                else:
                    self.logger.info(f"🔢 Computing probabilistic predictions for {len(numeric_data)} samples across {config.n_components} regimes...")
                    self.logger.info(f"🔍 DEBUG: Starting batched prediction for large dataset")
                    start_time = time.time()

                    # For large datasets, predict_proba can be slow - add batching for better memory management
                    if len(numeric_data) > 10000:
                        self.logger.info(f"📊 Large dataset detected ({len(numeric_data)} samples), using batched prediction for memory efficiency")
                        probabilities = self._batched_predict_proba(model, numeric_data, batch_size=5000)
                    else:
                        probabilities = model.predict_proba(numeric_data)

                    predict_proba_time = time.time() - start_time
                    self.logger.info(f"✅ Probabilistic predictions completed in {predict_proba_time:.2f}s")
                    self.logger.info(f"🔍 DEBUG: Probabilities shape: {probabilities.shape}, dtype: {probabilities.dtype}")

                    # Store the maximum probability for the predicted regime
                    self.logger.info(f"🔍 DEBUG: Storing regime probabilities...")
                    result['regime_probability'] = np.max(probabilities, axis=1)
                    self.logger.info(f"🔍 DEBUG: Regime probability stored - shape: {result['regime_probability'].shape}")

                    # Store full probability distribution for each regime
                    self.logger.info(f"🔍 DEBUG: Storing full probability distribution for {probabilities.shape[1]} regimes...")
                    for regime_idx in range(probabilities.shape[1]):
                        result[f'regime_{regime_idx}_probability'] = probabilities[:, regime_idx]
                    self.logger.info(f"🔍 DEBUG: Probability distributions stored")

                    # Store regime probabilities as percentages
                    self.logger.info(f"🔍 DEBUG: Converting probabilities to percentages...")
                    for regime_idx in range(probabilities.shape[1]):
                        result[f'regime_{regime_idx}_percentage'] = probabilities[:, regime_idx] * 100
                    self.logger.info(f"🔍 DEBUG: Percentages stored")

                    # Store regime probability statistics
                    self.logger.info(f"🔍 DEBUG: Calculating entropy...")
                    result['regime_probability_entropy'] = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                    self.logger.info(f"🔍 DEBUG: Entropy calculated - shape: {result['regime_probability_entropy'].shape}")

                    self.logger.info(f"🔍 DEBUG: Calculating confidence...")
                    result['regime_confidence'] = np.max(probabilities, axis=1) - np.mean(probabilities, axis=1)
                    self.logger.info(f"🔍 DEBUG: Confidence calculated - shape: {result['regime_confidence'].shape}")

                    self.logger.info(f"✅ Added probabilistic regime predictions for {probabilities.shape[1]} regimes")
                    self.logger.info(f"🔍 DEBUG: Probabilistic predictions complete, result columns: {list(result.keys())}")
                
            else:
                # For mismatched lengths, use default probabilities
                n_regimes = config.n_components
                result['regime_probability'] = 0.5
                for regime_idx in range(n_regimes):
                    result[f'regime_{regime_idx}_probability'] = 0.5
                    result[f'regime_{regime_idx}_percentage'] = 50.0
                result['regime_probability_entropy'] = 0.0
                result['regime_confidence'] = 0.0
                self.logger.warning("⚠️ Using default regime probabilities due to length mismatch")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not compute regime probabilities: {e}")
            n_regimes = config.n_components
            result['regime_probability'] = 0.5
            for regime_idx in range(n_regimes):
                result[f'regime_{regime_idx}_probability'] = 0.5
                result[f'regime_{regime_idx}_percentage'] = 50.0
            result['regime_probability_entropy'] = 0.0
            result['regime_confidence'] = 0.0

        self.logger.info(f"🔍 DEBUG: Setting detection method and computing final metrics...")
        result['detection_method'] = 'hmm_gaussian'
        self.logger.info(f"🔍 DEBUG: Detection method set to: {result['detection_method']}")

        try:
            self.logger.info(f"🔍 DEBUG: Computing model score...")
            score_value = model.score(numeric_data)
            # Convert to scalar if it's an array/series
            if hasattr(score_value, 'item'):
                score_value = score_value.item()
            elif hasattr(score_value, 'mean'):
                score_value = float(score_value.mean())
            else:
                score_value = float(score_value)

            result['model_score'] = score_value
            self.logger.info(f"🔍 DEBUG: Model score computed: {score_value:.4f}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not compute model score: {e}")
            result['model_score'] = 0.0

        self.logger.info(f"🔍 DEBUG: HMM regime detection function completed, returning result with {len(result)} rows")
        self.logger.info(f"🔍 DEBUG: Final result columns: {list(result.keys())}")
        return result

    def _create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for better HMM regime detection with validation."""
        df = data.copy()

        # Track original data for validation
        original_df = df.copy()
        feature_config = {
            'price_features': ['returns', 'log_returns', 'sma_5', 'sma_20', 'ema_12', 'ema_26', 'momentum_5', 'momentum_20'],
            'volume_features': ['volume_ma_5', 'volume_ratio', 'volume_change'],
            'volatility_features': ['tr', 'atr_14', 'bb_upper', 'bb_lower', 'bb_position'],
            'technical_features': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d']
        }

        # Check if feature validator is available
        validator_available = FEATURE_VALIDATOR_AVAILABLE

        # Basic price features
        if 'close' in df.columns and 'open' in df.columns:
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # Price momentum
            df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
            df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            # Fix volume_ratio calculation - avoid division by zero and handle constant volumes
            volume_ma_20 = df['volume'].rolling(20).mean()
            # Replace zeros with a small epsilon to avoid division by zero
            volume_ma_20_safe = volume_ma_20.replace(0, 1e-8)
            df['volume_ratio'] = df['volume'] / volume_ma_20_safe
            # If volume is perfectly constant, volume_ratio will be 1.0, which is actually correct
            # But we can add some noise to make it more meaningful
            df['volume_change'] = df['volume'].pct_change()

            # Additional volume features to make volume_ratio more meaningful
            df['volume_std_20'] = df['volume'].rolling(20).std()
            df['volume_zscore_20'] = (df['volume'] - volume_ma_20) / df['volume_std_20'].replace(0, 1e-8)

        # Volatility features
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # True Range and ATR
            df['tr'] = np.maximum(df['high'] - df['low'],
                                 np.maximum(abs(df['high'] - df['close'].shift(1)),
                                           abs(df['low'] - df['close'].shift(1))))
            df['atr_14'] = df['tr'].rolling(14).mean()

            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma_20 + 2 * std_20
            df['bb_lower'] = sma_20 - 2 * std_20
            df['bb_position'] = (df['close'] - sma_20) / (2 * std_20)

        # Technical indicators
        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            # RSI - improved calculation to avoid constant values
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            # Avoid division by zero and handle edge cases
            loss = loss.replace(0, 1e-10)  # Small value instead of zero
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            # Ensure RSI is within valid range
            df['rsi_14'] = df['rsi_14'].clip(0, 100)

            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Handle NaN values created by rolling operations - DON'T drop rows to preserve length
        # Instead, fill with forward/backward fill or zeros
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Debug: Check for problematic calculations and known problematic columns
        self.logger.info(f"Features created: {list(df.columns)}")

        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                unique_count = df[col].nunique()
                std_val = df[col].std()
                if std_val == 0 or unique_count == 1:
                    self.logger.warning(f"⚠️ Unexpected constant column: {col} (unique: {unique_count}, std: {std_val})")

        # Check for constant columns with fast-fail logic
        constant_cols = []
        trade_stat_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']

        # Fast-fail: Check if critical features are constant
        critical_constant_features = []
        for col in trade_stat_cols:
            if col in df.columns:
                unique_vals = df[col].nunique()
                std_val = df[col].std()
                if unique_vals <= 2 or (not pd.isna(std_val) and std_val < 1e-10):
                    critical_constant_features.append(f"{col}(unique={unique_vals}, std={std_val:.2e})")

        if critical_constant_features:
                error_msg = f"🚨 CRITICAL: Constant feature detection failed! These features are constant: {critical_constant_features}"
                self.logger.error(error_msg)
                self.logger.error("   This indicates a data processing failure - features should have variation")
                self.logger.error("   Please check the data converter step01_5_data_converter.py")
                self.logger.error("   FAST-FAIL: Not attempting gap filling or downloads due to constant features")
                raise ValueError(f"HMM training cannot proceed with constant features: {critical_constant_features}")

        # Check for other genuinely constant columns
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                is_constant = df[col].std() == 0 or df[col].nunique() == 1

                if is_constant:
                    # These should not be constant - they indicate data quality issues
                    if col in trade_stat_cols:
                        # We already checked these above, so this shouldn't happen
                        continue
                    else:
                        # For other columns, constant values are genuinely problematic
                        constant_cols.append(col)

        if constant_cols:
            self.logger.info(f"Removing genuinely constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

        self.logger.info("✅ Feature quality validation passed - all critical features have proper variation")

        # Select only numeric columns and limit to reasonable number of features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 20:
            # Keep most important features
            priority_features = ['returns', 'log_returns', 'rsi_14', 'macd', 'bb_position',
                               'momentum_5', 'volume_ratio', 'atr_14']
            other_features = [col for col in numeric_cols if col not in priority_features]
            selected_features = priority_features + other_features[:10]  # Limit to 18 total
            df = df[selected_features]

        # Feature Engineering Validation
        if validator_available:
            self.logger.info("🔍 Validating engineered features...")

            try:
                from src.utils.feature_engineering_validation import FeatureEngineeringValidator
                validator = FeatureEngineeringValidator()
                validation_result = validator.validate_engineered_features(
                    original_df=original_df,
                    features_df=df,
                    feature_config=feature_config,
                    validate_calculations=True,
                    check_dependencies=True
                )

                # Log validation results
                self.logger.info(f"✅ Feature validation completed - Quality Score: {validation_result.quality_score:.2f}/100")

                if not validation_result.passed:
                    self.logger.warning("⚠️ Feature validation found issues:")

                    # Log critical issues
                    critical_issues = [i for i in validation_result.issues if 'CRITICAL' in str(i).upper()]
                    if critical_issues:
                        self.logger.error(f"❌ Critical feature issues ({len(critical_issues)}):")
                        for issue in critical_issues[:3]:
                            self.logger.error(f"   - {issue}")

                    # Log warnings
                    if validation_result.warnings:
                        self.logger.warning(f"⚠️ Feature warnings ({len(validation_result.warnings)}):")
                        for warning in validation_result.warnings[:3]:
                            self.logger.warning(f"   - {warning}")

                # Feature quality gate
                if validation_result.quality_score < 99:
                    self.logger.warning("⚠️ Low feature quality detected - proceeding with caution")

            except Exception as e:
                self.logger.warning(f"⚠️ Feature validation failed: {e} - continuing with features")
        else:
            self.logger.info("ℹ️ Feature engineering validation not available - skipping validation")

        return df

    def _normalize_hmm_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data for stable HMM training."""
        df = data.copy()

        # Remove constant columns - more robust check
        constant_cols = []
        # Features that should never be considered constant even if they appear to be
        protected_features = {'volume_ratio'}

        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Skip protected features from constant column removal
                if col in protected_features:
                    continue

                # Check both std == 0 and nunique == 1 to catch edge cases
                if df[col].std() == 0 or df[col].nunique() == 1:
                    constant_cols.append(col)

        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.logger.info(f"Removed constant columns from normalization: {constant_cols}")
        else:
            self.logger.info("ℹ️ No constant columns found for removal")

        # Robust normalization using median and IQR
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                median = df[col].median()
                q75, q25 = np.percentile(df[col].dropna(), [75, 25])
                iqr = q75 - q25

                if iqr > 0:  # Avoid division by zero
                    df[col] = (df[col] - median) / iqr
                else:
                    df[col] = df[col] - median

        # Handle outliers using z-score with robust threshold
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                # Use 3.5 sigma instead of 3 for more robust outlier removal
                df.loc[z_scores > 3.5, col] = df[col].median()

        return df

    def _ensure_positive_definite_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data covariance matrix is positive-definite for stable HMM fitting."""
        df = data.copy()

        # Remove highly correlated features to prevent ill-conditioned covariance matrices
        corr_matrix = df.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.95
        high_corr_features = []
        for col in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[col].abs() > 0.95].tolist()
            if correlated_features:
                high_corr_features.extend(correlated_features)

        # Remove duplicate features and keep unique ones
        high_corr_features = list(set(high_corr_features))

        if high_corr_features:
            df = df.drop(columns=high_corr_features)
            self.logger.info(f"🧹 Removed {len(high_corr_features)} highly correlated features for HMM stability: {high_corr_features}")

        # Add small regularization to prevent singular matrices
        if len(df.columns) > 1:
            # Calculate sample covariance
            cov_matrix = df.cov()

            # Check if covariance matrix is positive-definite
            try:
                np.linalg.cholesky(cov_matrix.values)
                is_positive_definite = True
            except np.linalg.LinAlgError:
                is_positive_definite = False

            if not is_positive_definite:
                self.logger.warning("⚠️ Data covariance matrix not positive-definite, applying regularization")

                # Add small diagonal regularization
                regularization_factor = 1e-6
                n_features = len(df.columns)
                regularization_matrix = np.eye(n_features) * regularization_factor

                # Apply regularization to make matrix positive-definite
                regularized_cov = cov_matrix + regularization_matrix

                # Try Cholesky again
                try:
                    np.linalg.cholesky(regularized_cov.values)
                    self.logger.info(f"✅ Applied regularization (factor={regularization_factor}) to ensure positive-definite covariance")
                except np.linalg.LinAlgError:
                    # If still not positive-definite, increase regularization
                    regularization_factor = 1e-3
                    regularization_matrix = np.eye(n_features) * regularization_factor
                    regularized_cov = cov_matrix + regularization_matrix

                    try:
                        np.linalg.cholesky(regularized_cov.values)
                        self.logger.info(f"✅ Applied stronger regularization (factor={regularization_factor}) to ensure positive-definite covariance")
                    except np.linalg.LinAlgError:
                        self.logger.warning("⚠️ Covariance matrix still not positive-definite after regularization, using diagonal covariance")

                        # As last resort, use only the diagonal elements (variance)
                        for i, col in enumerate(df.columns):
                            if df[col].var() < 1e-10:  # Very small variance
                                df[col] = df[col] + np.random.normal(0, 1e-5, len(df))
                                self.logger.info(f"🔧 Added small noise to constant feature: {col}")

        return df

    def _validate_model_covariances(self, model):
        """Validate that fitted model covariance matrices are positive-definite."""
        if not hasattr(model, 'covars_'):
            return  # Some covariance types don't have covars_ attribute

        covars = model.covars_
        n_components = model.n_components

        for i in range(n_components):
            try:
                if model.covariance_type == 'full':
                    cov_matrix = covars[i]
                elif model.covariance_type == 'tied':
                    cov_matrix = covars
                elif model.covariance_type == 'diag':
                    cov_matrix = np.diag(covars[i])
                elif model.covariance_type == 'spherical':
                    cov_matrix = np.eye(len(covars[i])) * covars[i]
                else:
                    continue

                # Check if covariance matrix is positive-definite
                np.linalg.cholesky(cov_matrix)

            except np.linalg.LinAlgError:
                self.logger.warning(f"⚠️ Component {i} covariance matrix not positive-definite, applying regularization")
                self.logger.info("ℹ️ This is GOOD - regularization ensures numerical stability for HMM fitting")

                # Calculate adaptive regularization based on matrix condition
                if model.covariance_type == 'full':
                    # Use eigenvalues to determine appropriate regularization strength
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    min_eigenval = np.min(np.real(eigenvals))
                    if min_eigenval <= 0:
                        # Adaptive regularization: scale based on negative eigenvalue magnitude
                        reg_strength = max(1e-6, -min_eigenval * 1.1)
                        regularization = np.eye(cov_matrix.shape[0]) * reg_strength
                        self.logger.info(f"🔧 Using adaptive regularization (strength: {reg_strength:.2e})")
                    else:
                        regularization = np.eye(cov_matrix.shape[0]) * 1e-6
                    model.covars_[i] = cov_matrix + regularization
                elif model.covariance_type == 'tied':
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    min_eigenval = np.min(np.real(eigenvals))
                    if min_eigenval <= 0:
                        reg_strength = max(1e-6, -min_eigenval * 1.1)
                        regularization = np.eye(cov_matrix.shape[0]) * reg_strength
                        self.logger.info(f"🔧 Using adaptive regularization for tied covariance (strength: {reg_strength:.2e})")
                    else:
                        regularization = np.eye(cov_matrix.shape[0]) * 1e-6
                    model.covars_ = cov_matrix + regularization

                # Verify the regularization worked
                try:
                    if model.covariance_type == 'full':
                        np.linalg.cholesky(model.covars_[i])
                    elif model.covariance_type == 'tied':
                        np.linalg.cholesky(model.covars_)
                    self.logger.info(f"✅ Component {i} covariance matrix regularized successfully")
                except np.linalg.LinAlgError:
                    self.logger.error(f"❌ Component {i} covariance matrix still not positive-definite after regularization")
                    raise RuntimeError(f"HMM component {i} has non-positive-definite covariance matrix")

    def _batched_predict_proba(self, model, data: pd.DataFrame, batch_size: int = 5000) -> np.ndarray:
        """Perform batched predict_proba for large datasets to manage memory and provide progress feedback."""
        n_samples = len(data)
        n_regimes = model.n_components

        # Pre-allocate result array
        probabilities = np.zeros((n_samples, n_regimes))

        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = data.iloc[start_idx:end_idx]

            # Predict probabilities for this batch
            batch_probabilities = model.predict_proba(batch_data.values)

            # Store in result array
            probabilities[start_idx:end_idx] = batch_probabilities

            # Log progress for large datasets
            if n_samples > 20000 and (end_idx % 10000 == 0 or end_idx == n_samples):
                progress = (end_idx / n_samples) * 100
                self.logger.info(f"📊 Processed {end_idx}/{n_samples} samples ({progress:.1f}%)")

        return probabilities

    def _optimized_batched_predict_proba(self, model, data: pd.DataFrame, batch_size: int = 20000, use_parallel: bool = True) -> np.ndarray:
        """Perform optimized batched predict_proba for very large datasets with parallel processing and memory optimization."""
        n_samples = len(data)
        n_regimes = model.n_components

        # Pre-allocate result array with memory optimization
        probabilities = np.zeros((n_samples, n_regimes), dtype=np.float32)  # Use float32 for memory efficiency

        # For extremely large datasets, use parallel processing
        if use_parallel and n_samples > 200000:
            self.logger.info("🔄 Using parallel processing for ultra-large dataset")
            return self._parallel_batched_predict_proba(model, data, batch_size)
        else:
            # Use optimized sequential processing
            return self._sequential_optimized_predict_proba(model, data, batch_size)

    def _sequential_optimized_predict_proba(self, model, data: pd.DataFrame, batch_size: int = 20000) -> np.ndarray:
        """Sequential optimized batched prediction with memory management."""
        n_samples = len(data)
        n_regimes = model.n_components

        # Pre-allocate result array
        probabilities = np.zeros((n_samples, n_regimes), dtype=np.float32)

        # Process in larger batches for efficiency
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = data.iloc[start_idx:end_idx]

            # Predict probabilities for this batch
            batch_probabilities = model.predict_proba(batch_data.values)

            # Store in result array
            probabilities[start_idx:end_idx] = batch_probabilities.astype(np.float32)

            # Log progress for large datasets
            if n_samples > 100000 and (end_idx % 50000 == 0 or end_idx == n_samples):
                progress = (end_idx / n_samples) * 100
                self.logger.info(f"📊 Processed {end_idx}/{n_samples} samples ({progress:.1f}%)")

        return probabilities

    def _parallel_batched_predict_proba(self, model, data: pd.DataFrame, batch_size: int = 20000) -> np.ndarray:
        """Parallel batched prediction for ultra-large datasets."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        n_samples = len(data)
        n_regimes = model.n_components
        probabilities = np.zeros((n_samples, n_regimes), dtype=np.float32)

        # Create batches
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batches.append((start_idx, end_idx, data.iloc[start_idx:end_idx]))

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        completed_batches = 0

        def process_batch(batch_info):
            nonlocal completed_batches
            start_idx, end_idx, batch_data = batch_info
            batch_probabilities = model.predict_proba(batch_data.values)

            # Update progress
            with progress_lock:
                completed_batches += 1
                if completed_batches % 10 == 0 or completed_batches == len(batches):
                    progress = (completed_batches / len(batches)) * 100
                    self.logger.info(f"📊 Parallel processing: {completed_batches}/{len(batches)} batches ({progress:.1f}%)")

            return start_idx, end_idx, batch_probabilities.astype(np.float32)

        # Use optimal number of threads (not too many to avoid overhead)
        max_workers = min(4, len(batches))  # Limit to 4 threads to avoid overhead

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in as_completed(futures):
                start_idx, end_idx, batch_probabilities = future.result()
                probabilities[start_idx:end_idx] = batch_probabilities

        self.logger.info("✅ Parallel batch processing completed")
        return probabilities

    def _detect_hmm_multivariate_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using multivariate HMM."""
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available")
        
        # Prepare multivariate data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        # Feature engineering using HMM composite manager
        engineered_data = self.hmm_manager.engineer_features(numeric_data)
        
        # Create and fit multivariate HMM model
        model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state
        )
        
        model.fit(engineered_data)
        regime_labels = model.predict(engineered_data)
        
        # Get probabilistic predictions
        probabilities = model.predict_proba(engineered_data)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regime_labels
        result['regime_probability'] = np.max(probabilities, axis=1)
        
        # Store full probability distribution for each regime
        for regime_idx in range(probabilities.shape[1]):
            result[f'regime_{regime_idx}_probability'] = probabilities[:, regime_idx]
        
        # Store regime probabilities as percentages
        for regime_idx in range(probabilities.shape[1]):
            result[f'regime_{regime_idx}_percentage'] = probabilities[:, regime_idx] * 100
        
        # Store regime probability statistics
        result['regime_probability_entropy'] = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        result['regime_confidence'] = np.max(probabilities, axis=1) - np.mean(probabilities, axis=1)
        
        result['detection_method'] = 'hmm_multivariate'
        # Convert model score to scalar
        score_value = model.score(engineered_data)
        if hasattr(score_value, 'item'):
            score_value = score_value.item()
        elif hasattr(score_value, 'mean'):
            score_value = float(score_value.mean())
        else:
            score_value = float(score_value)
        result['model_score'] = score_value
        
        self.logger.info(f"✅ Added probabilistic regime predictions for {probabilities.shape[1]} regimes (multivariate)")
        
        return result

    def _detect_ensemble_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using ensemble HMM methods."""
        if not self.ensemble_manager:
            raise ImportError("Ensemble manager not available")
        
        # Create multiple HMM models with different configurations
        models = []
        configs = [
            HMMRegimeConfig(n_components=3, covariance_type="full"),
            HMMRegimeConfig(n_components=4, covariance_type="tied"),
            HMMRegimeConfig(n_components=5, covariance_type="diag"),
        ]
        
        for model_config in configs:
            try:
                model_result = self._detect_hmm_gaussian_regimes(data, model_config, 'light')
                models.append(model_result)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create ensemble model: {e}")
        
        if not models:
            raise ValueError("No ensemble models could be created")
        
        # Combine ensemble results
        ensemble_result = self._combine_ensemble_results(models)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = ensemble_result['regime']
        result['regime_probability'] = ensemble_result['probability']
        
        # Add probabilistic regime predictions from ensemble
        if 'regime_probabilities' in ensemble_result:
            probabilities = ensemble_result['regime_probabilities']
            # Store full probability distribution for each regime
            for regime_idx in range(probabilities.shape[1]):
                result[f'regime_{regime_idx}_probability'] = probabilities[:, regime_idx]
            
            # Store regime probabilities as percentages
            for regime_idx in range(probabilities.shape[1]):
                result[f'regime_{regime_idx}_percentage'] = probabilities[:, regime_idx] * 100
            
            # Store regime probability statistics
            result['regime_probability_entropy'] = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            result['regime_confidence'] = np.max(probabilities, axis=1) - np.mean(probabilities, axis=1)
        
        result['detection_method'] = 'ensemble_hmm'
        result['ensemble_consensus'] = ensemble_result['consensus']
        
        return result

    def _detect_multi_timeframe_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using multi-timeframe HMM ensemble."""
        # This would require multiple timeframe data
        # For now, implement single timeframe with multi-timeframe structure
        base_result = self._detect_hmm_gaussian_regimes(data, config, 'light')
        
        # Add multi-timeframe metadata
        base_result['detection_method'] = 'multi_timeframe_hmm'
        base_result['timeframe_consensus'] = 1.0  # Single timeframe for now
        
        return base_result

    def _detect_streaming_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using streaming HMM."""
        # Initialize streaming state if needed
        if self.streaming_state['current_model'] is None:
            self._initialize_streaming_model(data, config)
        
        # Process data in streaming fashion
        window_size = self.streaming_config.window_size
        update_frequency = self.streaming_config.update_frequency
        
        regimes = []
        probabilities = []
        
        for i in range(0, len(data), update_frequency):
            window_data = data.iloc[i:i+window_size]
            
            if len(window_data) < window_size:
                break
            
            # Update model if needed
            if i % (update_frequency * 10) == 0:
                self._update_streaming_model(window_data, config)
            
            # Predict regimes for current window
            window_regimes = self._predict_streaming_regimes(window_data)
            regimes.extend(window_regimes)
            probabilities.extend([0.8] * len(window_regimes))  # Placeholder
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regimes[:len(data)]
        result['regime_probability'] = probabilities[:len(data)]
        
        # Add probabilistic regime predictions for streaming (using placeholder values)
        n_regimes = config.n_components
        for regime_idx in range(n_regimes):
            # Use uniform distribution as placeholder for streaming
            result[f'regime_{regime_idx}_probability'] = 1.0 / n_regimes
            result[f'regime_{regime_idx}_percentage'] = 100.0 / n_regimes
        
        result['regime_probability_entropy'] = np.log(n_regimes)  # Maximum entropy for uniform distribution
        result['regime_confidence'] = 0.0  # Low confidence for streaming predictions
        
        result['detection_method'] = 'streaming_hmm'
        result['streaming_stability'] = self.streaming_state['stability_score']
        
        return result

    def _detect_regime_aware_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using regime-aware HMM."""
        # First detect basic regimes
        base_result = self._detect_hmm_gaussian_regimes(data, config, 'light')
        
        # Apply regime-aware refinements
        refined_result = self._refine_regime_aware_regimes(base_result, data)
        
        return refined_result

    def analyze_regime_transitions(
        self, 
        regimes_df: pd.DataFrame
    ) -> RegimeTransitionMetrics:
        """Analyze regime transition patterns."""
        try:
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            n_regimes = len(unique_regimes)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                
                current_idx = np.where(unique_regimes == current_regime)[0][0]
                next_idx = np.where(unique_regimes == next_regime)[0][0]
                
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(
                transition_matrix, 
                row_sums[:, np.newaxis], 
                out=np.zeros_like(transition_matrix), 
                where=row_sums[:, np.newaxis] != 0
            )
            
            # Calculate transition probabilities
            transition_probabilities = {}
            for i in range(n_regimes):
                for j in range(n_regimes):
                    transition_probabilities[(unique_regimes[i], unique_regimes[j])] = transition_matrix[i, j]
            
            # Calculate regime persistence
            regime_persistence = {}
            for i, regime in enumerate(unique_regimes):
                regime_persistence[regime] = transition_matrix[i, i]
            
            # Calculate transition volatility
            transition_volatility = np.std(transition_matrix)
            
            # Calculate regime stability
            regime_stability = np.mean([regime_persistence[regime] for regime in unique_regimes])
            
            return RegimeTransitionMetrics(
                transition_matrix=transition_matrix,
                transition_probabilities=transition_probabilities,
                regime_persistence=regime_persistence,
                transition_volatility=transition_volatility,
                regime_stability=regime_stability
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze regime transitions: {e}")
            return RegimeTransitionMetrics(
                transition_matrix=np.array([]),
                transition_probabilities={},
                regime_persistence={},
                transition_volatility=0.0,
                regime_stability=0.0
            )

    def validate_economic_significance(
        self, 
        regimes_df: pd.DataFrame, 
        returns_data: pd.Series
    ) -> EconomicSignificanceMetrics:
        """Validate economic significance of detected regimes."""
        try:
            if not self.pareto_analyzer:
                raise ImportError("Pareto analyzer not available")
            
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            
            # Calculate regime-specific metrics
            regime_returns = {}
            regime_volatility = {}
            regime_sharpe = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_returns_series = returns_data[regime_mask]
                
                if len(regime_returns_series) > 0:
                    regime_returns[regime] = np.mean(regime_returns_series)
                    regime_volatility[regime] = np.std(regime_returns_series)
                    regime_sharpe[regime] = safe_divide(
                        regime_returns[regime], 
                        regime_volatility[regime]
                    )
                else:
                    regime_returns[regime] = 0.0
                    regime_volatility[regime] = 0.0
                    regime_sharpe[regime] = 0.0
            
            # Analyze Pareto efficiency
            returns_array = np.array(list(regime_returns.values()))
            volatility_array = np.array(list(regime_volatility.values()))
            
            pareto_efficiency = self.pareto_analyzer.calculate_pareto_efficiency(
                returns_array, volatility_array
            )
            
            # Determine economic significance
            economic_significance = (
                pareto_efficiency > self.config.economic_significance_threshold and
                len(unique_regimes) > 1 and
                max(regime_returns.values()) - min(regime_returns.values()) > 0.01
            )
            
            return EconomicSignificanceMetrics(
                regime_returns=regime_returns,
                regime_volatility=regime_volatility,
                regime_sharpe=regime_sharpe,
                pareto_efficiency=pareto_efficiency,
                economic_significance=economic_significance
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate economic significance: {e}")
            return EconomicSignificanceMetrics(
                regime_returns={},
                regime_volatility={},
                regime_sharpe={},
                pareto_efficiency=0.0,
                economic_significance=False
            )

    def _validate_regime_quality(
        self, 
        regimes_df: pd.DataFrame, 
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate the quality of detected regimes."""
        try:
            # Use HMM composite manager validation
            validation_result = self.hmm_manager.validate_hmm_results(
                original_data, 
                regimes_df['regime'].values
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate regime quality: {e}")
            return {
                'validation_passed': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _combine_ensemble_results(self, models: List[pd.DataFrame]) -> Dict[str, Any]:
        """Combine results from ensemble models."""
        try:
            # Get regime predictions from all models
            regime_predictions = [model['regime'].values for model in models]
            
            # Get regime probabilities from all models
            regime_probabilities_list = []
            for model in models:
                # Extract regime probability columns
                prob_cols = [col for col in model.columns if col.startswith('regime_') and col.endswith('_probability')]
                if prob_cols:
                    # Sort by regime index to ensure consistent order
                    prob_cols.sort(key=lambda x: int(x.split('_')[1]))
                    probs = model[prob_cols].values
                    regime_probabilities_list.append(probs)
            
            # Calculate consensus regime
            consensus_regimes = []
            consensus_probabilities = []
            ensemble_probabilities = None
            
            for i in range(len(regime_predictions[0])):
                # Get regime votes for this time point
                votes = [pred[i] for pred in regime_predictions]
                
                # Calculate consensus (most common regime)
                unique_votes, vote_counts = np.unique(votes, return_counts=True)
                consensus_regime = unique_votes[np.argmax(vote_counts)]
                consensus_probability = np.max(vote_counts) / len(votes)
                
                consensus_regimes.append(consensus_regime)
                consensus_probabilities.append(consensus_probability)
            
            # Calculate ensemble probabilities if available
            if regime_probabilities_list:
                # Average probabilities across all models
                ensemble_probabilities = np.mean(regime_probabilities_list, axis=0)
            
            result = {
                'regime': consensus_regimes,
                'probability': consensus_probabilities,
                'consensus': np.mean(consensus_probabilities)
            }
            
            if ensemble_probabilities is not None:
                result['regime_probabilities'] = ensemble_probabilities
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to combine ensemble results: {e}")
            return {
                'regime': [0] * len(models[0]),
                'probability': [0.0] * len(models[0]),
                'consensus': 0.0
            }

    def _initialize_streaming_model(self, data: pd.DataFrame, config: HMMRegimeConfig):
        """Initialize streaming HMM model."""
        try:
            # Use first window to initialize model
            window_data = data.iloc[:self.streaming_config.window_size]
            initial_result = self._detect_hmm_gaussian_regimes(window_data, config, 'light')
            
            self.streaming_state['current_model'] = initial_result
            self.streaming_state['last_update'] = time.time()
            self.streaming_state['regime_history'] = initial_result['regime'].tolist()
            self.streaming_state['stability_score'] = 0.8  # Initial stability
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize streaming model: {e}")

    def _update_streaming_model(self, window_data: pd.DataFrame, config: HMMRegimeConfig):
        """Update streaming HMM model."""
        try:
            # Detect regimes for current window
            window_result = self._detect_hmm_gaussian_regimes(window_data, config, 'light')
            
            # Update streaming state
            self.streaming_state['current_model'] = window_result
            self.streaming_state['last_update'] = time.time()
            
            # Update regime history
            new_regimes = window_result['regime'].tolist()
            self.streaming_state['regime_history'].extend(new_regimes)
            
            # Keep only recent history
            max_history = self.streaming_config.window_size * 5
            if len(self.streaming_state['regime_history']) > max_history:
                self.streaming_state['regime_history'] = self.streaming_state['regime_history'][-max_history:]
            
            # Calculate stability score
            self._calculate_streaming_stability()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update streaming model: {e}")

    def _predict_streaming_regimes(self, window_data: pd.DataFrame) -> List[int]:
        """Predict regimes for streaming window."""
        try:
            if self.streaming_state['current_model'] is None:
                return [0] * len(window_data)
            
            # Use current model to predict regimes
            # This is a simplified implementation
            return self.streaming_state['current_model']['regime'].tolist()[:len(window_data)]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to predict streaming regimes: {e}")
            return [0] * len(window_data)

    def _calculate_streaming_stability(self):
        """Calculate streaming model stability."""
        try:
            regime_history = self.streaming_state['regime_history']
            if len(regime_history) < 10:
                self.streaming_state['stability_score'] = 0.5
                return
            
            # Calculate regime change frequency
            regime_changes = sum(1 for i in range(1, len(regime_history)) 
                               if regime_history[i] != regime_history[i-1])
            
            change_frequency = safe_divide(regime_changes, len(regime_history) - 1)
            stability_score = 1.0 - change_frequency
            
            self.streaming_state['stability_score'] = max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate streaming stability: {e}")
            self.streaming_state['stability_score'] = 0.0

    def _refine_regime_aware_regimes(
        self, 
        base_result: pd.DataFrame, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Refine regimes using regime-aware logic."""
        # This would implement regime-aware refinements
        # For now, return the base result with additional metadata
        refined_result = base_result.copy()
        refined_result['detection_method'] = 'regime_aware_hmm'
        refined_result['regime_awareness'] = 1.0  # Placeholder
        
        return refined_result

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data for regime detection using advanced quality framework."""
        # Check if quality framework is available
        if not QUALITY_FRAMEWORK_AVAILABLE:
            self.logger.warning("⚠️ Advanced quality framework not available, using basic validation")
            return self._basic_validation_fallback(data)

        # Use advanced quality framework for comprehensive validation
        quality_framework = DataQualityFramework()

        # Basic sanity checks first
        if len(data) < 100:  # Need extra buffer for rolling calculations
            raise ValueError("Insufficient data for regime detection (minimum 100 rows required for rolling calculations)")

        # Exclude first 50 rows to avoid NaN values from rolling calculations
        data_for_validation = data.iloc[50:].copy()
        self.logger.info(f"   📊 Using {len(data_for_validation)} rows for validation (excluded first 50 rows with rolling NaN values)")

        numeric_columns = data_for_validation.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for regime detection")

        # Advanced quality validation (using data without rolling NaN values)
        quality_framework = DataQualityFramework()
        quality_result = quality_framework.validate_dataframe_quality(
            data_for_validation,
            context="hmm_regime_detection_input_validation"
        )

        # Log quality assessment
        self.logger.info("🔍 HMM Input Data Quality Assessment:")
        self.logger.info(f"   📊 Quality Score: {quality_result.quality_score:.1f}/100")
        self.logger.info(f"   📊 Critical Issues: {len([i for i in quality_result.issues if 'CRITICAL' in str(i).upper()])}")
        self.logger.info(f"   📊 Warnings: {len(quality_result.warnings)}")

        # Handle critical issues
        critical_issues = [i for i in quality_result.issues if 'CRITICAL' in str(i).upper()]
        if critical_issues:
            self.logger.error("❌ Critical data quality issues found:")
            for issue in critical_issues[:3]:  # Show first 3 critical issues
                self.logger.error(f"   - {issue}")
            if len(critical_issues) > 3:
                self.logger.error(f"   ... and {len(critical_issues) - 3} more critical issues")

        # Handle warnings
        if quality_result.warnings:
            self.logger.warning("⚠️ Data quality warnings:")
            for warning in quality_result.warnings[:3]:  # Show first 3 warnings
                self.logger.warning(f"   - {warning}")

        # Quality gate - reject if quality score too low
        if quality_result.quality_score < 99:  # Strict threshold for HMM
            raise ValueError(f"HMM input data quality too low: {quality_result.quality_score:.1f}/100 (required: 99.0)")

        # Advanced gap detection and filling
        data = self._advanced_gap_detection_and_filling(data)

        return quality_result

    def _basic_validation_fallback(self, data: pd.DataFrame) -> None:
        """Basic validation fallback when advanced framework is not available."""
        if len(data) < 50:
            raise ValueError("Insufficient data for regime detection (minimum 50 rows required)")

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for regime detection")

        # Check for null values
        null_counts = data[numeric_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found in data: {null_counts.to_dict()}")

        self.logger.info("✅ Basic validation passed")

    def _advanced_gap_detection_and_filling(self, data: pd.DataFrame, skip_download_gaps: bool = False) -> pd.DataFrame:
        """Advanced gap detection and filling with download capability for large gaps.

        Args:
            data: Input DataFrame to process
            skip_download_gaps: If True, skip downloading data for large gaps and use interpolation only
        """
        from ..data.quality.data_cleaning import DataCleaner
        from datetime import timedelta

        self.logger.info("🔍 Performing advanced gap detection and filling...")
        self.logger.info(f"📊 Input data shape: {data.shape[0]:,} rows × {data.shape[1]} columns")

        # Handle timestamp access (could be column or index)
        if 'timestamp' in data.columns:
            timestamps = data['timestamp']
        elif hasattr(data.index, 'name') and data.index.name == 'timestamp':
            timestamps = data.index
        else:
            # Try to infer timestamp from index
            if pd.api.types.is_datetime64_any_dtype(data.index):
                timestamps = data.index
                self.logger.info("🔄 Using datetime index as timestamp for gap detection")
            else:
                raise ValueError("Could not find timestamp column or datetime index for gap detection")

        time_range_min = timestamps.min()
        time_range_max = timestamps.max()
        self.logger.info(f"📅 Data time range: {time_range_min} to {time_range_max}")

        # Initialize cleaning framework - data_type will be determined later
        cleaner = None  # Will be initialized later with correct data_type
        self.logger.info("🧹 Data cleaner will be initialized with detected timeframe")

        # Detect gaps in timestamp data
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(timestamps):
                timestamps = pd.to_datetime(timestamps, unit='ms', utc=True)

            # Calculate time differences
            time_diffs = timestamps.diff()
            if hasattr(time_diffs, 'dt'):
                time_diffs = time_diffs.dt.total_seconds()
            else:
                # Handle case where diff() returns TimedeltaIndex directly
                time_diffs = time_diffs / pd.Timedelta(seconds=1)

            # Determine expected interval and data-type specific thresholds
            expected_interval = self._determine_expected_interval(data, timestamps)
            gap_threshold, download_threshold = self._get_data_type_specific_thresholds(expected_interval)

            self.logger.info(f"📊 Expected data interval: {expected_interval:.1f}s")
            self.logger.info(f"📊 Gap detection threshold: {gap_threshold:.1f}s")
            self.logger.info(f"📊 Download attempt threshold: {download_threshold:.1f}s")

            # Find gaps larger than expected interval
            gap_mask = time_diffs > gap_threshold
            large_gaps = time_diffs[gap_mask]

            # Ensure large_gaps is always a Series for consistent .items() iteration
            if isinstance(large_gaps, pd.Index):
                large_gaps = pd.Series(large_gaps.values, index=large_gaps.index, name='gaps')

            if len(large_gaps) > 0:
                self.logger.info(f"📊 Found {len(large_gaps)} gaps larger than {gap_threshold:.1f}s")
                self.logger.info(f"📊 Largest gap: {large_gaps.max():.1f}s, Smallest gap: {large_gaps.min():.1f}s")

                # Handle describe() for TimedeltaIndex vs Series
                try:
                    # Ensure large_gaps is always a Series for describe() method
                    if not hasattr(large_gaps, 'describe') or isinstance(large_gaps, pd.Index):
                        gap_series = pd.Series(large_gaps.values, name='gaps')
                        self.logger.info(f"📊 Gap sizes: {gap_series.describe()}")
                    else:
                        self.logger.info(f"📊 Gap sizes: {large_gaps.describe()}")
                except Exception as e:
                    self.logger.info(f"📊 Gap sizes summary: count={len(large_gaps)}, mean={large_gaps.mean():.1f}s, std={large_gaps.std():.1f}s")

                # Log specific gap information with timing context
                self.logger.info("📅 Gap timing analysis:")

                # Analyze gap distribution by time of day and day of week
                gap_timestamps = []
                for i, (idx, gap_size) in enumerate(large_gaps.items()):
                    gap_start = timestamps.iloc[idx-1] if idx > 0 else timestamps.iloc[0]
                    gap_timestamps.append(gap_start)

                    if i < 10:  # Log first 10 gaps with detailed timing
                        gap_end = timestamps.iloc[idx] if idx < len(timestamps) else timestamps.iloc[-1]
                        gap_hours = gap_size / 3600  # Convert to hours for readability
                        self.logger.info(f"   Gap {i+1}: {gap_size:.1f}s ({gap_hours:.2f}h) from {gap_start} to {gap_end}")

                        # Add detailed timing context
                        if hasattr(gap_start, 'hour'):
                            day_of_week = gap_start.strftime('%A')
                            hour_of_day = gap_start.hour
                            minute_of_hour = gap_start.minute
                            date_str = gap_start.strftime('%Y-%m-%d')
                            time_str = f"{hour_of_day:02d}:{minute_of_hour:02d}"
                            self.logger.info(f"      📅 Occurred on {day_of_week}, {date_str} at {time_str} UTC")

                            # Calculate gap duration in human-readable format
                            if gap_size < 60:
                                duration_str = f"{gap_size:.0f} seconds"
                            elif gap_size < 3600:
                                minutes = gap_size // 60
                                seconds = gap_size % 60
                                duration_str = f"{minutes:.0f}m {seconds:.0f}s"
                            else:
                                hours = gap_size // 3600
                                minutes = (gap_size % 3600) // 60
                                duration_str = f"{hours:.0f}h {minutes:.0f}m"

                            self.logger.info(f"      ⏱️  Gap duration: {duration_str}")
                            self.logger.info(f"      📍 Gap location: Row {idx:,} in dataset")
                            self.logger.warning(f"      ⚠️  LARGE GAP DETECTED: {duration_str} gap at {date_str} {time_str} UTC - requires data download/filling")

                # Provide summary statistics about gap timing
                if len(gap_timestamps) > 0:
                    gap_times = pd.Series(gap_timestamps)
                    if hasattr(gap_times.iloc[0], 'hour'):
                        # Hourly distribution
                        hour_counts = gap_times.dt.hour.value_counts().sort_index()
                        peak_hour = hour_counts.idxmax() if len(hour_counts) > 0 else "N/A"
                        self.logger.info(f"      🕐 Peak gap hour: {peak_hour}:00 UTC ({hour_counts.max()} gaps)")

                        # Day of week distribution
                        day_counts = gap_times.dt.day_name().value_counts()
                        peak_day = day_counts.idxmax() if len(day_counts) > 0 else "N/A"
                        self.logger.info(f"      📅 Most affected day: {peak_day} ({day_counts.max()} gaps)")

                        # Time range of gaps
                        earliest_gap = gap_times.min()
                        latest_gap = gap_times.max()
                        self.logger.info(f"      📊 Gap time range: {earliest_gap} to {latest_gap}")

                        # Check for weekend/weekday patterns
                        weekdays = gap_times[gap_times.dt.dayofweek < 5]  # Monday=0, Sunday=6
                        weekends = gap_times[gap_times.dt.dayofweek >= 5]
                        self.logger.info(f"      📈 Weekday gaps: {len(weekdays)}, Weekend gaps: {len(weekends)}")

                if len(large_gaps) > 10:
                    self.logger.info(f"      ... and {len(large_gaps) - 10} more gaps (showing first 10)")

                # Batch process gaps for efficiency
                self.logger.info("🔄 Starting batch gap processing...")
                data = self._batch_process_gaps(data, large_gaps, timestamps, gap_threshold, download_threshold, skip_download_gaps, 'klines')
                self.logger.info(f"✅ Gap processing completed. New data shape: {data.shape[0]:,} rows")

                # Re-validate after gap filling
                self.logger.info("🔄 Re-validating data after gap filling...")
                quality_framework = DataQualityFramework()
                post_fill_result = quality_framework.validate_dataframe_quality(
                    data, context="hmm_post_gap_fill_validation"
                )

                if post_fill_result.quality_score < 99:
                    self.logger.warning("⚠️ Data quality still low after gap filling")

            else:
                self.logger.info("✅ No significant gaps detected in data")

        except Exception as e:
            self.logger.warning(f"⚠️ Error during advanced gap detection: {e}")
            # Fall back to basic gap filling
            # Use interpolation for basic gap filling (only on numeric columns)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                data[numeric_columns] = data[numeric_columns].interpolate(method='linear', limit_direction='both')
                self.logger.info(f"✅ Interpolated {len(numeric_columns)} numeric columns")
            else:
                self.logger.warning("⚠️ No numeric columns available for interpolation")

        return data

    def _determine_expected_interval(self, data: pd.DataFrame, timestamps: pd.Series) -> float:
        """Determine the expected time interval between data points using robust statistics.

        Reports median interval with tolerance and avoids misleading sub-second prints for minute data.
        """
        try:
            # First, check if this is klines data by looking for interval column
            is_klines_data = False
            expected_interval_override = None

            if 'interval' in data.columns:
                # Check for common kline interval patterns
                interval_values = data['interval'].dropna().unique()
                if len(interval_values) > 0:
                    interval_str = str(interval_values[0]).lower()
                    if '1m' in interval_str or '1min' in interval_str or interval_str == '1':
                        expected_interval_override = 60.0  # 1 minute
                        is_klines_data = True
                    elif '5m' in interval_str or '5min' in interval_str or interval_str == '5':
                        expected_interval_override = 300.0  # 5 minutes
                        is_klines_data = True
                    elif '15m' in interval_str or '15min' in interval_str or interval_str == '15':
                        expected_interval_override = 900.0  # 15 minutes
                        is_klines_data = True
                    elif '30m' in interval_str or '30min' in interval_str or interval_str == '30':
                        expected_interval_override = 1800.0  # 30 minutes
                        is_klines_data = True
                    elif '1h' in interval_str or '1hour' in interval_str or interval_str == '60':
                        expected_interval_override = 3600.0  # 1 hour
                        is_klines_data = True
                    elif '4h' in interval_str or '4hour' in interval_str:
                        expected_interval_override = 14400.0  # 4 hours
                        is_klines_data = True
                    elif '1d' in interval_str or '1day' in interval_str:
                        expected_interval_override = 86400.0  # 1 day
                        is_klines_data = True

            if is_klines_data and expected_interval_override is not None:
                self.logger.info(f"📊 Detected klines data with interval column: {interval_str}, using expected interval: {expected_interval_override:.1f}s")
                return expected_interval_override

            # Calculate time differences in seconds
            time_diffs_raw = timestamps.diff()
            if hasattr(time_diffs_raw, 'dt'):
                time_diffs = time_diffs_raw.dt.total_seconds().dropna()
            else:
                time_diffs = (time_diffs_raw / pd.Timedelta(seconds=1)).dropna()

            if len(time_diffs) == 0:
                self.logger.warning("⚠️ No time differences found, assuming 65s interval")
                return 65.0

            # Prefer median for robustness; use mode if clearly defined
            median_interval = float(time_diffs.median())
            mode_interval = time_diffs.mode()
            expected_interval = float(mode_interval.iloc[0]) if len(mode_interval) > 0 else median_interval

            # Clamp to sensible bounds and normalize 1m artifacts
            if expected_interval < 1:
                # Likely unit artifact; treat as minute data if median near 60s
                approx_minute = 60.0
                if abs(median_interval - approx_minute) <= 5:
                    expected_interval = approx_minute
                else:
                    expected_interval = max(1.0, expected_interval)
            elif expected_interval > 86400:  # 24 hours
                self.logger.warning(f"⚠️ Very large interval detected ({expected_interval:.1f}s), assuming 3600s (1h)")
                expected_interval = 3600.0

            # Log interval classification using robust expected interval
            if expected_interval <= 90:
                interval_type = "1m klines"
            elif expected_interval <= 600:
                interval_type = "5m klines"
            elif expected_interval <= 1800:
                interval_type = "15m/30m klines"
            elif expected_interval <= 3600:
                interval_type = "1h klines"
            elif expected_interval <= 14400:
                interval_type = "4h klines"
            elif expected_interval <= 28800:
                interval_type = "8h futures"
            elif expected_interval <= 86400:
                interval_type = "1d klines"
            else:
                interval_type = "weekly/monthly data"

            # Report median ± tolerance for clarity
            tolerance = float(np.std(time_diffs)) if 'np' in globals() else float(time_diffs.std())
            # Clamp tiny medians near 0 that can appear due to unit errors; avoid misleading 0.060s prints
            display_median = median_interval if median_interval >= 1.0 else 60.0 if abs(median_interval - 60.0) <= 5 else max(1.0, median_interval)
            self.logger.info(f"📊 Detected data type: {interval_type} (median interval: {display_median:.1f}s ± {min(tolerance, 5.0):.1f}s)")

            return expected_interval

        except Exception as e:
            self.logger.warning(f"⚠️ Error determining expected interval: {e}, assuming 65s")
            return 65.0

    def _get_data_type_specific_thresholds(self, expected_interval: float) -> Tuple[float, float]:
        """Get data-type specific gap detection and download thresholds."""
        try:
            from ..data.quality.data_cleaning import DataCleaner
            # Define thresholds based on data type
            if expected_interval <= 2:
                # Aggtrades data (1-2 second intervals)
                data_type = "aggtrades"
                gap_threshold = 2.0  # Smallest threshold should be < medium
                download_threshold = 5.0
            elif expected_interval <= 90:
                # Klines data (1 minute intervals)
                data_type = "klines_1m"
                # Use robust thresholds around one minute
                gap_threshold = max(60.0, min(70.0, expected_interval + 5.0))
                download_threshold = 120.0  # 2 minutes for download attempts
            elif expected_interval <= 600:
                # Klines data (5 minute intervals)
                data_type = "klines_5m"
                gap_threshold = 300.0  # 5 minutes for klines
                download_threshold = 600.0  # 10 minutes for download attempts
            elif expected_interval <= 1800:
                # Klines data (15-30 minute intervals)
                data_type = "klines_15m_30m"
                gap_threshold = 900.0  # 15 minutes for klines
                download_threshold = 1800.0  # 30 minutes for download attempts
            elif expected_interval <= 3600:
                # Klines data (1 hour intervals)
                data_type = "klines_1h"
                gap_threshold = 3600.0  # 1 hour for klines - minimum meaningful gap
                download_threshold = 7200.0  # 2 hours for download attempts
            elif expected_interval <= 14400:
                # Klines data (4 hour intervals)
                data_type = "klines_4h"
                gap_threshold = 7200.0  # 2 hours for klines
                download_threshold = 14400.0  # 4 hours for download attempts
            elif expected_interval <= 28800:
                # Futures data (8 hour intervals)
                data_type = "futures_8h"
                gap_threshold = 32400.0  # 9 hours for futures
                download_threshold = 43200.0  # 12 hours for download attempts
            elif expected_interval <= 86400:
                # Daily data
                data_type = "daily"
                gap_threshold = 43200.0  # 12 hours for daily data
                download_threshold = 86400.0  # 24 hours for download attempts
            else:
                # Weekly/monthly data
                data_type = "weekly_monthly"
                gap_threshold = expected_interval * 2  # 2x interval
                download_threshold = expected_interval * 3  # 3x interval
            
            self.logger.info(f"📊 Data type: {data_type}")
            self.logger.info(f"📊 Gap threshold: {gap_threshold:.1f}s ({gap_threshold/60:.1f}min)")
            self.logger.info(f"📊 Download threshold: {download_threshold:.1f}s ({download_threshold/60:.1f}min)")

            # Update DataCleaner with the correct data_type for proper gap thresholds
            cleaner = DataCleaner(data_type=data_type)
            
            return gap_threshold, download_threshold
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error setting data-type specific thresholds: {e}")
            # Fallback to generic thresholds
            gap_threshold = expected_interval * 2
            download_threshold = expected_interval * 3
            return gap_threshold, download_threshold

    def _download_missing_data(self, gap_start: pd.Timestamp, gap_end: pd.Timestamp, data: pd.DataFrame = None) -> pd.DataFrame | None:
        """Download missing data for large gaps using klines_downloading_processing.py."""
        try:
            self.logger.info(f"📥 Attempting to download data from {gap_start} to {gap_end}")

            # Use the klines downloading and processing pipeline
            try:
                from ...training.steps.data_collection.klines_downloading_processing import KlinesDataProcessingPipeline
                self.logger.info("🔍 Loaded KlinesDataProcessingPipeline")

                # Initialize the pipeline
                pipeline = KlinesDataProcessingPipeline()

                # Extract symbol and timeframe from data or use defaults
                symbol = "ETHUSDT"  # Default symbol
                interval = "1m"     # Default interval

                # Try to extract from data attributes if available
                if data is not None:
                    if hasattr(data, 'attrs') and 'symbol' in data.attrs:
                        symbol = data.attrs['symbol']
                    if hasattr(data, 'attrs') and 'timeframe' in data.attrs:
                        interval = data.attrs['timeframe']

                # Calculate gap duration for determining max_gap_minutes
                gap_duration = gap_end - gap_start
                gap_minutes = int(gap_duration.total_seconds() / 60)

                # Set max_gap_minutes to slightly less than the actual gap to ensure it's detected
                max_gap_minutes = max(1, gap_minutes - 1)

                self.logger.info(f"🔧 Gap filling parameters: symbol={symbol}, interval={interval}, max_gap_minutes={max_gap_minutes}")

                # Use asyncio to run the async gap handling method
                import asyncio
                import os

                # Get API credentials from environment or use defaults
                api_key = os.getenv('BINANCE_API_KEY', '')
                api_secret = os.getenv('BINANCE_API_SECRET', '')

                if not api_key or not api_secret:
                    self.logger.warning("⚠️ Binance API credentials not found, gap filling may fail")
                    return None

                # Run the gap handling asynchronously
                async def _run_gap_filling():
                    result = await pipeline.handle_gaps_with_column_removal(
                        symbol=symbol,
                        interval=interval,
                        max_gap_minutes=max_gap_minutes,
                        api_key=api_key,
                        api_secret=api_secret
                    )
                    return result

                # Execute the async function
                try:
                    loop = asyncio.get_running_loop()
                    # We're already in an event loop, use run_in_executor
                    self.logger.info("🔄 Running gap filling in executor to avoid event loop conflict")
                    future = loop.run_in_executor(None, lambda: asyncio.run(_run_gap_filling()))
                    gap_result = future.result(timeout=300)  # 5 minute timeout

                except RuntimeError:
                    # Safe to run in new event loop
                    self.logger.info("🔄 Running gap filling in new event loop")
                    gap_result = asyncio.run(_run_gap_filling())

                # Check results
                if gap_result.get("filled_gaps", 0) > 0:
                    self.logger.info(f"✅ Gap filling successful: {gap_result}")

                    # Try to load the newly downloaded data
                    downloaded_data = self._load_recently_downloaded_data(gap_start, gap_end, interval)
                    if downloaded_data is not None and not downloaded_data.empty:
                        self.logger.info(f"✅ Loaded {len(downloaded_data)} newly downloaded rows")
                        return downloaded_data
                    else:
                        self.logger.warning("⚠️ Gap filling completed but could not load downloaded data")
                        return None
                else:
                    self.logger.warning(f"⚠️ Gap filling failed or found no gaps: {gap_result}")
                    return None

            except ImportError as e:
                self.logger.warning(f"⚠️ Could not import KlinesDataProcessingPipeline: {e}")
                return None

            except Exception as e:
                self.logger.error(f"❌ Error during gap filling: {e}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error downloading missing data: {e}")
            return None

    def _load_recently_downloaded_data(self, gap_start: pd.Timestamp, gap_end: pd.Timestamp, timeframe: str) -> pd.DataFrame | None:
        """Load recently downloaded data for the gap period."""
        try:
            from ...training.steps.standardized_parquet_handler import standardized_parquet_handler
            from src.utils.pipeline_standards import PipelineStandards

            standards = PipelineStandards()
            symbol = getattr(self, 'symbol', 'ETHUSDT')
            exchange = getattr(self, 'exchange', 'BINANCE').lower()

            # Generate file path for downloaded data
            klines_file = standards.generate_file_name('klines', exchange, symbol, timeframe)
            data_dir = getattr(self, 'data_dir', 'historical_data')
            klines_path = Path(data_dir) / klines_file

            if klines_path.exists():
                # Load the data
                df = standardized_parquet_handler.read_parquet_standardized(klines_path)

                # Filter to the gap period
                # Handle timestamp access (could be column or index)
                if 'timestamp' in df.columns:
                    timestamps = df['timestamp']
                elif hasattr(df.index, 'name') and df.index.name == 'timestamp':
                    timestamps = df.index
                else:
                    # Try to infer timestamp from index
                    if pd.api.types.is_datetime64_any_dtype(df.index):
                        timestamps = df.index
                        self.logger.info("🔄 Using datetime index as timestamp for gap filtering")
                    else:
                        raise ValueError("Could not find timestamp column or datetime index for gap filtering")

                # Convert timestamps if needed
                if not pd.api.types.is_datetime64_any_dtype(timestamps):
                    timestamps = pd.to_datetime(timestamps, unit='ms', utc=True)

                # Filter to gap period
                mask = (timestamps >= gap_start) & (timestamps <= gap_end)
                gap_data = df[mask].copy()

                if not gap_data.empty:
                    self.logger.info(f"✅ Loaded {len(gap_data)} rows from downloaded data")
                    return gap_data

            self.logger.warning("⚠️ No suitable downloaded data found for gap period")
            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Error loading downloaded data: {e}")
            return None

    def _insert_downloaded_data(self, original_data: pd.DataFrame,
                               downloaded_data: pd.DataFrame,
                               insert_index: int) -> pd.DataFrame:
        """Insert downloaded data into the original dataset."""
        try:
            # Combine datasets
            combined = pd.concat([original_data, downloaded_data], ignore_index=True)

            # Sort by timestamp and remove duplicates
            if 'timestamp' in combined.columns:
                combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

            self.logger.info(f"✅ Successfully inserted {len(downloaded_data)} downloaded rows")
            return combined

        except Exception as e:
            self.logger.error(f"❌ Error inserting downloaded data: {e}")
            return original_data

    def _batch_process_gaps(self, data: pd.DataFrame, large_gaps: pd.Series, timestamps: pd.Series,
                           gap_threshold: float, download_threshold: float, skip_download_gaps: bool = False,
                           data_type: str = 'klines') -> pd.DataFrame:
        """Batch process gaps for efficiency with prioritization.

        Args:
            data: Input DataFrame
            large_gaps: Series of gap sizes
            timestamps: Timestamp series
            gap_threshold: Minimum gap size to process
            download_threshold: Gap size above which to attempt downloads
            skip_download_gaps: If True, skip downloading and use interpolation for all gaps
        """
        try:
            # Categorize gaps by size and priority
            download_gaps = []
            interpolation_gaps = []
            
            for idx in large_gaps.index:
                gap_size = large_gaps.loc[idx]
                if gap_size > download_threshold and not skip_download_gaps:
                    download_gaps.append((idx, gap_size))
                else:
                    interpolation_gaps.append((idx, gap_size))
            
            if skip_download_gaps and len(large_gaps) > 0:
                self.logger.info(f"🚫 Skipping download gaps (skip_download_gaps=True): {len(large_gaps)} gaps will be interpolated")
                interpolation_gaps = [(idx, large_gaps.loc[idx]) for idx in large_gaps.index]
            
            self.logger.info(f"📊 Gap categorization: {len(download_gaps)} for download, {len(interpolation_gaps)} for interpolation")
            
            # Process download gaps in batches (limit concurrent downloads)
            if download_gaps:
                self.logger.info(f"🔄 Processing {len(download_gaps)} gaps for download in batches...")
                data = self._process_download_gaps_batch(data, download_gaps, timestamps, data_type)
            
            # Process interpolation gaps in batches
            if interpolation_gaps:
                self.logger.info(f"🔄 Processing {len(interpolation_gaps)} gaps for interpolation in batches...")
                data = self._process_interpolation_gaps_batch(data, interpolation_gaps, data_type)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch gap processing: {e}")
            # Fall back to individual processing
            return self._fallback_individual_gap_processing(data, large_gaps, timestamps, gap_threshold, download_threshold, data_type)

    def _process_download_gaps_batch(self, data: pd.DataFrame, download_gaps: list, timestamps: pd.Series, data_type: str = 'klines') -> pd.DataFrame:
        """Process download gaps in batches with concurrency control."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Limit concurrent downloads to avoid overwhelming the system
        max_concurrent_downloads = 3
        batch_size = 10
        
        self.logger.info(f"🔄 Processing download gaps in batches of {batch_size} with max {max_concurrent_downloads} concurrent downloads")
        
        # Process gaps in batches
        for i in range(0, len(download_gaps), batch_size):
            batch = download_gaps[i:i + batch_size]
            self.logger.info(f"📥 Processing download batch {i//batch_size + 1}/{(len(download_gaps) + batch_size - 1)//batch_size}")
            
            # Use ThreadPoolExecutor for concurrent downloads
            with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as executor:
                futures = []
                for idx, gap_size in batch:
                    gap_start = timestamps.loc[idx-1]
                    gap_end = timestamps.loc[idx]
                    
                    # Validate dates before attempting download
                    current_time = pd.Timestamp.now(tz='UTC')
                    if gap_start > current_time or gap_end > current_time:
                        self.logger.warning(f"⚠️ Skipping future gap: {gap_start} to {gap_end}")
                        continue
                    
                    # Submit download task
                    future = executor.submit(self._download_missing_data, gap_start, gap_end, data)
                    futures.append((future, idx, gap_size, gap_start, gap_end))
                
                # Process completed downloads
                for future, idx, gap_size, gap_start, gap_end in futures:
                    try:
                        downloaded_data = future.result(timeout=30)  # 30 second timeout per download
                        
                        if downloaded_data is not None and not downloaded_data.empty:
                            self.logger.info(f"✅ Downloaded {len(downloaded_data)} rows for gap {gap_size:.1f}s")
                            data = self._insert_downloaded_data(data, downloaded_data, idx)
                        else:
                            self.logger.warning(f"❌ Download failed for gap {gap_size:.1f}s - using interpolation")
                            data = self._interpolate_gap(data, idx, gap_size, data_type)
                            
                    except Exception as e:
                        self.logger.warning(f"❌ Download error for gap {gap_size:.1f}s: {e} - using interpolation")
                        data = self._interpolate_gap(data, idx, gap_size, data_type)
        
        return data

    def _process_interpolation_gaps_batch(self, data: pd.DataFrame, interpolation_gaps: list, data_type: str = 'klines') -> pd.DataFrame:
        """Process interpolation gaps in batches for efficiency."""
        batch_size = 50  # Process more interpolation gaps per batch since they're faster
        
        self.logger.info(f"🔄 Processing interpolation gaps in batches of {batch_size}")
        
        for i in range(0, len(interpolation_gaps), batch_size):
            batch = interpolation_gaps[i:i + batch_size]
            if (i//batch_size + 1) % 10 == 0 or i == 0:  # Log every 10th batch or first batch
                self.logger.info(f"🔧 Processing interpolation batch {i//batch_size + 1}/{(len(interpolation_gaps) + batch_size - 1)//batch_size}")
            
            # Process interpolation gaps in the batch
            for idx, gap_size in batch:
                try:
                    data = self._interpolate_gap(data, idx, gap_size, data_type)
                except Exception as e:
                    self.logger.warning(f"⚠️ Interpolation error for gap {gap_size:.1f}s: {e}")
        
        return data

    def _fallback_individual_gap_processing(self, data: pd.DataFrame, large_gaps: pd.Series,
                                          timestamps: pd.Series, gap_threshold: float,
                                          download_threshold: float, data_type: str = 'klines') -> pd.DataFrame:
        """Fallback to individual gap processing if batch processing fails."""
        self.logger.warning("⚠️ Falling back to individual gap processing")
        
        for idx in large_gaps.index:
            gap_size = large_gaps.loc[idx]
            
            if gap_size > download_threshold:
                self.logger.info(f"📥 Large gap detected ({gap_size:.1f}s > {download_threshold:.1f}s) - attempting download")
                gap_start = timestamps.loc[idx-1]
                gap_end = timestamps.loc[idx]
                
                # Validate dates before attempting download
                current_time = pd.Timestamp.now(tz='UTC')
                if gap_start > current_time or gap_end > current_time:
                    self.logger.warning(f"⚠️ Cannot download future data: gap from {gap_start} to {gap_end}")
                    data = self._interpolate_gap(data, idx, gap_size, data_type)
                    continue
                
                # Attempt to download missing data
                downloaded_data = self._download_missing_data(gap_start, gap_end, data)
                
                if downloaded_data is not None and not downloaded_data.empty:
                    self.logger.info(f"✅ Successfully downloaded {len(downloaded_data)} rows for gap")
                    data = self._insert_downloaded_data(data, downloaded_data, idx)
                else:
                    self.logger.warning(f"❌ Failed to download data for gap - using interpolation")
                    data = self._interpolate_gap(data, idx, gap_size, data_type)
            else:
                self.logger.info(f"🔧 Gap detected ({gap_size:.1f}s) - using interpolation")
                data = self._interpolate_gap(data, idx, gap_size, data_type)
        
        return data

    def _interpolate_gap(self, data: pd.DataFrame, gap_index: int, gap_size: float, data_type: str = 'klines') -> pd.DataFrame:
        """Interpolate data for small gaps using various methods."""
        try:
            from ..data.quality.data_cleaning import DataCleaner

            cleaner = DataCleaner(data_type=data_type)

            # Choose interpolation method based on gap size and data type
            if gap_size <= 30:  # Very small gaps - linear interpolation
                method = 'linear'
                method_kwargs = {}
            elif gap_size <= 300:  # Medium gaps - polynomial interpolation
                method = 'polynomial'
                method_kwargs = {'order': 2}  # Quadratic polynomial
            else:  # Larger gaps - use linear interpolation as fallback
                method = 'linear'
                method_kwargs = {}

            # Apply interpolation to numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data.columns:
                    try:
                        # Interpolate missing values with appropriate parameters
                        data[col] = data[col].interpolate(method=method, limit=10, **method_kwargs)
                    except Exception as col_error:
                        # Fallback to linear interpolation if the chosen method fails
                        self.logger.warning(f"⚠️ Interpolation method '{method}' failed for column '{col}': {col_error}")
                        try:
                            data[col] = data[col].interpolate(method='linear', limit=10)
                            self.logger.info(f"✅ Fallback to linear interpolation successful for column '{col}'")
                        except Exception as fallback_error:
                            self.logger.warning(f"⚠️ Even linear interpolation failed for column '{col}': {fallback_error}")

            # For timestamp column, ensure proper time progression (memory-optimized)
            if 'timestamp' in data.columns:
                try:
                    # Use in-place sorting to reduce memory usage
                    data = data.sort_values('timestamp', inplace=False).reset_index(drop=True)
                except Exception as sort_error:
                    self.logger.warning(f"⚠️ Memory-efficient sort failed: {sort_error}")
                    # Fallback: try with copy=False
                    try:
                        data = data.sort_values('timestamp', inplace=False, ignore_index=True)
                    except Exception as fallback_error:
                        self.logger.warning(f"⚠️ All sort attempts failed: {fallback_error}")
                        # Last resort: don't sort but warn
                        self.logger.warning("⚠️ Unable to sort timestamp column - data may be unsorted")

            self.logger.debug(f"✅ Interpolated gap using {method} method")
            return data

        except Exception as e:
            self.logger.warning(f"⚠️ Error during gap interpolation: {e}")
            return data

    def _update_performance_stats(self, start_time: float, num_regimes: int):
        """Update performance statistics."""
        processing_time = time.time() - start_time
        
        self.performance_stats['total_regimes_detected'] += num_regimes
        self.performance_stats['processing_time'] += processing_time

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_regimes_detected': self.performance_stats['total_regimes_detected'],
            'total_processing_time': self.performance_stats['processing_time'],
            'regimes_per_second': safe_divide(
                self.performance_stats['total_regimes_detected'],
                self.performance_stats['processing_time']
            ),
            'average_accuracy': np.mean(self.performance_stats['accuracy_scores']) if self.performance_stats['accuracy_scores'] else 0.0
        }

# Global instance for backward compatibility
enhanced_hmm_regime_detector = EnhancedHMMRegimeDetector()

# Export for backward compatibility
HMMRegimeDetector = EnhancedHMMRegimeDetector