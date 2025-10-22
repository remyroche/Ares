"""
Consolidated Volatility-Aware Multi-Horizon Profit Labeler

This module consolidates the functionality from both volatility_aware_profit_labeler.py
and enhanced_multi_horizon_labeler.py into a single, comprehensive implementation.

Key Features:
1. Volatility-normalized targets (k·σ_t) instead of fixed percentages
2. Data-driven horizons via first-passage time quantiles
3. Multi-target labeling (small/medium/high) with separate optimization
4. Noise gating to filter microstructure-dominated periods
5. Label quality optimization for ML learnability (AUC, stability, balance, SNR)
6. Enhanced data cleaning and quality assessment
7. Trading-aware label definitions (Analyst & Tactician)
8. Label stability monitoring and leakage detection
9. Full backward compatibility with existing pipeline

Author: AI Assistant
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
import time
import hashlib
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from abc import ABC

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Core utilities
from src.utils.logger import get_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview, tprint_data_format
from src.utils.math_validation import safe_divide, validate_finite
from src.core.decorators import handles_errors, traced, validates

# Matrix operations for performance
from src.utils.matrix_operations import UnifiedMatrixOperations
from src.feature_generation.utils.enhanced_matrix_operations import EnhancedMatrixOperations

# Enhanced utilities integration
from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager, StorageConfig
from src.utils.serialization_utils import UniversalSerializer, safe_serialize, safe_deserialize
from src.utils.hardware.hardware_optimizer import HardwareOptimizer
from src.utils.ml_common.feature_selection import FeatureSelector
from src.utils.data.processing.transformers import DataTransformer

# Phase 1 optimization tools
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.memory_management import MemoryManager, MemoryManagerConfig, MemoryStrategy
    PHASE1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    PHASE1_OPTIMIZATIONS_AVAILABLE = False

# Statistical and ML utilities
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt, argrelextrema
# import xgboost as xgb  # Removed unused import
from sklearn.linear_model import LogisticRegression

# Import enhanced data and labels system
from .enhanced_data_labels_system import (
    EnhancedDataLabelsSystem, EnhancedDataLabelsConfig,
    TradingObjectiveConfig, LabelStabilityConfig, DataCleaningConfig,
    create_trading_optimized_config, create_research_optimized_config
)

@dataclass
class ConsolidatedLabelerConfig:
    """
    Consolidated configuration for volatility-aware profit labeling.
    
    Combines the best features from both VolatilityAwareConfig and EnhancedMultiHorizonConfig.
    """
    
    # Base timeframe for analysis
    base_timeframe_minutes: int = 5

    # Volatility modeling parameters
    rv_window_minutes: int = 30        # Rolling volatility window
    atr_window_bars: int = 14          # ATR window in bars
    volatility_ewma_lambda: float = 0.94  # EWMA smoothing for volatility

    # Multi-target configuration (small/medium/high)
    target_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'small': (0.4, 0.8),    # k ∈ [0.4, 0.8]
        'medium': (0.8, 1.3),   # k ∈ [0.8, 1.3]
        'high': (1.3, 2.0)      # k ∈ [1.3, 2.0]
    })

    # First-passage time quantile for horizon selection
    fpt_quantile: float = 0.65  # Q_0.65 of historical FPT

    # Noise gating parameters
    micro_range_alpha: float = 1.5     # k·σ_t ≥ α·mTR_t
    variance_ratio_threshold: float = 1.2  # VR threshold for microstructure filter
    liquidity_percentile: float = 10.0     # Minimum volume percentile
    spread_filter_enabled: bool = True

    # Label quality constraints
    min_positive_balance: float = 0.35    # Minimum 35% positive class
    max_positive_balance: float = 0.65    # Maximum 65% positive class
    min_aic_threshold: float = 0.55       # Minimum AUC for acceptance
    max_auc_std_threshold: float = 0.08   # Maximum AUC standard deviation

    # Hysteresis and conflict resolution
    hysteresis_bars: int = 2
    flip_override_beta: float = 0.3

    # Optimization parameters
    search_grid_k: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])
    search_grid_quantile: List[float] = field(default_factory=lambda: [0.5, 0.65, 0.8])
    search_grid_alpha: List[float] = field(default_factory=lambda: [1.2, 1.5, 1.8])

    # Cross-target correlation constraint
    max_target_correlation: float = 0.6

    # Processing parameters
    min_bars_for_labeling: int = 50
    max_horizon_bars: int = 100
    outlier_cap_percentile: float = 99.9

    # Quality scoring weights
    lqs_weights: Dict[str, float] = field(default_factory=lambda: {
        'predictability': 0.3,
        'stability': 0.25,
        'balance': 0.2,
        'snr': 0.15,
        'consistency': 0.1
    })

    # Enhanced features
    enable_enhanced_data_cleaning: bool = True
    enable_enhanced_stability_monitoring: bool = True
    enable_trading_aware_labels: bool = True
    enable_peak_trough_detection: bool = True
    
    # Phase 1 optimization settings
    enable_vectorbt_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    vectorbt_chunk_size: int = 1000
    memory_limit_gb: float = 2.0
    cache_size_mb: int = 100
    
    # Enhanced label definitions
    analyst_config: Optional[Dict[str, Any]] = None
    tactician_config: Optional[Dict[str, Any]] = None
    
    # Label stability monitoring
    stability_check_config: Optional[Dict[str, Any]] = None
    
    # Data quality assessment
    data_quality_config: Optional[Dict[str, Any]] = None
    
    # Peak/trough detection parameters
    peak_detection_method: str = "find_peaks"  # "find_peaks", "find_peaks_cwt", "argrelextrema"
    peak_prominence: float = 0.001  # Minimum prominence for peak detection (as fraction of price)
    peak_distance: int = 5  # Minimum distance between peaks (in bars)
    peak_width: Optional[Tuple[int, int]] = None  # (min_width, max_width) for peak width filtering
    peak_height_threshold: Optional[float] = None  # Minimum height threshold
    smoothing_window: int = 3  # Window for smoothing before peak detection
    use_relative_extrema: bool = True  # Use relative extrema instead of absolute peaks

@dataclass
class LabelQualityMetrics:
    """Container for label quality metrics."""

    # Core KPIs
    auc_mean: float = 0.0
    auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0

    # Stability metrics
    psi_score: float = 0.0  # Population Stability Index
    flip_rate: float = 0.0  # Label flip rate within windows

    # Balance metrics
    positive_balance: float = 0.0
    class_balance_score: float = 0.0  # How close to 50/50

    # Signal-to-noise metrics
    feature_ic_mean: float = 0.0  # Spearman correlation with features
    mutual_information: float = 0.0  # MI with adjacent horizons

    # Composite score
    label_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for reporting."""
        return {
            'auc_mean': self.auc_mean,
            'auc_std': self.auc_std,
            'pr_auc_mean': self.pr_auc_mean,
            'pr_auc_std': self.pr_auc_std,
            'psi_score': self.psi_score,
            'flip_rate': self.flip_rate,
            'positive_balance': self.positive_balance,
            'class_balance_score': self.class_balance_score,
            'feature_ic_mean': self.feature_ic_mean,
            'mutual_information': self.mutual_information,
            'label_quality_score': self.label_quality_score
        }

@dataclass
class LabelingResult:
    """Result container for labeling operations."""
    
    # Core results
    labels: pd.DataFrame
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    
    # Quality metrics
    quality_metrics: Dict[str, LabelQualityMetrics]
    overall_quality_score: float
    
    # Metadata
    config_used: ConsolidatedLabelerConfig
    processing_time: float
    n_samples: int
    n_targets: int
    timestamp: datetime = field(default_factory=datetime.now)

class ConsolidatedProfitLabeler(BaseStep):
    """
    Consolidated volatility-aware multi-horizon profit labeler.
    
    This class combines the best features from both VolatilityAwareProfitLabeler
    and EnhancedMultiHorizonProfitLabeler into a single, comprehensive implementation.
    """

    def __init__(self, config: Optional[ConsolidatedLabelerConfig] = None):
        """Initialize the consolidated labeler."""
        super().__init__("consolidated_profit_labeler")
        self.config = config or ConsolidatedLabelerConfig()
        self.logger = get_logger('ConsolidatedProfitLabeler')

        # Initialize matrix operations
        self.matrix_ops = UnifiedMatrixOperations()
        self.enhanced_ops = EnhancedMatrixOperations()

        # Initialize enhanced utilities
        self.vectorization_manager = UnifiedVectorizationManager(
            VectorizationConfig(
                enable_vectorization=True,
                vectorization_method="numpy",
                batch_size=1000,
                enable_parallel_processing=True,
                enable_optimization=True
            )
        )
        
        # Initialize Bayesian TPE optimizer for hyperparameter tuning
        self.tpe_optimizer = BayesianTPEOptimizer(
            n_trials=100,
            n_startup_trials=20,
            n_warmup_steps=50
        )
        
        # Initialize data storage and serialization
        self.klines_manager = KlinesParquetManager(
            StorageConfig(
                compression="zstd",
                compression_level=3,
                enable_metadata=True,
                enable_validation=True
            )
        )
        self.serializer = UniversalSerializer()
        
        # Initialize hardware optimizer
        self.hardware_optimizer = HardwareOptimizer()
        
        # Initialize feature selector
        self.feature_selector = FeatureSelector()
        
        # Initialize data transformer
        self.data_transformer = DataTransformer()
        
        # Initialize Phase 1 optimization tools
        if PHASE1_OPTIMIZATIONS_AVAILABLE and self.config.enable_vectorbt_optimization:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=self.config.vectorbt_chunk_size,
                fast_fail=True
            )
            tprint_info("   → VectorBTRollingOptimizer: Available")
        else:
            self.vectorbt_optimizer = None
            tprint_warning("   → VectorBTRollingOptimizer: Not available")
        
        if PHASE1_OPTIMIZATIONS_AVAILABLE and self.config.enable_memory_optimization:
            self.memory_optimizer = M1MemoryOptimizer(
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.cpu_optimizer = M1CPUOptimizer()
            memory_config = MemoryManagerConfig(
                strategy=MemoryStrategy.MODERATE,
                enable_monitoring=True,
                memory_threshold_mb=self.config.memory_limit_gb * 1024 * 0.8,
                max_memory_mb=self.config.memory_limit_gb * 1024
            )
            self.memory_manager = MemoryManager(memory_config)
            tprint_info("   → Hardware optimizations: Available")
        else:
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.memory_manager = None
            tprint_warning("   → Hardware optimizations: Not available")
        
        # Initialize caching for repeated calculations
        self._calculation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._performance_metrics = {
            'vectorbt_operations': 0,
            'cached_operations': 0,
            'memory_optimizations': 0,
            'labels_generated': 0,
            'total_time': 0.0
        }

        # Initialize enhanced data and labels system if enabled
        self.enhanced_system = None
        if self.config.enable_enhanced_data_cleaning or self.config.enable_trading_aware_labels:
            # Create proper EnhancedDataLabelsConfig with correct fields
            enhanced_config = EnhancedDataLabelsConfig(
                trading_objective=TradingObjectiveConfig(
                    primary_objective="risk_adjusted_returns",
                    enable_regime_conditioning=True
                ),
                label_definitions={
                    'analyst_config': self.config.analyst_config,
                    'tactician_config': self.config.tactician_config,
                    'enable_trading_aware': self.config.enable_trading_aware_labels
                },
                data_cleaning=DataCleaningConfig(
                    enable_cleaning=self.config.enable_enhanced_data_cleaning
                ),
                label_stability=LabelStabilityConfig(
                    enable_monitoring=self.config.enable_enhanced_stability_monitoring
                )
            )
            self.enhanced_system = EnhancedDataLabelsSystem(config=enhanced_config)

        # Cache for expensive calculations
        self._volatility_cache: Dict[str, pd.Series] = {}
        self._fpt_cache: Dict[str, np.ndarray] = {}
        self._quality_cache: Dict[str, LabelQualityMetrics] = {}
        
        # Performance tracking
        self._performance_metrics = {
            'vectorization_time': 0.0,
            'optimization_time': 0.0,
            'serialization_time': 0.0,
            'hardware_optimization_time': 0.0
        }

        self.tprint("🔧 Initialized Consolidated Profit Labeler with Enhanced Utilities")
        tprint_info(f"📊 Config: {len(self.config.target_bands)} target bands, "
                   f"RV window: {self.config.rv_window_minutes}min, "
                   f"Enhanced features: {self.config.enable_enhanced_data_cleaning}")
        tprint_info(f"🚀 Enhanced utilities: Vectorization, TPE Optimization, Hardware acceleration")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the consolidated profit labeling step.
        
        Args:
            config: Configuration dictionary containing:
                - data: DataFrame with OHLCV data
                - symbol: Optional symbol for context
                - exchange: Optional exchange for context
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - labeling_result: LabelingResult object
                - artifacts: List of generated artifacts
        """
        try:
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract data from config
            data = config.get('data')
            
            if data is None:
                return {
                    'success': False,
                    'error': 'Missing required parameter: data'
                }
            
            # Validate inputs
            if not isinstance(data, pd.DataFrame):
                return {
                    'success': False,
                    'error': 'data must be a pandas DataFrame'
                }
            
            # Preview input data
            self.tprint_data_preview(data, "input_data", max_rows=5)
            self.tprint_data_format(data, "input_data")
            
            # Apply hardware optimization to input data
            if self.hardware_utils and self.hardware_utils.get('optimize_dataframe'):
                data = self.hardware_utils['optimize_dataframe'](data)
                self.tprint_info("🔧 Input data optimized for hardware acceleration")
            
            # Check for cached results first
            cache_key = self._generate_cache_key(data, config)
            cached_result = self._load_cached_result(cache_key)
            
            if cached_result and not config.get('force_recompute', False):
                self.tprint_info("📦 Using cached labeling result")
                labeling_result = cached_result
            else:
                # Generate labels with enhanced processing
                labeling_result = self.generate_labels(data)
                
                # Cache the result for future use
                self._cache_result(cache_key, labeling_result)
            
            # Save artifacts
            artifacts = []
            
            # Apply hardware optimization to labeled data
            if self.hardware_optimizer:
                labeling_result.labels = self.hardware_optimizer.optimize_dataframe(labeling_result.labels)
                self.tprint_info("🔧 Labeled data optimized for hardware acceleration")
            
            # Preview labeled data
            self.tprint_data_preview(labeling_result.labels, "consolidated_labeled_data", max_rows=5)
            self.tprint_data_format(labeling_result.labels, "consolidated_labeled_data")
            
            # Save labeled data using enhanced serialization
            labeled_data_path = self._save_dataframe_enhanced(
                labeling_result.labels, 
                'consolidated_labeled_data'
            )
            if labeled_data_path:
                artifacts.append(labeled_data_path)
            
            # Save quality metrics with enhanced serialization
            quality_data = {
                'overall_quality_score': labeling_result.overall_quality_score,
                'quality_metrics': {k: v.to_dict() for k, v in labeling_result.quality_metrics.items()},
                'processing_time': labeling_result.processing_time,
                'n_samples': labeling_result.n_samples,
                'n_targets': labeling_result.n_targets,
                'performance_metrics': self._performance_metrics,
                'cache_key': cache_key,
                'timestamp': datetime.now().isoformat()
            }
            
            # Preview quality data
            self.tprint_data_format(quality_data, "consolidated_quality_metrics")
            
            # Save quality data using enhanced serialization
            quality_path = self._save_metadata_enhanced(
                quality_data, 
                'consolidated_quality_metrics'
            )
            if quality_path:
                artifacts.append(quality_path)
            
            # Log metrics
            self.tprint_metrics({
                'input_samples': len(data),
                'labeled_samples': len(labeling_result.labels),
                'quality_score': labeling_result.overall_quality_score,
                'target_columns': len([col for col in labeling_result.labels.columns if 'target' in col.lower()])
            }, "consolidated_labeling_metrics")
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(labeling_result, artifacts)
            self._save_outcome_file(outcome_content, 'consolidated_labeling_outcome')
            
            return {
                'success': True,
                'labeling_result': labeling_result,
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Consolidated labeling failed: {str(e)}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, labeling_result: LabelingResult, artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Consolidated Profit Labeling Outcome

## Summary
- **Status**: Success
- **Samples Processed**: {labeling_result.n_samples}
- **Targets Generated**: {labeling_result.n_targets}
- **Overall Quality Score**: {labeling_result.overall_quality_score:.3f}
- **Processing Time**: {labeling_result.processing_time:.2f} seconds
- **Artifacts Generated**: {len(artifacts)}

## Data Overview
- **Columns**: {list(labeling_result.labels.columns)}
- **Memory Usage**: {labeling_result.labels.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## Quality Metrics by Target
"""
        
        for target_name, metrics in labeling_result.quality_metrics.items():
            content += f"""
### {target_name}
- **AUC Mean**: {metrics.auc_mean:.3f} ± {metrics.auc_std:.3f}
- **PR-AUC Mean**: {metrics.pr_auc_mean:.3f} ± {metrics.pr_auc_std:.3f}
- **Positive Balance**: {metrics.positive_balance:.3f}
- **Class Balance Score**: {metrics.class_balance_score:.3f}
- **Label Quality Score**: {metrics.label_quality_score:.3f}
"""
        
        content += f"""
## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}

## Configuration
- **Target Bands**: {len(self.config.target_bands)}
- **RV Window**: {self.config.rv_window_minutes} minutes
- **ATR Window**: {self.config.atr_window_bars} bars
- **Enhanced Features**: {self.config.enable_enhanced_data_cleaning}
- **Trading-Aware Labels**: {self.config.enable_trading_aware_labels}
"""
        
        return content

    def _generate_cache_key(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Generate a cache key for the labeling operation."""
        import hashlib
        
        # Create a hash based on data characteristics and config
        data_hash = hashlib.md5(
            f"{len(data)}_{data.index[0]}_{data.index[-1]}_{data.shape}".encode()
        ).hexdigest()[:16]
        
        config_hash = hashlib.md5(
            str(sorted(config.items())).encode()
        ).hexdigest()[:16]
        
        return f"labeling_{data_hash}_{config_hash}"

    def _load_cached_result(self, cache_key: str) -> Optional[LabelingResult]:
        """Load cached labeling result."""
        try:
            cache_path = f"cache/{cache_key}.pkl"
            if os.path.exists(cache_path):
                cached_data = safe_deserialize(cache_path)
                if cached_data and isinstance(cached_data, dict):
                    # Reconstruct LabelingResult from cached data
                    tprint_data_preview(cached_data, "cached_labeling_result")
                    return LabelingResult(**cached_data)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load cached result: {e}")
        return None

    def _cache_result(self, cache_key: str, result: LabelingResult) -> None:
        """Cache labeling result."""
        try:
            os.makedirs("cache", exist_ok=True)
            cache_path = f"cache/{cache_key}.pkl"
            
            # Convert result to dictionary for serialization
            result_dict = {
                'labels': result.labels,
                'confidence_scores': result.confidence_scores,
                'eligibility_masks': result.eligibility_masks,
                'quality_metrics': result.quality_metrics,
                'overall_quality_score': result.overall_quality_score,
                'config_used': result.config_used,
                'processing_time': result.processing_time,
                'n_samples': result.n_samples,
                'n_targets': result.n_targets,
                'timestamp': result.timestamp
            }
            
            tprint_data_format(result_dict, "labeling_result_cache")
            safe_serialize(result_dict, cache_path, format='pickle')
            tprint_info(f"💾 Cached result: {cache_path}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to cache result: {e}")

    def _save_dataframe_enhanced(self, df: pd.DataFrame, name: str) -> Optional[str]:
        """Save DataFrame using enhanced serialization."""
        try:
            # Use KlinesParquetManager for efficient storage
            if hasattr(self, 'klines_manager'):
                # Create a temporary DataFrame with required columns
                temp_df = df.copy()
                if 'timestamp' not in temp_df.columns and hasattr(temp_df.index, 'to_pydatetime'):
                    temp_df['timestamp'] = temp_df.index
                
                # Use parquet for better compression and performance
                success = self.klines_manager.store_klines(
                    temp_df, 
                    symbol="temp", 
                    exchange="cache", 
                    interval="1m",
                    batch_id=name
                )
                
                if success:
                    return f"cache/temp/cache/klines/{name}.parquet"
            
            # Fallback to regular DataFrame saving
            return self._save_dataframe(df, name)
            
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced DataFrame save failed: {e}")
            return self._save_dataframe(df, name)

    def _save_metadata_enhanced(self, metadata: Dict[str, Any], name: str) -> Optional[str]:
        """Save metadata using enhanced serialization."""
        try:
            # Use enhanced serialization with compression
            metadata_path = f"artifacts/{name}.json"
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            # Add performance metrics
            metadata['performance_metrics'] = self._performance_metrics
            metadata['vectorization_config'] = self.vectorization_manager.get_performance_metrics()
            
            success = safe_serialize(metadata, metadata_path, format='json')
            if success:
                return metadata_path
            else:
                return self._save_metadata(metadata, name)
                
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced metadata save failed: {e}")
            return self._save_metadata(metadata, name)

    def generate_labels(self, data: pd.DataFrame) -> LabelingResult:
        """
        Generate consolidated volatility-aware multi-horizon labels.

        Args:
            data: OHLCV DataFrame with proper columns

        Returns:
            LabelingResult with comprehensive labeling data
        """
        start_time = datetime.now()
        tprint_data_preview(data, "input_data")
        self.tprint("🚀 Starting consolidated profit labeling...")

        # Check cache first if caching is enabled
        if self.config.enable_caching:
            cache_key = f"labels_{hash(data.values.tobytes())}_{self.config.base_timeframe_minutes}"
            if cache_key in self._calculation_cache:
                self._cache_hits += 1
                self._performance_metrics['cached_operations'] += 1
                self.tprint("💾 Using cached labeling result")
                return self._calculation_cache[cache_key]

        if len(data) < self.config.min_bars_for_labeling:
            self.tprint(f"⚠️ Insufficient data: {len(data)} < {self.config.min_bars_for_labeling}")
            return LabelingResult(
                labels=pd.DataFrame(),
                confidence_scores=pd.DataFrame(),
                eligibility_masks=pd.DataFrame(),
                quality_metrics={},
                overall_quality_score=0.0,
                config_used=self.config,
                processing_time=0.0,
                n_samples=0,
                n_targets=0
            )

        # Step 0: Data preparation and cleaning
        data_clean = self._prepare_and_clean_data(data)
        
        # Apply memory optimization if available
        if self.memory_optimizer and self.config.enable_memory_optimization:
            data_clean = self.memory_optimizer.optimize_dataframe(data_clean)
            self._performance_metrics['memory_optimizations'] += 1
            self.tprint("🧠 Applied memory optimization to data")

        # Step 1: Enhanced data cleaning if enabled
        if self.config.enable_enhanced_data_cleaning and self.enhanced_system:
            data_clean = self._apply_enhanced_data_cleaning(data_clean)

        # Step 2: Volatility modeling
        volatility_data = self._compute_volatility_series(data_clean)

        # Step 3: Noise gating
        noise_gates = self._compute_noise_gates(data_clean, volatility_data)

        # Step 4: Multi-target configuration optimization
        optimal_configs = self._optimize_target_configurations(
            data_clean, volatility_data, noise_gates
        )

        # Step 5: Generate labels for each target
        labeled_data = data_clean.copy()
        quality_metrics = {}
        confidence_scores = pd.DataFrame(index=data_clean.index)
        eligibility_masks = pd.DataFrame(index=data_clean.index)

        for target_name, config in optimal_configs.items():
            self.tprint(f"🎯 Generating {target_name} target labels...")

            target_result = self._generate_single_target_labels(
                data_clean, volatility_data, noise_gates, config, target_name
            )

            # Merge into main dataframe
            for col in target_result.columns:
                if col not in labeled_data.columns:
                    labeled_data[col] = target_result[col]

            # Store confidence scores and eligibility masks
            if 'confidence' in target_result.columns:
                confidence_scores[f'{target_name}_confidence'] = target_result['confidence']
            if 'eligibility' in target_result.columns:
                eligibility_masks[f'{target_name}_eligibility'] = target_result['eligibility']

            # Compute quality metrics
            quality_metrics[target_name] = self._compute_label_quality(
                target_result, data_clean, target_name
            )

        # Step 6: Enhanced stability monitoring if enabled
        if self.config.enable_enhanced_stability_monitoring and self.enhanced_system:
            stability_results = self._apply_enhanced_stability_monitoring(
                labeled_data, quality_metrics
            )
            # Merge stability results into quality metrics
            for target_name, stability_data in stability_results.items():
                if target_name in quality_metrics:
                    quality_metrics[target_name].mutual_information = stability_data.get('mutual_information', 0.0)

        # Step 7: Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality_score(quality_metrics)

        # Step 8: Create final result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = LabelingResult(
            labels=labeled_data,
            confidence_scores=confidence_scores,
            eligibility_masks=eligibility_masks,
            quality_metrics=quality_metrics,
            overall_quality_score=overall_quality_score,
            config_used=self.config,
            processing_time=processing_time,
            n_samples=len(labeled_data),
            n_targets=len(optimal_configs)
        )

        self.tprint("✅ Consolidated labeling completed")
        tprint_info(f"📊 Generated labels for {len(optimal_configs)} targets")
        tprint_info(f"📊 Overall quality score: {overall_quality_score:.3f}")
        tprint_data_preview(result.labels, "final_labeled_data")
        tprint_data_format(result, "final_labeling_result")

        # Store result in cache if caching is enabled
        if self.config.enable_caching:
            cache_key = f"labels_{hash(data.values.tobytes())}_{self.config.base_timeframe_minutes}"
            self._calculation_cache[cache_key] = result
            self._cache_misses += 1

        return result

    def _prepare_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Step 0: Clean and prepare data for labeling with enhanced utilities."""
        self.tprint("🔍 Preparing and cleaning data with enhanced utilities...")

        # Apply hardware optimization to input data
        if self.hardware_optimizer:
            data = self.hardware_optimizer.optimize_dataframe(data)
            self.tprint_info("🔧 Applied hardware optimization to input data")

        # Use data transformer for comprehensive cleaning
        data_clean = self.data_transformer.transform(
            data,
            operations=['outlier_detection', 'missing_value_handling', 'data_validation']
        )

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data_clean.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Add volume column if missing (for compatibility)
        if 'volume' not in data_clean.columns:
            data_clean['volume'] = 1000.0  # Default volume for compatibility

        # Use vectorized operations for better performance
        data_clean = self.vectorization_manager.vectorize_data(data_clean)
        
        # Compute returns with enhanced outlier handling
        data_clean['returns'] = data_clean['close'].pct_change()

        # Use feature selector for outlier detection
        outlier_mask = self.feature_selector.detect_outliers(
            data_clean['returns'].dropna(),
            method='isolation_forest'
        )
        
        # Cap extreme returns using percentile-based approach
        return_cap = np.percentile(data_clean['returns'].dropna(), self.config.outlier_cap_percentile)
        data_clean['returns'] = np.clip(data_clean['returns'], -return_cap, return_cap)

        # Compute true range for microstructure filtering using vectorized operations
        data_clean['true_range'] = np.maximum(
            data_clean['high'] - data_clean['low'],
            np.maximum(
                abs(data_clean['high'] - data_clean['close'].shift(1)),
                abs(data_clean['low'] - data_clean['close'].shift(1))
            )
        )

        # Enhanced missing value handling
        data_clean = self.data_transformer.handle_missing_values(
            data_clean, 
            strategy='forward_fill_interpolation'
        )

        # Apply final data validation
        data_clean = self.data_transformer.validate_data_integrity(data_clean)

        self.tprint(f"✅ Data prepared with enhanced utilities: {len(data_clean)} bars, {data_clean.shape[1]} columns")
        return data_clean

    def _apply_enhanced_data_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced data cleaning using the enhanced data and labels system."""
        if not self.enhanced_system:
            return data
        
        try:
            # Use the enhanced system's process_market_data method
            result = self.enhanced_system.process_market_data(
                market_data=data,
                force_recompute=True
            )
            if result.get('success', True) and 'processed_data' in result:
                cleaned_data = result['processed_data']
                self.tprint("✅ Enhanced data cleaning applied")
                return cleaned_data
            else:
                tprint_warning(f"⚠️ Enhanced data cleaning failed: {result.get('error', 'Unknown error')}")
                return data
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced data cleaning failed: {e}")
            return data

    def _apply_enhanced_stability_monitoring(self, labeled_data: pd.DataFrame, 
                                           quality_metrics: Dict[str, LabelQualityMetrics]) -> Dict[str, Dict[str, float]]:
        """Apply enhanced stability monitoring using the enhanced data and labels system."""
        if not self.enhanced_system:
            return {}
        
        try:
            # Use the enhanced system's process_market_data method to get stability metrics
            result = self.enhanced_system.process_market_data(
                market_data=labeled_data,
                force_recompute=True
            )
            if result.get('success', True) and 'stability_metrics' in result:
                stability_results = result['stability_metrics']
                self.tprint("✅ Enhanced stability monitoring applied")
                return stability_results
            else:
                tprint_warning(f"⚠️ Enhanced stability monitoring failed: {result.get('error', 'Unknown error')}")
                return {}
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced stability monitoring failed: {e}")
            return {}

    def _compute_volatility_series(self, data: pd.DataFrame) -> pd.Series:
        """Step 1: Compute volatility series using enhanced methods with optimization."""
        self.tprint("📊 Computing volatility series with enhanced optimization...")

        # Use vectorized operations for better performance
        data_vectorized = self.vectorization_manager.vectorize_data(data)
        
        # Ensure returns are computed
        if 'returns' not in data_vectorized.columns:
            data_vectorized = data_vectorized.copy()
            data_vectorized['returns'] = data_vectorized['close'].pct_change()

        # Use optimized rolling operations
        returns = data_vectorized['returns'].dropna()
        if len(returns) < self.config.rv_window_minutes:
            # Fallback for short data
            rv_window = min(len(returns), 20)
        else:
            rv_window = self.config.rv_window_minutes

        # Enhanced realized volatility computation with better numerical stability
        rolling_rv = returns.rolling(window=rv_window, min_periods=rv_window//2).std()
        rolling_rv = rolling_rv * np.sqrt(rv_window)  # Scale to return units

        # ATR (Average True Range) - ensure it exists
        if 'true_range' not in data_vectorized.columns:
            data_vectorized['true_range'] = np.maximum(
                data_vectorized['high'] - data_vectorized['low'],
                np.maximum(
                    abs(data_vectorized['high'] - data_vectorized['close'].shift(1)),
                    abs(data_vectorized['low'] - data_vectorized['close'].shift(1))
                )
            )

        # Use optimized ATR computation
        atr = data_vectorized['true_range'].rolling(
            window=self.config.atr_window_bars, 
            min_periods=self.config.atr_window_bars//2
        ).mean()
        
        # Convert ATR to return-scale by normalizing by price
        atr_returns = atr / data_vectorized['close']

        # Enhanced volatility combination using weighted approach
        # Use adaptive weights based on data characteristics
        rv_weight = 0.6  # Higher weight for realized volatility
        atr_weight = 0.4  # Lower weight for ATR
        
        # Check for data quality and adjust weights
        rv_quality = 1.0 - (rolling_rv.isnull().sum() / len(rolling_rv))
        atr_quality = 1.0 - (atr_returns.isnull().sum() / len(atr_returns))
        
        if rv_quality < 0.8:
            rv_weight = 0.3
            atr_weight = 0.7
        elif atr_quality < 0.8:
            rv_weight = 0.8
            atr_weight = 0.2

        combined_vol = (rv_weight * rolling_rv + atr_weight * atr_returns)

        # Apply EWMA smoothing with optimized parameters
        vol_ewma = combined_vol.ewm(
            alpha=1-self.config.volatility_ewma_lambda,
            min_periods=10
        ).mean()

        # Enhanced flooring using percentile-based approach
        vol_percentile_5 = vol_ewma.quantile(0.05)
        vol_floor = max(1e-6, vol_percentile_5 * 0.1)  # 10% of 5th percentile
        vol_ewma = vol_ewma.clip(lower=vol_floor)

        # Apply hardware optimization if available
        if self.hardware_optimizer:
            vol_ewma = self.hardware_optimizer.optimize_series(vol_ewma)

        self.tprint(f"✅ Enhanced volatility computed: mean={vol_ewma.mean():.6f}, "
               f"std={vol_ewma.std():.6f}, range=[{vol_ewma.min():.6f}, {vol_ewma.max():.6f}]")
        self.tprint(f"   → RV weight: {rv_weight:.2f}, ATR weight: {atr_weight:.2f}")
        self.tprint(f"   → Data quality - RV: {rv_quality:.2f}, ATR: {atr_quality:.2f}")

        return vol_ewma

    def _compute_noise_gates(self, data: pd.DataFrame, volatility: pd.Series) -> Dict[str, pd.Series]:
        """Step 2: Compute noise gates to filter microstructure effects."""
        self.tprint("🔇 Computing noise gates...")

        # Ensure required columns exist
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()

        if 'true_range' not in data.columns:
            data['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )

        gates = {}

        # 1. Micro-range gate: k·σ_t ≥ α·mTR_t (median true range)
        median_tr = data['true_range'].rolling(window=20).median()
        # We'll compute this per target when we have k values

        # 2. Variance ratio test for microstructure
        def compute_variance_ratio(returns: pd.Series, m: int = 5) -> pd.Series:
            """Compute variance ratio VR = Var(r_Δ) / (m·Var(r_Δ/m))."""
            if len(returns) < 2*m:
                return pd.Series(1.0, index=returns.index)

            # Compute returns at different scales
            r_delta = returns.diff(m).dropna()
            r_delta_m = returns.diff(1).dropna()

            # Align indices for proper division
            aligned_idx = r_delta.index.intersection(r_delta_m.index[:len(r_delta)])
            r_delta = r_delta.loc[aligned_idx]
            r_delta_m = r_delta_m.loc[aligned_idx]

            var_delta = r_delta.var()
            var_delta_m = r_delta_m.var()

            if var_delta_m > 0:
                vr = var_delta / (m * var_delta_m)
            else:
                vr = 1.0

            return pd.Series(vr, index=aligned_idx)

        # Vectorized variance ratio computation
        def compute_variance_ratio_vectorized(returns: pd.Series, m: int = 5) -> pd.Series:
            """Vectorized variance ratio computation."""
            if len(returns) < 2*m:
                return pd.Series(1.0, index=returns.index)
            
            # Compute returns at different scales
            r_delta = returns.diff(m).dropna()
            r_delta_m = returns.diff(1).dropna()
            
            # Align indices
            aligned_idx = r_delta.index.intersection(r_delta_m.index[:len(r_delta)])
            if len(aligned_idx) == 0:
                return pd.Series(1.0, index=returns.index)
            
            r_delta = r_delta.loc[aligned_idx]
            r_delta_m = r_delta_m.loc[aligned_idx]
            
            # Compute variances
            var_delta = r_delta.var()
            var_delta_m = r_delta_m.var()
            
            if var_delta_m > 0:
                vr = var_delta / (m * var_delta_m)
            else:
                vr = 1.0
            
            return pd.Series(vr, index=aligned_idx)
        
        # Compute variance ratio for the entire series
        vr_series = compute_variance_ratio_vectorized(data['returns'])
        gates['variance_ratio'] = vr_series.reindex(data.index, fill_value=1.0)

        # 3. Liquidity gate: volume percentile filter
        rolling_volume_pct = data['volume'].rolling(window=50).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50.0
        )
        gates['liquidity_gate'] = rolling_volume_pct >= self.config.liquidity_percentile

        # 4. Spread filter: ultra-tight ranges
        relative_spread = (data['high'] - data['low']) / data['close'].shift(1)
        median_spread = relative_spread.rolling(window=20).median()
        gates['spread_filter'] = relative_spread >= (median_spread * 0.5)  # Not ultra-tight

        # Combined eligibility gate (all filters must pass)
        gates['eligibility'] = (
            gates['liquidity_gate'] &
            gates['spread_filter'] &
            (gates['variance_ratio'] >= 0.8)  # Not dominated by microstructure
        )

        self.tprint(f"✅ Noise gates computed: {gates['eligibility'].mean():.1%} eligible bars")
        return gates

    def _optimize_target_configurations(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step 3: Optimize target configurations using Bayesian TPE optimization.

        This uses Tree-structured Parzen Estimator for efficient hyperparameter search
        within each band to maximize label quality, then filters for cross-target correlation constraints.
        """
        self.tprint("🔍 Optimizing target configurations with Bayesian TPE...")

        # Step 1: Find best config for each target using TPE optimization
        per_target_configs = {}

        for target_name, (k_min, k_max) in self.config.target_bands.items():
            self.tprint(f"🎯 Optimizing {target_name} target (k ∈ [{k_min}, {k_max}]) with TPE...")

            # Define search space for TPE optimization
            search_space = {
                'k': (k_min, k_max),
                'fpt_quantile': (0.5, 0.8),
                'micro_range_alpha': (1.0, 2.0)
            }

            # Define objective function for TPE
            def objective(trial):
                config = {
                    'k': trial.suggest_float('k', k_min, k_max),
                    'fpt_quantile': trial.suggest_float('fpt_quantile', 0.5, 0.8),
                    'micro_range_alpha': trial.suggest_float('micro_range_alpha', 1.0, 2.0)
                }

                try:
                    quality_metrics = self._evaluate_config_quality(
                        data, volatility, noise_gates, config, target_name
                    )
                    return quality_metrics.label_quality_score
                except Exception as e:
                    tprint_warning(f"⚠️ TPE trial failed: {e}")
                    return 0.0

            # Run TPE optimization
            try:
                best_trial = self.tpe_optimizer.optimize(
                    objective=objective,
                    search_space=search_space,
                    n_trials=50,  # Reduced for efficiency
                    timeout=300   # 5 minutes timeout
                )

                if best_trial and best_trial.value > 0:
                    best_config = {
                        'k': best_trial.params['k'],
                        'fpt_quantile': best_trial.params['fpt_quantile'],
                        'micro_range_alpha': best_trial.params['micro_range_alpha']
                    }
                    
                    # Get final quality metrics
                    quality_metrics = self._evaluate_config_quality(
                        data, volatility, noise_gates, best_config, target_name
                    )
                    best_config['quality_metrics'] = quality_metrics
                    
                    per_target_configs[target_name] = best_config
                    self.tprint(f"✅ TPE optimized {target_name}: k={best_config['k']:.3f}, "
                           f"q={best_config['fpt_quantile']:.3f}, "
                           f"LQS={quality_metrics.label_quality_score:.3f}")
                else:
                    raise ValueError("TPE optimization failed")
                    
            except Exception as e:
                tprint_warning(f"⚠️ TPE optimization failed for {target_name}: {e}")
                # Fallback to grid search
                best_config = self._fallback_grid_search(
                    data, volatility, noise_gates, target_name, k_min, k_max
                )
                per_target_configs[target_name] = best_config
                self.tprint(f"⚠️ Using fallback grid search for {target_name}")

        # Step 2: Filter configurations based on cross-target correlations
        optimal_configs = self._filter_by_correlation(per_target_configs, data, volatility, noise_gates)

        return optimal_configs

    def _fallback_grid_search(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series],
        target_name: str,
        k_min: float,
        k_max: float
    ) -> Dict[str, Any]:
        """Fallback grid search when TPE optimization fails."""
        best_config = None
        best_lqs = -np.inf

        # Generate candidate k values (grid search within band)
        k_candidates = [k for k in self.config.search_grid_k if k_min <= k <= k_max]
        k_candidates.extend(np.linspace(k_min, k_max, 5))

        for k in sorted(set(k_candidates)):
            for q in self.config.search_grid_quantile:
                for alpha in self.config.search_grid_alpha:
                    config = {
                        'k': k,
                        'fpt_quantile': q,
                        'micro_range_alpha': alpha
                    }

                    try:
                        quality_metrics = self._evaluate_config_quality(
                            data, volatility, noise_gates, config, target_name
                        )

                        if quality_metrics.label_quality_score > best_lqs:
                            best_lqs = quality_metrics.label_quality_score
                            best_config = config.copy()
                            best_config['quality_metrics'] = quality_metrics

                    except Exception as e:
                        continue

        if best_config:
            return best_config
        else:
            # Ultimate fallback
            return {
                'k': (k_min + k_max) / 2,
                'fpt_quantile': self.config.fpt_quantile,
                'micro_range_alpha': self.config.micro_range_alpha,
                'quality_metrics': LabelQualityMetrics()
            }

    def _filter_by_correlation(
        self,
        per_target_configs: Dict[str, Dict[str, Any]],
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter configurations to ensure targets are not too correlated."""

        self.tprint("🔗 Filtering configurations by cross-target correlation...")

        if len(per_target_configs) <= 1:
            return per_target_configs

        # Generate labels for all targets to compute correlations
        target_labels = {}
        for target_name, config in per_target_configs.items():
            labels = self._generate_single_target_labels(
                data, volatility, noise_gates, config, target_name, max_samples=2000
            )
            target_labels[target_name] = labels['target']

        # Compute correlation matrix
        label_series = pd.DataFrame(target_labels)

        # Only consider valid labels (non-zero)
        valid_mask = (label_series != 0).any(axis=1)
        if valid_mask.sum() < 50:
            self.tprint("⚠️ Insufficient overlapping labels for correlation filtering")
            return per_target_configs

        valid_labels = label_series[valid_mask]

        # Compute pairwise correlations
        correlation_matrix = valid_labels.corr(method='spearman')

        # Greedy selection: start with best LQS, add others if correlation < threshold
        selected_configs = {}

        # Sort by LQS descending
        sorted_targets = sorted(
            per_target_configs.items(),
            key=lambda x: x[1]['quality_metrics'].label_quality_score,
            reverse=True
        )

        # Always include the best target unconditionally
        if sorted_targets:
            best_target_name, best_config = sorted_targets[0]
            selected_configs[best_target_name] = best_config
            self.tprint(f"✅ Selected {best_target_name} (LQS={best_config['quality_metrics'].label_quality_score:.3f}) - best target")

        # Add other targets if correlation is acceptable
        for target_name, config in sorted_targets[1:]:
            # Check correlation with already selected targets
            should_include = True

            for selected_target in selected_configs.keys():
                corr = abs(correlation_matrix.loc[target_name, selected_target])
                if corr > self.config.max_target_correlation:
                    should_include = False
                    self.tprint(f"🚫 Excluding {target_name} due to high correlation "
                           f"({corr:.3f}) with {selected_target}")
                    break

            if should_include:
                selected_configs[target_name] = config
                self.tprint(f"✅ Selected {target_name} (LQS={config['quality_metrics'].label_quality_score:.3f})")

        self.tprint(f"✅ Correlation filtering complete: {len(selected_configs)}/{len(per_target_configs)} targets selected")
        return selected_configs

    def _evaluate_config_quality(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series],
        config: Dict[str, Any],
        target_name: str
    ) -> LabelQualityMetrics:
        """Evaluate label quality for a specific configuration."""

        # Generate labels for this config (subset for speed)
        sample_size = max(0, min(5000, len(data) - self.config.max_horizon_bars))
        if sample_size < 200:  # Minimum samples for quality evaluation
            return LabelQualityMetrics()
        sample_data = data.iloc[-sample_size:].copy()

        # Generate sample labels
        sample_labels = self._generate_single_target_labels(
            sample_data, volatility.iloc[-sample_size:], noise_gates, config, target_name,
            max_samples=1000  # Limit for quality evaluation
        )

        # Compute quality metrics
        return self._compute_label_quality(sample_labels, sample_data, target_name)

    def _generate_single_target_labels(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series],
        config: Dict[str, Any],
        target_name: str,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate labels for a single target configuration with hysteresis and conflict resolution."""

        labels = pd.DataFrame(index=data.index)
        labels['target'] = 0  # -1, 0, +1

        # Compute FPT-based horizon for this k
        k = config['k']
        horizon_bars = self._compute_adaptive_horizon(data, volatility, k, config['fpt_quantile'])

        # Micro-range gate for this specific k (both in return units)
        median_tr = data['true_range'].rolling(window=20).median()
        median_tr_returns = median_tr / data['close']  # Convert to return scale
        micro_gate = (k * volatility) >= (config['micro_range_alpha'] * median_tr_returns)

        # Combined eligibility
        eligibility = noise_gates['eligibility'] & micro_gate

        # Track active instances to prevent overlap
        active_instances = {}  # timestamp -> (target, expiry_bar)
        next_free_idx = 0  # Track next available index for labeling

        # Generate labels bar by bar
        max_idx = min(len(data) - self.config.max_horizon_bars,
                     max_samples if max_samples else len(data))

        for i in range(self.config.min_bars_for_labeling, max_idx):
            current_time = data.index[i]

            # Skip if we're before the next free index (overlap prevention)
            if i < next_free_idx:
                continue

            if not eligibility.iloc[i]:
                continue

            current_price = data['close'].iloc[i]
            current_vol = volatility.iloc[i]
            horizon = horizon_bars.iloc[i]

            # Define barriers
            target_price_up = current_price * (1 + k * current_vol)
            target_price_down = current_price * (1 - k * current_vol)
            max_horizon_idx = min(i + int(horizon) + 1, len(data))

            # Look forward to find first barrier hit
            window_data = data.iloc[i:max_horizon_idx]

            # Check for upper barrier hit (long target)
            upper_hit = np.any(window_data['high'] >= target_price_up)
            if upper_hit:
                hit_idx = np.where(window_data['high'] >= target_price_up)[0][0]
                raw_target = 1

                # Enhanced labeling with local extrema detection
                if self.config.enable_peak_trough_detection:
                    # Find local peak within the hit window for more precise labeling
                    peak_idx, trough_idx = self._find_local_extrema_in_window(
                        data, i, i + hit_idx, 'high'
                    )
                    
                    if peak_idx is not None:
                        # Use the local peak for more precise labeling
                        actual_hit_idx = peak_idx - i
                        labels.loc[current_time, 'target'] = raw_target
                        labels.loc[current_time, 'time_to_hit'] = actual_hit_idx
                        labels.loc[current_time, 'confidence'] = max(0.0, 1.0 - actual_hit_idx / horizon)
                        labels.loc[current_time, 'extrema_type'] = 'peak'
                        labels.loc[current_time, 'extrema_price'] = data.iloc[peak_idx]['high']
                        
                        # Register active instance and update next_free_idx
                        expiry_bar = peak_idx + 1  # End after peak
                        active_instances[current_time] = (raw_target, expiry_bar)
                        next_free_idx = max(next_free_idx, expiry_bar)
                    else:
                        # Fallback to original barrier hit logic
                        labels.loc[current_time, 'target'] = raw_target
                        labels.loc[current_time, 'time_to_hit'] = hit_idx
                        labels.loc[current_time, 'confidence'] = max(0.0, 1.0 - hit_idx / horizon)
                        labels.loc[current_time, 'extrema_type'] = 'barrier'
                        labels.loc[current_time, 'extrema_price'] = target_price_up
                        
                        # Register active instance and update next_free_idx
                        expiry_bar = i + hit_idx + 1  # End after hit
                        active_instances[current_time] = (raw_target, expiry_bar)
                        next_free_idx = max(next_free_idx, expiry_bar)
                else:
                    # Original barrier hit logic without extrema detection
                    labels.loc[current_time, 'target'] = raw_target
                    labels.loc[current_time, 'time_to_hit'] = hit_idx
                    labels.loc[current_time, 'confidence'] = max(0.0, 1.0 - hit_idx / horizon)
                    labels.loc[current_time, 'extrema_type'] = 'barrier'
                    labels.loc[current_time, 'extrema_price'] = target_price_up
                    
                    # Register active instance and update next_free_idx
                    expiry_bar = i + hit_idx + 1  # End after hit
                    active_instances[current_time] = (raw_target, expiry_bar)
                    next_free_idx = max(next_free_idx, expiry_bar)

                # Hysteresis check: if recent label flip, require stronger signal
                if self._check_hysteresis_violation(labels, i, raw_target):
                    # Check if opposite barrier was hit by more than beta threshold
                    if self._check_flip_override(data, i, raw_target, target_price_up, target_price_down):
                        # Override allowed - proceed with flip
                        pass
                    else:
                        # Hysteresis violation without strong override - skip
                        continue

            else:
                # Check for lower barrier hit (short target)
                lower_hit = np.any(window_data['low'] <= target_price_down)
                if lower_hit:
                    hit_idx = np.where(window_data['low'] <= target_price_down)[0][0]
                    raw_target = -1

                    # Enhanced labeling with local extrema detection
                    if self.config.enable_peak_trough_detection:
                        # Find local trough within the hit window for more precise labeling
                        peak_idx, trough_idx = self._find_local_extrema_in_window(
                            data, i, i + hit_idx, 'low'
                        )
                        
                        if trough_idx is not None:
                            # Use the local trough for more precise labeling
                            actual_hit_idx = trough_idx - i
                            labels.loc[current_time, 'target'] = raw_target
                            labels.loc[current_time, 'time_to_hit'] = actual_hit_idx
                            labels.loc[current_time, 'confidence'] = max(0.0, 1.0 - actual_hit_idx / horizon)
                            labels.loc[current_time, 'extrema_type'] = 'trough'
                            labels.loc[current_time, 'extrema_price'] = data.iloc[trough_idx]['low']
                            
                            # Register active instance and update next_free_idx
                            expiry_bar = trough_idx + 1  # End after trough
                            active_instances[current_time] = (raw_target, expiry_bar)
                            next_free_idx = max(next_free_idx, expiry_bar)
                        else:
                            # Fallback to original barrier hit logic
                            labels.loc[current_time, 'target'] = raw_target
                            labels.loc[current_time, 'time_to_hit'] = hit_idx
                            labels.loc[current_time, 'confidence'] = max(0.0, 1.0 - hit_idx / horizon)
                            labels.loc[current_time, 'extrema_type'] = 'barrier'
                            labels.loc[current_time, 'extrema_price'] = target_price_down
                            
                            # Register active instance and update next_free_idx
                            expiry_bar = i + hit_idx + 1  # End after hit
                            active_instances[current_time] = (raw_target, expiry_bar)
                            next_free_idx = max(next_free_idx, expiry_bar)
                    else:
                        # Original barrier hit logic without extrema detection
                        labels.loc[current_time, 'target'] = raw_target
                        labels.loc[current_time, 'time_to_hit'] = hit_idx
                        labels.loc[current_time, 'confidence'] = max(0.0, 1.0 - hit_idx / horizon)
                        labels.loc[current_time, 'extrema_type'] = 'barrier'
                        labels.loc[current_time, 'extrema_price'] = target_price_down
                        
                        # Register active instance and update next_free_idx
                        expiry_bar = i + hit_idx + 1  # End after hit
                        active_instances[current_time] = (raw_target, expiry_bar)
                        next_free_idx = max(next_free_idx, expiry_bar)

                    # Hysteresis check: if recent label flip, require stronger signal
                    if self._check_hysteresis_violation(labels, i, raw_target):
                        # Check if opposite barrier was hit by more than beta threshold
                        if self._check_flip_override(data, i, raw_target, target_price_up, target_price_down):
                            # Override allowed - proceed with flip
                            pass
                        else:
                            # Hysteresis violation without strong override - skip
                            continue

        # Add metadata
        labels['k'] = k
        labels['horizon_bars'] = horizon_bars
        labels['eligibility'] = eligibility

        # Clean up expired instances (for long-running scenarios)
        current_bar = max_idx
        expired_times = [t for t, (_, expiry) in active_instances.items() if expiry <= current_bar]
        for t in expired_times:
            del active_instances[t]

        return labels

    def _check_hysteresis_violation(self, labels: pd.DataFrame, current_idx: int, new_target: int) -> bool:
        """Check if assigning new_target would violate hysteresis constraints."""

        # Look back at recent labels within hysteresis window
        lookback_window = self.config.hysteresis_bars

        for lookback in range(1, lookback_window + 1):
            if current_idx - lookback < 0:
                break

            prev_target = labels.iloc[current_idx - lookback]['target']
            # Handle uninitialized labels (NaN or 0)
            if pd.isna(prev_target):
                prev_target = 0
            if prev_target != 0 and prev_target != new_target:
                return True  # Would be a flip

        return False  # No hysteresis violation

    def _check_flip_override(self, data: pd.DataFrame, current_idx: int,
                           new_target: int, target_up: float, target_down: float) -> bool:
        """Check if flip should be allowed due to strong opposite signal."""

        current_price = data['close'].iloc[current_idx]

        if new_target == 1:  # Trying to flip to long (was short)
            # Check if short barrier was hit by more than beta threshold
            short_barrier_distance = (current_price - target_down) / current_price
            return short_barrier_distance >= self.config.flip_override_beta

        elif new_target == -1:  # Trying to flip to short (was long)
            # Check if long barrier was hit by more than beta threshold
            long_barrier_distance = (target_up - current_price) / current_price
            return long_barrier_distance >= self.config.flip_override_beta

        return False

    def _compute_adaptive_horizon(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        k: float,
        quantile: float
    ) -> pd.Series:
        """Compute adaptive horizons based on first-passage time quantiles."""

        horizons = pd.Series(index=data.index, dtype=float)

        # For each point, estimate FPT to ±k·σ_t
        for i in range(max(50, len(data) - 1000), len(data)):
            # Look back to estimate FPT distribution
            lookback = min(i, 500)
            historical_data = data.iloc[i-lookback:i]

            if len(historical_data) < 50:
                horizons.iloc[i] = 20  # Default
                continue

            # Compute historical FPTs for this k
            fpt_values = []
            for j in range(20, len(historical_data)):  # Skip first 20 for stability
                current_price = historical_data['close'].iloc[j]
                current_vol = volatility.iloc[i-lookback+j] if i-lookback+j < len(volatility) else volatility.iloc[i-1]

                if pd.isna(current_vol) or current_vol <= 0:
                    continue

                target_up = current_price * (1 + k * current_vol)
                target_down = current_price * (1 - k * current_vol)

                # Find first hit within reasonable horizon
                window_end = min(j + self.config.max_horizon_bars, len(historical_data))
                window = historical_data.iloc[j:window_end]

                upper_hit = np.where(window['high'] >= target_up)[0]
                lower_hit = np.where(window['low'] <= target_down)[0]

                if len(upper_hit) > 0 or len(lower_hit) > 0:
                    first_hit = min(upper_hit[0] if len(upper_hit) > 0 else self.config.max_horizon_bars,
                                  lower_hit[0] if len(lower_hit) > 0 else self.config.max_horizon_bars)
                    fpt_values.append(first_hit)

            if fpt_values:
                # Use quantile of FPT distribution
                fpt_array = np.array(fpt_values)
                horizon = np.quantile(fpt_array, quantile)
                horizons.iloc[i] = max(1, min(horizon, self.config.max_horizon_bars))
            else:
                horizons.iloc[i] = 20  # Default fallback

        # Forward fill and smooth with EMA
        horizons = horizons.fillna(method='ffill').fillna(20)
        horizons = horizons.ewm(alpha=0.1).mean()  # Light smoothing

        return horizons

    def _detect_peaks_troughs(self, data: pd.DataFrame, price_column: str = 'close') -> Tuple[pd.Series, pd.Series]:
        """
        Detect peaks and troughs in price data using scipy.signal methods.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Column to use for peak detection ('close', 'high', 'low')
            
        Returns:
            Tuple of (peaks_series, troughs_series) where 1 indicates peak/trough
        """
        if not self.config.enable_peak_trough_detection:
            # Return empty series if peak detection is disabled
            return pd.Series(0, index=data.index), pd.Series(0, index=data.index)
        
        try:
            prices = data[price_column].values
            if len(prices) < 10:  # Need minimum data for peak detection
                return pd.Series(0, index=data.index), pd.Series(0, index=data.index)
            
            # Smooth the data if specified
            if self.config.smoothing_window > 1:
                from scipy.ndimage import uniform_filter1d
                prices_smooth = uniform_filter1d(prices, size=self.config.smoothing_window)
            else:
                prices_smooth = prices
            
            # Calculate prominence threshold as fraction of price range
            price_range = np.max(prices_smooth) - np.min(prices_smooth)
            prominence_threshold = price_range * self.config.peak_prominence
            
            # Detect peaks and troughs based on method
            if self.config.peak_detection_method == "find_peaks":
                peaks, _ = find_peaks(
                    prices_smooth,
                    prominence=prominence_threshold,
                    distance=self.config.peak_distance,
                    width=self.config.peak_width,
                    height=self.config.peak_height_threshold
                )
                
                # For troughs, detect peaks in inverted signal
                troughs, _ = find_peaks(
                    -prices_smooth,
                    prominence=prominence_threshold,
                    distance=self.config.peak_distance,
                    width=self.config.peak_width,
                    height=-self.config.peak_height_threshold if self.config.peak_height_threshold else None
                )
                
            elif self.config.peak_detection_method == "find_peaks_cwt":
                # Use continuous wavelet transform for peak detection
                from scipy.signal import find_peaks_cwt
                peaks = find_peaks_cwt(prices_smooth, widths=np.arange(1, 10))
                troughs = find_peaks_cwt(-prices_smooth, widths=np.arange(1, 10))
                
            elif self.config.peak_detection_method == "argrelextrema":
                # Use relative extrema detection
                peaks = argrelextrema(prices_smooth, np.greater, order=self.config.peak_distance)[0]
                troughs = argrelextrema(prices_smooth, np.less, order=self.config.peak_distance)[0]
                
            else:
                raise ValueError(f"Unknown peak detection method: {self.config.peak_detection_method}")
            
            # Create boolean series for peaks and troughs
            peaks_series = pd.Series(0, index=data.index)
            troughs_series = pd.Series(0, index=data.index)
            
            # Mark detected peaks and troughs
            if len(peaks) > 0:
                peaks_series.iloc[peaks] = 1
            if len(troughs) > 0:
                troughs_series.iloc[troughs] = 1
            
            self.tprint(f"✅ Peak/trough detection: {peaks_series.sum()} peaks, {troughs_series.sum()} troughs")
            return peaks_series, troughs_series
            
        except Exception as e:
            tprint_warning(f"⚠️ Peak/trough detection failed: {e}")
            return pd.Series(0, index=data.index), pd.Series(0, index=data.index)

    def _find_local_extrema_in_window(self, data: pd.DataFrame, start_idx: int, end_idx: int, 
                                    price_column: str = 'close') -> Tuple[Optional[int], Optional[int]]:
        """
        Find local peaks and troughs within a specific time window.
        
        Args:
            data: DataFrame with OHLCV data
            start_idx: Start index of the window
            end_idx: End index of the window
            price_column: Column to use for extrema detection
            
        Returns:
            Tuple of (peak_idx, trough_idx) within the window, or (None, None) if not found
        """
        if end_idx <= start_idx or start_idx < 0 or end_idx >= len(data):
            return None, None
        
        window_data = data.iloc[start_idx:end_idx+1]
        if len(window_data) < 3:  # Need at least 3 points for extrema
            return None, None
        
        try:
            prices = window_data[price_column].values
            
            # Use argrelextrema for local extrema within the window
            peak_indices = argrelextrema(prices, np.greater, order=1)[0]
            trough_indices = argrelextrema(prices, np.less, order=1)[0]
            
            # Convert to absolute indices
            peak_idx = start_idx + peak_indices[0] if len(peak_indices) > 0 else None
            trough_idx = start_idx + trough_indices[0] if len(trough_indices) > 0 else None
            
            return peak_idx, trough_idx
            
        except Exception as e:
            tprint_warning(f"⚠️ Local extrema detection in window failed: {e}")
            return None, None

    def _detect_opportunity_patterns(self, data: pd.DataFrame, i: int, horizon: int) -> Dict[str, Any]:
        """
        Detect opportunity patterns using peak/trough analysis within a time window.
        
        Args:
            data: DataFrame with OHLCV data
            i: Current index
            horizon: Look-forward horizon in bars
            
        Returns:
            Dictionary with opportunity pattern information
        """
        if not self.config.enable_peak_trough_detection:
            return {'has_opportunity': False}
        
        try:
            end_idx = min(i + horizon, len(data) - 1)
            if end_idx <= i + 2:  # Need at least 3 bars for pattern detection
                return {'has_opportunity': False}
            
            window_data = data.iloc[i:end_idx+1]
            prices = window_data['close'].values
            highs = window_data['high'].values
            lows = window_data['low'].values
            
            # Detect local extrema in the window
            peak_indices = argrelextrema(highs, np.greater, order=1)[0]
            trough_indices = argrelextrema(lows, np.less, order=1)[0]
            
            # Look for specific patterns
            patterns = {
                'has_opportunity': False,
                'peak_indices': peak_indices,
                'trough_indices': trough_indices,
                'pattern_type': None,
                'confidence': 0.0
            }
            
            # Pattern 1: Peak followed by decline (short opportunity)
            if len(peak_indices) > 0 and len(trough_indices) > 0:
                first_peak = peak_indices[0]
                first_trough = trough_indices[0]
                
                if first_peak < first_trough:
                    # Peak first, then trough - potential short opportunity
                    patterns['has_opportunity'] = True
                    patterns['pattern_type'] = 'peak_trough'
                    patterns['confidence'] = 0.7
                elif first_trough < first_peak:
                    # Trough first, then peak - potential long opportunity
                    patterns['has_opportunity'] = True
                    patterns['pattern_type'] = 'trough_peak'
                    patterns['confidence'] = 0.7
            
            # Pattern 2: Strong directional move with local extrema
            elif len(peak_indices) > 0:
                # Only peaks detected - potential short opportunity
                patterns['has_opportunity'] = True
                patterns['pattern_type'] = 'peak_only'
                patterns['confidence'] = 0.5
            elif len(trough_indices) > 0:
                # Only troughs detected - potential long opportunity
                patterns['has_opportunity'] = True
                patterns['pattern_type'] = 'trough_only'
                patterns['confidence'] = 0.5
            
            return patterns
            
        except Exception as e:
            tprint_warning(f"⚠️ Opportunity pattern detection failed: {e}")
            return {'has_opportunity': False}

    def _compute_label_quality(
        self,
        labels: pd.DataFrame,
        data: pd.DataFrame,
        target_name: str
    ) -> LabelQualityMetrics:
        """Compute comprehensive label quality metrics with enhanced feature engineering."""

        metrics = LabelQualityMetrics()

        # Filter to valid labels only
        valid_labels = labels[labels['target'] != 0].copy()

        if len(valid_labels) < 100:
            tprint_warning(f"⚠️ Insufficient labels for quality evaluation: {len(valid_labels)}")
            return metrics

        # 1. Balance metrics
        positive_count = (valid_labels['target'] == 1).sum()
        negative_count = (valid_labels['target'] == -1).sum()
        total_count = len(valid_labels)

        metrics.positive_balance = positive_count / total_count if total_count > 0 else 0.5
        # Balance score: closer to 0.5 is better (penalize imbalance)
        metrics.class_balance_score = 1.0 - abs(metrics.positive_balance - 0.5) * 2

        # 2. Enhanced feature-based predictability using feature engineering
        try:
            # Use feature selector to create comprehensive features
            feature_data = self._create_enhanced_features(data, valid_labels)
            
            if len(feature_data) >= 50:
                X = feature_data.values
                y = valid_labels['target'].values
                
                # Use feature selection to identify most relevant features
                selected_features = self.feature_selector.select_features(
                    X, y, method='mutual_info', k_best=10
                )
                
                if len(selected_features) > 0:
                    X_selected = X[:, selected_features]
                else:
                    X_selected = X

                # Enhanced model evaluation with cross-validation
                from sklearn.model_selection import TimeSeriesSplit
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                tscv = TimeSeriesSplit(n_splits=min(5, len(X_selected)//50))
                scaler = StandardScaler()

                auc_scores = []
                pr_scores = []

                for train_idx, test_idx in tscv.split(X_selected):
                    if len(train_idx) < 10 or len(test_idx) < 10:
                        continue

                    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Use Random Forest for better feature importance
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        max_depth=10,
                        min_samples_split=5
                    )
                    model.fit(X_train_scaled, y_train)

                    # Predict probabilities
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                    # AUC
                    auc = roc_auc_score(y_test, y_pred_proba)
                    auc_scores.append(auc)

                    # PR-AUC
                    pr_auc = average_precision_score(y_test, y_pred_proba)
                    pr_scores.append(pr_auc)

                if auc_scores:
                    metrics.auc_mean = np.mean(auc_scores)
                    metrics.auc_std = np.std(auc_scores)
                    metrics.pr_auc_mean = np.mean(pr_scores)
                    metrics.pr_auc_std = np.std(pr_scores)

        except Exception as e:
            tprint_warning(f"⚠️ Failed to compute enhanced predictability metrics: {e}")

        # 3. Enhanced stability metrics using statistical tests
        if len(valid_labels) > 200:
            try:
                # Use more sophisticated stability measures
                from scipy import stats
                
                # Split into two halves for stability analysis
                mid_point = len(valid_labels) // 2
                first_half = valid_labels.iloc[:mid_point]
                second_half = valid_labels.iloc[mid_point:]

                # Kolmogorov-Smirnov test for distribution stability
                ks_stat, ks_pvalue = stats.ks_2samp(
                    first_half['target'], 
                    second_half['target']
                )
                
                # PSI calculation with better binning
                def compute_psi_enhanced(labels1, labels2, bins=10):
                    # Create bins based on quantiles
                    all_labels = pd.concat([labels1, labels2])
                    bin_edges = np.quantile(all_labels, np.linspace(0, 1, bins + 1))
                    bin_edges = np.unique(bin_edges)  # Remove duplicates
                    
                    # Compute distributions
                    dist1 = np.histogram(labels1, bins=bin_edges)[0] / len(labels1)
                    dist2 = np.histogram(labels2, bins=bin_edges)[0] / len(labels2)
                    
                    # Add small epsilon to avoid log(0)
                    dist1 = np.maximum(dist1, 1e-8)
                    dist2 = np.maximum(dist2, 1e-8)
                    
                    # PSI calculation
                    psi = np.sum((dist1 - dist2) * np.log(dist1 / dist2))
                    return psi

                psi_score = compute_psi_enhanced(
                    first_half['target'], 
                    second_half['target']
                )
                
                metrics.psi_score = min(psi_score, 1.0)  # Cap at 1.0
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to compute enhanced stability metrics: {e}")

        # 4. Enhanced flip rate with temporal analysis
        try:
            # Calculate flip rate with different time windows
            flip_rates = []
            for window in [1, 2, 5]:  # Different time windows
                flip_count = 0
                for i in range(window, len(valid_labels)):
                    if valid_labels.iloc[i]['target'] != valid_labels.iloc[i-window]['target']:
                        flip_count += 1
                flip_rate = flip_count / (len(valid_labels) - window) if len(valid_labels) > window else 0.0
                flip_rates.append(flip_rate)
            
            # Use average flip rate across windows
            metrics.flip_rate = np.mean(flip_rates)
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to compute enhanced flip rate: {e}")

        # 5. Enhanced SNR using mutual information
        try:
            if len(feature_data) >= 50:
                from sklearn.feature_selection import mutual_info_classif
                
                # Calculate mutual information between features and labels
                mi_scores = mutual_info_classif(
                    feature_data.values, 
                    valid_labels['target'].values,
                    random_state=42
                )
                
                metrics.feature_ic_mean = np.mean(mi_scores) if len(mi_scores) > 0 else 0.0
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to compute enhanced SNR metrics: {e}")

        # 6. Enhanced composite Label Quality Score (LQS)
        weights = self.config.lqs_weights

        # Normalize components to [0,1] range where applicable
        auc_score = min(metrics.auc_mean, 1.0) if metrics.auc_mean > 0 else 0.0
        stability_score = 1.0 - min(metrics.psi_score, 1.0)  # Lower PSI is better
        balance_score = metrics.class_balance_score
        snr_score = min(metrics.feature_ic_mean, 1.0) if metrics.feature_ic_mean > 0 else 0.0
        consistency_score = 1.0 - min(metrics.flip_rate, 1.0)  # Lower flip rate is better

        metrics.label_quality_score = (
            weights['predictability'] * auc_score +
            weights['stability'] * stability_score +
            weights['balance'] * balance_score +
            weights['snr'] * snr_score +
            weights['consistency'] * consistency_score
        )

        return metrics

    def _create_enhanced_features(self, data: pd.DataFrame, valid_labels: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for label quality evaluation."""
        try:
            features = []
            
            for idx in valid_labels.index:
                if idx not in data.index:
                    continue
                    
                # Get data up to current index (no lookahead)
                current_data = data.loc[:idx]
                
                if len(current_data) < 10:
                    continue
                
                feature_row = []
                
                # Basic price features
                if 'close' in current_data.columns:
                    feature_row.extend([
                        current_data['close'].iloc[-1],  # Current price
                        current_data['close'].pct_change().iloc[-1],  # Current return
                        current_data['close'].pct_change().rolling(5).mean().iloc[-1],  # 5-period return
                        current_data['close'].pct_change().rolling(10).mean().iloc[-1],  # 10-period return
                        current_data['close'].pct_change().rolling(20).std().iloc[-1],  # 20-period volatility
                    ])
                
                # Volume features
                if 'volume' in current_data.columns:
                    feature_row.extend([
                        current_data['volume'].iloc[-1],  # Current volume
                        current_data['volume'].rolling(5).mean().iloc[-1],  # 5-period avg volume
                        current_data['volume'].rolling(10).std().iloc[-1],  # Volume volatility
                    ])
                
                # Technical indicators
                if 'high' in current_data.columns and 'low' in current_data.columns:
                    # True range
                    tr = np.maximum(
                        current_data['high'] - current_data['low'],
                        np.maximum(
                            abs(current_data['high'] - current_data['close'].shift(1)),
                            abs(current_data['low'] - current_data['close'].shift(1))
                        )
                    )
                    feature_row.extend([
                        tr.rolling(14).mean().iloc[-1],  # ATR
                        tr.iloc[-1],  # Current TR
                    ])
                
                # Momentum features
                if 'close' in current_data.columns:
                    # RSI-like momentum
                    returns = current_data['close'].pct_change()
                    gains = returns.where(returns > 0, 0)
                    losses = -returns.where(returns < 0, 0)
                    
                    avg_gain = gains.rolling(14).mean().iloc[-1]
                    avg_loss = losses.rolling(14).mean().iloc[-1]
                    
                    if not pd.isna(avg_gain) and not pd.isna(avg_loss) and avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        feature_row.append(rsi)
                    else:
                        feature_row.append(50)  # Neutral RSI
                
                # Time-based features
                if hasattr(current_data.index, 'hour'):
                    feature_row.extend([
                        current_data.index[-1].hour,
                        current_data.index[-1].dayofweek,
                    ])
                
                features.append(feature_row)
            
            if features:
                feature_df = pd.DataFrame(features, index=valid_labels.index[:len(features)])
                return feature_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to create enhanced features: {e}")
            return pd.DataFrame()

    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, LabelQualityMetrics]) -> float:
        """Calculate overall quality score across all targets."""
        if not quality_metrics:
            return 0.0
        
        # Calculate weighted average of LQS scores
        lqs_scores = [metrics.label_quality_score for metrics in quality_metrics.values()]
        return np.mean(lqs_scores) if lqs_scores else 0.0

# Factory functions for backward compatibility
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses) 
            if (self._cache_hits + self._cache_misses) > 0 else 0.0
        )
        
        return {
            **self._performance_metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'phase1_optimizations_available': PHASE1_OPTIMIZATIONS_AVAILABLE,
            'vectorbt_available': self.vectorbt_optimizer is not None,
            'memory_optimization_enabled': self.memory_optimizer is not None
        }
    
    def clear_cache(self):
        """Clear calculation cache to free memory."""
        self._calculation_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.tprint("🧹 Labeling cache cleared")


def create_consolidated_labeler(config: Optional[ConsolidatedLabelerConfig] = None) -> ConsolidatedProfitLabeler:
    """Factory function to create consolidated labeler."""
    return ConsolidatedProfitLabeler(config)

def create_enhanced_analyst_labeler(config: Optional[ConsolidatedLabelerConfig] = None) -> ConsolidatedProfitLabeler:
    """Factory function to create enhanced analyst labeler for feature generation integration."""
    # Create a config optimized for analyst labeling if none provided
    if config is None:
        analyst_config = ConsolidatedLabelerConfig(
            # Analyst-optimized parameters
            target_bands={
                'analyst': (0.6, 1.2),  # Single target band for analyst
                'small': (0.4, 0.8),    # Keep other bands for compatibility
                'medium': (0.8, 1.3),
                'high': (1.3, 2.0)
            },
            fpt_quantile=0.65,  # Standard quantile
            min_positive_balance=0.35,
            max_positive_balance=0.65,
            min_aic_threshold=0.55,
            max_auc_std_threshold=0.08,
            enable_trading_aware_labels=True
        )
        config = analyst_config
    
    return ConsolidatedProfitLabeler(config)

def create_volatility_aware_labeler(config: Optional[ConsolidatedLabelerConfig] = None) -> ConsolidatedProfitLabeler:
    """Factory function to create volatility-aware labeler (backward compatibility)."""
    return ConsolidatedProfitLabeler(config)

def apply_consolidated_labeling(
    data: pd.DataFrame,
    config: Optional[ConsolidatedLabelerConfig] = None,
    validate_robustness: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply consolidated volatility-aware multi-horizon profit labeling.

    Args:
        data: OHLCV DataFrame
        config: Configuration (optional)
        validate_robustness: Whether to run comprehensive validation

    Returns:
        Tuple of (labeled_dataframe, quality_report)
    """
    labeler = ConsolidatedProfitLabeler(config)
    result = labeler.generate_labels(data)
    
    # Convert result to expected format for backward compatibility
    quality_report = {
        'overall_quality_score': result.overall_quality_score,
        'quality_metrics': {k: v.to_dict() for k, v in result.quality_metrics.items()},
        'processing_time': result.processing_time,
        'n_samples': result.n_samples,
        'n_targets': result.n_targets
    }
    
    return result.labels, quality_report

# Backward compatibility aliases
VolatilityAwareProfitLabeler = ConsolidatedProfitLabeler
VolatilityAwareConfig = ConsolidatedLabelerConfig
LabelQualityScore = LabelQualityMetrics
MultiHorizonProfitLabeler = ConsolidatedProfitLabeler
MultiHorizonConfig = ConsolidatedLabelerConfig

# Test function
if __name__ == '__main__':
    # Simple test
    tprint('🧪 Testing Consolidated Profit Labeler')

    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)

    # Generate realistic price data with trends and volatility
    base_price = 100.0
    prices = [base_price]

    for i in range(999):
        # Add trend and volatility
        trend = 0.0001 * (i // 100 - 5)  # Changing trend
        vol = 0.002 + 0.001 * np.sin(i / 50)  # Changing volatility
        ret = np.random.normal(trend, vol)
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)

    # Test labeling
    tprint('\n🔍 Testing consolidated labeling...')
    config = ConsolidatedLabelerConfig()
    labeled_data, report = apply_consolidated_labeling(data, config)

    tprint(f'✅ Labeling completed:')
    tprint(f'   → Input shape: {data.shape}')
    tprint(f'   → Output shape: {labeled_data.shape}')

    # Show sample quality metrics
    if report and 'quality_metrics' in report:
        tprint(f'\n📊 Quality Report Summary:')
        for target, metrics in report['quality_metrics'].items():
            tprint(f'   → {target}: LQS={metrics.get("label_quality_score", 0):.3f}, '
                   f'AUC={metrics.get("auc_mean", 0):.3f}±{metrics.get("auc_std", 0):.3f}')

    tprint('✅ Consolidated Profit Labeler test completed!')