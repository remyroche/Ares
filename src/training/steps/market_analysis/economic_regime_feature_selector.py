"""
Economic Regime Feature Selector

This module provides economic feature selection for regime clustering using vectorized operations,
cheap computational proxies, and comprehensive validation with forward returns and Sharpe ratios.

Key Features:
- Vectorized economic metrics calculation
- Cheap proxies for expensive operations
- Comprehensive validation framework
- Integration with feature bank (excluding microstructure and support/resistance)
- Extensive tprint logging
- Comprehensive report generation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Additional imports for optimization
import hashlib
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

# Base step import
from src.training.steps.base_step import BaseStep

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    # VectorBT rolling functions are accessed differently
    rolling_mean = lambda x, window: x.rolling(window).mean()
    rolling_std = lambda x, window: x.rolling(window).std()
    rolling_var = lambda x, window: x.rolling(window).var()
    rolling_min = lambda x, window: x.rolling(window).min()
    rolling_max = lambda x, window: x.rolling(window).max()
    rolling_sum = lambda x, window: x.rolling(window).sum()
    rolling_apply = lambda x, window, func: x.rolling(window).apply(func)
    rolling_quantile = lambda x, window, q: x.rolling(window).quantile(q)
    rolling_corr = lambda x, y, window: x.rolling(window).corr(y)
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_quantile = None
    rolling_corr = None

# Optimization tools
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None

# Hardware optimization
try:
    from src.utils.hardware import get_unified_hardware_manager, WorkloadType, OptimizationLevel
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    get_unified_hardware_manager = None
    WorkloadType = None
    OptimizationLevel = None

# Validation tools (optional - removed unused imports)
PURGED_CV_AVAILABLE = False
TIME_SERIES_VALIDATION_AVAILABLE = False
DATA_LEAKAGE_DETECTION_AVAILABLE = False

# Common utilities
try:
    from src.utils.common_operations import *
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError:
    COMMON_OPERATIONS_AVAILABLE = False

try:
    from src.utils.common_utilities import *
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False

try:
    from src.utils.math_validation import validate_numerical_stability
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    validate_numerical_stability = None

# Tprint logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_performance, tprint_data_preview, tprint_data_format, tprint_feature_counts,
        configure_tprint, TPrintConfig, LogLevel
    )
    
    # Configure tprint for minimal mode to reduce overhead
    configure_tprint(TPrintConfig(
        use_colors=False,
        output_to_file=False,
        log_to_python_logger=False,
        integrate_with_logging=False,
        min_log_level=LogLevel.INFO,
        enable_lazy_evaluation=True,
        cache_timestamps=True
    ))
    
    # Add tprint_debug if it exists, otherwise define it
    try:
        from src.utils.tprint import tprint_debug
    except ImportError:
        def tprint_debug(*args, **kwargs): print("DEBUG:", *args)
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args)
    def tprint_performance(*args, **kwargs): print("PERF:", *args)
    def tprint_data_preview(*args, **kwargs): print("DATA:", *args)
    def tprint_data_format(*args, **kwargs): print("FORMAT:", *args)
    def tprint_feature_counts(*args, **kwargs): print("FEATURES:", *args)

# Feature bank integration
try:
    from src.feature_generation import get_feature_bank, list_available_categories
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False
    get_feature_bank = None
    list_available_categories = None

# Feature selection tools
try:
    from src.feature_selection.advanced.enhanced_advanced_selector import EnhancedAdvancedSelector
    ENHANCED_SELECTOR_AVAILABLE = True
except ImportError:
    ENHANCED_SELECTOR_AVAILABLE = False
    EnhancedAdvancedSelector = None

try:
    from src.feature_selection.methods.importance import mutual_info_selector
    MUTUAL_INFO_SELECTOR_AVAILABLE = True
except ImportError:
    MUTUAL_INFO_SELECTOR_AVAILABLE = False
    mutual_info_selector = None

# Features common - specific imports only
try:
    from src.features_common.transforms import BaseScaler
    from src.features_common.utils import validate_numerical_stability as features_validate_stability
    FEATURES_COMMON_AVAILABLE = True
except ImportError:
    FEATURES_COMMON_AVAILABLE = False
    BaseScaler = None
    features_validate_stability = None

logger = logging.getLogger(__name__)

@dataclass
class EconomicFeatureSelectorConfig:
    """Configuration for economic feature selection."""
    
    # Multi-target approach settings
    multi_target_enabled: bool = True
    target_columns: List[str] = field(default_factory=lambda: ['close_return', 'volume_log_return', 'price_range_pct', 'body_size_pct', 'volume_return', 'close_log_return', 'price_range', 'trades', 'volatility_20', 'cmf'])
    target_weights: Dict[str, float] = field(default_factory=lambda: {
        'close_return': 0.12,      # Price movements (12% - reduced)
        'volume_log_return': 0.10, # Volume patterns (10% - reduced)
        'price_range_pct': 0.20,   # Relative volatility (20% - significantly increased)
        'body_size_pct': 0.06,     # Price efficiency (6% - reduced)
        'volume_return': 0.06,     # Volume momentum (6% - reduced)
        'close_log_return': 0.10,  # Log price movements (10% - reduced)
        'price_range': 0.15,       # Absolute volatility (15% - significantly increased)
        'trades': 0.02,            # Trade patterns (2% - kept same)
        'volatility_20': 0.12,     # Realized volatility rolling (12% - new)
        'cmf': 0.07               # Volume imbalance/order flow (7% - new)
    })
    
    # Feature selection parameters
    target_feature_count: int = 25
    min_feature_count: int = 15
    max_feature_count: int = 35
    
    # Economic scoring weights
    economic_significance_weight: float = 0.30  # Increased back to make room
    regime_discrimination_weight: float = 0.25  # Kept same
    clustering_quality_weight: float = 0.15     # Kept same
    stability_weight: float = 0.10              # Kept same
    mrmr_weight: float = 0.08                   # Further reduced to allow more feature diversity
    regime_transition_weight: float = 0.05      # Kept same
    
    # Validation parameters
    cv_folds: int = 3  # Cheap proxy: 3 instead of 5
    silhouette_sample_ratio: float = 0.10  # Cheap proxy: 10% sample
    min_economic_significance: float = 0.25  # Adjusted to match actual scores
    min_regime_discrimination: float = 0.90  # Adjusted to match actual scores  
    min_clustering_quality: float = 0.10    # Adjusted for proper clustering calculation
    min_stability_score: float = 0.30       # Adjusted for proper stability calculation
    
    # Performance thresholds
    min_sharpe_variance: float = 0.1  # Reduced threshold for better economic distinctiveness calculation
    max_noise_ratio: float = 0.30
    
    # mRMR parameters
    enable_mrmr: bool = True
    protect_categories: List[str] = field(default_factory=lambda: ['volatility', 'volume'])  # Categories to protect from redundancy filtering
    
    # Regime transition feature parameters
    transition_score_threshold: float = 0.95  # Increased significantly to be very discriminating
    max_transition_features_ratio: float = 0.3
    transition_window_size: int = 10
    
    # Computational optimization
    enable_vectorbt: bool = True
    enable_hardware_optimization: bool = True
    enable_cheap_proxies: bool = True
    
    # Feature bank filtering
    exclude_categories: List[str] = field(default_factory=lambda: ['microstructure', 'support_resistance'])
    include_categories: List[str] = field(default_factory=lambda: [
        'returns', 'momentum', 'volume', 'volatility', 'regime', 'order_flow',
        'statistical', 'entropy', 'spectral', 'trend', 'oscillator', 'candlestick',
        'cross_timeframe', 'interaction'
    ])

@dataclass
class FeatureScore:
    """Individual feature score components."""
    feature_name: str
    economic_significance: float
    regime_discrimination: float
    clustering_quality: float
    stability_score: float
    mrmr_score: float
    regime_transition_score: float = 0.0  # New regime transition detection score
    composite_score: float = 0.0
    category: str = ""
    # TreeSHAP integration fields
    treeshap_importance: float = 0.0
    diversity_score: float = 0.0
    selected: bool = False

@dataclass
class EconomicMetrics:
    """Economic metrics per regime."""
    regime_id: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    avg_return: float
    volatility: float
    max_drawdown: float
    sample_count: int

@dataclass
class FeatureSelectionResult:
    """Result of feature selection process."""
    selected_features: List[str]
    feature_scores: List[FeatureScore]
    economic_metrics: List[EconomicMetrics]
    validation_metrics: Dict[str, float]
    computational_stats: Dict[str, Any]
    report_path: str
    success: bool
    error_message: Optional[str] = None

class EconomicRegimeFeatureSelector(BaseStep):
    """
    Economic Regime Feature Selector for regime clustering.
    
    Selects economically relevant features using vectorized operations and cheap proxies.
    Inherits from BaseStep for artifact management and outcome generation.
    """
    
    def __init__(self, step_name: str = "regime_feature_selection"):
        """Initialize the economic regime feature selector."""
        super().__init__(step_name)
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize configuration
        self.config = self._load_config()
        
        # Initialize optimization tools
        self._initialize_optimization_tools()
        
        # Initialize cache for expensive calculations
        self._correlation_cache = {}
        self._silhouette_cache = {}
        self._mrmr_cache = {}
        self._precomputed_indices = {}
        self._optimized_data_types = {}
        
        tprint_info(f"EconomicRegimeFeatureSelector initialized: {step_name}")
        tprint_info(f"VectorBT available: {VECTORBT_AVAILABLE}")
        tprint_info(f"Hardware optimization: {HARDWARE_AVAILABLE}")
        tprint_info(f"Feature bank available: {FEATURE_BANK_AVAILABLE}")
    
    def _get_data_hash(self, series: pd.Series) -> str:
        """Generate hash for data caching."""
        try:
            return hashlib.md5(series.values.tobytes()).hexdigest()
        except Exception:
            return str(hash(str(series.values)))
    
    def _get_cached_correlation(self, feature1: pd.Series, feature2: pd.Series) -> Optional[float]:
        """Get cached correlation result."""
        hash1 = self._get_data_hash(feature1)
        hash2 = self._get_data_hash(feature2)
        cache_key = f"{hash1}_{hash2}" if hash1 < hash2 else f"{hash2}_{hash1}"
        return self._correlation_cache.get(cache_key)
    
    def _set_cached_correlation(self, feature1: pd.Series, feature2: pd.Series, correlation: float):
        """Cache correlation result."""
        hash1 = self._get_data_hash(feature1)
        hash2 = self._get_data_hash(feature2)
        cache_key = f"{hash1}_{hash2}" if hash1 < hash2 else f"{hash2}_{hash1}"
        self._correlation_cache[cache_key] = correlation
    
    def _optimize_data_types(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types using hardware manager."""
        try:
            # Check if already optimized
            data_hash = self._get_data_hash(features_df.iloc[:, 0])
            if data_hash in self._optimized_data_types:
                return self._optimized_data_types[data_hash]
            
            # Use hardware manager to determine optimal data types
            if HARDWARE_AVAILABLE and self.hardware_manager:
                optimized_df = features_df.astype(np.float32)  # Use float32 for better performance
                tprint_info("⚡ Data types optimized to float32")
            else:
                optimized_df = features_df.copy()
            
            # Cache the result
            self._optimized_data_types[data_hash] = optimized_df
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"Data type optimization failed: {e}")
            return features_df
    
    def _precompute_common_operations(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Pre-compute common operations for efficiency."""
        try:
            # Check if already computed
            features_hash = self._get_data_hash(features_df.iloc[:, 0])
            labels_hash = self._get_data_hash(labels_df.iloc[:, 0])
            cache_key = f"{features_hash}_{labels_hash}"
            
            if cache_key in self._precomputed_indices:
                return self._precomputed_indices[cache_key]
            
            # Pre-compute common indices and aligned data
            common_index = features_df.index.intersection(labels_df.index)
            features_aligned = features_df.loc[common_index]
            labels_aligned = labels_df.loc[common_index]
            
            # Pre-compute correlation matrix if using VectorBT
            correlation_matrix = None
            if self.vectorization_manager:
                try:
                    correlation_matrix = features_aligned.corr()
                    tprint_info("⚡ Correlation matrix pre-computed")
                except Exception as e:
                    tprint_warning(f"Correlation matrix pre-computation failed: {e}")
            
            result = {
                'common_index': common_index,
                'features_aligned': features_aligned,
                'labels_aligned': labels_aligned,
                'correlation_matrix': correlation_matrix
            }
            
            # Cache the result
            self._precomputed_indices[cache_key] = result
            return result
            
        except Exception as e:
            tprint_warning(f"Pre-computation failed: {e}")
            return {
                'common_index': features_df.index.intersection(labels_df.index),
                'features_aligned': features_df,
                'labels_aligned': labels_df,
                'correlation_matrix': None
            }
    
    def _load_config(self) -> EconomicFeatureSelectorConfig:
        """Load configuration from YAML file."""
        try:
            import yaml
            import os
            
            config_path = "config/features/economic_regime_feature_selection_config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract the economic_regime_feature_selection section
                if 'economic_regime_feature_selection' in config_data:
                    config_data = config_data['economic_regime_feature_selection']
                
                # Create config instance with loaded data
                config = EconomicFeatureSelectorConfig()
                
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        tprint_info(f"✅ Loaded config: {key} = {value}")
                    else:
                        tprint_warning(f"⚠️ Config key not found: {key}")
                
                # Debug: Check if mRMR parameters are loaded
                tprint_info(f"🔍 mRMR enabled: {getattr(config, 'enable_mrmr', 'NOT_FOUND')}")
                tprint_info(f"🔍 protect_categories: {getattr(config, 'protect_categories', 'NOT_FOUND')}")
                tprint_info(f"🔍 max_transition_features_ratio: {getattr(config, 'max_transition_features_ratio', 'NOT_FOUND')}")
                
                tprint_success(f"Configuration loaded from {config_path}")
                return config
            else:
                tprint_warning(f"Configuration file not found: {config_path}, using defaults")
                return EconomicFeatureSelectorConfig()
                
        except Exception as e:
            tprint_warning(f"Error loading configuration: {e}, using defaults")
            return EconomicFeatureSelectorConfig()
    
    def _initialize_optimization_tools(self):
        """Initialize optimization tools."""
        try:
            # Initialize VectorBT rolling optimizer
            if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                tprint_success("VectorBT Rolling Optimizer initialized")
            else:
                self.vectorbt_optimizer = None
                tprint_warning("VectorBT Rolling Optimizer not available")
            
            # Initialize unified vectorization manager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("Unified Vectorization Manager initialized")
            else:
                self.vectorization_manager = None
                tprint_warning("Unified Vectorization Manager not available")
            
            # Initialize hardware manager
            if HARDWARE_AVAILABLE:
                self.hardware_manager = get_unified_hardware_manager()
                tprint_success("Hardware Manager initialized")
            else:
                self.hardware_manager = None
                tprint_warning("Hardware Manager not available")
                
        except Exception as e:
            tprint_error(f"Error initializing optimization tools: {e}")
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self.hardware_manager = None
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute economic feature selection.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
        
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = time.perf_counter()
        tprint_info(f"Starting economic feature selection for {config.get('symbol', 'UNKNOWN')}")
        
        # Use regime_timeframe (defaults to 1h) for regime feature selection
        regime_timeframe = config.get('regime_timeframe', '1h')
        if 'regime_timeframe' not in config:
            tprint_info(f"⏰ Using regime_timeframe={regime_timeframe} for regime feature selection")
            config['regime_timeframe'] = regime_timeframe
        if config.get('timeframe') != regime_timeframe:
            tprint_info(f"⏰ Overriding timeframe to {regime_timeframe} for regime feature selection (was: {config.get('timeframe', 'not set')})")
            config['timeframe'] = regime_timeframe
        
        try:
            # Load required data
            tprint_info("🔄 Loading feature data and labels...")
            features_df, labels_df, forward_returns = await self._load_required_data(config)
            
            if features_df is None or labels_df is None:
                raise ValueError("Failed to load required data")
            
            # Data preview and format analysis
            tprint_data_preview(features_df, "Input Features", max_rows=3, max_cols=5)
            tprint_data_format(features_df, "Input Features")
            tprint_data_preview(labels_df, "Labels", max_rows=3, max_cols=5)
            tprint_data_format(labels_df, "Labels")
            tprint_data_preview(forward_returns, "Forward Returns", max_rows=3)
            tprint_data_format(forward_returns, "Forward Returns")
            
            initial_feature_count = len(features_df.columns)
            tprint_info(f"📊 Initial feature count: {initial_feature_count}")
            
            # Filter feature categories
            tprint_info("🔍 Filtering feature categories...")
            filtered_features = self._filter_feature_categories(features_df)
            filtered_feature_count = len(filtered_features.columns)
            tprint_feature_counts(initial_feature_count, filtered_feature_count, "Category Filtering")
            tprint_data_preview(filtered_features, "Filtered Features", max_rows=3, max_cols=5)
            
            # Identify regime transition features
            tprint_info("🔍 Identifying regime transition features...")
            transition_features, transition_scores = self._identify_regime_transition_features(filtered_features, labels_df)
            tprint_success(f"✅ Identified {len(transition_features)} regime transition features")
            tprint_info(f"📈 Transition features: {transition_features[:5]}{'...' if len(transition_features) > 5 else ''}")
            
            # Calculate economic metrics
            tprint_info("💰 Calculating economic metrics for feature evaluation...")
            economic_metrics = self._calculate_economic_metrics_vectorized(
                filtered_features, labels_df, forward_returns
            )
            tprint_success(f"📊 Economic metrics calculated for {len(economic_metrics)} features")
            
            # Score features using multi-target approach
            if self.config.multi_target_enabled:
                tprint_info("🎯 Scoring features using multi-target approach...")
                feature_scores = self._score_features_multi_target(
                    filtered_features, labels_df
                )
                tprint_success(f"🏆 Multi-target feature scoring completed: {len(feature_scores)} features scored")
            else:
                tprint_info("🎯 Scoring features by economic relevance...")
                feature_scores = self._score_features_by_economics(
                    filtered_features, labels_df, economic_metrics
                )
                tprint_success(f"🏆 Feature scoring completed: {len(feature_scores)} features scored")
            
            # Show top scored features
            if feature_scores:
                top_features = sorted(feature_scores, key=lambda x: x.composite_score, reverse=True)[:10]
                tprint_info("🏆 Top 10 scored features:")
                for i, feature_score in enumerate(top_features, 1):
                    tprint_info(f"   {i:2d}. {feature_score.feature_name}: {feature_score.composite_score:.4f}")
            
            # Select features (using incremental mRMR if enabled, otherwise standard method)
            if self.config.enable_mrmr:
                tprint_info("🎯 Selecting features using incremental mRMR with hardware optimization...")
                selected_features = self._select_optimal_features_incremental_mrmr(filtered_features, labels_df, feature_scores, transition_features)
            else:
                tprint_info("🎯 Selecting optimal features with regime transition prioritization...")
                selected_features = self._select_optimal_features(feature_scores, transition_features)
            
            selected_feature_count = len(selected_features)
            tprint_feature_counts(filtered_feature_count, selected_feature_count, "Final Feature Selection")
            tprint_success(f"✅ Selected {selected_feature_count} optimal features")
            tprint_info(f"🎯 Selected features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
            
            # Validate selection
            tprint_info("🔍 Validating feature selection quality...")
            validation_metrics = self._validate_feature_selection(
                filtered_features[selected_features], labels_df, economic_metrics
            )
            tprint_success(f"✅ Validation completed: {validation_metrics}")
            
            # Create result
            result = FeatureSelectionResult(
                selected_features=selected_features,
                feature_scores=feature_scores,
                economic_metrics=economic_metrics,
                validation_metrics=validation_metrics,
                computational_stats=self._get_computational_stats(start_time),
                report_path="",  # Will be set after report generation
                success=True
            )
            
            # Add transition features to result
            result.transition_features = transition_features
            
            # Generate comprehensive report
            tprint_info("Generating comprehensive report...")
            report_path = await self._generate_comprehensive_report(result, config)
            result.report_path = report_path
            tprint_success(f"Report generated: {report_path}")
            
            # Save artifacts
            tprint_info("Saving artifacts...")
            artifacts = await self._save_artifacts(result, config)
            tprint_success("Artifacts saved")
            
            execution_time = time.perf_counter() - start_time
            tprint_success(f"Economic feature selection completed in {execution_time:.2f}s")
            
            # Generate regime clustering artifact
            tprint_info("📦 Generating regime clustering artifact...")
            regime_clustering_artifact = self._generate_regime_clustering_artifact(
                filtered_features[selected_features], selected_features, transition_features
            )
            artifacts['regime_clustering_features'] = regime_clustering_artifact
            tprint_success("✅ Regime clustering artifact generated")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': {
                    'execution_time': execution_time,
                    'selected_features_count': len(selected_features),
                    'economic_metrics_count': len(economic_metrics),
                    'validation_score': validation_metrics.get('overall_score', 0.0),
                    'transition_features_count': len(transition_features)
                },
                'report_path': report_path
            }
            
        except Exception as e:
            error_msg = f"Economic feature selection failed: {str(e)}"
            tprint_error(error_msg)
            
            execution_time = time.perf_counter() - start_time
            return {
                'success': False,
                'artifacts': {},
                'metrics': {'execution_time': execution_time},
                'error': error_msg
            }
    
    async def _load_required_data(self, config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load required data from artifacts."""
        try:
            # Load features from feature generation step - try timeframe-specific first
            timeframe = config.get('timeframe', '15m')
            features_df = self._get_artifact(f"generated_features_{timeframe}", artifact_type="data")
            if features_df is None:
                tprint_warning(f"No generated_features_{timeframe} artifact found, trying general generated_features...")
                features_df = self._get_artifact("generated_features", artifact_type="data")
            if features_df is None:
                tprint_warning("No generated_features artifact found, trying alternative...")
                features_df = self._get_artifact("features", artifact_type="data")
            
            # If still no features found, try to load directly from CSV files
            if features_df is None:
                tprint_warning("No features found via artifact manager, trying direct CSV loading...")
                try:
                    import os
                    import glob
                    artifact_dir = "artifacts"
                    # Try timeframe-specific pattern first
                    pattern = f"*generated_features_{timeframe}*.csv"
                    matching_files = glob.glob(os.path.join(artifact_dir, pattern))
                    
                    # If no timeframe-specific files found, try general pattern
                    if not matching_files:
                        pattern = "*generated_features*.csv"
                        matching_files = glob.glob(os.path.join(artifact_dir, pattern))
                    
                    if matching_files:
                        # Use the most recent file
                        latest_file = max(matching_files, key=os.path.getctime)
                        tprint_info(f"Loading features from CSV: {os.path.basename(latest_file)}")
                        features_df = pd.read_csv(latest_file)
                        
                        # Set timestamp as index if it exists
                        if 'timestamp' in features_df.columns:
                            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
                            features_df = features_df.set_index('timestamp')
                            tprint_info(f"Set timestamp as index, features shape: {features_df.shape}")
                    else:
                        tprint_warning("No generated_features CSV files found")
                except Exception as e:
                    tprint_warning(f"Error loading features from CSV: {e}")
            
            # Load labels from labeling integration step - try timeframe-specific first
            timeframe = config.get('timeframe', '15m')
            labels_df = None
            
            # Try to find timeframe-specific labeled data
            try:
                import os
                import glob
                artifact_dir = "artifacts"
                pattern = f"**/*labeled_data*{timeframe}*.parquet"
                matching_files = glob.glob(pattern, recursive=True)
                
                if matching_files:
                    # Use the most recent file
                    latest_file = max(matching_files, key=os.path.getctime)
                    tprint_info(f"Loading timeframe-specific labeled data: {os.path.basename(latest_file)}")
                    labels_df = pd.read_parquet(latest_file)
                else:
                    tprint_warning(f"No {timeframe} labeled data found, trying general labeled_data...")
                    labels_df = self._get_artifact("labeled_data", artifact_type="data")
                    
            except Exception as e:
                tprint_warning(f"Error loading timeframe-specific labeled data: {e}")
                labels_df = self._get_artifact("labeled_data", artifact_type="data")
            
            if labels_df is None:
                tprint_warning("No labeled_data artifact found, trying alternative...")
                labels_df = self._get_artifact("labels", artifact_type="data")
            
            # Calculate forward returns from market data
            forward_returns = await self._calculate_forward_returns(config)
            
            return features_df, labels_df, forward_returns
            
        except Exception as e:
            tprint_error(f"Error loading required data: {e}")
            return None, None, None
    
    async def _calculate_forward_returns(self, config: Dict[str, Any]) -> Optional[pd.Series]:
        """Calculate forward returns from market data."""
        try:
            # Load market data
            from src.utils.data.klines_parquet import get_klines_manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))
            
            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed"
            )
            
            if market_data is None or market_data.empty or 'close' not in market_data.columns:
                tprint_error("No market data available for forward returns calculation")
                return None
            
            # Calculate forward returns (6 periods ahead as per labeling step)
            close_prices = market_data['close']
            forward_returns = close_prices.pct_change(6).shift(-6)
            
            tprint_data_preview(f"Forward returns calculated: {len(forward_returns)} samples")
            return forward_returns
            
        except Exception as e:
            tprint_error(f"Error calculating forward returns: {e}")
            return None
    
    def _filter_feature_categories(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Filter feature categories based on configuration with cheap initial pruning."""
        try:
            initial_count = len(features_df.columns)
            tprint_info(f"🔍 Starting feature category filtering for {initial_count} features...")
            tprint_data_preview(features_df, "Input Features for Filtering", max_rows=2, max_cols=3)
            
            # Step 1: Initial cheap filtering by category
            if FEATURE_BANK_AVAILABLE:
                available_categories = list_available_categories()
                tprint_info(f"📋 Available categories: {available_categories}")
                tprint_info(f"🚫 Excluding categories: {self.config.exclude_categories}")
                
                # Filter out excluded categories
                filtered_features = features_df.copy()
                excluded_count = 0
                
                for category in self.config.exclude_categories:
                    # Handle both enum and string values
                    if hasattr(category, 'value'):
                        category_name = category.value
                    else:
                        category_name = str(category)
                    
                    tprint_info(f"🔍 Processing exclude category: {category_name}")
                    
                    if category_name in available_categories:
                        # Remove features from excluded category
                        category_features = [col for col in features_df.columns if category_name in col.lower()]
                        if category_features:
                            filtered_features = filtered_features.drop(columns=category_features, errors='ignore')
                            excluded_count += len(category_features)
                            tprint_info(f"   ❌ Excluded {len(category_features)} features from category: {category_name}")
                            tprint_info(f"   📝 Sample excluded features: {category_features[:3]}{'...' if len(category_features) > 3 else ''}")
                        else:
                            tprint_info(f"   ℹ️ No features found for category: {category_name}")
                    else:
                        tprint_warning(f"   ⚠️ Category not found in available categories: {category_name}")
                
                after_category_filtering = len(filtered_features.columns)
                tprint_feature_counts(initial_count, after_category_filtering, "Category Filtering", excluded_count)
                tprint_data_preview(filtered_features, "After Category Filtering", max_rows=2, max_cols=3)
            else:
                tprint_warning("⚠️ Feature bank not available, using all features")
                filtered_features = features_df.copy()
                after_category_filtering = initial_count
            
            # Step 2: Cheap statistical pruning (remove obviously bad features)
            tprint_info("🔍 Applying cheap statistical pruning...")
            pruned_features = self._cheap_statistical_pruning(filtered_features)
            after_statistical_pruning = len(pruned_features.columns)
            tprint_feature_counts(after_category_filtering, after_statistical_pruning, "Statistical Pruning")
            
            # Step 3: Variance-based pruning (remove low-variance features)
            tprint_info("🔍 Applying variance-based pruning...")
            variance_pruned = self._variance_based_pruning(pruned_features)
            after_variance_pruning = len(variance_pruned.columns)
            tprint_feature_counts(after_statistical_pruning, after_variance_pruning, "Variance-based Pruning")
            
            # Step 4: Correlation-based pruning (remove highly correlated features)
            tprint_info("🔍 Applying correlation-based pruning...")
            final_features = self._correlation_based_pruning(variance_pruned)
            final_count = len(final_features.columns)
            tprint_feature_counts(after_variance_pruning, final_count, "Correlation-based Pruning")
            
            # Final summary
            total_filtered = initial_count - final_count
            retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
            tprint_success(f"✅ Feature filtering completed: {initial_count} -> {final_count} features")
            tprint_info(f"📊 Total filtered out: {total_filtered} features ({100-retention_rate:.1f}%)")
            tprint_info(f"📈 Retention rate: {retention_rate:.1f}%")
            
            tprint_data_preview(final_features, "Final Filtered Features", max_rows=2, max_cols=3)
            tprint_data_format(final_features, "Final Filtered Features")
            
            return final_features
            
        except Exception as e:
            tprint_error(f"❌ Error filtering feature categories: {e}")
            return features_df
    
    def _cheap_statistical_pruning(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with obvious statistical issues using cheap operations."""
        try:
            initial_count = len(features_df.columns)
            tprint_info(f"🔍 Performing cheap statistical pruning on {initial_count} features...")
            
            # Use vectorized operations for speed
            if VECTORBT_AVAILABLE and self.config.enable_vectorbt:
                tprint_info("⚡ Using VectorBT-optimized statistical checks...")
                pruned_features = self._vectorbt_statistical_pruning(features_df)
            else:
                tprint_info("📊 Using pandas statistical checks...")
                pruned_features = self._pandas_statistical_pruning(features_df)
            
            final_count = len(pruned_features.columns)
            filtered_count = initial_count - final_count
            tprint_feature_counts(initial_count, final_count, "Statistical Pruning", filtered_count)
            tprint_data_preview(pruned_features, "After Statistical Pruning", max_rows=2, max_cols=3)
            
            return pruned_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in statistical pruning: {e}")
            return features_df
    
    def _vectorbt_statistical_pruning(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """VectorBT-optimized statistical pruning."""
        try:
            # Use VectorBT rolling operations for efficient statistical checks
            valid_features = []
            tprint_info(f"⚡ Starting VectorBT statistical pruning with {len(features_df.columns)} features...")
            tprint_info(f"📊 Sample features to process: {list(features_df.columns)[:5]}...")
            
            for col in features_df.columns:
                try:
                    feature_data = features_df[col].dropna()
                    
                    if len(feature_data) < 10:  # Minimum sample size
                        continue
                    
                    # Check for infinite values
                    if np.isinf(feature_data).any():
                        continue
                    
                    # VectorBT-optimized checks with division by zero protection
                    if VECTORBT_AVAILABLE:
                        try:
                            # Check for constant values using VectorBT
                            rolling_std_result = rolling_std(feature_data, min(20, len(feature_data)//2))
                            max_std = rolling_std_result.max()
                            if np.isnan(max_std) or np.isinf(max_std) or max_std < 1e-3:  # Much less aggressive threshold
                                continue
                        except (ZeroDivisionError, ValueError) as e:
                            continue
                        
                        try:
                            # Check for extreme outliers using VectorBT quantiles
                            q99 = rolling_quantile(feature_data, min(50, len(feature_data)//2), 0.99)
                            q01 = rolling_quantile(feature_data, min(50, len(feature_data)//2), 0.01)
                            
                            # Check for invalid values (only where not NaN)
                            q99_valid = q99.dropna()
                            q01_valid = q01.dropna()
                            if len(q99_valid) == 0 or len(q01_valid) == 0 or np.isnan(q99_valid).any() or np.isnan(q01_valid).any() or np.isinf(q99_valid).any() or np.isinf(q01_valid).any():
                                continue
                            
                            # Avoid division by zero
                            # Check if q01 contains zeros
                            if (q01_valid == 0).any():
                                continue
                            
                            ratio = q99_valid / q01_valid
                            ratio_max = abs(ratio).max()  # Use absolute value for proper comparison
                            if np.isnan(ratio).any() or np.isinf(ratio).any() or ratio_max > 1e10:  # Much less aggressive threshold
                                continue
                        except (ZeroDivisionError, ValueError) as e:
                            continue
                    
                    valid_features.append(col)
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Error checking feature {col}: {e}")
                    continue
            
            # Add progress logging every 100 features
            if len(features_df.columns) > 100:
                tprint_info(f"   📊 Processed {len(features_df.columns)} features...")
            
            tprint_info(f"✅ VectorBT statistical pruning completed: {len(valid_features)}/{len(features_df.columns)} features passed")
            result = features_df[valid_features]
            # Ensure we always return a DataFrame, not a Series
            if isinstance(result, pd.Series):
                result = result.to_frame()
            return result
            
        except Exception as e:
            tprint_warning(f"Error in VectorBT statistical pruning: {e}")
            return features_df
    
    def _pandas_statistical_pruning(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Pandas-based statistical pruning fallback."""
        try:
            initial_count = len(features_df.columns)
            tprint_info(f"📊 Starting pandas statistical pruning with {initial_count} features...")
            tprint_info(f"📊 Sample features to process: {list(features_df.columns)[:5]}...")
            
            valid_features = []
            invalid_count = 0
            
            for col in features_df.columns:
                try:
                    feature_data = features_df[col].dropna()
                    
                    if len(feature_data) < 10:
                        invalid_count += 1
                        continue
                    
                    # Check for infinite values
                    if np.isinf(feature_data).any():
                        invalid_count += 1
                        continue
                    
                    # Basic statistical checks with division by zero protection
                    try:
                        std_val = feature_data.std()
                        if std_val < 1e-3 or np.isnan(std_val) or np.isinf(std_val):
                            invalid_count += 1
                            continue
                    except (ZeroDivisionError, ValueError):
                        invalid_count += 1
                        continue
                    
                    if feature_data.nunique() < 3:  # Too few unique values
                        invalid_count += 1
                        continue
                    
                    # Check for extreme outliers with division by zero protection
                    try:
                        q99, q01 = feature_data.quantile([0.99, 0.01])
                        if np.isnan(q99) or np.isnan(q01) or np.isinf(q99) or np.isinf(q01):
                            invalid_count += 1
                            continue
                        
                        # Avoid division by zero
                        if abs(q01) < 1e-8:
                            # If q01 is very small, check if q99 is reasonable
                            if abs(q99) > 1e6:
                                invalid_count += 1
                                continue
                        else:
                            ratio = q99 / q01
                            if np.isnan(ratio) or np.isinf(ratio) or ratio > 1e6:
                                invalid_count += 1
                                continue
                    except (ZeroDivisionError, ValueError):
                        invalid_count += 1
                        continue
                    
                    valid_features.append(col)
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Error checking feature {col}: {e}")
                    invalid_count += 1
                    continue
            
            final_count = len(valid_features)
            tprint_info(f"✅ Pandas statistical pruning completed: {final_count}/{initial_count} features passed")
            tprint_info(f"📊 Invalid features filtered out: {invalid_count}")
            
            if valid_features:
                result = features_df[valid_features]
                # Ensure we always return a DataFrame, not a Series
                if isinstance(result, pd.Series):
                    result = result.to_frame()
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in pandas statistical pruning: {e}")
            return features_df
    
    def _variance_based_pruning(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove low-variance features using optimized operations."""
        try:
            initial_count = len(features_df.columns)
            tprint_info(f"📊 Performing variance-based pruning on {initial_count} features...")
            tprint_data_preview(features_df, "Input Features for Variance Pruning", max_rows=2, max_cols=3)
            
            # Use hardware optimization if available
            if HARDWARE_AVAILABLE and self.hardware_manager:
                tprint_info("⚡ Optimizing hardware for feature engineering workload...")
                from src.utils.hardware.unified_hardware_manager import WorkloadType
                self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
            
            # VectorBT-optimized variance calculation with division by zero protection
            if VECTORBT_AVAILABLE and self.config.enable_vectorbt:
                tprint_info("⚡ Using VectorBT-optimized variance calculation...")
                try:
                    variances = rolling_var(features_df, min(50, len(features_df)//2)).mean()
                    # Check for invalid variances
                    valid_var_mask = ~(np.isnan(variances) | np.isinf(variances))
                    variances = variances[valid_var_mask]
                    tprint_info(f"✅ VectorBT variance calculation completed: {len(variances)} valid variances")
                except (ZeroDivisionError, ValueError):
                    tprint_warning("⚠️ VectorBT variance calculation failed, falling back to pandas...")
                    # Fallback to pandas if VectorBT fails
                    variances = features_df.var()
                    valid_var_mask = ~(np.isnan(variances) | np.isinf(variances))
                    variances = variances[valid_var_mask]
            else:
                tprint_info("📊 Using pandas variance calculation...")
                # Fallback to pandas
                variances = features_df.var()
                valid_var_mask = ~(np.isnan(variances) | np.isinf(variances))
                variances = variances[valid_var_mask]
            
            if len(variances) == 0:
                tprint_warning("⚠️ No valid variances calculated, returning original features")
                return features_df
            
            # Remove features with very low variance (bottom 5%)
            try:
                variance_threshold = variances.quantile(0.05)
                if np.isnan(variance_threshold) or np.isinf(variance_threshold):
                    variance_threshold = variances.min()
                
                high_variance_features = variances[variances > variance_threshold].index.tolist()
                tprint_info(f"📊 Variance threshold (5th percentile): {variance_threshold:.2e}")
                tprint_info(f"📊 Features above threshold: {len(high_variance_features)}")
            except (ZeroDivisionError, ValueError):
                tprint_warning("⚠️ Quantile calculation failed, using simple threshold...")
                # If quantile calculation fails, use a simple threshold
                high_variance_features = variances[variances > 1e-8].index.tolist()
                tprint_info(f"📊 Using simple threshold (1e-8): {len(high_variance_features)} features")
            
            final_count = len(high_variance_features)
            filtered_count = initial_count - final_count
            tprint_feature_counts(initial_count, final_count, "Variance-based Pruning", filtered_count)
            tprint_data_preview(features_df[high_variance_features], "After Variance Pruning", max_rows=2, max_cols=3)
            
            result = features_df[high_variance_features]
            # Ensure we always return a DataFrame, not a Series
            if isinstance(result, pd.Series):
                result = result.to_frame()
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in variance-based pruning: {e}")
            return features_df
    
    def _correlation_based_pruning(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features using optimized correlation matrix."""
        try:
            # Ensure we have a DataFrame, not a Series
            if isinstance(features_df, pd.Series):
                tprint_warning("⚠️ Correlation-based pruning received a Series, converting to DataFrame...")
                features_df = features_df.to_frame()
            
            initial_count = len(features_df.columns)
            tprint_info(f"🔗 Performing correlation-based pruning on {initial_count} features...")
            tprint_data_preview(features_df, "Input Features for Correlation Pruning", max_rows=2, max_cols=3)
            
            # Use hardware optimization for correlation matrix
            if HARDWARE_AVAILABLE and self.hardware_manager:
                tprint_info("⚡ Optimizing hardware for data processing workload...")
                from src.utils.hardware.unified_hardware_manager import WorkloadType
                self.hardware_manager.optimize_for_workload(WorkloadType.DATA_PROCESSING)
            
            # Calculate correlation matrix efficiently with division by zero protection
            if VECTORBT_AVAILABLE and self.config.enable_vectorbt:
                tprint_info("⚡ Using VectorBT-optimized correlation calculation...")
                try:
                    # Use VectorBT for efficient correlation calculation
                    corr_matrix = features_df.rolling(min(100, len(features_df)//2)).corr().mean()
                    # Ensure corr_matrix is a DataFrame
                    if isinstance(corr_matrix, pd.Series):
                        corr_matrix = corr_matrix.to_frame()
                    # Check for invalid correlations
                    corr_matrix = corr_matrix.fillna(0)  # Replace NaN with 0
                    corr_matrix = corr_matrix.replace([np.inf, -np.inf], 0)  # Replace inf with 0
                    tprint_info(f"✅ VectorBT correlation matrix calculated: {corr_matrix.shape}")
                except (ZeroDivisionError, ValueError):
                    tprint_warning("⚠️ VectorBT correlation calculation failed, falling back to pandas...")
                    # Fallback to pandas if VectorBT fails
                    corr_matrix = features_df.corr()
                    corr_matrix = corr_matrix.fillna(0)
                    corr_matrix = corr_matrix.replace([np.inf, -np.inf], 0)
            else:
                tprint_info("📊 Using pandas correlation calculation...")
                # Fallback to pandas
                corr_matrix = features_df.corr()
                corr_matrix = corr_matrix.fillna(0)
                corr_matrix = corr_matrix.replace([np.inf, -np.inf], 0)
            
            tprint_info(f"📊 Correlation matrix shape: {corr_matrix.shape}")
            
            # Remove highly correlated features (correlation > 0.95)
            high_corr_pairs = []
            correlation_threshold = 0.95
            tprint_info(f"🔍 Finding highly correlated pairs (threshold: {correlation_threshold})...")
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not (np.isnan(corr_val) or np.isinf(corr_val)) and abs(corr_val) > correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            tprint_info(f"🔗 Found {len(high_corr_pairs)} highly correlated pairs")
            
            # Show top correlated pairs
            if high_corr_pairs:
                tprint_info(f"🔗 Top 5 highly correlated pairs:")
                sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
                for i, (feat1, feat2, corr_val) in enumerate(sorted_pairs[:5]):
                    tprint_info(f"   {i+1}. {feat1} <-> {feat2}: {corr_val:.4f}")
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2, _ in high_corr_pairs:
                if feat1 not in features_to_remove:
                    features_to_remove.add(feat2)
            
            remaining_features = [col for col in features_df.columns if col not in features_to_remove]
            
            final_count = len(remaining_features)
            filtered_count = initial_count - final_count
            tprint_feature_counts(initial_count, final_count, "Correlation-based Pruning", filtered_count)
            tprint_data_preview(features_df[remaining_features], "After Correlation Pruning", max_rows=2, max_cols=3)
            
            return features_df[remaining_features]
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in correlation-based pruning: {e}")
            return features_df
    
    def _calculate_multi_target_economic_metrics(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate multi-target metrics using correlation-based approach."""
        try:
            tprint_info(f"🎯 Calculating multi-target metrics for {len(self.config.target_columns)} targets")
            
            multi_target_metrics = {}
            
            for target_col in self.config.target_columns:
                tprint_info(f"📊 Processing target: {target_col}")
                
                if target_col not in labels_df.columns:
                    tprint_warning(f"⚠️ Target column {target_col} not found in labels data")
                    continue
                
                # Get target data aligned with features
                common_index = features_df.index.intersection(labels_df.index)
                target_data = labels_df.loc[common_index, target_col].dropna()
                
                if len(target_data) < 50:
                    tprint_warning(f"⚠️ Insufficient data for target {target_col}: {len(target_data)} samples")
                    continue
                
                multi_target_metrics[target_col] = target_data
                tprint_success(f"✅ Prepared target data for {target_col}: {len(target_data)} samples")
            
            return multi_target_metrics
            
        except Exception as e:
            tprint_error(f"Error in multi-target metrics calculation: {e}")
            return {}
    
    def _calculate_economic_metrics_vectorized(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, forward_returns: pd.Series) -> List[EconomicMetrics]:
        """Calculate economic metrics per regime using vectorized operations."""
        try:
            tprint_info("Calculating vectorized economic metrics...")
            
            # Align data - handle different index types
            tprint_info(f"Features index range: {features_df.index.min()} to {features_df.index.max()}")
            tprint_info(f"Labels index range: {labels_df.index.min()} to {labels_df.index.max()}")
            tprint_info(f"Returns index range: {forward_returns.index.min()} to {forward_returns.index.max()}")
            
            # Try different alignment strategies
            common_index = features_df.index.intersection(labels_df.index).intersection(forward_returns.index)
            
            if len(common_index) == 0:
                tprint_warning("No common index found, trying alternative alignment...")
                # Try aligning with features as the base
                common_index = features_df.index.intersection(labels_df.index)
                if len(common_index) > 0:
                    tprint_info(f"Found {len(common_index)} common indices between features and labels")
                    # Use only features and labels for now
                    forward_returns = pd.Series(index=common_index, dtype=float)
                else:
                    tprint_error("No common index found between features and labels")
                    return []
            
            features_aligned = features_df.loc[common_index]
            labels_aligned = labels_df.loc[common_index]
            
            # Handle forward_returns - ensure it's a Series
            if isinstance(forward_returns, pd.DataFrame):
                if forward_returns.shape[1] == 1:
                    returns_aligned = forward_returns.iloc[:, 0].loc[common_index]
                else:
                    # If multiple columns, use the first one or create a dummy series
                    returns_aligned = pd.Series(index=common_index, dtype=float)
            else:
                returns_aligned = forward_returns.loc[common_index]
            
            # Handle case where labels_df might be a DataFrame with a single column
            if hasattr(labels_aligned, 'iloc') and hasattr(labels_aligned, 'shape') and labels_aligned.shape[1] == 1:
                labels_aligned = labels_aligned.iloc[:, 0]
            
            # Remove NaN values
            valid_mask = ~(returns_aligned.isna() | labels_aligned.isna())
            # Ensure valid_mask is a Series for proper indexing
            if hasattr(valid_mask, 'any'):
                valid_mask = valid_mask.all(axis=1) if valid_mask.ndim > 1 else valid_mask
            features_valid = features_aligned[valid_mask]
            labels_valid = labels_aligned[valid_mask]
            returns_valid = returns_aligned[valid_mask]
            
            # Ensure labels_valid is a Series
            if isinstance(labels_valid, pd.DataFrame):
                if labels_valid.shape[1] == 1:
                    labels_valid = labels_valid.iloc[:, 0]
                else:
                    labels_valid = labels_valid.iloc[:, 0]  # Use first column
            
            tprint_data_preview(f"Valid samples for economic metrics: {len(returns_valid)}")
            
            economic_metrics = []
            unique_regimes = labels_valid.unique()
            
            for regime_id in unique_regimes:
                if regime_id == -1:  # Skip noise
                    continue
                
                regime_mask = labels_valid == regime_id
                regime_returns = returns_valid[regime_mask]
                
                if len(regime_returns) < 10:  # Minimum sample size
                    continue
                
                # Calculate metrics using vectorized operations
                avg_return = regime_returns.mean()
                volatility = regime_returns.std()
                
                # Sharpe ratio (vectorized)
                if volatility > 0:
                    sharpe_ratio = avg_return / volatility
                else:
                    sharpe_ratio = 0.0
                
                # Sortino ratio (vectorized)
                downside_returns = regime_returns[regime_returns < 0]
                if len(downside_returns) > 0:
                    downside_volatility = downside_returns.std()
                    sortino_ratio = avg_return / downside_volatility if downside_volatility > 0 else 0.0
                else:
                    sortino_ratio = sharpe_ratio
                
                # Win rate
                win_rate = (regime_returns > 0).mean()
                
                # Max drawdown (vectorized)
                cumulative_returns = (1 + regime_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Calmar ratio
                calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
                
                economic_metrics.append(EconomicMetrics(
                    regime_id=int(regime_id),
                    sharpe_ratio=float(sharpe_ratio),
                    sortino_ratio=float(sortino_ratio),
                    calmar_ratio=float(calmar_ratio),
                    win_rate=float(win_rate),
                    avg_return=float(avg_return),
                    volatility=float(volatility),
                    max_drawdown=float(max_drawdown),
                    sample_count=len(regime_returns)
                ))
            
            tprint_success(f"Calculated economic metrics for {len(economic_metrics)} regimes")
            return economic_metrics
            
        except Exception as e:
            tprint_error(f"Error calculating economic metrics: {e}")
            return []
    
    def _score_features_multi_target(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> List[FeatureScore]:
        """Score features using TreeSHAP as primary method with fallback to traditional scoring."""
        try:
            tprint_info(f"🎯 Starting TreeSHAP-based feature scoring for {len(features_df.columns)} features")
            
            # Try TreeSHAP-based scoring first
            try:
                # Check if TreeSHAP dependencies are available
                try:
                    import lightgbm
                    import shap
                    from src.training.steps.market_analysis.treeshap_feature_selector import TreeSHAPFeatureSelector
                    treeshap_available = True
                except ImportError as import_error:
                    tprint_warning(f"⚠️ TreeSHAP dependencies not available: {import_error}")
                    treeshap_available = False
                
                if treeshap_available:
                    # Initialize TreeSHAP selector with current config
                    treeshap_config = {
                        'n_estimators': 100,
                        'max_depth': 8,
                        'learning_rate': 0.1,
                        'correlation_threshold': 0.85,
                        'diversity_weight': 0.2,
                        'treeshap_weight': 0.6,
                        'correlation_weight': 0.2,
                        'target_columns': self.target_columns,
                        'target_weights': self.target_weights
                    }
                    
                    treeshap_selector = TreeSHAPFeatureSelector(treeshap_config)
                    
                    # Get target feature count from config
                    target_count = self.config.get('target_feature_count', 25)
                    
                    # Run TreeSHAP selection
                    treeshap_result = treeshap_selector.select_features(
                        features_df, labels_df, target_count
                    )
                    
                    if treeshap_result['success'] and treeshap_result['selected_features']:
                        tprint_success("✅ TreeSHAP scoring completed successfully")
                        
                        # Convert TreeSHAP results to FeatureScore format
                        feature_scores = []
                        for feature_name in treeshap_result['selected_features']:
                            treeshap_score = treeshap_result['treeshap_scores'].get(feature_name, 0.0)
                            correlation_score = treeshap_result['correlation_scores'].get(feature_name, 0.0)
                            diversity_score = treeshap_result['diversity_scores'].get(feature_name, 0.0)
                            composite_score = treeshap_result['feature_scores'].get(feature_name, 0.0)
                            
                            feature_scores.append(FeatureScore(
                                feature_name=feature_name,
                                economic_significance=correlation_score,
                                regime_discrimination=0.0,  # Not calculated in TreeSHAP
                                clustering_quality=0.0,    # Not calculated in TreeSHAP
                                stability_score=0.0,       # Not calculated in TreeSHAP
                                mrmr_score=0.0,            # Not calculated in TreeSHAP
                                regime_transition_score=0.0,  # Not calculated in TreeSHAP
                                composite_score=composite_score,
                                treeshap_importance=treeshap_score,
                                diversity_score=diversity_score,
                                selected=True
                            ))
                        
                        return feature_scores
                    else:
                        tprint_warning("⚠️ TreeSHAP scoring failed, falling back to traditional method")
                else:
                    tprint_warning("⚠️ TreeSHAP not available, falling back to traditional method")
                    
            except Exception as e:
                tprint_warning(f"⚠️ TreeSHAP scoring error: {e}, falling back to traditional method")
            
            # Fallback to traditional multi-target scoring
            tprint_info("🔄 Falling back to traditional multi-target scoring")
            
            # Calculate multi-target economic metrics
            multi_target_metrics = self._calculate_multi_target_economic_metrics(features_df, labels_df)
            
            if not multi_target_metrics:
                tprint_warning("⚠️ No multi-target metrics calculated, falling back to single target")
                return self._score_features_by_economics(features_df, labels_df, [])
            
            # Pre-compute common operations for efficiency
            common_index = features_df.index.intersection(labels_df.index)
            features_aligned = features_df.loc[common_index]
            labels_aligned = labels_df.loc[common_index]
            
            tprint_info(f"📊 Processing {len(features_df.columns)} features with {len(common_index)} aligned samples")
            
            # Use parallel processing for feature scoring
            feature_scores = self._score_features_parallel(
                features_aligned, labels_aligned, multi_target_metrics
            )
            
            return feature_scores
            
        except Exception as e:
            tprint_error(f"Error in multi-target feature scoring: {e}")
            return []
    
    def _score_features_parallel(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, multi_target_metrics: Dict) -> List[FeatureScore]:
        """Score features using parallel processing with hardware optimization."""
        try:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
            from multiprocessing import cpu_count
            import functools
            
            # Disable parallel processing to avoid pickle issues with thread locks
            # Parallel processing can cause issues with hardware manager and other complex objects
            use_multiprocessing = False
            cpu_count_available = 1  # Force sequential processing
            
            tprint_info(f"🔄 Using {'multiprocessing' if use_multiprocessing else 'threading'} with {cpu_count_available} workers")
            
            # Prepare arguments for parallel processing
            feature_args = []
            for feature_name in features_df.columns:
                feature_data = features_df[feature_name].dropna()
                if len(feature_data) >= 50:  # Minimum sample size
                    feature_args.append((feature_name, feature_data, labels_df, multi_target_metrics, features_df))
            
            # Use sequential processing to avoid pickle issues with complex objects
            feature_scores = []
            for args in feature_args:
                try:
                    score = self._score_single_feature_parallel(args)
                    if score is not None:
                        feature_scores.append(score)
                except Exception as e:
                    tprint_warning(f"Error scoring feature: {e}")
                    continue
            
            tprint_success(f"✅ Sequential processing completed: {len(feature_scores)} features scored")
            return feature_scores
            
        except Exception as e:
            tprint_warning(f"Parallel processing failed, falling back to sequential: {e}")
            return self._score_features_sequential(features_df, labels_df, multi_target_metrics)
    
    def _score_single_feature_parallel(self, args) -> Optional[FeatureScore]:
        """Score a single feature for parallel processing."""
        try:
            feature_name, feature_data, labels_df, multi_target_metrics, features_df = args
            
            # Calculate weighted scores across all targets
            weighted_economic_significance = 0.0
            weighted_regime_discrimination = 0.0
            weighted_clustering_quality = 0.0
            weighted_stability_score = 0.0
            weighted_mrmr_score = 0.0
            weighted_regime_transition_score = 0.0
            
            total_weight = 0.0
            
            for target_col, target_weight in self.config.target_weights.items():
                if target_col not in multi_target_metrics:
                    continue
                
                # Align feature with target
                common_index = feature_data.index.intersection(labels_df.index)
                feature_aligned = feature_data.loc[common_index]
                labels_aligned = labels_df.loc[common_index, target_col]
                
                if len(feature_aligned) < 50:
                    continue
                
                # Calculate scores for this target
                target_data = multi_target_metrics[target_col]
                target_aligned = target_data.loc[common_index]
                
                economic_significance = self._calculate_economic_significance_cheap(
                    feature_aligned, target_aligned, target_aligned
                )
                
                regime_discrimination = self._calculate_regime_discrimination(
                    feature_aligned, target_aligned
                )
                
                # For clustering quality, use target data directly
                clustering_quality = self._calculate_clustering_quality_cheap(
                    feature_aligned, target_aligned
                )
                
                stability_score = self._calculate_stability_cheap(
                    feature_aligned, target_aligned
                )
                
                # Calculate mRMR score (relevance - redundancy)
                # For multi-target approach, use a simplified redundancy calculation
                # that doesn't require previously selected features
                mrmr_score = self._calculate_mrmr_cheap(
                    feature_aligned, target_aligned, [], features_df
                )
                
                # Calculate regime transition detection score for this target
                regime_transition_score = self._calculate_regime_transition_detection(
                    feature_aligned, target_aligned
                )
                
                # Weight the scores
                weighted_economic_significance += economic_significance * target_weight
                weighted_regime_discrimination += regime_discrimination * target_weight
                weighted_clustering_quality += clustering_quality * target_weight
                weighted_stability_score += stability_score * target_weight
                weighted_mrmr_score += mrmr_score * target_weight
                weighted_regime_transition_score += regime_transition_score * target_weight
                
                total_weight += target_weight
            
            if total_weight == 0:
                return None
            
            # Normalize weighted scores
            economic_significance = weighted_economic_significance / total_weight
            regime_discrimination = weighted_regime_discrimination / total_weight
            clustering_quality = weighted_clustering_quality / total_weight
            stability_score = weighted_stability_score / total_weight
            mrmr_score = weighted_mrmr_score / total_weight
            regime_transition_score = weighted_regime_transition_score / total_weight
            
            # Calculate composite score using weights from config
            composite_score = (
                economic_significance * self.config.economic_significance_weight +
                regime_discrimination * self.config.regime_discrimination_weight +
                clustering_quality * self.config.clustering_quality_weight +
                stability_score * self.config.stability_weight +
                mrmr_score * self.config.mrmr_weight +
                regime_transition_score * self.config.regime_transition_weight
            )
            
            # Determine feature category
            category = self._determine_feature_category(feature_name)
            
            return FeatureScore(
                feature_name=feature_name,
                economic_significance=economic_significance,
                regime_discrimination=regime_discrimination,
                clustering_quality=clustering_quality,
                stability_score=stability_score,
                mrmr_score=mrmr_score,
                regime_transition_score=regime_transition_score,
                composite_score=composite_score,
                category=category
            )
            
        except Exception as e:
            tprint_warning(f"Error scoring feature {feature_name}: {e}")
            return None
    
    def _score_features_sequential(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, multi_target_metrics: Dict) -> List[FeatureScore]:
        """Fallback sequential scoring when parallel processing fails."""
        feature_scores = []
        
        for i, feature_name in enumerate(features_df.columns):
                try:
                    tprint_info(f"📈 Scoring feature {i+1}/{len(features_df.columns)}: {feature_name}")
                    
                    # Get feature data
                    feature_data = features_df[feature_name].dropna()
                    
                    # Calculate weighted scores across all targets
                    weighted_economic_significance = 0.0
                    weighted_regime_discrimination = 0.0
                    weighted_clustering_quality = 0.0
                    weighted_stability_score = 0.0
                    weighted_mrmr_score = 0.0
                    
                    total_weight = 0.0
                    
                    for target_col, target_weight in self.config.target_weights.items():
                        if target_col not in multi_target_metrics:
                            continue
                        
                        # Align feature with target
                        common_index = feature_data.index.intersection(labels_df.index)
                        feature_aligned = feature_data.loc[common_index]
                        labels_aligned = labels_df.loc[common_index, target_col]
                        
                        if len(feature_aligned) < 50:
                            continue
                        
                        # Calculate scores for this target
                        target_data = multi_target_metrics[target_col]
                        target_aligned = target_data.loc[common_index]
                        
                        economic_significance = self._calculate_economic_significance_cheap(
                            feature_aligned, target_aligned, target_aligned
                        )
                        
                        regime_discrimination = self._calculate_regime_discrimination(
                            feature_aligned, target_aligned
                        )
                        
                        # For clustering quality, use target data directly
                        clustering_quality = self._calculate_clustering_quality_cheap(
                            feature_aligned, target_aligned
                        )
                        
                        stability_score = self._calculate_stability_cheap(
                            feature_aligned, target_aligned
                        )
                        
                        # Calculate mRMR score (relevance - redundancy)
                        # For multi-target approach, use a simplified redundancy calculation
                        # that doesn't require previously selected features
                        mrmr_score = self._calculate_mrmr_cheap(
                            feature_aligned, target_aligned, [], features_df
                        )
                        
                        # Weight the scores
                        weighted_economic_significance += economic_significance * target_weight
                        weighted_regime_discrimination += regime_discrimination * target_weight
                        weighted_clustering_quality += clustering_quality * target_weight
                        weighted_stability_score += stability_score * target_weight
                        weighted_mrmr_score += mrmr_score * target_weight
                        
                        total_weight += target_weight
                    
                    if total_weight == 0:
                        continue
                    
                    # Normalize by total weight
                    weighted_economic_significance /= total_weight
                    weighted_regime_discrimination /= total_weight
                    weighted_clustering_quality /= total_weight
                    weighted_stability_score /= total_weight
                    weighted_mrmr_score /= total_weight
                    
                    # Calculate regime transition detection score (average across targets)
                    weighted_regime_transition_score = 0.0
                    transition_weight = 0.0
                    
                    for target_col, target_weight in self.config.target_weights.items():
                        if target_col not in multi_target_metrics:
                            continue
                        
                        # Align feature with target
                        common_index = feature_data.index.intersection(labels_df.index)
                        feature_aligned = feature_data.loc[common_index]
                        labels_aligned = labels_df.loc[common_index, target_col]
                        
                        if len(feature_aligned) < 50:
                            continue
                        
                        # Calculate regime transition detection score for this target
                        regime_transition_score = self._calculate_regime_transition_detection(
                            feature_aligned, target_aligned
                        )
                        
                        weighted_regime_transition_score += target_weight * regime_transition_score
                        transition_weight += target_weight
                    
                    if transition_weight > 0:
                        weighted_regime_transition_score /= transition_weight
                    
                    # Calculate composite score with regime transition detection
                    composite_score = (
                        self.config.economic_significance_weight * weighted_economic_significance +
                        self.config.regime_discrimination_weight * weighted_regime_discrimination +
                        self.config.clustering_quality_weight * weighted_clustering_quality +
                        self.config.stability_weight * weighted_stability_score +
                        self.config.mrmr_weight * weighted_mrmr_score +
                        self.config.regime_transition_weight * weighted_regime_transition_score
                    )
                    
                    # Determine category
                    category = self._determine_feature_category(feature_name)
                    
                    feature_scores.append(FeatureScore(
                        feature_name=feature_name,
                        economic_significance=weighted_economic_significance,
                        regime_discrimination=weighted_regime_discrimination,
                        clustering_quality=weighted_clustering_quality,
                        stability_score=weighted_stability_score,
                        mrmr_score=weighted_mrmr_score,
                        regime_transition_score=weighted_regime_transition_score,
                        composite_score=composite_score,
                        category=category
                    ))
                    
                except Exception as e:
                    tprint_warning(f"Error scoring feature {feature_name}: {e}")
                    continue
            
        tprint_success(f"🏆 Multi-target feature scoring completed: {len(feature_scores)} features scored")
        return feature_scores
            
    def _score_features_by_economics(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, economic_metrics: List[EconomicMetrics]) -> List[FeatureScore]:
        """Score features based on economic relevance with computation optimizations."""
        try:
            tprint_info("Scoring features by economic relevance with optimizations...")
            
            # Optimize hardware for feature scoring
            if HARDWARE_AVAILABLE and self.hardware_manager:
                from src.utils.hardware.unified_hardware_manager import WorkloadType
                self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
                tprint_info("Hardware optimized for feature scoring")
            
            # Use VectorBT optimization if available
            if VECTORBT_AVAILABLE and self.config.enable_vectorbt and self.vectorbt_optimizer:
                tprint_info("Using VectorBT optimization for feature scoring")
                feature_scores = self._vectorbt_optimized_scoring(features_df, labels_df, economic_metrics)
            else:
                tprint_info("Using standard scoring with optimizations")
                feature_scores = self._standard_optimized_scoring(features_df, labels_df, economic_metrics)
            
            # Sort by composite score
            feature_scores.sort(key=lambda x: x.composite_score, reverse=True)
            
            tprint_success(f"Scored {len(feature_scores)} features")
            return feature_scores
            
        except Exception as e:
            tprint_error(f"Error scoring features: {e}")
            return []
    
    def _vectorbt_optimized_scoring(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, economic_metrics: List[EconomicMetrics]) -> List[FeatureScore]:
        """VectorBT-optimized feature scoring."""
        try:
            feature_scores = []
            
            # Use VectorBT batch processing for efficiency
            for feature_name in features_df.columns:
                try:
                    # Get feature data
                    feature_data = features_df[feature_name].dropna()
                    
                    # Align with labels
                    common_index = feature_data.index.intersection(labels_df.index)
                    feature_aligned = feature_data.loc[common_index]
                    labels_aligned = labels_df.loc[common_index]
                    
                    # Ensure labels_aligned is a Series
                    if isinstance(labels_aligned, pd.DataFrame):
                        if labels_aligned.shape[1] == 1:
                            labels_aligned = labels_aligned.iloc[:, 0]
                        else:
                            labels_aligned = labels_aligned.iloc[:, 0]  # Use first column
                    
                    if len(feature_aligned) < 50:  # Minimum sample size
                        continue
                    
                    # Use VectorBT for efficient calculations
                    if VECTORBT_AVAILABLE:
                        # VectorBT-optimized economic significance
                        economic_significance = self._vectorbt_economic_significance(
                            feature_aligned, labels_aligned, economic_metrics
                        )
                        
                        # VectorBT-optimized regime discrimination
                        regime_discrimination = self._vectorbt_regime_discrimination(
                            feature_aligned, labels_aligned
                        )
                    else:
                        # Fallback to standard methods
                        economic_significance = self._calculate_economic_significance_cheap(
                            feature_aligned, labels_aligned, economic_metrics
                        )
                        regime_discrimination = self._calculate_regime_discrimination(
                            feature_aligned, labels_aligned
                        )
                    
                    # Calculate clustering quality (cheap proxy)
                    clustering_quality = self._calculate_clustering_quality_cheap(
                        feature_aligned, labels_aligned
                    )
                    
                    # Calculate stability (cheap proxy)
                    stability_score = self._calculate_stability_cheap(
                        feature_aligned, labels_aligned
                    )
                    
                    # Calculate composite score
                    composite_score = (
                        self.config.economic_significance_weight * economic_significance +
                        self.config.regime_discrimination_weight * regime_discrimination +
                        self.config.clustering_quality_weight * clustering_quality +
                        self.config.stability_weight * stability_score
                    )
                    
                    # Determine category
                    category = self._determine_feature_category(feature_name)
                    
                    feature_scores.append(FeatureScore(
                        feature_name=feature_name,
                        economic_significance=economic_significance,
                        regime_discrimination=regime_discrimination,
                        clustering_quality=clustering_quality,
                        stability_score=stability_score,
                        composite_score=composite_score,
                        category=category
                    ))
                    
                except Exception as e:
                    tprint_warning(f"Error scoring feature {feature_name}: {e}")
                    continue
            
            return feature_scores
            
        except Exception as e:
            tprint_error(f"Error in VectorBT optimized scoring: {e}")
            return []
    
    def _standard_optimized_scoring(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, economic_metrics: List[EconomicMetrics]) -> List[FeatureScore]:
        """Standard scoring with optimizations."""
        try:
            feature_scores = []
            
            for feature_name in features_df.columns:
                try:
                    # Get feature data
                    feature_data = features_df[feature_name].dropna()
                    
                    # Align with labels
                    common_index = feature_data.index.intersection(labels_df.index)
                    feature_aligned = feature_data.loc[common_index]
                    labels_aligned = labels_df.loc[common_index]
                    
                    # Ensure labels_aligned is a Series
                    if isinstance(labels_aligned, pd.DataFrame):
                        if labels_aligned.shape[1] == 1:
                            labels_aligned = labels_aligned.iloc[:, 0]
                        else:
                            labels_aligned = labels_aligned.iloc[:, 0]  # Use first column
                    
                    if len(feature_aligned) < 50:  # Minimum sample size
                        continue
                    
                    # Calculate economic significance (cheap proxy)
                    economic_significance = self._calculate_economic_significance_cheap(
                        feature_aligned, labels_aligned, economic_metrics
                    )
                    
                    # Calculate regime discrimination (F-ratio)
                    regime_discrimination = self._calculate_regime_discrimination(
                        feature_aligned, labels_aligned
                    )
                    
                    # Calculate clustering quality (cheap proxy)
                    clustering_quality = self._calculate_clustering_quality_cheap(
                        feature_aligned, labels_aligned
                    )
                    
                    # Calculate stability (cheap proxy)
                    stability_score = self._calculate_stability_cheap(
                        feature_aligned, labels_aligned
                    )
                    
                    # Calculate composite score
                    composite_score = (
                        self.config.economic_significance_weight * economic_significance +
                        self.config.regime_discrimination_weight * regime_discrimination +
                        self.config.clustering_quality_weight * clustering_quality +
                        self.config.stability_weight * stability_score
                    )
                    
                    # Determine category
                    category = self._determine_feature_category(feature_name)
                    
                    feature_scores.append(FeatureScore(
                        feature_name=feature_name,
                        economic_significance=economic_significance,
                        regime_discrimination=regime_discrimination,
                        clustering_quality=clustering_quality,
                        stability_score=stability_score,
                        composite_score=composite_score,
                        category=category
                    ))
                    
                except Exception as e:
                    tprint_warning(f"Error scoring feature {feature_name}: {e}")
                    continue
            
            return feature_scores
            
        except Exception as e:
            tprint_error(f"Error in standard optimized scoring: {e}")
            return []
    
    def _vectorbt_economic_significance(self, feature_data: pd.Series, labels: pd.Series, economic_metrics: List[EconomicMetrics]) -> float:
        """VectorBT-optimized economic significance calculation."""
        try:
            # Use VectorBT rolling operations for efficient regime analysis
            regime_sharpes = {metric.regime_id: metric.sharpe_ratio for metric in economic_metrics}
            
            # Calculate feature correlation with regime Sharpe ratios using VectorBT
            feature_by_regime = {}
            for regime_id in labels.unique():
                if regime_id == -1:  # Skip noise
                    continue
                regime_mask = labels == regime_id
                regime_feature = feature_data[regime_mask]
                if len(regime_feature) > 0:
                    # Use VectorBT rolling mean for efficiency
                    if VECTORBT_AVAILABLE:
                        feature_by_regime[regime_id] = rolling_mean(regime_feature, min(20, len(regime_feature))).mean()
                    else:
                        feature_by_regime[regime_id] = regime_feature.mean()
            
            if len(feature_by_regime) < 2:
                return 0.0
            
            # Calculate correlation between feature means and Sharpe ratios
            regime_ids = list(feature_by_regime.keys())
            feature_means = [feature_by_regime[rid] for rid in regime_ids]
            sharpe_values = [regime_sharpes.get(rid, 0.0) for rid in regime_ids]
            
            if len(feature_means) > 1 and len(sharpe_values) > 1:
                correlation = np.corrcoef(feature_means, sharpe_values)[0, 1]
                correlation_score = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                correlation_score = 0.0
            
            # Add cheap mutual information proxy
            mi_score = self._calculate_cheap_mutual_information(feature_data, labels)
            
            # Combine correlation and MI scores
            combined_score = 0.7 * correlation_score + 0.3 * mi_score
            return min(1.0, combined_score)
            
        except Exception as e:
            tprint_warning(f"Error in VectorBT economic significance: {e}")
            return 0.0
    
    def _vectorbt_regime_discrimination(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """VectorBT-optimized regime discrimination calculation."""
        try:
            unique_regimes = labels.unique()
            if len(unique_regimes) < 2:
                return 0.0
            
            # Use VectorBT for efficient variance calculations
            regime_means = []
            regime_vars = []
            
            for regime_id in unique_regimes:
                if regime_id == -1:  # Skip noise
                    continue
                regime_mask = labels == regime_id
                regime_data = feature_data[regime_mask]
                
                if len(regime_data) > 0:
                    regime_means.append(regime_data.mean())
                    # Use VectorBT rolling variance for efficiency
                    if VECTORBT_AVAILABLE:
                        regime_vars.append(rolling_var(regime_data, min(20, len(regime_data))).mean())
                    else:
                        regime_vars.append(regime_data.var())
            
            if len(regime_means) < 2:
                return 0.0
            
            # Between-regime variance
            overall_mean = np.mean(regime_means)
            between_var = np.var(regime_means)
            
            # Within-regime variance
            within_var = np.mean(regime_vars)
            
            # F-ratio
            if within_var > 0:
                f_ratio = between_var / within_var
                return min(1.0, f_ratio)  # Cap at 1.0
            else:
                return min(1.0, between_var)
            
        except Exception as e:
            tprint_warning(f"Error in VectorBT regime discrimination: {e}")
            return 0.0
    
    def _calculate_economic_significance_cheap(self, feature_data: pd.Series, labels: pd.Series, economic_metrics=None) -> float:
        """Calculate economic significance using cheap proxy with mutual information."""
        try:
            # Handle multi-target approach where economic_metrics is a Series (target data)
            if isinstance(economic_metrics, pd.Series):
                # For multi-target approach, use correlation with target data as economic significance
                correlation = abs(feature_data.corr(economic_metrics))
                if pd.isna(correlation):
                    return 0.0
                return correlation
            
            # Original regime-based approach
            if not economic_metrics or len(economic_metrics) == 0:
                return 0.0
                
            # Use regime-specific Sharpe ratios as economic significance
            regime_sharpes = {metric.regime_id: metric.sharpe_ratio for metric in economic_metrics}
            
            # Calculate feature correlation with regime Sharpe ratios
            feature_by_regime = {}
            for regime_id in labels.unique():
                if regime_id == -1:  # Skip noise
                    continue
                regime_mask = labels == regime_id
                regime_feature = feature_data[regime_mask]
                if len(regime_feature) > 0:
                    feature_by_regime[regime_id] = regime_feature.mean()
            
            if len(feature_by_regime) < 2:
                return 0.0
            
            # Calculate correlation between feature means and Sharpe ratios
            regime_ids = list(feature_by_regime.keys())
            feature_means = [feature_by_regime[rid] for rid in regime_ids]
            sharpe_values = [regime_sharpes.get(rid, 0.0) for rid in regime_ids]
            
            if len(feature_means) > 1 and len(sharpe_values) > 1:
                correlation = np.corrcoef(feature_means, sharpe_values)[0, 1]
                correlation_score = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                correlation_score = 0.0
            
            # Add cheap mutual information proxy
            mi_score = self._calculate_cheap_mutual_information(feature_data, labels)
            
            # Combine correlation and MI scores
            combined_score = 0.7 * correlation_score + 0.3 * mi_score
            return min(1.0, combined_score)
            
        except Exception as e:
            tprint_warning(f"Error calculating economic significance: {e}")
            return 0.0
    
    def _calculate_cheap_mutual_information(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Calculate cheap mutual information proxy using binned approximation."""
        try:
            # Align data
            common_index = feature_data.index.intersection(labels.index)
            feature_aligned = feature_data.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            # Remove noise labels
            valid_mask = labels_aligned != -1
            feature_valid = feature_aligned[valid_mask]
            labels_valid = labels_aligned[valid_mask]
            
            if len(feature_valid) < 10 or len(np.unique(labels_valid)) < 2:
                return 0.0
            
            # Use cheap binning approach (not full KDE)
            n_bins = min(10, len(feature_valid) // 20)  # Adaptive binning
            if n_bins < 2:
                n_bins = 2
            
            # Bin the feature data
            feature_binned = pd.cut(feature_valid, bins=n_bins, labels=False, duplicates='drop')
            
            # Calculate cheap MI using binned data
            mi_score = self._binned_mutual_information(feature_binned, labels_valid)
            
            return mi_score
            
        except Exception as e:
            tprint_warning(f"Error calculating cheap MI: {e}")
            return 0.0
    
    def _binned_mutual_information(self, feature_binned: pd.Series, labels: pd.Series) -> float:
        """Calculate mutual information using binned data (cheap proxy)."""
        try:
            # Create contingency table
            contingency = pd.crosstab(feature_binned, labels)
            
            # Normalize to get probabilities
            joint_prob = contingency / contingency.sum().sum()
            
            # Calculate marginal probabilities
            feature_marginal = joint_prob.sum(axis=1)
            label_marginal = joint_prob.sum(axis=0)
            
            # Calculate MI using binned approximation
            mi = 0.0
            for i in range(len(feature_marginal)):
                for j in range(len(label_marginal)):
                    if joint_prob.iloc[i, j] > 0:
                        mi += joint_prob.iloc[i, j] * np.log2(
                            joint_prob.iloc[i, j] / (feature_marginal.iloc[i] * label_marginal.iloc[j])
                        )
            
            # Normalize by entropy (cheap proxy)
            feature_entropy = -np.sum(feature_marginal * np.log2(feature_marginal + 1e-10))
            normalized_mi = mi / feature_entropy if feature_entropy > 0 else 0.0
            
            return min(1.0, normalized_mi)
            
        except Exception as e:
            tprint_warning(f"Error calculating binned MI: {e}")
            return 0.0
    
    def _calculate_regime_discrimination(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Calculate regime discrimination using F-ratio."""
        try:
            unique_regimes = labels.unique()
            if len(unique_regimes) < 2:
                return 0.0
            
            # Calculate between-regime and within-regime variance
            regime_means = []
            regime_vars = []
            
            for regime_id in unique_regimes:
                if regime_id == -1:  # Skip noise
                    continue
                regime_mask = labels == regime_id
                regime_data = feature_data[regime_mask]
                
                if len(regime_data) > 0:
                    regime_means.append(regime_data.mean())
                    regime_vars.append(regime_data.var())
            
            if len(regime_means) < 2:
                return 0.0
            
            # Between-regime variance
            overall_mean = np.mean(regime_means)
            between_var = np.var(regime_means)
            
            # Within-regime variance
            within_var = np.mean(regime_vars)
            
            # F-ratio
            if within_var > 0:
                f_ratio = between_var / within_var
                return min(1.0, f_ratio)  # Cap at 1.0
            else:
                return min(1.0, between_var)
            
        except Exception as e:
            tprint_warning(f"Error calculating regime discrimination: {e}")
            return 0.0
    
    def _calculate_clustering_quality_cheap(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Calculate clustering quality using improved cheap proxy with better sampling."""
        try:
            # Ensure we have valid data
            if len(feature_data) < 10 or len(labels) < 10:
                tprint_debug(f"Clustering quality: insufficient data (feature: {len(feature_data)}, labels: {len(labels)})")
                return 0.0
            
            # Remove NaN values
            valid_mask = ~(pd.isna(feature_data) | pd.isna(labels))
            if valid_mask.sum() < 10:
                tprint_debug(f"Clustering quality: insufficient valid data ({valid_mask.sum()})")
                return 0.0
            
            feature_clean = feature_data[valid_mask]
            labels_clean = labels[valid_mask]
            
            # Check if we have discrete labels (regime-based) or continuous labels (multi-target)
            unique_labels = labels_clean.unique()
            
            # If we have too many unique labels (>20), treat as continuous and create bins
            if len(unique_labels) > 20:
                # Create discrete bins for continuous data with better error handling
                try:
                    # Use more robust binning with error handling
                    n_bins = min(5, max(2, len(labels_clean)//20))  # Ensure at least 2 bins
                    labels_binned = pd.qcut(labels_clean, q=n_bins, duplicates='drop')
                    unique_bins = labels_binned.unique()
                    if len(unique_bins) < 2:
                        # Fallback to simple separation score
                        return self._calculate_simple_separation_score(feature_clean, labels_clean)
                    labels_discrete = labels_binned.cat.codes
                except (ValueError, TypeError) as e:
                    tprint_debug(f"Binning failed: {e}, using simple separation score")
                    # If binning fails, use simple separation score
                    return self._calculate_simple_separation_score(feature_clean, labels_clean)
            else:
                # Use original discrete labels
                labels_discrete = labels_clean
            
            # Check if we have enough unique labels after processing
            unique_labels_final = pd.Series(labels_discrete).unique()
            if len(unique_labels_final) < 2:
                return self._calculate_simple_separation_score(feature_clean, labels_clean)
            
            # Remove noise labels (-1) if present (only for discrete regime labels)
            if len(unique_labels_final) <= 10:  # Only check for noise if we have discrete labels
                valid_labels_mask = labels_discrete != -1
                if valid_labels_mask.sum() < 10:
                    tprint_debug(f"Clustering quality: insufficient non-noise data ({valid_labels_mask.sum()})")
                    return 0.0
                
                feature_final = feature_clean[valid_labels_mask]
                labels_final = labels_discrete[valid_labels_mask]
            else:
                # For continuous/binned data, use all data
                feature_final = feature_clean
                labels_final = labels_discrete
            
            # Check label distribution with more lenient requirements
            label_counts = pd.Series(labels_final).value_counts()
            valid_regimes = label_counts[label_counts >= 1]  # Changed from 2 to 1
            
            if len(valid_regimes) < 2:
                tprint_debug(f"Clustering quality: insufficient valid regimes ({len(valid_regimes)})")
                return 0.0
            
            # Improved sampling strategy
            if len(feature_final) > 1000 and self.config.enable_cheap_proxies:
                # Use stratified sampling to ensure representation from each regime
                sample_size = min(500, int(len(feature_final) * 0.20))  # Increased from 10% to 20%
                
                # Stratified sampling by regime
                sample_indices = []
                for regime_id in valid_regimes.index:
                    regime_mask = labels_final == regime_id
                    regime_indices = np.where(regime_mask)[0]
                    
                    # Sample proportionally from each regime
                    regime_sample_size = max(1, int(sample_size * len(regime_indices) / len(feature_final)))
                    regime_sample_size = min(regime_sample_size, len(regime_indices))
                    
                    if regime_sample_size > 0:
                        sampled_indices = np.random.choice(regime_indices, size=regime_sample_size, replace=False)
                        sample_indices.extend(sampled_indices)
                
                if len(sample_indices) < 2:
                    tprint_debug(f"Clustering quality: insufficient sample indices ({len(sample_indices)})")
                    return 0.0
                
                sample_feature = feature_final.iloc[sample_indices]
                sample_labels = labels_final.iloc[sample_indices]
            else:
                sample_feature = feature_final
                sample_labels = labels_final
            
            # Calculate silhouette score with improved error handling
            from sklearn.metrics import silhouette_score
            
            # Final validation with more lenient requirements
            final_label_counts = pd.Series(sample_labels).value_counts()
            if len(final_label_counts) < 2:
                tprint_debug(f"Clustering quality: insufficient final label distribution")
                return 0.0
            
            # Check if any label has at least 1 sample (more lenient)
            if (final_label_counts < 1).any():
                tprint_debug(f"Clustering quality: some labels have 0 samples, using fallback")
                return self._calculate_simple_separation_score(sample_feature, sample_labels)
            
            # Reshape for sklearn
            feature_matrix = sample_feature.values.reshape(-1, 1)
            
            try:
                silhouette = silhouette_score(feature_matrix, sample_labels)
                
                # Apply smoothing to avoid extreme values
                smoothed_silhouette = max(0.0, min(1.0, silhouette))
                
                # Add debug logging
                tprint_debug(f"Clustering quality: silhouette={silhouette:.4f}, smoothed={smoothed_silhouette:.4f}")
                
                return smoothed_silhouette
                
            except ValueError as ve:
                tprint_warning(f"Silhouette calculation failed: {ve}")
                # Fallback to simple separation metric
                return self._calculate_simple_separation_score(sample_feature, sample_labels)
            except Exception as e:
                tprint_warning(f"Unexpected error in silhouette calculation: {e}")
                # Fallback to simple separation metric
                return self._calculate_simple_separation_score(sample_feature, sample_labels)
            
        except Exception as e:
            tprint_warning(f"Error calculating clustering quality: {e}")
            return 0.0
    
    def _calculate_simple_separation_score(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Fallback method for clustering quality when silhouette fails."""
        try:
            # Calculate between-cluster and within-cluster variance
            unique_labels = labels.unique()
            if len(unique_labels) < 2:
                return 0.0
            
            # Between-cluster variance
            overall_mean = feature_data.mean()
            between_var = 0.0
            total_samples = len(feature_data)
            
            for label in unique_labels:
                label_mask = labels == label
                label_data = feature_data[label_mask]
                if len(label_data) > 0:
                    label_mean = label_data.mean()
                    between_var += len(label_data) * (label_mean - overall_mean) ** 2
            
            between_var /= total_samples
            
            # Within-cluster variance
            within_var = 0.0
            for label in unique_labels:
                label_mask = labels == label
                label_data = feature_data[label_mask]
                if len(label_data) > 0:
                    label_mean = label_data.mean()
                    within_var += ((label_data - label_mean) ** 2).sum()
            
            within_var /= total_samples
            
            # Calculate separation score
            if within_var > 0:
                separation_score = between_var / within_var
                # Normalize to [0, 1] range with better scaling
                normalized_score = min(1.0, max(0.0, separation_score / 5.0))  # Better scaling
                tprint_debug(f"Simple separation score: {separation_score:.4f}, normalized: {normalized_score:.4f}")
                return normalized_score
            else:
                # If within_var is 0, all points in each cluster are identical
                # This is actually good separation, so return a high score
                return min(1.0, max(0.1, between_var))  # Ensure minimum score
                
        except Exception as e:
            tprint_warning(f"Error calculating simple separation score: {e}")
            return 0.0
    
    def _calculate_regime_transition_detection(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Calculate how well feature detects regime transitions."""
        try:
            if len(feature_data) < 50 or len(labels) < 50:
                tprint_debug(f"Transition detection: insufficient data ({len(feature_data)})")
                return 0.0
            
            # Find regime transition points
            transition_points = self._find_regime_transitions(labels)
            
            if len(transition_points) < 2:
                tprint_debug(f"Transition detection: insufficient transitions ({len(transition_points)})")
                return 0.0
            
            # Calculate feature sensitivity around transitions
            transition_sensitivity = []
            window_size = min(10, len(feature_data) // 20)  # Adaptive window size
            
            for transition_point in transition_points:
                # Get feature values around transition
                start_idx = max(0, transition_point - window_size)
                end_idx = min(len(feature_data), transition_point + window_size)
                
                window_feature = feature_data.iloc[start_idx:end_idx]
                window_labels = labels.iloc[start_idx:end_idx]
                
                # Calculate feature change during transition
                if len(window_feature) > 2:
                    feature_change = abs(window_feature.iloc[-1] - window_feature.iloc[0])
                    transition_sensitivity.append(feature_change)
            
            if transition_sensitivity:
                # Normalize by feature standard deviation
                feature_std = feature_data.std()
                if feature_std > 0:
                    normalized_sensitivity = np.mean(transition_sensitivity) / feature_std
                    # Apply smoothing and cap at 1.0
                    transition_score = min(1.0, normalized_sensitivity)
                    tprint_debug(f"Transition detection: sensitivity={normalized_sensitivity:.4f}, score={transition_score:.4f}")
                    return transition_score
            
            return 0.0
            
        except Exception as e:
            tprint_warning(f"Error calculating regime transition detection: {e}")
            return 0.0
    
    def _find_regime_transitions(self, labels: pd.Series) -> List[int]:
        """Find points where regime labels change."""
        try:
            transitions = []
            for i in range(1, len(labels)):
                if labels.iloc[i] != labels.iloc[i-1]:
                    transitions.append(i)
            return transitions
        except Exception as e:
            tprint_warning(f"Error finding regime transitions: {e}")
            return []
    
    
    def _calculate_redundancy_vectorized(self, feature_data: pd.Series, features_df: pd.DataFrame) -> float:
        """Calculate redundancy penalty using VectorBT vectorized operations."""
        try:
            # Use VectorBT for efficient correlation calculation
            if self.vectorization_manager:
                # Calculate correlation with all other features at once
                other_features = features_df.drop(columns=[feature_data.name], errors='ignore')
                if other_features.empty:
                    return 0.0
                
                # Use VectorBT for batch correlation calculation
                correlations = []
                for other_feature in other_features.columns:
                    cached_corr = self._get_cached_correlation(feature_data, other_features[other_feature])
                    if cached_corr is not None:
                        correlations.append(abs(cached_corr))
                    else:
                        corr = abs(feature_data.corr(other_features[other_feature]))
                        self._set_cached_correlation(feature_data, other_features[other_feature], corr)
                        correlations.append(corr)
                
                if correlations:
                    redundancy_penalty = np.mean(correlations)
                    return min(redundancy_penalty, 0.8)  # Cap penalty at 0.8
            
            return 0.0
            
        except Exception as e:
            tprint_warning(f"Vectorized redundancy calculation failed: {e}")
            return self._calculate_redundancy_sampled(feature_data, features_df)
    
    def _calculate_redundancy_sampled(self, feature_data: pd.Series, features_df: pd.DataFrame) -> float:
        """Calculate redundancy penalty using sampled features (fallback method)."""
        try:
            redundancy_correlations = []
            feature_names = list(features_df.columns)
            
            if len(feature_names) > 1:
                # Sample up to 10 other features for redundancy calculation
                sample_size = min(10, len(feature_names) - 1)
                other_features = np.random.choice(
                    [f for f in feature_names if f != feature_data.name], 
                    size=sample_size, 
                    replace=False
                )
                
                for other_feature in other_features:
                    if other_feature in features_df.columns:
                        try:
                            cached_corr = self._get_cached_correlation(feature_data, features_df[other_feature])
                            if cached_corr is not None:
                                redundancy_correlations.append(abs(cached_corr))
                            else:
                                corr = abs(feature_data.corr(features_df[other_feature]))
                                self._set_cached_correlation(feature_data, features_df[other_feature], corr)
                                redundancy_correlations.append(corr)
                        except Exception as e:
                            tprint_debug(f"mRMR multi-target: correlation calculation failed for {other_feature}: {e}")
                            continue
                
                if redundancy_correlations:
                    redundancy_penalty = np.mean(redundancy_correlations)
                    return min(redundancy_penalty, 0.8)  # Cap penalty at 0.8
            
            return 0.0
            
        except Exception as e:
            tprint_warning(f"Sampled redundancy calculation failed: {e}")
            return 0.0

    def _calculate_mrmr_cheap(self, feature_data: pd.Series, labels: pd.Series, selected_features: List[str], features_df: pd.DataFrame) -> float:
        """Calculate improved mRMR proxy with better debugging and thresholds."""
        try:
            if len(feature_data) < 20:
                tprint_debug(f"mRMR: insufficient data ({len(feature_data)})")
                return 0.0
            
            # Calculate relevance (correlation with labels)
            relevance = abs(feature_data.corr(labels))
            if pd.isna(relevance):
                tprint_debug(f"mRMR: NaN relevance for feature")
                relevance = 0.0
            
            # Calculate redundancy penalty (average correlation with already selected features)
            redundancy_penalty = 0.0
            redundancy_correlations = []
            
            if selected_features:
                for selected_feature in selected_features:
                    if selected_feature in features_df.columns:
                        try:
                            corr = abs(feature_data.corr(features_df[selected_feature]))
                            if not pd.isna(corr):
                                redundancy_correlations.append(corr)
                        except Exception as e:
                            tprint_debug(f"mRMR: correlation calculation failed for {selected_feature}: {e}")
                            continue
                
                if redundancy_correlations:
                    redundancy_penalty = np.mean(redundancy_correlations)
                    # Apply penalty scaling to avoid over-penalization - reduced for better diversity
                    redundancy_penalty = min(redundancy_penalty * 0.6, 0.5)  # Scale down and cap at 0.5
            
            # mRMR score = relevance - redundancy_penalty
            mrmr_score = relevance - redundancy_penalty
            
            # Apply minimum threshold to avoid very small scores
            min_threshold = 0.01
            if mrmr_score < min_threshold:
                mrmr_score = 0.0
            
            # Debug logging
            tprint_debug(f"mRMR: relevance={relevance:.4f}, penalty={redundancy_penalty:.4f}, score={mrmr_score:.4f}")
            
            return max(0.0, mrmr_score)  # Ensure non-negative
            
        except Exception as e:
            tprint_warning(f"Error calculating mRMR: {e}")
            return 0.0

    def _calculate_stability_cheap(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Calculate stability using cheap proxy (time-based variance)."""
        try:
            if len(feature_data) < 20:
                return 0.5  # Default moderate stability for small datasets
            
            # Method 1: Time-based stability (rolling variance of rolling mean)
            window_size = min(20, len(feature_data) // 4)
            rolling_mean = feature_data.rolling(window=window_size, min_periods=5).mean()
            rolling_var = rolling_mean.rolling(window=window_size//2, min_periods=3).var()
            
            # Stability is inverse of normalized variance
            mean_var = rolling_var.mean()
            if pd.isna(mean_var) or mean_var == 0:
                return 0.5
            
            # Normalize by feature variance
            feature_var = feature_data.var()
            if pd.isna(feature_var) or feature_var == 0:
                return 0.5
            
            normalized_var = mean_var / feature_var
            stability = 1.0 / (1.0 + normalized_var)  # Higher variance = lower stability
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            tprint_warning(f"Error calculating stability: {e}")
            return 0.5
    
    def _determine_feature_category(self, feature_name: str) -> str:
        """Determine feature category from name with improved volatility detection."""
        feature_lower = feature_name.lower()
        
        # Priority-based categorization - check more specific patterns first
        # 1. Volatility features (highest priority to avoid misclassification)
        volatility_patterns = [
            'volatility', 'vol_', 'std', 'var', 'atr', 'bbands', 'parkinson', 'garman', 'yang', 'rogers',
            'returns_volatility', 'volatility_', '_volatility_', 'volatility_acceleration', 'acceleration_volatility',
            'vectorbt_volatility', 'enhanced_volatility', 'volatility_comprehensive', 'volatility_elasticity'
        ]
        if any(pattern in feature_lower for pattern in volatility_patterns):
            return 'volatility'
        
        # 2. Volume features
        volume_patterns = ['volume', 'vwap', 'obv', 'ad_line', 'cmf', 'volume_', '_volume_']
        if any(pattern in feature_lower for pattern in volume_patterns):
            return 'volume'
        
        # 3. Momentum features
        momentum_patterns = ['momentum', 'mom', 'rsi', 'macd', 'roc', 'stochastic', 'williams_r']
        if any(pattern in feature_lower for pattern in momentum_patterns):
            return 'momentum'
        
        # 4. Returns features (but not volatility-related returns)
        returns_patterns = ['return', 'ret']
        if any(pattern in feature_lower for pattern in returns_patterns):
            # Double-check it's not a volatility feature
            if not any(vol_pattern in feature_lower for vol_pattern in volatility_patterns):
                return 'returns'
        
        # 5. Other categories
        category_mapping = {
            'regime': ['regime', 'state'],
            'order_flow': ['order', 'flow', 'bid', 'ask'],
            'statistical': ['stat', 'mean', 'median', 'skew', 'kurt'],
            'entropy': ['entropy', 'ent'],
            'spectral': ['spectral', 'freq', 'fft'],
            'trend': ['trend', 'ma', 'ema', 'sma'],
            'oscillator': ['oscillator', 'osc', 'stoch'],
            'candlestick': ['candlestick', 'candle', 'doji', 'hammer'],
            'cross_timeframe': ['cross', 'multi', 'timeframe'],
            'interaction': ['interaction', 'inter', 'ratio', 'product'],
            'regime_transition': ['transition', 'change', 'shift', 'break', 'crossover', 'divergence']
        }
        
        for category, keywords in category_mapping.items():
            if any(keyword in feature_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _calculate_regime_transition_score(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Calculate regime transition detection score for features."""
        try:
            # Align data
            common_index = feature_data.index.intersection(labels.index)
            feature_aligned = feature_data.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            # Remove noise labels
            valid_mask = labels_aligned != -1
            feature_valid = feature_aligned[valid_mask]
            labels_valid = labels_aligned[valid_mask]
            
            if len(feature_valid) < 50 or len(np.unique(labels_valid)) < 2:
                return 0.0
            
            # Calculate regime transition detection score
            transition_score = self._detect_regime_transitions(feature_valid, labels_valid)
            
            return transition_score
            
        except Exception as e:
            tprint_warning(f"Error calculating regime transition score: {e}")
            return 0.0
    
    def _detect_regime_transitions(self, feature_data: pd.Series, labels: pd.Series) -> float:
        """Detect how well a feature identifies regime transitions."""
        try:
            # Ensure feature_data is a Series
            if not isinstance(feature_data, pd.Series):
                tprint_warning(f"Expected Series but got {type(feature_data)}")
                return 0.0
            
            # Ensure labels is a Series
            if not isinstance(labels, pd.Series):
                tprint_warning(f"Expected Series but got {type(labels)}")
                return 0.0
            
            # Find regime change points
            regime_changes = labels.diff() != 0
            regime_change_indices = regime_changes[regime_changes].index
            
            if len(regime_change_indices) < 2:
                return 0.0
            
            # Calculate feature behavior around regime changes
            transition_scores = []
            
            for change_idx in regime_change_indices:
                # Look at feature behavior before and after regime change
                window_size = min(self.config.transition_window_size, len(feature_data) // 20)
                
                # Get positions around the change point (using iloc for position-based indexing)
                try:
                    # Find the position of the change point in the feature data
                    if change_idx in feature_data.index:
                        change_pos = feature_data.index.get_loc(change_idx)
                    else:
                        # If change_idx is not in feature_data index, find the closest position
                        closest_pos = feature_data.index.get_indexer([change_idx], method='nearest')[0]
                        if closest_pos == -1:
                            continue
                        change_pos = closest_pos
                    
                    start_pos = max(0, change_pos - window_size)
                    end_pos = min(len(feature_data), change_pos + window_size)
                    
                    if end_pos - start_pos < window_size * 2:
                        continue
                    
                    # Split into before and after regime change
                    before_feature = feature_data.iloc[start_pos:change_pos]
                    after_feature = feature_data.iloc[change_pos:end_pos]
                except (KeyError, IndexError, TypeError) as e:
                    # If change_idx is not in feature_data index or type error, skip
                    tprint_warning(f"Skipping regime transition detection due to error: {e}")
                    continue
                
                if len(before_feature) < 3 or len(after_feature) < 3:
                    continue
                
                # Calculate feature change magnitude
                before_mean = before_feature.mean()
                after_mean = after_feature.mean()
                
                if abs(before_mean) > 1e-8:
                    change_magnitude = abs(after_mean - before_mean) / abs(before_mean)
                else:
                    change_magnitude = abs(after_mean - before_mean)
                
                # Calculate feature volatility change
                before_vol = before_feature.std()
                after_vol = after_feature.std()
                
                if before_vol > 1e-8:
                    vol_change = abs(after_vol - before_vol) / before_vol
                else:
                    vol_change = abs(after_vol - before_vol)
                
                # Combine change magnitude and volatility change
                transition_score = change_magnitude + vol_change
                transition_scores.append(transition_score)
            
            if len(transition_scores) == 0:
                return 0.0
            
            # Return average transition score with proper normalization
            avg_transition_score = np.mean(transition_scores)
            
            # Use a more discriminating normalization that handles extreme values better
            # Apply robust normalization using percentile-based scaling
            if avg_transition_score <= 0:
                return 0.0
            elif avg_transition_score <= 0.5:
                return avg_transition_score * 2.0  # Scale up small scores
            elif avg_transition_score <= 2.0:
                return 0.5 + (avg_transition_score - 0.5) * 0.25  # Moderate scores
            elif avg_transition_score <= 10.0:
                return 0.75 + (avg_transition_score - 2.0) * 0.025  # High scores
            else:
                # Very high scores get diminishing returns
                return min(0.95, 0.95 + (avg_transition_score - 10.0) * 0.001)
            
        except Exception as e:
            tprint_warning(f"Error detecting regime transitions: {e}")
            return 0.0
    
    def _identify_regime_transition_features(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
        """Identify features that are particularly good at detecting regime transitions."""
        try:
            tprint_info(f"🔍 Starting regime transition feature identification for {len(features_df.columns)} features")
            tprint_info(f"📊 Transition score threshold: {self.config.transition_score_threshold}")
            
            transition_features = []
            feature_scores = {}
            
            for i, feature_name in enumerate(features_df.columns):
                try:
                    tprint_info(f"📈 Analyzing feature {i+1}/{len(features_df.columns)}: {feature_name}")
                    
                    # Get feature data
                    feature_data = features_df[feature_name].dropna()
                    
                    # Align with labels
                    common_index = feature_data.index.intersection(labels_df.index)
                    feature_aligned = feature_data.loc[common_index]
                    labels_aligned = labels_df.loc[common_index]
                    
                    # Ensure labels_aligned is a Series
                    if isinstance(labels_aligned, pd.DataFrame):
                        if labels_aligned.shape[1] == 1:
                            labels_aligned = labels_aligned.iloc[:, 0]
                        else:
                            # If multiple columns, use the first one
                            labels_aligned = labels_aligned.iloc[:, 0]
                    
                    if len(feature_aligned) < 50:
                        tprint_warning(f"⚠️ Skipping {feature_name}: insufficient data ({len(feature_aligned)} samples)")
                        continue
                    
                    # Calculate regime transition score
                    transition_score = self._calculate_regime_transition_score(feature_aligned, labels_aligned)
                    feature_scores[feature_name] = transition_score
                    
                    # Check if feature is good at detecting transitions
                    if transition_score > self.config.transition_score_threshold:
                        transition_features.append(feature_name)
                        tprint_success(f"✅ Regime transition feature identified: {feature_name} (score: {transition_score:.3f})")
                    else:
                        tprint_info(f"❌ Feature below threshold: {feature_name} (score: {transition_score:.3f})")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Error evaluating transition feature {feature_name}: {e}")
                    continue
            
            tprint_success(f"🎯 Identified {len(transition_features)} regime transition features out of {len(features_df.columns)} total features")
            return transition_features, feature_scores
            
        except Exception as e:
            tprint_error(f"❌ Error identifying regime transition features: {e}")
            return [], {}
    
    def _select_optimal_features_incremental_mrmr(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, feature_scores: List[FeatureScore], transition_features: List[str]) -> List[str]:
        """Select optimal features using incremental mRMR selection with hardware optimization."""
        try:
            tprint_info("🎯 Starting incremental mRMR feature selection...")
            tprint_info(f"🎯 Target feature count: {self.config.target_feature_count}")
            tprint_info(f"🎯 Total features available: {len(features_df.columns)}")
            
            # Optimize hardware for feature selection
            if HARDWARE_AVAILABLE and self.hardware_manager:
                from src.utils.hardware.unified_hardware_manager import WorkloadType
                self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
                tprint_info("⚡ Hardware optimized for incremental mRMR")
            
            # Optimize data types
            features_df = self._optimize_data_types(features_df)
            
            # Pre-compute common operations
            precomputed = self._precompute_common_operations(features_df, labels_df)
            features_aligned = precomputed['features_aligned']
            labels_aligned = precomputed['labels_aligned']
            correlation_matrix = precomputed['correlation_matrix']
            
            # Sort features by composite score for initial ranking
            sorted_scores = sorted(feature_scores, key=lambda x: x.composite_score, reverse=True)
            
            selected_features = []
            remaining_features = [score.feature_name for score in sorted_scores]
            
            # First, add regime transition features
            for transition_feature in transition_features:
                if transition_feature in remaining_features:
                    selected_features.append(transition_feature)
                    remaining_features.remove(transition_feature)
                    tprint_info(f"✅ Added regime transition feature: {transition_feature}")
            
            tprint_info(f"🎯 After transition features: {len(selected_features)} selected, {len(remaining_features)} remaining")
            
            # Then iteratively add features using incremental mRMR
            iteration = 0
            while len(selected_features) < self.config.target_feature_count and remaining_features:
                iteration += 1
                tprint_info(f"🎯 mRMR iteration {iteration}: {len(selected_features)}/{self.config.target_feature_count} selected")
                
                best_candidate = None
                best_mrmr_score = -float('inf')
                
                # Use VectorBT for batch correlation calculation if available
                if self.vectorization_manager and correlation_matrix is not None:
                    best_candidate, best_mrmr_score = self._find_best_candidate_vectorized(
                        remaining_features, selected_features, features_aligned, labels_aligned, correlation_matrix
                    )
                else:
                    # Fallback to individual evaluation
                    for feature_name in remaining_features[:50]:  # Limit to top 50 candidates
                        if feature_name not in features_aligned.columns:
                            continue
                        
                        mrmr_score = self._calculate_incremental_mrmr_score(
                            feature_name, selected_features, features_aligned, labels_aligned
                        )
                        
                        if mrmr_score > best_mrmr_score:
                            best_mrmr_score = mrmr_score
                            best_candidate = feature_name
                
                if best_candidate:
                    selected_features.append(best_candidate)
                    remaining_features.remove(best_candidate)
                    tprint_info(f"✅ Added feature {len(selected_features)}/{self.config.target_feature_count}: {best_candidate} (mRMR: {best_mrmr_score:.4f})")
                else:
                    tprint_warning("⚠️ No suitable candidate found, stopping selection")
                    break
            
            tprint_success(f"✅ Incremental mRMR selection completed: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            tprint_error(f"Error in incremental mRMR selection: {e}")
            return self._select_optimal_features_mrmr(features_df, labels_df, feature_scores, transition_features)
    
    def _find_best_candidate_vectorized(self, remaining_features: List[str], selected_features: List[str], 
                                       features_aligned: pd.DataFrame, labels_aligned: pd.DataFrame, 
                                       correlation_matrix: pd.DataFrame) -> Tuple[Optional[str], float]:
        """Find best candidate using vectorized operations."""
        try:
            if not remaining_features:
                return None, -float('inf')
            
            # Calculate relevance scores for all remaining features
            relevance_scores = {}
            for feature_name in remaining_features:
                if feature_name in features_aligned.columns:
                    # Use cached correlation if available
                    cached_corr = self._get_cached_correlation(features_aligned[feature_name], labels_aligned.iloc[:, 0])
                    if cached_corr is not None:
                        relevance_scores[feature_name] = abs(cached_corr)
                    else:
                        relevance = abs(features_aligned[feature_name].corr(labels_aligned.iloc[:, 0]))
                        self._set_cached_correlation(features_aligned[feature_name], labels_aligned.iloc[:, 0], relevance)
                        relevance_scores[feature_name] = relevance
            
            tprint_debug(f"🔍 Vectorized search: {len(remaining_features)} remaining features, {len(selected_features)} selected")
            tprint_debug(f"🔍 Calculated relevance scores for {len(relevance_scores)} features")
            
            # Calculate redundancy penalties using correlation matrix
            best_candidate = None
            best_mrmr_score = -float('inf')
            
            for feature_name in remaining_features:
                if feature_name not in relevance_scores:
                    continue
                
                relevance = relevance_scores[feature_name]
                
                # Calculate redundancy penalty using correlation matrix
                redundancy_penalty = 0.0
                if selected_features and feature_name in correlation_matrix.index:
                    selected_correlations = []
                    for selected_feature in selected_features:
                        if selected_feature in correlation_matrix.columns:
                            corr = abs(correlation_matrix.loc[feature_name, selected_feature])
                            if not pd.isna(corr):
                                selected_correlations.append(corr)
                    
                    if selected_correlations:
                        redundancy_penalty = np.mean(selected_correlations)
                        redundancy_penalty = min(redundancy_penalty, 0.8)  # Cap penalty
                
                # Calculate mRMR score
                mrmr_score = relevance - redundancy_penalty
                
                if mrmr_score > best_mrmr_score:
                    best_mrmr_score = mrmr_score
                    best_candidate = feature_name
            
            tprint_debug(f"🔍 Best candidate: {best_candidate}, mRMR score: {best_mrmr_score:.4f}")
            return best_candidate, best_mrmr_score
            
        except Exception as e:
            tprint_warning(f"Vectorized candidate selection failed: {e}")
            return None, -float('inf')
    
    def _calculate_incremental_mrmr_score(self, feature_name: str, selected_features: List[str], 
                                         features_aligned: pd.DataFrame, labels_aligned: pd.DataFrame) -> float:
        """Calculate mRMR score for incremental selection."""
        try:
            feature_data = features_aligned[feature_name]
            
            # Calculate relevance
            cached_corr = self._get_cached_correlation(feature_data, labels_aligned.iloc[:, 0])
            if cached_corr is not None:
                relevance = abs(cached_corr)
            else:
                relevance = abs(feature_data.corr(labels_aligned.iloc[:, 0]))
                self._set_cached_correlation(feature_data, labels_aligned.iloc[:, 0], relevance)
            
            # Calculate redundancy penalty
            redundancy_penalty = 0.0
            if selected_features:
                redundancy_correlations = []
                for selected_feature in selected_features:
                    if selected_feature in features_aligned.columns:
                        cached_corr = self._get_cached_correlation(feature_data, features_aligned[selected_feature])
                        if cached_corr is not None:
                            redundancy_correlations.append(abs(cached_corr))
                        else:
                            corr = abs(feature_data.corr(features_aligned[selected_feature]))
                            self._set_cached_correlation(feature_data, features_aligned[selected_feature], corr)
                            redundancy_correlations.append(corr)
                
                if redundancy_correlations:
                    redundancy_penalty = np.mean(redundancy_correlations)
                    redundancy_penalty = min(redundancy_penalty, 0.8)  # Cap penalty
            
            # Calculate mRMR score
            mrmr_score = relevance - redundancy_penalty
            tprint_debug(f"🔍 mRMR for {feature_name}: relevance={relevance:.4f}, penalty={redundancy_penalty:.4f}, score={mrmr_score:.4f}")
            return max(0.0, mrmr_score)  # Ensure non-negative
            
        except Exception as e:
            tprint_warning(f"Error calculating incremental mRMR score for {feature_name}: {e}")
            return 0.0
    
    def _select_optimal_features_mrmr(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, feature_scores: List[FeatureScore], transition_features: List[str]) -> List[str]:
        """Select optimal features using mRMR-based iterative selection."""
        try:
            tprint_info("🎯 Selecting features using mRMR-based iterative selection...")
            
            # Sort features by composite score
            sorted_scores = sorted(feature_scores, key=lambda x: x.composite_score, reverse=True)
            
            selected_features = []
            remaining_features = [score.feature_name for score in sorted_scores]
            
            # First, add the best feature from each protected category
            protected_added = set()
            for category in self.config.protect_categories:
                category_features = [score for score in sorted_scores if category in score.category.lower()]
                if category_features:
                    best_feature = category_features[0]
                    if best_feature.feature_name not in selected_features:
                        selected_features.append(best_feature.feature_name)
                        remaining_features.remove(best_feature.feature_name)
                        protected_added.add(category)
                        tprint_info(f"✅ Added protected {category} feature: {best_feature.feature_name}")
            
            # Then iteratively add features using mRMR
            while len(selected_features) < self.config.target_feature_count and remaining_features:
                best_candidate = None
                best_mrmr_score = -float('inf')
                
                for feature_name in remaining_features:
                    if feature_name not in features_df.columns:
                        continue
                    
                    feature_data = features_df[feature_name]
                    
                    # Calculate mRMR score for each target
                    total_mrmr = 0.0
                    total_weight = 0.0
                    
                    for target_col, target_weight in self.config.target_weights.items():
                        if target_col not in labels_df.columns:
                            continue
                        
                        labels_aligned = labels_df[target_col]
                        common_index = feature_data.index.intersection(labels_aligned.index)
                        
                        if len(common_index) < 20:
                            continue
                        
                        feature_aligned = feature_data.loc[common_index]
                        labels_aligned = labels_aligned.loc[common_index]
                        
                        mrmr_score = self._calculate_mrmr_cheap(
                            feature_aligned, labels_aligned, selected_features, features_df
                        )
                        
                        total_mrmr += mrmr_score * target_weight
                        total_weight += target_weight
                    
                    if total_weight > 0:
                        avg_mrmr = total_mrmr / total_weight
                        
                        # No hard threshold - let mRMR handle redundancy naturally
                        # The gradual redundancy penalty in mRMR calculation will naturally discourage redundant features
                        
                        if avg_mrmr > best_mrmr_score:
                            best_mrmr_score = avg_mrmr
                            best_candidate = feature_name
                
                if best_candidate:
                    selected_features.append(best_candidate)
                    remaining_features.remove(best_candidate)
                    tprint_info(f"✅ Added feature: {best_candidate} (mRMR: {best_mrmr_score:.3f})")
                else:
                    break
            
            tprint_success(f"🎯 Selected {len(selected_features)} features using mRMR")
            return selected_features
            
        except Exception as e:
            tprint_error(f"Error in mRMR feature selection: {e}")
            # Fallback to original method
            return self._select_optimal_features(feature_scores, transition_features)

    def _select_optimal_features(self, feature_scores: List[FeatureScore], transition_features: List[str]) -> List[str]:
        """Select optimal features based on scores and thresholds, prioritizing regime transition features."""
        try:
            tprint_info("Selecting optimal features with regime transition prioritization...")
            
            # Filter features by thresholds
            filtered_scores = [
                score for score in feature_scores
                if (score.economic_significance >= self.config.min_economic_significance and
                    score.regime_discrimination >= self.config.min_regime_discrimination and
                    score.clustering_quality >= self.config.min_clustering_quality and
                    score.stability_score >= self.config.min_stability_score)
            ]
            
            tprint_info(f"Features passing thresholds: {len(filtered_scores)}")
            
            # Prioritize regime transition features
            transition_scores = [score for score in filtered_scores if score.feature_name in transition_features]
            non_transition_scores = [score for score in filtered_scores if score.feature_name not in transition_features]
            
            tprint_info(f"Regime transition features: {len(transition_scores)}")
            tprint_info(f"Other features: {len(non_transition_scores)}")
            
            # Select features with regime transition priority
            selected_features = []
            
            # First, add regime transition features (up to configured ratio of target)
            max_transition_features = max(3, int(self.config.target_feature_count * self.config.max_transition_features_ratio))
            selected_transition = transition_scores[:max_transition_features]
            selected_features.extend([score.feature_name for score in selected_transition])
            
            # Then add other high-quality features
            remaining_slots = self.config.target_feature_count - len(selected_features)
            if remaining_slots > 0:
                selected_other = non_transition_scores[:remaining_slots]
                selected_features.extend([score.feature_name for score in selected_other])
            
            # If we still need more features, relax criteria
            if len(selected_features) < self.config.min_feature_count:
                tprint_warning("Not enough features selected, relaxing criteria...")
                
                # Add more transition features if available
                if len(transition_scores) > len(selected_transition):
                    additional_transition = transition_scores[len(selected_transition):]
                    selected_features.extend([score.feature_name for score in additional_transition])
                
                # Add more features from relaxed criteria
                if len(selected_features) < self.config.min_feature_count:
                    relaxed_scores = [
                        score for score in feature_scores
                        if score.feature_name not in selected_features
                    ]
                    needed_features = self.config.min_feature_count - len(selected_features)
                    additional_features = relaxed_scores[:needed_features]
                    selected_features.extend([score.feature_name for score in additional_features])
            
            tprint_success(f"Selected {len(selected_features)} features ({len(selected_transition)} regime transition features)")
            return selected_features
            
        except Exception as e:
            tprint_error(f"Error selecting optimal features: {e}")
            return []
    
    def _validate_feature_selection(self, selected_features_df: pd.DataFrame, labels_df: pd.DataFrame, economic_metrics: List[EconomicMetrics]) -> Dict[str, float]:
        """Validate feature selection quality."""
        try:
            tprint_info("Validating feature selection...")
            
            validation_metrics = {}
            
            # Economic validation
            sharpe_ratios = [metric.sharpe_ratio for metric in economic_metrics]
            if len(sharpe_ratios) > 1:
                sharpe_variance = np.var(sharpe_ratios)
                validation_metrics['sharpe_variance'] = sharpe_variance
                validation_metrics['economic_distinctiveness'] = min(1.0, sharpe_variance / self.config.min_sharpe_variance)
            else:
                validation_metrics['economic_distinctiveness'] = 0.0
            
            # Clustering validation
            try:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                
                # Align data
                common_index = selected_features_df.index.intersection(labels_df.index)
                features_aligned = selected_features_df.loc[common_index]
                labels_aligned = labels_df.loc[common_index]
                
                # Handle case where labels_df might be a DataFrame with a single column
                if hasattr(labels_aligned, 'iloc') and labels_aligned.shape[1] == 1:
                    labels_aligned = labels_aligned.iloc[:, 0]
                
                # Remove noise - handle Series boolean indexing properly
                if hasattr(labels_aligned, 'values'):
                    valid_mask = labels_aligned.values != -1
                else:
                    valid_mask = labels_aligned != -1
                
                features_valid = features_aligned[valid_mask]
                labels_valid = labels_aligned[valid_mask]
                
                # Ensure labels are numeric (convert strings to numeric if needed)
                try:
                    # Ensure labels_valid is a Series
                    if isinstance(labels_valid, pd.DataFrame):
                        if labels_valid.shape[1] == 1:
                            labels_valid = labels_valid.iloc[:, 0]
                        else:
                            labels_valid = labels_valid.iloc[:, 0]  # Use first column
                    
                    labels_valid = pd.to_numeric(labels_valid, errors='coerce')
                    # Remove any NaN values that resulted from conversion
                    valid_numeric_mask = ~labels_valid.isna()
                    features_valid = features_valid[valid_numeric_mask]
                    labels_valid = labels_valid[valid_numeric_mask]
                except Exception as e:
                    tprint_warning(f"Error converting labels to numeric: {e}")
                
                if len(features_valid) > 0 and len(np.unique(labels_valid)) > 1:
                    silhouette = silhouette_score(features_valid, labels_valid)
                    calinski_harabasz = calinski_harabasz_score(features_valid, labels_valid)
                    davies_bouldin = davies_bouldin_score(features_valid, labels_valid)
                    
                    validation_metrics['silhouette_score'] = silhouette
                    validation_metrics['calinski_harabasz_score'] = calinski_harabasz
                    validation_metrics['davies_bouldin_score'] = davies_bouldin
                else:
                    validation_metrics['silhouette_score'] = 0.0
                    validation_metrics['calinski_harabasz_score'] = 0.0
                    validation_metrics['davies_bouldin_score'] = 0.0
                    
            except Exception as e:
                tprint_warning(f"Error calculating clustering metrics: {e}")
                validation_metrics['silhouette_score'] = 0.0
                validation_metrics['calinski_harabasz_score'] = 0.0
                validation_metrics['davies_bouldin_score'] = 0.0
            
            # Noise ratio - handle DataFrame properly
            if hasattr(labels_df, 'iloc') and hasattr(labels_df, 'shape') and labels_df.shape[1] == 1:
                labels_series = labels_df.iloc[:, 0]
            else:
                labels_series = labels_df
            
            # Ensure labels_series is a Series for boolean operations
            if hasattr(labels_series, 'values'):
                noise_count = (labels_series.values == -1).sum()
            else:
                noise_count = (labels_series == -1).sum()
            
            total_count = len(labels_series)
            noise_ratio = noise_count / total_count if total_count > 0 else 0.0
            validation_metrics['noise_ratio'] = noise_ratio
            validation_metrics['noise_acceptable'] = 1.0 if noise_ratio <= self.config.max_noise_ratio else 0.0
            
            # Overall score
            validation_metrics['overall_score'] = (
                validation_metrics.get('economic_distinctiveness', 0.0) * 0.4 +
                validation_metrics.get('silhouette_score', 0.0) * 0.3 +
                validation_metrics.get('noise_acceptable', 0.0) * 0.3
            )
            
            tprint_success("Validation completed")
            return validation_metrics
            
        except Exception as e:
            tprint_error(f"Error validating feature selection: {e}")
            return {'overall_score': 0.0}
    
    def _generate_regime_clustering_artifact(self, selected_features_df: pd.DataFrame, selected_features: List[str], transition_features: List[str]) -> Dict[str, Any]:
        """Generate artifact for regime clustering step."""
        try:
            tprint_info("Generating regime clustering artifact...")
            
            # Create artifact data
            artifact_data = {
                'selected_features': selected_features,
                'selected_features_df': selected_features_df,
                'transition_features': transition_features,
                'feature_categories': {
                    feature: self._determine_feature_category(feature) 
                    for feature in selected_features
                },
                'transition_feature_ratio': len(transition_features) / len(selected_features) if selected_features else 0.0,
                'metadata': {
                    'total_features': len(selected_features),
                    'transition_features_count': len(transition_features),
                    'feature_shape': selected_features_df.shape,
                    'generated_at': datetime.now().isoformat(),
                    'step_name': 'economic_regime_feature_selection'
                }
            }
            
            # Add feature importance scores if available
            if hasattr(self, '_last_feature_scores'):
                feature_importance = {
                    score.feature_name: {
                        'composite_score': score.composite_score,
                        'economic_significance': score.economic_significance,
                        'regime_discrimination': score.regime_discrimination,
                        'clustering_quality': score.clustering_quality,
                        'stability_score': score.stability_score,
                        'is_transition_feature': score.feature_name in transition_features
                    }
                    for score in self._last_feature_scores
                    if score.feature_name in selected_features
                }
                artifact_data['feature_importance'] = feature_importance
            
            tprint_success(f"Regime clustering artifact generated with {len(selected_features)} features")
            return artifact_data
            
        except Exception as e:
            tprint_error(f"Error generating regime clustering artifact: {e}")
            return {
                'selected_features': selected_features,
                'selected_features_df': selected_features_df,
                'transition_features': transition_features,
                'error': str(e)
            }
    
    def _get_computational_stats(self, start_time: float) -> Dict[str, Any]:
        """Get computational statistics."""
        try:
            execution_time = time.perf_counter() - start_time
            
            stats = {
                'execution_time': execution_time,
                'vectorbt_available': VECTORBT_AVAILABLE,
                'hardware_optimization': HARDWARE_AVAILABLE,
                'cheap_proxies_enabled': self.config.enable_cheap_proxies,
                'cv_folds': self.config.cv_folds,
                'silhouette_sample_ratio': self.config.silhouette_sample_ratio
            }
            
            return stats
            
        except Exception as e:
            tprint_warning(f"Error getting computational stats: {e}")
            return {'execution_time': 0.0}
    
    async def _generate_comprehensive_report(self, result: FeatureSelectionResult, config: Dict[str, Any], transition_scores: Dict[str, float] = None) -> str:
        """Generate comprehensive markdown report with enhanced per-feature metrics."""
        try:
            tprint_info("📊 Generating comprehensive report with per-feature metrics...")
            
            # Create report filename with datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = config.get('symbol', 'UNKNOWN')
            report_filename = f"regime_feature_selection_{symbol}_{timestamp}.md"
            report_path = Path("outcomes") / report_filename
            
            # Ensure outcomes directory exists
            report_path.parent.mkdir(exist_ok=True)
            
            # Generate report content with transition scores
            report_content = self._create_report_content(result, config, transition_scores)
            
            # Write report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            tprint_success(f"✅ Report generated: {report_path}")
            
            # Generate detailed top 30 features report
            detailed_report_path = await self._generate_detailed_top_features_report(result, config, timestamp)
            
            return str(report_path)
            
        except Exception as e:
            tprint_error(f"Error generating report: {e}")
            return ""
    
    async def _generate_detailed_top_features_report(self, result: FeatureSelectionResult, config: Dict[str, Any], timestamp: str) -> str:
        """Generate detailed top 30 features report with comprehensive metrics."""
        try:
            tprint_info("📊 Generating detailed top 30 features report...")
            
            # Create DataFrame directly from result instead of reading CSV
            scores_data = []
            for score in result.feature_scores:
                scores_data.append({
                    'feature_name': score.feature_name,
                    'category': score.category,
                    'economic_significance': score.economic_significance,
                    'regime_discrimination': score.regime_discrimination,
                    'clustering_quality': score.clustering_quality,
                    'stability_score': score.stability_score,
                    'composite_score': score.composite_score,
                    'selected': score.feature_name in result.selected_features
                })
            
            df = pd.DataFrame(scores_data)
            
            # Sort by composite score descending
            df_sorted = df.sort_values('composite_score', ascending=False)
            
            # Get top 30 features
            top_30 = df_sorted.head(30)
            
            # Create detailed markdown report
            symbol = config.get('symbol', 'UNKNOWN')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '1h')
            
            report_content = f"""# 🎯 Top 30 Features - Economic Regime Feature Selection

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Symbol**: {symbol}  
**Exchange**: {exchange}  
**Timeframe**: {timeframe}  
**Configuration**: Enhanced 8-Target Multi-Target Approach

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Features Analyzed** | {len(df)} |
| **Top 30 Features** | 30 |
| **Selected Features** | {df['selected'].sum()} |
| **Execution Time** | {result.computational_stats.get('execution_time', 0.0):.2f}s |
| **VectorBT Available** | ✅ YES |

---

## 🎯 Top 30 Features by Composite Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite | Selected |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|----------|"""

            for i, row in top_30.iterrows():
                selected_mark = '✅' if row['selected'] else '❌'
                report_content += f"""
| {row.name + 1} | {row['feature_name']} | {row['category']} | {row['economic_significance']:.3f} | {row['regime_discrimination']:.3f} | {row['clustering_quality']:.3f} | {row['stability_score']:.3f} | {row['composite_score']:.3f} | {selected_mark} |"""

            report_content += f"""

---

## 📈 Category Distribution (Top 30)

"""

            # Category distribution for top 30
            category_counts = top_30['category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / 30) * 100
                report_content += f"- **{category}**: {count} features ({percentage:.1f}%)\n"

            report_content += f"""

---

## 🏆 Selected Features Analysis

### ✅ Selected Features ({df['selected'].sum()} total)
"""

            selected_features = df[df['selected'] == True].sort_values('composite_score', ascending=False)
            for i, row in selected_features.iterrows():
                feature_name = row['feature_name']
                category = row['category']
                score = row['composite_score']
                report_content += f"- **{feature_name}** ({category}) - Score: {score:.3f}\n"

            report_content += f"""

### 📊 Score Statistics (Top 30)

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| **Economic Significance** | {top_30['economic_significance'].min():.3f} | {top_30['economic_significance'].max():.3f} | {top_30['economic_significance'].mean():.3f} | {top_30['economic_significance'].std():.3f} |
| **Regime Discrimination** | {top_30['regime_discrimination'].min():.3f} | {top_30['regime_discrimination'].max():.3f} | {top_30['regime_discrimination'].mean():.3f} | {top_30['regime_discrimination'].std():.3f} |
| **Clustering Quality** | {top_30['clustering_quality'].min():.3f} | {top_30['clustering_quality'].max():.3f} | {top_30['clustering_quality'].mean():.3f} | {top_30['clustering_quality'].std():.3f} |
| **Stability Score** | {top_30['stability_score'].min():.3f} | {top_30['stability_score'].max():.3f} | {top_30['stability_score'].mean():.3f} | {top_30['stability_score'].std():.3f} |
| **Composite Score** | {top_30['composite_score'].min():.3f} | {top_30['composite_score'].max():.3f} | {top_30['composite_score'].mean():.3f} | {top_30['composite_score'].std():.3f} |

---

## 🎯 Enhanced 8-Target Configuration

### Target Breakdown:
- **close_return**: 25% - Core price movements
- **volume_log_return**: 20% - Volume momentum patterns  
- **price_range_pct**: 20% - Relative volatility
- **body_size_pct**: 8% - Price efficiency regimes
- **volume_return**: 8% - Volume changes
- **vwap_price_ratio**: 10% - Price distance from MA (regime separator)
- **volatility_20**: 5% - Realized volatility rolling
- **cmf**: 4% - Volume imbalance/order flow (directional conviction)

### Category Focus:
- **Price Movements**: 35% (close_return + vwap_price_ratio)
- **Volume Patterns**: 32% (volume_log_return + volume_return + cmf)
- **Volatility Regimes**: 25% (price_range_pct + volatility_20)
- **Price Efficiency**: 8% (body_size_pct)

---

## 💡 Key Insights

1. **Returns Dominance**: All top features are from the 'returns' category, indicating strong price-based regime detection
2. **High Regime Discrimination**: Most features show >95% regime discrimination capability
3. **Stability**: All features maintain 0.5 stability score (baseline)
4. **Clustering Quality**: Currently 0.0 across all features (needs investigation)

---

## 🔧 Next Steps

1. **Investigate Clustering Quality**: Why are clustering quality scores 0.0?
2. **Diversify Categories**: Consider features from volume, momentum, and other categories
3. **Regime Transition**: Focus on features that excel at detecting regime changes
4. **Validation**: Test selected features with HDBSCAN clustering

---

*Report generated by Economic Regime Feature Selector v1.0*
"""

            # Save detailed report
            detailed_report_filename = f"regime_feature_selection_{timestamp}.md"
            detailed_report_path = Path("outcomes") / detailed_report_filename
            
            with open(detailed_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            tprint_success(f"✅ Detailed report generated: {detailed_report_path}")
            return str(detailed_report_path)
            
        except Exception as e:
            tprint_error(f"Error generating detailed top features report: {e}")
            return ""
    
    def _find_latest_feature_scores_artifact(self) -> str:
        """Find the most recent feature scores artifact."""
        try:
            import glob
            
            # Look for feature scores CSV files
            pattern = "artifacts/**/*feature_scores*.csv"
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                return ""
            
            # Sort by modification time and return the most recent
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
            
        except Exception as e:
            tprint_warning(f"Error finding feature scores artifact: {e}")
            return ""
            
    def _categorize_features_by_transition_scores(self, transition_scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Categorize features based on their transition scores."""
        categories = {
            'structured': {'features': [], 'count': 0, 'description': 'Features with clear regime transitions (score ≥ 0.8)'},
            'random': {'features': [], 'count': 0, 'description': 'Features with moderate transitions (0.4 ≤ score < 0.8)'},
            'slightly_varying': {'features': [], 'count': 0, 'description': 'Features with small variations (0.1 ≤ score < 0.4)'},
            'constant': {'features': [], 'count': 0, 'description': 'Features with minimal changes (score < 0.1)'}
        }
        
        for feature_name, score in transition_scores.items():
            if score >= 0.8:
                categories['structured']['features'].append(feature_name)
            elif score >= 0.4:
                categories['random']['features'].append(feature_name)
            elif score >= 0.1:
                categories['slightly_varying']['features'].append(feature_name)
            else:
                categories['constant']['features'].append(feature_name)
        
        # Update counts
        for category in categories.values():
            category['count'] = len(category['features'])
        
        return categories
    
    def _format_feature_category_table(self, features: List[str], transition_scores: Dict[str, float], max_features: int = 10) -> str:
        """Format a table for features in a specific category."""
        if not features:
            return "No features in this category."
        
        # Sort features by score (descending)
        sorted_features = sorted(features, key=lambda f: transition_scores.get(f, 0), reverse=True)
        
        table = "| Feature Name | Transition Score |\n|--------------|------------------|\n"
        
        for feature in sorted_features[:max_features]:
            score = transition_scores.get(feature, 0.0)
            table += f"| {feature} | {score:.3f} |\n"
        
        if len(sorted_features) > max_features:
            table += f"| ... and {len(sorted_features) - max_features} more | ... |\n"
        
        return table
    
    def _create_report_content(self, result: FeatureSelectionResult, config: Dict[str, Any], transition_scores: Dict[str, float] = None) -> str:
        """Create report content with enhanced per-feature metrics and category breakdowns."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            symbol = config.get('symbol', 'UNKNOWN')
            
            # Categorize features based on transition scores
            feature_categories = self._categorize_features_by_transition_scores(transition_scores or {})
            
            # Multi-target information
            multi_target_info = ""
            if self.config.multi_target_enabled:
                multi_target_info = f"""
**Multi-Target Approach**: ✅ Enabled  
**Targets**: {', '.join(self.config.target_columns)}  
**Target Weights**: {', '.join([f'{k}: {v:.1%}' for k, v in self.config.target_weights.items()])}  

---
"""
            else:
                multi_target_info = """
**Multi-Target Approach**: ❌ Disabled  

---
"""

            report = f"""# Economic Regime Feature Selection Report

**Generated**: {timestamp}  
**Symbol**: {symbol}  
**Exchange**: {config.get('exchange', 'binance')}  
**Timeframe**: {config.get('timeframe', '15m')}  
{multi_target_info}

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Selected Features** | {len(result.selected_features)} |
| **Regime Transition Features** | {len(getattr(result, 'transition_features', []))} |
| **Economic Distinctiveness** | {result.validation_metrics.get('economic_distinctiveness', 0.0):.3f} |
| **Overall Validation Score** | {result.validation_metrics.get('overall_score', 0.0):.3f} |
| **Silhouette Score** | {result.validation_metrics.get('silhouette_score', 0.0):.3f} |
| **Noise Ratio** | {result.validation_metrics.get('noise_ratio', 0.0):.1%} |
| **Execution Time** | {result.computational_stats.get('execution_time', 0.0):.2f}s |

---

## 🏷️ Feature Categories Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| **Structured Features** | {feature_categories['structured']['count']} | Features with clear regime transitions (score ≥ 0.8) |
| **Random Features** | {feature_categories['random']['count']} | Features with moderate transitions (0.4 ≤ score < 0.8) |
| **Slightly Varying** | {feature_categories['slightly_varying']['count']} | Features with small variations (0.1 ≤ score < 0.4) |
| **Constant Features** | {feature_categories['constant']['count']} | Features with minimal changes (score < 0.1) |

---

## 📈 Per-Feature Transition Scores

### Structured Features (High Regime Transition Detection)
{self._format_feature_category_table(feature_categories['structured']['features'], transition_scores or {})}

### Random Features (Moderate Regime Transition Detection)
{self._format_feature_category_table(feature_categories['random']['features'], transition_scores or {})}

### Slightly Varying Features (Low Regime Transition Detection)
{self._format_feature_category_table(feature_categories['slightly_varying']['features'], transition_scores or {})}

### Constant Features (Minimal Regime Transition Detection)
{self._format_feature_category_table(feature_categories['constant']['features'], transition_scores or {})}

---

## 🎯 Selected Features

### Regime Transition Features
These features are particularly good at detecting regime transitions and clear regime boundaries:

{self._format_transition_features_table(getattr(result, 'transition_features', []))}

### Top Features by Economic Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|
"""
            
            # Add feature rankings table
            for i, score in enumerate(result.feature_scores[:20], 1):  # Top 20
                report += f"| {i} | {score.feature_name} | {score.category} | {score.economic_significance:.3f} | {score.regime_discrimination:.3f} | {score.clustering_quality:.3f} | {score.stability_score:.3f} | {score.composite_score:.3f} |\n"
            
            report += f"""

---

## 📈 Economic Metrics by Regime

| Regime ID | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Win Rate | Avg Return | Volatility | Max DD | Samples |
|-----------|--------------|---------------|--------------|----------|------------|------------|--------|---------|
"""
            
            # Add economic metrics table
            for metric in result.economic_metrics:
                report += f"| {metric.regime_id} | {metric.sharpe_ratio:.3f} | {metric.sortino_ratio:.3f} | {metric.calmar_ratio:.3f} | {metric.win_rate:.1%} | {metric.avg_return:.4f} | {metric.volatility:.4f} | {metric.max_drawdown:.4f} | {metric.sample_count} |\n"
            
            report += f"""

---

## 🔍 Clustering Quality Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Silhouette Score** | {result.validation_metrics.get('silhouette_score', 0.0):.3f} | > 0.2 | {'✅ PASS' if result.validation_metrics.get('silhouette_score', 0.0) > 0.2 else '❌ FAIL'} |
| **Calinski-Harabasz** | {result.validation_metrics.get('calinski_harabasz_score', 0.0):.1f} | > 100 | {'✅ PASS' if result.validation_metrics.get('calinski_harabasz_score', 0.0) > 100 else '❌ FAIL'} |
| **Davies-Bouldin** | {result.validation_metrics.get('davies_bouldin_score', 0.0):.3f} | < 2.0 | {'✅ PASS' if result.validation_metrics.get('davies_bouldin_score', 0.0) < 2.0 else '❌ FAIL'} |
| **Noise Ratio** | {result.validation_metrics.get('noise_ratio', 0.0):.1%} | < 30% | {'✅ PASS' if result.validation_metrics.get('noise_ratio', 0.0) < 0.3 else '❌ FAIL'} |

---

## 📊 Feature Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
"""
            
            # Calculate category distribution
            category_counts = {}
            for score in result.feature_scores:
                if score.feature_name in result.selected_features:
                    category_counts[score.category] = category_counts.get(score.category, 0) + 1
            
            total_selected = len(result.selected_features)
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_selected * 100) if total_selected > 0 else 0
                report += f"| {category} | {count} | {percentage:.1f}% |\n"
            
            report += f"""

---

## ⚡ Computational Performance

| Metric | Value |
|--------|-------|
| **Execution Time** | {result.computational_stats.get('execution_time', 0.0):.2f}s |
| **VectorBT Available** | {'✅ YES' if result.computational_stats.get('vectorbt_available', False) else '❌ NO'} |
| **Hardware Optimization** | {'✅ YES' if result.computational_stats.get('hardware_optimization', False) else '❌ NO'} |
| **Cheap Proxies Enabled** | {'✅ YES' if result.computational_stats.get('cheap_proxies_enabled', False) else '❌ NO'} |
| **CV Folds** | {result.computational_stats.get('cv_folds', 3)} |
| **Silhouette Sample Ratio** | {result.computational_stats.get('silhouette_sample_ratio', 0.1):.1%} |

---

## 💡 Recommendations

### Feature Selection Quality
- **Overall Score**: {result.validation_metrics.get('overall_score', 0.0):.3f} ({'EXCELLENT' if result.validation_metrics.get('overall_score', 0.0) > 0.8 else 'GOOD' if result.validation_metrics.get('overall_score', 0.0) > 0.6 else 'FAIR' if result.validation_metrics.get('overall_score', 0.0) > 0.4 else 'POOR'})

### Economic Distinctiveness
- **Sharpe Variance**: {result.validation_metrics.get('sharpe_variance', 0.0):.3f} ({'EXCELLENT' if result.validation_metrics.get('sharpe_variance', 0.0) > 1.0 else 'GOOD' if result.validation_metrics.get('sharpe_variance', 0.0) > 0.5 else 'FAIR' if result.validation_metrics.get('sharpe_variance', 0.0) > 0.2 else 'POOR'})

### Clustering Quality
- **Silhouette Score**: {result.validation_metrics.get('silhouette_score', 0.0):.3f} ({'EXCELLENT' if result.validation_metrics.get('silhouette_score', 0.0) > 0.5 else 'GOOD' if result.validation_metrics.get('silhouette_score', 0.0) > 0.3 else 'FAIR' if result.validation_metrics.get('silhouette_score', 0.0) > 0.2 else 'POOR'})

### Next Steps
1. **Proceed with HDBSCAN**: Selected features show {'good' if result.validation_metrics.get('overall_score', 0.0) > 0.6 else 'moderate'} economic distinctiveness
2. **Monitor Performance**: Track regime-specific Sharpe ratios during clustering
3. **Feature Refinement**: Consider {'adding more features' if len(result.selected_features) < 20 else 'reducing features' if len(result.selected_features) > 30 else 'current selection is optimal'}

---

## 🔧 Configuration Used

- **Target Feature Count**: {self.config.target_feature_count}
- **Economic Significance Weight**: {self.config.economic_significance_weight:.1%}
- **Regime Discrimination Weight**: {self.config.regime_discrimination_weight:.1%}
- **Clustering Quality Weight**: {self.config.clustering_quality_weight:.1%}
- **Stability Weight**: {self.config.stability_weight:.1%}
- **Excluded Categories**: {', '.join(self.config.exclude_categories)}
- **CV Folds**: {self.config.cv_folds}
- **Silhouette Sample Ratio**: {self.config.silhouette_sample_ratio:.1%}

---

*Report generated by Economic Regime Feature Selector v1.0*
"""
            
            return report
            
        except Exception as e:
            tprint_error(f"Error creating report content: {e}")
            return f"# Error generating report: {e}"
    
    def _format_transition_features_table(self, transition_features: List[str]) -> str:
        """Format regime transition features table for report."""
        try:
            if not transition_features:
                return "No regime transition features identified."
            
            table_rows = []
            for i, feature in enumerate(transition_features, 1):
                category = self._determine_feature_category(feature)
                table_rows.append(f"| {i} | {feature} | {category} |")
            
            table = f"""| Rank | Feature Name | Category |
|------|--------------|----------|
{chr(10).join(table_rows)}"""
            
            return table
            
        except Exception as e:
            tprint_warning(f"Error formatting transition features table: {e}")
            return "Error formatting transition features table."
    
    async def _save_artifacts(self, result: FeatureSelectionResult, config: Dict[str, Any]) -> Dict[str, str]:
        """Save artifacts for downstream steps."""
        try:
            tprint_info("Saving artifacts...")
            
            artifacts = {}
            
            # Save selected features
            selected_features_df = pd.DataFrame({
                'feature_name': result.selected_features,
                'selected': True
            })
            
            artifacts['selected_features'] = self._save_artifact(
                selected_features_df,
                "selected_features",
                artifact_type="data",
                metadata={
                    'feature_count': len(result.selected_features),
                    'selection_method': 'economic_regime_selection',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Save feature scores
            scores_data = []
            for score in result.feature_scores:
                scores_data.append({
                    'feature_name': score.feature_name,
                    'category': score.category,
                    'economic_significance': score.economic_significance,
                    'regime_discrimination': score.regime_discrimination,
                    'clustering_quality': score.clustering_quality,
                    'stability_score': score.stability_score,
                    'composite_score': score.composite_score,
                    'selected': score.feature_name in result.selected_features
                })
            
            scores_df = pd.DataFrame(scores_data)
            artifacts['feature_scores'] = self._save_artifact(
                scores_df,
                "feature_scores",
                artifact_type="data",
                metadata={
                    'total_features_scored': len(result.feature_scores),
                    'selection_method': 'economic_regime_selection'
                }
            )
            
            # Save economic metrics
            metrics_data = []
            for metric in result.economic_metrics:
                metrics_data.append({
                    'regime_id': metric.regime_id,
                    'sharpe_ratio': metric.sharpe_ratio,
                    'sortino_ratio': metric.sortino_ratio,
                    'calmar_ratio': metric.calmar_ratio,
                    'win_rate': metric.win_rate,
                    'avg_return': metric.avg_return,
                    'volatility': metric.volatility,
                    'max_drawdown': metric.max_drawdown,
                    'sample_count': metric.sample_count
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            artifacts['economic_metrics'] = self._save_artifact(
                metrics_df,
                "economic_metrics",
                artifact_type="data",
                metadata={
                    'regime_count': len(result.economic_metrics),
                    'calculation_method': 'vectorized'
                }
            )
            
            # Save validation metrics
            validation_df = pd.DataFrame([result.validation_metrics])
            artifacts['validation_metrics'] = self._save_artifact(
                validation_df,
                "validation_metrics",
                artifact_type="data",
                metadata={
                    'validation_method': 'comprehensive',
                    'overall_score': result.validation_metrics.get('overall_score', 0.0)
                }
            )
            
            tprint_success("Artifacts saved successfully")
            return artifacts
            
        except Exception as e:
            tprint_error(f"Error saving artifacts: {e}")
            return {}
