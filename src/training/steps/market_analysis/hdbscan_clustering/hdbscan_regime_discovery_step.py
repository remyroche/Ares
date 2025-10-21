"""
HDBSCAN Regime Discovery Step

BaseClass-based step that replaces NAS/TAS regime discovery with HDBSCAN-based approach.
Integrates with ares_launcher.py and uses the comprehensive regime discovery system.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import time
import gc
import psutil

# Import BaseClass and step registry
from src.training.steps.base_step import BaseStep

# Import HDBSCAN regime discovery system
from src.training.steps.market_analysis.hdbscan_clustering import (
    HDBSCANRegimeDiscovery, 
    RegimeDiscoveryConfig,
    RegimeResult
)

# Import new data-driven clustering system
from src.training.steps.market_analysis.hdbscan_clustering.optimization.data_driven_clustering_optimizer import (
    DataDrivenClusteringOptimizer, DataDrivenClusteringResult
)
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import (
    DataDrivenClusteringConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering.advanced_financial_features import (
    AdvancedFinancialFeatureEngineer, AdvancedFeatureConfig
)

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, tprint_data_preview, tprint_data_format, LogLevel
)
from src.utils.hardware import get_memory_usage, optimize_dataframe_default
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.serialization_utils import save_pickle, load_pickle

# Import enhanced artifact management
from src.utils.artifact_manager import ArtifactManager, ArtifactConfig

# Import enhanced common operations and utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    safe_numeric_operation, optimize_dataframe_memory,
    validate_dataframe_structure, safe_dataframe_merge
)
from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols,
    optimize_dataframe_memory as optimize_df_memory
)
from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power,
    validate_array, validate_numeric_range, safe_statistical_operation
)
# Memory optimization now handled by hardware module

logger = logging.getLogger(__name__)


class HDBSCANRegimeDiscoveryStep(BaseStep):
    """
    HDBSCAN-based regime discovery step that replaces NAS/TAS approach.
    
    Features:
    - 5 feature families (Returns, Volatility, Volume/Flow, Entropy, Spectral)
    - Multi-mode dimensionality reduction (PCA/UMAP/densMAP)
    - HDBSCAN clustering with tree export
    - Post-clustering optimization with change budget
    - Economic validation and profiling
    - Temporal stabilization with causal/acausal modes
    - Deterministic reproducibility
    - Hardware optimization for M1 systems
    """
    
    def __init__(self, step_name: str = "hdbscan_regime_discovery", config: Optional[Dict[str, Any]] = None):
        """Initialize the HDBSCAN regime discovery step with economic validation."""
        tprint_debug(f"Initializing HDBSCANRegimeDiscoveryStep with name: {step_name}")
        start_time = time.perf_counter()
        
        super().__init__(step_name, config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regime discovery system
        self.regime_discovery = None
        self.config = None
        
        # Initialize data-driven clustering optimizer
        self.enable_data_driven = config.get('enable_data_driven', True) if config else True
        self.enable_economic_validation = config.get('enable_economic_validation', True) if config else True
        
        if self.enable_data_driven:
            self.data_driven_config = DataDrivenClusteringConfig()
            self.data_driven_optimizer = DataDrivenClusteringOptimizer(self.data_driven_config)
            self.advanced_feature_engineer = AdvancedFinancialFeatureEngineer(AdvancedFeatureConfig())
            tprint_info("✅ Data-driven clustering optimizer initialized")
        
        # Performance tracking
        self.performance_stats = {
            'initialization_time': 0.0,
            'memory_usage_mb': 0.0,
            'total_operations': 0
        }
        
        # Track initialization time
        init_time = time.perf_counter() - start_time
        self.performance_stats['initialization_time'] = init_time
        self.performance_stats['memory_usage_mb'] = get_memory_usage()
        
        tprint_success(f"🚀 HDBSCANRegimeDiscoveryStep initialized in {init_time:.3f}s")
        tprint_debug(f"Initial memory usage: {self.performance_stats['memory_usage_mb']:.2f}MB")
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HDBSCAN regime discovery step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - execution_mode: 'full', 'light', or 'blank'
                - live_mode: Whether this is live trading (default: False)
                
        Returns:
            Dictionary with execution results, artifacts, and metrics
        """
        start_time = datetime.now()
        perf_start = time.perf_counter()
        
        # Enhanced data format analysis for troubleshooting
        tprint_data_format(config, "step_config", level=LogLevel.INFO)
        
        try:
            symbol = config.get('symbol', 'UNKNOWN')
            tprint_info(f"🔍 Starting HDBSCAN regime discovery for {symbol}")
            tprint_debug(f"Configuration: {config}")
            
            # Memory optimization: Clean up before starting
            gc.collect()
            initial_memory = get_memory_usage()
            tprint_debug(f"Initial memory usage: {initial_memory:.2f}MB")
            
            # Validate required parameters
            with tprint_timer("Configuration validation"):
                self._validate_config(config)
            
            # Create regime discovery configuration
            with tprint_timer("Regime discovery configuration creation"):
                self.config = self._create_regime_discovery_config(config)
                tprint_debug(f"Created config with execution_mode: {config.get('execution_mode', 'light')}")
            
            # Initialize regime discovery system
            with tprint_timer("Regime discovery system initialization"):
                self.regime_discovery = HDBSCANRegimeDiscovery(self.config)
                tprint_debug("Regime discovery system initialized successfully")
            
            # Load market data
            with tprint_timer("Market data loading"):
                market_data = self._load_market_data(config)
                if market_data is None or len(market_data) == 0:
                    raise ValueError("Failed to load market data")
                
                # Data preview of loaded market data
                tprint_data_preview(market_data, "loaded_market_data", max_rows=10, level="INFO")
                
                # Optimize memory usage
                market_data = optimize_dataframe_default(market_data)
                tprint_success(f"✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
                tprint_debug(f"Data memory usage: {market_data.memory_usage(deep=True).sum() / 1024**2:.2f}MB")
            
            # Extract returns for economic validation
            with tprint_timer("Returns extraction"):
                returns = self._extract_returns(market_data)
                if returns is not None:
                    tprint_data_preview(returns, "extracted_returns", max_rows=10, level="DEBUG")
                    tprint_debug(f"Extracted returns: {len(returns)} samples")
                else:
                    tprint_warning("No returns data available for economic validation")
            
            # Execute regime discovery
            with tprint_timer("Regime discovery execution"):
                # Data preview before regime discovery
                tprint_data_preview(market_data, "regime_discovery_input", max_rows=5, level="DEBUG")
                
                regime_result = await self.regime_discovery.discover_regimes(
                    data=market_data,
                    fit=True,
                    is_live=config.get('live_mode', False),
                    returns=returns
                )
                
                if not regime_result.success:
                    raise ValueError(f"Regime discovery failed: {regime_result.error_message}")
                
                # Data preview of regime discovery results
                tprint_data_preview(regime_result.labels, "regime_discovery_labels", max_rows=10, level="INFO")
                tprint_data_preview(regime_result.validation_metrics, "regime_validation_metrics", level="DEBUG")
                
                tprint_success(f"✅ Regime discovery completed: {regime_result.validation_metrics['n_regimes']} regimes")
            
            # Economic validation and data-driven optimization
            if self.enable_data_driven and self.enable_economic_validation:
                with tprint_timer("Economic validation and data-driven optimization"):
                    economic_validation_result = await self._perform_economic_validation(
                        regime_result, market_data, config
                    )
                    tprint_success("✅ Economic validation completed")
            else:
                economic_validation_result = None
            
            # Create artifacts
            with tprint_timer("Artifacts creation"):
                artifacts = self._create_artifacts(regime_result, config)
                tprint_data_preview(artifacts, "created_artifacts", level="INFO")
                tprint_debug(f"Created {len(artifacts)} artifact categories")
            
            # Save artifacts
            with tprint_timer("Artifacts saving"):
                self._save_artifacts(artifacts, config)
                tprint_debug("Artifacts saved successfully")
            
            # Calculate metrics
            with tprint_timer("Metrics calculation"):
                metrics = self._calculate_metrics(regime_result, start_time)
                tprint_debug(f"Calculated metrics: {list(metrics.keys())}")
            
            # Create comprehensive outcome report
            with tprint_timer("Outcome report creation"):
                outcome_report = self._create_outcome_report(regime_result, metrics, config)
                tprint_debug(f"Outcome report created: {len(outcome_report)} characters")
            
            # Save outcome report to outcomes/ directory
            with tprint_timer("Outcome report saving"):
                report_path = self._save_outcome_report(outcome_report, config)
                tprint_info(f"📊 Comprehensive report saved to: {report_path}")
            
            # Final performance metrics
            total_time = (datetime.now() - start_time).total_seconds()
            perf_time = time.perf_counter() - perf_start
            final_memory = get_memory_usage()
            
            tprint_performance("HDBSCAN regime discovery", total_time)
            tprint_success(f"✅ HDBSCAN regime discovery completed: {regime_result.validation_metrics['n_regimes']} regimes")
            tprint_debug(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (delta: {final_memory - initial_memory:+.2f}MB)")
            
            # Update performance stats
            self.performance_stats['total_operations'] += 1
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'report_path': report_path,
                'regime_result': regime_result,
                'processing_time': total_time,
                'performance_time': perf_time,
                'memory_usage': {
                    'initial_mb': initial_memory,
                    'final_mb': final_memory,
                    'delta_mb': final_memory - initial_memory
                }
            }
            
        except Exception as e:
            error_msg = f"HDBSCAN regime discovery failed: {str(e)}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            # Data preview for error case
            error_data = {
                'error_message': error_msg,
                'config': config,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'performance_time': time.perf_counter() - perf_start
            }
            tprint_data_preview(error_data, "comprehensive_error_case_data", level="ERROR")
            
            # Clean up on error
            gc.collect()
            
            return {
                'success': False,
                'error': error_msg,
                'artifacts': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'performance_time': time.perf_counter() - perf_start
            }
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        tprint_debug("Validating configuration parameters")
        
        required_params = ['symbol', 'exchange', 'timeframe']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            error_msg = f"Missing required parameters: {missing_params}"
            tprint_error(error_msg)
            raise ValueError(error_msg)
        
        tprint_debug(f"Configuration validation passed. Symbol: {config.get('symbol')}, Exchange: {config.get('exchange')}, Timeframe: {config.get('timeframe')}")
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def _create_regime_discovery_config(self, config: Dict[str, Any]) -> RegimeDiscoveryConfig:
        """Create regime discovery configuration from step config."""
        execution_mode = config.get('execution_mode', 'light')
        tprint_info(f"Creating regime discovery config for execution mode: {execution_mode}")
        
        if execution_mode == 'full':
            tprint_debug("Using FULL execution mode configuration")
            # Full mode: comprehensive regime discovery
            return RegimeDiscoveryConfig(
                # Feature extraction
                enabled_feature_families=[
                    "returns_momentum", "volatility", "volume_flow", 
                    "entropy_complexity", "spectral"
                ],
                total_max_features=300,
                enable_pid_features=True,
                enable_hybrid_features=True,
                enable_hardware_optimization=True,
                
                # Preprocessing
                transformer_type="quantile",
                correlation_threshold=0.95,
                mutual_info_threshold=0.05,
                
                # Dimensionality reduction
                dim_reduction_mode="umap",  # Use UMAP for full mode
                pca_variance_threshold=0.98,
                umap_n_components=8,
                umap_n_neighbors=30,
                
                # HDBSCAN clustering
                min_cluster_size_pct=0.01,
                min_cluster_size_floor=12,
                cluster_selection_method="eom",
                
                # Post-clustering optimization
                change_budget_pct=0.10,
                max_optimization_rounds=100,
                use_condensed_tree=True,
                
                # Economic validation
                min_economic_separation_pct=0.7,
                interpretable_axes=[
                    "trend_pc", "vol_pc", "breadth_pc", "skew_pc", "liquidity_stress_pc"
                ],
                
                # Temporal stabilization
                smoothing_window=5,
                min_dwell_bars=3,
                cooldown_bars_after_switch=2,
                
                # Determinism
                random_state=42,
                numpy_seed=42,
                pin_blas_threads=True
            )
            
        elif execution_mode == 'light':
            tprint_debug("Using LIGHT execution mode configuration")
            # Light mode: essential regime discovery
            return RegimeDiscoveryConfig(
                # Feature extraction
                enabled_feature_families=[
                    "returns_momentum", "volatility", "volume_flow"
                ],
                total_max_features=150,
                enable_pid_features=True,
                enable_hybrid_features=False,
                enable_hardware_optimization=True,
                
                # Preprocessing
                transformer_type="robust",
                correlation_threshold=0.95,
                
                # Dimensionality reduction
                dim_reduction_mode="pca_only",  # Use PCA-only for light mode
                pca_variance_threshold=0.95,
                
                # HDBSCAN clustering
                min_cluster_size_pct=0.02,
                min_cluster_size_floor=20,
                cluster_selection_method="eom",
                
                # Post-clustering optimization
                change_budget_pct=0.05,
                max_optimization_rounds=50,
                use_condensed_tree=False,
                
                # Economic validation
                min_economic_separation_pct=0.5,
                interpretable_axes=["trend_pc", "vol_pc"],
                
                # Temporal stabilization
                smoothing_window=3,
                min_dwell_bars=2,
                cooldown_bars_after_switch=1,
                
                # Determinism
                random_state=42,
                numpy_seed=42,
                pin_blas_threads=True
            )
            
        else:  # blank mode
            tprint_debug("Using BLANK execution mode configuration")
            # Blank mode: minimal regime discovery
            return RegimeDiscoveryConfig(
                # Feature extraction
                enabled_feature_families=["returns_momentum", "volatility"],
                total_max_features=50,
                enable_pid_features=False,
                enable_hybrid_features=False,
                enable_hardware_optimization=False,
                
                # Preprocessing
                transformer_type="standard",
                correlation_threshold=0.99,
                
                # Dimensionality reduction
                dim_reduction_mode="pca_only",
                pca_variance_threshold=0.90,
                
                # HDBSCAN clustering
                min_cluster_size_pct=0.05,
                min_cluster_size_floor=50,
                cluster_selection_method="leaf",
                
                # Post-clustering optimization
                change_budget_pct=0.01,
                max_optimization_rounds=10,
                use_condensed_tree=False,
                
                # Economic validation
                min_economic_separation_pct=0.3,
                interpretable_axes=["trend_pc"],
                
                # Temporal stabilization
                smoothing_window=1,
                min_dwell_bars=1,
                cooldown_bars_after_switch=0,
                
                # Determinism
                random_state=42,
                numpy_seed=42,
                pin_blas_threads=False
            )
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def _load_market_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load market data using klines manager with comprehensive optimization and validation."""
        try:
            tprint_info("📂 Loading market data...")
            
            # Get klines manager
            data_dir = config.get('data_dir', 'historical_data')
            tprint_debug(f"Using data directory: {data_dir}")
            klines_manager = get_klines_manager(data_dir=data_dir)
            
            # Parse date filters if provided
            start_date = None
            end_date = None
            
            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
                tprint_info(f"📅 Using start_date filter: {start_date}")
            
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])
                tprint_info(f"📅 Using end_date filter: {end_date}")
            
            # Load data
            symbol = config['symbol']
            timeframe = config['timeframe']
            tprint_debug(f"Loading data for {symbol} {timeframe}")
            
            with tprint_timer("Market data loading"):
                market_data = klines_manager.read_data(
                    symbol=symbol,
                    interval=timeframe,
                    data_type="processed",
                    start_date=start_date,
                    end_date=end_date
                )
            
            if market_data is not None and len(market_data) > 0:
                # Enhanced data format analysis for troubleshooting
                tprint_data_format(market_data, "loaded_market_data", level=LogLevel.INFO)
                
                # Enhanced data validation using common operations
                def validate_and_optimize_data(df):
                    # Memory optimization
                    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
                    tprint_debug(f"Initial data memory usage: {initial_memory:.2f}MB")
                    
                    # Ensure timestamp column exists
                    if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                        df = df.copy()
                        df['timestamp'] = df.index
                        tprint_success("✅ Added timestamp column from DatetimeIndex")
                    
                    # Validate dataframe structure
                    if not validate_dataframe_structure(df):
                        tprint_warning("⚠️ DataFrame structure validation failed, applying fixes")
                        # Apply basic fixes
                        df = df.dropna(how='all')  # Remove completely empty rows
                        df = df.select_dtypes(include=[np.number, 'datetime64[ns]'])  # Keep only numeric and datetime columns
                    
                    # Comprehensive memory optimization
                    df = optimize_dataframe_memory(df)
                    final_memory = df.memory_usage(deep=True).sum() / 1024**2
                    memory_saved = initial_memory - final_memory
                    
                    tprint_debug(f"Memory optimization: {initial_memory:.2f}MB -> {final_memory:.2f}MB (saved {memory_saved:.2f}MB)")
                    
                    return df
                
                # Use safe dataframe operation for validation and optimization
                market_data = safe_dataframe_operation(market_data, validate_and_optimize_data)
                
                # Additional memory cleanup
                gc.collect()
                post_gc_memory = get_memory_usage()
                tprint_debug(f"System memory after GC: {post_gc_memory:.2f}MB")
                
                # Enhanced data quality validation
                def validate_data_quality(df):
                    null_count = df.isnull().sum().sum()
                    inf_count = df.isin([np.inf, -np.inf]).sum().sum()
                    
                    tprint_debug(f"Data quality check: {null_count} null values, {inf_count} infinite values")
                    
                    # Check for finite values in numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if not validate_finite(df[col].values):
                            tprint_warning(f"⚠️ Column {col} contains non-finite values")
                    
                    return df
                
                market_data = safe_dataframe_operation(market_data, validate_data_quality)
                
                tprint_success(f"✅ Market data loaded: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
                tprint_info(f"📅 Date range: {market_data.index.min()} to {market_data.index.max()}")
                
                return market_data
            else:
                tprint_error("❌ No market data loaded")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to load market data: {e}")
            return None
    
    def _extract_returns(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract returns from market data for economic validation with enhanced math validation."""
        try:
            if 'close' in market_data.columns:
                # Use safe numeric operation for returns calculation
                def calculate_returns():
                    returns = market_data['close'].pct_change().dropna()
                    
                    # Validate returns are finite
                    if not validate_finite(returns.values):
                        tprint_warning("⚠️ Non-finite values in returns, applying safe operations")
                        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Validate returns range
                    if not validate_numeric_range(returns.values, min_val=-1.0, max_val=1.0):
                        tprint_warning("⚠️ Returns outside expected range [-1, 1], clipping")
                        returns = returns.clip(-1.0, 1.0)
                    
                    return returns.values
                
                returns = safe_numeric_operation(calculate_returns, default=None)
                
                if returns is not None and validate_finite(returns):
                    tprint_debug(f"✅ Returns extracted: {len(returns)} samples, range: [{np.min(returns):.4f}, {np.max(returns):.4f}]")
                    return returns
                else:
                    tprint_warning("⚠️ Returns calculation failed or produced invalid values")
                    return None
            else:
                tprint("⚠️ No 'close' column found for returns calculation", "WARNING")
                return None
        except Exception as e:
            tprint(f"⚠️ Failed to extract returns: {e}", "WARNING")
            return None
    
    def _create_artifacts(self, regime_result: RegimeResult, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create artifacts from regime discovery result with enhanced artifact management."""
        try:
            # Initialize artifact manager
            artifact_config = ArtifactConfig(
                enable_compression=True,
                enable_versioning=True,
                enable_metadata=True,
                compression_level=6
            )
            artifact_manager = ArtifactManager(artifact_config)
            
            # Create comprehensive artifacts
            artifacts = {
                # Core regime data
                'regime_labels': regime_result.labels,
                'regime_probabilities': regime_result.probabilities,
                'cluster_persistence': regime_result.cluster_persistence,
                
                # Economic profiles
                'economic_profiles': [
                    {
                        'regime_id': profile.regime_id,
                        'name': profile.name,
                        'key_stats': profile.key_stats,
                        'confidence_intervals': profile.confidence_intervals,
                        'avg_duration': profile.avg_duration,
                        'transitions': profile.transitions,
                        'works_best_for': profile.works_best_for,
                        'risk_caveats': profile.risk_caveats,
                        'radar_plot_data': profile.radar_plot_data
                    }
                    for profile in regime_result.economic_profiles
                ],
                
                # Validation metrics
                'validation_metrics': regime_result.validation_metrics,
                
                # Metadata
                'metadata': regime_result.metadata,
                
                # Configuration
                'config': config,
                
                # Timestamps
                'created_at': datetime.now().isoformat(),
                'symbol': config['symbol'],
                'exchange': config['exchange'],
                'timeframe': config['timeframe'],
                
                # Enhanced artifact metadata
                'artifact_metadata': {
                    'artifact_type': 'hdbscan_regime_discovery',
                    'version': '1.0.0',
                    'created_by': 'HDBSCANRegimeDiscoveryStep',
                    'data_quality_score': self._calculate_data_quality_score(regime_result),
                    'compression_ratio': 0.0,  # Will be calculated during saving
                    'file_size_bytes': 0,  # Will be calculated during saving
                    'checksum': '',  # Will be calculated during saving
                }
            }
            
            # Add artifact manager metadata
            artifacts['artifact_manager_metadata'] = {
                'compression_enabled': artifact_config.enable_compression,
                'versioning_enabled': artifact_config.enable_versioning,
                'metadata_enabled': artifact_config.enable_metadata,
                'compression_level': artifact_config.compression_level
            }
            
            return artifacts
            
        except Exception as e:
            tprint(f"⚠️ Failed to create artifacts: {e}", "WARNING")
            return {}
    
    def _save_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save artifacts to disk with enhanced artifact management."""
        try:
            # Initialize artifact manager
            artifact_config = ArtifactConfig(
                enable_compression=True,
                enable_versioning=True,
                enable_metadata=True,
                compression_level=6
            )
            artifact_manager = ArtifactManager(artifact_config)
            
            # Create output directory
            output_dir = Path(config.get('data_dir', 'historical_data')) / 'hdbscan_regime_discovery' / config['symbol']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save regime labels as parquet with enhanced metadata
            if 'regime_labels' in artifacts:
                labels_df = pd.DataFrame({
                    'regime_label': artifacts['regime_labels'],
                    'regime_probability': artifacts['regime_probabilities'] if 'regime_probabilities' in artifacts else None,
                    'cluster_persistence': artifacts['cluster_persistence'] if 'cluster_persistence' in artifacts else None
                })
                
                labels_file = output_dir / f"hdbscan_regime_labels_{config['symbol']}_{config['timeframe']}_{timestamp}.parquet"
                labels_df.to_parquet(labels_file, compression='snappy')
                
                # Add metadata to the parquet file
                labels_df.attrs.update({
                    'symbol': config['symbol'],
                    'exchange': config['exchange'],
                    'timeframe': config['timeframe'],
                    'created_at': datetime.now().isoformat(),
                    'data_quality_score': artifacts.get('artifact_metadata', {}).get('data_quality_score', 0.0)
                })
                
                tprint(f"✅ Regime labels saved to {labels_file}", "SUCCESS")
            
            # Save full artifacts with enhanced compression
            artifacts_file = output_dir / f"hdbscan_regime_artifacts_{config['symbol']}_{config['timeframe']}_{timestamp}.pkl"
            
            # Use artifact manager for enhanced saving
            if hasattr(artifact_manager, 'save_artifact'):
                artifact_manager.save_artifact(
                    data=artifacts,
                    path=str(artifacts_file),
                    artifact_type='hdbscan_regime_discovery',
                    metadata=artifacts.get('artifact_metadata', {})
                )
            else:
                # Fallback to standard pickle saving
                save_pickle(artifacts, artifacts_file)
            
            tprint(f"✅ Full artifacts saved to {artifacts_file}", "SUCCESS")
            
            # Save economic profiles as JSON with enhanced formatting
            if 'economic_profiles' in artifacts:
                import json
                profiles_file = output_dir / f"hdbscan_economic_profiles_{config['symbol']}_{config['timeframe']}_{timestamp}.json"
                
                # Enhanced JSON formatting
                profiles_data = {
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'created_at': datetime.now().isoformat(),
                        'n_profiles': len(artifacts['economic_profiles'])
                    },
                    'economic_profiles': artifacts['economic_profiles']
                }
                
                with open(profiles_file, 'w') as f:
                    json.dump(profiles_data, f, indent=2, default=str)
                tprint(f"✅ Economic profiles saved to {profiles_file}", "SUCCESS")
            
            # Save configuration as YAML for better readability
            if 'config' in artifacts:
                import yaml
                config_file = output_dir / f"hdbscan_config_{config['symbol']}_{config['timeframe']}_{timestamp}.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(artifacts['config'], f, default_flow_style=False, indent=2)
                tprint(f"✅ Configuration saved to {config_file}", "SUCCESS")
            
            # Calculate and update artifact metadata
            self._update_artifact_metadata(artifacts, output_dir, timestamp)
            
        except Exception as e:
            tprint(f"⚠️ Failed to save artifacts: {e}", "WARNING")
    
    def _update_artifact_metadata(self, artifacts: Dict[str, Any], output_dir: Path, timestamp: str) -> None:
        """Update artifact metadata with file information."""
        try:
            # Calculate file sizes and compression ratios
            total_size = 0
            file_info = {}
            
            for file_path in output_dir.glob(f"*{timestamp}*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_info[file_path.name] = {
                        'size_bytes': file_size,
                        'size_mb': file_size / 1024 / 1024,
                        'created_at': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                    }
            
            # Update artifact metadata
            if 'artifact_metadata' in artifacts:
                artifacts['artifact_metadata']['file_size_bytes'] = total_size
                artifacts['artifact_metadata']['file_size_mb'] = total_size / 1024 / 1024
                artifacts['artifact_metadata']['file_info'] = file_info
                artifacts['artifact_metadata']['compression_ratio'] = self._calculate_compression_ratio(artifacts, total_size)
            
            tprint(f"📊 Artifact metadata updated: {total_size / 1024 / 1024:.2f}MB total size", "INFO")
            
        except Exception as e:
            tprint(f"⚠️ Failed to update artifact metadata: {e}", "WARNING")
    
    def _calculate_compression_ratio(self, artifacts: Dict[str, Any], compressed_size: int) -> float:
        """Calculate compression ratio for artifacts."""
        try:
            # Estimate uncompressed size
            import sys
            uncompressed_size = sys.getsizeof(artifacts)
            
            if uncompressed_size > 0:
                return 1.0 - (compressed_size / uncompressed_size)
            else:
                return 0.0
                
        except Exception as e:
            tprint(f"⚠️ Failed to calculate compression ratio: {e}", "WARNING")
            return 0.0
    
    def _calculate_metrics(self, regime_result: RegimeResult, start_time: datetime) -> Dict[str, Any]:
        """Calculate step execution metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'n_regimes': regime_result.validation_metrics.get('n_regimes', 0),
                'noise_ratio': regime_result.validation_metrics.get('noise_ratio', 0.0),
                'economic_separation': regime_result.validation_metrics.get('economic_separation', 0.0),
                'validation_passed': regime_result.validation_metrics.get('validation_passed', False),
                'reallocation_moves': regime_result.validation_metrics.get('reallocation_moves', 0),
                'merges_performed': regime_result.validation_metrics.get('merges_performed', 0),
                'stabilization_changes': regime_result.validation_metrics.get('stabilization_changes', 0),
                'success': regime_result.success
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    async def _perform_economic_validation(self, 
                                         regime_result: RegimeResult, 
                                         market_data: pd.DataFrame, 
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive economic validation."""
        try:
            tprint_info("💰 Starting economic validation...")
            
            # Extract cluster labels
            cluster_labels = regime_result.regime_labels
            
            # Engineer advanced features
            advanced_features, feature_names, feature_categories = self.advanced_feature_engineer.engineer_features(market_data)
            tprint_data_preview(advanced_features, "economic_validation_features", max_rows=5, level="DEBUG")
            
            # Perform data-driven optimization
            optimization_result = self.data_driven_optimizer.optimize_all_parameters(
                market_data=market_data,
                features=advanced_features,
                feature_names=feature_names,
                clustering_func=lambda x: cluster_labels  # Use existing labels
            )
            
            # Data preview of optimization results
            tprint_data_preview(optimization_result, "economic_optimization_result", level="DEBUG")
            
            # Extract economic validation results
            economic_validation_result = {
                'overall_economic_score': optimization_result.economic_validation_result.overall_economic_score if optimization_result.economic_validation_result else 0.0,
                'return_separation_score': optimization_result.economic_validation_result.return_separation_score if optimization_result.economic_validation_result else 0.0,
                'volatility_discrimination_score': optimization_result.economic_validation_result.volatility_discrimination_score if optimization_result.economic_validation_result else 0.0,
                'risk_discrimination_score': optimization_result.economic_validation_result.risk_discrimination_score if optimization_result.economic_validation_result else 0.0,
                'drawdown_discrimination_score': optimization_result.economic_validation_result.drawdown_discrimination_score if optimization_result.economic_validation_result else 0.0,
                'volume_discrimination_score': optimization_result.economic_validation_result.volume_discrimination_score if optimization_result.economic_validation_result else 0.0,
                'strategy_performance_score': optimization_result.economic_validation_result.strategy_performance_score if optimization_result.economic_validation_result else 0.0,
                'overall_persistence_score': optimization_result.regime_persistence_result.overall_persistence_score if optimization_result.regime_persistence_result else 0.0,
                'lifespan_score': optimization_result.regime_persistence_result.lifespan_score if optimization_result.regime_persistence_result else 0.0,
                'transition_score': optimization_result.regime_persistence_result.transition_score if optimization_result.regime_persistence_result else 0.0,
                'economic_coherence_score': optimization_result.regime_persistence_result.economic_coherence_score if optimization_result.regime_persistence_result else 0.0,
                'volatility_persistence_score': optimization_result.regime_persistence_result.volatility_persistence_score if optimization_result.regime_persistence_result else 0.0,
                'optimal_parameters': optimization_result.optimal_parameters,
                'optimization_summary': optimization_result.optimization_summary,
                'success': True
            }
            
            tprint_success(f"✅ Economic validation completed - Score: {economic_validation_result['overall_economic_score']:.3f}")
            
            return economic_validation_result
            
        except Exception as e:
            tprint_warning(f"⚠️ Economic validation failed: {e}")
            return {
                'overall_economic_score': 0.0,
                'return_separation_score': 0.0,
                'volatility_discrimination_score': 0.0,
                'risk_discrimination_score': 0.0,
                'drawdown_discrimination_score': 0.0,
                'volume_discrimination_score': 0.0,
                'strategy_performance_score': 0.0,
                'overall_persistence_score': 0.0,
                'lifespan_score': 0.0,
                'transition_score': 0.0,
                'economic_coherence_score': 0.0,
                'volatility_persistence_score': 0.0,
                'optimal_parameters': {},
                'optimization_summary': {},
                'success': False,
                'error': str(e)
            }
    
    def _create_outcome_report(self, regime_result: RegimeResult, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create comprehensive outcome report markdown."""
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            
            # Extract detailed information from regime result
            n_regimes = metrics.get('n_regimes', 0)
            noise_ratio = metrics.get('noise_ratio', 0.0)
            economic_separation = metrics.get('economic_separation', 0.0)
            validation_passed = metrics.get('validation_passed', False)
            
            # Get processing metadata
            processing_metadata = regime_result.metadata
            feature_metadata = processing_metadata.get('feature_extraction', {})
            preprocessing_metadata = processing_metadata.get('preprocessing', {})
            clustering_metadata = processing_metadata.get('clustering', {})
            economic_metadata = processing_metadata.get('economic_validation', {})
            
            report = f"""# HDBSCAN Regime Discovery Comprehensive Report

**Generated**: {timestamp.isoformat()}  
**Report ID**: `hdbscan_regime_discovery_{config['symbol']}_{config['timeframe']}_{timestamp_str}`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | {config['symbol']} |
| **Exchange** | {config['exchange']} |
| **Timeframe** | {config['timeframe']} |
| **Execution Mode** | {config.get('execution_mode', 'light')} |
| **Processing Time** | {metrics.get('processing_time_seconds', 0):.2f} seconds |
| **Success Status** | {'✅ SUCCESS' if regime_result.success else '❌ FAILED'} |
| **Regimes Discovered** | {n_regimes} |
| **Noise Ratio** | {noise_ratio:.1%} |
| **Economic Separation** | {economic_separation:.1%} |
| **Validation Status** | {'✅ PASSED' if validation_passed else '❌ FAILED'} |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: {n_regimes}
- **Noise Points**: {noise_ratio:.1%} of total samples
- **Average Regime Size**: {((1 - noise_ratio) * len(regime_result.labels) / max(n_regimes, 1)):.0f} samples per regime
- **Economic Separation Score**: {economic_separation:.3f} (0.0 = identical, 1.0 = completely separated)

### Regime Distribution
"""
            
            # Add regime distribution analysis
            if len(regime_result.labels) > 0:
                unique_labels, counts = np.unique(regime_result.labels, return_counts=True)
                total_samples = len(regime_result.labels)
                
                report += "| Regime ID | Sample Count | Percentage |\n"
                report += "|-----------|--------------|------------|\n"
                
                for label, count in zip(unique_labels, counts):
                    percentage = (count / total_samples) * 100
                    if label == -1:
                        report += f"| **Noise (-1)** | {count:,} | {percentage:.1f}% |\n"
                    else:
                        report += f"| **Regime {label}** | {count:,} | {percentage:.1f}% |\n"
            
            report += f"""

---

## 🏗️ Processing Pipeline Details

### 1. Feature Extraction
- **Feature Families Enabled**: {', '.join(self.config.enabled_feature_families)}
- **Total Features Generated**: {self.config.total_max_features}
- **PID Features**: {'✅ Enabled' if self.config.enable_pid_features else '❌ Disabled'}
- **Hybrid Features**: {'✅ Enabled' if self.config.enable_hybrid_features else '❌ Disabled'}
- **Hardware Optimization**: {'✅ Enabled' if self.config.enable_hardware_optimization else '❌ Disabled'}

#### Feature Family Breakdown
"""
            
            # Add feature family details
            for family in self.config.enabled_feature_families:
                report += f"- **{family.replace('_', ' ').title()}**: Features capturing {family.replace('_', ' ')} patterns\n"
            
            report += f"""

### 2. Preprocessing Pipeline
- **Transformer Type**: {self.config.transformer_type.title()}
- **Correlation Threshold**: {self.config.correlation_threshold}
- **Mutual Information Threshold**: {self.config.mutual_info_threshold}
- **HSIC Threshold**: {self.config.hsic_threshold}
- **Per-Asset Transformers**: {'✅ Enabled' if self.config.per_asset_transformers else '❌ Disabled'}

### 3. Dimensionality Reduction
- **Method**: {self.config.dim_reduction_mode.upper()}
- **PCA Variance Threshold**: {self.config.pca_variance_threshold:.1%}
- **UMAP Components**: {self.config.umap_n_components}
- **UMAP Neighbors**: {self.config.umap_n_neighbors}
- **UMAP Min Distance**: {self.config.umap_min_dist}

### 4. HDBSCAN Clustering
- **Min Cluster Size**: {self.config.min_cluster_size_pct:.1%} ({self.config.min_cluster_size_floor} minimum)
- **Cluster Selection Method**: {self.config.cluster_selection_method.upper()}
- **Selection Epsilon**: {self.config.cluster_selection_epsilon}
- **Prediction Data**: {'✅ Enabled' if self.config.prediction_data else '❌ Disabled'}

### 5. Post-Clustering Optimization
- **Change Budget**: {self.config.change_budget_pct:.1%} of samples
- **Max Optimization Rounds**: {self.config.max_optimization_rounds}
- **Condensed Tree Usage**: {'✅ Enabled' if self.config.use_condensed_tree else '❌ Disabled'}
- **Reallocation Moves**: {metrics.get('reallocation_moves', 0)}
- **Merges Performed**: {metrics.get('merges_performed', 0)}

### 6. Temporal Stabilization
- **Smoothing Window**: {self.config.smoothing_window} periods
- **Min Dwell Time**: {self.config.min_dwell_bars} bars
- **Cooldown Period**: {self.config.cooldown_bars_after_switch} bars
- **Stabilization Changes**: {metrics.get('stabilization_changes', 0)}

---

## 💰 Economic Analysis

### Economic Validation Results
- **Minimum Separation Required**: {self.config.min_economic_separation_pct:.1%}
- **Actual Separation Achieved**: {economic_separation:.1%}
- **Validation Status**: {'✅ PASSED' if validation_passed else '❌ FAILED'}

### Interpretable Economic Axes
"""
            
            for axis in self.config.interpretable_axes:
                report += f"- **{axis.replace('_', ' ').title()}**: {axis}\n"
            
            report += "\n### Economic Profiles by Regime\n"
            
            if regime_result.economic_profiles:
                for profile in regime_result.economic_profiles:
                    report += f"""
#### Regime {profile.regime_id}: {profile.name}

**Key Economic Statistics:**
"""
                    for stat_name, stat_value in profile.key_stats.items():
                        if isinstance(stat_value, (int, float)):
                            if isinstance(stat_value, float):
                                report += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value:.4f}\n"
                            else:
                                report += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value:,}\n"
                        else:
                            report += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value}\n"
                    
                    report += f"""
**Confidence Intervals:**
"""
                    for ci_name, ci_value in profile.confidence_intervals.items():
                        if isinstance(ci_value, tuple) and len(ci_value) == 2:
                            report += f"- **{ci_name.replace('_', ' ').title()}**: [{ci_value[0]:.4f}, {ci_value[1]:.4f}]\n"
                    
                    report += f"""
**Temporal Characteristics:**
- **Average Duration**: {profile.avg_duration:.1f} periods
- **Transitions From Others**: {profile.transitions.get('from_other', 0)}
- **Transitions To Others**: {profile.transitions.get('to_other', 0)}
- **Self-Transitions**: {profile.transitions.get('self_transitions', 0)}

**Trading Recommendations:**
- **Works Best For**: {', '.join(profile.works_best_for)}
- **Risk Caveats**: {', '.join(profile.risk_caveats)}

**Radar Plot Data:**
"""
                    for radar_name, radar_value in profile.radar_plot_data.items():
                        report += f"- **{radar_name.replace('_', ' ').title()}**: {radar_value:.3f}\n"
                    
                    report += "\n---\n"
            else:
                report += "\n*No economic profiles generated.*\n"
            
            report += f"""

---

## 🔧 Technical Configuration

### Hardware Optimization
- **M1 GPU Acceleration**: {'✅ Available' if hasattr(self.regime_discovery, 'm1_gpu_manager') and self.regime_discovery.m1_gpu_manager else '❌ Not Available'}
- **Matrix Operations**: {'✅ Available' if hasattr(self.regime_discovery, 'matrix_ops') and self.regime_discovery.matrix_ops else '❌ Not Available'}
- **Memory Optimization**: {'✅ Enabled' if self.config.enable_hardware_optimization else '❌ Disabled'}

### Determinism Settings
- **Random State**: {self.config.random_state}
- **NumPy Seed**: {self.config.numpy_seed}
- **BLAS Threading**: {'✅ Pinned' if self.config.pin_blas_threads else '❌ Not Pinned'}
- **Numba Threads**: {self.config.numba_threads}

### Data Quality Metrics
- **Effective Sample Size**: {processing_metadata.get('n_effective_samples', 'N/A')}
- **Window Size**: {processing_metadata.get('window_size', 'N/A')}
- **Overlap Percentage**: {processing_metadata.get('overlap_pct', 'N/A'):.1% if 'overlap_pct' in processing_metadata else 'N/A'}

---

## 📈 Performance Metrics

### Processing Times
"""
            
            # Add detailed processing times
            processing_times = {
                'Feature Extraction': processing_metadata.get('feature_extraction', {}).get('processing_time', 0),
                'Preprocessing': processing_metadata.get('preprocessing', {}).get('processing_time', 0),
                'Dimensionality Reduction': processing_metadata.get('dimensionality_reduction', {}).get('processing_time', 0),
                'Clustering': processing_metadata.get('clustering', {}).get('processing_time', 0),
                'Optimization': processing_metadata.get('reallocation', {}).get('processing_time', 0),
                'Economic Validation': processing_metadata.get('economic_validation', {}).get('processing_time', 0),
                'Temporal Stabilization': processing_metadata.get('temporal_stabilization', {}).get('processing_time', 0)
            }
            
            total_processing_time = sum(processing_times.values())
            
            for step_name, step_time in processing_times.items():
                percentage = (step_time / total_processing_time * 100) if total_processing_time > 0 else 0
                report += f"- **{step_name}**: {step_time:.2f}s ({percentage:.1f}%)\n"
            
            report += f"- **Total Processing Time**: {total_processing_time:.2f}s\n"
            
            report += f"""

### Memory Usage
- **Peak Memory Usage**: {processing_metadata.get('peak_memory_mb', 'N/A')} MB
- **Final Memory Usage**: {processing_metadata.get('final_memory_mb', 'N/A')} MB

---

## 📁 Generated Artifacts

### Data Files
- **Regime Labels**: `hdbscan_regime_labels_{config['symbol']}_{config['timeframe']}_{timestamp_str}.parquet`
- **Full Artifacts**: `hdbscan_regime_artifacts_{config['symbol']}_{config['timeframe']}_{timestamp_str}.pkl`
- **Economic Profiles**: `hdbscan_economic_profiles_{config['symbol']}_{config['timeframe']}_{timestamp_str}.json`

### Report Files
- **This Report**: `hdbscan_regime_discovery_report_{config['symbol']}_{config['timeframe']}_{timestamp_str}.md`

### Data Directory Structure
```
{config.get('data_dir', 'historical_data')}/hdbscan_regime_discovery/{config['symbol']}/
├── hdbscan_regime_labels_{config['symbol']}_{config['timeframe']}_{timestamp_str}.parquet
├── hdbscan_regime_artifacts_{config['symbol']}_{config['timeframe']}_{timestamp_str}.pkl
├── hdbscan_economic_profiles_{config['symbol']}_{config['timeframe']}_{timestamp_str}.json
└── hdbscan_regime_discovery_report_{config['symbol']}_{config['timeframe']}_{timestamp_str}.md
```

---

## 🎯 Key Insights

### Regime Characteristics
"""
            
            if regime_result.economic_profiles:
                # Add insights about regime characteristics
                regime_names = [profile.name for profile in regime_result.economic_profiles]
                regime_durations = [profile.avg_duration for profile in regime_result.economic_profiles]
                
                if regime_durations:
                    avg_duration = np.mean(regime_durations)
                    min_duration = np.min(regime_durations)
                    max_duration = np.max(regime_durations)
                    
                    report += f"""
- **Average Regime Duration**: {avg_duration:.1f} periods
- **Shortest Regime Duration**: {min_duration:.1f} periods
- **Longest Regime Duration**: {max_duration:.1f} periods
- **Regime Types Discovered**: {', '.join(set(regime_names))}
"""
            
            report += f"""

### Trading Implications
- **Number of Actionable Regimes**: {n_regimes} (excluding noise)
- **Regime Stability**: {'High' if noise_ratio < 0.1 else 'Medium' if noise_ratio < 0.3 else 'Low'}
- **Economic Separation**: {'Excellent' if economic_separation > 0.8 else 'Good' if economic_separation > 0.6 else 'Fair' if economic_separation > 0.4 else 'Poor'}

### Model Performance
- **Validation Status**: {'✅ PASSED - Ready for Production' if validation_passed else '❌ FAILED - Requires Review'}
- **Economic Significance**: {'High' if economic_separation > 0.7 else 'Medium' if economic_separation > 0.5 else 'Low'}

---

## 🔄 Next Steps

### Immediate Actions
1. **Review Economic Profiles**: Examine each regime's characteristics and trading recommendations
2. **Validate Regime Stability**: Monitor regime transitions and duration patterns
3. **Test Trading Strategies**: Implement regime-aware trading strategies based on economic profiles

### Model Integration
1. **Feature Engineering**: Use regime labels as features in downstream models
2. **Regime-Aware Training**: Train models with regime-specific parameters
3. **Risk Management**: Implement regime-based position sizing and risk controls

### Monitoring
1. **Regime Drift Detection**: Monitor for changes in regime characteristics over time
2. **Performance Tracking**: Track strategy performance across different regimes
3. **Model Retraining**: Schedule periodic regime discovery updates

---

## 📊 Appendix

### Full Configuration
```yaml
regime_discovery_config:
  enabled_feature_families: {self.config.enabled_feature_families}
  total_max_features: {self.config.total_max_features}
  transformer_type: {self.config.transformer_type}
  dim_reduction_mode: {self.config.dim_reduction_mode}
  min_cluster_size_pct: {self.config.min_cluster_size_pct}
  change_budget_pct: {self.config.change_budget_pct}
  min_economic_separation_pct: {self.config.min_economic_separation_pct}
  random_state: {self.config.random_state}
```

### Processing Metadata
```json
{processing_metadata}
```

---

*Report generated by HDBSCAN Regime Discovery Step v1.0.0*  
*Generated at: {timestamp.isoformat()}*  
*Processing completed in: {metrics.get('processing_time_seconds', 0):.2f} seconds*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create comprehensive outcome report: {e}", "WARNING")
            return f"# HDBSCAN Regime Discovery Outcome Report\n\nError creating comprehensive report: {str(e)}"

    def _save_outcome_report(self, report: str, config: Dict[str, Any]) -> str:
        """Save comprehensive outcome report to outcomes/ directory with timestamp."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"hdbscan_regime_discovery_report_{config['symbol']}_{config['timeframe']}_{timestamp}.md"
            
            # Ensure outcomes directory exists
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Create symbol-specific subdirectory
            symbol_dir = outcomes_dir / f"hdbscan_regime_discovery_{config['symbol']}"
            symbol_dir.mkdir(exist_ok=True)
            
            # Save report to outcomes directory
            report_path = symbol_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            tprint(f"📄 Comprehensive outcome report saved: {report_path}", "INFO")
            tprint(f"📁 Report location: outcomes/hdbscan_regime_discovery_{config['symbol']}/{report_filename}", "INFO")
            
            return str(report_path)
            
        except Exception as e:
            tprint(f"⚠️ Failed to save comprehensive outcome report: {e}", "WARNING")
            return ""
    
    def _calculate_data_quality_score(self, regime_result: RegimeResult) -> float:
        """Calculate data quality score for artifacts."""
        try:
            score = 0.0
            
            # Check if regime result is valid
            if not regime_result.success:
                return 0.0
            
            # Check data completeness
            if len(regime_result.labels) > 0:
                score += 0.3
            
            # Check economic profiles
            if len(regime_result.economic_profiles) > 0:
                score += 0.3
            
            # Check validation metrics
            validation_metrics = regime_result.validation_metrics
            if validation_metrics.get('validation_passed', False):
                score += 0.2
            
            # Check economic separation
            economic_separation = validation_metrics.get('economic_separation', 0.0)
            if economic_separation > 0.5:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate data quality score: {e}", "WARNING")
            return 0.0


# Register the step
def register_hdbscan_regime_discovery_step():
    """Register the HDBSCAN regime discovery step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)
    tprint("✅ HDBSCAN regime discovery step registered", "SUCCESS")


# Auto-register when module is imported
register_hdbscan_regime_discovery_step()
