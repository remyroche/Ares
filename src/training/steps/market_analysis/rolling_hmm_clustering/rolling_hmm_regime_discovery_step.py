"""
Rolling HMM Regime Discovery Step

This step performs regime discovery using Sticky HMM with:
- EWMA-based rolling feature engineering (8+24h combinations)
- Sticky priors for persistent regimes
- Diagonal covariance with regularization
- Hierarchical parameter optimization
- PCA dimensionality reduction
- Comprehensive quality assessment

Optimized for Mac M1 with VectorBT, hardware acceleration, and Numba JIT.

Inherits from BaseStep for standardized artifact management and execution.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_debug

# Import Rolling HMM components
from .feature_engineering import (
    RollingHMMFeatureEngineer,
    FeatureEngineeringConfig,
    DEFAULT_EWMA_CONFIGS
)
from .sticky_hmm_model import (
    StickyHMMModel,
    StickyHMMConfig
)
from .hpo_config import (
    RollingHMMOptimizer,
    HPOConfig,
    DEFAULT_HPO_CONFIG
)

# Import quality assessor
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics
)

# Import economic relevance analyzer
from src.training.steps.market_analysis.clusters.regime_economic_relevance_analyzer import (
    RegimeEconomicRelevanceAnalyzer
)

# Import execution mode configuration
from src.training.steps.market_analysis.shared_utils.execution_mode_lookback_config import (
    get_execution_mode_config
)

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager,
    WorkloadType,
    OptimizationLevel
)

logger = logging.getLogger(__name__)


class RollingHMMRegimeDiscoveryStep(BaseStep):
    """
    Rolling HMM Regime Discovery Step.

    Performs regime discovery using Sticky HMM with comprehensive feature engineering,
    hierarchical parameter optimization, and quality assessment. Optimized for Mac M1.

    Key features:
    - EWMA-based rolling features (8+16, 8+20, 8+24, 12+16, 12+20, 12+24)
    - Returns, volatility, trend, and volume features
    - PCA dimensionality reduction (3-5 components for 80-90% variance)
    - Sticky HMM with diagonal covariance and regularization
    - Hierarchical HPO (EWMA periods, model structure, regularization)
    - VectorBT and hardware optimization for M1
    - Comprehensive quality assessment with ClusterQualityAssessor

    Inherits from BaseStep to provide:
    - Standardized artifact management
    - Automatic context setting
    - Market data access by default
    - Consistent result saving
    """

    def __init__(self, step_name: str = "rolling_hmm_regime_discovery"):
        """
        Initialize the Rolling HMM regime discovery step.

        Args:
            step_name: Name for this step (used for artifact organization)
        """
        super().__init__(step_name, use_versioned_artifacts=True)  # Enable HDF5 storage for regime probabilities
        self.logger = system_logger.getChild('RollingHMMRegimeDiscovery')

        # Quality assessor will be created lazily when first accessed
        self._quality_assessor = None

        # Hardware manager
        self.hardware_manager = None

        tprint(f"Ă˘ÂÂ Initialized {step_name} step", "SUCCESS")

    @property
    def quality_assessor(self) -> ClusterQualityAssessor:
        """Lazy property for quality assessor."""
        tprint_debug("ÄÂÂÂ Accessing quality assessor instance")
        if self._quality_assessor is None:
            tprint_info("  Ă˘ÂÂ Initializing ClusterQualityAssessor")
            self._quality_assessor = ClusterQualityAssessor(
                artifact_manager=self.artifact_manager,
                enable_hardware_optimization=True,
                enable_vectorization=True
            )
        return self._quality_assessor

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Rolling HMM regime discovery with optional HPO.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'BTCUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Optional timeframe override (defaults to regime_timeframe)
                - regime_timeframe: Timeframe for regime detection (default: '1h')
                - execution_mode: 'full', 'light', or 'blank'
                - rolling_hmm_params: Optional parameters override
                - enable_auto_tuning: Whether to run HPO (default: True)
                - hpo_config: Optional HPO configuration override
                - feature_config: Optional feature engineering configuration
        
        Returns:
            Dict containing:
                - 'success': bool indicating if step completed successfully
                - 'artifacts': dict of created artifacts
                - 'metrics': dict of performance metrics
                - 'error': error message if step failed (optional)
                - 'execution_time': float seconds taken to execute
                - 'hpo_results': dict of HPO results if enabled (optional)
        """
        start_time = time.time()
        
        # Validate configuration
        try:
            self._validate_config(config)
        except Exception as e:
            execution_time = time.time() - start_time
            return self._handle_execution_error(e, config, execution_time)
        
        # Extract configuration
        symbol = config.get('symbol', 'BTCUSDT')
        exchange = config.get('exchange', 'binance')
        
        # Use regime_timeframe for regime detection
        regime_timeframe = config.get('regime_timeframe', '1h')
        timeframe = config.get('timeframe', regime_timeframe)
        
        # Override to regime_timeframe
        if timeframe != regime_timeframe:
            tprint(
                f"Ă˘ÂÂ° Using regime_timeframe={regime_timeframe} for Rolling HMM "
                f"(overriding timeframe={timeframe})",
                "INFO"
            )
            timeframe = regime_timeframe
        
        tprint(
            f"ÄÂÂÂ Starting Rolling HMM Regime Discovery for {symbol} on {exchange} "
            f"(timeframe: {timeframe})",
            "INFO"
        )
        tprint_debug("ÄÂÂ§Â  Enhanced Features: EWMA rolling (8+16 to 12+24), Returns/Vol/Trend/Volume, PCA (3-5), Sticky HMM, VectorBT+M1")
        
        try:
            # Initialize hardware optimization
            self._initialize_hardware_optimization()
            
            # Set context for versioned artifacts (HDF5 storage)
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction="long",
                model="regime"
            )
            
            # Load market data
            tprint("ÄÂÂÄ˝ Loading market data...", "INFO")

            market_data = self._load_market_data(symbol, exchange, timeframe, config)

            if market_data is None or market_data.empty:
                tprint_error(f"Ă˘ÂÂ No market data available for {symbol} on {timeframe}")
                raise ValueError(f"No market data available for {symbol} on {timeframe}")

            tprint(f"ÄÂÂÂ DEBUG: [STEP 1] market_data after _load_market_data: {market_data.shape}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 1] Index range: {market_data.index.min()} to {market_data.index.max()}", "INFO")
            tprint(f"Ă˘ÂÂ Loaded {len(market_data)} samples of market data", "SUCCESS")

            try:
                if self.hardware_manager is not None and hasattr(self.hardware_manager, 'memory_optimizer'):
                    optimizer = self.hardware_manager.memory_optimizer
                    if hasattr(optimizer, 'optimize_dataframe_memory'):
                        market_data = optimizer.optimize_dataframe_memory(market_data)
                        tprint_debug("ÄÂÂ§Â  Applied M1 memory optimization to market_data")
            except Exception as e:
                tprint_debug(f"Ă˘ÂÂ ÄÂ¸Â M1 memory optimization skipped due to error: {e}")

            # Check execution mode and if HPO is enabled
            execution_mode = config.get('execution_mode', 'full')
            enable_auto_tuning = config.get('enable_auto_tuning', True)  # Default to True
            hpo_results: Optional[Dict[str, Any]] = None
            best_params: Optional[Dict[str, Any]] = None

            tprint_info(f"ÄÂÂÂ§ Configuration check: execution_mode={execution_mode}, enable_auto_tuning={enable_auto_tuning}")

            # Check if manual params are provided
            has_manual_params = 'rolling_hmm_params' in config and config['rolling_hmm_params']
            if has_manual_params:
                tprint_info(f"ÄÂÂÂ Manual params found: {list(config['rolling_hmm_params'].keys())}")

            # Apply execution mode data limits (blank=20d, light=20d, full=all)
            market_data = self._apply_execution_mode_filter(market_data, execution_mode, timeframe)
            tprint(f"ÄÂÂÂ DEBUG: [STEP 2] market_data after _apply_execution_mode_filter: {market_data.shape}", "INFO")
            tprint(f"   Ă˘ÂÂ After execution mode filter ({execution_mode}): {len(market_data)} samples")
            
            # Initialize feature engineer
            feature_config = self._get_feature_config(
                config,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                execution_mode=execution_mode
            )
            feature_engineer = RollingHMMFeatureEngineer(feature_config)
            
            # Pre-compute ALL features for ALL EWMA windows ONCE (cached for HPO)
            if enable_auto_tuning:
                tprint("")
                tprint("ÄÂÂÂ Pre-computing features for HPO efficiency")
                all_cached_features = feature_engineer.precompute_all_features(market_data)
                tprint(f"Ă˘ÂÂ Cached features for {len(all_cached_features)} EWMA configurations")
                tprint("")
            
            # Only skip if user provided manual params AND explicitly disabled auto-tuning
            if 'rolling_hmm_params' in config and config['rolling_hmm_params']:
                if config.get('enable_auto_tuning') is False:
                    enable_auto_tuning = False
                    tprint_info("Ă˘ÂĹĄÄÂ¸Â  Manual params provided with enable_auto_tuning=False, skipping HPO")
                else:
                    tprint_info("Ă˘ÂĹĄÄÂ¸Â  Manual params provided but enable_auto_tuning=True (default), running HPO")
            
            # Show execution mode info
            if enable_auto_tuning and execution_mode in ['light', 'blank']:
                tprint_info(f"Ă˘ÂĹĄÄÂ¸Â  HPO enabled in '{execution_mode}' mode (will use reduced trials for speed)")
            
            if enable_auto_tuning:
                tprint("", "INFO")
                tprint("ÄÂÂĹť HPO Enabled - Finding Optimal Hyperparameters", "INFO")
                tprint("=" * 80, "INFO")

                # Run HPO synchronously
                hpo_results, best_params = await self._run_hpo(
                    market_data, feature_engineer, symbol, exchange, timeframe, config
                )

                # Update config with best parameters
                if hpo_results and best_params:
                    tprint("", "INFO")
                    tprint("Ă˘ÂÂ HPO Complete - Using Optimal Parameters", "SUCCESS")
                    self._log_best_params(best_params)
                    if 'rolling_hmm_params' not in config:
                        config['rolling_hmm_params'] = {}
                    config['rolling_hmm_params'].update(best_params)
                else:
                    tprint_warning("Ă˘ÂÂ ÄÂ¸Â  HPO did not complete, using default parameters")
                
                tprint("=" * 80, "INFO")
                tprint("", "INFO")
            
            # Run clustering
            tprint("ÄÂÂÂ Running Rolling HMM clustering...", "INFO")
            result = await self._run_clustering(
                market_data,
                feature_engineer,
                symbol,
                exchange,
                timeframe,
                config,
                hpo_results=hpo_results,
                best_params=best_params,
            )

            # Save results
            labels_df, probs_df = await self._save_results(result, symbol, exchange, timeframe, config)

            # Add HPO results to result dict for report generation
            if hpo_results:
                result['hpo_results'] = hpo_results

            # Generate reports
            await self._generate_reports(result, market_data, symbol, exchange, timeframe, config)
            
            # Cleanup quality assessor to prevent resource leaks
            if self._quality_assessor is not None:
                try:
                    del self._quality_assessor
                    self._quality_assessor = None
                except Exception:
                    pass  # Ignore cleanup errors
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            tprint("", "SUCCESS")
            tprint(f"Ă˘ÂÂ Rolling HMM Regime Discovery completed in {execution_time:.2f}s", "SUCCESS")
            tprint(f"   - Identified {result.get('n_regimes', 0)} regimes", "SUCCESS")
            tprint(f"   - Quality score: {result.get('quality_metrics', {}).get('quality_score', 0):.4f}", "SUCCESS")
            
            # Return standardized result
            return_dict = {
                'success': True,
                'artifacts': {
                    'labels': labels_df,
                    'probabilities': probs_df,
                    'transition_matrix': result.get('transition_matrix'),
                    'feature_importance': result.get('feature_importance')
                },
                'metrics': result.get('quality_metrics', {}),
                'execution_time': execution_time,
                'n_regimes': result.get('n_regimes', 0)
            }
            
            if hpo_results:
                return_dict['hpo_results'] = hpo_results
            
            return return_dict
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._handle_execution_error(e, config, execution_time)

    def _load_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """Load market data from config, historical storage (all modes), or artifacts."""
        if 'market_data' in config and config['market_data'] is not None:
            external_data = config['market_data']

            # CRITICAL: Validate data size to prevent truncation
            execution_mode = config.get('execution_mode', 'full')

            # Calculate expected minimum samples for this mode
            samples_per_day_map = {'1m': 1440, '3m': 480, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1}
            samples_per_day = samples_per_day_map.get(timeframe, 24)

            # Get lookback days based on execution mode
            mode_config = get_execution_mode_config()
            lookback_config = mode_config.get_configuration(execution_mode)
            lookback_days = lookback_config.optimization_window_days
            
            # Expected samples for the configured lookback period
            expected_min_samples = lookback_days * samples_per_day
            actual_samples = len(external_data)

            tprint(f"ÄÂÂÂ [REGIME_DISCOVERY] Validating config market_data:", "INFO")
            tprint(f"   Ă˘ÂÂ Execution mode: {execution_mode}", "INFO")
            tprint(f"   Ă˘ÂÂ Timeframe: {timeframe}", "INFO")
            tprint(f"   Ă˘ÂÂ Lookback days for mode: {lookback_days}", "INFO")
            tprint(f"   Ă˘ÂÂ Expected min samples ({lookback_days} days): {expected_min_samples:,}", "INFO")
            tprint(f"   Ă˘ÂÂ Actual samples in config: {actual_samples:,}", "INFO")

            if actual_samples < expected_min_samples * 0.3:  # Allow 30% tolerance
                tprint(
                    f"Ă˘ÂÂ [REGIME_DISCOVERY] CRITICAL: config['market_data'] has only {actual_samples:,} samples!\n"
                    f"   Expected at least {int(expected_min_samples * 0.3):,} samples (30% of {lookback_days} days)\n"
                    f"   This data appears TRUNCATED - bypassing config and loading from historical storage instead!",
                    "ERROR"
                )
                # Fall through to normal loading to get full dataset
            else:
                tprint(f"Ă˘ÂÂ Using market data from config ({len(external_data)} samples)", "SUCCESS")
                return external_data

        # Load using KlinesParquetManager (same method as regime_models_training)
        tprint(f"ÄÂÂÄ˝ [REGIME_DISCOVERY] Loading fresh data for {symbol} from historical storage", "INFO")

        try:
            from src.utils.kline_parquet import KlinesParquetManager, StorageConfig

            klines_manager = KlinesParquetManager(config=StorageConfig(base_dir='historical_data'))

            # Get lookback days based on execution mode
            execution_mode = config.get('execution_mode', 'full')
            mode_config = get_execution_mode_config()
            lookback_config = mode_config.get_configuration(execution_mode)
            lookback_days = lookback_config.optimization_window_days
            
            # Use the smart last_n_days parameter to load only the period we need
            # This automatically finds the latest available data and loads the configured lookback period
            tprint(f"ÄÂÂÄ˝ [REGIME_DISCOVERY] Loading last {lookback_days} days from latest available data (mode: {execution_mode})", "INFO")
            
            fresh_data = klines_manager.load_klines(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,
                last_n_days=lookback_days  # Dynamic based on execution mode
            )
            
            if fresh_data is None or len(fresh_data) == 0:
                tprint(f"Ă˘ÂÂ [REGIME_DISCOVERY] No data found in historical storage", "ERROR")
                raise ValueError(f"No historical data found for {symbol} {exchange} {timeframe}")

            tprint(f"ÄÂÂÂ DEBUG: KlinesParquetManager returned data: {fresh_data.shape if fresh_data is not None else 'None'}", "INFO")

            if fresh_data is not None and len(fresh_data) > 0:
                tprint(f"Ă˘ÂÂ [REGIME_DISCOVERY] Loaded {len(fresh_data):,} rows from historical storage (last {lookback_days} days)", "SUCCESS")

                # Check for and remove duplicate index labels
                if fresh_data.index.duplicated().any():
                    n_duplicates = fresh_data.index.duplicated().sum()
                    tprint(f"Ă˘ÂÂ ÄÂ¸Â [REGIME_DISCOVERY] Found {n_duplicates} duplicate timestamps, removing duplicates", "WARNING")
                    fresh_data = fresh_data[~fresh_data.index.duplicated(keep='first')]
                    tprint(f"Ă˘ÂÂ [REGIME_DISCOVERY] After deduplication: {len(fresh_data):,} rows", "SUCCESS")

                # Validate sample count
                expected_samples_per_day = 24 if timeframe == '1h' else (24 * 4 if timeframe == '15m' else 24)
                expected_samples = lookback_days * expected_samples_per_day
                actual_samples = len(fresh_data)

                tprint(f"ÄÂÂÂ [REGIME_DISCOVERY] Data validation: Expected ~{expected_samples:,} samples for {lookback_days} days of {timeframe} data", "INFO")
                tprint(f"ÄÂÂÂ [REGIME_DISCOVERY] Data validation: Actual samples: {actual_samples:,}", "INFO")

                if actual_samples < expected_samples * 0.5:
                    tprint(f"Ă˘ÂÂ ÄÂ¸Â [REGIME_DISCOVERY] WARNING: Only {actual_samples:,} samples available (expected ~{expected_samples:,})", "WARNING")

                tprint(f"ÄÂÂÂ [REGIME_DISCOVERY] Date range: {fresh_data.index.min()} to {fresh_data.index.max()}", "INFO")
                return fresh_data

        except Exception as e:
            tprint(f"Ă˘ÂÂ [REGIME_DISCOVERY] Failed to load from KlinesParquetManager: {e}", "ERROR")
            import traceback
            tprint(f"Ă˘ÂÂ [REGIME_DISCOVERY] Traceback: {traceback.format_exc()}", "ERROR")
            self.logger.error(f"Failed to load from KlinesParquetManager: {e}", exc_info=True)

        # Fall back to artifact sources
        artifact_sources = [
            ('klines_downloading_processing', 'klines_data'),
            ('data_collection', 'market_data'),
            ('data_reading', 'ohlcv_data'),
        ]

        original_context = self._current_context.copy()

        try:
            for step_name, artifact_name in artifact_sources:
                try:
                    self.artifact_manager.set_context(
                        step_name=step_name,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                    )

                    market_data = self._get_artifact(
                        artifact_name=artifact_name,
                        artifact_type='data',
                    )
                    
                    tprint(f"ÄÂÂÂ DEBUG: Artifact {step_name}/{artifact_name} returned: {market_data.shape if market_data is not None and hasattr(market_data, 'shape') else 'None/Invalid'}", "INFO")

                    if market_data is not None and not market_data.empty:
                        # Log data size but don't reject it - let the pipeline handle insufficient data
                        actual_samples = len(market_data)
                        tprint(f"Ă˘ÂÂ Loaded market data from {step_name}/{artifact_name} ({actual_samples:,} samples)", "SUCCESS")
                        
                        # Warn if data seems insufficient but still use it
                        samples_per_day_map = {'1m': 1440, '3m': 480, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1}
                        samples_per_day = samples_per_day_map.get(timeframe, 24)
                        expected_min_samples = 180 * samples_per_day
                        
                        if actual_samples < expected_min_samples * 0.5:
                            tprint(
                                f"Ă˘ÂÂ ÄÂ¸Â [REGIME_DISCOVERY] WARNING: Only {actual_samples:,} samples available\n"
                                f"   (expected ~{expected_min_samples:,} for 180 days of {timeframe} data)\n"
                                f"   Proceeding anyway - results may be suboptimal",
                                "WARNING"
                            )
                        
                        return market_data
                except Exception as load_error:
                    self.logger.debug(
                        f"Could not load market data from {step_name}/{artifact_name}: {load_error}"
                    )
        finally:
            self.artifact_manager.set_context(**original_context)

        tprint(
            f"Ă˘ÂÂ ÄÂ¸Â Could not load market data for {symbol} on {timeframe} from artifacts",
            "WARNING",
        )

        raise ValueError(
            "Market data not available via artifact manager. "
            "Run the data collection or klines processing steps before rolling HMM discovery."
        )

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization for M1."""
        tprint_info("Ă˘ÂÄ Initializing hardware optimization for M1")

        self.hardware_manager = get_unified_hardware_manager()
        self.hardware_manager.optimize_for_workload(
            WorkloadType.ML_TRAINING,
            OptimizationLevel.BALANCED
        )

    def _get_feature_config(
        self,
        config: Dict[str, Any],
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
        execution_mode: str
    ) -> FeatureEngineeringConfig:
        """Get feature engineering configuration."""
        tprint_debug("Fetching feature engineering configuration")
        feature_config = config.get('feature_config', {})

        cache_dir = Path(feature_config.get('cache_dir', 'artifacts/cache/rolling_hmm'))
        cache_namespace = feature_config.get('cache_namespace') or \
            f"{symbol}_{exchange}_{timeframe}_{execution_mode}"

        return FeatureEngineeringConfig(
            ewma_configs=DEFAULT_EWMA_CONFIGS,
            use_log_returns=feature_config.get('use_log_returns', True),
            use_volatility_features=feature_config.get('use_volatility_features', True),
            use_trend_features=feature_config.get('use_trend_features', True),
            use_volume_features=feature_config.get('use_volume_features', True),
            pca_components=feature_config.get('pca_components', 4),
            normalize_method=feature_config.get('normalize_method', 'robust'),
            rolling_normalize_window=feature_config.get('rolling_normalize_window', 100),
            enable_vectorbt_optimization=feature_config.get('enable_vectorbt_optimization', True),
            enable_hardware_optimization=feature_config.get('enable_hardware_optimization', True),
            enable_numba_jit=feature_config.get('enable_numba_jit', True),
            cache_dir=cache_dir,
            enable_persistent_cache=feature_config.get('enable_persistent_cache', True),
            cache_version=feature_config.get('cache_version', 'v1'),
            cache_namespace=cache_namespace
        )

    def _get_hpo_config(self, config: Dict[str, Any]) -> HPOConfig:
        """Get HPO configuration."""
        tprint_debug("Fetching HPO configuration with execution mode adjustments")
        hpo_config = config.get('hpo_config', {})
        execution_mode = config.get('execution_mode', 'full')

        resource_constrained = False
        try:
            hw = getattr(self, 'hardware_manager', None)
            if hw is not None:
                cpu_usage = hw.get_cpu_usage()
                memory_pressure = hw.get_memory_pressure()
                if cpu_usage is not None and memory_pressure is not None:
                    if cpu_usage > 85.0 or memory_pressure > 0.80:
                        resource_constrained = True
        except Exception:
            resource_constrained = False

        # Adjust HPO config based on execution mode
        if execution_mode == 'light':
            hpo_config['final_refinement_trials'] = hpo_config.get('final_refinement_trials', 20)
            hpo_config['cv_folds'] = hpo_config.get('cv_folds', 3)
        elif execution_mode == 'blank':
            hpo_config['final_refinement_trials'] = hpo_config.get('final_refinement_trials', 5)
            hpo_config['cv_folds'] = hpo_config.get('cv_folds', 2)
        else:  # full
            if resource_constrained:
                default_trials = 25
            else:
                default_trials = 50
            hpo_config['final_refinement_trials'] = hpo_config.get('final_refinement_trials', default_trials)
            hpo_config['cv_folds'] = hpo_config.get('cv_folds', 5)

        default_n_rounds = DEFAULT_HPO_CONFIG.n_rounds
        if execution_mode == 'full' and resource_constrained and 'n_rounds' not in hpo_config:
            default_n_rounds = 1

        return HPOConfig(
            stages=hpo_config.get('stages', DEFAULT_HPO_CONFIG.stages),
            n_rounds=hpo_config.get('n_rounds', default_n_rounds),
            enable_final_refinement=hpo_config.get('enable_final_refinement', True),
            final_refinement_trials=hpo_config['final_refinement_trials'],
            cv_folds=hpo_config['cv_folds'],
            weight_between_within_cv=hpo_config.get('weight_between_within_cv', 0.40),
            weight_temporal=hpo_config.get('weight_temporal', 0.10),
            weight_economic=hpo_config.get('weight_economic', 0.50),
            direction=hpo_config.get('direction', 'maximize'),
            use_custom_balanced_score=hpo_config.get('use_custom_balanced_score', True),
            verbose=hpo_config.get('verbose', True)
        )

    async def _run_hpo(
        self,
        market_data: pd.DataFrame,
        feature_engineer: RollingHMMFeatureEngineer,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Run hierarchical parameter optimization."""
        try:
            # Get HPO config
            hpo_config = self._get_hpo_config(config)

            hpo_market_data = market_data
            try:
                rolling_params = config.get('rolling_hmm_params', {}) or {}
                max_samples = rolling_params.get('max_samples_for_hpo')
                sample_fraction = rolling_params.get('hpo_sample_fraction')
                total_samples = len(market_data)
                cap = total_samples
                if isinstance(max_samples, int) and max_samples > 0:
                    cap = min(cap, max_samples)
                if isinstance(sample_fraction, (float, int)) and 0 < float(sample_fraction) < 1.0:
                    cap = min(cap, int(total_samples * float(sample_fraction)))
                if cap < total_samples and cap > 0:
                    hpo_market_data = market_data.tail(cap)
                    tprint_info(f"ÄÂÂÂ§ HPO using subsample of {len(hpo_market_data)} rows out of {total_samples} (rolling_hmm_params cap)")
            except Exception as e:
                tprint_debug(f"Ă˘ÂÂ ÄÂ¸Â HPO subsampling disabled due to error: {e}")

            # Create optimizer
            optimizer = RollingHMMOptimizer(hpo_config)

            # Run optimization
            result = optimizer.optimize(
                hpo_market_data,
                feature_engineer,
                StickyHMMModel,
                self.quality_assessor
            )

            if result and result['best_params']:
                best_params = result['best_params']

                # Save HPO results as artifact
                self._save_artifact(
                    data=result,
                    artifact_name='rolling_hmm_hpo_results',
                    artifact_type='metadata',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )

                return result, best_params
            else:
                tprint_warning("Ă˘ÂÂ ÄÂ¸Â  HPO returned no results")
                return None, None

        except Exception as e:
            tprint_error(f"Ă˘ÂÂ HPO failed: {e}")
            self.logger.error(f"HPO failed: {e}", exc_info=True)
            return None, None

    async def _run_clustering(
        self,
        market_data: pd.DataFrame,
        feature_engineer: RollingHMMFeatureEngineer,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any],
        hpo_results: Optional[Dict[str, Any]] = None,
        best_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run Rolling HMM clustering."""
        try:
            # Get parameters
            params = config.get('rolling_hmm_params', {})

            # Extract EWMA config
            ewma_config_idx = params.get('ewma_config_idx', 0)
            ewma_config = DEFAULT_EWMA_CONFIGS[int(ewma_config_idx)]

            tprint_info(f"  Ă˘ÂÂ Using EWMA config: {ewma_config.name}")

            # Generate features
            tprint(f"ÄÂÂÂ DEBUG: [STEP 3] market_data before feature generation: {market_data.shape}", "INFO")
            features = feature_engineer.generate_features(market_data, ewma_config)
            tprint(f"ÄÂÂÂ DEBUG: [STEP 4] features after generation: {features.shape}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 4] features index range: {features.index.min()} to {features.index.max()}", "INFO")

            if len(features) < 50:
                raise ValueError(f"Insufficient data after feature engineering: {len(features)} samples")

            # Extract economic features instead of PCA for better economic interpretability
            features_economic = feature_engineer.extract_economic_features(
                features,
                market_data,
                ewma_config
            )
            tprint(f"ÄÂÂÂ DEBUG: [STEP 5] features_economic after extraction: {features_economic.shape}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 5] features_economic index range: {features_economic.index.min()} to {features_economic.index.max()}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 5] Economic features: {list(features_economic.columns)}", "INFO")

            # Apply PCA on compact economic feature set for HMM emissions
            features_pca, pca_model, pca_explained = feature_engineer.apply_pca(
                features_economic,
                use_cache=True,
                cache_key=f"economic_{ewma_config.name}",
            )
            tprint_info(f"  Ă˘ÂÂ PCA explained variance (economic features): {pca_explained:.2%}")

            # Create HMM config
            n_components = params.get('n_components', 5)
            min_covar = params.get('min_covar', 1e-3)
            kappa = params.get('kappa', 10.0)

            # OPTIMIZED: Reduced n_iter from 200 to 100 for faster convergence
            # Using diag covariance (already set) for speed
            # Relaxed tol from 1e-4 to 1e-3 for faster convergence
            hmm_config = StickyHMMConfig(
                n_components=int(n_components),
                min_covar=float(min_covar),
                kappa=float(kappa),
                n_iter=params.get('n_iter', 100),  # Reduced from 200 for faster training
                tol=params.get('tol', 1e-3),  # Relaxed from 1e-4
                covariance_type='diag',  # 5-10x faster than 'full'
                kmeans_init=True,
                use_sticky_priors=True,
                post_fit_regularization=True,
                random_state=params.get('random_state', 42)
            )

            tprint_info(f"  Ă˘ÂÂ HMM config: n_components={n_components}, kappa={kappa}, min_covar={min_covar}")

            # Fit HMM model using PCA-transformed economic features
            hmm_model = StickyHMMModel(hmm_config)
            hmm_model.fit(
                features_pca.values,
                ewma_config_name=ewma_config.name,
                pca_components=features_pca.shape[1],
            )

            # Predict regime labels
            regime_labels = hmm_model.predict(features_pca.values)
            regime_probs = hmm_model.predict_proba(features_pca.values)
            tprint(f"ÄÂÂÂ DEBUG: [STEP 6] regime_labels after HMM predict: shape={regime_labels.shape}, unique={np.unique(regime_labels)}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 6] regime_probs after HMM predict: {regime_probs.shape}", "INFO")

            # Get model summary
            model_summary = hmm_model.get_model_summary()

            # Calculate forward returns for quality assessment
            forward_returns = market_data['close'].pct_change().shift(-1)
            forward_returns = forward_returns.loc[features_economic.index]

            # Assess quality
            tprint_info("  Ă˘ÂÂ Assessing regime quality")

            # Diagnostic: Check regime transitions
            unique_regimes = np.unique(regime_labels)
            regime_transitions = np.sum(regime_labels[1:] != regime_labels[:-1])
            tprint_info(f"  Ă˘ÂÂ Found {len(unique_regimes)} unique regimes, {regime_transitions} transitions in {len(regime_labels)} samples")

            metrics = self.quality_assessor.assess_hmm_regime_quality(
                regime_labels=regime_labels,
                feature_data=features_economic,
                transition_matrix=model_summary['transition_matrix'],
                hmm_model=None,
                forward_returns=forward_returns,
                timestamps=features_economic.index,
                timeframe=timeframe,
                min_regime_size=10,
                run_validators=True,
                temporal_sensitivity_mode="standard"
            )

            # Create result
            result = {
                'regime_labels': regime_labels,
                'regime_probs': regime_probs,
                'features': features_economic,
                'features_pca': features_pca,
                'original_features': features,
                'economic_feature_names': list(features_economic.columns),  # Store economic feature names
                'hmm_model': hmm_model,
                'transition_matrix': model_summary['transition_matrix'],
                'stationary_distribution': model_summary['stationary_distribution'],
                'expected_durations': model_summary['expected_durations'],
                'model_summary': model_summary,
                'quality_metrics': metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics,
                'n_regimes': n_components,
                'timestamps': features_economic.index,
                'pca_explained_variance': pca_explained,
                'hpo_results': hpo_results,  # Include HPO results in the output
                'best_params': best_params   # Include best params for reference
            }

            return result

        except Exception as e:
            tprint_error(f"Ă˘ÂÂ Clustering failed: {e}")
            self.logger.error(f"Clustering failed: {e}", exc_info=True)
            raise

    async def _save_results(
        self,
        result: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Save clustering results as artifacts."""
        try:
            tprint(f"ÄÂÂÂ DEBUG: [STEP 7] _save_results called", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 7] result['timestamps'] length: {len(result['timestamps'])}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 7] result['regime_labels'] length: {len(result['regime_labels'])}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 7] result['regime_probs'] shape: {result['regime_probs'].shape}", "INFO")

            # Create labels DataFrame (raw Viterbi regimes)
            labels_df = pd.DataFrame({
                'timestamp': result['timestamps'],
                'regime_label': result['regime_labels'],
            })
            labels_df.set_index('timestamp', inplace=True)

            # Build ML-friendly labels: posterior-filtered + min run-length smoothing
            try:
                probs_array = np.asarray(result['regime_probs'], dtype=float)
                viterbi_labels = np.asarray(result['regime_labels'], dtype=int)
                top_prob = probs_array.max(axis=1)
                ml_labels = viterbi_labels.copy()

                # 1) Posterior confidence filter
                low_conf_mask = top_prob < 0.55
                ml_labels[low_conf_mask] = -1

                # 2) Enforce minimum run length for non-noise regimes
                min_run = 2
                n = ml_labels.shape[0]
                start = 0
                while start < n:
                    label = ml_labels[start]
                    end = start + 1
                    while end < n and ml_labels[end] == label:
                        end += 1
                    run_len = end - start
                    if label != -1 and run_len < min_run:
                        ml_labels[start:end] = -1
                    start = end

                # 3) Apply hysteresis on regime switches using per-regime probabilities
                try:
                    hysteresis_factor = float(config.get("regime_hysteresis_factor", 1.05))
                except Exception:
                    hysteresis_factor = 1.05

                if hysteresis_factor > 1.0:
                    n_samples, n_regimes = probs_array.shape
                    for i in range(1, n_samples):
                        prev_label = ml_labels[i - 1]
                        curr_label = ml_labels[i]

                        # Only apply hysteresis between valid non-noise regimes
                        if prev_label < 0 or curr_label < 0 or prev_label == curr_label:
                            continue
                        if prev_label >= n_regimes or curr_label >= n_regimes:
                            continue

                        prev_prob = probs_array[i, prev_label]
                        curr_prob = probs_array[i, curr_label]
                        if not np.isfinite(prev_prob) or not np.isfinite(curr_prob):
                            continue

                        # Require extra confidence to switch regimes
                        if curr_prob < hysteresis_factor * prev_prob:
                            ml_labels[i] = prev_label

                labels_df['regime_label_ml'] = ml_labels
            except Exception:
                # Fall back gracefully if anything goes wrong
                labels_df['regime_label_ml'] = labels_df['regime_label']
            tprint(f"ÄÂÂÂ DEBUG: [STEP 8] labels_df shape after creation: {labels_df.shape}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 8] labels_df index range: {labels_df.index.min()} to {labels_df.index.max()}", "INFO")

            # Create probabilities DataFrame
            probs_columns = [f'regime_{i}_prob' for i in range(result['n_regimes'])]
            
            # Ensure timestamps is a proper DatetimeIndex
            timestamps = result['timestamps']
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.to_datetime(timestamps)
            
            # Debug: Check timestamps before creating DataFrame
            tprint_info(f"  Ă˘ÂÂ Creating probs_df with {len(timestamps)} timestamps")
            tprint_info(f"  Ă˘ÂÂ Timestamp type: {type(timestamps)}, dtype: {timestamps.dtype if hasattr(timestamps, 'dtype') else 'N/A'}")
            tprint_info(f"  Ă˘ÂÂ Timestamp range: {timestamps.min()} to {timestamps.max()}")
            tprint_info(f"  Ă˘ÂÂ Regime probs shape: {result['regime_probs'].shape}")
            
            probs_df = pd.DataFrame(
                result['regime_probs'],
                index=timestamps,
                columns=pd.Index(probs_columns)
            )
            
            # Debug: Verify DataFrame after creation
            tprint_info(f"  Ă˘ÂÂ probs_df shape: {probs_df.shape}")
            tprint_info(f"  Ă˘ÂÂ probs_df index type: {type(probs_df.index)}")
            tprint_info(f"  Ă˘ÂÂ probs_df index range: {probs_df.index.min()} to {probs_df.index.max()}")

            try:
                probs_values = probs_df.to_numpy(copy=False)
                top_prob = probs_values.max(axis=1)
                eps = 1e-12
                p_safe = np.clip(probs_values, eps, 1.0)
                entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
                regime_indices = np.arange(result['n_regimes'], dtype=float)
                expected_index = probs_values.dot(regime_indices)
                confidence_df = pd.DataFrame(
                    {
                        'timestamp': probs_df.index,
                        'regime_top_prob': top_prob,
                        'regime_entropy': entropy,
                        'regime_expected_index': expected_index,
                    }
                )
                confidence_df_to_save = confidence_df.reset_index(drop=True)
                self._save_artifact(
                    data=confidence_df_to_save,
                    artifact_name='rolling_hmm_regime_confidence_features',
                    artifact_type='data',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
            except Exception as _:
                pass

            # Save economic features used for HMM emissions so supervised models can reuse
            # the exact same normalized economic feature space.
            try:
                features_economic = result.get('features', None)
                if isinstance(features_economic, pd.DataFrame) and not features_economic.empty:
                    tprint_info("  [32mSaving economic features used by Rolling HMM[0m")

                    economic_df_to_save = features_economic.reset_index()
                    economic_df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)

                    self._save_artifact(
                        data=economic_df_to_save,
                        artifact_name='rolling_hmm_economic_features',
                        artifact_type='data',
                        metadata={
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            'economic_feature_names': list(features_economic.columns),
                        },
                    )
                else:
                    tprint_info("  [33mNo economic features found in result['features']; skipping economic artifact save[0m")
            except Exception as _:
                # Economic features are a convenience artifact; failure to save them
                # must not break the main HMM regime discovery pipeline.
                pass

            # Save labels
            # CRITICAL: Reset index to ensure it's saved correctly in HDF5 (same as probs_df)
            labels_df_to_save = labels_df.reset_index()
            labels_df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
            
            tprint(f"ÄÂÂÂ DEBUG: [STEP 9] About to save labels_df to HDF5: {labels_df_to_save.shape}", "INFO")
            tprint(f"ÄÂÂÂ DEBUG: [STEP 9] labels_df columns: {list(labels_df_to_save.columns)}", "INFO")
            self._save_artifact(
                data=labels_df_to_save,
                artifact_name='rolling_hmm_regime_labels',
                artifact_type='data',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )
            tprint(f"Ă˘ÂÂ DEBUG: [STEP 9] Successfully saved rolling_hmm_regime_labels", "SUCCESS")

            # Save probabilities
            # CRITICAL: Reset index to ensure it's saved correctly in HDF5
            # HDF5 can have issues with certain DatetimeIndex formats
            probs_df_to_save = probs_df.reset_index()
            probs_df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
            
            tprint_info(f"  Ă˘ÂÂ Saving probs_df with {len(probs_df_to_save)} rows")
            tprint_info(f"  Ă˘ÂÂ Columns: {list(probs_df_to_save.columns)}")
            
            self._save_artifact(
                data=probs_df_to_save,
                artifact_name='rolling_hmm_regime_probabilities',
                artifact_type='data',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )

            # Save transition matrix
            transition_matrix_df = pd.DataFrame(
                result['transition_matrix'],
                columns=pd.Index([f'to_regime_{i}' for i in range(result['n_regimes'])]),
                index=pd.Index([f'from_regime_{i}' for i in range(result['n_regimes'])])
            )
            self._save_artifact(
                data=transition_matrix_df,
                artifact_name='rolling_hmm_transition_matrix',
                artifact_type='data',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )

            # Save model summary
            self._save_artifact(
                data=result['model_summary'],
                artifact_name='rolling_hmm_model_summary',
                artifact_type='metadata',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )

            # Save quality metrics
            self._save_artifact(
                data=result['quality_metrics'],
                artifact_name='rolling_hmm_quality_metrics',
                artifact_type='metadata',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )

            return labels_df, probs_df

        except Exception as e:
            tprint_error(f"Ă˘ÂÂ Failed to save results: {e}")
            self.logger.error(f"Failed to save results: {e}", exc_info=True)
            raise

    async def _generate_reports(
        self,
        result: Dict[str, Any],
        market_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ):
        """Generate quality assessment and economic relevance reports."""
        try:
            metrics = result['quality_metrics']

            # Print summary
            tprint("", "INFO")
            tprint("=" * 80, "INFO")
            tprint("ÄÂÂÂ Rolling HMM Clustering Quality Report", "INFO")
            tprint("=" * 80, "INFO")
            tprint(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}", "INFO")
            tprint(f"Number of Regimes: {result['n_regimes']}", "INFO")
            # Note: Rolling HMM uses economic features, not PCA
            if 'pca_explained_variance' in result:
                tprint(f"PCA Explained Variance: {result['pca_explained_variance']:.2%}", "INFO")
            tprint("", "INFO")
            tprint("Quality Metrics:", "INFO")
            tprint(f"  - Quality Score: {metrics.get('quality_score', 0):.4f}", "INFO")
            tprint(f"  - Silhouette Score: {metrics.get('silhouette_score', 0):.4f}", "INFO")
            tprint(f"  - Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.4f}", "INFO")
            tprint(f"  - Temporal Smoothness: {metrics.get('temporal_smoothness', 0):.4f}", "INFO")
            tprint(f"  - Regime Persistence: {metrics.get('regime_persistence', 0):.2f} bars", "INFO")
            tprint("", "INFO")

            # Winning configuration details
            winning_params = config.get('rolling_hmm_params', {})
            hmm_model = result.get('hmm_model')
            hmm_config = getattr(hmm_model, 'config', None)
            tprint("Winning Configuration:", "INFO")
            if hmm_config is not None:
                tprint(f"  - Rolling HMM n_components: {hmm_config.n_components}", "INFO")
                tprint(f"  - Rolling HMM kappa: {hmm_config.kappa}", "INFO")
                tprint(f"  - Rolling HMM min_covar: {hmm_config.min_covar}", "INFO")
                tprint(f"  - Rolling HMM n_iter: {hmm_config.n_iter}", "INFO")
                tprint(f"  - Rolling HMM tol: {hmm_config.tol}", "INFO")
            if winning_params:
                for key, value in winning_params.items():
                    if key in {'n_components', 'kappa', 'min_covar', 'n_iter', 'tol'}:
                        continue  # already covered above
                    tprint(f"  - {key}: {value}", "INFO")
            else:
                tprint("  - No HPO overrides applied (using default configuration)", "INFO")
            tprint("", "INFO")

            tprint("Expected Durations per Regime:", "INFO")
            expected_durations = result.get('expected_durations', [])
            expected_total = float(np.sum(expected_durations)) if len(expected_durations) else 0.0
            for i, duration in enumerate(expected_durations):
                pct = (duration / expected_total * 100.0) if expected_total > 0 else 0.0
                tprint(f"  - Regime {i}: {duration:.2f} bars ({pct:.1f}%)", "INFO")
            tprint("=" * 80, "INFO")

            # Persist detailed reports via quality assessor
            metrics_obj = ClusterQualityMetrics(**metrics) if isinstance(metrics, dict) else metrics
            
            # Extract EWMA config for readable output
            ewma_config_idx = config.get('rolling_hmm_params', {}).get('ewma_config_idx', 0)
            ewma_config = DEFAULT_EWMA_CONFIGS[int(ewma_config_idx)]
            
            # Format rolling_hmm_params for readable output
            rolling_hmm_params = config.get('rolling_hmm_params', {})
            formatted_params = {}
            if rolling_hmm_params:
                for key, value in rolling_hmm_params.items():
                    if key == 'ewma_config_idx':
                        formatted_params[key] = f"{value} ({ewma_config.name})"
                    else:
                        formatted_params[key] = value
            
            method_config = {
                'rolling_hmm_params': formatted_params if formatted_params else "Default parameters used",
                'ewma_config': {
                    'name': ewma_config.name,
                    'short_window': ewma_config.short_window,
                    'long_window': ewma_config.long_window
                }
            }
            self.quality_assessor.generate_markdown_report(
                metrics_obj,
                symbol=symbol,
                method_specific_config=method_config,
                report_prefix="rolling_hmm_quality"
            )

            all_trials = None
            hpo_results = result.get('hpo_results') or config.get('hpo_results')
            if not hpo_results:
                hpo_results = config.get('hpo_summary')
            
            # Debug: Check HPO results more thoroughly
            if hpo_results is None:
                tprint("Ă˘ÂÂ ÄÂ¸Â No HPO results found in result or config", "WARNING")
            elif not hpo_results:
                tprint("Ă˘ÂÂ ÄÂ¸Â HPO results found but empty dictionary", "WARNING")
            elif not isinstance(hpo_results, dict):
                tprint(f"Ă˘ÂÂ ÄÂ¸Â HPO results found but wrong type: {type(hpo_results)}", "WARNING")
                hpo_results = None  # Reset to None to avoid further issues
            
            if hpo_results:
                tprint(f"ÄÂÂÂ Found HPO results with keys: {list(hpo_results.keys())}", "INFO")
                trial_keys = ['coarse_results', 'fine_results', 'refinement_results', 'second_round_results']
                all_trials = []
                for key in trial_keys:
                    trials = hpo_results.get(key)
                    if isinstance(trials, list):
                        tprint(f"ÄÂÂÂ Processing {len(trials)} trials from {key}", "INFO")
                        for trial in trials:
                            if isinstance(trial, dict):
                                params = trial.get('params', {})
                                trial_dict = params.copy() if isinstance(params, dict) else {}
                                trial_dict['score'] = trial.get('score')

                                quality_metrics = trial.get('quality_metrics')
                                if isinstance(quality_metrics, dict):
                                    # Preserve full metrics dict for downstream consumers
                                    trial_dict['quality_metrics'] = quality_metrics

                                    # Also flatten a shallow copy for convenience in CSV export
                                    for k, v in quality_metrics.items():
                                        if isinstance(v, dict):
                                            for sub_k, sub_v in v.items():
                                                trial_dict[f'{k}_{sub_k}'] = sub_v
                                        else:
                                            trial_dict[k] = v

                                trial_dict['trial_number'] = trial.get('trial_number')
                                all_trials.append(trial_dict)
                            elif isinstance(trial, tuple) and len(trial) >= 2:
                                params, score = trial[:2]
                                trial_dict = params.copy() if isinstance(params, dict) else {}
                                trial_dict['score'] = score
                                all_trials.append(trial_dict)
                    else:
                        tprint(f"ÄÂÂÂ No trials found for key {key} (type: {type(trials)})", "INFO")
            else:
                tprint("Ă˘ÂÂ ÄÂ¸Â No HPO results found in result or config", "WARNING")
                # Debug: check what keys are available
                available_keys = list(result.keys()) if result else []
                if available_keys:
                    tprint(f"ÄÂÂÂ Available result keys: {available_keys}", "INFO")
                config_keys = list(config.keys()) if config else []
                if config_keys:
                    tprint(f"ÄÂÂÂ Available config keys: {config_keys}", "INFO")

            if all_trials:
                tprint(f"ÄÂÂÂ Passing {len(all_trials)} trials to comprehensive CSV report generation", "INFO")
            else:
                tprint("Ă˘ÂÂ ÄÂ¸Â No trials available for all-trials CSV export", "WARNING")
            self.quality_assessor.generate_comprehensive_csv_report(
                metrics_obj,
                all_trials=all_trials,
                symbol=symbol,
                method_specific_config=method_config
            )

            # Generate Economic Relevance Analysis
            tprint("", "INFO")
            tprint("=" * 80, "INFO")
            tprint("ÄÂÂÂ° Generating Economic Relevance Analysis", "INFO")
            tprint("=" * 80, "INFO")

            try:
                # Extract regime labels and align with market data
                regime_labels = result['regime_labels']
                timestamps = result['timestamps']

                # Align market data with regime labels
                aligned_market_data = market_data.loc[timestamps]
                prices = aligned_market_data['close']

                # Create regime labels as pandas Series with timestamps
                regime_labels_series = pd.Series(regime_labels, index=timestamps, name='regime')

                # Extract regime type annotations from quality metrics if available
                regime_types = None
                to_dict_method = getattr(metrics, "to_dict", None)
                metrics_dict = to_dict_method() if callable(to_dict_method) else metrics
                if isinstance(metrics_dict, dict):
                    detected_types = metrics_dict.get('regime_type_per_cluster')
                    if not detected_types and metrics_dict.get('per_regime_metrics'):
                        detected_types = {
                            int(regime_id): regime_data.get('regime_type')
                            for regime_id, regime_data in metrics_dict['per_regime_metrics'].items()
                            if regime_data.get('regime_type')
                        }
                    if detected_types:
                        regime_types = {
                            int(regime_id): regime_type
                            for regime_id, regime_type in detected_types.items()
                            if regime_type is not None
                        }
                    if not regime_types:
                        fallback_types = self._infer_regime_types_from_metrics(metrics_dict)
                        if fallback_types:
                            regime_types = fallback_types
                            tprint_info(
                                f"  Ă˘ÂÂ Fallback regime classification inferred from returns: {regime_types}"
                            )
                        else:
                            detected_types = None
                    if regime_types:
                        tprint_info(
                            f"  Ă˘ÂÂ Using regime type mapping for economic analysis: {regime_types}"
                        )

                # Initialize economic analyzer (use 100 permutations for broad-strokes significance)
                economic_analyzer = RegimeEconomicRelevanceAnalyzer(
                    risk_free_rate=0.02,
                    trading_days_per_year=365 * 24 if timeframe == '1h' else 365,  # Adjust for hourly data
                    transaction_cost=0.001,
                    significance_tests=True,
                    n_permutations=100
                )

                tprint_info("  Ă˘ÂÂ Evaluating trading strategies based on regimes")

                # Evaluate strategies
                strategies = economic_analyzer.evaluate_strategies(
                    prices=prices,
                    regime_labels=regime_labels_series,
                    regime_types=regime_types,
                    use_dynamic_mapping=True,
                )

                # Optionally perform significance tests on strategies
                economic_analyzer.perform_significance_test(strategies)

            except Exception as e:
                tprint_warning(f"Ă˘ÂÂ ÄÂ¸Â  Economic relevance analysis failed: {e}")
                self.logger.warning(f"Economic relevance analysis failed: {e}", exc_info=True)

        except Exception as e:
            tprint_warning(f"Ă˘ÂÂ ÄÂ¸Â  Failed to generate reports: {e}")
            self.logger.warning(f"Failed to generate reports: {e}", exc_info=True)

    def _infer_regime_types_from_metrics(self, metrics_dict: Dict[str, Any]) -> Dict[int, str]:
        """Infer regime types using per-regime economic metrics."""

        per_regime_metrics = metrics_dict.get("per_regime_metrics")
        if not per_regime_metrics:
            return {}

        regime_data_list: List[Dict[str, float]] = []
        for regime_id, regime_data in per_regime_metrics.items():
            if not isinstance(regime_data, dict):
                continue
            mean_ret = regime_data.get("mean_return")
            if mean_ret is None:
                continue

            regime_data_list.append(
                {
                    "id": int(regime_id),
                    "mean_return": float(mean_ret),
                    "volatility": float(regime_data.get("volatility", 0.0) or 0.0),
                    "sharpe": float(regime_data.get("sharpe", 0.0) or 0.0),
                }
            )

        if not regime_data_list:
            return {}

        # Percentile thresholds
        mean_returns = [r["mean_return"] for r in regime_data_list]
        vols = [r["volatility"] for r in regime_data_list]
        sharpes = [r["sharpe"] for r in regime_data_list]

        mean_p70 = float(np.percentile(mean_returns, 70))
        mean_p30 = float(np.percentile(mean_returns, 30))
        vol_p70 = float(np.percentile(vols, 70))
        sharpe_p50 = float(np.percentile(sharpes, 50))

        # Absolute guards: avoid degenerate thresholds when dispersion is tiny
        mean_abs_threshold = max(0.00015, abs(mean_p30))
        high_vol_threshold = max(vol_p70, 0.007)
        positive_sharpe_threshold = max(sharpe_p50, 0.05)
        negative_sharpe_threshold = min(sharpe_p50, -0.02)

        tprint_debug(
            "  [RegimeType] Thresholds Ă˘ÂÂ "
            f"mean_p70={mean_p70:.6f}, mean_p30={mean_p30:.6f}, "
            f"vol_p70={vol_p70:.6f}, sharpe_p50={sharpe_p50:.4f}, "
            f"high_vol_abs={high_vol_threshold:.6f}"
        )

        inferred: Dict[int, str] = {}
        for r in regime_data_list:
            rid = r["id"]
            mean_ret = r["mean_return"]
            vol_val = r["volatility"]
            sharpe_val = r["sharpe"]

            tprint_debug(
                "  [RegimeType] Regime {} Ă˘ÂÂ mean={:.6f}, vol={:.6f}, sharpe={:.4f}".format(
                    rid, mean_ret, vol_val, sharpe_val
                )
            )

            # Risk-off / volatile regimes: poor returns, elevated vol, negative Sharpe
            if (
                (mean_ret < mean_p30 or mean_ret < -mean_abs_threshold)
                and vol_val > high_vol_threshold
                and sharpe_val <= negative_sharpe_threshold
            ):
                inferred[rid] = "volatile"

            # Trending (risk-on): strong returns with healthy Sharpe
            elif (
                mean_ret > max(mean_p70, mean_abs_threshold * 1.5)
                and sharpe_val >= positive_sharpe_threshold
            ):
                inferred[rid] = "trending"

            # Stable / low-vol: muted returns and low volatility
            elif abs(mean_ret) <= mean_abs_threshold and vol_val < high_vol_threshold * 0.85:
                inferred[rid] = "stable"

            # Fallback classifications
            elif sharpe_val < 0 and vol_val > vol_p70:
                inferred[rid] = "volatile"
            else:
                inferred[rid] = "neutral"

        return inferred

    def _log_best_params(self, best_params: Dict[str, Any]):
        """Log best parameters from HPO."""
        tprint("Best Parameters:", "INFO")
        for key, value in best_params.items():
            if key == "ewma_config_idx":
                ewma_config = DEFAULT_EWMA_CONFIGS[int(value)]
                tprint(f"  - EWMA Config: {ewma_config.name} (idx={value})", "INFO")
            else:
                tprint(f"  - {key}: {value}", "INFO")

    def _handle_execution_error(
        self,
        error: Exception,
        config: Dict[str, Any],
        execution_time: float,
    ) -> Dict[str, Any]:
        """Handle execution errors consistently with logging and structured result."""
        error_msg = str(error)
        self.logger.error(
            f"Rolling HMM execution failed: {error_msg}",
            exc_info=True,
        )
        tprint(f"Ă˘ÂÂ Rolling HMM Regime Discovery failed: {error_msg}", "ERROR")

        return {
            'success': False,
            'error': error_msg,
            'artifacts': {},
            'metrics': {},
            'execution_time': execution_time,
        }

    def _apply_execution_mode_filter(
        self,
        data: pd.DataFrame,
        execution_mode: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Apply execution mode data filtering using centralized lookback days.

        Args:
            data: Market data DataFrame
            execution_mode: Execution mode ('blank', 'light', 'full')
            timeframe: Timeframe string (e.g., '1h', '15m')

        Returns:
            Filtered DataFrame
        """
        # Samples per day mapping
        samples_per_day_map = {
            '1m': 1440,   # 60 * 24
            '3m': 480,    # 20 * 24
            '5m': 288,    # 12 * 24
            '15m': 96,    # 4 * 24
            '30m': 48,    # 2 * 24
            '1h': 24,     # 1 * 24
            '4h': 6,      # 24 / 4
            '1d': 1
        }

        exec_mode = str(execution_mode or 'full').lower()

        # Full mode: no filtering
        if exec_mode == 'full':
            tprint_info("  Ă˘ÂÂ Full mode: Using all available data (no filtering)")
            return data

        # Determine days limit based on centralized execution mode configuration
        try:
            from src.training.steps.market_analysis.shared_utils.execution_mode_lookback_config import (
                get_execution_mode_config,
            )

            exec_config = get_execution_mode_config()
            days_limit = exec_config.get_data_loading_days(exec_mode)
        except Exception:
            # Fallback: no filtering if centralized config is unavailable
            days_limit = None

        if days_limit is None:
            tprint_info(f"  Ă˘ÂÂ {exec_mode.capitalize()} mode: No explicit day limit configured, using all available data")
            return data

        if exec_mode == 'blank':
            tprint_info(f"  Ă˘ÂÂ Blank mode: Using {days_limit} days of data (centralized config)")
        elif exec_mode == 'light':
            tprint_info(f"  Ă˘ÂÂ Light mode: Using {days_limit} days of data (centralized config)")
        else:
            tprint_info(f"  Ă˘ÂÂ {exec_mode.capitalize()} mode: Using {days_limit} days of data (centralized config)")

        # Calculate sample limit
        samples_per_day = samples_per_day_map.get(timeframe, 24)
        limit = days_limit * samples_per_day
        
        tprint_info(f"  Ă˘ÂÂ Data filtering: {days_limit} days ÄÂ {samples_per_day} samples/day = {limit} samples limit")

        # Apply filter
        if len(data) > limit:
            filtered = data.tail(limit).copy()  # Keep most recent data
            tprint_info(f"  Ă˘ÂÂ {execution_mode.capitalize()} mode: Filtered to {days_limit} days ({limit} samples)")
            tprint_info(f"  Ă˘ÂÂ Filtered data range: {filtered.index.min()} to {filtered.index.max()}")
            return filtered

        tprint_info(f"  Ă˘ÂÂ {execution_mode.capitalize()} mode: Data size ({len(data)}) within limit ({limit} samples)")
        tprint_info(f"  Ă˘ÂÂ Data range: {data.index.min()} to {data.index.max()}")
        return data

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration."""
        tprint_debug("Validating Rolling HMM configuration input")
        required_keys = ['symbol', 'exchange']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")


# Auto-register step
from src.training.steps.base_step import step_registry
step_registry.register('rolling_hmm_regime_discovery', RollingHMMRegimeDiscoveryStep)
