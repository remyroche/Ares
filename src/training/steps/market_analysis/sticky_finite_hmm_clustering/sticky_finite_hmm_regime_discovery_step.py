"""
Sticky Finite HMM Regime Discovery Step

This step performs regime discovery using Sticky Finite HMM (K=5) with Variational Bayes
using Pyro + PyTorch. It integrates with artifact management, quality assessment,
and the BaseStep framework for standardized execution.

Inherits from BaseStep for standardized artifact management and autonomous execution.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

# Runtime imports
try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import Sticky Finite HMM components
from .sticky_finite_hmm_clusterer import (
    StickyFiniteHMMClusterer,
    StickyFiniteHMMConfig,
    DEPENDENCIES_AVAILABLE
)
from .standalone_runner import run_sticky_finite_hmm_clustering

# Import quality assessor
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics
)

from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS
)

logger = logging.getLogger(__name__)


class StickyFiniteHMMRegimeDiscoveryStep(BaseStep):
    """
    Sticky Finite HMM Regime Discovery Step.
    
    Performs regime discovery using Sticky Finite HMM with fixed K=5 states
    and Variational Bayes inference (Pyro + PyTorch).
    
    Key features:
    - Fixed K=5 states (not nonparametric)
    - Dirichlet priors with stickiness for persistent regimes
    - VB/SVI inference (faster than Gibbs sampling)
    - KMeans warm start for fast convergence
    - ELBO tracking with early stopping
    
    Inherits from BaseStep to provide:
    - Standardized artifact management
    - Automatic context setting
    - Market data access by default
    - Consistent result saving
    """
    
    def __init__(self, step_name: str = "sticky_finite_hmm_regime_discovery"):
        """
        Initialize the Sticky Finite HMM regime discovery step.

        Args:
            step_name: Name for this step (used for artifact organization)
        """
        super().__init__(step_name)
        self.logger = system_logger.getChild('StickyFiniteHMMRegimeDiscovery')

        # Validate dependencies
        if not DEPENDENCIES_AVAILABLE:
            self.logger.error("Pyro and PyTorch not available")
            tprint_error("❌ Pyro and PyTorch required for Sticky Finite HMM")
            raise ImportError(
                "Sticky Finite HMM requires pyro-ppl and torch. "
                "Install with: pip install pyro-ppl torch"
            )

        # Quality assessor will be created lazily when first accessed
        self._quality_assessor = None

        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    @property
    def quality_assessor(self) -> ClusterQualityAssessor:
        """Lazy property for quality assessor."""
        if self._quality_assessor is None:
            self._quality_assessor = ClusterQualityAssessor(
                artifact_manager=self.artifact_manager,
                enable_hardware_optimization=True,
                enable_vectorization=True
            )
        return self._quality_assessor
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Sticky Finite HMM regime discovery with optional auto-tuning.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'BTCUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Optional timeframe override (defaults to regime_timeframe)
                - regime_timeframe: Timeframe for regime detection (default: '1h')
                - execution_mode: 'full', 'light', or 'blank'
                - sticky_finite_hmm_params: Optional parameters override
                - enable_auto_tuning: Whether to run auto-tuning (default: True in all modes,
                                                                           False only if manual params provided)
                - auto_tuning_config: Optional auto-tuning configuration:
                    - use_hierarchical: Use hierarchical optimization (default: True)
                    - use_multi_objective: Use multi-objective Pareto optimization (default: False)
                    - n_rounds: Number of optimization rounds (default: 2)
                    - tpe_trials: Number of TPE trials (default: 100)
                    - timeout: Timeout in seconds (default: 3600)
                    - cache_dir: Directory to cache results (default: None)
                
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
            - 'execution_time': float seconds taken to execute
            - 'auto_tuning_results': dict of auto-tuning results if enabled (optional)
        """
        start_time = time.time()
        
        # Validate configuration
        try:
            self._validate_config(config)
        except Exception as e:
            return self._handle_execution_error(e, config)
        
        # Extract configuration
        symbol = config.get('symbol', 'BTCUSDT')
        exchange = config.get('exchange', 'binance')
        
        # Use regime_timeframe for regime detection
        regime_timeframe = config.get('regime_timeframe', '1h')
        timeframe = config.get('timeframe', regime_timeframe)
        
        # Override to regime_timeframe
        if timeframe != regime_timeframe:
            tprint(
                f"⏰ Using regime_timeframe={regime_timeframe} for Sticky Finite HMM "
                f"(overriding timeframe={timeframe})",
                "INFO"
            )
            timeframe = regime_timeframe
        
        tprint(
            f"🚀 Starting Enhanced Sticky Finite HMM Regime Discovery for {symbol} on {exchange} "
            f"(timeframe: {timeframe})",
            "INFO"
        )
        tprint("🧠 Enhanced Features:", "INFO")
        tprint("   - Structured variational inference with forward-backward", "INFO")
        tprint("   - Natural gradient updates for reduced variance", "INFO") 
        tprint("   - Rao-Blackwellization for exact sufficient statistics", "INFO")
        tprint("   - Vectorized computations for optimal performance", "INFO")
        
        try:
            # Set artifact manager context
            self.artifact_manager.set_context(
                step_name=self.step_name,
                symbol=symbol,
                exchange=exchange,
                datetime=datetime.now(),
                information="regime_discovery",
                direction="long",
                model="Analyst"
            )
            
            # Load market data (from config or artifacts)
            tprint("📥 Loading market data...", "INFO")
            
            # Check if market_data provided directly in config (for testing/external use)
            if 'market_data' in config and config['market_data'] is not None:
                market_data = config['market_data']
                tprint(f"✅ Using market data from config ({len(market_data)} samples)", "SUCCESS")
            else:
                market_data = self._load_market_data(symbol, exchange, timeframe, config)
                
                if market_data is None or market_data.empty:
                    tprint_error(f"❌ No market data available for {symbol} on {timeframe}")
                    raise ValueError(f"No market data available for {symbol} on {timeframe}")
                
                tprint(f"✅ Loaded {len(market_data)} samples of market data", "SUCCESS")
            
            # Check if auto-tuning is enabled (default: True for all modes)
            enable_auto_tuning = config.get('enable_auto_tuning', True)
            auto_tuning_results = None
            
            # Only skip if user provided manual params (unless explicitly enabled)
            if 'sticky_finite_hmm_params' in config and config['sticky_finite_hmm_params']:
                # If user provided manual params, auto-tuning is off by default
                # (unless explicitly enabled)
                if 'enable_auto_tuning' not in config:
                    enable_auto_tuning = False
                    tprint_info("ℹ️  Manual params provided, skipping auto-tuning (set enable_auto_tuning=True to override)")
            
            # Show execution mode info if auto-tuning in light/blank
            execution_mode = config.get('execution_mode', 'full')
            if enable_auto_tuning and execution_mode in ['light', 'blank']:
                tprint_info(f"ℹ️  Auto-tuning enabled in '{execution_mode}' mode (will use reduced trials for speed)")
            
            if enable_auto_tuning:
                tprint("", "INFO")
                tprint("🎯 Auto-Tuning Enabled - Finding Optimal Hyperparameters", "INFO")
                tprint("=" * 80, "INFO")
                
                # Run auto-tuning
                auto_tuning_results, best_params = await self._run_auto_tuning(
                    market_data, symbol, exchange, timeframe, config
                )
                
                # Update config with best parameters
                if auto_tuning_results and best_params:
                    tprint("", "INFO")
                    tprint("✅ Auto-Tuning Complete - Using Optimal Parameters", "SUCCESS")
                    if 'sticky_finite_hmm_params' not in config:
                        config['sticky_finite_hmm_params'] = {}
                    config['sticky_finite_hmm_params'].update(best_params)
                else:
                    tprint_warning("⚠️ Auto-tuning did not complete, using default parameters")
                
                tprint("=" * 80, "INFO")
                tprint("", "INFO")
            
            # Run clustering (with auto-tuned or default params)
            tprint("🔍 Running Sticky Finite HMM clustering...", "INFO")
            result = await self._run_clustering(
                market_data, symbol, exchange, timeframe, config
            )
            
            # Save artifacts
            labels_df, probs_df = await self._save_results(result, symbol, exchange, timeframe, config)
            
            # Generate comprehensive CSV exports and markdown report
            await self._generate_reports(result, symbol, exchange, timeframe, config)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            tprint(
                f"✅ Sticky Finite HMM Regime Discovery completed in {execution_time:.2f}s",
                "SUCCESS"
            )
            
            # Prepare artifacts dictionary with consistent naming
            step_artifacts = {
                'hdp_hmm_regime_labels': labels_df,  # Use consistent naming with HDP-HMM
                'hdp_hmm_regime_probabilities': probs_df,
                'hdp_hmm_cluster_statistics': self._get_artifact(
                    "sticky_finite_hmm_cluster_statistics", "metadata"
                ),
                'hdp_hmm_transition_matrix': self._get_artifact(
                    "sticky_finite_hmm_transition_matrix", "data"
                ),
                'sticky_finite_hmm_regime_labels': labels_df,  # Keep both for compatibility
                'sticky_finite_hmm_regime_probabilities': probs_df
            }
            
            # Prepare final result
            final_result = {
                'success': True,
                'artifacts': step_artifacts,
                'metrics': result.get('quality_metrics', {}),
                'execution_time': execution_time,
                'n_regimes': result.get('n_clusters', 0),
                'composite_score': result.get('quality_metrics', {}).get('composite_score', 0.0)
            }
            
            # Add auto-tuning results if available
            if auto_tuning_results:
                final_result['auto_tuning_results'] = auto_tuning_results
            
            return final_result
            
        except Exception as e:
            return self._handle_execution_error(e, config)
    
    def _load_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Load market data from artifacts or historical_data directory.
        
        Looks for market data in the following order:
        1. Current step's artifacts
        2. data_collection step artifacts
        3. klines_downloading_processing step artifacts
        4. historical_data directory (via DataLoader)
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe for data
            config: Configuration dictionary
            
        Returns:
            DataFrame with market data or None if not found
        """
        # Try multiple artifact sources first
        artifact_sources = [
            ('klines_downloading_processing', 'klines_data'),
            ('data_collection', 'market_data'),
            ('data_reading', 'ohlcv_data'),
        ]
        
        for step_name, artifact_name in artifact_sources:
            try:
                # Set context to look for data
                self.artifact_manager.set_context(
                    step_name=step_name,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                
                # Try to load market data
                market_data = self._get_artifact(
                    artifact_name=artifact_name,
                    artifact_type="data"
                )
                
                if market_data is not None and not market_data.empty:
                    tprint(
                        f"✅ Loaded market data from {step_name}/{artifact_name}",
                        "SUCCESS"
                    )
                    
                    # Apply light mode filter if needed
                    market_data = self._apply_light_mode_filter(
                        market_data, config, timeframe
                    )
                    
                    return market_data
                    
            except Exception as e:
                self.logger.debug(
                    f"Could not load from {step_name}/{artifact_name}: {e}"
                )
                continue
        
        # Fallback: Try to load from historical_data directory
        tprint(
            f"⚠️ Could not load from artifacts, trying historical_data directory...",
            "WARNING"
        )
        
        try:
            from src.utils.data_loader import DataLoader
            
            data_loader = DataLoader()
            
            # Currently supports ETHUSDT 1h data
            if symbol.upper() == 'ETHUSDT' and timeframe == '1h':
                market_data = data_loader.load_ethusdt_1h_data()
                
                if market_data is not None and not market_data.empty:
                    tprint(
                        f"✅ Loaded {symbol} {timeframe} data from historical_data ({len(market_data)} rows)",
                        "SUCCESS"
                    )
                    
                    # Apply light mode filter if needed
                    market_data = self._apply_light_mode_filter(
                        market_data, config, timeframe
                    )
                    
                    return market_data
            else:
                tprint(
                    f"⚠️ DataLoader currently only supports ETHUSDT 1h data (requested: {symbol} {timeframe})",
                    "WARNING"
                )
        
        except Exception as e:
            self.logger.warning(f"Could not load from historical_data: {e}")
        
        # No data found anywhere
        tprint(
            f"⚠️ Could not load market data for {symbol} from any source",
            "WARNING"
        )
        return None
    
    async def _run_auto_tuning(
        self,
        market_data: Any,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Run automatic hyperparameter tuning for Sticky Finite HMM.
        
        Args:
            market_data: Market data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
            
        Returns:
            Tuple of (tuning_results, best_params)
        """
        try:
            # Import auto-tuner
            from .sticky_finite_hmm_auto_tuner import run_sticky_finite_hmm_auto_tuning
            
            tprint("🔧 Auto-Tuner Loaded", "INFO")
            
            # Get auto-tuning configuration
            auto_tuning_config = config.get('auto_tuning_config', {})
            use_hierarchical = auto_tuning_config.get('use_hierarchical', True)
            use_multi_objective = auto_tuning_config.get('use_multi_objective', False)  # ✨ Multi-objective optimization
            
            # Adjust trials based on execution mode
            execution_mode = config.get('execution_mode', 'full')
            if execution_mode == 'light':
                # Light mode: faster auto-tuning (50% trials)
                default_n_rounds = 1
                default_tpe_trials = 50
                default_timeout = 1800  # 30 min
            elif execution_mode == 'blank':
                # Blank mode: minimal auto-tuning (25% trials)
                default_n_rounds = 1
                default_tpe_trials = 25
                default_timeout = 900  # 15 min
            else:
                # Full mode: complete auto-tuning
                default_n_rounds = 2
                default_tpe_trials = 100
                default_timeout = 3600  # 1 hour
            
            n_rounds = auto_tuning_config.get('n_rounds', default_n_rounds)
            tpe_trials = auto_tuning_config.get('tpe_trials', default_tpe_trials)
            timeout = auto_tuning_config.get('timeout', default_timeout)
            cache_dir = auto_tuning_config.get('cache_dir', None)
            
            tprint(f"   - Strategy: {'Hierarchical' if use_hierarchical else 'Standard'}", "INFO")
            tprint(f"   - Multi-Objective: {use_multi_objective}", "INFO")
            tprint(f"   - Rounds: {n_rounds}", "INFO")
            tprint(f"   - TPE Trials: {tpe_trials}", "INFO")
            tprint(f"   - Timeout: {timeout}s ({timeout/60:.1f} min)", "INFO")
            tprint("", "INFO")
            
            # Run auto-tuning (in executor to not block async)
            loop = asyncio.get_event_loop()
            best_params, best_score, tuning_results = await loop.run_in_executor(
                None,
                run_sticky_finite_hmm_auto_tuning,
                market_data,
                symbol,
                exchange,
                timeframe,
                None,  # search_space (use default)
                use_hierarchical,
                use_multi_objective,  # ✨ Pass multi-objective flag
                n_rounds,
                5,  # coarse_grid_points
                5,  # fine_grid_points
                tpe_trials,
                timeout,
                3,  # cv_folds
                DEFAULT_CLUSTERING_GOALS,  # optimization_goals
                DEFAULT_OPTIMIZATION_TARGETS,  # optimization_targets
                42,  # random_state
                cache_dir,
                True  # verbose
            )
            
            # Prepare results summary
            results_summary = {
                'best_score': best_score,
                'best_params': best_params,
                'method': tuning_results.get('method', 'unknown'),
                'total_time': tuning_results.get('total_time', 0.0),
                'total_trials': tuning_results.get('total_trials', 0)
            }
            
            tprint("", "INFO")
            tprint("=" * 80, "SUCCESS")
            tprint(f"🎉 Auto-Tuning Complete!", "SUCCESS")
            tprint(f"   Best Score: {best_score:.4f}", "SUCCESS")
            tprint(f"   Total Time: {tuning_results.get('total_time', 0.0):.1f}s", "SUCCESS")
            tprint(f"   Total Trials: {tuning_results.get('total_trials', 0)}", "SUCCESS")
            tprint("", "SUCCESS")
            tprint("Best Parameters Found:", "SUCCESS")
            tprint(f"   - K: {best_params.get('K', 5)}", "SUCCESS")
            tprint(f"   - n_mixtures: {best_params.get('n_mixtures', 1)}", "SUCCESS")
            tprint(f"   - kappa: {best_params.get('kappa', 10.0):.2f}", "SUCCESS")
            tprint(f"   - base_alpha: {best_params.get('base_alpha', 0.5):.3f}", "SUCCESS")
            tprint(f"   - lr: {best_params.get('lr', 1e-2):.5f}", "SUCCESS")
            tprint(f"   - pca_components: {best_params.get('pca_components', 15)}", "SUCCESS")
            tprint("=" * 80, "SUCCESS")
            
            # Show Pareto front if multi-objective was used
            if use_multi_objective and 'pareto_front' in results_summary:
                pareto = results_summary['pareto_front']
                tprint("", "SUCCESS")
                tprint(f"🎯 Pareto Front: {pareto['n_solutions']} non-dominated solutions found", "SUCCESS")
            
            return results_summary, best_params
            
        except ImportError as e:
            tprint_error(f"❌ Auto-tuner not available: {e}")
            tprint_warning("⚠️ Install optimization dependencies: pip install optuna")
            return None, None
        except Exception as e:
            self.logger.error(f"Auto-tuning failed: {e}", exc_info=True)
            tprint_error(f"❌ Auto-tuning failed: {e}")
            return None, None
    
    async def _run_clustering(
        self,
        market_data: Any,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Sticky Finite HMM clustering with provided or default parameters.
        
        Args:
            market_data: Market data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
            
        Returns:
            Dictionary with clustering results
        """
        tprint_info("🔧 Configuring Sticky Finite HMM parameters...")
        
        # Get parameters
        params = config.get('sticky_finite_hmm_params', {})
        
        # Extract parameters with defaults
        K = params.get('K', 5)
        n_mixtures = params.get('n_mixtures', 1)
        base_alpha = params.get('base_alpha', 0.5)
        kappa = params.get('kappa', 10.0)
        num_iters = params.get('num_iters', 150)  # Reduced from 800 for faster training
        lr = params.get('lr', 1e-2)
        
        # Feature selection
        min_features = params.get('min_features', 50)
        max_features = params.get('max_features', 100)
        
        # PCA (default to 15 components, matching core model)
        enable_pca = params.get('enable_pca', True)
        pca_components = params.get('pca_components', 15)
        
        # Optimization flag (skip posteriors during auto-tuning)
        compute_posteriors = params.get('compute_posteriors', True)
        
        tprint(f"📊 Sticky Finite HMM Parameters:", "INFO")
        tprint(f"  - K: {K}", "INFO")
        tprint(f"  - n_mixtures: {n_mixtures}", "INFO")
        tprint(f"  - base_alpha: {base_alpha}", "INFO")
        tprint(f"  - kappa: {kappa}", "INFO")
        tprint(f"  - num_iters: {num_iters}", "INFO")
        tprint(f"  - lr: {lr}", "INFO")
        tprint(f"  - features: {min_features}-{max_features}", "INFO")
        tprint(f"  - PCA: {enable_pca} (components: {pca_components})", "INFO")
        tprint(f"  - compute_posteriors: {compute_posteriors} ({'FULL' if compute_posteriors else 'FAST mode'})", "INFO")
        
        # Run clustering (async wrapper for sync function)
        tprint_info("🚀 Launching clustering in executor...")
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            run_sticky_finite_hmm_clustering,
            market_data,
            symbol,
            exchange,
            timeframe,
            min_features,
            max_features,
            K,
            n_mixtures,
            base_alpha,
            kappa,
            num_iters,
            lr,
            enable_pca,
            pca_components,
            False,  # save_results (we handle saving in this step)
            None,  # output_dir
            compute_posteriors  # Pass optimization flag
        )
        
        tprint_success(f"✅ Clustering complete: {results.get('n_clusters', 0)} regimes discovered")
        
        # Run comprehensive quality assessment using ClusterQualityAssessor
        tprint_info("🔍 Running comprehensive quality assessment...")
        try:
            # Extract data for quality assessment
            cluster_labels = np.array(results['cluster_labels'])
            feature_matrix = results.get('feature_matrix')
            
            if feature_matrix is None:
                tprint_warning("⚠️ No feature matrix found in results, creating basic features")
                # Create basic features from market data if needed
                feature_matrix = self._create_basic_features(market_data)
            
            # Ensure data alignment
            min_length = min(len(cluster_labels), len(feature_matrix))
            cluster_labels = cluster_labels[:min_length]
            feature_matrix = feature_matrix.iloc[:min_length].reset_index(drop=True)
            timestamps = market_data.index[:min_length]
            
            # Calculate forward returns for economic validation
            forward_returns = market_data['close'].pct_change().shift(-1).iloc[:min_length]
            
            # Run comprehensive quality assessment
            quality_metrics = self.quality_assessor.assess_quality(
                regime_labels=cluster_labels,
                feature_data=feature_matrix,
                forward_returns=forward_returns,
                timestamps=timestamps,
                min_regime_size=10,
                temporal_sensitivity_mode="standard"
            )
            
            # Add comprehensive quality metrics to results
            results['comprehensive_quality_metrics'] = quality_metrics.to_dict()
            results['quality_score'] = quality_metrics.quality_score or 0.0
            
            tprint_success(f"✅ Quality assessment complete: Score = {quality_metrics.quality_score:.4f}")
            
            # Save detailed CSV reports
            self._save_quality_reports(quality_metrics, symbol, exchange, timeframe)
            
        except Exception as e:
            tprint_error(f"❌ Quality assessment failed: {e}")
            self.logger.warning(f"Quality assessment failed: {e}")
        
        return results
    
    def _create_basic_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic features from market data for quality assessment.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            DataFrame with basic features
        """
        features = pd.DataFrame({
            'returns': market_data['close'].pct_change(),
            'volume': market_data['volume'],
            'high_low_ratio': market_data['high'] / market_data['low'],
            'open_close_ratio': market_data['open'] / market_data['close'],
            'price_change': market_data['close'] - market_data['open'],
            'volatility': market_data['high'] - market_data['low'],
            'volume_price_trend': market_data['volume'] * market_data['close'].pct_change(),
            'price_momentum': market_data['close'].pct_change(5),
            'volume_sma': market_data['volume'].rolling(20).mean(),
            'price_position': (market_data['close'] - market_data['low']) / (market_data['high'] - market_data['low'])
        }).fillna(0)
        
        return features
    
    def _save_quality_reports(self, quality_metrics: ClusterQualityMetrics, 
                             symbol: str, exchange: str, timeframe: str) -> None:
        """
        Save detailed quality assessment reports to CSV files.
        
        Args:
            quality_metrics: ClusterQualityMetrics object
            symbol: Trading symbol
            exchange: Exchange name  
            timeframe: Timeframe
        """
        tprint_info("💾 Generating detailed quality assessment CSV...")
        
        try:
            # Create output directory
            from pathlib import Path
            from datetime import datetime
            
            output_dir = Path("artifacts") / "quality_reports" / symbol / exchange / timeframe
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Main summary report
            summary_data = {
                'Metric': [
                    'Silhouette Score',
                    'Davies-Bouldin Index', 
                    'Calinski-Harabasz Index',
                    'Within-Regime CV',
                    'Between-Regime CV',
                    'Temporal Smoothness',
                    'Regime Persistence',
                    'Number of Regimes',
                    'Noise Ratio',
                    'Balance Score',
                    'Overall Quality Score'
                ],
                'Value': [
                    quality_metrics.silhouette_score,
                    quality_metrics.davies_bouldin_score,
                    quality_metrics.calinski_harabasz_score,
                    quality_metrics.within_regime_cv,
                    quality_metrics.between_regime_cv,
                    quality_metrics.temporal_smoothness,
                    quality_metrics.regime_persistence,
                    quality_metrics.n_regimes,
                    quality_metrics.noise_ratio,
                    quality_metrics.balance_score,
                    quality_metrics.quality_score
                ],
                'Description': [
                    'Cluster separation quality (-1 to 1, higher better)',
                    'Cluster separation quality (lower better)',
                    'Cluster separation quality (higher better)',
                    'Within-regime feature consistency (lower better)',
                    'Between-regime feature separation (higher better)',
                    'Temporal stability of regimes (0 to 1, higher better)',
                    'Average regime duration in periods',
                    'Number of discovered regimes',
                    'Ratio of noise points (lower better)',
                    'Balance of cluster sizes (0 to 1, higher better)',
                    'Composite quality score (0 to 1, higher better)'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = output_dir / f"clustering_quality_summary_{timestamp}.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            
            # 2. Per-regime detailed metrics
            if quality_metrics.per_regime_metrics:
                regime_data = []
                for regime_id, regime_metrics in quality_metrics.per_regime_metrics.items():
                    regime_data.append({
                        'Regime_ID': regime_id,
                        'Size': regime_metrics.get('size', 0),
                        'Size_Percentage': regime_metrics.get('size_pct', 0),
                        'Mean_Return': regime_metrics.get('mean_return', 0),
                        'Volatility': regime_metrics.get('volatility', 0),
                        'Sharpe_Ratio': regime_metrics.get('sharpe', 0),
                        'Max_Drawdown': regime_metrics.get('max_drawdown', 0),
                        'Win_Rate': regime_metrics.get('win_rate', 0),
                        'Regime_Type': regime_metrics.get('regime_type', 'unknown'),
                        'Duration_Mean': regime_metrics.get('duration_mean', 0),
                        'Duration_Std': regime_metrics.get('duration_std', 0)
                    })
                
                regime_df = pd.DataFrame(regime_data)
                regime_csv_path = output_dir / f"regime_detailed_metrics_{timestamp}.csv"
                regime_df.to_csv(regime_csv_path, index=False)
            
            # 3. Economic validation metrics
            if quality_metrics.economic_validation:
                econ_data = {
                    'Economic_Metric': [
                        'Portfolio Return',
                        'Portfolio Sharpe Ratio',
                        'Max Drawdown',
                        'Volatility',
                        'Hit Rate',
                        'Profit Factor',
                        'Average Trade Return',
                        'Target Return Achievement'
                    ],
                    'Value': [
                        quality_metrics.economic_validation.get('portfolio_return', 0),
                        quality_metrics.economic_validation.get('portfolio_sharpe', 0),
                        quality_metrics.economic_validation.get('max_drawdown', 0),
                        quality_metrics.economic_validation.get('portfolio_volatility', 0),
                        quality_metrics.economic_validation.get('hit_rate', 0),
                        quality_metrics.economic_validation.get('profit_factor', 0),
                        quality_metrics.economic_validation.get('avg_trade_return', 0),
                        quality_metrics.economic_validation.get('target_return_achievement', 0)
                    ],
                    'Benchmark': [
                        'Higher better',
                        'Higher better',
                        'Lower better', 
                        'Lower better',
                        'Higher better',
                        'Higher better',
                        'Higher better',
                        'Higher better'
                    ]
                }
                
                econ_df = pd.DataFrame(econ_data)
                econ_csv_path = output_dir / f"economic_validation_{timestamp}.csv"
                econ_df.to_csv(econ_csv_path, index=False)
            
            # 4. Temporal analysis metrics
            temporal_data = {
                'Temporal_Metric': [
                    'Temporal Smoothness',
                    'Temporal Smoothness (Raw)',
                    'Flip-Flop Ratio',
                    'Regime Persistence',
                    'Average Duration',
                    'Duration Std Dev',
                    'Min Duration',
                    'Max Duration'
                ],
                'Value': [
                    quality_metrics.temporal_smoothness,
                    quality_metrics.temporal_smoothness_raw,
                    quality_metrics.flip_flop_ratio,
                    quality_metrics.regime_persistence,
                    quality_metrics.regime_duration_distribution.get('mean_duration', 0),
                    quality_metrics.regime_duration_distribution.get('std_duration', 0),
                    quality_metrics.regime_duration_distribution.get('min_duration', 0),
                    quality_metrics.regime_duration_distribution.get('max_duration', 0)
                ],
                'Interpretation': [
                    'Higher = more stable regimes',
                    'Higher = more stable (no penalty)',
                    'Lower = fewer rapid switches',
                    'Higher = longer lasting regimes',
                    'Average regime length in periods',
                    'Variability in regime duration',
                    'Shortest regime observed',
                    'Longest regime observed'
                ]
            }
            
            temporal_df = pd.DataFrame(temporal_data)
            temporal_csv_path = output_dir / f"temporal_analysis_{timestamp}.csv"
            temporal_df.to_csv(temporal_csv_path, index=False)
            
            # Save comprehensive metrics as artifact
            self._save_artifact(
                data=quality_metrics.to_dict(),
                artifact_name="comprehensive_quality_metrics",
                artifact_type="metadata"
            )
            
            tprint_success(f"✅ Detailed CSV reports saved to {output_dir}")
            tprint_info(f"   📄 Summary: {summary_csv_path.name}")
            if quality_metrics.per_regime_metrics:
                tprint_info(f"   📄 Regimes: {regime_csv_path.name}")
            if quality_metrics.economic_validation:
                tprint_info(f"   📄 Economic: {econ_csv_path.name}")
            tprint_info(f"   📄 Temporal: {temporal_csv_path.name}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save quality reports: {e}")
            self.logger.error(f"Failed to save quality reports: {e}")
    
    async def _save_results(
        self,
        results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Tuple[Any, Optional[Any]]:
        """
        Save clustering results to artifacts.
        
        Args:
            results: Clustering results dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
            
        Returns:
            Tuple of (labels_df, probabilities_df)
        """
        tprint("💾 Saving Sticky Finite HMM results to artifacts...", "INFO")
        
        # Reset context to current step
        self.artifact_manager.set_context(
            step_name=self.step_name,
            symbol=symbol,
            exchange=exchange,
            datetime=datetime.now(),
            information="regime_discovery"
        )
        
        # Save cluster labels with consistent naming (HDP-HMM compatible)
        cluster_labels_df = pd.DataFrame({
            'regime_label': results['cluster_labels']
        })
        # Primary naming (HDP-HMM compatible)
        self._save_artifact(
            data=cluster_labels_df,
            artifact_name="hdp_hmm_regime_labels",
            artifact_type="data"
        )
        # Secondary naming (sticky_finite_hmm compatibility)
        self._save_artifact(
            data=cluster_labels_df,
            artifact_name="sticky_finite_hmm_regime_labels",
            artifact_type="data"
        )
        
        # Save cluster probabilities
        cluster_probs_df = None
        if results.get('cluster_probabilities') is not None:
            cluster_probs_df = pd.DataFrame(results['cluster_probabilities'])
            # Primary naming (HDP-HMM compatible)
            self._save_artifact(
                data=cluster_probs_df,
                artifact_name="hdp_hmm_regime_probabilities",
                artifact_type="data"
            )
            # Secondary naming (sticky_finite_hmm compatibility)
            self._save_artifact(
                data=cluster_probs_df,
                artifact_name="sticky_finite_hmm_regime_probabilities",
                artifact_type="data"
            )
            tprint("✅ Saved regime probabilities (soft labels)", "SUCCESS")
        else:
            tprint_warning("⚠️ Cluster probabilities not available")
        
        # Save transition matrix
        if results.get('transition_matrix') is not None:
            transition_df = pd.DataFrame(results['transition_matrix'])
            # Primary naming (HDP-HMM compatible)
            self._save_artifact(
                data=transition_df,
                artifact_name="hdp_hmm_transition_matrix",
                artifact_type="data"
            )
            # Secondary naming (sticky_finite_hmm compatibility)
            self._save_artifact(
                data=transition_df,
                artifact_name="sticky_finite_hmm_transition_matrix",
                artifact_type="data"
            )
        
        # Save quality metrics
        self._save_artifact(
            data=results['quality_metrics'],
            artifact_name="hdp_hmm_quality_metrics",
            artifact_type="metadata"
        )
        
        # Save cluster statistics
        cluster_stats = {
            'n_regimes': results['n_clusters'],
            'regime_sizes': pd.Series(results['cluster_labels']).value_counts().to_dict(),
            'transition_persistence': results['quality_metrics'].get('transition_persistence', 0.0),
            'composite_score': results['quality_metrics'].get('composite_score', 0.0),
            'final_elbo': results.get('final_elbo', 0.0)
        }
        # Primary naming (HDP-HMM compatible)
        self._save_artifact(
            data=cluster_stats,
            artifact_name="hdp_hmm_cluster_statistics",
            artifact_type="metadata"
        )
        # Secondary naming (sticky_finite_hmm compatibility)
        self._save_artifact(
            data=cluster_stats,
            artifact_name="sticky_finite_hmm_cluster_statistics",
            artifact_type="metadata"
        )
        
        # Save feature names if available
        if 'feature_names' in results:
            self._save_artifact(
                data={'features': results['feature_names']},
                artifact_name="hdp_hmm_features_used",
                artifact_type="metadata"
            )
        
        # Save emission parameters
        if 'emission_params' in results:
            self._save_artifact(
                data=results['emission_params'],
                artifact_name="hdp_hmm_emission_params",
                artifact_type="metadata"
            )
        
        tprint(f"✅ Saved {results['n_clusters']} regime labels and metrics", "SUCCESS")
        return cluster_labels_df, cluster_probs_df
    
    async def _generate_reports(
        self,
        results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Generate comprehensive CSV exports and markdown reports.
        
        Exports include:
        - CV metrics (within/between clusters, CV ratio, per feature type)
        - Temporal smoothness (detailed calculation with flip-flop detection)
        - Quality metrics (silhouette, DBI, CH)
        - Regime statistics (durations, persistence, transitions)
        - Per-regime metrics (human-readable)
        
        Args:
            results: Clustering results dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
        """
        tprint("📊 Generating comprehensive CSV exports and markdown report...", "INFO")
        
        try:
            # Create outcomes directory
            outcomes_dir = Path("outcomes") / "sticky_finite_hmm_clustering" / symbol / exchange / timeframe
            outcomes_dir.mkdir(parents=True, exist_ok=True)
            tprint_info(f"   Output directory: {outcomes_dir}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tprint_info(f"   Timestamp: {timestamp}")
            
            # Extract quality assessment from results
            quality_metrics_dict = results.get('quality_metrics', {})
            quality_assessment = quality_metrics_dict.get('quality_assessment')
            
            if quality_assessment is None:
                tprint_warning("⚠️ No quality assessment available for report generation")
                return
            
            # Convert dict to ClusterQualityMetrics object if needed
            if isinstance(quality_assessment, dict):
                from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
                    ClusterQualityMetrics
                )
                # The quality_assessment dict already contains all the fields
                # We'll use it directly for CSV export and get the object for markdown
            
            # 1. Generate comprehensive all-in-one CSV (matches HDP-HMM format + enriched)
            tprint_info("   Exporting comprehensive metrics CSV...")
            self._export_comprehensive_all_metrics_csv(
                results, quality_assessment, outcomes_dir, timestamp
            )
            
            # 2. Generate ELBO history CSV (time series - unique to VB)
            tprint_info("   Exporting ELBO history CSV...")
            self._export_elbo_history_csv(
                results, outcomes_dir, timestamp
            )
            
            # 3. Generate transition matrix CSV (2D matrix)
            tprint_info("   Exporting transition matrix CSV...")
            self._export_transition_matrix_csv(
                results, outcomes_dir, timestamp
            )
            
            # 4. Generate enhanced markdown report with variance reduction details
            tprint_info("   Generating enhanced markdown report with variance reduction details...")

            self._generate_enhanced_markdown_report(
                results, quality_assessment, symbol, outcomes_dir, timestamp
            )
            
            # 5. Export variance reduction metrics CSV
            tprint_info("   Exporting variance reduction metrics CSV...")
            self._export_variance_reduction_metrics_csv(
                results, outcomes_dir, timestamp
            )
            
            tprint_success(f"✅ Enhanced reports generated in: {outcomes_dir}")
            tprint_success(f"   📊 Comprehensive metrics with variance reduction analysis")
            
        except Exception as e:
            tprint_warning(f"⚠️ Report generation failed: {e}")
            self.logger.error(f"Report generation error: {e}", exc_info=True)
    
    def _export_comprehensive_all_metrics_csv(
        self,
        results: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        output_dir: Path,
        timestamp: str
    ) -> None:
        """
        Export comprehensive quality metrics to CSV - ONE LARGE FILE with ALL metrics.
        
        Matches HDP-HMM format from hdp_hmm_isolated_tuning.py + enriched with:
        - All quality metrics (silhouette, DBI, CH, CV, temporal, balance, ELBO)
        - Temporal smoothness (raw + flip-flop penalized)
        - Flip-flop ratio, regime persistence
        - Duration distribution (mean, median, std, min, max)
        - Per-regime metrics as columns (size, duration, silhouette, returns, Sharpe)
        - Per-category CV ratios
        """
        csv_path = output_dir / f"sticky_finite_hmm_all_results_{timestamp}.csv"
        
        # Extract per-category CV metrics
        feature_category_cv = quality_assessment.get('feature_category_cv_metrics', {})
        per_regime_metrics = quality_assessment.get('per_regime_metrics', {})
        duration_dist = quality_assessment.get('regime_duration_distribution', {})
        
        # Build comprehensive metrics dictionary (matching HDP-HMM format + enrichments)
        metrics_data = {
            # ===== CORE METRICS (HDP-HMM format) =====
            'composite_score': [quality_assessment.get('quality_score', 0.0)],
            
            # Hyperparameters
            'K': [results.get('n_clusters', 5)],
            'base_alpha': [results.get('metadata', {}).get('config', {}).get('base_alpha', 0.5)],
            'kappa': [results.get('metadata', {}).get('config', {}).get('kappa', 10.0)],
            'num_iters': [results.get('metadata', {}).get('config', {}).get('num_iters', 800)],
            'lr': [results.get('metadata', {}).get('config', {}).get('lr', 1e-2)],
            
            # Clustering results
            'n_clusters': [results.get('n_clusters', 0)],
            'silhouette_score': [quality_assessment.get('silhouette_score', 0.0)],
            'davies_bouldin_score': [quality_assessment.get('davies_bouldin_score', 0.0)],
            'calinski_harabasz_score': [quality_assessment.get('calinski_harabasz_score', 0.0)],
            'temporal_smoothness': [quality_assessment.get('temporal_smoothness', 0.0)],
            'balance_score': [quality_assessment.get('balance_score', 0.0)],
            
            # ===== CV METRICS =====
            'between_regime_cv': [quality_assessment.get('between_regime_cv', 0.0)],
            'within_regime_cv': [quality_assessment.get('within_regime_cv', 0.0)],
            'within_regime_cv_std': [quality_assessment.get('within_regime_cv_std', 0.0)],
            'between_regime_cv_std': [quality_assessment.get('between_regime_cv_std', 0.0)],
            'cv_ratio': [
                quality_assessment.get('between_regime_cv', 0.0) / 
                max(quality_assessment.get('within_regime_cv', 1e-10), 1e-10)
            ],
            
            # Economic CV
            'economic_cv_ratio': [
                quality_assessment.get('economic_cv_metrics', {}).get('economic_cv_ratio_mean_return', 0.0)
            ],
            
            # ===== ENRICHED: TEMPORAL METRICS =====
            'temporal_smoothness_raw': [quality_assessment.get('temporal_smoothness_raw', 0.0)],
            'flip_flop_ratio': [quality_assessment.get('flip_flop_ratio', 0.0)],
            'regime_persistence_bars': [quality_assessment.get('regime_persistence', 0.0)],
            'transition_persistence': [results['quality_metrics'].get('transition_persistence', 0.0)],
            
            # ===== ENRICHED: DURATION DISTRIBUTION =====
            'duration_mean': [duration_dist.get('mean', 0.0)],
            'duration_median': [duration_dist.get('median', 0.0)],
            'duration_std': [duration_dist.get('std', 0.0)],
            'duration_min': [duration_dist.get('min', 0.0)],
            'duration_max': [duration_dist.get('max', 0.0)],
            
            # ===== ENRICHED: BALANCE METRICS =====
            'min_cluster_size_pct': [quality_assessment.get('min_cluster_size_pct', 0.0)],
            'max_cluster_size_pct': [quality_assessment.get('max_cluster_size_pct', 0.0)],
            'cluster_size_std': [quality_assessment.get('cluster_size_std', 0.0)],
            
            # Runtime and convergence
            'runtime': [results.get('processing_time', 0.0)],
            'memory_usage_mb': [results.get('memory_usage_mb', 0.0)],
            'converged': [results.get('metadata', {}).get('convergence_info', {}).get('converged', False)],
            'convergence_iteration': [results.get('metadata', {}).get('convergence_info', {}).get('final_iteration', 0)],
            'final_elbo': [results.get('final_elbo', 0.0)],
            
            # ===== PER-CATEGORY CV RATIOS (matching HDP-HMM) =====
            'cv_order_flow': [feature_category_cv.get('order_flow', {}).get('cv_ratio', 0.0)],
            'cv_microstructure': [feature_category_cv.get('microstructure', {}).get('cv_ratio', 0.0)],
            'cv_momentum': [feature_category_cv.get('momentum', {}).get('cv_ratio', 0.0)],
            'cv_volatility': [feature_category_cv.get('volatility', {}).get('cv_ratio', 0.0)],
            'cv_volume': [feature_category_cv.get('volume', {}).get('cv_ratio', 0.0)],
            'cv_trend': [feature_category_cv.get('trend', {}).get('cv_ratio', 0.0)],
            'cv_temporal': [feature_category_cv.get('temporal', {}).get('cv_ratio', 0.0)],
            
            # Status
            'success': [True],
            'error': ['']
        }
        
        # ===== ENRICHED: PER-REGIME METRICS AS COLUMNS =====
        for regime_id, regime_metrics in per_regime_metrics.items():
            prefix = f'regime_{regime_id}_'
            metrics_data[f'{prefix}size'] = [regime_metrics.get('size', 0)]
            metrics_data[f'{prefix}size_pct'] = [regime_metrics.get('size_pct', 0.0)]
            metrics_data[f'{prefix}duration_mean'] = [regime_metrics.get('duration_mean', 0.0)]
            metrics_data[f'{prefix}duration_std'] = [regime_metrics.get('duration_std', 0.0)]
            metrics_data[f'{prefix}silhouette_mean'] = [regime_metrics.get('silhouette_mean', 0.0)]
            metrics_data[f'{prefix}silhouette_std'] = [regime_metrics.get('silhouette_std', 0.0)]
            
            # Economic metrics if available (comprehensive economic validation)
            if 'mean_return' in regime_metrics:
                # Basic return metrics
                metrics_data[f'{prefix}mean_return'] = [regime_metrics.get('mean_return', 0.0)]
                metrics_data[f'{prefix}volatility'] = [regime_metrics.get('volatility', 0.0)]
                metrics_data[f'{prefix}sharpe'] = [regime_metrics.get('sharpe', 0.0)]
                metrics_data[f'{prefix}skewness'] = [regime_metrics.get('skewness', 0.0)]
                
                # Risk metrics
                metrics_data[f'{prefix}max_drawdown'] = [regime_metrics.get('max_drawdown', 0.0)]
                
                # Target-based metrics
                metrics_data[f'{prefix}pct_above_target'] = [regime_metrics.get('pct_above_target', 0.0)]
                metrics_data[f'{prefix}pct_below_neg_target'] = [regime_metrics.get('pct_below_neg_target', 0.0)]
                metrics_data[f'{prefix}pct_target_hits'] = [regime_metrics.get('pct_target_hits', 0.0)]
                
                # Advanced risk-adjusted metrics
                metrics_data[f'{prefix}risk_adj_target_hits'] = [regime_metrics.get('risk_adj_target_hits', 0.0)]
                metrics_data[f'{prefix}win_rate'] = [regime_metrics.get('win_rate', 0.0)]
                metrics_data[f'{prefix}return_per_vol'] = [regime_metrics.get('return_per_vol', 0.0)]
                metrics_data[f'{prefix}profit_factor'] = [regime_metrics.get('profit_factor', 0.0)]
        
        df = pd.DataFrame(metrics_data)
        
        # Reorder columns: core metrics first, then enrichments, then per-regime
        core_columns = [
            'composite_score', 'K', 'base_alpha', 'kappa', 'num_iters', 'lr', 'n_clusters',
            'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score',
            'temporal_smoothness', 'balance_score',
            'between_regime_cv', 'within_regime_cv', 'within_regime_cv_std', 'between_regime_cv_std', 'cv_ratio',
            'economic_cv_ratio',
            'temporal_smoothness_raw', 'flip_flop_ratio', 'regime_persistence_bars', 'transition_persistence',
            'duration_mean', 'duration_median', 'duration_std', 'duration_min', 'duration_max',
            'min_cluster_size_pct', 'max_cluster_size_pct', 'cluster_size_std',
            'runtime', 'memory_usage_mb', 'converged', 'convergence_iteration', 'final_elbo',
            'cv_order_flow', 'cv_microstructure', 'cv_momentum', 'cv_volatility',
            'cv_volume', 'cv_trend', 'cv_temporal',
        ]
        
        # Add per-regime columns
        regime_columns = [col for col in df.columns if col.startswith('regime_')]
        regime_columns.sort()  # Sort by regime ID
        
        # Final column order
        column_order = [col for col in core_columns if col in df.columns] + regime_columns + ['success', 'error']
        df = df[column_order]
        
        df.to_csv(csv_path, index=False)
        tprint_success(f"✅ Comprehensive all-in-one results saved: {csv_path.absolute()}")
        tprint_info(f"   📊 {len(df.columns)} columns including per-regime metrics")
    
    def _export_transition_matrix_csv(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        timestamp: str
    ) -> None:
        """Export transition matrix to CSV."""
        transition_matrix = results.get('transition_matrix')
        if transition_matrix is None:
            tprint_warning("⚠️ No transition matrix available")
            return
            
        K = transition_matrix.shape[0]
        regime_labels = [f'Regime_{i}' for i in range(K)]
        
        df = pd.DataFrame(
            transition_matrix,
            index=pd.Index(regime_labels),
            columns=pd.Index(regime_labels)
        )
        
        csv_path = output_dir / f"transition_matrix_{timestamp}.csv"
        df.to_csv(csv_path, index=True)
        tprint_success(f"✅ Transition matrix saved: {csv_path.absolute()}")
    
    def _generate_enhanced_markdown_report(
        self,
        results: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        symbol: str,
        output_dir: Path,
        timestamp: str
    ) -> Path:
        """
        Generate enhanced markdown report with variance reduction details.
        
        Args:
            results: Clustering results
            quality_assessment: Quality assessment dictionary
            symbol: Trading symbol
            output_dir: Output directory
            timestamp: Timestamp for filenames
            
        Returns:
            Path to generated markdown report
        """
        report_path = output_dir / f"enhanced_sticky_finite_hmm_report_{timestamp}.md"
        
        # Extract metrics
        quality_metrics = results.get('quality_metrics', {})
        convergence_info = results.get('metadata', {}).get('convergence_info', {})
        
        with open(report_path, 'w') as f:
            f.write(f"# Enhanced Sticky Finite HMM Regime Analysis Report\n\n")
            f.write(f"**Symbol:** {symbol}  \n")
            f.write(f"**Timestamp:** {timestamp}  \n")
            f.write(f"**Analysis Method:** Enhanced Sticky Finite HMM with Variance Reduction\n\n")
            
            f.write("## 🧠 Enhanced Features\n\n")
            f.write("This analysis utilizes advanced variance reduction techniques:\n\n")
            f.write("- **Structured Variational Inference:** Forward-backward message passing for exact marginals\n")
            f.write("- **Natural Gradient Updates:** Closed-form parameter updates in mean-parameter space\n")
            f.write("- **Rao-Blackwellization:** Zero Monte Carlo variance for sufficient statistics\n")
            f.write("- **Vectorized Computations:** Optimized NumPy operations for performance\n\n")
            
            f.write("## 📊 Model Configuration\n\n")
            config = results.get('metadata', {}).get('config', {})
            f.write(f"- **Number of States (K):** {config.get('K', 'N/A')}\n")
            f.write(f"- **Stickiness (κ):** {config.get('kappa', 'N/A')}\n")
            f.write(f"- **Base Concentration (α):** {config.get('base_alpha', 'N/A')}\n")
            f.write(f"- **Learning Rate:** {config.get('lr', 'N/A')}\n")
            f.write(f"- **Iterations:** {config.get('num_iters', 'N/A')}\n")
            f.write(f"- **PCA Components:** {config.get('pca_components', 'N/A')}\n\n")
            
            f.write("## 🎯 Quality Metrics\n\n")
            f.write(f"- **Composite Score:** {quality_metrics.get('composite_score', 0):.4f}\n")
            f.write(f"- **Transition Persistence:** {quality_metrics.get('transition_persistence', 0):.4f}\n")
            f.write(f"- **Final ELBO:** {results.get('final_elbo', 0):.2f}\n\n")
            
            if convergence_info:
                f.write("## 📈 Convergence Information\n\n")
                f.write(f"- **Converged:** {convergence_info.get('converged', False)}\n")
                f.write(f"- **Final Iteration:** {convergence_info.get('final_iteration', 'N/A')}\n")
                f.write(f"- **Best ELBO:** {convergence_info.get('best_elbo', 0):.2f}\n\n")
            
            f.write("## 📂 Generated Files\n\n")
            f.write(f"- `all_metrics_{timestamp}.csv` - Comprehensive metrics\n")
            f.write(f"- `elbo_history_{timestamp}.csv` - ELBO convergence trace\n")
            f.write(f"- `transition_matrix_{timestamp}.csv` - Learned transition probabilities\n")
            f.write(f"- `variance_reduction_metrics_{timestamp}.csv` - Variance reduction analysis\n\n")
            
            f.write("---\n")
            f.write("*Generated by Enhanced Sticky Finite HMM with variance reduction techniques*\n")
        
        tprint_info(f"   ✅ Enhanced markdown report: {report_path}")
        return report_path
    
    def _export_variance_reduction_metrics_csv(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        timestamp: str
    ) -> None:
        """
        Export variance reduction metrics analysis.
        
        Args:
            results: Clustering results
            output_dir: Output directory  
            timestamp: Timestamp for filename
        """
        convergence_info = results.get('metadata', {}).get('convergence_info', {})
        elbo_history = results.get('elbo_history', [])
        
        # Calculate variance reduction metrics
        variance_metrics = {
            'metric': [
                'final_elbo',
                'convergence_achieved', 
                'elbo_variance_final_10',
                'total_iterations',
                'variance_reduction_techniques'
            ],
            'value': [
                results.get('final_elbo', 0),
                str(convergence_info.get('converged', False)),
                np.var(elbo_history[-10:]) if len(elbo_history) >= 10 else 0,
                len(elbo_history),
                'structured_vi;natural_gradients;rao_blackwellization;vectorized'
            ]
        }
        
        variance_df = pd.DataFrame(variance_metrics)
        csv_path = output_dir / f"variance_reduction_metrics_{timestamp}.csv"
        variance_df.to_csv(csv_path, index=False)
        tprint_info(f"   ✅ Variance reduction metrics: {csv_path}")
    
    def _export_elbo_history_csv(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        timestamp: str
    ) -> None:
        """Export ELBO history to CSV (unique to VB models)."""
        csv_path = output_dir / f"elbo_history_{timestamp}.csv"
        
        elbo_history = results.get('elbo_history', [])
        if not elbo_history:
            tprint_warning("⚠️ No ELBO history available")
            return
        
        df = pd.DataFrame({
            'iteration': range(len(elbo_history)),
            'elbo': elbo_history
        })
        
        # Add moving average for trend visualization
        window = min(10, len(elbo_history) // 10)
        if window > 1:
            df['elbo_ma'] = df['elbo'].rolling(window=window, min_periods=1).mean()
        
        # Add improvement per iteration
        df['elbo_improvement'] = df['elbo'].diff()
        
        df.to_csv(csv_path, index=False)
        tprint_success(f"✅ ELBO history saved: {csv_path.absolute()}")
        
    def _generate_markdown_report(
        self,
        results: Dict[str, Any],  # <-- ADD 'results'
        quality_assessment: Dict[str, Any],
        symbol: str,
        output_dir: Path,
        hmm_params: Dict[str, Any]
    ) -> Optional[str]:
        """Generate comprehensive markdown report using quality assessor."""
        try:
            # Convert dict to ClusterQualityMetrics object
            from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
                ClusterQualityMetrics
            )
            
            metrics_obj = ClusterQualityMetrics(**quality_assessment)
            
            # Generate markdown report
            report_path = self.quality_assessor.generate_markdown_report(
                metrics=metrics_obj,
                symbol=f"{symbol}_StickyFiniteHMM",
                output_dir=str(output_dir),
                method_specific_config=hmm_params  # <-- 3. PASS TO ASSESSOR
            )
            
            if report_path:
                # --- START: FULFILLS REQUEST #5 ---
                try:
                    # Extract PCA loadings from the results metadata
                    pca_loadings = results.get('metadata', {}).get('pca_loadings')
                    
                    if pca_loadings:
                        # Convert to Path if string
                        if isinstance(report_path, str):
                            report_path = Path(report_path)
                        
                        # Create markdown string for PCA loadings
                        md_string = "\n\n---\n\n## PCA Component Analysis\n\n"
                        md_string += "Top 5 features contributing to each Principal Component:\n\n"
                        
                        for component, features in pca_loadings.items():
                            md_string += f"### {component.upper()}\n"
                            md_string += "| Feature | Loading |\n"
                            md_string += "|---|---|\n"
                            for feature, loading in features.items():
                                md_string += f"| `{feature}` | {loading:.4f} |\n"
                            md_string += "\n"
                        
                        # Append this string to the generated report
                        with open(report_path, "a", encoding="utf-8") as f:
                            f.write(md_string)
                        
                        tprint_success(f"✅ Appended PCA loadings to markdown report: {report_path.absolute()}")
                
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to append PCA loadings to report: {e}")
                # --- END: FULFILLS REQUEST #5 ---

            if report_path and not isinstance(report_path, Path):
                report_path = Path(report_path)

            tprint_success(f"✅ Markdown report generated: {report_path.absolute() if report_path else 'N/A'}")
            
            return str(report_path) if report_path else None
            
        except Exception as e:
            tprint_warning(f"⚠️ Markdown report generation failed: {e}")
            self.logger.error(f"Markdown report error: {e}", exc_info=True)
            return None
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['symbol', 'exchange']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            tprint_error(f"❌ Missing required configuration keys: {missing_keys}")
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate symbol
        symbol = config.get('symbol')
        if not symbol or not isinstance(symbol, str):
            tprint_error(f"❌ Invalid symbol: {symbol}")
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Validate exchange
        exchange = config.get('exchange')
        valid_exchanges = ['binance', 'bybit', 'okx', 'coinbase']
        if exchange not in valid_exchanges:
            tprint(
                f"⚠️ Exchange '{exchange}' not in validated list: {valid_exchanges}",
                "WARNING"
            )
    
    def _handle_execution_error(
        self,
        error: Exception,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle execution errors and return error result.
        
        Args:
            error: Exception that occurred
            config: Configuration dictionary
            
        Returns:
            Error result dictionary
        """
        error_msg = str(error)
        self.logger.error(f"Sticky Finite HMM execution failed: {error_msg}", exc_info=True)
        tprint(f"❌ Sticky Finite HMM Regime Discovery failed: {error_msg}", "ERROR")
        
        return {
            'success': False,
            'error': error_msg,
            'artifacts': {},
            'metrics': {},
            'execution_time': 0.0
        }


# Convenience function for direct step execution
async def run_sticky_finite_hmm_step(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run Sticky Finite HMM step directly.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Step execution results
        
    Example:
        ```python
        config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'regime_timeframe': '1h',
            'sticky_finite_hmm_params': {
                'K': 5,
                'base_alpha': 0.5,
                'kappa': 10.0,
                'num_iters': 800,
                'lr': 1e-2
            }
        }
        
        results = await run_sticky_finite_hmm_step(config)
        tprint_info(f"Success: {results['success']}")
        tprint_info(f"Regimes: {results.get('n_regimes', 0)}")
        ```
    """
    step = StickyFiniteHMMRegimeDiscoveryStep()
    return await step.execute(config)


__all__ = [
    'StickyFiniteHMMRegimeDiscoveryStep',
    'run_sticky_finite_hmm_step'
]


if __name__ == "__main__":
    """
    Direct execution interface for Sticky Finite HMM Regime Discovery Step.
    
    Usage:
        python sticky_finite_hmm_regime_discovery_step.py --symbol BTCUSDT --exchange binance --timeframe 1h
        
    This allows the step to be called directly without going through the launcher,
    automatically triggering the whole pipeline with BaseStep-standard data loading:
    1. Artifacts from previous steps (klines_downloading_processing, data_collection, etc.)
    2. Fallback to DataLoader for legacy support
    """
    import argparse
    import sys
    
    async def main():
        """Main function for direct execution."""
        parser = argparse.ArgumentParser(
            description="Sticky Finite HMM Regime Discovery Step - Direct Execution"
        )
        parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
        parser.add_argument("--exchange", default="binance", help="Exchange name (default: binance)")
        parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")
        parser.add_argument("--execution-mode", default="full", choices=["full", "light", "blank"],
                          help="Execution mode (default: full)")
        parser.add_argument("--enable-auto-tuning", action="store_true", default=True,
                          help="Enable hyperparameter auto-tuning (default: True)")
        parser.add_argument("--disable-auto-tuning", dest="enable_auto_tuning", action="store_false",
                          help="Disable hyperparameter auto-tuning")
        
        args = parser.parse_args()
        
        # Build configuration
        config = {
            'symbol': args.symbol,
            'exchange': args.exchange,
            'regime_timeframe': args.timeframe,
            'execution_mode': args.execution_mode,
            'enable_auto_tuning': args.enable_auto_tuning,
            'direction': 'long',
            'interaction_generation_mode': 'analyst'
        }
        
        tprint_info("🚀 Starting Sticky Finite HMM Regime Discovery Step (Direct Execution)")
        tprint_info(f"📊 Configuration: {args.symbol} {args.exchange} {args.timeframe} ({args.execution_mode} mode)")
        
        try:
            # Run the step
            results = await run_sticky_finite_hmm_step(config)
            
            # Report results
            if results.get('success', False):
                tprint_success("✅ Sticky Finite HMM Regime Discovery completed successfully!")
                
                # Print key metrics
                n_regimes = results.get('n_regimes', 0)
                composite_score = results.get('quality_metrics', {}).get('composite_score', 0)
                execution_time = results.get('execution_time', 0)
                
                tprint_info(f"📈 Results Summary:")
                tprint_info(f"   • Number of Regimes: {n_regimes}")
                tprint_info(f"   • Composite Quality Score: {composite_score:.4f}")
                tprint_info(f"   • Execution Time: {execution_time:.2f}s")
                
                # Report paths
                if 'artifacts' in results:
                    tprint_info(f"💾 Artifacts saved:")
                    for artifact_name, artifact_path in results['artifacts'].items():
                        tprint_info(f"   • {artifact_name}: {artifact_path}")
                
                sys.exit(0)
            else:
                tprint_error(f"❌ Sticky Finite HMM Regime Discovery failed: {results.get('error', 'Unknown error')}")
                sys.exit(1)
                
        except KeyboardInterrupt:
            tprint_warning("⚠️ Execution interrupted by user")
            sys.exit(130)
        except Exception as e:
            tprint_error(f"❌ Unexpected error: {e}")
            sys.exit(1)
    
    # Run the async main function
    asyncio.run(main())

