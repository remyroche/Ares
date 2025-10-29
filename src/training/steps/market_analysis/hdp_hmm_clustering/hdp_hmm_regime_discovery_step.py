"""
HDP-HMM Regime Discovery Step.

This step performs regime discovery using Hierarchical Dirichlet Process Hidden Markov Models
with integrated artifact management, quality assessment, and performance enhancements.

Inherits from BaseStep for standardized artifact management and autonomous execution.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import HDP-HMM components
from .hdp_hmm_clusterer import HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE
from .hdp_hmm_auto_tuner import HDPHMMAutoTuner, HDPHMMSearchSpace, run_hdp_hmm_auto_tuning
from .standalone_runner import run_hdp_hmm_clustering

# Import quality assessor
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor
)

logger = logging.getLogger(__name__)


class HDPHMMRegimeDiscoveryStep(BaseStep):
    """
    HDP-HMM Regime Discovery Step.
    
    Performs nonparametric Bayesian regime discovery using Sticky HDP-HMM
    with the following enhancements:
    - Hierarchical hyperparameter optimization (3-5x faster)
    - Unified vectorization manager (2-10x faster)
    - Hardware optimization (M1/M2, GPU support)
    - VectorBT integration (3-5x faster rolling ops)
    - Memory management (handles large datasets)
    
    Inherits from BaseStep to provide:
    - Standardized artifact management
    - Automatic context setting
    - Market data access by default
    - Consistent result saving
    """
    
    def __init__(self, step_name: str = "hdp_hmm_regime_discovery"):
        """
        Initialize the HDP-HMM regime discovery step.
        
        Args:
            step_name: Name for this step (used for artifact organization)
        """
        super().__init__(step_name)
        self.logger = system_logger.getChild('HDPHMMRegimeDiscovery')
        
        # Validate HMM library availability
        if not HMM_AVAILABLE:
            self.logger.error("HMM libraries (pyhsmm or ssm) not available")
            raise ImportError(
                "HMM libraries required for HDP-HMM clustering. "
                "Install with: pip install ssm-jax or pyhsmm"
            )
        
        # Initialize quality assessor
        self.quality_assessor = create_cluster_quality_assessor(
            artifact_manager=self.artifact_manager
        )
        
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HDP-HMM regime discovery.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'BTCUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Optional timeframe override (defaults to regime_timeframe)
                - regime_timeframe: Timeframe for regime detection (default: '1h')
                - execution_mode: 'full', 'light', or 'blank'
                - run_optimization: Whether to run HPO (default: False)
                - hdp_hmm_params: Optional HDP-HMM parameters override
                
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
            - 'execution_time': float seconds taken to execute
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
        
        # Use regime_timeframe (default: 1h or 60m) for regime detection
        regime_timeframe = config.get('regime_timeframe', '1h')
        timeframe = config.get('timeframe', regime_timeframe)
        
        # Override timeframe to regime_timeframe for consistency
        if timeframe != regime_timeframe:
            tprint(
                f"⏰ Using regime_timeframe={regime_timeframe} for HDP-HMM "
                f"(overriding timeframe={timeframe})", 
                "INFO"
            )
            timeframe = regime_timeframe
        
        tprint(
            f"🚀 Starting HDP-HMM Regime Discovery for {symbol} on {exchange} "
            f"(timeframe: {timeframe})",
            "INFO"
        )
        
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
            
            # Load market data
            tprint("📥 Loading market data...", "INFO")
            market_data = self._load_market_data(symbol, exchange, timeframe, config)
            
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {symbol} on {timeframe}")
            
            tprint(f"✅ Loaded {len(market_data)} samples of market data", "SUCCESS")
            
            # Check if optimization is requested
            run_optimization = config.get('run_optimization', False)
            
            if run_optimization:
                # Run hyperparameter optimization
                tprint("🎯 Running hyperparameter optimization...", "INFO")
                result = await self._run_optimization(
                    market_data, symbol, exchange, timeframe, config
                )
            else:
                # Run clustering with default or provided parameters
                tprint("🔍 Running HDP-HMM clustering...", "INFO")
                result = await self._run_clustering(
                    market_data, symbol, exchange, timeframe, config
                )
            
            # Save artifacts
            await self._save_results(result, symbol, exchange, timeframe, config)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            tprint(
                f"✅ HDP-HMM Regime Discovery completed in {execution_time:.2f}s",
                "SUCCESS"
            )
            
            return {
                'success': True,
                'artifacts': result.get('artifacts', {}),
                'metrics': result.get('quality_metrics', {}),
                'execution_time': execution_time,
                'n_regimes': result.get('n_clusters', 0),
                'composite_score': result.get('quality_metrics', {}).get('composite_score', 0.0)
            }
            
        except Exception as e:
            return self._handle_execution_error(e, config)
    
    def _load_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Load market data from artifacts using BaseStep's artifact manager.
        
        Looks for market data in the following order:
        1. Current step's artifacts
        2. data_collection step artifacts
        3. klines_downloading_processing step artifacts
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe for data
            config: Configuration dictionary
            
        Returns:
            DataFrame with market data or None if not found
        """
        # Try multiple artifact sources
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
        
        # No data found
        tprint(
            f"⚠️ Could not load market data for {symbol} from any source",
            "WARNING"
        )
        return None
    
    async def _run_clustering(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run HDP-HMM clustering with provided or default parameters.
        
        Args:
            market_data: Market data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
            
        Returns:
            Dictionary with clustering results
        """
        # Get HDP-HMM parameters
        hdp_params = config.get('hdp_hmm_params', {})
        
        # Extract parameters with defaults
        alpha = hdp_params.get('alpha', 3.0)
        kappa = hdp_params.get('kappa', 50.0)
        gamma = hdp_params.get('gamma', 3.0)
        n_iterations = hdp_params.get('n_iterations', 100)
        max_states = hdp_params.get('max_states', 20)
        
        # Feature selection parameters
        min_features = hdp_params.get('min_features', 50)
        max_features = hdp_params.get('max_features', 100)
        
        # PCA parameters
        enable_pca = hdp_params.get('enable_pca', True)
        pca_components = hdp_params.get('pca_components', 10)
        
        # Enhancement parameters (defaults to enabled)
        enable_vectorization = config.get('enable_vectorization', True)
        enable_hardware_optimization = config.get('enable_hardware_optimization', True)
        enable_memory_optimization = config.get('enable_memory_optimization', True)
        enable_vectorbt = config.get('enable_vectorbt', True)
        memory_budget_mb = config.get('memory_budget_mb', 2048.0)
        
        tprint(f"📊 HDP-HMM Parameters:", "INFO")
        tprint(f"  - alpha: {alpha}", "INFO")
        tprint(f"  - kappa: {kappa}", "INFO")
        tprint(f"  - gamma: {gamma}", "INFO")
        tprint(f"  - n_iterations: {n_iterations}", "INFO")
        tprint(f"  - max_states: {max_states}", "INFO")
        tprint(f"  - features: {min_features}-{max_features}", "INFO")
        tprint(f"  - enhancements: vectorization={enable_vectorization}, "
               f"hardware={enable_hardware_optimization}, "
               f"memory={enable_memory_optimization}, "
               f"vectorbt={enable_vectorbt}", "INFO")
        
        # Run clustering (async wrapper for sync function)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            run_hdp_hmm_clustering,
            market_data,
            symbol,
            exchange,
            timeframe,
            min_features,
            max_features,
            alpha,
            kappa,
            gamma,
            n_iterations,
            max_states,
            enable_pca,
            pca_components,
            False,  # save_results (we handle saving in this step)
            None,  # output_dir
            enable_vectorization,
            enable_hardware_optimization,
            enable_memory_optimization,
            enable_vectorbt,
            memory_budget_mb
        )
        
        return results
    
    async def _run_optimization(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run hierarchical hyperparameter optimization then cluster with best params.
        
        Args:
            market_data: Market data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
            
        Returns:
            Dictionary with clustering results using best parameters
        """
        # Get optimization parameters
        opt_params = config.get('optimization_params', {})
        
        tpe_trials = opt_params.get('tpe_trials', 50)
        timeout = opt_params.get('timeout', 3600)  # 1 hour default
        use_hierarchical = opt_params.get('use_hierarchical', True)
        
        tprint(f"🎯 Starting hyperparameter optimization:", "INFO")
        tprint(f"  - TPE trials: {tpe_trials}", "INFO")
        tprint(f"  - Timeout: {timeout}s", "INFO")
        tprint(f"  - Method: {'hierarchical (3-5x faster)' if use_hierarchical else 'standard'}", "INFO")
        
        # Run optimization (async wrapper for sync function)
        loop = asyncio.get_event_loop()
        best_params, best_score, tuning_results = await loop.run_in_executor(
            None,
            run_hdp_hmm_auto_tuning,
            market_data,
            symbol,
            exchange,
            timeframe,
            None,  # search_space (use default)
            3,  # coarse_grid_points
            3,  # fine_grid_points
            tpe_trials,
            timeout,
            False,  # save_results (we handle saving)
            use_hierarchical
        )
        
        tprint(f"✅ Optimization complete:", "SUCCESS")
        tprint(f"  - Best score: {best_score:.4f}", "SUCCESS")
        tprint(f"  - Total trials: {tuning_results.n_trials}", "SUCCESS")
        tprint(f"  - Time: {tuning_results.total_time:.2f}s", "SUCCESS")
        
        # Save optimization results
        self._save_artifact(
            data={'best_params': best_params, 'best_score': best_score},
            artifact_name="hdp_hmm_optimization_results",
            artifact_type="metadata"
        )
        
        # Now run clustering with best parameters
        tprint("🔍 Running clustering with optimized parameters...", "INFO")
        
        # Update config with best parameters
        config_with_best = config.copy()
        config_with_best['hdp_hmm_params'] = best_params
        
        results = await self._run_clustering(
            market_data, symbol, exchange, timeframe, config_with_best
        )
        
        # Add optimization info to results
        results['optimization'] = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': tuning_results.n_trials,
            'optimization_time': tuning_results.total_time
        }
        
        return results
    
    async def _save_results(
        self,
        results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Save clustering results to artifacts.
        
        Args:
            results: Clustering results dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            config: Configuration dictionary
        """
        tprint("💾 Saving HDP-HMM results to artifacts...", "INFO")
        
        # Reset context to current step
        self.artifact_manager.set_context(
            step_name=self.step_name,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            datetime=datetime.now(),
            information="regime_discovery"
        )
        
        # Save cluster labels
        cluster_labels_df = pd.DataFrame({
            'regime_label': results['cluster_labels']
        })
        self._save_artifact(
            data=cluster_labels_df,
            artifact_name="hdp_hmm_regime_labels",
            artifact_type="data"
        )
        
        # Save transition matrix
        if results.get('transition_matrix') is not None:
            transition_df = pd.DataFrame(results['transition_matrix'])
            self._save_artifact(
                data=transition_df,
                artifact_name="hdp_hmm_transition_matrix",
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
            'composite_score': results['quality_metrics'].get('composite_score', 0.0)
        }
        self._save_artifact(
            data=cluster_stats,
            artifact_name="hdp_hmm_cluster_statistics",
            artifact_type="metadata"
        )
        
        # Save feature names if available
        if 'feature_names' in results:
            self._save_artifact(
                data={'features': results['feature_names']},
                artifact_name="hdp_hmm_features_used",
                artifact_type="metadata"
            )
        
        tprint(f"✅ Saved {results['n_clusters']} regime labels and metrics", "SUCCESS")
    
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
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate symbol
        symbol = config.get('symbol')
        if not symbol or not isinstance(symbol, str):
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
        self.logger.error(f"HDP-HMM execution failed: {error_msg}", exc_info=True)
        tprint(f"❌ HDP-HMM Regime Discovery failed: {error_msg}", "ERROR")
        
        return {
            'success': False,
            'error': error_msg,
            'artifacts': {},
            'metrics': {},
            'execution_time': 0.0
        }


# Convenience function for direct step execution
async def run_hdp_hmm_step(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run HDP-HMM step directly.
    
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
            'run_optimization': False,
            'hdp_hmm_params': {
                'alpha': 3.0,
                'kappa': 50.0,
                'n_iterations': 100
            }
        }
        
        results = await run_hdp_hmm_step(config)
        tprint_info(f"Success: {results['success']}")
        tprint_info(f"Regimes: {results.get('n_regimes', 0)}")
        ```
    """
    step = HDPHMMRegimeDiscoveryStep()
    return await step.execute(config)


__all__ = [
    'HDPHMMRegimeDiscoveryStep',
    'run_hdp_hmm_step'
]
