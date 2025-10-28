"""
MS-DR Clustering Step

BaseClass-based step for Markov-Switching Dynamic Regression clustering.
Integrates with ares_launcher.py and uses artifact_manager for data persistence.

Features:
- Automated market data loading with configurable timeframes (default: 60m/1h)
- Regime-switching model clustering
- Artifact management for results persistence
- Hierarchical hyperparameter optimization support
- Memory and hardware optimization
- Comprehensive quality assessment
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Import BaseClass and artifact management
from src.training.steps.base_step import BaseStep

# Import MS-DR clustering components
from src.training.steps.market_analysis.ms_dr_clustering import (
    MSDRClusterer,
    MSDRConfig,
    MSDRResult,
    MSDRAutoTuner,
    MSDRTuningConfig,
    MS_AVAILABLE
)

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured
)
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.serialization_utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class MSDRClusteringStep(BaseStep):
    """
    MS-DR (Markov-Switching Dynamic Regression) clustering step for regime discovery.
    
    Features:
    - Markov-Switching models with regime-dependent dynamics
    - Automatic regime selection using information criteria
    - Hierarchical hyperparameter optimization (50-70% faster)
    - Memory optimization and hardware acceleration
    - Safe mathematical operations
    - VectorBT acceleration support
    - Comprehensive quality assessment
    - Artifact management for persistence
    
    Configuration:
        - symbol: Trading symbol (e.g., 'ETHUSDT')
        - exchange: Exchange name (e.g., 'binance')
        - timeframe: Timeframe (default: '60m', alternatives: '1h', '15m', '5m')
        - enable_hyperparameter_optimization: Auto-tune parameters (default: False)
        - use_hierarchical_optimization: Use hierarchical HPO (default: True)
        - execution_mode: 'full', 'light', or 'blank' (default: 'light')
    """
    
    def __init__(self, step_name: str = "ms_dr_clustering"):
        """Initialize the MS-DR clustering step."""
        super().__init__(step_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not MS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for MS-DR clustering. "
                "Install with: pip install statsmodels>=0.13.0"
            )
        
        # Initialize clusterer and tuner
        self.clusterer = None
        self.tuner = None
        self.config = None
        
        tprint("🚀 MS-DR Clustering Step initialized", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MS-DR clustering step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (default: '60m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - execution_mode: 'full', 'light', or 'blank' (default: 'light')
                - enable_hyperparameter_optimization: Enable HPO (default: False)
                - use_hierarchical_optimization: Use hierarchical HPO (default: True)
                - live_mode: Whether this is live trading (default: False)
                
        Returns:
            Dictionary with execution results, artifacts, and metrics
        """
        start_time = datetime.now()
        artifacts_created = []
        
        try:
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '60m')  # Default to 60m (1h)
            
            tprint(f"🔍 Starting MS-DR clustering for {symbol} ({timeframe})", "INFO")
            
            # Validate required parameters
            self._validate_config(config)
            
            # Create MS-DR configuration
            self.config = self._create_msdr_config(config)
            
            # Load market data with default timeframe
            market_data = await self._load_market_data(config)
            if market_data is None or len(market_data) == 0:
                raise ValueError("Failed to load market data")
            
            tprint(f"✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            
            # Execute clustering (with or without hyperparameter optimization)
            if config.get('enable_hyperparameter_optimization', False):
                msdr_result, best_params = await self._execute_with_optimization(
                    market_data, config
                )
            else:
                msdr_result = await self._execute_clustering(market_data)
                best_params = None
            
            if not msdr_result.success:
                raise RuntimeError(f"MS-DR clustering failed: {msdr_result.error_message}")
            
            tprint(f"✅ Clustering successful: {msdr_result.n_clusters} regimes discovered", "SUCCESS")
            
            # Save artifacts using artifact manager
            artifacts_created = await self._save_clustering_artifacts(
                msdr_result, market_data, best_params, config
            )
            
            # Generate metrics
            metrics = self._generate_metrics(msdr_result, market_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            tprint_success(f"🎉 MS-DR Clustering Step completed in {execution_time:.2f}s")
            
            return {
                'success': True,
                'artifacts': artifacts_created,
                'metrics': metrics,
                'execution_time': execution_time,
                'n_regimes': msdr_result.n_clusters,
                'best_params': best_params
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"MS-DR clustering failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'error': error_msg,
                'artifacts': artifacts_created,
                'execution_time': execution_time
            }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_params = ['symbol', 'exchange']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Set default timeframe if not provided
        if 'timeframe' not in config:
            config['timeframe'] = '60m'  # Default to 60m (1 hour)
            tprint_info(f"📊 Using default timeframe: 60m (1h)")
    
    def _create_msdr_config(self, config: Dict[str, Any]) -> MSDRConfig:
        """Create MS-DR configuration from step config."""
        execution_mode = config.get('execution_mode', 'light')
        
        # Base configuration
        base_config = {
            'random_state': config.get('random_state', 42),
            'use_safe_math': True,
            'use_memory_optimization': True,
            'use_hardware_acceleration': True,
            'use_vectorbt_operations': True,
            'use_parallel_selection': True
        }
        
        if execution_mode == 'full':
            # Full mode: comprehensive regime discovery
            return MSDRConfig(
                n_regimes=5,
                model_type='autoregression',
                order=2,
                switching_variance=True,
                auto_select_regimes=True,
                min_regimes=2,
                max_regimes=10,
                ic_criterion='bic',
                enable_pca=True,
                pca_components=15,
                pca_variance_threshold=0.98,
                min_samples_required=200,
                show_progress=True,
                **base_config
            )
        
        elif execution_mode == 'blank':
            # Blank mode: minimal processing
            return MSDRConfig(
                n_regimes=3,
                model_type='autoregression',
                order=1,
                switching_variance=True,
                auto_select_regimes=True,
                min_regimes=2,
                max_regimes=5,
                ic_criterion='aic',
                enable_pca=True,
                pca_components=8,
                pca_variance_threshold=0.90,
                min_samples_required=100,
                show_progress=False,
                **base_config
            )
        
        else:  # light mode (default)
            # Light mode: balanced performance
            return MSDRConfig(
                n_regimes=4,
                model_type='autoregression',
                order=1,
                switching_variance=True,
                auto_select_regimes=True,
                min_regimes=2,
                max_regimes=8,
                ic_criterion='aic',
                enable_pca=True,
                pca_components=10,
                pca_variance_threshold=0.95,
                min_samples_required=150,
                show_progress=True,
                **base_config
            )
    
    async def _load_market_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load market data using klines manager with default 60m/1h timeframe."""
        try:
            tprint("📂 Loading market data...", "INFO")
            
            # Get klines manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))
            
            # Set default timeframe if not specified
            timeframe = config.get('timeframe', '60m')
            if timeframe == '1h':
                timeframe = '60m'  # Normalize 1h to 60m
            
            tprint_info(f"📊 Loading data with timeframe: {timeframe}")
            
            # Parse date filters based on execution_mode
            start_date = None
            end_date = None
            
            execution_mode = config.get('execution_mode', 'light')
            if execution_mode == 'light' and not ('start_date' in config or 'end_date' in config):
                # Light mode: use last 30 days
                end_date = pd.Timestamp.now(tz='UTC').normalize()
                start_date = end_date - pd.Timedelta(days=30)
                tprint(f"📅 Light mode: Auto-filtering to last 30 days ({start_date.date()} to {end_date.date()})", "INFO")
            elif execution_mode == 'blank' and not ('start_date' in config or 'end_date' in config):
                # Blank mode: use last 90 days
                end_date = pd.Timestamp.now(tz='UTC').normalize()
                start_date = end_date - pd.Timedelta(days=90)
                tprint(f"📅 Blank mode: Auto-filtering to last 90 days ({start_date.date()} to {end_date.date()})", "INFO")
            # Full mode: no automatic filtering
            
            # Override with explicit dates if provided
            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
                tprint(f"📅 Using start_date filter: {start_date}", "INFO")
            
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])
                tprint(f"📅 Using end_date filter: {end_date}", "INFO")
            
            # Load data
            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=timeframe,
                data_type="processed",
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data is None or len(market_data) == 0:
                tprint_error("❌ No market data returned from klines manager")
                return None
            
            # Ensure index is datetime
            if not isinstance(market_data.index, pd.DatetimeIndex):
                if 'open_time' in market_data.columns:
                    market_data = market_data.set_index('open_time')
                elif 'timestamp' in market_data.columns:
                    market_data = market_data.set_index('timestamp')
            
            return market_data
            
        except Exception as e:
            tprint_error(f"❌ Failed to load market data: {e}")
            self.logger.error(f"Market data loading error: {e}", exc_info=True)
            return None
    
    async def _execute_clustering(self, market_data: pd.DataFrame) -> MSDRResult:
        """Execute MS-DR clustering without hyperparameter optimization."""
        tprint_info("🔄 Running MS-DR clustering...")
        
        # Initialize clusterer
        self.clusterer = MSDRClusterer(self.config)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.clusterer.fit_predict,
            market_data.values
        )
        
        return result
    
    async def _execute_with_optimization(
        self, 
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> tuple[MSDRResult, Dict[str, Any]]:
        """Execute MS-DR clustering with hyperparameter optimization."""
        tprint_info("🎯 Running MS-DR clustering with hyperparameter optimization...")
        
        # Create tuning configuration
        use_hierarchical = config.get('use_hierarchical_optimization', True)
        
        tuning_config = MSDRTuningConfig(
            n_trials=config.get('n_trials', 50),
            timeout_minutes=config.get('timeout_minutes', 30.0),
            use_hierarchical=use_hierarchical,
            n_trials_per_group=config.get('n_trials_per_group', 20),
            random_state=config.get('random_state', 42)
        )
        
        # Initialize tuner
        self.tuner = MSDRAutoTuner(tuning_config)
        
        # Run optimization in executor
        loop = asyncio.get_event_loop()
        
        if use_hierarchical:
            tprint_info("⚡ Using hierarchical optimization (50-70% faster!)")
            optimization_result = await loop.run_in_executor(
                None,
                self.tuner.auto_tune_hierarchical,
                market_data
            )
        else:
            tprint_info("📊 Using standard optimization")
            optimization_result = await loop.run_in_executor(
                None,
                self.tuner.auto_tune,
                market_data
            )
        
        best_params = optimization_result['best_params']
        
        # Create clusterer with best params
        best_config = MSDRConfig(**best_params)
        best_config.use_safe_math = self.config.use_safe_math
        best_config.use_memory_optimization = self.config.use_memory_optimization
        best_config.use_hardware_acceleration = self.config.use_hardware_acceleration
        
        self.clusterer = MSDRClusterer(best_config)
        
        # Run final clustering with best params
        result = await loop.run_in_executor(
            None,
            self.clusterer.fit_predict,
            market_data.values
        )
        
        return result, best_params
    
    async def _save_clustering_artifacts(
        self,
        msdr_result: MSDRResult,
        market_data: pd.DataFrame,
        best_params: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[str]:
        """Save clustering artifacts using artifact manager."""
        artifacts = []
        
        try:
            # Save regime labels
            regime_labels_df = pd.DataFrame({
                'timestamp': market_data.index,
                'regime_label': msdr_result.cluster_labels,
                'n_regimes': msdr_result.n_clusters
            })
            artifact_path = self._save_artifact(
                data=regime_labels_df,
                artifact_name="ms_dr_regime_labels",
                artifact_type="data",
                compression="auto",
                metadata={
                    'symbol': config.get('symbol'),
                    'timeframe': config.get('timeframe', '60m'),
                    'n_regimes': msdr_result.n_clusters,
                    'silhouette_score': msdr_result.silhouette_score,
                    'aic': msdr_result.aic,
                    'bic': msdr_result.bic
                }
            )
            artifacts.append(artifact_path)
            
            # Save regime probabilities
            regime_probs_df = pd.DataFrame(
                msdr_result.cluster_probabilities,
                index=market_data.index,
                columns=[f'regime_{i}_prob' for i in range(msdr_result.n_clusters)]
            )
            artifact_path = self._save_artifact(
                data=regime_probs_df,
                artifact_name="ms_dr_regime_probabilities",
                artifact_type="data",
                compression="auto"
            )
            artifacts.append(artifact_path)
            
            # Save transition matrix
            if msdr_result.transition_matrix is not None:
                transition_df = pd.DataFrame(
                    msdr_result.transition_matrix,
                    index=[f'from_regime_{i}' for i in range(msdr_result.n_clusters)],
                    columns=[f'to_regime_{i}' for i in range(msdr_result.n_clusters)]
                )
                artifact_path = self._save_artifact(
                    data=transition_df,
                    artifact_name="ms_dr_transition_matrix",
                    artifact_type="data"
                )
                artifacts.append(artifact_path)
            
            # Save comprehensive results
            results_dict = {
                'n_regimes': msdr_result.n_clusters,
                'regime_labels': msdr_result.cluster_labels.tolist(),
                'transition_matrix': msdr_result.transition_matrix.tolist() if msdr_result.transition_matrix is not None else None,
                'regime_params': msdr_result.regime_params,
                'regime_variances': msdr_result.regime_variances.tolist() if msdr_result.regime_variances is not None else None,
                'regime_durations': msdr_result.regime_durations.tolist() if msdr_result.regime_durations is not None else None,
                'transition_persistence': msdr_result.transition_persistence,
                'metrics': {
                    'silhouette_score': msdr_result.silhouette_score,
                    'calinski_harabasz_score': msdr_result.calinski_harabasz_score,
                    'davies_bouldin_score': msdr_result.davies_bouldin_score,
                    'aic': msdr_result.aic,
                    'bic': msdr_result.bic,
                    'hqic': msdr_result.hqic,
                    'log_likelihood': msdr_result.log_likelihood
                },
                'performance': {
                    'processing_time': msdr_result.processing_time,
                    'memory_usage_mb': msdr_result.memory_usage_mb
                },
                'best_params': best_params,
                'metadata': msdr_result.metadata
            }
            
            artifact_path = self._save_artifact(
                data=results_dict,
                artifact_name="ms_dr_clustering_results",
                artifact_type="metadata",
                metadata={
                    'symbol': config.get('symbol'),
                    'timeframe': config.get('timeframe', '60m'),
                    'execution_mode': config.get('execution_mode', 'light')
                }
            )
            artifacts.append(artifact_path)
            
            tprint_success(f"✅ Saved {len(artifacts)} artifacts")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save some artifacts: {e}")
            self.logger.error(f"Artifact saving error: {e}", exc_info=True)
        
        return artifacts
    
    def _generate_metrics(
        self,
        msdr_result: MSDRResult,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive metrics from MS-DR results."""
        metrics = {
            'n_regimes': msdr_result.n_clusters,
            'noise_ratio': msdr_result.noise_ratio,
            'processing_time_seconds': msdr_result.processing_time,
            'memory_usage_mb': msdr_result.memory_usage_mb,
            
            # Quality metrics
            'silhouette_score': msdr_result.silhouette_score,
            'calinski_harabasz_score': msdr_result.calinski_harabasz_score,
            'davies_bouldin_score': msdr_result.davies_bouldin_score,
            
            # Model selection metrics
            'aic': msdr_result.aic,
            'bic': msdr_result.bic,
            'hqic': msdr_result.hqic,
            'log_likelihood': msdr_result.log_likelihood,
            
            # Transition metrics
            'transition_persistence': msdr_result.transition_persistence,
            'avg_regime_duration': float(np.mean(msdr_result.regime_durations)) if msdr_result.regime_durations is not None else None,
            
            # Data metrics
            'n_samples': len(market_data),
            'n_features': len(msdr_result.feature_names),
            'feature_names': msdr_result.feature_names
        }
        
        return metrics


# Register with step registry (if available)
try:
    from src.training.steps.base_step import step_registry
    step_registry.register("ms_dr_clustering", MSDRClusteringStep)
except ImportError:
    pass


__all__ = [
    'MSDRClusteringStep'
]
