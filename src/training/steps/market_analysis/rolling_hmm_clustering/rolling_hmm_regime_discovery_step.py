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

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
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
    EWMAConfig,
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
        super().__init__(step_name)
        self.logger = system_logger.getChild('RollingHMMRegimeDiscovery')

        # Quality assessor will be created lazily when first accessed
        self._quality_assessor = None

        # Hardware manager
        self.hardware_manager = None

        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    @property
    def quality_assessor(self) -> ClusterQualityAssessor:
        """Lazy property for quality assessor."""
        tprint_debug("🔍 Accessing quality assessor instance")
        if self._quality_assessor is None:
            tprint_info("  → Initializing ClusterQualityAssessor")
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
                f"⏰ Using regime_timeframe={regime_timeframe} for Rolling HMM "
                f"(overriding timeframe={timeframe})",
                "INFO"
            )
            timeframe = regime_timeframe
        
        tprint(
            f"🚀 Starting Rolling HMM Regime Discovery for {symbol} on {exchange} "
            f"(timeframe: {timeframe})",
            "INFO"
        )
        tprint("🧠 Enhanced Features:", "INFO")
        tprint("   - EWMA-based rolling features (8+16, 8+20, 8+24, 12+16, 12+20, 12+24)", "INFO")
        tprint("   - Returns, volatility, trend, and volume features", "INFO")
        tprint("   - PCA dimensionality reduction (3-5 components)", "INFO")
        tprint("   - Sticky HMM with diagonal covariance and regularization", "INFO")
        tprint("   - VectorBT and hardware optimization for M1", "INFO")
        
        try:
            # Initialize hardware optimization
            self._initialize_hardware_optimization()
            
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
                tprint_error(f"❌ No market data available for {symbol} on {timeframe}")
                raise ValueError(f"No market data available for {symbol} on {timeframe}")
            
            tprint(f"✅ Loaded {len(market_data)} samples of market data", "SUCCESS")
            
            # Check execution mode and if HPO is enabled
            execution_mode = config.get('execution_mode', 'full')
            enable_auto_tuning = config.get('enable_auto_tuning', True)
            hpo_results: Optional[Dict[str, Any]] = None
            best_params: Optional[Dict[str, Any]] = None
            
            # Apply execution mode data limits (blank=20d, light=20d, full=all)
            market_data = self._apply_execution_mode_filter(market_data, execution_mode, timeframe)
            tprint(f"   → After execution mode filter ({execution_mode}): {len(market_data)} samples")
            
            # Initialize feature engineer
            feature_config = self._get_feature_config(config)
            feature_engineer = RollingHMMFeatureEngineer(feature_config)
            
            # Pre-compute ALL features for ALL EWMA windows ONCE (cached for HPO)
            if enable_auto_tuning:
                tprint("")
                tprint("🔄 Pre-computing features for HPO efficiency")
                all_cached_features = feature_engineer.precompute_all_features(market_data)
                tprint(f"✅ Cached features for {len(all_cached_features)} EWMA configurations")
                tprint("")
            
            # Only skip if user provided manual params
            if 'rolling_hmm_params' in config and config['rolling_hmm_params']:
                if 'enable_auto_tuning' not in config:
                    enable_auto_tuning = False
                    tprint_info("ℹ️  Manual params provided, skipping HPO (set enable_auto_tuning=True to override)")
            
            # Show execution mode info
            if enable_auto_tuning and execution_mode in ['light', 'blank']:
                tprint_info(f"ℹ️  HPO enabled in '{execution_mode}' mode (will use reduced trials for speed)")
            
            if enable_auto_tuning:
                tprint("", "INFO")
                tprint("🎯 HPO Enabled - Finding Optimal Hyperparameters", "INFO")
                tprint("=" * 80, "INFO")

                # Run HPO synchronously
                hpo_results, best_params = await self._run_hpo(
                    market_data, feature_engineer, symbol, exchange, timeframe, config
                )

                # Update config with best parameters
                if hpo_results and best_params:
                    tprint("", "INFO")
                    tprint("✅ HPO Complete - Using Optimal Parameters", "SUCCESS")
                    self._log_best_params(best_params)
                    if 'rolling_hmm_params' not in config:
                        config['rolling_hmm_params'] = {}
                    config['rolling_hmm_params'].update(best_params)
                else:
                    tprint_warning("⚠️  HPO did not complete, using default parameters")
                
                tprint("=" * 80, "INFO")
                tprint("", "INFO")
            
            # Run clustering
            tprint("🔍 Running Rolling HMM clustering...", "INFO")
            result = await self._run_clustering(
                market_data, feature_engineer, symbol, exchange, timeframe, config
            )

            # Save results
            labels_df, probs_df = await self._save_results(result, symbol, exchange, timeframe, config)

            # Generate reports
            await self._generate_reports(result, symbol, exchange, timeframe, config)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            tprint("", "SUCCESS")
            tprint(f"✅ Rolling HMM Regime Discovery completed in {execution_time:.2f}s", "SUCCESS")
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
        """Load market data from config or artifacts, restoring step context afterwards."""
        if 'market_data' in config and config['market_data'] is not None:
            external_data = config['market_data']
            tprint(f"✅ Using market data from config ({len(external_data)} samples)", "SUCCESS")
            return external_data

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

                    if market_data is not None and not market_data.empty:
                        tprint(
                            f"✅ Loaded market data from {step_name}/{artifact_name}",
                            "SUCCESS",
                        )
                        return market_data
                except Exception as load_error:
                    self.logger.debug(
                        f"Could not load market data from {step_name}/{artifact_name}: {load_error}"
                    )
        finally:
            self.artifact_manager.set_context(**original_context)

        tprint(
            f"⚠️ Could not load market data for {symbol} on {timeframe} from artifacts",
            "WARNING",
        )

        raise ValueError(
            "Market data not available via artifact manager. "
            "Run the data collection or klines processing steps before rolling HMM discovery."
        )

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization for M1."""
        tprint_info("⚡ Initializing hardware optimization for M1")

        self.hardware_manager = get_unified_hardware_manager()
        self.hardware_manager.optimize_for_workload(
            WorkloadType.ML_TRAINING,
            OptimizationLevel.BALANCED
        )

    def _get_feature_config(self, config: Dict[str, Any]) -> FeatureEngineeringConfig:
        """Get feature engineering configuration."""
        tprint_debug("Fetching feature engineering configuration")
        feature_config = config.get('feature_config', {})

        return FeatureEngineeringConfig(
            ewma_configs=DEFAULT_EWMA_CONFIGS,
            use_log_returns=feature_config.get('use_log_returns', True),
            use_volatility_features=feature_config.get('use_volatility_features', True),
            use_trend_features=feature_config.get('use_trend_features', True),
            use_volume_features=feature_config.get('use_volume_features', True),
            pca_components=feature_config.get('pca_components', 4),
            normalize_method=feature_config.get('normalize_method', 'zscore'),
            rolling_normalize_window=feature_config.get('rolling_normalize_window', 100),
            enable_vectorbt_optimization=feature_config.get('enable_vectorbt_optimization', True),
            enable_hardware_optimization=feature_config.get('enable_hardware_optimization', True),
            enable_numba_jit=feature_config.get('enable_numba_jit', True)
        )

    def _get_hpo_config(self, config: Dict[str, Any]) -> HPOConfig:
        """Get HPO configuration."""
        tprint_debug("Fetching HPO configuration with execution mode adjustments")
        hpo_config = config.get('hpo_config', {})
        execution_mode = config.get('execution_mode', 'full')

        # Adjust HPO config based on execution mode
        if execution_mode == 'light':
            hpo_config['final_refinement_trials'] = hpo_config.get('final_refinement_trials', 20)
            hpo_config['cv_folds'] = hpo_config.get('cv_folds', 3)
        elif execution_mode == 'blank':
            hpo_config['final_refinement_trials'] = hpo_config.get('final_refinement_trials', 5)
            hpo_config['cv_folds'] = hpo_config.get('cv_folds', 2)
        else:  # full
            hpo_config['final_refinement_trials'] = hpo_config.get('final_refinement_trials', 50)
            hpo_config['cv_folds'] = hpo_config.get('cv_folds', 5)

        return HPOConfig(
            stages=hpo_config.get('stages', DEFAULT_HPO_CONFIG.stages),
            n_rounds=hpo_config.get('n_rounds', DEFAULT_HPO_CONFIG.n_rounds),
            enable_final_refinement=hpo_config.get('enable_final_refinement', True),
            final_refinement_trials=hpo_config['final_refinement_trials'],
            cv_folds=hpo_config['cv_folds'],
            weight_predictive_ll=hpo_config.get('weight_predictive_ll', 0.33),
            weight_temporal=hpo_config.get('weight_temporal', 0.33),
            weight_economic=hpo_config.get('weight_economic', 0.34),
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

            # Create optimizer
            optimizer = RollingHMMOptimizer(hpo_config)

            # Run optimization
            result = optimizer.optimize(
                market_data,
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
                tprint_warning("⚠️  HPO returned no results")
                return None, None

        except Exception as e:
            tprint_error(f"❌ HPO failed: {e}")
            self.logger.error(f"HPO failed: {e}", exc_info=True)
            return None, None

    async def _run_clustering(
        self,
        market_data: pd.DataFrame,
        feature_engineer: RollingHMMFeatureEngineer,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Rolling HMM clustering."""
        try:
            # Get parameters
            params = config.get('rolling_hmm_params', {})

            # Extract EWMA config
            ewma_config_idx = params.get('ewma_config_idx', 0)
            ewma_config = DEFAULT_EWMA_CONFIGS[int(ewma_config_idx)]

            tprint_info(f"  → Using EWMA config: {ewma_config.name}")

            # Generate features
            features = feature_engineer.generate_features(market_data, ewma_config)

            if len(features) < 50:
                raise ValueError(f"Insufficient data after feature engineering: {len(features)} samples")

            # Apply PCA
            pca_components = params.get('pca_components', 4)
            features_pca, pca_model, explained_var = feature_engineer.apply_pca(
                features,
                n_components=int(pca_components)
            )

            # Create HMM config
            n_components = params.get('n_components', 5)
            min_covar = params.get('min_covar', 1e-3)
            kappa = params.get('kappa', 10.0)

            hmm_config = StickyHMMConfig(
                n_components=int(n_components),
                min_covar=float(min_covar),
                kappa=float(kappa),
                n_iter=params.get('n_iter', 200),
                tol=params.get('tol', 1e-4),
                covariance_type='diag',
                kmeans_init=True,
                use_sticky_priors=True,
                post_fit_regularization=True,
                random_state=params.get('random_state', 42)
            )

            tprint_info(f"  → HMM config: n_components={n_components}, kappa={kappa}, min_covar={min_covar}")

            # Fit HMM model
            hmm_model = StickyHMMModel(hmm_config)
            hmm_model.fit(features_pca.values)

            # Predict regime labels
            regime_labels = hmm_model.predict(features_pca.values)
            regime_probs = hmm_model.predict_proba(features_pca.values)

            # Get model summary
            model_summary = hmm_model.get_model_summary()

            # Calculate forward returns for quality assessment
            forward_returns = market_data['close'].pct_change().shift(-1)
            forward_returns = forward_returns.loc[features_pca.index]

            # Assess quality
            tprint_info("  → Assessing regime quality")
            metrics = self.quality_assessor.assess_hmm_regime_quality(
                regime_labels=regime_labels,
                feature_data=features_pca,
                transition_matrix=model_summary['transition_matrix'],
                hmm_model=None,
                forward_returns=forward_returns,
                timestamps=features_pca.index,
                timeframe=timeframe,
                min_regime_size=10,
                run_validators=True,
                temporal_sensitivity_mode="standard"
            )

            # Create result
            result = {
                'regime_labels': regime_labels,
                'regime_probs': regime_probs,
                'features': features_pca,
                'original_features': features,
                'pca_model': pca_model,
                'pca_explained_variance': explained_var,
                'hmm_model': hmm_model,
                'transition_matrix': model_summary['transition_matrix'],
                'stationary_distribution': model_summary['stationary_distribution'],
                'expected_durations': model_summary['expected_durations'],
                'model_summary': model_summary,
                'quality_metrics': metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics,
                'n_regimes': n_components,
                'timestamps': features_pca.index
            }

            return result

        except Exception as e:
            tprint_error(f"❌ Clustering failed: {e}")
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
            # Create labels DataFrame
            labels_df = pd.DataFrame({
                'timestamp': result['timestamps'],
                'regime_label': result['regime_labels']
            })
            labels_df.set_index('timestamp', inplace=True)

            # Create probabilities DataFrame
            probs_columns = [f'regime_{i}_prob' for i in range(result['n_regimes'])]
            probs_df = pd.DataFrame(
                result['regime_probs'],
                index=result['timestamps'],
                columns=probs_columns
            )

            # Save labels
            self._save_artifact(
                data=labels_df,
                artifact_name='rolling_hmm_regime_labels',
                artifact_type='data',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )

            # Save probabilities
            self._save_artifact(
                data=probs_df,
                artifact_name='rolling_hmm_regime_probabilities',
                artifact_type='data',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            )

            # Save transition matrix
            transition_matrix_df = pd.DataFrame(
                result['transition_matrix'],
                columns=[f'to_regime_{i}' for i in range(result['n_regimes'])],
                index=[f'from_regime_{i}' for i in range(result['n_regimes'])]
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
            tprint_error(f"❌ Failed to save results: {e}")
            self.logger.error(f"Failed to save results: {e}", exc_info=True)
            raise

    async def _generate_reports(
        self,
        result: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any]
    ):
        """Generate quality assessment reports."""
        try:
            metrics = result['quality_metrics']

            # Print summary
            tprint("", "INFO")
            tprint("=" * 80, "INFO")
            tprint("📊 Rolling HMM Clustering Quality Report", "INFO")
            tprint("=" * 80, "INFO")
            tprint(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}", "INFO")
            tprint(f"Number of Regimes: {result['n_regimes']}", "INFO")
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
            method_config = {
                'rolling_hmm_params': config.get('rolling_hmm_params', {}),
                'ewma_config': getattr(hmm_model, 'config', None)
            }
            self.quality_assessor.generate_markdown_report(
                metrics_obj,
                symbol=symbol,
                method_specific_config=method_config
            )

            all_trials = None
            hpo_results = result.get('hpo_results') or config.get('hpo_results')
            if not hpo_results:
                hpo_results = config.get('hpo_summary')
            if hpo_results:
                trial_keys = ['coarse_results', 'fine_results', 'refinement_results']
                all_trials = []
                for key in trial_keys:
                    trials = hpo_results.get(key)
                    if isinstance(trials, list):
                        all_trials.extend(trials)

            self.quality_assessor.generate_comprehensive_csv_report(
                metrics_obj,
                all_trials=all_trials,
                symbol=symbol,
                method_specific_config=method_config
            )

        except Exception as e:
            tprint_warning(f"⚠️  Failed to generate reports: {e}")
            self.logger.warning(f"Failed to generate reports: {e}", exc_info=True)

# ... (rest of the code remains the same)
    def _log_best_params(self, best_params: Dict[str, Any]):
        """Log best parameters from HPO."""
        tprint("Best Parameters:", "INFO")
        for key, value in best_params.items():
            if key == 'ewma_config_idx':
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
        tprint(f"❌ Rolling HMM Regime Discovery failed: {error_msg}", "ERROR")

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
        """
        Apply execution mode data filtering (matching statsmodel_clustering pattern).

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

        # Determine days limit based on execution mode
        if execution_mode == 'blank':
            days_limit = 20  # 20 days for blank mode
        elif execution_mode == 'light':
            days_limit = 180  # 180 days (6 months) for light mode
        else:  # 'full'
            tprint_info("  → Full mode: Using all available data (no filtering)")
            return data  # No filtering for full mode

        # Calculate sample limit
        samples_per_day = samples_per_day_map.get(timeframe, 24)
        limit = days_limit * samples_per_day

        # Apply filter
        if len(data) > limit:
            filtered = data.tail(limit).copy()  # Keep most recent data
            tprint_info(f"  → {execution_mode.capitalize()} mode: Filtered to {days_limit} days ({limit} samples)")
            return filtered

        tprint_info(f"  → {execution_mode.capitalize()} mode: Data size ({len(data)}) within limit ({limit} samples)")
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
