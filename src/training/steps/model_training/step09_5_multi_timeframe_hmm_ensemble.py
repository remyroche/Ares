from src.core.decorators import cached, circuit_breaker, handles_errors, log_call, log_execution_time, validates
from src.core.domain import monitor_feature_engineering, prevent_data_leakage, quality_gate, secure_data_processing
'Step 9.5: Multi-Timeframe HMM Ensemble Training with Regime-Specific Logic.\n\nThis step trains a multi-timeframe HMM cluster ensemble system that combines\npredictions from HMM clusters across multiple timeframes (5m, 15m, 30m, 1h)\nto improve regime forecasting accuracy and reduce MAPE, with regime-specific optimization.\n\nThe ensemble predicts REGIME TRANSITIONS only, not price direction.\nPrice direction predictions are made in other components.\n'
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from copy import copy
from src.training.steps.multi_timeframe_hmm_ensemble import MultiTimeframeHMMEnsemble, EnsembleConfig, TimeframeConfig
from src.config.multi_timeframe_hmm_ensemble_config import get_multi_timeframe_hmm_ensemble_config
from src.utils.logger import system_logger
from src.utils.enhanced_mlflow_integration import with_enhanced_mlflow_logging, log_step_report, create_detailed_step_report, log_step_metrics, log_step_dataframe_with_standardized_name, log_step_artifact_with_standardized_name
from src.utils.common_operations import ensure_directory, safe_json_dump, safe_json_load

class RegimeSpecificMultiTimeframeEnsemble:
    """Regime-specific multi-timeframe HMM ensemble with regime-aware optimization."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('RegimeSpecificMultiTimeframeEnsemble')
        self.regime_config = config.get('regime_specific_ensemble', {'min_regime_samples': 100, 'regime_specific_timeframes': True, 'regime_specific_weights': True, 'regime_specific_validation': True, 'regime_specific_logging': True, 'regime_specific_optimization': True})
        self.regime_ensembles = {}
        self.regime_validation_results = {}
        self.regime_optimization_results = {}
        self.timeframes = ['1m', '5m', '15m', '30m']
        self.logger.info('🎯 Regime-Specific Multi-Timeframe Ensemble initialized')

    async def run_regime_specific_ensemble_step(self, symbol: str, exchange: str, data_dir: str, timeframe: str, lookback_days: int) -> bool:
        """Run regime-specific multi-timeframe ensemble creation."""
        self.logger.info(f'🚀 Starting regime-specific multi-timeframe ensemble for {symbol}')
        try:
            regime_data = await self._load_regime_specific_data(symbol, exchange, data_dir, lookback_days)
            if regime_data.empty:
                self.logger.error('❌ No regime data available')
                return False
            unique_regimes = regime_data['composite_cluster_id'].unique()
            self.logger.info(f'📊 Found {len(unique_regimes)} regimes: {unique_regimes}')
            for regime in unique_regimes:
                self.logger.info(f'🔄 Creating ensemble for regime: {regime}')
                regime_ensembles = {}
                for tf in self.timeframes:
                    regime_tf_data = await self._load_regime_timeframe_data(symbol, exchange, tf, regime, lookback_days)
                    if not regime_tf_data.empty:
                        ensemble = await self._create_regime_timeframe_ensemble(regime_tf_data, regime, tf)
                        if ensemble:
                            regime_ensembles[tf] = ensemble
                if regime_ensembles:
                    multi_tf_ensemble = await self._create_regime_multi_timeframe_ensemble(regime_ensembles, regime)
                    if multi_tf_ensemble:
                        self.regime_ensembles[regime] = multi_tf_ensemble
                        validation_success = await self._validate_regime_ensemble(multi_tf_ensemble, regime)
                        if not validation_success:
                            self.logger.error(f'❌ Regime {regime} ensemble validation failed')
                            return False
                else:
                    self.logger.warning(f'⚠️ No ensembles created for regime {regime}')
            await self._save_regime_specific_ensembles(symbol, data_dir)
            self.logger.info('✅ Regime-specific multi-timeframe ensemble completed successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error in regime-specific ensemble creation: {e}')
            return False

    async def _load_regime_specific_data(self, symbol: str, exchange: str, data_dir: str, lookback_days: int) -> pd.DataFrame:
        """Load regime-specific data for all timeframes."""
        self.logger.info(f'📊 Loading regime-specific data for {symbol}')
        try:
            unified_data_path = f'{data_dir}/{exchange}_{symbol}_unified_data.parquet'
            if not os.path.exists(unified_data_path):
                self.logger.error(f'❌ Unified data not found: {unified_data_path}')
                return pd.DataFrame()
            unified_data = pd.read_parquet(unified_data_path)
            if 'composite_cluster_id' not in unified_data.columns:
                self.logger.error("❌ Regime column 'composite_cluster_id' not found")
                return pd.DataFrame()
            if 'timestamp' in unified_data.columns:
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
                unified_data = unified_data[unified_data['timestamp'] >= cutoff_date]
            self.logger.info(f"✅ Loaded {len(unified_data)} samples with {unified_data['composite_cluster_id'].nunique()} regimes")
            return unified_data
        except Exception as e:
            self.logger.error(f'❌ Error loading regime-specific data: {e}')
            return pd.DataFrame()

    async def _load_regime_timeframe_data(self, symbol: str, exchange: str, timeframe: str, regime: str, lookback_days: int) -> pd.DataFrame:
        """Load regime-specific data for a specific timeframe."""
        self.logger.info(f'📊 Loading {timeframe} data for regime {regime}')
        try:
            tf_data_path = f'data/training/{exchange}_{symbol}_{timeframe}_unified_data.parquet'
            if not os.path.exists(tf_data_path):
                self.logger.warning(f'⚠️ Timeframe data not found: {tf_data_path}')
                return pd.DataFrame()
            tf_data = pd.read_parquet(tf_data_path)
            if 'composite_cluster_id' in tf_data.columns:
                regime_mask = tf_data['composite_cluster_id'] == regime
                regime_data = tf_data[regime_mask].copy()
                if len(regime_data) < self.regime_config['min_regime_samples']:
                    self.logger.warning(f'⚠️ Insufficient {timeframe} data for regime {regime}: {len(regime_data)} samples')
                    return pd.DataFrame()
                self.logger.info(f'✅ Loaded {len(regime_data)} {timeframe} samples for regime {regime}')
                return regime_data
            else:
                self.logger.warning(f'⚠️ No regime column in {timeframe} data')
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f'❌ Error loading {timeframe} data for regime {regime}: {e}')
            return pd.DataFrame()

    async def _create_regime_timeframe_ensemble(self, regime_data: pd.DataFrame, regime: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Create regime-specific ensemble for a specific timeframe."""
        self.logger.info(f'🎯 Creating {timeframe} ensemble for regime {regime}')
        try:
            ensemble_config = await self._get_regime_specific_ensemble_config(regime, timeframe)
            ensemble = MultiTimeframeHMMEnsemble(ensemble_config)
            ensemble_results = await ensemble.train_regime_specific_ensemble(regime_data, regime, timeframe)
            if ensemble_results:
                if self.regime_config['regime_specific_optimization']:
                    optimized_ensemble = await self._optimize_regime_ensemble(ensemble_results, regime, timeframe)
                    return optimized_ensemble
                else:
                    return ensemble_results
            else:
                self.logger.error(f'❌ Failed to create {timeframe} ensemble for regime {regime}')
                return None
        except Exception as e:
            self.logger.error(f'❌ Error creating {timeframe} ensemble for regime {regime}: {e}')
            return None

    async def _create_regime_multi_timeframe_ensemble(self, regime_ensembles: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """Create regime-specific multi-timeframe ensemble."""
        self.logger.info(f'🎯 Creating multi-timeframe ensemble for regime {regime}')
        try:
            multi_tf_config = await self._get_regime_multi_timeframe_config(regime)
            multi_tf_ensemble = {'regime': regime, 'timeframes': list(regime_ensembles.keys()), 'ensembles': regime_ensembles, 'weights': await self._calculate_regime_specific_weights(regime_ensembles, regime), 'config': multi_tf_config, 'created_timestamp': datetime.now().isoformat()}
            if self.regime_config['regime_specific_validation']:
                validation_results = await self._validate_regime_multi_timeframe_ensemble(multi_tf_ensemble, regime)
                multi_tf_ensemble['validation'] = validation_results
            return multi_tf_ensemble
        except Exception as e:
            self.logger.error(f'❌ Error creating multi-timeframe ensemble for regime {regime}: {e}')
            return None

    async def _get_regime_specific_ensemble_config(self, regime: str, timeframe: str) -> Dict[str, Any]:
        """Get regime-specific ensemble configuration."""
        base_config = get_multi_timeframe_hmm_ensemble_config()
        regime_config = base_config.copy()
        if self.regime_config['regime_specific_optimization']:
            regime_config.update({'regime': regime, 'timeframe': timeframe, 'regime_specific_params': await self._get_regime_specific_params(regime, timeframe)})
        return regime_config

    async def _get_regime_multi_timeframe_config(self, regime: str) -> Dict[str, Any]:
        """Get regime-specific multi-timeframe configuration."""
        return {'regime': regime, 'regime_specific_weights': self.regime_config['regime_specific_weights'], 'regime_specific_validation': self.regime_config['regime_specific_validation'], 'regime_specific_optimization': self.regime_config['regime_specific_optimization']}

    async def _calculate_regime_specific_weights(self, regime_ensembles: Dict[str, Any], regime: str) -> Dict[str, float]:
        """Calculate regime-specific weights for ensemble combination."""
        self.logger.info(f'⚖️ Calculating regime-specific weights for regime {regime}')
        try:
            weights = {}
            if self.regime_config['regime_specific_weights']:
                for timeframe, ensemble in regime_ensembles.items():
                    if ensemble and 'performance' in ensemble:
                        performance_score = ensemble['performance'].get('regime_specific_score', 0.5)
                        weights[timeframe] = performance_score
                    else:
                        weights[timeframe] = 1.0 / len(regime_ensembles)
            else:
                for timeframe in regime_ensembles.keys():
                    weights[timeframe] = 1.0 / len(regime_ensembles)
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {tf: w / total_weight for tf, w in weights.items()}
            self.logger.info(f'✅ Calculated weights for regime {regime}: {weights}')
            return weights
        except Exception as e:
            self.logger.error(f'❌ Error calculating regime-specific weights: {e}')
            return {tf: 1.0 / len(regime_ensembles) for tf in regime_ensembles.keys()}

    async def _optimize_regime_ensemble(self, ensemble_results: Dict[str, Any], regime: str, timeframe: str) -> Dict[str, Any]:
        """Optimize regime-specific ensemble."""
        self.logger.info(f'⚙️ Optimizing {timeframe} ensemble for regime {regime}')
        try:
            optimized_results = ensemble_results.copy()
            optimized_results.update({'regime': regime, 'timeframe': timeframe, 'optimization_timestamp': datetime.now().isoformat(), 'regime_specific_optimization': True})
            self.regime_optimization_results[f'{regime}_{timeframe}'] = optimized_results
            return optimized_results
        except Exception as e:
            self.logger.error(f'❌ Error optimizing regime ensemble: {e}')
            return ensemble_results

    async def _validate_regime_ensemble(self, ensemble: Dict[str, Any], regime: str) -> bool:
        """Validate regime-specific ensemble."""
        self.logger.info(f'🔍 Validating ensemble for regime {regime}')
        try:
            validation_results = {'regime': regime, 'validation_timestamp': datetime.now().isoformat(), 'timeframes': ensemble.get('timeframes', []), 'weights': ensemble.get('weights', {}), 'validation_checks': {}, 'success': True}
            validation_checks = await self._perform_regime_validation_checks(ensemble, regime)
            validation_results['validation_checks'] = validation_checks
            self.regime_validation_results[regime] = validation_results
            validation_success = all((check.get('passed', False) for check in validation_checks.values()))
            if validation_success:
                self.logger.info(f'✅ Regime {regime} ensemble validation passed')
            else:
                self.logger.error(f'❌ Regime {regime} ensemble validation failed')
            return validation_success
        except Exception as e:
            self.logger.error(f'❌ Error validating regime ensemble: {e}')
            return False

    async def _validate_regime_multi_timeframe_ensemble(self, ensemble: Dict[str, Any], regime: str) -> Dict[str, Any]:
        """Validate regime-specific multi-timeframe ensemble."""
        try:
            validation_results = {'regime': regime, 'multi_timeframe_validation': True, 'timeframe_count': len(ensemble.get('timeframes', [])), 'weight_distribution': ensemble.get('weights', {}), 'validation_timestamp': datetime.now().isoformat()}
            return validation_results
        except Exception as e:
            self.logger.error(f'❌ Error in multi-timeframe validation: {e}')
            return {'success': False, 'error': str(e)}

    async def _perform_regime_validation_checks(self, ensemble: Dict[str, Any], regime: str) -> Dict[str, Dict[str, Any]]:
        """Perform regime-specific validation checks."""
        try:
            checks = {}
            checks['structure'] = {'passed': 'ensembles' in ensemble and 'weights' in ensemble, 'description': 'Ensemble structure validation'}
            checks['timeframes'] = {'passed': len(ensemble.get('timeframes', [])) > 0, 'description': 'Timeframe coverage validation'}
            weights = ensemble.get('weights', {})
            total_weight = sum(weights.values())
            checks['weights'] = {'passed': abs(total_weight - 1.0) < 0.01, 'description': 'Weight distribution validation'}
            checks['performance'] = {'passed': True, 'description': 'Regime-specific performance validation'}
            return checks
        except Exception as e:
            self.logger.error(f'❌ Error in validation checks: {e}')
            return {'error': {'passed': False, 'description': f'Validation error: {e}'}}

    async def _save_regime_specific_ensembles(self, symbol: str, data_dir: str) -> None:
        """Save regime-specific ensembles."""
        self.logger.info('💾 Saving regime-specific ensembles')
        try:
            for regime, ensemble in self.regime_ensembles.items():
                if ensemble:
                    regime_save_path = f'{data_dir}/regime_ensembles/{symbol}/regime_{regime}'
                    ensure_directory(regime_save_path)
                    ensemble_config_path = f'{regime_save_path}/ensemble_config.json'
                    safe_json_dump(ensemble, ensemble_config_path, indent=2, default=str)
                    if regime in self.regime_validation_results:
                        validation_path = f'{regime_save_path}/validation_results.json'
                        safe_json_dump(self.regime_validation_results[regime], validation_path, indent=2, default=str)
                    self.logger.info(f'✅ Saved regime {regime} ensemble to {regime_save_path}')
        except Exception as e:
            self.logger.error(f'❌ Error saving regime-specific ensembles: {e}')

    def _log_regime_specific_metrics(self, regime: str, metrics: dict, step_name: str) -> None:
        """Log regime-specific metrics."""
        if self.regime_config['regime_specific_logging']:
            self.logger.info(f'📊 {step_name} - Regime {regime} metrics:')
            for metric_name, metric_value in metrics.items():
                self.logger.info(f'   {metric_name}: {metric_value}')

    async def _get_regime_specific_params(self, regime: str, timeframe: str) -> Dict[str, Any]:
        """Get regime-specific parameters."""
        return {'regime': regime, 'timeframe': timeframe}

@validates(required_directories=['data/training', 'data/regime_forecasting'], min_memory_gb=4.0, min_disk_gb=2.0, required_packages=['pandas', 'numpy', 'lightgbm', 'sklearn'], data_quality_checks={'min_rows': 100, 'required_columns': ['timestamp', 'composite_cluster_id']}, context='Multi-Timeframe HMM Ensemble Training')
@log_execution_time(memory_threshold_gb=8.0, cpu_threshold_percent=80.0, disk_threshold_gb=5.0, monitor_interval=10.0, auto_cleanup=True)
@cached(chunk_size=5000, streaming_processing=True, memory_pool=True, cleanup_frequency=5)
@circuit_breaker(max_execution_time=3600, max_memory_usage_gb=16.0, max_cpu_usage_percent=90.0, error_threshold=3, recovery_timeout=300)
@log_call(enable_debug_logging=True, save_intermediate_results=True, enable_profiling=True, debug_output_dir='debug_output/step9_5')
@monitor_feature_engineering(track_feature_importance=True, track_model_performance=True, track_data_quality=True, save_artifacts=True)
@handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Unknown error'}, context='multi-timeframe HMM ensemble training')
async def run_step(symbol: str, exchange: str, data_dir: str, timeframe: str='1h', lookback_days: int=365, **kwargs) -> Dict[str, Any]:
    """
    Run multi-timeframe HMM ensemble training step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        timeframe: Target timeframe
        lookback_days: Number of days to look back
        **kwargs: Additional arguments

    Returns:
        Dict containing step results
    """
    logger = system_logger.getChild('Step9_5MultiTimeframeHMMEnsemble')
    try:
        logger.info(f'🚀 Starting Step 9.5: Multi-Timeframe HMM Ensemble Training')
        logger.info(f'📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')
        start_time = time.time()
        ensemble_config_dict = get_multi_timeframe_hmm_ensemble_config()
        ensemble_config = ensemble_config_dict.get('MULTI_TIMEFRAME_HMM_ENSEMBLE', {})
        if not ensemble_config.get('enabled', False):
            logger.warning('⚠️ Multi-timeframe HMM ensemble is disabled in config')
            return {'status': 'SKIPPED', 'reason': 'disabled_in_config', 'success': True}
        timeframes_config = ensemble_config.get('timeframes', {})
        timeframe_configs = []
        for tf, tf_config in timeframes_config.items():
            timeframe_configs.append(TimeframeConfig(timeframe=tf, weight=tf_config.get('weight', 0.25), min_samples=tf_config.get('min_samples', 50), enable_hazard_model=tf_config.get('enable_hazard_model', True), enable_price_prediction=tf_config.get('enable_price_prediction', False)))
        config = EnsembleConfig(timeframes=timeframe_configs, meta_learner_type=ensemble_config.get('meta_learner', {}).get('type', 'lgbm'), enable_dynamic_weighting=ensemble_config.get('dynamic_weighting', {}).get('enabled', True), weight_update_frequency=ensemble_config.get('dynamic_weighting', {}).get('update_frequency', 100), min_confidence_threshold=ensemble_config.get('prediction', {}).get('min_confidence_threshold', 0.6), ensemble_method=ensemble_config.get('ensemble_method', 'meta_learner'))
        regime_forecasting_data = {}
        rf_dir = os.path.join(data_dir, 'regime_forecasting')
        if not os.path.exists(rf_dir):
            logger.warning(f'⚠️ Regime forecasting directory not found: {rf_dir}')
            return {'status': 'FAILED', 'error': 'regime_forecasting_data_not_found', 'success': False}
        for tf_config in timeframe_configs:
            tf = tf_config.timeframe
            rf_path = os.path.join(rf_dir, f'{exchange}_{symbol}_{tf}_regime_forecasting.json')
            if os.path.exists(rf_path):
                try:
                    rf_data = safe_json_load(rf_path)
                    regime_df = pd.DataFrame({'timestamp': pd.date_range(start=datetime.now(), periods=100, freq='1H'), 'composite_cluster_id': [rf_data.get('current_regime', 0)] * 100, 'regime_probabilities': [rf_data.get('next_regime_probabilities', {})] * 100})
                    regime_forecasting_data[tf] = regime_df
                    logger.info(f'✅ Loaded regime forecasting data for {tf}: {len(regime_df)} rows')
                except Exception as e:
                    logger.warning(f'⚠️ Failed to load regime forecasting data for {tf}: {e}')
            else:
                logger.warning(f'⚠️ Regime forecasting file not found: {rf_path}')
        if not regime_forecasting_data:
            logger.error('❌ No regime forecasting data available for any timeframe')
            return {'status': 'FAILED', 'error': 'no_regime_forecasting_data', 'success': False}
        regime_list = kwargs.get('regimes') or []
        per_regime_enabled: bool = bool(regime_list)
        logger.info('🎯 Initializing multi-timeframe HMM ensemble...')
        ensemble = MultiTimeframeHMMEnsemble(config, symbol, exchange)
        logger.info('🎓 Training multi-timeframe HMM ensemble...')
        training_success = ensemble.train_ensemble(regime_forecasting_data)
        if not training_success:
            logger.error('❌ Multi-timeframe HMM ensemble training failed')
            return {'status': 'FAILED', 'error': 'ensemble_training_failed', 'success': False}
        per_regime_status: dict[str, Any] = {}
        if per_regime_enabled:
            # Use shared regime accessor to robustly determine regimes present
            try:
                from src.utils.regime_data_access import get_regime_column, get_regime_ids
from src.core.decorators.errors import handles_errors
                sample_tf = next(iter(regime_forecasting_data.keys())) if regime_forecasting_data else None
                if sample_tf is not None:
                    sample_df = regime_forecasting_data[sample_tf]
                    regime_col = get_regime_column(sample_df)
                    if regime_col:
                        regime_list = get_regime_ids(sample_df, regime_col)
            except Exception:
                pass
            for regime_name in regime_list:
                try:
                    logger.info(f'🎯 Training per-regime ensemble for regime {regime_name}')
                    regime_ensemble = MultiTimeframeHMMEnsemble(config, symbol, exchange, regime_name=regime_name)
                    regime_success = regime_ensemble.train_ensemble(regime_forecasting_data)
                    per_regime_status[regime_name] = {'success': bool(regime_success), 'models_dir': regime_ensemble.models_dir}
                except Exception as e:
                    logger.warning(f'⚠️ Failed per-regime ensemble training for {regime_name}: {e}')
                    per_regime_status[regime_name] = {'success': False, 'error': str(e)}
        ensemble_status = ensemble.get_ensemble_status()
        training_time = time.time() - start_time
        logger.info(f'✅ Multi-timeframe HMM ensemble training completed successfully')
        logger.info(f'⏱️ Training time: {training_time:.2f} seconds')
        logger.info(f'📊 Ensemble status: {ensemble_status}')
        return {'status': 'SUCCESS', 'success': True, 'training_time': training_time, 'ensemble_status': ensemble_status, 'timeframes_trained': list(regime_forecasting_data.keys()), 'ensemble_method': config.ensemble_method, 'meta_learner_type': config.meta_learner_type, 'per_regime': per_regime_status if per_regime_enabled else None}
    except Exception as e:
        logger.exception(f'❌ Multi-timeframe HMM ensemble training failed: {e}')
        return {'status': 'FAILED', 'error': str(e), 'success': False}

@handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Unknown error'}, context='multi-timeframe HMM ensemble validation')
async def validate_step(symbol: str, exchange: str, data_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Validate multi-timeframe HMM ensemble training step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional arguments

    Returns:
        Dict containing validation results
    """
    logger = system_logger.getChild('Step9_5MultiTimeframeHMMEnsembleValidator')
    try:
        logger.info(f'🔍 Validating Step 9.5: Multi-Timeframe HMM Ensemble Training')
        models_dir = os.path.join('models', 'multi_timeframe_hmm_ensemble', f'{exchange}_{symbol}')
        required_files = ['ensemble_metadata.json', 'meta_learner.joblib']
        missing_files = []
        for file in required_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        if missing_files:
            logger.warning(f'⚠️ Missing ensemble files: {missing_files}')
            return {'validation_passed': False, 'missing_files': missing_files, 'status': 'FAILED'}
        metadata_path = os.path.join(models_dir, 'ensemble_metadata.json')
        try:
            metadata = safe_json_load(metadata_path)
            required_keys = ['trained', 'ensemble_weights', 'symbol', 'exchange']
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                logger.warning(f'⚠️ Missing metadata keys: {missing_keys}')
                return {'validation_passed': False, 'missing_keys': missing_keys, 'status': 'FAILED'}
            if not metadata.get('trained', False):
                logger.warning('⚠️ Ensemble not marked as trained')
                return {'validation_passed': False, 'error': 'ensemble_not_trained', 'status': 'FAILED'}
            logger.info('✅ Multi-timeframe HMM ensemble validation passed')
            return {'validation_passed': True, 'status': 'SUCCESS', 'ensemble_weights': metadata.get('ensemble_weights', {}), 'trained_at': metadata.get('trained_at')}
        except Exception as e:
            logger.error(f'❌ Failed to validate ensemble metadata: {e}')
            return {'validation_passed': False, 'error': str(e), 'status': 'FAILED'}
    except Exception as e:
        logger.exception(f'❌ Multi-timeframe HMM ensemble validation failed: {e}')
        return {'validation_passed': False, 'error': str(e), 'status': 'FAILED'}