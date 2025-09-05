"""Step 3: HMM Regime Discovery with Comprehensive Monitoring."""
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.graceful_module_handler import graceful_handler
from src.utils.pipeline_standards import PipelineStandards
logger = system_logger.getChild('Step03HMMRegimeDiscovery')

class Step03HMMRegimeDiscovery(BaseStep):
    """Step 3: HMM Regime Discovery for market regime identification."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__('step03_hmm_regime_discovery', config)
        self.logger = system_logger.getChild('Step03HMMRegimeDiscovery')
        self.hmm_config = self.config.get('hmm_regime_discovery', {'n_components': 3, 'n_iter': 100, 'random_state': 42, 'covariance_type': 'full', 'min_regime_samples': 1000})
        graceful_handler.setup_graceful_imports()
        self.standards = PipelineStandards(self.logger)
        self.hmm_model = self._setup_hmm_model()

    def _setup_hmm_model(self) -> None:
        """Setup HMM model with graceful fallback."""
        try:
            from sklearn.mixture import GaussianMixture
import logging

            self.logger.info('✅ Using GaussianMixture for regime discovery')
            return GaussianMixture
        except ImportError:
            self.logger.warning('⚠️ sklearn not available, using fallback regime discovery')
            return None

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HMM regime discovery step."""
        step_start = time.time()
        self.logger.info('🎯 Starting HMM regime discovery...')
        try:
            data = pipeline_state.get('dataframe')
            if data is None:
                data = training_input.get('validated_data')
            if data is None:
                raise ValueError('No DataFrame available for regime discovery')
            data = self._validate_input_data(data)
            self.logger.info(f'📊 Processing {len(data)} rows for regime discovery')
            features = self._prepare_regime_features(data)
            regime_results = await self._discover_regimes(features, data)
            output_path = self._save_regime_results(regime_results, training_input)
            pipeline_state['regime_discovery'] = regime_results
            pipeline_state['regime_data_path'] = str(output_path)
            execution_time = time.time() - step_start
            self.logger.info(f'✅ HMM regime discovery completed in {execution_time:.2f}s')
            return {'success': True, 'regime_results': regime_results, 'execution_time': execution_time, 'output_path': str(output_path)}
        except Exception as e:
            self.logger.exception(f'❌ HMM regime discovery failed: {e}')
            return {'success': False, 'error': str(e), 'execution_time': time.time() - step_start}

    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime discovery."""
        self.logger.info('🔧 Preparing features for regime discovery...')
        features = []
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            features.append(returns.values)
        if 'volume' in data.columns:
            volume_returns = data['volume'].pct_change().dropna()
            features.append(volume_returns.values)
        if 'high' in data.columns and 'low' in data.columns:
            volatility = ((data['high'] - data['low']) / data['close']).dropna()
            features.append(volatility.values)
        if features:
            min_length = min((len(f) for f in features))
            aligned_features = np.array([f[:min_length] for f in features]).T
            self.logger.info(f'📊 Prepared {aligned_features.shape[1]} features with {aligned_features.shape[0]} samples')
            return aligned_features
        else:
            self.logger.warning('⚠️ No suitable features found, using fallback')
            if 'close' in data.columns:
                prices = data['close'].values
                returns = np.diff(prices) / prices[:-1]
                return returns.reshape(-1, 1)
            else:
                raise ValueError('No suitable features available for regime discovery')

    async def _discover_regimes(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover market regimes using HMM."""
        self.logger.info('🎯 Discovering market regimes...')
        if self.hmm_model is None:
            return self._fallback_regime_discovery(features, data)
        try:
            n_components = self.hmm_config['n_components']
            model = self.hmm_model(n_components=n_components, random_state=self.hmm_config['random_state'], covariance_type=self.hmm_config['covariance_type'])
            model.fit(features)
            regime_labels = model.predict(features)
            regime_stats = self._calculate_regime_statistics(features, regime_labels, data)
            self.logger.info(f'✅ Discovered {n_components} market regimes')
            for i, stats in regime_stats.items():
                self.logger.info(f"   Regime {i}: {stats['count']} samples, mean return: {stats['mean_return']:.4f}")
            return {'regime_labels': regime_labels.tolist(), 'regime_stats': regime_stats, 'model_params': {'n_components': n_components, 'means': model.means_.tolist() if hasattr(model, 'means_') else [], 'covariances': model.covariances_.tolist() if hasattr(model, 'covariances_') else []}, 'discovery_method': 'gaussian_mixture'}
        except Exception as e:
            self.logger.warning(f'⚠️ HMM regime discovery failed: {e}, using fallback')
            return self._fallback_regime_discovery(features, data)

    def _fallback_regime_discovery(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback regime discovery using simple statistical methods."""
        self.logger.info('🔄 Using fallback regime discovery...')
        if features.shape[1] > 0:
            returns = features[:, 0]
            window = min(100, len(returns) // 10)
            if window > 1:
                volatility = pd.Series(returns).rolling(window=window).std().fillna(0)
                low_threshold = volatility.quantile(0.33)
                high_threshold = volatility.quantile(0.67)
                regime_labels = np.zeros(len(returns))
                regime_labels[volatility > high_threshold] = 2
                regime_labels[(volatility > low_threshold) & (volatility <= high_threshold)] = 1
            else:
                regime_labels = (returns > np.median(returns)).astype(int)
        else:
            regime_labels = np.zeros(len(data))
        regime_stats = self._calculate_regime_statistics(features, regime_labels, data)
        self.logger.info(f'✅ Fallback regime discovery completed: {len(np.unique(regime_labels))} regimes')
        return {'regime_labels': regime_labels.tolist(), 'regime_stats': regime_stats, 'model_params': {'n_components': len(np.unique(regime_labels)), 'discovery_method': 'fallback'}, 'discovery_method': 'fallback'}

    def _calculate_regime_statistics(self, features: np.ndarray, regime_labels: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for each regime."""
        regime_stats = {}
        unique_regimes = np.unique(regime_labels)
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_features = features[mask]
            stats = {'count': int(np.sum(mask)), 'percentage': float(np.sum(mask) / len(regime_labels) * 100)}
            if len(regime_features) > 0:
                stats['mean_return'] = float(np.mean(regime_features[:, 0])) if regime_features.shape[1] > 0 else 0.0
                stats['volatility'] = float(np.std(regime_features[:, 0])) if regime_features.shape[1] > 0 else 0.0
            regime_stats[str(regime)] = stats
        return regime_stats

    def _save_regime_results(self, regime_results: Dict[str, Any], training_input: Dict[str, Any]) -> Path:
        """Save regime discovery results."""
        symbol = training_input.get('symbol', 'UNKNOWN')
        exchange = training_input.get('exchange', 'UNKNOWN')
        output_dir = Path(f'data/training/regimes/{exchange}_{symbol}')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"regime_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(output_path, 'w') as f:
            json.dump(regime_results, f, indent=2)
        self.logger.info(f'💾 Saved regime results to {output_path}')
        return output_path

    def _validate_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input data using pipeline standards.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        self.logger.info('🔍 Validating input data using pipeline standards...')
        validation_result = self.standards.validate_data_quality(data, 'unified')
        if not validation_result.passed:
            self.logger.warning(f'⚠️ Data quality issues detected: {validation_result.quality_score:.2f}')
            for issue in validation_result.issues:
                self.logger.warning(f'   - {issue.message}')
        fixed_data = data.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in fixed_data.columns]
        if missing_columns:
            self.logger.error(f'❌ Missing required columns: {missing_columns}')
            raise ValueError(f'Missing required columns for regime discovery: {missing_columns}')
        for col in required_columns:
            if col in fixed_data.columns:
                if not pd.api.types.is_numeric_dtype(fixed_data[col]):
                    self.logger.info(f'🔢 Converting {col} to numeric')
                    fixed_data[col] = pd.to_numeric(fixed_data[col], errors='coerce')
        initial_count = len(fixed_data)
        fixed_data = fixed_data.dropna(subset=required_columns)
        removed_count = initial_count - len(fixed_data)
        if removed_count > 0:
            self.logger.info(f'🗑️ Removed {removed_count} rows with NaN values')
        if len(fixed_data) < 100:
            self.logger.error(f'❌ Insufficient data after cleaning: {len(fixed_data)} rows')
            raise ValueError(f'Insufficient data for regime discovery: {len(fixed_data)} rows')
        self.logger.info(f'✅ Input validation completed: {len(fixed_data)} rows')
        return fixed_data
__all__ = ['Step03HMMRegimeDiscovery']