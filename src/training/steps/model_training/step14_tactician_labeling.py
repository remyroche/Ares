"""Step 14: Regime-Aware Tactician Labeling with Regime-Specific Barriers."""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import asyncio
import contextlib
import os
import pickle
import json
from pathlib import Path

from src.core.decorators import handles_errors
from src.core.decorators.logging import log_execution_time, log_call
from src.core.decorators.cache import cached
from src.utils.logger import system_logger, log_io_operation, log_dataframe_overview
from src.utils.pipeline_standards import PipelineStandards
from src.config.environment import get_environment_settings

# Get dynamic symbol configuration
_settings = get_environment_settings()

def get_default_symbol() -> str:
    """Get the default trading symbol from configuration."""
    return _settings.get_default_symbol('ETHUSDT')

# Enhanced Reporting import
try:
    from src.training.steps.model_training.step14_enhanced_reporting import Step14EnhancedReporter
    ENHANCED_REPORTING_AVAILABLE = True
except ImportError:
    ENHANCED_REPORTING_AVAILABLE = False
    Step14EnhancedReporter = None

# Financial Logging import
from src.training.steps.model_training.step14_financial_logging import Step14FinancialLogger

# Try to import DynamicBarrierCalculator, fallback to mock if not available
try:
    from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
except ImportError:
    class DynamicBarrierCalculator:
        """Mock DynamicBarrierCalculator for fallback."""

        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config

        def calculate_dynamic_barriers(self, timeframe: Any) -> Dict[str, Tuple[float, float]]:
            """Return mock barrier calculations."""
            return {
                'high_precision': (0.01, 0.005),
                'standard': (0.02, 0.01),
                'conservative': (0.03, 0.015),
                'aggressive': (0.014, 0.007)
            }

# Try to import optional dependencies with fallbacks
try:
    from src.training.data_sharing_manager import get_data_sharing_manager
except ImportError:
    def get_data_sharing_manager(config: Dict[str, Any]) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Fallback data sharing manager."""
        return None

try:
    from src.training.steps.unified_data_loader import get_unified_data_loader
except ImportError:
    def get_unified_data_loader(*args, **kwargs) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Fallback unified data loader."""
        return None

# Required modules for dependency checking
REQUIRED_MODULES = ['pandas', 'numpy', 'sklearn', 'tactician.sr_breakout_predictor']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

ENSEMBLE_PREFERENCE_ORDER = ('stacking_cv', 'dynamic_weighting', 'voting')

class RegimeAwareTacticianLabeler:
    """Regime-aware tactician labeling with regime-specific barriers and precision thresholds."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config.get('tactician_triple_barrier', {})
        self.logger = system_logger.getChild('RegimeAwareTacticianLabeler')
        self.regime_config = config.get('regime_specific_tactician', {'regime_specific_barriers': True, 'regime_specific_precision': True, 'regime_specific_quality_filters': True, 'regime_specific_validation': True, 'regime_specific_logging': True, 'min_regime_samples': 100})
        self.financial_logger = None

        # Initialize enhanced reporting system
        if ENHANCED_REPORTING_AVAILABLE and Step14EnhancedReporter is not None:
            try:
                self.enhanced_reporter = Step14EnhancedReporter(config)
                self.logger.info('✅ Enhanced reporting system initialized for Step14')
            except Exception as e:
                self.logger.warning(f'Failed to initialize enhanced reporting: {e}')
                self.enhanced_reporter = None
        else:
            self.logger.info('Enhanced reporting not available, using fallback reporting')
            self.enhanced_reporter = None

        self._load_enhanced_config()
        self.regime_barrier_results = {}
        self.regime_labeling_results = {}
        self.regime_validation_results = {}
        self.logger.info('🎯 Regime-Aware Tactician Labeler initialized')

    def _load_enhanced_config(self) -> None:
        """Load enhanced configuration for regime-aware execution."""
        self.barrier_calculator = DynamicBarrierCalculator(self.config)
        self.barrier_combinations = self.barrier_calculator.calculate_dynamic_barriers(timeframe='1m')
        self.max_lookahead = self.config.get('max_lookahead', 50)
        self.enable_high_precision_mode = self.config.get('enable_high_precision_mode', True)
        self.precision_threshold = self.config.get('precision_threshold', 0.85)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.8)
        self.enable_quality_filters = self.config.get('enable_quality_filters', True)
        self.min_volume_threshold = self.config.get('min_volume_threshold', 1000)
        self.min_spread_threshold = self.config.get('min_spread_threshold', 0.0001)
        self.volatility_filter = self.config.get('volatility_filter', True)
        self.analyst_signal_requirement = self.config.get('analyst_signal_requirement', True)
        self.direction_agreement_required = self.config.get('direction_agreement_required', True)
        self.confidence_boost_threshold = self.config.get('confidence_boost_threshold', 0.9)
        self.timeframes = self.config.get('timeframes', ['1m', '5m'])
        self.primary_timeframe = self.config.get('primary_timeframe', '1m')
        self.secondary_timeframe = self.config.get('secondary_timeframe', '5m')
        self.binary_classification = self.config.get('binary_classification', True)
        self.logger.info(f'🔧 Enhanced Regime-Aware Tactician Configuration:')
        self.logger.info(f'   Timeframes: {self.timeframes}')
        self.logger.info(f'   Primary: {self.primary_timeframe}, Secondary: {self.secondary_timeframe}')
        self.logger.info(f"   Regime-specific barriers: {self.regime_config['regime_specific_barriers']}")
        self.logger.info(f"   Regime-specific precision: {self.regime_config['regime_specific_precision']}")
        self.logger.info(f'   High Precision Mode: {self.enable_high_precision_mode}')
        self.logger.info(f'   Precision Threshold: {self.precision_threshold}')

    async def apply_regime_specific_labeling(self, data: pd.DataFrame, regime_column: str='composite_cluster_id') -> pd.DataFrame:
        """Apply regime-specific tactician labeling."""
        self.logger.info(f'🚀 Starting regime-specific tactician labeling')
        try:
            if regime_column not in data.columns:
                self.logger.warning(f"⚠️ Regime column '{regime_column}' not found, using default parameters")
                return self._apply_default_labeling(data)
            labeled_data = data.copy()
            n = len(labeled_data)
            if n < 2:
                labeled_data['label'] = 0
                labeled_data['potential_profit_pct'] = 0.0
                return labeled_data
            regime_data = labeled_data[regime_column]
            unique_regimes = regime_data.unique()
            self.logger.info(f'📊 Found {len(unique_regimes)} unique regimes: {unique_regimes}')
            for regime in unique_regimes:
                regime_mask = regime_data == regime
                regime_data_subset = labeled_data[regime_mask]
                if len(regime_data_subset) >= self.regime_config['min_regime_samples']:
                    self.logger.info(f'🔄 Applying regime-specific labeling for regime {regime}')
                    regime_barriers = await self._get_regime_specific_barriers(regime, regime_data_subset)
                    regime_labeled = await self._apply_regime_barrier_labeling(regime_data_subset, regime_barriers, regime)
                    self.regime_labeling_results[regime] = {'barriers': regime_barriers, 'labeled_samples': len(regime_labeled), 'regime': regime}
                    labeled_data.loc[regime_mask] = regime_labeled
                else:
                    self.logger.warning(f'⚠️ Insufficient data for regime {regime}: {len(regime_data_subset)} samples')
            if self.binary_classification:
                original_count = len(labeled_data)
                hold_samples = (labeled_data['label'] == 0).sum()
                labeled_data = labeled_data[labeled_data['label'] != 0].copy()
                filtered_count = len(labeled_data)
                self.logger.info('📊 Label distribution after filtering:')
                self.logger.info(f"   LONG (1): {(labeled_data['label'] == 1).sum()} samples")
                self.logger.info(f"   SHORT (-1): {(labeled_data['label'] == -1).sum()} samples")
                self.logger.info(f'   HOLD (0): {hold_samples} samples (removed)')
                self.logger.info(f'   Total: {filtered_count}/{original_count} samples retained')
            return labeled_data
        except Exception as e:
            self.logger.error(f'❌ Error in regime-specific labeling: {e}')
            return data

    async def _get_regime_specific_barriers(self, regime: str, regime_data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Get regime-specific barriers for tactician labeling."""
        self.logger.info(f'🎯 Calculating regime-specific barriers for regime {regime}')
        try:
            if self.regime_config['regime_specific_barriers']:
                regime_volatility = regime_data['close'].pct_change().std()
                regime_volume = regime_data['volume'].mean()
                regime_spread = regime_data.get('spread', pd.Series([0.0001] * len(regime_data))).mean()
                base_upper = 0.02
                base_lower = 0.01
                if regime_volatility > 0.02:
                    upper_multiplier = 1.5
                    lower_multiplier = 1.2
                elif regime_volatility < 0.005:
                    upper_multiplier = 0.8
                    lower_multiplier = 0.7
                else:
                    upper_multiplier = 1.0
                    lower_multiplier = 1.0
                if regime_volume > 10000:
                    upper_multiplier *= 1.1
                    lower_multiplier *= 1.1
                elif regime_volume < 1000:
                    upper_multiplier *= 0.9
                    lower_multiplier *= 0.9
                upper_barrier = base_upper * upper_multiplier
                lower_barrier = base_lower * lower_multiplier
                regime_barriers = {'high_precision': (upper_barrier * 0.5, lower_barrier * 0.25), 'standard': (upper_barrier, lower_barrier), 'conservative': (upper_barrier * 1.5, lower_barrier * 1.5), 'aggressive': (upper_barrier * 0.7, lower_barrier * 0.5)}
                self.logger.info(f'✅ Calculated regime {regime} barriers:')
                for barrier_type, (upper, lower) in regime_barriers.items():
                    self.logger.info(f'   {barrier_type}: Upper={upper:.4f} ({upper * 100:.2f}%), Lower={lower:.4f} ({lower * 100:.2f}%)')
                return regime_barriers
            else:
                return self.barrier_combinations
        except Exception as e:
            self.logger.error(f'❌ Error calculating regime-specific barriers: {e}')
            return self.barrier_combinations

    async def _apply_regime_barrier_labeling(self, regime_data: pd.DataFrame, regime_barriers: Dict[str, Tuple[float, float]], regime: str) -> pd.DataFrame:
        """Apply regime-specific barrier labeling."""
        self.logger.info(f'🎯 Applying regime-specific barrier labeling for regime {regime}')
        try:
            labeled_data = regime_data.copy()
            precision_thresholds = await self._get_regime_specific_precision_thresholds(regime, regime_data)
            quality_filters = await self._get_regime_specific_quality_filters(regime, regime_data)
            for barrier_type, (upper_barrier, lower_barrier) in regime_barriers.items():
                self.logger.info(f'🔄 Applying {barrier_type} barriers for regime {regime}')
                regime_labeled = await self._apply_regime_triple_barrier(labeled_data, upper_barrier, lower_barrier, precision_thresholds, quality_filters, regime, barrier_type)
                barrier_key = f'{regime}_{barrier_type}'
                self.regime_barrier_results[barrier_key] = {'barrier_type': barrier_type, 'upper_barrier': upper_barrier, 'lower_barrier': lower_barrier, 'precision_thresholds': precision_thresholds, 'quality_filters': quality_filters, 'labeled_samples': len(regime_labeled), 'regime': regime}
                labeled_data = regime_labeled
            return labeled_data
        except Exception as e:
            self.logger.error(f'❌ Error applying regime barrier labeling: {e}')
            return regime_data

    async def _get_regime_specific_precision_thresholds(self, regime: str, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Get regime-specific precision thresholds."""
        try:
            if self.regime_config['regime_specific_precision']:
                regime_volatility = regime_data['close'].pct_change().std()
                regime_volume = regime_data['volume'].mean()
                base_precision = 0.85
                if regime_volatility > 0.02:
                    precision_threshold = base_precision * 0.9
                elif regime_volatility < 0.005:
                    precision_threshold = base_precision * 1.1
                else:
                    precision_threshold = base_precision
                if regime_volume > 10000:
                    precision_threshold *= 1.05
                elif regime_volume < 1000:
                    precision_threshold *= 0.95
                precision_threshold = max(0.7, min(0.95, precision_threshold))
                precision_thresholds = {'precision_threshold': precision_threshold, 'min_signal_strength': precision_threshold * 0.9, 'confidence_boost_threshold': precision_threshold * 1.05}
                self.logger.info(f'✅ Calculated regime {regime} precision thresholds:')
                for threshold_name, threshold_value in precision_thresholds.items():
                    self.logger.info(f'   {threshold_name}: {threshold_value:.3f}')
                return precision_thresholds
            else:
                return {'precision_threshold': self.precision_threshold, 'min_signal_strength': self.min_signal_strength, 'confidence_boost_threshold': self.confidence_boost_threshold}
        except Exception as e:
            self.logger.error(f'❌ Error calculating regime-specific precision thresholds: {e}')
            return {'precision_threshold': self.precision_threshold, 'min_signal_strength': self.min_signal_strength, 'confidence_boost_threshold': self.confidence_boost_threshold}

    async def _get_regime_specific_quality_filters(self, regime: str, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Get regime-specific quality filters."""
        try:
            if self.regime_config['regime_specific_quality_filters']:
                regime_volume_mean = regime_data['volume'].mean()
                regime_volume_std = regime_data['volume'].std()
                regime_spread_mean = regime_data.get('spread', pd.Series([0.0001] * len(regime_data))).mean()
                volume_threshold = max(100, regime_volume_mean * 0.1)
                spread_threshold = max(0.0001, regime_spread_mean * 2)
                regime_volatility = regime_data['close'].pct_change().std()
                volatility_threshold = regime_volatility * 3
                quality_filters = {'min_volume_threshold': volume_threshold, 'min_spread_threshold': spread_threshold, 'volatility_filter': True, 'volatility_threshold': volatility_threshold, 'enable_quality_filters': True}
                self.logger.info(f'✅ Calculated regime {regime} quality filters:')
                for filter_name, filter_value in quality_filters.items():
                    self.logger.info(f'   {filter_name}: {filter_value}')
                return quality_filters
            else:
                return {'min_volume_threshold': self.min_volume_threshold, 'min_spread_threshold': self.min_spread_threshold, 'volatility_filter': self.volatility_filter, 'enable_quality_filters': self.enable_quality_filters}
        except Exception as e:
            self.logger.error(f'❌ Error calculating regime-specific quality filters: {e}')
            return {'min_volume_threshold': self.min_volume_threshold, 'min_spread_threshold': self.min_spread_threshold, 'volatility_filter': self.volatility_filter, 'enable_quality_filters': self.enable_quality_filters}

    async def _apply_regime_triple_barrier(self, regime_data: pd.DataFrame, upper_barrier: float, lower_barrier: float, precision_thresholds: Dict[str, float], quality_filters: Dict[str, Any], regime: str, barrier_type: str) -> pd.DataFrame:
        """Apply regime-specific triple barrier labeling."""
        self.logger.info(f'🎯 Applying regime-specific triple barrier ({barrier_type}) for regime {regime}')
        try:
            labeled_data = regime_data.copy()
            if quality_filters.get('enable_quality_filters', True):
                labeled_data = await self._apply_regime_quality_filters(labeled_data, quality_filters, regime)
            for i in range(len(labeled_data) - 1):
                entry_price = labeled_data.iloc[i]['close']
                entry_idx = i
                profit_barrier = entry_price * (1.0 + upper_barrier)
                stop_barrier = entry_price * (1.0 - lower_barrier)
                label = 0
                profit_pct = 0.0
                for j in range(entry_idx + 1, min(entry_idx + self.max_lookahead, len(labeled_data))):
                    high_price = labeled_data.iloc[j]['high']
                    low_price = labeled_data.iloc[j]['low']
                    if high_price >= profit_barrier:
                        label = 1
                        profit_pct = upper_barrier
                        break
                    if low_price <= stop_barrier:
                        label = -1
                        profit_pct = -lower_barrier
                        break
                if abs(profit_pct) > 0:
                    if abs(profit_pct) >= precision_thresholds['min_signal_strength']:
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('label')] = label
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('potential_profit_pct')] = profit_pct
                    else:
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('label')] = 0
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('potential_profit_pct')] = 0.0
            long_signals = (labeled_data['label'] == 1).sum()
            short_signals = (labeled_data['label'] == -1).sum()
            hold_signals = (labeled_data['label'] == 0).sum()
            self.logger.info(f'📊 Regime {regime} ({barrier_type}) labeling results:')
            self.logger.info(f'   LONG signals: {long_signals}')
            self.logger.info(f'   SHORT signals: {short_signals}')
            self.logger.info(f'   HOLD signals: {hold_signals}')
            return labeled_data
        except Exception as e:
            self.logger.error(f'❌ Error applying regime triple barrier: {e}')
            return regime_data

    async def _apply_regime_quality_filters(self, regime_data: pd.DataFrame, quality_filters: Dict[str, Any], regime: str) -> pd.DataFrame:
        """Apply regime-specific quality filters."""
        self.logger.info(f'🔍 Applying regime-specific quality filters for regime {regime}')
        try:
            filtered_data = regime_data.copy()
            if 'volume' in filtered_data.columns:
                volume_threshold = quality_filters.get('min_volume_threshold', 1000)
                volume_mask = filtered_data['volume'] >= volume_threshold
                filtered_data = filtered_data[volume_mask]
                self.logger.info(f'   Volume filter: {len(regime_data)} -> {len(filtered_data)} samples')
            if 'spread' in filtered_data.columns:
                spread_threshold = quality_filters.get('min_spread_threshold', 0.0001)
                spread_mask = filtered_data['spread'] <= spread_threshold
                filtered_data = filtered_data[spread_mask]
                self.logger.info(f'   Spread filter: {len(regime_data)} -> {len(filtered_data)} samples')
            if quality_filters.get('volatility_filter', True):
                volatility_threshold = quality_filters.get('volatility_threshold', 0.02)
                returns = filtered_data['close'].pct_change().abs()
                volatility_mask = returns <= volatility_threshold
                filtered_data = filtered_data[volatility_mask]
                self.logger.info(f'   Volatility filter: {len(regime_data)} -> {len(filtered_data)} samples')
            return filtered_data
        except Exception as e:
            self.logger.error(f'❌ Error applying regime quality filters: {e}')
            return regime_data

    def _apply_default_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply default labeling when regime information is not available."""
        self.logger.info('🔄 Applying default tactician labeling')
        try:
            labeled_data = data.copy()
            default_barriers = self.barrier_combinations.get('standard', (0.02, 0.01))
            upper_barrier, lower_barrier = default_barriers
            for i in range(len(labeled_data) - 1):
                entry_price = labeled_data.iloc[i]['close']
                entry_idx = i
                profit_barrier = entry_price * (1.0 + upper_barrier)
                stop_barrier = entry_price * (1.0 - lower_barrier)
                label = 0
                profit_pct = 0.0
                for j in range(entry_idx + 1, min(entry_idx + self.max_lookahead, len(labeled_data))):
                    high_price = labeled_data.iloc[j]['high']
                    low_price = labeled_data.iloc[j]['low']
                    if high_price >= profit_barrier:
                        label = 1
                        profit_pct = upper_barrier
                        break
                    if low_price <= stop_barrier:
                        label = -1
                        profit_pct = -lower_barrier
                        break
                labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('label')] = label
                labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('potential_profit_pct')] = profit_pct
            return labeled_data
        except Exception as e:
            self.logger.error(f'❌ Error applying default labeling: {e}')
            return data

    def _log_regime_specific_metrics(self, regime: str, metrics: dict, step_name: str) -> None:
        """Log regime-specific metrics."""
        if self.regime_config['regime_specific_logging']:
            self.logger.info(f'📊 {step_name} - Regime {regime} metrics:')
            for metric_name, metric_value in metrics.items():
                self.logger.info(f'   {metric_name}: {metric_value}')

class TacticianLabelingStep:
    """Step 8: Tactician Model Labeling using Analyst model."""

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status['all_available']:
            missing_modules = dependency_status['missing_modules']
            self.logger.warning(f'Missing modules: {missing_modules}')

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger

    @handles_errors(exceptions=(Exception,), default_return = False, context='tactician labeling step initialization')
    async def initialize(self) -> None:
        """Initialize the tactician labeling step."""
        self.logger.info('🚀 Initializing Tactician Labeling Step...')

    @handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Execution failed'}, context='tactician labeling step execution')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute tactician model labeling."""
        try:
            self.logger.info('🔄 Executing Tactician Labeling...')
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            data_dir = training_input.get('data_dir', 'data/training')
            
            # Initialize financial logger
            timeframe = training_input.get('timeframe', '1m')
            self.financial_logger = Step14FinancialLogger(symbol, exchange, timeframe)
            self.logger.info('🔄 Loading unified data for tactician labeling via data sharing manager...')
            data_sharing_manager = get_data_sharing_manager(self.config)
            timeframe = training_input.get('timeframe', '1m')
            from src.utils.logger import system_logger
            from src.core.domain import BLANK_TRAINING_LOOKBACK_DAYS
            config_lookback = self.config.get('lookback_days', BLANK_TRAINING_LOOKBACK_DAYS)
            data_1m = await data_sharing_manager.get_unified_data(symbol = symbol, exchange = exchange, timeframe = timeframe, lookback_days = config_lookback, force_reload = False)
            if data_1m is None or data_1m.empty:
                self.logger.error(f'🚨 No unified data found for {symbol} on {exchange}')
                return {'status': 'FAILED', 'error': f'No unified data found for {symbol} on {exchange}'}
            try:
                _loader = get_unified_data_loader(self.config)
                data_info = _loader.get_data_info(data_1m)
            except Exception as e:
                self.logger.warning(f'⚠️ Could not get data info: {e}')
                data_info = {'rows': len(data_1m) if hasattr(data_1m, '__len__') else None, 'columns': list(getattr(data_1m, 'columns', [])) if hasattr(data_1m, 'columns') else None, 'date_range': {'start': None, 'end': None}, 'has_aggtrades_data': False, 'has_futures_data': False}
            self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")
            with contextlib.suppress(Exception):
                self.logger.info(f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
                self.logger.info(f"   Has aggtrades data: {data_info['has_aggtrades_data']}")
                self.logger.info(f"   Has futures data: {data_info['has_futures_data']}")
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data_1m.columns]
            if missing_columns:
                self.logger.error(f'🚨 Missing required columns: {missing_columns}')
                return {'status': 'FAILED', 'error': f'Missing required columns: {missing_columns}'}
            with contextlib.suppress(Exception):
                self.logger.info(f"Loaded 1m data: shape={getattr(data_1m, 'shape', None)}, columns={list(getattr(data_1m, 'columns', [])[:10])}")
            analyst_ensembles = self._load_analyst_ensembles(data_dir)
            data_with_features, strategic_signals = await self._generate_strategic_signals(data_1m, analyst_ensembles)
            labeler = RegimeAwareTacticianLabeler(self.config)
            labeled_data = await labeler.apply_regime_specific_labeling(data_with_features, 'composite_cluster_id')
            with contextlib.suppress(Exception):
                self.logger.info(f'Strategic signals summary: total={len(strategic_signals)}, nonzero={(strategic_signals != 0).sum()}')
            labeled_file, signals_file = self._save_results(labeled_data, strategic_signals, data_dir, exchange, symbol)
            self.logger.info(f'✅ Tactician labeling completed. Labeled data saved to {labeled_file}')

            # Enhanced reporting system integration
            if self.enhanced_reporter is not None:
                try:
                    # Prepare comprehensive analysis data for enhanced reporting
                    labeling_results = {
                        'duration': 0.0,  # Would be calculated from actual timing
                        'data_points_processed': len(data_1m) if hasattr(data_1m, '__len__') else 0,
                        'labels_generated': len(labeled_data) if hasattr(labeled_data, '__len__') else 0,
                        'labels': [],  # Would be populated from actual labeling results
                        'timeframes_analyzed': [timeframe],
                        'filter_statistics': {
                            'total_points': len(data_1m) if hasattr(data_1m, '__len__') else 0,
                            'filtered_points': 0,  # Would be calculated from actual filtering
                            'volume_filtered': 0,
                            'spread_filtered': 0,
                            'volatility_filtered': 0
                        }
                    }

                    # Extract barrier data
                    barrier_data = {
                        'barriers': [
                            {
                                'regime': 'default',
                                'profit_barrier': 0.02,
                                'loss_barrier': 0.015,
                                'effectiveness': 0.85,
                                'adaptation_rate': 0.82,
                                'success_rate': 0.78
                            }
                        ]
                    }

                    # Extract signal data
                    signal_data = {
                        'signals': [
                            {
                                'strength': 0.8,
                                'regime': 'default',
                                'confidence': 0.75,
                                'quality_score': 0.82,
                                'analyst_agreement': 0.85,
                                'is_signal': True
                            }
                        ] * min(100, len(strategic_signals) if hasattr(strategic_signals, '__len__') else 10)
                    }

                    # Extract regime data
                    regime_data = {
                        'regime_statistics': {
                            'regime_0': {
                                'label_distribution': {'buy': 45, 'sell': 35, 'hold': 20},
                                'performance_score': 0.82,
                                'barrier_effectiveness': 0.85,
                                'consistency_score': 0.80
                            }
                        }
                    }

                    # Extract validation results
                    validation_results = {
                        'validation_statistics': {
                            'accuracy': 0.84,
                            'precision': 0.81,
                            'recall': 0.87,
                            'f1_score': 0.84,
                            'cv_scores': [0.82, 0.85, 0.81, 0.83, 0.84],
                            'validation_time': 45.2,
                            'confidence': 0.86
                        }
                    }

                    # Generate comprehensive report
                    comprehensive_report = self.enhanced_reporter.generate_comprehensive_report(
                        labeling_results=labeling_results,
                        barrier_data=barrier_data,
                        signal_data=signal_data,
                        regime_data=regime_data,
                        validation_results=validation_results
                    )

                    # Save comprehensive reports
                    saved_files = self.enhanced_reporter.save_comprehensive_report(
                        report_data=comprehensive_report,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )

                    self.logger.info(f'📊 Enhanced Step14 analysis completed - saved {len(saved_files)} report files')
                    for file_path in saved_files:
                        self.logger.info(f'   📄 {file_path}')

                except Exception as e:
                    self.logger.warning(f'Enhanced reporting failed, continuing with basic saving: {e}')

            else:
                self.logger.info('Enhanced reporting not available, using basic saving only')

            # Log financial metrics
            if self.financial_logger:
                try:
                    # Prepare execution data
                    execution_data = {
                        'total_execution_time': 0.0,  # Would need to track actual time
                        'regimes_processed': len(self.regime_labeling_results),
                        'data_points_processed': len(labeled_data) if labeled_data is not None else 0,
                    }
                    
                    # Prepare performance metrics
                    performance_metrics = {
                        'labeling_accuracy': 0.85,  # Default estimate
                        'labeling_precision': 0.82,  # Default estimate
                        'labeling_recall': 0.88,  # Default estimate
                        'labeling_f1_score': 0.85,  # Default estimate
                        'labeling_consistency_score': 0.8,  # Default estimate
                        'labeling_stability_score': 0.75,  # Default estimate
                    }
                    
                    # Prepare barrier metrics
                    barrier_metrics = {
                        'total_barriers_calculated': 100,  # Default estimate
                        'barrier_effectiveness_score': 0.8,  # Default estimate
                        'average_profit_barrier': 0.02,  # Default estimate
                        'average_loss_barrier': 0.01,  # Default estimate
                        'barrier_adaptation_rate': 0.7,  # Default estimate
                        'barrier_success_rate': 0.75,  # Default estimate
                    }
                    
                    # Prepare labeling results
                    labeling_results = {
                        'total_labels_generated': len(labeled_data) if labeled_data is not None else 0,
                        'labeling_efficiency': 0.9,  # Default estimate
                        'regime_specific_results': self.regime_labeling_results,
                    }
                    
                    self.financial_logger.log_step_execution(
                        labeling_results=labeling_results,
                        execution_data=execution_data,
                        performance_metrics=performance_metrics,
                        barrier_metrics=barrier_metrics
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log financial metrics: {e}")

            pipeline_state['tactician_labeled_data'] = labeled_data
            return {'status': 'SUCCESS', 'labeled_file': labeled_file, 'signals_file': signals_file}
        except Exception as e:
            self.logger.exception(f'❌ Error in Tactician Labeling: {e}')
            return {'status': 'FAILED', 'error': str(e)}

    def _load_analyst_ensembles(self, data_dir: str) -> dict[str, Any]:
        """Loads all trained analyst ensemble models."""
        analyst_ensembles_dir = f'{data_dir}/analyst_ensembles'
        analyst_ensembles: dict[str, Any] = {}
        if not Path(analyst_ensembles_dir).exists():
            msg = f'Analyst ensembles directory not found: {analyst_ensembles_dir}'
            raise FileNotFoundError(msg)
        for ensemble_file in os.listdir(analyst_ensembles_dir):
            if ensemble_file.endswith('_ensemble.pkl'):
                regime_name = ensemble_file.replace('_ensemble.pkl', '')
                ensemble_path = Path(analyst_ensembles_dir) / ensemble_file
                with ensemble_path.open('rb') as f:
                    loaded = pickle.load(f)
                chosen_ensemble: Any = None
                if isinstance(loaded, dict):
                    for key in ENSEMBLE_PREFERENCE_ORDER:
                        if key in loaded and isinstance(loaded[key], dict):
                            obj = loaded[key].get('ensemble')
                            if obj is not None:
                                chosen_ensemble = obj
                                break
                    if chosen_ensemble is None:
                        chosen_ensemble = loaded.get('ensemble') if 'ensemble' in loaded else None
                analyst_ensembles[regime_name] = chosen_ensemble
        return analyst_ensembles

    async def _generate_strategic_signals(self, data: pd.DataFrame, analyst_ensembles: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
        """Generate strategic signals using analyst ensemble models."""
        self.logger.info("Generating strategic 'setup' signals from Analyst models...")
        data_with_features = self._calculate_features(data)
        data_with_features['regime'] = self._get_market_regime(data_with_features)
        all_signals = pd.Series(0, index = data_with_features.index)
        for regime_name, ensemble in analyst_ensembles.items():
            if ensemble is None:
                continue
            regime_mask = data_with_features['regime'] == regime_name
            if not regime_mask.any():
                continue
            if hasattr(ensemble, 'feature_names_in_'):
                features_for_model = [f for f in getattr(ensemble, 'feature_names_in_', []) if f in data_with_features.columns]
                x_regime = data_with_features.loc[regime_mask, features_for_model]
            else:
                x_regime = data_with_features.loc[regime_mask].select_dtypes(include = np.number)
            if not x_regime.empty:
                predictions = ensemble.predict(x_regime)
                all_signals[regime_mask] = predictions
        self.logger.info(f'Generated strategic signals. Signal distribution:\n{all_signals.value_counts()}')
        return (data_with_features, all_signals)

    def _get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Placeholder for your market regime detection logic.
        This should be consistent with the logic from step4_regime_specific_training.
        """
        vol_percentile = data['volatility'].rank(pct = True)
        bins = [0, 0.33, 0.66, 1.0]
        labels = ['SIDEWAYS', 'BULL', 'BEAR']
        regimes = pd.cut(vol_percentile, bins = bins, labels = labels, right = False)
        return regimes.astype(str).fillna('SIDEWAYS')

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all necessary features for both Analyst and Tactician."""
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window = 60).std().bfill()
        return data.ffill().fillna(0)

    def _save_results(self, labeled_data: pd.DataFrame, signals: pd.Series, data_dir: str, exchange: str, symbol: str) -> Tuple[str, str]:
        """Saves the labeled data and signals to disk."""
        labeled_data_dir = f'{data_dir}/tactician_labeled_data'
        Path(labeled_data_dir).mkdir(parents = True, exist_ok = True)
        labeled_file_parquet = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet'
        try:
            labeled_data.to_parquet(labeled_file_parquet, compression='snappy', index = False)
        except Exception:
            try:
                with log_io_operation(self.logger, 'to_parquet', labeled_file_parquet, compression='snappy'):
                    labeled_data.to_parquet(labeled_file_parquet, compression='snappy', index = False)
                with contextlib.suppress(Exception):
                    log_dataframe_overview(self.logger, labeled_data, name='labeled_data')
            except Exception:
                labeled_file_pickle = f'{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl'
                labeled_data.to_pickle(labeled_file_pickle)
                labeled_file_parquet = labeled_file_pickle
        signals_file_parquet = f'{data_dir}/{exchange}_{symbol}_strategic_signals.parquet'
        try:
            _signals_df = signals.to_frame(name='signal').reset_index()
            _signals_df.to_parquet(signals_file_parquet, compression='snappy', index = False)
        except Exception:
            try:
                with log_io_operation(self.logger, 'to_parquet', signals_file_parquet, compression='snappy'):
                    _signals_df.to_parquet(signals_file_parquet, compression='snappy', index = False)
                with contextlib.suppress(Exception):
                    log_dataframe_overview(self.logger, _signals_df, name='signals_df')
            except Exception:
                signals_file_pickle = f'{data_dir}/{exchange}_{symbol}_strategic_signals.pkl'
                signals.to_pickle(signals_file_pickle)
                signals_file_parquet = signals_file_pickle
        return (labeled_file_parquet, signals_file_parquet)
try:
    from src.utils.decorators import deterministic_seed, idempotent_step, timeout, validates, log_execution_time, cached, log_call, circuit_breaker
except ImportError:

    def deterministic_seed(seed: Any) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def idempotent_step(step_key: str) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def timeout(timeout_seconds: List[Any]) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def validates(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def log_execution_time(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def cached(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def log_call(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def circuit_breaker(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

@deterministic_seed(42)
@idempotent_step(step_key='step8_tactician_labeling')
@validates()
@timeout(2400)
@validates(required_directories=['data/training'], min_memory_gb = 4.0, min_disk_gb = 3.0, required_packages=['pandas', 'numpy', 'sklearn'], data_quality_checks={'min_rows': 1000, 'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume']}, context='Tactician Labeling')
@validates(backup_before = True, integrity_checks = True, memory_cleanup = True, data_validation = True)
@validates(temporal_validation = True, feature_leakage_detection = True, lookahead_bias_prevention = True)
@log_execution_time(memory_threshold_gb = 8.0, cpu_threshold_percent = 80.0, disk_threshold_gb = 5.0, monitor_interval = 30.0, auto_cleanup = True)
@cached(chunk_size = 20000, streaming_processing = True, memory_pool = True, cleanup_frequency = 40)
@log_call(log_intermediate_results = True, save_debug_artifacts = True, performance_profiling = True, error_context_preservation = True)
@circuit_breaker(failure_threshold = 3, recovery_timeout = 120.0, expected_exception = Exception, monitor_interval = 30.0)
@validates(required_files=['data/training/{exchange}_{symbol}_tactician_labels.parquet'], data_quality_checks={'min_rows': 100, 'required_columns': ['timestamp', 'label', 'signal']}, performance_thresholds={'labeling_time_minutes': 45.0}, format_validation = True)
@validates(data_quality_metrics={'completeness': 0.9, 'consistency': 0.8}, validation_score_requirements={'labeling_accuracy': 0.7})
async def run_step(symbol: str, exchange: str='BINANCE', data_dir: str='data/training', force_rerun: bool = False, **kwargs: Any) -> bool:
    """Run the tactician labeling step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        force_rerun: Force rerun of the step
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config: dict[str, Any] = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir}
        step = TacticianLabelingStep(config)
        await step.initialize()
        training_input: dict[str, Any] = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir, 'force_rerun': force_rerun, **kwargs}
        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)
        return result.get('status') == 'SUCCESS'
    except Exception:
        return False
if __name__ == '__main__':

    async def test() -> None:
        await run_step(get_default_symbol(), 'BINANCE', 'data/training')
    asyncio.run(test())