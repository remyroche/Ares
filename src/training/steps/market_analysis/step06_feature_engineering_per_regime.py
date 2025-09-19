from src.utils.tprint import tprint

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import time
from src.core.decorators import traced, validates, handles_errors
from src.utils.data.klines_parquet import get_klines_manager
from src.training.steps.market_analysis.enhanced_validation_framework import EnhancedValidator, ValidationLevel
from src.utils.enhanced_error_handler import ErrorRecord, ErrorContext
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

"""Enhanced Step 6: Per-Regime Feature Engineering.

This module provides per-HMM regime feature engineering functionality, ensuring that
features are engineered specifically for each regime's characteristics.
"""
import asyncio
import logging
from pathlib import Path
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Import feature generation optimization
try:
    from src.feature_generation.utils.feature_generation_optimization import (
        FeatureGenerationOptimizer, 
        FeatureOptimizationConfig, 
        OptimizationMethod
    )
    FEATURE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    FEATURE_OPTIMIZATION_AVAILABLE = False

class FeatureInteractionEngine:

        @log_important_calls
        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config
            self.logger = logging.getLogger(__name__)

        async def create_interactions(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Any:
            return data
try:
    from .regime_handler import regime_handler
except ImportError:

    def regime_handler(*args, **kwargs) -> None:
        return {}
try:
    from .regime_processing_decorator import per_regime_processing, aggregate_regime_results, RegimeProcessingContext
except ImportError:

    def per_regime_processing(*args, **kwargs):
        """Fallback decorator for per-regime processing."""
        def decorator(func: Callable):
            return func
        return decorator

    def aggregate_regime_results(*args, **kwargs):
        """Fallback function for aggregating regime results."""
        return {}

    class RegimeProcessingContext:
        """Fallback context manager for regime processing."""

        @log_important_calls
        def __init__(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
            self.symbol = symbol
            self.exchange = exchange
            self.timeframe = timeframe
            self.data_dir = data_dir
            self.regime_data = None
            self.regime_ids = []
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False
            
        def get_regime_data(self, regime_id: int, preserve_context: bool = False):
            return pd.DataFrame()
try:
    from src.utils.pipeline_standards import pipeline_standards
    import datetime
except ImportError:
    def pipeline_standards(*args, **kwargs) -> Dict[str, Any]:
        """Fallback pipeline standards."""
        return {
            'data_quality_threshold': 0.95,
            'min_samples': 100,
            'max_missing_ratio': 0.05
        }

from src.core.decorators.logging import log_execution_time, log_call

try:
    from src.utils.logger import get_logger
    logger = get_logger('Step6FeatureEngineeringPerRegime')
except ImportError:
    logger = logging.getLogger(__name__)

class PerRegimeFeatureEngineeringStep(FeatureInteractionEngine):
    """Enhanced feature engineering step that processes each regime separately."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_feature_engineering', True)
        self.regime_specific_features = config.get('regime_specific_features', {})
        self.adaptive_lookback = config.get('adaptive_lookback_per_regime', True)
        step6_config = config.get('step06_feature_engineering', {})
        step6_config['force_regime_specific_periods'] = True
        config['step06_feature_engineering'] = step6_config
        self.config = config
        self.force_regime_specific_periods = True
        self.validator = EnhancedValidator()
        
        # Initialize feature optimization
        if FEATURE_OPTIMIZATION_AVAILABLE:
            optimization_config = FeatureOptimizationConfig(
                optimization_method=OptimizationMethod.REGIME_AWARE,
                regime_aware=True,
                parallel_processing=True
            )
            self.feature_optimizer = FeatureGenerationOptimizer(optimization_config)
            self.logger.info('✅ Feature generation optimizer initialized')
        else:
            self.feature_optimizer = None
            self.logger.warning('⚠️ Feature generation optimizer not available')
        
        self.logger.info('🎯 Per-regime feature engineering initialized with regime-specific optimization enabled')

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="per-regime feature engineering"
    )
    @traced(span_name='execute_per_regime_feature_engineering')
    async def execute_per_regime_feature_engineering(self, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool = False) -> bool:
        """Execute feature engineering on a per-regime basis.
        
        Each regime may have different market dynamics, so features should be
        engineered specifically for each regime's characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            Success status
        """
        try:
            # Comprehensive input validation
            if not symbol or not isinstance(symbol, str) or len(symbol.strip()) == 0:
                raise ValueError("Symbol must be a non-empty string")
            
            if not exchange or not isinstance(exchange, str) or len(exchange.strip()) == 0:
                raise ValueError("Exchange must be a non-empty string")
            
            if not timeframe or not isinstance(timeframe, str) or len(timeframe.strip()) == 0:
                raise ValueError("Timeframe must be a non-empty string")
            
            if not data_dir or not isinstance(data_dir, str):
                raise ValueError("Data directory must be a non-empty string")
            
            # Validate data directory exists and is accessible
            data_path = Path(data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
            
            if not data_path.is_dir():
                raise ValueError(f"Data path is not a directory: {data_dir}")
            
            # Check directory permissions
            if not data_path.readable():
                raise PermissionError(f"Data directory is not readable: {data_dir}")
            
            # Sanitize inputs
            symbol = symbol.strip().upper()
            exchange = exchange.strip().upper()
            timeframe = timeframe.strip().lower()
            
            self.logger.info('🚀 Starting per-regime feature engineering process')
            labeled_data = await self._load_labeled_data(symbol, exchange, timeframe, data_dir)
            
            if labeled_data is None:
                raise ValueError("Failed to load labeled data")
            
            # Validate labeled data quality
            validation_result = await self.validator.validate_data_quality(
                labeled_data, ValidationLevel.CRITICAL, "feature_generation"
            )
            
            if not validation_result.passed:
                raise ValueError(f"Labeled data quality validation failed: {validation_result.message}")
            
            self.logger.info(f'✅ Labeled data validation passed: {len(labeled_data)} rows, {len(labeled_data.columns)} columns')
            async with RegimeProcessingContext(symbol, exchange, timeframe, data_dir) as ctx:
                if ctx.regime_data is None:
                    self.logger.error('❌ Failed to load regime data')
                    return False
                self.logger.info(f'📊 Engineering features for {len(ctx.regime_ids)} regimes')
                regime_results = {}
                regime_feature_info = {}
                for regime_id in ctx.regime_ids:
                    self.logger.info(f'🔄 Processing regime {regime_id}')
                    regime_config = self._get_regime_feature_config(regime_id)
                    result, feature_info = await self._engineer_features_single_regime(ctx = ctx, regime_id = regime_id, labeled_data = labeled_data, regime_config = regime_config)
                    if result is not None:
                        regime_results[regime_id] = result
                        regime_feature_info[regime_id] = feature_info
                        self.logger.info(f'✅ Successfully engineered features for regime {regime_id}')
                    else:
                        self.logger.error(f'❌ Failed to engineer features for regime {regime_id}')
                success = await regime_handler.save_regime_results(results = regime_results, step_name='step06_feature_engineering', symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, result_type='feature_engineered_data')
                
                if not success:
                    raise RuntimeError("Failed to save regime results")
                
                await self._save_regime_feature_metadata(regime_feature_info, symbol, exchange, timeframe, data_dir)
                aggregated = self._aggregate_regime_features(regime_results)
                
                if aggregated is None or aggregated.empty:
                    raise ValueError("Feature aggregation produced no results")
                
                output_path = Path(data_dir) / exchange.lower() / symbol.lower() / 'processed' / f'{symbol.lower()}_{timeframe}_features_per_regime.parquet'
                standardized_parquet_handler.write_parquet_standardized(aggregated, output_path, index=False)
                self.logger.info(f'✅ Saved aggregated feature data: {output_path}')
                
                # Validate expected outputs were created
                expected_outputs = [
                    f'{exchange}_{symbol}_{timeframe}_features_per_regime.parquet',
                    f'{exchange}_{symbol}_{timeframe}_feature_metadata.json'
                ]
                
                validation_result = await self.validator.validate_process_completion(
                    'feature_generation', expected_outputs, str(Path(data_dir) / exchange.lower() / symbol.lower() / 'processed'), ValidationLevel.CRITICAL
                )
                
                if not validation_result.passed:
                    raise RuntimeError(
                        f"Feature generation completed but validation failed: {validation_result.message}"
                    )
                
                self._log_feature_statistics(aggregated, regime_feature_info)
                self.logger.info('✅ Feature generation completed successfully')
                return True
        except RuntimeError as e:
            self.logger.critical(f'🚨 CRITICAL PROCESS ERROR in Feature Generation: {e}')
            # Re-raise to trigger fail-fast behavior
            raise
        except Exception as e:
            self.logger.critical(f'🚨 CRITICAL ERROR in Feature Generation: {e}')
            
            # Convert to RuntimeError for fail-fast behavior
            raise RuntimeError(
                f"Feature generation failed with critical error: {e}"
            )

    async def _engineer_features_single_regime(self, ctx: RegimeProcessingContext, regime_id: int, labeled_data: pd.DataFrame, regime_config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Engineer features for a single regime.
        
        Args:
            ctx: Regime processing context
            regime_id: Regime ID to process
            labeled_data: Labeled data from previous step
            regime_config: Configuration for this regime
            
        Returns:
            Tuple of (feature engineered DataFrame, feature metadata)
        """
        try:
            regime_data = ctx.get_regime_data(regime_id, preserve_context = True)
            if regime_data.empty:
                self.logger.warning(f'⚠️ No data for regime {regime_id}')
                return (None, None)
            regime_labeled = pd.merge(regime_data, labeled_data[['timestamp', 'label', 'label_type']], on='timestamp', how='left')
            if 'is_regime_context' in regime_labeled.columns:
                context_mask = regime_labeled['is_regime_context']
                regime_labeled = regime_labeled.drop(columns=['is_regime_context'])
            else:
                context_mask = pd.Series(False, index = regime_labeled.index)
            if self.adaptive_lookback:
                await self._optimize_regime_lookback_periods(regime_id, regime_labeled, context_mask, regime_config)
            features_df = await self._apply_feature_engineering(regime_labeled, regime_config)
            if features_df is None:
                return (None, None)
            features_df['feature_regime_id'] = regime_id
            feature_info = {'regime_id': regime_id, 'num_features': len([c for c in features_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]), 'feature_names': [c for c in features_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']], 'regime_config': regime_config, 'data_shape': features_df.shape, 'context_rows': int(context_mask.sum()) if context_mask is not None else 0, 'optimization_info': {'adaptive_lookback_enabled': self.adaptive_lookback, 'optimized_periods': regime_config.get('optimized_periods', {}), 'optimization_priority': regime_config.get('optimization_priority', 'unknown'), 'emphasis': regime_config.get('emphasis', 'unknown')}}
            return (features_df, feature_info)
        except Exception as e:
            self.logger.error(f'❌ Error engineering features for regime {regime_id}: {e}')
            return (None, None)
    @log_all_calls

    def _get_regime_feature_config(self, regime_id: int) -> Dict[str, Any]:
        """Get feature engineering configuration for a specific regime.
        
        Different regimes may benefit from different feature sets and parameters.
        This method creates regime-specific configurations that will be used
        to optimize lookback periods and feature interactions.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific feature configuration
        """
        if f'regime_{regime_id}' in self.regime_specific_features:
            custom_config = self.regime_specific_features[f'regime_{regime_id}']
            self.logger.info(f'📋 Using custom configuration for regime {regime_id}')
            return custom_config
        base_config = {
            'enable_technical_indicators': True, 
            'enable_price_features': True, 
            'enable_volume_features': True, 
            'enable_volatility_features': True, 
            'enable_microstructure_features': True, 
            'force_regime_specific_periods': True, 
            'regime_id': regime_id
        }
        
        # Get regime characteristics dynamically instead of hard-coding
        regime_characteristics = self._analyze_regime_characteristics(regime_id)
        
        if regime_characteristics['type'] == 'trending':
            config = {
                **base_config, 
                'lookback_periods': [10, 20, 50, 100, 200], 
                'emphasis': 'trend', 
                'additional_features': ['SMA_cross_features', 'EMA_ribbon', 'ADX_features', 'trend_strength'], 
                'interaction_patterns': {
                    'trend_momentum': {
                        'features': ['SMA_20', 'SMA_100', 'EMA_21', 'ADX_14'], 
                        'weight': 2.0, 
                        'enabled': True
                    }, 
                    'trend_volume': {
                        'features': ['OBV_20', 'Volume_Ratio', 'SMA_20', 'ATR_14'], 
                        'weight': 1.8, 
                        'enabled': True
                    }
                }, 
                'optimization_priority': 'trend_strength',
                'volatility_threshold': regime_characteristics.get('volatility_threshold', 0.02),
                'trend_strength': regime_characteristics.get('trend_strength', 0.7)
            }
            self.logger.info(f'📈 Configured regime {regime_id} for trending markets (volatility: {regime_characteristics.get("avg_volatility", "N/A"):.4f})')
            
        elif regime_characteristics['type'] == 'volatile':
            config = {
                **base_config, 
                'lookback_periods': [5, 10, 20, 30], 
                'emphasis': 'mean_reversion', 
                'additional_features': ['RSI_divergence', 'Bollinger_bands_features', 'ATR_bands', 'volatility_cones'], 
                'interaction_patterns': {
                    'mean_reversion': {
                        'features': ['RSI_14', 'BB_Position_20', 'Williams_R_14', 'CCI_20'], 
                        'weight': 2.2, 
                        'enabled': True
                    }, 
                    'volatility_regime': {
                        'features': ['ATR_14', 'BB_Squeeze_20', 'Volatility', 'Volume_Ratio'], 
                        'weight': 1.9, 
                        'enabled': True
                    }
                }, 
                'optimization_priority': 'volatility_capture',
                'volatility_threshold': regime_characteristics.get('volatility_threshold', 0.05),
                'mean_reversion_strength': regime_characteristics.get('mean_reversion_strength', 0.6)
            }
            self.logger.info(f'📊 Configured regime {regime_id} for volatile/ranging markets (volatility: {regime_characteristics.get("avg_volatility", "N/A"):.4f})')
            
        else:  # balanced/transitional
            config = {
                **base_config, 
                'lookback_periods': [7, 14, 30, 60], 
                'emphasis': 'balanced', 
                'additional_features': ['momentum_features', 'volume_profile', 'market_microstructure'], 
                'interaction_patterns': {
                    'momentum_volume': {
                        'features': ['RSI_14', 'MACD_12_26', 'OBV_20', 'Volume_Ratio'], 
                        'weight': 1.6, 
                        'enabled': True
                    }, 
                    'oscillator_trend': {
                        'features': ['RSI_14', 'Williams_R_14', 'CCI_20', 'EMA_21'], 
                        'weight': 1.4, 
                        'enabled': True
                    }
                }, 
                'optimization_priority': 'balanced_performance',
                'regime_stability': regime_characteristics.get('stability_score', 0.5)
            }
            self.logger.info(f'⚖️ Configured regime {regime_id} for balanced approach (stability: {regime_characteristics.get("stability_score", "N/A"):.4f})')
        return config

    def _analyze_regime_characteristics(self, regime_id: int) -> Dict[str, Any]:
        """Analyze regime characteristics dynamically instead of using hard-coded assumptions.
        
        Args:
            regime_id: Regime ID to analyze
            
        Returns:
            Dictionary with regime characteristics
        """
        try:
            # Try to load regime metadata if available
            regime_metadata = self._load_regime_metadata(regime_id)
            if regime_metadata:
                return regime_metadata
            
            # Fallback to heuristic analysis based on regime ID patterns
            # This is still better than hard-coding as it's based on general patterns
            characteristics = {
                'regime_id': regime_id,
                'type': 'balanced',  # default
                'avg_volatility': 0.02,  # default moderate volatility
                'volatility_threshold': 0.02,
                'stability_score': 0.5,
                'trend_strength': 0.5,
                'mean_reversion_strength': 0.5,
                'analysis_method': 'heuristic_fallback'
            }
            
            # Heuristic patterns (better than hard-coding specific IDs)
            total_regimes = self._estimate_total_regimes()
            
            if total_regimes > 0:
                regime_percentile = regime_id / total_regimes
                
                # Low regime IDs often represent stable/trending conditions
                if regime_percentile < 0.3:
                    characteristics.update({
                        'type': 'trending',
                        'avg_volatility': 0.015,
                        'volatility_threshold': 0.02,
                        'trend_strength': 0.8,
                        'stability_score': 0.7
                    })
                # High regime IDs often represent volatile/ranging conditions
                elif regime_percentile > 0.7:
                    characteristics.update({
                        'type': 'volatile',
                        'avg_volatility': 0.045,
                        'volatility_threshold': 0.05,
                        'mean_reversion_strength': 0.7,
                        'stability_score': 0.3
                    })
                # Middle regimes are transitional/balanced
                else:
                    characteristics.update({
                        'type': 'balanced',
                        'avg_volatility': 0.025,
                        'volatility_threshold': 0.03,
                        'stability_score': 0.5
                    })
            
            self.logger.info(f'🔍 Analyzed regime {regime_id} characteristics: {characteristics["type"]} (method: {characteristics["analysis_method"]})')
            return characteristics
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to analyze regime {regime_id} characteristics: {e}')
            # Return safe defaults
            return {
                'regime_id': regime_id,
                'type': 'balanced',
                'avg_volatility': 0.02,
                'volatility_threshold': 0.02,
                'stability_score': 0.5,
                'analysis_method': 'error_fallback'
            }
    
    def _load_regime_metadata(self, regime_id: int) -> Optional[Dict[str, Any]]:
        """Load regime metadata if available from previous analysis.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Regime metadata dictionary or None
        """
        try:
            # Try to load from various possible sources
            metadata_paths = [
                f'data/regimes/regime_{regime_id}_metadata.json',
                f'data/training/regime_{regime_id}_characteristics.json',
                f'data/hmm/regime_{regime_id}_analysis.json'
            ]
            
            for metadata_path in metadata_paths:
                if Path(metadata_path).exists():
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Validate metadata structure
                    if isinstance(metadata, dict) and 'type' in metadata:
                        self.logger.info(f'✅ Loaded regime {regime_id} metadata from {metadata_path}')
                        metadata['analysis_method'] = 'metadata_file'
                        return metadata
            
            return None
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to load regime {regime_id} metadata: {e}')
            return None
    
    def _estimate_total_regimes(self) -> int:
        """Estimate total number of regimes in the system.
        
        Returns:
            Estimated total number of regimes
        """
        try:
            # Try to determine from configuration
            if hasattr(self, 'config') and isinstance(self.config, dict):
                hmm_config = self.config.get('hmm_clustering', {})
                if 'n_components' in hmm_config:
                    return hmm_config['n_components']
                if 'max_regimes' in hmm_config:
                    return hmm_config['max_regimes']
            
            # Try to determine from regime data files
            regime_files = list(Path('data/regimes').glob('regime_*_metadata.json')) if Path('data/regimes').exists() else []
            if regime_files:
                regime_ids = []
                for file_path in regime_files:
                    try:
                        regime_id = int(file_path.stem.split('_')[1])
                        regime_ids.append(regime_id)
                    except (IndexError, ValueError):
                        continue
                if regime_ids:
                    return max(regime_ids) + 1  # Assuming 0-based indexing
            
            # Default fallback
            return 5
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to estimate total regimes: {e}')
            return 5  # Safe default

    @log_all_calls

    def _update_regime_interaction_patterns(self, regime_config: Dict[str, Any], regime_id: int) -> None:
        """Update interaction patterns with regime-specific optimized periods.
        
        Args:
            regime_config: Regime configuration dictionary
            regime_id: Regime ID
        """
        try:
            optimized_periods = regime_config.get('optimized_periods', {})
            if not optimized_periods:
                self.logger.warning(f'⚠️ No optimized periods available for regime {regime_id}')
                return
            interaction_patterns = regime_config.get('interaction_patterns', {})
            for pattern_name, pattern_config in interaction_patterns.items():
                updated_features = []
                for feature in pattern_config.get('features', []):
                    base_indicator = feature.split('_')[0]
                    if base_indicator in optimized_periods:
                        optimized_period = optimized_periods[base_indicator].get('selected_periods', [None])[0]
                        if optimized_period:
                            if '_' in feature:
                                parts = feature.split('_')
                                parts[1] = str(optimized_period)
                                updated_feature = '_'.join(parts)
                            else:
                                updated_feature = f'{base_indicator}_{optimized_period}'
                            updated_features.append(updated_feature)
                            self.logger.debug(f'🔄 Updated {feature} -> {updated_feature} for regime {regime_id}')
                        else:
                            updated_features.append(feature)
                    else:
                        updated_features.append(feature)
                pattern_config['features'] = updated_features
            regime_config['interaction_patterns'] = interaction_patterns
            self.logger.info(f'✅ Updated interaction patterns for regime {regime_id} with optimized periods')
        except Exception as e:
            self.logger.error(f'❌ Error updating interaction patterns for regime {regime_id}: {e}')

    async def _apply_feature_engineering(self, regime_data: pd.DataFrame, regime_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Apply feature engineering to regime data.
        
        Args:
            regime_data: Regime-specific data
            regime_config: Regime configuration
            
        Returns:
            Feature engineered DataFrame or None
        """
        try:
            self.logger.info(f"🔧 Applying feature engineering for regime {regime_config.get('regime_id', 'unknown')}")
            technical_features = self.extract_optimal_technical_indicators(regime_data)
            if technical_features.empty:
                self.logger.warning('⚠️ No technical features extracted')
                return None
            features_df = pd.concat([regime_data, technical_features], axis = 1)
            interaction_patterns = regime_config.get('interaction_patterns', {})
            if interaction_patterns:
                self.logger.info(f'🔄 Applying {len(interaction_patterns)} interaction patterns')
                feature_names = [col for col in features_df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']]
                interaction_features = self.extract_interaction_features(features_df[feature_names].values, feature_names, regime_data)
                if interaction_features is not None and interaction_features.size > 0:
                    interaction_names = [f'interaction_{i}' for i in range(interaction_features.shape[1])]
                    interaction_df = pd.DataFrame(interaction_features, index = features_df.index, columns = interaction_names)
                    features_df = pd.concat([features_df, interaction_df], axis = 1)
                    self.logger.info(f'✅ Added {len(interaction_names)} interaction features')
            features_df['regime_emphasis'] = regime_config.get('emphasis', 'unknown')
            features_df['optimization_priority'] = regime_config.get('optimization_priority', 'unknown')
            self.logger.info(f'✅ Feature engineering completed: {features_df.shape[1]} total features')
            return features_df
        except Exception as e:
            self.logger.error(f'❌ Error applying feature engineering: {e}')
            return None
    @log_all_calls

    def _validate_regime_optimization(self, regime_id: int, optimization_results: Dict[str, Any], regime_config: Dict[str, Any]) -> bool:
        """Validate that regime-specific optimization is working correctly.
        
        Args:
            regime_id: Regime ID
            optimization_results: Optimization results
            regime_config: Regime configuration
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            validation_passed = True
            validation_issues = []
            if optimization_results.get('status') != 'optimized':
                validation_issues.append(f"Optimization status: {optimization_results.get('status', 'unknown')}")
                validation_passed = False
            regime_specific_periods = optimization_results.get('optimization_results', {}).get('regime_specific_periods', {})
            regime_key = f'regime_{regime_id}'
            if regime_key not in regime_specific_periods:
                validation_issues.append(f'No regime-specific periods found for {regime_key}')
            optimized_periods = regime_config.get('optimized_periods', {})
            if not optimized_periods:
                validation_issues.append('No optimized periods in regime config')
                validation_passed = False
            interaction_patterns = regime_config.get('interaction_patterns', {})
            if not interaction_patterns:
                validation_issues.append('No interaction patterns configured')
                validation_passed = False
            if validation_passed:
                self.logger.info(f'✅ Regime {regime_id} optimization validation passed')
            else:
                self.logger.warning(f'⚠️ Regime {regime_id} optimization validation issues: {validation_issues}')
            return validation_passed
        except Exception as e:
            self.logger.error(f'❌ Error validating regime {regime_id} optimization: {e}')
            return False

    async def _load_labeled_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load labeled data from previous step.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Labeled DataFrame or None
        """
        try:
            labeled_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_labeled_per_regime.parquet'
            if not labeled_path.exists():
                labeled_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_labeled.parquet'
            if labeled_path.exists():
                data = standardized_parquet_handler.read_parquet_standardized(labeled_path)

                # 🔧 INTEGRATE DATA CLEANING UTILITY
                # Clean corrupted data before feature engineering to prevent infinity values
                try:
                    from src.utils.ml_common.data_processing.data_cleaning_utils import exclude_corrupted_periods

                    # Ensure datetime column exists
                    if 'timestamp' in data.columns and data['timestamp'].dtype == 'int64':
                        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                    elif 'datetime' not in data.columns:
                        # Try to infer datetime column
                        datetime_cols = [col for col in data.columns if 'time' in col.lower()]
                        if datetime_cols:
                            data['datetime'] = pd.to_datetime(data[datetime_cols[0]])
                        else:
                            data['datetime'] = data.index

                    # Apply data cleaning
                    original_count = len(data)
                    data = exclude_corrupted_periods(data)
                    cleaned_count = len(data)

                    if original_count != cleaned_count:
                        excluded_count = original_count - cleaned_count
                        self.logger.info(f"🧹 Feature Engineering Data cleaning applied: Excluded {excluded_count:,} corrupted rows ({100*excluded_count/original_count:.4f}%)")

                except ImportError as e:
                    self.logger.warning(f"⚠️ Data cleaning utility not available for feature engineering: {e}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Data cleaning failed for feature engineering, proceeding with original data: {e}")

                self.logger.info(f'✅ Loaded labeled data: {len(data)} rows')
                return data
            else:
                self.logger.error(f'❌ Labeled data not found: {labeled_path}')
                return None
        except Exception as e:
            self.logger.error(f'❌ Error loading labeled data: {e}')
            return None
    @log_all_calls

    def _aggregate_regime_features(self, regime_results: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate per-regime feature results intelligently.
        
        Args:
            regime_results: Dictionary of regime results
            
        Returns:
            Aggregated DataFrame with all features
        """
        if not regime_results:
            return pd.DataFrame()
        all_columns = set()
        for df in regime_results.values():
            if df is not None:
                all_columns.update(df.columns)
        base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'composite_cluster_id', 'feature_regime_id']
        dfs = []
        for regime_id, df in regime_results.items():
            if df is not None and (not df.empty):
                for col in all_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        aggregated = pd.concat(dfs, ignore_index = True)
        aggregated = aggregated.sort_values('timestamp').reset_index(drop = True)
        return aggregated

    async def _save_regime_feature_metadata(self, regime_feature_info: Dict[int, Dict[str, Any]], symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
        """Save metadata about regime-specific features.
        
        Args:
            regime_feature_info: Feature information for each regime
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            metadata = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'total_regimes': len(regime_feature_info), 'regime_features': regime_feature_info, 'timestamp': pd.Timestamp.now().isoformat()}
            metadata_path = Path(data_dir) / exchange.lower() / symbol.lower() / 'processed' / f'{symbol.lower()}_{timeframe}_regime_features_metadata.json'
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent = 2)
            self.logger.info(f'✅ Saved regime feature metadata: {metadata_path}')
        except Exception as e:
            self.logger.error(f'❌ Error saving feature metadata: {e}')
    @log_all_calls

    def _log_feature_statistics(self, aggregated_data: pd.DataFrame, regime_feature_info: Dict[int, Dict[str, Any]]) -> None:
        """Log statistics about the engineered features.
        
        Args:
            aggregated_data: Aggregated feature data
            regime_feature_info: Feature information per regime
        """
        try:
            self.logger.info('📊 Feature Engineering Statistics:')
            self.logger.info(f'   Total samples: {len(aggregated_data)}')
            feature_cols = [c for c in aggregated_data.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'composite_cluster_id', 'feature_regime_id', 'label']]
            self.logger.info(f'   Total features: {len(feature_cols)}')
            self.logger.info('📊 Per-Regime Feature Statistics:')
            for regime_id, info in regime_feature_info.items():
                if info is None:
                    continue
                self.logger.info(f'   Regime {regime_id}:')
                self.logger.info(f"      Features: {info.get('num_features', 0)}")
                self.logger.info(f"      Samples: {info.get('data_shape', [0])[0]}")
                self.logger.info(f"      Context rows: {info.get('context_rows', 0)}")
                config = info.get('regime_config', {})
                optimization_info = info.get('optimization_info', {})
                if 'emphasis' in config:
                    self.logger.info(f"      Emphasis: {config['emphasis']}")
                if optimization_info:
                    self.logger.info(f"      Optimization Priority: {optimization_info.get('optimization_priority', 'N/A')}")
                    self.logger.info(f"      Adaptive Lookback: {optimization_info.get('adaptive_lookback_enabled', False)}")
                    optimized_periods = optimization_info.get('optimized_periods', {})
                    if optimized_periods:
                        period_count = sum((len(periods.get('selected_periods', [])) for periods in optimized_periods.values()))
                        self.logger.info(f'      Optimized Indicators: {len(optimized_periods)} ({period_count} total periods)')
                    else:
                        self.logger.info(f'      Optimized Indicators: None (using fallback)')
        except Exception as e:
            self.logger.error(f'❌ Error logging feature statistics: {e}')

    async def _optimize_regime_lookback_periods(self, regime_id: int, regime_labeled: pd.DataFrame, context_mask: pd.Series, regime_config: Dict[str, Any]) -> None:
        """Optimize lookback periods for a specific regime.
        
        Args:
            regime_id: Regime ID
            regime_labeled: Labeled regime data
            context_mask: Context mask for filtering data
            regime_config: Regime configuration to update
        """
        self.logger.info(f'🔍 Optimizing lookback periods for regime {regime_id}')
        optimization_data = regime_labeled[~context_mask]
        if len(optimization_data) <= 100:
            self.logger.warning(f'⚠️ Insufficient data for regime {regime_id} optimization ({len(optimization_data)} rows)')
            regime_config['optimized_periods'] = {}
            return
        target = optimization_data['label']
        regime_series = pd.Series(regime_id, index = optimization_data.index)
        optimization_results = await self.optimize_lookback_periods(optimization_data, target, regimes = regime_series)
        status = optimization_results.get('status')
        if status == 'optimized':
            self._process_optimized_results(regime_id, optimization_results, regime_config)
        elif status == 'fallback':
            self.logger.warning(f'⚠️ Regime {regime_id} using fallback periods')
            regime_config['optimized_periods'] = optimization_results.get('periods', {})
        else:
            self.logger.error(f'❌ Regime {regime_id} optimization failed')
            regime_config['optimized_periods'] = {}
    @log_all_calls
    async def optimize_lookback_periods(self, data: pd.DataFrame, target: pd.Series, regimes: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Optimize lookback periods for features using the feature generation optimizer.
        
        Args:
            data: Input data DataFrame
            target: Target variable series
            regimes: Optional regime series for regime-aware optimization
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.feature_optimizer:
            self.logger.warning('⚠️ Feature optimizer not available, using fallback periods')
            return self._get_fallback_periods()
        
        try:
            # Get feature columns (exclude common non-feature columns)
            feature_columns = [col for col in data.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'regime']]
            
            if not feature_columns:
                self.logger.warning('⚠️ No feature columns found for optimization')
                return self._get_fallback_periods()
            
            self.logger.info(f'🔬 Optimizing lookback periods for {len(feature_columns)} features')
            
            # Create feature configurations for optimization
            feature_configs = {}
            for feature_name in feature_columns:
                # Create a simple feature generator that returns the feature as-is
                def create_feature_generator(feature_col):
                    def feature_generator(data_df, lookback):
                        return data_df[feature_col]
                    return feature_generator
                
                feature_configs[feature_name] = {
                    'generator': create_feature_generator(feature_name)
                }
            
            # Perform optimization
            regime_column = 'regime' if regimes is not None else None
            if regime_column and regime_column not in data.columns and regimes is not None:
                data = data.copy()
                data[regime_column] = regimes
            
            results = await self.feature_optimizer.optimize_multiple_features(
                data, feature_configs, 'label', regime_column
            )
            
            # Process results
            optimized_periods = {}
            for feature_name, result in results.items():
                optimized_periods[feature_name] = {
                    'selected_periods': [result.optimal_lookback],
                    'performance_score': result.performance_score,
                    'stability_score': result.stability_score,
                    'method': result.optimization_method
                }
            
            return {
                'status': 'optimized',
                'periods': optimized_periods,
                'optimization_results': {
                    'regime_specific_periods': {f'regime_{regime}': optimized_periods for regime in data[regime_column].unique()} if regime_column else {}
                }
            }
            
        except Exception as e:
            self.logger.error(f'❌ Error in lookback period optimization: {e}')
            return self._get_fallback_periods()
    
    def _get_fallback_periods(self) -> Dict[str, Any]:
        """Get fallback periods when optimization fails."""
        return {
            'status': 'fallback',
            'periods': {
                'price': {'selected_periods': [10, 20, 50]},
                'volume': {'selected_periods': [10, 20]},
                'technical': {'selected_periods': [14, 21]},
                'volatility': {'selected_periods': [20, 50]},
                'momentum': {'selected_periods': [10, 20]}
            }
        }

    @log_all_calls

    def _process_optimized_results(self, regime_id: int, optimization_results: Dict[str, Any], regime_config: Dict[str, Any]) -> None:
        """Process successful optimization results.
        
        Args:
            regime_id: Regime ID
            optimization_results: Optimization results
            regime_config: Regime configuration to update
        """
        self.logger.info(f'✅ Regime {regime_id} optimization successful')
        regime_specific_periods = optimization_results.get('optimization_results', {}).get('regime_specific_periods', {})
        regime_key = f'regime_{regime_id}'
        if regime_key in regime_specific_periods:
            regime_periods = regime_specific_periods[regime_key]
            regime_config['optimized_periods'] = regime_periods
            self.logger.info(f'📊 Regime {regime_id} specific periods: {list(regime_periods.keys())}')
        else:
            global_periods = optimization_results.get('periods', {})
            regime_config['optimized_periods'] = global_periods
            self.logger.info(f'📊 Using global optimized periods for regime {regime_id}')
        self._update_regime_interaction_patterns(regime_config, regime_id)
        validation_passed = self._validate_regime_optimization(regime_id, optimization_results, regime_config)
        if not validation_passed:
            self.logger.warning(f'⚠️ Regime {regime_id} optimization validation failed, but continuing')

@traced(span_name='run_per_regime_feature_engineering_step')
@validates()
@handles_errors
async def run_per_regime_step(symbol: str, exchange: str, timeframe: str, data_dir: str = None, force_rerun: bool = False, config: Optional[Dict[str, Any]]=None) -> bool:
    """Run the enhanced per-regime feature engineering step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info('🚀 Starting Step 6: Per-Regime Feature Engineering')
    if config is None:
        config = {}
    try:
        config_path = Path(__file__).parent / 'step06_per_regime_config.json'
        if config_path.exists():
            import json

            with open(config_path, 'r') as f:
                default_config = json.load(f)
                config = {**default_config, **config}
                logger.info('✅ Loaded per-regime feature engineering configuration')
        else:
            logger.warning('⚠️ Per-regime config file not found, using defaults')
    except Exception as e:
        logger.warning(f'⚠️ Error loading per-regime config: {e}, using defaults')
    if data_dir is None:
        data_dir = standardized_parquet_handler.get_standardized_path('processed_data', exchange, symbol)
    config['per_regime_feature_engineering'] = True
    step = PerRegimeFeatureEngineeringStep(config)
    await step.initialize()
    success = await step.execute_per_regime_feature_engineering(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, force_rerun = force_rerun)
    if success:
        logger.info('✅ Step 6: Per-Regime Feature Engineering completed successfully')
    else:
        logger.error('❌ Step 6: Per-Regime Feature Engineering failed')
    return success
if __name__ == '__main__':

    async def test() -> None:
        """Test the per-regime feature engineering step."""
        success = await run_per_regime_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='historical_data')
        tprint(f'Per-regime feature engineering result: {success}')
    asyncio.run(test())