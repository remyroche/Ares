from typing import List
from typing import Any
import pandas as pd
from datetime import datetime
from pathlib import Path
from ..utils.logger import system_logger
from ..utils.warning_symbols import invalid, warning, failed

import numpy as np
from ..core.decorators import handles_errors
from ..core.decorators.validate import validates

try:
    from src.training.steps.market_analysis.model_persistence_components.model_serializer import ModelSerializer
    from src.training.steps.market_analysis.model_persistence_components.version_manager import VersionManager
    _PERSISTENCE_AVAILABLE = True
except Exception:
    _PERSISTENCE_AVAILABLE = False

import json
import logging
import time

class MLTacticsManager:
    """
    ML Tactics Manager responsible for ML-based tactics and decision making.
    This module handles all ML tactics logic and decision making.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML tactics manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild('MLTacticsManager')
        self.is_initialized: bool = False
        self.ml_predictions: dict[str, Any] = {}
        self.ml_decisions: dict[str, Any] = {}
        self.ml_config: dict[str, Any] = self.config.get('ml_tactics_manager', {})
        step17_config = self.config.get('step17_optimization', {})
        ml_tactics_optimization = step17_config.get('ml_tactics', {})
        self.enable_ml_tactics: bool = ml_tactics_optimization.get('enable_ml_tactics', True)
        self.confidence_threshold: float = ml_tactics_optimization.get('confidence_threshold', 0.7)
        self.regime_threshold: float = ml_tactics_optimization.get('regime_threshold', 0.6)
        self.ml_weight: float = ml_tactics_optimization.get('ml_weight', 0.8)
        self.regime_weight: float = ml_tactics_optimization.get('regime_weight', 0.2)
        self.multi_output_models: dict[str, Any] = {}
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None
        # Updated to use 0.3% micro movement predictions instead of barrier-based approach
        self.micro_movement_config = {
            'micro_immediate': {'target': 0.003, 'timeframe': '1m', 'horizon': 'immediate'},  # 0.3% within 10 minutes
            'micro_short': {'target': 0.003, 'timeframe': '5m', 'horizon': 'short'},  # 0.3% within 20 minutes
            'small_immediate': {'target': 0.005, 'timeframe': '1m', 'horizon': 'immediate'},  # 0.5% within 10 minutes
            'small_short': {'target': 0.005, 'timeframe': '5m', 'horizon': 'short'}  # 0.5% within 20 minutes
        }
        # Updated thresholds for 0.3% micro movement predictions
        self.green_light_thresholds = {
            'micro_immediate_long': ml_tactics_optimization.get('micro_immediate_long_threshold', 0.75),
            'micro_immediate_short': ml_tactics_optimization.get('micro_immediate_short_threshold', 0.78),
            'micro_short_long': ml_tactics_optimization.get('micro_short_long_threshold', 0.70),
            'micro_short_short': ml_tactics_optimization.get('micro_short_short_threshold', 0.73),
            'combined_threshold': ml_tactics_optimization.get('combined_threshold', 0.70)
        }
        # Enhanced exit thresholds with adaptive features
        self.exit_thresholds = {
            'micro_immediate_long': ml_tactics_optimization.get('exit_micro_immediate_long_threshold', 0.40),
            'micro_immediate_short': ml_tactics_optimization.get('exit_micro_immediate_short_threshold', 0.35),
            'micro_short_long': ml_tactics_optimization.get('exit_micro_short_long_threshold', 0.45),
            'micro_short_short': ml_tactics_optimization.get('exit_micro_short_short_threshold', 0.40),
            'combined_exit_threshold': ml_tactics_optimization.get('combined_exit_threshold', 0.45),
            'directional_confidence_min': ml_tactics_optimization.get('directional_confidence_min', 0.10),
            # New adaptive thresholds for regime-based optimization
            'high_volatility_multiplier': ml_tactics_optimization.get('high_volatility_multiplier', 0.8),  # Tighter thresholds in high vol
            'low_volatility_multiplier': ml_tactics_optimization.get('low_volatility_multiplier', 1.2),    # Looser thresholds in low vol
            'trending_market_multiplier': ml_tactics_optimization.get('trending_market_multiplier', 0.9), # Tighter in trending markets
            'ranging_market_multiplier': ml_tactics_optimization.get('ranging_market_multiplier', 1.1),   # Looser in ranging markets
        }
        self.confidence_weights = {
            'analyst_weight': ml_tactics_optimization.get('analyst_confidence_weight', 0.3),
            'micro_immediate_weight': ml_tactics_optimization.get('micro_immediate_weight', 0.4),
            'micro_short_weight': ml_tactics_optimization.get('micro_short_weight', 0.2),
            'directional_weight': ml_tactics_optimization.get('directional_weight', 0.1)
        }
        self.model_storage_dir: str = self.config.get('model_storage_dir', 'models')

    class ProbabilityAveragingEnsemble:
        """Simple ensemble that averages predict_proba outputs across models."""

        def __init__(self, models: List[Any]) -> None:
            self.models = [m for m in models if hasattr(m, 'predict_proba')]

        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Any:
            if not self.models:
                if hasattr(X, 'shape') and len(getattr(X, 'shape', [])) > 0:
                    n = X.shape[0]
                else:
                    n = 1
                return np.tile(np.array([[0.5, 0.5]]), (n, 1))
            probs: List[np.ndarray] = []
            for model in self.models:
                try:
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        proba = np.vstack([1 - proba, proba]).T
                    probs.append(proba)
                except Exception:
                    continue
            if not probs:
                if hasattr(X, 'shape') and len(getattr(X, 'shape', [])) > 0:
                    n = X.shape[0]
                else:
                    n = 1
                return np.tile(np.array([[0.5, 0.5]]), (n, 1))
            return np.mean(probs, axis = 0)

    @handles_errors(error_handlers={ValueError: (False, 'Invalid ML tactics manager configuration'), AttributeError: (False, 'Missing required ML tactics parameters'), KeyError: (False, 'Missing configuration keys')}, default_return = False, context='ML tactics manager initialization')
    async def initialize(self) -> bool:
        """
        Initialize ML tactics manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info('Initializing ML Tactics Manager...')
            if not self._validate_configuration():
                self.logger.error(invalid('Invalid configuration for ML tactics manager'))
                return False
            await self._initialize_ml_models()
            self.is_initialized = True
            self.logger.info('✅ ML Tactics Manager initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ ML Tactics Manager initialization failed: {e}'))
            return False

    @handles_errors(fallback = False)
    def _validate_configuration(self) -> bool:
        """
        Validate ML tactics manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.logger.error(invalid('Invalid confidence_threshold configuration'))
                return False
            if self.regime_threshold <= 0 or self.regime_threshold > 1:
                self.logger.error(invalid('Invalid regime_threshold configuration'))
                return False
            if self.ml_weight <= 0 or self.ml_weight > 1:
                self.logger.error(invalid('Invalid ml_weight configuration'))
                return False
            if self.regime_weight <= 0 or self.regime_weight > 1:
                self.logger.error(invalid('Invalid regime_weight configuration'))
                return False
            for barrier_type, config in self.barrier_config.items():
                if config['profit_target_multiplier'] <= 0 or config['stop_loss_multiplier'] <= 0:
                    self.logger.error(invalid(f'Invalid barrier configuration for {barrier_type}'))
                    return False
            for threshold_type, threshold in self.green_light_thresholds.items():
                if threshold <= 0 or threshold > 1:
                    self.logger.error(invalid(f'Invalid green light threshold for {threshold_type}'))
                    return False
            for threshold_type, threshold in self.exit_thresholds.items():
                if threshold <= 0 or threshold > 1:
                    self.logger.error(invalid(f'Invalid exit threshold for {threshold_type}'))
                    return False
            total_weight = sum(self.confidence_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.logger.error(invalid(f'Confidence weights must sum to 1.0, got {total_weight}'))
                return False
            for weight_name, weight in self.confidence_weights.items():
                if weight < 0 or weight > 1:
                    self.logger.error(invalid(f'Invalid confidence weight for {weight_name}: {weight}'))
                    return False
            return True
        except Exception as e:
            self.logger.exception(failed(f'Configuration validation failed: {e}'))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            if 'ml_tactics' in step17_results:
                ml_tactics_optimization = step17_results['ml_tactics']
                self.enable_ml_tactics = ml_tactics_optimization.get('enable_ml_tactics', self.enable_ml_tactics)
                self.confidence_threshold = ml_tactics_optimization.get('confidence_threshold', self.confidence_threshold)
                self.regime_threshold = ml_tactics_optimization.get('regime_threshold', self.regime_threshold)
                self.ml_weight = ml_tactics_optimization.get('ml_weight', self.ml_weight)
                self.regime_weight = ml_tactics_optimization.get('regime_weight', self.regime_weight)
                self.barrier_config = {'fifty_percent': {'profit_target_multiplier': ml_tactics_optimization.get('fifty_percent_profit_target_multiplier', 0.5), 'stop_loss_multiplier': ml_tactics_optimization.get('fifty_percent_stop_loss_multiplier', 0.5), 'timeframe': ml_tactics_optimization.get('fifty_percent_timeframe', '1m')}, 'twenty_five_percent': {'profit_target_multiplier': ml_tactics_optimization.get('twenty_five_percent_profit_target_multiplier', 0.25), 'stop_loss_multiplier': ml_tactics_optimization.get('twenty_five_percent_stop_loss_multiplier', 0.25), 'timeframe': ml_tactics_optimization.get('twenty_five_percent_timeframe', '1m')}, 'fifty_percent_5m': {'profit_target_multiplier': ml_tactics_optimization.get('fifty_percent_5m_profit_target_multiplier', 0.5), 'stop_loss_multiplier': ml_tactics_optimization.get('fifty_percent_5m_stop_loss_multiplier', 0.5), 'timeframe': ml_tactics_optimization.get('fifty_percent_5m_timeframe', '5m')}, 'twenty_five_percent_5m': {'profit_target_multiplier': ml_tactics_optimization.get('twenty_five_percent_5m_profit_target_multiplier', 0.25), 'stop_loss_multiplier': ml_tactics_optimization.get('twenty_five_percent_5m_stop_loss_multiplier', 0.25), 'timeframe': ml_tactics_optimization.get('twenty_five_percent_5m_timeframe', '5m')}}
                self.green_light_thresholds = {'fifty_percent': ml_tactics_optimization.get('fifty_percent_threshold', 0.75), 'twenty_five_percent': ml_tactics_optimization.get('twenty_five_percent_threshold', 0.8), 'combined_threshold': ml_tactics_optimization.get('combined_threshold', 0.7)}
                self.exit_thresholds = {'fifty_percent': ml_tactics_optimization.get('exit_fifty_percent_threshold', 0.4), 'twenty_five_percent': ml_tactics_optimization.get('exit_twenty_five_percent_threshold', 0.35), 'combined_exit_threshold': ml_tactics_optimization.get('combined_exit_threshold', 0.45)}
                self.confidence_weights = {'analyst_weight': ml_tactics_optimization.get('analyst_confidence_weight', 0.3), 'fifty_percent_1m_weight': ml_tactics_optimization.get('fifty_percent_1m_weight', 0.25), 'twenty_five_percent_1m_weight': ml_tactics_optimization.get('twenty_five_percent_1m_weight', 0.15), 'fifty_percent_5m_weight': ml_tactics_optimization.get('fifty_percent_5m_weight', 0.2), 'twenty_five_percent_5m_weight': ml_tactics_optimization.get('twenty_five_percent_5m_weight', 0.1)}
                self.logger.info('✅ ML tactics manager configuration refreshed from step17 results')
        except Exception as e:
            self.logger.exception(f'Error refreshing step17 configuration: {e}')

    @handles_errors(fallback = False)
    async def _initialize_ml_models(self) -> bool:
        """
        Initialize multi-output prediction models.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info('Initializing multi-output prediction models...')
            for barrier_type in ['fifty_percent', 'twenty_five_percent', 'fifty_percent_5m', 'twenty_five_percent_5m']:
                self.multi_output_models[barrier_type] = {'model': None, 'calibrator': None, 'is_trained': False, 'feature_importance': {}, 'performance_metrics': {}}
            await self._load_pretrained_models()
            if not self.is_trained:
                self.logger.warning('No pre-trained models found, using fallback models')
                await self._initialize_fallback_models()
            self.logger.info('✅ Multi-output prediction models initialized')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ ML models initialization failed: {e}'))
            return False

    @handles_errors(fallback = False)
    async def _load_pretrained_models(self) -> bool:
        """
        Load pre-trained multi-output models.

        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            if not _PERSISTENCE_AVAILABLE:
                self.logger.warning('Model persistence components unavailable - using fallback')
                return False
            version_manager = VersionManager({'versioning': {'base_dir': self.model_storage_dir}})
            symbol = self.config.get('symbol')
            exchange = self.config.get('exchange')
            latest = await version_manager.get_latest_version(symbol = symbol, exchange = exchange)
            if not latest:
                self.logger.warning('No versions found in model registry - using fallback')
                return False
            version_dir = Path(self.model_storage_dir) / latest['version'] / 'models'
            loaded_models: List[Any] = []

            async def _scan_and_load(subdir: str) -> None:
                nonlocal loaded_models
                model_dir = version_dir / subdir
                if not model_dir.exists():
                    return
                for fp in sorted(model_dir.glob('*.pkl')):
                    model = await ModelSerializer({'serialization': {'base_dir': self.model_storage_dir}}).load_model(str(fp))
                    if model is not None:
                        loaded_models.append(model)
                for fp in sorted(model_dir.glob('*.joblib')):
                    model = await ModelSerializer({'serialization': {'base_dir': self.model_storage_dir}}).load_model(str(fp))
                    if model is not None:
                        loaded_models.append(model)
            await _scan_and_load('pickle')
            await _scan_and_load('joblib')
            if not loaded_models:
                self.logger.warning('No persisted models found under latest version - using fallback')
                return False
            ensemble = self.ProbabilityAveragingEnsemble(loaded_models)
            for barrier_type in ['fifty_percent', 'twenty_five_percent', 'fifty_percent_5m', 'twenty_five_percent_5m']:
                self.multi_output_models[barrier_type]['model'] = ensemble
                self.multi_output_models[barrier_type]['calibrator'] = None
                self.multi_output_models[barrier_type]['is_trained'] = True
            self.is_trained = True
            self.logger.info(f"✅ Loaded {len(loaded_models)} persisted model(s) from version {latest['version']}")
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Failed to load pre-trained models: {e}'))
            return False

    @handles_errors(fallback = False)
    async def _initialize_fallback_models(self) -> bool:
        """
        Initialize fallback models for testing.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info('Initializing fallback models...')
            for barrier_type in ['fifty_percent', 'twenty_five_percent', 'fifty_percent_5m', 'twenty_five_percent_5m']:
                self.multi_output_models[barrier_type]['is_trained'] = True
                self.multi_output_models[barrier_type]['model'] = 'fallback'
            self.is_trained = True
            self.logger.info('✅ Fallback models initialized')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Fallback models initialization failed: {e}'))
            return False

    @handles_errors(error_handlers={ValueError: (False, 'Invalid ML tactics parameters'), AttributeError: (False, 'Missing ML tactics components'), KeyError: (False, 'Missing required ML tactics data')}, default_return = False, context='ML tactics execution')
    async def execute_ml_tactics(self, tactics_input: dict[str, Any]) -> dict[str, Any]:
        """
        Execute ML-based tactics.

        Args:
            tactics_input: ML tactics input parameters

        Returns:
            dict: ML tactics results
        """
        try:
            self.logger.info('🤖 Executing ML tactics...')
            if not self._validate_tactics_input(tactics_input):
                return {}
            ml_predictions = self._get_ml_predictions()
            if not ml_predictions:
                self.logger.warning(warning('⚠️ No ML predictions available'))
                return {}
            regime_tactics = self._apply_regime_and_location_tactics(ml_predictions)
            entry_decisions = self._make_ml_entry_decisions(ml_predictions)
            sizing_decisions = self._make_ml_sizing_decisions(ml_predictions)
            leverage_decisions = self._make_ml_leverage_decisions(ml_predictions)
            directional_decisions = self._make_ml_directional_decisions(ml_predictions)
            liquidation_decisions = self._make_ml_liquidation_risk_decisions(ml_predictions)
            position_size = await self._calculate_position_size(ml_predictions)
            leverage = await self._calculate_leverage(ml_predictions)
            ml_results = {'regime_tactics': regime_tactics, 'entry_decisions': entry_decisions, 'sizing_decisions': sizing_decisions, 'leverage_decisions': leverage_decisions, 'directional_decisions': directional_decisions, 'liquidation_decisions': liquidation_decisions, 'position_size': position_size, 'leverage': leverage, 'ml_predictions': ml_predictions, 'timestamp': datetime.now()}
            self.ml_decisions = ml_results
            self.logger.info('✅ ML tactics execution completed successfully')
            return ml_results
        except Exception as e:
            self.logger.exception(failed(f'❌ ML tactics execution failed: {e}'))
            return {}

    @validates(strict = True)
    @handles_errors(fallback = False)
    def _validate_tactics_input(self, tactics_input: dict[str, Any]) -> bool:
        """
        Validate ML tactics input parameters.

        Args:
            tactics_input: ML tactics input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ['symbol', 'exchange', 'timeframe', 'current_price']
            for field in required_fields:
                if field not in tactics_input:
                    self.logger.error(f'Missing required ML tactics input field: {field}')
                    return False
            if tactics_input.get('current_price', 0) <= 0:
                self.logger.error(invalid('Invalid current_price value'))
                return False
            return True
        except Exception as e:
            self.logger.exception(failed(f'ML tactics input validation failed: {e}'))
            return False

    @handles_errors(fallback = None)
    def _get_ml_predictions(self) -> dict[str, Any] | None:
        """
        Get ML predictions.

        Returns:
            dict: ML predictions or None if not available
        """
        try:
            return {'regime_prediction': {'BULL_TREND': 0.7, 'BEAR_TREND': 0.2, 'SIDEWAYS_RANGE': 0.1}, 'location_prediction': {'NEAR_SUPPORT': 0.8, 'NEAR_RESISTANCE': 0.1, 'MIDDLE': 0.1}, 'entry_prediction': {'confidence': 0.85, 'direction': 'LONG', 'strength': 0.8}, 'sizing_prediction': {'confidence': 0.75, 'size_multiplier': 1.2, 'risk_level': 'MEDIUM'}, 'leverage_prediction': {'confidence': 0.7, 'leverage_multiplier': 1.5, 'risk_level': 'HIGH'}, 'directional_prediction': {'confidence': 0.8, 'direction': 'UP', 'strength': 0.75}, 'liquidation_risk_prediction': {'confidence': 0.6, 'risk_level': 'LOW', 'time_to_liquidation': 24}}
        except Exception as e:
            self.logger.exception(failed(f'❌ Failed to get ML predictions: {e}'))
            return None

    @handles_errors(fallback = None)
    def _apply_regime_and_location_tactics(self, regime_info: dict[str, Any]) -> dict[str, Any]:
        """
        Apply regime and location tactics.

        Args:
            regime_info: Regime information

        Returns:
            dict: Regime and location tactics
        """
        try:
            regime_prediction = regime_info.get('regime_prediction', {})
            location_prediction = regime_info.get('location_prediction', {})
            dominant_regime = max(regime_prediction.items(), key=lambda x: x[1])[0]
            regime_confidence = regime_prediction.get(dominant_regime, 0)
            dominant_location = max(location_prediction.items(), key=lambda x: x[1])[0]
            location_confidence = location_prediction.get(dominant_location, 0)
            regime_tactics = self._get_regime_tactics(dominant_regime, regime_confidence)
            location_tactics = self._get_location_tactics(dominant_location, location_confidence)
            return {'dominant_regime': dominant_regime, 'regime_confidence': regime_confidence, 'dominant_location': dominant_location, 'location_confidence': location_confidence, 'regime_tactics': regime_tactics, 'location_tactics': location_tactics, 'combined_tactics': self._combine_regime_location_tactics(regime_tactics, location_tactics)}
        except Exception as e:
            self.logger.exception(f'❌ Regime and location tactics application failed: {e}')
            return {}

    @handles_errors(fallback = None)
    def _make_ml_entry_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make ML-based entry decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Entry decisions
        """
        try:
            entry_prediction = ml_predictions.get('entry_prediction', {})
            confidence = entry_prediction.get('confidence', 0)
            direction = entry_prediction.get('direction', 'NEUTRAL')
            strength = entry_prediction.get('strength', 0)
            if confidence >= self.confidence_threshold:
                if direction == 'LONG' and strength > 0.6:
                    decision = 'ENTER_LONG'
                elif direction == 'SHORT' and strength > 0.6:
                    decision = 'ENTER_SHORT'
                else:
                    decision = 'HOLD'
            else:
                decision = 'HOLD_LOW_CONFIDENCE'
            return {'decision': decision, 'confidence': confidence, 'direction': direction, 'strength': strength, 'reasoning': f'ML prediction: {direction} with {confidence:.2f} confidence'}
        except Exception as e:
            self.logger.exception(failed(f'❌ ML entry decisions making failed: {e}'))
            return {}

    @handles_errors(fallback = None)
    def _make_ml_sizing_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make ML-based sizing decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Sizing decisions
        """
        try:
            sizing_prediction = ml_predictions.get('sizing_prediction', {})
            confidence = sizing_prediction.get('confidence', 0)
            size_multiplier = sizing_prediction.get('size_multiplier', 1.0)
            risk_level = sizing_prediction.get('risk_level', 'MEDIUM')
            if confidence >= self.confidence_threshold:
                if risk_level == 'LOW':
                    adjusted_multiplier = size_multiplier * 1.2
                elif risk_level == 'HIGH':
                    adjusted_multiplier = size_multiplier * 0.8
                else:
                    adjusted_multiplier = size_multiplier
                decision = 'ADJUST_SIZE'
            else:
                adjusted_multiplier = 1.0
                decision = 'MAINTAIN_SIZE'
            return {'decision': decision, 'confidence': confidence, 'size_multiplier': adjusted_multiplier, 'risk_level': risk_level, 'reasoning': f'ML sizing: {adjusted_multiplier:.2f}x with {confidence:.2f} confidence'}
        except Exception as e:
            self.logger.exception(failed(f'❌ ML sizing decisions making failed: {e}'))
            return {}

    @handles_errors(fallback = None)
    def _make_ml_leverage_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make ML-based leverage decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Leverage decisions
        """
        try:
            leverage_prediction = ml_predictions.get('leverage_prediction', {})
            confidence = leverage_prediction.get('confidence', 0)
            leverage_multiplier = leverage_prediction.get('leverage_multiplier', 1.0)
            risk_level = leverage_prediction.get('risk_level', 'MEDIUM')
            if confidence >= self.confidence_threshold:
                if risk_level == 'LOW':
                    adjusted_leverage = leverage_multiplier * 1.3
                elif risk_level == 'HIGH':
                    adjusted_leverage = leverage_multiplier * 0.7
                else:
                    adjusted_leverage = leverage_multiplier
                decision = 'ADJUST_LEVERAGE'
            else:
                adjusted_leverage = 1.0
                decision = 'MAINTAIN_LEVERAGE'
            return {'decision': decision, 'confidence': confidence, 'leverage_multiplier': adjusted_leverage, 'risk_level': risk_level, 'reasoning': f'ML leverage: {adjusted_leverage:.2f}x with {confidence:.2f} confidence'}
        except Exception as e:
            self.logger.exception(failed(f'❌ ML leverage decisions making failed: {e}'))
            return {}

    @handles_errors(fallback = None)
    def _make_ml_directional_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make ML-based directional decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Directional decisions
        """
        try:
            directional_prediction = ml_predictions.get('directional_prediction', {})
            confidence = directional_prediction.get('confidence', 0)
            direction = directional_prediction.get('direction', 'NEUTRAL')
            strength = directional_prediction.get('strength', 0)
            if confidence >= self.confidence_threshold:
                if direction == 'UP' and strength > 0.6:
                    decision = 'BULLISH'
                elif direction == 'DOWN' and strength > 0.6:
                    decision = 'BEARISH'
                else:
                    decision = 'NEUTRAL'
            else:
                decision = 'UNCERTAIN'
            return {'decision': decision, 'confidence': confidence, 'direction': direction, 'strength': strength, 'reasoning': f'ML direction: {direction} with {confidence:.2f} confidence'}
        except Exception as e:
            self.logger.exception(failed(f'❌ ML directional decisions making failed: {e}'))
            return {}

    @handles_errors(fallback = None)
    def _make_ml_liquidation_risk_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make ML-based liquidation risk decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Liquidation risk decisions
        """
        try:
            liquidation_prediction = ml_predictions.get('liquidation_risk_prediction', {})
            confidence = liquidation_prediction.get('confidence', 0)
            risk_level = liquidation_prediction.get('risk_level', 'MEDIUM')
            time_to_liquidation = liquidation_prediction.get('time_to_liquidation', 24)
            if confidence >= self.confidence_threshold:
                if risk_level == 'HIGH':
                    decision = 'REDUCE_POSITION'
                elif risk_level == 'MEDIUM':
                    decision = 'MONITOR_CLOSELY'
                else:
                    decision = 'MAINTAIN_POSITION'
            else:
                decision = 'UNCERTAIN_RISK'
            return {'decision': decision, 'confidence': confidence, 'risk_level': risk_level, 'time_to_liquidation': time_to_liquidation, 'reasoning': f'ML liquidation risk: {risk_level} with {confidence:.2f} confidence'}
        except Exception as e:
            self.logger.exception(f'❌ ML liquidation risk decisions making failed: {e}')
            return {}

    @handles_errors(fallback = None)
    async def _calculate_position_size(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate position size based on ML predictions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Position size calculation results
        """
        try:
            sizing_decisions = self._make_ml_sizing_decisions(ml_predictions)
            base_position_size = 0.05
            size_multiplier = sizing_decisions.get('size_multiplier', 1.0)
            calculated_size = base_position_size * size_multiplier
            max_position_size = 0.3
            calculated_size = min(calculated_size, max_position_size)
            return {'base_size': base_position_size, 'size_multiplier': size_multiplier, 'calculated_size': calculated_size, 'max_size': max_position_size, 'confidence': sizing_decisions.get('confidence', 0), 'decision': sizing_decisions.get('decision', 'MAINTAIN_SIZE')}
        except Exception as e:
            self.logger.exception(failed(f'❌ Position size calculation failed: {e}'))
            return {}

    @handles_errors(fallback = None)
    async def _calculate_leverage(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate leverage based on ML predictions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Leverage calculation results
        """
        try:
            leverage_decisions = self._make_ml_leverage_decisions(ml_predictions)
            base_leverage = 1.0
            leverage_multiplier = leverage_decisions.get('leverage_multiplier', 1.0)
            calculated_leverage = base_leverage * leverage_multiplier
            max_leverage = 10.0
            calculated_leverage = min(calculated_leverage, max_leverage)
            return {'base_leverage': base_leverage, 'leverage_multiplier': leverage_multiplier, 'calculated_leverage': calculated_leverage, 'max_leverage': max_leverage, 'confidence': leverage_decisions.get('confidence', 0), 'decision': leverage_decisions.get('decision', 'MAINTAIN_LEVERAGE')}
        except Exception as e:
            self.logger.exception(failed(f'❌ Leverage calculation failed: {e}'))
            return {}

    def _get_regime_tactics(self, regime: str, confidence: float) -> dict[str, Any]:
        """Get tactics for a specific regime."""
        tactics = {'BULL_TREND': {'position_multiplier': 1.2, 'risk_tolerance': 'HIGH'}, 'BEAR_TREND': {'position_multiplier': 0.8, 'risk_tolerance': 'LOW'}, 'SIDEWAYS_RANGE': {'position_multiplier': 1.0, 'risk_tolerance': 'MEDIUM'}}
        return tactics.get(regime, {'position_multiplier': 1.0, 'risk_tolerance': 'MEDIUM'})

    def _get_location_tactics(self, location: str, confidence: float) -> dict[str, Any]:
        """Get tactics for a specific location."""
        tactics = {'NEAR_SUPPORT': {'entry_aggression': 'HIGH', 'stop_distance': 'TIGHT'}, 'NEAR_RESISTANCE': {'entry_aggression': 'LOW', 'stop_distance': 'WIDE'}, 'MIDDLE': {'entry_aggression': 'MEDIUM', 'stop_distance': 'MEDIUM'}}
        return tactics.get(location, {'entry_aggression': 'MEDIUM', 'stop_distance': 'MEDIUM'})

    def _combine_regime_location_tactics(self, regime_tactics: dict[str, Any], location_tactics: dict[str, Any]) -> dict[str, Any]:
        """Combine regime and location tactics."""
        return {'position_multiplier': regime_tactics.get('position_multiplier', 1.0), 'risk_tolerance': regime_tactics.get('risk_tolerance', 'MEDIUM'), 'entry_aggression': location_tactics.get('entry_aggression', 'MEDIUM'), 'stop_distance': location_tactics.get('stop_distance', 'MEDIUM')}

    def get_ml_decisions(self) -> dict[str, Any]:
        """
        Get the latest ML decisions.

        Returns:
            dict: ML decisions
        """
        return self.ml_decisions.copy()

    @handles_errors(fallback = None)
    async def stop(self) -> None:
        """Stop the ML tactics manager and cleanup resources."""
        try:
            self.logger.info('🛑 Stopping ML Tactics Manager...')
            self.is_initialized = False
            self.logger.info('✅ ML Tactics Manager stopped successfully')
        except Exception as e:
            self.logger.exception(failed(f'❌ Failed to stop ML Tactics Manager: {e}'))

    @handles_errors(fallback = None)
    async def cleanup(self) -> None:
        """Cleanup ML tactics manager resources."""
        try:
            self.logger.info('Cleaning up ML Tactics Manager...')
            await self.stop()
            self.ml_decisions.clear()
            self.logger.info('✅ ML Tactics Manager cleanup completed')
        except Exception as e:
            self.logger.exception(f'Error cleaning up ML Tactics Manager: {e}')

    @handles_errors(fallback = None)
    async def generate_multi_output_predictions(self, market_data: pd.DataFrame, analyst_barriers: dict[str, float], symbol: str, timeframe: str, analyst_confidence: float = 0.5) -> dict[str, Any]:
        """
        Generate multi-output predictions for 0.3% micro movements.

        Args:
            market_data: Market data with OHLCV
            analyst_barriers: Analyst's barrier values (for reference - not used in new approach)
            symbol: Trading symbol
            timeframe: Current timeframe
            analyst_confidence: Analyst confidence score

        Returns:
            dict: Multi-output predictions with 0.3% micro movement probabilities and directional confidence
        """
        try:
            if not self.is_trained:
                self.logger.warning('Models not trained, using fallback predictions')
                return self._generate_fallback_predictions()
            # Generate 0.3% micro movement predictions
            predictions = {}
            for movement_type in ['micro_immediate_long', 'micro_immediate_short', 'micro_short_long', 'micro_short_short']:
                movement_prediction = await self._generate_micro_movement_prediction(
                    movement_type=movement_type, 
                    market_data=market_data, 
                    symbol=symbol, 
                    timeframe=timeframe
                )
                if movement_prediction:
                    predictions[movement_type] = movement_prediction
            
            # Calculate directional analysis
            directional_analysis = self._calculate_directional_analysis(predictions)
            combined_confidence = self._calculate_combined_micro_confidence(predictions, analyst_confidence)
            green_light_signal = self._evaluate_micro_movement_signal(predictions, combined_confidence, directional_analysis)
            
            result = {
                **predictions, 
                'combined_confidence': combined_confidence, 
                'directional_analysis': directional_analysis,
                'green_light_signal': green_light_signal, 
                'metadata': {
                    'symbol': symbol, 
                    'timeframe': timeframe, 
                    'generation_timestamp': datetime.now().isoformat(), 
                    'model_type': 'tactician_micro_movement', 
                    'micro_movement_config': self.micro_movement_config
                }
            }
            self.logger.info(f"Generated multi-output predictions for {symbol}: {green_light_signal['signal']}")
            return result
        except Exception as e:
            self.logger.exception(failed(f'❌ Multi-output predictions generation failed: {e}'))
            return self._generate_fallback_predictions()

    def _calculate_tactician_barriers(self, analyst_barriers: dict[str, float]) -> dict[str, dict[str, float]]:
        """
        Calculate Tactician barriers as 50% and 25% of Analyst barriers.

        Args:
            analyst_barriers: Analyst's barrier values

        Returns:
            dict: Tactician barriers for 50% and 25% levels
        """
        try:
            analyst_upper = analyst_barriers.get('upper_barrier', 0.02)
            analyst_lower = analyst_barriers.get('lower_barrier', -0.01)
            tactician_barriers = {}
            tactician_barriers['fifty_percent'] = {'upper_barrier': analyst_upper * self.barrier_config['fifty_percent']['profit_target_multiplier'], 'lower_barrier': analyst_lower * self.barrier_config['fifty_percent']['stop_loss_multiplier'], 'timeframe': self.barrier_config['fifty_percent']['timeframe']}
            tactician_barriers['twenty_five_percent'] = {'upper_barrier': analyst_upper * self.barrier_config['twenty_five_percent']['profit_target_multiplier'], 'lower_barrier': analyst_lower * self.barrier_config['twenty_five_percent']['stop_loss_multiplier'], 'timeframe': self.barrier_config['twenty_five_percent']['timeframe']}
            tactician_barriers['fifty_percent_5m'] = {'upper_barrier': analyst_upper * self.barrier_config['fifty_percent_5m']['profit_target_multiplier'], 'lower_barrier': analyst_lower * self.barrier_config['fifty_percent_5m']['stop_loss_multiplier'], 'timeframe': self.barrier_config['fifty_percent_5m']['timeframe']}
            tactician_barriers['twenty_five_percent_5m'] = {'upper_barrier': analyst_upper * self.barrier_config['twenty_five_percent_5m']['profit_target_multiplier'], 'lower_barrier': analyst_lower * self.barrier_config['twenty_five_percent_5m']['stop_loss_multiplier'], 'timeframe': self.barrier_config['twenty_five_percent_5m']['timeframe']}
            return tactician_barriers
        except Exception as e:
            self.logger.exception(failed(f'❌ Barrier calculation failed: {e}'))
            return {'fifty_percent': {'upper_barrier': 0.01, 'lower_barrier': -0.005, 'timeframe': '1m'}, 'twenty_five_percent': {'upper_barrier': 0.005, 'lower_barrier': -0.0025, 'timeframe': '1m'}, 'fifty_percent_5m': {'upper_barrier': 0.01, 'lower_barrier': -0.005, 'timeframe': '5m'}, 'twenty_five_percent_5m': {'upper_barrier': 0.005, 'lower_barrier': -0.0025, 'timeframe': '5m'}}

    async def _generate_barrier_prediction(self, barrier_type: str, market_data: pd.DataFrame, barriers: dict[str, float], symbol: str, timeframe: str) -> dict[str, Any]:
        """
        Generate prediction for a specific barrier type.

        Args:
            barrier_type: "fifty_percent" or "twenty_five_percent"
            market_data: Market data
            barriers: Barrier values
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            dict: Barrier prediction with confidence and direction
        """
        try:
            features = self._extract_features(market_data)
            if self.multi_output_models[barrier_type]['model'] == 'fallback':
                confidence = self._generate_fallback_confidence(barrier_type, features)
                direction = self._determine_direction(features)
            else:
                confidence = self._predict_with_model(barrier_type, features)
                direction = self._determine_direction(features)
            confidence = np.clip(confidence, 0.0, 1.0)
            return {'confidence': confidence, 'direction': direction, 'upper_barrier': barriers['upper_barrier'], 'lower_barrier': barriers['lower_barrier'], 'timeframe': barriers['timeframe'], 'barrier_type': barrier_type}
        except Exception as e:
            self.logger.exception(failed(f'❌ Barrier prediction failed for {barrier_type}: {e}'))
            return None

    def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract enhanced features from market data for prediction using FeatureBank.

        Args:
            market_data: Market data with OHLCV

        Returns:
            np.ndarray: Feature array
        """
        try:
            if len(market_data) < 20:
                return np.array([0.5] * 10)

            # Use enhanced feature bank for feature extraction
            features_df = self._extract_enhanced_features(market_data)

            if features_df.empty or len(features_df.columns) == 0:
                self.logger.warning("No enhanced features generated, falling back to basic features")
                return self._extract_basic_features(market_data)

            # Select most relevant features for ML prediction
            selected_features = self._select_relevant_features(features_df)

            # Convert to numpy array and ensure consistent length
            feature_array = selected_features.values[-1]  # Take latest values

            # Pad or truncate to expected length (10 features for compatibility)
            if len(feature_array) < 10:
                # Pad with zeros if fewer features
                padded_features = np.zeros(10)
                padded_features[:len(feature_array)] = feature_array
                feature_array = padded_features
            elif len(feature_array) > 10:
                # Take first 10 features if more available
                feature_array = feature_array[:10]

            return feature_array

        except Exception as e:
            self.logger.exception(failed(f'❌ Enhanced feature extraction failed: {e}'))
            return self._extract_basic_features(market_data)

    def _extract_enhanced_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract enhanced features using the FeatureBank.

        Args:
            market_data: Market data with OHLCV

        Returns:
            DataFrame with enhanced features
        """
        try:
            from src.feature_generation import get_feature_bank

            # Get global feature bank or create one
            try:
                feature_bank = get_feature_bank()
            except:
                # Create new feature bank if none exists
                from src.feature_generation.core.feature_bank import FeatureBank, FeatureBankConfig
                config = FeatureBankConfig(
                    enable_matrix_operations=True,
                    enable_gpu_acceleration=True,
                    enable_parallel_processing=True,
                    auto_normalize=True,
                    normalization_method='zscore'
                )
                feature_bank = FeatureBank(config)

            # Generate enhanced features for key categories
            categories = [
                'normalization',  # Rolling z-scores, volatility scaling
                'momentum',       # RSI, MACD, momentum indicators
                'volatility',     # Bollinger Bands, ATR
                'volume',         # Volume ratios, OBV
                'interaction',    # Feature interactions
                'cross_timeframe' # Multi-timeframe features
            ]

            enhanced_features = feature_bank.generate_features(
                data=market_data,
                categories=categories,
                lookback_optimization=False  # Use default lookbacks for speed
            )

            self.logger.info(f"✅ Generated {len(enhanced_features.columns)} enhanced features")
            return enhanced_features

        except Exception as e:
            self.logger.warning(f"Error generating enhanced features: {e}")
            return pd.DataFrame()

    def _select_relevant_features(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Select most relevant features for ML prediction.

        Args:
            features_df: DataFrame with all generated features

        Returns:
            Series with selected features (latest values)
        """
        # Feature selection strategy for Tactician ML models
        priority_features = [
            # Normalization features (most important for stationarity)
            'zscore_close_20', 'zscore_close_50', 'zscore_volume_20',
            'vol_scaled_returns_20', 'regime_norm_close_30',

            # Momentum features
            'rsi_14', 'macd_12_26_9', 'momentum_5', 'roc_10',

            # Volatility features
            'volatility_20', 'atr_14', 'bb_width_20',

            # Volume features
            'volume_ratio_20', 'obv', 'volume_momentum_5',

            # Interaction features
            'momentum_volume_5', 'price_volume_divergence_20',

            # Cross-timeframe features
            'ctf_fractional_volatility_5m_15m', 'ctf_aligned_1m_to_5m'
        ]

        selected_values = []

        for feature in priority_features:
            if feature in features_df.columns:
                # Get latest value, handle NaN
                latest_value = features_df[feature].iloc[-1]
                if pd.isna(latest_value):
                    latest_value = 0.0
                selected_values.append(latest_value)

        # If we don't have enough priority features, fill with available features
        if len(selected_values) < 10:
            available_features = [col for col in features_df.columns
                                if col not in priority_features]

            for feature in available_features[:10-len(selected_values)]:
                latest_value = features_df[feature].iloc[-1]
                if pd.isna(latest_value):
                    latest_value = 0.0
                selected_values.append(latest_value)

        # Ensure we have exactly 10 features
        while len(selected_values) < 10:
            selected_values.append(0.0)

        return pd.Series(selected_values[:10])

    def _extract_basic_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract basic features as fallback (original implementation).

        Args:
            market_data: Market data with OHLCV

        Returns:
            np.ndarray: Basic feature array
        """
        try:
            features = []
            if len(market_data) < 20:
                return np.array([0.5] * 10)

            close_prices = market_data['close'].values
            high_prices = market_data['high'].values
            low_prices = market_data['low'].values
            volumes = market_data['volume'].values

            # Basic momentum and volatility features
            price_momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if close_prices[-5] > 0 else 0
            features.append(price_momentum)

            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-20:])
            features.append(volatility)

            volume_trend = (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
            features.append(volume_trend)

            price_range = (high_prices[-1] - low_prices[-1]) / close_prices[-1] if close_prices[-1] > 0 else 0
            features.append(price_range)

            ma_short = np.mean(close_prices[-5:])
            ma_long = np.mean(close_prices[-20:])
            ma_ratio = ma_short / ma_long if ma_long > 0 else 1.0
            features.append(ma_ratio)

            # RSI calculation
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
            rsi = 100 - 100 / (1 + rs)
            features.append(rsi / 100)

            # Additional basic features
            features.extend([
                close_prices[-1] / close_prices[-2] - 1 if close_prices[-2] > 0 else 0,
                np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1.0,
                (high_prices[-1] - close_prices[-1]) / close_prices[-1] if close_prices[-1] > 0 else 0,
                (close_prices[-1] - low_prices[-1]) / close_prices[-1] if close_prices[-1] > 0 else 0
            ])

            return np.array(features)

        except Exception as e:
            self.logger.exception(failed(f'❌ Basic feature extraction failed: {e}'))
            return np.array([0.5] * 10)

    def _generate_fallback_confidence(self, barrier_type: str, features: np.ndarray) -> float:
        """
        Generate fallback confidence score.

        Args:
            barrier_type: Barrier type
            features: Feature array

        Returns:
            float: Confidence score
        """
        try:
            base_confidence = 0.5
            if len(features) > 0:
                momentum = features[0]
                if abs(momentum) > 0.01:
                    base_confidence += 0.2
                elif abs(momentum) > 0.005:
                    base_confidence += 0.1
            if len(features) > 1:
                volatility = features[1]
                if volatility < 0.01:
                    base_confidence += 0.1
                elif volatility > 0.03:
                    base_confidence -= 0.1
            if len(features) > 5:
                rsi = features[5]
                if 0.3 < rsi < 0.7:
                    base_confidence += 0.1
                elif rsi < 0.2 or rsi > 0.8:
                    base_confidence -= 0.1
            if barrier_type == 'twenty_five_percent':
                base_confidence *= 0.9
            return np.clip(base_confidence, 0.0, 1.0)
        except Exception as e:
            self.logger.exception(failed(f'❌ Fallback confidence generation failed: {e}'))
            return 0.5

    def _determine_direction(self, features: np.ndarray) -> str:
        """
        Determine price direction based on features.

        Args:
            features: Feature array

        Returns:
            str: "UP" or "DOWN"
        """
        try:
            if len(features) > 0:
                momentum = features[0]
                if momentum > 0:
                    return 'UP'
                return 'DOWN'
            return 'UP'
        except Exception as e:
            self.logger.exception(failed(f'❌ Direction determination failed: {e}'))
            return 'UP'

    def _predict_with_model(self, barrier_type: str, features: np.ndarray) -> float:
        """
        Predict confidence using actual model.

        Args:
            barrier_type: Barrier type
            features: Feature array

        Returns:
            float: Confidence score
        """
        try:
            model_entry = self.multi_output_models.get(barrier_type, {})
            model = model_entry.get('model')
            calibrator = model_entry.get('calibrator')
            target = calibrator or model
            if target is None:
                return self._generate_fallback_confidence(barrier_type, features)
            if hasattr(target, 'predict_proba'):
                proba = target.predict_proba(features.reshape(1, -1))
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return float(proba[0, 1])
                return float(np.clip(proba.ravel()[0], 0.0, 1.0))
            if hasattr(target, 'decision_function'):
                score = float(target.decision_function(features.reshape(1, -1))[0])
                import math
                return float(1.0 / (1.0 + math.exp(-score)))
            if hasattr(target, 'predict'):
                pred = target.predict(features.reshape(1, -1))
                try:
                    return float(np.clip(float(pred[0]), 0.0, 1.0))
                except Exception:
                    return self._generate_fallback_confidence(barrier_type, features)
            return self._generate_fallback_confidence(barrier_type, features)
        except Exception as e:
            self.logger.exception(failed(f'❌ Model prediction failed: {e}'))
            return 0.5

    def _calibrate_prediction(self, barrier_type: str, confidence: float) -> float:
        """
        Calibrate prediction using calibrator.

        Args:
            barrier_type: Barrier type
            confidence: Raw confidence

        Returns:
            float: Calibrated confidence
        """
        try:
            return confidence
        except Exception as e:
            self.logger.exception(failed(f'❌ Prediction calibration failed: {e}'))
            return confidence

    def _calculate_combined_confidence(self, predictions: dict[str, Any], analyst_confidence: float = 0.5) -> float:
        """
        Calculate combined confidence from Analyst and Tactician predictions.

        Args:
            predictions: Tactician predictions dictionary
            analyst_confidence: Analyst confidence score

        Returns:
            float: Combined confidence score
        """
        try:
            combined_confidence = analyst_confidence * self.confidence_weights['analyst_weight']
            for barrier_type, prediction in predictions.items():
                if prediction and 'confidence' in prediction:
                    confidence = prediction['confidence']
                    if barrier_type == 'fifty_percent':
                        weight = self.confidence_weights['fifty_percent_1m_weight']
                    elif barrier_type == 'twenty_five_percent':
                        weight = self.confidence_weights['twenty_five_percent_1m_weight']
                    elif barrier_type == 'fifty_percent_5m':
                        weight = self.confidence_weights['fifty_percent_5m_weight']
                    elif barrier_type == 'twenty_five_percent_5m':
                        weight = self.confidence_weights['twenty_five_percent_5m_weight']
                    else:
                        weight = 0.0
                    combined_confidence += confidence * weight
            return np.clip(combined_confidence, 0.0, 1.0)
        except Exception as e:
            self.logger.exception(failed(f'❌ Combined confidence calculation failed: {e}'))
            return 0.5

    def _evaluate_green_light_signal(self, predictions: dict[str, Any], combined_confidence: float) -> dict[str, Any]:
        """
        Evaluate green light signal based on predictions and thresholds.

        Args:
            predictions: Predictions dictionary
            combined_confidence: Combined confidence score

        Returns:
            dict: Green light signal evaluation
        """
        try:
            fifty_percent_ok = False
            twenty_five_percent_ok = False
            fifty_percent_confidences = []
            if 'fifty_percent' in predictions and predictions['fifty_percent']:
                fifty_percent_confidences.append(predictions['fifty_percent']['confidence'])
            if 'fifty_percent_5m' in predictions and predictions['fifty_percent_5m']:
                fifty_percent_confidences.append(predictions['fifty_percent_5m']['confidence'])
            if fifty_percent_confidences:
                fifty_percent_ok = max(fifty_percent_confidences) >= self.green_light_thresholds['fifty_percent']
            twenty_five_percent_confidences = []
            if 'twenty_five_percent' in predictions and predictions['twenty_five_percent']:
                twenty_five_percent_confidences.append(predictions['twenty_five_percent']['confidence'])
            if 'twenty_five_percent_5m' in predictions and predictions['twenty_five_percent_5m']:
                twenty_five_percent_confidences.append(predictions['twenty_five_percent_5m']['confidence'])
            if twenty_five_percent_confidences:
                twenty_five_percent_ok = max(twenty_five_percent_confidences) >= self.green_light_thresholds['twenty_five_percent']
            combined_ok = combined_confidence >= self.green_light_thresholds['combined_threshold']
            if fifty_percent_ok and twenty_five_percent_ok and combined_ok:
                signal = 'GREEN_LIGHT'
                reason = 'All thresholds met'
            elif combined_ok:
                signal = 'YELLOW_LIGHT'
                reason = 'Combined threshold met, individual thresholds partial'
            else:
                signal = 'RED_LIGHT'
                reason = 'Thresholds not met'
            return {'signal': signal, 'reason': reason, 'fifty_percent_ok': fifty_percent_ok, 'twenty_five_percent_ok': twenty_five_percent_ok, 'combined_ok': combined_ok, 'combined_confidence': combined_confidence, 'thresholds': self.green_light_thresholds}
        except Exception as e:
            self.logger.exception(failed(f'❌ Green light signal evaluation failed: {e}'))
            return {'signal': 'RED_LIGHT', 'reason': 'Evaluation failed', 'fifty_percent_ok': False, 'twenty_five_percent_ok': False, 'combined_ok': False, 'combined_confidence': 0.0, 'thresholds': self.green_light_thresholds}

    def _generate_tactician_triple_barrier_analysis(self, predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Generate triple barrier analysis for tactician predictions.
        Converts tactician predictions to price target format and applies triple barrier logic.
        
        Args:
            predictions: Tactician predictions dictionary
            
        Returns:
            dict: Triple barrier analysis results
        """
        try:
            upside_probabilities = {}
            downside_probabilities = {}
            for barrier_type, prediction in predictions.items():
                if not prediction:
                    continue
                confidence = prediction.get('confidence', 0.5)
                upper_barrier = prediction.get('upper_barrier', 0.01)
                lower_barrier = prediction.get('lower_barrier', -0.005)
                upper_pct = f'{upper_barrier * 100:.1f}%'
                lower_pct = f'{abs(lower_barrier) * 100:.1f}%'
                if upper_barrier > 0:
                    upside_probabilities[upper_pct] = confidence
                if lower_barrier < 0:
                    downside_probabilities[lower_pct] = confidence
            tactician_profit_take = 0.002
            tactician_stop_loss = 0.001
            tactician_confidence_threshold = 0.6
            cumulative_upper_confidence = 0.0
            upper_barrier_targets = []
            for target, prob in upside_probabilities.items():
                target_value = float(target.replace('%', ''))
                upper_barrier_value = tactician_profit_take * 100
                if target_value >= upper_barrier_value:
                    cumulative_upper_confidence += prob
                    upper_barrier_targets.append({'target': target, 'probability': prob, 'contribution': prob})
            cumulative_lower_confidence = 0.0
            lower_barrier_targets = []
            for target, prob in downside_probabilities.items():
                target_value = float(target.replace('%', ''))
                lower_barrier_value = tactician_stop_loss * 100
                if target_value >= lower_barrier_value:
                    cumulative_lower_confidence += prob
                    lower_barrier_targets.append({'target': target, 'probability': prob, 'contribution': prob})
            threshold_met = cumulative_upper_confidence >= tactician_confidence_threshold
            green_light = threshold_met and cumulative_upper_confidence > cumulative_lower_confidence and (cumulative_upper_confidence > 0.5)
            risk_reward_ratio = cumulative_upper_confidence / cumulative_lower_confidence if cumulative_lower_confidence > 0 else float('inf')
            return {'upper_barrier_threshold': f'{tactician_profit_take * 100:.1f}%', 'lower_barrier_threshold': f'{tactician_stop_loss * 100:.1f}%', 'confidence_threshold': tactician_confidence_threshold, 'cumulative_upper_confidence': float(cumulative_upper_confidence), 'cumulative_lower_confidence': float(cumulative_lower_confidence), 'threshold_met': threshold_met, 'green_light': green_light, 'risk_reward_ratio': float(risk_reward_ratio), 'upper_barrier_targets': upper_barrier_targets, 'lower_barrier_targets': lower_barrier_targets, 'decision_reasoning': self._get_tactician_ml_decision_reasoning(cumulative_upper_confidence, cumulative_lower_confidence, threshold_met, green_light), 'tactician_specific': {'barrier_types_analyzed': list(predictions.keys()), 'upside_probabilities': upside_probabilities, 'downside_probabilities': downside_probabilities}}
        except Exception as e:
            self.logger.error(f'Error generating tactician triple barrier analysis: {e}')
            return {'upper_barrier_threshold': '0.2%', 'lower_barrier_threshold': '0.1%', 'confidence_threshold': 0.6, 'cumulative_upper_confidence': 0.0, 'cumulative_lower_confidence': 0.0, 'threshold_met': False, 'green_light': False, 'risk_reward_ratio': 0.0, 'upper_barrier_targets': [], 'lower_barrier_targets': [], 'decision_reasoning': f'Error in calculation: {str(e)}', 'tactician_specific': {'error': str(e)}}

    def _get_tactician_ml_decision_reasoning(self, cumulative_upper_confidence: float, cumulative_lower_confidence: float, threshold_met: bool, green_light: bool) -> str:
        """
        Generate human-readable decision reasoning for tactician ML predictions.
        
        Args:
            cumulative_upper_confidence: Cumulative confidence for upper barrier
            cumulative_lower_confidence: Cumulative confidence for lower barrier
            threshold_met: Whether confidence threshold is met
            green_light: Whether green light decision is made
            
        Returns:
            str: Decision reasoning
        """
        if green_light:
            return f'TACTICIAN ML GREEN LIGHT: Upper barrier confidence ({cumulative_upper_confidence:.1%}) exceeds threshold (60.0%) and is higher than lower barrier confidence ({cumulative_lower_confidence:.1%})'
        elif threshold_met:
            return f'TACTICIAN ML THRESHOLD MET but NO GREEN LIGHT: Upper barrier confidence ({cumulative_upper_confidence:.1%}) meets threshold but lower barrier confidence ({cumulative_lower_confidence:.1%}) is too high'
        else:
            return f'TACTICIAN ML NO GREEN LIGHT: Upper barrier confidence ({cumulative_upper_confidence:.1%}) below threshold (60.0%)'

    async def _generate_micro_movement_prediction(self, movement_type: str, market_data: pd.DataFrame, symbol: str, timeframe: str) -> dict[str, Any]:
        """
        Generate prediction for a specific 0.3% micro movement type.

        Args:
            movement_type: "micro_immediate_long", "micro_immediate_short", "micro_short_long", "micro_short_short"
            market_data: Market data
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            dict: Micro movement prediction with probability
        """
        try:
            features = self._extract_features(market_data)
            
            if self.multi_output_models.get(movement_type, {}).get('model') == 'fallback':
                probability = self._generate_fallback_micro_probability(movement_type, features)
            else:
                probability = self._predict_micro_movement_with_model(movement_type, features)
            
            probability = np.clip(probability, 0.0, 1.0)
            
            config = self.micro_movement_config.get(
                movement_type.replace('_long', '').replace('_short', ''), 
                {'target': 0.003, 'timeframe': '1m', 'horizon': 'immediate'}
            )
            
            return {
                'probability': probability,
                'movement_type': movement_type,
                'target': config['target'],
                'timeframe': config['timeframe'],
                'horizon': config['horizon'],
                'direction': 'LONG' if '_long' in movement_type else 'SHORT'
            }
        except Exception as e:
            self.logger.exception(failed(f'❌ Micro movement prediction failed for {movement_type}: {e}'))
            return None

    def _calculate_directional_analysis(self, predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate directional analysis from micro movement predictions.

        Args:
            predictions: Dictionary of micro movement predictions

        Returns:
            dict: Directional analysis with bias and confidence
        """
        try:
            long_immediate = predictions.get('micro_immediate_long', {}).get('probability', 0.5)
            short_immediate = predictions.get('micro_immediate_short', {}).get('probability', 0.5)
            long_short = predictions.get('micro_short_long', {}).get('probability', 0.5)
            short_short = predictions.get('micro_short_short', {}).get('probability', 0.5)
            
            # Calculate overall directional probabilities
            long_overall = (long_immediate + long_short) / 2
            short_overall = (short_immediate + short_short) / 2
            
            # Calculate directional bias and confidence
            directional_difference = long_overall - short_overall
            directional_confidence = abs(directional_difference)
            directional_bias = 'LONG' if directional_difference > 0 else 'SHORT'
            
            # Calculate opportunity asymmetry
            opportunity_asymmetry = abs(long_overall - short_overall)
            
            return {
                'long_overall_probability': long_overall,
                'short_overall_probability': short_overall,
                'directional_bias': directional_bias,
                'directional_confidence': directional_confidence,
                'opportunity_asymmetry': opportunity_asymmetry,
                'immediate_vs_short_ratio': {
                    'long': long_immediate / max(long_short, 0.01),
                    'short': short_immediate / max(short_short, 0.01)
                }
            }
        except Exception as e:
            self.logger.exception(failed(f'❌ Directional analysis calculation failed: {e}'))
            return {
                'long_overall_probability': 0.5,
                'short_overall_probability': 0.5,
                'directional_bias': 'NEUTRAL',
                'directional_confidence': 0.0,
                'opportunity_asymmetry': 0.0,
                'immediate_vs_short_ratio': {'long': 1.0, 'short': 1.0}
            }

    def _generate_fallback_predictions(self) -> dict[str, Any]:
        """
        Generate fallback predictions when models are not available.

        Returns:
            dict: Fallback predictions for 0.3% micro movements
        """
        return {
            'micro_immediate_long': {'probability': 0.5, 'movement_type': 'micro_immediate_long', 'target': 0.003, 'timeframe': '1m', 'horizon': 'immediate', 'direction': 'LONG'},
            'micro_immediate_short': {'probability': 0.5, 'movement_type': 'micro_immediate_short', 'target': 0.003, 'timeframe': '1m', 'horizon': 'immediate', 'direction': 'SHORT'},
            'micro_short_long': {'probability': 0.5, 'movement_type': 'micro_short_long', 'target': 0.003, 'timeframe': '5m', 'horizon': 'short', 'direction': 'LONG'},
            'micro_short_short': {'probability': 0.5, 'movement_type': 'micro_short_short', 'target': 0.003, 'timeframe': '5m', 'horizon': 'short', 'direction': 'SHORT'},
            'combined_confidence': 0.5,
            'directional_analysis': {
                'long_overall_probability': 0.5,
                'short_overall_probability': 0.5,
                'directional_bias': 'NEUTRAL',
                'directional_confidence': 0.0,
                'opportunity_asymmetry': 0.0,
                'immediate_vs_short_ratio': {'long': 1.0, 'short': 1.0}
            },
            'green_light_signal': {
                'signal': 'RED_LIGHT', 
                'reason': 'Fallback mode - models not trained', 
                'micro_immediate_long_ok': False, 
                'micro_immediate_short_ok': False, 
                'combined_ok': False, 
                'combined_confidence': 0.5, 
                'thresholds': self.green_light_thresholds
            },
            'metadata': {
                'model_type': 'fallback_micro_movement', 
                'generation_timestamp': datetime.now().isoformat()
            }
        }

    @handles_errors(fallback = None)
    async def evaluate_exit_signal(self, current_predictions: dict[str, Any], position_context: dict[str, Any], market_conditions: dict[str, Any] = None) -> dict[str, Any]:
        """
        Evaluate exit signal based on 0.3% micro movement predictions and position context.

        Args:
            current_predictions: Current multi-output predictions with micro movements
            position_context: Current position context (including position direction)
            market_conditions: Optional market conditions (volatility, regime) for adaptive thresholds

        Returns:
            dict: Exit signal evaluation based on probability degradation with adaptive thresholds
        """
        try:
            combined_confidence = current_predictions.get('combined_confidence', 0.5)
            directional_analysis = current_predictions.get('directional_analysis', {})

            # Apply adaptive thresholds based on market conditions
            current_thresholds = self.exit_thresholds.copy()
            if market_conditions:
                current_volatility = market_conditions.get('volatility', 0.5)
                market_regime = market_conditions.get('regime', 'neutral')
                current_thresholds = self._apply_adaptive_thresholds(current_volatility, market_regime)

            # Get position direction from context
            position_side = position_context.get('side', '').upper()
            
            # Evaluate exit conditions based on position direction (immediate probabilities only)
            if position_side == 'LONG':
                micro_immediate_prob = current_predictions.get('micro_immediate_long', {}).get('probability', 0.5)
                
                # Check if immediate probability drops below exit threshold (using adaptive thresholds)
                immediate_exit = micro_immediate_prob <= current_thresholds['micro_immediate_long']
                
            elif position_side == 'SHORT':
                micro_immediate_prob = current_predictions.get('micro_immediate_short', {}).get('probability', 0.5)
                
                # Check if immediate probability drops below exit threshold (using adaptive thresholds)
                immediate_exit = micro_immediate_prob <= current_thresholds['micro_immediate_short']
                
            else:
                # No position or unknown direction
                immediate_exit = False
                micro_immediate_prob = 0.5
            
            # Check directional confidence degradation (MAIN EXIT TRIGGER for price reversals)
            directional_confidence = directional_analysis.get('directional_confidence', 0.0)
            directional_bias = directional_analysis.get('directional_bias', 'NEUTRAL')
            
            # Directional reversal detection - exit when direction confidence drops OR bias changes against position
            directional_reversal = (
                directional_confidence < current_thresholds['directional_confidence_min'] or
                (position_side == 'LONG' and directional_bias == 'SHORT') or
                (position_side == 'SHORT' and directional_bias == 'LONG')
            )

            # Check combined confidence exit
            combined_exit = combined_confidence <= current_thresholds['combined_exit_threshold']
            
            # Determine exit signal - PRIORITIZE directional reversal as main exit trigger
            if directional_reversal:
                exit_signal = 'EXIT'
                if directional_bias != 'NEUTRAL' and ((position_side == 'LONG' and directional_bias == 'SHORT') or (position_side == 'SHORT' and directional_bias == 'LONG')):
                    reason = f'DIRECTIONAL REVERSAL: Price direction changed from {position_side} to {directional_bias} (confidence: {directional_confidence:.3f})'
                else:
                    reason = f'DIRECTIONAL CONFIDENCE LOSS: Direction confidence ({directional_confidence:.3f}) below minimum ({current_thresholds["directional_confidence_min"]:.3f})'
            elif combined_exit:
                exit_signal = 'EXIT'
                reason = f'Combined confidence ({combined_confidence:.3f}) below threshold ({current_thresholds["combined_exit_threshold"]:.3f})'
            elif immediate_exit:
                exit_signal = 'EXIT'
                reason = f'Immediate probability degraded: {micro_immediate_prob:.3f} below threshold ({current_thresholds[f"micro_immediate_{position_side.lower()}"]:.3f})'
            else:
                exit_signal = 'HOLD'
                reason = f'No exit signals - immediate: {micro_immediate_prob:.3f}, directional: {directional_confidence:.3f} ({directional_bias}), combined: {combined_confidence:.3f}'
            
            return {
                'exit_signal': exit_signal,
                'reason': reason,
                'immediate_exit': immediate_exit,
                'directional_reversal': directional_reversal,
                'combined_exit': combined_exit,
                'combined_confidence': combined_confidence,
                'directional_confidence': directional_confidence,
                'directional_bias': directional_bias,
                'micro_immediate_prob': micro_immediate_prob,
                'exit_thresholds': current_thresholds,  # Return adaptive thresholds
                'position_side': position_side,
                'market_conditions_applied': market_conditions is not None,
                'adaptive_multiplier': current_thresholds.get('adaptive_multiplier', 1.0)
            }
        except Exception as e:
            self.logger.exception(failed(f'❌ Exit signal evaluation failed: {e}'))
            return {'exit_signal': 'HOLD', 'reason': 'Evaluation failed', 'fifty_percent_exit': False, 'twenty_five_percent_exit': False, 'combined_exit': False, 'combined_confidence': 0.5, 'exit_thresholds': self.exit_thresholds}

    def _apply_adaptive_thresholds(self, current_volatility: float, market_regime: str = 'neutral') -> dict[str, float]:
        """
        Apply adaptive thresholds based on market conditions.

        Args:
            current_volatility: Current market volatility level (0.0 to 1.0)
            market_regime: Market regime ('trending', 'ranging', 'volatile', 'neutral')

        Returns:
            dict: Adjusted thresholds for current market conditions
        """
        try:
            # Get base thresholds
            adjusted_thresholds = self.exit_thresholds.copy()

            # Volatility adjustment
            if current_volatility > 0.7:  # High volatility
                vol_multiplier = self.exit_thresholds.get('high_volatility_multiplier', 0.8)
            elif current_volatility < 0.3:  # Low volatility
                vol_multiplier = self.exit_thresholds.get('low_volatility_multiplier', 1.2)
            else:
                vol_multiplier = 1.0

            # Market regime adjustment
            if market_regime == 'trending':
                regime_multiplier = self.exit_thresholds.get('trending_market_multiplier', 0.9)
            elif market_regime == 'ranging':
                regime_multiplier = self.exit_thresholds.get('ranging_market_multiplier', 1.1)
            else:
                regime_multiplier = 1.0

            # Apply adjustments to key thresholds
            combined_multiplier = vol_multiplier * regime_multiplier
            adjusted_thresholds['combined_exit_threshold'] *= combined_multiplier
            adjusted_thresholds['directional_confidence_min'] *= combined_multiplier

            # Adjust immediate thresholds
            for key in ['micro_immediate_long', 'micro_immediate_short']:
                if key in adjusted_thresholds:
                    adjusted_thresholds[key] *= combined_multiplier

            self.logger.info(f"Applied adaptive thresholds: vol={current_volatility:.2f}, regime={market_regime}, multiplier={combined_multiplier:.2f}")
            adjusted_thresholds['adaptive_multiplier'] = combined_multiplier
            return adjusted_thresholds

        except Exception as e:
            self.logger.exception(f"❌ Adaptive threshold adjustment failed: {e}")
            return self.exit_thresholds.copy()


class ExitStrategyOptimizer:
    """
    Enhanced exit strategy optimizer with adaptive and ensemble methods.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize exit strategy optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('ExitStrategyOptimizer')

        # Exit strategy parameters
        self.exit_strategies = {
            'directional_reversal': {
                'weight': 0.4,
                'description': 'Exit on directional reversal'
            },
            'combined_confidence': {
                'weight': 0.3,
                'description': 'Exit on combined confidence degradation'
            },
            'immediate_probability': {
                'weight': 0.2,
                'description': 'Exit on immediate probability degradation'
            },
            'time_based': {
                'weight': 0.1,
                'description': 'Exit based on time constraints'
            }
        }

        # Market regime strategies
        self.regime_strategies = {
            'high_volatility': {
                'directional_weight': 0.5,
                'immediate_weight': 0.3,
                'confidence_weight': 0.2
            },
            'low_volatility': {
                'directional_weight': 0.3,
                'immediate_weight': 0.2,
                'confidence_weight': 0.5
            },
            'trending': {
                'directional_weight': 0.6,
                'immediate_weight': 0.2,
                'confidence_weight': 0.2
            },
            'ranging': {
                'directional_weight': 0.2,
                'immediate_weight': 0.4,
                'confidence_weight': 0.4
            }
        }

    def optimize_exit_strategy(self, market_data: pd.DataFrame, current_predictions: dict[str, Any],
                             position_context: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize exit strategy based on current market conditions.

        Args:
            market_data: Current market data
            current_predictions: ML predictions
            position_context: Current position information

        Returns:
            dict: Optimized exit strategy recommendations
        """
        try:
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(market_data)

            # Get base exit signals
            base_signals = self._get_base_exit_signals(current_predictions, position_context)

            # Apply ensemble methods
            ensemble_signals = self._apply_ensemble_methods(base_signals, market_conditions)

            # Generate recommendations
            recommendations = self._generate_exit_recommendations(ensemble_signals, market_conditions)

            return {
                'exit_strategy': recommendations,
                'market_conditions': market_conditions,
                'base_signals': base_signals,
                'ensemble_signals': ensemble_signals,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.exception(f"❌ Exit strategy optimization failed: {e}")
            return {'exit_strategy': 'HOLD', 'reason': 'Optimization failed'}

    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze current market conditions for adaptive thresholds."""
        try:
            # Calculate volatility
            if len(market_data) > 20:
                recent_data = market_data.tail(20)
                volatility = recent_data['close'].pct_change().std()
                volatility_score = min(1.0, volatility * 10)  # Normalize to 0-1
            else:
                volatility_score = 0.5

            # Determine regime (simplified)
            if volatility_score > 0.7:
                regime = 'high_volatility'
            elif volatility_score < 0.3:
                regime = 'low_volatility'
            elif self._is_trending(market_data):
                regime = 'trending'
            else:
                regime = 'ranging'

            return {
                'volatility': volatility_score,
                'regime': regime,
                'trend_strength': self._calculate_trend_strength(market_data)
            }

        except Exception as e:
            self.logger.exception(f"❌ Market condition analysis failed: {e}")
            return {'volatility': 0.5, 'regime': 'neutral', 'trend_strength': 0.5}

    def _is_trending(self, market_data: pd.DataFrame, window: int = 10) -> bool:
        """Check if market is trending."""
        try:
            if len(market_data) < window * 2:
                return False

            recent_prices = market_data['close'].tail(window).values
            older_prices = market_data['close'].tail(window * 2).head(window).values

            # Simple trend detection: check if recent prices are consistently higher/lower
            recent_trend = np.mean(np.diff(recent_prices))
            older_trend = np.mean(np.diff(older_prices))

            # If recent trend is stronger and consistent, consider it trending
            return abs(recent_trend) > abs(older_trend) * 0.5

        except Exception:
            return False

    def _calculate_trend_strength(self, market_data: pd.DataFrame, window: int = 20) -> float:
        """Calculate trend strength (0-1)."""
        try:
            if len(market_data) < window:
                return 0.5

            prices = market_data['close'].tail(window).values
            trend_strength = abs(np.polyfit(np.arange(window), prices, 1)[0])
            return min(1.0, trend_strength * 100)  # Normalize

        except Exception:
            return 0.5

    def _get_base_exit_signals(self, current_predictions: dict[str, Any], position_context: dict[str, Any]) -> dict[str, Any]:
        """Get base exit signals from different strategies."""
        try:
            # Extract position information
            position_side = position_context.get('side', 'LONG')
            entry_time = position_context.get('entry_time')
            hold_time = (datetime.now() - entry_time).total_seconds() if entry_time else 0

            # Time-based signal
            time_signal = 0.0
            if hold_time > 10800:  # 3 hours
                time_signal = 1.0
            elif hold_time > 7200:  # 2 hours
                time_signal = 0.5

            # ML-based signals (simplified)
            directional_signal = current_predictions.get('directional_reversal', 0.0)
            confidence_signal = current_predictions.get('combined_confidence', 0.5)
            immediate_signal = current_predictions.get('immediate_probability', 0.5)

            return {
                'directional': directional_signal,
                'confidence': 1.0 - confidence_signal,  # Invert confidence for exit signal
                'immediate': 1.0 - immediate_signal,   # Invert probability for exit signal
                'time_based': time_signal
            }

        except Exception as e:
            self.logger.exception(f"❌ Base exit signals calculation failed: {e}")
            return {'directional': 0.0, 'confidence': 0.0, 'immediate': 0.0, 'time_based': 0.0}

    def _apply_ensemble_methods(self, base_signals: dict[str, float], market_conditions: dict[str, Any]) -> dict[str, float]:
        """Apply ensemble methods to combine exit signals."""
        try:
            regime = market_conditions.get('regime', 'neutral')

            # Get regime-specific weights
            regime_weights = self.regime_strategies.get(regime, {
                'directional_weight': 0.4,
                'immediate_weight': 0.3,
                'confidence_weight': 0.3
            })

            # Calculate weighted ensemble signal
            ensemble_signal = (
                base_signals['directional'] * regime_weights['directional_weight'] +
                base_signals['immediate'] * regime_weights['immediate_weight'] +
                base_signals['confidence'] * regime_weights['confidence_weight'] +
                base_signals['time_based'] * 0.1  # Time always has small weight
            )

            # Apply volatility adjustment
            volatility = market_conditions.get('volatility', 0.5)
            if volatility > 0.7:
                ensemble_signal *= 1.2  # Higher sensitivity in high volatility
            elif volatility < 0.3:
                ensemble_signal *= 0.8  # Lower sensitivity in low volatility

            return {
                'ensemble_signal': min(1.0, ensemble_signal),
                'base_signals': base_signals,
                'regime_weights': regime_weights,
                'volatility_adjustment': 1.2 if volatility > 0.7 else (0.8 if volatility < 0.3 else 1.0)
            }

        except Exception as e:
            self.logger.exception(f"❌ Ensemble methods failed: {e}")
            return {'ensemble_signal': 0.0, 'base_signals': base_signals}

    def _generate_exit_recommendations(self, ensemble_signals: dict[str, float], market_conditions: dict[str, Any]) -> dict[str, Any]:
        """Generate exit recommendations based on ensemble signals."""
        try:
            ensemble_signal = ensemble_signals.get('ensemble_signal', 0.0)

            # Exit thresholds based on market conditions
            regime = market_conditions.get('regime', 'neutral')
            volatility = market_conditions.get('volatility', 0.5)

            # Adaptive exit threshold
            base_threshold = 0.6
            if regime == 'high_volatility':
                threshold = base_threshold * 0.8  # Lower threshold in high volatility
            elif regime == 'low_volatility':
                threshold = base_threshold * 1.2  # Higher threshold in low volatility
            else:
                threshold = base_threshold

            # Adjust for volatility
            threshold *= (1.0 - volatility * 0.2)  # Lower threshold when volatility increases

            if ensemble_signal >= threshold:
                recommendation = 'EXIT'
                reason = f'Ensemble signal ({ensemble_signal:.3f}) exceeds threshold ({threshold:.3f}) in {regime} regime'
            else:
                recommendation = 'HOLD'
                reason = f'Ensemble signal ({ensemble_signal:.3f}) below threshold ({threshold:.3f}) in {regime} regime'

            return {
                'action': recommendation,
                'confidence': ensemble_signal,
                'threshold': threshold,
                'reason': reason,
                'market_regime': regime,
                'volatility': volatility
            }

        except Exception as e:
            self.logger.exception(f"❌ Exit recommendations failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Recommendation failed'}

@handles_errors(fallback = None)
async def setup_ml_tactics_manager(config: dict[str, Any] | None = None) -> MLTacticsManager | None:
    """
    Setup and return a configured MLTacticsManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        MLTacticsManager: Configured ML tactics manager instance
    """
    try:
        manager = MLTacticsManager(config or {})
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.exception(failed(f'Failed to setup ML Tactics Manager: {e}'))
        return None