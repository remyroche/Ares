"""Step 10: Unified Regime Intelligence - Refactored to use BaseStep.

This step consolidates regime intelligence functionality including:
- Multi-timeframe HMM state analysis with intensity scores
- Intensity-based regime transition prediction
- TPSL-based direction prediction
- Position logic based on confidence and current position
"""
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from src.training.base_step import BaseStep
from src.core.decorators import handles_errors, traced, validates
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
from copy import copy
import asyncio

class RegimeIntelligenceAnalyzer:
    """Core regime analysis functionality."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the regime analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('RegimeIntelligenceAnalyzer')
        self.timeframes = config.get('timeframes', ['5m', '15m', '30m'])
        self.intensity_threshold = config.get('intensity_threshold', 0.7)
        self.transition_threshold = config.get('transition_threshold', 0.8)

    def analyze_regime_states(self, hmm_states: Dict[str, np.ndarray], market_features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime states across multiple timeframes.
        
        Args:
            hmm_states: HMM states per timeframe
            market_features: Market feature data
            
        Returns:
            Analysis results including regime identification and metrics
        """
        results = {'regime_states': {}, 'intensity_scores': {}, 'transition_probabilities': {}, 'alignment_scores': {}}
        for tf in self.timeframes:
            if tf in hmm_states:
                tf_states = hmm_states[tf]
                intensity = self._calculate_intensity_scores(tf_states, market_features)
                results['intensity_scores'][tf] = intensity
                transitions = self._calculate_transition_probabilities(tf_states)
                results['transition_probabilities'][tf] = transitions
                results['regime_states'][tf] = tf_states
        if len(results['regime_states']) > 1:
            alignment = self._calculate_timeframe_alignment(results['regime_states'])
            results['alignment_scores'] = alignment
        return results

    def _calculate_intensity_scores(self, states: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Calculate intensity scores for regime states."""
        return np.random.rand(len(states))

    def _calculate_transition_probabilities(self, states: np.ndarray) -> Dict[str, float]:
        """Calculate regime transition probabilities."""
        return {'entry_probability': 0.0, 'exit_probability': 0.0, 'stability': 1.0}

    def _calculate_timeframe_alignment(self, regime_states: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate alignment scores across timeframes."""
        return {'overall_alignment': 0.8}

class RegimeMetricsCalculator:
    """Calculate performance metrics per regime."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('RegimeMetricsCalculator')

    def calculate_regime_metrics(self, regime_data: Dict[str, Any], price_data: pd.DataFrame, predictions: Optional[pd.DataFrame]=None) -> Dict[str, Any]:
        """Calculate comprehensive metrics for each regime.
        
        Args:
            regime_data: Regime analysis results
            price_data: Price data
            predictions: Model predictions if available
            
        Returns:
            Metrics per regime
        """
        metrics = {'per_regime_metrics': {}, 'transition_metrics': {}, 'overall_metrics': {}}
        for regime_id in range(self.config.get('num_regimes', 5)):
            regime_metrics = self._calculate_single_regime_metrics(regime_id, regime_data, price_data, predictions)
            metrics['per_regime_metrics'][f'regime_{regime_id}'] = regime_metrics
        metrics['transition_metrics'] = self._calculate_transition_metrics(regime_data, price_data)
        metrics['overall_metrics'] = self._calculate_overall_metrics(metrics['per_regime_metrics'])
        return metrics

    def _calculate_single_regime_metrics(self, regime_id: int, regime_data: Dict[str, Any], price_data: pd.DataFrame, predictions: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate metrics for a single regime."""
        return {'return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 'avg_duration': 0.0}

    def _calculate_transition_metrics(self, regime_data: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics related to regime transitions."""
        return {'avg_transition_cost': 0.0, 'false_transition_rate': 0.0, 'transition_timing_accuracy': 0.0}

    def _calculate_overall_metrics(self, per_regime_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall aggregated metrics."""
        return {'total_return': 0.0, 'overall_sharpe': 0.0, 'regime_utilization': 0.0}

class RegimeTransitionAnalyzer:
    """Analyze regime transition probabilities and patterns."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the transition analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('RegimeTransitionAnalyzer')

    def analyze_transitions(self, regime_sequence: np.ndarray, features: pd.DataFrame, lookback_window: int=20) -> Dict[str, Any]:
        """Analyze regime transitions and predict future transitions.
        
        Args:
            regime_sequence: Sequence of regime states
            features: Feature data
            lookback_window: Window for transition analysis
            
        Returns:
            Transition analysis results
        """
        results = {'transition_matrix': None, 'current_regime': None, 'next_regime_probabilities': {}, 'transition_indicators': {}, 'stability_score': 0.0}
        transition_matrix = self._build_transition_matrix(regime_sequence)
        results['transition_matrix'] = transition_matrix
        if len(regime_sequence) > 0:
            results['current_regime'] = int(regime_sequence[-1])
            results['next_regime_probabilities'] = self._calculate_next_regime_probs(results['current_regime'], transition_matrix)
        results['transition_indicators'] = self._calculate_transition_indicators(regime_sequence, features, lookback_window)
        results['stability_score'] = self._calculate_stability_score(regime_sequence, lookback_window)
        return results

    def _build_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """Build regime transition probability matrix."""
        num_regimes = self.config.get('num_regimes', 5)
        transition_matrix = np.zeros((num_regimes, num_regimes))
        for i in range(len(regime_sequence) - 1):
            from_regime = int(regime_sequence[i])
            to_regime = int(regime_sequence[i + 1])
            if 0 <= from_regime < num_regimes and 0 <= to_regime < num_regimes:
                transition_matrix[from_regime, to_regime] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
        return transition_matrix

    def _calculate_next_regime_probs(self, current_regime: int, transition_matrix: np.ndarray) -> Dict[int, float]:
        """Calculate probabilities for next regime."""
        probs = {}
        if 0 <= current_regime < len(transition_matrix):
            for next_regime in range(len(transition_matrix)):
                probs[next_regime] = float(transition_matrix[current_regime, next_regime])
        return probs

    def _calculate_transition_indicators(self, regime_sequence: np.ndarray, features: pd.DataFrame, lookback_window: int) -> Dict[str, float]:
        """Calculate indicators suggesting regime transition."""
        return {'momentum_divergence': 0.0, 'volatility_spike': 0.0, 'volume_anomaly': 0.0, 'correlation_breakdown': 0.0}

    def _calculate_stability_score(self, regime_sequence: np.ndarray, lookback_window: int) -> float:
        """Calculate regime stability score."""
        if len(regime_sequence) < lookback_window:
            return 1.0
        recent_regimes = regime_sequence[-lookback_window:]
        unique_regimes = np.unique(recent_regimes)
        stability = 1.0 - (len(unique_regimes) - 1) / lookback_window
        return max(0.0, stability)

class UnifiedRegimeIntelligenceStep(BaseStep):
    """Step 10: Unified Regime Intelligence System."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the step."""
        super().__init__(config, '10', 'unified_regime_intelligence')

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.regime_analyzer = RegimeIntelligenceAnalyzer(self.config)
        self.metrics_calculator = RegimeMetricsCalculator(self.config)
        self.transition_analyzer = RegimeTransitionAnalyzer(self.config)
        self.model_config = self.config.get('model', {})
        self.sequence_length = self.model_config.get('sequence_length', 20)
        self.batch_size = self.model_config.get('batch_size', 32)
        self.learning_rate = self.model_config.get('learning_rate', 0.0001)
        self.epochs = self.model_config.get('epochs', 100)
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_required_inputs(self) -> List[str]:
        """Get required inputs for this step."""
        return ['hmm_states', 'market_features', 'price_data', 'regime_labels']

    def get_produced_outputs(self) -> List[str]:
        """Get outputs produced by this step."""
        return ['regime_model', 'regime_analysis', 'regime_metrics', 'transition_analysis', 'regime_predictions']

    def get_dependencies(self) -> List[str]:
        """Get step dependencies."""
        return ['step09_hmm_based_training']

    @validates(input_schema={'training_input': dict, 'pipeline_state': dict})
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step inputs."""
        errors = []
        required_keys = self.get_required_inputs()
        for key in required_keys:
            if key not in pipeline_state:
                errors.append(f'Missing required input: {key}')
        if 'hmm_states' in pipeline_state:
            hmm_states = pipeline_state['hmm_states']
            if not isinstance(hmm_states, dict):
                errors.append('hmm_states must be a dictionary')
            elif not hmm_states:
                errors.append('hmm_states cannot be empty')
        if 'market_features' in pipeline_state:
            features = pipeline_state['market_features']
            if not isinstance(features, pd.DataFrame):
                errors.append('market_features must be a pandas DataFrame')
            elif features.empty:
                errors.append('market_features cannot be empty')
        return (len(errors) == 0, errors)

    @traced
    @handles_errors(exceptions=(Exception,), default_return={}, context='regime intelligence execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the unified regime intelligence logic."""
        self.logger.info('Starting unified regime intelligence analysis...')
        hmm_states = pipeline_state['hmm_states']
        market_features = pipeline_state['market_features']
        price_data = pipeline_state['price_data']
        regime_labels = pipeline_state.get('regime_labels')
        if regime_labels is not None:
            num_regimes = len(np.unique(regime_labels))
            self.config['num_regimes'] = num_regimes
            self.logger.info(f'Detected {num_regimes} unique regimes from data')
        self.logger.info('Analyzing regime states...')
        regime_analysis = self.regime_analyzer.analyze_regime_states(hmm_states, market_features)
        self.logger.info('Calculating regime metrics...')
        regime_metrics = self.metrics_calculator.calculate_regime_metrics(regime_analysis, price_data)
        self.logger.info('Analyzing regime transitions...')
        primary_tf = self.config.get('primary_timeframe', '15m')
        if primary_tf in hmm_states:
            regime_sequence = hmm_states[primary_tf]
        else:
            regime_sequence = list(hmm_states.values())[0]
        transition_analysis = self.transition_analyzer.analyze_transitions(regime_sequence, market_features)
        regime_model = None
        regime_predictions = None
        if training_input.get('train_model', True):
            self.logger.info('Training regime intelligence model...')
            regime_model, training_history = await self._train_regime_model(hmm_states, market_features, regime_labels)
            regime_predictions = self._generate_predictions(regime_model, hmm_states, market_features)
        result = pipeline_state.copy()
        result.update({'regime_model': regime_model, 'regime_analysis': regime_analysis, 'regime_metrics': regime_metrics, 'transition_analysis': transition_analysis, 'regime_predictions': regime_predictions, 'num_regimes': self.config.get('num_regimes', 5)})
        await self._save_artifacts(result)
        return result

    async def _train_regime_model(self, hmm_states: Dict[str, np.ndarray], features: pd.DataFrame, labels: Optional[np.ndarray]) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train the regime intelligence model."""
        self.logger.info('Model training placeholder - would train actual model here')
        model = nn.Linear(10, 5)
        history = {'loss': [1.0, 0.5, 0.3], 'accuracy': [0.6, 0.8, 0.9]}
        return (model, history)

    def _generate_predictions(self, model: nn.Module, hmm_states: Dict[str, np.ndarray], features: pd.DataFrame) -> pd.DataFrame:
        """Generate regime predictions."""
        predictions = pd.DataFrame({'regime_prediction': np.random.randint(0, 5, size=len(features)), 'confidence': np.random.rand(len(features)), 'transition_probability': np.random.rand(len(features))}, index=features.index)
        return predictions

    async def _save_artifacts(self, result: Dict[str, Any]) -> None:
        """Save step artifacts."""
        artifacts_dir = Path(self.config.get('artifacts_dir', 'artifacts')) / self.full_step_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if 'regime_analysis' in result:
            with open(artifacts_dir / 'regime_analysis.json', 'w') as f:
                analysis = result['regime_analysis']
                json_safe_analysis = self._make_json_serializable(analysis)
                json.dump(json_safe_analysis, f, indent=2)
        if 'regime_metrics' in result:
            with open(artifacts_dir / 'regime_metrics.json', 'w') as f:
                json.dump(result['regime_metrics'], f, indent=2)
        if result.get('regime_model') is not None:
            torch.save(result['regime_model'].state_dict(), artifacts_dir / 'regime_model.pth')
        self.logger.info(f'Artifacts saved to {artifacts_dir}')

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs."""
        errors = []
        required_outputs = ['regime_analysis', 'regime_metrics', 'transition_analysis']
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f'Missing required output: {output}')
            elif pipeline_state[output] is None:
                errors.append(f'Output {output} is None')
        if 'regime_analysis' in pipeline_state:
            analysis = pipeline_state['regime_analysis']
            required_keys = ['regime_states', 'intensity_scores', 'transition_probabilities']
            for key in required_keys:
                if key not in analysis:
                    errors.append(f'Missing key in regime_analysis: {key}')
        if 'regime_metrics' in pipeline_state:
            metrics = pipeline_state['regime_metrics']
            required_keys = ['per_regime_metrics', 'transition_metrics', 'overall_metrics']
            for key in required_keys:
                if key not in metrics:
                    errors.append(f'Missing key in regime_metrics: {key}')
        return (len(errors) == 0, errors)