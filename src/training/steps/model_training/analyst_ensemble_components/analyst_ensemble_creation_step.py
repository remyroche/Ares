"""Step 13: Analyst Ensemble Creation - Migrated to use BaseStep pattern.

This step combines multiple analyst models into ensemble predictions with advanced voting mechanisms.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.core.decorators import handles_errors, log_execution_time, validates
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from .ensemble_aggregator import EnsembleAggregator
from .voting_mechanism import VotingMechanism
from .weight_optimizer import WeightOptimizer
from .ensemble_evaluator import EnsembleEvaluator
from copy import copy
from src.core.decorators.errors import handles_errors

class AnalystEnsembleCreationStep(BaseStep):
    """Step 13: Analyst Ensemble Creation with advanced voting mechanisms."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Analyst Ensemble Creation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '13', 'analyst_ensemble_creation')

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.ensemble_aggregator = EnsembleAggregator(self.config)
        self.voting_mechanism = VotingMechanism(self.config)
        self.weight_optimizer = WeightOptimizer(self.config)
        self.ensemble_evaluator = EnsembleEvaluator(self.config)
        self.ensemble_config = self._initialize_ensemble_config()
        self.ensemble_models: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, Dict[str, float]] = {}
        self.ensemble_metrics: Dict[str, Any] = {}

    def _initialize_ensemble_config(self) -> Dict[str, Any]:
        """Initialize ensemble-specific configuration."""
        return {'aggregation_methods': ['voting', 'weighted', 'stacking', 'blending'], 'voting_types': ['hard', 'soft', 'weighted_soft'], 'weight_optimization': True, 'dynamic_weighting': True, 'regime_aware_aggregation': True, 'min_models_for_ensemble': 3, 'ensemble_diversity_threshold': 0.3, 'cross_validation_folds': 5, 'meta_learner_type': 'logistic_regression'}

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'step12_analyst_enhancement_completed' not in pipeline_state:
            errors.append('Step 12 (Analyst Enhancement) must be completed before ensemble creation')
        if 'enhanced_analyst_models' not in pipeline_state:
            errors.append('No enhanced analyst models found in pipeline state')
        if 'features' not in pipeline_state:
            errors.append('No feature data found for ensemble validation')
        required_config = ['ensemble_creation', 'voting_mechanism', 'weight_optimization']
        for key in required_config:
            if key not in self.config:
                errors.append(f'Missing required configuration: {key}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='analyst ensemble creation logic')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main analyst ensemble creation logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with ensemble models
        """
        self.logger.info('🎯 Starting analyst ensemble creation...')
        enhanced_models = pipeline_state['enhanced_analyst_models']
        features = pipeline_state.get('features', {})
        regime_data = pipeline_state.get('regime_data', {})
        regime_ensembles = {}
        regime_ensemble_metrics = {}
        for regime_id, regime_models in enhanced_models.items():
            self.logger.info(f'📊 Creating ensemble for regime {regime_id}...')
            if len(regime_models) < self.ensemble_config['min_models_for_ensemble']:
                self.logger.warning(f"Regime {regime_id} has only {len(regime_models)} models, minimum {self.ensemble_config['min_models_for_ensemble']} required")
                continue
            regime_features = self._get_regime_features(features, regime_data.get(regime_id, {}))
            regime_ensemble = await self._create_regime_ensemble(regime_id, regime_models, regime_features)
            if regime_ensemble:
                regime_ensembles[regime_id] = regime_ensemble['ensemble']
                regime_ensemble_metrics[regime_id] = regime_ensemble['metrics']
        if len(regime_ensembles) > 1:
            self.logger.info('🔄 Creating cross-regime ensemble...')
            cross_regime_ensemble = await self._create_cross_regime_ensemble(regime_ensembles, features)
            regime_ensembles['cross_regime'] = cross_regime_ensemble['ensemble']
            regime_ensemble_metrics['cross_regime'] = cross_regime_ensemble['metrics']
        if self.ensemble_config['weight_optimization']:
            self.logger.info('⚖️ Optimizing ensemble weights...')
            optimized_weights = await self.weight_optimizer.optimize_weights(regime_ensembles, features)
            self.ensemble_weights = optimized_weights
        result = pipeline_state.copy()
        result['analyst_ensembles'] = regime_ensembles
        result['ensemble_weights'] = self.ensemble_weights
        result['ensemble_metrics'] = regime_ensemble_metrics
        result['ensemble_summary'] = self._create_ensemble_summary(regime_ensembles, regime_ensemble_metrics)
        return result

    async def _create_regime_ensemble(self, regime_id: str, regime_models: Dict[str, Any], regime_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create ensemble for a specific regime.
        
        Args:
            regime_id: Regime identifier
            regime_models: Models for this regime
            regime_features: Features for this regime
            
        Returns:
            Dictionary with ensemble and metrics
        """
        try:
            prepared_models = self._prepare_models_for_aggregation(regime_models)
            diversity_score = await self._calculate_model_diversity(prepared_models, regime_features)
            if diversity_score < self.ensemble_config['ensemble_diversity_threshold']:
                self.logger.warning(f'Low model diversity ({diversity_score:.3f}) for regime {regime_id}')
            ensembles = {}
            if 'voting' in self.ensemble_config['aggregation_methods']:
                voting_ensemble = await self.voting_mechanism.create_voting_ensemble(prepared_models, self.ensemble_config['voting_types'])
                ensembles['voting'] = voting_ensemble
            if 'weighted' in self.ensemble_config['aggregation_methods']:
                weighted_ensemble = await self.ensemble_aggregator.create_weighted_ensemble(prepared_models, regime_features)
                ensembles['weighted'] = weighted_ensemble
            if 'stacking' in self.ensemble_config['aggregation_methods']:
                stacking_ensemble = await self.ensemble_aggregator.create_stacking_ensemble(prepared_models, regime_features, self.ensemble_config['meta_learner_type'])
                ensembles['stacking'] = stacking_ensemble
            if 'blending' in self.ensemble_config['aggregation_methods']:
                blending_ensemble = await self.ensemble_aggregator.create_blending_ensemble(prepared_models, regime_features)
                ensembles['blending'] = blending_ensemble
            metrics = await self.ensemble_evaluator.evaluate_ensembles(ensembles, regime_features)
            best_ensemble_type = max(metrics.items(), key=lambda x: x[1]['accuracy'])[0]
            self.logger.info(f"✅ Created {len(ensembles)} ensemble types for regime {regime_id}, best: {best_ensemble_type} (accuracy: {metrics[best_ensemble_type]['accuracy']:.4f})")
            return {'ensemble': ensembles, 'metrics': metrics, 'best_type': best_ensemble_type, 'diversity_score': diversity_score}
        except Exception as e:
            self.logger.error(f'Failed to create ensemble for regime {regime_id}: {str(e)}')
            return None

    async def _create_cross_regime_ensemble(self, regime_ensembles: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """Create an ensemble across different regimes.
        
        Args:
            regime_ensembles: Ensembles for each regime
            features: All feature data
            
        Returns:
            Dictionary with cross-regime ensemble and metrics
        """
        try:
            best_regime_models = {}
            for regime_id, ensemble_data in regime_ensembles.items():
                if isinstance(ensemble_data, dict) and 'ensemble' in ensemble_data:
                    best_type = ensemble_data.get('best_type', 'voting')
                    best_model = ensemble_data['ensemble'].get(best_type)
                    if best_model:
                        best_regime_models[f'{regime_id}_{best_type}'] = best_model
            meta_ensemble = await self.ensemble_aggregator.create_meta_ensemble(best_regime_models, features)
            metrics = await self.ensemble_evaluator.evaluate_single_ensemble(meta_ensemble, features)
            return {'ensemble': {'meta': meta_ensemble}, 'metrics': {'meta': metrics}, 'best_type': 'meta', 'regime_models': list(best_regime_models.keys())}
        except Exception as e:
            self.logger.error(f'Failed to create cross-regime ensemble: {str(e)}')
            return {'ensemble': {}, 'metrics': {}, 'best_type': None, 'error': str(e)}

    def _prepare_models_for_aggregation(self, regime_models: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare models for ensemble aggregation."""
        prepared = {}
        for model_name, model_info in regime_models.items():
            if isinstance(model_info, dict) and 'model' in model_info:
                prepared[model_name] = {'model': model_info['model'], 'features': model_info.get('features', []), 'performance': model_info.get('validation_accuracy', 0.5)}
        return prepared

    async def _calculate_model_diversity(self, models: Dict[str, Any], features: pd.DataFrame) -> float:
        """Calculate diversity score for a set of models."""
        if len(models) < 2 or features.empty:
            return 0.0
        try:
            predictions = []
            sample_size = min(1000, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            X_sample = features.iloc[sample_indices]
            for model_info in models.values():
                model = model_info['model']
                model_features = model_info.get('features', features.columns.tolist())
                X_model = X_sample[model_features] if model_features else X_sample
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_model)[:, 1]
                else:
                    pred = model.predict(X_model)
                predictions.append(pred)
            predictions = np.array(predictions)
            n_models = len(predictions)
            disagreement = 0.0
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    disagreement += np.mean(predictions[i] != predictions[j])
            diversity_score = disagreement / (n_models * (n_models - 1) / 2)
            return diversity_score
        except Exception as e:
            self.logger.warning(f'Failed to calculate model diversity: {str(e)}')
            return 0.0

    def _get_regime_features(self, features: pd.DataFrame, regime_info: Dict[str, Any]) -> pd.DataFrame:
        """Extract features for a specific regime."""
        if features.empty:
            return pd.DataFrame()
        regime_mask = regime_info.get('mask', [])
        if isinstance(regime_mask, list) and regime_mask:
            regime_mask = np.array(regime_mask)
            return features[regime_mask]
        return features

    def _create_ensemble_summary(self, regime_ensembles: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of all ensembles."""
        summary = {'total_regimes': len(regime_ensembles), 'total_ensemble_types': 0, 'best_performers': {}, 'average_metrics': {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}}
        for regime_id, ensemble_data in regime_ensembles.items():
            if isinstance(ensemble_data, dict) and 'ensemble' in ensemble_data:
                summary['total_ensemble_types'] += len(ensemble_data['ensemble'])
                if 'best_type' in ensemble_data and regime_id in metrics:
                    best_metrics = metrics[regime_id].get(ensemble_data['best_type'], {})
                    summary['best_performers'][regime_id] = {'type': ensemble_data['best_type'], 'accuracy': best_metrics.get('accuracy', 0.0)}
                    for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                        if metric in best_metrics:
                            summary['average_metrics'][metric].append(best_metrics[metric])
        for metric, values in summary['average_metrics'].items():
            if values:
                summary['average_metrics'][metric] = np.mean(values)
            else:
                summary['average_metrics'][metric] = 0.0
        return summary

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        required_outputs = ['analyst_ensembles', 'ensemble_weights', 'ensemble_metrics', 'ensemble_summary']
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f'Missing required output: {output}')
        if 'analyst_ensembles' in pipeline_state:
            ensembles = pipeline_state['analyst_ensembles']
            if not isinstance(ensembles, dict):
                errors.append('Analyst ensembles must be a dictionary')
            elif len(ensembles) == 0:
                errors.append('No ensembles were created')
        if 'ensemble_summary' in pipeline_state:
            summary = pipeline_state['ensemble_summary']
            if summary.get('total_ensemble_types', 0) == 0:
                errors.append('No ensemble types were created')
        return (len(errors) == 0, errors)

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['enhanced_analyst_models', 'features', 'step12_analyst_enhancement_completed']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return ['analyst_ensembles', 'ensemble_weights', 'ensemble_metrics', 'ensemble_summary']

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['step12_analyst_enhancement']