"""Step 12: Analyst Enhancement - Migrated to use BaseStep pattern.

This step refines the trained analyst models through a regime-specific sequential process.
"""
import asyncio
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.core.decorators import handles_errors, log_execution_time, validates
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from .hyperparameter_optimizer import HyperparameterOptimizer
from .feature_selector import FeatureSelector
from .model_optimizer import ModelOptimizer
from .ensemble_creator import EnsembleCreator
from copy import copy
from src.core.decorators.errors import handles_errors

class AnalystEnhancementStep(BaseStep):
    """Step 12: Analyst Enhancement with regime-aware optimization."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Analyst Enhancement step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '12', 'analyst_enhancement')

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        self.ensemble_creator = EnsembleCreator(self.config)
        self.regime_config = self._initialize_regime_config()
        self.regime_enhancement_results: Dict[str, Dict[str, Any]] = {}
        self.regime_optimized_models: Dict[str, Any] = {}

    def _initialize_regime_config(self) -> Dict[str, Any]:
        """Initialize regime-specific configuration."""
        return {'regime_specific_hpo': True, 'regime_specific_feature_selection': True, 'regime_specific_model_optimization': True, 'min_regime_samples': 100, 'regime_validation_split': 0.2, 'regime_parallel_processing': True, 'regime_memory_optimization': True, 'max_regime_models': 10, 'regime_ensemble_voting': 'weighted'}

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'step11_analyst_creation_completed' not in pipeline_state:
            errors.append('Step 11 (Analyst Creation) must be completed before enhancement')
        if 'analyst_models' not in pipeline_state:
            errors.append('No analyst models found in pipeline state')
        if 'regime_data' not in pipeline_state:
            errors.append('No regime data found in pipeline state')
        if 'features' not in pipeline_state:
            errors.append('No feature data found in pipeline state')
        required_config = ['hyperparameter_optimization', 'feature_selection', 'model_optimization']
        for key in required_config:
            if key not in self.config:
                errors.append(f'Missing required configuration: {key}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='analyst enhancement logic')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main analyst enhancement logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with enhanced models
        """
        self.logger.info('🔧 Starting analyst model enhancement...')
        analyst_models = pipeline_state['analyst_models']
        regime_data = pipeline_state['regime_data']
        features = pipeline_state['features']
        enhanced_models = {}
        enhancement_results = {}
        for regime_id, regime_info in regime_data.items():
            self.logger.info(f'📊 Processing regime {regime_id}...')
            regime_features = self._get_regime_features(features, regime_info)
            regime_models = self._get_regime_models(analyst_models, regime_id)
            if not regime_models:
                self.logger.warning(f'No models found for regime {regime_id}, skipping...')
                continue
            regime_enhanced = await self._enhance_regime_models(regime_id, regime_models, regime_features, regime_info)
            enhanced_models[regime_id] = regime_enhanced['models']
            enhancement_results[regime_id] = regime_enhanced['results']
        self.logger.info('🔄 Creating ensemble models...')
        ensemble_models = await self.ensemble_creator.create_ensembles(enhanced_models, regime_data)
        result = pipeline_state.copy()
        result['enhanced_analyst_models'] = enhanced_models
        result['analyst_enhancement_results'] = enhancement_results
        result['analyst_ensemble_models'] = ensemble_models
        result['analyst_enhancement_metrics'] = self._calculate_enhancement_metrics(enhancement_results)
        return result

    async def _enhance_regime_models(self, regime_id: str, models: Dict[str, Any], features: pd.DataFrame, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance models for a specific regime.
        
        Args:
            regime_id: Regime identifier
            models: Dictionary of models for this regime
            features: Feature data for this regime
            regime_info: Regime metadata
            
        Returns:
            Dictionary with enhanced models and results
        """
        enhanced_models = {}
        enhancement_results = {}
        X_train, y_train, X_val, y_val = self._split_regime_data(features, regime_info)
        for model_name, model in models.items():
            self.logger.info(f'  🎯 Enhancing {model_name} for regime {regime_id}...')
            try:
                if self.config.get('hyperparameter_optimization', {}).get('enabled', True):
                    optimized_params = await self.hyperparameter_optimizer.optimize(model, X_train, y_train, X_val, y_val, regime_id)
                else:
                    optimized_params = {}
                if self.config.get('feature_selection', {}).get('enabled', True):
                    selected_features = await self.feature_selector.select_features(model, X_train, y_train, X_val, y_val, regime_id)
                    X_train_selected = X_train[selected_features]
                    X_val_selected = X_val[selected_features]
                else:
                    selected_features = X_train.columns.tolist()
                    X_train_selected = X_train
                    X_val_selected = X_val
                if self.config.get('model_optimization', {}).get('enabled', True):
                    optimized_model = await self.model_optimizer.optimize(model, X_train_selected, y_train, optimized_params, regime_id)
                else:
                    optimized_model = model
                val_predictions = optimized_model.predict(X_val_selected)
                val_accuracy = accuracy_score(y_val, val_predictions)
                enhanced_models[model_name] = {'model': optimized_model, 'params': optimized_params, 'features': selected_features, 'validation_accuracy': val_accuracy}
                enhancement_results[model_name] = {'original_features': X_train.shape[1], 'selected_features': len(selected_features), 'validation_accuracy': val_accuracy, 'optimization_params': optimized_params}
                self.logger.info(f'    ✅ Enhanced {model_name}: Features {X_train.shape[1]}→{len(selected_features)}, Accuracy: {val_accuracy:.4f}')
            except Exception as e:
                self.logger.error(f'    ❌ Failed to enhance {model_name}: {str(e)}')
                enhanced_models[model_name] = models[model_name]
                enhancement_results[model_name] = {'error': str(e)}
        return {'models': enhanced_models, 'results': enhancement_results}

    def _get_regime_features(self, features: pd.DataFrame, regime_info: Dict[str, Any]) -> pd.DataFrame:
        """Extract features for a specific regime."""
        regime_mask = regime_info.get('mask', [])
        if isinstance(regime_mask, list):
            regime_mask = np.array(regime_mask)
        return features[regime_mask]

    def _get_regime_models(self, analyst_models: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
        """Get models for a specific regime."""
        return analyst_models.get(regime_id, {})

    def _split_regime_data(self, features: pd.DataFrame, regime_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split regime data into training and validation sets."""
        target_col = regime_info.get('target_column', 'target')
        if target_col not in features.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")
        X = features.drop(columns=[target_col])
        y = features[target_col]
        split_idx = int(len(X) * (1 - self.regime_config['regime_validation_split']))
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        return (X_train, y_train, X_val, y_val)

    def _calculate_enhancement_metrics(self, enhancement_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall enhancement metrics."""
        metrics = {'total_regimes': len(enhancement_results), 'successful_enhancements': 0, 'average_feature_reduction': [], 'average_accuracy_improvement': [], 'regime_metrics': {}}
        for regime_id, regime_results in enhancement_results.items():
            regime_metrics = {'models_enhanced': len(regime_results), 'successful': 0, 'failed': 0, 'average_accuracy': []}
            for model_name, results in regime_results.items():
                if 'error' not in results:
                    regime_metrics['successful'] += 1
                    if 'validation_accuracy' in results:
                        regime_metrics['average_accuracy'].append(results['validation_accuracy'])
                    if 'original_features' in results and 'selected_features' in results:
                        reduction = 1 - results['selected_features'] / results['original_features']
                        metrics['average_feature_reduction'].append(reduction)
                else:
                    regime_metrics['failed'] += 1
            if regime_metrics['average_accuracy']:
                regime_metrics['average_accuracy'] = np.mean(regime_metrics['average_accuracy'])
            else:
                regime_metrics['average_accuracy'] = 0.0
            metrics['regime_metrics'][regime_id] = regime_metrics
            if regime_metrics['successful'] > 0:
                metrics['successful_enhancements'] += 1
        if metrics['average_feature_reduction']:
            metrics['average_feature_reduction'] = np.mean(metrics['average_feature_reduction'])
        else:
            metrics['average_feature_reduction'] = 0.0
        return metrics

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        required_outputs = ['enhanced_analyst_models', 'analyst_enhancement_results', 'analyst_ensemble_models', 'analyst_enhancement_metrics']
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f'Missing required output: {output}')
        if 'enhanced_analyst_models' in pipeline_state:
            enhanced_models = pipeline_state['enhanced_analyst_models']
            if not isinstance(enhanced_models, dict):
                errors.append('Enhanced models must be a dictionary')
            elif len(enhanced_models) == 0:
                errors.append('No enhanced models were created')
        if 'analyst_enhancement_metrics' in pipeline_state:
            metrics = pipeline_state['analyst_enhancement_metrics']
            if metrics.get('successful_enhancements', 0) == 0:
                errors.append('No successful model enhancements')
        return (len(errors) == 0, errors)

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['analyst_models', 'regime_data', 'features', 'step11_analyst_creation_completed']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return ['enhanced_analyst_models', 'analyst_enhancement_results', 'analyst_ensemble_models', 'analyst_enhancement_metrics']

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['step11_analyst_creation']