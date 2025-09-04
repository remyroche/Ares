"""Model selector component for tactician specialist training."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger

class ModelSelector:
    """Handles model selection for specialist tactics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('model_selection', {})
        self.logger = system_logger.getChild('model_selector')
        self.selection_metric = self.config.get('selection_metric', 'precision')
        self.min_performance_threshold = self.config.get('min_performance_threshold', 0.85)
        self.max_models_per_tactic = self.config.get('max_models_per_tactic', 3)
        self.diversity_weight = self.config.get('diversity_weight', 0.2)
        self.metric_weights = self.config.get('metric_weights', {'precision': 0.4, 'recall': 0.2, 'accuracy': 0.2, 'stability': 0.2})

    @handles_errors(exceptions=(Exception,), default_return={}, context='best model selection')
    async def select_best_models(self, models: Dict[str, Any], validation_results: Dict[str, Dict[str, float]], tactic_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select best models for a tactic based on performance.
        
        Args:
            models: Dictionary of trained models
            validation_results: Validation metrics for each model
            tactic_config: Tactic configuration
            
        Returns:
            Dictionary of selected models
        """
        if not models or not validation_results:
            return {}
        model_scores = await self._calculate_model_scores(models, validation_results, tactic_config)
        min_threshold = tactic_config.get('min_confidence', self.min_performance_threshold)
        qualified_models = {name: score for name, score in model_scores.items() if validation_results[name].get(self.selection_metric, 0) >= min_threshold}
        if not qualified_models:
            self.logger.warning(f'No models meet minimum threshold of {min_threshold}')
            best_model_name = max(model_scores.items(), key=lambda x: x[1])[0]
            return {best_model_name: models[best_model_name]}
        selected_models = await self._select_diverse_models(models, qualified_models, validation_results)
        return selected_models

    async def _calculate_model_scores(self, models: Dict[str, Any], validation_results: Dict[str, Dict[str, float]], tactic_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate composite scores for models.
        
        Args:
            models: Dictionary of models
            validation_results: Validation metrics
            tactic_config: Tactic configuration
            
        Returns:
            Dictionary of model scores
        """
        scores = {}
        for model_name in models:
            if model_name not in validation_results:
                continue
            metrics = validation_results[model_name]
            if 'error' in metrics:
                continue
            score = 0.0
            total_weight = 0.0
            for metric_name, weight in self.metric_weights.items():
                if metric_name in metrics:
                    score += metrics[metric_name] * weight
                    total_weight += weight
            if total_weight > 0:
                scores[model_name] = score / total_weight
            else:
                scores[model_name] = metrics.get(self.selection_metric, 0)
        return scores

    async def _select_diverse_models(self, models: Dict[str, Any], qualified_scores: Dict[str, float], validation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Select diverse set of models.
        
        Args:
            models: All models
            qualified_scores: Scores of qualified models
            validation_results: Validation metrics
            
        Returns:
            Selected diverse models
        """
        sorted_models = sorted(qualified_scores.items(), key=lambda x: x[1], reverse=True)
        selected = {}
        best_name = sorted_models[0][0]
        selected[best_name] = models[best_name]
        for model_name, score in sorted_models[1:]:
            if len(selected) >= self.max_models_per_tactic:
                break
            diversity_score = await self._calculate_diversity_score(model_name, selected, validation_results)
            combined_score = score * (1 - self.diversity_weight) + diversity_score * self.diversity_weight
            if combined_score > 0.5:
                selected[model_name] = models[model_name]
        self.logger.info(f'Selected {len(selected)} models from {len(models)} candidates')
        return selected

    async def _calculate_diversity_score(self, candidate_name: str, selected_models: Dict[str, Any], validation_results: Dict[str, Dict[str, float]]) -> float:
        """Calculate diversity score for a candidate model.
        
        Args:
            candidate_name: Name of candidate model
            selected_models: Already selected models
            validation_results: Validation metrics
            
        Returns:
            Diversity score (0-1)
        """
        if not selected_models:
            return 1.0
        candidate_type = candidate_name.split('_')[0] if '_' in candidate_name else candidate_name
        type_diversity = 1.0
        for selected_name in selected_models:
            selected_type = selected_name.split('_')[0] if '_' in selected_name else selected_name
            if candidate_type == selected_type:
                type_diversity *= 0.5
        candidate_metrics = validation_results.get(candidate_name, {})
        perf_diversity = 0.0
        for selected_name in selected_models:
            selected_metrics = validation_results.get(selected_name, {})
            metric_diffs = []
            for metric in ['precision', 'recall', 'accuracy']:
                if metric in candidate_metrics and metric in selected_metrics:
                    diff = abs(candidate_metrics[metric] - selected_metrics[metric])
                    metric_diffs.append(diff)
            if metric_diffs:
                perf_diversity += np.mean(metric_diffs)
        if selected_models:
            perf_diversity /= len(selected_models)
        diversity_score = (type_diversity + perf_diversity) / 2
        return min(1.0, diversity_score)

    @handles_errors(exceptions=(Exception,), default_return={}, context='model ranking')
    async def rank_models_by_tactic(self, all_models: Dict[str, Dict[str, Any]], evaluation_results: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
        """Rank models for each tactic.
        
        Args:
            all_models: All specialist models by regime
            evaluation_results: Evaluation results
            
        Returns:
            Ranked models for each tactic
        """
        tactic_rankings = {}
        all_tactics = set()
        for regime_models in all_models.values():
            all_tactics.update(regime_models.keys())
        for tactic in all_tactics:
            tactic_models = []
            for regime_id, regime_models in all_models.items():
                if tactic in regime_models:
                    models = regime_models[tactic]
                    if regime_id in evaluation_results and tactic in evaluation_results[regime_id]:
                        metrics = evaluation_results[regime_id][tactic]
                        if isinstance(models, dict):
                            for model_name, model in models.items():
                                score = metrics.get(self.selection_metric, 0)
                                tactic_models.append((f'{regime_id}_{model_name}', score))
                        else:
                            score = metrics.get(self.selection_metric, 0)
                            tactic_models.append((f'{regime_id}_{tactic}', score))
            tactic_models.sort(key=lambda x: x[1], reverse=True)
            tactic_rankings[tactic] = tactic_models
        return tactic_rankings

    @handles_errors(exceptions=(Exception,), default_return={}, context='optimal model combination')
    async def select_optimal_combination(self, models: Dict[str, Any], performance_data: Dict[str, Any], constraints: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """Select optimal combination of models.
        
        Args:
            models: Available models
            performance_data: Performance metrics
            constraints: Optional constraints
            
        Returns:
            Optimal model combination
        """
        constraints = constraints or {}
        max_total_models = constraints.get('max_total_models', 10)
        model_values = []
        for tactic_name, tactic_models in models.items():
            tactic_performance = performance_data.get(tactic_name, {})
            if isinstance(tactic_models, dict):
                for model_name, model in tactic_models.items():
                    precision = tactic_performance.get('precision', 0.5)
                    recall = tactic_performance.get('recall', 0.1)
                    value = precision * recall
                    model_values.append({'tactic': tactic_name, 'model': model_name, 'value': value, 'precision': precision, 'recall': recall})
        model_values.sort(key=lambda x: x['value'], reverse=True)
        selected_combination = {}
        total_models = 0
        for model_info in model_values:
            if total_models >= max_total_models:
                break
            tactic = model_info['tactic']
            if tactic not in selected_combination:
                selected_combination[tactic] = []
            selected_combination[tactic].append({'model': model_info['model'], 'value': model_info['value'], 'metrics': {'precision': model_info['precision'], 'recall': model_info['recall']}})
            total_models += 1
        return selected_combination