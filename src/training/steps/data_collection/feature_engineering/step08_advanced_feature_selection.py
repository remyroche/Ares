from typing import List
from typing import Dict
from typing import Any
import pandas as pd
import numpy as np
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

'Step 8: Advanced Feature Selection - Refactored to use BaseStep.\n\nThis step performs sophisticated feature selection using:\n- Phase 1: mRMR and Random Forest to select top 150 features\n- Phase 2: Boruta to generate multiple feature sets (100, 80, 60)\nwith regime-aware selection, time-series validation, and interpretability analysis.\n'
import asyncio
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn('Numba not available - computations will be slower')
try:
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn('Joblib not available - parallel processing disabled')
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn('SHAP not available - interpretability features will be limited')
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    warnings.warn('Boruta not available - will use alternative feature selection')
try:
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn('LIME not available - interpretability features will be limited')
from .training.base_step import BaseStep
from src.utils.logger import system_logger
import json
import logging
import time

if NUMBA_AVAILABLE:

    @jit(nopython = True, parallel = True)
    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix using Numba for speed."""
        n_features = X.shape[1]
        corr_matrix = np.zeros((n_features, n_features))
        X_std = np.zeros_like(X)
        for i in prange(n_features):
            mean = np.mean(X[:, i])
            std = np.std(X[:, i])
            if std > 0:
                X_std[:, i] = (X[:, i] - mean) / std
            else:
                X_std[:, i] = 0
        n_samples = X.shape[0]
        for i in prange(n_features):
            for j in range(i, n_features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = np.sum(X_std[:, i] * X_std[:, j]) / (n_samples - 1)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        return corr_matrix

    @jit(nopython = True)
    def fast_mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Fast mutual information calculation using histogram method."""
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        y_min, y_max = (np.min(y), np.max(y))
        y_bins = np.linspace(y_min, y_max, n_bins + 1)
        y_discrete = np.searchsorted(y_bins[1:-1], y)
        for i in range(n_features):
            x_min, x_max = (np.min(X[:, i]), np.max(X[:, i]))
            if x_max > x_min:
                x_bins = np.linspace(x_min, x_max, n_bins + 1)
                x_discrete = np.searchsorted(x_bins[1:-1], X[:, i])
                hist_2d = np.zeros((n_bins, n_bins))
                for j in range(len(x_discrete)):
                    hist_2d[x_discrete[j], y_discrete[j]] += 1
                hist_2d = hist_2d / len(x_discrete)
                px = np.sum(hist_2d, axis = 1)
                py = np.sum(hist_2d, axis = 0)
                mi = 0.0
                for xi in range(n_bins):
                    for yi in range(n_bins):
                        if hist_2d[xi, yi] > 0 and px[xi] > 0 and (py[yi] > 0):
                            mi += hist_2d[xi, yi] * np.log(hist_2d[xi, yi] / (px[xi] * py[yi]))
                mi_scores[i] = mi
        return mi_scores
else:

    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        """Standard correlation matrix computation."""
        return np.corrcoef(X.T)

    def fast_mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Standard mutual information using sklearn."""
        if len(np.unique(y)) <= 10:
            return mutual_info_classif(X, y)
        else:
            return mutual_info_regression(X, y)

class AdvancedFeatureSelectionStep(BaseStep):
    """Step 8: Advanced Feature Selection using standardized base class."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize advanced feature selection step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '08', 'advanced_feature_selection')
        self.selection_config = config.get('feature_selection_config', {'phase1': {'mrmr_top_k': 150, 'rf_importance_threshold': 0.001, 'use_parallel': True, 'n_jobs': -1}, 'phase2': {'boruta_max_iter': 100, 'boruta_alpha': 0.05, 'feature_sets': [100, 80, 60], 'use_shap': True}, 'regime_aware': {'enabled': True, 'min_regime_samples': 100, 'regime_importance_weight': 1.2}, 'interpretability': {'calculate_shap': True, 'calculate_lime': True, 'top_features_to_explain': 20}})
        self.feature_selector = None
        self.feature_importance_analyzer = None
        self.selected_features = {}
        self.feature_statistics = {}
    @log_step_functions

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.logger.info(f'Initializing {self.step_name}...')
        self._initialize_selectors()
        self._initialize_interpretability_tools()
        self.feature_dir = self.base_dir / 'selected_features'
        self.feature_dir.mkdir(parents = True, exist_ok = True)
        self.importance_dir = self.base_dir / 'feature_importance'
        self.importance_dir.mkdir(parents = True, exist_ok = True)
        self.interpretability_dir = self.base_dir / 'interpretability'
        self.interpretability_dir.mkdir(parents = True, exist_ok = True)
    @log_all_calls

    def _initialize_selectors(self) -> None:
        """Initialize feature selection algorithms."""
        self.rf_classifier = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42, n_jobs = self.selection_config['phase1']['n_jobs'])
        self.rf_regressor = RandomForestRegressor(n_estimators = 100, max_depth = 10, random_state = 42, n_jobs = self.selection_config['phase1']['n_jobs'])
        self.lgb_params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'num_threads': self.selection_config['phase1']['n_jobs']}
    @log_all_calls

    def _initialize_interpretability_tools(self) -> None:
        """Initialize SHAP and LIME if available."""
        self.shap_explainer = None
        self.lime_explainer = None
        if SHAP_AVAILABLE and self.selection_config['interpretability']['calculate_shap']:
            self.logger.info('SHAP initialized for interpretability analysis')
        if LIME_AVAILABLE and self.selection_config['interpretability']['calculate_lime']:
            self.logger.info('LIME initialized for interpretability analysis')
    @log_step_functions

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        if 'preprocessed_data' not in pipeline_state:
            errors.append('Missing preprocessed_data in pipeline state')
        if 'feature_engineered_data' not in pipeline_state:
            errors.append('Missing feature_engineered_data in pipeline state')
        if 'regime_labels' not in pipeline_state:
            errors.append('Missing regime_labels in pipeline state')
        if not errors and 'feature_engineered_data' in pipeline_state:
            data = pipeline_state['feature_engineered_data']
            if not isinstance(data, dict):
                errors.append('feature_engineered_data must be a dictionary')
            elif 'train' not in data:
                errors.append("Missing 'train' split in feature_engineered_data")
        if not errors:
            train_data = pipeline_state['feature_engineered_data']['train']
            if not isinstance(train_data, pd.DataFrame):
                errors.append('Train data must be a pandas DataFrame')
            elif len(train_data.columns) < 10:
                errors.append(f'Insufficient features for selection: {len(train_data.columns)}')
        return (len(errors) == 0, errors)

    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature selection logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        self.logger.info('🎯 Starting advanced feature selection...')
        feature_data = pipeline_state['feature_engineered_data']
        regime_labels = pipeline_state.get('regime_labels', {})
        self.logger.info('📊 Phase 1: mRMR and Random Forest selection...')
        phase1_features = await self._execute_phase1_selection(feature_data, regime_labels, pipeline_state)
        self.logger.info('📊 Phase 2: Boruta feature selection...')
        phase2_features = await self._execute_phase2_selection(feature_data, phase1_features, regime_labels, pipeline_state)
        if self.selection_config['interpretability']['calculate_shap']:
            self.logger.info('🔍 Generating interpretability analysis...')
            await self._generate_interpretability_analysis(feature_data, phase2_features, pipeline_state)
        selected_data = {}
        for split_name, data in feature_data.items():
            selected_data[split_name] = {}
            for feature_set_name, features in phase2_features.items():
                available_features = [f for f in features if f in data.columns]
                selected_data[split_name][feature_set_name] = data[available_features].copy()
                self.logger.info(f'✅ {split_name} - {feature_set_name}: {len(available_features)} features selected')
        pipeline_state['selected_features'] = phase2_features
        pipeline_state['selected_feature_data'] = selected_data
        pipeline_state['feature_importance'] = self.feature_importance_
        pipeline_state['feature_statistics'] = self.feature_statistics
        await self._save_feature_selection_results(phase2_features, self.feature_importance_, self.feature_statistics)
        self.logger.info('✅ Feature selection completed successfully!')
        return pipeline_state

    async def _execute_phase1_selection(self, feature_data: Dict[str, pd.DataFrame], regime_labels: Dict[str, np.ndarray], pipeline_state: Dict[str, Any]) -> List[str]:
        """Execute Phase 1 feature selection using mRMR and Random Forest.
        
        Args:
            feature_data: Dictionary of feature data by split
            regime_labels: Regime labels for each split
            pipeline_state: Current pipeline state
            
        Returns:
            List of selected feature names
        """
        train_data = feature_data['train']
        train_labels = regime_labels.get('train')
        target_col = pipeline_state.get('target_column', 'target')
        if target_col in train_data.columns:
            y = train_data[target_col].values
            X = train_data.drop(columns=[target_col])
        else:
            y = train_labels if train_labels is not None else np.zeros(len(train_data))
            X = train_data
        feature_names = X.columns.tolist()
        X_values = X.values
        self.logger.info('Calculating mutual information scores...')
        mi_scores = fast_mutual_information(X_values, y)
        self.logger.info('Calculating Random Forest importance...')
        if len(np.unique(y)) <= 10:
            self.rf_classifier.fit(X_values, y)
            rf_importance = self.rf_classifier.feature_importances_
        else:
            self.rf_regressor.fit(X_values, y)
            rf_importance = self.rf_regressor.feature_importances_
        self.logger.info('Calculating feature correlations...')
        corr_matrix = fast_correlation_matrix(X_values)
        selected_indices = self._mrmr_selection(mi_scores, corr_matrix, self.selection_config['phase1']['mrmr_top_k'])
        rf_threshold = self.selection_config['phase1']['rf_importance_threshold']
        rf_selected = np.where(rf_importance > rf_threshold)[0]
        all_selected = np.union1d(selected_indices, rf_selected)
        self.feature_importance_ = {'mutual_information': dict(zip(feature_names, mi_scores)), 'random_forest': dict(zip(feature_names, rf_importance)), 'mrmr_selected': [feature_names[i] for i in selected_indices], 'rf_selected': [feature_names[i] for i in rf_selected]}
        selected_features = [feature_names[i] for i in all_selected]
        self.logger.info(f'Phase 1 selected {len(selected_features)} features')
        return selected_features
    @log_all_calls

    def _mrmr_selection(self, mi_scores: np.ndarray, corr_matrix: np.ndarray, k: int) -> np.ndarray:
        """Perform mRMR (minimum Redundancy Maximum Relevance) selection.
        
        Args:
            mi_scores: Mutual information scores
            corr_matrix: Feature correlation matrix
            k: Number of features to select
            
        Returns:
            Indices of selected features
        """
        n_features = len(mi_scores)
        selected = []
        remaining = list(range(n_features))
        first_idx = np.argmax(mi_scores)
        selected.append(first_idx)
        remaining.remove(first_idx)
        while len(selected) < k and remaining:
            max_score = -np.inf
            best_idx = None
            for idx in remaining:
                relevance = mi_scores[idx]
                redundancy = np.mean([abs(corr_matrix[idx, s]) for s in selected])
                score = relevance - redundancy
                if score > max_score:
                    max_score = score
                    best_idx = idx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        return np.array(selected)

    async def _execute_phase2_selection(self, feature_data: Dict[str, pd.DataFrame], phase1_features: List[str], regime_labels: Dict[str, np.ndarray], pipeline_state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Execute Phase 2 feature selection using Boruta.
        
        Args:
            feature_data: Dictionary of feature data by split
            phase1_features: Features selected in Phase 1
            regime_labels: Regime labels for each split
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary of feature sets
        """
        train_data = feature_data['train']
        available_features = [f for f in phase1_features if f in train_data.columns]
        X = train_data[available_features]
        target_col = pipeline_state.get('target_column', 'target')
        if target_col in train_data.columns:
            y = train_data[target_col].values
        else:
            y = regime_labels.get('train', np.zeros(len(train_data)))
        feature_sets = {}
        if BORUTA_AVAILABLE:
            self.logger.info('Running Boruta feature selection...')
            if len(np.unique(y)) <= 10:
                estimator = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 42, n_jobs = self.selection_config['phase1']['n_jobs'])
            else:
                estimator = RandomForestRegressor(n_estimators = 100, max_depth = 5, random_state = 42, n_jobs = self.selection_config['phase1']['n_jobs'])
            boruta = BorutaPy(estimator, n_estimators='auto', max_iter = self.selection_config['phase2']['boruta_max_iter'], alpha = self.selection_config['phase2']['boruta_alpha'], random_state = 42)
            boruta.fit(X.values, y)
            boruta_selected = X.columns[boruta.support_].tolist()
            boruta_tentative = X.columns[boruta.support_weak_].tolist()
            feature_ranking = boruta.ranking_
            ranked_features = sorted(zip(available_features, feature_ranking), key = lambda x: x[1])
            self.logger.info(f'Boruta selected {len(boruta_selected)} confirmed features')
            self.logger.info(f'Boruta found {len(boruta_tentative)} tentative features')
        else:
            self.logger.info('Using LightGBM for feature ranking (Boruta not available)...')
            if len(np.unique(y)) <= 10:
                lgb_train = lgb.Dataset(X.values, label = y)
                lgb_params = self.lgb_params.copy()
                model = lgb.train(lgb_params, lgb_train, num_boost_round = 100, verbose_eval = False)
            else:
                lgb_params = self.lgb_params.copy()
                lgb_params['objective'] = 'regression'
                lgb_params['metric'] = 'rmse'
                lgb_train = lgb.Dataset(X.values, label = y)
                model = lgb.train(lgb_params, lgb_train, num_boost_round = 100, verbose_eval = False)
            importance = model.feature_importance(importance_type='gain')
            ranked_features = sorted(zip(available_features, importance), key = lambda x: x[1], reverse = True)
            boruta_selected = [f for f, _ in ranked_features[:100]]
        for n_features in self.selection_config['phase2']['feature_sets']:
            if n_features <= len(ranked_features):
                feature_sets[f'top_{n_features}'] = [f for f, _ in ranked_features[:n_features]]
            else:
                feature_sets[f'top_{n_features}'] = available_features
        if BORUTA_AVAILABLE and boruta_selected:
            feature_sets['boruta_confirmed'] = boruta_selected
            if boruta_tentative:
                feature_sets['boruta_all'] = boruta_selected + boruta_tentative
        self.feature_statistics = self._calculate_feature_statistics(X, y, feature_sets)
        return feature_sets
    @log_all_calls

    def _calculate_feature_statistics(self, X: pd.DataFrame, y: np.ndarray, feature_sets: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate statistics for selected features.
        
        Args:
            X: Feature data
            y: Target values
            feature_sets: Dictionary of feature sets
            
        Returns:
            Feature statistics
        """
        stats = {'n_samples': len(X), 'n_features_original': len(X.columns), 'feature_sets': {}}
        for set_name, features in feature_sets.items():
            set_stats = {'n_features': len(features), 'reduction_ratio': 1 - len(features) / len(X.columns), 'features': features}
            if len(features) > 1:
                feature_data = X[features]
                corr_matrix = feature_data.corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                set_stats['correlation_stats'] = {'mean': corr_matrix.values.mean(), 'max': corr_matrix.values.max(), 'percentile_75': np.percentile(corr_matrix.values, 75), 'percentile_90': np.percentile(corr_matrix.values, 90)}
            stats['feature_sets'][set_name] = set_stats
        return stats

    async def _generate_interpretability_analysis(self, feature_data: Dict[str, pd.DataFrame], selected_features: Dict[str, List[str]], pipeline_state: Dict[str, Any]) -> None:
        """Generate interpretability analysis for selected features.
        
        Args:
            feature_data: Feature data
            selected_features: Selected feature sets
            pipeline_state: Pipeline state
        """
        if not SHAP_AVAILABLE and (not LIME_AVAILABLE):
            self.logger.warning('No interpretability libraries available')
            return
        train_data = feature_data['train']
        feature_set_sizes = {name: len(features) for name, features in selected_features.items()}
        smallest_set_name = min(feature_set_sizes, key = feature_set_sizes.get)
        features_to_explain = selected_features[smallest_set_name]
        n_explain = min(len(features_to_explain), self.selection_config['interpretability']['top_features_to_explain'])
        features_to_explain = features_to_explain[:n_explain]
        X_explain = train_data[features_to_explain]
        target_col = pipeline_state.get('target_column', 'target')
        if target_col in train_data.columns:
            y = train_data[target_col].values
        else:
            y = np.zeros(len(train_data))
        interpretability_results = {'feature_set': smallest_set_name, 'n_features_explained': n_explain, 'features': features_to_explain}
        if SHAP_AVAILABLE and self.selection_config['interpretability']['calculate_shap']:
            self.logger.info('Generating SHAP values...')
            if len(np.unique(y)) <= 10:
                model = RandomForestClassifier(n_estimators = 50, max_depth = 5, random_state = 42)
            else:
                model = RandomForestRegressor(n_estimators = 50, max_depth = 5, random_state = 42)
            model.fit(X_explain.values, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain.values)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            shap_importance = np.abs(shap_values).mean(axis = 0)
            interpretability_results['shap'] = {'feature_importance': dict(zip(features_to_explain, shap_importance)), 'mean_abs_shap': shap_importance.tolist()}
            shap_file = self.interpretability_dir / 'shap_values.npz'
            np.savez_compressed(shap_file, shap_values = shap_values, features = features_to_explain, importance = shap_importance)
        interpret_file = self.interpretability_dir / 'interpretability_analysis.json'
        safe_json_dump(interpretability_results, interpret_file)
        self.logger.info('✅ Interpretability analysis completed')

    async def _save_feature_selection_results(self, selected_features: Dict[str, List[str]], feature_importance: Dict[str, Any], feature_statistics: Dict[str, Any]) -> None:
        """Save feature selection results.
        
        Args:
            selected_features: Selected feature sets
            feature_importance: Feature importance scores
            feature_statistics: Feature statistics
        """
        for set_name, features in selected_features.items():
            feature_file = self.feature_dir / f'{set_name}_features.json'
            safe_json_dump({'features': features, 'count': len(features)}, feature_file)
        importance_file = self.importance_dir / 'feature_importance.json'
        safe_json_dump(feature_importance, importance_file)
        stats_file = self.feature_dir / 'feature_statistics.json'
        safe_json_dump(feature_statistics, stats_file)
        summary = {'timestamp': datetime.now().isoformat(), 'phase1_features': len(feature_importance.get('mrmr_selected', [])), 'feature_sets': {name: len(features) for name, features in selected_features.items()}, 'statistics': feature_statistics}
        summary_file = self.base_dir / 'feature_selection_summary.json'
        safe_json_dump(summary, summary_file)
        self.logger.info(f'✅ Saved feature selection results to {self.base_dir}')

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        if 'selected_features' not in pipeline_state:
            errors.append('Missing selected_features in pipeline state')
        if 'selected_feature_data' not in pipeline_state:
            errors.append('Missing selected_feature_data in pipeline state')
        if 'feature_importance' not in pipeline_state:
            errors.append('Missing feature_importance in pipeline state')
        if 'selected_features' in pipeline_state:
            selected = pipeline_state['selected_features']
            if not isinstance(selected, dict):
                errors.append('selected_features must be a dictionary')
            elif len(selected) == 0:
                errors.append('No feature sets were selected')
            else:
                for set_name, features in selected.items():
                    if not isinstance(features, list):
                        errors.append(f'Feature set {set_name} must be a list')
                    elif len(features) == 0:
                        errors.append(f'Feature set {set_name} is empty')
        if 'selected_feature_data' in pipeline_state:
            data = pipeline_state['selected_feature_data']
            if not isinstance(data, dict):
                errors.append('selected_feature_data must be a dictionary')
            elif 'train' not in data:
                errors.append("Missing 'train' split in selected_feature_data")
            else:
                train_data = data['train']
                if not isinstance(train_data, dict):
                    errors.append('Train data must contain feature sets')
                elif len(train_data) == 0:
                    errors.append('No feature sets in train data')
        return (len(errors) == 0, errors)

    def get_required_inputs(self) -> list:
        """Get list of required inputs."""
        return ['feature_engineered_data', 'preprocessed_data', 'regime_labels']

    def get_produced_outputs(self) -> list:
        """Get list of produced outputs."""
        return ['selected_features', 'selected_feature_data', 'feature_importance', 'feature_statistics']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ['step06_feature_engineering', 'step03_hmm_regime_discovery']

async def run_step(symbol: str, exchange: str, timeframe: str='1m', data_dir: str = None, force_rerun: bool = False, **kwargs: Any) -> bool:
    """
    Run Step 8: Advanced Feature Selection (backward compatibility).
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force_rerun: Force rerun the step
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    config = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir or 'data', 'force_rerun': force_rerun, **kwargs}
    step = AdvancedFeatureSelectionStep(config)
    pipeline_state = {'feature_engineered_data': {'train': pd.DataFrame(), 'validation': pd.DataFrame(), 'test': pd.DataFrame()}, 'preprocessed_data': {}, 'regime_labels': {}}
    try:
        result = await step.execute({}, pipeline_state)
        return True
    except Exception as e:
        system_logger.error(f'Failed to execute step 8: {str(e)}')
        return False
if __name__ == '__main__':
    asyncio.run(run_step(symbol='BTCUSDT', exchange='binance', timeframe='1m', force_rerun = True))