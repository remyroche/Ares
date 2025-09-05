from typing import Any
import pandas as pd
from typing import Optional
from typing import Tuple
from typing import List, Dict
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
'Step 8: Advanced Feature Selection with Two-Phase Approach.\n\nThis step performs sophisticated feature selection using:\n- Phase 1: mRMR and Random Forest to select top 150 features\n- Phase 2: Boruta to generate multiple feature sets (100, 80, 60)\nwith regime-aware selection, time-series validation, and interpretability analysis.\n'
import asyncio
import json
import os
import warnings
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn('Numba not available - computations will be slower')
try:
    from joblib import Parallel, delayed
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
    import lime
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn('LIME not available - interpretability features will be limited')
try:
    import lightgbm as lgb
import logging
import time

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn('LightGBM not available - using RandomForest fallback for importance')
try:
    from src.core.decorators import handles_errors
except Exception:
    from src.utils.decorators import handles_errors
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True)
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

    @jit(nopython=True)
    def fast_mutual_info_discrete(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast mutual information calculation for discrete targets."""
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        for i in range(n_features):
            x_bins = np.percentile(X[:, i], np.linspace(0, 100, 11))
            x_discrete = np.searchsorted(x_bins[1:-1], X[:, i])
            mi_scores[i] = _calculate_mi_discrete(x_discrete, y)
        return mi_scores

    @jit(nopython=True)
    def _calculate_mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate MI between two discrete variables."""
        xy_counts = np.zeros((10, 2))
        for i in range(len(x)):
            if y[i] < 2:
                xy_counts[min(x[i], 9), int(y[i])] += 1
        n = len(x)
        mi = 0.0
        for i in range(10):
            for j in range(2):
                pxy = xy_counts[i, j] / n
                if pxy > 0:
                    px = np.sum(xy_counts[i, :]) / n
                    py = np.sum(xy_counts[:, j]) / n
                    if px > 0 and py > 0:
                        mi += pxy * np.log(pxy / (px * py))
        return mi
else:

    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        return np.corrcoef(X.T)

    def fast_mutual_info_discrete(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return mutual_info_classif(X, y, random_state=42)

class Step08AdvancedFeatureSelection:
    """Advanced two-phase feature selection with regime awareness and interpretability."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Step 8 Advanced Feature Selection."""
        self.config = config
        self.logger = system_logger.getChild('Step08AdvancedFeatureSelection')
        self.standards = pipeline_standards
        self.step_config = config.get('step08_advanced_feature_selection', {})
        self.output_dir = ensure_directory(self.step_config.get('output_dir', 'data/selected_features'))
        self.phase1_target_features = self.step_config.get('phase1_target_features', 150)
        self.enable_mrmr = self.step_config.get('enable_mrmr', True)
        self.enable_rf_importance = self.step_config.get('enable_rf_importance', True)
        self.phase2_targets = self.step_config.get('phase2_targets', [100, 80, 60])
        self.boruta_max_iter = self.step_config.get('boruta_max_iter', 100)
        self.boruta_alpha = self.step_config.get('boruta_alpha', 0.05)
        self.enable_redundancy_analysis = self.step_config.get('enable_redundancy_analysis', True)
        self.min_redundancy_correlation = self.step_config.get('min_redundancy_correlation', 0.7)
        self.redundancy_groups_per_concept = self.step_config.get('redundancy_groups_per_concept', 2)
        self.feature_concept_patterns = self.step_config.get('feature_concept_patterns', {'momentum': ['rsi', 'macd', 'momentum', 'roc'], 'volatility': ['bb_', 'atr', 'volatility', 'std'], 'volume': ['volume', 'vwap', 'obv', 'mfi'], 'trend': ['ema', 'sma', 'trend', 'adx'], 'microstructure': ['spread', 'imbalance', 'flow', 'tick'], 'regime': ['regime', 'cluster', 'state'], 'support_resistance': ['sr_', 'support', 'resistance', 'level']})
        self.n_splits_ts = self.step_config.get('n_splits_ts', 5)
        self.min_regime_samples = self.step_config.get('min_regime_samples', 100)
        self.enable_shap = self.step_config.get('enable_shap', True) and SHAP_AVAILABLE
        self.enable_lime = self.step_config.get('enable_lime', True) and LIME_AVAILABLE
        self.n_lime_samples = self.step_config.get('n_lime_samples', 10)
        self.n_jobs = self.step_config.get('n_jobs', -1)
        self.use_parallel = JOBLIB_AVAILABLE and self.n_jobs != 1
        self.logger.info('🚀 Step 8 Advanced Feature Selection initialized')
        self.logger.info(f'   Phase 1 target: {self.phase1_target_features} features')
        self.logger.info(f'   Phase 2 targets: {self.phase2_targets}')
        self.logger.info(f'   Computational optimizations:')
        self.logger.info(f'     - Numba: {NUMBA_AVAILABLE}')
        self.logger.info(f'     - Joblib: {JOBLIB_AVAILABLE}')
        self.logger.info(f'     - Parallel jobs: {self.n_jobs}')
        self.logger.info(f'   Feature selection methods:')
        self.logger.info(f'     - Boruta: {BORUTA_AVAILABLE}')
        self.logger.info(f'     - SHAP: {SHAP_AVAILABLE}')
        self.logger.info(f'     - LIME: {LIME_AVAILABLE}')

    @handles_errors(exceptions=(ValueError, RuntimeError), default_return=False)
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute Step 8: Advanced Feature Selection.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with selected features
        """
        try:
            start_time = datetime.now()
            self.logger.info('🚀 Starting Step 8: Advanced Feature Selection...')
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            filtered_train_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet'
            filtered_val_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet'
            if not os.path.exists(filtered_train_path):
                self.logger.warning('⚠️ Filtered features not found, using original features')
                filtered_train_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet'
                filtered_val_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet'
            self.logger.info(f'📊 Loading features from: {filtered_train_path}')
            df_train = pd.read_parquet(filtered_train_path)
            df_val = pd.read_parquet(filtered_val_path)
            df = pd.concat([df_train, df_val], ignore_index=True)
            self.logger.info(f'📈 Loaded {len(df)} rows with {len(df.columns)} columns')
            label_columns = ['target', 'direction', 'profit', 'outcome', 'returns', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df.columns if col not in label_columns]
            features_df = df[feature_columns]
            labels_df = df[[col for col in label_columns if col in df.columns]]
            if 'target' in labels_df.columns:
                y = labels_df['target']
            elif 'direction' in labels_df.columns:
                y = labels_df['direction']
            else:
                raise ValueError('No target or direction column found')
            if y.dtype != int:
                y = (y > 0).astype(int)
            regime_labels = None
            hmm_path = f'data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            if os.path.exists(hmm_path):
                self.logger.info(f'🎭 Loading regime labels from: {hmm_path}')
                hmm_data = pd.read_parquet(hmm_path)
                try:
                    from .utils.regime_data_access import get_regime_column
                    regime_col = get_regime_column(hmm_data)
                except Exception:
                    regime_col = 'composite_cluster_id' if 'composite_cluster_id' in hmm_data.columns else None
                if regime_col and regime_col in hmm_data.columns:
                    regime_labels = hmm_data[regime_col].iloc[:len(df)]
            self.logger.info('📊 Starting Phase 1: mRMR/RF Selection...')
            phase1_features, phase1_metadata = await self.phase1_mrmr_rf_selection(features_df, y, regime_labels)
            self.logger.info('🎯 Starting Phase 2: Boruta Multi-Target Selection...')
            phase2_results, interpretability_results = await self.phase2_boruta_multi_target(phase1_features, y, regime_labels)
            output_files = await self._save_selection_results(phase1_features, phase1_metadata, phase2_results, interpretability_results, symbol, exchange, timeframe, df_train, df_val, labels_df)
            pipeline_state['step08_advanced_feature_selection'] = {'status': 'completed', 'start_time': start_time.isoformat(), 'end_time': datetime.now().isoformat(), 'output_files': output_files, 'phase1_metadata': phase1_metadata, 'phase2_results': {k: v for k, v in phase2_results.items() if k != 'features'}, 'interpretability_results': interpretability_results, 'original_features': len(feature_columns), 'phase1_features': len(phase1_features.columns), 'phase2_feature_sets': {f'top_{k}': len(v['features']) for k, v in phase2_results.items()}, 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            self.logger.info('✅ Step 8: Advanced Feature Selection completed successfully')
            return pipeline_state
        except Exception as e:
            self.logger.error(f'❌ Step 8 failed: {str(e)}')
            pipeline_state['step08_advanced_feature_selection'] = {'status': 'failed', 'error': str(e), 'timestamp': datetime.now().isoformat()}
            return pipeline_state

    async def phase1_mrmr_rf_selection(self, X: pd.DataFrame, y: pd.Series, regime_labels: Optional[pd.Series]=None) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Phase 1: Select top 150 features using mRMR and Random Forest.
        
        Args:
            X: Feature dataframe
            y: Target series
            regime_labels: Optional regime labels
            
        Returns:
            Selected features and metadata
        """
        metadata = {}
        mrmr_features = []
        if self.enable_mrmr:
            self.logger.info('🔍 Running mRMR selection...')
            mrmr_features = self._mrmr_selection(X, y, self.phase1_target_features)
            metadata['mrmr_features'] = mrmr_features
            self.logger.info(f'   mRMR selected {len(mrmr_features)} features')
        rf_features = []
        if self.enable_rf_importance:
            self.logger.info('🌳 Running Random Forest selection with TS validation...')
            rf_features = self._time_series_rf_selection(X, y, self.phase1_target_features)
            metadata['rf_features'] = rf_features
            self.logger.info(f'   RF selected {len(rf_features)} features')
        regime_validated_features = []
        if regime_labels is not None:
            self.logger.info('🎭 Validating features per regime...')
            candidate_features = list(set(mrmr_features) | set(rf_features))
            regime_validated_features = self._validate_features_per_regime(X, y, regime_labels, candidate_features)
            metadata['regime_validated_features'] = regime_validated_features
        consensus_features = list(set(mrmr_features) & set(rf_features))
        metadata['consensus_features'] = consensus_features
        final_features = list(consensus_features)
        remaining_slots = self.phase1_target_features - len(final_features)
        for feature in regime_validated_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        for feature in mrmr_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        for feature in rf_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        if len(final_features) < self.phase1_target_features:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            for feature in mi_ranking.index:
                if feature not in final_features and len(final_features) < self.phase1_target_features:
                    final_features.append(feature)
        metadata['final_features_count'] = len(final_features)
        metadata['consensus_ratio'] = len(consensus_features) / len(final_features) if final_features else 0
        metadata['regime_specific_additions'] = len([f for f in final_features if f in regime_validated_features])
        self.logger.info(f'✅ Phase 1 complete: {len(X.columns)} → {len(final_features)} features')
        self.logger.info(f'   Consensus features: {len(consensus_features)}')
        self.logger.info(f"   Regime-specific additions: {metadata['regime_specific_additions']}")
        return (X[final_features], metadata)

    def _mrmr_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Optimized Minimum Redundancy Maximum Relevance feature selection.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        X_values = X.values
        y_values = y.values
        feature_names = X.columns.tolist()
        n_total_features = len(feature_names)
        if NUMBA_AVAILABLE and y.dtype == int:
            relevance_scores = fast_mutual_info_discrete(X_values, y_values)
        else:
            relevance_scores = mutual_info_classif(X, y, random_state=42)
        if NUMBA_AVAILABLE:
            corr_matrix = np.abs(fast_correlation_matrix(X_values))
        else:
            corr_matrix = np.abs(X.corr().values)
        selected_indices = []
        remaining_indices = list(range(n_total_features))
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        while len(selected_indices) < n_features and remaining_indices:
            redundancy_matrix = corr_matrix[np.ix_(remaining_indices, selected_indices)]
            redundancy_scores = np.mean(redundancy_matrix, axis=1)
            remaining_relevance = relevance_scores[remaining_indices]
            mrmr_scores = remaining_relevance - redundancy_scores
            best_idx_in_remaining = np.argmax(mrmr_scores)
            best_idx = remaining_indices[best_idx_in_remaining]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        return [feature_names[idx] for idx in selected_indices]

    def _time_series_rf_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Random Forest feature selection with time-series cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        tscv = TimeSeriesSplit(n_splits=min(self.n_splits_ts, 3))
        feature_importances = np.zeros(X.shape[1])
        for train_idx, val_idx in tscv.split(X):
            X_train, y_train = (X.iloc[train_idx], y.iloc[train_idx])
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            feature_importances += rf.feature_importances_
        feature_importances /= tscv.get_n_splits()
        top_indices = np.argsort(feature_importances)[-n_features:]
        return X.columns[top_indices].tolist()

    def _validate_features_per_regime(self, X: pd.DataFrame, y: pd.Series, regime_labels: pd.Series, candidate_features: List[str]) -> List[str]:
        """
        Optimized regime validation using parallel processing.
        
        Args:
            X: Feature dataframe
            y: Target series
            regime_labels: Regime labels
            candidate_features: Features to validate
            
        Returns:
            List of regime-validated features
        """
        unique_regimes = np.unique(regime_labels)
        valid_regimes = [r for r in unique_regimes if (regime_labels == r).sum() >= self.min_regime_samples]
        if not valid_regimes:
            return candidate_features
        X_values = X[candidate_features].values
        y_values = y.values
        if JOBLIB_AVAILABLE and len(valid_regimes) > 1:
            regime_scores_list = Parallel(n_jobs=-1)((delayed(self._evaluate_regime_features)(regime, X_values, y_values, regime_labels) for regime in valid_regimes))
        else:
            regime_scores_list = [self._evaluate_regime_features(regime, X_values, y_values, regime_labels) for regime in valid_regimes]
        regime_scores_matrix = np.array(regime_scores_list)
        mean_scores = np.mean(regime_scores_matrix, axis=0)
        min_scores = np.min(regime_scores_matrix, axis=0)
        validated_indices = np.where((mean_scores > 0.01) & (min_scores > 0.005))[0]
        return [candidate_features[idx] for idx in validated_indices]

    def _evaluate_regime_features(self, regime: Any, X_values: List[Any], y_values: List[Any], regime_labels: List[Any]) -> None:
        """Evaluate features for a single regime."""
        regime_mask = (regime_labels == regime).values
        X_regime = X_values[regime_mask]
        y_regime = y_values[regime_mask]
        if NUMBA_AVAILABLE and y_values.dtype == int:
            mi_scores = fast_mutual_info_discrete(X_regime, y_regime)
        else:
            mi_scores = mutual_info_classif(X_regime, y_regime, random_state=42)
        return mi_scores

    async def phase2_boruta_multi_target(self, X: pd.DataFrame, y: pd.Series, regime_labels: Optional[pd.Series]=None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Phase 2: Boruta selection with redundancy analysis for multiple target sizes.
        
        Args:
            X: Feature dataframe (already filtered to ~150 features)
            y: Target series
            regime_labels: Optional regime labels
            
        Returns:
            Feature sets and interpretability results
        """
        feature_sets = {}
        redundancy_groups = {}
        feature_clusters = {}
        if self.enable_redundancy_analysis:
            self.logger.info('🔄 Analyzing feature redundancy...')
            redundancy_groups = self._analyze_feature_redundancy(X)
            self.logger.info(f'   Found {len(redundancy_groups)} redundancy groups')
            self.logger.info('🔍 Performing hierarchical clustering for redundancy...')
            feature_clusters = self._hierarchical_feature_clustering(X)
            self.logger.info(f'   Identified {len(feature_clusters)} feature clusters')
        if BORUTA_AVAILABLE:
            self.logger.info('🔍 Running Boruta for all-relevant features...')
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            boruta_selector = BorutaPy(rf, n_estimators='auto', alpha=self.boruta_alpha, max_iter=self.boruta_max_iter, random_state=42)
            boruta_selector.fit(X.values, y.values)
            feature_ranks = boruta_selector.ranking_
            feature_importance = pd.Series(1 / feature_ranks, index=X.columns).sort_values(ascending=False)
            confirmed_features = X.columns[boruta_selector.support_].tolist()
            self.logger.info(f'   Boruta confirmed {len(confirmed_features)} features')
        else:
            if LGB_AVAILABLE:
                self.logger.warning('⚠️ Boruta not available, using LightGBM importance')
                lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, verbose=-1)
                lgb_model.fit(X, y)
                feature_importance = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            else:
                self.logger.warning('⚠️ Boruta and LightGBM not available, using RandomForest importance fallback')
                rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            threshold = feature_importance.quantile(0.2)
            confirmed_features = feature_importance[feature_importance > threshold].index.tolist()
        for target_size in self.phase2_targets:
            self.logger.info(f'📊 Creating redundancy-aware feature set with {target_size} features...')
            if self.enable_redundancy_analysis and (redundancy_groups or feature_clusters):
                all_redundancy_groups = dict(redundancy_groups)
                for cluster_id, cluster_features in feature_clusters.items():
                    all_redundancy_groups[f'cluster_{cluster_id}'] = cluster_features
                selected_features = self._select_features_with_redundancy_advanced(feature_importance, all_redundancy_groups, target_size, confirmed_features, boruta_selector if BORUTA_AVAILABLE else None)
            else:
                selected_features = feature_importance.head(target_size).index.tolist()
            ts_validation = self._time_series_validate_features(X[selected_features], y, n_splits=self.n_splits_ts)
            regime_validation = {}
            if regime_labels is not None:
                regime_validation = self._per_regime_validate_features(X[selected_features], y, regime_labels)
            redundancy_stats = self._calculate_redundancy_stats(selected_features, redundancy_groups) if redundancy_groups else {}
            feature_sets[target_size] = {'features': selected_features, 'importance_scores': feature_importance[selected_features].to_dict(), 'ts_validation': ts_validation, 'regime_validation': regime_validation, 'boruta_confirmed': len([f for f in selected_features if f in confirmed_features]), 'boruta_confirmed_ratio': len([f for f in selected_features if f in confirmed_features]) / len(selected_features), 'redundancy_stats': redundancy_stats}
            self.logger.info(f"   TS validation score: {ts_validation['mean_score']:.4f} ± {ts_validation['std_score']:.4f}")
            self.logger.info(f"   Boruta confirmed: {feature_sets[target_size]['boruta_confirmed']} features")
            if redundancy_stats:
                self.logger.info(f"   Redundancy groups: {redundancy_stats['groups_represented']}")
                self.logger.info(f"   Average redundancy: {redundancy_stats['average_redundancy']:.1f} features/group")
                self.logger.info(f"   Concept coverage: {sum(redundancy_stats['concept_coverage'].values())} features across {len([v for v in redundancy_stats['concept_coverage'].values() if v > 0])} concepts")
        self.logger.info('🔮 Generating interpretability analysis...')
        interpretability_results = await self._generate_interpretability_report(X, y, feature_sets)
        return (feature_sets, interpretability_results)

    def _time_series_validate_features(self, X: pd.DataFrame, y: pd.Series, n_splits: int=5) -> dict[str, Any]:
        """Time-series aware feature validation."""
        tscv = TimeSeriesSplit(n_splits=min(n_splits, 3))
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
            y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
            model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            scores.append(score)
        return {'mean_score': np.mean(scores), 'std_score': np.std(scores), 'scores': scores, 'n_splits': len(scores)}

    def _per_regime_validate_features(self, X: pd.DataFrame, y: pd.Series, regime_labels: pd.Series) -> dict[str, float]:
        """Validate features perform well in each regime."""
        regime_scores = {}
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            if regime_mask.sum() < self.min_regime_samples:
                continue
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            try:
                scores = cross_val_score(lgb.LGBMClassifier(n_estimators=50, max_depth=5, verbose=-1), X_regime, y_regime, cv=min(3, len(np.unique(y_regime))), scoring='roc_auc')
                regime_scores[f'regime_{regime}'] = scores.mean()
            except:
                continue
        return regime_scores

    async def _generate_interpretability_report(self, X: pd.DataFrame, y: pd.Series, feature_sets: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """Generate SHAP/LIME interpretability analysis."""
        report = {}
        for size, feature_data in feature_sets.items():
            self.logger.info(f'🔍 Analyzing interpretability for {size}-feature set...')
            features = feature_data['features']
            X_subset = X[features]
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_subset, y)
            feature_report = {}
            if self.enable_shap and SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    sample_size = min(1000, len(X_subset))
                    sample_idx = np.random.choice(len(X_subset), sample_size, replace=False)
                    X_sample = X_subset.iloc[sample_idx]
                    shap_values = explainer.shap_values(X_sample)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
                    feature_report['shap_importance'] = shap_importance.head(20).to_dict()
                    feature_report['feature_interactions'] = self._detect_feature_interactions(shap_values, features)
                except Exception as e:
                    self.logger.warning(f'⚠️ SHAP analysis failed: {e}')
                    feature_report['shap_error'] = str(e)
            if self.enable_lime and LIME_AVAILABLE:
                try:
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_subset.values, feature_names=features, class_names=['0', '1'], mode='classification')
                    sample_explanations = []
                    for i in range(min(self.n_lime_samples, len(X_subset))):
                        exp = lime_explainer.explain_instance(X_subset.iloc[i].values, model.predict_proba, num_features=min(10, len(features)))
                        sample_explanations.append(exp.as_list())
                    feature_report['lime_explanations'] = sample_explanations[:3]
                except Exception as e:
                    self.logger.warning(f'⚠️ LIME analysis failed: {e}')
                    feature_report['lime_error'] = str(e)
            y_pred = model.predict_proba(X_subset)[:, 1]
            feature_report['model_performance'] = {'roc_auc': roc_auc_score(y, y_pred), 'accuracy': accuracy_score(y, model.predict(X_subset)), 'f1_score': f1_score(y, model.predict(X_subset))}
            report[f'feature_set_{size}'] = feature_report
        return report

    def _detect_feature_interactions(self, shap_values: np.ndarray, feature_names: List[str], top_k: int=10) -> List[Tuple[str, str, float]]:
        """Detect top feature interactions from SHAP values."""
        interactions = []
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        corr_matrix = shap_df.corr().abs()
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                interactions.append((feature_names[i], feature_names[j], corr_matrix.iloc[i, j]))
        interactions.sort(key=lambda x: x[2], reverse=True)
        return [(f1, f2, round(score, 3)) for f1, f2, score in interactions[:top_k]]

    def _hierarchical_feature_clustering(self, X: pd.DataFrame, n_clusters: int=None) -> dict[int, List[str]]:
        """
        Perform hierarchical clustering on features to identify redundant groups.
        Uses correlation distance and Ward linkage.
        
        Args:
            X: Feature dataframe
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary mapping cluster IDs to feature lists
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        corr_matrix = X.corr().abs()
        distance_matrix = 1 - corr_matrix
        condensed_distances = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_distances, method='ward')
        if n_clusters is None:
            distances = Z[:, 2]
            gaps = np.diff(distances)
            optimal_idx = np.argmax(gaps) + 1
            distance_threshold = distances[optimal_idx]
            clusters = fcluster(Z, distance_threshold, criterion='distance')
        else:
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
        feature_clusters = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in feature_clusters:
                feature_clusters[cluster_id] = []
            feature_clusters[cluster_id].append(X.columns[idx])
        feature_clusters = {k: v for k, v in feature_clusters.items() if len(v) > 1}
        return feature_clusters

    def _analyze_feature_redundancy(self, X: pd.DataFrame) -> dict[str, List[str]]:
        """
        Optimized feature redundancy analysis using vectorized operations.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        redundancy_groups = {}
        n_features = len(X.columns)
        if NUMBA_AVAILABLE:
            corr_matrix = np.abs(fast_correlation_matrix(X.values))
        else:
            corr_matrix = X.corr().abs().values
        triu_indices = np.triu_indices(n_features, k=1)
        high_corr_mask = corr_matrix[triu_indices] >= self.min_redundancy_correlation
        high_corr_i = triu_indices[0][high_corr_mask]
        high_corr_j = triu_indices[1][high_corr_mask]
        if len(high_corr_i) > 0:
            adjacency = np.zeros((n_features, n_features), dtype=bool)
            adjacency[high_corr_i, high_corr_j] = True
            adjacency[high_corr_j, high_corr_i] = True
            visited = np.zeros(n_features, dtype=bool)
            corr_group_id = 0
            for start_idx in range(n_features):
                if not visited[start_idx] and np.any(adjacency[start_idx]):
                    component_mask = np.zeros(n_features, dtype=bool)
                    component_mask[start_idx] = True
                    prev_size = 0
                    while np.sum(component_mask) > prev_size:
                        prev_size = np.sum(component_mask)
                        component_mask |= np.any(adjacency[component_mask], axis=0)
                    visited |= component_mask
                    component_features = [X.columns[i] for i in np.where(component_mask)[0]]
                    if len(component_features) > 1:
                        redundancy_groups[f'corr_group_{corr_group_id}'] = component_features
                        corr_group_id += 1
        for concept, patterns in self.feature_concept_patterns.items():
            concept_features = []
            for feature in X.columns:
                feature_lower = feature.lower()
                if any((pattern in feature_lower for pattern in patterns)):
                    concept_features.append(feature)
            if len(concept_features) > 1:
                new_features = []
                for f in concept_features:
                    if not any((f in group for group in redundancy_groups.values())):
                        new_features.append(f)
                if len(new_features) > 1:
                    redundancy_groups[f'concept_{concept}'] = new_features
        return redundancy_groups

    def _select_features_with_redundancy(self, feature_importance: pd.Series, redundancy_groups: dict[str, List[str]], target_size: int, confirmed_features: List[str]) -> List[str]:
        """
        Select features considering redundancy to ensure robustness.
        
        Args:
            feature_importance: Feature importance scores
            redundancy_groups: Dictionary of redundancy groups
            target_size: Target number of features
            confirmed_features: Boruta-confirmed features
            
        Returns:
            List of selected features
        """
        selected_features = []
        used_groups = set()
        feature_to_groups = {}
        for group_name, features in redundancy_groups.items():
            for feature in features:
                if feature not in feature_to_groups:
                    feature_to_groups[feature] = []
                feature_to_groups[feature].append(group_name)
        for feature in feature_importance.index:
            if len(selected_features) >= target_size:
                break
            if feature in feature_to_groups:
                groups = feature_to_groups[feature]
                group_counts = {}
                for group in groups:
                    group_features = redundancy_groups[group]
                    count = sum((1 for f in selected_features if f in group_features))
                    group_counts[group] = count
                min_count = min(group_counts.values()) if group_counts else 0
                if min_count < self.redundancy_groups_per_concept:
                    selected_features.append(feature)
                    for group in groups:
                        used_groups.add(group)
            else:
                selected_features.append(feature)
        if len(selected_features) < target_size:
            for group_name, group_features in redundancy_groups.items():
                if len(selected_features) >= target_size:
                    break
                current_count = sum((1 for f in selected_features if f in group_features))
                if current_count < self.redundancy_groups_per_concept:
                    group_importance = feature_importance[feature_importance.index.isin(group_features)].sort_values(ascending=False)
                    for feature in group_importance.index:
                        if feature not in selected_features and len(selected_features) < target_size:
                            selected_features.append(feature)
                            current_count += 1
                            if current_count >= self.redundancy_groups_per_concept:
                                break
        while len(selected_features) < target_size:
            for feature in feature_importance.index:
                if feature not in selected_features:
                    selected_features.append(feature)
                    break
            else:
                break
        confirmed_selected = [f for f in selected_features if f in confirmed_features]
        unconfirmed_selected = [f for f in selected_features if f not in confirmed_features]
        final_features = confirmed_selected + unconfirmed_selected
        return final_features[:target_size]

    def _select_features_with_redundancy_advanced(self, feature_importance: pd.Series, all_redundancy_groups: dict[str, List[str]], target_size: int, confirmed_features: List[str], boruta_selector: Any=None) -> List[str]:
        """
        Advanced feature selection that combines Boruta's all-relevant features
        with redundancy reduction using multiple strategies.
        
        Args:
            feature_importance: Feature importance scores
            all_redundancy_groups: Combined redundancy groups from multiple methods
            target_size: Target number of features
            confirmed_features: Boruta-confirmed features
            boruta_selector: Fitted Boruta selector (optional)
            
        Returns:
            List of selected features with optimal redundancy
        """
        selected_features = []
        if confirmed_features:
            confirmed_by_group = {}
            ungrouped_confirmed = []
            for feature in confirmed_features:
                assigned = False
                for group_name, group_features in all_redundancy_groups.items():
                    if feature in group_features:
                        if group_name not in confirmed_by_group:
                            confirmed_by_group[group_name] = []
                        confirmed_by_group[group_name].append(feature)
                        assigned = True
                        break
                if not assigned:
                    ungrouped_confirmed.append(feature)
            for group_name, group_confirmed in confirmed_by_group.items():
                group_importance = feature_importance[group_confirmed].sort_values(ascending=False)
                n_to_take = min(self.redundancy_groups_per_concept, len(group_importance))
                selected_features.extend(group_importance.head(n_to_take).index.tolist())
            selected_features.extend(ungrouped_confirmed)
        remaining_slots = target_size - len(selected_features)
        if remaining_slots > 0:
            remaining_features = [f for f in feature_importance.index if f not in selected_features]
            vif_selected = self._select_low_vif_features(feature_importance[remaining_features], all_redundancy_groups, remaining_slots, selected_features)
            selected_features.extend(vif_selected)
        if len(selected_features) < target_size:
            concept_coverage = {}
            for concept, patterns in self.feature_concept_patterns.items():
                concept_features = [f for f in selected_features if any((p in f.lower() for p in patterns))]
                concept_coverage[concept] = len(concept_features)
            for concept, count in sorted(concept_coverage.items(), key=lambda x: x[1]):
                if len(selected_features) >= target_size:
                    break
                if count < 2:
                    patterns = self.feature_concept_patterns[concept]
                    concept_candidates = [f for f in feature_importance.index if any((p in f.lower() for p in patterns)) and f not in selected_features]
                    for feature in feature_importance[concept_candidates].sort_values(ascending=False).index:
                        if len(selected_features) < target_size:
                            selected_features.append(feature)
                            count += 1
                            if count >= 2:
                                break
        if boruta_selector is not None and hasattr(boruta_selector, 'ranking_'):
            boruta_ranks = dict(zip(feature_importance.index, boruta_selector.ranking_))
            redundant_pairs = []
            for i, f1 in enumerate(selected_features):
                for j, f2 in enumerate(selected_features[i + 1:], i + 1):
                    for group_features in all_redundancy_groups.values():
                        if f1 in group_features and f2 in group_features:
                            if boruta_ranks.get(f1, float('inf')) > boruta_ranks.get(f2, float('inf')):
                                redundant_pairs.append((i, f1))
                            else:
                                redundant_pairs.append((j, f2))
                            break
            removed_indices = set()
            for idx, feature in redundant_pairs:
                if idx not in removed_indices and len(selected_features) > target_size:
                    removed_indices.add(idx)
            for idx in sorted(removed_indices, reverse=True):
                selected_features.pop(idx)
        return selected_features[:target_size]

    def _select_low_vif_features(self, candidate_importance: pd.Series, redundancy_groups: dict[str, List[str]], n_features: int, already_selected: List[str]) -> List[str]:
        """
        Select features with low VIF (Variance Inflation Factor) to minimize multicollinearity.
        """
        selected = []
        for feature in candidate_importance.index:
            if len(selected) >= n_features:
                break
            redundancy_score = 0
            for group_name, group_features in redundancy_groups.items():
                if feature in group_features:
                    existing_count = sum((1 for f in already_selected + selected if f in group_features))
                    redundancy_score += existing_count
            if redundancy_score < self.redundancy_groups_per_concept:
                selected.append(feature)
        return selected

    def _calculate_redundancy_stats(self, selected_features: List[str], redundancy_groups: dict[str, List[str]]) -> dict[str, Any]:
        """
        Calculate redundancy statistics for selected features.
        
        Args:
            selected_features: List of selected features
            redundancy_groups: Dictionary of redundancy groups
            
        Returns:
            Dictionary of redundancy statistics
        """
        stats = {'groups_represented': 0, 'average_redundancy': 0, 'min_redundancy': float('inf'), 'max_redundancy': 0, 'concept_coverage': {}, 'group_feature_counts': {}}
        for group_name, group_features in redundancy_groups.items():
            count = sum((1 for f in selected_features if f in group_features))
            if count > 0:
                stats['groups_represented'] += 1
                stats['group_feature_counts'][group_name] = count
                stats['min_redundancy'] = min(stats['min_redundancy'], count)
                stats['max_redundancy'] = max(stats['max_redundancy'], count)
        if stats['group_feature_counts']:
            stats['average_redundancy'] = sum(stats['group_feature_counts'].values()) / len(stats['group_feature_counts'])
        else:
            stats['min_redundancy'] = 0
        for concept in self.feature_concept_patterns:
            concept_features = [f for f in selected_features if any((p in f.lower() for p in self.feature_concept_patterns[concept]))]
            stats['concept_coverage'][concept] = len(concept_features)
        return stats

    async def _save_selection_results(self, phase1_features: pd.DataFrame, phase1_metadata: dict[str, Any], phase2_results: dict[int, dict[str, Any]], interpretability_results: dict[str, Any], symbol: str, exchange: str, timeframe: str, df_train: pd.DataFrame, df_val: pd.DataFrame, labels_df: pd.DataFrame) -> dict[str, str]:
        """Save all selection results and create output datasets."""
        output_files = {}
        phase1_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_phase1_features.json')
        safe_json_dump({'features': phase1_features.columns.tolist(), 'metadata': phase1_metadata, 'timestamp': datetime.now().isoformat()}, phase1_path)
        output_files['phase1_results'] = phase1_path
        for target_size, results in phase2_results.items():
            phase2_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_top{target_size}_features.json')
            safe_json_dump({'features': results['features'], 'importance_scores': results['importance_scores'], 'validation': {'ts_validation': results['ts_validation'], 'regime_validation': results['regime_validation']}, 'boruta_stats': {'confirmed': results['boruta_confirmed'], 'confirmed_ratio': results['boruta_confirmed_ratio']}, 'timestamp': datetime.now().isoformat()}, phase2_path)
            output_files[f'top{target_size}_features'] = phase2_path
            selected_features = results['features']
            train_size = len(df_train)
            train_features = phase1_features[selected_features].iloc[:train_size]
            train_data = pd.concat([train_features, labels_df.iloc[:train_size]], axis=1)
            train_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_top{target_size}_train.parquet')
            train_data.to_parquet(train_path)
            output_files[f'top{target_size}_train'] = train_path
            val_features = phase1_features[selected_features].iloc[train_size:]
            val_data = pd.concat([val_features, labels_df.iloc[train_size:]], axis=1)
            val_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_top{target_size}_val.parquet')
            val_data.to_parquet(val_path)
            output_files[f'top{target_size}_val'] = val_path
        interp_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_interpretability_report.json')
        safe_json_dump(interpretability_results, interp_path)
        output_files['interpretability_report'] = interp_path
        report_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_selection_report.json')
        safe_json_dump({'phase1_summary': {'input_features': len(df_train.columns) - len(labels_df.columns), 'output_features': len(phase1_features.columns), 'consensus_features': len(phase1_metadata.get('consensus_features', [])), 'regime_validated': phase1_metadata.get('regime_specific_additions', 0)}, 'phase2_summary': {f'top_{size}': {'features': len(results['features']), 'ts_score': results['ts_validation']['mean_score'], 'boruta_confirmed': results['boruta_confirmed']} for size, results in phase2_results.items()}, 'timestamp': datetime.now().isoformat()}, report_path)
        output_files['selection_report'] = report_path
        self.logger.info(f'💾 Saved all selection results to {self.output_dir}')
        return output_files

async def run_step(symbol: str, exchange: str, timeframe: str='1m', data_dir: str=None, force_rerun: bool=False, **kwargs: Any) -> bool:
    """
    Run Step 8: Advanced Feature Selection.
    
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
    try:
        config_path = 'config/training_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        config.update(kwargs)
        training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir or f'data/{exchange}/{symbol}'}
        pipeline_state = {}
        step = Step08AdvancedFeatureSelection(config)
        result = await step.execute(training_input, pipeline_state)
        if result.get('step08_advanced_feature_selection', {}).get('status') == 'completed':
            system_logger.info('✅ Step 8: Advanced Feature Selection completed successfully')
            return True
        else:
            system_logger.error('❌ Step 8: Advanced Feature Selection failed')
            return False
    except Exception as e:
        system_logger.error(f'❌ Error running Step 8: {e}')
        return False
if __name__ == '__main__':
    asyncio.run(run_step(symbol='BTCUSDT', exchange='binance', timeframe='1m', force_rerun=True))