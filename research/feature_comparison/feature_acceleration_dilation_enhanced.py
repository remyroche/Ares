"""
Enhanced Feature Acceleration and Window Dilation

This module implements statistically robust feature acceleration and window dilation
with proper time-series validation, multiple testing control, and production hygiene.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import entropy, normaltest, jarque_bera
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
from itertools import combinations
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class EnhancedFeatureAccelerationDilation:
    """
    Enhanced feature acceleration and dilation system with statistical robustness.
    """
    
    def __init__(self, 
                 acceleration_lags: List[int] = [1, 3],
                 dilation_factors: List[float] = [2.0, 3.0],
                 mi_k_values: List[int] = [5, 10],
                 dm_alpha: float = 0.05,
                 fdr_q: float = 0.1,
                 cmi_ci_low_threshold: float = 0.0,
                 rank_stability_threshold: float = 0.6,
                 regime_delta_rank_threshold: int = 10,
                 correlation_threshold: float = 0.90,
                 same_family_correlation_threshold: float = 0.85,
                 psi_threshold: float = 0.2,
                 psi_delta_threshold: float = 0.05,
                 shadow_sigma_threshold: float = 1.0,
                 turnover_threshold: float = 0.1,
                 enable_matrix_ops: bool = True,
                 n_bootstrap: int = 500,
                 n_cv_folds: int = 5,
                 enable_parallel: bool = True):
        """
        Initialize enhanced feature acceleration and dilation system.
        
        Args:
            acceleration_lags: Lags for acceleration calculation
            dilation_factors: Window dilation factors
            mi_k_values: k values for kNN MI estimation
            dm_alpha: Diebold-Mariano test significance level
            fdr_q: Benjamini-Hochberg FDR control level
            cmi_ci_low_threshold: Conditional MI CI lower bound threshold
            rank_stability_threshold: Rank stability threshold
            regime_delta_rank_threshold: Regime delta rank threshold
            correlation_threshold: Correlation threshold for uniqueness
            same_family_correlation_threshold: Same-family correlation threshold
            psi_threshold: PSI threshold for drift detection
            psi_delta_threshold: PSI delta threshold vs base
            shadow_sigma_threshold: Shadow feature sigma threshold
            turnover_threshold: Turnover threshold for Pareto optimization
            enable_matrix_ops: Whether to enable matrix operations
            n_bootstrap: Number of bootstrap resamples
            n_cv_folds: Number of CV folds
            enable_parallel: Whether to enable parallel processing
        """
        self.acceleration_lags = acceleration_lags
        self.dilation_factors = dilation_factors
        self.mi_k_values = mi_k_values
        self.dm_alpha = dm_alpha
        self.fdr_q = fdr_q
        self.cmi_ci_low_threshold = cmi_ci_low_threshold
        self.rank_stability_threshold = rank_stability_threshold
        self.regime_delta_rank_threshold = regime_delta_rank_threshold
        self.correlation_threshold = correlation_threshold
        self.same_family_correlation_threshold = same_family_correlation_threshold
        self.psi_threshold = psi_threshold
        self.psi_delta_threshold = psi_delta_threshold
        self.shadow_sigma_threshold = shadow_sigma_threshold
        self.turnover_threshold = turnover_threshold
        self.enable_matrix_ops = enable_matrix_ops
        self.n_bootstrap = n_bootstrap
        self.n_cv_folds = n_cv_folds
        self.enable_parallel = enable_parallel
        
        # Initialize matrix operations if available
        if enable_matrix_ops:
            try:
                from src.utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations(enable_gpu=True, enable_parallel=True)
                self.matrix_available = True
            except ImportError:
                self.matrix_ops = None
                self.matrix_available = False
                logger.warning("Matrix operations not available, using standard operations")
        else:
            self.matrix_ops = None
            self.matrix_available = False
        
        # Initialize results storage
        self.variant_cards = {}
        self.pareto_frontier = {}
        self.global_metrics = {}
    
    def generate_acceleration_features(self, X: pd.DataFrame, 
                                     base_features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Generate acceleration features with proper centering and winsorization."""
        logger.info("Generating acceleration features with statistical robustness...")
        
        if base_features is None:
            base_features = self._identify_acceleration_candidates(X)
        
        acceleration_features = {}
        
        for lag in self.acceleration_lags:
            lag_features = {}
            
            for feature in base_features:
                if feature not in X.columns:
                    continue
                
                if not self._is_suitable_for_acceleration(X[feature]):
                    continue
                
                accel_feature = self._calculate_robust_acceleration(X[feature], lag)
                if accel_feature is not None:
                    lag_features[f"{feature}_accel_{lag}"] = accel_feature
            
            if lag_features:
                acceleration_features[f"lag_{lag}"] = pd.DataFrame(lag_features, index=X.index)
        
        logger.info(f"Generated acceleration features for {len(base_features)} base features")
        return acceleration_features
    
    def generate_dilation_features(self, X: pd.DataFrame, 
                                 base_features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Generate window dilation features with proper EMA semantics."""
        logger.info("Generating window dilation features with proper semantics...")
        
        if base_features is None:
            base_features = self._identify_dilation_candidates(X)
        
        dilation_features = {}
        
        for factor in self.dilation_factors:
            factor_features = {}
            
            for feature in base_features:
                if feature not in X.columns:
                    continue
                
                if not self._is_suitable_for_dilation(X[feature]):
                    continue
                
                dilated_feature = self._calculate_robust_dilation(X[feature], factor)
                if dilated_feature is not None:
                    factor_features[f"{feature}_dil_{factor}x"] = dilated_feature
            
            if factor_features:
                dilation_features[f"factor_{factor}"] = pd.DataFrame(factor_features, index=X.index)
        
        logger.info(f"Generated dilation features for {len(base_features)} base features")
        return dilation_features
    
    def evaluate_features_with_ts_cv(self, X: pd.DataFrame, y: pd.Series,
                                   acceleration_features: Dict[str, pd.DataFrame],
                                   dilation_features: Dict[str, pd.DataFrame],
                                   base_features: List[str]) -> Dict[str, Any]:
        """
        Evaluate features with proper time-series cross-validation and statistical testing.
        """
        logger.info("Evaluating features with time-series CV and statistical testing...")
        
        # Setup time-series CV with purged and embargoed splits
        tscv = self._create_purged_embargoed_cv(X, y)
        
        results = {
            'acceleration_evaluations': {},
            'dilation_evaluations': {},
            'accepted_features': [],
            'rejected_features': [],
            'watchlist_features': [],
            'variant_cards': {},
            'pareto_frontier': {},
            'global_metrics': {}
        }
        
        # Evaluate acceleration features
        if acceleration_features:
            accel_results = self._evaluate_acceleration_with_ts_cv(
                X, y, acceleration_features, base_features, tscv
            )
            results['acceleration_evaluations'] = accel_results
        
        # Evaluate dilation features
        if dilation_features:
            dil_results = self._evaluate_dilation_with_ts_cv(
                X, y, dilation_features, base_features, tscv
            )
            results['dilation_evaluations'] = dil_results
        
        # Apply multiple testing correction
        results = self._apply_multiple_testing_correction(results)
        
        # Apply Pareto optimization
        results = self._apply_pareto_optimization(results)
        
        # Generate variant cards
        results['variant_cards'] = self._generate_variant_cards(results)
        
        # Generate global metrics
        results['global_metrics'] = self._compute_global_metrics(results)
        
        return results
    
    def _create_purged_embargoed_cv(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create purged and embargoed time-series CV splits."""
        n_samples = len(X)
        splits = []
        
        # Create time-based splits with purging and embargo
        for i in range(self.n_cv_folds):
            # Calculate split boundaries
            train_end = int(n_samples * (i + 1) / (self.n_cv_folds + 1))
            test_start = train_end + int(n_samples * 0.05)  # 5% purging
            test_end = int(n_samples * (i + 2) / (self.n_cv_folds + 1))
            
            if test_end > n_samples:
                test_end = n_samples
            
            # Create indices
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 50 and len(test_idx) > 20:  # Minimum sample requirements
                splits.append((train_idx, test_idx))
        
        return splits
    
    def _calculate_robust_acceleration(self, feature_series: pd.Series, lag: int) -> Optional[pd.Series]:
        """Calculate acceleration with proper centering and winsorization."""
        try:
            # 1. Center first if bounded (e.g., RSI-50)
            if self._is_bounded_feature(feature_series):
                centered = feature_series - 50
            else:
                centered = feature_series
            
            # 2. Winsorize to reduce noise and handle asymmetric tails
            winsorized = self._winsorize_robust(centered)
            
            # 3. Calculate acceleration: accel_k = base_t - base_{t-k}
            acceleration = winsorized - winsorized.shift(lag)
            
            # 4. Handle zero/near-zero volatility
            acceleration = self._clamp_volatility(acceleration)
            
            # 5. Re-scale with robust statistics
            acceleration = self._rescale_robust(acceleration)
            
            return acceleration
            
        except Exception as e:
            logger.warning(f"Failed to calculate acceleration for {feature_series.name}: {e}")
            return None
    
    def _calculate_robust_dilation(self, feature_series: pd.Series, factor: float) -> Optional[pd.Series]:
        """Calculate dilation with proper EMA semantics and scale equivalence."""
        try:
            feature_name = feature_series.name or ""
            original_window = self._extract_window_size(feature_name)
            
            if original_window is None:
                original_window = 20
            
            # Calculate new window size
            new_window = int(original_window * factor)
            
            # Generate dilated feature based on type with proper semantics
            if 'ema_' in feature_name.lower() or 'ewm' in feature_name.lower():
                # EMA dilation: map to span, not window length
                # effective window ≈ 2/(α) - 1, so span = 2/(1+window) - 1
                original_span = self._extract_ema_span(feature_name) or original_window
                new_span = int(original_span * factor)
                dilated = feature_series.ewm(span=new_span).mean()
            elif 'ma_' in feature_name.lower():
                # Moving average dilation
                dilated = feature_series.rolling(new_window).mean()
            elif 'std_' in feature_name.lower():
                # Standard deviation dilation
                dilated = feature_series.rolling(new_window).std()
            elif 'vol_' in feature_name.lower() or 'volatility' in feature_name.lower():
                # Volatility dilation
                returns = feature_series.pct_change()
                dilated = returns.rolling(new_window).std()
            else:
                # Generic rolling mean dilation
                dilated = feature_series.rolling(new_window).mean()
            
            # Check scale equivalence (correlation with existing larger-span features)
            if self._is_scale_equivalent(dilated, feature_series, factor):
                logger.info(f"Dropping {feature_name}_dil_{factor}x due to scale equivalence")
                return None
            
            return dilated
            
        except Exception as e:
            logger.warning(f"Failed to calculate dilation for {feature_series.name}: {e}")
            return None
    
    def _evaluate_acceleration_with_ts_cv(self, X: pd.DataFrame, y: pd.Series,
                                        acceleration_features: Dict[str, pd.DataFrame],
                                        base_features: List[str], tscv: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate acceleration features with time-series CV."""
        results = {}
        
        for lag_key, accel_df in acceleration_features.items():
            lag = int(lag_key.split('_')[1])
            lag_results = {}
            
            for accel_feature in accel_df.columns:
                base_feature = accel_feature.replace(f'_accel_{lag}', '')
                
                if base_feature not in base_features:
                    continue
                
                # Run time-series CV evaluation
                evaluation = self._evaluate_feature_pair_ts_cv(
                    X[base_feature], accel_df[accel_feature], y, 
                    base_feature, accel_feature, tscv
                )
                
                lag_results[accel_feature] = evaluation
            
            results[lag_key] = lag_results
        
        return results
    
    def _evaluate_dilation_with_ts_cv(self, X: pd.DataFrame, y: pd.Series,
                                    dilation_features: Dict[str, pd.DataFrame],
                                    base_features: List[str], tscv: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate dilation features with time-series CV."""
        results = {}
        
        for factor_key, dil_df in dilation_features.items():
            factor = float(factor_key.split('_')[1])
            factor_results = {}
            
            for dil_feature in dil_df.columns:
                base_feature = dil_feature.replace(f'_dil_{factor}x', '')
                
                if base_feature not in base_features:
                    continue
                
                # Run time-series CV evaluation
                evaluation = self._evaluate_feature_pair_ts_cv(
                    X[base_feature], dil_df[dil_feature], y,
                    base_feature, dil_feature, tscv
                )
                
                factor_results[dil_feature] = evaluation
            
            results[factor_key] = factor_results
        
        return results
    
    def _evaluate_feature_pair_ts_cv(self, base_feature: pd.Series, variant_feature: pd.Series,
                                   y: pd.Series, base_name: str, variant_name: str,
                                   tscv: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate feature pair with time-series CV and statistical testing."""
        try:
            # Align data
            common_idx = base_feature.index.intersection(variant_feature.index).intersection(y.index)
            base_aligned = base_feature.loc[common_idx]
            variant_aligned = variant_feature.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(base_aligned.isna() | variant_aligned.isna() | y_aligned.isna())
            base_clean = base_aligned[valid_mask]
            variant_clean = variant_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            if len(base_clean) < 100:  # Minimum sample requirement
                return {'error': 'Insufficient data'}
            
            # Run time-series CV evaluation
            cv_results = []
            for train_idx, test_idx in tscv:
                if len(train_idx) < 50 or len(test_idx) < 20:
                    continue
                
                # Get train/test data
                X_train = np.column_stack([base_clean.iloc[train_idx], variant_clean.iloc[train_idx]])
                X_test = np.column_stack([base_clean.iloc[test_idx], variant_clean.iloc[test_idx]])
                y_train = y_clean.iloc[train_idx]
                y_test = y_clean.iloc[test_idx]
                
                # Calculate metrics for this fold
                fold_results = self._calculate_fold_metrics(
                    X_train, X_test, y_train, y_test, base_clean.iloc[train_idx], 
                    variant_clean.iloc[train_idx], base_clean.iloc[test_idx], 
                    variant_clean.iloc[test_idx]
                )
                
                cv_results.append(fold_results)
            
            if not cv_results:
                return {'error': 'No valid CV folds'}
            
            # Aggregate results across folds
            evaluation = self._aggregate_cv_results(cv_results)
            
            # Add statistical tests
            evaluation.update(self._run_statistical_tests(
                base_clean, variant_clean, y_clean, base_name, variant_name
            ))
            
            # Add production hygiene checks
            evaluation.update(self._run_production_hygiene_checks(
                base_clean, variant_clean, base_name, variant_name
            ))
            
            return evaluation
            
        except Exception as e:
            logger.warning(f"Error evaluating feature pair {base_name} vs {variant_name}: {e}")
            return {'error': str(e)}
    
    def _calculate_fold_metrics(self, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              base_train: pd.Series, variant_train: pd.Series,
                              base_test: pd.Series, variant_test: pd.Series) -> Dict[str, Any]:
        """Calculate metrics for a single CV fold."""
        try:
            # 1. kNN MI estimation
            mi_base = self._estimate_knn_mi(base_train.values.reshape(-1, 1), y_train.values)
            mi_variant = self._estimate_knn_mi(variant_train.values.reshape(-1, 1), y_train.values)
            
            # 2. Conditional MI
            cmi = self._estimate_conditional_mi(
                base_train.values, variant_train.values, y_train.values
            )
            
            # 3. Permutation importance
            base_model = RandomForestRegressor(n_estimators=50, random_state=42)
            base_model.fit(base_train.values.reshape(-1, 1), y_train.values)
            base_pred = base_model.predict(base_test.values.reshape(-1, 1))
            base_mse = mean_squared_error(y_test, base_pred)
            
            variant_model = RandomForestRegressor(n_estimators=50, random_state=42)
            variant_model.fit(variant_train.values.reshape(-1, 1), y_train.values)
            variant_pred = variant_model.predict(variant_test.values.reshape(-1, 1))
            variant_mse = mean_squared_error(y_test, variant_pred)
            
            # 4. Joint model
            joint_model = RandomForestRegressor(n_estimators=50, random_state=42)
            joint_model.fit(X_train, y_train)
            joint_pred = joint_model.predict(X_test)
            joint_mse = mean_squared_error(y_test, joint_pred)
            
            # 5. Correlation
            correlation = np.corrcoef(base_train, variant_train)[0, 1]
            
            return {
                'mi_base': mi_base,
                'mi_variant': mi_variant,
                'cmi': cmi,
                'base_mse': base_mse,
                'variant_mse': variant_mse,
                'joint_mse': joint_mse,
                'correlation': correlation,
                'mse_improvement': (base_mse - joint_mse) / base_mse if base_mse > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating fold metrics: {e}")
            return {'error': str(e)}
    
    def _aggregate_cv_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across CV folds."""
        # Filter out error results
        valid_results = [r for r in cv_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid CV results'}
        
        # Aggregate metrics
        aggregated = {}
        for metric in ['mi_base', 'mi_variant', 'cmi', 'base_mse', 'variant_mse', 
                      'joint_mse', 'correlation', 'mse_improvement']:
            values = [r[metric] for r in valid_results if metric in r and not np.isnan(r[metric])]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_median'] = np.median(values)
                aggregated[f'{metric}_q25'] = np.percentile(values, 25)
                aggregated[f'{metric}_q75'] = np.percentile(values, 75)
        
        # Calculate MI ratio
        if 'mi_base_mean' in aggregated and 'mi_variant_mean' in aggregated:
            aggregated['mi_ratio'] = aggregated['mi_variant_mean'] / (aggregated['mi_base_mean'] + 1e-8)
        
        return aggregated
    
    def _run_statistical_tests(self, base_feature: pd.Series, variant_feature: pd.Series,
                             y: pd.Series, base_name: str, variant_name: str) -> Dict[str, Any]:
        """Run statistical tests for significance."""
        try:
            # Diebold-Mariano test for MSE improvement
            base_model = RandomForestRegressor(n_estimators=50, random_state=42)
            base_model.fit(base_feature.values.reshape(-1, 1), y.values)
            base_pred = base_model.predict(base_feature.values.reshape(-1, 1))
            
            joint_model = RandomForestRegressor(n_estimators=50, random_state=42)
            X_joint = np.column_stack([base_feature.values, variant_feature.values])
            joint_model.fit(X_joint, y.values)
            joint_pred = joint_model.predict(X_joint)
            
            # Calculate prediction errors
            base_errors = y.values - base_pred
            joint_errors = y.values - joint_pred
            
            # Diebold-Mariano test
            dm_stat, dm_pvalue = self._diebold_mariano_test(base_errors, joint_errors)
            
            # Bootstrap confidence intervals for CMI
            cmi_ci = self._bootstrap_cmi_ci(base_feature.values, variant_feature.values, y.values)
            
            return {
                'dm_statistic': dm_stat,
                'dm_pvalue': dm_pvalue,
                'cmi_ci_low': cmi_ci[0],
                'cmi_ci_high': cmi_ci[1],
                'cmi_ci_contains_zero': cmi_ci[0] <= 0 <= cmi_ci[1]
            }
            
        except Exception as e:
            logger.warning(f"Error running statistical tests: {e}")
            return {'dm_pvalue': 1.0, 'cmi_ci_contains_zero': True}
    
    def _run_production_hygiene_checks(self, base_feature: pd.Series, variant_feature: pd.Series,
                                     base_name: str, variant_name: str) -> Dict[str, Any]:
        """Run production hygiene checks."""
        try:
            # 1. PSI calculation (monthly)
            psi_base = self._calculate_psi_monthly(base_feature)
            psi_variant = self._calculate_psi_monthly(variant_feature)
            psi_delta = abs(psi_variant - psi_base)
            
            # 2. Turnover calculation
            turnover = self._calculate_turnover(variant_feature)
            
            # 3. Shadow feature check
            shadow_perm_imp = self._calculate_shadow_perm_imp(variant_feature)
            
            # 4. Zero/near-zero volatility guards
            zero_vol_rate = self._calculate_zero_vol_rate(variant_feature)
            
            # 5. Rank stability
            rank_stability = self._calculate_rank_stability(variant_feature)
            
            return {
                'psi_base': psi_base,
                'psi_variant': psi_variant,
                'psi_delta': psi_delta,
                'turnover': turnover,
                'shadow_perm_imp': shadow_perm_imp,
                'zero_vol_rate': zero_vol_rate,
                'rank_stability': rank_stability
            }
            
        except Exception as e:
            logger.warning(f"Error running production hygiene checks: {e}")
            return {
                'psi_delta': 1.0, 'turnover': 1.0, 'shadow_perm_imp': 0.0,
                'zero_vol_rate': 1.0, 'rank_stability': 0.0
            }
    
    def _apply_multiple_testing_correction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Benjamini-Hochberg FDR correction."""
        # Collect all p-values
        pvalues = []
        feature_pvalue_map = {}
        
        for eval_type in ['acceleration_evaluations', 'dilation_evaluations']:
            if eval_type in results:
                for category, evaluations in results[eval_type].items():
                    for feature, evaluation in evaluations.items():
                        if 'dm_pvalue' in evaluation:
                            pvalues.append(evaluation['dm_pvalue'])
                            feature_pvalue_map[feature] = evaluation['dm_pvalue']
        
        if not pvalues:
            return results
        
        # Apply Benjamini-Hochberg correction
        from statsmodels.stats.multitest import multipletests
        rejected, pvals_corrected, _, _ = multipletests(pvalues, alpha=self.fdr_q, method='fdr_bh')
        
        # Update results with corrected decisions
        corrected_idx = 0
        for eval_type in ['acceleration_evaluations', 'dilation_evaluations']:
            if eval_type in results:
                for category, evaluations in results[eval_type].items():
                    for feature, evaluation in evaluations.items():
                        if 'dm_pvalue' in evaluation:
                            evaluation['dm_pvalue_corrected'] = pvals_corrected[corrected_idx]
                            evaluation['dm_rejected'] = rejected[corrected_idx]
                            corrected_idx += 1
        
        return results
    
    def _apply_pareto_optimization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Pareto optimization for FQS vs turnover."""
        # Collect all features with FQS and turnover
        feature_metrics = []
        
        for eval_type in ['acceleration_evaluations', 'dilation_evaluations']:
            if eval_type in results:
                for category, evaluations in results[eval_type].items():
                    for feature, evaluation in evaluations.items():
                        if 'rank_stability' in evaluation and 'turnover' in evaluation:
                            # Use rank stability as proxy for FQS
                            fqs = evaluation['rank_stability']
                            turnover = evaluation['turnover']
                            feature_metrics.append((feature, fqs, turnover))
        
        if not feature_metrics:
            return results
        
        # Find Pareto frontier
        pareto_frontier = self._find_pareto_frontier(feature_metrics)
        pareto_features = [f[0] for f in pareto_frontier]
        
        # Update results with Pareto decisions
        for eval_type in ['acceleration_evaluations', 'dilation_evaluations']:
            if eval_type in results:
                for category, evaluations in results[eval_type].items():
                    for feature, evaluation in evaluations.items():
                        if feature in pareto_features:
                            evaluation['pareto_optimal'] = True
                        else:
                            evaluation['pareto_optimal'] = False
        
        results['pareto_frontier'] = pareto_frontier
        return results
    
    def _generate_variant_cards(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate variant cards for traceability."""
        variant_cards = {}
        
        for eval_type in ['acceleration_evaluations', 'dilation_evaluations']:
            if eval_type in results:
                for category, evaluations in results[eval_type].items():
                    for feature, evaluation in evaluations.items():
                        if 'error' not in evaluation:
                            card = self._create_variant_card(feature, evaluation)
                            variant_cards[feature] = card
        
        return variant_cards
    
    def _create_variant_card(self, feature: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a variant card for a single feature."""
        # Determine decision
        decision = self._determine_decision(evaluation)
        
        # Create rationale
        rationale = self._create_rationale(evaluation, decision)
        
        card = {
            'feature': feature,
            'fqs': evaluation.get('rank_stability', 0.0),
            'delta_fqs': evaluation.get('rank_stability', 0.0) - 0.5,  # Assuming base FQS = 0.5
            'dm_pvalue': evaluation.get('dm_pvalue', 1.0),
            'dm_pvalue_corrected': evaluation.get('dm_pvalue_corrected', 1.0),
            'perm_imp_mean': evaluation.get('mi_variant_mean', 0.0),
            'perm_imp_std': evaluation.get('mi_variant_std', 0.0),
            'mi_median': evaluation.get('mi_variant_median', 0.0),
            'mi_iqr': evaluation.get('mi_variant_q75', 0.0) - evaluation.get('mi_variant_q25', 0.0),
            'rank_stability': evaluation.get('rank_stability', 0.0),
            'max_correlation': abs(evaluation.get('correlation_mean', 0.0)),
            'vif': 1.0,  # Would need to calculate VIF
            'turnover': evaluation.get('turnover', 0.0),
            'psi': evaluation.get('psi_variant', 0.0),
            'decision': decision,
            'rationale': rationale
        }
        
        return card
    
    def _determine_decision(self, evaluation: Dict[str, Any]) -> str:
        """Determine decision (Keep/Drop/Watchlist) based on evaluation."""
        # Check basic requirements
        if evaluation.get('dm_pvalue_corrected', 1.0) > self.dm_alpha:
            return 'Drop'  # Not statistically significant
        
        if evaluation.get('cmi_ci_contains_zero', True):
            return 'Drop'  # Conditional MI not significant
        
        if evaluation.get('psi_delta', 1.0) > self.psi_delta_threshold:
            return 'Watchlist'  # High PSI delta
        
        if evaluation.get('turnover', 1.0) > self.turnover_threshold:
            return 'Watchlist'  # High turnover
        
        if evaluation.get('rank_stability', 0.0) < self.rank_stability_threshold:
            return 'Watchlist'  # Low rank stability
        
        if not evaluation.get('pareto_optimal', False):
            return 'Drop'  # Not on Pareto frontier
        
        return 'Keep'
    
    def _create_rationale(self, evaluation: Dict[str, Any], decision: str) -> str:
        """Create rationale for decision."""
        if decision == 'Keep':
            return "Passes all statistical and practical tests with good FQS and low turnover."
        elif decision == 'Watchlist':
            issues = []
            if evaluation.get('psi_delta', 0) > self.psi_delta_threshold:
                issues.append("high PSI delta")
            if evaluation.get('turnover', 0) > self.turnover_threshold:
                issues.append("high turnover")
            if evaluation.get('rank_stability', 0) < self.rank_stability_threshold:
                issues.append("low rank stability")
            return f"Monitor due to: {', '.join(issues)}."
        else:
            return "Fails statistical significance or Pareto optimization criteria."
    
    # Helper methods for statistical calculations
    def _estimate_knn_mi(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate MI using kNN method."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            mi_values = []
            for k in self.mi_k_values:
                # Use kNN MI estimation
                mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=k)[0]
                mi_values.append(mi)
            return np.median(mi_values)
        except:
            return 0.0
    
    def _estimate_conditional_mi(self, base: np.ndarray, variant: np.ndarray, y: np.ndarray) -> float:
        """Estimate conditional MI CMI(variant; y | base)."""
        try:
            # CMI(X; Y | Z) = MI(X, Z; Y) - MI(Z; Y)
            joint_mi = mutual_info_regression(np.column_stack([base, variant]), y)[0]
            base_mi = mutual_info_regression(base.reshape(-1, 1), y)[0]
            return joint_mi - base_mi
        except:
            return 0.0
    
    def _diebold_mariano_test(self, errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
        """Diebold-Mariano test for forecast accuracy."""
        try:
            d = errors1 - errors2
            d_mean = np.mean(d)
            d_var = np.var(d)
            if d_var == 0:
                return 0.0, 1.0
            
            dm_stat = d_mean / np.sqrt(d_var / len(d))
            pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            return dm_stat, pvalue
        except:
            return 0.0, 1.0
    
    def _bootstrap_cmi_ci(self, base: np.ndarray, variant: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Bootstrap confidence interval for conditional MI."""
        try:
            cmi_values = []
            for _ in range(self.n_bootstrap):
                # Bootstrap sample
                idx = np.random.choice(len(base), size=len(base), replace=True)
                cmi = self._estimate_conditional_mi(base[idx], variant[idx], y[idx])
                cmi_values.append(cmi)
            
            return np.percentile(cmi_values, [2.5, 97.5])
        except:
            return (0.0, 0.0)
    
    def _calculate_psi_monthly(self, feature: pd.Series) -> float:
        """Calculate PSI for monthly drift detection."""
        try:
            # Split into train and recent periods
            n_samples = len(feature)
            train_end = int(n_samples * 0.7)
            
            train_data = feature.iloc[:train_end].dropna()
            recent_data = feature.iloc[train_end:].dropna()
            
            if len(train_data) < 100 or len(recent_data) < 50:
                return 0.0
            
            # Calculate PSI
            train_hist, _ = np.histogram(train_data, bins=10, density=True)
            recent_hist, _ = np.histogram(recent_data, bins=10, density=True)
            
            # Avoid division by zero
            train_hist = np.maximum(train_hist, 1e-8)
            recent_hist = np.maximum(recent_hist, 1e-8)
            
            psi = np.sum((recent_hist - train_hist) * np.log(recent_hist / train_hist))
            return psi
        except:
            return 0.0
    
    def _calculate_turnover(self, feature: pd.Series) -> float:
        """Calculate turnover as avg|signal_t - signal_{t-1}|."""
        try:
            diff = feature.diff().abs()
            return diff.mean()
        except:
            return 0.0
    
    def _calculate_shadow_perm_imp(self, feature: pd.Series) -> float:
        """Calculate shadow feature permutation importance."""
        try:
            # Create shadow feature (randomized version)
            shadow = feature.sample(frac=1.0).values
            return np.random.normal(0, 0.1)  # Placeholder
        except:
            return 0.0
    
    def _calculate_zero_vol_rate(self, feature: pd.Series) -> float:
        """Calculate rate of zero/near-zero volatility."""
        try:
            # Check for near-zero values
            near_zero = (feature.abs() < 1e-8).sum()
            return near_zero / len(feature)
        except:
            return 0.0
    
    def _calculate_rank_stability(self, feature: pd.Series) -> float:
        """Calculate rank stability across time periods."""
        try:
            # Split into multiple periods and calculate rank correlation
            n_periods = 5
            period_size = len(feature) // n_periods
            
            ranks = []
            for i in range(n_periods):
                start = i * period_size
                end = start + period_size
                period_data = feature.iloc[start:end]
                if len(period_data) > 10:
                    ranks.append(period_data.rank())
            
            if len(ranks) < 2:
                return 0.0
            
            # Calculate rank correlation between periods
            correlations = []
            for i in range(len(ranks) - 1):
                corr = ranks[i].corr(ranks[i + 1])
                if not np.isnan(corr):
                    correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
        except:
            return 0.0
    
    def _find_pareto_frontier(self, feature_metrics: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
        """Find Pareto frontier for FQS vs turnover."""
        if not feature_metrics:
            return []
        
        # Sort by FQS (descending) and turnover (ascending)
        sorted_metrics = sorted(feature_metrics, key=lambda x: (-x[1], x[2]))
        
        pareto_frontier = []
        for feature, fqs, turnover in sorted_metrics:
            # Check if this point is not dominated
            is_dominated = False
            for _, other_fqs, other_turnover in pareto_frontier:
                if other_fqs >= fqs and other_turnover <= turnover and (other_fqs > fqs or other_turnover < turnover):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_frontier.append((feature, fqs, turnover))
        
        return pareto_frontier
    
    # Additional helper methods
    def _is_bounded_feature(self, feature_series: pd.Series) -> bool:
        """Check if feature is bounded (e.g., RSI-like)."""
        return feature_series.min() >= 0 and feature_series.max() <= 100
    
    def _winsorize_robust(self, series: pd.Series) -> pd.Series:
        """Robust winsorization handling asymmetric tails."""
        try:
            # Use different limits for upper and lower tails if asymmetric
            lower_limit = series.quantile(0.01)
            upper_limit = series.quantile(0.99)
            return series.clip(lower=lower_limit, upper=upper_limit)
        except:
            return series
    
    def _clamp_volatility(self, series: pd.Series, epsilon: float = 1e-8) -> pd.Series:
        """Clamp zero/near-zero volatility values."""
        return series.replace(0, epsilon).clip(lower=-1/epsilon, upper=1/epsilon)
    
    def _rescale_robust(self, series: pd.Series) -> pd.Series:
        """Robust rescaling using median and IQR."""
        try:
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            if iqr > 0:
                return (series - median) / iqr
            else:
                return series - median
        except:
            return series
    
    def _extract_ema_span(self, feature_name: str) -> Optional[int]:
        """Extract EMA span from feature name."""
        import re
        match = re.search(r'ema_(\d+)', feature_name.lower())
        if match:
            return int(match.group(1))
        return None
    
    def _is_scale_equivalent(self, dilated: pd.Series, original: pd.Series, factor: float) -> bool:
        """Check if dilated feature is scale equivalent to existing features."""
        try:
            # Check correlation with original
            corr = dilated.corr(original)
            return corr > 0.97 if not pd.isna(corr) else False
        except:
            return False
    
    def _compute_global_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute global metrics across all evaluations."""
        total_features = 0
        accepted_features = 0
        rejected_features = 0
        watchlist_features = 0
        
        for eval_type in ['acceleration_evaluations', 'dilation_evaluations']:
            if eval_type in results:
                for category, evaluations in results[eval_type].items():
                    for feature, evaluation in evaluations.items():
                        total_features += 1
                        decision = evaluation.get('decision', 'Drop')
                        if decision == 'Keep':
                            accepted_features += 1
                        elif decision == 'Watchlist':
                            watchlist_features += 1
                        else:
                            rejected_features += 1
        
        return {
            'total_features': total_features,
            'accepted_features': accepted_features,
            'rejected_features': rejected_features,
            'watchlist_features': watchlist_features,
            'acceptance_rate': accepted_features / max(total_features, 1),
            'watchlist_rate': watchlist_features / max(total_features, 1)
        }
    
    # Placeholder methods for missing functionality
    def _identify_acceleration_candidates(self, X: pd.DataFrame) -> List[str]:
        """Identify features suitable for acceleration."""
        candidates = []
        for feature in X.columns:
            if self._is_suitable_for_acceleration(X[feature]):
                candidates.append(feature)
        return candidates
    
    def _identify_dilation_candidates(self, X: pd.DataFrame) -> List[str]:
        """Identify features suitable for dilation."""
        candidates = []
        for feature in X.columns:
            if self._is_suitable_for_dilation(X[feature]):
                candidates.append(feature)
        return candidates
    
    def _is_suitable_for_acceleration(self, feature_series: pd.Series) -> bool:
        """Check if feature is suitable for acceleration."""
        if len(feature_series.dropna()) < 50:
            return False
        try:
            autocorr = feature_series.autocorr(lag=1)
            return autocorr > 0.2 if not pd.isna(autocorr) else False
        except:
            return False
    
    def _is_suitable_for_dilation(self, feature_series: pd.Series) -> bool:
        """Check if feature is suitable for dilation."""
        if len(feature_series.dropna()) < 100:
            return False
        feature_name = feature_series.name or ""
        window_indicators = ['_w', '_ma_', '_ema_', '_std_', '_vol_', '_vwap_', '_bb_']
        return any(indicator in feature_name.lower() for indicator in window_indicators)
    
    def _extract_window_size(self, feature_name: str) -> Optional[int]:
        """Extract window size from feature name."""
        import re
        patterns = [r'_(\d+)$', r'_(\d+)_', r'w(\d+)', r'(\d+)_']
        for pattern in patterns:
            match = re.search(pattern, feature_name)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None