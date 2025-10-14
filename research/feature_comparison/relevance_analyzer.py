"""
Feature Relevance Analyzer

This module provides tools to analyze feature relevance using multiple methods:
- LGBM with SHAP values
- LASSO regression
- Mutual Information
- Other statistical measures
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, Bootstrap
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr
from scipy import stats
import warnings
from .robust_scaling import RobustFeatureScaler, MultiMethodScaler

# Try to import LGBM and SHAP
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not available. LGBM analysis will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. SHAP analysis will be skipped.")

logger = logging.getLogger(__name__)

class RelevanceAnalyzer:
    """
    Analyzes feature relevance using multiple methods.
    """
    
    def __init__(self, random_state: int = 42, scaling_method: str = 'robust'):
        """
        Initialize the relevance analyzer.
        
        Args:
            random_state: Random state for reproducibility
            scaling_method: Scaling method ('standard', 'robust', 'minmax', 'quantile', 'power')
        """
        self.random_state = random_state
        self.scaling_method = scaling_method
        self.scaler = RobustFeatureScaler(method=scaling_method)
        self.multi_scaler = MultiMethodScaler()
        self.bootstrap_results = {}
        self.temporal_results = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, robust_scaling: bool = True) -> Tuple[pd.DataFrame, pd.Series, 
                                                    pd.DataFrame, pd.Series]:
        """
        Prepare data for analysis by splitting and scaling.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test set size
            robust_scaling: Whether to use robust scaling
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Remove any rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=self.random_state
        )
        
        if robust_scaling:
            # Use robust scaling
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Use standard scaling
            X_train_scaled = pd.DataFrame(
                StandardScaler().fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                StandardScaler().fit_transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def lgbm_shap_analysis(self, X: pd.DataFrame, y: pd.Series, 
                          task_type: str = 'regression') -> Dict[str, Any]:
        """
        Perform LGBM analysis with SHAP values.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with LGBM and SHAP results
        """
        if not LGBM_AVAILABLE:
            logger.warning("LightGBM not available. Skipping LGBM analysis.")
            return {}
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP analysis.")
            return {}
        
        try:
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data(X, y)
            
            # Configure LGBM parameters
            if task_type == 'regression':
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': self.random_state
                }
            else:
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': self.random_state
                }
            
            # Train LGBM model
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Calculate mean absolute SHAP values
            if task_type == 'regression':
                mean_shap_values = np.abs(shap_values).mean(axis=0)
            else:
                mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            shap_importance = pd.DataFrame({
                'feature': X_test.columns,
                'shap_importance': mean_shap_values
            }).sort_values('shap_importance', ascending=False)
            
            # Calculate test performance
            y_pred = model.predict(X_test)
            if task_type == 'regression':
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                performance = {'mse': mse, 'r2': r2}
            else:
                from sklearn.metrics import accuracy_score, roc_auc_score
                y_pred_binary = (y_pred > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred_binary)
                try:
                    auc = roc_auc_score(y_test, y_pred)
                    performance = {'accuracy': accuracy, 'auc': auc}
                except:
                    performance = {'accuracy': accuracy}
            
            return {
                'model': model,
                'feature_importance': feature_importance,
                'shap_importance': shap_importance,
                'performance': performance,
                'shap_values': shap_values
            }
            
        except Exception as e:
            logger.error(f"Error in LGBM-SHAP analysis: {e}")
            return {}
    
    def lasso_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform LASSO regression analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with LASSO results
        """
        try:
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data(X, y)
            
            # Fit LASSO with cross-validation
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=2000)
            lasso.fit(X_train, y_train)
            
            # Get feature coefficients
            feature_coef = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': lasso.coef_,
                'abs_coefficient': np.abs(lasso.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Calculate performance
            y_pred = lasso.predict(X_test)
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': lasso,
                'feature_coefficients': feature_coef,
                'performance': {'mse': mse, 'r2': r2},
                'alpha': lasso.alpha_,
                'selected_features': feature_coef[feature_coef['abs_coefficient'] > 0]['feature'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in LASSO analysis: {e}")
            return {}
    
    def mutual_information_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                  task_type: str = 'regression') -> Dict[str, Any]:
        """
        Perform mutual information analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with mutual information results
        """
        try:
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data(X, y)
            
            # Calculate mutual information
            if task_type == 'regression':
                mi_scores = mutual_info_regression(X_train, y_train, random_state=self.random_state)
            else:
                mi_scores = mutual_info_classif(X_train, y_train, random_state=self.random_state)
            
            # Create results DataFrame
            mi_results = pd.DataFrame({
                'feature': X_train.columns,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
            
            return {
                'mutual_info_scores': mi_results,
                'mean_mi': mi_scores.mean(),
                'std_mi': mi_scores.std()
            }
            
        except Exception as e:
            logger.error(f"Error in mutual information analysis: {e}")
            return {}
    
    def correlation_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform correlation analysis between features and target.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with correlation results
        """
        try:
            # Calculate correlations
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            
            # Create results DataFrame
            corr_results = pd.DataFrame({
                'feature': correlations.index,
                'correlation': correlations.values
            })
            
            return {
                'correlations': corr_results,
                'mean_correlation': correlations.mean(),
                'max_correlation': correlations.max()
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def comprehensive_analysis(self, X: pd.DataFrame, y: pd.Series, 
                             task_type: str = 'regression') -> Dict[str, Any]:
        """
        Perform comprehensive feature relevance analysis using all methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive feature relevance analysis...")
        
        results = {
            'task_type': task_type,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # LGBM-SHAP analysis
        logger.info("Running LGBM-SHAP analysis...")
        lgbm_results = self.lgbm_shap_analysis(X, y, task_type)
        results['lgbm_shap'] = lgbm_results
        
        # LASSO analysis
        logger.info("Running LASSO analysis...")
        lasso_results = self.lasso_analysis(X, y)
        results['lasso'] = lasso_results
        
        # Mutual Information analysis
        logger.info("Running mutual information analysis...")
        mi_results = self.mutual_information_analysis(X, y, task_type)
        results['mutual_info'] = mi_results
        
        # Correlation analysis
        logger.info("Running correlation analysis...")
        corr_results = self.correlation_analysis(X, y)
        results['correlation'] = corr_results
        
        # Create combined ranking
        results['combined_ranking'] = self._create_combined_ranking(results)
        
        logger.info("Comprehensive analysis completed.")
        return results
    
    def _create_combined_ranking(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a combined ranking of features based on all methods.
        
        Args:
            results: Results from comprehensive analysis
            
        Returns:
            DataFrame with combined rankings
        """
        try:
            # Get all unique features
            all_features = set()
            if 'lgbm_shap' in results and 'feature_importance' in results['lgbm_shap']:
                all_features.update(results['lgbm_shap']['feature_importance']['feature'])
            if 'lasso' in results and 'feature_coefficients' in results['lasso']:
                all_features.update(results['lasso']['feature_coefficients']['feature'])
            if 'mutual_info' in results and 'mutual_info_scores' in results['mutual_info']:
                all_features.update(results['mutual_info']['mutual_info_scores']['feature'])
            if 'correlation' in results and 'correlations' in results['correlation']:
                all_features.update(results['correlation']['correlations']['feature'])
            
            # Create ranking DataFrame
            ranking_df = pd.DataFrame({'feature': list(all_features)})
            
            # Add rankings from each method
            if 'lgbm_shap' in results and 'feature_importance' in results['lgbm_shap']:
                lgbm_rank = results['lgbm_shap']['feature_importance'].set_index('feature')['importance'].rank(ascending=False)
                ranking_df['lgbm_rank'] = ranking_df['feature'].map(lgbm_rank)
            
            if 'lasso' in results and 'feature_coefficients' in results['lasso']:
                lasso_rank = results['lasso']['feature_coefficients'].set_index('feature')['abs_coefficient'].rank(ascending=False)
                ranking_df['lasso_rank'] = ranking_df['feature'].map(lasso_rank)
            
            if 'mutual_info' in results and 'mutual_info_scores' in results['mutual_info']:
                mi_rank = results['mutual_info']['mutual_info_scores'].set_index('feature')['mutual_info'].rank(ascending=False)
                ranking_df['mi_rank'] = ranking_df['feature'].map(mi_rank)
            
            if 'correlation' in results and 'correlations' in results['correlation']:
                corr_rank = results['correlation']['correlations'].set_index('feature')['correlation'].rank(ascending=False)
                ranking_df['corr_rank'] = ranking_df['feature'].map(corr_rank)
            
            # Calculate average rank
            rank_cols = [col for col in ranking_df.columns if col.endswith('_rank')]
            ranking_df['avg_rank'] = ranking_df[rank_cols].mean(axis=1)
            ranking_df = ranking_df.sort_values('avg_rank')
            
            return ranking_df
            
        except Exception as e:
            logger.error(f"Error creating combined ranking: {e}")
            return pd.DataFrame()
    
    def bootstrap_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                   n_bootstrap: int = 100, 
                                   task_type: str = 'regression') -> Dict[str, Any]:
        """
        Perform bootstrap resampling to assess feature importance variance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_bootstrap: Number of bootstrap samples
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with bootstrap results
        """
        logger.info(f"Running bootstrap analysis with {n_bootstrap} samples...")
        
        bootstrap_results = {
            'n_bootstrap': n_bootstrap,
            'feature_importance_variance': {},
            'method_results': {}
        }
        
        # Bootstrap for LGBM
        if LGBM_AVAILABLE:
            lgbm_importances = []
            lgbm_performances = []
            
            for i in range(n_bootstrap):
                try:
                    # Bootstrap sample
                    bootstrap_indices = np.random.choice(
                        len(X), size=len(X), replace=True, random_state=self.random_state + i
                    )
                    X_boot = X.iloc[bootstrap_indices]
                    y_boot = y.iloc[bootstrap_indices]
                    
                    # Run LGBM analysis
                    lgbm_result = self.lgbm_shap_analysis(X_boot, y_boot, task_type)
                    
                    if 'feature_importance' in lgbm_result:
                        importance_df = lgbm_result['feature_importance'].set_index('feature')['importance']
                        lgbm_importances.append(importance_df)
                        
                    if 'performance' in lgbm_result:
                        perf = lgbm_result['performance']
                        lgbm_performances.append(perf.get('r2', 0) if task_type == 'regression' else perf.get('accuracy', 0))
                        
                except Exception as e:
                    logger.warning(f"Bootstrap iteration {i} failed: {e}")
                    continue
            
            if lgbm_importances:
                # Combine importance results
                importance_df = pd.DataFrame(lgbm_importances)
                bootstrap_results['method_results']['lgbm'] = {
                    'mean_importance': importance_df.mean(),
                    'std_importance': importance_df.std(),
                    'cv_importance': importance_df.std() / (importance_df.mean() + 1e-8),
                    'mean_performance': np.mean(lgbm_performances) if lgbm_performances else 0,
                    'std_performance': np.std(lgbm_performances) if lgbm_performances else 0
                }
        
        # Bootstrap for LASSO
        lasso_importances = []
        lasso_performances = []
        
        for i in range(n_bootstrap):
            try:
                # Bootstrap sample
                bootstrap_indices = np.random.choice(
                    len(X), size=len(X), replace=True, random_state=self.random_state + i
                )
                X_boot = X.iloc[bootstrap_indices]
                y_boot = y.iloc[bootstrap_indices]
                
                # Run LASSO analysis
                lasso_result = self.lasso_analysis(X_boot, y_boot)
                
                if 'feature_coefficients' in lasso_result:
                    coef_df = lasso_result['feature_coefficients'].set_index('feature')['abs_coefficient']
                    lasso_importances.append(coef_df)
                    
                if 'performance' in lasso_result:
                    perf = lasso_result['performance']
                    lasso_performances.append(perf.get('r2', 0))
                    
            except Exception as e:
                logger.warning(f"LASSO bootstrap iteration {i} failed: {e}")
                continue
        
        if lasso_importances:
            # Combine LASSO results
            importance_df = pd.DataFrame(lasso_importances)
            bootstrap_results['method_results']['lasso'] = {
                'mean_importance': importance_df.mean(),
                'std_importance': importance_df.std(),
                'cv_importance': importance_df.std() / (importance_df.mean() + 1e-8),
                'mean_performance': np.mean(lasso_performances) if lasso_performances else 0,
                'std_performance': np.std(lasso_performances) if lasso_performances else 0
            }
        
        self.bootstrap_results = bootstrap_results
        return bootstrap_results
    
    def calculate_rank_correlation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Spearman rank correlation between different methods.
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            Dictionary with rank correlation results
        """
        logger.info("Calculating rank correlations between methods...")
        
        rank_correlations = {}
        method_rankings = {}
        
        # Extract rankings from each method
        if 'lgbm_shap' in analysis_results and 'feature_importance' in analysis_results['lgbm_shap']:
            lgbm_ranking = analysis_results['lgbm_shap']['feature_importance'].set_index('feature')['importance']
            method_rankings['lgbm'] = lgbm_ranking
        
        if 'lasso' in analysis_results and 'feature_coefficients' in analysis_results['lasso']:
            lasso_ranking = analysis_results['lasso']['feature_coefficients'].set_index('feature')['abs_coefficient']
            method_rankings['lasso'] = lasso_ranking
        
        if 'mutual_info' in analysis_results and 'mutual_info_scores' in analysis_results['mutual_info']:
            mi_ranking = analysis_results['mutual_info']['mutual_info_scores'].set_index('feature')['mutual_info']
            method_rankings['mutual_info'] = mi_ranking
        
        if 'correlation' in analysis_results and 'correlations' in analysis_results['correlation']:
            corr_ranking = analysis_results['correlation']['correlations'].set_index('feature')['correlation']
            method_rankings['correlation'] = corr_ranking
        
        # Calculate pairwise correlations
        method_names = list(method_rankings.keys())
        correlation_matrix = pd.DataFrame(index=method_names, columns=method_names)
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i <= j:
                    try:
                        # Get common features
                        common_features = set(method_rankings[method1].index) & set(method_rankings[method2].index)
                        
                        if len(common_features) > 1:
                            # Get rankings for common features
                            rank1 = method_rankings[method1].loc[list(common_features)].rank(ascending=False)
                            rank2 = method_rankings[method2].loc[list(common_features)].rank(ascending=False)
                            
                            # Calculate Spearman correlation
                            corr, p_value = spearmanr(rank1, rank2)
                            correlation_matrix.loc[method1, method2] = corr
                            correlation_matrix.loc[method2, method1] = corr
                            
                            # Store detailed results
                            pair_key = f"{method1}_vs_{method2}"
                            rank_correlations[pair_key] = {
                                'spearman_correlation': corr,
                                'p_value': p_value,
                                'n_common_features': len(common_features),
                                'is_significant': p_value < 0.05
                            }
                        else:
                            correlation_matrix.loc[method1, method2] = np.nan
                            correlation_matrix.loc[method2, method1] = np.nan
                            
                    except Exception as e:
                        logger.warning(f"Error calculating correlation between {method1} and {method2}: {e}")
                        correlation_matrix.loc[method1, method2] = np.nan
                        correlation_matrix.loc[method2, method1] = np.nan
        
        rank_correlations['correlation_matrix'] = correlation_matrix
        rank_correlations['mean_correlation'] = correlation_matrix.replace(np.nan, 0).mean().mean()
        
        return rank_correlations
    
    def temporal_stability_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                  n_windows: int = 5, 
                                  task_type: str = 'regression') -> Dict[str, Any]:
        """
        Analyze temporal stability of feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_windows: Number of time windows to analyze
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with temporal stability results
        """
        logger.info(f"Running temporal stability analysis with {n_windows} windows...")
        
        # Sort by index to ensure temporal order
        X_sorted = X.sort_index()
        y_sorted = y.sort_index()
        
        n_samples = len(X_sorted)
        window_size = n_samples // n_windows
        
        temporal_results = {
            'n_windows': n_windows,
            'window_size': window_size,
            'window_results': {},
            'stability_metrics': {}
        }
        
        window_rankings = {}
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < n_windows - 1 else n_samples
            
            X_window = X_sorted.iloc[start_idx:end_idx]
            y_window = y_sorted.iloc[start_idx:end_idx]
            
            window_name = f"window_{i+1}"
            logger.info(f"Analyzing {window_name} ({start_idx}:{end_idx})")
            
            try:
                # Run analysis on this window
                window_analysis = self.comprehensive_analysis(X_window, y_window, task_type)
                temporal_results['window_results'][window_name] = window_analysis
                
                # Extract rankings for stability analysis
                if 'combined_ranking' in window_analysis:
                    ranking = window_analysis['combined_ranking'].set_index('feature')['avg_rank']
                    window_rankings[window_name] = ranking
                    
            except Exception as e:
                logger.warning(f"Error analyzing {window_name}: {e}")
                temporal_results['window_results'][window_name] = {'error': str(e)}
        
        # Calculate stability metrics
        if len(window_rankings) > 1:
            # Calculate ranking stability across windows
            all_features = set()
            for ranking in window_rankings.values():
                all_features.update(ranking.index)
            
            stability_scores = {}
            for feature in all_features:
                feature_ranks = []
                for ranking in window_rankings.values():
                    if feature in ranking.index:
                        feature_ranks.append(ranking[feature])
                
                if len(feature_ranks) > 1:
                    # Calculate coefficient of variation for ranking stability
                    mean_rank = np.mean(feature_ranks)
                    std_rank = np.std(feature_ranks)
                    cv_rank = std_rank / (mean_rank + 1e-8)
                    stability_scores[feature] = {
                        'mean_rank': mean_rank,
                        'std_rank': std_rank,
                        'cv_rank': cv_rank,
                        'stability_score': 1 / (1 + cv_rank)  # Higher is more stable
                    }
            
            temporal_results['stability_metrics'] = {
                'feature_stability': stability_scores,
                'mean_stability': np.mean([s['stability_score'] for s in stability_scores.values()]),
                'stable_features': [f for f, s in stability_scores.items() if s['stability_score'] > 0.7]
            }
        
        self.temporal_results = temporal_results
        return temporal_results
    
    def robust_comprehensive_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                    task_type: str = 'regression',
                                    include_bootstrap: bool = True,
                                    include_temporal: bool = True,
                                    n_bootstrap: int = 10,
                                    n_temporal_windows: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive analysis with robust evaluation methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'regression' or 'classification'
            include_bootstrap: Whether to include bootstrap analysis
            include_temporal: Whether to include temporal stability analysis
            n_bootstrap: Number of bootstrap samples
            n_temporal_windows: Number of temporal windows
            
        Returns:
            Dictionary with comprehensive robust analysis results
        """
        logger.info("Starting robust comprehensive feature relevance analysis...")
        
        # Run standard comprehensive analysis
        results = self.comprehensive_analysis(X, y, task_type)
        
        # Add rank correlation analysis
        logger.info("Adding rank correlation analysis...")
        rank_correlations = self.calculate_rank_correlation(results)
        results['rank_correlations'] = rank_correlations
        
        # Add bootstrap analysis
        if include_bootstrap:
            logger.info("Adding bootstrap analysis...")
            bootstrap_results = self.bootstrap_feature_importance(X, y, n_bootstrap, task_type)
            results['bootstrap_analysis'] = bootstrap_results
        
        # Add temporal stability analysis
        if include_temporal and len(X) > n_temporal_windows * 10:  # Ensure enough data
            logger.info("Adding temporal stability analysis...")
            temporal_results = self.temporal_stability_analysis(X, y, n_temporal_windows, task_type)
            results['temporal_stability'] = temporal_results
        
        # Add scaling validation
        logger.info("Adding scaling validation...")
        if hasattr(self.scaler, 'scaling_params'):
            results['scaling_validation'] = self.scaler.scaling_params
        
        logger.info("Robust comprehensive analysis completed.")
        return results