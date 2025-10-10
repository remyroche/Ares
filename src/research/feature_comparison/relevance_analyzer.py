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
from sklearn.model_selection import train_test_split
import warnings

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
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the relevance analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series, 
                                                    pd.DataFrame, pd.Series]:
        """
        Prepare data for analysis by splitting and scaling.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test set size
            
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
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
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