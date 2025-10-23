"""
Diagnostics to Catch Pitfalls

This module provides comprehensive diagnostics to identify common pitfalls
in feature engineering and model validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

logger = logging.getLogger(__name__)

class FeatureDiagnostics:
    """
    Comprehensive diagnostics for feature engineering pitfalls.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize feature diagnostics.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def test_target_leakage(self, X: pd.DataFrame, y: pd.Series, 
                           model: Any, test_name: str = "target_leakage") -> Dict[str, Any]:
        """
        Test for target leakage using shuffle labels test.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            test_name: Name of the test
            
        Returns:
            Leakage test results
        """
        logger.info(f"Running {test_name} test...")
        
        # Original performance
        model.fit(X, y)
        y_pred_original = model.predict(X)
        original_r2 = r2_score(y, y_pred_original)
        original_mse = mean_squared_error(y, y_pred_original)
        
        # Shuffled labels performance
        y_shuffled = y.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        model.fit(X, y_shuffled)
        y_pred_shuffled = model.predict(X)
        shuffled_r2 = r2_score(y_shuffled, y_pred_shuffled)
        shuffled_mse = mean_squared_error(y_shuffled, y_pred_shuffled)
        
        # Calculate leakage indicators
        r2_drop = original_r2 - shuffled_r2
        mse_increase = shuffled_mse - original_mse
        
        # Leakage flags
        has_leakage = r2_drop < 0.1 or mse_increase < 0.1  # Suspicious if performance doesn't drop much
        
        results = {
            'test_name': test_name,
            'original_r2': original_r2,
            'shuffled_r2': shuffled_r2,
            'r2_drop': r2_drop,
            'original_mse': original_mse,
            'shuffled_mse': shuffled_mse,
            'mse_increase': mse_increase,
            'has_leakage': has_leakage,
            'leakage_severity': 'high' if r2_drop < 0.05 else 'medium' if r2_drop < 0.1 else 'low'
        }
        
        return results
    
    def test_forward_fill_leakage(self, X: pd.DataFrame, y: pd.Series,
                                 timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Test for forward-fill leakage in time series data.
        
        Args:
            X: Feature matrix
            y: Target vector
            timestamp_col: Name of timestamp column
            
        Returns:
            Forward-fill leakage test results
        """
        logger.info("Running forward-fill leakage test...")
        
        if timestamp_col not in X.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return {'test_name': 'forward_fill_leakage', 'error': 'No timestamp column'}
        
        # Sort by timestamp
        X_sorted = X.sort_values(timestamp_col)
        y_sorted = y.loc[X_sorted.index]
        
        # Check for forward-filled values
        forward_fill_indicators = []
        
        for col in X_sorted.columns:
            if col == timestamp_col:
                continue
            
            # Check for consecutive identical values (potential forward-fill)
            diff = X_sorted[col].diff()
            consecutive_zeros = (diff == 0).sum()
            total_values = len(X_sorted) - 1
            
            if total_values > 0:
                forward_fill_ratio = consecutive_zeros / total_values
                forward_fill_indicators.append({
                    'column': col,
                    'forward_fill_ratio': forward_fill_ratio,
                    'consecutive_zeros': consecutive_zeros,
                    'total_values': total_values
                })
        
        # Calculate overall forward-fill score
        if forward_fill_indicators:
            mean_forward_fill_ratio = np.mean([ind['forward_fill_ratio'] for ind in forward_fill_indicators])
            high_forward_fill_features = [ind for ind in forward_fill_indicators 
                                        if ind['forward_fill_ratio'] > 0.5]
        else:
            mean_forward_fill_ratio = 0
            high_forward_fill_features = []
        
        results = {
            'test_name': 'forward_fill_leakage',
            'mean_forward_fill_ratio': mean_forward_fill_ratio,
            'high_forward_fill_features': len(high_forward_fill_features),
            'forward_fill_indicators': forward_fill_indicators,
            'has_forward_fill_leakage': mean_forward_fill_ratio > 0.3
        }
        
        return results
    
    def test_vwap_window_leakage(self, X: pd.DataFrame, vwap_cols: List[str],
                                timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Test for VWAP window leakage (ensure windows end at t).
        
        Args:
            X: Feature matrix
            vwap_cols: List of VWAP column names
            timestamp_col: Name of timestamp column
            
        Returns:
            VWAP window leakage test results
        """
        logger.info("Running VWAP window leakage test...")
        
        if timestamp_col not in X.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return {'test_name': 'vwap_window_leakage', 'error': 'No timestamp column'}
        
        vwap_issues = []
        
        for vwap_col in vwap_cols:
            if vwap_col not in X.columns:
                continue
            
            # Check for potential future data leakage
            # This is a simplified check - in practice, you'd need to verify
            # that VWAP calculations only use data up to time t
            
            # Check for suspiciously smooth VWAP values (might indicate future data)
            vwap_values = X[vwap_col].dropna()
            if len(vwap_values) > 1:
                # Calculate rolling correlation with future values (should be low)
                vwap_shifted = vwap_values.shift(1)
                correlation_with_past = vwap_values.corr(vwap_shifted)
                
                # Check for sudden jumps (might indicate window boundary issues)
                vwap_diff = vwap_values.diff()
                jump_threshold = vwap_diff.std() * 3
                large_jumps = (vwap_diff.abs() > jump_threshold).sum()
                
                vwap_issues.append({
                    'column': vwap_col,
                    'correlation_with_past': correlation_with_past,
                    'large_jumps': large_jumps,
                    'jump_threshold': jump_threshold
                })
        
        results = {
            'test_name': 'vwap_window_leakage',
            'vwap_issues': vwap_issues,
            'has_vwap_leakage': len([issue for issue in vwap_issues 
                                   if issue['correlation_with_past'] > 0.9]) > 0
        }
        
        return results
    
    def test_scaling_sensitivity(self, X: pd.DataFrame, y: pd.Series, 
                                model: Any) -> Dict[str, Any]:
        """
        Test sensitivity to different scaling methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            
        Returns:
            Scaling sensitivity test results
        """
        logger.info("Running scaling sensitivity test...")
        
        # Test different scaling methods
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaling_results = {}
        
        for scaler_name, scaler in scalers.items():
            try:
                # Scale features
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                
                # Train model
                model.fit(X_scaled_df, y)
                y_pred = model.predict(X_scaled_df)
                
                # Calculate performance
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                
                scaling_results[scaler_name] = {
                    'r2': r2,
                    'mse': mse,
                    'scaler_params': scaler.get_params()
                }
                
            except Exception as e:
                logger.warning(f"Scaling test failed for {scaler_name}: {e}")
                scaling_results[scaler_name] = {'error': str(e)}
        
        # Calculate sensitivity metrics
        r2_scores = [result['r2'] for result in scaling_results.values() 
                    if 'r2' in result and not isinstance(result['r2'], str)]
        
        if len(r2_scores) > 1:
            r2_std = np.std(r2_scores)
            r2_range = np.max(r2_scores) - np.min(r2_scores)
            sensitivity_score = r2_std / (np.mean(r2_scores) + 1e-8)
        else:
            r2_std = 0
            r2_range = 0
            sensitivity_score = 0
        
        results = {
            'test_name': 'scaling_sensitivity',
            'scaling_results': scaling_results,
            'r2_std': r2_std,
            'r2_range': r2_range,
            'sensitivity_score': sensitivity_score,
            'is_sensitive': sensitivity_score > 0.1
        }
        
        return results
    
    def test_collinearity_after_pruning(self, X: pd.DataFrame, 
                                       feature_selector: Any = None) -> Dict[str, Any]:
        """
        Test for collinearity after feature selection.
        
        Args:
            X: Feature matrix
            feature_selector: Feature selector to use
            
        Returns:
            Collinearity test results
        """
        logger.info("Running collinearity after pruning test...")
        
        # Select features if selector provided
        if feature_selector is not None:
            try:
                X_selected = feature_selector.fit_transform(X)
                if hasattr(feature_selector, 'get_support'):
                    selected_features = X.columns[feature_selector.get_support()].tolist()
                else:
                    selected_features = X.columns.tolist()
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")
                X_selected = X
                selected_features = X.columns.tolist()
        else:
            X_selected = X
            selected_features = X.columns.tolist()
        
        # Calculate VIF for selected features
        vif_results = []
        
        for i, feature in enumerate(selected_features):
            try:
                vif = variance_inflation_factor(X_selected.values, i)
                vif_results.append({
                    'feature': feature,
                    'vif': vif
                })
            except:
                vif_results.append({
                    'feature': feature,
                    'vif': np.inf
                })
        
        # Calculate collinearity metrics
        vif_values = [result['vif'] for result in vif_results if not np.isinf(result['vif'])]
        
        if vif_values:
            mean_vif = np.mean(vif_values)
            max_vif = np.max(vif_values)
            high_vif_count = len([vif for vif in vif_values if vif > 10])
            very_high_vif_count = len([vif for vif in vif_values if vif > 20])
        else:
            mean_vif = 0
            max_vif = 0
            high_vif_count = 0
            very_high_vif_count = 0
        
        results = {
            'test_name': 'collinearity_after_pruning',
            'vif_results': vif_results,
            'mean_vif': mean_vif,
            'max_vif': max_vif,
            'high_vif_count': high_vif_count,
            'very_high_vif_count': very_high_vif_count,
            'has_collinearity': high_vif_count > 0
        }
        
        return results
    
    def test_shadow_features(self, X: pd.DataFrame, y: pd.Series, 
                            model: Any, n_shadow: int = 10) -> Dict[str, Any]:
        """
        Test using shadow features to identify suspicious features.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            n_shadow: Number of shadow features to generate
            
        Returns:
            Shadow feature test results
        """
        logger.info("Running shadow feature test...")
        
        # Generate shadow features (randomized versions of real features)
        shadow_features = {}
        for i in range(n_shadow):
            # Randomly shuffle each real feature
            for col in X.columns:
                shadow_col = f"shadow_{col}_{i}"
                shadow_features[shadow_col] = X[col].sample(frac=1, random_state=self.random_state + i).values
        
        # Create dataset with real and shadow features
        X_with_shadow = X.copy()
        for shadow_col, shadow_values in shadow_features.items():
            X_with_shadow[shadow_col] = shadow_values
        
        # Train model and get feature importance
        model.fit(X_with_shadow, y)
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=X_with_shadow.columns)
        elif hasattr(model, 'coef_'):
            importance = pd.Series(np.abs(model.coef_), index=X_with_shadow.columns)
        else:
            logger.warning("Model does not support feature importance")
            return {'test_name': 'shadow_features', 'error': 'No feature importance support'}
        
        # Separate real and shadow features
        real_features = [col for col in X.columns if col in importance.index]
        shadow_features_list = [col for col in importance.index if col.startswith('shadow_')]
        
        real_importance = importance[real_features]
        shadow_importance = importance[shadow_features_list]
        
        # Calculate shadow feature statistics
        shadow_mean = shadow_importance.mean()
        shadow_std = shadow_importance.std()
        shadow_threshold = shadow_mean + shadow_std
        
        # Identify suspicious features (worse than shadow features)
        suspicious_features = real_importance[real_importance < shadow_threshold].index.tolist()
        
        # Calculate feature quality scores
        feature_quality = {}
        for feature in real_features:
            feature_imp = real_importance[feature]
            quality_score = (feature_imp - shadow_mean) / (shadow_std + 1e-8)
            feature_quality[feature] = {
                'importance': feature_imp,
                'quality_score': quality_score,
                'above_shadow': feature_imp > shadow_threshold
            }
        
        results = {
            'test_name': 'shadow_features',
            'shadow_mean': shadow_mean,
            'shadow_std': shadow_std,
            'shadow_threshold': shadow_threshold,
            'suspicious_features': suspicious_features,
            'feature_quality': feature_quality,
            'n_suspicious': len(suspicious_features),
            'n_real_features': len(real_features)
        }
        
        return results
    
    def run_comprehensive_diagnostics(self, X: pd.DataFrame, y: pd.Series, 
                                    model: Any, timestamp_col: str = 'timestamp',
                                    vwap_cols: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics suite.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            timestamp_col: Name of timestamp column
            vwap_cols: List of VWAP column names
            
        Returns:
            Comprehensive diagnostics results
        """
        logger.info("Running comprehensive diagnostics suite...")
        
        diagnostics = {}
        
        # Target leakage tests
        diagnostics['target_leakage'] = self.test_target_leakage(X, y, model)
        
        # Forward-fill leakage test
        diagnostics['forward_fill_leakage'] = self.test_forward_fill_leakage(X, y, timestamp_col)
        
        # VWAP window leakage test
        if vwap_cols:
            diagnostics['vwap_window_leakage'] = self.test_vwap_window_leakage(X, vwap_cols, timestamp_col)
        
        # Scaling sensitivity test
        diagnostics['scaling_sensitivity'] = self.test_scaling_sensitivity(X, y, model)
        
        # Collinearity after pruning test
        diagnostics['collinearity_after_pruning'] = self.test_collinearity_after_pruning(X)
        
        # Shadow features test
        diagnostics['shadow_features'] = self.test_shadow_features(X, y, model)
        
        # Overall diagnostics summary
        diagnostics['summary'] = self._generate_diagnostics_summary(diagnostics)
        
        return diagnostics
    
    def _generate_diagnostics_summary(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate diagnostics summary."""
        summary = {
            'total_tests': len(diagnostics) - 1,  # Exclude summary itself
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0,
            'critical_issues': []
        }
        
        for test_name, test_results in diagnostics.items():
            if test_name == 'summary':
                continue
            
            if 'error' in test_results:
                summary['failed_tests'] += 1
                summary['critical_issues'].append(f"{test_name}: {test_results['error']}")
            elif 'has_leakage' in test_results and test_results['has_leakage']:
                summary['failed_tests'] += 1
                summary['critical_issues'].append(f"{test_name}: Potential leakage detected")
            elif 'is_sensitive' in test_results and test_results['is_sensitive']:
                summary['warnings'] += 1
            elif 'has_collinearity' in test_results and test_results['has_collinearity']:
                summary['warnings'] += 1
            else:
                summary['passed_tests'] += 1
        
        return summary