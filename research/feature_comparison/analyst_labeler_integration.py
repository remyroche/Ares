"""
Analyst-Labeler Integration for Feature Relevance Analysis

This module integrates feature relevance analysis with analyst-labeler approaches
for predicting price action and market movements.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import warnings

logger = logging.getLogger(__name__)

class AnalystLabelerIntegration:
    """
    Integrates feature relevance analysis with analyst-labeler approaches
    for price action prediction.
    """
    
    def __init__(self, price_threshold: float = 0.001, 
                 lookforward_periods: int = 1):
        """
        Initialize analyst-labeler integration.
        
        Args:
            price_threshold: Minimum price movement to consider significant
            lookforward_periods: Number of periods to look forward for labeling
        """
        self.price_threshold = price_threshold
        self.lookforward_periods = lookforward_periods
    
    def create_price_action_labels(self, prices: pd.Series, 
                                 method: str = 'directional') -> pd.Series:
        """
        Create price action labels for analyst-labeler approach.
        
        Args:
            prices: Price series
            method: Labeling method ('directional', 'magnitude')
            
        Returns:
            Price action labels
        """
        if method == 'directional':
            return self._create_directional_labels(prices)
        elif method == 'magnitude':
            return self._create_magnitude_labels(prices)
        else:
            raise ValueError(f"Unknown labeling method: {method}. Use 'directional' or 'magnitude'")
    
    def _create_directional_labels(self, prices: pd.Series) -> pd.Series:
        """Create directional price movement labels."""
        returns = prices.pct_change()
        
        # Forward-looking labels (predict future direction)
        future_returns = returns.shift(-self.lookforward_periods)
        
        # Create binary labels
        labels = pd.Series(index=prices.index, dtype='category')
        labels[future_returns > self.price_threshold] = 'up'
        labels[future_returns < -self.price_threshold] = 'down'
        labels[(future_returns >= -self.price_threshold) & 
               (future_returns <= self.price_threshold)] = 'sideways'
        
        return labels
    
    def _create_magnitude_labels(self, prices: pd.Series) -> pd.Series:
        """Create magnitude-based price movement labels."""
        returns = prices.pct_change()
        future_returns = returns.shift(-self.lookforward_periods)
        
        # Create magnitude categories
        labels = pd.Series(index=prices.index, dtype='category')
        labels[future_returns > 0.01] = 'large_up'      # > 1%
        labels[(future_returns > 0.002) & (future_returns <= 0.01)] = 'small_up'    # 0.2% - 1%
        labels[(future_returns >= -0.002) & (future_returns <= 0.002)] = 'sideways'  # -0.2% to 0.2%
        labels[(future_returns >= -0.01) & (future_returns < -0.002)] = 'small_down' # -1% to -0.2%
        labels[future_returns < -0.01] = 'large_down'   # < -1%
        
        return labels
    
    
    def create_analyst_style_targets(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create analyst-style targets for feature relevance analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of different target types
        """
        targets = {}
        
        # Price-based targets
        if 'close' in data.columns:
            targets['price_direction'] = self.create_price_action_labels(
                data['close'], method='directional'
            )
            targets['price_magnitude'] = self.create_price_action_labels(
                data['close'], method='magnitude'
            )
        
        # VWAP-based targets
        if 'vwap_w20' in data.columns:
            targets['vwap_direction'] = self.create_price_action_labels(
                data['vwap_w20'], method='directional'
            )
        
        # Volume-based targets
        if 'volume' in data.columns:
            targets['volume_direction'] = self.create_price_action_labels(
                data['volume'], method='directional'
            )
        
        return targets
    
    def evaluate_feature_relevance_for_targets(self, X: pd.DataFrame, 
                                             targets: Dict[str, pd.Series],
                                             methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate feature relevance for different analyst-style targets.
        
        Args:
            X: Feature matrix
            targets: Dictionary of target variables
            methods: List of methods to use ('lgbm', 'lasso', 'mi', 'permutation')
            
        Returns:
            Feature relevance results for each target
        """
        if methods is None:
            methods = ['lgbm', 'lasso', 'mi', 'permutation']
        
        from .method_settings import MethodSettings
        method_settings = MethodSettings()
        
        results = {}
        
        for target_name, target_series in targets.items():
            logger.info(f"Evaluating features for target: {target_name}")
            
            # Align data
            common_idx = X.index.intersection(target_series.index)
            X_aligned = X.loc[common_idx]
            y_aligned = target_series.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
            X_clean = X_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            if len(X_clean) == 0:
                logger.warning(f"No valid data for target {target_name}")
                continue
            
            target_results = {}
            
            # Determine if this is classification or regression
            is_classification = y_clean.dtype.name == 'category' or y_clean.dtype == 'object'
            task_type = 'classification' if is_classification else 'regression'
            
            # LGBM analysis
            if 'lgbm' in methods:
                try:
                    lgbm_model = method_settings.create_lgbm_model(task_type)
                    lgbm_model.fit(X_clean, y_clean)
                    
                    # Get feature importance
                    if hasattr(lgbm_model, 'feature_importances_'):
                        importance = pd.Series(lgbm_model.feature_importances_, index=X_clean.columns)
                    else:
                        importance = pd.Series(0, index=X_clean.columns)
                    
                    # Get performance metrics
                    y_pred = lgbm_model.predict(X_clean)
                    
                    if is_classification:
                        from sklearn.metrics import accuracy_score, f1_score
                        performance = {
                            'accuracy': accuracy_score(y_clean, y_pred),
                            'f1_score': f1_score(y_clean, y_pred, average='weighted')
                        }
                    else:
                        from sklearn.metrics import r2_score, mean_squared_error
                        performance = {
                            'r2': r2_score(y_clean, y_pred),
                            'mse': mean_squared_error(y_clean, y_pred)
                        }
                    
                    target_results['lgbm'] = {
                        'feature_importance': importance,
                        'performance': performance,
                        'model': lgbm_model
                    }
                    
                except Exception as e:
                    logger.warning(f"LGBM analysis failed for {target_name}: {e}")
                    target_results['lgbm'] = {'error': str(e)}
            
            # LASSO analysis
            if 'lasso' in methods:
                try:
                    lasso_model = method_settings.create_lasso_model()
                    lasso_model.fit(X_clean, y_clean)
                    
                    # Get feature importance (absolute coefficients)
                    importance = pd.Series(np.abs(lasso_model.coef_), index=X_clean.columns)
                    
                    # Get performance
                    y_pred = lasso_model.predict(X_clean)
                    
                    if is_classification:
                        from sklearn.metrics import accuracy_score, f1_score
                        performance = {
                            'accuracy': accuracy_score(y_clean, y_pred),
                            'f1_score': f1_score(y_clean, y_pred, average='weighted')
                        }
                    else:
                        from sklearn.metrics import r2_score, mean_squared_error
                        performance = {
                            'r2': r2_score(y_clean, y_pred),
                            'mse': mean_squared_error(y_clean, y_pred)
                        }
                    
                    target_results['lasso'] = {
                        'feature_importance': importance,
                        'performance': performance,
                        'model': lasso_model
                    }
                    
                except Exception as e:
                    logger.warning(f"LASSO analysis failed for {target_name}: {e}")
                    target_results['lasso'] = {'error': str(e)}
            
            # Mutual Information analysis
            if 'mi' in methods:
                try:
                    # For classification, use mutual_info_classif
                    if is_classification:
                        from sklearn.feature_selection import mutual_info_classif
                        mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
                    else:
                        from sklearn.feature_selection import mutual_info_regression
                        mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
                    
                    importance = pd.Series(mi_scores, index=X_clean.columns)
                    
                    target_results['mi'] = {
                        'feature_importance': importance,
                        'performance': None  # MI doesn't provide performance metrics
                    }
                    
                except Exception as e:
                    logger.warning(f"MI analysis failed for {target_name}: {e}")
                    target_results['mi'] = {'error': str(e)}
            
            # Permutation Importance analysis
            if 'permutation' in methods:
                try:
                    # Use LGBM model for permutation importance
                    if 'lgbm' in target_results and 'model' in target_results['lgbm']:
                        model = target_results['lgbm']['model']
                    else:
                        model = method_settings.create_lgbm_model(task_type)
                        model.fit(X_clean, y_clean)
                    
                    importance = method_settings.calculate_permutation_importance(
                        model, X_clean, y_clean
                    )
                    
                    target_results['permutation'] = {
                        'feature_importance': importance,
                        'performance': None  # Permutation importance doesn't provide performance metrics
                    }
                    
                except Exception as e:
                    logger.warning(f"Permutation importance analysis failed for {target_name}: {e}")
                    target_results['permutation'] = {'error': str(e)}
            
            results[target_name] = target_results
        
        return results
    
    def create_analyst_style_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create analyst-style report for feature relevance analysis.
        
        Args:
            results: Feature relevance results for different targets
            
        Returns:
            Analyst-style report
        """
        report = {
            'summary': {},
            'target_analysis': {},
            'feature_rankings': {},
            'method_agreement': {}
        }
        
        # Summary statistics
        report['summary'] = {
            'n_targets': len(results),
            'target_names': list(results.keys()),
            'methods_used': set()
        }
        
        # Analyze each target
        for target_name, target_results in results.items():
            target_analysis = {
                'n_features': 0,
                'best_method': None,
                'best_performance': 0,
                'top_features': []
            }
            
            # Collect method results
            method_results = {}
            for method, method_data in target_results.items():
                if 'error' not in method_data and 'feature_importance' in method_data:
                    method_results[method] = method_data
                    report['summary']['methods_used'].add(method)
            
            if method_results:
                # Find best performing method
                best_method = None
                best_performance = 0
                
                for method, method_data in method_results.items():
                    if 'performance' in method_data and method_data['performance']:
                        perf = method_data['performance']
                        # Use R² for regression, F1 for classification
                        if 'r2' in perf:
                            performance_score = perf['r2']
                        elif 'f1_score' in perf:
                            performance_score = perf['f1_score']
                        else:
                            performance_score = 0
                        
                        if performance_score > best_performance:
                            best_performance = performance_score
                            best_method = method
                
                target_analysis['best_method'] = best_method
                target_analysis['best_performance'] = best_performance
                
                # Get top features from best method
                if best_method and best_method in method_results:
                    importance = method_results[best_method]['feature_importance']
                    target_analysis['n_features'] = len(importance)
                    target_analysis['top_features'] = importance.nlargest(10).to_dict()
            
            report['target_analysis'][target_name] = target_analysis
        
        # Feature rankings across targets
        all_features = set()
        for target_results in results.values():
            for method_data in target_results.values():
                if 'feature_importance' in method_data:
                    all_features.update(method_data['feature_importance'].index)
        
        feature_rankings = {}
        for feature in all_features:
            rankings = []
            for target_name, target_results in results.items():
                for method, method_data in target_results.items():
                    if 'feature_importance' in method_data and feature in method_data['feature_importance'].index:
                        importance = method_data['feature_importance'][feature]
                        rankings.append(importance)
            
            if rankings:
                feature_rankings[feature] = {
                    'mean_importance': np.mean(rankings),
                    'std_importance': np.std(rankings),
                    'n_observations': len(rankings)
                }
        
        # Sort by mean importance
        feature_rankings = dict(sorted(feature_rankings.items(), 
                                     key=lambda x: x[1]['mean_importance'], 
                                     reverse=True))
        
        report['feature_rankings'] = feature_rankings
        
        # Method agreement analysis
        if len(results) > 1:
            # Calculate correlation between methods across targets
            method_correlations = {}
            methods = list(report['summary']['methods_used'])
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    correlations = []
                    for target_name, target_results in results.items():
                        if (method1 in target_results and method2 in target_results and
                            'feature_importance' in target_results[method1] and
                            'feature_importance' in target_results[method2]):
                            
                            imp1 = target_results[method1]['feature_importance']
                            imp2 = target_results[method2]['feature_importance']
                            
                            # Align features
                            common_features = imp1.index.intersection(imp2.index)
                            if len(common_features) > 1:
                                corr = imp1[common_features].corr(imp2[common_features])
                                if not np.isnan(corr):
                                    correlations.append(corr)
                    
                    if correlations:
                        method_correlations[f"{method1}_vs_{method2}"] = {
                            'mean_correlation': np.mean(correlations),
                            'std_correlation': np.std(correlations),
                            'n_targets': len(correlations)
                        }
            
            report['method_agreement'] = method_correlations
        
        return report
    
    def print_analyst_style_summary(self, report: Dict[str, Any]) -> None:
        """Print analyst-style summary of feature relevance analysis."""
        print("\n" + "="*80)
        print("ANALYST-LABELER FEATURE RELEVANCE ANALYSIS")
        print("="*80)
        
        # Summary
        summary = report['summary']
        print(f"\nAnalysis Summary:")
        print(f"  Targets analyzed: {summary['n_targets']}")
        print(f"  Methods used: {', '.join(summary['methods_used'])}")
        print(f"  Target types: {', '.join(summary['target_names'])}")
        
        # Target analysis
        print(f"\nTarget Analysis:")
        print("-" * 60)
        for target_name, analysis in report['target_analysis'].items():
            print(f"\n{target_name.upper()}:")
            print(f"  Best method: {analysis['best_method']}")
            print(f"  Best performance: {analysis['best_performance']:.4f}")
            print(f"  Features analyzed: {analysis['n_features']}")
            
            if analysis['top_features']:
                print(f"  Top 5 features:")
                for i, (feature, importance) in enumerate(list(analysis['top_features'].items())[:5]):
                    print(f"    {i+1}. {feature}: {importance:.4f}")
        
        # Feature rankings
        print(f"\nOverall Feature Rankings (Top 10):")
        print("-" * 60)
        for i, (feature, metrics) in enumerate(list(report['feature_rankings'].items())[:10]):
            print(f"{i+1:2d}. {feature:20s} | "
                  f"Mean: {metrics['mean_importance']:.4f} | "
                  f"Std: {metrics['std_importance']:.4f} | "
                  f"Obs: {metrics['n_observations']}")
        
        # Method agreement
        if report['method_agreement']:
            print(f"\nMethod Agreement:")
            print("-" * 60)
            for comparison, metrics in report['method_agreement'].items():
                print(f"  {comparison}: {metrics['mean_correlation']:.4f} ± {metrics['std_correlation']:.4f} "
                      f"(n={metrics['n_targets']})")
        
        print("\n" + "="*80)