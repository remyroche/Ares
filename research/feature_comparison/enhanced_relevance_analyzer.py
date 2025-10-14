"""
Enhanced Relevance Analyzer

This module provides comprehensive feature relevance analysis with time-series
safe validation, stability metrics, diagnostics, and standardized method settings.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

# Import our custom modules
from .relevance_analyzer import RelevanceAnalyzer
from .time_series_validation import TimeSeriesValidator
from .stability_metrics import FeatureStabilityAnalyzer
from .diagnostics import FeatureDiagnostics
from .method_settings import MethodSettings
from .robust_scaling import RobustFeatureScaler

logger = logging.getLogger(__name__)

class EnhancedRelevanceAnalyzer:
    """
    Enhanced relevance analyzer with comprehensive validation and diagnostics.
    """
    
    def __init__(self, scaling_method: str = 'robust', random_state: int = 42,
                 enable_diagnostics: bool = True, enable_stability: bool = True):
        """
        Initialize enhanced relevance analyzer.
        
        Args:
            scaling_method: Scaling method for features
            random_state: Random state for reproducibility
            enable_diagnostics: Whether to enable diagnostics
            enable_stability: Whether to enable stability analysis
        """
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.enable_diagnostics = enable_diagnostics
        self.enable_stability = enable_stability
        
        # Initialize components
        self.base_analyzer = RelevanceAnalyzer(scaling_method=scaling_method)
        self.time_series_validator = TimeSeriesValidator(random_state=random_state)
        self.stability_analyzer = FeatureStabilityAnalyzer()
        self.diagnostics = FeatureDiagnostics(random_state=random_state)
        self.method_settings = MethodSettings(random_state=random_state)
        self.scaler = RobustFeatureScaler(method=scaling_method)
        
        # Store results
        self.analysis_results = {}
        self.validation_results = {}
        self.stability_results = {}
        self.diagnostics_results = {}
    
    def comprehensive_analysis(self, X: pd.DataFrame, y: pd.Series,
                             task_type: str = 'regression',
                             groups: Optional[pd.DataFrame] = None,
                             vwap_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive feature relevance analysis.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'regression' or 'classification'
            groups: Group information (timestamps, assets, regimes)
            vwap_cols: List of VWAP column names for diagnostics
            
        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting comprehensive feature relevance analysis...")
        
        # Step 1: Prepare data with robust scaling
        logger.info("Step 1: Preparing data with robust scaling...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Step 2: Run base analysis
        logger.info("Step 2: Running base relevance analysis...")
        base_results = self.base_analyzer.comprehensive_analysis(
            X_scaled_df, y, task_type
        )
        self.analysis_results = base_results
        
        # Step 3: Time-series validation
        logger.info("Step 3: Running time-series validation...")
        validation_results = self._run_time_series_validation(
            X_scaled_df, y, task_type, groups
        )
        self.validation_results = validation_results
        
        # Step 4: Stability analysis
        if self.enable_stability:
            logger.info("Step 4: Running stability analysis...")
            stability_results = self._run_stability_analysis(base_results)
            self.stability_results = stability_results
        
        # Step 5: Diagnostics
        if self.enable_diagnostics:
            logger.info("Step 5: Running diagnostics...")
            diagnostics_results = self._run_diagnostics(
                X_scaled_df, y, groups, vwap_cols
            )
            self.diagnostics_results = diagnostics_results
        
        # Step 6: Compile comprehensive results
        logger.info("Step 6: Compiling comprehensive results...")
        comprehensive_results = self._compile_comprehensive_results(
            base_results, validation_results, stability_results, diagnostics_results
        )
        
        logger.info("Comprehensive analysis completed!")
        return comprehensive_results
    
    def _run_time_series_validation(self, X: pd.DataFrame, y: pd.Series,
                                   task_type: str, groups: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run time-series validation."""
        validation_results = {}
        
        # Get standardized models
        lgbm_model = self.method_settings.create_lgbm_model(task_type)
        lasso_model = self.method_settings.create_lasso_model()
        
        # Test different validation methods
        models_to_test = {
            'lgbm': lgbm_model,
            'lasso': lasso_model
        }
        
        for model_name, model in models_to_test.items():
            try:
                # Run all validation methods
                model_validation = self.time_series_validator.run_all_validations(
                    model, X, y, groups
                )
                validation_results[model_name] = model_validation
                
            except Exception as e:
                logger.warning(f"Time-series validation failed for {model_name}: {e}")
                validation_results[model_name] = {'error': str(e)}
        
        return validation_results
    
    def _run_stability_analysis(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run stability analysis."""
        stability_results = {}
        
        # Extract analysis results for different methods
        method_results = {}
        for method, results in base_results.items():
            if isinstance(results, dict) and 'feature_importance' in results:
                method_results[method] = results
        
        # Calculate stability metrics
        stability_results['rank_consistency'] = self.stability_analyzer.calculate_rank_consistency(method_results)
        stability_results['jaccard_overlap'] = self.stability_analyzer.calculate_jaccard_overlap(method_results)
        
        # Bootstrap stability (if available)
        if 'bootstrap_analysis' in base_results:
            stability_results['bootstrap_stability'] = self.stability_analyzer.calculate_bootstrap_stability(
                base_results['bootstrap_analysis']
            )
        
        # Temporal stability (if available)
        if 'temporal_stability' in base_results:
            stability_results['temporal_drift'] = self.stability_analyzer.calculate_temporal_drift(
                base_results['temporal_stability']
            )
        
        return stability_results
    
    def _run_diagnostics(self, X: pd.DataFrame, y: pd.Series,
                        groups: Optional[pd.DataFrame] = None,
                        vwap_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive diagnostics."""
        diagnostics_results = {}
        
        # Get standardized model for diagnostics
        lgbm_model = self.method_settings.create_lgbm_model('regression')
        
        # Run comprehensive diagnostics
        diagnostics_results = self.diagnostics.run_comprehensive_diagnostics(
            X, y, lgbm_model, 
            timestamp_col='timestamp' if groups is not None and 'timestamp' in groups.columns else None,
            vwap_cols=vwap_cols
        )
        
        return diagnostics_results
    
    def _compile_comprehensive_results(self, base_results: Dict[str, Any],
                                     validation_results: Dict[str, Any],
                                     stability_results: Dict[str, Any],
                                     diagnostics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive results."""
        comprehensive_results = {
            'base_analysis': base_results,
            'time_series_validation': validation_results,
            'stability_analysis': stability_results,
            'diagnostics': diagnostics_results,
            'summary': self._generate_comprehensive_summary(
                base_results, validation_results, stability_results, diagnostics_results
            )
        }
        
        return comprehensive_results
    
    def _generate_comprehensive_summary(self, base_results: Dict[str, Any],
                                       validation_results: Dict[str, Any],
                                       stability_results: Dict[str, Any],
                                       diagnostics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        summary = {
            'analysis_completed': True,
            'timestamp': pd.Timestamp.now().isoformat(),
            'scaling_method': self.scaling_method,
            'random_state': self.random_state,
            'diagnostics_enabled': self.enable_diagnostics,
            'stability_enabled': self.enable_stability
        }
        
        # Base analysis summary
        if 'lgbm_shap' in base_results and 'performance' in base_results['lgbm_shap']:
            summary['best_r2_score'] = base_results['lgbm_shap']['performance'].get('r2', 0)
        
        # Validation summary
        if validation_results:
            summary['validation_methods'] = list(validation_results.keys())
            summary['validation_successful'] = len([v for v in validation_results.values() 
                                                   if 'error' not in v]) > 0
        
        # Stability summary
        if stability_results:
            if 'rank_consistency' in stability_results and 'overall_consistency' in stability_results['rank_consistency']:
                summary['mean_rank_correlation'] = stability_results['rank_consistency']['overall_consistency'].get('mean_spearman_corr', 0)
            
            if 'jaccard_overlap' in stability_results:
                jaccard_scores = [metrics['mean_jaccard'] for metrics in stability_results['jaccard_overlap'].values()]
                summary['mean_jaccard_overlap'] = np.mean(jaccard_scores) if jaccard_scores else 0
        
        # Diagnostics summary
        if diagnostics_results and 'summary' in diagnostics_results:
            diag_summary = diagnostics_results['summary']
            summary['diagnostics_passed'] = diag_summary['passed_tests']
            summary['diagnostics_failed'] = diag_summary['failed_tests']
            summary['diagnostics_warnings'] = diag_summary['warnings']
            summary['critical_issues'] = diag_summary['critical_issues']
        
        return summary
    
    def get_feature_ranking_summary(self) -> pd.DataFrame:
        """Get comprehensive feature ranking summary."""
        if not self.analysis_results:
            return pd.DataFrame()
        
        # Collect rankings from different methods
        rankings = {}
        
        for method, results in self.analysis_results.items():
            if isinstance(results, dict) and 'feature_importance' in results:
                importance = results['feature_importance']
                if isinstance(importance, pd.Series):
                    rankings[method] = importance.rank(ascending=False)
        
        if not rankings:
            return pd.DataFrame()
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame(rankings)
        
        # Calculate average ranking
        ranking_df['avg_rank'] = ranking_df.mean(axis=1)
        ranking_df['rank_std'] = ranking_df.std(axis=1)
        ranking_df['rank_cv'] = ranking_df['rank_std'] / (ranking_df['avg_rank'] + 1e-8)
        
        # Sort by average rank
        ranking_df = ranking_df.sort_values('avg_rank')
        
        return ranking_df
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get comprehensive stability report."""
        if not self.stability_results:
            return {}
        
        report = {
            'stability_metrics': self.stability_results,
            'overall_stability_score': 0.0
        }
        
        # Calculate overall stability score
        stability_scores = []
        
        # Add rank consistency score
        if 'rank_consistency' in self.stability_results and 'overall_consistency' in self.stability_results['rank_consistency']:
            mean_corr = self.stability_results['rank_consistency']['overall_consistency'].get('mean_spearman_corr', 0)
            if not np.isnan(mean_corr):
                stability_scores.append(mean_corr)
        
        # Add Jaccard overlap score
        if 'jaccard_overlap' in self.stability_results:
            jaccard_scores = [metrics['mean_jaccard'] for metrics in self.stability_results['jaccard_overlap'].values()]
            if jaccard_scores:
                stability_scores.append(np.mean(jaccard_scores))
        
        # Add bootstrap stability score
        if 'bootstrap_stability' in self.stability_results:
            for method_stability in self.stability_results['bootstrap_stability'].values():
                if 'overall_stability' in method_stability:
                    stability_scores.append(method_stability['overall_stability'])
        
        if stability_scores:
            report['overall_stability_score'] = np.mean(stability_scores)
        
        return report
    
    def get_diagnostics_report(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics report."""
        if not self.diagnostics_results:
            return {}
        
        return self.diagnostics_results
    
    def print_comprehensive_summary(self) -> None:
        """Print comprehensive analysis summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FEATURE RELEVANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Base analysis summary
        if self.analysis_results:
            print("\nBase Analysis Results:")
            print("-" * 40)
            for method, results in self.analysis_results.items():
                if isinstance(results, dict) and 'performance' in results:
                    perf = results['performance']
                    print(f"{method.upper()}:")
                    for metric, value in perf.items():
                        print(f"  {metric}: {value:.4f}")
        
        # Validation summary
        if self.validation_results:
            print("\nTime-Series Validation Results:")
            print("-" * 40)
            for model_name, validation in self.validation_results.items():
                if 'error' not in validation:
                    print(f"{model_name.upper()}:")
                    for val_type, val_results in validation.items():
                        if 'error' not in val_results and 'mean_scores' in val_results:
                            mean_scores = val_results['mean_scores']
                            print(f"  {val_type}: R² = {mean_scores.get('r2', 0):.4f}")
        
        # Stability summary
        if self.stability_results:
            print("\nStability Analysis Results:")
            print("-" * 40)
            if 'rank_consistency' in self.stability_results and 'overall_consistency' in self.stability_results['rank_consistency']:
                mean_corr = self.stability_results['rank_consistency']['overall_consistency'].get('mean_spearman_corr', 0)
                print(f"Mean Rank Correlation: {mean_corr:.4f}")
            
            if 'jaccard_overlap' in self.stability_results:
                jaccard_scores = [metrics['mean_jaccard'] for metrics in self.stability_results['jaccard_overlap'].values()]
                if jaccard_scores:
                    print(f"Mean Jaccard Overlap: {np.mean(jaccard_scores):.4f}")
        
        # Diagnostics summary
        if self.diagnostics_results and 'summary' in self.diagnostics_results:
            diag_summary = self.diagnostics_results['summary']
            print("\nDiagnostics Results:")
            print("-" * 40)
            print(f"Tests Passed: {diag_summary['passed_tests']}")
            print(f"Tests Failed: {diag_summary['failed_tests']}")
            print(f"Warnings: {diag_summary['warnings']}")
            
            if diag_summary['critical_issues']:
                print("\nCritical Issues:")
                for issue in diag_summary['critical_issues']:
                    print(f"  - {issue}")
        
        print("\n" + "="*80)