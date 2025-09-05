"""Matrix Operations Module for Step 7 Enhanced Matrix Operations.

This module provides comprehensive matrix operations including standard operations,
SR-specific analysis, and enhanced SR analysis.
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import logging
import time

# Optional dependencies with fallback handling
try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class MatrixOperations:
    """Matrix operations for enhanced analysis."""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def execute_standard_matrix_operations(self, numeric_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute standard matrix operations."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return {'error': 'NumPy or Pandas not available'}
        
        results = {}
        self.logger.info('📊 Performing correlation analysis...')
        correlation_matrix = numeric_df.corr()
        results['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': self._find_high_correlations(correlation_matrix, config['correlation_threshold'])
        }
        
        self.logger.info('🔍 Checking condition number...')
        condition_number = np.linalg.cond(numeric_df.values)
        results['condition_number_check'] = {
            'condition_number': float(condition_number),
            'is_well_conditioned': condition_number < config['condition_number_threshold']
        }
        
        self.logger.info('📈 Performing eigenvalue analysis...')
        eigenvalues = np.linalg.eigvals(numeric_df.values)
        results['eigenvalue_analysis'] = {
            'eigenvalues': eigenvalues.tolist(),
            'min_eigenvalue': float(np.min(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues)),
            'eigenvalue_ratio': float(np.max(eigenvalues) / np.min(eigenvalues)),
            'small_eigenvalues': int(np.sum(np.abs(eigenvalues) < config['min_eigenvalue_threshold']))
        }
        
        self.logger.info('🔧 Performing SVD analysis...')
        try:
            U, s, Vt = np.linalg.svd(numeric_df.values, full_matrices=False)
            results['singular_value_decomposition'] = {
                'singular_values': s.tolist(),
                'rank': int(np.sum(s > config['min_eigenvalue_threshold'])),
                'condition_number_svd': float(s[0] / s[-1]) if len(s) > 1 else float('inf')
            }
        except Exception as e:
            self.logger.warning(f'⚠️ SVD failed: {str(e)}')
            results['singular_value_decomposition'] = {'error': str(e)}
        
        self.logger.info('📊 Analyzing matrix rank...')
        try:
            rank = np.linalg.matrix_rank(numeric_df.values)
            results['matrix_rank_analysis'] = {
                'rank': int(rank),
                'full_rank': rank == min(numeric_df.shape),
                'rank_deficiency': min(numeric_df.shape) - rank
            }
        except Exception as e:
            self.logger.warning(f'⚠️ Rank analysis failed: {str(e)}')
            results['matrix_rank_analysis'] = {'error': str(e)}
        
        return results
    
    async def execute_sr_matrix_operations(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SR-specific matrix operations."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return {'error': 'NumPy or Pandas not available'}
        
        try:
            sr_features = config.get('sr_features', [])
            if not sr_features:
                return {'error': 'No SR features found'}
            
            sr_df = df[sr_features].select_dtypes(include=[np.number])
            if len(sr_df.columns) == 0:
                return {'error': 'No numeric SR features found'}
            
            self.logger.info(f'🎯 Analyzing {len(sr_df.columns)} SR features')
            results = {}
            
            self.logger.info('📊 Performing SR feature correlation analysis...')
            sr_correlation_matrix = sr_df.corr()
            results['sr_correlation_analysis'] = {
                'correlation_matrix': sr_correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(sr_correlation_matrix, config['sr_correlation_threshold']),
                'sr_feature_count': len(sr_df.columns)
            }
            
            self.logger.info('🔍 Checking SR feature condition number...')
            sr_condition_number = np.linalg.cond(sr_df.values)
            results['sr_condition_number'] = {
                'condition_number': float(sr_condition_number),
                'is_well_conditioned': sr_condition_number < config['sr_condition_number_threshold']
            }
            
            self.logger.info('📈 Performing SR feature eigenvalue analysis...')
            sr_eigenvalues = np.linalg.eigvals(sr_df.values)
            results['sr_eigenvalue_analysis'] = {
                'eigenvalues': sr_eigenvalues.tolist(),
                'min_eigenvalue': float(np.min(sr_eigenvalues)),
                'max_eigenvalue': float(np.max(sr_eigenvalues)),
                'eigenvalue_ratio': float(np.max(sr_eigenvalues) / np.min(sr_eigenvalues)),
                'small_eigenvalues': int(np.sum(np.abs(sr_eigenvalues) < config['min_eigenvalue_threshold']))
            }
            
            self.logger.info('🔧 Performing SR feature clustering analysis...')
            results['sr_clustering_analysis'] = self._analyze_sr_feature_clusters(sr_df)
            
            self.logger.info('📊 Analyzing SR feature stability...')
            results['sr_stability_analysis'] = self._analyze_sr_feature_stability(sr_df)
            
            self.logger.info('🎯 Analyzing SR feature importance...')
            results['sr_importance_analysis'] = self._analyze_sr_feature_importance(sr_df)
            
            return results
        except Exception as e:
            self.logger.error(f'Error in SR matrix operations: {e}')
            return {'error': str(e)}
    
    async def execute_enhanced_sr_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced SR analysis using SR breakout predictor features."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return {'error': 'NumPy or Pandas not available'}
        
        try:
            enhanced_sr_features = [col for col in df.columns if any(
                keyword in col.lower() for keyword in [
                    'sr_enhanced_', 'sr_clusters_', 'sr_fibonacci_', 'sr_elliott_', 
                    'sr_order_flow_', 'sr_pivot_', 'sr_support_1_pct', 'sr_support_2_pct', 
                    'sr_resistance_1_pct', 'sr_resistance_2_pct'
                ]
            )]
            
            if not enhanced_sr_features:
                return {'error': 'No enhanced SR features found'}
            
            enhanced_sr_df = df[enhanced_sr_features].select_dtypes(include=[np.number])
            if len(enhanced_sr_df.columns) == 0:
                return {'error': 'No numeric enhanced SR features found'}
            
            self.logger.info(f'🎯 Analyzing {len(enhanced_sr_df.columns)} enhanced SR features')
            results = {}
            
            self.logger.info('📊 Performing enhanced SR feature correlation analysis...')
            enhanced_correlation_matrix = enhanced_sr_df.corr()
            results['enhanced_sr_correlation_analysis'] = {
                'correlation_matrix': enhanced_correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(enhanced_correlation_matrix, config['sr_correlation_threshold']),
                'enhanced_sr_feature_count': len(enhanced_sr_df.columns)
            }
            
            self.logger.info('🔧 Performing enhanced SR feature clustering analysis...')
            results['enhanced_sr_clustering_analysis'] = self._analyze_enhanced_sr_feature_clusters(enhanced_sr_df)
            
            self.logger.info('📊 Analyzing enhanced SR feature stability...')
            results['enhanced_sr_stability_analysis'] = self._analyze_enhanced_sr_feature_stability(enhanced_sr_df)
            
            self.logger.info('🎯 Analyzing enhanced SR feature importance...')
            results['enhanced_sr_importance_analysis'] = self._analyze_enhanced_sr_feature_importance(enhanced_sr_df)
            
            return results
        except Exception as e:
            self.logger.error(f'Error in enhanced SR analysis: {e}')
            return {'error': str(e)}
    
    async def execute_sr_optimization_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SR optimization analysis using optimization features."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return {'error': 'NumPy or Pandas not available'}
        
        try:
            optimization_features = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['sr_optimized_', 'sr_optimization_']
            )]
            
            if not optimization_features:
                return {'error': 'No SR optimization features found'}
            
            optimization_df = df[optimization_features].select_dtypes(include=[np.number])
            if len(optimization_df.columns) == 0:
                return {'error': 'No numeric SR optimization features found'}
            
            self.logger.info(f'🎯 Analyzing {len(optimization_df.columns)} SR optimization features')
            results = {}
            
            self.logger.info('📊 Performing SR optimization feature correlation analysis...')
            optimization_correlation_matrix = optimization_df.corr()
            results['sr_optimization_correlation_analysis'] = {
                'correlation_matrix': optimization_correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(optimization_correlation_matrix, config['sr_correlation_threshold']),
                'optimization_feature_count': len(optimization_df.columns)
            }
            
            self.logger.info('🔧 Analyzing SR optimization parameters...')
            results['sr_optimization_parameter_analysis'] = self._analyze_sr_optimization_parameters(optimization_df)
            
            return results
        except Exception as e:
            self.logger.error(f'Error in SR optimization analysis: {e}')
            return {'error': str(e)}
    
    def _find_high_correlations(self, correlation_matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Find high correlation pairs."""
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        return high_correlations
    
    def _analyze_sr_feature_clusters(self, sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SR feature clusters."""
        try:
            correlation_matrix = sr_df.corr()
            high_corr_groups = []
            processed_features = set()
            
            for i, feature1 in enumerate(sr_df.columns):
                if feature1 in processed_features:
                    continue
                group = [feature1]
                processed_features.add(feature1)
                for feature2 in sr_df.columns[i + 1:]:
                    if feature2 not in processed_features:
                        corr = abs(correlation_matrix.loc[feature1, feature2])
                        if corr > 0.8:
                            group.append(feature2)
                            processed_features.add(feature2)
                if len(group) > 1:
                    high_corr_groups.append(group)
            
            return {
                'high_correlation_groups': high_corr_groups,
                'group_count': len(high_corr_groups),
                'total_grouped_features': sum(len(group) for group in high_corr_groups)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_sr_feature_stability(self, sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SR feature stability over time."""
        try:
            stability_metrics = {}
            for column in sr_df.columns:
                values = sr_df[column].dropna()
                if len(values) > 1:
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                    range_stability = 1.0 / (1.0 + (values.max() - values.min()))
                    stability_metrics[column] = {
                        'coefficient_of_variation': float(cv),
                        'range_stability': float(range_stability),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
            
            overall_stability = {
                'mean_cv': np.mean([metrics['coefficient_of_variation'] for metrics in stability_metrics.values()]),
                'mean_range_stability': np.mean([metrics['range_stability'] for metrics in stability_metrics.values()]),
                'stable_features': len([cv for cv in [metrics['coefficient_of_variation'] for metrics in stability_metrics.values()] if cv < 0.5]),
                'unstable_features': len([cv for cv in [metrics['coefficient_of_variation'] for metrics in stability_metrics.values()] if cv > 1.0])
            }
            
            return {
                'feature_stability': stability_metrics,
                'overall_stability': overall_stability
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_sr_feature_importance(self, sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SR feature importance based on variance and correlation."""
        try:
            variances = sr_df.var()
            variance_importance = variances.sort_values(ascending=False)
            
            correlation_matrix = sr_df.corr()
            avg_correlations = correlation_matrix.abs().mean()
            correlation_importance = (1.0 / (1.0 + avg_correlations)).sort_values(ascending=False)
            
            combined_importance = (variance_importance + correlation_importance) / 2
            combined_importance = combined_importance.sort_values(ascending=False)
            
            return {
                'variance_importance': variance_importance.to_dict(),
                'correlation_importance': correlation_importance.to_dict(),
                'combined_importance': combined_importance.to_dict(),
                'top_features': combined_importance.head(10).index.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_enhanced_sr_feature_clusters(self, enhanced_sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enhanced SR feature clusters."""
        try:
            feature_groups = {
                'enhanced_strength': [col for col in enhanced_sr_df.columns if 'enhanced_strength' in col],
                'clustering': [col for col in enhanced_sr_df.columns if 'clusters' in col or 'noise' in col],
                'fibonacci': [col for col in enhanced_sr_df.columns if 'fibonacci' in col],
                'elliott': [col for col in enhanced_sr_df.columns if 'elliott' in col],
                'order_flow': [col for col in enhanced_sr_df.columns if 'order_flow' in col],
                'pivot': [col for col in enhanced_sr_df.columns if 'pivot' in col or 'support_1' in col or 'resistance_1' in col]
            }
            
            group_stats = {}
            for group_name, group_features in feature_groups.items():
                if group_features:
                    group_data = enhanced_sr_df[group_features]
                    group_stats[group_name] = {
                        'feature_count': len(group_features),
                        'mean_correlation': group_data.corr().abs().mean().mean(),
                        'mean_variance': group_data.var().mean(),
                        'features': group_features
                    }
            
            return {
                'feature_groups': group_stats,
                'total_groups': len([g for g in group_stats.values() if g['feature_count'] > 0]),
                'group_correlations': self._calculate_group_correlations(enhanced_sr_df, feature_groups)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_enhanced_sr_feature_stability(self, enhanced_sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enhanced SR feature stability."""
        try:
            stability_metrics = {}
            for column in enhanced_sr_df.columns:
                values = enhanced_sr_df[column].dropna()
                if len(values) > 1:
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                    feature_type = 'unknown'
                    if 'enhanced_strength' in column:
                        feature_type = 'enhanced_strength'
                    elif 'clusters' in column or 'noise' in column:
                        feature_type = 'clustering'
                    elif 'fibonacci' in column:
                        feature_type = 'fibonacci'
                    elif 'elliott' in column:
                        feature_type = 'elliott'
                    elif 'order_flow' in column:
                        feature_type = 'order_flow'
                    elif 'pivot' in column or 'support_' in column or 'resistance_' in column:
                        feature_type = 'pivot'
                    elif 'momentum_pct' in column or 'volatility_pct' in column or 'trend_pct' in column:
                        feature_type = 'momentum'
                    
                    stability_metrics[column] = {
                        'coefficient_of_variation': float(cv),
                        'feature_type': feature_type,
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'stability_score': 1.0 / (1.0 + cv) if cv != float('inf') else 0.0
                    }
            
            type_stability = {}
            for metrics in stability_metrics.values():
                feature_type = metrics['feature_type']
                if feature_type not in type_stability:
                    type_stability[feature_type] = []
                type_stability[feature_type].append(metrics['stability_score'])
            
            for feature_type, scores in type_stability.items():
                type_stability[feature_type] = {
                    'average_stability': np.mean(scores),
                    'stability_count': len(scores)
                }
            
            return {
                'feature_stability': stability_metrics,
                'type_stability': type_stability,
                'overall_stability': np.mean([m['stability_score'] for m in stability_metrics.values()])
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_enhanced_sr_feature_importance(self, enhanced_sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enhanced SR feature importance."""
        try:
            variances = enhanced_sr_df.var()
            variance_importance = variances.sort_values(ascending=False)
            
            correlation_matrix = enhanced_sr_df.corr()
            avg_correlations = correlation_matrix.abs().mean()
            correlation_importance = (1.0 / (1.0 + avg_correlations)).sort_values(ascending=False)
            
            combined_importance = (variance_importance + correlation_importance) / 2
            combined_importance = combined_importance.sort_values(ascending=False)
            
            feature_importance_by_type = {
                'enhanced_strength': [], 'clustering': [], 'fibonacci': [], 
                'elliott': [], 'order_flow': [], 'pivot': [], 'momentum': []
            }
            
            for feature, importance in combined_importance.items():
                if 'enhanced_strength' in feature:
                    feature_importance_by_type['enhanced_strength'].append((feature, importance))
                elif 'clusters' in feature or 'noise' in feature:
                    feature_importance_by_type['clustering'].append((feature, importance))
                elif 'fibonacci' in feature:
                    feature_importance_by_type['fibonacci'].append((feature, importance))
                elif 'elliott' in feature:
                    feature_importance_by_type['elliott'].append((feature, importance))
                elif 'order_flow' in feature:
                    feature_importance_by_type['order_flow'].append((feature, importance))
                elif 'pivot' in feature or 'support_' in feature or 'resistance_' in feature:
                    feature_importance_by_type['pivot'].append((feature, importance))
                elif 'momentum_pct' in feature or 'volatility_pct' in feature or 'trend_pct' in feature:
                    feature_importance_by_type['momentum'].append((feature, importance))
            
            for feature_type in feature_importance_by_type:
                feature_importance_by_type[feature_type].sort(key=lambda x: x[1], reverse=True)
            
            return {
                'variance_importance': variance_importance.to_dict(),
                'correlation_importance': correlation_importance.to_dict(),
                'combined_importance': combined_importance.to_dict(),
                'importance_by_type': feature_importance_by_type,
                'top_features': combined_importance.head(10).index.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_sr_optimization_parameters(self, optimization_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SR optimization parameters."""
        try:
            parameter_features = [col for col in optimization_df.columns if 'sr_optimized_' in col and any(
                param in col for param in ['method_weights', 'strength_weights', 'dbscan', 'fibonacci', 'elliott', 'order_flow', 'tf_']
            )]
            
            if not parameter_features:
                return {'error': 'No parameter features found'}
            
            parameter_data = optimization_df[parameter_features]
            parameter_stats = {}
            
            for col in parameter_data.columns:
                values = parameter_data[col].dropna()
                if len(values) > 0:
                    parameter_stats[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }
            
            parameter_groups = {
                'weights': [col for col in parameter_features if 'weights' in col],
                'dbscan': [col for col in parameter_features if 'dbscan' in col],
                'advanced': [col for col in parameter_features if any(adv in col for adv in ['fibonacci', 'elliott', 'order_flow'])],
                'timeframe': [col for col in parameter_features if 'tf_' in col]
            }
            
            return {
                'parameter_features': parameter_features,
                'parameter_statistics': parameter_stats,
                'parameter_groups': parameter_groups,
                'parameter_correlations': parameter_data.corr().to_dict()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_group_correlations(self, df: pd.DataFrame, feature_groups: Dict[str, List]) -> Dict[str, float]:
        """Calculate correlations between feature groups."""
        try:
            group_correlations = {}
            for group1_name, group1_features in feature_groups.items():
                for group2_name, group2_features in feature_groups.items():
                    if group1_name < group2_name and group1_features and group2_features:
                        group1_data = df[group1_features]
                        group2_data = df[group2_features]
                        cross_corr = group1_data.corrwith(group2_data, axis=0)
                        avg_correlation = cross_corr.abs().mean()
                        group_correlations[f'{group1_name}_vs_{group2_name}'] = float(avg_correlation)
            return group_correlations
        except Exception as e:
            return {'error': str(e)}


__all__ = ['MatrixOperations']