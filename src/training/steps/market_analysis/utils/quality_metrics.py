"""Quality Metrics Calculation Module for Step 7 Enhanced Matrix Operations.

This module provides comprehensive quality metrics calculation and reporting
for feature matrices and matrix operations results.
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd

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


class QualityMetricsCalculator:
    """Quality metrics calculation for feature matrices."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def calculate_quality_metrics(self, df: pd.DataFrame, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for the feature matrix."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return {'error': 'NumPy or Pandas not available'}
        
        try:
            self.logger.info('📊 Calculating quality metrics...')
            numeric_df = df.select_dtypes(include=[np.number])
            quality_metrics = {}
            
            # Completeness metrics
            quality_metrics['completeness'] = {
                'total_cells': numeric_df.size,
                'missing_cells': numeric_df.isnull().sum().sum(),
                'missing_ratio': float(numeric_df.isnull().sum().sum() / numeric_df.size),
                'complete_rows': int(numeric_df.dropna().shape[0]),
                'complete_columns': int(numeric_df.dropna(axis=1).shape[1])
            }
            
            # Variance metrics
            variances = numeric_df.var()
            quality_metrics['variance'] = {
                'mean_variance': float(variances.mean()),
                'median_variance': float(variances.median()),
                'min_variance': float(variances.min()),
                'max_variance': float(variances.max()),
                'low_variance_features': int((variances < 1e-06).sum()),
                'zero_variance_features': int((variances == 0).sum())
            }
            
            # Correlation metrics
            if 'correlation_analysis' in matrix_results:
                corr_matrix = pd.DataFrame(matrix_results['correlation_analysis']['correlation_matrix'])
                high_corrs = matrix_results['correlation_analysis']['high_correlations']
                quality_metrics['correlation'] = {
                    'mean_correlation': float(corr_matrix.abs().mean().mean()),
                    'max_correlation': float(corr_matrix.abs().max().max()),
                    'high_correlation_pairs': len(high_corrs),
                    'correlation_threshold': 0.8
                }
            
            # Numerical stability metrics
            if 'condition_number_check' in matrix_results:
                quality_metrics['numerical_stability'] = {
                    'condition_number': matrix_results['condition_number_check']['condition_number'],
                    'is_well_conditioned': matrix_results['condition_number_check']['is_well_conditioned'],
                    'condition_threshold': 1000000000000.0
                }
            
            # Dimensionality metrics
            if 'matrix_rank_analysis' in matrix_results:
                quality_metrics['dimensionality'] = {
                    'matrix_rank': matrix_results['matrix_rank_analysis']['rank'],
                    'full_rank': matrix_results['matrix_rank_analysis']['full_rank'],
                    'rank_deficiency': matrix_results['matrix_rank_analysis']['rank_deficiency'],
                    'effective_dimensions': matrix_results['matrix_rank_analysis']['rank']
                }
            
            # Distribution metrics
            quality_metrics['distribution'] = {
                'skewness_mean': float(numeric_df.skew().mean()),
                'skewness_std': float(numeric_df.skew().std()),
                'kurtosis_mean': float(numeric_df.kurtosis().mean()),
                'kurtosis_std': float(numeric_df.kurtosis().std()),
                'high_skew_features': int((abs(numeric_df.skew()) > 3).sum()),
                'high_kurtosis_features': int((numeric_df.kurtosis() > 10).sum())
            }
            
            # Outlier metrics
            quality_metrics['outliers'] = self._calculate_outlier_metrics(numeric_df)
            
            # Memory metrics
            quality_metrics['memory'] = {
                'memory_usage_mb': float(numeric_df.memory_usage(deep=True).sum() / 1024 / 1024),
                'memory_per_feature_kb': float(numeric_df.memory_usage(deep=True).sum() / len(numeric_df.columns) / 1024),
                'data_types': numeric_df.dtypes.value_counts().to_dict()
            }
            
            # Overall quality score
            quality_metrics['overall_score'] = self._calculate_overall_quality_score(quality_metrics)
            
            self.logger.info(f"✅ Quality metrics calculated. Overall score: {quality_metrics['overall_score']:.2f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f'❌ Error calculating quality metrics: {str(e)}')
            return {'error': str(e)}
    
    def _calculate_outlier_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate outlier metrics for features."""
        outlier_metrics = {}
        try:
            outlier_counts = []
            outlier_ratios = []
            
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts.append(outliers)
                outlier_ratios.append(outliers / len(df))
            
            outlier_metrics = {
                'total_outliers': sum(outlier_counts),
                'mean_outliers_per_feature': float(np.mean(outlier_counts)),
                'max_outliers_in_feature': max(outlier_counts),
                'mean_outlier_ratio': float(np.mean(outlier_ratios)),
                'high_outlier_features': int(sum(1 for ratio in outlier_ratios if ratio > 0.1))
            }
        except Exception as e:
            outlier_metrics = {'error': str(e)}
        
        return outlier_metrics
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            score = 0.0
            max_score = 0.0
            
            # Completeness score (25 points)
            completeness = quality_metrics.get('completeness', {})
            if 'missing_ratio' in completeness:
                completeness_score = max(0, 25 * (1 - completeness['missing_ratio']))
                score += completeness_score
                max_score += 25
            
            # Variance score (20 points)
            variance = quality_metrics.get('variance', {})
            if 'zero_variance_features' in variance:
                zero_var_ratio = variance['zero_variance_features'] / len(quality_metrics.get('completeness', {}).get('total_cells', 1))
                variance_score = max(0, 20 * (1 - zero_var_ratio))
                score += variance_score
                max_score += 20
            
            # Correlation score (20 points)
            correlation = quality_metrics.get('correlation', {})
            if 'high_correlation_pairs' in correlation:
                corr_score = max(0, 20 * (1 - correlation['high_correlation_pairs'] / 100))
                score += corr_score
                max_score += 20
            
            # Stability score (15 points)
            stability = quality_metrics.get('numerical_stability', {})
            if 'is_well_conditioned' in stability:
                stability_score = 15 if stability['is_well_conditioned'] else 5
                score += stability_score
                max_score += 15
            
            # Dimensionality score (10 points)
            dimensionality = quality_metrics.get('dimensionality', {})
            if 'rank_deficiency' in dimensionality:
                rank_score = max(0, 10 * (1 - dimensionality['rank_deficiency'] / 100))
                score += rank_score
                max_score += 10
            
            # Distribution score (10 points)
            distribution = quality_metrics.get('distribution', {})
            if 'high_skew_features' in distribution:
                skew_penalty = min(10, distribution['high_skew_features'] / 10)
                distribution_score = max(0, 10 - skew_penalty)
                score += distribution_score
                max_score += 10
            
            return score / max_score if max_score > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f'Error calculating overall quality score: {str(e)}')
            return 0.0
    
    def _generate_report_header(self, overall_score: float) -> List[str]:
        """Generate report header and overall score section."""
        report = []
        report.append('=' * 80)
        report.append('📊 DETAILED FEATURE MATRIX QUALITY REPORT')
        report.append('=' * 80)
        
        report.append(f'🎯 OVERALL QUALITY SCORE: {overall_score:.2f}/1.00')
        
        if overall_score >= 0.9:
            report.append('✅ EXCELLENT - Feature matrix is of very high quality')
        elif overall_score >= 0.8:
            report.append('🟢 GOOD - Feature matrix is of good quality with minor issues')
        elif overall_score >= 0.7:
            report.append('🟡 ACCEPTABLE - Feature matrix has some quality issues')
        elif overall_score >= 0.6:
            report.append('🟠 POOR - Feature matrix has significant quality issues')
        else:
            report.append('🔴 CRITICAL - Feature matrix has severe quality issues')
        
        report.append('')
        return report
    
    def _generate_completeness_analysis(self, completeness: Dict[str, Any]) -> List[str]:
        """Generate completeness analysis section."""
        report = []
        report.append('📋 1. DATA COMPLETENESS ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Total cells: {completeness.get('total_cells', 0):,}")
        report.append(f"   Missing cells: {completeness.get('missing_cells', 0):,}")
        report.append(f"   Missing ratio: {completeness.get('missing_ratio', 0):.2%}")
        report.append(f"   Complete rows: {completeness.get('complete_rows', 0):,}")
        report.append(f"   Complete columns: {completeness.get('complete_columns', 0):,}")
        
        if completeness.get('missing_ratio', 0) > 0.05:
            report.append('   ⚠️  RECOMMENDATION: High missing data ratio - consider imputation')
        else:
            report.append('   ✅ Data completeness is acceptable')
        
        report.append('')
        return report
    
    def _generate_variance_analysis(self, variance: Dict[str, Any]) -> List[str]:
        """Generate variance analysis section."""
        report = []
        report.append('📊 2. FEATURE VARIANCE ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Mean variance: {variance.get('mean_variance', 0):.6f}")
        report.append(f"   Median variance: {variance.get('median_variance', 0):.6f}")
        report.append(f"   Min variance: {variance.get('min_variance', 0):.6f}")
        report.append(f"   Max variance: {variance.get('max_variance', 0):.6f}")
        report.append(f"   Low variance features: {variance.get('low_variance_features', 0)}")
        report.append(f"   Zero variance features: {variance.get('zero_variance_features', 0)}")
        
        if variance.get('zero_variance_features', 0) > 0:
            report.append('   ⚠️  RECOMMENDATION: Remove zero-variance features')
        else:
            report.append('   ✅ Feature variance is acceptable')
        
        report.append('')
        return report
    
    def _generate_correlation_analysis(self, correlation: Dict[str, Any]) -> List[str]:
        """Generate correlation analysis section."""
        report = []
        report.append('🔗 3. FEATURE CORRELATION ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Mean correlation: {correlation.get('mean_correlation', 0):.4f}")
        report.append(f"   Max correlation: {correlation.get('max_correlation', 0):.4f}")
        report.append(f"   High correlation pairs: {correlation.get('high_correlation_pairs', 0)}")
        report.append(f"   Correlation threshold: {correlation.get('correlation_threshold', 0.8)}")
        
        if correlation.get('high_correlation_pairs', 0) > 10:
            report.append('   ⚠️  RECOMMENDATION: Many highly correlated features - consider feature selection')
        elif correlation.get('high_correlation_pairs', 0) > 0:
            report.append('   ⚠️  RECOMMENDATION: Some highly correlated features - review for redundancy')
        else:
            report.append('   ✅ Feature correlations are acceptable')
        
        report.append('')
        return report
    
    def _generate_stability_analysis(self, stability: Dict[str, Any]) -> List[str]:
        """Generate numerical stability analysis section."""
        report = []
        report.append('🔢 4. NUMERICAL STABILITY ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Condition number: {stability.get('condition_number', 0):.2e}")
        report.append(f"   Well-conditioned: {stability.get('is_well_conditioned', False)}")
        report.append(f"   Condition threshold: {stability.get('condition_threshold', 1000000000000.0):.2e}")
        
        if not stability.get('is_well_conditioned', False):
            report.append('   ⚠️  RECOMMENDATION: Matrix is ill-conditioned - consider regularization or feature scaling')
        else:
            report.append('   ✅ Numerical stability is good')
        
        report.append('')
        return report
    
    def _generate_dimensionality_analysis(self, dimensionality: Dict[str, Any]) -> List[str]:
        """Generate dimensionality analysis section."""
        report = []
        report.append('📐 5. DIMENSIONALITY ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Matrix rank: {dimensionality.get('matrix_rank', 0)}")
        report.append(f"   Full rank: {dimensionality.get('full_rank', False)}")
        report.append(f"   Rank deficiency: {dimensionality.get('rank_deficiency', 0)}")
        report.append(f"   Effective dimensions: {dimensionality.get('effective_dimensions', 0)}")
        
        if dimensionality.get('rank_deficiency', 0) > 0:
            report.append('   ⚠️  RECOMMENDATION: Rank-deficient matrix - consider dimensionality reduction')
        else:
            report.append('   ✅ Matrix has full rank')
        
        report.append('')
        return report
    
    def _generate_distribution_analysis(self, distribution: Dict[str, Any]) -> List[str]:
        """Generate distribution analysis section."""
        report = []
        report.append('📈 6. FEATURE DISTRIBUTION ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Mean skewness: {distribution.get('skewness_mean', 0):.4f}")
        report.append(f"   Skewness std: {distribution.get('skewness_std', 0):.4f}")
        report.append(f"   Mean kurtosis: {distribution.get('kurtosis_mean', 0):.4f}")
        report.append(f"   Kurtosis std: {distribution.get('kurtosis_std', 0):.4f}")
        report.append(f"   High skew features: {distribution.get('high_skew_features', 0)}")
        report.append(f"   High kurtosis features: {distribution.get('high_kurtosis_features', 0)}")
        
        if distribution.get('high_skew_features', 0) > 10:
            report.append('   ⚠️  RECOMMENDATION: Many skewed features - consider transformations')
        else:
            report.append('   ✅ Feature distributions are generally acceptable')
        
        report.append('')
        return report
    
    def _generate_outlier_analysis(self, outliers: Dict[str, Any]) -> List[str]:
        """Generate outlier analysis section."""
        report = []
        report.append('🎯 7. OUTLIER ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Total outliers: {outliers.get('total_outliers', 0):,}")
        report.append(f"   Mean outliers per feature: {outliers.get('mean_outliers_per_feature', 0):.1f}")
        report.append(f"   Max outliers in feature: {outliers.get('max_outliers_in_feature', 0)}")
        report.append(f"   Mean outlier ratio: {outliers.get('mean_outlier_ratio', 0):.2%}")
        report.append(f"   High outlier features: {outliers.get('high_outlier_features', 0)}")
        
        if outliers.get('high_outlier_features', 0) > 5:
            report.append('   ⚠️  RECOMMENDATION: Many features with high outlier ratios - consider outlier handling')
        else:
            report.append('   ✅ Outlier levels are acceptable')
        
        report.append('')
        return report
    
    def _generate_memory_analysis(self, memory: Dict[str, Any]) -> List[str]:
        """Generate memory usage analysis section."""
        report = []
        report.append('💾 8. MEMORY USAGE ANALYSIS')
        report.append('-' * 40)
        report.append(f"   Total memory usage: {memory.get('memory_usage_mb', 0):.1f} MB")
        report.append(f"   Memory per feature: {memory.get('memory_per_feature_kb', 0):.1f} KB")
        report.append(f"   Data types: {memory.get('data_types', {})}")
        
        if memory.get('memory_usage_mb', 0) > 1000:
            report.append('   ⚠️  RECOMMENDATION: High memory usage - consider data type optimization')
        else:
            report.append('   ✅ Memory usage is reasonable')
        
        report.append('')
        return report
    
    def _generate_sr_analysis(self, matrix_results: Dict[str, Any]) -> List[str]:
        """Generate SR-specific analysis section."""
        report = []
        if not (matrix_results and ('sr_analysis' in matrix_results or 'sr_enhanced_analysis' in matrix_results or 'sr_optimization_analysis' in matrix_results)):
            return report
            
        report.append('🎯 9. SR-SPECIFIC ANALYSIS')
        report.append('-' * 40)
        
        if 'sr_analysis' in matrix_results:
            sr_analysis = matrix_results['sr_analysis']
            if 'sr_feature_count' in sr_analysis:
                report.append(f"   SR Features: {sr_analysis['sr_feature_count']}")
            if 'sr_correlation_analysis' in sr_analysis:
                high_corrs = sr_analysis['sr_correlation_analysis'].get('high_correlations', [])
                report.append(f'   SR High Correlations: {len(high_corrs)}')
        
        if 'sr_enhanced_analysis' in matrix_results:
            enhanced_analysis = matrix_results['sr_enhanced_analysis']
            if 'enhanced_sr_feature_count' in enhanced_analysis:
                report.append(f"   Enhanced SR Features: {enhanced_analysis['enhanced_sr_feature_count']}")
            if 'enhanced_sr_importance_analysis' in enhanced_analysis:
                importance = enhanced_analysis['enhanced_sr_importance_analysis']
                if 'top_features' in importance:
                    report.append(f"   Top Enhanced SR Features: {len(importance['top_features'])}")
        
        if 'sr_optimization_analysis' in matrix_results:
            opt_analysis = matrix_results['sr_optimization_analysis']
            if 'optimization_feature_count' in opt_analysis:
                report.append(f"   SR Optimization Features: {opt_analysis['optimization_feature_count']}")
        
        report.append('')
        return report
    
    def _generate_recommendations(self, quality_metrics: Dict[str, Any], matrix_results: Dict[str, Any] = None) -> List[str]:
        """Generate actionable recommendations section."""
        report = []
        report.append('🚀 10. ACTIONABLE RECOMMENDATIONS')
        report.append('-' * 40)
        recommendations = []
        
        completeness = quality_metrics.get('completeness', {})
        variance = quality_metrics.get('variance', {})
        correlation = quality_metrics.get('correlation', {})
        stability = quality_metrics.get('numerical_stability', {})
        dimensionality = quality_metrics.get('dimensionality', {})
        distribution = quality_metrics.get('distribution', {})
        outliers = quality_metrics.get('outliers', {})
        memory = quality_metrics.get('memory', {})
        
        if completeness.get('missing_ratio', 0) > 0.05:
            recommendations.append('• Implement data imputation for missing values')
        if variance.get('zero_variance_features', 0) > 0:
            recommendations.append('• Remove zero-variance features')
        if correlation.get('high_correlation_pairs', 0) > 5:
            recommendations.append('• Apply feature selection to reduce multicollinearity')
        if not stability.get('is_well_conditioned', False):
            recommendations.append('• Apply feature scaling or regularization')
        if dimensionality.get('rank_deficiency', 0) > 0:
            recommendations.append('• Consider PCA or other dimensionality reduction techniques')
        if distribution.get('high_skew_features', 0) > 10:
            recommendations.append('• Apply log or power transformations to skewed features')
        if outliers.get('high_outlier_features', 0) > 5:
            recommendations.append('• Implement outlier detection and handling strategies')
        if memory.get('memory_usage_mb', 0) > 1000:
            recommendations.append('• Optimize data types to reduce memory usage')
        
        if matrix_results and ('sr_analysis' in matrix_results or 'sr_enhanced_analysis' in matrix_results):
            recommendations.append('• Review SR feature correlations and consider feature selection')
            recommendations.append('• Validate SR feature stability across different market conditions')
            recommendations.append('• Consider SR feature importance for model training prioritization')
        
        if not recommendations:
            recommendations.append('• No immediate actions required - feature matrix is in good condition')
        
        for rec in recommendations:
            report.append(f'   {rec}')
        
        report.append('')
        return report
    
    def _generate_summary(self, overall_score: float, matrix_results: Dict[str, Any] = None) -> List[str]:
        """Generate summary section."""
        report = []
        report.append('📋 11. SUMMARY')
        report.append('-' * 40)
        report.append(f'   Overall Quality Score: {overall_score:.2f}/1.00')
        
        if matrix_results and ('sr_analysis' in matrix_results or 'sr_enhanced_analysis' in matrix_results or 'sr_optimization_analysis' in matrix_results):
            report.append('   SR Analysis: ✅ COMPREHENSIVE SR FEATURES ANALYZED')
            total_sr_features = 0
            if 'sr_analysis' in matrix_results:
                total_sr_features += matrix_results['sr_analysis'].get('sr_feature_count', 0)
            if 'sr_enhanced_analysis' in matrix_results:
                total_sr_features += matrix_results['sr_enhanced_analysis'].get('enhanced_sr_feature_count', 0)
            if 'sr_optimization_analysis' in matrix_results:
                total_sr_features += matrix_results['sr_optimization_analysis'].get('optimization_feature_count', 0)
            report.append(f'   Total SR Features: {total_sr_features}')
            
            if 'sr_optimization_analysis' in matrix_results:
                opt_analysis = matrix_results['sr_optimization_analysis']
                if 'sr_optimization_performance_analysis' in opt_analysis:
                    perf_score = opt_analysis['sr_optimization_performance_analysis'].get('overall_performance_score', 0)
                    if perf_score >= 0.7:
                        report.append('   SR Optimization: ✅ HIGH PERFORMANCE')
                    elif perf_score >= 0.5:
                        report.append('   SR Optimization: ⚠️  MODERATE PERFORMANCE')
                    else:
                        report.append('   SR Optimization: 🔴 LOW PERFORMANCE')
        else:
            report.append('   SR Analysis: ⚠️  NO SR FEATURES DETECTED')
        
        if overall_score >= 0.8:
            report.append('   Status: ✅ READY FOR MODEL TRAINING')
        elif overall_score >= 0.6:
            report.append('   Status: ⚠️  NEEDS IMPROVEMENT BEFORE TRAINING')
        else:
            report.append('   Status: 🔴 REQUIRES SIGNIFICANT IMPROVEMENT')
        
        report.append('=' * 80)
        return report
    
    def generate_detailed_quality_report(self, quality_metrics: Dict[str, Any], matrix_results: Dict[str, Any] = None) -> str:
        """Generate detailed quality report with recommendations."""
        try:
            overall_score = quality_metrics.get('overall_score', 0.0)
            
            # Generate all report sections
            report = []
            report.extend(self._generate_report_header(overall_score))
            report.extend(self._generate_completeness_analysis(quality_metrics.get('completeness', {})))
            report.extend(self._generate_variance_analysis(quality_metrics.get('variance', {})))
            report.extend(self._generate_correlation_analysis(quality_metrics.get('correlation', {})))
            report.extend(self._generate_stability_analysis(quality_metrics.get('numerical_stability', {})))
            report.extend(self._generate_dimensionality_analysis(quality_metrics.get('dimensionality', {})))
            report.extend(self._generate_distribution_analysis(quality_metrics.get('distribution', {})))
            report.extend(self._generate_outlier_analysis(quality_metrics.get('outliers', {})))
            report.extend(self._generate_memory_analysis(quality_metrics.get('memory', {})))
            report.extend(self._generate_sr_analysis(matrix_results))
            report.extend(self._generate_recommendations(quality_metrics, matrix_results))
            report.extend(self._generate_summary(overall_score, matrix_results))
            
            return '\n'.join(report)
            
        except Exception as e:
            self.logger.error(f'Error generating detailed quality report: {str(e)}')
            return f'Error generating report: {str(e)}'


__all__ = ['QualityMetricsCalculator']