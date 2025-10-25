"""
Feature Generation Final Validation Step

This step performs comprehensive validation of the final feature datasets to ensure they are
ready for model training. It validates data quality, feature distributions, target relationships,
and generates final validation reports.

Features:
- Data quality validation (NaN, outliers, distributions)
- Feature-target correlation analysis
- Cross-validation readiness assessment
- Statistical validation of feature sets
- Final dataset integrity checks
- Comprehensive validation reporting
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path
import json
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Import BaseStep and step registry
from src.training.steps.base_step import BaseStep

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


class FeatureGenerationFinalValidationStep(BaseStep):
    """
    Final validation step for the feature generation pipeline.

    This step validates the final feature datasets and ensures they are suitable
    for model training with comprehensive quality checks.
    """

    def __init__(self, step_name: str = "feature_generation_final_validation_step"):
        """Initialize the final validation step."""
        super().__init__(step_name)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final validation of feature datasets.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Timeframe
                - execution_mode: Execution mode (light, full, etc.)
                - validation_config: Optional validation configuration overrides

        Returns:
            Dict containing execution results and artifacts
        """
        try:
            tprint_info(f"🔍 Starting {self.step_name} execution...")

            # Get final datasets from previous step
            final_datasets = self._get_final_datasets()

            if not final_datasets:
                raise ValueError("No final datasets available for validation")

            # Perform comprehensive validation
            validation_results = self._perform_comprehensive_validation(final_datasets, config)

            # Assess validation success
            success = self._assess_validation_success(validation_results, config)

            # Generate artifacts
            artifacts = self._generate_artifacts(validation_results, config)

            # Create comprehensive outcome report
            outcome_report = self._create_outcome_report(validation_results, config)

            # Save artifacts
            saved_artifacts = []
            for artifact_name, artifact_data in artifacts.items():
                artifact_path = self._save_artifact(
                    artifact_data,
                    artifact_name,
                    artifact_type="data"
                )
                saved_artifacts.append({
                    'name': artifact_name,
                    'path': artifact_path,
                    'type': 'data'
                })

            # Save outcome report
            report_path = self._save_artifact(
                outcome_report,
                "final_validation_outcome_report",
                artifact_type="report"
            )

            # Calculate metrics
            metrics = self._calculate_metrics(validation_results, config)

            execution_result = {
                'success': success,
                'artifacts': saved_artifacts,
                'metrics': metrics,
                'validation_results': validation_results,
                'validation_summary': self._summarize_validation(validation_results),
                'outcome_report_path': report_path,
                'execution_time': 0.0  # Will be set by base class
            }

            if success:
                tprint_success(f"✅ {self.step_name} completed successfully")
                tprint_info(f"📊 Validated {len(final_datasets)} feature sets")
            else:
                tprint_warning(f"⚠️ {self.step_name} completed with validation warnings")

            return execution_result

        except Exception as e:
            error_msg = f"Final validation step failed: {str(e)}"
            tprint_error(error_msg)
            logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {},
                'execution_time': 0.0
            }

    def _get_final_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get final datasets from previous steps."""
        final_datasets = {}

        # Get the final feature selection artifacts
        feature_set_sizes = [60, 50, 40]  # Standard sizes

        for size in feature_set_sizes:
            # Try to get the selected feature dataframe
            try:
                dataset = self._get_artifact(f'selected_feature_dataframe_{size}')
                if dataset is not None and isinstance(dataset, pd.DataFrame):
                    final_datasets[f'final_dataset_{size}'] = dataset
                    tprint_info(f"📊 Retrieved final dataset with {size} features")
            except Exception as e:
                tprint_warning(f"⚠️ Could not retrieve final dataset for {size} features: {e}")

        # Also try to get labeled dataframe as fallback
        try:
            labeled_df = self._get_artifact('labeled_dataframe')
            if labeled_df is not None and len(final_datasets) == 0:
                final_datasets['labeled_dataframe'] = labeled_df
                tprint_info("📊 Using labeled dataframe as fallback for validation")
        except Exception as e:
            tprint_warning(f"⚠️ Could not retrieve labeled dataframe: {e}")

        return final_datasets

    def _perform_comprehensive_validation(self, datasets: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive validation of all datasets."""
        validation_results = {}

        for dataset_name, dataset in datasets.items():
            tprint_info(f"🔍 Validating {dataset_name}...")

            dataset_validation = self._validate_single_dataset(dataset, dataset_name, config)
            validation_results[dataset_name] = dataset_validation

        return validation_results

    def _validate_single_dataset(self, dataset: pd.DataFrame, dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single dataset comprehensively."""
        validation = {
            'dataset_name': dataset_name,
            'shape': dataset.shape,
            'columns': list(dataset.columns),
            'data_types': dataset.dtypes.to_dict(),
            'memory_usage': dataset.memory_usage(deep=True).sum(),
            'validations': {}
        }

        # Basic data quality validation
        validation['validations']['data_quality'] = self._validate_data_quality(dataset, config)

        # Feature distribution validation
        validation['validations']['feature_distributions'] = self._validate_feature_distributions(dataset, config)

        # Target relationship validation
        validation['validations']['target_relationships'] = self._validate_target_relationships(dataset, config)

        # Cross-validation readiness
        validation['validations']['cv_readiness'] = self._validate_cv_readiness(dataset, config)

        # Statistical validation
        validation['validations']['statistical'] = self._validate_statistical_properties(dataset, config)

        # Overall assessment
        validation['overall_assessment'] = self._assess_dataset_quality(validation['validations'], config)

        return validation

    def _validate_data_quality(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic data quality metrics."""
        quality = {}

        # NaN analysis
        nan_counts = dataset.isnull().sum()
        nan_percentages = (nan_counts / len(dataset)) * 100

        quality['total_nans'] = int(nan_counts.sum())
        quality['nan_columns'] = nan_counts[nan_counts > 0].to_dict()
        quality['high_nan_columns'] = nan_percentages[nan_percentages > 10].to_dict()  # >10% NaN
        quality['max_nan_percentage'] = float(nan_percentages.max())

        # Data type validation
        quality['numeric_columns'] = len(dataset.select_dtypes(include=[np.number]).columns)
        quality['categorical_columns'] = len(dataset.select_dtypes(include=['object', 'category']).columns)
        quality['datetime_columns'] = len(dataset.select_dtypes(include=['datetime']).columns)

        # Basic statistics
        numeric_data = dataset.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            quality['numeric_stats'] = {
                'mean': numeric_data.mean().to_dict(),
                'std': numeric_data.std().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict()
            }

        # Outlier detection (simple IQR method)
        outlier_counts = {}
        for col in numeric_data.columns:
            if col not in ['target', 'label', 'return']:  # Skip target columns for outlier detection
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)).sum()
                outlier_counts[col] = int(outliers)

        quality['outlier_counts'] = outlier_counts
        quality['total_outliers'] = sum(outlier_counts.values())

        return quality

    def _validate_feature_distributions(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature distributions for normality and other properties."""
        distributions = {}

        numeric_data = dataset.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_data.columns
                       if col not in ['target', 'label', 'return', 'timestamp']]

        # Distribution statistics
        distribution_stats = {}
        for col in feature_cols[:20]:  # Limit to first 20 features for performance
            try:
                # Basic distribution tests
                data = numeric_data[col].dropna()

                if len(data) > 30:  # Need sufficient data for tests
                    # Shapiro-Wilk test for normality
                    stat, p_value = stats.shapiro(data)
                    distribution_stats[col] = {
                        'shapiro_statistic': float(stat),
                        'shapiro_p_value': float(p_value),
                        'is_normal': p_value > 0.05,
                        'skewness': float(stats.skew(data)),
                        'kurtosis': float(stats.kurtosis(data)),
                        'variance': float(data.var())
                    }
                else:
                    distribution_stats[col] = {
                        'insufficient_data': True,
                        'sample_size': len(data)
                    }
            except Exception as e:
                distribution_stats[col] = {'error': str(e)}

        distributions['feature_distribution_stats'] = distribution_stats

        # Overall distribution assessment
        if distribution_stats:
            normal_features = len([col for col, stats in distribution_stats.items()
                                 if isinstance(stats, dict) and stats.get('is_normal', False)])
            distributions['normal_features_count'] = normal_features
            distributions['normal_features_percentage'] = normal_features / len(distribution_stats)

        return distributions

    def _validate_target_relationships(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate relationships between features and targets."""
        relationships = {}

        # Identify target and feature columns
        target_cols = [col for col in ['target', 'label', 'return'] if col in dataset.columns]
        feature_cols = [col for col in dataset.columns
                       if col not in target_cols + ['timestamp']]

        if not target_cols or not feature_cols:
            return {'error': 'No target or feature columns found'}

        target_col = target_cols[0]
        numeric_features = dataset[feature_cols].select_dtypes(include=[np.number])

        # Feature-target correlations
        correlations = {}
        for col in numeric_features.columns:
            try:
                corr = dataset[col].corr(dataset[target_col])
                correlations[col] = float(corr) if not np.isnan(corr) else 0.0
            except Exception as e:
                correlations[col] = {'error': str(e)}

        relationships['feature_target_correlations'] = correlations

        # Strong correlations (>0.8 or <-0.8)
        strong_correlations = {k: v for k, v in correlations.items()
                             if isinstance(v, (int, float)) and abs(v) > 0.8}
        relationships['strong_correlations'] = strong_correlations

        # Weak correlations (<0.1)
        weak_correlations = {k: v for k, v in correlations.items()
                           if isinstance(v, (int, float)) and abs(v) < 0.1}
        relationships['weak_correlations'] = weak_correlations

        # Target distribution
        try:
            target_data = dataset[target_col].dropna()
            relationships['target_stats'] = {
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
                'skewness': float(stats.skew(target_data)),
                'kurtosis': float(stats.kurtosis(target_data)),
                'positive_ratio': float((target_data > 0).mean()),
                'negative_ratio': float((target_data < 0).mean())
            }
        except Exception as e:
            relationships['target_stats'] = {'error': str(e)}

        return relationships

    def _validate_cv_readiness(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if dataset is ready for cross-validation."""
        cv_readiness = {}

        # Identify target and feature columns
        target_cols = [col for col in ['target', 'label', 'return'] if col in dataset.columns]
        feature_cols = [col for col in dataset.columns
                       if col not in target_cols + ['timestamp']]

        if not target_cols or not feature_cols:
            return {'error': 'No target or feature columns found'}

        target_col = target_cols[0]
        X = dataset[feature_cols].select_dtypes(include=[np.number]).dropna()
        y = dataset[target_col].dropna()

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 100:  # Need minimum data for CV
            return {'insufficient_data': True, 'sample_size': len(X)}

        try:
            # Simple model for CV testing
            model = RandomForestRegressor(n_estimators=10, random_state=42)

            # Time series aware cross-validation (if timestamp available)
            if 'timestamp' in dataset.columns:
                # Sort by timestamp for time series CV
                dataset_sorted = dataset.loc[common_idx].sort_values('timestamp')
                X_sorted = dataset_sorted[feature_cols].select_dtypes(include=[np.number])
                y_sorted = dataset_sorted[target_col]

                # Simple time series split
                split_idx = int(len(X_sorted) * 0.8)
                X_train, X_test = X_sorted[:split_idx], X_sorted[split_idx:]
                y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                cv_readiness['cv_type'] = 'time_series_split'
                cv_readiness['train_size'] = len(X_train)
                cv_readiness['test_size'] = len(X_test)
            else:
                # Regular cross-validation
                scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                predictions = model.fit(X, y).predict(X)

                cv_readiness['cv_type'] = 'kfold'
                cv_readiness['cv_scores'] = scores.tolist()
                cv_readiness['mean_cv_score'] = float(scores.mean())

            # Calculate performance metrics
            mse = mean_squared_error(y if 'timestamp' not in dataset.columns else y_test, predictions)
            r2 = r2_score(y if 'timestamp' not in dataset.columns else y_test, predictions)

            cv_readiness['mse'] = float(mse)
            cv_readiness['r2_score'] = float(r2)
            cv_readiness['rmse'] = float(np.sqrt(mse))

        except Exception as e:
            cv_readiness['error'] = str(e)

        return cv_readiness

    def _validate_statistical_properties(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical properties of the dataset."""
        stats_validation = {}

        # Multicollinearity check (correlation matrix)
        numeric_data = dataset.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_data.columns
                       if col not in ['target', 'label', 'return']]

        if len(feature_cols) > 1:
            try:
                correlation_matrix = numeric_data[feature_cols].corr()
                # Check for high correlations between features
                high_corr_pairs = []
                for i in range(len(feature_cols)):
                    for j in range(i+1, len(feature_cols)):
                        corr = correlation_matrix.iloc[i, j]
                        if abs(corr) > 0.9:  # Very high correlation
                            high_corr_pairs.append({
                                'feature1': feature_cols[i],
                                'feature2': feature_cols[j],
                                'correlation': float(corr)
                            })

                stats_validation['high_correlation_pairs'] = high_corr_pairs
                stats_validation['max_correlation'] = float(correlation_matrix.abs().max().max())
            except Exception as e:
                stats_validation['correlation_error'] = str(e)

        # Feature variance check
        variances = numeric_data[feature_cols].var()
        low_variance_features = variances[variances < 1e-6].to_dict()
        stats_validation['low_variance_features'] = low_variance_features
        stats_validation['min_variance'] = float(variances.min())
        stats_validation['max_variance'] = float(variances.max())

        return stats_validation

    def _assess_dataset_quality(self, validations: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall dataset quality based on all validations."""
        assessment = {
            'overall_score': 0.0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }

        total_checks = 0
        passed_checks = 0

        # Assess data quality
        if 'data_quality' in validations:
            dq = validations['data_quality']
            total_checks += 1

            if dq.get('max_nan_percentage', 100) < 5:  # Less than 5% NaN
                passed_checks += 1
            else:
                assessment['issues'].append(f"High NaN percentage: {dq['max_nan_percentage']:.1f}%")

            if dq.get('total_outliers', 0) < len(dataset) * 0.1:  # Less than 10% outliers
                passed_checks += 0.5
            else:
                assessment['warnings'].append(f"High outlier count: {dq['total_outliers']}")

        # Assess feature distributions
        if 'feature_distributions' in validations:
            fd = validations['feature_distributions']
            total_checks += 1

            if fd.get('normal_features_percentage', 0) > 0.3:  # More than 30% normal features
                passed_checks += 1
            else:
                assessment['recommendations'].append("Consider feature transformations for non-normal distributions")

        # Assess target relationships
        if 'target_relationships' in validations:
            tr = validations['target_relationships']
            total_checks += 1

            strong_correlations = len(tr.get('strong_correlations', {}))
            if strong_correlations > 0:
                passed_checks += 1
            else:
                assessment['warnings'].append("No strong feature-target correlations found")

        # Assess CV readiness
        if 'cv_readiness' in validations:
            cv = validations['cv_readiness']
            total_checks += 1

            if 'error' not in cv and cv.get('r2_score', -1) > 0:
                passed_checks += 1
            else:
                assessment['issues'].append("Poor cross-validation performance")

        # Calculate overall score
        if total_checks > 0:
            assessment['overall_score'] = passed_checks / total_checks

        # Determine quality level
        if assessment['overall_score'] >= 0.8:
            assessment['quality_level'] = 'excellent'
        elif assessment['overall_score'] >= 0.6:
            assessment['quality_level'] = 'good'
        elif assessment['overall_score'] >= 0.4:
            assessment['quality_level'] = 'fair'
        else:
            assessment['quality_level'] = 'poor'

        return assessment

    def _assess_validation_success(self, validation_results: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Assess if validation was successful based on results."""
        execution_mode = config.get('execution_mode', 'light')

        for dataset_name, validation in validation_results.items():
            assessment = validation.get('overall_assessment', {})
            quality_level = assessment.get('quality_level', 'unknown')

            # Check for critical issues
            issues = assessment.get('issues', [])
            if issues and execution_mode == 'strict':
                tprint_error(f"❌ Critical validation issues in {dataset_name}: {issues}")
                return False

            # Check quality level
            if quality_level == 'poor':
                tprint_warning(f"⚠️ Poor quality dataset {dataset_name}: {assessment.get('overall_score', 0):.2f}")
                if execution_mode == 'strict':
                    return False

        return True

    def _generate_artifacts(self, validation_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate artifacts from validation results."""
        artifacts = {}

        # Final validated datasets
        final_datasets = self._get_final_datasets()
        for dataset_name, dataset in final_datasets.items():
            artifacts[dataset_name] = dataset

        # Validation results summary
        artifacts['final_validation_metrics'] = validation_results

        # Quality scores
        quality_scores = {}
        for dataset_name, validation in validation_results.items():
            assessment = validation.get('overall_assessment', {})
            quality_scores[dataset_name] = {
                'overall_score': assessment.get('overall_score', 0.0),
                'quality_level': assessment.get('quality_level', 'unknown'),
                'issues_count': len(assessment.get('issues', [])),
                'warnings_count': len(assessment.get('warnings', [])),
                'recommendations_count': len(assessment.get('recommendations', []))
            }
        artifacts['final_quality_scores'] = quality_scores

        # Validation warnings
        all_warnings = []
        for dataset_name, validation in validation_results.items():
            assessment = validation.get('overall_assessment', {})
            for warning in assessment.get('warnings', []):
                all_warnings.append(f"{dataset_name}: {warning}")
        artifacts['final_validation_warnings'] = all_warnings

        # Performance metrics
        performance_metrics = {}
        for dataset_name, validation in validation_results.items():
            cv_readiness = validation.get('validations', {}).get('cv_readiness', {})
            if 'r2_score' in cv_readiness:
                performance_metrics[dataset_name] = {
                    'r2_score': cv_readiness['r2_score'],
                    'mse': cv_readiness.get('mse', 0),
                    'rmse': cv_readiness.get('rmse', 0)
                }
        artifacts['final_performance_metrics'] = performance_metrics

        return artifacts

    def _calculate_metrics(self, validation_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for the validation."""
        metrics = {
            'datasets_validated': len(validation_results),
            'execution_timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'execution_mode': config.get('execution_mode', 'light')
        }

        # Quality statistics
        quality_scores = []
        quality_levels = []
        total_issues = 0
        total_warnings = 0

        for validation in validation_results.values():
            assessment = validation.get('overall_assessment', {})
            quality_scores.append(assessment.get('overall_score', 0.0))
            quality_levels.append(assessment.get('quality_level', 'unknown'))
            total_issues += len(assessment.get('issues', []))
            total_warnings += len(assessment.get('warnings', []))

        if quality_scores:
            metrics.update({
                'avg_quality_score': float(np.mean(quality_scores)),
                'min_quality_score': float(np.min(quality_scores)),
                'max_quality_score': float(np.max(quality_scores)),
                'quality_levels': quality_levels,
                'total_issues': total_issues,
                'total_warnings': total_warnings
            })

        return metrics

    def _summarize_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of validation results."""
        summary = {
            'total_datasets': len(validation_results),
            'quality_levels': {},
            'top_issues': [],
            'top_warnings': []
        }

        for dataset_name, validation in validation_results.items():
            assessment = validation.get('overall_assessment', {})
            quality_level = assessment.get('quality_level', 'unknown')
            summary['quality_levels'][dataset_name] = quality_level

            # Collect issues and warnings
            for issue in assessment.get('issues', []):
                summary['top_issues'].append(f"{dataset_name}: {issue}")
            for warning in assessment.get('warnings', []):
                summary['top_warnings'].append(f"{dataset_name}: {warning}")

        return summary

    def _create_outcome_report(self, validation_results: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create comprehensive outcome report."""
        try:
            report = f"""# Final Dataset Validation Outcome Report

**Execution Details:**
- **Symbol:** {config.get('symbol', 'unknown')}
- **Exchange:** {config.get('exchange', 'binance')}
- **Timeframe:** {config.get('timeframe', '15m')}
- **Execution Mode:** {config.get('execution_mode', 'light')}
- **Timestamp:** {datetime.now().isoformat()}

## Validation Summary

**Datasets Validated:** {len(validation_results)}

**Overall Results:**
"""

            # Overall metrics
            quality_scores = []
            for validation in validation_results.values():
                assessment = validation.get('overall_assessment', {})
                quality_scores.append(assessment.get('overall_score', 0.0))

            if quality_scores:
                avg_score = np.mean(quality_scores)
                report += f"- **Average Quality Score:** {avg_score:.2f}\n"
                report += f"- **Quality Range:** {np.min(quality_scores):.2f} - {np.max(quality_scores):.2f}\n"

            # Dataset-specific results
            report += "\n## Dataset Quality Assessment\n"

            for dataset_name, validation in validation_results.items():
                assessment = validation.get('overall_assessment', {})
                quality_level = assessment.get('quality_level', 'unknown')
                score = assessment.get('overall_score', 0.0)

                quality_icon = {
                    'excellent': '🟢',
                    'good': '🟡',
                    'fair': '🟠',
                    'poor': '🔴'
                }.get(quality_level, '⚪')

                report += f"\n### {quality_icon} {dataset_name}\n"
                report += f"- **Quality Level:** {quality_level.upper()}\n"
                report += f"- **Quality Score:** {score:.2f}\n"
                report += f"- **Issues:** {len(assessment.get('issues', []))}\n"
                report += f"- **Warnings:** {len(assessment.get('warnings', []))}\n"

                # Key metrics
                validations = validation.get('validations', {})
                if 'data_quality' in validations:
                    dq = validations['data_quality']
                    report += f"- **Max NaN %:** {dq.get('max_nan_percentage', 0):.1f}%\n"
                    report += f"- **Outliers:** {dq.get('total_outliers', 0)}\n"

                if 'cv_readiness' in validations:
                    cv = validations['cv_readiness']
                    if 'r2_score' in cv:
                        report += f"- **CV R² Score:** {cv['r2_score']:.3f}\n"

            # Issues and warnings
            all_issues = []
            all_warnings = []
            for validation in validation_results.values():
                assessment = validation.get('overall_assessment', {})
                for issue in assessment.get('issues', []):
                    all_issues.append(issue)
                for warning in assessment.get('warnings', []):
                    all_warnings.append(warning)

            if all_issues:
                report += f"\n## Critical Issues ({len(all_issues)})\n"
                for issue in all_issues[:10]:  # Show top 10
                    report += f"- ❌ {issue}\n"

            if all_warnings:
                report += f"\n## Warnings ({len(all_warnings)})\n"
                for warning in all_warnings[:10]:  # Show top 10
                    report += f"- ⚠️ {warning}\n"

            # Recommendations
            all_recommendations = []
            for validation in validation_results.values():
                assessment = validation.get('overall_assessment', {})
                all_recommendations.extend(assessment.get('recommendations', []))

            if all_recommendations:
                report += f"\n## Recommendations ({len(all_recommendations)})\n"
                for rec in all_recommendations[:5]:  # Show top 5
                    report += f"- 💡 {rec}\n"

            # Generated artifacts
            artifact_count = len(validation_results)  # datasets
            artifact_count += 3  # validation_metrics, quality_scores, validation_warnings

            report += f"\n## Generated Artifacts\n"
            report += f"- **Final datasets:** {len(validation_results)}\n"
            report += f"- **Validation metrics:** 1\n"
            report += f"- **Quality scores:** 1\n"
            report += f"- **Validation warnings:** 1\n"
            report += f"- **Performance metrics:** 1\n"
            report += f"- **Total artifacts:** {artifact_count + 1}\n"  # +1 for report

            report += f"""

---
*Generated by Feature Generation Final Validation Step at {datetime.now().isoformat()}*
"""

            return report

        except Exception as e:
            tprint_error(f"⚠️ Failed to create outcome report: {e}")
            return f"# Final Validation Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_feature_generation_final_validation_step():
    """Register the feature generation final validation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_final_validation_step", FeatureGenerationFinalValidationStep)
    tprint("✅ Feature generation final validation step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_final_validation_step()
