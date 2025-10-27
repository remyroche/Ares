"""
Feature Generation Final Validation Step

This step performs comprehensive validation of the final feature datasets to ensure they are
ready for model training. It validates data quality, feature distributions, target relationships,
and generates final validation reports.

IMPORTANT: This step analyzes the OPPOSITE mode's artifacts:
- When called in Analyst mode, it analyzes Tactician artifacts
- When called in Tactician mode, it analyzes Analyst artifacts
- Default is Analyst mode (analyzes Tactician artifacts)

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
    
    IMPORTANT: This step analyzes the OPPOSITE mode's artifacts:
    - When called in Analyst mode, it analyzes Tactician artifacts
    - When called in Tactician mode, it analyzes Analyst artifacts
    - Default is Analyst mode (analyzes Tactician artifacts)
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
            # Determine analysis mode for logging
            execution_mode = self._determine_execution_mode()
            analysis_mode = 'tactician' if execution_mode == 'analyst' else 'analyst'
            
            tprint_info(f"🔍 Starting {self.step_name} execution...")
            tprint_info(f"🔍 Running in {execution_mode} mode, analyzing {analysis_mode} artifacts")

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

            # Get analysis mode for reporting
            execution_mode = self._determine_execution_mode()
            analysis_mode = 'tactician' if execution_mode == 'analyst' else 'analyst'
            
            if success:
                tprint_success(f"✅ {self.step_name} completed successfully")
                tprint_info(f"📊 Validated {len(final_datasets)} {analysis_mode} feature sets")
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
        """Get final datasets from previous steps based on execution mode.
        
        This function analyzes the OPPOSITE mode's artifacts:
        - When called in Analyst mode, it analyzes Tactician artifacts
        - When called in Tactician mode, it analyzes Analyst artifacts
        - Default is Analyst mode (analyzes Tactician artifacts)
        """
        final_datasets = {}

        # Determine execution mode (Analyst vs Tactician)
        execution_mode = self._determine_execution_mode()
        
        # Determine which mode's artifacts to analyze (opposite of execution mode)
        analysis_mode = 'tactician' if execution_mode == 'analyst' else 'analyst'
        tprint_info(f"🔍 Execution mode: {execution_mode}, analyzing {analysis_mode} artifacts")

        # Get the final feature selection artifacts based on analysis mode
        feature_set_sizes = [60, 50, 40]  # Standard sizes

        for size in feature_set_sizes:
            # Try to get the selected feature dataframe with analysis mode-specific naming
            artifact_names_to_try = [
                f'selected_feature_dataframe_{size}',  # Generic fallback
                f'{analysis_mode}_selected_feature_dataframe_{size}',  # Analysis mode-specific
                f'final_{analysis_mode}_dataset_{size}'  # Alternative naming
            ]
            
            dataset = None
            for artifact_name in artifact_names_to_try:
                try:
                    dataset = self._get_artifact(artifact_name)
                    if dataset is not None and isinstance(dataset, pd.DataFrame):
                        final_datasets[f'final_dataset_{size}'] = dataset
                        tprint_info(f"📊 Retrieved {analysis_mode} final dataset with {size} features from '{artifact_name}'")
                        break
                except Exception as e:
                    continue
            
            if dataset is None:
                tprint_warning(f"⚠️ Could not retrieve final dataset for {size} features in {analysis_mode} mode")

        # Also try to get labeled dataframe for target relationships with analysis mode-specific naming
        labeled_df = None
        artifact_names_to_try = [
            'labeled_dataframe', 'labeled_data', 'labeled_dataset', 'target_dataframe',  # Generic fallbacks
            f'{analysis_mode}_labeled_dataframe', f'{analysis_mode}_labeled_data',  # Analysis mode-specific
            f'{analysis_mode}_target_dataframe', f'{analysis_mode}_dataset'  # Alternative naming
        ]
        
        for artifact_name in artifact_names_to_try:
            try:
                labeled_df = self._get_artifact(artifact_name)
                if labeled_df is not None:
                    tprint_info(f"📊 Retrieved {analysis_mode} labeled data from '{artifact_name}' with {len(labeled_df.columns)} columns")
                    break
            except Exception as e:
                continue
        
        if labeled_df is not None:
            final_datasets['labeled_dataframe'] = labeled_df
        else:
            tprint_warning(f"⚠️ Could not retrieve labeled data from any known artifact names for {analysis_mode} mode")

        return final_datasets

    def _determine_execution_mode(self) -> str:
        """
        Determine execution mode (Analyst vs Tactician) from configuration.
        
        Returns:
            str: 'analyst' or 'tactician' (defaults to 'analyst')
        """
        # Check for execution_context in config (set by ares_launcher)
        if hasattr(self, 'config') and self.config:
            execution_context = self.config.get('execution_context', '').lower()
            
            if 'tactician' in execution_context:
                return 'tactician'
            elif 'analyst' in execution_context:
                return 'analyst'
        
        # Fallback: check for mode-specific artifacts to infer the mode
        try:
            # Check if tactician-specific artifacts exist
            tactician_artifacts = [
                'tactician_interaction_features',
                'tactician_selected_feature_dataframe_60',
                'tactician_labeled_dataframe'
            ]
            
            analyst_artifacts = [
                'analyst_interaction_features', 
                'analyst_selected_feature_dataframe_60',
                'analyst_labeled_dataframe'
            ]
            
            # Count available artifacts for each mode
            tactician_count = 0
            analyst_count = 0
            
            for artifact_name in tactician_artifacts:
                try:
                    artifact = self._get_artifact(artifact_name)
                    if artifact is not None:
                        tactician_count += 1
                except Exception:
                    continue
            
            for artifact_name in analyst_artifacts:
                try:
                    artifact = self._get_artifact(artifact_name)
                    if artifact is not None:
                        analyst_count += 1
                except Exception:
                    continue
            
            # Determine mode based on artifact availability
            if tactician_count > analyst_count:
                tprint_info(f"🔍 Inferred tactician mode from artifacts ({tactician_count} tactician vs {analyst_count} analyst artifacts)")
                return 'tactician'
            elif analyst_count > tactician_count:
                tprint_info(f"🔍 Inferred analyst mode from artifacts ({analyst_count} analyst vs {tactician_count} tactician artifacts)")
                return 'analyst'
            else:
                # Equal or no artifacts found, default to analyst
                tprint_info(f"🔍 No clear mode indicators found, defaulting to analyst mode")
                return 'analyst'
                
        except Exception as e:
            tprint_warning(f"⚠️ Error determining execution mode from artifacts: {e}")
            return 'analyst'  # Default fallback

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

        # Filter normalization validation
        validation['validations']['filter_normalization'] = self._validate_filter_normalization(dataset, config)

        # Overall assessment
        validation['overall_assessment'] = self._assess_dataset_quality(validation['validations'], config, dataset)

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

        # Outlier detection (3-sigma method for financial data)
        outlier_counts = {}
        for col in numeric_data.columns:
            if col not in ['target', 'label', 'return']:  # Skip target columns for outlier detection
                mean_val = numeric_data[col].mean()
                std_val = numeric_data[col].std()
                
                # Use 3-sigma rule for financial data (more lenient than IQR)
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

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
            
            # Calculate average statistics
            skewness_values = [stats.get('skewness', 0) for stats in distribution_stats.values() 
                              if isinstance(stats, dict) and 'skewness' in stats]
            kurtosis_values = [stats.get('kurtosis', 0) for stats in distribution_stats.values() 
                              if isinstance(stats, dict) and 'kurtosis' in stats]
            
            distributions['normal_features_count'] = normal_features
            distributions['total_features_tested'] = len(distribution_stats)
            distributions['normal_features_percentage'] = (normal_features / len(distribution_stats)) * 100
            distributions['average_skewness'] = np.mean(skewness_values) if skewness_values else 0
            distributions['average_kurtosis'] = np.mean(kurtosis_values) if kurtosis_values else 0
            distributions['skewness_range'] = (min(skewness_values), max(skewness_values)) if skewness_values else (0, 0)
            distributions['kurtosis_range'] = (min(kurtosis_values), max(kurtosis_values)) if kurtosis_values else (0, 0)

        return distributions

    def _validate_target_relationships(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate relationships between features and targets."""
        relationships = {}

        # Identify target and feature columns
        # Look for common target column patterns in financial data
        target_patterns = ['target', 'label', 'return', 'quality_scores', 'price_target', 'volatility_target', 'direction']
        target_cols = []
        
        for col in dataset.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in target_patterns):
                target_cols.append(col)
        
        # If no obvious targets found, look for columns that might be targets based on naming
        if not target_cols:
            for col in dataset.columns:
                col_lower = col.lower()
                if any(suffix in col_lower for suffix in ['_target', '_label', '_score', '_signal', '_direction']):
                    target_cols.append(col)
        
        feature_cols = [col for col in dataset.columns
                       if col not in target_cols + ['timestamp', 'open_time', 'close_time']]

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

    def _validate_filter_normalization(self, dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all filter grades are properly normalized/scaled."""
        filter_validation = {
            'filter_grades_found': False,
            'normalization_status': {},
            'scaling_issues': [],
            'recommendations': []
        }

        # Identify filter grade columns
        filter_grade_patterns = [
            'efficiency_grade', 'bar_efficiency_grade',
            'clv_grade', 'close_location_value_grade',
            'atr_grade', 'atr_volatility_grade',
            'trend_coherence_grade', 'trend_grade',
            'filter_grade', 'quality_grade'
        ]
        
        filter_grade_cols = []
        for col in dataset.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in filter_grade_patterns):
                filter_grade_cols.append(col)

        if not filter_grade_cols:
            filter_validation['recommendations'].append("No filter grade columns found - filters may not be generating normalized grades")
            return filter_validation

        filter_validation['filter_grades_found'] = True
        filter_validation['filter_grade_columns'] = filter_grade_cols

        # Validate each filter grade column
        for col in filter_grade_cols:
            try:
                grade_data = dataset[col].dropna()
                if len(grade_data) == 0:
                    filter_validation['scaling_issues'].append(f"Column '{col}' has no valid data")
                    continue

                # Check if values are in expected range [0, 1]
                min_val = float(grade_data.min())
                max_val = float(grade_data.max())
                mean_val = float(grade_data.mean())
                std_val = float(grade_data.std())

                # Validate range
                range_valid = 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0
                
                # Check for proper distribution (not all same value)
                distribution_valid = std_val > 1e-6
                
                # Check for reasonable mean (not all 0s or 1s)
                mean_reasonable = 0.1 <= mean_val <= 0.9

                col_status = {
                    'range_valid': range_valid,
                    'distribution_valid': distribution_valid,
                    'mean_reasonable': mean_reasonable,
                    'min_value': min_val,
                    'max_value': max_val,
                    'mean_value': mean_val,
                    'std_value': std_val,
                    'sample_count': len(grade_data)
                }

                filter_validation['normalization_status'][col] = col_status

                # Generate recommendations
                if not range_valid:
                    filter_validation['scaling_issues'].append(f"Column '{col}' values outside [0,1] range: [{min_val:.3f}, {max_val:.3f}]")
                    filter_validation['recommendations'].append(f"Normalize '{col}' to [0,1] range using MinMaxScaler or manual scaling")
                
                if not distribution_valid:
                    filter_validation['scaling_issues'].append(f"Column '{col}' has no variance (all values identical)")
                    filter_validation['recommendations'].append(f"Check if '{col}' filter is working correctly - no grade variation detected")
                
                if not mean_reasonable:
                    if mean_val < 0.1:
                        filter_validation['scaling_issues'].append(f"Column '{col}' mean too low ({mean_val:.3f}) - may indicate overly strict filtering")
                        filter_validation['recommendations'].append(f"Consider adjusting '{col}' filter thresholds to allow more samples")
                    elif mean_val > 0.9:
                        filter_validation['scaling_issues'].append(f"Column '{col}' mean too high ({mean_val:.3f}) - may indicate overly lenient filtering")
                        filter_validation['recommendations'].append(f"Consider tightening '{col}' filter thresholds for better quality control")

                # Check for extreme skewness (indicating poor normalization)
                if std_val > 0:
                    skewness = float(stats.skew(grade_data))
                    if abs(skewness) > 2.0:
                        filter_validation['scaling_issues'].append(f"Column '{col}' has extreme skewness ({skewness:.3f}) - may need better normalization")
                        filter_validation['recommendations'].append(f"Consider using RobustScaler or log transformation for '{col}' normalization")

            except Exception as e:
                filter_validation['scaling_issues'].append(f"Error validating column '{col}': {str(e)}")

        # Overall assessment
        total_issues = len(filter_validation['scaling_issues'])
        total_columns = len(filter_grade_cols)
        
        if total_issues == 0:
            filter_validation['overall_status'] = 'excellent'
            filter_validation['recommendations'].append("✅ All filter grades are properly normalized and scaled")
        elif total_issues <= total_columns * 0.3:  # Less than 30% of columns have issues
            filter_validation['overall_status'] = 'good'
            filter_validation['recommendations'].append("✅ Most filter grades are properly normalized, minor issues detected")
        elif total_issues <= total_columns * 0.6:  # Less than 60% of columns have issues
            filter_validation['overall_status'] = 'fair'
            filter_validation['recommendations'].append("⚠️ Some filter grades need normalization/scaling improvements")
        else:
            filter_validation['overall_status'] = 'poor'
            filter_validation['recommendations'].append("❌ Multiple filter grades need normalization/scaling fixes")

        return filter_validation

    def _assess_dataset_quality(self, validations: Dict[str, Any], config: Dict[str, Any], dataset: pd.DataFrame) -> Dict[str, Any]:
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

            # Detailed NaN analysis
            max_nan_pct = dq.get('max_nan_percentage', 100)
            total_nans = dq.get('total_nans', 0)
            high_nan_cols = dq.get('high_nan_columns', {})
            
            if max_nan_pct < 5:  # Less than 5% NaN
                passed_checks += 1
                assessment['recommendations'].append(f"✅ Good data completeness: max NaN {max_nan_pct:.1f}%")
            else:
                assessment['issues'].append(f"❌ High NaN percentage: {max_nan_pct:.1f}% (total: {total_nans:,} missing values)")
                if high_nan_cols:
                    worst_cols = sorted(high_nan_cols.items(), key=lambda x: x[1], reverse=True)[:5]
                    assessment['issues'].append(f"   Worst columns: {', '.join([f'{col}({pct:.1f}%)' for col, pct in worst_cols])}")

            # Detailed outlier analysis
            total_outliers = dq.get('total_outliers', 0)
            outlier_percentage = (total_outliers / len(dataset)) * 100 if len(dataset) > 0 else 0
            outlier_counts = dq.get('outlier_counts', {})
            
            if total_outliers < len(dataset) * 0.3:  # Less than 30% outliers (more realistic threshold)
                passed_checks += 0.5
                assessment['recommendations'].append(f"✅ Acceptable outlier level: {outlier_percentage:.1f}% ({total_outliers:,} outliers)")
            else:
                assessment['warnings'].append(f"⚠️ High outlier count: {outlier_percentage:.1f}% ({total_outliers:,} outliers)")
                
                # Show which features have the most outliers
                if outlier_counts:
                    top_outlier_features = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    outlier_details = ', '.join([f'{col}({count})' for col, count in top_outlier_features])
                    assessment['warnings'].append(f"   Top outlier features: {outlier_details}")
                    assessment['recommendations'].append("✅ Using 3-sigma outlier detection (more lenient than IQR) for financial data")
                
            # Data type analysis
            numeric_cols = dq.get('numeric_columns', 0)
            categorical_cols = dq.get('categorical_columns', 0)
            datetime_cols = dq.get('datetime_columns', 0)
            assessment['recommendations'].append(f"📊 Data types: {numeric_cols} numeric, {categorical_cols} categorical, {datetime_cols} datetime")

        # Assess feature distributions
        if 'feature_distributions' in validations:
            fd = validations['feature_distributions']
            total_checks += 1

            normal_pct = fd.get('normal_features_percentage', 0)
            total_features_tested = fd.get('total_features_tested', 0)
            normal_features_count = fd.get('normal_features_count', 0)
            
            # For financial data, we expect many features to be non-normal
            if normal_pct > 0.2:  # More than 20% normal features (lowered threshold for financial data)
                passed_checks += 1
                assessment['recommendations'].append(f"✅ Good distribution normality: {normal_pct:.1f}% ({normal_features_count}/{total_features_tested}) features are normal")
            else:
                # Check if non-normal features are percentage-based, naturally bounded, or raw market data
                distribution_stats = fd.get('feature_distribution_stats', {})
                problematic_features = []
                acceptable_features = []
                
                for col, stats in distribution_stats.items():
                    if isinstance(stats, dict) and not stats.get('is_normal', False):
                        skewness = stats.get('skewness', 0)
                        col_lower = col.lower()
                        
                        # Skip percentage features, naturally bounded features, and raw market data
                        is_percentage = any(suffix in col_lower for suffix in ['_pct', '_percent', '_ratio', '_rate', '_return', '_log_return'])
                        is_raw_market_data = any(raw_feature in col_lower for raw_feature in [
                            'price_range', 'body_size', 'volume', 'quote_volume', 'trades', 
                            'high', 'low', 'open', 'close', 'amount', 'count'
                        ])
                        
                        if is_percentage or is_raw_market_data:
                            feature_type = "percentage" if is_percentage else "raw market data"
                            acceptable_features.append(f"{col}(skew={skewness:.2f}, {feature_type})")
                        else:
                            problematic_features.append(f"{col}(skew={skewness:.2f})")
                
                if problematic_features:
                    assessment['warnings'].append(f"⚠️ Low normality: only {normal_pct:.1f}% ({normal_features_count}/{total_features_tested}) features are normally distributed")
                    # Show top 5 most skewed non-percentage features
                    top_skewed = sorted(problematic_features, key=lambda x: abs(float(x.split('skew=')[1].rstrip(')'))), reverse=True)[:5]
                    assessment['warnings'].append(f"   Most skewed features: {', '.join(top_skewed)}")
                    assessment['recommendations'].append("💡 Consider feature transformations (log, sqrt, Box-Cox) for engineered features only")
                else:
                    assessment['recommendations'].append(f"✅ Acceptable distributions: {normal_pct:.1f}% normal, {len(acceptable_features)} percentage/raw market data features")
                
            # Distribution statistics summary
            avg_skewness = fd.get('average_skewness', 0)
            avg_kurtosis = fd.get('average_kurtosis', 0)
            assessment['recommendations'].append(f"📈 Distribution stats: avg skewness={avg_skewness:.2f}, avg kurtosis={avg_kurtosis:.2f}")

        # Assess target relationships
        if 'target_relationships' in validations:
            tr = validations['target_relationships']
            total_checks += 1

            strong_correlations = tr.get('strong_correlations', {})
            all_correlations = tr.get('feature_target_correlations', {})
            strong_count = len(strong_correlations)
            total_features = len(all_correlations)
            
            if strong_count > 0:
                passed_checks += 1
                assessment['recommendations'].append(f"✅ Strong correlations found: {strong_count}/{total_features} features have |corr| > 0.8")
                # Show top correlations
                top_correlations = sorted(strong_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                corr_details = ', '.join([f'{col}({corr:.3f})' for col, corr in top_correlations])
                assessment['recommendations'].append(f"   Top correlations: {corr_details}")
            else:
                # Check if this is due to missing targets
                error_msg = tr.get('error', '')
                if 'No target' in error_msg or 'No feature' in error_msg:
                    assessment['issues'].append(f"❌ Missing target variables: {error_msg}")
                    assessment['recommendations'].append("💡 Target variables are required for supervised learning and correlation analysis")
                    assessment['recommendations'].append("💡 Check if labeled data contains target columns with expected naming patterns")
                else:
                    assessment['warnings'].append(f"⚠️ No strong correlations: {total_features} features tested, none with |corr| > 0.8")
                    
                    # Show best correlations even if not strong
                    if all_correlations:
                        best_correlations = sorted(all_correlations.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)[:5]
                        best_details = ', '.join([f'{col}({corr:.3f})' for col, corr in best_correlations if isinstance(corr, (int, float))])
                        assessment['recommendations'].append(f"💡 Best correlations: {best_details}")
                        assessment['recommendations'].append("💡 Consider feature engineering to create more predictive features")

        # Assess CV readiness
        if 'cv_readiness' in validations:
            cv = validations['cv_readiness']
            total_checks += 1

            if 'error' not in cv and cv.get('r2_score', -1) > 0:
                r2_score = cv.get('r2_score', 0)
                mse = cv.get('mse', 0)
                rmse = cv.get('rmse', 0)
                passed_checks += 1
                assessment['recommendations'].append(f"✅ Good CV performance: R²={r2_score:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}")
            else:
                error_msg = cv.get('error', 'Unknown error')
                r2_score = cv.get('r2_score', 0)
                mse = cv.get('mse', 0)
                rmse = cv.get('rmse', 0)
                
                if 'insufficient_data' in cv:
                    sample_size = cv.get('sample_size', 0)
                    assessment['issues'].append(f"❌ Insufficient data for CV: only {sample_size} samples (need ≥100)")
                elif 'No target' in error_msg or 'No feature' in error_msg:
                    assessment['issues'].append(f"❌ Cannot perform CV: {error_msg}")
                    assessment['recommendations'].append("💡 Cross-validation requires both features and targets")
                    assessment['recommendations'].append("💡 Check if labeled data contains target columns with expected naming patterns")
                else:
                    assessment['issues'].append(f"❌ Poor CV performance: R²={r2_score:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}")
                    if error_msg != 'Unknown error':
                        assessment['issues'].append(f"   Error details: {error_msg}")
                
                assessment['recommendations'].append("💡 Consider: 1) More data collection, 2) Feature selection, 3) Different algorithms, 4) Target variable refinement")

        # Assess filter normalization
        if 'filter_normalization' in validations:
            fn = validations['filter_normalization']
            total_checks += 1
            
            if fn.get('filter_grades_found', False):
                overall_status = fn.get('overall_status', 'unknown')
                scaling_issues = fn.get('scaling_issues', [])
                recommendations = fn.get('recommendations', [])
                
                if overall_status == 'excellent':
                    passed_checks += 1
                    assessment['recommendations'].extend([r for r in recommendations if r.startswith('✅')])
                elif overall_status == 'good':
                    passed_checks += 0.8
                    assessment['recommendations'].extend([r for r in recommendations if r.startswith('✅')])
                    assessment['warnings'].extend([r for r in recommendations if r.startswith('⚠️')])
                elif overall_status == 'fair':
                    passed_checks += 0.5
                    assessment['warnings'].extend([r for r in recommendations if r.startswith('⚠️')])
                    assessment['issues'].extend(scaling_issues[:3])  # Show top 3 issues
                else:  # poor
                    assessment['issues'].extend(scaling_issues[:5])  # Show top 5 issues
                    assessment['recommendations'].extend([r for r in recommendations if r.startswith('❌')])
                
                # Add specific filter grade statistics
                normalization_status = fn.get('normalization_status', {})
                if normalization_status:
                    valid_grades = sum(1 for status in normalization_status.values() if status.get('range_valid', False))
                    total_grades = len(normalization_status)
                    assessment['recommendations'].append(f"📊 Filter grades: {valid_grades}/{total_grades} properly normalized")
            else:
                assessment['warnings'].append("⚠️ No filter grades found - advanced filters may not be generating normalized outputs")
                assessment['recommendations'].append("💡 Consider enabling filter grade generation in advanced filters configuration")

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
                score = assessment.get('overall_score', 0)
                issues = assessment.get('issues', [])
                warnings = assessment.get('warnings', [])
                
                tprint_warning(f"⚠️ Poor quality dataset {dataset_name}: {score:.2f}")
                
                # Show specific issues
                if issues:
                    tprint_error(f"   Critical issues: {len(issues)}")
                    for issue in issues[:3]:  # Show top 3 issues
                        tprint_error(f"   - {issue}")
                
                if warnings:
                    tprint_warning(f"   Warnings: {len(warnings)}")
                    for warning in warnings[:3]:  # Show top 3 warnings
                        tprint_warning(f"   - {warning}")
                
                # Show recommendations
                recommendations = assessment.get('recommendations', [])
                if recommendations:
                    tprint_info(f"   Recommendations: {len(recommendations)}")
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        tprint_info(f"   - {rec}")
                
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

        # Validation results summary (flattened for PyArrow compatibility)
        flattened_metrics = self._flatten_validation_results(validation_results)
        artifacts['final_validation_metrics'] = flattened_metrics

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

    def _flatten_validation_results(self, validation_results: Dict[str, Any]) -> pd.DataFrame:
        """Flatten validation results into a PyArrow-compatible DataFrame."""
        flattened_data = []
        
        for dataset_name, validation in validation_results.items():
            # Extract basic information
            row = {
                'dataset_name': dataset_name,
                'shape_rows': validation.get('shape', (0, 0))[0],
                'shape_cols': validation.get('shape', (0, 0))[1],
                'memory_usage': validation.get('memory_usage', 0),
                'columns_count': len(validation.get('columns', [])),
            }
            
            # Extract overall assessment
            assessment = validation.get('overall_assessment', {})
            row.update({
                'overall_score': assessment.get('overall_score', 0.0),
                'quality_level': assessment.get('quality_level', 'unknown'),
                'issues_count': len(assessment.get('issues', [])),
                'warnings_count': len(assessment.get('warnings', [])),
                'recommendations_count': len(assessment.get('recommendations', [])),
            })
            
            # Extract validation details
            validations = validation.get('validations', {})
            for validation_type, validation_data in validations.items():
                if isinstance(validation_data, dict):
                    # Flatten nested validation data
                    for key, value in validation_data.items():
                        if isinstance(value, (int, float, str, bool)):
                            row[f"{validation_type}_{key}"] = value
                        else:
                            row[f"{validation_type}_{key}"] = str(value)
                else:
                    row[validation_type] = str(validation_data)
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)

    def _calculate_metrics(self, validation_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for the validation."""
        execution_mode = self._determine_execution_mode()
        analysis_mode = 'tactician' if execution_mode == 'analyst' else 'analyst'
        
        metrics = {
            'datasets_validated': len(validation_results),
            'execution_timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'execution_mode': config.get('execution_mode', 'light'),
            'execution_mode_type': execution_mode,  # analyst or tactician (the mode we're running in)
            'analysis_mode_type': analysis_mode,  # analyst or tactician (the mode whose artifacts we're analyzing)
            'execution_context': config.get('execution_context', 'unknown')
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
            # Get execution mode and analysis mode for reporting
            execution_mode = self._determine_execution_mode()
            analysis_mode = 'tactician' if execution_mode == 'analyst' else 'analyst'
            
            report = f"""# Final Dataset Validation Outcome Report

**Execution Details:**
- **Symbol:** {config.get('symbol', 'unknown')}
- **Exchange:** {config.get('exchange', 'binance')}
- **Timeframe:** {config.get('timeframe', '15m')}
- **Execution Mode:** {config.get('execution_mode', 'light')}
- **Running in:** {execution_mode.title()} mode
- **Analyzing:** {analysis_mode.title()} artifacts
- **Timestamp:** {datetime.now().isoformat()}

## Validation Summary

**Datasets Validated:** {len(validation_results)} ({analysis_mode.title()} artifacts)

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
                report += f"- **Shape:** {validation.get('shape', 'unknown')}\n"
                report += f"- **Memory Usage:** {validation.get('memory_usage', 0):,} bytes\n"

                # Detailed validation results
                validations = validation.get('validations', {})
                
                # Data Quality Details
                if 'data_quality' in validations:
                    dq = validations['data_quality']
                    report += f"\n#### 📊 Data Quality Metrics\n"
                    report += f"- **NaN Percentage:** {dq.get('max_nan_percentage', 0):.1f}% (total: {dq.get('total_nans', 0):,} missing)\n"
                    report += f"- **Outliers:** {dq.get('total_outliers', 0):,} detected\n"
                    report += f"- **Data Types:** {dq.get('numeric_columns', 0)} numeric, {dq.get('categorical_columns', 0)} categorical\n"
                    
                    if dq.get('high_nan_columns'):
                        report += f"- **High NaN Columns:** {', '.join([f'{col}({pct:.1f}%)' for col, pct in list(dq['high_nan_columns'].items())[:5]])}\n"

                # Feature Distribution Details
                if 'feature_distributions' in validations:
                    fd = validations['feature_distributions']
                    report += f"\n#### 📈 Feature Distribution Analysis\n"
                    report += f"- **Normal Features:** {fd.get('normal_features_count', 0)}/{fd.get('total_features_tested', 0)} ({fd.get('normal_features_percentage', 0):.1f}%)\n"
                    report += f"- **Average Skewness:** {fd.get('average_skewness', 0):.2f}\n"
                    report += f"- **Average Kurtosis:** {fd.get('average_kurtosis', 0):.2f}\n"

                # Target Relationship Details
                if 'target_relationships' in validations:
                    tr = validations['target_relationships']
                    strong_corr = tr.get('strong_correlations', {})
                    all_corr = tr.get('feature_target_correlations', {})
                    report += f"\n#### 🎯 Feature-Target Relationships\n"
                    report += f"- **Strong Correlations:** {len(strong_corr)}/{len(all_corr)} features with |corr| > 0.8\n"
                    
                    if strong_corr:
                        top_corr = sorted(strong_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                        report += f"- **Top Correlations:** {', '.join([f'{col}({corr:.3f})' for col, corr in top_corr])}\n"
                    elif all_corr:
                        best_corr = sorted(all_corr.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)[:3]
                        report += f"- **Best Correlations:** {', '.join([f'{col}({corr:.3f})' for col, corr in best_corr if isinstance(corr, (int, float))])}\n"

                # CV Performance Details
                if 'cv_readiness' in validations:
                    cv = validations['cv_readiness']
                    report += f"\n#### 🔄 Cross-Validation Performance\n"
                    if 'error' not in cv:
                        report += f"- **R² Score:** {cv.get('r2_score', 0):.3f}\n"
                        report += f"- **MSE:** {cv.get('mse', 0):.3f}\n"
                        report += f"- **RMSE:** {cv.get('rmse', 0):.3f}\n"
                    else:
                        report += f"- **Status:** Failed - {cv.get('error', 'Unknown error')}\n"
                        if 'insufficient_data' in cv:
                            report += f"- **Sample Size:** {cv.get('sample_size', 0)} (minimum required: 100)\n"

                # Issues and warnings
                issues = assessment.get('issues', [])
                warnings = assessment.get('warnings', [])
                recommendations = assessment.get('recommendations', [])

                if issues:
                    report += f"\n#### ❌ Critical Issues ({len(issues)})\n"
                    for issue in issues:
                        report += f"- {issue}\n"

                if warnings:
                    report += f"\n#### ⚠️ Warnings ({len(warnings)})\n"
                    for warning in warnings:
                        report += f"- {warning}\n"

                if recommendations:
                    report += f"\n#### 💡 Recommendations ({len(recommendations)})\n"
                    for rec in recommendations:
                        report += f"- {rec}\n"

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
