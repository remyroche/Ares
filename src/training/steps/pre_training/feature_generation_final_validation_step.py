"""
Feature Generation Final Validation Step

This step performs final validation of the selected features,
ensuring they meet quality standards and are ready for model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from scipy import stats

from src.training.steps.base_step import BaseStep
from src.training.steps.pre_training.validation.data_contracts import validate_selection_artifact
from src.training.steps.pre_training.standardized_labeling_interface import validate_dataframe_schema


class FeatureGenerationFinalValidationStep(BaseStep):
    """
    Validates final features before model training.
    
    Validation checks:
    - Data quality (no NaN, inf values)
    - Feature distribution analysis
    - Label alignment
    - Statistical properties
    - Ready for training pipeline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the final validation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_final_validation_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final validation.
        
        Args:
            config: Configuration containing:
                - min_samples: Minimum required samples
                - max_nan_ratio: Maximum allowed NaN ratio
                - validate_distribution: Whether to validate distributions
        
        Returns:
            Dictionary containing:
                - success: bool
                - validation_report: Dict
                - ready_for_training: bool
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("✅ Starting final validation")
            
            # Load final features
            final_features = self._load_dataframe('final_selected_features')
            if final_features is None:
                return {
                    'success': False,
                    'error': 'No final selected features found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Get label column
            label_column = config.get('label_column', 'label')
            if label_column not in final_features.columns:
                return {
                    'success': False,
                    'error': f'Label column {label_column} not found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Initialize validation report
            validation_report = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(final_features),
                'total_features': len(final_features.columns) - 1,  # Excluding label
                'checks': {},
                'warnings': [],
                'errors': []
            }
            
            # 1. Check minimum samples
            min_samples = config.get('min_samples', 1000)
            if len(final_features) < min_samples:
                validation_report['errors'].append(
                    f"Insufficient samples: {len(final_features)} < {min_samples}"
                )
                validation_report['checks']['min_samples'] = False
            else:
                validation_report['checks']['min_samples'] = True
            
            # 2. Check for NaN values
            nan_counts = final_features.isnull().sum()
            total_nans = nan_counts.sum()
            max_nan_ratio = config.get('max_nan_ratio', 0.01)
            nan_ratio = total_nans / (len(final_features) * len(final_features.columns))
            
            if nan_ratio > max_nan_ratio:
                validation_report['errors'].append(
                    f"Too many NaN values: {nan_ratio:.4f} > {max_nan_ratio}"
                )
                validation_report['checks']['nan_check'] = False
            else:
                validation_report['checks']['nan_check'] = True
            
            validation_report['nan_ratio'] = float(nan_ratio)
            validation_report['nan_counts'] = {
                col: int(count) for col, count in nan_counts.items() if count > 0
            }
            
            # 3. Check for infinite values
            inf_counts = np.isinf(final_features.select_dtypes(include=[np.number])).sum()
            total_infs = inf_counts.sum()
            
            if total_infs > 0:
                validation_report['errors'].append(
                    f"Found {total_infs} infinite values"
                )
                validation_report['checks']['inf_check'] = False
            else:
                validation_report['checks']['inf_check'] = True
            
            # 4. Check feature distributions
            if config.get('validate_distribution', True):
                distribution_stats = self._validate_distributions(final_features, label_column)
                validation_report['distribution_stats'] = distribution_stats
                
                # Check for constant features
                constant_features = [
                    feat for feat, stats_dict in distribution_stats.items()
                    if stats_dict.get('std', 0) == 0
                ]
                
                if constant_features:
                    validation_report['warnings'].append(
                        f"Found {len(constant_features)} constant features"
                    )
                    validation_report['checks']['no_constant_features'] = False
                else:
                    validation_report['checks']['no_constant_features'] = True
            
            # 5. Validate label properties
            y = final_features[label_column]
            label_stats = {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'skewness': float(stats.skew(y)),
                'kurtosis': float(stats.kurtosis(y))
            }
            validation_report['label_stats'] = label_stats
            
            # Check if labels are well-distributed
            if abs(label_stats['skewness']) > 5:
                validation_report['warnings'].append(
                    f"Labels are highly skewed: {label_stats['skewness']:.2f}"
                )
            
            # 6. Check for data leakage (perfect correlations)
            X = final_features.drop(columns=[label_column])
            high_corr_features = []
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    corr = abs(X[col].corr(y))
                    if corr > 0.99:  # Suspiciously high correlation
                        high_corr_features.append((col, corr))
            
            if high_corr_features:
                validation_report['warnings'].append(
                    f"Found {len(high_corr_features)} features with suspiciously high "
                    f"correlation (>0.99) - possible data leakage"
                )
                validation_report['high_correlation_features'] = [
                    {'feature': feat, 'correlation': float(corr)}
                    for feat, corr in high_corr_features
                ]
            
            # Determine if ready for training
            ready_for_training = (
                len(validation_report['errors']) == 0 and
                validation_report['checks'].get('min_samples', False) and
                validation_report['checks'].get('nan_check', False) and
                validation_report['checks'].get('inf_check', False)
            )
            
            validation_report['ready_for_training'] = ready_for_training
            
            # Save validation report
            report_path = self._save_metadata(
                validation_report,
                'final_validation_report'
            )
            
            # If validation passed, save the final dataset as training-ready
            if ready_for_training:
                training_data_path = self._save_dataframe(
                    final_features,
                    'training_ready_data',
                    metadata={'validation_passed': True}
                )
                artifacts = [report_path, training_data_path]
            else:
                artifacts = [report_path]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'validation_report': validation_report,
                'ready_for_training': ready_for_training,
                'artifacts': artifacts,
                'metrics': {
                    'total_samples': len(final_features),
                    'total_features': len(final_features.columns) - 1,
                    'nan_ratio': nan_ratio,
                    'validation_errors': len(validation_report['errors']),
                    'validation_warnings': len(validation_report['warnings']),
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    def _validate_distributions(
        self, df: pd.DataFrame, label_column: str
    ) -> Dict[str, Dict[str, float]]:
        """Validate feature distributions."""
        distribution_stats = {}
        
        for col in df.columns:
            if col == label_column:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    distribution_stats[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'skewness': float(stats.skew(col_data)),
                        'kurtosis': float(stats.kurtosis(col_data)),
                        'zeros_pct': float((col_data == 0).sum() / len(col_data))
                    }
        
        return distribution_stats
