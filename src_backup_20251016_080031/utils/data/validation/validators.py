"""Cross-step data consistency validation system."""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
import logging
import time

# Import DataValidator from ml_common for backwards compatibility
try:
    from src.utils.ml_common.validation.validation_utils import DataValidator
except ImportError:
    # Fallback DataValidator implementation
    class DataValidator:
        """Fallback DataValidator implementation."""
        def __init__(self, logger=None):
            self.logger = logger or logging.getLogger(__name__)

        def validate_dataframe(self, data, validation_level="comprehensive"):
            """Basic validation fallback."""
            return {"valid": True, "issues": [], "warnings": []}

        def validate_input_data(self, data, labels=None):
            """Validate input data for ML processing."""
            result = {
                'is_valid': True,
                'warnings': [],
                'errors': []
            }

            if data is None:
                result['is_valid'] = False
                result['errors'].append("Data is None")
                return result

            # Check for empty data
            if hasattr(data, 'shape') and data.shape[0] == 0:
                result['is_valid'] = False
                result['errors'].append("Data is empty")
                return result

            # Check minimum samples
            if hasattr(data, 'shape') and data.shape[0] < 10:
                result['warnings'].append(f"Low number of samples: {data.shape[0]}")

            # Check for NaN values
            if hasattr(data, 'isna'):
                nan_count = data.isna().sum().sum()
                if nan_count > 0:
                    result['warnings'].append(f"Found {nan_count} NaN values")

            return result

@dataclass
class DataLineage:
    """Track data lineage and transformations."""
    step_name: str
    timestamp: datetime
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    columns_added: List[str] = field(default_factory = list)
    columns_removed: List[str] = field(default_factory = list)
    columns_modified: List[str] = field(default_factory = list)
    transformations_applied: List[str] = field(default_factory = list)
    data_quality_score: float = 100.0
    metadata: Dict[str, Any] = field(default_factory = dict)

@dataclass
class ConsistencyIssue:
    """Represents a data consistency issue."""
    issue_type: str
    severity: str
    message: str
    affected_columns: List[str] = field(default_factory = list)
    suggested_fix: Optional[str] = None
    step_context: Optional[str] = None

class CrossStepValidator:
    """Validates data consistency across pipeline steps."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('CrossStepValidator')
        self.standards = PipelineStandards(self.logger)
        self.data_lineage: List[DataLineage] = []
        self.consistency_issues: List[ConsistencyIssue] = []
        self.consistency_rules = {'timestamp_continuity': {'enabled': True, 'max_gap_minutes': 60, 'tolerance_percentage': 0.05}, 'volume_consistency': {'enabled': True, 'max_volume_change_percentage': 1000.0, 'min_volume_threshold': 0.0}, 'price_range_validation': {'enabled': True, 'max_price_change_percentage': 50.0, 'min_price_threshold': 0.001}, 'column_preservation': {'enabled': True, 'critical_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'], 'allow_additional_columns': True}, 'data_shape_consistency': {'enabled': True, 'max_row_loss_percentage': 10.0, 'max_row_gain_percentage': 20.0}}
        self.step_specific_rules = {'data_reading': {'expected_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'], 'min_rows': 100, 'max_null_percentage': 0.1}, 'sr_optimization': {'expected_new_columns': ['sr_levels', 'sr_strength'], 'preserve_original_columns': True}, 'hmm_regime_discovery': {'expected_new_columns': ['regime_labels', 'regime_probabilities'], 'regime_count_range': (2, 10)}, 'regime_data_splitting': {'expected_outputs': ['regime_data'], 'preserve_data_integrity': True}, 'labeling': {'expected_new_columns': ['labels', 'label_confidence'], 'label_value_range': (-1, 0, 1)}, 'feature_engineering': {'allow_column_removal': False, 'min_new_features': 1, 'max_feature_correlation': 0.95}}
        self.logger.info('🔍 CrossStepValidator initialized with comprehensive consistency rules')

    def validate_step_transition(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, step_metadata: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Validate data consistency between pipeline steps.
        
        Args:
            from_step: Name of the source step
            to_step: Name of the destination step
            input_data: Input DataFrame
            output_data: Output DataFrame
            step_metadata: Additional step metadata
            
        Returns:
            Validation results dictionary
        """
        self.logger.info(f'🔍 Validating transition: {from_step} → {to_step}')
        validation_result = {'passed': True, 'issues': [], 'warnings': [], 'data_lineage': None, 'consistency_score': 100.0}
        try:
            lineage = self._record_data_lineage(from_step, input_data, output_data, step_metadata)
            validation_result['data_lineage'] = lineage
            consistency_checks = [self._check_timestamp_continuity, self._check_volume_consistency, self._check_price_range_validation, self._check_column_preservation, self._check_data_shape_consistency, self._check_step_specific_rules]
            total_issues = 0
            total_warnings = 0
            for check_func in consistency_checks:
                try:
                    issues, warnings = check_func(from_step, to_step, input_data, output_data, step_metadata)
                    validation_result['issues'].extend(issues)
                    validation_result['warnings'].extend(warnings)
                    total_issues += len(issues)
                    total_warnings += len(warnings)
                except Exception as e:
                    self.logger.warning(f'⚠️ Consistency check failed: {e}')
                    validation_result['warnings'].append(f'Check failed: {str(e)}')
            validation_result['consistency_score'] = self._calculate_consistency_score(total_issues, total_warnings, input_data, output_data)
            critical_issues = [issue for issue in validation_result['issues'] if issue.get('severity') == 'critical']
            if critical_issues:
                validation_result['passed'] = False
                self.logger.error(f'❌ Critical consistency issues found: {len(critical_issues)}')
            elif total_issues > 5:
                validation_result['passed'] = False
                self.logger.warning(f'⚠️ Too many consistency issues: {total_issues}')
            self.logger.info(f"✅ Consistency validation completed: score={validation_result['consistency_score']:.1f}, issues={total_issues}, warnings={total_warnings}")
            return validation_result
        except Exception as e:
            self.logger.exception(f'❌ Error in consistency validation: {e}')
            return {'passed': False, 'issues': [{'type': 'validation_error', 'message': str(e)}], 'warnings': [], 'data_lineage': None, 'consistency_score': 0.0}

    def _record_data_lineage(self, step_name: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> DataLineage:
        """Record data lineage information."""
        lineage = DataLineage(step_name = step_name, timestamp = datetime.now(), input_shape = input_data.shape, output_shape = output_data.shape, data_quality_score = self.standards.validate_data_quality(output_data, 'unified').quality_score, metadata = metadata or {})
        input_cols = set(input_data.columns)
        output_cols = set(output_data.columns)
        lineage.columns_added = list(output_cols - input_cols)
        lineage.columns_removed = list(input_cols - output_cols)
        lineage.columns_modified = list(input_cols & output_cols)
        if len(input_data) != len(output_data):
            lineage.transformations_applied.append(f'row_count_change: {len(input_data)} → {len(output_data)}')
        if lineage.columns_added:
            lineage.transformations_applied.append(f'columns_added: {lineage.columns_added}')
        if lineage.columns_removed:
            lineage.transformations_applied.append(f'columns_removed: {lineage.columns_removed}')
        self.data_lineage.append(lineage)
        return lineage

    def _check_timestamp_continuity(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Check timestamp continuity between steps."""
        issues = []
        warnings = []
        if not self.consistency_rules['timestamp_continuity']['enabled']:
            return (issues, warnings)
        if 'timestamp' not in input_data.columns or 'timestamp' not in output_data.columns:
            warnings.append({'type': 'timestamp_continuity', 'severity': 'low', 'message': 'Timestamp column not found in input or output data'})
            return (issues, warnings)
        input_timestamps = pd.to_datetime(input_data['timestamp']).sort_values()
        output_timestamps = pd.to_datetime(output_data['timestamp']).sort_values()
        if len(output_timestamps) < len(input_timestamps) * 0.9:
            issues.append({'type': 'timestamp_continuity', 'severity': 'medium', 'message': f'Significant timestamp loss: {len(input_timestamps)} → {len(output_timestamps)}', 'suggested_fix': 'Check for data filtering or aggregation issues'})
        new_timestamps = set(output_timestamps) - set(input_timestamps)
        if len(new_timestamps) > len(input_timestamps) * 0.05:
            warnings.append({'type': 'timestamp_continuity', 'severity': 'low', 'message': f'Unexpected new timestamps: {len(new_timestamps)} new timestamps found', 'suggested_fix': 'Verify timestamp generation logic'})
        return (issues, warnings)

    def _check_volume_consistency(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Check volume consistency between steps."""
        issues = []
        warnings = []
        if not self.consistency_rules['volume_consistency']['enabled']:
            return (issues, warnings)
        if 'volume' not in input_data.columns or 'volume' not in output_data.columns:
            return (issues, warnings)
        input_volume = input_data['volume']
        output_volume = output_data['volume']
        input_volume_mean = input_volume.mean()
        output_volume_mean = output_volume.mean()
        if input_volume_mean > 0:
            volume_change_percentage = abs(output_volume_mean - input_volume_mean) / input_volume_mean * 100
            max_change = self.consistency_rules['volume_consistency']['max_volume_change_percentage']
            if volume_change_percentage > max_change:
                issues.append({'type': 'volume_consistency', 'severity': 'high', 'message': f'Extreme volume change: {volume_change_percentage:.1f}%', 'suggested_fix': 'Check for volume calculation errors or data corruption'})
        negative_volumes = (output_volume < 0).sum()
        if negative_volumes > 0:
            issues.append({'type': 'volume_consistency', 'severity': 'critical', 'message': f'Negative volumes detected: {negative_volumes} rows', 'suggested_fix': 'Fix volume calculation or data source'})
        return (issues, warnings)

    def _check_price_range_validation(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Check price range validation between steps."""
        issues = []
        warnings = []
        if not self.consistency_rules['price_range_validation']['enabled']:
            return (issues, warnings)
        price_columns = ['open', 'high', 'low', 'close']
        missing_price_cols = [col for col in price_columns if col not in input_data.columns or col not in output_data.columns]
        if missing_price_cols:
            warnings.append({'type': 'price_range_validation', 'severity': 'low', 'message': f'Missing price columns: {missing_price_cols}'})
            return (issues, warnings)
        for col in price_columns:
            if col in input_data.columns and col in output_data.columns:
                input_prices = input_data[col]
                output_prices = output_data[col]
                input_price_mean = input_prices.mean()
                output_price_mean = output_prices.mean()
                if input_price_mean > 0:
                    price_change_percentage = abs(output_price_mean - input_price_mean) / input_price_mean * 100
                    max_change = self.consistency_rules['price_range_validation']['max_price_change_percentage']
                    if price_change_percentage > max_change:
                        issues.append({'type': 'price_range_validation', 'severity': 'high', 'message': f'Extreme {col} price change: {price_change_percentage:.1f}%', 'suggested_fix': 'Check for price calculation errors or data corruption'})
                negative_prices = (output_prices < 0).sum()
                if negative_prices > 0:
                    issues.append({'type': 'price_range_validation', 'severity': 'critical', 'message': f'Negative {col} prices detected: {negative_prices} rows', 'suggested_fix': 'Fix price calculation or data source'})
        return (issues, warnings)

    def _check_column_preservation(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Check column preservation between steps."""
        issues = []
        warnings = []
        if not self.consistency_rules['column_preservation']['enabled']:
            return (issues, warnings)
        critical_columns = self.consistency_rules['column_preservation']['critical_columns']
        input_cols = set(input_data.columns)
        output_cols = set(output_data.columns)
        missing_critical = [col for col in critical_columns if col in input_cols and col not in output_cols]
        if missing_critical:
            issues.append({'type': 'column_preservation', 'severity': 'critical', 'message': f'Critical columns removed: {missing_critical}', 'suggested_fix': 'Ensure critical columns are preserved in step processing'})
        removed_columns = input_cols - output_cols
        if removed_columns and (not self.consistency_rules['column_preservation']['allow_additional_columns']):
            warnings.append({'type': 'column_preservation', 'severity': 'medium', 'message': f'Columns removed: {list(removed_columns)}', 'suggested_fix': 'Verify column removal is intentional'})
        return (issues, warnings)

    def _check_data_shape_consistency(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Check data shape consistency between steps."""
        issues = []
        warnings = []
        if not self.consistency_rules['data_shape_consistency']['enabled']:
            return (issues, warnings)
        input_rows = len(input_data)
        output_rows = len(output_data)
        if input_rows > 0:
            row_loss_percentage = (input_rows - output_rows) / input_rows * 100
            max_loss = self.consistency_rules['data_shape_consistency']['max_row_loss_percentage']
            if row_loss_percentage > max_loss:
                issues.append({'type': 'data_shape_consistency', 'severity': 'high', 'message': f'Excessive row loss: {row_loss_percentage:.1f}% ({input_rows} → {output_rows})', 'suggested_fix': 'Check for data filtering or processing errors'})
        if input_rows > 0:
            row_gain_percentage = (output_rows - input_rows) / input_rows * 100
            max_gain = self.consistency_rules['data_shape_consistency']['max_row_gain_percentage']
            if row_gain_percentage > max_gain:
                warnings.append({'type': 'data_shape_consistency', 'severity': 'medium', 'message': f'Unexpected row gain: {row_gain_percentage:.1f}% ({input_rows} → {output_rows})', 'suggested_fix': 'Verify data expansion logic'})
        return (issues, warnings)

    def _check_step_specific_rules(self, from_step: str, to_step: str, input_data: pd.DataFrame, output_data: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Check step-specific validation rules."""
        issues = []
        warnings = []
        if to_step not in self.step_specific_rules:
            return (issues, warnings)
        rules = self.step_specific_rules[to_step]
        if 'expected_columns' in rules:
            expected_cols = set(rules['expected_columns'])
            actual_cols = set(output_data.columns)
            missing_cols = expected_cols - actual_cols
            if missing_cols:
                issues.append({'type': 'step_specific_rules', 'severity': 'high', 'message': f'Missing expected columns for {to_step}: {list(missing_cols)}', 'suggested_fix': f'Ensure {to_step} produces required columns'})
        if 'expected_new_columns' in rules:
            expected_new_cols = set(rules['expected_new_columns'])
            input_cols = set(input_data.columns)
            output_cols = set(output_data.columns)
            new_cols = output_cols - input_cols
            missing_new_cols = expected_new_cols - new_cols
            if missing_new_cols:
                warnings.append({'type': 'step_specific_rules', 'severity': 'medium', 'message': f'Missing expected new columns for {to_step}: {list(missing_new_cols)}', 'suggested_fix': f'Verify {to_step} feature generation logic'})
        if 'min_rows' in rules:
            min_rows = rules['min_rows']
            if len(output_data) < min_rows:
                issues.append({'type': 'step_specific_rules', 'severity': 'high', 'message': f'Insufficient rows for {to_step}: {len(output_data)} < {min_rows}', 'suggested_fix': 'Check data filtering or processing logic'})
        return (issues, warnings)

    def _calculate_consistency_score(self, total_issues: int, total_warnings: int, input_data: pd.DataFrame, output_data: pd.DataFrame) -> float:
        """Calculate overall consistency score."""
        base_score = 100.0
        issue_penalty = total_issues * 10
        warning_penalty = total_warnings * 2
        if len(input_data) > 0:
            shape_change_penalty = abs(len(output_data) - len(input_data)) / len(input_data) * 20
        else:
            shape_change_penalty = 0
        final_score = max(0.0, base_score - issue_penalty - warning_penalty - shape_change_penalty)
        return final_score

    def get_data_lineage(self) -> List[DataLineage]:
        """Get complete data lineage."""
        return self.data_lineage.copy()

    def get_consistency_issues(self) -> List[ConsistencyIssue]:
        """Get all consistency issues."""
        return self.consistency_issues.copy()

    def get_consistency_summary(self) -> Dict[str, Any]:
        """Get consistency validation summary."""
        total_issues = len(self.consistency_issues)
        critical_issues = len([issue for issue in self.consistency_issues if issue.severity == 'critical'])
        high_issues = len([issue for issue in self.consistency_issues if issue.severity == 'high'])
        return {'total_steps_validated': len(self.data_lineage), 'total_issues': total_issues, 'critical_issues': critical_issues, 'high_issues': high_issues, 'consistency_score_avg': np.mean([lineage.data_quality_score for lineage in self.data_lineage]) if self.data_lineage else 100.0, 'data_lineage_length': len(self.data_lineage)}

    def reset_validation_state(self) -> None:
        """Reset validation state."""
        self.data_lineage.clear()
        self.consistency_issues.clear()
        self.logger.info('🔄 Cross-step validation state reset')
cross_step_validator = CrossStepValidator()