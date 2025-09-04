"""Regime Continuity Validator for Pipeline Steps.

This module validates that regime continuity is maintained throughout the pipeline,
ensuring that regime information flows correctly between steps.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass

from src.utils.logger import getChild as get_logger
from src.utils.common_operations import ensure_directory, safe_json_dump, safe_json_load
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, cached, validates, handles_errors, log_execution_time
from src.training.steps.regime_continuity_manager import regime_continuity_manager, RegimeStatus


logger = get_logger('RegimeContinuityValidator')


@dataclass
class ContinuityValidationResult:
    """Result of regime continuity validation."""
    step_name: str
    is_valid: bool
    validation_score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    validated_at: datetime


class RegimeContinuityValidator:
    """Validates regime continuity throughout the pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the regime continuity validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('RegimeContinuityValidator')
        self.standards = pipeline_standards
        
        # Validation thresholds
        self.validation_config = self.config.get('regime_continuity_validation', {
            'min_regime_coverage': 0.8,  # Minimum 80% of regimes must be processed
            'max_regime_failure_rate': 0.2,  # Maximum 20% regime failure rate
            'min_data_continuity': 0.9,  # Minimum 90% data continuity
            'max_temporal_gaps': 0.1,  # Maximum 10% temporal gaps
            'correlation_threshold': 0.7,  # Minimum correlation between regime data
            'feature_consistency_threshold': 0.8  # Minimum feature consistency
        })
    
    @traced(span_name='validate_step_continuity')
    async def validate_step_continuity(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> ContinuityValidationResult:
        """Validate regime continuity for a specific step.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Validation result
        """
        try:
            self.logger.info(f"🔍 Validating regime continuity for {step_name}")
            
            issues = []
            warnings = []
            recommendations = []
            validation_score = 1.0
            
            # Check regime coverage
            coverage_result = await self._validate_regime_coverage(step_name, symbol, exchange, timeframe, data_dir)
            if not coverage_result['is_valid']:
                issues.extend(coverage_result['issues'])
                validation_score *= 0.5
            
            # Check data continuity
            continuity_result = await self._validate_data_continuity(step_name, symbol, exchange, timeframe, data_dir)
            if not continuity_result['is_valid']:
                issues.extend(continuity_result['issues'])
                validation_score *= 0.7
            
            # Check temporal consistency
            temporal_result = await self._validate_temporal_consistency(step_name, symbol, exchange, timeframe, data_dir)
            if not temporal_result['is_valid']:
                warnings.extend(temporal_result['warnings'])
                validation_score *= 0.9
            
            # Check feature consistency
            feature_result = await self._validate_feature_consistency(step_name, symbol, exchange, timeframe, data_dir)
            if not feature_result['is_valid']:
                warnings.extend(feature_result['warnings'])
                validation_score *= 0.9
            
            # Check regime metadata consistency
            metadata_result = await self._validate_regime_metadata_consistency(step_name, symbol, exchange, timeframe, data_dir)
            if not metadata_result['is_valid']:
                warnings.extend(metadata_result['warnings'])
                validation_score *= 0.95
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, warnings, validation_score)
            
            is_valid = len(issues) == 0 and validation_score >= 0.8
            
            result = ContinuityValidationResult(
                step_name=step_name,
                is_valid=is_valid,
                validation_score=validation_score,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                validated_at=datetime.now()
            )
            
            self.logger.info(f"📊 Continuity validation for {step_name}: {'✅ PASSED' if is_valid else '❌ FAILED'} (score: {validation_score:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating continuity for {step_name}: {e}")
            return ContinuityValidationResult(
                step_name=step_name,
                is_valid=False,
                validation_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                recommendations=["Fix validation error and retry"],
                validated_at=datetime.now()
            )
    
    @traced(span_name='validate_regime_coverage')
    async def _validate_regime_coverage(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate that all regimes are properly covered.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Coverage validation result
        """
        try:
            # Get regime metadata
            if not regime_continuity_manager.regime_metadata:
                return {
                    'is_valid': False,
                    'issues': ['No regime metadata available'],
                    'coverage_rate': 0.0
                }
            
            total_regimes = len(regime_continuity_manager.regime_metadata)
            
            # Check step contexts
            if step_name not in regime_continuity_manager.step_contexts:
                return {
                    'is_valid': False,
                    'issues': [f'No step contexts found for {step_name}'],
                    'coverage_rate': 0.0
                }
            
            step_contexts = regime_continuity_manager.step_contexts[step_name]
            completed_regimes = len([ctx for ctx in step_contexts.values() if ctx.status == RegimeStatus.COMPLETED])
            failed_regimes = len([ctx for ctx in step_contexts.values() if ctx.status == RegimeStatus.FAILED])
            
            coverage_rate = completed_regimes / total_regimes if total_regimes > 0 else 0.0
            failure_rate = failed_regimes / total_regimes if total_regimes > 0 else 0.0
            
            issues = []
            if coverage_rate < self.validation_config['min_regime_coverage']:
                issues.append(f'Insufficient regime coverage: {coverage_rate:.1%} < {self.validation_config["min_regime_coverage"]:.1%}')
            
            if failure_rate > self.validation_config['max_regime_failure_rate']:
                issues.append(f'Excessive regime failure rate: {failure_rate:.1%} > {self.validation_config["max_regime_failure_rate"]:.1%}')
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'coverage_rate': coverage_rate,
                'failure_rate': failure_rate,
                'total_regimes': total_regimes,
                'completed_regimes': completed_regimes,
                'failed_regimes': failed_regimes
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime coverage: {e}")
            return {
                'is_valid': False,
                'issues': [f'Coverage validation error: {str(e)}'],
                'coverage_rate': 0.0
            }
    
    @traced(span_name='validate_data_continuity')
    async def _validate_data_continuity(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate data continuity between steps.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Data continuity validation result
        """
        try:
            training_dir = Path(data_dir) / 'training'
            
            # Check for aggregated output
            aggregated_file = training_dir / f'{exchange}_{symbol}_{timeframe}_{step_name}_aggregated.parquet'
            
            if not aggregated_file.exists():
                return {
                    'is_valid': False,
                    'issues': [f'No aggregated output found for {step_name}'],
                    'continuity_rate': 0.0
                }
            
            # Load aggregated data
            aggregated_data = pd.read_parquet(aggregated_file)
            
            # Check for regime-specific outputs
            regime_files = list(training_dir.glob(f'{exchange}_{symbol}_{timeframe}_{step_name}_regime_*.parquet'))
            regime_files.extend(list(training_dir.glob(f'{exchange}_{symbol}_{timeframe}_{step_name}_regime_*.json')))
            
            if not regime_files:
                return {
                    'is_valid': False,
                    'issues': [f'No regime-specific outputs found for {step_name}'],
                    'continuity_rate': 0.0
                }
            
            # Validate data consistency
            total_regime_data_points = 0
            for regime_file in regime_files:
                try:
                    if regime_file.suffix == '.parquet':
                        regime_data = pd.read_parquet(regime_file)
                        total_regime_data_points += len(regime_data)
                    elif regime_file.suffix == '.json':
                        with open(regime_file, 'r') as f:
                            regime_data = json.load(f)
                        # Estimate data points from JSON structure
                        if isinstance(regime_data, dict) and 'data_shape' in regime_data:
                            total_regime_data_points += regime_data['data_shape'][0]
                except Exception as e:
                    self.logger.warning(f"⚠️ Error reading regime file {regime_file}: {e}")
            
            # Calculate continuity rate
            aggregated_data_points = len(aggregated_data)
            continuity_rate = min(1.0, total_regime_data_points / aggregated_data_points) if aggregated_data_points > 0 else 0.0
            
            issues = []
            if continuity_rate < self.validation_config['min_data_continuity']:
                issues.append(f'Insufficient data continuity: {continuity_rate:.1%} < {self.validation_config["min_data_continuity"]:.1%}')
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'continuity_rate': continuity_rate,
                'aggregated_data_points': aggregated_data_points,
                'regime_data_points': total_regime_data_points,
                'regime_files_count': len(regime_files)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating data continuity: {e}")
            return {
                'is_valid': False,
                'issues': [f'Data continuity validation error: {str(e)}'],
                'continuity_rate': 0.0
            }
    
    @traced(span_name='validate_temporal_consistency')
    async def _validate_temporal_consistency(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate temporal consistency of regime data.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Temporal consistency validation result
        """
        try:
            training_dir = Path(data_dir) / 'training'
            
            # Check aggregated data for temporal consistency
            aggregated_file = training_dir / f'{exchange}_{symbol}_{timeframe}_{step_name}_aggregated.parquet'
            
            if not aggregated_file.exists():
                return {
                    'is_valid': True,  # Not applicable
                    'warnings': [],
                    'temporal_gaps': 0.0
                }
            
            aggregated_data = pd.read_parquet(aggregated_file)
            
            if 'timestamp' not in aggregated_data.columns:
                return {
                    'is_valid': True,  # Not applicable
                    'warnings': [],
                    'temporal_gaps': 0.0
                }
            
            # Check for temporal gaps
            timestamps = pd.to_datetime(aggregated_data['timestamp']).sort_values()
            time_diffs = timestamps.diff().dropna()
            
            # Calculate expected time difference based on timeframe
            expected_diff = self._get_expected_time_diff(timeframe)
            if expected_diff is None:
                return {
                    'is_valid': True,  # Cannot validate
                    'warnings': [f'Unknown timeframe: {timeframe}'],
                    'temporal_gaps': 0.0
                }
            
            # Find gaps (differences significantly larger than expected)
            gap_threshold = expected_diff * 2  # Allow for some tolerance
            gaps = time_diffs[time_diffs > gap_threshold]
            
            temporal_gaps = len(gaps) / len(time_diffs) if len(time_diffs) > 0 else 0.0
            
            warnings = []
            if temporal_gaps > self.validation_config['max_temporal_gaps']:
                warnings.append(f'Excessive temporal gaps: {temporal_gaps:.1%} > {self.validation_config["max_temporal_gaps"]:.1%}')
            
            return {
                'is_valid': len(warnings) == 0,
                'warnings': warnings,
                'temporal_gaps': temporal_gaps,
                'total_gaps': len(gaps),
                'total_intervals': len(time_diffs)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating temporal consistency: {e}")
            return {
                'is_valid': False,
                'warnings': [f'Temporal consistency validation error: {str(e)}'],
                'temporal_gaps': 1.0
            }
    
    def _get_expected_time_diff(self, timeframe: str) -> Optional[pd.Timedelta]:
        """Get expected time difference for a timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Expected time difference or None
        """
        timeframe_mapping = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }
        
        return timeframe_mapping.get(timeframe)
    
    @traced(span_name='validate_feature_consistency')
    async def _validate_feature_consistency(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate feature consistency across regimes.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Feature consistency validation result
        """
        try:
            training_dir = Path(data_dir) / 'training'
            
            # Get regime-specific feature files
            regime_feature_files = list(training_dir.glob(f'{exchange}_{symbol}_{timeframe}_{step_name}_regime_*.json'))
            
            if len(regime_feature_files) < 2:
                return {
                    'is_valid': True,  # Not enough regimes to compare
                    'warnings': [],
                    'consistency_score': 1.0
                }
            
            # Load feature information from each regime
            regime_features = {}
            for regime_file in regime_feature_files:
                try:
                    with open(regime_file, 'r') as f:
                        regime_data = json.load(f)
                    
                    # Extract regime ID from filename
                    regime_id = regime_file.stem.split('_')[-1]
                    
                    # Extract feature information
                    if 'feature_columns' in regime_data:
                        regime_features[regime_id] = set(regime_data['feature_columns'])
                    elif 'selected_features' in regime_data:
                        regime_features[regime_id] = set(regime_data['selected_features'])
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Error reading regime file {regime_file}: {e}")
            
            if len(regime_features) < 2:
                return {
                    'is_valid': True,  # Not enough regimes to compare
                    'warnings': [],
                    'consistency_score': 1.0
                }
            
            # Calculate feature consistency
            all_features = set.union(*regime_features.values())
            common_features = set.intersection(*regime_features.values())
            
            consistency_score = len(common_features) / len(all_features) if all_features else 1.0
            
            warnings = []
            if consistency_score < self.validation_config['feature_consistency_threshold']:
                warnings.append(f'Low feature consistency: {consistency_score:.1%} < {self.validation_config["feature_consistency_threshold"]:.1%}')
            
            return {
                'is_valid': len(warnings) == 0,
                'warnings': warnings,
                'consistency_score': consistency_score,
                'total_features': len(all_features),
                'common_features': len(common_features),
                'regime_count': len(regime_features)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating feature consistency: {e}")
            return {
                'is_valid': False,
                'warnings': [f'Feature consistency validation error: {str(e)}'],
                'consistency_score': 0.0
            }
    
    @traced(span_name='validate_regime_metadata_consistency')
    async def _validate_regime_metadata_consistency(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate regime metadata consistency.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Metadata consistency validation result
        """
        try:
            # Check if regime metadata is consistent
            if not regime_continuity_manager.regime_metadata:
                return {
                    'is_valid': False,
                    'warnings': ['No regime metadata available'],
                    'metadata_consistency': 0.0
                }
            
            # Check for missing regime metadata
            missing_metadata = []
            for regime_id, metadata in regime_continuity_manager.regime_metadata.items():
                if not metadata.market_characteristics:
                    missing_metadata.append(f'regime_{regime_id}')
            
            warnings = []
            if missing_metadata:
                warnings.append(f'Missing market characteristics for regimes: {missing_metadata}')
            
            # Check for inconsistent step status
            if step_name in regime_continuity_manager.step_contexts:
                step_contexts = regime_continuity_manager.step_contexts[step_name]
                inconsistent_statuses = []
                
                for regime_id, context in step_contexts.items():
                    if context.status not in [RegimeStatus.COMPLETED, RegimeStatus.FAILED, RegimeStatus.SKIPPED]:
                        inconsistent_statuses.append(f'regime_{regime_id}: {context.status.value}')
                
                if inconsistent_statuses:
                    warnings.append(f'Inconsistent step statuses: {inconsistent_statuses}')
            
            metadata_consistency = 1.0 - (len(warnings) * 0.1)  # Reduce score for each warning
            
            return {
                'is_valid': len(warnings) == 0,
                'warnings': warnings,
                'metadata_consistency': max(0.0, metadata_consistency),
                'total_regimes': len(regime_continuity_manager.regime_metadata),
                'missing_metadata_count': len(missing_metadata)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating metadata consistency: {e}")
            return {
                'is_valid': False,
                'warnings': [f'Metadata consistency validation error: {str(e)}'],
                'metadata_consistency': 0.0
            }
    
    def _generate_recommendations(
        self,
        issues: List[str],
        warnings: List[str],
        validation_score: float
    ) -> List[str]:
        """Generate recommendations based on validation results.
        
        Args:
            issues: List of issues found
            warnings: List of warnings found
            validation_score: Overall validation score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if validation_score < 0.8:
            recommendations.append("Overall validation score is low. Review and fix all issues.")
        
        if any('coverage' in issue.lower() for issue in issues):
            recommendations.append("Improve regime coverage by ensuring all regimes are processed successfully.")
        
        if any('continuity' in issue.lower() for issue in issues):
            recommendations.append("Improve data continuity by ensuring proper data flow between steps.")
        
        if any('failure' in issue.lower() for issue in issues):
            recommendations.append("Reduce regime failure rate by improving error handling and data quality.")
        
        if any('temporal' in warning.lower() for warning in warnings):
            recommendations.append("Address temporal gaps in the data to improve consistency.")
        
        if any('consistency' in warning.lower() for warning in warnings):
            recommendations.append("Improve feature consistency across regimes.")
        
        if not recommendations:
            recommendations.append("Regime continuity validation passed. Continue with next steps.")
        
        return recommendations
    
    @traced(span_name='validate_pipeline_continuity')
    async def validate_pipeline_continuity(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate regime continuity for the entire pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            steps: List of steps to validate (default: all regime-aware steps)
            
        Returns:
            Pipeline continuity validation results
        """
        try:
            if steps is None:
                steps = list(regime_continuity_manager.regime_aware_steps)
            
            self.logger.info(f"🔍 Validating pipeline continuity for {len(steps)} steps")
            
            step_results = {}
            overall_score = 0.0
            total_steps = 0
            
            for step_name in steps:
                result = await self.validate_step_continuity(
                    step_name, symbol, exchange, timeframe, data_dir
                )
                step_results[step_name] = result
                
                if result.validation_score > 0:
                    overall_score += result.validation_score
                    total_steps += 1
            
            overall_score = overall_score / total_steps if total_steps > 0 else 0.0
            
            # Generate pipeline-level recommendations
            all_issues = []
            all_warnings = []
            for result in step_results.values():
                all_issues.extend(result.issues)
                all_warnings.extend(result.warnings)
            
            pipeline_recommendations = self._generate_recommendations(all_issues, all_warnings, overall_score)
            
            pipeline_result = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'validated_at': datetime.now().isoformat(),
                'overall_score': overall_score,
                'total_steps': len(steps),
                'validated_steps': total_steps,
                'step_results': {name: {
                    'is_valid': result.is_valid,
                    'validation_score': result.validation_score,
                    'issues_count': len(result.issues),
                    'warnings_count': len(result.warnings)
                } for name, result in step_results.items()},
                'pipeline_issues': all_issues,
                'pipeline_warnings': all_warnings,
                'pipeline_recommendations': pipeline_recommendations
            }
            
            # Save validation results
            await self._save_validation_results(pipeline_result, symbol, exchange, timeframe, data_dir)
            
            self.logger.info(f"📊 Pipeline continuity validation completed: {'✅ PASSED' if overall_score >= 0.8 else '❌ FAILED'} (score: {overall_score:.2f})")
            
            return pipeline_result
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating pipeline continuity: {e}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'validated_at': datetime.now().isoformat()
            }
    
    async def _save_validation_results(
        self,
        validation_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> None:
        """Save validation results to file.
        
        Args:
            validation_results: Validation results
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            training_dir = Path(data_dir) / 'training'
            training_dir.mkdir(parents=True, exist_ok=True)
            
            validation_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_continuity_validation.json'
            
            safe_json_dump(validation_results, validation_file)
            
            self.logger.info(f"✅ Saved validation results: {validation_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving validation results: {e}")


# Global instance
regime_continuity_validator = RegimeContinuityValidator()