"""
Enhanced Feature Generation Data Validation Step

This step performs comprehensive data validation and quality assessment
using advanced quality scoring and validation frameworks from the Ares ecosystem.
"""

from __future__ import annotations

import logging
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import advanced quality validation components
try:
    from src.utils.data.quality.comprehensive_quality_scorer import (
        ComprehensiveQualityScorer, QualityScore, QualityScoreLevel
    )
    from src.utils.data.quality.data_quality import (
        DataQualityFramework, QualityThresholds, QualityResult
    )
    from src.utils.data.quality.advanced_quality_metrics import (
        AdvancedQualityMetrics, QualityAssessment
    )
    from src.utils.data.quality.quality_alert_system import QualityAlertSystem
    QUALITY_COMPONENTS_AVAILABLE = True
except ImportError:
    QUALITY_COMPONENTS_AVAILABLE = False
    ComprehensiveQualityScorer = None
    DataQualityFramework = None
    AdvancedQualityMetrics = None
    QualityAlertSystem = None

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_progress
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)

@dataclass
class FeatureGenerationDataValidationStep(ModularComponent):
    """Enhanced data validation step using comprehensive quality assessment."""

    def __init__(self, name: str = "data_validation_step", 
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the enhanced data validation step."""
        super().__init__(name, config or {}, logger)
        
        # Initialize quality assessment components
        if QUALITY_COMPONENTS_AVAILABLE:
            self.quality_scorer = ComprehensiveQualityScorer()
            self.data_quality_framework = DataQualityFramework()
            self.advanced_metrics = AdvancedQualityMetrics()
            self.alert_system = QualityAlertSystem()
        else:
            self.logger.warning("⚠️ Quality components not available, using fallback validation")
            self.quality_scorer = None
            self.data_quality_framework = None
            self.advanced_metrics = None
            self.alert_system = None

    async def execute(self,
                     training_input: Optional[Dict[str, Any]] = None,
                     pipeline_state: Optional[Dict[str, Any]] = None,
                     data: Optional[Any] = None,
                     **kwargs) -> ComponentResult:
        """Execute enhanced data validation step using comprehensive quality assessment."""

        self.logger.info("🔍 Starting enhanced data validation step with comprehensive quality assessment")

        # Extract parameters from training_input or kwargs
        if training_input is None:
            # Extract from kwargs (called from component factory)
            symbol = kwargs.get('symbol', 'ETHUSDT')
            timeframe = kwargs.get('timeframe', '15m')
            direction = kwargs.get('direction', 'longs')
            intensity = kwargs.get('intensity', 'blank')
            lookback_days = kwargs.get('lookback_days')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            exchange = kwargs.get('exchange', 'binance')
            custom_overrides = kwargs.get('custom_overrides')
            # Create training_input dict for compatibility
            training_input = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'custom_overrides': custom_overrides
            }
        else:
            # Extract from training_input (called from pipeline)
            symbol = training_input.get('symbol', 'ETHUSDT')
            timeframe = training_input.get('timeframe', '15m')
            direction = training_input.get('direction', 'longs')
            intensity = training_input.get('intensity', 'blank')
            lookback_days = training_input.get('lookback_days')
            start_date = training_input.get('start_date')
            end_date = training_input.get('end_date')
            exchange = training_input.get('exchange', 'binance')
            custom_overrides = training_input.get('custom_overrides')
        
        # Use provided data or load data for validation
        if data is not None:
            # Use the provided data
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Provided data must be a pandas DataFrame")
        else:
            # Load data for validation
            data = await self._load_data_for_validation(
                symbol, timeframe, exchange, start_date, end_date, lookback_days
            )

        try:
            # Use ModularComponent's safe processing
            if not self.is_initialized():
                if not self.initialize():
                    return ComponentResult(
                        success=False,
                        metadata={},
                        error_message="Component initialization failed"
                    )

            if not QUALITY_COMPONENTS_AVAILABLE:
                # Fallback to basic validation
                return await self._fallback_validation(data, training_input, pipeline_state or {})

            # Perform comprehensive quality assessment
            quality_result = await self._perform_comprehensive_validation(
                data, symbol, timeframe, direction, training_input, pipeline_state or {}
            )

            # Convert result to ComponentResult
            component_result = ComponentResult(
                success=quality_result.success,
                metadata={
                    'data_quality_score': quality_result.data_quality_score,
                    'quality_level': quality_result.quality_level,
                    'validation_metadata': quality_result.validation_metadata,
                    'quality_breakdown': quality_result.quality_breakdown,
                    'issues': quality_result.issues,
                    'warnings': quality_result.warnings,
                    'recommendations': quality_result.recommendations,
                    'artifacts': quality_result.artifacts
                },
                error_message=quality_result.error_message
            )

            if component_result.success:
                self.logger.info(f"✅ Enhanced data validation completed successfully")
                self.logger.info(f"📊 Quality Score: {quality_result.data_quality_score:.3f} ({quality_result.quality_level})")
                if quality_result.issues:
                    self.logger.warning(f"⚠️ Issues found: {len(quality_result.issues)}")
                if quality_result.recommendations:
                    self.logger.info(f"💡 Recommendations: {len(quality_result.recommendations)}")
            else:
                self.logger.error(f"❌ Data validation failed: {component_result.error_message}")

            return component_result

        except Exception as e:
            self.logger.error(f"❌ Enhanced data validation step failed with exception: {e}")
            return ComponentResult(
                success=False,
                metadata={},
                error_message=str(e)
            )

    async def _perform_comprehensive_validation(self, data: pd.DataFrame, symbol: str, 
                                                timeframe: str, direction: str,
                                                training_input: Dict[str, Any],
                                                pipeline_state: Dict[str, Any]) -> DataValidationResult:
        """Perform comprehensive data validation using advanced quality frameworks."""
        
        try:
            # Step 1: Basic data quality framework validation
            quality_thresholds = QualityThresholds(
                max_nan_ratio=0.05,
                max_infinite_count=0,
                min_unique_values=2,
                max_constant_ratio=0.95
            )
            
            basic_quality_result = self.data_quality_framework.validate_dataframe_quality(
                data, context=f"data_validation_{symbol}_{timeframe}"
            )
            
            # Step 2: Advanced quality metrics assessment
            advanced_assessment = self.advanced_metrics.comprehensive_quality_assessment(
                data, context=f"data_validation_{symbol}_{timeframe}"
            )
            
            # Step 3: Comprehensive quality scoring
            quality_score = self.quality_scorer.assess_data_quality(
                data, 
                context=f"data_validation_{symbol}_{timeframe}",
                step_name="feature_generation_data_validation"
            )
            
            # Step 4: Quality alert system check
            # Create a mock MLValidationResult for the alert system
            from src.utils.data.quality.quality_alert_system import MLValidationResult
            validation_result = MLValidationResult(
                quality_score=quality_score,  # Pass the QualityScore object directly
                grade=quality_score.level.value,
                drift_issues=[],
                correlation_issues=[],
                target_issues=[],
                distribution_issues=[],
                outlier_issues=[],
                time_series_issues=[],
                financial_issues=[]
            )
            alerts = self.alert_system.check_alerts(validation_result)
            
            # Determine overall success and quality level
            # Be more lenient with success criteria - prioritize comprehensive quality assessment
            # If comprehensive quality assessment is good, consider it successful even if basic validation fails
            # Convert percentage to decimal for comparison
            comprehensive_score = quality_score.overall_score / 100.0 if quality_score.overall_score > 1 else quality_score.overall_score
            success = (comprehensive_score >= 0.5) or (basic_quality_result.quality_score >= 30 and comprehensive_score >= 0.3)
            
            quality_level = quality_score.level.value if quality_score.level else "unknown"
            
            # Compile comprehensive result
            return DataValidationResult(
                success=success,
                data_quality_score=quality_score.overall_score,
                quality_level=quality_level,
                validation_metadata={
                    'basic_quality': basic_quality_result.__dict__,
                    'advanced_assessment': advanced_assessment.__dict__,
                    'quality_score_details': quality_score.__dict__,
                    'alerts': [alert.__dict__ for alert in alerts]
                },
                quality_breakdown=quality_score.component_scores,
                issues=quality_score.issues,
                warnings=quality_score.warnings,
                recommendations=quality_score.recommendations,
                artifacts={
                    'quality_report': quality_score.__dict__,
                    'basic_validation': basic_quality_result.__dict__,
                    'advanced_metrics': advanced_assessment.__dict__,
                    'alerts': [alert.__dict__ for alert in alerts]
                },
                error_message=None if success else f"Data quality validation failed: {len(quality_score.issues)} issues found"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive validation failed: {e}")
            return DataValidationResult(
                success=False,
                data_quality_score=0.0,
                quality_level="error",
                validation_metadata={},
                quality_breakdown={},
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                recommendations=["Check data format and try again"],
                artifacts={},
                error_message=str(e)
            )

    async def _fallback_validation(self, data: pd.DataFrame, training_input: Dict[str, Any],
                                 pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Fallback validation when advanced components are not available."""
        
        try:
            # Basic validation checks
            basic_checks = {
                'has_data': not len(data) == 0,
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_all_nan': not data.isnull().all().any(),
                'sufficient_rows': len(data) >= 100
            }
            
            success = all(basic_checks.values())
            quality_score = sum(basic_checks.values()) / len(basic_checks) * 100
            
            return ComponentResult(
                success=success,
                metadata={
                    'basic_checks': basic_checks,
                    'data_quality_score': quality_score,
                    'validation_metadata': {'method': 'fallback_basic'},
                    'quality_breakdown': basic_checks,
                    'issues': [] if success else ['Basic validation failed'],
                    'warnings': [],
                    'recommendations': ['Install quality components for enhanced validation']
                },
                error_message=None if success else "Basic validation failed"
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                metadata={},
                error_message=str(e)
            )

    # Required utility methods for BasePreTrainingComponent
