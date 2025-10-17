"""
Enhanced Final Validation Step

This step performs comprehensive final validation using QualityAlertSystem
and advanced validation frameworks from the Ares ecosystem.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import enum

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import advanced validation components
try:
    from src.utils.data.quality.quality_alert_system import QualityAlertSystem
    from src.utils.data.quality.comprehensive_quality_scorer import (
        ComprehensiveQualityScorer, QualityScore, QualityScoreLevel
    )
    from src.utils.data.quality.advanced_quality_metrics import (
        AdvancedQualityMetrics, QualityAssessment
    )
    from src.utils.ml_common.validation import (
        ValidationManager, ValidationResult
    )
    VALIDATION_COMPONENTS_AVAILABLE = True
except ImportError:
    VALIDATION_COMPONENTS_AVAILABLE = False
    QualityAlertSystem = None
    ComprehensiveQualityScorer = None
    QualityScore = None
    QualityScoreLevel = None
    AdvancedQualityMetrics = None
    QualityAssessment = None
    ValidationManager = None
    ValidationResult = None

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result
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
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)

def make_json_safe(obj: Any) -> Any:
    """
    Convert objects to JSON-safe format by handling common serialization issues.
    
    Args:
        obj: Object to convert to JSON-safe format
        
    Returns:
        JSON-safe version of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, enum.Enum):
        return obj.value
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        # Convert object to dict and make it JSON-safe
        return make_json_safe(obj.__dict__)
    else:
        # For other types, try to convert to string
        return str(obj)

@dataclass
class FinalValidationResult:
    """Enhanced result of final validation step."""

    success: bool
    validation_score: float
    quality_level: str
    validation_metadata: Dict[str, Any]
    quality_alerts: List[Dict[str, Any]]
    comprehensive_metrics: Dict[str, Any]
    validation_recommendations: List[str]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationFinalValidationStep(BasePreTrainingComponent):
    """Enhanced final validation step using QualityAlertSystem."""

    # Type hints for conditionally initialized attributes
    quality_alert_system: Optional[QualityAlertSystem]
    quality_scorer: Optional[ComprehensiveQualityScorer]
    advanced_metrics: Optional[AdvancedQualityMetrics]
    validation_manager: Optional[ValidationManager]

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the enhanced final validation step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Extract validation-specific parameters from config
        custom_params = self.config.custom_params or {}
        self.min_validation_score = custom_params.get('min_validation_score', 70)
        self.min_rows = custom_params.get('min_rows', 100)
        self.blocking_severities = custom_params.get('blocking_severities', ['critical', 'blocker', 'error'])
        
        # Initialize validation components
        if VALIDATION_COMPONENTS_AVAILABLE:
            # Initialize quality alert system
            self.quality_alert_system = QualityAlertSystem()
            
            # Initialize comprehensive quality scorer
            self.quality_scorer = ComprehensiveQualityScorer()
            
            # Initialize advanced quality metrics
            self.advanced_metrics = AdvancedQualityMetrics()
            
            # Initialize validation manager
            self.validation_manager = ValidationManager()
        else:
            tprint_warning("⚠️ Advanced validation components not available, using fallback")
            self.quality_alert_system = None
            self.quality_scorer = None
            self.advanced_metrics = None
            self.validation_manager = None

    async def execute(self,
                     training_input: Dict[str, Any],
                     pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute enhanced final validation step using QualityAlertSystem.
        
        Uses advanced validation components when available, falls back to basic validation otherwise.
        
        Returns:
            ComponentResult with success status, artifacts, and metadata containing:
            - validation_score: Overall quality score (0-100)
            - quality_level: Quality level assessment
            - validation_metadata: Detailed validation results
            - quality_alerts: List of quality issues found
            - comprehensive_metrics: Detailed metrics breakdown
            - validation_recommendations: Actionable recommendations
        """

        tprint_info("🔍 Starting enhanced final validation step with comprehensive quality assessment")

        # Extract parameters from training_input
        data = training_input.get('data')
        symbol = training_input.get('symbol', 'ETHUSDT')
        timeframe = training_input.get('timeframe', '15m')
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = training_input.get('exchange', 'binance')
        custom_overrides = training_input.get('custom_overrides')

        try:
            if not VALIDATION_COMPONENTS_AVAILABLE:
                # Fallback to basic validation
                return await self._fallback_final_validation(data, training_input, pipeline_state)

            # Perform comprehensive final validation
            validation_result = await self._perform_enhanced_final_validation(
                data, symbol, timeframe, direction, custom_overrides
            )

            # Convert result to ComponentResult
            component_result = ComponentResult(
                success=validation_result.success,
                artifacts=validation_result.artifacts,
                metadata={
                    'validation_score': validation_result.validation_score,
                    'quality_level': validation_result.quality_level,
                    'validation_metadata': validation_result.validation_metadata,
                    'quality_alerts': validation_result.quality_alerts,
                    'comprehensive_metrics': validation_result.comprehensive_metrics,
                    'validation_recommendations': validation_result.validation_recommendations
                },
                error_message=validation_result.error_message
            )

            if component_result.success:
                tprint_success("✅ Enhanced final validation completed successfully")
                tprint_info(f"📊 Validation Score: {validation_result.validation_score:.3f} ({validation_result.quality_level})")
                tprint_info(f"🚨 Quality Alerts: {len(validation_result.quality_alerts)}")
                tprint_info(f"💡 Recommendations: {len(validation_result.validation_recommendations)}")
            else:
                tprint_error(f"❌ Final validation failed: {component_result.error_message}")

            return component_result

        except Exception as e:
            tprint_error(f"❌ Enhanced final validation step failed with exception: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=str(e)
            )

    async def _perform_enhanced_final_validation(self, data: pd.DataFrame, symbol: str,
                                                  timeframe: str, direction: str,
                                                  custom_overrides: Optional[Dict[str, Any]]) -> FinalValidationResult:
        """Perform enhanced final validation using QualityAlertSystem."""
        
        try:
            # Step 1: Comprehensive quality scoring
            quality_score = self.quality_scorer.score_data_quality(
                data, 
                symbol=symbol,
                timeframe=timeframe,
                direction=direction
            )
            
            # Step 2: Advanced quality metrics assessment
            advanced_assessment = self.advanced_metrics.assess_data_quality(data)
            
            # Step 3: Quality alert system check
            quality_alerts = self.quality_alert_system.check_quality_alerts(data, quality_score)
            
            # Step 4: Comprehensive validation using validation manager
            validation_result = await self.validation_manager.perform_comprehensive_validation(
                data, symbol=symbol, timeframe=timeframe, direction=direction
            )
            
            # Step 5: Generate recommendations
            recommendations = self._generate_validation_recommendations(
                quality_score, advanced_assessment, quality_alerts, validation_result
            )
            
            # Determine overall success and quality level
            # Only consider configured severity levels as blocking
            blocking_alerts = [alert for alert in quality_alerts 
                             if hasattr(alert, 'severity') and 
                             alert.severity in self.blocking_severities]
            
            success = (quality_score.overall_score >= self.min_validation_score and 
                      len(blocking_alerts) == 0 and 
                      validation_result.success)
            
            quality_level = quality_score.level.value if quality_score.level else "unknown"
            
            # Compile comprehensive result
            return FinalValidationResult(
                success=success,
                validation_score=quality_score.overall_score,
                quality_level=quality_level,
                validation_metadata={
                    'quality_score_details': make_json_safe(quality_score),
                    'advanced_assessment': make_json_safe(advanced_assessment),
                    'validation_result': make_json_safe(validation_result)
                },
                quality_alerts=[make_json_safe(alert) for alert in quality_alerts],
                comprehensive_metrics={
                    'quality_breakdown': quality_score.component_scores,
                    'advanced_metrics': advanced_assessment.metrics,
                    'validation_metrics': validation_result.metrics
                },
                validation_recommendations=recommendations,
                artifacts={
                    'quality_score': make_json_safe(quality_score),
                    'advanced_assessment': make_json_safe(advanced_assessment),
                    'quality_alerts': [make_json_safe(alert) for alert in quality_alerts],
                    'validation_result': make_json_safe(validation_result)
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Enhanced final validation failed: {e}")
            return FinalValidationResult(
                success=False,
                validation_score=0.0,
                quality_level="error",
                validation_metadata={},
                quality_alerts=[],
                comprehensive_metrics={},
                validation_recommendations=["Check data format and try again"],
                artifacts={},
                error_message=str(e)
            )

    def _generate_validation_recommendations(self, quality_score, advanced_assessment, 
                                              quality_alerts, validation_result) -> List[str]:
        """Generate validation recommendations based on assessment results."""
        recommendations = []
        
        # Quality score recommendations with specific guidance
        if quality_score.overall_score < 80:
            recommendations.append(f"Data quality score {quality_score.overall_score:.1f} is below 80 - review data completeness and accuracy")
            if hasattr(quality_score, 'component_scores'):
                low_scores = [(k, v) for k, v in quality_score.component_scores.items() if v < 70]
                if low_scores:
                    recommendations.append(f"Focus on improving: {', '.join([f'{k} ({v:.1f})' for k, v in low_scores])}")
        
        # Alert-based recommendations with specific alert details
        if quality_alerts:
            recommendations.append(f"Address {len(quality_alerts)} quality alerts")
            # Include top 2-3 specific alerts for guidance
            top_alerts = quality_alerts[:3]
            for i, alert in enumerate(top_alerts, 1):
                alert_type = getattr(alert, 'type', 'unknown')
                alert_message = getattr(alert, 'message', 'No details available')
                recommendations.append(f"  {i}. {alert_type}: {alert_message}")
        
        # Advanced assessment recommendations with specific issues
        if hasattr(advanced_assessment, 'issues') and advanced_assessment.issues:
            recommendations.append(f"Resolve {len(advanced_assessment.issues)} data issues")
            # Include top 2-3 specific issues
            top_issues = advanced_assessment.issues[:3]
            for i, issue in enumerate(top_issues, 1):
                issue_desc = str(issue) if not isinstance(issue, dict) else issue.get('description', str(issue))
                recommendations.append(f"  {i}. {issue_desc}")
        
        # Validation result recommendations with specific failures
        if not validation_result.success:
            recommendations.append("Review validation failures and data integrity")
            if hasattr(validation_result, 'failures') and validation_result.failures:
                for i, failure in enumerate(validation_result.failures[:2], 1):
                    failure_desc = str(failure) if not isinstance(failure, dict) else failure.get('description', str(failure))
                    recommendations.append(f"  {i}. Validation failure: {failure_desc}")
        
        return recommendations

    async def _fallback_final_validation(self, data: pd.DataFrame, training_input: Dict[str, Any],
                                         pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Fallback final validation when advanced components are not available."""
        
        try:
            # Basic validation checks
            basic_checks = {
                'has_data': not data.empty,
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_all_nan': not data.isnull().all().any(),
                'sufficient_rows': len(data) >= self.min_rows,
                'no_infinite_values': not np.isinf(data.select_dtypes(include=[np.number])).any().any()
            }
            
            # Identify failing checks
            failing_checks = [check_name for check_name, passed in basic_checks.items() if not passed]
            
            success = all(basic_checks.values())
            validation_score = sum(basic_checks.values()) / len(basic_checks) * 100
            
            # Generate specific recommendations based on failing checks
            recommendations = []
            if not success:
                if 'has_data' in failing_checks:
                    recommendations.append("Ensure data is loaded and not empty")
                if 'has_required_columns' in failing_checks:
                    missing_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in data.columns]
                    recommendations.append(f"Add missing required columns: {missing_cols}")
                if 'no_all_nan' in failing_checks:
                    nan_cols = data.columns[data.isnull().all()].tolist()
                    recommendations.append(f"Remove or fix columns with all NaN values: {nan_cols}")
                if 'sufficient_rows' in failing_checks:
                    recommendations.append(f"Ensure at least {self.min_rows} rows of data (current: {len(data)})")
                if 'no_infinite_values' in failing_checks:
                    recommendations.append("Remove infinite values from numeric columns")
                recommendations.append("Install validation components for enhanced assessment")
            
            return ComponentResult(
                success=success,
                artifacts={'basic_checks': basic_checks, 'failing_checks': failing_checks},
                metadata={
                    'validation_score': validation_score,
                    'quality_level': 'basic',
                    'validation_metadata': {'method': 'fallback_basic', 'failing_checks': failing_checks},
                    'quality_alerts': [] if success else [{'type': 'basic_validation_failed', 'failing_checks': failing_checks}],
                    'comprehensive_metrics': basic_checks,
                    'validation_recommendations': recommendations
                },
                error_message=None if success else f"Basic validation failed: {', '.join(failing_checks)}"
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=str(e)
            )

    # Required utility methods for BasePreTrainingComponent
    def safe_dataframe_operation(self, operation_func, *args, **kwargs):
        """Safe dataframe operation wrapper."""
        return safe_dataframe_operation(operation_func, *args, **kwargs)

    def safe_matrix_multiply(self, a, b):
        """Safe matrix multiplication."""
        return safe_matrix_multiply(a, b)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize dataframe for matrix operations."""
        return optimize_dataframe(df)

# Command handler for ares_launcher integration
async def handle_feature_generation_final_validation_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ComponentResult:
    """
    Handle feature generation final validation step command.

    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments

    Returns:
        ComponentResult with validation results
    """
    # Create sample data for validation (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Create step instance and execute
    step = FeatureGenerationFinalValidationStep()

    # Build proper training_input dict and pipeline_state
    training_input = {
        'data': sample_data,
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
    
    pipeline_state = {}

    return await step.execute(training_input, pipeline_state)

# Register component with factory
def _register_feature_generation_final_validation_step():
    """Register the feature generation final validation step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_final_validation_step',
            FeatureGenerationFinalValidationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_final_validation_step()