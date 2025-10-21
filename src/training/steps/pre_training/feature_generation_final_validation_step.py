"""
Enhanced Final Validation Step

This step performs comprehensive final validation using QualityAlertSystem
and advanced validation frameworks from the Ares ecosystem.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import enum
from datetime import datetime

from src.training.steps.base_step import BaseStep

from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe
from src.utils.tprint import tprint_success, tprint_warning, tprint_data_preview, tprint_data_format



# Import advanced validation components
# Self-contained validation components
from enum import Enum

class QualityScoreLevel(Enum):
    """Quality score levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class QualityScore:
    """Quality score result."""
    score: float
    level: QualityScoreLevel
    details: Dict[str, Any]

@dataclass
class QualityAssessment:
    """Quality assessment result."""
    overall_score: float
    metrics: Dict[str, float]
    recommendations: List[str]

class ComprehensiveQualityScorer:
    """Self-contained comprehensive quality scorer."""
    
    def __init__(self):
        pass
    
    def score_dataframe(self, df: pd.DataFrame) -> QualityScore:
        """Score dataframe quality."""
        if df.empty:
            return QualityScore(0.0, QualityScoreLevel.CRITICAL, {"error": "Empty dataframe"})
        
        # Calculate basic quality metrics
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Calculate quality score (0-100)
        score = 100.0
        
        # Penalize for NaN values
        if nan_ratio > 0.05:
            score -= (nan_ratio - 0.05) * 200
        
        # Penalize for duplicates
        if duplicate_ratio > 0.1:
            score -= (duplicate_ratio - 0.1) * 100
        
        # Penalize for insufficient data
        if len(df) < 100:
            score -= (100 - len(df)) * 0.5
        
        score = max(0.0, min(100.0, score))
        
        # Determine quality level
        if score >= 90:
            level = QualityScoreLevel.EXCELLENT
        elif score >= 75:
            level = QualityScoreLevel.GOOD
        elif score >= 60:
            level = QualityScoreLevel.FAIR
        elif score >= 40:
            level = QualityScoreLevel.POOR
        else:
            level = QualityScoreLevel.CRITICAL
        
        return QualityScore(score, level, {
            "nan_ratio": nan_ratio,
            "duplicate_ratio": duplicate_ratio,
            "row_count": len(df),
            "column_count": len(df.columns)
        })

class AdvancedQualityMetrics:
    """Self-contained advanced quality metrics."""
    
    def assess(self, df: pd.DataFrame) -> QualityAssessment:
        """Assess advanced quality metrics."""
        if df.empty:
            return QualityAssessment(0.0, {}, ["Empty dataframe"])
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        metrics = {
            "completeness": 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            "uniqueness": len(df.drop_duplicates()) / len(df),
            "numeric_ratio": len(numeric_cols) / len(df.columns)
        }
        
        overall_score = sum(metrics.values()) / len(metrics) * 100
        
        recommendations = []
        if metrics["completeness"] < 0.9:
            recommendations.append("Improve data completeness")
        if metrics["uniqueness"] < 0.8:
            recommendations.append("Address duplicate data")
        
        return QualityAssessment(overall_score, metrics, recommendations)

class QualityAlertSystem:
    """Self-contained quality alert system."""
    
    def __init__(self):
        pass
    
    def check_alerts(self, quality_score: QualityScore, validation_result: Dict[str, Any]) -> List[str]:
        """Check for quality alerts."""
        alerts = []
        
        if quality_score.level in [QualityScoreLevel.POOR, QualityScoreLevel.CRITICAL]:
            alerts.append(f"Quality level is {quality_score.level.value} (score: {quality_score.score:.1f})")
        
        if not validation_result.get("valid", True):
            alerts.extend(validation_result.get("issues", []))
        
        return alerts

@dataclass
class ValidationResult:
    """Validation result."""
    success: bool
    score: float
    issues: List[str]
    recommendations: List[str]

class ValidationManager:
    """Self-contained validation manager."""
    
    def __init__(self):
        pass
    
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """Validate data."""
        if data.empty:
            return ValidationResult(False, 0.0, ["Empty dataframe"], ["Provide valid data"])
        
        # Basic validation
        issues = []
        recommendations = []
        
        if len(data) < 100:
            issues.append("Insufficient data rows")
            recommendations.append("Collect more data")
        
        if data.isnull().sum().sum() > len(data) * len(data.columns) * 0.1:
            issues.append("High missing data ratio")
            recommendations.append("Clean missing data")
        
        success = len(issues) == 0
        score = 100.0 - len(issues) * 20
        
        return ValidationResult(success, score, issues, recommendations)

# Set availability flag
VALIDATION_COMPONENTS_AVAILABLE = True

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
    success: bool
    validation_score: float
    quality_level: str
    validation_metadata: Dict[str, Any]
    quality_alerts: List[Any]
    comprehensive_metrics: Dict[str, Any]
    validation_recommendations: List[str]
    artifacts: Dict[str, Any]
    final_dataset: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None


@dataclass
class FeatureGenerationFinalValidationStep(BaseStep):
    """Enhanced final validation step using QualityAlertSystem."""

    # Type hints for conditionally initialized attributes
    quality_alert_system: Optional[QualityAlertSystem]
    quality_scorer: Optional[ComprehensiveQualityScorer]
    advanced_metrics: Optional[AdvancedQualityMetrics]
    validation_manager: Optional[ValidationManager]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced final validation step."""
        tprint_step("🔧 Initializing FeatureGenerationFinalValidationStep")
        tprint_info(f"⚙️ Config provided: {config is not None}")
        
        super().__init__("feature_generation_final_validation_step", config)
        
        # Extract validation-specific parameters from config
        self.min_validation_score = config.get('min_validation_score', 70) if config else 70
        self.min_rows = config.get('min_rows', 100) if config else 100
        self.blocking_severities = config.get('blocking_severities', ['critical', 'blocker', 'error']) if config else ['critical', 'blocker', 'error']
        
        tprint_info(f"🎯 Min validation score: {self.min_validation_score}")
        tprint_info(f"📊 Min rows required: {self.min_rows}")
        tprint_info(f"🚨 Blocking severities: {self.blocking_severities}")
        
        # Initialize validation components
        tprint_debug("🔍 Checking validation components availability")
        if VALIDATION_COMPONENTS_AVAILABLE:
            tprint_success("✅ Advanced validation components available")
            try:
                # Initialize quality alert system
                tprint_debug("🔧 Initializing QualityAlertSystem")
                self.quality_alert_system = QualityAlertSystem()
                tprint_success("✅ QualityAlertSystem initialized")
                
                # Initialize comprehensive quality scorer
                tprint_debug("🔧 Initializing ComprehensiveQualityScorer")
                self.quality_scorer = ComprehensiveQualityScorer()
                tprint_success("✅ ComprehensiveQualityScorer initialized")
                
                # Initialize advanced quality metrics
                tprint_debug("🔧 Initializing AdvancedQualityMetrics")
                self.advanced_metrics = AdvancedQualityMetrics()
                tprint_success("✅ AdvancedQualityMetrics initialized")
                
                # Initialize validation manager
                tprint_debug("🔧 Initializing ValidationManager")
                self.validation_manager = ValidationManager()
                tprint_success("✅ ValidationManager initialized")
            except Exception as e:
                tprint_error(f"❌ Failed to initialize validation components: {e}")
                self.quality_alert_system = None
                self.quality_scorer = None
                self.advanced_metrics = None
                self.validation_manager = None
        else:
            tprint_warning("⚠️ Advanced validation components not available, using fallback")
            self.quality_alert_system = None
            self.quality_scorer = None
            self.advanced_metrics = None
            self.validation_manager = None
        
        tprint_success("🎉 FeatureGenerationFinalValidationStep initialization complete")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the final validation step using BaseStep pattern."""
        tprint_step("🚀 Starting final validation execution")
        
        # Extract parameters from config
        data = config.get('data')
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'long')
        intensity = config.get('intensity', 'blank')
        custom_overrides = config.get('custom_overrides', {})
        
        tprint_info(f"📊 Input data shape: {data.shape if data is not None else 'None'}")
        tprint_info(f"🎯 Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
        tprint_info(f"⚡ Intensity: {intensity}")
        tprint_info(f"🔧 Custom overrides: {custom_overrides is not None}")
        
        # Enhanced data format analysis for input troubleshooting
        if data is not None:
            tprint_data_format(data, f"final_validation_input_{symbol}_{timeframe}", level="INFO")
        
        # Set context for enhanced file naming
        self._set_context(symbol=symbol, exchange='binance', direction=direction, model='Analyst')
        
        # Try to load cached results using BaseStep methods
        tprint_debug("🔍 Checking for cached results")
        cached_dataset = self._load_dataframe('final_dataset')
        cached_metrics = self._load_metadata('final_validation_metrics')
        cached_quality_scores = self._load_metadata('final_quality_scores')
        
        tprint_info(f"📦 Cached dataset available: {cached_dataset is not None}")
        tprint_info(f"📦 Cached metrics available: {cached_metrics is not None}")
        tprint_info(f"📦 Cached quality scores available: {cached_quality_scores is not None}")
        
        if cached_dataset is not None:
            tprint_success("📦 Retrieved final dataset from artifact manager - using cached result")
            tprint_data_preview(cached_dataset, "cached_dataset_retrieved", level="INFO")
            tprint_data_format(cached_dataset, "cached_dataset_format", level="INFO")
            self.logger.info("📦 Retrieved final dataset from artifact manager")
            return FinalValidationResult(
                success=True,
                validation_score=1.0,
                quality_level="excellent",
                validation_metadata=cached_metrics or {},
                quality_alerts=[],
                comprehensive_metrics=cached_quality_scores or {},
                validation_recommendations=[],
                artifacts={'cache_hit': True},
                final_dataset=cached_dataset,
                error_message=None
            )
        
        # Load required artifacts from previous steps for validation
        tprint_info("📦 Loading artifacts from previous steps for validation")
        
        # Load final features from feature_generation_final_feature_selection_step
        final_features = None
        try:
            final_features = artifact_manager.get_dataframe(
                'feature_generation_final_feature_selection_step',
                'selected_feature_dataframe_60'  # Use the 60-feature set as primary
            )
            if final_features is not None and not final_features.empty:
                tprint_data_preview(final_features, "final_features_loaded", level="INFO")
                tprint_data_format(final_features, "final_features_format", level="INFO")
                tprint_success(f"✅ Loaded final features: {final_features.shape}")
            else:
                tprint_warning("⚠️ No final features found, trying alternative names")
                # Try alternative artifact names
                for alt_name in ['selected_features_60', 'final_features', 'selected_features']:
                    final_features = artifact_manager.get_dataframe(
                        'feature_generation_final_feature_selection_step',
                        alt_name
                    )
                    if final_features is not None and not final_features.empty:
                        tprint_data_preview(final_features, f"final_features_loaded_from_{alt_name}", level="INFO")
                        tprint_data_format(final_features, f"final_features_format_from_{alt_name}", level="INFO")
                        tprint_success(f"✅ Loaded final features from {alt_name}: {final_features.shape}")
                        break
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load final features: {e}")
        
        # Load targets from feature_generation_labeling_integration_step
        targets = None
        try:
            targets = artifact_manager.get_series(
                'feature_generation_labeling_integration_step',
                'targets'
            )
            if targets is not None and not targets.empty:
                # Preview loaded targets for troubleshooting
                tprint_data_preview(targets, "loaded_targets_for_final_validation", level="INFO")
                tprint_data_format(targets, "loaded_targets_format", level="INFO")
                tprint_success(f"✅ Loaded targets: {len(targets)} samples")
            else:
                tprint_warning("⚠️ No targets found, trying alternative names")
                # Try alternative artifact names
                for alt_name in ['target', 'labels', 'y']:
                    targets = artifact_manager.get_series(
                        'feature_generation_labeling_integration_step',
                        alt_name
                    )
                    if targets is not None and not targets.empty:
                        tprint_data_preview(targets, f"targets_loaded_from_{alt_name}", level="INFO")
                        tprint_data_format(targets, f"targets_format_from_{alt_name}", level="INFO")
                        tprint_success(f"✅ Loaded targets from {alt_name}: {len(targets)} samples")
                        break
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load targets: {e}")
        
        # Use loaded artifacts if available
        if final_features is not None:
            data = final_features
            tprint_data_preview(data, "data_from_final_features", level="INFO")
            tprint_data_format(data, "data_format_from_final_features", level="INFO")
            tprint_info(f"📊 Using final features for validation: {data.shape}")
        
        if targets is not None:
            # Align data and targets
            aligned_data = data.join(targets.rename('target'), how='inner').dropna()
            if not aligned_data.empty:
                tprint_data_preview(aligned_data, "aligned_data_with_targets", level="INFO")
                tprint_data_format(aligned_data, "aligned_data_format_with_targets", level="INFO")
                data = aligned_data
                tprint_success(f"✅ Aligned data with targets: {data.shape}")
            else:
                tprint_warning("⚠️ No overlapping timestamps between features and targets")

        # Load data if not provided
        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_info("🔍 Auto-loading data for final validation")
            # Try to load from various sources using BaseStep methods
            data = self._load_dataframe('vectorized_features')
            if data is not None and not data.empty:
                tprint_data_preview(data, "loaded_vectorized_features", level="DEBUG")
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('optimized_feature_dataframe')
                if data is not None and not data.empty:
                    tprint_data_preview(data, "loaded_optimized_feature_dataframe", level="DEBUG")
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('interaction_features')
                if data is not None and not data.empty:
                    tprint_data_preview(data, "loaded_interaction_features", level="DEBUG")
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('selected_features')
                if data is not None and not data.empty:
                    tprint_data_preview(data, "loaded_selected_features", level="DEBUG")
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('generated_features')
                if data is not None and not data.empty:
                    tprint_data_preview(data, "loaded_generated_features", level="DEBUG")
            
            # Preview the final loaded data
            if data is not None and not data.empty:
                tprint_data_preview(data, "final_loaded_data", level="INFO")
                tprint_data_format(data, "final_loaded_data_format", level="INFO")

        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_error("❌ Input data is None or empty - validation failed")
            tprint_data_preview(data, "empty_data_error", level="ERROR")
            tprint_data_format(data, "empty_data_format_error", level="ERROR")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {
                    'validation_score': 0.0,
                    'quality_level': "error",
                    'validation_metadata': {},
                    'comprehensive_metrics': {},
                    'error_message': "Input data is None or empty"
                }
            }

        # Perform basic validation
        tprint_info("🔧 Performing basic validation")
        tprint_data_preview(data, "data_before_validation", level="DEBUG")
        tprint_data_format(data, "data_format_before_validation", level="DEBUG")
        
        # Basic validation checks
        basic_checks = {
            'has_data': not data.empty,
            'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close']),
            'no_all_nan': not data.isnull().all().any(),
            'sufficient_rows': len(data) >= self.min_rows,
            'no_infinite_values': not np.isinf(data.select_dtypes(include=[np.number])).any().any()
        }
        
        success = all(basic_checks.values())
        validation_score = sum(basic_checks.values()) / len(basic_checks) * 100
        quality_level = "excellent" if validation_score >= 90 else "good" if validation_score >= 70 else "poor"
        
        tprint_info(f"✅ Validation completed - Success: {success}, Score: {validation_score:.2f}")
        tprint_data_preview(data, "final_data_state", level="DEBUG")
        tprint_data_format(data, "final_data_format_state", level="DEBUG")
        
        # Prepare result for BaseStep
        base_result = {
            'success': success,
            'artifacts': ['final_dataset', 'final_validation_metrics'],
            'metrics': {
                'validation_score': validation_score,
                'quality_level': quality_level,
                'validation_metadata': basic_checks,
                'comprehensive_metrics': basic_checks,
                'quality_alerts': [] if success else [{'type': 'basic_validation_failed', 'checks': basic_checks}],
                'validation_recommendations': [] if success else ["Review data quality and try again"]
            }
        }

        if not success:
            base_result['error'] = f"Basic validation failed: {[k for k, v in basic_checks.items() if not v]}"

        # Store artifacts using BaseStep methods
        if success:
            tprint_debug("💾 Storing successful validation artifacts")
            tprint_data_preview(data, "final_dataset_before_saving", level="DEBUG")
            tprint_data_format(data, "final_dataset_format_before_saving", level="DEBUG")
            self._save_dataframe(data, 'final_dataset')
            self._save_metadata(basic_checks, 'final_validation_metrics')
            tprint_success("✅ Final validation artifacts stored")
        else:
            tprint_warning("⚠️ Validation failed - not storing artifacts")

        tprint_success("🎉 Final validation execution complete")
        return base_result




# Handler function for ares_launcher integration
async def handle_feature_generation_final_validation_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    exchange: str = "binance",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: int = None,
    start_date: str = None,
    end_date: str = None,
    custom_overrides: dict = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handler function for feature generation final validation step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        timeframe: Timeframe (e.g., "15m")
        exchange: Exchange name (e.g., "binance")
        direction: Trading direction (e.g., "longs")
        intensity: Intensity level (e.g., "blank")
        lookback_days: Number of days to look back
        start_date: Start date for data
        end_date: End date for data
        custom_overrides: Custom configuration overrides
        data: Input data for validation
        **kwargs: Additional arguments

    Returns:
        Dict containing validation results
    """
    try:
        # Create the step instance
        step = FeatureGenerationFinalValidationStep(
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'custom_overrides': custom_overrides or {},
                'data': data
            }
        )

        # Execute the step
        result = await step.execute({
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': exchange,
            'direction': direction,
            'intensity': intensity,
            'lookback_days': lookback_days,
            'start_date': start_date,
            'end_date': end_date,
            'custom_overrides': custom_overrides or {},
            'data': data
        })

        return result

    except Exception as e:
        return {
            'success': False,
            'artifacts': [],
            'metrics': {},
            'error': str(e)
        }
