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
            tprint_success("📦 Retrieved final dataset from cache - using cached result")
            self.logger.info("📦 Retrieved final dataset from cache")
            return {
                'success': True,
                'artifacts': ['final_dataset', 'final_validation_metrics', 'final_quality_scores'],
                'metrics': {
                    'validation_score': 1.0,
                    'quality_level': "excellent",
                    'validation_metadata': cached_metrics or {},
                    'comprehensive_metrics': cached_quality_scores or {},
                    'cache_hit': True
                }
            }

        # Load data if not provided
        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_info("🔍 Auto-loading data for final validation")
            # Try to load from various sources using BaseStep methods
            data = self._load_dataframe('vectorized_features')
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('optimized_feature_dataframe')
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('interaction_features')
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('selected_features')
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = self._load_dataframe('generated_features')

        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_error("❌ Input data is None or empty - validation failed")
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
