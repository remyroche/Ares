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

from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)
from src.training.common.component_result import ComponentResult
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe
from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactKeys,
)

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
class FeatureGenerationFinalValidationStep(ModularComponent):
    """Enhanced final validation step using QualityAlertSystem."""

    # Type hints for conditionally initialized attributes
    quality_alert_system: Optional[QualityAlertSystem]
    quality_scorer: Optional[ComprehensiveQualityScorer]
    advanced_metrics: Optional[AdvancedQualityMetrics]
    validation_manager: Optional[ValidationManager]

    def __init__(self, name: str = "final_validation_step", 
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the enhanced final validation step."""
        tprint_step("🔧 Initializing FeatureGenerationFinalValidationStep")
        tprint_info(f"📝 Step name: {name}")
        tprint_info(f"⚙️ Config provided: {config is not None}")
        tprint_info(f"📋 Logger provided: {logger is not None}")
        
        super().__init__(name, config or {}, logger)
        
        # Extract validation-specific parameters from config
        self.min_validation_score = self.get_config('min_validation_score', 70)
        self.min_rows = self.get_config('min_rows', 100)
        self.blocking_severities = self.get_config('blocking_severities', ['critical', 'blocker', 'error'])
        
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

    def _initialize_resources(self) -> bool:
        """Initialize validation components."""
        tprint_step("🔧 Initializing validation resources")
        try:
            if VALIDATION_COMPONENTS_AVAILABLE:
                tprint_debug("🔍 Advanced validation components available, initializing...")
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
            else:
                tprint_warning("⚠️ Advanced validation components not available, using fallback")
                self.quality_alert_system = None
                self.quality_scorer = None
                self.advanced_metrics = None
                self.validation_manager = None
            
            self.set_state('initialized_at', time.time())
            tprint_success("🎉 Validation resources initialization complete")
            return True
        except Exception as e:
            tprint_error(f"❌ Failed to initialize validation components: {e}")
            self.logger.error(f"Failed to initialize validation components: {e}")
            return False

    async def execute(
        self,
        data: pd.DataFrame,
        symbol: str = "ETHUSDT",
        timeframe: str = "15m",
        direction: str = "longs",
        intensity: str = "blank",
        custom_overrides: Optional[Dict[str, Any]] = None,
    ) -> FinalValidationResult:
        tprint_step("🚀 Starting final validation execution")
        tprint_info(f"📊 Input data shape: {data.shape if data is not None else 'None'}")
        tprint_info(f"🎯 Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
        tprint_info(f"⚡ Intensity: {intensity}")
        tprint_info(f"🔧 Custom overrides: {custom_overrides is not None}")
        
        # Get artifact manager
        tprint_debug("🔍 Getting artifact manager")
        artifact_manager = get_pretraining_artifact_manager()
        tprint_success("✅ Artifact manager retrieved")
        
        # Try to load from artifact manager first (pretraining store)
        tprint_debug("🔍 Checking for cached results")
        cached_dataset = artifact_manager.get_artifact('final_validation', 'final_dataset')
        cached_metrics = artifact_manager.get_artifact('final_validation', 'final_validation_metrics')
        cached_quality_scores = artifact_manager.get_artifact('final_validation', 'final_quality_scores')
        # Backwards-compatible step name
        if cached_dataset is None:
            cached_dataset = artifact_manager.get_artifact('feature_generation_final_validation_step', 'final_dataset')
            cached_metrics = cached_metrics or artifact_manager.get_artifact('feature_generation_final_validation_step', 'final_validation_metrics')
            cached_quality_scores = cached_quality_scores or artifact_manager.get_artifact('feature_generation_final_validation_step', 'final_quality_scores')
        # Final fallback: enhanced artifact manager cache
        if cached_dataset is None:
            try:
                from src.utils.artifact_manager import ArtifactManager as _EnhancedAM
                _enh = _EnhancedAM(config={})
                cached_dataset = _enh.retrieve_enhanced(ArtifactKeys.FINAL_DATASET)
                cached_metrics = cached_metrics or _enh.retrieve_enhanced(ArtifactKeys.FINAL_VALIDATION_METRICS)
                cached_quality_scores = cached_quality_scores or _enh.retrieve_enhanced(ArtifactKeys.FINAL_QUALITY_SCORES)
            except Exception:
                pass
        
        tprint_info(f"📦 Cached dataset available: {cached_dataset is not None}")
        tprint_info(f"📦 Cached metrics available: {cached_metrics is not None}")
        tprint_info(f"📦 Cached quality scores available: {cached_quality_scores is not None}")
        
        if cached_dataset is not None:
            tprint_success("📦 Retrieved final dataset from artifact manager - using cached result")
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

        if data is None or (hasattr(data, 'empty') and data.empty):
            # Auto-load from vectorization outputs
            tprint_info("🔍 Auto-loading vectorized features for final validation")
            data = artifact_manager.get_dataframe('feature_generation_vectorization_step', 'vectorized_features')
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = artifact_manager.get_dataframe('vectorization', 'vectorized_features')

        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_info("🔍 Vectorized features unavailable; combining period + interaction features")
            period_features = artifact_manager.get_dataframe('feature_generation_period_lookback_optimization_step', ArtifactKeys.OPTIMIZED_FEATURE_DATAFRAME)
            interaction_features = artifact_manager.get_dataframe('feature_generation_interaction_generation_step', ArtifactKeys.INTERACTION_FEATURES)
            frames = [df for df in (period_features, interaction_features) if isinstance(df, pd.DataFrame) and not df.empty]
            if frames:
                combined = pd.concat(frames, axis=1, join='outer')
                combined = combined.loc[:, ~combined.columns.duplicated()]
                combined = combined.dropna(axis=0, how='all')
                data = combined
                tprint_success(f"✅ Combined period + interaction features for validation: shape={data.shape}")

        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_info("⚠️ Falling back to selected/generation artifacts for validation")
            data = artifact_manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
            if data is None or (hasattr(data, 'empty') and data.empty):
                data = artifact_manager.get_dataframe('feature_generation_feature_generation_step', ArtifactKeys.FEATURE_DATAFRAME)
        if data is None or (hasattr(data, 'empty') and data.empty):
            tprint_error("❌ Input data is None or empty - validation failed")
            return FinalValidationResult(
                success=False,
                validation_score=0.0,
                quality_level="error",
                validation_metadata={},
                quality_alerts=[],
                comprehensive_metrics={},
                validation_recommendations=[],
                artifacts={},
                final_dataset=data,
                error_message="Input data is None or empty"
            )

        if VALIDATION_COMPONENTS_AVAILABLE:
            tprint_info("🔧 Using enhanced validation workflow")
            result = await self._perform_enhanced_final_validation(
                data, symbol, timeframe, direction, custom_overrides
            )
        else:
            tprint_info("🔧 Using fallback validation workflow")
            result = await self._fallback_final_validation(data)

        tprint_info(f"✅ Validation completed - Success: {result.success}, Score: {result.validation_score:.2f}")
        
        if result.final_dataset is None:
            tprint_debug("🔧 Setting final dataset to input data")
            result.final_dataset = data
        result.artifacts.setdefault(ArtifactKeys.FINAL_DATASET, result.final_dataset)

        # Store artifacts in artifact manager
        if result.success:
            tprint_debug("💾 Storing successful validation artifacts")
            artifact_manager.save(
                step_name='final_validation',
                artifacts={
                    'final_dataset': result.final_dataset,
                    'final_validation_metrics': result.validation_metadata,
                    'final_quality_scores': result.comprehensive_metrics,
                    'final_validation_warnings': result.quality_alerts
                },
                metadata={
                    'step': 'final_validation',
                    'shape': result.final_dataset.shape if hasattr(result.final_dataset, 'shape') else None,
                    'created_at': datetime.now().isoformat()
                }
            )
            artifact_manager.save(
                step_name='feature_generation_final_validation_step',
                artifacts={
                    'final_dataset': result.final_dataset,
                    'final_validation_metrics': result.validation_metadata,
                    'final_quality_scores': result.comprehensive_metrics,
                    'final_validation_warnings': result.quality_alerts
                },
                metadata={
                    'step': 'feature_generation_final_validation_step',
                    'shape': result.final_dataset.shape if hasattr(result.final_dataset, 'shape') else None,
                    'created_at': datetime.now().isoformat()
                }
            )
            tprint_success("✅ Final validation artifacts stored")
        else:
            tprint_warning("⚠️ Validation failed - not storing artifacts")

        # Generate human-readable report (best-effort)
        try:
            tprint_debug("📝 Generating validation report")
            report = self._generate_final_validation_report(
                result,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                data_shape=(len(data), len(data.columns) if isinstance(data, pd.DataFrame) else 0)
            )
            tprint_debug("📝 Formatting report as markdown")
            md = self._format_final_validation_markdown(report)
            tprint_debug("💾 Storing validation report")
            self._store_final_validation_report(report, md, symbol, timeframe)
            tprint_success("✅ Validation report generated and stored")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate report: {e}")
            pass

        tprint_success("🎉 Final validation execution complete")
        return result

    def _cleanup_resources(self) -> None:
        """Cleanup validation components."""
        tprint_step("🧹 Cleaning up validation resources")
        try:
            tprint_debug("🔧 Clearing validation component references")
            self.quality_alert_system = None
            self.quality_scorer = None
            self.advanced_metrics = None
            self.validation_manager = None
            self.set_state('cleaned_up_at', time.time())
            tprint_success("✅ Validation resources cleaned up successfully")
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
            self.logger.error(f"Error during cleanup: {e}")

    def _process_data(self, data, **kwargs):
        """Process data through final validation."""
        tprint_step("⚙️ Processing data through final validation")
        tprint_info(f"📊 Data shape: {data.shape if data is not None else 'None'}")
        try:
            if not VALIDATION_COMPONENTS_AVAILABLE:
                tprint_info("🔧 Using fallback validation (advanced components not available)")
                return self._fallback_validation(data, **kwargs)

            # Perform comprehensive final validation
            tprint_info("🔧 Using enhanced validation workflow")
            validation_result = self._perform_enhanced_validation(data, **kwargs)
            tprint_success("✅ Data processing completed successfully")
            return validation_result

        except Exception as e:
            tprint_error(f"❌ Final validation failed: {e}")
            self.logger.error(f"Final validation failed: {e}")
            raise

    def _get_validation_rules(self):
        """Get validation rules for this component."""
        tprint_debug("📋 Getting validation rules")
        rules = {
            'min_validation_score': self.min_validation_score,
            'min_rows': self.min_rows,
            'blocking_severities': self.blocking_severities,
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close']
        }
        tprint_info(f"✅ Validation rules: {rules}")
        return rules

    def _validate_component_specific(self, data):
        """Validate component-specific requirements."""
        tprint_debug("🔍 Validating component-specific requirements")
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            tprint_info(f"📊 Validating DataFrame with shape: {data.shape}")
            if len(data) < self.min_rows:
                error_msg = f"Data has {len(data)} rows, minimum required: {self.min_rows}"
                errors.append(error_msg)
                tprint_warning(f"⚠️ {error_msg}")
            else:
                tprint_success(f"✅ Row count check passed: {len(data)} >= {self.min_rows}")
            
            metadata['shape'] = data.shape
            metadata['columns'] = list(data.columns)
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                errors.append(error_msg)
                tprint_error(f"❌ {error_msg}")
            else:
                tprint_success("✅ All required columns present")
        else:
            tprint_error(f"❌ Invalid data type: {type(data)}, expected pandas.DataFrame")
            errors.append(f"Invalid data type: {type(data)}, expected pandas.DataFrame")
        
        result = {'errors': errors, 'warnings': warnings, 'metadata': metadata}
        tprint_info(f"✅ Component validation completed - errors: {len(errors)}, warnings: {len(warnings)}")
        return result

    # --- Reporting helpers ---
    def _generate_final_validation_report(self, result: FinalValidationResult, symbol: str, timeframe: str, direction: str, data_shape: tuple) -> Dict[str, Any]:
        from datetime import datetime as _dt
        alerts = result.quality_alerts or []
        severity_counts = {}
        for a in alerts:
            sev = None
            try:
                if isinstance(a, dict):
                    sev = a.get('severity') or a.get('level')
                elif hasattr(a, 'severity'):
                    sev = getattr(a, 'severity')
            except Exception:
                sev = None
            sev = str(sev) if sev is not None else 'unknown'
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        top_alerts = []
        for a in alerts[:20]:
            try:
                if isinstance(a, dict):
                    top_alerts.append({k: a.get(k) for k in ('message','severity','code','rule','metric','value') if k in a})
                else:
                    top_alerts.append(str(a))
            except Exception:
                continue

        return {
            'title': 'Final Validation Report',
            'timestamp': _dt.now().isoformat(),
            'configuration': {'symbol': symbol, 'timeframe': timeframe, 'direction': direction},
            'summary': {
                'rows': int(data_shape[0]) if data_shape else 0,
                'columns': int(data_shape[1]) if data_shape and len(data_shape)>1 else 0,
                'validation_score': float(result.validation_score),
                'quality_level': str(result.quality_level)
            },
            'validation_metadata': result.validation_metadata or {},
            'comprehensive_metrics': result.comprehensive_metrics or {},
            'severity_counts': severity_counts,
            'top_alerts': top_alerts,
            'recommendations': result.validation_recommendations or []
        }

    def _format_final_validation_markdown(self, report: Dict[str, Any]) -> str:
        md = f"# {report['title']}\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        cfg = report.get('configuration', {})
        md += "## 📌 Configuration\n\n"
        md += f"- Symbol: {cfg.get('symbol','?')}\n"
        md += f"- Timeframe: {cfg.get('timeframe','?')}\n"
        md += f"- Direction: {cfg.get('direction','?')}\n"
        summ = report.get('summary', {})
        md += "\n## 📊 Summary\n\n"
        md += f"- Rows: {summ.get('rows',0):,}\n"
        md += f"- Columns: {summ.get('columns',0)}\n"
        md += f"- Validation Score: {summ.get('validation_score',0.0):.4f}\n"
        md += f"- Quality Level: {summ.get('quality_level','?')}\n"
        # Severity
        md += "\n## 🚨 Alert Severity Counts\n\n"
        sev = report.get('severity_counts', {})
        if sev:
            for k, v in sev.items():
                md += f"- {k}: {v}\n"
        else:
            md += "- None\n"
        # Top alerts
        md += "\n## 🔎 Top Alerts (sample)\n\n"
        if report.get('top_alerts'):
            for a in report['top_alerts']:
                md += f"- {a}\n"
        else:
            md += "- None\n"
        # Metadata sections
        def _dump(title, obj):
            nonlocal md
            if obj:
                md += f"\n## {title}\n\n"
                for k, v in obj.items():
                    md += f"- {k}: {v}\n"
        _dump('🛠️ Validation Metadata', report.get('validation_metadata', {}))
        _dump('📈 Comprehensive Metrics', report.get('comprehensive_metrics', {}))
        # Recs
        if report.get('recommendations'):
            md += "\n## 💡 Recommendations\n\n"
            for r in report['recommendations']:
                md += f"- {r}\n"
        return md

    def _store_final_validation_report(self, report: Dict[str, Any], markdown: str, symbol: str, timeframe: str) -> None:
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json
        out_dir = _Path('outcomes')
        out_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        md_path = out_dir / f"final_validation_report_{symbol}_{timeframe}_{ts}.md"
        json_path = out_dir / f"final_validation_report_{symbol}_{timeframe}_{ts}.json"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)

    async def _perform_enhanced_final_validation(self, data: pd.DataFrame, symbol: str,
                                                  timeframe: str, direction: str,
                                                  custom_overrides: Optional[Dict[str, Any]]) -> FinalValidationResult:
        """Perform enhanced final validation using QualityAlertSystem."""
        tprint_step("🔍 Starting enhanced final validation")
        tprint_info(f"📊 Data shape: {data.shape}")
        tprint_info(f"🎯 Validation parameters: symbol={symbol}, timeframe={timeframe}, direction={direction}")
        
        try:
            # Step 1: Comprehensive quality scoring
            tprint_debug("📊 Step 1: Performing comprehensive quality scoring")
            quality_score = self.quality_scorer.score_data_quality(
                data, 
                symbol=symbol,
                timeframe=timeframe,
                direction=direction
            )
            tprint_info(f"✅ Quality score calculated: {quality_score.overall_score:.2f}")
            tprint_info(f"📈 Quality level: {quality_score.level.value if quality_score.level else 'unknown'}")
            
            # Step 2: Advanced quality metrics assessment
            tprint_debug("📊 Step 2: Performing advanced quality metrics assessment")
            advanced_assessment = self.advanced_metrics.assess_data_quality(data)
            tprint_info(f"✅ Advanced assessment completed: {len(advanced_assessment.metrics)} metrics calculated")
            if hasattr(advanced_assessment, 'issues') and advanced_assessment.issues:
                tprint_warning(f"⚠️ Found {len(advanced_assessment.issues)} data issues")
            else:
                tprint_success("✅ No data issues found")
            
            # Step 3: Quality alert system check
            tprint_debug("🚨 Step 3: Checking quality alerts")
            quality_alerts = self.quality_alert_system.check_quality_alerts(data, quality_score)
            tprint_info(f"✅ Quality alert check completed: {len(quality_alerts)} alerts found")
            if quality_alerts:
                tprint_warning(f"⚠️ Quality alerts detected: {len(quality_alerts)} total")
                # Log severity breakdown
                severity_counts = {}
                for alert in quality_alerts:
                    severity = getattr(alert, 'severity', 'unknown')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                for severity, count in severity_counts.items():
                    tprint_warning(f"   - {severity}: {count} alerts")
            else:
                tprint_success("✅ No quality alerts found")
            
            # Step 4: Comprehensive validation using validation manager
            tprint_debug("🔍 Step 4: Performing comprehensive validation")
            validation_result = await self.validation_manager.perform_comprehensive_validation(
                data, symbol=symbol, timeframe=timeframe, direction=direction
            )
            tprint_info(f"✅ Comprehensive validation completed: success={validation_result.success}")
            if hasattr(validation_result, 'metrics') and validation_result.metrics:
                tprint_info(f"📊 Validation metrics: {len(validation_result.metrics)} calculated")
            if not validation_result.success and hasattr(validation_result, 'failures'):
                tprint_error(f"❌ Validation failures: {len(validation_result.failures)} found")
                for i, failure in enumerate(validation_result.failures[:3], 1):
                    tprint_error(f"   {i}. {failure}")
            else:
                tprint_success("✅ All validation checks passed")
            
            # Step 5: Generate recommendations
            tprint_debug("💡 Step 5: Generating validation recommendations")
            recommendations = self._generate_validation_recommendations(
                quality_score, advanced_assessment, quality_alerts, validation_result
            )
            tprint_info(f"✅ Generated {len(recommendations)} recommendations")
            if recommendations:
                tprint_info("📋 Top recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    tprint_info(f"   {i}. {rec}")
            
            # Determine overall success and quality level
            tprint_debug("🎯 Determining overall validation success")
            # Only consider configured severity levels as blocking
            blocking_alerts = [alert for alert in quality_alerts 
                             if hasattr(alert, 'severity') and 
                             alert.severity in self.blocking_severities]
            
            tprint_info(f"🚨 Blocking alerts found: {len(blocking_alerts)} (threshold: {self.min_validation_score})")
            tprint_info(f"📊 Quality score: {quality_score.overall_score:.2f} (min required: {self.min_validation_score})")
            tprint_info(f"✅ Validation manager success: {validation_result.success}")
            
            success = (quality_score.overall_score >= self.min_validation_score and 
                      len(blocking_alerts) == 0 and 
                      validation_result.success)
            
            quality_level = quality_score.level.value if quality_score.level else "unknown"
            tprint_info(f"🎯 Overall validation success: {success}")
            tprint_info(f"📈 Final quality level: {quality_level}")
            
            # Compile comprehensive result
            tprint_debug("📦 Compiling comprehensive validation result")
            result = FinalValidationResult(
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
                },
                final_dataset=data
            )
            tprint_success("🎉 Enhanced final validation completed successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Enhanced final validation failed: {e}")
            tprint_error(f"📊 Data shape at failure: {data.shape if data is not None else 'None'}")
            tprint_error(f"🔧 Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
            import traceback
            tprint_error(f"📋 Full traceback: {traceback.format_exc()}")
            return FinalValidationResult(
                success=False,
                validation_score=0.0,
                quality_level="error",
                validation_metadata={},
                quality_alerts=[],
                comprehensive_metrics={},
                validation_recommendations=["Check data format and try again"],
                artifacts={},
                final_dataset=data,
                error_message=str(e)
            )

    def _generate_validation_recommendations(self, quality_score, advanced_assessment, 
                                              quality_alerts, validation_result) -> List[str]:
        """Generate validation recommendations based on assessment results."""
        tprint_debug("💡 Generating validation recommendations")
        recommendations = []
        tprint_info(f"📊 Input parameters:")
        tprint_info(f"   - Quality score: {quality_score.overall_score:.2f}")
        tprint_info(f"   - Quality alerts: {len(quality_alerts)}")
        tprint_info(f"   - Validation success: {validation_result.success}")
        tprint_info(f"   - Advanced assessment issues: {len(getattr(advanced_assessment, 'issues', [])) if hasattr(advanced_assessment, 'issues') else 'N/A'}")
        
        # Quality score recommendations with specific guidance
        if quality_score.overall_score < 80:
            rec_msg = f"Data quality score {quality_score.overall_score:.1f} is below 80 - review data completeness and accuracy"
            recommendations.append(rec_msg)
            tprint_warning(f"⚠️ Quality score recommendation: {rec_msg}")
            if hasattr(quality_score, 'component_scores'):
                low_scores = [(k, v) for k, v in quality_score.component_scores.items() if v < 70]
                if low_scores:
                    rec_msg = f"Focus on improving: {', '.join([f'{k} ({v:.1f})' for k, v in low_scores])}"
                    recommendations.append(rec_msg)
                    tprint_warning(f"⚠️ Component score recommendation: {rec_msg}")
        else:
            tprint_success(f"✅ Quality score acceptable: {quality_score.overall_score:.1f}")
        
        # Alert-based recommendations with specific alert details
        if quality_alerts:
            rec_msg = f"Address {len(quality_alerts)} quality alerts"
            recommendations.append(rec_msg)
            tprint_warning(f"⚠️ Alert recommendation: {rec_msg}")
            # Include top 2-3 specific alerts for guidance
            top_alerts = quality_alerts[:3]
            for i, alert in enumerate(top_alerts, 1):
                alert_type = getattr(alert, 'type', 'unknown')
                alert_message = getattr(alert, 'message', 'No details available')
                rec_msg = f"  {i}. {alert_type}: {alert_message}"
                recommendations.append(rec_msg)
                tprint_warning(f"   ⚠️ {rec_msg}")
        else:
            tprint_success("✅ No quality alerts found")
        
        # Advanced assessment recommendations with specific issues
        if hasattr(advanced_assessment, 'issues') and advanced_assessment.issues:
            rec_msg = f"Resolve {len(advanced_assessment.issues)} data issues"
            recommendations.append(rec_msg)
            tprint_warning(f"⚠️ Data issues recommendation: {rec_msg}")
            # Include top 2-3 specific issues
            top_issues = advanced_assessment.issues[:3]
            for i, issue in enumerate(top_issues, 1):
                issue_desc = str(issue) if not isinstance(issue, dict) else issue.get('description', str(issue))
                rec_msg = f"  {i}. {issue_desc}"
                recommendations.append(rec_msg)
                tprint_warning(f"   ⚠️ {rec_msg}")
        else:
            tprint_success("✅ No data issues found in advanced assessment")
        
        # Validation result recommendations with specific failures
        if not validation_result.success:
            rec_msg = "Review validation failures and data integrity"
            recommendations.append(rec_msg)
            tprint_error(f"❌ Validation failure recommendation: {rec_msg}")
            if hasattr(validation_result, 'failures') and validation_result.failures:
                for i, failure in enumerate(validation_result.failures[:2], 1):
                    failure_desc = str(failure) if not isinstance(failure, dict) else failure.get('description', str(failure))
                    rec_msg = f"  {i}. Validation failure: {failure_desc}"
                    recommendations.append(rec_msg)
                    tprint_error(f"   ❌ {rec_msg}")
        else:
            tprint_success("✅ All validation checks passed")
        
        tprint_info(f"✅ Generated {len(recommendations)} total recommendations")
        return recommendations

    async def _fallback_final_validation(self, data: pd.DataFrame) -> FinalValidationResult:
        """Fallback final validation when advanced components are not available."""
        tprint_step("🔧 Starting fallback final validation")
        tprint_info(f"📊 Data shape: {data.shape}")
        tprint_info(f"📋 Using basic validation checks (advanced components not available)")
        
        try:
            # Basic validation checks
            tprint_debug("🔍 Performing basic validation checks")
            basic_checks = {
                'has_data': not data.empty,
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_all_nan': not data.isnull().all().any(),
                'sufficient_rows': len(data) >= self.min_rows,
                'no_infinite_values': not np.isinf(data.select_dtypes(include=[np.number])).any().any()
            }
            
            tprint_info("📊 Basic validation check results:")
            for check_name, passed in basic_checks.items():
                status = "✅" if passed else "❌"
                tprint_info(f"   {status} {check_name}: {passed}")
            
            # Identify failing checks
            failing_checks = [check_name for check_name, passed in basic_checks.items() if not passed]
            tprint_info(f"❌ Failing checks: {failing_checks if failing_checks else 'None'}")
            
            success = all(basic_checks.values())
            validation_score = sum(basic_checks.values()) / len(basic_checks) * 100
            tprint_info(f"🎯 Overall success: {success}")
            tprint_info(f"📊 Validation score: {validation_score:.2f}%")
            
            # Generate specific recommendations based on failing checks
            tprint_debug("💡 Generating recommendations based on failing checks")
            recommendations = []
            if not success:
                tprint_warning("⚠️ Generating recommendations for failed validation")
                if 'has_data' in failing_checks:
                    recommendations.append("Ensure data is loaded and not empty")
                    tprint_warning("   - Data is empty or None")
                if 'has_required_columns' in failing_checks:
                    missing_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in data.columns]
                    recommendations.append(f"Add missing required columns: {missing_cols}")
                    tprint_warning(f"   - Missing columns: {missing_cols}")
                if 'no_all_nan' in failing_checks:
                    nan_cols = data.columns[data.isnull().all()].tolist()
                    recommendations.append(f"Remove or fix columns with all NaN values: {nan_cols}")
                    tprint_warning(f"   - All-NaN columns: {nan_cols}")
                if 'sufficient_rows' in failing_checks:
                    recommendations.append(f"Ensure at least {self.min_rows} rows of data (current: {len(data)})")
                    tprint_warning(f"   - Insufficient rows: {len(data)} < {self.min_rows}")
                if 'no_infinite_values' in failing_checks:
                    recommendations.append("Remove infinite values from numeric columns")
                    tprint_warning("   - Infinite values detected in numeric columns")
                recommendations.append("Install validation components for enhanced assessment")
                tprint_info(f"💡 Generated {len(recommendations)} recommendations")
            else:
                tprint_success("✅ All basic validation checks passed - no recommendations needed")
            
            tprint_info("📦 Compiling fallback validation result")
            result = FinalValidationResult(
                success=success,
                validation_score=validation_score,
                quality_level='basic',
                validation_metadata={'method': 'fallback_basic', 'failing_checks': failing_checks},
                quality_alerts=[] if success else [{'type': 'basic_validation_failed', 'failing_checks': failing_checks}],
                comprehensive_metrics=basic_checks,
                validation_recommendations=recommendations,
                artifacts={'basic_checks': basic_checks, 'failing_checks': failing_checks},
                final_dataset=data,
                error_message=None if success else f"Basic validation failed: {', '.join(failing_checks)}"
            )
            tprint_success("🎉 Fallback validation completed")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Fallback validation failed with exception: {e}")
            tprint_error(f"📊 Data shape at failure: {data.shape if data is not None else 'None'}")
            import traceback
            tprint_error(f"📋 Full traceback: {traceback.format_exc()}")
            return FinalValidationResult(
                success=False,
                validation_score=0.0,
                quality_level="error",
                validation_metadata={},
                quality_alerts=[],
                comprehensive_metrics={},
                validation_recommendations=[],
                artifacts={},
                final_dataset=data,
                error_message=str(e)
            )

    # Required utility methods for BasePreTrainingComponent

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
    **kwargs
) -> ComponentResult:
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
        **kwargs: Additional arguments

    Returns:
        ComponentResult: Result of the final validation step
    """
    tprint_step("🚀 Starting final validation step handler")
    tprint_info(f"📋 Handler parameters:")
    tprint_info(f"   - Symbol: {symbol}")
    tprint_info(f"   - Timeframe: {timeframe}")
    tprint_info(f"   - Exchange: {exchange}")
    tprint_info(f"   - Direction: {direction}")
    tprint_info(f"   - Intensity: {intensity}")
    tprint_info(f"   - Lookback days: {lookback_days}")
    tprint_info(f"   - Start date: {start_date}")
    tprint_info(f"   - End date: {end_date}")
    tprint_info(f"   - Custom overrides: {custom_overrides is not None}")
    tprint_info(f"   - Additional kwargs: {len(kwargs)} items")
    
    tprint_debug("🔍 Getting artifact manager")
    artifact_manager = get_pretraining_artifact_manager()
    tprint_success("✅ Artifact manager retrieved")

    try:
        # Create the step instance
        tprint_debug("🔧 Creating FeatureGenerationFinalValidationStep instance")
        step = FeatureGenerationFinalValidationStep(
            name="final_validation_step",
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'custom_overrides': custom_overrides or {}
            }
        )
        tprint_success("✅ Step instance created successfully")

        # Load data for processing
        tprint_debug("📊 Loading data for processing")
        data = await step.load_data(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        tprint_info(f"✅ Data loaded successfully - shape: {data.shape if data is not None else 'None'}")

        # Process the data
        tprint_debug("⚙️ Processing data through final validation")
        result = await step.process_data_async(
            data,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            custom_overrides=custom_overrides or {},
            **kwargs
        )
        tprint_info(f"✅ Data processing completed - success: {result.get('success', False)}")

        # Create result object
        tprint_debug("📦 Creating FinalValidationResult object")
        step_result = FinalValidationResult(
            success=result.get('success', False),
            validation_score=result.get('validation_score', 0.0),
            quality_level=result.get('quality_level', 'unknown'),
            validation_metadata=result.get('validation_metadata', {}),
            quality_alerts=result.get('quality_alerts', []),
            comprehensive_metrics=result.get('comprehensive_metrics', {}),
            validation_recommendations=result.get('validation_recommendations', []),
            artifacts=result.get('artifacts', {}),
            final_dataset=result.get('final_dataset'),
            error_message=result.get('error_message')
        )
        tprint_info(f"✅ Step result created - success: {step_result.success}, score: {step_result.validation_score:.2f}")

        # Convert to ComponentResult
        tprint_debug("🔄 Converting to ComponentResult")
        component_result = ComponentResult(
            success=step_result.success,
            data=step_result.final_dataset,
            metadata={
                'step_name': 'feature_generation_final_validation_step',
                'validation_score': step_result.validation_score,
                'quality_level': step_result.quality_level,
                'validation_metadata': step_result.validation_metadata
            },
            artifacts=step_result.artifacts,
            error_message=step_result.error_message
        )
        tprint_success("✅ ComponentResult created successfully")

        # Save artifacts
        tprint_debug("💾 Saving step result artifacts")
        await artifact_manager.save_step_result(
            step_name='feature_generation_final_validation_step',
            result=component_result,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction
        )
        tprint_success("✅ Step result artifacts saved")
        
        tprint_success("🎉 Final validation step handler completed successfully")
        return component_result

    except Exception as e:
        error_message = f"Final validation step failed: {str(e)}"
        tprint_error(f"❌ Handler function failed: {error_message}")
        tprint_error(f"📊 Parameters at failure: symbol={symbol}, timeframe={timeframe}, direction={direction}")
        import traceback
        tprint_error(f"📋 Full traceback: {traceback.format_exc()}")

        # Return failed result
        tprint_debug("📦 Creating failed ComponentResult")
        component_result = ComponentResult(
            success=False,
            data=None,
            metadata={'step_name': 'feature_generation_final_validation_step'},
            artifacts={},
            error_message=error_message
        )
        tprint_warning("⚠️ Returning failed result from handler")

        return component_result
