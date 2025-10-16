"""
Comprehensive Data Quality Scoring System

This module provides a comprehensive data quality scoring system that integrates
with all tools in src/utils/data/quality/ to assess data quality throughout
the training pipeline, particularly in market_analysis steps.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from src.utils.logger import system_logger
from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds, QualityResult
from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics, QualityAssessment
from src.utils.data.quality.data_cleaning import DataCleaner
from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
from src.utils.data.quality.quality_alert_system import QualityAlertSystem

logger = system_logger.getChild('ComprehensiveQualityScorer')

class QualityScoreLevel(Enum):
    """Quality score levels."""
    EXCELLENT = "excellent"    # 90-100
    GOOD = "good"             # 80-89
    FAIR = "fair"             # 70-79
    POOR = "poor"             # 60-69
    CRITICAL = "critical"     # 0-59

@dataclass
class QualityScore:
    """Comprehensive quality score with detailed breakdown."""
    overall_score: float
    level: QualityScoreLevel
    component_scores: Dict[str, float]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    assessment_timestamp: datetime
    data_shape: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityTrend:
    """Quality trend analysis over time."""
    scores: List[float]
    timestamps: List[datetime]
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0-1, how strong the trend is
    volatility: float  # Score volatility over time

class ComprehensiveQualityScorer:
    """Comprehensive data quality scoring system."""

    def __init__(self):
        self.logger = logger.getChild('ComprehensiveQualityScorer')

        # Initialize quality assessment tools
        self.advanced_metrics = AdvancedQualityMetrics()
        self.data_cleaner = DataCleaner()
        self.statistical_validator = StatisticalValidator()
        self.alert_system = QualityAlertSystem()

        # Quality score history for trend analysis
        self.quality_history: List[QualityScore] = []

        # Component weights for overall score calculation
        self.component_weights = {
            'completeness': 0.25,      # Data completeness
            'accuracy': 0.20,          # Data accuracy
            'consistency': 0.20,       # Data consistency
            'validity': 0.15,          # Data validity
            'timeliness': 0.10,        # Data timeliness
            'uniqueness': 0.10         # Data uniqueness
        }

        self.logger.info("📊 Comprehensive Quality Scorer initialized")

    def assess_data_quality(self,
                           data: pd.DataFrame,
                           context: str = "general",
                           step_name: Optional[str] = None,
                           data_type: str = "klines") -> QualityScore:
        """
        Perform comprehensive data quality assessment.

        Args:
            data: DataFrame to assess
            context: Assessment context (e.g., 'market_analysis', 'data_collection')
            step_name: Name of the pipeline step
            data_type: Type of data ('klines', 'aggtrades', 'futures')

        Returns:
            Comprehensive quality score
        """
        self.logger.info(f"🔍 Assessing data quality for {context} - {step_name}")

        # Initialize component scores
        component_scores = {}
        issues = []
        warnings = []
        recommendations = []

        try:
            # 1. Completeness Assessment
            completeness_score = self._assess_completeness(data, data_type)
            component_scores['completeness'] = completeness_score

            if completeness_score < 0.8:
                issues.append(f"Low data completeness: {completeness_score:.2f}")
                recommendations.append("Check data source for missing data")

            # 2. Accuracy Assessment
            accuracy_score = self._assess_accuracy(data, data_type)
            component_scores['accuracy'] = accuracy_score

            if accuracy_score < 0.8:
                issues.append(f"Data accuracy concerns: {accuracy_score:.2f}")
                recommendations.append("Validate data against known sources")

            # 3. Consistency Assessment
            consistency_score = self._assess_consistency(data, data_type)
            component_scores['consistency'] = consistency_score

            if consistency_score < 0.8:
                issues.append(f"Data consistency issues: {consistency_score:.2f}")
                recommendations.append("Check for data format inconsistencies")

            # 4. Validity Assessment
            validity_score = self._assess_validity(data, data_type)
            component_scores['validity'] = validity_score

            if validity_score < 0.8:
                issues.append(f"Data validity problems: {validity_score:.2f}")
                recommendations.append("Validate data ranges and formats")

            # 5. Timeliness Assessment
            timeliness_score = self._assess_timeliness(data, data_type)
            component_scores['timeliness'] = timeliness_score

            if timeliness_score < 0.8:
                warnings.append(f"Data timeliness concerns: {timeliness_score:.2f}")
                recommendations.append("Check data freshness and update frequency")

            # 6. Uniqueness Assessment
            uniqueness_score = self._assess_uniqueness(data, data_type)
            component_scores['uniqueness'] = uniqueness_score

            if uniqueness_score < 0.9:
                warnings.append(f"Data uniqueness issues: {uniqueness_score:.2f}")
                recommendations.append("Check for duplicate records")

            # Calculate overall score
            overall_score = self._calculate_overall_score(component_scores)
            level = self._determine_quality_level(overall_score)

            # Create quality score
            quality_score = QualityScore(
                overall_score=overall_score,
                level=level,
                component_scores=component_scores,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                assessment_timestamp=datetime.now(),
                data_shape=(len(data), len(data.columns)),
                metadata={
                    'context': context,
                    'step_name': step_name,
                    'data_type': data_type
                }
            )

            # Store in history
            self.quality_history.append(quality_score)

            # Trigger alerts if needed
            self._trigger_quality_alerts(quality_score, context, step_name)

            self.logger.info(f"✅ Quality assessment completed: {overall_score:.2f} ({level.value})")

            return quality_score

        except Exception as e:
            self.logger.error(f"❌ Error in quality assessment: {e}")
            # Return minimal quality score on error
            return QualityScore(
                overall_score=0.0,
                level=QualityScoreLevel.CRITICAL,
                component_scores={},
                issues=[f"Assessment error: {str(e)}"],
                warnings=[],
                recommendations=["Fix assessment error and retry"],
                assessment_timestamp=datetime.now(),
                data_shape=(len(data) if data is not None else 0, len(data.columns) if data is not None else 0),
                metadata={'error': str(e)}
            )

    def _assess_completeness(self, data: pd.DataFrame, data_type: str) -> float:
        """Assess data completeness."""
        if data.empty:
            return 0.0

        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        completeness = 1.0 - missing_ratio

        # Check for required columns based on data type
        required_columns = self._get_required_columns(data_type)
        missing_required = sum(1 for col in required_columns if col not in data.columns)
        if missing_required > 0:
            completeness *= (len(required_columns) - missing_required) / len(required_columns)

        return max(0.0, completeness)

    def _assess_accuracy(self, data: pd.DataFrame, data_type: str) -> float:
        """Assess data accuracy using advanced metrics."""
        try:
            # Use advanced quality metrics for accuracy assessment
            assessment = self.advanced_metrics.comprehensive_quality_assessment(data, "accuracy", data_type)

            # Convert assessment to accuracy score
            accuracy_score = 1.0 - (assessment.issues_found / max(1, len(data)))
            return max(0.0, min(1.0, accuracy_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Error in accuracy assessment: {e}")
            return 0.5  # Default score on error

    def _assess_consistency(self, data: pd.DataFrame, data_type: str) -> float:
        """Assess data consistency."""
        if data.empty:
            return 0.0

        consistency_score = 1.0

        # Check data type consistency
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for mixed types in object columns
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    type_consistency = len(set(type(val).__name__ for val in non_null_values))
                    if type_consistency > 1:
                        consistency_score *= 0.8  # Reduce score for mixed types

        # Check for temporal consistency if timestamp column exists
        if 'timestamp' in data.columns:
            try:
                timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
                valid_timestamps = timestamps.dropna()
                if len(valid_timestamps) > 1:
                    # Check for reasonable time gaps
                    time_diffs = valid_timestamps.diff().dropna()
                    if not time_diffs.empty:
                        # Check for extreme time gaps (more than 1 day for most data types)
                        max_gap = time_diffs.max()
                        if max_gap > pd.Timedelta(days=1):
                            consistency_score *= 0.7
            except Exception:
                consistency_score *= 0.5

        return max(0.0, consistency_score)

    def _assess_validity(self, data: pd.DataFrame, data_type: str) -> float:
        """Assess data validity using data quality validator."""
        try:
            # Create appropriate thresholds based on data type
            thresholds = QualityThresholds(
                max_nan_ratio=0.1,
                max_infinite_count=0,
                min_unique_values=2,
                max_constant_ratio=0.95
            )

            validator = DataQualityFramework(thresholds)
            result = validator.validate_dataframe_quality(data)

            # Convert validation result to validity score
            if result.passed:
                validity_score = 1.0
            else:
                # Reduce score based on number of issues
                issue_penalty = len(result.issues) * 0.1
                validity_score = max(0.0, 1.0 - issue_penalty)

            return validity_score

        except Exception as e:
            self.logger.warning(f"⚠️ Error in validity assessment: {e}")
            return 0.5  # Default score on error

    def _assess_timeliness(self, data: pd.DataFrame, data_type: str) -> float:
        """Assess data timeliness."""
        if data.empty or 'timestamp' not in data.columns:
            return 0.5  # Default score if no timestamp

        try:
            timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
            valid_timestamps = timestamps.dropna()

            if valid_timestamps.empty:
                return 0.0

            # Check how recent the data is
            latest_timestamp = valid_timestamps.max()
            now = pd.Timestamp.now()

            # Calculate age of latest data
            age = now - latest_timestamp

            # Define acceptable age based on data type
            acceptable_age = {
                'aggtrades': pd.Timedelta(minutes=5),
                'klines': pd.Timedelta(minutes=15),
                'futures': pd.Timedelta(hours=1)
            }.get(data_type, pd.Timedelta(hours=1))

            if age <= acceptable_age:
                timeliness_score = 1.0
            elif age <= acceptable_age * 2:
                timeliness_score = 0.8
            elif age <= acceptable_age * 4:
                timeliness_score = 0.6
            else:
                timeliness_score = 0.3

            return timeliness_score

        except Exception as e:
            self.logger.warning(f"⚠️ Error in timeliness assessment: {e}")
            return 0.5  # Default score on error

    def _assess_uniqueness(self, data: pd.DataFrame, data_type: str) -> float:
        """Assess data uniqueness."""
        if data.empty:
            return 0.0

        # Check for duplicate rows
        total_rows = len(data)
        unique_rows = len(data.drop_duplicates())
        uniqueness_ratio = unique_rows / total_rows

        # Check for duplicate timestamps (if applicable)
        if 'timestamp' in data.columns:
            unique_timestamps = data['timestamp'].nunique()
            timestamp_uniqueness = unique_timestamps / total_rows
            uniqueness_ratio = (uniqueness_ratio + timestamp_uniqueness) / 2

        return uniqueness_ratio

    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        if not component_scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for component, score in component_scores.items():
            weight = self.component_weights.get(component, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_quality_level(self, score: float) -> QualityScoreLevel:
        """Determine quality level based on score."""
        if score >= 0.9:
            return QualityScoreLevel.EXCELLENT
        elif score >= 0.8:
            return QualityScoreLevel.GOOD
        elif score >= 0.7:
            return QualityScoreLevel.FAIR
        elif score >= 0.6:
            return QualityScoreLevel.POOR
        else:
            return QualityScoreLevel.CRITICAL

    def _get_required_columns(self, data_type: str) -> List[str]:
        """Get required columns for data type."""
        required_columns = {
            'klines': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'aggtrades': ['timestamp', 'price', 'quantity'],
            'futures': ['timestamp']
        }
        return required_columns.get(data_type, ['timestamp'])

    def _trigger_quality_alerts(self, quality_score: QualityScore, context: str, step_name: Optional[str]):
        """Trigger quality alerts if needed."""
        try:
            if quality_score.level in [QualityScoreLevel.POOR, QualityScoreLevel.CRITICAL]:
                self.alert_system.trigger_alert(
                    level="critical" if quality_score.level == QualityScoreLevel.CRITICAL else "warning",
                    message=f"Low data quality detected: {quality_score.overall_score:.2f}",
                    context=f"{context} - {step_name}",
                    details={
                        'score': quality_score.overall_score,
                        'level': quality_score.level.value,
                        'issues': quality_score.issues,
                        'component_scores': quality_score.component_scores
                    }
                )
        except Exception as e:
            self.logger.warning(f"⚠️ Error triggering quality alert: {e}")

    def get_quality_trend(self, context: Optional[str] = None, step_name: Optional[str] = None) -> Optional[QualityTrend]:
        """Get quality trend analysis."""
        if len(self.quality_history) < 2:
            return None

        # Filter by context and step if specified
        filtered_scores = self.quality_history
        if context:
            filtered_scores = [s for s in filtered_scores if s.metadata.get('context') == context]
        if step_name:
            filtered_scores = [s for s in filtered_scores if s.metadata.get('step_name') == step_name]

        if len(filtered_scores) < 2:
            return None

        scores = [s.overall_score for s in filtered_scores]
        timestamps = [s.assessment_timestamp for s in filtered_scores]

        # Calculate trend
        if len(scores) >= 2:
            # Simple linear trend calculation
            x = np.arange(len(scores))
            y = np.array(scores)
            trend_slope = np.polyfit(x, y, 1)[0]

            if trend_slope > 0.01:
                trend_direction = "improving"
            elif trend_slope < -0.01:
                trend_direction = "declining"
            else:
                trend_direction = "stable"

            trend_strength = abs(trend_slope)
            volatility = np.std(scores)

            return QualityTrend(
                scores=scores,
                timestamps=timestamps,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                volatility=volatility
            )

        return None

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary."""
        if not self.quality_history:
            return {'total_assessments': 0}

        recent_scores = [s.overall_score for s in self.quality_history[-10:]]
        all_scores = [s.overall_score for s in self.quality_history]

        # Count by quality level
        level_counts = {}
        for level in QualityScoreLevel:
            level_counts[level.value] = sum(1 for s in self.quality_history if s.level == level)

        return {
            'total_assessments': len(self.quality_history),
            'average_score': np.mean(all_scores),
            'recent_average': np.mean(recent_scores),
            'best_score': np.max(all_scores),
            'worst_score': np.min(all_scores),
            'level_distribution': level_counts,
            'trend': self.get_quality_trend()
        }

# Global instance
_quality_scorer: Optional[ComprehensiveQualityScorer] = None

def get_quality_scorer() -> ComprehensiveQualityScorer:
    """Get the global quality scorer instance."""
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = ComprehensiveQualityScorer()
    return _quality_scorer

def assess_quality(data: pd.DataFrame,
                  context: str = "general",
                  step_name: Optional[str] = None,
                  data_type: str = "klines") -> QualityScore:
    """Convenience function for quality assessment."""
    scorer = get_quality_scorer()
    return scorer.assess_data_quality(data, context, step_name, data_type)
