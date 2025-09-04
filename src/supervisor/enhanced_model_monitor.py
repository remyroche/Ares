from __future__ import annotations
'\nEnhanced Model Monitor\n\nThis module provides comprehensive model behavior monitoring, feature importance tracking,\ndecision path analysis, and ensemble performance monitoring that integrates with the\nexisting performance monitoring infrastructure.\n'
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from dataclasses_json import dataclass_json
from src.utils.logger import system_logger
if TYPE_CHECKING:
    import asyncio

class ModelDriftType(Enum):
    """Model drift types."""
    CONCEPT_DRIFT = 'concept_drift'
    DATA_DRIFT = 'data_drift'
    LABEL_DRIFT = 'label_drift'
    FEATURE_DRIFT = 'feature_drift'

@dataclass_json
@dataclass
class ModelDriftAlert:
    """Model drift alert."""
    model_id: str
    model_type: str
    drift_type: ModelDriftType
    drift_score: float
    threshold: float
    timestamp: datetime
    features_affected: list[str]
    severity: str
    description: str

@dataclass_json
@dataclass
class FeatureDriftMetrics:
    """Feature drift metrics."""
    feature_name: str
    current_distribution: dict[str, float]
    reference_distribution: dict[str, float]
    drift_score: float
    ks_statistic: float
    p_value: float
    is_drifted: bool

@dataclass_json
@dataclass
class ModelPerformanceSnapshot:
    """Model performance snapshot."""
    model_id: str
    model_type: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    prediction_confidence: float
    feature_importance_stability: float
    concept_drift_score: float
    data_drift_score: float

@dataclass_json
@dataclass
class EnsemblePerformanceMetrics:
    """Ensemble performance metrics."""
    ensemble_id: str
    timestamp: datetime
    ensemble_accuracy: float
    individual_model_accuracies: dict[str, float]
    ensemble_weights: dict[str, float]
    diversity_score: float
    agreement_score: float
    meta_learner_performance: float | None = None

class EnhancedModelMonitor:
    """
    Enhanced model monitor that integrates with existing performance monitoring
    to provide comprehensive model behavior tracking.
    """

    def __init__(self, config: dict[str, Any], performance_monitor: PerformanceMonitor) -> None:
        """
        Initialize enhanced model monitor.

        Args:
            config: Configuration dictionary
            performance_monitor: Existing performance monitor instance
        """
        self.config = config
        self.performance_monitor = performance_monitor
        self.logger = system_logger.getChild('EnhancedModelMonitor')
        self.monitor_config = config.get('enhanced_model_monitor', {})
        self.drift_detection_enabled = self.monitor_config.get('drift_detection_enabled', True)
        self.feature_importance_tracking = self.monitor_config.get('feature_importance_tracking', True)
        self.decision_path_analysis = self.monitor_config.get('decision_path_analysis', True)
        self.ensemble_monitoring = self.monitor_config.get('ensemble_monitoring', True)
        self.drift_check_interval = self.monitor_config.get('drift_check_interval', 300)
        self.performance_snapshot_interval = self.monitor_config.get('performance_snapshot_interval', 60)
        self.feature_analysis_interval = self.monitor_config.get('feature_analysis_interval', 600)
        self.model_performance_history: dict[str, list[ModelPerformanceSnapshot]] = {}
        self.ensemble_performance_history: dict[str, list[EnsemblePerformanceMetrics]] = {}
        self.drift_alerts: list[ModelDriftAlert] = []
        self.feature_drift_history: dict[str, list[FeatureDriftMetrics]] = {}
        self.reference_distributions: dict[str, dict[str, float]] = {}
        self.reference_performance: dict[str, float] = {}
        self.is_monitoring = False
        self.monitoring_tasks: list[asyncio.Task] = []
        self.logger.info('🚀 Enhanced Model Monitor initialized')

    @handles_errors(error_handlers={ValueError: (False, 'Invalid model monitor configuration'), AttributeError: (False, 'Missing required monitor parameters')}, default_return=False, context='model monitor initialization')
    async def initialize(self) -> bool:
        """Initialize the enhanced model monitor."""
        try:
            self.logger.info('Initializing Enhanced Model Monitor...')
            await self._load_reference_data()
            await self._initialize_drift_detection()
            await self._initialize_feature_tracking()
            await self._initialize_ensemble_monitoring()
            self.logger.info('✅ Enhanced Model Monitor initialization completed')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Enhanced Model Monitor initialization failed: {e}')
            return False

    @handles_errors(fallback=None)
    async def _load_reference_data(self) -> None:
        """Load reference data for drift detection."""
        try:
            self.logger.info('Loading reference data for drift detection...')
        except Exception as e:
            self.logger.exception(f'Error loading reference data: {e}')

    @handles_errors(fallback=None)
    async def _initialize_drift_detection(self) -> None:
        """Initialize drift detection components."""
        try:
            self.logger.info('Initializing drift detection components...')
        except Exception as e:
            self.logger.exception(f'Error initializing drift detection: {e}')

    @handles_errors(fallback=None)
    async def _initialize_feature_tracking(self) -> None:
        """Initialize feature importance tracking."""
        try:
            self.logger.info('Initializing feature importance tracking...')
        except Exception as e:
            self.logger.exception(f'Error initializing feature tracking: {e}')

    @handles_errors(fallback=None)
    async def _initialize_ensemble_monitoring(self) -> None:
        """Initialize ensemble performance monitoring."""
        try:
            self.logger.info('Initializing ensemble monitoring...')
        except Exception as e:
            self.logger.exception(f'Error initializing ensemble monitoring: {e}')